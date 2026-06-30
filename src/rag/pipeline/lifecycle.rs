//! Construction (`from_parts`, `open_or_create`), persistence (`save`),
//! accessors (`chunk_count`, `store`, `turbo`), and the `build` view.

use super::state::RagPipeline;
use crate::rag::{DocumentStore, ErasedEmbedder, TurboIndex, TurboVectorIndex};
use anyhow::{Context, Result, bail};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

impl RagPipeline {
    /// Construct a pipeline from an explicit store and turbo index.
    ///
    /// Used by `open_or_create` and tests that want full control. Most
    /// users should call [`Self::open_or_create`].
    #[allow(dead_code)]
    pub(crate) fn from_parts(
        store: Arc<DocumentStore>,
        turbo: Arc<RwLock<TurboIndex>>,
        supported_extensions: Option<HashSet<String>>,
    ) -> Self {
        Self::new(store, turbo, supported_extensions)
    }

    /// Open (or initialise) a RAG pipeline at the given on-disk paths.
    ///
    /// * If both `db_path` and `index_path` exist, both are loaded and their
    ///   chunk counts are validated to match. A mismatch returns an error.
    /// * If neither exists, fresh empty artifacts are created (the index
    ///   file is NOT written to disk until [`Self::save`] is called).
    /// * If only one exists, an error is returned — the artifacts must
    ///   stay in sync. Caller can delete both and retry.
    ///
    /// `embedding_dim` and `bit_width` are required when creating a fresh
    /// index; they're ignored when loading.
    pub(crate) async fn open_or_create(
        db_path: &Path,
        index_path: &Path,
        embedding_dim: usize,
        bit_width: usize,
        supported_extensions: Option<HashSet<String>>,
    ) -> Result<Self> {
        let db_exists = db_path.exists();
        let idx_exists = index_path.exists();
        let store = Arc::new(DocumentStore::open(db_path).await?);

        let turbo = if idx_exists {
            if !db_exists {
                bail!(
                    "RAG index file exists at {:?} but database is missing at {:?}",
                    index_path,
                    db_path
                );
            }
            TurboIndex::load(index_path).context("failed to load turbovec index")?
        } else {
            if db_exists && store.chunk_count().await? > 0 {
                bail!(
                    "RAG database has rows but index file is missing at {:?}",
                    index_path
                );
            }
            TurboIndex::new(embedding_dim, bit_width)
                .map_err(|e| anyhow::anyhow!("failed to create turbovec index: {e:?}"))?
        };

        let store_count = store.chunk_count().await? as usize;
        if store_count != turbo.len() {
            bail!(
                "RAG store/index out of sync: SQLite has {} chunks, turbovec has {}. \
                 Delete both files and retry.",
                store_count,
                turbo.len()
            );
        }

        Ok(Self::new(
            store,
            Arc::new(RwLock::new(turbo)),
            supported_extensions,
        ))
    }

    /// Persist the turbovec index to disk. The SQLite store is durable
    /// per write — only the index needs explicit persistence.
    pub async fn save(&self, index_path: &Path) -> Result<()> {
        let turbo = self.turbo.read().await;
        turbo
            .save(index_path)
            .context("failed to write turbovec index")?;
        Ok(())
    }

    /// Number of chunks currently persisted (per SQLite).
    pub async fn chunk_count(&self) -> Result<i64> {
        self.store.chunk_count().await
    }

    /// List unique source names currently persisted in the store.
    ///
    /// Filenames only — full canonical paths are not stored (see
    /// [`RagSourceRegistry::hydrate_from_store`] for the lossy hydration
    /// note).
    pub(crate) async fn list_sources(&self) -> Result<Vec<String>> {
        self.store.list_sources().await
    }

    /// Access the underlying store (for advanced use).
    pub fn store(&self) -> &Arc<DocumentStore> {
        &self.store
    }

    /// Access the underlying turbo index (for advanced use).
    pub fn turbo(&self) -> &Arc<RwLock<TurboIndex>> {
        &self.turbo
    }

    /// Build a [`TurboVectorIndex`] view that can be passed to
    /// `Agent::dynamic_context`. The view shares the same underlying
    /// `Arc`s as this pipeline, so updates via `add_source` / `remove_source`
    /// are visible immediately to live agents.
    pub fn build(&self, embedder: Arc<dyn ErasedEmbedder>) -> TurboVectorIndex {
        TurboVectorIndex::new(Arc::clone(&self.turbo), Arc::clone(&self.store), embedder)
    }
}
