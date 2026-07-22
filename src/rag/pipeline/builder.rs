//! Public builder API for constructing a [`RagPipeline`].

use super::state::RagPipeline;
use super::walker;
use crate::agent::embeddings::EmbeddingService;
use crate::agent::permission::PermissionPolicy;
use crate::agent::tools::{ManageRagTool, RagSourceRegistry};
use crate::domain::rag::{ChunkingOptions, RagSource, RagSourceType};
use crate::rag::{ErasedEmbedder, TurboVectorIndex};
use crate::security::SharedSandbox;
use anyhow::{Result, anyhow};
use rig_core::embeddings::EmbeddingModel;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Builder for [`RagPipeline`] with a fluent API.
///
/// Call [`RagPipeline::builder()`] to create one, chain setters, then
/// call [`.build()`](RagPipelineBuilder::build) to produce a [`BuiltRag`].
pub struct RagPipelineBuilder {
    embedder: Option<Arc<dyn ErasedEmbedder>>,
    db_path: Option<PathBuf>,
    index_path: Option<PathBuf>,
    store_at: Option<PathBuf>,
    extensions: Option<HashSet<String>>,
    chunking: ChunkingOptions,
    bit_width: usize,
    sandbox: Option<Arc<SharedSandbox>>,
}

impl RagPipelineBuilder {
    pub(super) fn new() -> Self {
        Self {
            embedder: None,
            db_path: None,
            index_path: None,
            store_at: None,
            extensions: None,
            chunking: ChunkingOptions::default(),
            bit_width: 4,
            sandbox: None,
        }
    }

    /// Set the embedding service. The generic `M` is erased internally;
    /// callers pass `EmbeddingService<M>` without seeing `dyn ErasedEmbedder`.
    pub fn embedder<M>(mut self, service: EmbeddingService<M>) -> Self
    where
        M: EmbeddingModel + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.embedder = Some(Arc::new(service));
        self
    }

    /// Set the directory for on-disk artifacts. Derives `<dir>/rag.db` and
    /// `<dir>/rag.tvim`. Explicit `db_path`/`index_path` setters override
    /// the corresponding derived path.
    pub fn store_at(mut self, dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref().to_path_buf();
        if self.db_path.is_none() {
            self.db_path = Some(dir.join("rag.db"));
        }
        if self.index_path.is_none() {
            self.index_path = Some(dir.join("rag.tvim"));
        }
        self.store_at = Some(dir);
        self
    }

    /// Set the SQLite database path explicitly. Overrides the `db_path`
    /// component of [`.store_at()`](Self::store_at).
    pub fn db_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.db_path = Some(path.into());
        self
    }

    /// Set the turbovec index path explicitly. Overrides the `index_path`
    /// component of [`.store_at()`](Self::store_at).
    pub fn index_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.index_path = Some(path.into());
        self
    }

    /// Set the file extensions to index (without the dot, e.g. `"txt"`).
    /// Defaults to `["txt", "md", "pdf"]`.
    pub fn extensions(mut self, exts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.extensions = Some(exts.into_iter().map(Into::into).collect());
        self
    }

    /// Maximum number of words per chunk (default: 220).
    pub fn chunk_words(mut self, n: usize) -> Self {
        self.chunking.chunk_words = n;
        self
    }

    /// Number of words to overlap between consecutive chunks (default: 40).
    pub fn chunk_overlap_words(mut self, n: usize) -> Self {
        self.chunking.chunk_overlap_words = n;
        self
    }

    /// Turbovec quantization bit width (default: 4).
    pub fn bit_width(mut self, n: usize) -> Self {
        self.bit_width = n;
        self
    }

    /// Set the sandbox for path validation. Defaults to CWD.
    pub fn sandbox(mut self, s: impl Into<Arc<SharedSandbox>>) -> Self {
        self.sandbox = Some(s.into());
        self
    }

    /// Consume the builder and produce a [`BuiltRag`].
    ///
    /// # Errors
    ///
    /// Returns an error if no embedder was set, or if the underlying
    /// store/index cannot be opened.
    pub async fn build(self) -> Result<BuiltRag> {
        let embedder = self
            .embedder
            .ok_or_else(|| anyhow!("RagPipelineBuilder::embedder() is required"))?;

        let dim = embedder.ndims();

        let exts = self
            .extensions
            .unwrap_or_else(|| ["txt", "md", "pdf"].into_iter().map(String::from).collect());

        let db = self.db_path.unwrap_or_else(|| "rag_data/rag.db".into());
        let idx = self
            .index_path
            .unwrap_or_else(|| "rag_data/rag.tvim".into());

        let mut pipeline =
            RagPipeline::open_or_create(&db, &idx, dim, self.bit_width, Some(exts.clone())).await?;
        pipeline.chunking = self.chunking;

        let vector_index = pipeline.build(Arc::clone(&embedder));

        let registry = RagSourceRegistry::hydrate_from_store(&pipeline, exts).await?;

        let indexer = RagIndexer {
            pipeline: Arc::new(pipeline),
            embedder,
            registry: Arc::new(Mutex::new(registry)),
            sandbox: self
                .sandbox
                .unwrap_or_else(|| Arc::new(SharedSandbox::default())),
        };

        Ok(BuiltRag {
            vector_index,
            indexer,
        })
    }
}

/// The output of [`RagPipelineBuilder::build`].
pub struct BuiltRag {
    /// Plug into `agent.dynamic_context(top_k, rag.vector_index)`.
    pub vector_index: TurboVectorIndex,
    /// Ingestion handle — owns pipeline + embedder + registry.
    pub indexer: RagIndexer,
}

/// Ingestion handle and tool factory. Owns the pipeline, embedder, source
/// registry, and sandbox.
#[derive(Clone)]
pub struct RagIndexer {
    pub(crate) pipeline: Arc<RagPipeline>,
    pub(crate) embedder: Arc<dyn ErasedEmbedder>,
    pub(crate) registry: Arc<Mutex<RagSourceRegistry>>,
    pub(crate) sandbox: Arc<SharedSandbox>,
}

impl RagIndexer {
    /// Register and index a file or directory.
    ///
    /// Resolves the path via the sandbox, validates the extension via the
    /// registry, then embeds and persists the chunks. Returns the number
    /// of chunks added. Re-adding an existing source returns `Ok(0)` without
    /// re-embedding.
    pub async fn add(&self, path: &Path) -> Result<usize> {
        let canonical = self.sandbox.resolve_path_unchecked(path);

        // Fast-path: if the canonical path is already registered, there is
        // nothing to do. This keeps re-adds cheap and avoids re-embedding.
        {
            let registry = self
                .registry
                .lock()
                .map_err(|e| anyhow!("registry mutex poisoned: {e}"))?;
            if registry.sources().iter().any(|s| s.path == canonical) {
                return Ok(0);
            }
        }

        let source_type = if canonical.is_file() {
            RagSourceType::File
        } else if canonical.is_dir() {
            RagSourceType::Directory
        } else {
            return Err(anyhow!("Path does not exist: {}", canonical.display()));
        };

        let added = self
            .pipeline
            .add_source_dyn(&canonical, self.embedder.as_ref())
            .await?;

        self.pipeline
            .register_source(&canonical, source_type)
            .await?;

        {
            let mut registry = self
                .registry
                .lock()
                .map_err(|e| anyhow!("registry mutex poisoned: {e}"))?;
            registry
                .add_source(path, &self.sandbox)
                .map_err(|e| anyhow!("{e}"))?;
        }

        Ok(added)
    }

    /// Remove a source and all its chunks.
    ///
    /// Resolves the canonical path via the sandbox, deletes the persisted
    /// chunks, then removes the registry entry. Returns the number of
    /// chunks removed.
    pub async fn remove(&self, path: &Path) -> Result<usize> {
        let canonical = self.sandbox.resolve_path_unchecked(path);

        // Verify the source is registered before mutating persistence.
        {
            let registry = self
                .registry
                .lock()
                .map_err(|e| anyhow!("registry mutex poisoned: {e}"))?;
            if !registry.sources().iter().any(|s| s.path == canonical) {
                return Err(anyhow!("Source not found: {}", canonical.display()));
            }
        }

        let removed = if canonical.is_file() {
            self.pipeline
                .remove_source(&canonical.display().to_string())
                .await?
        } else if canonical.is_dir() {
            let extensions = self.pipeline.effective_extensions();
            let files = tokio::task::spawn_blocking({
                let dir = canonical.clone();
                move || walker::walk_indexable(&dir, &extensions)
            })
            .await
            .map_err(|e| anyhow!("directory walk task failed: {e}"))??;

            let mut total = 0usize;
            for file in files {
                let file_canonical = file.canonicalize().unwrap_or_else(|_| file.clone());
                total += self
                    .pipeline
                    .remove_source(&file_canonical.display().to_string())
                    .await?;
            }
            total
        } else {
            return Err(anyhow!("Path does not exist: {}", canonical.display()));
        };

        self.pipeline.unregister_source(&canonical).await?;

        {
            let mut registry = self
                .registry
                .lock()
                .map_err(|e| anyhow!("registry mutex poisoned: {e}"))?;
            registry
                .remove_source(&canonical)
                .map_err(|e| anyhow!("{e}"))?;
        }

        Ok(removed)
    }

    /// List all registered sources.
    pub fn list(&self) -> Vec<RagSource> {
        self.registry
            .lock()
            .map(|g| g.sources().to_vec())
            .unwrap_or_default()
    }

    /// Returns `true` if no sources are registered.
    pub fn is_empty(&self) -> bool {
        self.registry.lock().map(|g| g.is_empty()).unwrap_or(true)
    }

    /// Number of chunks currently persisted in the pipeline.
    pub async fn chunk_count(&self) -> Result<i64> {
        self.pipeline.chunk_count().await
    }

    /// Access the underlying pipeline (staging API escape hatch).
    pub fn pipeline(&self) -> &Arc<RagPipeline> {
        &self.pipeline
    }

    /// Mutable access to the pipeline (for staging API use in tests).
    /// Returns `None` if the pipeline Arc is shared (e.g., another clone
    /// of this indexer exists).
    pub fn pipeline_mut(&mut self) -> Option<&mut RagPipeline> {
        Arc::get_mut(&mut self.pipeline)
    }

    /// Access the sandbox (for `ManageRagTool` permission gating).
    pub(crate) fn sandbox(&self) -> &Arc<SharedSandbox> {
        &self.sandbox
    }

    /// Create a [`ManageRagTool`] that delegates to this indexer.
    pub fn tool(&self, policy: PermissionPolicy) -> ManageRagTool {
        ManageRagTool::new(self.clone(), policy)
    }
}
