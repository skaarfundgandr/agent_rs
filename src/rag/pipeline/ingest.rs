//! File and directory ingestion plus source removal.

use super::state::RagPipeline;
use super::{sync, walker};
use crate::agent::embeddings::EmbeddingService;
use crate::domain::rag::{RagSource, RagSourceType};
use crate::rag::ErasedEmbedder;
use crate::rag::loader::{DEFAULT_EXTENSIONS, DocumentLoader, PdfLoader, TextLoader};
use crate::rag::splitter::{TextSplitter, WordSplitter};
use anyhow::{Result, bail};
use rig_core::embeddings::EmbeddingModel;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

impl RagPipeline {
    /// Load a file from disk, chunk it, embed via the given service, and
    /// persist into both store and turbo index. Returns the number of
    /// chunks added.
    ///
    /// File type is selected by extension: `.pdf` uses [`PdfLoader`],
    /// everything else uses [`TextLoader`].
    pub async fn add_source<M>(&self, path: &Path, embedder: &EmbeddingService<M>) -> Result<usize>
    where
        M: EmbeddingModel + WasmCompatSend + WasmCompatSync + 'static,
    {
        // Delegate to the dyn variant via the blanket ErasedEmbedder impl.
        self.add_source_dyn(path, embedder).await
    }

    /// Erased-embedder variant of [`Self::add_source`]. Used by tools and
    /// any code that already holds a `&dyn ErasedEmbedder`.
    pub async fn add_source_dyn(
        &self,
        path: &Path,
        embedder: &dyn ErasedEmbedder,
    ) -> Result<usize> {
        if path.is_dir() {
            return self.add_directory(path, embedder).await;
        }
        self.add_single_file(path, embedder).await
    }

    async fn add_single_file(&self, path: &Path, embedder: &dyn ErasedEmbedder) -> Result<usize> {
        let document = if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("pdf"))
            .unwrap_or(false)
        {
            PdfLoader::new().load(path).await?
        } else {
            TextLoader::new().load(path).await?
        };

        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let source = canonical.display().to_string();
        let file_type = document
            .metadata
            .get("file_type")
            .cloned()
            .unwrap_or_else(|| "txt".to_string());

        // Remove any prior chunks for this source so re-add is idempotent.
        // This is necessary because hydration from the store makes re-adds
        // observable after a restart.
        let _ = self.remove_source(&source).await?;

        let splitter =
            WordSplitter::new(self.chunking.chunk_words, self.chunking.chunk_overlap_words);
        let chunks = splitter.split(&document);
        if chunks.is_empty() {
            bail!("source {:?} produced no chunks", path);
        }

        sync::persist_chunks(
            &self.store,
            &self.turbo,
            &chunks,
            &source,
            &file_type,
            embedder,
        )
        .await
    }

    async fn add_directory(&self, dir: &Path, embedder: &dyn ErasedEmbedder) -> Result<usize> {
        let extensions = self.effective_extensions();
        let dir_owned = dir.to_path_buf();
        let walk_extensions = extensions.clone();
        let files = tokio::task::spawn_blocking(move || {
            walker::walk_indexable(&dir_owned, &walk_extensions)
        })
        .await
        .map_err(|e| anyhow::anyhow!("directory walk task failed: {e}"))??;
        let mut total_chunks = 0usize;
        let mut files_indexed = 0usize;

        for path in files {
            match self.add_single_file(&path, embedder).await {
                Ok(n) => {
                    total_chunks += n;
                    files_indexed += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        "skipping file {} during directory walk: {e}",
                        path.display()
                    );
                }
            }
        }

        if files_indexed == 0 {
            bail!(
                "no supported files found in directory {:?} (extensions: {:?})",
                dir,
                extensions
            );
        }

        Ok(total_chunks)
    }

    pub(crate) fn effective_extensions(&self) -> HashSet<String> {
        self.supported_extensions
            .clone()
            .unwrap_or_else(|| DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect())
    }

    /// Remove every chunk whose `source` matches the given string. The
    /// `source` is the canonical path string of the indexed file as written
    /// by `add_single_file`; the loaders' bare-filename metadata is
    /// no longer used as the chunk key.
    ///
    /// Returns the number of chunks removed.
    pub async fn remove_source(&self, source: &str) -> Result<usize> {
        let ids = self.store.delete_by_source(source).await?;
        if ids.is_empty() {
            return Ok(0);
        }
        let mut turbo = self.turbo.write().await;
        let mut count = 0usize;
        for id in &ids {
            if turbo.remove(*id as u64) {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Persist a registered source (canonical path + type) to the SQLite
    /// `rag_sources` table. This table is used to rebuild the in-memory
    /// [`RagSourceRegistry`] after a restart.
    pub(crate) async fn register_source(
        &self,
        path: &Path,
        source_type: RagSourceType,
    ) -> Result<()> {
        let source_type_str = match source_type {
            RagSourceType::File => "file",
            RagSourceType::Directory => "directory",
        };
        self.store
            .insert_source(&path.display().to_string(), source_type_str)
            .await
    }

    /// Remove a registered source from the SQLite `rag_sources` table.
    pub(crate) async fn unregister_source(&self, path: &Path) -> Result<()> {
        self.store.delete_source(&path.display().to_string()).await
    }

    /// Load all registered sources from the SQLite `rag_sources` table.
    pub(crate) async fn list_registered_sources(&self) -> Result<Vec<RagSource>> {
        let rows = self.store.list_sources_with_types().await?;
        Ok(rows
            .into_iter()
            .map(|(path, source_type)| RagSource {
                path: PathBuf::from(path),
                source_type: match source_type.as_str() {
                    "directory" => RagSourceType::Directory,
                    _ => RagSourceType::File,
                },
            })
            .collect())
    }
}
