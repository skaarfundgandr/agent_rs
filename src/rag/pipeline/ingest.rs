//! File and directory ingestion plus source removal.

use super::state::RagPipeline;
use super::{sync, walker};
use crate::agent::embeddings::EmbeddingService;
use crate::rag::ErasedEmbedder;
use crate::rag::loader::{DEFAULT_EXTENSIONS, DocumentLoader, PdfLoader, TextLoader};
use crate::rag::splitter::{TextSplitter, WordSplitter};
use anyhow::{Result, bail};
use rig_core::embeddings::EmbeddingModel;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::collections::HashSet;
use std::path::Path;

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
            PdfLoader::new().load(path)?
        } else {
            TextLoader::new().load(path)?
        };

        let source = document
            .metadata
            .get("source")
            .cloned()
            .unwrap_or_else(|| path.display().to_string());
        let file_type = document
            .metadata
            .get("file_type")
            .cloned()
            .unwrap_or_else(|| "txt".to_string());

        let splitter = WordSplitter::default();
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
        let mut total_chunks = 0usize;
        let mut files_indexed = 0usize;

        for path in walker::walk_indexable(dir, &extensions)? {
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

    fn effective_extensions(&self) -> HashSet<String> {
        self.supported_extensions
            .clone()
            .unwrap_or_else(|| DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect())
    }

    /// Remove every chunk whose `source` matches the given string. The
    /// `source` is typically the file name component (as stored by the
    /// loaders), not the full canonical path — see [`PdfLoader::load`] /
    /// [`TextLoader::load`] for what gets stored.
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
}
