//! Public builder API for constructing a [`RagPipeline`].

use super::state::RagPipeline;
use crate::agent::embeddings::EmbeddingService;
use crate::agent::tools::RagSourceRegistry;
use crate::domain::rag::ChunkingOptions;
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
    sandbox: Option<SharedSandbox>,
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
    pub fn sandbox(mut self, s: impl Into<SharedSandbox>) -> Self {
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
            RagPipeline::open_or_create(&db, &idx, dim, self.bit_width, Some(exts.clone()))
                .await?;
        pipeline.chunking = self.chunking;

        let vector_index = pipeline.build(Arc::clone(&embedder));

        let indexer = RagIndexer {
            pipeline: Arc::new(pipeline),
            embedder,
            registry: Arc::new(Mutex::new(RagSourceRegistry::new(exts))),
            sandbox: Arc::new(self.sandbox.unwrap_or_default()),
        };

        Ok(BuiltRag { vector_index, indexer })
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
/// registry, and sandbox. Methods for `add`/`remove`/`list`/`tool` are
/// added in Phase 2.
#[derive(Clone)]
pub struct RagIndexer {
    pub(crate) pipeline: Arc<RagPipeline>,
    pub(crate) embedder: Arc<dyn ErasedEmbedder>,
    pub(crate) registry: Arc<Mutex<RagSourceRegistry>>,
    pub(crate) sandbox: Arc<SharedSandbox>,
}
