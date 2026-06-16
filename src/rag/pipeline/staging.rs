//! Lower-level chunk-staging builder API, kept for backwards compat with
//! unit tests. Staged chunks are embedded and persisted only when
//! [`RagPipeline::commit_pending`] is called.

use super::state::RagPipeline;
use super::sync;
use crate::agent::embeddings::EmbeddingService;
use crate::rag::{Chunk, Document, ErasedEmbedder, TextSplitter};
use anyhow::Result;
use rig_core::embeddings::EmbeddingModel;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::collections::BTreeMap;

impl RagPipeline {
    /// Stage a collection of chunks. Does NOT persist — use
    /// [`Self::add_source`](super::super::RagPipeline::add_source) for the
    /// high-level path. Chunks staged this way are embedded and persisted
    /// only when [`Self::commit_pending`] is called.
    pub fn add_chunks(mut self, chunks: Vec<Chunk>) -> Self {
        self.pending.extend(chunks);
        self
    }

    /// Stage a document by splitting via the given splitter.
    pub fn add_document<S: TextSplitter>(mut self, document: &Document, splitter: &S) -> Self {
        self.pending.extend(splitter.split(document));
        self
    }

    /// Stage multiple documents by splitting via the given splitter.
    pub fn add_documents<S: TextSplitter>(mut self, documents: &[Document], splitter: &S) -> Self {
        for doc in documents {
            self.pending.extend(splitter.split(doc));
        }
        self
    }

    /// Embed and persist all staged chunks into the store + index.
    /// Returns the number of chunks committed and clears the pending buffer.
    /// Uses the chunk's `source` and `file_type` metadata (defaulting to
    /// `"unknown"`/`"txt"`).
    pub async fn commit_pending<M>(&mut self, embedder: &EmbeddingService<M>) -> Result<usize>
    where
        M: EmbeddingModel + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.commit_pending_dyn(embedder).await
    }

    /// Erased-embedder variant of [`Self::commit_pending`]. Groups staged
    /// chunks by `(source, file_type)` and persists each group via the
    /// shared `sync::persist_chunks` helper.
    pub async fn commit_pending_dyn(&mut self, embedder: &dyn ErasedEmbedder) -> Result<usize> {
        if self.pending.is_empty() {
            return Ok(0);
        }
        let chunks = std::mem::take(&mut self.pending);

        let mut grouped: BTreeMap<(String, String), Vec<Chunk>> = BTreeMap::new();
        for c in chunks {
            let src = c
                .metadata
                .get("source")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            let ft = c
                .metadata
                .get("file_type")
                .cloned()
                .unwrap_or_else(|| "txt".to_string());
            grouped.entry((src, ft)).or_default().push(c);
        }

        let mut total = 0usize;
        for ((source, file_type), group) in grouped {
            sync::persist_chunks(
                &self.store,
                &self.turbo,
                &group,
                &source,
                &file_type,
                embedder,
            )
            .await?;
            total += group.len();
        }
        Ok(total)
    }

    /// Number of chunks currently staged (not yet persisted).
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}
