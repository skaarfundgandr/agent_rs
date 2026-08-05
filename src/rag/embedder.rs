//! Object-safe embedder adapter.
//!
//! [`TurboVectorIndex`](crate::rag::TurboVectorIndex) is a non-generic struct
//! that needs to embed query strings without being parameterised over the
//! underlying embedding model. This trait is the bridge: any
//! `EmbeddingService<M>` automatically implements it via the blanket impl in
//! `src/agent/embeddings.rs`.

use crate::rag::alias::{QueryFuture, TextsFuture};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Object-safe embedding adapter implemented by every
/// [`EmbeddingService<M>`](crate::agent::embeddings::EmbeddingService).
///
/// Lets non-generic consumers (e.g.
/// [`TurboVectorIndex`](crate::rag::TurboVectorIndex)) embed query strings
/// without being parameterised over the concrete embedding model.
pub trait ErasedEmbedder: WasmCompatSend + WasmCompatSync {
    /// Embed a single query string. Returns the embedding as `f32` vectors
    /// (cast from rig's native `f64` embeddings for turbovec compatibility).
    fn embed_query<'a>(&'a self, text: &'a str) -> QueryFuture<'a>;

    /// Embed multiple texts in one call. Implementations should batch
    /// according to the underlying provider's `MAX_DOCUMENTS` limit.
    fn embed_texts<'a>(&'a self, texts: Vec<String>) -> TextsFuture<'a>;

    /// Dimensionality of the embedding vectors this embedder produces.
    fn ndims(&self) -> usize;
}
