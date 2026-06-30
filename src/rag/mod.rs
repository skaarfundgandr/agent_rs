#![cfg(feature = "rag")]

//! RAG subsystem: document loading, chunking, embedding, vector indexing, and
//! SQLite-backed metadata storage. Exposed at the crate root as
//! `agent_rs::rag`.
//!
//! All code in this module is compiled only with the `rag` Cargo feature
//! (the lib default has no RAG code at all).

pub mod alias;
pub mod embedder;
pub mod index;
pub mod loader;
pub mod pipeline;
pub mod splitter;
pub mod store;
pub mod vector_index;

pub use crate::domain::rag::{Chunk, Document, RagSource, RagSourceType};
pub use alias::{QueryFuture, TextsFuture};
pub use embedder::ErasedEmbedder;
pub use index::TurboIndex;
pub use loader::{DocumentLoader, PdfLoader, TextLoader};
pub use pipeline::{BuiltRag, RagIndexer, RagPipeline, RagPipelineBuilder};
pub use splitter::{TextSplitter, WordSplitter};
pub use store::{DocumentStore, RagChunkRow};
pub use vector_index::TurboVectorIndex;
