#![cfg(feature = "rag")]

use std::collections::HashMap;
use std::path::PathBuf;

/// A loaded document with its raw text content and associated metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Document {
    /// The full text content of the document.
    pub content: String,
    /// Arbitrary key-value metadata (e.g. source path, file type).
    pub metadata: HashMap<String, String>,
}

/// Typed metadata attached to each [`Chunk`].
///
/// Replaces the former `HashMap<String, String>` to provide compile-time
/// field access for the keys that the pipeline actually stores and queries.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChunkMetadata {
    /// Source filename (e.g. `"doc.txt"`).
    pub source: String,
    /// File type / extension (e.g. `"txt"`, `"pdf"`).
    pub file_type: String,
    /// Zero-based index of this chunk within its source document.
    pub chunk_index: usize,
}

impl Default for ChunkMetadata {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            file_type: "txt".to_string(),
            chunk_index: 0,
        }
    }
}

/// A single chunk of text produced by splitting a [`Document`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    /// The chunk text content.
    pub text: String,
    /// Metadata inherited from the source document plus chunk-specific keys.
    pub metadata: ChunkMetadata,
}

/// Options controlling how a document is split into chunks.
///
/// Defaults: [`chunk_words`](Self::chunk_words) 220,
/// [`chunk_overlap_words`](Self::chunk_overlap_words) 40.
#[derive(Debug, Clone)]
pub struct ChunkingOptions {
    /// Maximum number of words per chunk.
    pub chunk_words: usize,
    /// Number of words to overlap between consecutive chunks.
    pub chunk_overlap_words: usize,
}

impl Default for ChunkingOptions {
    fn default() -> Self {
        Self {
            chunk_words: 220,
            chunk_overlap_words: 40,
        }
    }
}

/// The kind of RAG source (individual file or directory).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RagSourceType {
    File,
    Directory,
}

/// A registered RAG source entry: the path and whether it is a file or directory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RagSource {
    /// Canonical path to the source on disk.
    pub path: PathBuf,
    /// Whether the source is an individual file or a directory.
    pub source_type: RagSourceType,
}
