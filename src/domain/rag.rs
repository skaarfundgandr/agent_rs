use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Document {
    pub content: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    pub text: String,
    pub metadata: HashMap<String, String>,
}

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
