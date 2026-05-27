use std::collections::HashMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Document {
    pub content: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

