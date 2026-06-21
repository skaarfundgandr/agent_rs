//! Text splitters: turn a `Document` into `Vec<Chunk>`.

use crate::rag::{Chunk, Document};
use std::collections::HashMap;

/// Trait for splitting a document into chunks.
pub trait TextSplitter {
    /// Splits a document into a collection of chunks.
    fn split(&self, document: &Document) -> Vec<Chunk>;
}

/// Splitter that chunks text by word count with a sliding window.
#[derive(Clone, Debug)]
pub struct WordSplitter {
    chunk_words: usize,
    chunk_overlap_words: usize,
}

impl WordSplitter {
    /// Creates a new `WordSplitter` with the given chunk and overlap sizes.
    ///
    /// # Arguments
    ///
    /// * `chunk_words` - Maximum number of words per chunk (clamped to at least 1).
    /// * `chunk_overlap_words` - Number of words to overlap between consecutive
    ///   chunks (clamped to at most `chunk_words - 1`).
    ///
    /// # Returns
    ///
    /// Returns a new `WordSplitter` instance.
    pub fn new(chunk_words: usize, chunk_overlap_words: usize) -> Self {
        Self {
            chunk_words: chunk_words.max(1),
            chunk_overlap_words: chunk_overlap_words.min(chunk_words.saturating_sub(1)),
        }
    }
}

impl Default for WordSplitter {
    fn default() -> Self {
        Self::new(220, 40)
    }
}

impl TextSplitter for WordSplitter {
    fn split(&self, document: &Document) -> Vec<Chunk> {
        let words: Vec<&str> = document.content.split_whitespace().collect();

        if words.is_empty() {
            return Vec::new();
        }

        let step = self.chunk_words - self.chunk_overlap_words;
        let mut chunks = Vec::new();
        let mut start = 0;

        let source = document
            .metadata
            .get("source")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        let file_type = document
            .metadata
            .get("file_type")
            .cloned()
            .unwrap_or_else(|| "txt".to_string());

        while start < words.len() {
            let end = (start + self.chunk_words).min(words.len());
            let chunk_text = words[start..end].join(" ");

            if !chunk_text.trim().is_empty() {
                let chunk_idx = chunks.len();
                let mut metadata = HashMap::new();
                metadata.insert("source".to_string(), source.clone());
                metadata.insert("file_type".to_string(), file_type.clone());
                metadata.insert("chunk_index".to_string(), chunk_idx.to_string());

                chunks.push(Chunk {
                    text: chunk_text,
                    metadata,
                });
            }

            if end == words.len() {
                break;
            }

            start += step;
        }

        chunks
    }
}
