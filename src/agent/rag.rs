use anyhow::{bail, Context, Result};
use pdf_extract::extract_text;
use rig::{
    embeddings::EmbeddingModel,
    vector_store::in_memory_store::InMemoryVectorStore,
    OneOrMany,
};
use std::path::Path;
use crate::agent::embeddings::EmbeddingService;
pub use crate::domain::rag::ChunkingOptions;

/// A builder for creating a RAG-enabled vector store.
pub struct RagStoreBuilder<M: EmbeddingModel> {
    embedding_service: EmbeddingService<M>,
    chunking_options: ChunkingOptions,
    documents: Vec<String>,
}

impl<M: EmbeddingModel> RagStoreBuilder<M> {
    /// Create a new builder with the given embedding service.
    pub fn new(embedding_service: EmbeddingService<M>) -> Self {
        Self {
            embedding_service,
            chunking_options: ChunkingOptions::default(),
            documents: Vec::new(),
        }
    }

    /// Set the chunking options for the builder.
    pub fn with_chunking(mut self, options: ChunkingOptions) -> Self {
        self.chunking_options = options;
        self
    }

    /// Load and chunk a PDF file, adding it to the store.
    pub fn add_pdf<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let path = path.as_ref();
        let source = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| path.to_str().unwrap_or("unknown"));

        let pdf_text = extract_pdf_text(path)?;
        let chunks = chunk_text(
            &pdf_text,
            self.chunking_options.chunk_words,
            self.chunking_options.chunk_overlap_words,
        );

        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            if chunk.trim().is_empty() {
                continue;
            }

            self.documents.push(format!(
                "[source: {source} | chunk: {chunk_idx}]\n{chunk}"
            ));
        }

        Ok(self)
    }

    /// Add raw text documents directly.
    pub fn add_documents(mut self, docs: Vec<String>) -> Self {
        self.documents.extend(docs);
        self
    }

    /// Build the vector store by embedding all added documents.
    pub async fn build(self) -> Result<InMemoryVectorStore<String>> {
        if self.documents.is_empty() {
            bail!("no embeddable text was provided or extracted");
        }

        let embeddings = self.embedding_service.embed_texts(self.documents.clone()).await?;
        
        let mut vector_store = InMemoryVectorStore::<String>::default();
        vector_store.add_documents(
            self.documents
                .into_iter()
                .zip(embeddings)
                .map(|(doc, emb)| (doc, OneOrMany::one(emb))),
        );

        Ok(vector_store)
    }

    /// Build the vector store and return an index for the given model.
    pub async fn build_index(self) -> Result<rig::vector_store::in_memory_store::InMemoryVectorIndex<M, String>> 
    where M: Clone {
        let model = self.embedding_service.model().clone();
        let vector_store = self.build().await?;
        Ok(vector_store.index(model))
    }
}

/// Helper function to chunk text into smaller fragments based on word count.
pub fn chunk_text(text: &str, max_words: usize, overlap_words: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return Vec::new();
    }

    let max_words = max_words.max(1);
    let overlap_words = overlap_words.min(max_words.saturating_sub(1));
    let step = max_words - overlap_words;

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + max_words).min(words.len());
        let chunk = words[start..end].join(" ");
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }

        if end == words.len() {
            break;
        }

        start += step;
    }

    chunks
}

/// Extract plain text from a PDF file.
pub fn extract_pdf_text<P: AsRef<Path>>(path: P) -> Result<String> {
    extract_text(path).context("Failed to extract text from PDF")
}
