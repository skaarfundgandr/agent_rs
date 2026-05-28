use crate::agent::embeddings::EmbeddingService;
pub use crate::domain::rag::{Chunk, ChunkingOptions, Document};
use anyhow::{Context, Result, bail};
use pdf_extract::extract_text;
use rig::{
    OneOrMany, embeddings::EmbeddingModel, vector_store::in_memory_store::InMemoryVectorStore,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Trait for loading documents from the file system.
pub trait DocumentLoader {
    /// Loads a file and returns a `Document`.
    fn load(&self, path: &Path) -> Result<Document>;
}

/// Loader for PDF documents.
#[derive(Default, Clone, Copy, Debug)]
pub struct PdfLoader;

impl PdfLoader {
    pub fn new() -> Self {
        Self
    }
}

impl DocumentLoader for PdfLoader {
    fn load(&self, path: &Path) -> Result<Document> {
        let text = extract_pdf_text(path)?;
        let source_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), source_name);
        metadata.insert("file_type".to_string(), "pdf".to_string());

        Ok(Document {
            content: text,
            metadata,
        })
    }
}

/// Loader for plain text and Markdown documents.
#[derive(Default, Clone, Copy, Debug)]
pub struct TextLoader;

impl TextLoader {
    pub fn new() -> Self {
        Self
    }
}

impl DocumentLoader for TextLoader {
    fn load(&self, path: &Path) -> Result<Document> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read text file: {:?}", path))?;

        let source_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();

        let file_type = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("txt")
            .to_string();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), source_name);
        metadata.insert("file_type".to_string(), file_type);

        Ok(Document { content, metadata })
    }
}

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

/// Pipeline to collect chunks and build vector index stores.
#[derive(Default, Clone, Debug)]
pub struct RagPipeline {
    chunks: Vec<Chunk>,
}

impl RagPipeline {
    /// Create a new, empty RAG pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a collection of chunks directly to the pipeline.
    pub fn add_chunks(mut self, chunks: Vec<Chunk>) -> Self {
        self.chunks.extend(chunks);
        self
    }

    /// Add a document to the pipeline, using the specified splitter.
    pub fn add_document<S: TextSplitter>(mut self, document: &Document, splitter: &S) -> Self {
        let doc_chunks = splitter.split(document);
        self.chunks.extend(doc_chunks);
        self
    }

    /// Add a collection of documents using the specified splitter.
    pub fn add_documents<S: TextSplitter>(mut self, documents: &[Document], splitter: &S) -> Self {
        for doc in documents {
            let doc_chunks = splitter.split(doc);
            self.chunks.extend(doc_chunks);
        }
        self
    }

    /// Build a Rig InMemoryVectorStore<String> by formatting chunk text + metadata.
    pub async fn build_store<M: EmbeddingModel>(
        &self,
        embedding_service: &EmbeddingService<M>,
    ) -> Result<InMemoryVectorStore<String>> {
        self.build_store_with_formatter(embedding_service, |chunk| {
            let source = chunk
                .metadata
                .get("source")
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let chunk_idx = chunk
                .metadata
                .get("chunk_index")
                .map(|s| s.as_str())
                .unwrap_or("0");
            format!("[source: {source} | chunk: {chunk_idx}]\n{}", chunk.text)
        })
        .await
    }

    /// Build a Rig InMemoryVectorStore<String> by formatting chunk text + metadata using a custom formatter.
    pub async fn build_store_with_formatter<M: EmbeddingModel, F>(
        &self,
        embedding_service: &EmbeddingService<M>,
        formatter: F,
    ) -> Result<InMemoryVectorStore<String>>
    where
        F: Fn(&Chunk) -> String,
    {
        if self.chunks.is_empty() {
            bail!("cannot build store with empty chunks");
        }

        let formatted_docs: Vec<String> = self.chunks.iter().map(formatter).collect();

        let embeddings = embedding_service
            .embed_texts(formatted_docs.clone())
            .await?;

        let mut vector_store = InMemoryVectorStore::<String>::default();
        vector_store.add_documents(
            formatted_docs
                .into_iter()
                .zip(embeddings)
                .map(|(doc, emb)| (doc, OneOrMany::one(emb))),
        );

        Ok(vector_store)
    }

    /// Build a Rig InMemoryVectorIndex<M, String> using the provided embedding service.
    pub async fn build_index<M: EmbeddingModel + Clone>(
        &self,
        embedding_service: &EmbeddingService<M>,
    ) -> Result<rig::vector_store::in_memory_store::InMemoryVectorIndex<M, String>> {
        let model = embedding_service.model().clone();
        let store = self.build_store(embedding_service).await?;
        Ok(store.index(model))
    }

    /// Build a Rig InMemoryVectorIndex<M, String> using the provided embedding service and a custom formatter.
    pub async fn build_index_with_formatter<M: EmbeddingModel + Clone, F>(
        &self,
        embedding_service: &EmbeddingService<M>,
        formatter: F,
    ) -> Result<rig::vector_store::in_memory_store::InMemoryVectorIndex<M, String>>
    where
        F: Fn(&Chunk) -> String,
    {
        let model = embedding_service.model().clone();
        let store = self
            .build_store_with_formatter(embedding_service, formatter)
            .await?;
        Ok(store.index(model))
    }
}

/// A builder for creating a RAG-enabled vector store.
///
/// # Deprecated
/// Prefer using [`PdfLoader`], [`WordSplitter`], and [`RagPipeline`] instead.
#[deprecated(since = "0.2.0", note = "use PdfLoader and RagPipeline instead")]
pub struct RagStoreBuilder<M: EmbeddingModel> {
    embedding_service: EmbeddingService<M>,
    chunking_options: ChunkingOptions,
    documents: Vec<String>,
}

#[allow(deprecated)]
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
        let doc = PdfLoader.load(path.as_ref())?;
        let splitter = WordSplitter::new(
            self.chunking_options.chunk_words,
            self.chunking_options.chunk_overlap_words,
        );
        let chunks = splitter.split(&doc);
        for chunk in chunks {
            let source = chunk
                .metadata
                .get("source")
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let chunk_idx = chunk
                .metadata
                .get("chunk_index")
                .map(|s| s.as_str())
                .unwrap_or("0");
            self.documents.push(format!(
                "[source: {source} | chunk: {chunk_idx}]\n{}",
                chunk.text
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

        let embeddings = self
            .embedding_service
            .embed_texts(self.documents.clone())
            .await?;

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
    pub async fn build_index(
        self,
    ) -> Result<rig::vector_store::in_memory_store::InMemoryVectorIndex<M, String>>
    where
        M: Clone,
    {
        let model = self.embedding_service.model().clone();
        let vector_store = self.build().await?;
        Ok(vector_store.index(model))
    }
}

/// Extract plain text from a PDF file.
pub fn extract_pdf_text<P: AsRef<Path>>(path: P) -> Result<String> {
    extract_text(path).context("Failed to extract text from PDF")
}
