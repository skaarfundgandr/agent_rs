//! Tests for custom loader registration on RagPipelineBuilder.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

#[path = "common/mod.rs"]
mod common;

use agent_rs::agent::embeddings::EmbeddingService;
use agent_rs::rag::{Document, DocumentLoader, RagPipeline};
use rig_core::embeddings::{Embedding, EmbeddingModel};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::result::Result as StdResult;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
struct MockEmbeddingModel;

impl EmbeddingModel for MockEmbeddingModel {
    const MAX_DOCUMENTS: usize = 8;
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        8
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> StdResult<Vec<Embedding>, rig_core::embeddings::EmbeddingError> {
        Ok(texts
            .into_iter()
            .map(|text| Embedding {
                document: text.clone(),
                vec: vec![text.len() as f64, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            })
            .collect())
    }
}

struct UpperCaseLoader {
    called: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl DocumentLoader for UpperCaseLoader {
    async fn load(&self, path: &Path) -> anyhow::Result<Document> {
        self.called.store(true, Ordering::SeqCst);
        let content = tokio::fs::read_to_string(path).await?.to_uppercase();
        let source_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let file_type = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("txt")
            .to_string();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), source_name);
        metadata.insert("file_type".to_string(), file_type);
        Ok(Document { content, metadata })
    }
}

struct CsvLoader;

#[async_trait::async_trait]
impl DocumentLoader for CsvLoader {
    async fn load(&self, path: &Path) -> anyhow::Result<Document> {
        let content = tokio::fs::read_to_string(path).await?;
        let content = format!("CSV:{content}");
        let source_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), source_name);
        metadata.insert("file_type".to_string(), "csv".to_string());
        Ok(Document { content, metadata })
    }
}

#[tokio::test]
async fn custom_loader_overrides_builtin() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("test.txt");
    fs::write(&file, "hello world").unwrap();

    let called = Arc::new(AtomicBool::new(false));
    let loader = UpperCaseLoader {
        called: Arc::clone(&called),
    };

    let rag = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .store_at(dir.path())
        .loader("txt", Arc::new(loader))
        .build()
        .await
        .unwrap();

    let added = rag.indexer.add(&file).await.unwrap();
    assert!(added > 0);
    assert!(called.load(Ordering::SeqCst));
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn unregistered_extension_falls_back_to_builtin() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("test.txt");
    fs::write(&file, "hello world").unwrap();

    let rag = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .store_at(dir.path())
        .build()
        .await
        .unwrap();

    let added = rag.indexer.add(&file).await.unwrap();
    assert!(added > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn custom_loader_for_novel_extension() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("data.csv");
    fs::write(&file, "col1,col2,col3\n1,2,3").unwrap();

    let rag = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .store_at(dir.path())
        .extensions(["txt", "md", "pdf", "csv"])
        .loader("csv", Arc::new(CsvLoader))
        .build()
        .await
        .unwrap();

    let added = rag.indexer.add(&file).await.unwrap();
    assert!(added > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn loader_extension_is_case_insensitive() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("test.txt");
    fs::write(&file, "hello world").unwrap();

    let called = Arc::new(AtomicBool::new(false));
    let loader = UpperCaseLoader {
        called: Arc::clone(&called),
    };

    let rag = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .store_at(dir.path())
        .loader("TXT", Arc::new(loader))
        .build()
        .await
        .unwrap();

    rag.indexer.add(&file).await.unwrap();
    assert!(called.load(Ordering::SeqCst));
}
