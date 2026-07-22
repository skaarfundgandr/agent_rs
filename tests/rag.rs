#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

#[path = "common/mod.rs"]
mod common;

use agent_rs::agent::embeddings::EmbeddingService;
use agent_rs::rag::{DocumentLoader, RagPipeline, TextLoader, TextSplitter, WordSplitter};
use common::rag_pipeline;
use rig_core::embeddings::{Embedding, EmbeddingModel};
use std::fs;
use std::result::Result as StdResult;

/// Deterministic 8-dim mock embedder. Output is `[len, 1, 0, 0, 0, 0, 0, 0]`
/// where len is the input string length. Picked dim=8 because turbovec requires
/// dim to be a positive multiple of 8.
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

// ---------- Existing-style tests (loaders + splitter) — unchanged conceptually ----------

#[tokio::test]
async fn test_text_loader_and_splitter() {
    let temp_file = tempfile::Builder::new().suffix(".md").tempfile().unwrap();
    let test_file = temp_file.path();
    fs::write(test_file, "This is a test file for the RAG text loader.").unwrap();

    let loader = TextLoader::new();
    let doc = loader.load(test_file).await.unwrap();

    assert_eq!(doc.content, "This is a test file for the RAG text loader.");
    let source_name = test_file.file_name().unwrap().to_str().unwrap();
    assert_eq!(doc.metadata.get("source").unwrap(), source_name);
    assert_eq!(doc.metadata.get("file_type").unwrap(), "md");

    let splitter = WordSplitter::new(5, 1);
    let chunks = splitter.split(&doc);
    assert!(chunks.len() >= 2);
    assert_eq!(chunks[0].text, "This is a test file");
    assert_eq!(chunks[0].metadata.source, source_name);
    assert_eq!(chunks[0].metadata.chunk_index, 0);
}

// ---------- New tests against the turbovec/SQLite pipeline ----------

#[tokio::test]
async fn pipeline_add_source_and_search() {
    use rig_core::vector_store::VectorStoreIndex;
    use rig_core::vector_store::request::VectorSearchRequest;

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("doc.txt");
    fs::write(
        &file,
        "Rust is a systems language. It is safe and fast and Rust is popular.",
    )
    .unwrap();

    let rag = rag_pipeline(&dir).await;
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = rag
        .indexer
        .pipeline()
        .add_source(&file, &embedder)
        .await
        .unwrap();
    assert!(added > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added);

    let req = VectorSearchRequest::builder()
        .query("Rust")
        .samples(2)
        .build();
    let hits = rag.vector_index.top_n::<String>(req).await.unwrap();
    assert!(!hits.is_empty());
    let (_score, _id, doc) = &hits[0];
    assert!(doc.contains("[source: doc.txt"));
}

#[tokio::test]
async fn pipeline_remove_source_clears_store_and_index() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("removable.txt");
    fs::write(&file, "alpha beta gamma delta epsilon zeta eta theta").unwrap();

    let rag = rag_pipeline(&dir).await;
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    rag.indexer
        .pipeline()
        .add_source(&file, &embedder)
        .await
        .unwrap();
    assert!(rag.indexer.chunk_count().await.unwrap() > 0);

    let removed = rag
        .indexer
        .pipeline()
        .remove_source("removable.txt")
        .await
        .unwrap();
    assert!(removed > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap(), 0);
    assert_eq!(rag.indexer.pipeline().turbo().read().await.len(), 0);
}

#[tokio::test]
async fn pipeline_save_and_reopen_preserves_chunks() {
    use rig_core::vector_store::VectorStoreIndex;
    use rig_core::vector_store::request::VectorSearchRequest;

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("persist.txt");
    fs::write(&file, "one two three four five six seven eight nine ten").unwrap();

    // First session: add + save.
    {
        let rag = rag_pipeline(&dir).await;
        let embedder = EmbeddingService::new(MockEmbeddingModel);
        rag.indexer
            .pipeline()
            .add_source(&file, &embedder)
            .await
            .unwrap();
        rag.indexer
            .pipeline()
            .save(&dir.path().join("rag.tvim"))
            .await
            .unwrap();
        assert!(dir.path().join("rag.db").exists());
        assert!(dir.path().join("rag.tvim").exists());
    }

    // Second session: reopen and verify.
    let rag2 = rag_pipeline(&dir).await;
    assert!(rag2.indexer.chunk_count().await.unwrap() > 0);

    let req = VectorSearchRequest::builder()
        .query("one")
        .samples(1)
        .build();
    let hits = rag2.vector_index.top_n::<String>(req).await.unwrap();
    assert!(!hits.is_empty());
}

#[tokio::test]
async fn pipeline_open_or_create_rejects_mismatched_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");

    // Create index only with garbage data, then try to build — should error.
    fs::write(&idx, "garbage").unwrap();
    let err = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .db_path(&db)
        .index_path(&idx)
        .build()
        .await;
    assert!(err.is_err());
}

#[tokio::test]
async fn pipeline_commit_pending_persists_staged_chunks() {
    use agent_rs::rag::Document;
    use std::collections::HashMap;

    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "manual.txt".to_string());
    metadata.insert("file_type".to_string(), "txt".to_string());
    let doc = Document {
        content: "manually staged chunk text for unit testing".to_string(),
        metadata,
    };

    let mut rag = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .db_path(&db)
        .index_path(&idx)
        .build()
        .await
        .unwrap();
    let pipeline = rag.indexer.pipeline_mut().expect("unique indexer in test");
    let splitter = WordSplitter::new(4, 1);
    pipeline.add_document(&doc, &splitter);
    assert!(pipeline.pending_count() > 0);

    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let committed = pipeline.commit_pending(&embedder).await.unwrap();
    assert!(committed > 0);
    assert_eq!(pipeline.pending_count(), 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, committed);
}

#[tokio::test]
async fn pipeline_add_source_walks_directory() {
    let dir = tempfile::tempdir().unwrap();

    // Create a subdirectory with multiple text files.
    let sub = dir.path().join("docs");
    fs::create_dir(&sub).unwrap();
    fs::write(
        sub.join("a.txt"),
        "alpha bravo charlie delta echo foxtrot golf",
    )
    .unwrap();
    fs::write(
        sub.join("b.md"),
        "hotel india juliet kilo lima mike november",
    )
    .unwrap();
    fs::write(
        sub.join("c.txt"),
        "oscar papa quebec romeo sierra tango uniform victor",
    )
    .unwrap();

    let rag = rag_pipeline(&dir).await;
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = rag
        .indexer
        .pipeline()
        .add_source(&sub, &embedder)
        .await
        .unwrap();

    // All three files should have been indexed.
    assert!(added > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn pipeline_add_source_dir_skips_unsupported_extensions() {
    let dir = tempfile::tempdir().unwrap();

    let sub = dir.path().join("mixed");
    fs::create_dir(&sub).unwrap();
    fs::write(
        sub.join("good.txt"),
        "one two three four five six seven eight",
    )
    .unwrap();
    fs::write(sub.join("skip.csv"), "col1,col2,col3").unwrap();
    fs::write(sub.join("also_skip.json"), r#"{"key": "value"}"#).unwrap();

    let rag = rag_pipeline(&dir).await;
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = rag
        .indexer
        .pipeline()
        .add_source(&sub, &embedder)
        .await
        .unwrap();

    // Only the .txt file should have been indexed.
    assert!(added > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn pipeline_add_source_dir_respects_custom_extensions() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");

    let sub = dir.path().join("custom");
    fs::create_dir(&sub).unwrap();
    fs::write(sub.join("a.txt"), "alpha bravo charlie delta echo foxtrot").unwrap();
    fs::write(sub.join("b.rs"), "fn main() { println!(\"hello\"); }").unwrap();

    // Only index .rs files.
    let rag = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .db_path(&db)
        .index_path(&idx)
        .extensions(["rs"])
        .build()
        .await
        .unwrap();
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = rag
        .indexer
        .pipeline()
        .add_source(&sub, &embedder)
        .await
        .unwrap();

    // Only the .rs file should have been indexed.
    assert!(added > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn pipeline_add_source_dir_skips_hidden_files() {
    let dir = tempfile::tempdir().unwrap();

    let sub = dir.path().join("hidden");
    fs::create_dir(&sub).unwrap();
    fs::write(
        sub.join("visible.txt"),
        "one two three four five six seven eight",
    )
    .unwrap();
    fs::write(sub.join(".hidden.txt"), "secret hidden content here").unwrap();

    let rag = rag_pipeline(&dir).await;
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = rag
        .indexer
        .pipeline()
        .add_source(&sub, &embedder)
        .await
        .unwrap();

    // Only the visible file should have been indexed.
    assert!(added > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn pipeline_add_source_dir_errors_on_empty_directory() {
    let dir = tempfile::tempdir().unwrap();

    let sub = dir.path().join("empty");
    fs::create_dir(&sub).unwrap();

    let rag = rag_pipeline(&dir).await;
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let err = rag.indexer.pipeline().add_source(&sub, &embedder).await;

    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("no supported files"));
}

#[tokio::test]
async fn pipeline_add_source_dir_nested_walks_recursively() {
    let dir = tempfile::tempdir().unwrap();

    // Create nested directory structure.
    let sub = dir.path().join("nested");
    let deep = sub.join("deep");
    fs::create_dir_all(&deep).unwrap();
    fs::write(
        sub.join("top.txt"),
        "alpha bravo charlie delta echo foxtrot",
    )
    .unwrap();
    fs::write(
        deep.join("bottom.md"),
        "hotel india juliet kilo lima mike november",
    )
    .unwrap();

    let rag = rag_pipeline(&dir).await;
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = rag
        .indexer
        .pipeline()
        .add_source(&sub, &embedder)
        .await
        .unwrap();

    // Both files from different nesting levels should be indexed.
    assert!(added > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn test_top_n_ids_honors_threshold() {
    use rig_core::vector_store::VectorStoreIndex;
    use rig_core::vector_store::request::VectorSearchRequest;

    let dir = tempfile::tempdir().unwrap();

    // Write 3 files with different content lengths so their embeddings differ.
    let short = dir.path().join("short.txt");
    fs::write(&short, "Hello world.").unwrap();
    let medium = dir.path().join("medium.txt");
    fs::write(&medium, "Rust is a systems programming language that focuses on safety, speed, and concurrency without a garbage collector.").unwrap();
    let long = dir.path().join("long.txt");
    fs::write(
        &long,
        "This is a longer document with more content. "
            .repeat(50)
            .as_str(),
    )
    .unwrap();

    let rag = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .store_at(dir.path())
        .build()
        .await
        .unwrap();

    rag.indexer.add(&short).await.unwrap();
    rag.indexer.add(&medium).await.unwrap();
    rag.indexer.add(&long).await.unwrap();

    // Without a threshold — should return some hits.
    let base_req = VectorSearchRequest::builder()
        .query("test query text")
        .samples(10)
        .build();
    let base = rag.vector_index.top_n_ids(base_req).await.unwrap();
    assert!(!base.is_empty(), "expected hits without threshold");

    // With a threshold above max score — should be empty.
    let impossible_req = VectorSearchRequest::builder()
        .query("test query text")
        .samples(10)
        .threshold(f64::MAX)
        .build();
    let impossible = rag.vector_index.top_n_ids(impossible_req).await.unwrap();
    assert!(
        impossible.is_empty(),
        "expected no hits with threshold f64::MAX"
    );

    // With a threshold that keeps only the top hit — every score tracked >= threshold.
    let base_max = base
        .iter()
        .map(|(s, _)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let cut_req = VectorSearchRequest::builder()
        .query("test query text")
        .samples(10)
        .threshold(base_max)
        .build();
    let cut = rag.vector_index.top_n_ids(cut_req).await.unwrap();
    assert!(
        cut.len() <= base.len(),
        "threshold filter should not increase result count"
    );
    for (score, _id) in &cut {
        assert!(
            *score >= base_max,
            "expected score >= {base_max}, got {score}"
        );
    }
    assert_eq!(
        cut.len(),
        1,
        "threshold = max score should keep exactly 1 result"
    );
}
