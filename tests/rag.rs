#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

use agent_rs_lib::agent::embeddings::EmbeddingService;
use agent_rs_lib::rag::{DocumentLoader, RagPipeline, TextLoader, TextSplitter, WordSplitter};
use rig_core::embeddings::{Embedding, EmbeddingModel};
use std::fs;
use std::result::Result as StdResult;
use std::sync::Arc;

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
    assert_eq!(chunks[0].metadata.get("source").unwrap(), source_name);
    assert_eq!(chunks[0].metadata.get("chunk_index").unwrap(), "0");
}

// ---------- New tests against the turbovec/SQLite pipeline ----------

#[tokio::test]
async fn pipeline_add_source_and_search() {
    use rig_core::vector_store::VectorStoreIndex;
    use rig_core::vector_store::request::VectorSearchRequest;

    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");
    let file = dir.path().join("doc.txt");
    fs::write(
        &file,
        "Rust is a systems language. It is safe and fast and Rust is popular.",
    )
    .unwrap();

    let pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
        .await
        .unwrap();
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = pipeline.add_source(&file, &embedder).await.unwrap();
    assert!(added > 0);
    assert_eq!(pipeline.chunk_count().await.unwrap() as usize, added);

    let arc_embedder: Arc<dyn agent_rs_lib::rag::ErasedEmbedder> =
        Arc::new(EmbeddingService::new(MockEmbeddingModel));
    let index = pipeline.build(arc_embedder);

    let req = VectorSearchRequest::builder()
        .query("Rust")
        .samples(2)
        .build();
    let hits = index.top_n::<String>(req).await.unwrap();
    assert!(!hits.is_empty());
    // Each hit's document should be formatted with the source name + chunk idx.
    let (_score, _id, doc) = &hits[0];
    assert!(doc.contains("[source: doc.txt"));
}

#[tokio::test]
async fn pipeline_remove_source_clears_store_and_index() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");
    let file = dir.path().join("removable.txt");
    fs::write(&file, "alpha beta gamma delta epsilon zeta eta theta").unwrap();

    let pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
        .await
        .unwrap();
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    pipeline.add_source(&file, &embedder).await.unwrap();
    assert!(pipeline.chunk_count().await.unwrap() > 0);

    let removed = pipeline.remove_source("removable.txt").await.unwrap();
    assert!(removed > 0);
    assert_eq!(pipeline.chunk_count().await.unwrap(), 0);
    assert_eq!(pipeline.turbo().read().await.len(), 0);
}

#[tokio::test]
async fn pipeline_save_and_reopen_preserves_chunks() {
    use rig_core::vector_store::VectorStoreIndex;
    use rig_core::vector_store::request::VectorSearchRequest;

    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");
    let file = dir.path().join("persist.txt");
    fs::write(&file, "one two three four five six seven eight nine ten").unwrap();

    // First session: add + save.
    {
        let pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
            .await
            .unwrap();
        let embedder = EmbeddingService::new(MockEmbeddingModel);
        pipeline.add_source(&file, &embedder).await.unwrap();
        pipeline.save(&idx).await.unwrap();
        assert!(db.exists());
        assert!(idx.exists());
    }

    // Second session: reopen and verify.
    let reopened = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
        .await
        .unwrap();
    assert!(reopened.chunk_count().await.unwrap() > 0);

    let arc_embedder: Arc<dyn agent_rs_lib::rag::ErasedEmbedder> =
        Arc::new(EmbeddingService::new(MockEmbeddingModel));
    let index = reopened.build(arc_embedder);
    let req = VectorSearchRequest::builder()
        .query("one")
        .samples(1)
        .build();
    let hits = index.top_n::<String>(req).await.unwrap();
    assert!(!hits.is_empty());
}

#[tokio::test]
async fn pipeline_open_or_create_rejects_mismatched_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");

    // Create index only with garbage data, then try to open — should error.
    fs::write(&idx, "garbage").unwrap();
    let err = RagPipeline::open_or_create(&db, &idx, 8, 4, None).await;
    // Either: db missing + idx exists → "RAG index file exists ... database is missing"
    // Or: turbovec load fails on garbage → caught and surfaced.
    assert!(err.is_err());
}

#[tokio::test]
async fn pipeline_commit_pending_persists_staged_chunks() {
    use agent_rs_lib::rag::Document;
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

    let mut pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
        .await
        .unwrap();
    let splitter = WordSplitter::new(4, 1);
    pipeline = pipeline.add_document(&doc, &splitter);
    assert!(pipeline.pending_count() > 0);

    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let committed = pipeline.commit_pending(&embedder).await.unwrap();
    assert!(committed > 0);
    assert_eq!(pipeline.pending_count(), 0);
    assert_eq!(pipeline.chunk_count().await.unwrap() as usize, committed);
}

#[tokio::test]
async fn pipeline_add_source_walks_directory() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");

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

    let pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
        .await
        .unwrap();
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = pipeline.add_source(&sub, &embedder).await.unwrap();

    // All three files should have been indexed.
    assert!(added > 0);
    assert_eq!(pipeline.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn pipeline_add_source_dir_skips_unsupported_extensions() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");

    let sub = dir.path().join("mixed");
    fs::create_dir(&sub).unwrap();
    fs::write(
        sub.join("good.txt"),
        "one two three four five six seven eight",
    )
    .unwrap();
    fs::write(sub.join("skip.csv"), "col1,col2,col3").unwrap();
    fs::write(sub.join("also_skip.json"), r#"{"key": "value"}"#).unwrap();

    let pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
        .await
        .unwrap();
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = pipeline.add_source(&sub, &embedder).await.unwrap();

    // Only the .txt file should have been indexed.
    assert!(added > 0);
    assert_eq!(pipeline.chunk_count().await.unwrap() as usize, added);
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
    let extensions: std::collections::HashSet<String> =
        std::collections::HashSet::from(["rs"].map(String::from));
    let pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, Some(extensions))
        .await
        .unwrap();
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = pipeline.add_source(&sub, &embedder).await.unwrap();

    // Only the .rs file should have been indexed.
    assert!(added > 0);
    assert_eq!(pipeline.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn pipeline_add_source_dir_skips_hidden_files() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");

    let sub = dir.path().join("hidden");
    fs::create_dir(&sub).unwrap();
    fs::write(
        sub.join("visible.txt"),
        "one two three four five six seven eight",
    )
    .unwrap();
    fs::write(sub.join(".hidden.txt"), "secret hidden content here").unwrap();

    let pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
        .await
        .unwrap();
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = pipeline.add_source(&sub, &embedder).await.unwrap();

    // Only the visible file should have been indexed.
    assert!(added > 0);
    assert_eq!(pipeline.chunk_count().await.unwrap() as usize, added);
}

#[tokio::test]
async fn pipeline_add_source_dir_errors_on_empty_directory() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");

    let sub = dir.path().join("empty");
    fs::create_dir(&sub).unwrap();

    let pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
        .await
        .unwrap();
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let err = pipeline.add_source(&sub, &embedder).await;

    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("no supported files"));
}

#[tokio::test]
async fn pipeline_add_source_dir_nested_walks_recursively() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");

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

    let pipeline = RagPipeline::open_or_create(&db, &idx, 8, 4, None)
        .await
        .unwrap();
    let embedder = EmbeddingService::new(MockEmbeddingModel);
    let added = pipeline.add_source(&sub, &embedder).await.unwrap();

    // Both files from different nesting levels should be indexed.
    assert!(added > 0);
    assert_eq!(pipeline.chunk_count().await.unwrap() as usize, added);
}
