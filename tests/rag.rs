use agent_rs_lib::agent::embeddings::EmbeddingService;
use agent_rs_lib::agent::rag::{
    DocumentLoader, RagPipeline, TextLoader, TextSplitter, WordSplitter,
};
use rig::embeddings::{Embedding, EmbeddingModel};
use std::fs;
use std::result::Result as StdResult;

#[derive(Clone)]
struct MockEmbeddingModel;

impl EmbeddingModel for MockEmbeddingModel {
    const MAX_DOCUMENTS: usize = 2;

    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        3
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> StdResult<Vec<Embedding>, rig::embeddings::EmbeddingError> {
        Ok(texts
            .into_iter()
            .map(|text| Embedding {
                document: text.clone(),
                vec: vec![text.len() as f64, 1.0, 0.0],
            })
            .collect())
    }
}

#[test]
fn test_text_loader_and_splitter() {
    let temp_file = tempfile::Builder::new().suffix(".md").tempfile().unwrap();
    let test_file = temp_file.path();
    fs::write(test_file, "This is a test file for the RAG text loader.").unwrap();

    let loader = TextLoader::new();
    let doc = loader.load(test_file).unwrap();

    assert_eq!(doc.content, "This is a test file for the RAG text loader.");
    let source_name = test_file.file_name().unwrap().to_str().unwrap();
    assert_eq!(doc.metadata.get("source").unwrap(), source_name);
    assert_eq!(doc.metadata.get("file_type").unwrap(), "md");

    // Test WordSplitter
    let splitter = WordSplitter::new(5, 1);
    let chunks = splitter.split(&doc);

    // "This is a test file for the RAG text loader." has 10 words.
    // Chunk 1 (5 words): "This is a test file"
    // Chunk 2 (5 words): "file for the RAG text"
    // Chunk 3 (5 words): "text loader."
    assert!(chunks.len() >= 2);
    assert_eq!(chunks[0].text, "This is a test file");
    assert_eq!(chunks[0].metadata.get("source").unwrap(), source_name);
    assert_eq!(chunks[0].metadata.get("chunk_index").unwrap(), "0");
}

#[tokio::test]
async fn test_rag_pipeline_building() {
    let temp_file = tempfile::Builder::new().suffix(".txt").tempfile().unwrap();
    let test_file = temp_file.path();
    fs::write(
        test_file,
        "Rust is a systems programming language focusing on safety and speed.",
    )
    .unwrap();

    let loader = TextLoader::new();
    let doc = loader.load(test_file).unwrap();
    let splitter = WordSplitter::new(4, 1);

    let embedding_service = EmbeddingService::new(MockEmbeddingModel);
    let index = RagPipeline::new()
        .add_document(&doc, &splitter)
        .build_index(&embedding_service)
        .await
        .expect("Should build RAG index successfully");

    // The index implements rig::vector_store::VectorStoreIndex
    // Let's do a top_n search to check compatibility
    use rig::vector_store::{VectorStoreIndex, request::VectorSearchRequest};
    let req = VectorSearchRequest::builder()
        .query("Rust programming")
        .samples(1)
        .build();

    let results = index
        .top_n::<String>(req)
        .await
        .expect("Search should succeed");
    assert!(!results.is_empty());

    // First result should have the correct score and content
    let (_score, _id, document) = &results[0];
    let source_name = test_file.file_name().unwrap().to_str().unwrap();
    assert!(document.contains(source_name));
}

#[tokio::test]
async fn test_rag_pipeline_custom_formatter() {
    use agent_rs_lib::agent::rag::Document;

    let doc = Document {
        content: "Hello from custom formatter test".to_string(),
        metadata: std::collections::HashMap::new(),
    };
    let splitter = WordSplitter::new(5, 1);

    let embedding_service = EmbeddingService::new(MockEmbeddingModel);
    let store = RagPipeline::new()
        .add_document(&doc, &splitter)
        .build_store_with_formatter(&embedding_service, |chunk| {
            format!("CUSTOM FORMAT: {}", chunk.text)
        })
        .await
        .expect("Should build store with custom formatter");

    let index = store.index(MockEmbeddingModel);
    use rig::vector_store::{VectorStoreIndex, request::VectorSearchRequest};
    let req = VectorSearchRequest::builder()
        .query("custom")
        .samples(1)
        .build();
    let results = index.top_n::<String>(req).await.unwrap();
    assert!(!results.is_empty());
    let (_score, _id, document) = &results[0];
    assert_eq!(document, "CUSTOM FORMAT: Hello from custom formatter test");
}
