use agent_rs_lib::agent::embeddings::EmbeddingService;
use agent_rs_lib::agent::rag::{
    DocumentLoader, PdfLoader, RagPipeline, TextLoader, TextSplitter, WordSplitter,
};
use rig::embeddings::{Embedding, EmbeddingModel};
use std::fs;
use std::path::Path;
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
    let test_file = Path::new("tests/temp_test_doc.md");
    fs::write(test_file, "This is a test file for the RAG text loader.").unwrap();

    let loader = TextLoader::new();
    let doc = loader.load(test_file).unwrap();

    assert_eq!(doc.content, "This is a test file for the RAG text loader.");
    assert_eq!(doc.metadata.get("source").unwrap(), "temp_test_doc.md");
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
    assert_eq!(chunks[0].metadata.get("source").unwrap(), "temp_test_doc.md");
    assert_eq!(chunks[0].metadata.get("chunk_index").unwrap(), "0");

    fs::remove_file(test_file).unwrap();
}

#[tokio::test]
async fn test_rag_pipeline_building() {
    let test_file = Path::new("tests/temp_test_doc2.txt");
    fs::write(test_file, "Rust is a systems programming language focusing on safety and speed.").unwrap();

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

    let results = index.top_n::<String>(req).await.expect("Search should succeed");
    assert!(!results.is_empty());
    
    // First result should have the correct score and content
    let (_score, id, document) = &results[0];
    assert!(document.contains("temp_test_doc2.txt"));

    fs::remove_file(test_file).unwrap();
}
