#![cfg(feature = "rag")]

use agent_rs_lib::agent::embeddings::EmbeddingService;
use rig_core::Embed;
use rig_core::embeddings::embed::{EmbedError, TextEmbedder};
use rig_core::embeddings::{Embedding, EmbeddingModel};
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
    ) -> StdResult<Vec<Embedding>, rig_core::embeddings::EmbeddingError> {
        Ok(texts
            .into_iter()
            .map(|text| Embedding {
                document: text.clone(),
                vec: vec![text.len() as f64, 1.0, 0.0],
            })
            .collect())
    }
}

#[derive(Clone, Debug)]
struct SampleDocument {
    id: String,
    parts: Vec<String>,
}

impl Embed for SampleDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> StdResult<(), EmbedError> {
        for part in &self.parts {
            embedder.embed(part.clone());
        }

        Ok(())
    }
}

#[tokio::test]
async fn embeds_texts_in_original_order() {
    let service = EmbeddingService::new(MockEmbeddingModel);
    let embeddings = service
        .embed_texts(vec!["alpha", "beta", "gamma"])
        .await
        .expect("expected embeddings");

    let documents = embeddings
        .iter()
        .map(|embedding| embedding.document.as_str())
        .collect::<Vec<_>>();

    assert_eq!(documents, vec!["alpha", "beta", "gamma"]);
}

#[tokio::test]
async fn embeds_documents_and_preserves_fragments() {
    let service = EmbeddingService::new(MockEmbeddingModel);
    let docs = vec![
        SampleDocument {
            id: "doc-1".to_string(),
            parts: vec!["first".to_string(), "second".to_string()],
        },
        SampleDocument {
            id: "doc-2".to_string(),
            parts: vec!["third".to_string()],
        },
    ];

    let embeddings = service
        .embed_documents(docs)
        .await
        .expect("expected document embeddings");

    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].0.id, "doc-1");
    assert_eq!(embeddings[1].0.id, "doc-2");
    assert_eq!(embeddings[0].1.len(), 2);
    assert_eq!(embeddings[1].1.len(), 1);
    assert_eq!(embeddings[0].1.first_ref().document, "first");
    assert_eq!(embeddings[0].1.last_ref().document, "second");
}

#[tokio::test]
async fn rejects_empty_text_collections() {
    let service = EmbeddingService::new(MockEmbeddingModel);
    let err = service.embed_texts(Vec::<String>::new()).await;

    assert!(err.is_err());
}
