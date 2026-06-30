#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

use agent_rs::agent::embeddings::EmbeddingService;
use agent_rs::rag::RagPipeline;
use rig_core::embeddings::{Embedding, EmbeddingModel};
use std::fs;
use std::result::Result as StdResult;

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

#[tokio::test]
async fn remove_after_restart_preserves_canonical_paths() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("rag.db");
    let idx = dir.path().join("rag.tvim");
    let file = dir.path().join("doc.txt");
    fs::write(&file, "one two three four five six seven eight nine ten").unwrap();

    // First session: add and save.
    {
        let rag = RagPipeline::builder()
            .embedder(EmbeddingService::new(MockEmbeddingModel))
            .db_path(&db)
            .index_path(&idx)
            .build()
            .await
            .unwrap();
        rag.indexer.add(file.as_path()).await.unwrap();
        rag.indexer.pipeline().save(&idx).await.unwrap();
    }

    // Second session: try to remove by relative path.
    let rag2 = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .db_path(&db)
        .index_path(&idx)
        .build()
        .await
        .unwrap();

    // After restart the registry should still contain the canonical path.
    let before = rag2.indexer.list();
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].path, file);

    // Re-adding the same file should be a no-op.
    let re_added = rag2.indexer.add(file.as_path()).await.unwrap();
    assert_eq!(re_added, 0);
    assert_eq!(rag2.indexer.list().len(), 1);

    // Removing by the original path should succeed and clear chunks.
    let removed = rag2.indexer.remove(file.as_path()).await.unwrap();
    assert!(removed > 0);
    assert!(rag2.indexer.list().is_empty());
    assert_eq!(rag2.indexer.chunk_count().await.unwrap(), 0);
}
