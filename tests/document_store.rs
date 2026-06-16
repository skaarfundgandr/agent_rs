#![cfg(feature = "rag")]

use agent_rs_lib::rag::{Chunk, DocumentStore};
use std::collections::HashMap;
use tempfile::tempdir;

#[tokio::test]
async fn document_store_open_insert_fetch_delete_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rag_store.db");
    let store = DocumentStore::open(&path).await.expect("open");

    let chunks = vec![
        Chunk {
            text: "first chunk".to_string(),
            metadata: HashMap::new(),
        },
        Chunk {
            text: "second chunk".to_string(),
            metadata: HashMap::new(),
        },
    ];
    let ids = store
        .insert_chunks(&chunks, "test.txt", "txt")
        .await
        .unwrap();
    assert_eq!(ids.len(), 2);

    let rows = store.get_chunks_by_ids(&ids).await.unwrap();
    assert_eq!(rows.len(), 2);
    let texts: std::collections::HashSet<String> = rows.iter().map(|r| r.content.clone()).collect();
    assert!(texts.contains("first chunk"));
    assert!(texts.contains("second chunk"));

    assert_eq!(store.chunk_count().await.unwrap(), 2);
    assert_eq!(store.list_sources().await.unwrap(), vec!["test.txt"]);

    let deleted = store.delete_by_source("test.txt").await.unwrap();
    assert_eq!(deleted.len(), 2);
    assert_eq!(store.chunk_count().await.unwrap(), 0);
}
