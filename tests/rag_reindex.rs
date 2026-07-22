#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

#[path = "common/mod.rs"]
mod common;

use std::fs;

#[tokio::test]
async fn reindex_refreshes_modified_file() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("doc.txt");

    fs::write(&file, "one two three four five six seven eight nine ten").unwrap();

    let rag = common::rag_pipeline(&dir).await;
    let first = rag.indexer.add(&file).await.unwrap();
    assert!(first > 0, "first add should index chunks");

    let long_content: String = (0..300)
        .map(|i| format!("word_{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(&file, &long_content).unwrap();

    let re_add = rag.indexer.add(&file).await.unwrap();
    assert_eq!(re_add, 0, "re-add of registered source should return 0");
    let chunk_count_after_add = rag.indexer.chunk_count().await.unwrap();
    assert_eq!(
        chunk_count_after_add, first as i64,
        "chunk count should still match first add"
    );

    let second = rag.indexer.reindex(&file).await.unwrap();
    assert!(second > 0, "reindex should produce chunks");
    assert_ne!(
        second, first,
        "longer content should produce a different chunk count"
    );
    let chunk_count_after_reindex = rag.indexer.chunk_count().await.unwrap();
    assert_eq!(
        chunk_count_after_reindex, second as i64,
        "chunk count should match reindex result (no duplicate chunks)"
    );

    assert_eq!(
        rag.indexer.list().len(),
        1,
        "only one source should be registered"
    );
}
