#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

#[path = "common/mod.rs"]
mod common;

use std::fs;

#[tokio::test]
async fn remove_after_restart_preserves_canonical_paths() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("doc.txt");
    fs::write(&file, "one two three four five six seven eight nine ten").unwrap();

    // First session: add and save.
    {
        let rag = common::rag_pipeline(&dir).await;
        rag.indexer.add(file.as_path()).await.unwrap();
        rag.indexer
            .pipeline()
            .save(&dir.path().join("rag.tvim"))
            .await
            .unwrap();
    }

    // Second session: try to remove by relative path.
    let rag2 = common::rag_pipeline(&dir).await;

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
