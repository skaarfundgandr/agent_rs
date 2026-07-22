#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

#[path = "common/mod.rs"]
mod common;

use std::fs;

#[tokio::test]
async fn same_filename_in_different_directories_does_not_collide() {
    use rig_core::vector_store::VectorStoreIndex;
    use rig_core::vector_store::request::VectorSearchRequest;

    let dir = tempfile::tempdir().unwrap();

    let a_dir = dir.path().join("a");
    let b_dir = dir.path().join("b");
    fs::create_dir(&a_dir).unwrap();
    fs::create_dir(&b_dir).unwrap();

    let a_path = a_dir.join("notes.txt");
    let b_path = b_dir.join("notes.txt");

    fs::write(
        &a_path,
        "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu",
    )
    .unwrap();
    fs::write(
        &b_path,
        "Apple banana cherry date elderberry fig grape honeydew kiwi lemon mango nectarine orange papaya quince raspberry strawberry tangerine ugli vanilla watermelon",
    )
    .unwrap();

    let rag = common::rag_pipeline(&dir).await;

    let added_a = rag.indexer.add(&a_path).await.unwrap();
    let added_b = rag.indexer.add(&b_path).await.unwrap();
    assert!(added_a > 0);
    assert!(added_b > 0);

    // Regression: before the fix, adding a second file with the same
    // filename would wipe the first file's chunks because the key was
    // bare filename rather than canonical path.
    assert_eq!(
        rag.indexer.chunk_count().await.unwrap() as usize,
        added_a + added_b
    );

    // Remove a — only a's chunks should be removed.
    let removed = rag.indexer.remove(&a_path).await.unwrap();
    assert_eq!(removed, added_a);
    assert_eq!(rag.indexer.chunk_count().await.unwrap() as usize, added_b);

    // Registry lists only b.
    let sources = rag.indexer.list();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].path, b_path);

    // Search for b's content — hits should contain b's text and not a's.
    let req = VectorSearchRequest::builder()
        .query("banana cherry date")
        .samples(10)
        .build();
    let hits = rag.vector_index.top_n::<String>(req).await.unwrap();
    assert!(!hits.is_empty(), "expected search hits after removing a");
    let any_a = hits.iter().any(|(_, _, doc)| doc.contains("bravo charlie"));
    let any_b = hits.iter().any(|(_, _, doc)| doc.contains("banana cherry"));
    assert!(
        any_b,
        "expected hits containing b's text after a was removed"
    );
    assert!(
        !any_a,
        "expected no hits containing a's text after a was removed"
    );
}

#[tokio::test]
async fn directory_remove_deletes_by_canonical_paths() {
    let dir = tempfile::tempdir().unwrap();

    let sub = dir.path().join("mydir");
    fs::create_dir(&sub).unwrap();
    fs::write(
        sub.join("alpha.txt"),
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen",
    )
    .unwrap();
    fs::write(
        sub.join("beta.txt"),
        "sixteen seventeen eighteen nineteen twenty twentyone twentytwo twentythree twentyfour twentyfive",
    )
    .unwrap();

    let rag = common::rag_pipeline(&dir).await;

    rag.indexer.add(&sub).await.unwrap();
    assert!(rag.indexer.chunk_count().await.unwrap() > 0);
    assert!(!rag.indexer.list().is_empty());

    let removed = rag.indexer.remove(&sub).await.unwrap();
    assert!(removed > 0);
    assert_eq!(rag.indexer.chunk_count().await.unwrap(), 0);
    assert!(rag.indexer.list().is_empty());
}
