#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

#[path = "common/mod.rs"]
mod common;

use std::fs;

use rig_core::tool::Tool;

#[tokio::test]
async fn search_tool_returns_indexed_content() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("doc.txt");

    fs::write(
        &file,
        "Rust is a systems programming language focused on safety, speed, \
         and concurrency. It empowers everyone to build reliable and efficient \
         software. The Rust community is welcoming and inclusive.",
    )
    .unwrap();

    let rag = common::rag_pipeline(&dir).await;
    rag.indexer.add(&file).await.unwrap();

    let search = rag.indexer.search_tool();
    let args = agent_rs::agent::tools::rag::SearchRagArgs {
        query: "systems language safe concurrent".to_string(),
        samples: None,
        threshold: None,
    };
    let result = search.call(args).await.unwrap();
    assert!(
        result.contains("systems programming language"),
        "expected result to contain indexed content, got: {result}"
    );
    assert!(
        result.contains("hit 1"),
        "expected hit numbering, got: {result}"
    );
}

#[tokio::test]
async fn search_tool_threshold_above_max_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("doc.txt");

    fs::write(
        &file,
        "The quick brown fox jumps over the lazy dog near the bank of the river.",
    )
    .unwrap();

    let rag = common::rag_pipeline(&dir).await;
    rag.indexer.add(&file).await.unwrap();

    let search = rag.indexer.search_tool();
    let args = agent_rs::agent::tools::rag::SearchRagArgs {
        query: "fox dog river".to_string(),
        samples: None,
        threshold: Some(f64::MAX),
    };
    let result = search.call(args).await.unwrap();
    assert!(
        result.starts_with("No results"),
        "expected 'No results' for impossible threshold, got: {result}"
    );
}

#[tokio::test]
async fn search_tool_empty_index_reports_no_results() {
    let dir = tempfile::tempdir().unwrap();

    let rag = common::rag_pipeline(&dir).await;
    let search = rag.indexer.search_tool();
    let args = agent_rs::agent::tools::rag::SearchRagArgs {
        query: "anything".to_string(),
        samples: None,
        threshold: None,
    };
    let result = search.call(args).await.unwrap();
    assert!(
        result.starts_with("No results"),
        "expected 'No results' for empty index, got: {result}"
    );
}
