#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

use agent_rs::agent::embeddings::EmbeddingService;
use agent_rs::agent::permission::PermissionPolicy;
use agent_rs::agent::tools::ManageRagTool;
use agent_rs::rag::RagPipeline;
use agent_rs::security::{SandboxConfig, SharedSandbox};
use rig_core::embeddings::{Embedding, EmbeddingModel};
use rig_core::tool::Tool;
use std::fs;
use std::result::Result as StdResult;
use std::sync::Arc;

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

async fn build_tool(sandbox_root: &std::path::Path) -> (ManageRagTool, Arc<RagPipeline>) {
    let rag = RagPipeline::builder()
        .embedder(EmbeddingService::new(MockEmbeddingModel))
        .store_at(sandbox_root)
        .extensions(["txt", "md"])
        .sandbox(Arc::new(SharedSandbox::from(
            SandboxConfig::single(sandbox_root).unwrap(),
        )))
        .build()
        .await
        .unwrap();
    let pipeline = rag.indexer.pipeline().clone();
    let tool = rag.indexer.tool(PermissionPolicy::AllowAll);
    (tool, pipeline)
}

#[tokio::test]
async fn manage_rag_add_persists_to_pipeline() {
    use agent_rs::agent::tools::rag::ManageRagArgs;

    let tmp = tempfile::tempdir().unwrap();

    let file = tmp.path().join("a.txt");
    fs::write(
        &file,
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
    )
    .unwrap();

    let (tool, pipeline) = build_tool(tmp.path()).await;

    let result = tool
        .call(ManageRagArgs {
            action: "add".to_string(),
            path: Some("a.txt".to_string()),
            force: None,
        })
        .await
        .unwrap();
    assert!(result.contains("indexed"));

    assert!(pipeline.chunk_count().await.unwrap() > 0);
    assert!(!pipeline.turbo().read().await.is_empty());
}

#[tokio::test]
async fn manage_rag_list_includes_added_source() {
    use agent_rs::agent::tools::rag::ManageRagArgs;

    let tmp = tempfile::tempdir().unwrap();

    let file = tmp.path().join("listed.txt");
    fs::write(&file, "one two three four five six seven eight").unwrap();

    let (tool, _pipeline) = build_tool(tmp.path()).await;

    tool.call(ManageRagArgs {
        action: "add".to_string(),
        path: Some("listed.txt".to_string()),
        force: None,
    })
    .await
    .unwrap();

    let listed = tool
        .call(ManageRagArgs {
            action: "list".to_string(),
            path: None,
            force: None,
        })
        .await
        .unwrap();
    assert!(listed.contains("listed.txt"));
}

#[tokio::test]
async fn manage_rag_remove_clears_pipeline() {
    use agent_rs::agent::tools::rag::ManageRagArgs;

    let tmp = tempfile::tempdir().unwrap();

    let file = tmp.path().join("rm.txt");
    fs::write(&file, "alpha beta gamma delta epsilon zeta eta theta iota").unwrap();

    let (tool, pipeline) = build_tool(tmp.path()).await;

    tool.call(ManageRagArgs {
        action: "add".to_string(),
        path: Some("rm.txt".to_string()),
        force: None,
    })
    .await
    .unwrap();
    assert!(pipeline.chunk_count().await.unwrap() > 0);

    let removed = tool
        .call(ManageRagArgs {
            action: "remove".to_string(),
            path: Some("rm.txt".to_string()),
            force: None,
        })
        .await
        .unwrap();
    assert!(removed.contains("removed"));

    assert_eq!(pipeline.chunk_count().await.unwrap(), 0);
    assert_eq!(pipeline.turbo().read().await.len(), 0);
}

#[tokio::test]
async fn manage_rag_add_directory_persists_to_pipeline() {
    use agent_rs::agent::tools::rag::ManageRagArgs;

    let tmp = tempfile::tempdir().unwrap();

    let sub = tmp.path().join("docs");
    fs::create_dir(&sub).unwrap();
    fs::write(
        sub.join("a.txt"),
        "alpha bravo charlie delta echo foxtrot golf",
    )
    .unwrap();
    fs::write(
        sub.join("b.txt"),
        "hotel india juliet kilo lima mike november oscar",
    )
    .unwrap();

    let (tool, pipeline) = build_tool(tmp.path()).await;

    let result = tool
        .call(ManageRagArgs {
            action: "add".to_string(),
            path: Some("docs".to_string()),
            force: None,
        })
        .await
        .unwrap();
    assert!(result.contains("indexed"));

    assert!(pipeline.chunk_count().await.unwrap() > 0);
    assert!(!pipeline.turbo().read().await.is_empty());
}

#[tokio::test]
async fn manage_rag_add_force_reindexes() {
    use agent_rs::agent::tools::rag::ManageRagArgs;

    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("f.txt");
    fs::write(&file, "one two three four five six seven eight nine ten").unwrap();
    let (tool, pipeline) = build_tool(tmp.path()).await;

    let result = tool
        .call(ManageRagArgs {
            action: "add".to_string(),
            path: Some("f.txt".to_string()),
            force: None,
        })
        .await
        .unwrap();
    assert!(result.contains("indexed"));
    let first_chunks = pipeline.chunk_count().await.unwrap();
    assert!(first_chunks > 0);

    let long_content: String = (0..300)
        .map(|i| format!("word_{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(&file, &long_content).unwrap();

    let noop = tool
        .call(ManageRagArgs {
            action: "add".to_string(),
            path: Some("f.txt".to_string()),
            force: None,
        })
        .await
        .unwrap();
    assert!(noop.contains("indexed 0 chunks"));
    assert_eq!(pipeline.chunk_count().await.unwrap(), first_chunks);

    let forced = tool
        .call(ManageRagArgs {
            action: "add".to_string(),
            path: Some("f.txt".to_string()),
            force: Some(true),
        })
        .await
        .unwrap();
    assert!(forced.contains("indexed"));
    let forced_chunks: usize = forced
        .strip_prefix("indexed ")
        .and_then(|s| s.strip_suffix(" chunks"))
        .and_then(|s| s.parse().ok())
        .unwrap();
    assert!(forced_chunks > 0, "forced reindex should produce chunks");
    assert_eq!(
        pipeline.chunk_count().await.unwrap(),
        forced_chunks as i64,
        "chunk count should match forced reindex result"
    );
}
