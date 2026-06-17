#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

use agent_rs_lib::agent::embeddings::EmbeddingService;
use agent_rs_lib::agent::permission::PermissionPolicy;
use agent_rs_lib::agent::tools::{ManageRagTool, RagSourceRegistry};
use agent_rs_lib::rag::{ErasedEmbedder, RagPipeline};
use agent_rs_lib::security::{SandboxConfig, SharedSandbox};
use rig_core::embeddings::{Embedding, EmbeddingModel};
use rig_core::tool::Tool;
use std::collections::HashSet;
use std::fs;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};

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

fn build_tool(
    sandbox_root: &std::path::Path,
    pipeline: Arc<RagPipeline>,
) -> (Arc<Mutex<RagSourceRegistry>>, ManageRagTool) {
    let sandbox = Arc::new(SharedSandbox::from(SandboxConfig::single(sandbox_root).unwrap()));
    let registry = Arc::new(Mutex::new(RagSourceRegistry::new(HashSet::from(
        ["txt", "md"].map(String::from),
    ))));
    let embedder: Arc<dyn ErasedEmbedder> = Arc::new(EmbeddingService::new(MockEmbeddingModel));
    let tool = ManageRagTool::new(
        Arc::clone(&registry),
        pipeline,
        embedder,
        Arc::clone(&sandbox),
        PermissionPolicy::AllowAll,
    );
    (registry, tool)
}

#[tokio::test]
async fn manage_rag_add_persists_to_pipeline() {
    use agent_rs_lib::agent::tools::rag::ManageRagArgs;

    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("rag.db");
    let idx = tmp.path().join("rag.tvim");
    let pipeline = Arc::new(
        RagPipeline::open_or_create(&db, &idx, 8, 4, None)
            .await
            .unwrap(),
    );

    let file = tmp.path().join("a.txt");
    fs::write(
        &file,
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
    )
    .unwrap();

    let (_reg, tool) = build_tool(tmp.path(), Arc::clone(&pipeline));

    let result = tool
        .call(ManageRagArgs {
            action: "add".to_string(),
            path: Some("a.txt".to_string()),
        })
        .await
        .unwrap();
    assert!(result.contains("indexed"));

    assert!(pipeline.chunk_count().await.unwrap() > 0);
    assert!(!pipeline.turbo().read().await.is_empty());
}

#[tokio::test]
async fn manage_rag_list_includes_added_source() {
    use agent_rs_lib::agent::tools::rag::ManageRagArgs;

    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("rag.db");
    let idx = tmp.path().join("rag.tvim");
    let pipeline = Arc::new(
        RagPipeline::open_or_create(&db, &idx, 8, 4, None)
            .await
            .unwrap(),
    );

    let file = tmp.path().join("listed.txt");
    fs::write(&file, "one two three four five six seven eight").unwrap();

    let (_reg, tool) = build_tool(tmp.path(), Arc::clone(&pipeline));

    tool.call(ManageRagArgs {
        action: "add".to_string(),
        path: Some("listed.txt".to_string()),
    })
    .await
    .unwrap();

    let listed = tool
        .call(ManageRagArgs {
            action: "list".to_string(),
            path: None,
        })
        .await
        .unwrap();
    assert!(listed.contains("listed.txt"));
}

#[tokio::test]
async fn manage_rag_remove_clears_pipeline() {
    use agent_rs_lib::agent::tools::rag::ManageRagArgs;

    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("rag.db");
    let idx = tmp.path().join("rag.tvim");
    let pipeline = Arc::new(
        RagPipeline::open_or_create(&db, &idx, 8, 4, None)
            .await
            .unwrap(),
    );

    let file = tmp.path().join("rm.txt");
    fs::write(&file, "alpha beta gamma delta epsilon zeta eta theta iota").unwrap();

    let (_reg, tool) = build_tool(tmp.path(), Arc::clone(&pipeline));

    tool.call(ManageRagArgs {
        action: "add".to_string(),
        path: Some("rm.txt".to_string()),
    })
    .await
    .unwrap();
    assert!(pipeline.chunk_count().await.unwrap() > 0);

    let removed = tool
        .call(ManageRagArgs {
            action: "remove".to_string(),
            path: Some("rm.txt".to_string()),
        })
        .await
        .unwrap();
    assert!(removed.contains("removed"));

    assert_eq!(pipeline.chunk_count().await.unwrap(), 0);
    assert_eq!(pipeline.turbo().read().await.len(), 0);
}

#[tokio::test]
async fn manage_rag_add_directory_persists_to_pipeline() {
    use agent_rs_lib::agent::tools::rag::ManageRagArgs;

    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("rag.db");
    let idx = tmp.path().join("rag.tvim");
    let pipeline = Arc::new(
        RagPipeline::open_or_create(&db, &idx, 8, 4, None)
            .await
            .unwrap(),
    );

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

    let (_reg, tool) = build_tool(tmp.path(), Arc::clone(&pipeline));

    let result = tool
        .call(ManageRagArgs {
            action: "add".to_string(),
            path: Some("docs".to_string()),
        })
        .await
        .unwrap();
    assert!(result.contains("indexed"));

    assert!(pipeline.chunk_count().await.unwrap() > 0);
    assert!(!pipeline.turbo().read().await.is_empty());
}
