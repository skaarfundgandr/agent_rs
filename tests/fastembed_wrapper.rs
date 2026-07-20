#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "rag")]

use agent_rs::agent::embeddings::{EmbeddingService, FastembedModel};

#[tokio::test]
// Downloads the BGESmallENV15 model from Hugging Face on first run (~50MB); not CI-safe.
// Use a pre-populated FASTEMBED_CACHE_DIR or run with --include-ignored.
#[ignore]
async fn fastembed_wrapper_loads_and_embeds_bge_small_en_v15() -> anyhow::Result<()> {
    let service = EmbeddingService::from_fastembed(FastembedModel::BGESmallENV15)?;
    assert_eq!(service.ndims(), 384);
    let embedding = service.embed_text("hello fastembed").await?;
    assert_eq!(embedding.vec.len(), 384);
    Ok(())
}
