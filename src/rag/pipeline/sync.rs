//! Shared "embed chunks and persist into store + turbovec" path used by both
//! `ingest::add_single_file` and `staging::commit_pending_dyn`.

use crate::rag::{Chunk, DocumentStore, ErasedEmbedder, TurboIndex};
use anyhow::{Context, Result, bail};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Insert `chunks` into the SQLite store, embed them, validate the embedding
/// dimension against the turbovec index, and push the vectors into the index.
///
/// On dimension mismatch, the just-inserted rows are removed from the store
/// (best-effort) before returning the error, so a retry starts from a clean
/// slate.
///
/// Returns the number of chunks persisted.
pub(crate) async fn persist_chunks(
    store: &DocumentStore,
    turbo: &Arc<RwLock<TurboIndex>>,
    chunks: &[Chunk],
    source: &str,
    file_type: &str,
    embedder: &dyn ErasedEmbedder,
) -> Result<usize> {
    if chunks.is_empty() {
        return Ok(0);
    }

    let ids = store.insert_chunks(chunks, source, file_type).await?;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = embedder.embed_texts(texts).await?;

    let expected_dim = turbo.read().await.dim();
    if !embeddings.is_empty() && embeddings[0].len() != expected_dim {
        let _ = store.delete_by_source(source).await;
        bail!(
            "embedding dimension {} does not match turbovec index dim {}",
            embeddings[0].len(),
            expected_dim
        );
    }

    let mut flat: Vec<f32> = Vec::with_capacity(embeddings.len() * expected_dim);
    for emb in &embeddings {
        flat.extend_from_slice(emb);
    }
    let u64_ids: Vec<u64> = ids.iter().map(|i| *i as u64).collect();
    {
        let mut turbo = turbo.write().await;
        turbo.add(&flat, &u64_ids).context("turbovec add failed")?;
    }

    Ok(chunks.len())
}
