//! Bridge between turbovec and rig's `VectorStoreIndex` trait.

use crate::rag::{DocumentStore, ErasedEmbedder, RagChunkRow, TurboIndex};
use rig_core::vector_store::request::{Filter, VectorSearchRequest};
use rig_core::vector_store::{VectorStoreError, VectorStoreIndex};
use rig_core::wasm_compat::WasmCompatSend;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Bridge between [`TurboIndex`] (vector search) and [`DocumentStore`]
/// (chunk metadata), exposed via rig's [`VectorStoreIndex`] trait so it
/// can be wired into `Agent::dynamic_context`.
pub struct TurboVectorIndex {
    turbo: Arc<RwLock<TurboIndex>>,
    store: Arc<DocumentStore>,
    embedder: Arc<dyn ErasedEmbedder>,
}

impl TurboVectorIndex {
    /// Construct a new index. The `embedder`'s `ndims()` must match the
    /// turbovec index's `dim()` — validated lazily on first search.
    pub fn new(
        turbo: Arc<RwLock<TurboIndex>>,
        store: Arc<DocumentStore>,
        embedder: Arc<dyn ErasedEmbedder>,
    ) -> Self {
        Self {
            turbo,
            store,
            embedder,
        }
    }
}

impl VectorStoreIndex for TurboVectorIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> serde::Deserialize<'a> + WasmCompatSend,
    {
        // 1) Get (score, id_str) tuples.
        let id_hits = self.top_n_ids(req).await?;
        if id_hits.is_empty() {
            return Ok(Vec::new());
        }

        // 2) Parse id strings to i64 for the SQLite fetch.
        let ids_i64: Vec<i64> = id_hits
            .iter()
            .map(|(_, id_str)| {
                id_str
                    .parse::<i64>()
                    .map_err(|e| VectorStoreError::DatastoreError(Box::new(e)))
            })
            .collect::<Result<_, _>>()?;

        // 3) Batch-fetch chunk rows.
        let rows = self.store.get_chunks_by_ids(&ids_i64).await.map_err(|e| {
            VectorStoreError::DatastoreError(Box::new(std::io::Error::other(e.to_string())))
        })?;

        // 4) Build a id→content map, re-emit results in original hit order.
        let mut by_id: std::collections::HashMap<i64, RagChunkRow> =
            rows.into_iter().map(|r| (r.id, r)).collect();

        let mut out = Vec::with_capacity(id_hits.len());
        for (score, id_str) in id_hits {
            let id_i64: i64 = id_str.parse().map_err(|e: std::num::ParseIntError| {
                VectorStoreError::DatastoreError(Box::new(e))
            })?;
            let Some(row) = by_id.remove(&id_i64) else {
                continue; // row missing from store (stale index); skip
            };
            // Format the same way the old InMemoryVectorStore did:
            // "[source: ... | chunk: N]\n<content>"
            let formatted = format!(
                "[source: {} | chunk: {}]\n{}",
                row.source, row.chunk_index, row.content
            );
            let value = serde_json::Value::String(formatted);
            let typed: T = serde_json::from_value(value).map_err(VectorStoreError::JsonError)?;
            out.push((score, id_str, typed));
        }
        Ok(out)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        let k = req.samples() as usize;
        if k == 0 {
            return Ok(Vec::new());
        }

        // Embed the query.
        let query_vec = self.embedder.embed_query(req.query()).await.map_err(|e| {
            VectorStoreError::DatastoreError(Box::new(std::io::Error::other(e.to_string())))
        })?;

        let turbo = self.turbo.read().await;
        if turbo.is_empty() {
            return Ok(Vec::new());
        }
        let (scores, ids) = turbo.search(&query_vec, k);
        // Threshold is a minimum over the reported score — turbovec's quantized
        // inner-product estimate, which equals cosine similarity for normalized
        // embeddings. Filter only when requested; no implicit default.
        let hits = scores.into_iter().zip(ids);
        let out: Vec<(f64, String)> = match req.threshold() {
            Some(t) => hits
                .filter(|(s, _)| f64::from(*s) >= t)
                .map(|(s, id)| (f64::from(s), id.to_string()))
                .collect(),
            None => hits.map(|(s, id)| (f64::from(s), id.to_string())).collect(),
        };
        Ok(out)
    }
}
