//! SQLite-backed chunk metadata store.

use crate::rag::Chunk;
use anyhow::{Context, Result};
use std::path::Path;
use tokio_rusqlite::Connection;

/// A single row in the `rag_chunks` table.
#[derive(Debug, Clone)]
pub struct RagChunkRow {
    pub id: i64,
    pub source: String,
    pub file_type: String,
    pub chunk_index: i64,
    pub content: String,
}

/// SQLite-backed store for RAG chunk metadata.
///
/// Holds the canonical `(source, chunk_index, content)` rows. The matching
/// turbovec vectors live in [`TurboIndex`] and are linked by `id` (the
/// SQLite ROWID, cast to `u64` for turbovec).
pub struct DocumentStore {
    conn: Connection,
}

impl DocumentStore {
    /// Open or create the SQLite database at `path`, creating the schema
    /// if it does not already exist.
    pub async fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path.to_path_buf())
            .await
            .context("failed to open rag SQLite store")?;
        conn.call(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS rag_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_source ON rag_chunks(source);",
            )?;
            Ok(())
        })
        .await
        .context("failed to initialize rag_chunks schema")?;
        Ok(Self { conn })
    }

    /// Insert chunks under the given `source` + `file_type`, returning
    /// the SQLite ROWIDs assigned to each row (same order as input).
    pub async fn insert_chunks(
        &self,
        chunks: &[Chunk],
        source: &str,
        file_type: &str,
    ) -> Result<Vec<i64>> {
        let prepared: Vec<(String, String, i64, String)> = chunks
            .iter()
            .enumerate()
            .map(|(idx, chunk)| {
                (
                    source.to_string(),
                    file_type.to_string(),
                    idx as i64,
                    chunk.text.clone(),
                )
            })
            .collect();

        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;
                let mut ids = Vec::with_capacity(prepared.len());
                {
                    let mut stmt = tx.prepare(
                        "INSERT INTO rag_chunks (source, file_type, chunk_index, content)
                         VALUES (?1, ?2, ?3, ?4)",
                    )?;
                    for (src, ftype, idx, content) in &prepared {
                        stmt.execute(tokio_rusqlite::params![src, ftype, idx, content])?;
                        ids.push(tx.last_insert_rowid());
                    }
                }
                tx.commit()?;
                Ok(ids)
            })
            .await
            .context("failed to insert RAG chunks")
    }

    /// Fetch the rows for a batch of IDs. Order of the returned Vec is
    /// arbitrary — callers that need a specific order must reorder.
    pub async fn get_chunks_by_ids(&self, ids: &[i64]) -> Result<Vec<RagChunkRow>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids = ids.to_vec();
        self.conn
            .call(move |conn| {
                let placeholders = (0..ids.len())
                    .map(|i| format!("?{}", i + 1))
                    .collect::<Vec<_>>()
                    .join(",");
                let sql = format!(
                    "SELECT id, source, file_type, chunk_index, content
                     FROM rag_chunks WHERE id IN ({placeholders})"
                );
                let mut stmt = conn.prepare(&sql)?;
                let params: Vec<&dyn tokio_rusqlite::ToSql> = ids
                    .iter()
                    .map(|i| i as &dyn tokio_rusqlite::ToSql)
                    .collect();
                let rows = stmt
                    .query_map(params.as_slice(), |row| {
                        Ok(RagChunkRow {
                            id: row.get(0)?,
                            source: row.get(1)?,
                            file_type: row.get(2)?,
                            chunk_index: row.get(3)?,
                            content: row.get(4)?,
                        })
                    })?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(rows)
            })
            .await
            .context("failed to fetch RAG chunks by id")
    }

    /// Delete every row whose `source` matches. Returns the IDs that
    /// were deleted so the caller can drop them from the vector index.
    pub async fn delete_by_source(&self, source: &str) -> Result<Vec<i64>> {
        let source = source.to_string();
        self.conn
            .call(move |conn| {
                let tx = conn.transaction()?;
                let ids: Vec<i64> = {
                    let mut stmt = tx.prepare("SELECT id FROM rag_chunks WHERE source = ?1")?;
                    stmt.query_map([&source], |row| row.get::<_, i64>(0))?
                        .collect::<std::result::Result<Vec<_>, _>>()?
                };
                tx.execute("DELETE FROM rag_chunks WHERE source = ?1", [&source])?;
                tx.commit()?;
                Ok(ids)
            })
            .await
            .context("failed to delete RAG chunks by source")
    }

    /// Number of rows in `rag_chunks`.
    pub async fn chunk_count(&self) -> Result<i64> {
        self.conn
            .call(|conn| {
                let count: i64 =
                    conn.query_row("SELECT COUNT(*) FROM rag_chunks", [], |row| row.get(0))?;
                Ok(count)
            })
            .await
            .context("failed to count RAG chunks")
    }

    /// Unique sources currently stored.
    pub async fn list_sources(&self) -> Result<Vec<String>> {
        self.conn
            .call(|conn| {
                let mut stmt =
                    conn.prepare("SELECT DISTINCT source FROM rag_chunks ORDER BY source")?;
                let sources = stmt
                    .query_map([], |row| row.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(sources)
            })
            .await
            .context("failed to list RAG sources")
    }
}
