//! [`RagPipeline`] struct definition and private constructor.
//!
//! All other files in this module contain `impl RagPipeline { ... }` blocks
//! grouped by concern (lifecycle, ingest, staging). They share the fields
//! directly because the visibility is `pub(crate)` within the `pipeline`
//! parent module.

use crate::rag::{Chunk, DocumentStore, TurboIndex};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Pipeline that owns the persisted RAG state.
///
/// Wraps a [`DocumentStore`] (SQLite chunk metadata) and a [`TurboIndex`]
/// (turbovec ANN vectors) and keeps them in sync. Use [`Self::open_or_create`]
/// at startup to load existing artifacts (or initialise empty ones), then
/// [`Self::add_source`] / [`Self::remove_source`] to manage indexed content
/// at runtime. Call [`Self::save`] periodically (or on shutdown) to persist
/// the turbovec index — the SQLite store is durable per-write.
///
/// Two-file persistence: SQLite database (rows of metadata) + turbovec index
/// (.tvim binary). [`Self::open_or_create`] enforces they have a matching
/// chunk count; if not, an error is returned and the caller should clear
/// and rebuild.
///
/// Construction also retains a `Vec<Chunk>` buffer for the lower-level
/// add_chunks / add_document / add_documents builder methods kept for
/// backwards-compatible test ergonomics. Those methods do NOT persist
/// — they only stage chunks for a future `commit_pending` call.
pub struct RagPipeline {
    pub(crate) store: Arc<DocumentStore>,
    pub(crate) turbo: Arc<RwLock<TurboIndex>>,
    /// Staged chunks for the lower-level builder API (kept for test compat).
    /// Never written to SQLite/turbovec automatically.
    pub(crate) pending: Vec<Chunk>,
    /// File extensions to index when walking directories. `None` uses the
    /// built-in default set (see [`crate::rag::loader::DEFAULT_EXTENSIONS`]).
    pub(crate) supported_extensions: Option<HashSet<String>>,
}

impl RagPipeline {
    /// Private constructor used by `from_parts` and `open_or_create`.
    pub(crate) fn new(
        store: Arc<DocumentStore>,
        turbo: Arc<RwLock<TurboIndex>>,
        supported_extensions: Option<HashSet<String>>,
    ) -> Self {
        Self {
            store,
            turbo,
            pending: Vec::new(),
            supported_extensions,
        }
    }
}
