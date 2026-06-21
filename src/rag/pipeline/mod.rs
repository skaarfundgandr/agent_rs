//! [`RagPipeline`] — the orchestrator that ties together [`crate::rag::store::DocumentStore`],
//! [`crate::rag::index::TurboIndex`], and the loaders/splitters/embedders from the other
//! `rag` submodules.
//!
//! Use [`RagPipeline::open_or_create`] at startup to load existing artifacts
//! (or initialise empty ones), then [`RagPipeline::add_source`] /
//! [`RagPipeline::remove_source`] to manage indexed content at runtime.
//!
//! Implementation is split across sibling modules grouped by concern:
//!
//! - `state` — struct definition + private constructor
//! - `lifecycle` — `from_parts`, `open_or_create`, `save`, accessors, `build`
//! - `sync` — shared "embed + write to store + turbovec" helper
//! - `walker` — directory walking for `add_directory`
//! - `ingest` — `add_source`, `add_directory`, `remove_source`
//! - `staging` — lower-level builder API (`add_chunks`, `commit_pending`)

#![cfg(feature = "rag")]

mod ingest;
mod lifecycle;
mod staging;
mod state;
mod sync;
mod walker;

pub use state::RagPipeline;
