//! Boxed-future type aliases used by [`crate::rag::ErasedEmbedder`].

use std::future::Future;
use std::pin::Pin;

/// Boxed future resolving to a single query embedding (`Vec<f32>`), as used
/// by [`ErasedEmbedder::embed_query`](crate::rag::ErasedEmbedder::embed_query).
pub type QueryFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<f32>, anyhow::Error>> + Send + 'a>>;
/// Boxed future resolving to a batch of text embeddings (`Vec<Vec<f32>>`), as
/// used by [`ErasedEmbedder::embed_texts`](crate::rag::ErasedEmbedder::embed_texts).
pub type TextsFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, anyhow::Error>> + Send + 'a>>;
