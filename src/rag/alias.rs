//! Boxed-future type aliases used by [`crate::rag::ErasedEmbedder`].

use std::future::Future;
use std::pin::Pin;

pub type QueryFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<f32>, anyhow::Error>> + Send + 'a>>;
pub type TextsFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, anyhow::Error>> + Send + 'a>>;
