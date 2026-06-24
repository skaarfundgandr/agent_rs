#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#[cfg(test)]
pub mod agents_tests;
#[cfg(test)]
pub mod embeddings;
#[cfg(test)]
pub mod mcp_client;
#[cfg(test)]
pub mod mcp_registry;
#[cfg(feature = "opentelemetry")]
pub mod observability_tests;
#[cfg(feature = "opentelemetry")]
pub mod react_otel_tests;
#[cfg(test)]
pub mod react_recovery_tests;
#[cfg(test)]
pub mod react_tests;
