//! Build the OTLP/HTTP exporter and install the global `tracing` subscriber.
//!
//! # Batch exporter policy
//!
//! Spans are exported via a `BatchSpanProcessor` with a **1-second scheduled
//! delay** (down from the OTel SDK default of 5 s). This ensures shallow
//! research runs show spans within ~1 s of completion while still batching
//! efficiently for deep runs.
//!
//! Override the delay (in milliseconds) with `LANGSMITH_OTEL_BATCH_DELAY_MS`:
//! - `0` – uses a synchronous (simple) exporter, no batching — useful for
//!   local development and debugging.
//! - Any positive value – used as the batch scheduled delay (default: 1000 ms).
//!
//! Programmatic `BatchConfig` construction overrides any `OTEL_BSP_*`
//! environment variables the SDK would otherwise read.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use opentelemetry::global;
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::{WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{
    Resource,
    propagation::TraceContextPropagator,
    trace::{BatchConfigBuilder, BatchSpanProcessor, Sampler, SdkTracerProvider},
};
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt};

use crate::domain::observability::LangSmithConfig;

/// Holds the live OTLP `SdkTracerProvider`; calling [`shutdown`](Self::shutdown)
/// (or dropping) flushes pending spans.
pub struct TracerHandle {
    provider: Arc<SdkTracerProvider>,
}

impl TracerHandle {
    /// Flush and shut down the tracer provider explicitly.
    pub fn shutdown(&self) -> Result<()> {
        self.provider
            .shutdown()
            .context("failed to shut down OTLP tracer provider")
    }
}

impl Drop for TracerHandle {
    fn drop(&mut self) {
        let _ = self.provider.shutdown();
    }
}

/// Build the OTLP/HTTP exporter, install the global `tracing` subscriber,
/// and return a [`TracerHandle`] that owns the tracer provider.
///
/// Spans are batch-exported with a 1-second scheduled delay by default
/// (`LANGSMITH_OTEL_BATCH_DELAY_MS` overrides this; set to `0` for a
/// synchronous exporter).
///
/// Once installed, **rig's existing GenAI spans** (e.g. `invoke_agent`,
/// `chat`, `execute_tool`) export to LangSmith automatically — no call-site
/// changes are needed. The `x-api-key` and `Langsmith-Project` headers are
/// forwarded from [`LangSmithConfig`] via the
/// [`WithHttpConfig::with_headers`] builder method.
///
/// # Errors
/// Returns an error if the OTLP exporter, tracer provider, or global
/// subscriber cannot be built. `set_global_default` will fail if a global
/// subscriber has already been installed (call this exactly once per
/// process).
pub fn init_tracing(cfg: &LangSmithConfig) -> Result<TracerHandle> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let headers = HashMap::from([
        ("x-api-key".to_string(), cfg.api_key.clone()),
        ("Langsmith-Project".to_string(), cfg.project.clone()),
    ]);

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(&cfg.endpoint)
        .with_headers(headers)
        .build()
        .context("failed to build OTLP/HTTP span exporter")?;

    let resource = Resource::builder()
        .with_service_name(cfg.service_name.clone())
        .build();
    let batch_delay_ms = cfg.batch_delay_ms;

    let provider = if batch_delay_ms == 0 {
        SdkTracerProvider::builder()
            .with_resource(resource)
            .with_sampler(Sampler::AlwaysOn)
            .with_simple_exporter(exporter)
            .build()
    } else {
        let batch_config = BatchConfigBuilder::default()
            .with_scheduled_delay(Duration::from_millis(batch_delay_ms))
            .build();

        let processor = BatchSpanProcessor::builder(exporter)
            .with_batch_config(batch_config)
            .build();

        SdkTracerProvider::builder()
            .with_resource(resource)
            .with_sampler(Sampler::AlwaysOn)
            .with_span_processor(processor)
            .build()
    };

    global::set_tracer_provider(provider.clone());

    let tracer = provider.tracer(cfg.service_name.clone());
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let subscriber = Registry::default().with(env_filter).with(otel_layer);

    // Note: `cfg.console` is reserved for future use. A `tracing_subscriber::fmt`
    // layer always being installed would require runtime type-erasure (the fmt
    // layer's concrete type differs from the no-layer type). Today, users who
    // want console output alongside the OTel layer can do so by composing
    // their own subscriber, or by setting `RUST_LOG=info` to surface
    // log-level events through env-filter alone.
    if cfg.console {
        tracing::warn!(
            "LANGSMITH_OTEL_CONSOLE is not yet implemented — console output is not activated"
        );
    }

    tracing::subscriber::set_global_default(subscriber)
        .context("failed to install global tracing subscriber (one is already set?)")?;

    Ok(TracerHandle {
        provider: Arc::new(provider),
    })
}

/// Flush and shut down the tracer provider. Equivalent to
/// [`TracerHandle::shutdown`] but takes the handle by value for ergonomics.
pub async fn shutdown_tracing(handle: TracerHandle) -> Result<()> {
    handle.shutdown()
}
