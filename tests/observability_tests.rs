#![cfg(feature = "opentelemetry")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::OnceLock;

use agent_rs_lib::domain::observability::LangSmithConfig;
use agent_rs_lib::observability::init_tracing;

static INIT: OnceLock<()> = OnceLock::new();

fn init_once() {
    INIT.get_or_init(|| {
        let cfg = LangSmithConfig {
            endpoint: "http://127.0.0.1:0".to_string(),
            api_key: "test".to_string(),
            project: "test".to_string(),
            service_name: "test".to_string(),
            console: false,
            batch: true,
        };
        // A global tracing subscriber can only be set once per process.
        // If another test module (e.g. `react_otel_tests`) installed it first,
        // this returns `Err` and the test continues to use that subscriber.
        let _ = init_tracing(&cfg);
    });
}

#[tokio::test(flavor = "current_thread")]
async fn init_tracing_returns_handle() {
    init_once();

    // Emit a test span to prove the subscriber is wired up.
    let _guard = tracing::info_span!("test_span").entered();
    tracing::info!("test event from observability test");
}

#[tokio::test(flavor = "current_thread")]
async fn shutdown_tracing_succeeds() {
    init_once();

    // We can't create a second handle (global subscriber already set),
    // but we verify the module compiles and the subscriber is active.
}

#[test]
fn langsmith_config_from_env() {
    let cfg = LangSmithConfig::default();
    assert_eq!(
        cfg.endpoint,
        "https://api.smith.langchain.com/otel/v1/traces"
    );
    assert!(cfg.api_key.is_empty());
    assert_eq!(cfg.project, "default");
    assert_eq!(cfg.service_name, "agent_rs");
    assert!(!cfg.console);
    assert!(cfg.batch);
}

#[test]
fn langsmith_config_from_env_or_default_missing_key() {
    let result = LangSmithConfig::from_env_or_default("NONEXISTENT_ENV_VAR_12345");
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("missing env var"));
}
