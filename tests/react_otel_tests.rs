#![cfg(feature = "opentelemetry")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex, OnceLock};

use agent_rs_lib::agent::react::ReActSpanEmitter;
use agent_rs_lib::domain::agent::{
    Action, FinalAnswer, Observation, ReActStep, ReActTrace, Thought,
};
use agent_rs_lib::domain::errors::ReActError;
use agent_rs_lib::domain::observability::LangSmithConfig;
use agent_rs_lib::observability::{LangSmithAgentHook, LangSmithReActEmitter, init_tracing};
use tracing::field::Visit;
use tracing::span::{Id, Record};
use tracing::{Event, Subscriber};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

// ---------------------------------------------------------------------------
// OnceLock guard — only one global subscriber per process
// ---------------------------------------------------------------------------

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
        // Ignore "already set" errors — another test module may have
        // installed the subscriber first.
        let _ = init_tracing(&cfg);
    });
}

// ---------------------------------------------------------------------------
// Capture layer — observes tracing events and span field records
// ---------------------------------------------------------------------------

/// A `tracing_subscriber::Layer` that captures every emitted `Event` and
/// every `Record` applied to a span, for assertion in tests.
#[derive(Clone, Default)]
struct TraceCapture {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
    records: Arc<Mutex<Vec<CapturedRecord>>>,
}

#[derive(Debug, Clone)]
struct CapturedEvent {
    level: tracing::Level,
    #[allow(dead_code)]
    target: String,
    fields: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
struct CapturedRecord {
    fields: Vec<(String, String)>,
}

struct FieldVisitor(Vec<(String, String)>);

impl Visit for FieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .push((field.name().to_string(), format!("{value:?}")));
    }
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.push((field.name().to_string(), value.to_string()));
    }
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.0.push((field.name().to_string(), value.to_string()));
    }
    fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
        self.0.push((field.name().to_string(), value.to_string()));
    }
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.0.push((field.name().to_string(), value.to_string()));
    }
}

impl<S: Subscriber> Layer<S> for TraceCapture {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor(Vec::new());
        event.record(&mut visitor);
        self.events.lock().unwrap().push(CapturedEvent {
            level: *event.metadata().level(),
            target: event.metadata().target().to_string(),
            fields: visitor.0,
        });
    }

    fn on_record(&self, _id: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor(Vec::new());
        values.record(&mut visitor);
        if !visitor.0.is_empty() {
            self.records
                .lock()
                .unwrap()
                .push(CapturedRecord { fields: visitor.0 });
        }
    }
}

impl TraceCapture {
    fn error_events(&self) -> Vec<CapturedEvent> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.level == tracing::Level::ERROR)
            .cloned()
            .collect()
    }

    fn fields_named(&self, name: &str) -> Vec<String> {
        self.records
            .lock()
            .unwrap()
            .iter()
            .flat_map(|r| r.fields.iter())
            .filter(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn langsmith_react_emitter_default_compiles() {
    let _e = LangSmithReActEmitter;
}

#[test]
fn langsmith_agent_hook_default_compiles() {
    let _h = LangSmithAgentHook;
}

#[test]
fn langsmith_react_emitter_implements_trait() {
    let emitter = Arc::new(LangSmithReActEmitter) as Arc<dyn ReActSpanEmitter>;
    emitter.emit_cycle_start(0);
    emitter.emit_cycle_end(0, &ReActTrace::default());

    let action = Action {
        tool_name: "read_file".to_string(),
        args: r#"{"path":"foo.txt"}"#.to_string(),
        tool_call_id: Some("tc-1".to_string()),
        cycle: 0,
    };
    emitter.emit_action(&action);

    let observation = Observation {
        tool_name: "read_file".to_string(),
        result: "file contents".to_string(),
        is_error: false,
        cycle: 0,
        duration: std::time::Duration::from_millis(42),
    };
    emitter.emit_observation(&observation);
}

#[test]
fn langsmith_react_emitter_does_not_panic_on_recording() {
    init_once();

    let emitter = LangSmithReActEmitter;

    // These span.record() calls target fields that don't exist on the
    // "test_span" — tracing silently ignores them, no panic.
    let _guard = tracing::info_span!("test_span").entered();

    emitter.emit_cycle_start(0);
    emitter.emit_cycle_end(0, &ReActTrace::default());

    let action = Action {
        tool_name: "test_tool".to_string(),
        args: "{}".to_string(),
        tool_call_id: None,
        cycle: 0,
    };
    emitter.emit_action(&action);

    let observation = Observation {
        tool_name: "test_tool".to_string(),
        result: "ok".to_string(),
        is_error: false,
        cycle: 0,
        duration: std::time::Duration::from_millis(1),
    };
    emitter.emit_observation(&observation);
}

#[test]
fn re_exports_are_present() {
    use agent_rs_lib::observability::{LangSmithAgentHook, LangSmithReActEmitter};

    let _emitter = LangSmithReActEmitter;
    let _hook = LangSmithAgentHook;
}

#[test]
fn react_trace_has_all_step_types() {
    init_once();

    let emitter = LangSmithReActEmitter;
    let trace = ReActTrace {
        prompt: "test".to_string(),
        steps: vec![
            ReActStep::Thought(Thought {
                reasoning: "I need to read a file".to_string(),
                cycle: 0,
            }),
            ReActStep::Action(Action {
                tool_name: "read_file".to_string(),
                args: r#"{"path":"foo.txt"}"#.to_string(),
                tool_call_id: Some("tc-1".to_string()),
                cycle: 0,
            }),
            ReActStep::Observation(Observation {
                tool_name: "read_file".to_string(),
                result: "hello world".to_string(),
                is_error: false,
                cycle: 0,
                duration: std::time::Duration::from_millis(10),
            }),
            ReActStep::FinalAnswer(FinalAnswer {
                text: "done".to_string(),
                cycles: 1,
            }),
        ],
        final_answer: Some(FinalAnswer {
            text: "done".to_string(),
            cycles: 1,
        }),
    };

    // Should not panic; the trace serializes to a JSON string that gets
    // emitted via tracing::info! in emit_cycle_end.
    let _guard = tracing::info_span!("test_trace").entered();
    emitter.emit_cycle_end(0, &trace);
}

/// `LangSmithReActEmitter` must:
/// 1. Emit a `tracing::error!` event with the tool name and error message
///    when `emit_observation(is_error=true)` is called.
/// 2. Set `react.is_error` to `true` on the current span.
/// 3. Emit a `tracing::error!` event when `emit_error` is called.
/// 4. Set `react.is_error` to `true` on the current span via `emit_error`.
#[test]
fn langsmith_react_emitter_handles_errors_gracefully() {
    let capture = TraceCapture::default();
    let subscriber = Registry::default().with(capture.clone());

    tracing::subscriber::with_default(subscriber, || {
        // The span must declare the fields the emitter will record, otherwise
        // tracing silently drops the record() call.
        let span = tracing::info_span!(
            "test_error_span",
            "langsmith.span.kind" = tracing::field::Empty,
            "openinference.span.kind" = tracing::field::Empty,
            "gen_ai.tool.name" = tracing::field::Empty,
            "output.value" = tracing::field::Empty,
            "react.is_error" = tracing::field::Empty,
            "react.duration_ms" = tracing::field::Empty,
        );
        let _guard = span.enter();

        let emitter = LangSmithReActEmitter;

        // 1. Tool call error
        let observation = Observation {
            tool_name: "failing_tool".to_string(),
            result: "Permission denied".to_string(),
            is_error: true,
            cycle: 0,
            duration: std::time::Duration::from_millis(5),
        };
        emitter.emit_observation(&observation);

        // 2. Loop level error
        let err = ReActError::Model("API rate limit exceeded".to_string());
        emitter.emit_error(&err);

        // --- Assertion 1: tracing::error! events were emitted ----------
        let error_events = capture.error_events();
        assert!(
            error_events.len() >= 2,
            "expected at least 2 ERROR events (one from emit_observation, one from emit_error), got: {error_events:#?}"
        );

        // Find the emit_observation error event — it should mention the tool
        // name and the error result.
        let obs_error = error_events
            .iter()
            .find(|e| {
                e.fields
                    .iter()
                    .any(|(k, v)| k == "tool_name" && v == "failing_tool")
            })
            .expect("expected an error event mentioning tool_name=failing_tool");
        let has_error_text = obs_error
            .fields
            .iter()
            .any(|(k, v)| k == "error" && v.contains("Permission denied"));
        assert!(
            has_error_text,
            "expected error event to include the observation error text, got: {obs_error:#?}"
        );

        // Find the emit_error event — it should include the error message.
        let loop_error = error_events
            .iter()
            .find(|e| {
                e.fields
                    .iter()
                    .any(|(k, v)| k == "error" && v.contains("API rate limit exceeded"))
            })
            .expect("expected an error event mentioning the ReActError message");
        assert_eq!(loop_error.level, tracing::Level::ERROR);

        // --- Assertion 2: react.is_error was set to true --------------
        let is_error_values = capture.fields_named("react.is_error");
        assert!(
            is_error_values.iter().any(|v| v == "true"),
            "expected react.is_error to be set to 'true' on the span, got: {is_error_values:?}"
        );

        // --- Assertion 3: output.value was set to the observation result
        let output_values = capture.fields_named("output.value");
        assert!(
            output_values.iter().any(|v| v == "Permission denied"),
            "expected output.value to be set to the observation result, got: {output_values:?}"
        );

        // --- Assertion 4: gen_ai.tool.name was set to the tool name ----
        let tool_names = capture.fields_named("gen_ai.tool.name");
        assert!(
            tool_names.iter().any(|v| v == "failing_tool"),
            "expected gen_ai.tool.name to be set to 'failing_tool', got: {tool_names:?}"
        );
    });
}
