#![cfg(feature = "opentelemetry")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, OnceLock};

use agent_rs_lib::agent::react::ReActSpanEmitter;
use agent_rs_lib::domain::agent::{
    Action, FinalAnswer, Observation, ReActStep, ReActTrace, Thought,
};
use agent_rs_lib::domain::observability::LangSmithConfig;
use agent_rs_lib::observability::{LangSmithAgentHook, LangSmithReActEmitter, init_tracing};

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
