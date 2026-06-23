#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use agent_rs_lib::agent::react::{
    REACT_PREAMBLE, ReActLoop, ReActSpanEmitter, detect_final_answer,
};
use agent_rs_lib::domain::agent::{
    Action, FinalAnswer, Observation, ReActStep, ReActTrace, Thought,
};
use agent_rs_lib::domain::errors::ReActError;
use rig_core::client::CompletionClient;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// A no-op span emitter that tracks calls for assertion purposes.
struct TestSpanEmitter;

impl ReActSpanEmitter for TestSpanEmitter {}

/// Build a real Agent backed by the openai client (pointed at a non-existent
/// server) just for testing builder defaults. The model is never actually
/// called.
fn make_test_agent() -> rig_core::agent::Agent<
    rig_core::providers::openai::responses_api::ResponsesCompletionModel<reqwest::Client>,
> {
    let client = rig_core::providers::openai::Client::builder()
        .base_url("http://127.0.0.1:1")
        .api_key("test")
        .build()
        .expect("build openai client for test");
    client
        .agent(rig_core::providers::openai::GPT_4O)
        .preamble("test preamble")
        .build()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_detect_final_answer() {
    assert_eq!(
        detect_final_answer("Final Answer: The answer is 42"),
        Some("The answer is 42".to_string())
    );
    assert_eq!(
        detect_final_answer("FINAL ANSWER: done"),
        Some("done".to_string())
    );
    assert_eq!(detect_final_answer("no sentinel here"), None);
    assert_eq!(detect_final_answer("Final Answer:"), None);
    assert_eq!(detect_final_answer("Final Answer:   "), None);
}

#[test]
fn trace_is_serializable() {
    let trace = ReActTrace {
        prompt: "test prompt".to_string(),
        steps: vec![
            ReActStep::Thought(Thought {
                reasoning: "I need to think".to_string(),
                cycle: 0,
            }),
            ReActStep::Action(Action {
                tool_name: "read_file".to_string(),
                args: "{}".to_string(),
                tool_call_id: Some("tc-1".to_string()),
                cycle: 0,
            }),
            ReActStep::Observation(Observation {
                tool_name: "read_file".to_string(),
                result: "file contents here".to_string(),
                is_error: false,
                cycle: 0,
                duration: Duration::from_millis(42),
            }),
            ReActStep::FinalAnswer(FinalAnswer {
                text: "The answer is 42".to_string(),
                cycles: 1,
            }),
        ],
        final_answer: Some(FinalAnswer {
            text: "The answer is 42".to_string(),
            cycles: 1,
        }),
    };

    let json = serde_json::to_string_pretty(&trace).unwrap();
    assert!(json.contains(r#""kind": "thought""#), "missing thought tag");
    assert!(json.contains(r#""kind": "action""#), "missing action tag");
    assert!(
        json.contains(r#""kind": "observation""#),
        "missing observation tag"
    );
    assert!(
        json.contains(r#""kind": "final_answer""#),
        "missing final_answer tag"
    );

    // Round-trip
    let deserialized: ReActTrace = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.steps.len(), 4);
    assert_eq!(deserialized.prompt, "test prompt");
    let fa = deserialized.final_answer.as_ref().unwrap();
    assert_eq!(fa.text, "The answer is 42");
    assert_eq!(fa.cycles, 1);
}

#[test]
fn react_error_display_includes_cycles() {
    let err = ReActError::MaxCyclesExceeded { cycles: 7 };
    let msg = err.to_string();
    assert!(msg.contains("7"), "Expected '7' in: {msg}");
    assert!(
        msg.contains("max_cycles"),
        "Expected 'max_cycles' in: {msg}"
    );
}

#[test]
fn react_error_tool_execution_includes_tool_name() {
    let err = ReActError::ToolExecution {
        tool: "read_file".to_string(),
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        )),
    };
    let msg = err.to_string();
    assert!(msg.contains("read_file"), "Expected 'read_file' in: {msg}");
}

#[test]
fn react_error_no_tool_calls_message() {
    let err = ReActError::NoToolCallsAndNoFinalAnswer { cycle: 3 };
    let msg = err.to_string();
    assert!(msg.contains("3"), "Expected '3' in: {msg}");
}

#[test]
fn react_preamble_is_nonempty_and_under_300_tokens() {
    assert!(
        REACT_PREAMBLE.len() > 50,
        "Preamble too short: {} bytes",
        REACT_PREAMBLE.len()
    );
    let word_count = REACT_PREAMBLE.split_whitespace().count();
    assert!(
        word_count < 300,
        "Preamble too long: {word_count} words (limit 300)"
    );
}

#[test]
fn react_loop_builder_defaults() {
    let agent = make_test_agent();
    let mut history: Vec<rig_core::message::Message> = Vec::new();
    let loop_builder = ReActLoop::builder(&agent, "hello", &mut history);

    assert_eq!(loop_builder.max_cycles, 20);
    assert!(
        loop_builder.react_preamble.is_none(),
        "react_preamble should be None by default"
    );
}

#[test]
fn noop_span_emitter_is_inert() {
    let emitter = TestSpanEmitter;
    let trace = ReActTrace::default();

    // None of these should panic.
    emitter.emit_cycle_start(0);
    emitter.emit_cycle_end(0, &trace);

    let dummy_action = Action {
        tool_name: "tool_a".to_string(),
        args: "{}".to_string(),
        tool_call_id: None,
        cycle: 0,
    };
    emitter.emit_action(&dummy_action);

    let dummy_obs = Observation {
        tool_name: "tool_a".to_string(),
        result: "ok".to_string(),
        is_error: false,
        cycle: 0,
        duration: Duration::from_millis(1),
    };
    emitter.emit_observation(&dummy_obs);
}
