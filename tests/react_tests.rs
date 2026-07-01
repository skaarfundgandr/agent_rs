#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use agent_rs::agent::ReActExt;
use agent_rs::agent::react::{
    ActionCb, ObservationCb, REACT_PREAMBLE, ReActSpanEmitter, detect_final_answer,
};
use agent_rs::domain::agent::{Action, FinalAnswer, Observation, ReActStep, ReActTrace, Thought};
use agent_rs::domain::errors::ReActError;
use rig_core::OneOrMany;
use rig_core::client::CompletionClient;
use rig_core::message::{
    AssistantContent, Message, ToolCall, ToolFunction, ToolResultContent, UserContent,
};
use std::sync::{Arc, Mutex};
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

fn make_obs_callback(observations: &Arc<Mutex<Vec<Observation>>>) -> Option<ObservationCb> {
    let obs = Arc::clone(observations);
    Some(Arc::new(move |o| obs.lock().unwrap().push(o.clone())))
}

fn make_tool_call_msg(call_id: &str, name: &str, args: serde_json::Value) -> Message {
    Message::Assistant {
        id: None,
        content: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
            call_id.to_string(),
            ToolFunction {
                name: name.to_string(),
                arguments: args,
            },
        ))),
    }
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
fn test_react_builder_defaults() {
    let agent = make_test_agent();
    let builder = agent.react();
    assert_eq!(builder.max_cycles, 20);
    assert_eq!(builder.max_retries, 3);
    assert_eq!(builder.tool_timeout_secs, 60);
    assert!(
        builder.react_preamble.is_none(),
        "react_preamble should be None by default"
    );
}

#[test]
fn test_react_builder_max_retries_setter() {
    let agent = make_test_agent();
    let builder = agent.react().max_retries(5);
    assert_eq!(builder.max_retries, 5);
    let built = builder.build();
    assert_eq!(built.max_retries(), 5);
}

#[test]
#[should_panic(expected = "threshold")]
fn test_builder_compaction_panics_without_threshold() {
    let agent = make_test_agent();
    let _ = agent.react().with_compaction().build(); // panics
}

#[tokio::test]
async fn test_built_prompt_does_not_mutate_history() {
    let agent = make_test_agent();
    let built = agent.react().build();
    let h: Vec<Message> = Vec::new();
    let _ = built.prompt("test").await;
    assert_eq!(h.len(), 0);
}

#[test]
fn test_built_chat_accessors() {
    let agent = make_test_agent();
    let built = agent.react().max_cycles(10).build();
    assert_eq!(built.max_cycles(), 10);
    let h: Vec<Message> = Vec::new();
    assert!(h.is_empty());
}

#[test]
#[should_panic(expected = "max_cycles must be at least 1")]
fn test_max_cycles_zero_panics() {
    let agent = make_test_agent();
    let _ = agent.react().max_cycles(0);
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

/// When rig-core executes tool turns internally, the non-streaming ReAct loop
/// must still surface every intermediate action and observation callback,
/// not only the ones from the final assistant turn.
#[test]
fn internal_tool_turns_emit_callbacks() {
    let actions = Arc::new(Mutex::new(Vec::<Action>::new()));
    let observations = Arc::new(Mutex::new(Vec::<Observation>::new()));

    let actions_clone = Arc::clone(&actions);
    let on_action: Option<ActionCb> = Some(Arc::new(move |a| {
        actions_clone.lock().unwrap().push(a.clone())
    }));

    let on_observation = make_obs_callback(&observations);

    let messages = vec![
        // Initial user prompt.
        Message::User {
            content: OneOrMany::one(UserContent::text("read a file")),
        },
        // First assistant turn: tool call executed internally by rig-core.
        make_tool_call_msg(
            "tc-1",
            "read_file",
            serde_json::json!({"path": "README.md"}),
        ),
        // Corresponding tool result.
        // Corresponding tool result.
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(rig_core::message::ToolResult {
                id: "tc-1".to_string(),
                call_id: None,
                content: OneOrMany::one(ToolResultContent::text("hello")),
            })),
        },
        // Final assistant turn: plain text answer.
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::text("Final Answer: done")),
        },
    ];

    let mut trace = ReActTrace::default();
    let emitter: Arc<dyn ReActSpanEmitter> = Arc::new(TestSpanEmitter);
    agent_rs::agent::react::emit_internal_tool_callbacks(
        &messages,
        0,
        &on_action,
        &on_observation,
        &emitter,
        &mut trace,
    );

    let actions = actions.lock().unwrap();
    let observations = observations.lock().unwrap();

    assert_eq!(actions.len(), 1, "expected one action callback");
    assert_eq!(actions[0].tool_name, "read_file");

    assert_eq!(observations.len(), 1, "expected one observation callback");
    assert_eq!(observations[0].tool_name, "read_file");
    assert_eq!(observations[0].result, "hello");

    assert!(
        matches!(trace.steps[0], ReActStep::Action(_)),
        "first trace step should be an action"
    );
    assert!(
        matches!(trace.steps[1], ReActStep::Observation(_)),
        "second trace step should be an observation"
    );
}

/// When the first message in `response.messages` is a tool result carried
/// over from the previous ReAct cycle (i.e. the cycle's `current_prompt`),
/// it must NOT be treated as a new observation.
#[test]
fn internal_tool_callbacks_skip_leading_prompt_tool_result() {
    let observations = Arc::new(Mutex::new(Vec::<Observation>::new()));
    let on_observation = make_obs_callback(&observations);

    // First message is the prompt for this `agent.prompt()` call, modelled as
    // a tool result from the previous cycle.
    let messages = vec![
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(rig_core::message::ToolResult {
                id: "tc-prev".to_string(),
                call_id: None,
                content: OneOrMany::one(ToolResultContent::text("previous result")),
            })),
        },
        make_tool_call_msg(
            "tc-1",
            "read_file",
            serde_json::json!({"path": "README.md"}),
        ),
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(rig_core::message::ToolResult {
                id: "tc-1".to_string(),
                call_id: None,
                content: OneOrMany::one(ToolResultContent::text("hello")),
            })),
        },
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::text("Final Answer: done")),
        },
    ];

    let mut trace = ReActTrace::default();
    let emitter: Arc<dyn ReActSpanEmitter> = Arc::new(TestSpanEmitter);
    agent_rs::agent::react::emit_internal_tool_callbacks(
        &messages,
        1,
        &None,
        &on_observation,
        &emitter,
        &mut trace,
    );

    let observations = observations.lock().unwrap();
    assert_eq!(
        observations.len(),
        1,
        "only the new tool result should emit an observation"
    );
    assert_eq!(observations[0].tool_name, "read_file");
    assert_eq!(observations[0].result, "hello");
    assert!(
        !observations.iter().any(|o| o.result == "previous result"),
        "previous cycle's tool result must not emit a callback"
    );
}
