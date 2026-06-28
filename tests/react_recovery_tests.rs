#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/mod.rs"]
mod common;

use agent_rs::agent::ReActExt;
use agent_rs::agent::react::{ReActSpanEmitter, recover_turn_limit_history};
use agent_rs::domain::errors::ReActError;
use rig_core::client::CompletionClient;
use rig_core::completion::request::PromptError;
use rig_core::message::{Message, UserContent};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

struct TestSpanEmitter;

impl ReActSpanEmitter for TestSpanEmitter {}

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
fn test_turn_limit_max_cycles_overrides_default_max_turns() {
    let client = rig_core::providers::openai::Client::builder()
        .base_url("http://127.0.0.1:1")
        .api_key("test")
        .build()
        .expect("build openai client for test");
    let agent = client
        .agent(rig_core::providers::openai::GPT_4O)
        .preamble("test preamble")
        .default_max_turns(20)
        .build();
    let builder = agent.react().max_cycles(5);
    assert_eq!(builder.max_cycles, 5);
    let built = builder.build();
    assert_eq!(built.max_cycles(), 5);
}

#[tokio::test]
async fn test_model_http_error_is_retried() {
    let agent = make_test_agent();
    let built = agent.react().build();
    let before = built.history().len();
    let err = built.prompt("test").await.unwrap_err();
    match &err {
        ReActError::Model(s) => {
            let lower = s.to_lowercase();
            assert!(
                lower.contains("connection") || lower.contains("http") || lower.contains("connect"),
                "Expected connection/HTTP error in: {s}"
            );
        }
        other => panic!("Expected ReActError::Model, got: {other:?}"),
    }
    assert_eq!(
        built.history().len(),
        before,
        "history should not be mutated on error"
    );
}

#[tokio::test]
async fn test_model_non_transient_error_not_retried() {
    use common::{MockCompletionModel, mock_agent};

    let responses = vec![MockCompletionModel::json_error("non-transient json error")];

    let agent = mock_agent(responses);
    let built = agent.react().max_retries(3).build();
    let before = built.history().len();
    let err = built.prompt("test").await.unwrap_err();
    match &err {
        ReActError::Model(s) => {
            let lower = s.to_lowercase();
            assert!(
                lower.contains("json") || lower.contains("non-transient"),
                "Expected JSON/non-transient error in: {s}"
            );
        }
        other => panic!("Expected ReActError::Model, got: {other:?}"),
    }
    assert_eq!(
        built.history().len(),
        before,
        "history should not be mutated on error"
    );
}

#[tokio::test]
async fn test_empty_assistant_content_retried_once() {
    use common::{MockCompletionModel, mock_agent};

    // First call returns empty content (should trigger one retry).
    // Second call returns the final answer.
    let responses = vec![
        MockCompletionModel::text(""),
        MockCompletionModel::text("Final Answer: recovered"),
    ];

    let agent = mock_agent(responses);
    let built = agent.react().max_cycles(3).build();
    let trace = built.prompt("test").await.expect("prompt should succeed");

    let fa = trace
        .final_answer
        .as_ref()
        .expect("should have final answer");
    assert_eq!(fa.text, "Final Answer: recovered");
}

#[tokio::test]
async fn test_history_not_mutated_on_error() {
    let agent = make_test_agent();
    let built = agent.react().build();
    let before = built.history().len();
    let _ = built.prompt("test").await;
    assert_eq!(built.history().len(), before);
}

#[test]
fn test_noop_span_emitter_emit_error_is_inert() {
    let emitter = TestSpanEmitter;
    let err = ReActError::Model("test".into());
    emitter.emit_error(&err);
}

#[test]
fn test_observation_is_error_on_wrong_tool_name() {
    use agent_rs::domain::agent::Observation;
    let obs = Observation {
        tool_name: "listDirectory".to_string(),
        result: "tool not found".to_string(),
        is_error: true,
        cycle: 0,
        duration: Duration::from_millis(1),
    };
    assert!(obs.is_error);
    let json = serde_json::to_string(&obs).unwrap();
    assert!(json.contains("\"is_error\":true"));
}

// ---------------------------------------------------------------------------
// Turn-limit recovery tests
// ---------------------------------------------------------------------------

/// Build a simple user turn `Message` (mirrors how the ReAct loop constructs
/// the effective prompt).
fn user_msg(text: &str) -> Message {
    Message::User {
        content: rig_core::OneOrMany::one(UserContent::text(text)),
    }
}

/// `recover_turn_limit_history` must extract the partial progress carried by a
/// `MaxTurnsError`. This is the data the pre-fix code discarded, causing the
/// "hard stuck" loop where each cycle re-sent an identical request.
#[test]
fn recover_turn_limit_history_extracts_partial_progress() {
    // In rig-core's `MaxTurnsError`, `chat_history` is `build_full_history`
    // — i.e. the snapshot + the cycle's prompt + progress, with the final
    // pending message as its last element (mirrored by the `prompt` field).
    let snapshot = vec![user_msg("snapshot-1"), user_msg("snapshot-2")];
    let prompt = user_msg("pending tool result");
    let mut full_history = snapshot.clone();
    full_history.push(prompt.clone());
    let err = PromptError::MaxTurnsError {
        max_turns: 3,
        chat_history: Box::new(full_history.clone()),
        prompt: Box::new(prompt.clone()),
    };
    let recovered = recover_turn_limit_history(&err).expect("should recover from MaxTurnsError");
    assert_eq!(recovered, full_history);
    // The pending prompt is the last element of the recovered history — the
    // run_loop caller pops it to use as `current_prompt` for the next cycle,
    // leaving the snapshot + progress as the new working history.
    let mut working = recovered;
    let last = working.pop().expect("history non-empty");
    assert_eq!(working, snapshot);
    assert_eq!(last, prompt);
}

/// Other `PromptError` variants carry no recoverable history — must return
/// `None` so the loop falls through to the non-recoverable error path.
#[test]
fn recover_turn_limit_history_returns_none_for_other_errors() {
    let err = PromptError::PromptCancelled {
        chat_history: Vec::new(),
        reason: "test".to_string(),
    };
    assert!(recover_turn_limit_history(&err).is_none());
}
