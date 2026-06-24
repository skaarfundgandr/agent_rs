#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Integration tests for `ReActLoop::recover_from_prompt_error`.
//!
//! These tests exercise the `PromptError` recovery paths
//! (`UnknownToolCall`, `MaxTurnsError`, and the unrecoverable `Abort` arm)
//! without going through the full `execute()` API — that path requires a
//! live completion model, which would make the tests flaky. Instead the
//! tests construct synthetic `PromptError` values and a real
//! `ReActLoop` (with a no-op `Agent` reference) and assert on the
//! resulting `self.history`, `trace.steps`, and returned `PromptRecovery`.

use std::sync::{Arc, Mutex};

use agent_rs_lib::agent::react::{PromptRecovery, ReActLoop, ReActSpanEmitter};
use agent_rs_lib::domain::agent::{Action, Observation, ReActStep, ReActTrace, Thought};
use agent_rs_lib::domain::errors::ReActError;
use rig_core::OneOrMany;
use rig_core::client::CompletionClient;
use rig_core::completion::{CompletionError, PromptError};
use rig_core::message::{AssistantContent, Message, ToolCall, ToolFunction, UserContent};
use serde_json::json;

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

type TestAgent = rig_core::agent::Agent<
    rig_core::providers::openai::responses_api::ResponsesCompletionModel<reqwest::Client>,
>;

/// A recording span emitter that captures every call for later assertions.
#[derive(Clone, Default)]
struct RecordingSpanEmitter {
    thoughts: Arc<Mutex<Vec<Thought>>>,
    actions: Arc<Mutex<Vec<Action>>>,
    observations: Arc<Mutex<Vec<Observation>>>,
    cycle_starts: Arc<Mutex<Vec<usize>>>,
    cycle_ends: Arc<Mutex<Vec<usize>>>,
    /// Captures the Display string of each `ReActError` (the type itself
    /// does not implement `Clone`).
    errors: Arc<Mutex<Vec<String>>>,
}

impl ReActSpanEmitter for RecordingSpanEmitter {
    fn emit_thought(&self, thought: &Thought) {
        self.thoughts.lock().unwrap().push(thought.clone());
    }
    fn emit_cycle_start(&self, cycle: usize) {
        self.cycle_starts.lock().unwrap().push(cycle);
    }
    fn emit_cycle_end(&self, cycle: usize, _trace: &ReActTrace) {
        self.cycle_ends.lock().unwrap().push(cycle);
    }
    fn emit_action(&self, action: &Action) {
        self.actions.lock().unwrap().push(action.clone());
    }
    fn emit_observation(&self, observation: &Observation) {
        self.observations.lock().unwrap().push(observation.clone());
    }
    fn emit_error(&self, err: &ReActError) {
        self.errors.lock().unwrap().push(err.to_string());
    }
}

/// Build a real `Agent` backed by the openai client (pointed at a non-existent
/// server). The model is never actually called — these tests only exercise
/// `ReActLoop::recover_from_prompt_error`, which never invokes the agent.
fn make_test_agent() -> TestAgent {
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

fn make_assistant_with_tool_call(tool_name: &str, call_id: &str) -> Message {
    let tc = ToolCall {
        id: call_id.to_string(),
        call_id: Some(format!("call-{call_id}")),
        function: ToolFunction::new(tool_name.to_string(), json!({"path": "test"})),
        signature: None,
        additional_params: None,
    };
    Message::Assistant {
        id: Some(format!("asst-{call_id}")),
        content: OneOrMany::one(AssistantContent::ToolCall(tc)),
    }
}

fn make_user_message(text: &str) -> Message {
    Message::User {
        content: OneOrMany::one(UserContent::text(text)),
    }
}

/// Drop the `ReActLoop` so the `&mut Vec<Message>` borrow on `history` is
/// released, allowing post-recovery assertions on `history`.
fn drop_loop<M, P>(rl: ReActLoop<'_, M, P>)
where
    M: rig_core::completion::CompletionModel
        + rig_core::wasm_compat::WasmCompatSend
        + rig_core::wasm_compat::WasmCompatSync
        + 'static,
    P: rig_core::agent::PromptHook<M>
        + rig_core::wasm_compat::WasmCompatSend
        + rig_core::wasm_compat::WasmCompatSync
        + 'static,
{
    drop(rl);
}

// ---------------------------------------------------------------------------
// UnknownToolCall recovery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_tool_call_recovers_with_corrective_user_message() {
    let agent = make_test_agent();
    let mut history: Vec<Message> = Vec::new();
    let emitter = Arc::new(RecordingSpanEmitter::default());

    let mut react_loop =
        ReActLoop::builder(&agent, "test prompt", &mut history).with_span_emitter(emitter.clone());

    let mut trace = ReActTrace {
        prompt: "test prompt".to_string(),
        steps: Vec::new(),
        final_answer: None,
    };

    // Diagnostic chat_history from rig-core includes the rejected assistant
    // message (containing the bad tool call) as the last entry.
    let assistant_msg = make_assistant_with_tool_call("nonexistent_tool", "tc-1");
    let chat_history = vec![assistant_msg.clone()];
    let err = PromptError::UnknownToolCall {
        tool_name: "nonexistent_tool".to_string(),
        available_tools: vec!["read_file".to_string(), "write_file".to_string()],
        allowed_tools: vec!["read_file".to_string(), "write_file".to_string()],
        chat_history: Box::new(chat_history),
    };

    let recovery = react_loop
        .recover_from_prompt_error(err, 0, &mut trace, 0)
        .await;

    // --- Returned prompt: a User message with a tool result ---
    let next_prompt = match recovery {
        PromptRecovery::Recovered(msg) => msg,
        PromptRecovery::Abort(ReActError::Model(msg)) => {
            panic!("expected Recovered, got Abort(Model({msg}))")
        }
        PromptRecovery::Abort(other) => panic!("expected Recovered, got Abort({other:?})"),
    };
    let Message::User { content } = next_prompt else {
        panic!("expected Message::User, got {next_prompt:?}");
    };
    // The user message should contain tool_result content (the corrective
    // feedback), not plain text.
    let mut saw_tool_result = false;
    for item in content.iter() {
        if matches!(item, UserContent::ToolResult(_)) {
            saw_tool_result = true;
        }
    }
    assert!(
        saw_tool_result,
        "expected next prompt to contain a ToolResult, got: {content:?}"
    );

    // --- trace.steps: Action + Observation recorded for the invalid call ---
    let action_count = trace
        .steps
        .iter()
        .filter(|s| matches!(s, ReActStep::Action(_)))
        .count();
    let obs_count = trace
        .steps
        .iter()
        .filter(|s| matches!(s, ReActStep::Observation(_)))
        .count();
    assert_eq!(
        action_count, 1,
        "expected 1 Action step for the invalid tool call"
    );
    assert_eq!(
        obs_count, 1,
        "expected 1 Observation step for the invalid tool call"
    );

    // --- Emitter side effects ---
    let actions = emitter.actions.lock().unwrap();
    let observations = emitter.observations.lock().unwrap();
    assert_eq!(actions.len(), 1);
    assert_eq!(actions[0].tool_name, "nonexistent_tool");
    assert_eq!(observations.len(), 1);
    assert!(
        observations[0].is_error,
        "the invalid tool call's observation should be marked as error"
    );
    assert_eq!(observations[0].tool_name, "nonexistent_tool");

    // --- self.history: original assistant message pushed ---
    // Drop the loop first to release the &mut borrow on `history`.
    drop_loop(react_loop);
    let assistant_count = history
        .iter()
        .filter(|m| matches!(m, Message::Assistant { .. }))
        .count();
    assert_eq!(
        assistant_count, 1,
        "expected 1 assistant message in history after recovery"
    );
}

#[tokio::test]
async fn unknown_tool_call_with_no_assistant_message_in_history_aborts() {
    // If the diagnostic chat_history is empty (or doesn't contain a matching
    // assistant message), the recovery can't proceed and must abort.
    let agent = make_test_agent();
    let mut history: Vec<Message> = Vec::new();
    let emitter = Arc::new(RecordingSpanEmitter::default());

    let mut react_loop =
        ReActLoop::builder(&agent, "test prompt", &mut history).with_span_emitter(emitter.clone());

    let mut trace = ReActTrace::default();

    let err = PromptError::UnknownToolCall {
        tool_name: "nonexistent_tool".to_string(),
        available_tools: vec!["read_file".to_string()],
        allowed_tools: vec!["read_file".to_string()],
        chat_history: Box::new(Vec::new()),
    };

    let recovery = react_loop
        .recover_from_prompt_error(err, 0, &mut trace, 0)
        .await;

    // Should fall through to the unrecoverable arm.
    match recovery {
        PromptRecovery::Abort(ReActError::Model(msg)) => {
            assert!(
                msg.contains("nonexistent_tool") || msg.contains("UnknownToolCall"),
                "expected the abort message to mention the tool or error kind, got: {msg}"
            );
        }
        PromptRecovery::Abort(other) => {
            panic!("expected Abort(Model(_)), got Abort({other:?})")
        }
        PromptRecovery::Recovered(msg) => {
            panic!("expected Abort, got Recovered({msg:?})")
        }
    }
}

// ---------------------------------------------------------------------------
// MaxTurnsError recovery
// ---------------------------------------------------------------------------

#[tokio::test]
async fn max_turns_error_merges_history_and_returns_prompt() {
    let agent = make_test_agent();
    let emitter = Arc::new(RecordingSpanEmitter::default());

    // Build the original history and error chat_history BEFORE creating the
    // ReActLoop (which holds &mut history).
    let orig_history = vec![make_user_message("pre-existing user message")];
    let new_msgs = [
        make_assistant_with_tool_call("read_file", "tc-1"),
        make_user_message("tool result"),
        make_user_message("final user turn (the prompt)"),
    ];
    let orig_hist_len = orig_history.len();

    // chat_history = orig_history + new_msgs. The last element of chat_history
    // is the `prompt` field of the MaxTurnsError (rig-core convention).
    let chat_history: Vec<Message> = orig_history
        .iter()
        .cloned()
        .chain(new_msgs.iter().cloned())
        .collect();
    let prompt = chat_history.last().expect("non-empty").clone();

    let mut history = orig_history;
    let mut react_loop =
        ReActLoop::builder(&agent, "test prompt", &mut history).with_span_emitter(emitter.clone());
    let mut trace = ReActTrace::default();

    let err = PromptError::MaxTurnsError {
        max_turns: 1,
        chat_history: Box::new(chat_history),
        prompt: Box::new(prompt.clone()),
    };

    let recovery = react_loop
        .recover_from_prompt_error(err, 0, &mut trace, orig_hist_len)
        .await;

    // The returned prompt should equal the original `prompt`.
    match recovery {
        PromptRecovery::Recovered(msg) => {
            let Message::User { content: ref c } = msg else {
                panic!("expected Message::User, got {msg:?}");
            };
            let prompt_text = match c.iter().next().expect("non-empty") {
                UserContent::Text(t) => t.text.clone(),
                other => panic!("expected UserContent::Text, got {other:?}"),
            };
            assert_eq!(prompt_text, "final user turn (the prompt)");
        }
        PromptRecovery::Abort(ReActError::Model(msg)) => {
            panic!("expected Recovered, got Abort(Model({msg}))")
        }
        PromptRecovery::Abort(other) => panic!("expected Recovered, got Abort({other:?})"),
    }

    // Drop the loop to release the &mut history borrow.
    drop_loop(react_loop);

    // history should be: [pre-existing, assistant, user tool result].
    // The final prompt is NOT pushed (it becomes the next cycle's prompt).
    assert_eq!(
        history.len(),
        3,
        "expected history to have 3 messages (orig + 2 new, prompt excluded), got: {history:#?}"
    );
}

#[tokio::test]
async fn max_turns_error_with_no_new_messages_does_not_extend_history() {
    // If chat_history[orig_hist_len..] is empty (or has 1 element), the
    // history is not extended but the prompt is still returned.
    let agent = make_test_agent();
    let emitter = Arc::new(RecordingSpanEmitter::default());

    // Build the original history BEFORE creating the ReActLoop.
    let orig_history = vec![make_user_message("pre-existing")];
    let orig_hist_len = orig_history.len();

    // chat_history == orig_history (no new messages), prompt is unrelated.
    let prompt = make_user_message("the prompt");
    let err = PromptError::MaxTurnsError {
        max_turns: 1,
        chat_history: Box::new(orig_history.clone()),
        prompt: Box::new(prompt.clone()),
    };

    let mut history = orig_history;
    let mut react_loop =
        ReActLoop::builder(&agent, "test prompt", &mut history).with_span_emitter(emitter.clone());
    let mut trace = ReActTrace::default();

    let recovery = react_loop
        .recover_from_prompt_error(err, 0, &mut trace, orig_hist_len)
        .await;

    match recovery {
        PromptRecovery::Recovered(_) => {}
        PromptRecovery::Abort(ReActError::Model(msg)) => {
            panic!("expected Recovered, got Abort(Model({msg}))")
        }
        PromptRecovery::Abort(other) => panic!("expected Recovered, got Abort({other:?})"),
    }

    // History unchanged.
    assert_eq!(history.len(), 1);
}

// ---------------------------------------------------------------------------
// Unrecoverable errors
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completion_error_returns_abort() {
    let agent = make_test_agent();
    let mut history: Vec<Message> = Vec::new();
    let emitter = Arc::new(RecordingSpanEmitter::default());

    let mut react_loop =
        ReActLoop::builder(&agent, "test prompt", &mut history).with_span_emitter(emitter.clone());

    let mut trace = ReActTrace::default();

    let err = PromptError::CompletionError(CompletionError::ProviderError(
        "upstream timeout".to_string(),
    ));

    let recovery = react_loop
        .recover_from_prompt_error(err, 0, &mut trace, 0)
        .await;

    match recovery {
        PromptRecovery::Abort(ReActError::Model(msg)) => {
            // The error should be wrapped into ReActError::Model with the
            // PromptError's Display impl as the message.
            assert!(
                msg.contains("ProviderError") || msg.contains("upstream timeout"),
                "expected abort message to mention the underlying error, got: {msg}"
            );
        }
        PromptRecovery::Abort(other) => {
            panic!("expected Abort(Model(_)), got Abort({other:?})")
        }
        PromptRecovery::Recovered(msg) => {
            panic!("expected Abort for CompletionError, got Recovered({msg:?})")
        }
    }

    // History and trace should be untouched. (trace is no longer borrowed
    // after the call, so we can read it now.)
    assert!(trace.steps.is_empty());

    // Drop the loop to release the &mut history borrow.
    drop_loop(react_loop);
    assert!(history.is_empty());
}
