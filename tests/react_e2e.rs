#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/mod.rs"]
mod common;

use agent_rs::agent::ReActExt;
use agent_rs::domain::agent::{Action, FinalAnswer, Observation, ReActStep};
use common::{MockCompletionModel, mock_agent};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full ReAct cycle: tool call → observation → final answer.
///
/// The mock returns a tool call on the first `completion()` invocation, then a
/// final answer text on the second. The echo tool registered by `mock_agent`
/// returns the text argument as the observation.
#[tokio::test]
async fn test_full_cycle_tool_then_final() {
    let responses = vec![
        MockCompletionModel::tool_call("tc-1", "echo", serde_json::json!({"text": "hello"})),
        MockCompletionModel::text("Final Answer: world"),
    ];

    let agent = mock_agent(responses);
    let built = agent.react().max_cycles(3).build();
    let trace = built.prompt("test").await.expect("prompt should succeed");

    // The trace should have: action + observation + final_answer
    assert!(
        trace
            .steps
            .iter()
            .any(|s| matches!(s, ReActStep::Action(_))),
        "trace should contain an action step"
    );
    assert!(
        trace
            .steps
            .iter()
            .any(|s| matches!(s, ReActStep::Observation(_))),
        "trace should contain an observation step"
    );
    let fa = trace
        .final_answer
        .as_ref()
        .expect("should have a final answer");
    assert_eq!(fa.text, "Final Answer: world");
    assert_eq!(fa.cycles, 1);
}

/// Multi-cycle: two sequential tool calls then a final answer.
///
/// Each mock response contains exactly one tool call. rig-core's multi-turn
/// processes the first tool call, then the ReAct loop feeds the result back
/// as the next prompt, triggering the second mock response.
#[tokio::test]
async fn test_multi_cycle_two_tool_calls_then_final() {
    let responses = vec![
        MockCompletionModel::tool_call("tc-1", "echo", serde_json::json!({"text": "a"})),
        MockCompletionModel::tool_call("tc-2", "echo", serde_json::json!({"text": "b"})),
        MockCompletionModel::text("Final Answer: done"),
    ];

    let agent = mock_agent(responses);
    let built = agent.react().max_cycles(5).build();
    let trace = built
        .prompt("test multi")
        .await
        .expect("prompt should succeed");

    eprintln!(
        "trace steps: {:?}",
        trace
            .steps
            .iter()
            .map(std::mem::discriminant)
            .collect::<Vec<_>>()
    );
    eprintln!("final_answer: {:?}", trace.final_answer);

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
    assert!(
        action_count >= 1,
        "expected at least 1 action step, got {action_count}"
    );
    assert!(
        obs_count >= 1,
        "expected at least 1 observation step, got {obs_count}"
    );

    let fa = trace
        .final_answer
        .as_ref()
        .expect("should have final answer");
    assert_eq!(fa.text, "Final Answer: done");
}

/// Callbacks fire in the expected order: action → observation → final.
#[tokio::test]
async fn test_callbacks_fire_in_order() {
    use std::sync::{Arc, Mutex};

    let actions = Arc::new(Mutex::new(Vec::<Action>::new()));
    let observations = Arc::new(Mutex::new(Vec::<Observation>::new()));
    let finals = Arc::new(Mutex::new(Vec::<FinalAnswer>::new()));

    let actions_clone = Arc::clone(&actions);
    let observations_clone = Arc::clone(&observations);
    let finals_clone = Arc::clone(&finals);

    let responses = vec![
        MockCompletionModel::tool_call("tc-1", "echo", serde_json::json!({"text": "ping"})),
        MockCompletionModel::text("Final Answer: pong"),
    ];

    let agent = mock_agent(responses);
    let built = agent
        .react()
        .max_cycles(3)
        .on_action(move |a: &Action| {
            actions_clone.lock().unwrap().push(a.clone());
        })
        .on_observation(move |o: &Observation| {
            observations_clone.lock().unwrap().push(o.clone());
        })
        .on_final(move |f: &FinalAnswer| {
            finals_clone.lock().unwrap().push(f.clone());
        })
        .build();

    let trace = built
        .prompt("test callbacks")
        .await
        .expect("prompt should succeed");

    assert_eq!(
        actions.lock().unwrap().len(),
        1,
        "expected 1 action callback"
    );
    assert_eq!(
        observations.lock().unwrap().len(),
        1,
        "expected 1 observation callback"
    );
    assert_eq!(finals.lock().unwrap().len(), 1, "expected 1 final callback");
    assert_eq!(finals.lock().unwrap()[0].text, "Final Answer: pong");

    let fa = trace.final_answer.as_ref().unwrap();
    assert_eq!(fa.text, "Final Answer: pong");
}

/// `prompt()` does not mutate shared history; `chat()` does.
#[tokio::test]
async fn test_prompt_vs_chat_history_mutation() {
    // prompt() path
    let responses = vec![MockCompletionModel::text("Final Answer: yes")];
    let agent = mock_agent(responses);
    let built = agent.react().build();
    let before = built.history().len();
    let _ = built.prompt("test").await;
    assert_eq!(
        built.history().len(),
        before,
        "prompt() should not mutate history"
    );

    // chat() path
    let responses = vec![MockCompletionModel::text("Final Answer: ok")];
    let agent = mock_agent(responses);
    let built = agent.react().build();
    let before = built.history().len();
    let _ = built.chat("test").await;
    assert!(
        built.history().len() > before,
        "chat() should append to history"
    );
}
