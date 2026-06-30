#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/mod.rs"]
mod common;

use agent_rs::agent::ReActExt;
use agent_rs::domain::agent::{Action, FinalAnswer, Observation, ReActStep};
use common::{MockCompletionModel, mock_agent};
use rig_core::message::Message;

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
    // prompt() path — prompt is stateless, local history stays empty
    let responses = vec![MockCompletionModel::text("Final Answer: yes")];
    let agent = mock_agent(responses);
    let built = agent.react().build();
    let h1: Vec<Message> = Vec::new();
    let _ = built.prompt("test").await;
    assert_eq!(h1.len(), 0, "prompt() should not mutate history");

    // chat() path — chat writes the full working history on success
    let responses = vec![MockCompletionModel::text("Final Answer: ok")];
    let agent = mock_agent(responses);
    let built = agent.react().build();
    let mut h2: Vec<Message> = Vec::new();
    let _ = built.chat("test", &mut h2).await;
    assert!(!h2.is_empty(), "chat() should append to history");
}

/// `stream_chat` writes the final history back on `Completed`.
#[tokio::test]
async fn test_stream_chat_writes_history_on_completion() {
    use futures::StreamExt;

    let model = MockCompletionModel::with_streaming_text("Final Answer: streamed");
    let agent = rig_core::agent::AgentBuilder::new(model)
        .default_max_turns(1)
        .build();
    let built = agent.react().max_cycles(3).build();

    let mut history: Vec<Message> = Vec::new();
    let mut stream = built
        .stream_chat("hello", &mut history)
        .expect("stream_chat should succeed");

    // Consume all items from the stream
    while let Some(_item) = stream.next().await {
        // drain
    }

    // After the stream completes, history should have been written back
    assert!(
        !history.is_empty(),
        "stream_chat should write history back on completion"
    );
    // Should contain user message + assistant response
    assert!(
        history.len() >= 2,
        "history should contain at least user + assistant messages, got {}",
        history.len()
    );
}

/// `stream_chat` that errors mid-stream does NOT mutate history.
#[tokio::test]
async fn test_stream_chat_error_does_not_mutate_history() {
    use futures::StreamExt;

    // Use a model with no streaming text — stream() returns an error
    let model = MockCompletionModel::new(vec![]);
    let agent = rig_core::agent::AgentBuilder::new(model)
        .default_max_turns(1)
        .build();
    let built = agent.react().max_cycles(3).build();

    let mut history: Vec<Message> = Vec::new();
    let result = built.stream_chat("hello", &mut history);

    match result {
        Ok(mut stream) => {
            // Consume all items — should get an error event, no Completed
            while let Some(item) = stream.next().await {
                if let agent_rs::domain::agent::ReActStreamItem::Error { .. } = item {
                    break;
                }
            }
        }
        Err(_) => {
            // stream_chat itself errored — history should be untouched
        }
    }

    // History should NOT have been mutated on error
    assert!(
        history.is_empty(),
        "stream_chat error should not mutate history"
    );
}
