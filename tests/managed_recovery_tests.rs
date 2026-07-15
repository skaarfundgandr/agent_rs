#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/mod.rs"]
mod common;

use agent_rs::agent::{InvalidToolPolicy, ManagedExt, invalid_tool_feedback};
use common::{EchoTool, MockCompletionModel};
use rig_core::agent::AgentBuilder;

// ---------------------------------------------------------------------------
// Builder budget tests (unit, no model interaction)
// ---------------------------------------------------------------------------

#[test]
fn test_managed_invalid_tool_policy_defaults() {
    let model = MockCompletionModel::new(vec![]);
    let agent = AgentBuilder::new(model).tool(EchoTool).build();
    let b = agent.managed().build();
    assert_eq!(b.invalid_tool_policy(), InvalidToolPolicy::Skip);
    assert_eq!(b.max_invalid_tool_call_retries(), 0);
}

#[test]
fn test_managed_retry_auto_budget_2() {
    let model = MockCompletionModel::new(vec![]);
    let agent = AgentBuilder::new(model).tool(EchoTool).build();
    let b = agent
        .managed()
        .invalid_tool_policy(InvalidToolPolicy::Retry)
        .build();
    assert_eq!(b.max_invalid_tool_call_retries(), 2);
}

#[test]
fn test_managed_explicit_budget_wins() {
    let model = MockCompletionModel::new(vec![]);
    let agent = AgentBuilder::new(model).tool(EchoTool).build();
    let b = agent
        .managed()
        .max_invalid_tool_call_retries(5)
        .invalid_tool_policy(InvalidToolPolicy::Retry)
        .build();
    assert_eq!(b.max_invalid_tool_call_retries(), 5);
}

#[test]
fn test_managed_explicit_zero_after_retry() {
    let model = MockCompletionModel::new(vec![]);
    let agent = AgentBuilder::new(model).tool(EchoTool).build();
    let b = agent
        .managed()
        .invalid_tool_policy(InvalidToolPolicy::Retry)
        .max_invalid_tool_call_retries(0)
        .build();
    assert_eq!(b.max_invalid_tool_call_retries(), 0);
}

// ---------------------------------------------------------------------------
// Skip policy: invalid tool call skip-feedback is injected, loop continues.
// Managed does NOT recover MaxTurnsError, so default_max_turns must be >=3.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_managed_skip_recovers() {
    let model = MockCompletionModel::new(vec![
        MockCompletionModel::tool_call("tc-1", "bogus_tool", serde_json::json!({})),
        MockCompletionModel::text("ok recovered"),
    ]);
    let agent = AgentBuilder::new(model)
        .tool(EchoTool)
        .default_max_turns(3)
        .build();
    let built = agent.managed().build();
    let out = built
        .prompt("hi")
        .await
        .expect("should recover from invalid tool");
    assert!(
        out.contains("ok recovered") || out.contains("recovered"),
        "expected output mentioning recovery, got: {out}"
    );
}

// ---------------------------------------------------------------------------
// Fail policy: invalid tool call immediately errors.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_managed_fail_hard_fails() {
    let model = MockCompletionModel::new(vec![MockCompletionModel::tool_call(
        "tc-1",
        "bogus_tool",
        serde_json::json!({}),
    )]);
    let agent = AgentBuilder::new(model)
        .tool(EchoTool)
        .default_max_turns(3)
        .build();
    let built = agent
        .managed()
        .invalid_tool_policy(InvalidToolPolicy::Fail)
        .build();
    let err = built.prompt("hi").await.unwrap_err();
    let s = err.to_string();
    assert!(
        s.contains("bogus_tool") || s.contains("unknown") || s.contains("UnknownToolCall"),
        "expected error mentioning tool name, got: {s}"
    );
}

// ---------------------------------------------------------------------------
// Retry policy with auto-budget (2): invalid tool retried, then succeeds.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_managed_retry_recovers() {
    let model = MockCompletionModel::new(vec![
        MockCompletionModel::tool_call("tc-1", "bogus_tool", serde_json::json!({})),
        MockCompletionModel::tool_call("tc-2", "echo", serde_json::json!({"text": "hello"})),
        MockCompletionModel::text("Final Answer: done"),
    ]);
    let agent = AgentBuilder::new(model)
        .tool(EchoTool)
        .default_max_turns(5)
        .build();
    let built = agent
        .managed()
        .invalid_tool_policy(InvalidToolPolicy::Retry)
        .build();
    let out = built.prompt("hi").await.expect("retry should recover");
    assert!(
        out.contains("done") || out.contains("Final"),
        "expected final output, got: {out}"
    );
}

// ---------------------------------------------------------------------------
// Dead Retry (explicit budget 0): no retry budget, fails on invalid tool.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_managed_dead_retry_fails() {
    let model = MockCompletionModel::new(vec![MockCompletionModel::tool_call(
        "tc-1",
        "bogus_tool",
        serde_json::json!({}),
    )]);
    let agent = AgentBuilder::new(model)
        .tool(EchoTool)
        .default_max_turns(3)
        .build();
    let built = agent
        .managed()
        .invalid_tool_policy(InvalidToolPolicy::Retry)
        .max_invalid_tool_call_retries(0)
        .build();
    let err = built.prompt("hi").await.unwrap_err();
    let s = err.to_string();
    assert!(
        s.contains("bogus_tool") || s.contains("unknown") || s.contains("UnknownToolCall"),
        "expected error mentioning tool name, got: {s}"
    );
}

// ---------------------------------------------------------------------------
// Feedback unit test
// ---------------------------------------------------------------------------

#[test]
fn test_invalid_tool_feedback_string() {
    let s = invalid_tool_feedback("x", &["echo".into()]);
    assert!(s.contains("echo"));
    assert!(s.contains("x"));
    assert!(
        s.contains("unknown") || s.contains("not allowed"),
        "expected feedback mentioning unknown/not allowed, got: {s}"
    );
}
