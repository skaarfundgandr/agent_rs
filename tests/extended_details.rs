//! Integration tests for the opt-in `.extended_details()` telemetry APIs on
//! ReAct and managed agents: usage aggregation, globally contiguous
//! completion-call indices across cycles/runs, raw provider payload capture,
//! history semantics, serde round-trips, and idempotency.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/mod.rs"]
mod common;

use agent_rs::agent::{
    BuiltManagedAgent, BuiltReAct, ExtendedChatDetails, ExtendedReActTrace, ManagedChatDetails,
    ManagedExt, ManagedPromptDetails, ReActExt,
};
use agent_rs::domain::agent::ReActTrace;
use common::{EchoTool, mock_history};
use rig_core::OneOrMany;
use rig_core::agent::{AgentBuilder, CompletionCall, PromptResponse};
use rig_core::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Usage,
};
use rig_core::message::{AssistantContent, Message, ToolCall, ToolFunction, UserContent};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Telemetry-capable mock: scripted content + per-call usage + raw payloads.
// ---------------------------------------------------------------------------

/// One scripted outcome for [`TelemetryMockModel::completion`].
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
enum TelemetryScript {
    /// Successful completion with scripted content, usage, and raw payload.
    Ok {
        content: OneOrMany<AssistantContent>,
        usage: Usage,
        raw: serde_json::Value,
    },
    /// Failed completion; `retryable` maps to `ProviderError`, else `JsonError`.
    Fail { retryable: bool },
}

/// Queue-driven mock that reports per-call usage and a scripted raw payload.
///
/// `Response = serde_json::Value` so the scripted `raw` becomes the raw
/// provider payload captured by `CaptureTelemetryHook`.
#[derive(Clone, Debug)]
struct TelemetryMockModel {
    queue: Arc<Mutex<VecDeque<TelemetryScript>>>,
}

impl TelemetryMockModel {
    fn new(scripts: Vec<TelemetryScript>) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::from(scripts))),
        }
    }
}

impl TelemetryScript {
    fn text(content: &str, usage: Usage, raw: serde_json::Value) -> TelemetryScript {
        TelemetryScript::Ok {
            content: OneOrMany::one(AssistantContent::text(content)),
            usage,
            raw,
        }
    }

    fn tool_call(
        call_id: &str,
        name: &str,
        args: serde_json::Value,
        usage: Usage,
        raw: serde_json::Value,
    ) -> TelemetryScript {
        TelemetryScript::Ok {
            content: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                call_id.to_string(),
                ToolFunction::new(name.to_string(), args),
            ))),
            usage,
            raw,
        }
    }

    fn fail(retryable: bool) -> TelemetryScript {
        TelemetryScript::Fail { retryable }
    }
}

impl CompletionModel for TelemetryMockModel {
    type Response = serde_json::Value;
    type StreamingResponse = ();
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>) -> Self {
        Self::new(vec![])
    }

    fn completion(
        &self,
        _request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<CompletionResponse<Self::Response>, CompletionError>,
    > + Send {
        let queue = self.queue.clone();
        async move {
            let mut q = queue.lock().expect("telemetry mock lock poisoned");
            match q.pop_front() {
                Some(TelemetryScript::Ok {
                    content,
                    usage,
                    raw,
                }) => Ok(CompletionResponse {
                    choice: content,
                    usage,
                    raw_response: raw,
                    message_id: None,
                }),
                Some(TelemetryScript::Fail { retryable }) => {
                    if retryable {
                        Err(CompletionError::ProviderError(
                            "telemetry mock: scripted retryable failure".into(),
                        ))
                    } else {
                        let json_err: serde_json::Error =
                            serde_json::from_str::<serde_json::Value>("invalid").unwrap_err();
                        Err(CompletionError::JsonError(json_err))
                    }
                }
                None => Err(CompletionError::ProviderError(
                    "telemetry mock: no more responses configured".into(),
                )),
            }
        }
    }

    // Trait mandates the `impl Future` return form; `async fn` would not
    // match the declared signature.
    #[allow(clippy::manual_async_fn)]
    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<
            rig_core::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
            CompletionError,
        >,
    > + Send {
        async move {
            Err(CompletionError::ProviderError(
                "streaming not configured".into(),
            ))
        }
    }
}

/// Build an agent with an `echo` tool and the given scripted queue.
fn telemetry_agent(scripts: Vec<TelemetryScript>) -> rig_core::agent::Agent<TelemetryMockModel> {
    let model = TelemetryMockModel::new(scripts);
    AgentBuilder::new(model)
        .tool(EchoTool)
        .default_max_turns(20)
        .build()
}

fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + output,
        ..Usage::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Compile-level proof that the `Standard` surfaces keep their original
/// signatures: plain `ReActTrace` / `String` returns.
#[tokio::test]
async fn standard_signatures_unchanged() -> Result<(), Box<dyn std::error::Error>> {
    let agent = telemetry_agent(vec![TelemetryScript::text(
        "Final Answer: pong",
        usage(10, 5),
        serde_json::json!({"tag": "prompt"}),
    )]);
    let built: BuiltReAct<TelemetryMockModel> = agent.react().max_cycles(3).build();
    let trace: ReActTrace = built.prompt("ping").await?;
    assert_eq!(
        trace.final_answer.expect("final answer").text,
        "Final Answer: pong"
    );

    let agent = telemetry_agent(vec![TelemetryScript::text(
        "Final Answer: chat pong",
        usage(10, 5),
        serde_json::json!({"tag": "chat"}),
    )]);
    let built: BuiltReAct<TelemetryMockModel> = agent.react().max_cycles(3).build();
    let mut history = Vec::new();
    let text: String = built.chat("ping", &mut history).await?;
    assert_eq!(text, "Final Answer: chat pong");

    let agent = telemetry_agent(vec![TelemetryScript::text(
        "Final Answer: managed pong",
        usage(10, 5),
        serde_json::json!({"tag": "managed"}),
    )]);
    let managed: BuiltManagedAgent<TelemetryMockModel> = agent.managed().build();
    let s: String = managed.prompt("ping").await?;
    assert_eq!(s, "Final Answer: managed pong");

    Ok(())
}

/// One agent_rs cycle containing two rig-internal completions (tool call then
/// final text) aggregates usage field-wise, keeps call indices contiguous, and
/// captures both raw payloads in order.
#[tokio::test]
async fn react_extended_prompt_aggregates_usage() -> Result<(), Box<dyn std::error::Error>> {
    let u1 = usage(10, 5);
    let u2 = usage(20, 7);
    let r1 = serde_json::json!({"id": "completion-1", "kind": "tool_call"});
    let r2 = serde_json::json!({"id": "completion-2", "kind": "text"});
    let agent = telemetry_agent(vec![
        TelemetryScript::tool_call(
            "tc-1",
            "echo",
            serde_json::json!({"text": "hello"}),
            u1,
            r1.clone(),
        ),
        TelemetryScript::text("Final Answer: world", u2, r2.clone()),
    ]);
    let built = agent.react().max_cycles(3).build().extended_details();
    let details = built.prompt("test").await?;

    assert_eq!(details.completion_calls.len(), 2);
    assert_eq!(details.completion_calls[0].call_index, 0);
    assert_eq!(details.completion_calls[1].call_index, 1);
    assert_eq!(details.completion_calls[0].usage, u1);
    assert_eq!(details.completion_calls[1].usage, u2);
    assert_eq!(details.usage, u1 + u2);
    assert_eq!(details.raw_responses, vec![r1, r2]);
    let fa = details.trace.final_answer.expect("final answer");
    assert_eq!(fa.text, "Final Answer: world");
    assert_eq!(fa.cycles, 1);

    Ok(())
}

/// Two agent_rs-level model calls (empty-text output retried as a second rig
/// run) still produce a globally contiguous 0..n call index sequence and a
/// usage sum across both runs.
#[tokio::test]
async fn react_extended_offsets_indices_across_cycles() -> Result<(), Box<dyn std::error::Error>> {
    let u1 = usage(11, 6);
    let u2 = usage(22, 9);
    let r1 = serde_json::json!({"run": 1});
    let r2 = serde_json::json!({"run": 2});
    let agent = telemetry_agent(vec![
        TelemetryScript::text("", u1, r1.clone()),
        TelemetryScript::text("Final Answer: recovered", u2, r2.clone()),
    ]);
    let built = agent.react().max_cycles(3).build().extended_details();
    let details = built.prompt("test").await?;

    assert_eq!(details.completion_calls.len(), 2);
    assert_eq!(details.completion_calls[0].call_index, 0);
    assert_eq!(details.completion_calls[1].call_index, 1);
    assert_eq!(details.completion_calls[0].usage, u1);
    assert_eq!(details.completion_calls[1].usage, u2);
    assert_eq!(details.usage, u1 + u2);
    assert_eq!(details.raw_responses, vec![r1, r2]);
    assert_eq!(
        details.trace.final_answer.expect("final answer").text,
        "Final Answer: recovered"
    );

    Ok(())
}

/// Extended `chat` snapshots the final working history: `details.history`
/// equals the mutated `&mut` argument, which grew during the run.
#[tokio::test]
async fn react_extended_chat_history_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    let u = usage(13, 8);
    let raw = serde_json::json!({"id": "chat-run"});
    let agent = telemetry_agent(vec![TelemetryScript::text(
        "Final Answer: chat ok",
        u,
        raw.clone(),
    )]);
    let built = agent.react().max_cycles(3).build().extended_details();

    let mut history = mock_history(&["earlier turn"]);
    let before = history.len();
    let details: ExtendedChatDetails = built.chat("hello", &mut history).await?;

    assert!(!details.output.is_empty());
    assert_eq!(details.output, "Final Answer: chat ok");
    assert_eq!(details.usage, u);
    assert_eq!(details.completion_calls.len(), 1);
    assert_eq!(details.completion_calls[0].call_index, 0);
    assert_eq!(details.raw_responses, vec![raw]);
    assert_eq!(details.history, history);
    assert!(history.len() > before, "caller history should have grown");
    assert_eq!(history.len(), 3);

    Ok(())
}

/// Managed `prompt` retries a retryable failure; the failed attempt
/// contributes nothing to usage, call indices, or raw payloads.
#[tokio::test]
async fn managed_extended_prompt_retry_aggregates() -> Result<(), Box<dyn std::error::Error>> {
    let u = usage(30, 9);
    let r = serde_json::json!({"provider": "mock", "attempt": 2});
    let agent = telemetry_agent(vec![
        TelemetryScript::fail(true),
        TelemetryScript::text("Final Answer: after retry", u, r.clone()),
    ]);
    let built = agent.managed().max_retries(3).build().extended_details();
    let details: ManagedPromptDetails = built.prompt("test").await?;

    assert_eq!(details.response.output, "Final Answer: after retry");
    assert_eq!(details.response.usage, u);
    assert_eq!(details.response.completion_calls.len(), 1);
    assert_eq!(details.response.completion_calls[0].call_index, 0);
    assert_eq!(details.response.completion_calls[0].usage, u);
    assert_eq!(details.raw_responses, vec![r]);

    Ok(())
}

/// Extended managed `chat`: the caller's history grows by exactly one
/// user+assistant pair, while `details.history` holds the full rig transcript
/// (including internal tool turns) when tools ran.
#[tokio::test]
async fn managed_extended_chat_details() -> Result<(), Box<dyn std::error::Error>> {
    let u1 = usage(15, 4);
    let u2 = usage(25, 6);
    let r1 = serde_json::json!({"id": "m-tool"});
    let r2 = serde_json::json!({"id": "m-final"});
    let agent = telemetry_agent(vec![
        TelemetryScript::tool_call(
            "tc-1",
            "echo",
            serde_json::json!({"text": "hi"}),
            u1,
            r1.clone(),
        ),
        TelemetryScript::text("Final Answer: managed chat", u2, r2.clone()),
    ]);
    let built = agent.managed().max_retries(3).build().extended_details();

    let mut history = mock_history(&["earlier turn"]);
    let before = history.len();
    let details: ManagedChatDetails = built.chat("hi", &mut history).await?;

    // Caller history: exactly one user+assistant pair appended.
    assert_eq!(history.len(), before + 2);
    assert_eq!(history[0], Message::user("earlier turn"));
    assert_eq!(history[1], Message::user("hi"));
    assert_eq!(history[2], Message::assistant("Final Answer: managed chat"));

    assert_eq!(details.output, "Final Answer: managed chat");
    assert_eq!(details.usage, u1 + u2);
    assert_eq!(details.completion_calls.len(), 2);
    assert_eq!(details.completion_calls[0].call_index, 0);
    assert_eq!(details.completion_calls[1].call_index, 1);
    assert_eq!(details.completion_calls[0].usage, u1);
    assert_eq!(details.completion_calls[1].usage, u2);
    assert_eq!(details.raw_responses, vec![r1, r2]);

    // Full working transcript: longer than the caller's pair-only history and
    // it contains the internal tool call + tool result turns.
    assert_eq!(details.history.len(), 5);
    assert!(details.history.len() > history.len());
    assert_eq!(details.history[0], history[0]);
    assert!(details.history.iter().any(|m| matches!(
        m,
        Message::Assistant { content, .. }
            if content.iter().any(|c| matches!(c, AssistantContent::ToolCall(_)))
    )));
    assert!(details.history.iter().any(|m| matches!(
        m,
        Message::User { content }
            if content.iter().any(|c| matches!(c, UserContent::ToolResult(_)))
    )));

    Ok(())
}

/// All four detail types survive `to_value -> from_value -> to_value` with
/// byte-identical JSON.
#[tokio::test]
async fn serde_round_trip_all_detail_types() -> Result<(), Box<dyn std::error::Error>> {
    let u = usage(40, 12);
    let u2 = usage(8, 3);
    let completion_calls = vec![CompletionCall::new(0, u), CompletionCall::new(1, u2)];
    let raw = serde_json::json!({"provider": "mock", "usage": {"in": 40, "out": 12}});

    // 1. ExtendedReActTrace
    let trace_details = ExtendedReActTrace {
        trace: ReActTrace::default(),
        usage: u + u2,
        completion_calls: completion_calls.clone(),
        raw_responses: vec![raw.clone()],
    };
    round_trip(&trace_details)?;

    // 2. ExtendedChatDetails
    let chat_details = ExtendedChatDetails {
        output: "chat output".to_string(),
        usage: u + u2,
        completion_calls: completion_calls.clone(),
        raw_responses: vec![raw.clone()],
        history: mock_history(&["a", "b"]),
    };
    round_trip(&chat_details)?;

    // 3. ManagedChatDetails
    let managed_chat = ManagedChatDetails {
        output: "managed chat output".to_string(),
        usage: u + u2,
        completion_calls: completion_calls.clone(),
        raw_responses: vec![raw.clone()],
        history: mock_history(&["earlier"]),
    };
    round_trip(&managed_chat)?;

    // 4. ManagedPromptDetails — PromptResponse is #[non_exhaustive], so build
    // it by deserializing rig's documented JSON shape. `content` needs a
    // non-empty array (OneOrMany rejects empty), `messages` may be null.
    let usage_json = serde_json::json!({
        "input_tokens": 40,
        "output_tokens": 12,
        "total_tokens": 52,
        "cached_input_tokens": 1,
        "cache_creation_input_tokens": 2,
        "tool_use_prompt_tokens": 3,
        "reasoning_tokens": 4,
    });
    let prompt_response_json = serde_json::json!({
        "output": "hello world",
        "usage": usage_json,
        "completion_calls": [ { "call_index": 0, "usage": usage_json } ],
        "messages": null,
        "content": [ { "text": "hello world" } ],
    });
    let response: PromptResponse = serde_json::from_value(prompt_response_json.clone())?;
    let managed_prompt = ManagedPromptDetails {
        response,
        raw_responses: vec![raw],
    };
    let value = serde_json::to_value(&managed_prompt)?;
    let decoded: ManagedPromptDetails = serde_json::from_value(value.clone())?;
    let reencoded = serde_json::to_value(&decoded)?;
    assert_eq!(value, reencoded);
    assert_eq!(
        serde_json::to_value(&decoded.response)?,
        prompt_response_json
    );

    Ok(())
}

/// `extended_details()` is idempotent on both builders.
#[tokio::test]
async fn extended_details_idempotent() -> Result<(), Box<dyn std::error::Error>> {
    let agent = telemetry_agent(vec![TelemetryScript::text(
        "Final Answer: idem react",
        usage(5, 2),
        serde_json::json!({"idem": true}),
    )]);
    let details: ExtendedReActTrace = agent
        .react()
        .max_cycles(3)
        .build()
        .extended_details()
        .extended_details()
        .prompt("test")
        .await?;
    assert_eq!(
        details.trace.final_answer.expect("final answer").text,
        "Final Answer: idem react"
    );

    let agent = telemetry_agent(vec![TelemetryScript::text(
        "Final Answer: idem managed",
        usage(5, 2),
        serde_json::json!({"idem": true}),
    )]);
    let details: ManagedPromptDetails = agent
        .managed()
        .build()
        .extended_details()
        .extended_details()
        .prompt("test")
        .await?;
    assert_eq!(details.response.output, "Final Answer: idem managed");

    Ok(())
}

/// The compaction variant `prompt_compact` returns populated telemetry
/// details (no compaction triggered at a 1000-token threshold).
#[tokio::test]
async fn react_extended_prompt_compact() -> Result<(), Box<dyn std::error::Error>> {
    let u = usage(17, 6);
    let raw = serde_json::json!({"compact": true});
    let agent = telemetry_agent(vec![TelemetryScript::text(
        "Final Answer: compacted world",
        u,
        raw.clone(),
    )]);
    let built = agent
        .react()
        .with_compaction()
        .threshold(1000)
        .build()
        .extended_details();
    let details: ExtendedReActTrace = built.prompt_compact("test").await?;

    assert_eq!(
        details.trace.final_answer.expect("final answer").text,
        "Final Answer: compacted world"
    );
    assert_eq!(details.usage, u);
    assert_eq!(details.completion_calls.len(), 1);
    assert_eq!(details.completion_calls[0].call_index, 0);
    assert_eq!(details.completion_calls[0].usage, u);
    assert_eq!(details.raw_responses, vec![raw]);

    Ok(())
}

fn round_trip<T>(value: &T) -> Result<(), Box<dyn std::error::Error>>
where
    T: serde::Serialize + serde::de::DeserializeOwned + std::fmt::Debug,
{
    let encoded = serde_json::to_value(value)?;
    let decoded: T = serde_json::from_value(encoded.clone())?;
    let reencoded = serde_json::to_value(&decoded)?;
    assert_eq!(encoded, reencoded);
    Ok(())
}

/// A retryable failure mid-run discards the raw payloads the capture hook had
/// already collected for the failed run's successful turns: only the
/// successful attempt's payloads survive.
#[tokio::test]
async fn managed_extended_truncates_raws_of_failed_run() -> Result<(), Box<dyn std::error::Error>> {
    let u_failed = usage(9, 3);
    let r_failed = serde_json::json!({"id": "failed-run-turn"});
    let u_ok = usage(31, 11);
    let r_ok = serde_json::json!({"id": "successful-run"});
    let agent = telemetry_agent(vec![
        // Attempt 1: a successful tool-call turn (hook fires, raw captured)...
        TelemetryScript::tool_call(
            "tc-1",
            "echo",
            serde_json::json!({"text": "hi"}),
            u_failed,
            r_failed,
        ),
        // ...then a retryable failure aborts the run.
        TelemetryScript::fail(true),
        // Attempt 2 succeeds.
        TelemetryScript::text("Final Answer: survived", u_ok, r_ok.clone()),
    ]);
    let built = agent.managed().max_retries(3).build().extended_details();
    let details: ManagedPromptDetails = built.prompt("test").await?;

    assert_eq!(details.response.output, "Final Answer: survived");
    assert_eq!(details.response.usage, u_ok);
    assert_eq!(details.response.completion_calls.len(), 1);
    assert_eq!(details.response.completion_calls[0].usage, u_ok);
    assert_eq!(details.raw_responses, vec![r_ok]);

    Ok(())
}

/// Turn-limit recovery: raw payloads captured before a `MaxTurnsError` are
/// truncated away, and only the recovered run's telemetry survives with
/// contiguous indices.
#[tokio::test]
async fn react_extended_turn_limit_recovery_truncates() -> Result<(), Box<dyn std::error::Error>> {
    let u_lost1 = usage(7, 2);
    let r_lost1 = serde_json::json!({"id": "lost-1"});
    let u_lost2 = usage(8, 3);
    let r_lost2 = serde_json::json!({"id": "lost-2"});
    let u_ok = usage(19, 6);
    let r_ok = serde_json::json!({"id": "recovered-run"});

    let model = TelemetryMockModel::new(vec![
        // Run 1 exhausts its 2-call budget on tool calls -> MaxTurnsError.
        TelemetryScript::tool_call(
            "tc-1",
            "echo",
            serde_json::json!({"text": "a"}),
            u_lost1,
            r_lost1,
        ),
        TelemetryScript::tool_call(
            "tc-2",
            "echo",
            serde_json::json!({"text": "b"}),
            u_lost2,
            r_lost2,
        ),
        // Run 2 (after turn-limit recovery) reaches the final answer.
        TelemetryScript::text("Final Answer: recovered", u_ok, r_ok.clone()),
    ]);
    let agent = AgentBuilder::new(model)
        .tool(EchoTool)
        .default_max_turns(2)
        .build();
    let built = agent.react().max_cycles(5).build().extended_details();
    let details: ExtendedReActTrace = built.prompt("test").await?;

    assert_eq!(
        details.trace.final_answer.expect("final answer").text,
        "Final Answer: recovered"
    );
    assert_eq!(details.usage, u_ok);
    assert_eq!(details.completion_calls.len(), 1);
    assert_eq!(details.completion_calls[0].call_index, 0);
    assert_eq!(details.completion_calls[0].usage, u_ok);
    assert_eq!(details.raw_responses, vec![r_ok]);

    Ok(())
}

/// Managed compaction variants in the Extended state return populated details.
#[tokio::test]
async fn managed_extended_compact_variants() -> Result<(), Box<dyn std::error::Error>> {
    let u = usage(21, 7);
    let raw = serde_json::json!({"managed-compact": true});
    let agent = telemetry_agent(vec![TelemetryScript::text(
        "Final Answer: managed compact",
        u,
        raw.clone(),
    )]);
    let built = agent
        .managed()
        .with_compaction()
        .threshold(1000)
        .build()
        .extended_details();
    let details: ManagedPromptDetails = built.prompt_compact("test").await?;
    assert_eq!(details.response.output, "Final Answer: managed compact");
    assert_eq!(details.response.usage, u);
    assert_eq!(details.response.completion_calls.len(), 1);
    assert_eq!(details.raw_responses, vec![raw.clone()]);

    let agent = telemetry_agent(vec![TelemetryScript::text(
        "Final Answer: managed compact chat",
        u,
        raw.clone(),
    )]);
    let built = agent
        .managed()
        .with_compaction()
        .threshold(1000)
        .build()
        .extended_details();
    let mut history = mock_history(&["earlier"]);
    let details: ManagedChatDetails = built.chat_compact("hi", &mut history).await?;
    assert_eq!(details.output, "Final Answer: managed compact chat");
    assert_eq!(details.usage, u);
    assert_eq!(details.completion_calls.len(), 1);
    assert_eq!(details.raw_responses, vec![raw]);
    assert!(!details.history.is_empty());
    assert_eq!(
        history.last(),
        Some(&Message::assistant("Final Answer: managed compact chat"))
    );

    Ok(())
}

/// Failed extended chat runs leave the caller's history untouched.
#[tokio::test]
async fn extended_chat_error_leaves_history_untouched() -> Result<(), Box<dyn std::error::Error>> {
    let agent = telemetry_agent(vec![TelemetryScript::fail(false)]);
    let built = agent.react().max_cycles(3).build().extended_details();
    let mut history = mock_history(&["earlier"]);
    let snapshot = history.clone();
    assert!(built.chat("hi", &mut history).await.is_err());
    assert_eq!(history, snapshot);

    let agent = telemetry_agent(vec![TelemetryScript::fail(false)]);
    let built = agent.managed().max_retries(3).build().extended_details();
    let mut history = mock_history(&["earlier"]);
    let snapshot = history.clone();
    assert!(built.chat("hi", &mut history).await.is_err());
    assert_eq!(history, snapshot);

    Ok(())
}
