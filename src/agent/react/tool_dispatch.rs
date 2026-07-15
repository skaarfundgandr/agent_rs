use std::sync::Arc;
use std::time::{Duration, Instant};

use rig_core::agent::Agent;
use rig_core::completion::CompletionModel;
use rig_core::message::{Message, ToolCall, ToolResultContent, UserContent};
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::domain::agent::{Action, Observation, ReActStep, ReActTrace};
use crate::domain::errors::ReActError;

use super::callbacks::{ActionCb, ObservationCb};
use super::emitter::ReActSpanEmitter;
use super::helpers::tool_error_to_string;

/// Result of dispatching tool calls for a single ReAct cycle.
pub(crate) struct ToolDispatchResult {
    /// The prompt message for the next cycle (the last tool result).
    pub next_prompt: Option<Message>,
    /// Intermediate tool result messages to append to `working_history`
    /// (all tool results except the last one).
    pub history_extensions: Vec<Message>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn dispatch_tool_calls<M>(
    agent: &Agent<M>,
    tool_calls: &[&ToolCall],
    cycle: usize,
    tool_timeout_secs: u64,
    on_action: &Option<ActionCb>,
    on_observation: &Option<ObservationCb>,
    span_emitter: &Arc<dyn ReActSpanEmitter>,
    trace: &mut ReActTrace,
) -> Result<ToolDispatchResult, ReActError>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
{
    let mut next_prompt = None;
    let mut history_extensions = Vec::new();
    let num_tool_calls = tool_calls.len();

    for (i, tc) in tool_calls.iter().enumerate() {
        let args_str = tc.function.arguments.to_string();
        let action = Action {
            tool_name: tc.function.name.clone(),
            args: args_str.clone(),
            tool_call_id: Some(tc.id.clone()),
            cycle,
        };
        if let Some(cb) = on_action {
            cb(&action);
        }
        span_emitter.emit_action(&action);
        trace.steps.push(ReActStep::Action(action.clone()));

        let start = Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(tool_timeout_secs),
            agent
                .tool_server_handle
                .call_tool(&tc.function.name, &args_str),
        )
        .await;
        let duration = start.elapsed();

        let observation = match result {
            Ok(Ok(s)) => Observation {
                tool_name: tc.function.name.clone(),
                result: s,
                is_error: false,
                cycle,
                duration,
            },
            Ok(Err(e)) => Observation {
                tool_name: tc.function.name.clone(),
                result: tool_error_to_string(&e),
                is_error: true,
                cycle,
                duration,
            },
            Err(_elapsed) => Observation {
                tool_name: tc.function.name.clone(),
                result: format!(
                    "Tool '{}' timed out after {}s",
                    tc.function.name, tool_timeout_secs
                ),
                is_error: true,
                cycle,
                duration,
            },
        };

        if let Some(cb) = on_observation {
            cb(&observation);
        }
        span_emitter.emit_observation(&observation);
        trace
            .steps
            .push(ReActStep::Observation(observation.clone()));

        let call_id = tc
            .call_id
            .clone()
            .unwrap_or_else(|| format!("react-cycle-{cycle}"));
        let tool_result_content = ToolResultContent::text(observation.result.clone());
        let user_content = UserContent::tool_result_with_call_id(
            tc.id.clone(),
            call_id,
            rig_core::OneOrMany::one(tool_result_content),
        );
        let msg = Message::User {
            content: rig_core::OneOrMany::one(user_content),
        };

        if i == num_tool_calls - 1 {
            next_prompt = Some(msg);
        } else {
            history_extensions.push(msg);
        }
    }

    Ok(ToolDispatchResult {
        next_prompt,
        history_extensions,
    })
}
