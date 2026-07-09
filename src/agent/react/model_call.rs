use std::sync::Arc;

use rig_core::agent::{Agent, PromptHook, PromptResponse};
use rig_core::completion::{CompletionModel, Prompt, PromptError};
use rig_core::message::Message;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::domain::agent::ReActTrace;
use crate::domain::errors::ReActError;

use super::callbacks::ErrorCb;
use super::emitter::ReActSpanEmitter;
use super::helpers::recover_turn_limit_history;

/// Result of a single model call attempt within a ReAct cycle.
pub(crate) enum ModelCallResult {
    /// The model responded successfully.
    Ok(PromptResponse),
    /// A turn-limit error was recovered — the caller should update
    /// `working_history`/`current_prompt` from the returned values and
    /// `continue` the outer cycle loop.
    TurnLimitRecovery {
        recovered_history: Vec<Message>,
        recovered_prompt: Message,
    },
    /// A non-recoverable error occurred.
    Err(ReActError),
}

/// Append `suffix` to the agent's system preamble for one call only.
fn agent_with_system_suffix<M, P>(agent: &Agent<M, P>, suffix: &str) -> Agent<M, P>
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    let mut agent = agent.clone();
    agent.preamble = Some(match agent.preamble.take() {
        Some(p) => format!("{p}\n\n<system-reminder>{suffix}\n</system-reminder>"),
        None => suffix.to_string(),
    });
    agent
}

/// Execute the model call with retry logic and turn-limit recovery.
///
/// When `system_suffix` is `Some`, it is appended to the agent's system
/// preamble for this call only (history and prompt are left unchanged).
///
/// Returns a [`ModelCallResult`] indicating success, turn-limit recovery, or
/// a fatal error.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_model_call<M, P>(
    agent: &Agent<M, P>,
    current_prompt: &Message,
    working_history: &[Message],
    max_retries: u32,
    cycle: usize,
    span_emitter: &Arc<dyn ReActSpanEmitter>,
    on_error: &Option<ErrorCb>,
    trace: &ReActTrace,
    system_suffix: Option<&str>,
) -> ModelCallResult
where
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
{
    let agent_for_call;
    let agent = match system_suffix {
        Some(suffix) => {
            agent_for_call = agent_with_system_suffix(agent, suffix);
            &agent_for_call
        }
        None => agent,
    };

    let mut attempt = 0u32;
    loop {
        attempt += 1;
        match agent
            .prompt(current_prompt.clone())
            .with_history(working_history.iter().cloned())
            .extended_details()
            .await
        {
            Ok(resp) => return ModelCallResult::Ok(resp),
            Err(e) => {
                let is_transient = crate::agent::retry::is_retryable(&e);
                let is_turn_limit = matches!(&e, PromptError::MaxTurnsError { .. });
                if is_transient && attempt < max_retries {
                    let delay = std::time::Duration::from_millis(500 * 2u64.pow(attempt - 1));
                    tokio::time::sleep(delay).await;
                    continue;
                }
                if is_turn_limit {
                    let err = ReActError::Model(e.to_string());
                    if let Some(cb) = on_error {
                        cb(&err);
                    }
                    span_emitter.emit_error(&err);
                    span_emitter.emit_cycle_end(cycle, trace);
                    if let Some(mut recovered) = recover_turn_limit_history(&e)
                        && let Some(last) = recovered.pop()
                    {
                        return ModelCallResult::TurnLimitRecovery {
                            recovered_history: recovered,
                            recovered_prompt: last,
                        };
                    }
                    return ModelCallResult::Err(err);
                }
                let err = ReActError::Model(e.to_string());
                if let Some(cb) = on_error {
                    cb(&err);
                }
                span_emitter.emit_error(&err);
                span_emitter.emit_cycle_end(cycle, trace);
                return ModelCallResult::Err(err);
            }
        }
    }
}
