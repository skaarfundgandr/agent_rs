use rig_core::agent::{Agent, PromptRequest, PromptResponse, StreamingPromptRequest};
use rig_core::completion::{CompletionModel, PromptError};
use rig_core::message::Message;
use rig_core::streaming::StreamingChat;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

use crate::agent::telemetry::{CaptureTelemetryHook, TelemetryAccum};

pub async fn execute_chat<M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static>(
    agent: &Agent<M>,
    prompt: &str,
    history: &mut Vec<Message>,
    max_invalid_tool_call_retries: u32,
) -> Result<String, PromptError> {
    let mut req = PromptRequest::from_agent(agent, prompt)
        .history(history.clone())
        .extended_details();
    if max_invalid_tool_call_retries > 0 {
        req = req.max_invalid_tool_call_retries(max_invalid_tool_call_retries as usize);
    }
    let response = req.await?;
    if let Some(messages) = response.messages {
        history.extend(messages);
    }
    Ok(response.output)
}

/// Like [`execute_chat`] but returns the full [`PromptResponse`] and captures provider-native payloads into `accum`.
pub(crate) async fn execute_chat_details<
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
>(
    agent: &Agent<M>,
    prompt: &str,
    history: &mut Vec<Message>,
    max_invalid_tool_call_retries: u32,
    accum: &mut TelemetryAccum,
) -> Result<PromptResponse, PromptError> {
    let mut req = PromptRequest::from_agent(agent, prompt)
        .history(history.clone())
        .extended_details();
    if max_invalid_tool_call_retries > 0 {
        req = req.max_invalid_tool_call_retries(max_invalid_tool_call_retries as usize);
    }
    let raw_len_before = accum.raw_len();
    req = req.add_hook(CaptureTelemetryHook::new(accum.raw_handle()));
    let response = match req.await {
        Ok(response) => response,
        Err(e) => {
            accum.truncate_raw(raw_len_before);
            return Err(e);
        }
    };
    if let Some(messages) = &response.messages {
        history.extend(messages.iter().cloned());
    }
    Ok(response)
}

pub fn execute_stream_chat<M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static>(
    agent: &Agent<M>,
    prompt: &str,
    history: Vec<Message>,
) -> StreamingPromptRequest<M>
where
    M::StreamingResponse: rig_core::completion::GetTokenUsage,
{
    agent.stream_chat(prompt, history)
}
