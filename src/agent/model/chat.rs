use rig_core::agent::{Agent, StreamingPromptRequest};
use rig_core::completion::{Chat, CompletionModel, PromptError};
use rig_core::message::Message;
use rig_core::streaming::StreamingChat;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Executes a standard chat turn against the LLM.
///
/// # Arguments
///
/// * `agent` - The underlying standard Rig agent instance.
/// * `prompt` - The user input prompt text.
/// * `history` - The conversation history as a vector of messages.
///
/// # Returns
///
/// Returns the response text from the LLM.
///
/// # Errors
///
/// Returns a `PromptError` if the underlying model invocation fails.
pub async fn execute_chat<
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig_core::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
>(
    agent: &Agent<M, P>,
    prompt: &str,
    history: &mut Vec<Message>,
) -> Result<String, PromptError> {
    agent.chat(prompt, history).await
}

/// Executes a streaming chat turn against the LLM, preparing the StreamingPromptRequest.
///
/// # Arguments
///
/// * `agent` - The underlying standard Rig agent instance.
/// * `prompt` - The user input prompt text.
/// * `history` - The conversation history as a vector of messages.
///
/// # Returns
///
/// Returns a `StreamingPromptRequest` which can be configured or run to obtain a stream.
pub fn execute_stream_chat<M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static, P>(
    agent: &Agent<M, P>,
    prompt: &str,
    history: Vec<Message>,
) -> StreamingPromptRequest<M, P>
where
    M::StreamingResponse: rig_core::completion::GetTokenUsage,
    Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    P: rig_core::agent::PromptHook<M> + 'static,
{
    agent.stream_chat(prompt, history)
}

