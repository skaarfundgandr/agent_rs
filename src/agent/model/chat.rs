use rig::agent::{Agent, StreamingPromptRequest};
use rig::completion::{Chat, CompletionModel, PromptError};
use rig::message::Message;
use rig::streaming::StreamingChat;
use rig::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Executes a standard chat turn against the LLM.
pub async fn execute_chat<
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    P: rig::agent::PromptHook<M> + WasmCompatSend + WasmCompatSync + 'static,
>(
    agent: &Agent<M, P>,
    prompt: &str,
    history: Vec<Message>,
) -> Result<String, PromptError> {
    agent.chat(prompt, history).await
}

/// Executes a streaming chat turn against the LLM, preparing the StreamingPromptRequest.
pub fn execute_stream_chat<M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static, P>(
    agent: &Agent<M, P>,
    prompt: &str,
    history: Vec<Message>,
) -> StreamingPromptRequest<M, P>
where
    M::StreamingResponse: rig::completion::GetTokenUsage,
    Agent<M, P>: StreamingChat<M, M::StreamingResponse, Hook = P>,
    P: rig::agent::PromptHook<M> + 'static,
{
    agent.stream_chat(prompt, history)
}
