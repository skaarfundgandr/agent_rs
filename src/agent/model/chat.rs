use rig_core::agent::{Agent, StreamingPromptRequest};
use rig_core::completion::{Chat, CompletionModel, PromptError};
use rig_core::message::Message;
use rig_core::streaming::StreamingChat;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

pub async fn execute_chat<
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
>(
    agent: &Agent<M>,
    prompt: &str,
    history: &mut Vec<Message>,
) -> Result<String, PromptError> {
    agent.chat(prompt, history).await
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
