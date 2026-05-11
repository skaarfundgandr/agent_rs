use rig::agent::Agent;
use rig::completion::{Chat, CompletionModel, Prompt, PromptError};
use rig::message::Message;
use rig::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// An agent wrapper that automatically compacts conversation history
/// when it exceeds a specified token threshold.
pub struct ContextManagedAgent<M: CompletionModel, C: Prompt> {
    inner: Agent<M>,
    compaction_threshold: usize,
    compaction_model: C,
}

impl<
    M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static,
    C: Prompt + WasmCompatSend + WasmCompatSync + 'static,
> ContextManagedAgent<M, C>
{
    /// Send a chat prompt and automatically manage the context history.
    /// The history is mutated in-place:
    /// - If it exceeds the threshold, it is compacted into a summary.
    /// - The new prompt and the agent's response are automatically appended.
    pub async fn chat(
        &self,
        prompt: &str,
        history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        // 1. Approximate token count using JSON serialization
        let history_text = serde_json::to_string(&history).unwrap_or_default();
        let current_tokens = (history_text.chars().count() + prompt.chars().count()) / 4;

        // 2. Compact if threshold is exceeded and we have enough history to summarize
        if current_tokens > self.compaction_threshold && !history.is_empty() {
            let compaction_prompt = format!(
                "Summarize the following conversation history concisely, preserving key facts, \
                names, dates, and technical details. This summary will serve as the memory for \
                future interactions:\n\n{}",
                history_text
            );

            match self.compaction_model.prompt(&compaction_prompt).await {
                Ok(summary) => {
                    history.clear();
                    // Message::System has an `id` field in some rig versions? Let's use the standard `from` or try basic struct
                    history.push(Message::System { content: summary });
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        // 3. Call the underlying agent with the current (possibly compacted) history
        let response = self.inner.chat(prompt, history.clone()).await?;

        // Using rig's standard Message constructors if possible, or correct struct fields
        history.push(Message::User {
            content: rig::OneOrMany::one(rig::message::UserContent::text(prompt)),
        });
        history.push(Message::Assistant {
            content: rig::OneOrMany::one(rig::message::AssistantContent::text(&response)),
            id: None, // Rig's Assistant Message has an optional id for tool calls
        });

        Ok(response)
    }

    /// Access the underlying Rig Agent
    pub fn agent(&self) -> &Agent<M> {
        &self.inner
    }
}

/// Extension trait to easily add context management to an existing rig Agent
pub trait AgentContextExt<M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static> {
    /// Wraps the agent in a ContextManagedAgent that will automatically
    /// compact conversation history using the provided compaction model
    /// when the estimated token count exceeds the threshold.
    fn with_compaction<C: Prompt + WasmCompatSend + WasmCompatSync + 'static>(
        self,
        threshold: usize,
        compaction_model: C,
    ) -> ContextManagedAgent<M, C>;
}

impl<M: CompletionModel + WasmCompatSend + WasmCompatSync + 'static> AgentContextExt<M>
    for Agent<M>
{
    fn with_compaction<C: Prompt + WasmCompatSend + WasmCompatSync + 'static>(
        self,
        threshold: usize,
        compaction_model: C,
    ) -> ContextManagedAgent<M, C> {
        ContextManagedAgent {
            inner: self,
            compaction_threshold: threshold,
            compaction_model,
        }
    }
}
