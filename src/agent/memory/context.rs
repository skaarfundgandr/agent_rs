use rig::completion::{Prompt, PromptError};
use rig::message::Message;
use std::future::IntoFuture;
use tracing::Instrument;

use crate::agent::memory::tokenizer::{count_messages_tokens, count_string_tokens};

/// Manages context memory, token tracking, and automatic compaction of history.
pub struct ContextManager<C: Prompt> {
    compaction_threshold: usize,
    compaction_model: C,
    token_estimator: Option<fn(&[Message]) -> usize>,
    compaction_prompt_formatter: Option<fn(&str) -> String>,
}

fn default_compaction_prompt_formatter(history_text: &str) -> String {
    format!(
        "Summarize the following conversation history concisely, preserving key facts, \
        names, dates, and technical details. This summary will serve as the memory for \
        future interactions:\n\n{}",
        history_text
    )
}

impl<C: Prompt + rig::wasm_compat::WasmCompatSend + rig::wasm_compat::WasmCompatSync + 'static> ContextManager<C> {
    /// Creates a new `ContextManager` with a threshold and a compaction LLM.
    pub fn new(compaction_threshold: usize, compaction_model: C) -> Self {
        Self {
            compaction_threshold,
            compaction_model,
            token_estimator: None,
            compaction_prompt_formatter: None,
        }
    }

    /// Registers a custom token estimator callback.
    pub fn with_token_estimator(mut self, estimator: fn(&[Message]) -> usize) -> Self {
        self.token_estimator = Some(estimator);
        self
    }

    /// Registers a custom compaction prompt formatter callback.
    pub fn with_compaction_prompt_formatter(mut self, formatter: fn(&str) -> String) -> Self {
        self.compaction_prompt_formatter = Some(formatter);
        self
    }

    /// Estimates the total token count of the history and current prompt combined.
    pub fn estimate_tokens(&self, history: &[Message], prompt: &str) -> usize {
        let history_tokens = if let Some(estimator) = self.token_estimator {
            estimator(history)
        } else {
            count_messages_tokens(history)
        };

        history_tokens + count_string_tokens(prompt)
    }

    /// Checks if history exceeds the threshold and compacts it in-place using the compaction model.
    /// Returns `Ok(true)` if compaction occurred, `Ok(false)` otherwise.
    pub async fn compact_history_if_needed(
        &self,
        history: &mut Vec<Message>,
        prompt: &str,
    ) -> Result<bool, PromptError> {
        let current_tokens = self.estimate_tokens(history, prompt);
        if current_tokens > self.compaction_threshold && !history.is_empty() {
            let initial_tokens = count_messages_tokens(history);
            let compaction_span = tracing::info_span!(
                "context_compaction",
                initial_tokens = initial_tokens,
                threshold = self.compaction_threshold
            );

            let history_text = serde_json::to_string(&history).unwrap_or_default();
            let compaction_prompt = match self.compaction_prompt_formatter {
                Some(formatter) => formatter(&history_text),
                None => default_compaction_prompt_formatter(&history_text),
            };

            let summary = self.compaction_model.prompt(&compaction_prompt)
                .into_future()
                .instrument(compaction_span)
                .await?;

            let compacted_tokens = count_string_tokens(&summary);
            tracing::info!(
                compacted_tokens = compacted_tokens,
                "Conversation history compacted successfully"
            );

            history.clear();
            history.push(Message::System { content: summary });
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
