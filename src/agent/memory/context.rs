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
    ///
    /// # Arguments
    ///
    /// * `compaction_threshold` - The threshold token count above which conversation history is compacted.
    /// * `compaction_model` - The LLM/compactor model that implements `Prompt` used to generate the summary.
    ///
    /// # Returns
    ///
    /// Returns a new instance of `ContextManager<C>`.
    pub fn new(compaction_threshold: usize, compaction_model: C) -> Self {
        Self {
            compaction_threshold,
            compaction_model,
            token_estimator: None,
            compaction_prompt_formatter: None,
        }
    }

    /// Registers a custom token estimator callback.
    ///
    /// # Arguments
    ///
    /// * `estimator` - A function pointer that estimates the token count of a message slice.
    ///
    /// # Returns
    ///
    /// Returns `Self` with the updated estimator callback.
    pub fn with_token_estimator(mut self, estimator: fn(&[Message]) -> usize) -> Self {
        self.token_estimator = Some(estimator);
        self
    }

    /// Registers a custom compaction prompt formatter callback.
    ///
    /// # Arguments
    ///
    /// * `formatter` - A function pointer that takes the history JSON text representation and returns the custom prompt for the compaction LLM.
    ///
    /// # Returns
    ///
    /// Returns `Self` with the updated prompt formatter callback.
    pub fn with_compaction_prompt_formatter(mut self, formatter: fn(&str) -> String) -> Self {
        self.compaction_prompt_formatter = Some(formatter);
        self
    }

    /// Estimates the total token count of the history and current prompt combined.
    ///
    /// # Arguments
    ///
    /// * `history` - The slice of current conversation messages.
    /// * `prompt` - The new prompt text about to be sent.
    ///
    /// # Returns
    ///
    /// Returns the combined token estimate as a `usize`.
    pub fn estimate_tokens(&self, history: &[Message], prompt: &str) -> usize {
        let history_tokens = if let Some(estimator) = self.token_estimator {
            estimator(history)
        } else {
            count_messages_tokens(history)
        };

        history_tokens + count_string_tokens(prompt)
    }

    /// Checks if history exceeds the threshold and compacts it in-place using the compaction model.
    ///
    /// If compaction occurs:
    /// - The conversation history is cleared and replaced with a single system message containing the summary.
    /// - A `context_compaction` span is logged tracing token reduction.
    ///
    /// # Arguments
    ///
    /// * `history` - A mutable reference to the conversation history vector.
    /// * `prompt` - The current prompt text being submitted.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if compaction occurred and the history was updated, or `Ok(false)` if no compaction was needed.
    ///
    /// # Errors
    ///
    /// Returns a `PromptError` if calling the compaction model fails.
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
