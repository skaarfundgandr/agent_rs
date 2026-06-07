use rig::message::{AssistantContent, Message, UserContent};
use std::sync::LazyLock;
use tiktoken_rs::{cl100k_base, CoreBPE};

static LAZY_TOKENIZER: LazyLock<Result<CoreBPE, String>> = LazyLock::new(|| {
    cl100k_base().map_err(|e| e.to_string())
});

/// Count tokens in a plain text string using the cl100k_base BPE tokenizer.
///
/// Falls back to a character-based heuristic (char_count / 4) if loading the tokenizer fails.
///
/// # Arguments
///
/// * `text` - The plain text string slice to token-count.
///
/// # Returns
///
/// Returns the estimated token count of the string as a `usize`.
pub fn count_string_tokens(text: &str) -> usize {
    match &*LAZY_TOKENIZER {
        Ok(bpe) => bpe.count_with_special_tokens(text),
        Err(e) => {
            tracing::warn!("Failed to load cl100k_base tokenizer: {}. Falling back to character heuristic.", e);
            text.chars().count() / 4
        }
    }
}

/// Count total tokens for a slice of Rig Messages, accounting for ChatML/API overhead (~4 tokens per message).
///
/// Falls back to serializing complex content types (like images and documents) to JSON strings
/// to estimate their token usage when simple text matches are not available.
///
/// # Arguments
///
/// * `messages` - A slice of `Message` structs representing the conversation history.
///
/// # Returns
///
/// Returns the total estimated token count for the messages as a `usize`.
pub fn count_messages_tokens(messages: &[Message]) -> usize {
    let mut total = 0;

    for msg in messages {
        match msg {
            Message::System { content } => {
                total += count_string_tokens(content);
                total += 4;
            }
            Message::User { content } => {
                for item in content.iter() {
                    match item {
                        UserContent::Text(t) => {
                            total += count_string_tokens(&t.text);
                        }
                        other => {
                            // Serialize complex user content (e.g. Image, Document) to string
                            match serde_json::to_string(other) {
                                Ok(serialized) => {
                                    total += count_string_tokens(&serialized);
                                }
                                Err(err) => {
                                    tracing::warn!("Failed to serialize UserContent for token estimation: {:?}", err);
                                }
                            }
                        }
                    }
                }
                total += 4;
            }
            Message::Assistant { content, .. } => {
                for item in content.iter() {
                    match item {
                        AssistantContent::Text(t) => {
                            total += count_string_tokens(&t.text);
                        }
                        other => {
                            // Serialize complex assistant content (e.g. ToolCall, Reasoning) to string
                            match serde_json::to_string(other) {
                                Ok(serialized) => {
                                    total += count_string_tokens(&serialized);
                                }
                                Err(err) => {
                                    tracing::warn!("Failed to serialize AssistantContent for token estimation: {:?}", err);
                                }
                            }
                        }
                    }
                }
                total += 4;
            }
        }
    }
    total
}
