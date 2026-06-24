use rig_core::message::{AssistantContent, Message};
use rig_core::tool::server::ToolServerError;

/// Detect a "Final Answer:" / "FINAL ANSWER:" sentinel in trailing text.
///
/// Returns `Some(text_after_sentinel)` if found, `None` otherwise.
#[doc(hidden)]
pub fn detect_final_answer(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    // Look for "final answer:" (case-insensitive) and take everything after it.
    let sentinel_pos = lower.find("final answer:");
    match sentinel_pos {
        Some(pos) => {
            let answer = text[pos + "final answer:".len()..].trim();
            if answer.is_empty() {
                None
            } else {
                Some(answer.to_string())
            }
        }
        None => None,
    }
}

/// Convert a `ToolServerError` to a human-readable string.
pub(crate) fn tool_error_to_string(e: &ToolServerError) -> String {
    format!("{e}")
}

/// Find the last `Message::Assistant` in `chat_history` that contains a
/// `ToolCall` whose function name matches `tool_name`.
///
/// Returns a cloned copy so the caller can push it to its own history.
#[doc(hidden)]
pub fn find_assistant_with_tool_call(chat_history: &[Message], tool_name: &str) -> Option<Message> {
    chat_history.iter().rev().find_map(|msg| {
        let Message::Assistant { content, id } = msg else {
            return None;
        };
        let has_match = content.iter().any(|item| {
            if let AssistantContent::ToolCall(tc) = item {
                tc.function.name == tool_name
            } else {
                false
            }
        });
        if has_match {
            Some(Message::Assistant {
                content: content.clone(),
                id: id.clone(),
            })
        } else {
            None
        }
    })
}
