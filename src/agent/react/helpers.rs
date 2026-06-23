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
