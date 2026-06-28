use rig_core::completion::PromptError;
use rig_core::tool::server::ToolServerError;

/// Detect a "Final Answer:" / "FINAL ANSWER:" sentinel in trailing text.
///
/// Returns `Some(text_after_sentinel)` if found, `None` otherwise.
#[doc(hidden)]
pub fn detect_final_answer(text: &str) -> Option<String> {
    let sentinel_len = "final answer:".len();
    let prefix = text.get(..sentinel_len)?;
    if !prefix.eq_ignore_ascii_case("final answer:") {
        return None;
    }
    let answer = text[sentinel_len..].trim();
    if answer.is_empty() {
        None
    } else {
        Some(answer.to_string())
    }
}

/// Convert a `ToolServerError` to a human-readable string.
pub(crate) fn tool_error_to_string(e: &ToolServerError) -> String {
    format!("{e}")
}

/// Recover the partial conversation progress carried by rig-core's
/// [`MaxTurnsError`](PromptError::MaxTurnsError).
///
/// When the inner agent loop exhausts its per-`agent.prompt()` turn budget,
/// rig-core returns the *full* accumulated history (the caller's snapshot +
/// the prompt that was sent + every assistant turn and tool result gathered
/// before the limit) inside the error. Returning `Some(history)` here lets the
/// ReAct loop preserve that progress instead of discarding it (which would make
/// the next cycle redo identical work and reproduce the same turn-limit error).
/// Returns `None` for any non-`MaxTurnsError`.
#[doc(hidden)]
pub fn recover_turn_limit_history(e: &PromptError) -> Option<Vec<rig_core::message::Message>> {
    match e {
        PromptError::MaxTurnsError { chat_history, .. } => Some((**chat_history).clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use rig_core::completion::CompletionError;
    use rig_core::tool::ToolSetError;
    use rig_core::tool::server::ToolServerError;

    #[test]
    fn tool_error_to_string_roundtrip() {
        let err = ToolServerError::ToolsetError(ToolSetError::ToolNotFoundError(
            "missing_tool".to_string(),
        ));
        let s = tool_error_to_string(&err);
        assert!(
            s.contains("missing_tool"),
            "error string should contain tool name: {s}"
        );
    }

    #[test]
    fn detect_final_answer_returns_none_for_empty_after_colon() {
        assert_eq!(detect_final_answer("Final Answer:"), None);
        assert_eq!(detect_final_answer("Final Answer:   "), None);
    }

    #[test]
    fn recover_returns_none_for_non_max_turns_error() {
        let err = PromptError::CompletionError(CompletionError::ProviderError("test".into()));
        assert!(recover_turn_limit_history(&err).is_none());
    }
}
