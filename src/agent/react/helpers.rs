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
