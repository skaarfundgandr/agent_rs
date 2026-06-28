use rig_core::message::{Message, UserContent};

use crate::agent::react::Compact;
use crate::domain::errors::ReActError;

/// Run context compaction on `working_history` if a context manager is
/// configured. Returns `Ok(())` on success or `Err(ReActError::Model)` if
/// compaction fails.
pub(crate) async fn maybe_compact_history(
    context_manager: Option<&(dyn Compact + Send + Sync)>,
    working_history: &mut Vec<Message>,
    current_prompt: &Message,
    fallback_prompt: &str,
) -> Result<(), ReActError> {
    let Some(cm) = context_manager else {
        return Ok(());
    };
    let prompt_text = extract_prompt_text(current_prompt, fallback_prompt);
    cm.compact(working_history, prompt_text)
        .await
        .map(|_| ())
        .map_err(|e| ReActError::Model(e.to_string()))
}

/// Extract the text content from a `Message::User`, falling back to the
/// provided default.
fn extract_prompt_text<'a>(prompt: &'a Message, fallback: &'a str) -> &'a str {
    match prompt {
        Message::User { content } => content.iter().find_map(|c| match c {
            UserContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        }),
        _ => None,
    }
    .unwrap_or(fallback)
}
