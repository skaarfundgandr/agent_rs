use rig_core::message::Message;

/// Removes `AssistantContent::Reasoning` blocks from the history.
///
/// Use this on history that will be persisted and fed back to the model
/// on subsequent turns. Reasoning blocks are ephemeral chain-of-thought
/// that waste tokens when persisted.
///
/// Reasoning is still yielded by the stream for real-time display —
/// this function only affects the history vector.
///
/// # Arguments
///
/// * `history` - The conversation history vector containing messages to filter.
///
/// # Returns
///
/// Returns a new conversation history vector with all reasoning content blocks removed.
///
/// # Example
///
/// ```
/// use rig_core::message::{Message, AssistantContent, UserContent};
/// use rig_core::OneOrMany;
///
/// let history = vec![
///     Message::User {
///         content: OneOrMany::one(UserContent::text("hello")),
///     },
///     Message::Assistant {
///         id: None,
///         content: OneOrMany::many(vec![
///             AssistantContent::Reasoning(rig_core::message::Reasoning::new("thinking...")),
///             AssistantContent::text("Hi there!"),
///         ]).unwrap(),
///     },
/// ];
///
/// let clean = agent_rs::agent::strip_reasoning_from_history(history);
/// assert_eq!(clean.len(), 2);
/// // The reasoning block is removed; text remains.
/// if let Message::Assistant { content, .. } = &clean[1] {
///     assert_eq!(content.iter().count(), 1);
/// }
/// ```
pub fn strip_reasoning_from_history(history: Vec<Message>) -> Vec<Message> {
    history
        .into_iter()
        .filter_map(|msg| match msg {
            Message::Assistant { id, content } => {
                let filtered: Vec<_> = content
                    .into_iter()
                    .filter(|item| {
                        !matches!(item, rig_core::message::AssistantContent::Reasoning(_))
                    })
                    .collect();
                match rig_core::OneOrMany::many(filtered) {
                    Ok(new_content) => Some(Message::Assistant {
                        id,
                        content: new_content,
                    }),
                    Err(_) => None, // message was only reasoning
                }
            }
            other => Some(other),
        })
        .collect()
}
