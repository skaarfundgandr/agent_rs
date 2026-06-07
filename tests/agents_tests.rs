use agent_rs_lib::agent::memory::context::ContextManager;
use agent_rs_lib::agent::memory::tokenizer::{count_messages_tokens, count_string_tokens};
use rig::completion::{Prompt, PromptError};
use rig::message::{AssistantContent, Message, UserContent};
use rig::wasm_compat::WasmCompatSend;

#[derive(Clone)]
struct MockCompactor {
    response: String,
}

impl Prompt for MockCompactor {
    #[allow(refining_impl_trait)]
    async fn prompt(
        &self,
        prompt: impl Into<Message> + WasmCompatSend,
    ) -> Result<String, PromptError> {
        if self.response != "Fallback" {
            return Ok(self.response.clone());
        }
        let msg = prompt.into();
        match msg {
            Message::System { content } => Ok(format!("Mock response for: {}", content)),
            Message::User { content } => {
                let text = match content.first_ref() {
                    UserContent::Text(t) => t.text.clone(),
                    _ => "".to_string(),
                };
                Ok(format!("Mock response for: {}", text))
            }
            _ => Ok(self.response.clone()),
        }
    }
}

#[test]
fn test_count_string_tokens() {
    let text = "Hello, world!";
    let count = count_string_tokens(text);
    // "Hello, world!" is 4 tokens under cl100k_base
    assert_eq!(count, 4);
}

#[test]
fn test_count_messages_tokens_basic() {
    let messages = vec![
        Message::System {
            content: "You are a helpful assistant.".to_string(),
        },
        Message::User {
            content: rig::OneOrMany::one(UserContent::text("Hello")),
        },
        Message::Assistant {
            id: None,
            content: rig::OneOrMany::one(AssistantContent::text("Hi there!")),
        },
    ];

    let count = count_messages_tokens(&messages);
    // Expected token calculation:
    // system: "You are a helpful assistant." (6 tokens) + overhead (4 tokens) = 10 tokens
    // user: "Hello" (1 token) + overhead (4 tokens) = 5 tokens
    // assistant: "Hi there!" (3 tokens) + overhead (4 tokens) = 7 tokens
    // Total: 22 tokens
    assert_eq!(count, 22);
}

#[test]
fn test_count_messages_tokens_complex_fallback() {
    // We will test serialization fallback for complex content
    // Create a UserContent::Image
    let img = rig::completion::message::Image {
        data: rig::completion::message::DocumentSourceKind::String("dGVzdA==".to_string()),
        media_type: None,
        detail: None,
        additional_params: None,
    };
    let messages = vec![
        Message::User {
            content: rig::OneOrMany::one(UserContent::Image(img)),
        },
    ];

    let count = count_messages_tokens(&messages);
    // Since it serializes the image variant to string, it should be > 4 overhead tokens.
    assert!(count > 4);
}

#[tokio::test]
async fn test_context_manager_no_compaction_under_threshold() {
    let compactor = MockCompactor {
        response: "Compacted".to_string(),
    };
    // Threshold set high (1000 tokens)
    let manager = ContextManager::new(1000, compactor);

    let mut history = vec![
        Message::User {
            content: rig::OneOrMany::one(UserContent::text("Short message")),
        },
    ];

    let compacted = manager
        .compact_history_if_needed(&mut history, "Short prompt")
        .await
        .unwrap();

    assert!(!compacted);
    assert_eq!(history.len(), 1);
    if let Message::User { content } = &history[0] {
        if let UserContent::Text(t) = content.first_ref() {
            assert_eq!(t.text, "Short message");
        } else {
            panic!("Expected text user content");
        }
    } else {
        panic!("Expected user message");
    }
}

#[tokio::test]
async fn test_context_manager_compaction_above_threshold() {
    let compactor = MockCompactor {
        response: "Compacted summary history".to_string(),
    };
    // Threshold set low (15 tokens)
    let manager = ContextManager::new(15, compactor);

    let mut history = vec![
        Message::User {
            content: rig::OneOrMany::one(UserContent::text(
                "A very long message that is guaranteed to exceed the 15 token threshold",
            )),
        },
    ];

    let compacted = manager
        .compact_history_if_needed(&mut history, "Another prompt")
        .await
        .unwrap();

    assert!(compacted);
    assert_eq!(history.len(), 1);
    if let Message::System { content } = &history[0] {
        assert_eq!(content, "Compacted summary history");
    } else {
        panic!("Expected system message after compaction");
    }
}

#[tokio::test]
async fn test_context_manager_custom_compaction_prompt() {
    let compactor = MockCompactor {
        response: "Fallback".to_string(),
    };
    // Threshold low (15 tokens)
    let manager = ContextManager::new(15, compactor)
        .with_compaction_prompt_formatter(|history_text| {
            format!("CUSTOM HEADER: {}", history_text)
        });

    let mut history = vec![
        Message::User {
            content: rig::OneOrMany::one(UserContent::text(
                "A very long message that is guaranteed to exceed the 15 token threshold",
            )),
        },
    ];

    let compacted = manager
        .compact_history_if_needed(&mut history, "Another prompt")
        .await
        .unwrap();

    assert!(compacted);
    assert_eq!(history.len(), 1);
    if let Message::System { content } = &history[0] {
        assert!(content.starts_with("Mock response for: CUSTOM HEADER:"));
    } else {
        panic!("Expected system message after compaction");
    }
}

#[tokio::test]
async fn test_context_managed_chat_stream() {
    use futures::StreamExt;
    use tokio::sync::oneshot;
    use rig::agent::MultiTurnStreamItem;
    use rig::completion::Usage;
    use agent_rs_lib::agent::agents::ContextManagedChatStream;

    let final_history = vec![
        Message::User {
            content: rig::OneOrMany::one(UserContent::text("Hello")),
        },
        Message::Assistant {
            id: None,
            content: rig::OneOrMany::one(AssistantContent::text("Hi there!")),
        },
    ];

    let final_item: MultiTurnStreamItem<()> = MultiTurnStreamItem::final_response_with_history(
        "Hi there!",
        Usage::new(),
        Some(final_history.clone()),
    );

    let inner_stream = futures::stream::iter(vec![
        Ok(final_item),
    ]);

    let (tx, rx) = oneshot::channel();
    let mut managed_stream = ContextManagedChatStream::new(inner_stream, tx);

    while let Some(item) = managed_stream.next().await {
        assert!(item.is_ok());
    }

    let updated_history = rx.await.unwrap();
    assert_eq!(updated_history.len(), 2);
    if let Message::Assistant { content, .. } = &updated_history[1] {
        if let AssistantContent::Text(t) = content.first_ref() {
            assert_eq!(t.text, "Hi there!");
        } else {
            panic!("Expected text assistant content");
        }
    } else {
        panic!("Expected assistant message");
    }
}

#[tokio::test]
async fn test_context_managed_chat_stream_no_history() {
    use futures::StreamExt;
    use tokio::sync::oneshot;
    use rig::agent::MultiTurnStreamItem;
    use rig::completion::Usage;
    use agent_rs_lib::agent::agents::ContextManagedChatStream;

    let final_item: MultiTurnStreamItem<()> = MultiTurnStreamItem::final_response(
        "Hi there!",
        Usage::new(),
    );

    let inner_stream = futures::stream::iter(vec![
        Ok(final_item),
    ]);

    let (tx, rx) = oneshot::channel();
    let mut managed_stream = ContextManagedChatStream::new(inner_stream, tx);

    while let Some(item) = managed_stream.next().await {
        assert!(item.is_ok());
    }

    let updated_history = rx.await.unwrap();
    assert!(updated_history.is_empty());
}

