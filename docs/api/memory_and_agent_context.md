# Memory and Agent Context

Automates history size management to prevent context window overflows and excessive token costs.

> **Runtime reference:** See the [history compaction flowchart](../diagrams/flowchart.md) for the compaction algorithm, the [runtime sequence diagram](../diagrams/sequence-diagram.md) for how context management interacts with the chat loop, and the [class diagram](../diagrams/class-diagram.md) for `ContextManagedAgent` and `AgentContextExt`.

---

## `ContextManagedAgent<M, C, P>`

Wraps an `Agent<M, P>` (where `M: CompletionModel` and `P: PromptHook<M>`) and a compaction model `C: Prompt` to automatically summarize conversation history when it crosses a token threshold, calculated via the `cl100k_base` BPE tokenizer by default.

### Methods
- **`async chat(&self, prompt: &str, history: &mut Vec<Message>) -> Result<String, PromptError>`**
  Executes an LLM chat turn. Summarizes conversation history in-place if threshold is crossed, then appends the current user prompt and assistant response.
- **`async chat_with_owned_history(&self, prompt: &str, history: Vec<Message>) -> Result<(String, Vec<Message>), PromptError>`**
  Executes an LLM chat turn using owned history, returning the updated history rather than mutating it in-place.
- **`async stream_chat(&self, prompt: &str, history: &[Message]) -> Result<(ContextManagedChatStream<impl Stream, M::StreamingResponse>, oneshot::Receiver<Vec<Message>>), PromptError>`**
  Executes a streaming LLM chat turn. Compacts history if needed, returns a stream wrapper yielding elements, and a oneshot `Receiver` that resolves to the updated history once the stream is fully consumed.
- **`async stream_chat_with_owned_history(&self, prompt: &str, history: Vec<Message>) -> Result<(ContextManagedChatStream<impl Stream, M::StreamingResponse>, oneshot::Receiver<Vec<Message>>), PromptError>`**
  Executes a streaming LLM chat turn using owned history.

  > [!NOTE]
  > **Thought / Thinking Tokens:** The returned stream (`ContextManagedChatStream`) does not filter or modify the elements yielded by the underlying completion model. Therefore, any thought, reasoning, or thinking tokens produced by the model are preserved and passed through to the consumer (as part of `MultiTurnStreamItem::StreamAssistantItem`).
- **`with_token_estimator(mut self, estimator: fn(&[Message]) -> usize) -> Self`**
  Registers a custom token estimator callback to replace the default `cl100k_base` token counting.
- **`with_compaction_prompt_formatter(mut self, formatter: fn(&str) -> String) -> Self`**
  Registers a custom prompt formatter to format the compaction request sent to the compaction model.
- **`agent(&self) -> &Agent<M, P>`**
  Returns a reference to the inner wrapped `Agent`.

---

## `ContextManagedChatStream<S, R>`

A stream wrapper for streaming responses from a context-managed agent. Once the stream is polled to completion, the updated conversation history (including the final accumulated model response) is sent to the oneshot channel to be retrieved by the caller.

### Methods
- **`new(inner: S, history_tx: oneshot::Sender<Vec<Message>>, original_history: Vec<Message>) -> Self`**
  Creates a new `ContextManagedChatStream` wrapping an underlying stream, a oneshot sender, and the conversation history snapshot to merge into the final history.

---

## `ContextManager<C>`

Manages context memory, token estimation, and automatic compaction independently of the agent wrapper.

### Methods
- **`new(compaction_threshold: usize, compaction_model: C) -> Self`**
  Creates a new `ContextManager` with a compaction threshold and a compaction LLM.
- **`with_token_estimator(mut self, estimator: fn(&[Message]) -> usize) -> Self`**
  Registers a custom token estimator callback.
- **`with_compaction_prompt_formatter(mut self, formatter: fn(&str) -> String) -> Self`**
  Registers a custom compaction prompt formatter callback.
- **`estimate_tokens(&self, history: &[Message], prompt: &str) -> usize`**
  Estimates the total token count of the history and current prompt combined.
- **`async compact_history_if_needed(&self, history: &mut Vec<Message>, prompt: &str) -> Result<bool, PromptError>`**
  Checks if the conversation history exceeds the threshold and compacts it in-place using the compaction model. Returns `Ok(true)` if compaction occurred, or `Ok(false)` otherwise.

---

## Tokenizer Utilities

Located in `agent::memory::tokenizer`.

- **`count_string_tokens(text: &str) -> usize`**
  Counts tokens in a plain text string using the `cl100k_base` BPE tokenizer. Falls back to a character-based heuristic (character count / 4) if the tokenizer cannot be loaded.
- **`count_messages_tokens(messages: &[Message]) -> usize`**
  Counts total tokens for a slice of Rig `Message`s, accounting for ChatML/API framing overhead (~4 tokens per message). Falls back to JSON serialization for complex content types (e.g., images, tool calls).

---

## Model Execution Utilities

Located in `agent::model::chat`.

- **`async execute_chat(agent: &Agent<M, P>, prompt: &str, history: &mut Vec<Message>) -> Result<String, PromptError>`**
  Utility to execute a standard (non-streaming) chat turn against the LLM using the provided history. Updates history in-place with the assistant response.
- **`execute_stream_chat(agent: &Agent<M, P>, prompt: &str, history: Vec<Message>) -> StreamingPromptRequest<M, P>`**
  Utility to prepare a streaming chat turn request against the LLM.

---

## `AgentContextExt`

Extension trait implemented for standard Rig `Agent<M, P>` structs to easily wrap them in a context management layer.

- **`with_compaction<C: Prompt>(self, threshold: usize, compaction_model: C) -> ContextManagedAgent<M, C, P>`**
  Wraps the standard Rig agent in a `ContextManagedAgent` using the specified token threshold and compaction model.

### Example Usage: Context Compaction

```rust,no_run
use agent_rs_lib::agent::agents::AgentContextExt;
use rig_core::message::Message;

# async fn example() -> Result<(), rig_core::completion::PromptError> {
# let openai = todo!(); // your Rig client
let chat_agent = openai.agent("gpt-5").build();
let compaction_agent = openai.agent("gpt-5-mini").build();

// Wrap the chat agent to automatically compact context when it exceeds ~2000 tokens
let managed_agent = chat_agent.with_compaction(2000, compaction_agent);

let mut history = vec![];
let response = managed_agent.chat("What were my previous requests?", &mut history).await?;
# Ok(())
# }
```

### Example Usage: Streaming Chat & Thought Tokens

```rust,no_run
use agent_rs_lib::agent::agents::AgentContextExt;
use rig_core::message::Message;
use rig_core::agent::MultiTurnStreamItem;
use futures::StreamExt;

# async fn example() -> Result<(), rig_core::completion::PromptError> {
# let openai = todo!();
let chat_agent = openai.agent("gpt-5").build();
let compaction_agent = openai.agent("gpt-5-mini").build();

// Wrap the agent with compaction enabled
let managed_agent = chat_agent.with_compaction(2000, compaction_agent);

let history = vec![];

// Start streaming chat turn
let (mut stream, rx) = managed_agent.stream_chat("Explain quantum computing.", &history).await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => {
            // content can be standard text chunks or thought/reasoning tokens
            println!("Received assistant chunk: {:?}", content);
        }
        Ok(MultiTurnStreamItem::FinalResponse(response)) => {
            println!("Streaming finished! Final response: {}", response.choice);
        }
        Err(err) => {
            eprintln!("Error in stream: {:?}", err);
        }
    }
}

// Retrieve the updated history containing the final response
let updated_history = rx.await?;
# Ok(())
# }
```

---

## `strip_reasoning_from_history()`

A free function that removes `AssistantContent::Reasoning` blocks from a history vector.

Reasoning blocks are ephemeral chain-of-thought from the model. They are useful for real-time display (yielded by the stream as `StreamAssistantContent::Reasoning` / `ReasoningDelta`) but waste tokens when persisted and fed back to the model on subsequent turns. Calling this function before persisting history avoids that waste.

```rust
pub fn strip_reasoning_from_history(history: Vec<Message>) -> Vec<Message>
```

Assistant messages whose content consists entirely of reasoning are dropped from the result (since a reasoning-only message would be semantically empty). Non-assistant messages pass through unchanged. The message `id` field is preserved when filtering partial content.

### Example Usage

```rust,no_run
use agent_rs_lib::agent;

// Stream yields reasoning for live display — not affected by this function
let (stream, rx) = agent.stream_chat(prompt, &history).await?;
let display_history = consume_chat_stream(stream, rx, channel).await?;

// Strip reasoning before persisting
let persisted_history = agent::strip_reasoning_from_history(display_history);
repo.save_session_history(session_id, &persisted_history)?;
```

The function is available at:
- `agent_rs_lib::agent::strip_reasoning_from_history` (re-exported from `agent` module)
- `agent_rs_lib::agent::agents::strip_reasoning_from_history` (full module path)
