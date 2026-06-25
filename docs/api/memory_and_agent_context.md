# Memory and Agent Context

Automates history size management to prevent context window overflows and excessive token costs.

> **Runtime reference:** See the [history compaction flowchart](../diagrams/flowchart.md) for the compaction algorithm, the [runtime sequence diagram](../diagrams/sequence-diagram.md) for how context management interacts with the chat loop, and the [class diagram](../diagrams/class-diagram.md) for `BuiltManagedAgent` and `ManagedExt`.

---

## `BuiltManagedAgent<M, P, C>`

A fully configured managed agent, ready to run prompts and chats. Constructed by calling [`.build()`](ManagedBuilder::build) on a [`ManagedBuilder`].

Wraps an `Agent<M, P>` (where `M: CompletionModel` and `P: PromptHook<M>`) with shared conversation history (`Arc<Mutex<Vec<Message>>>`) and an optional compaction model `C: Prompt` that automatically summarizes history when it crosses a token threshold.

### Methods
- **`history(&self) -> Vec<Message>`**
  Returns a snapshot of the current conversation history.
- **`async prompt(&self, msg: impl Into<String>) -> Result<String, PromptError>`**
  Executes an LLM chat turn **without** mutating shared history. Returns the response text. When compaction is enabled (with-compaction variant), history is compacted before the call.
- **`async chat(&self, msg: impl Into<String>) -> Result<String, PromptError>`**
  Executes an LLM chat turn **with** history mutation on success. On success, the shared history is replaced with the new working history. On error, the shared history is not modified.
- **`async stream_prompt(&self, msg: impl Into<String>) -> Result<ManagedStream<R>, PromptError>`**
  Streams a chat turn **without** mutating shared history.
- **`async stream_chat(&self, msg: impl Into<String>) -> Result<ManagedStream<R>, PromptError>`**
  Streams a chat turn **with** history mutation on completion. The shared history is updated with the final accumulated messages when the stream finishes.

> [!NOTE]
> **With-compaction variants:** When built with `.with_compaction()`, additional methods are available: `prompt_compact()`, `chat_compact()`, `stream_prompt_compact()`, `stream_chat_compact()`. These compact the history before calling the LLM.

---

## `ManagedBuilder<'a, M, P, CompState>`

Builder for a managed agent. Constructed via [`ManagedExt::managed`].

### Methods
- **`with_history(self, history: Vec<Message>) -> Self`**
  Seeds the initial conversation history.
- **`with_compaction(self) -> ManagedBuilder<'a, M, P, CompactionConfig<Agent<M, P>>>`**
  Enables automatic context compaction. The compaction model defaults to a clone of the agent itself.
- **`build(self) -> BuiltManagedAgent<M, P, ()>`**
  Builds the agent without compaction.

When compaction is enabled, additional methods are available on `ManagedBuilder<'a, M, P, CompactionConfig<C>>`:
- **`threshold(self, n: usize) -> Self`** — Sets the compaction threshold (must be > 0).
- **`compaction_model<NewC: Prompt>(self, model: NewC) -> Self`** — Replaces the compaction model.
- **`compaction_prompt(self, formatter: fn(&str) -> String) -> Self`** — Sets a custom compaction prompt formatter.
- **`tokenizer(self, estimator: fn(&[Message]) -> usize) -> Self`** — Sets a custom token estimator.
- **`build(self) -> BuiltManagedAgent<M, P, C>`** — Builds the agent with compaction. Panics if threshold was not set.

---

## `ManagedStream<R>`

A stream wrapper for a managed agent chat session. Once the stream finishes and yields the `FinalResponse`, the shared history (if provided) is updated with the final accumulated messages.

Implements `futures::Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>`.

---

## `ManagedExt`

Extension trait implemented for standard Rig `Agent<M, P>` structs to start building a managed agent.

- **`fn managed(&self) -> ManagedBuilder<'_, M, P, NoCompaction>`**
  Entry point for building a managed agent.

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

### Example Usage: Basic Chat

```rust,no_run
use agent_rs_lib::agent::ManagedExt;
use rig_core::message::Message;

# async fn example() -> Result<(), rig_core::completion::PromptError> {
# let openai = todo!(); // your Rig client
let chat_agent = openai.agent("gpt-5").build();

// Build a managed agent with shared history
let managed = chat_agent.managed().build();

let response = managed.chat("Hello, world!").await?;
println!("Response: {}", response);

// History is automatically updated after chat()
let history = managed.history();
# Ok(())
# }
```

### Example Usage: With Context Compaction

```rust,no_run
use agent_rs_lib::agent::ManagedExt;
use rig_core::message::Message;

# async fn example() -> Result<(), rig_core::completion::PromptError> {
# let openai = todo!(); // your Rig client
let chat_agent = openai.agent("gpt-5").build();
let compaction_agent = openai.agent("gpt-5-mini").build();

// Build with compaction enabled at ~2000 token threshold
let managed = chat_agent
    .managed()
    .with_compaction()
    .threshold(2000)
    .compaction_model(compaction_agent)
    .build();

let response = managed.chat_compact("What were my previous requests?").await?;
# Ok(())
# }
```

### Example Usage: Streaming Chat

```rust,no_run
use agent_rs_lib::agent::ManagedExt;
use rig_core::agent::MultiTurnStreamItem;
use futures::StreamExt;

# async fn example() -> Result<(), rig_core::completion::PromptError> {
# let openai = todo!();
let chat_agent = openai.agent("gpt-5").build();
let managed = chat_agent.managed().build();

let mut stream = managed.stream_chat("Explain quantum computing.").await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => {
            println!("Received chunk: {:?}", content);
        }
        Ok(MultiTurnStreamItem::FinalResponse(response)) => {
            println!("Streaming finished!");
        }
        Err(err) => {
            eprintln!("Error in stream: {:?}", err);
        }
    }
}
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

let persisted_history = agent::strip_reasoning_from_history(display_history);
```

The function is available at:
- `agent_rs_lib::agent::strip_reasoning_from_history` (re-exported from `agent` module)
- `agent_rs_lib::agent::agents::strip_reasoning_from_history` (full module path)
