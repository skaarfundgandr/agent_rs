# Memory and Agent Context

> **v0.6.0 → v0.7.0 migration:**
>
> | Old (v0.6.0) | New (v0.7.0) |
> |---|---|
> | `chat(msg)` | `chat(msg, &mut history)` |
> | `chat_compact(msg)` | `chat_compact(msg, &mut history)` |
> | `stream_chat(msg)` | `stream_chat(msg, &mut history)` |
> | `stream_chat_compact(msg)` | `stream_chat_compact(msg, &mut history)` |
> | `with_history(vec)` on builder | **Removed.** Pass history to `chat()` directly. |
> | `history()` accessor | **Removed.** Caller owns the history vec. |
>
> `prompt()`, `stream_prompt()`, `prompt_compact()`, and `stream_prompt_compact()`
> are **unchanged** — they are stateless by rig-core convention.
>
> **Key behavioral difference (ReAct vs Managed):**
> - **ReAct `chat()`** replaces the caller's `&mut Vec<Message>` with the full working
>   trace (every tool call, observation, assistant turn).
> - **Managed `chat()`** pushes only `Message::user` + `Message::assistant` to the
>   caller's history on success — a simpler append pattern.
>
> In both cases, on error the caller's history is left untouched.

---

## `BuiltManagedAgent<M, P, C>`

A fully configured managed agent, ready to run prompts and chats. Constructed by calling [`.build()`](ManagedBuilder::build) on a [`ManagedBuilder`].

Wraps an `Agent<M, P>` (where `M: CompletionModel` and `P: PromptHook<M>`) with an optional compaction model `C: Prompt` that automatically summarizes history when it crosses a token threshold. History is now **caller-owned** — passed as `&mut Vec<Message>` to `chat()`.

### Methods
- **`max_retries(&self) -> u32`**
  Returns the configured retry limit for completion calls (default 3).
- **`async prompt(&self, msg: impl Into<String>) -> Result<String, PromptError>`**
  Stateless — returns the response text. No history interaction.
- **`async chat(&self, msg, &mut history) -> Result<String, PromptError>`**
  On success, pushes `Message::user` + `Message::assistant` to `*history`.
  On error, `history` is untouched.
- **`async stream_prompt(&self, msg) -> Result<ManagedStream<R>, PromptError>`**
  Streaming variant. Stateless — no history write-back.
- **`async stream_chat(&self, msg, &mut history) -> Result<ManagedStream<R>, PromptError>`**
  Streaming variant. On `FinalResponse`, pushes user+assistant to `*history`.

> [!NOTE]
> **With-compaction variants:** When built with `.with_compaction()`, additional methods are available: `prompt_compact()`, `chat_compact()`, `stream_prompt_compact()`, `stream_chat_compact()`. These compact the history in-place before calling the LLM.

---

## `ManagedBuilder<'a, M, P, CompState>`

Builder for a managed agent. Constructed via [`ManagedExt::managed`].

### Methods
- **`max_retries(self, n: u32) -> Self`**
  Sets the maximum number of retries for completion calls on transient errors
  (`HttpError`, `ProviderError`). Defaults to 3. Retries use exponential
  backoff (500ms × 2^attempt). Note: streaming methods are not retried at the
  construction level because stream errors only surface during polling.
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

## `ManagedStream<'h, R>`

A stream wrapper for a managed agent chat session. Once the stream finishes and yields the `FinalResponse`, the caller's history (if provided) is updated with the final user+assistant messages.

Implements `futures::Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>>`. Holds `history_out: Option<&'h mut Vec<Message>>` for write-back on completion.

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
use agent_rs::agent::ManagedExt;
use rig_core::message::Message;

# async fn example() -> Result<(), rig_core::completion::PromptError> {
# let openai = todo!(); // your Rig client
let chat_agent = openai.agent("gpt-5").build();

// Build a managed agent (no internal history — caller owns it)
let managed = chat_agent.managed().build();

let mut history = Vec::new();
let response = managed.chat("Hello, world!", &mut history).await?;
println!("Response: {}", response);
// history now contains user+assistant messages from this turn
# Ok(())
# }
```

### Example Usage: With Context Compaction

```rust,no_run
use agent_rs::agent::ManagedExt;
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

let mut history = Vec::new();
let response = managed.chat_compact("What were my previous requests?", &mut history).await?;
# Ok(())
# }
```

### Example Usage: Streaming Chat

```rust,no_run
use agent_rs::agent::ManagedExt;
use rig_core::agent::MultiTurnStreamItem;
use futures::StreamExt;

# async fn example() -> Result<(), rig_core::completion::PromptError> {
# let openai = todo!();
let chat_agent = openai.agent("gpt-5").build();
let managed = chat_agent.managed().build();

let mut history = Vec::new();
let mut stream = managed.stream_chat("Explain quantum computing.", &mut history).await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => {
            println!("Received chunk: {:?}", content);
        }
        Ok(MultiTurnStreamItem::FinalResponse(response)) => {
            println!("Streaming finished!");
            // history now contains user+assistant messages
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
use agent_rs::agent;

let persisted_history = agent::strip_reasoning_from_history(display_history);
```

The function is available at:
- `agent_rs::agent::strip_reasoning_from_history` (re-exported from `agent` module)
- `agent_rs::agent::agents::strip_reasoning_from_history` (full module path)
