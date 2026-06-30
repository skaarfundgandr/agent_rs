# ReAct Loop

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
> are **unchanged** — they are stateless by rig-core convention and never touch history.
>
> On success, `chat()` replaces the caller's `&mut Vec<Message>` with the full working
> trace (user turns, assistant responses, tool calls, final answer). On error the history
> is left untouched.

The ReAct (Reasoning + Acting) loop is the framework's per-cycle agent driver. The
loop drives a **single-turn completion per cycle** (not rig's `multi_turn` auto-loop):
it dispatches one prompt, classifies the assistant's content into thoughts / tool
calls / final answer, executes any tool calls, feeds observations back, and repeats
until the model emits a "Final Answer:" sentinel or `max_cycles` is reached.

## Public API

```rust
use agent_rs::agent::{ReActExt, ReActBuilder, BuiltReAct, ReActSpanEmitter, REACT_PREAMBLE};
```

- **`ReActExt::react(agent)`** — extension trait method on
  `rig::agent::Agent<M, P>`. Returns a `ReActBuilder`.
- **Builder methods** (all return `Self`, preserving typestate):
  - `max_cycles(usize)` — guard rail; default 20.
  - `max_retries(u32)` — retries for transient completion errors (`HttpError`,
    `ProviderError`) and `MaxTurnsError` within a single cycle; default 3. Uses
    exponential backoff (500ms × 2^attempt).
  - `react_preamble(Option<String>)` — preamble prepended to the prompt; default `None` (uses `REACT_PREAMBLE` at runtime).
  - `with_span_emitter(Arc<dyn ReActSpanEmitter>)` — for OTel integration.
  - `on_thought(impl Fn(&Thought) + Send + Sync + 'static)`
  - `on_action(impl Fn(&Action) + Send + Sync + 'static)`
  - `on_observation(impl Fn(&Observation) + Send + Sync + 'static)`
  - `on_final(impl Fn(&FinalAnswer) + Send + Sync + 'static)`
  - `on_error(impl Fn(&ReActError) + Send + Sync + 'static)`
  - `with_compaction()` — enables automatic context compaction (typestate transition).
  - `threshold(usize)`, `compaction_model(C)`, `compaction_prompt(fn)`, `tokenizer(fn)` — compaction config.
  - `build()` — returns `BuiltReAct<M, P, C>`.
- **`BuiltReAct`** methods:
  - `prompt(msg)` — stateless; returns `Result<ReActTrace, ReActError>`. No history interaction.
  - `chat(msg, &mut history)` — caller-owned history; on success writes the full working trace into `*history`. Returns `Result<String, ReActError>`.
  - `stream_prompt(msg)` / `stream_chat(msg, &mut history)` — streaming variants returning `ReActStream`. On `Completed`, `stream_chat` writes `*history = final_history`.
  - `max_cycles()` — accessor for the configured limit.
  - `max_retries()` — accessor for the configured retry limit.
- **`ReActSpanEmitter`** — trait with no-op defaults for `emit_cycle_start`,
  `emit_cycle_end`, `emit_action`, `emit_observation`. The `opentelemetry` feature
  provides `LangSmithReActEmitter` as a concrete impl.
- **`ReActStream<'h, M, P, C>`** — implements `Stream<Item = ReActStreamItem>` for streaming ReAct loops. Holds `history_out: Option<&'h mut Vec<Message>>` for write-back on `Completed`.
- **`ReActStreamItem`** — enum of streaming events: `CycleStart`, `ThoughtDelta`, `Action`,
  `Observation`, `FinalAnswerDelta`, `Completed { trace, final_history }`, `Error`.

## Data Types (in `domain::agent`)

```rust
use agent_rs::domain::agent::{Thought, Action, Observation, FinalAnswer, ReActStep, ReActTrace, ReActStreamItem};
```

All step types are `Serialize` + `Deserialize` (groundwork for M3 state persistence).
`ReActTrace` is the serializable record of one `react()` invocation.

## Termination Conditions

In order of precedence:
1. **Final Answer sentinel** — assistant text contains "Final Answer:" (case-insensitive).
2. **`max_cycles` exceeded** — returns `Err(ReActError::MaxCyclesExceeded { cycles })`.
3. **Plain text with no tool calls** — the response is treated as a final answer.
4. **Empty / no-final-answer with no tool calls** — returns
   `Err(ReActError::NoToolCallsAndNoFinalAnswer { cycle })`.

## Usage Example

```rust,no_run
use agent_rs::agent::ReActExt;

let react = agent
    .react()
    .max_cycles(20)
    .on_action(|a| println!("→ {}", a.tool_name))
    .on_observation(|o| println!("← {} ({}B)", o.tool_name, o.result.len()))
    .build();

// Stateless prompt — no history interaction
let trace = react.prompt("Summarise the README.").await?;

// Caller-owned history — replaces on success, untouched on error
let mut history = Vec::new();
let answer = react.chat("Summarise the README.", &mut history).await?;
println!("{answer}");
// history now contains the full working trace from this turn
```

## Per-Cycle Mechanics

1. `Span::current().record(...)` is augmented with `langsmith.span.kind = "chain"` / `react.cycle = N`.
2. `agent.prompt(preamble + prompt)` — single completion (stateless, no history interaction).
3. The returned messages are pushed into the working history clone.
4. The last assistant message is classified:
   - `AssistantContent::Reasoning(r)` / pre-tool-call `Text(t)` → emit `Thought`.
   - `AssistantContent::ToolCall(tc)` → emit `Action`, then call
     `agent.tool_server_handle.call_tool(name, args)`, build `Observation`, inject
     a `UserContent::ToolResult` message into working history for the next cycle.
   - Post-tool-call `Text(t)` containing "Final Answer:" sentinel → emit `FinalAnswer`.

The `ToolResult` correlation uses `tc.id` (provider-assigned) with a fallback of
`react-cycle-{n}` if absent. `tc.call_id` is also populated with the synthetic id
for providers that distinguish the two.
