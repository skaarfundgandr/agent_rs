# ReAct Loop

The ReAct (Reasoning + Acting) loop is the framework's per-cycle agent driver. The
loop drives a **single-turn completion per cycle** (not rig's `multi_turn` auto-loop):
it dispatches one prompt, classifies the assistant's content into thoughts / tool
calls / final answer, executes any tool calls, feeds observations back, and repeats
until the model emits a "Final Answer:" sentinel or `max_cycles` is reached.

## Public API

```rust
use agent_rs_lib::agent::{ReActExt, ReActBuilder, BuiltReAct, ReActSpanEmitter, REACT_PREAMBLE};
```

- **`ReActExt::react(agent)`** — extension trait method on
  `rig::agent::Agent<M, P>`. Returns a `ReActBuilder`.
- **Builder methods** (all return `Self`, preserving typestate):
  - `max_cycles(usize)` — guard rail; default 20.
  - `react_preamble(Option<String>)` — preamble prepended to the prompt; default `None` (uses `REACT_PREAMBLE` at runtime).
  - `with_history(Vec<Message>)` — seed the initial conversation history.
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
  - `prompt(msg)` — execute without mutating history; returns `Result<ReActTrace, ReActError>`.
  - `chat(msg)` — execute with history mutation on success; returns `Result<String, ReActError>`.
  - `stream_prompt(msg)` / `stream_chat(msg)` — streaming variants returning `ReActStream`.
  - `history()` — snapshot of current conversation history.
  - `max_cycles()` — accessor for the configured limit.
- **`ReActSpanEmitter`** — trait with no-op defaults for `emit_cycle_start`,
  `emit_cycle_end`, `emit_action`, `emit_observation`. The `opentelemetry` feature
  provides `LangSmithReActEmitter` as a concrete impl.
- **`ReActStream`** — implements `Stream<Item = ReActStreamItem>` for streaming ReAct loops.
- **`ReActStreamItem`** — enum of streaming events: `CycleStart`, `ThoughtDelta`, `Action`,
  `Observation`, `FinalAnswerDelta`, `Completed`, `Error`.

## Data Types (in `domain::agent`)

```rust
use agent_rs_lib::domain::agent::{Thought, Action, Observation, FinalAnswer, ReActStep, ReActTrace, ReActStreamItem};
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
use agent_rs_lib::agent::ReActExt;

let react = agent
    .react()
    .max_cycles(20)
    .on_action(|a| println!("→ {}", a.tool_name))
    .on_observation(|o| println!("← {} ({}B)", o.tool_name, o.result.len()))
    .build();

// Non-mutating prompt
let trace = react.prompt("Summarise the README.").await?;

// Mutating chat (appends to shared history)
let answer = react.chat("Summarise the README.").await?;
println!("{answer}");
```

## Per-Cycle Mechanics

1. `Span::current().record(...)` is augmented with `langsmith.span.kind = "chain"` / `react.cycle = N`.
2. `agent.prompt(preamble + prompt).with_history(history).extended_details().await` — single completion.
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
