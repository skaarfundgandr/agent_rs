# ReAct Loop

The ReAct (Reasoning + Acting) loop is the framework's per-cycle agent driver. The
loop drives a **single-turn completion per cycle** (not rig's `multi_turn` auto-loop):
it dispatches one prompt, classifies the assistant's content into thoughts / tool
calls / final answer, executes any tool calls, feeds observations back, and repeats
until the model emits a "Final Answer:" sentinel or `max_cycles` is reached.

## Public API

```rust
use agent_rs_lib::agent::{ReActExt, ReActLoop, ReActSpanEmitter, REACT_PREAMBLE};
```

- **`ReActExt::react(agent, prompt, history)`** — extension trait method on
  `rig::agent::Agent<M, P>`. Returns a `ReActLoop` builder.
- **`ReActLoop::builder(agent, prompt, history)`** — equivalent direct constructor.
- **Builder methods** (all return `Self`):
  - `max_cycles(usize)` — guard rail; default 20.
  - `react_preamble(Option<String>)` — preamble prepended to the prompt; default `Some(REACT_PREAMBLE)`.
  - `on_thought(impl Fn(&Thought) + Send + Sync + 'static)`
  - `on_action(impl Fn(&Action) + Send + Sync + 'static)`
  - `on_observation(impl Fn(&Observation) + Send + Sync + 'static)`
  - `on_final(impl Fn(&FinalAnswer) + Send + Sync + 'static)`
  - `on_error(impl Fn(&ReActError) + Send + Sync + 'static)`
  - `with_span_emitter(Arc<dyn ReActSpanEmitter>)` — for OTel integration.
- **`ReActLoop::execute()`** — `async`, returns `Result<ReActTrace, ReActError>`.
- **`ReActSpanEmitter`** — trait with no-op defaults for `emit_cycle_start`,
  `emit_cycle_end`, `emit_action`, `emit_observation`. The `opentelemetry` feature
  provides `LangSmithReActEmitter` as a concrete impl.

## Data Types (in `domain::agent`)

```rust
use agent_rs_lib::domain::agent::{Thought, Action, Observation, FinalAnswer, ReActStep, ReActTrace};
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
use rig_core::message::Message;
use std::sync::Arc;

let mut history: Vec<Message> = Vec::new();
let trace = agent
    .react("Summarise the README.", &mut history)
    .max_cycles(20)
    .on_action(|a| println!("→ {}", a.tool_name))
    .on_observation(|o| println!("← {} ({}B)", o.tool_name, o.result.len()))
    .execute()
    .await?;
println!("{}", serde_json::to_string_pretty(&trace)?);
```

## Per-Cycle Mechanics

1. `Span::current().record(...)` is augmented with `langsmith.span.kind = "chain"` / `react.cycle = N`.
2. `agent.prompt(preamble + prompt).with_history(history).extended_details().await` — single completion.
3. The returned messages are pushed into `history`.
4. The last assistant message is classified:
   - `AssistantContent::Reasoning(r)` / pre-tool-call `Text(t)` → emit `Thought`.
   - `AssistantContent::ToolCall(tc)` → emit `Action`, then call
     `agent.tool_server_handle.call_tool(name, args)`, build `Observation`, inject
     a `UserContent::ToolResult` message into `history` for the next cycle.
   - Post-tool-call `Text(t)` containing "Final Answer:" sentinel → emit `FinalAnswer`.

The `ToolResult` correlation uses `tc.id` (provider-assigned) with a fallback of
`react-cycle-{n}` if absent. `tc.call_id` is also populated with the synthetic id
for providers that distinguish the two.
