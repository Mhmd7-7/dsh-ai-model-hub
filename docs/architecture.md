# Architecture

## The one idea

DeepSeek Harness knows **what capabilities are available**. It never knows how to
launch a model, which interpreter runs it, how its process is supervised, or how
a particular engine works. Every one of those facts lives below a single
interface, and that interface is expressed in one currency: **capabilities**.

```
   "text_to_image"          ← the agent's entire vocabulary for this
        │
        ▼
   ┌─────────┐   ┌────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐
   │ Catalog │──▶│ Router │──▶│ Runtime │──▶│ Adapter │──▶│ Artifact │
   └─────────┘   └────────┘   └─────────┘   └─────────┘   └──────────┘
     what can      which one    make it      actually      typed,
     be done?      and why?     ready        run it        durable
```

## The dependency rule

```
dsh-plugin/  ──imports──▶  @deepseek-ai/*        (DSH: a 0.1.x-rc line)
     │
     └──imports──▶  src/  ──imports──▶  node builtins ONLY
```

`src/` imports nothing from DeepSeek Harness. That is not tidiness — it is the
survivability requirement. DSH is pre-1.0 and will change. Because the hub's
entire logic sits behind a facade the plugin merely *binds to*, a breaking DSH
change is confined to `dsh-plugin/`, and the hub's library tests keep running
without DSH installed at all.

This is verified rather than asserted: it is the reason `npm test` needs no DSH
process and completes in seconds.

## The three boundaries that carry the design

### 1. Capability ↔ Model — the router

The router's contract has two halves:

- it picks the *right* model, and
- it contains no knowledge of any *particular* model.

The second half is enforced by a test that routes against catalogs using invented
model ids and invented engine names. If routing works for
`zzz_completely_invented_engine`, the router cannot be cheating.

The router reads only declared facts: `capabilities`, `inputTypes`, `tags`,
`resources`, `enabled`, `priority`. There is no `if (modelId === …)`, no engine
name, and no capability-specific branch.

**Determinism is a requirement, not a nicety.** Given the same catalog and the
same request, the same model is chosen, so a routing surprise is reproducible.
The total order is: enabled first, then lower `priority`, then lexicographic id.

### 2. Model ↔ Process — the runtime manager

The runtime manager knows nothing about capabilities. It will start "model X"
because it was asked to; deciding *that* X is correct is the router's job. This
split means routing policy and process supervision can be replaced
independently.

Two facts are tracked separately, and conflating them was a real bug found by
the test suite:

| Fact | Question | Decided by |
|---|---|---|
| **availability** | may the router choose this? | health + config + resources |
| **lifecycle** | is there a process, and do we own it? | process ownership, endpoint liveness |

An in-process adapter — the mock, which exists only as a test double — is
*healthy* with no process at all. Treating
"healthy" as "already running" made the hub refuse to launch models it should
launch. The fix is that liveness is established only by an owned process or by a
live **endpoint** — never by an adapter's own opinion of itself.

### 3. Model ↔ Model — artifacts

Models never exchange bytes, file handles, or engine objects. They exchange
artifact references. That single decision is what makes multi-step workflows
compose without any participant understanding the others:

```ts
const image = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a spaceship' })
const mesh  = await hub.invokeModel({ capability: 'image_to_3d', inputs: [image.outputs[0].id] })
```

The router satisfies the second call by checking that some model declares
`image` among its `inputTypes`. Neither model knows what a PNG or a GLB is.

### 4. Host ↔ Model — runtime discovery

A **host** is the only thing an operator configures: "ComfyUI is at
`http://127.0.0.1:8188`". Everything about what that engine can do *right now* —
which checkpoints are on disk, which of them can see, which node packs are
installed — is answered by the engine itself and synthesized into
`ModelDescriptor` entries:

```
   host.runtime.engine
        │  ("comfyui", "ollama", "a1111", …)
        ▼
   ┌──────────────┐    ┌────────────────────┐    ┌───────────────┐
   │  Discoverer  │───▶│ parse → map        │───▶│ merge         │───▶ ModelCatalog
   │  /object_info│    │ (pure, testable)   │    │ static first  │
   └──────────────┘    └────────────────────┘    └───────────────┘
```

Three rules keep it from becoming a second, weaker catalog:

- **Discovery produces descriptors, never `ResolvedModel`.**
  `resolveDescriptor()` remains the only thing that turns a descriptor into a
  resolved model, so inheritance, defaults, and validation cannot diverge between
  a hand-written entry and a discovered one. The router, runtime manager, and
  adapters are unchanged by discovery — it only changes where descriptors come
  from.
- **No model identifier appears in discovery code.** Discoverers know an
  engine's *response shape* and apply general heuristics. Capabilities are
  decided by shape (does `/api/show` report a projector? does `/object_info`
  contain a mesh export node?) rather than by matching a name, because a name
  list is wrong the moment someone installs something new.
- **The merge enforces precedence, not the catalog.**
  `mergeCatalogConfig(static, discovered)` returns
  `[...static, ...discovered.filter(id not already claimed)]`. `ModelCatalog`'s
  constructor silently skips a later duplicate and logs a diagnostic; relying on
  that would turn a deliberate rule into a logged accident, and would report a
  conflict that was designed away as a load error.

Discovery is off by default, fail-soft (an unreachable engine is a warning and no
models from that host), cached per host for a TTL, and refreshable on demand —
which is the answer to "I just pulled a model and do not want to wait".

## Data flow: "Create a futuristic city image"

```
1  agent            invoke_model({ capability: 'text_to_image', prompt: '…' })
2  plugin           validates args through the real DSH tool schema
3  hub              resolveRequestInputs()      → no inputs
4  router           filter by capability        → ollama_text_model     rejected
                                                 → comfyui_z_image_turbo eligible
                    sort by priority, then id   → comfyui_z_image_turbo
5  runtime          ensureReady()               → endpoint live, no cold start needed
6  adapter          comfyui.invoke()            → queues the graph, returns a PNG
7  artifacts        store.put()                 → image_a-futuristic-city_1e2ca8.png
8  hub              InvocationResult { outputs: [artifact], decision, durationMs }
9  plugin           renders model-facing text naming the artifact id
10 agent            continues, optionally feeding that id into the next step
```

Steps 4–8 are capability-generic. Step 6 is the only one that knows anything
about images, and it is replaceable by a config edit. The two ids above are the
ones the shipped `config/models.json` declares; a catalog that declares others
produces the same shape with different names, because nothing above reads them.

## Routing policy

Candidates are filtered, then ordered. Every rejection is recorded with a reason
and returned in `decision.candidates`, so "why did it pick that?" is always
answerable.

**Filtering** — a model is rejected when it:

- does not declare the capability
- is disabled, or excluded by routing policy
- lacks a required tag
- does not accept one of the artifact kinds the request supplies
- exceeds the detected machine (VRAM / RAM / GPU)
- is running but failing its health check

A **stopped but startable** model is deliberately *eligible*: refusing cold models
would make the hub useless on a machine where nothing runs until asked. Set
`allowColdStarts: false` to route among warm models only.

**Ordering** is a single numeric score — currently `priority`. The scoring lives
in one function that cannot name a model, which is what keeps the rule honest.

## Failure handling

Failures are typed, coded, and actionable. `ModelHubError.code` is stable and
safe to branch on; the message is for humans and for the model.

| Situation | Code | What happens |
|---|---|---|
| Nothing serves the capability | `NO_COMPATIBLE_MODEL` | Every candidate's rejection reason is listed |
| Model is stopped and not startable | `MODEL_UNAVAILABLE` | Names the expected endpoint |
| Model needs more than the machine has | `INSUFFICIENT_RESOURCES` | Reports the shortfall |
| Engine failed to load | `START_FAILED` | Includes the engine's stderr |
| Invocation exceeded its budget | `INVOCATION_TIMEOUT` | — |
| Caller cancelled | `INVOCATION_ABORTED` | Never retried on another model |
| A model failed mid-workflow | — | Falls back to the next eligible candidate from the *same* deterministic decision |

Fallback breadth is bounded by the catalog, not by a constant: the candidate
order comes from one routing decision, so a fallback is as reproducible as the
primary choice. A cancellation is the caller's decision and is never retried.

## Extensibility: three levels of change

| You want to… | You change | Code required |
|---|---|---|
| Add a model on a supported engine | a JSON entry | none |
| Let a supported engine report its own models | a flag: `discoverModels: true` | none |
| Add a new kind of engine | one adapter + one `AdapterKind` value | ~1 file |
| Let a new engine report its own models | one discoverer | ~1 file |
| Add a capability | the vocabulary + an adapter handler | ~2 files |

At no level do you touch the router, the DSH plugin, or DeepSeek Harness. The
discovery row is deliberately parallel to the adapter row: a new engine family is
one adapter *and*, if it can introspect itself, one discoverer — and neither
touches anything above it.

## The DSH coupling surface

The entire dependency on DeepSeek Harness is this table. It is deliberately
small enough to audit in a minute and cheap enough to re-verify after a DSH
upgrade.

| DSH API | Used for | Where |
|---|---|---|
| `name` / `inject` / `apply` / `Config` | plugin registration (Loader contract) | `dsh-plugin/index.ts` |
| `ctx.tools.register` | exposing capability tools to the model | `dsh-plugin/tools/*` |
| `ctx.systemPrompt.context` | a dynamic snapshot of available capabilities | `dsh-plugin/index.ts` |
| `ctx.logger` | diagnostics | `dsh-plugin/index.ts` |
| `ctx.effect` | disposing the hub — and every process it started — with the plugin | `dsh-plugin/index.ts` |
| `defineTool` | typed tool definitions and output contracts | `dsh-plugin/tools/*` |
| `Service` | publishing the hub as `ctx.modelHub` | `dsh-plugin/service.ts` |

Verified against `@deepseek-ai/dsh@0.1.5-rc.2` by inspecting the installed
packages' type declarations and a real third-party plugin, not from memory or
assumption. `dsh-plugin/types.ts` records the one runtime mixin that is absent
from cordis's published `Context` type, with the source that establishes it.

### Why the plugin has no default export

The plugin is a **function plugin with named exports only**. A stray
`export default` would make the DSH Loader's `unwrapExports` collapse the module
and silently drop `inject` — a documented DSH failure mode. A test asserts the
absence of a default export specifically to prevent that regression.

## What happens when DSH changes

1. `npm test` runs without DSH. If it passes, the hub is intact.
2. `npm run typecheck` resolves the real DSH type declarations. A changed API
   surfaces here.
3. Fix `dsh-plugin/`. Nothing in `src/` needs to move.

The JSON Schema in `src/catalog/model-catalog.schema.json` is a fourth
stability anchor: a deployment's catalog file outlives any DSH version, so the
schema is kept honest by a drift test that requires the published schema and the
runtime validator to reach identical verdicts on paired accept/reject cases.
