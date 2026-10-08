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

For enumerating engines such as Ollama, a **host** declares the endpoint and the
engine's reported models can become `ModelDescriptor` entries. ComfyUI is a
deliberate exception: its host config explicitly lists API-format workflow
providers, with capabilities, bindings and output selectors. The hub does not
synthesize providers from checkpoints or saved workflows:

```
   host.runtime.engine
        │  ("ollama", "a1111", …)
        ▼
   ┌──────────────┐    ┌────────────────────┐    ┌───────────────┐
   │  Discoverer  │───▶│ parse → map        │───▶│ merge         │───▶ ModelCatalog
   │   API list   │    │ (pure, testable)   │    │ static first  │
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
  decided from the enumerating engine's reported API properties (for example,
  whether `/api/show` reports a projector), rather than a hardcoded model-name
  list. ComfyUI capabilities are instead declared by each configured workflow;
  graph structure and node class names are not used to invent providers.
- **The merge enforces precedence, not the catalog.**
  `mergeCatalogConfig(static, discovered)` returns
  `[...static, ...discovered.filter(id not already claimed)]`. `ModelCatalog`'s
  constructor silently skips a later duplicate and logs a diagnostic; relying on
  that would turn a deliberate rule into a logged accident, and would report a
  conflict that was designed away as a load error.

Discovery is off by default, fail-soft (an unreachable engine is a warning and no
models from that host), cached per host for a TTL, and refreshable on demand —
which is the answer to "I just pulled an Ollama model and do not want to wait".
ComfyUI providers declared on the host are unaffected by rescanning saved
workflows or loose checkpoint files.

### 4b. The Local models scan — an explicit, user-driven pass

The settings page's **Scan** is a second, deliberately separate discovery path,
because it answers a different question. Automatic discovery decides what the
*router* may select and is therefore conservative and configuration-driven; the
scan decides what the *user* can see and run right now, so it reads the machine
in front of them.

- `dsh-plugin/scan.ts` walks the declared engines: Ollama through its own
  `/api/tags`, ComfyUI through its saved-workflow route (`/userdata`) and/or a
  configured workflow directory. The two halves never share a failure path, so one
  engine being down cannot hide the other's results.
- **Readiness is engine-specific, and status precedes parsing.** A bare `GET /` is
  not a health check: Ollama answers it with the plain text `Ollama is running` and
  ComfyUI answers it with the editor's HTML, so probing the root and parsing JSON
  reports two healthy engines as stopped. Each engine is asked the route it
  implements, and a connection refusal, an HTTP error and a 200 that is not JSON
  are three separately reported failures.
- **A workflow is addressed, not guessed.** ComfyUI lists saved workflows as paths
  relative to a directory and serves one back through a single path segment, so the
  directory is part of the address and the whole relative path is percent-encoded
  (`/userdata/workflows%2F3d%2Fmodel.json`). Contents that could not be fetched are
  reported as `unreadable` with the status actually received — never as an invalid
  document nobody has seen.
- `src/comfy/scan.ts` turns a saved workflow into a public contract from the graph
  **and from ComfyUI's own node metadata**: `output_node` says which nodes are
  terminal, the declared `output` types say what a deliverable is, `input.required`
  says whether a conversion is complete, and a declared `default` fills a widget the
  editor omitted. Inputs come from evidence — a prompt that reaches a sampler, an
  image loader, a titled scalar control, a lone sampler or size node — and anything
  the metadata cannot support is withheld with a diagnostic naming the reason.
- Runnable workflows are published through `ModelHub.publishScannedModels`, which
  republishes the catalog as `static + scanned + discovered`. A repeated scan
  *replaces* the scanned half rather than growing it, so ids derived from source
  locations make a rescan a refresh and never a duplicate. Static configuration
  still wins the id collision, exactly as it does for automatic discovery.
- **Discovery owns only the engine it started.** A ComfyUI that is already
  answering is used and left alone; a stopped one is launched through the hub's own
  policy-gated path, waited for on its readiness route, and shut down afterwards
  only if this pass started it, no other scan still needs it, and nothing is queued
  or running. Concurrent scans share one pass, so they cannot start a second engine
  or stop each other's.
- **Discovered results outlive the engine.** The last successful pass per host is
  cached, so a later failure reports the same workflows with the engine marked down
  — they stay listed, stay registered, and run by starting the engine on demand —
  rather than emptying the page because a socket was refused.

The boundary the previous section states is unchanged and is what the scan is
built to respect: **a ComfyUI checkpoint is not a provider; a workflow is.**

### 5. Machine ↔ Router — measured resources

The router can only respect a model's declared `resources` if it knows what the
machine has, so `src/machine.ts` measures it and the catalog holds the result.
Two pairs of numbers exist on purpose, and conflating them is the mistake the
shape is built to prevent:

| | Question | Source |
|---|---|---|
| **capacity** — `vramGb`, `ramGb` | could this model *ever* run here? | total, from `nvidia-smi` and `os.totalmem()` |
| **headroom** — `availableVramGb`, `availableRamGb` | will it fit *right now*? | free, from `nvidia-smi`'s `memory.free` and `os.freemem()` |

The split shows up in three places:

- **Startup gating** asks about capacity. A model that does not fit an idle
  machine is `unsupported` and says so; rechecking it against headroom would let a
  busy moment mark a perfectly startable model permanently unroutable.
- **Routing** asks about headroom for a model that is *already resident* and about
  capacity for one the runtime can cold-start — because launching a stopped model
  is precisely what makes the whole machine's memory available to it. A model that
  is already loaded has, by definition, already paid for its memory, so counting
  its own footprint against it would be double-entry.
- **`availableResources()`** reports the probe's figures minus the declared
  footprint of every resident model, which is the number an operator should compare
  a model's requirements against.

Two properties keep this honest rather than clever. An unmeasurable figure is
**absent**, never guessed at, and the check falls back to capacity and says which
basis it used — `needs 12 GiB VRAM but only 6.5 GiB is available (of 8 GiB total)`.
And a measurement **expires**: the constructor starts one in the background (it
cannot await a subprocess), routing re-measures when the last one is older than
`resourceTtlMs`, and an invocation invalidates it outright, because a generation is
the one thing here that genuinely changes how much memory is free.

The probe is also the one place a command runs outside the model-command allowlist,
and deliberately: see the note on `GPU_PROBE_COMMAND` in `src/machine.ts`.

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

### The same flow for 3D

```
1  agent            invoke_model({ capability: 'image_to_3d', inputs: ['image_…'] })
2  router           filter by capability        → mock_text_model        rejected
                                                 → trellis_image_large   eligible
                    check availability           → cold but startable
                    check resources              → 12 GiB needed, 8 GiB available → rejected
                                                 → sf3d_image_to_mesh    eligible
3  runtime          ensureReady()               → starts the engine if configured to
4  adapter          three_d.invoke()            → speaks the engine's own two calls
5  artifacts        store.put()                 → model_3d_…glb
6  hub              InvocationResult { outputs: [mesh], decision, durationMs }
```

The 3D adapter is deliberately *protocol-driven rather than engine-driven*: the
call names, argument order, and result location come from the catalog entry, and
the transport is one of two shapes the local ecosystem actually uses (Gradio's
queue API, or a JSON HTTP route). That is what makes "add another 3D engine" a
JSON edit, and it is enforced by tests that drive the adapter against a fake engine
configured entirely by the test. See [three-d.md](three-d.md).

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
| Add another local 3D engine | a JSON entry (+ a steps file if it needs one) | none |
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
