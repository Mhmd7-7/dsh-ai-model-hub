# Components

Each section below gives one component's responsibility, its contract, and the
decisions behind it. The rule that governs all of them: **each component knows
one thing, and knows nothing about its neighbours' internals.**

---

## 1. Capability vocabulary — `src/catalog/capabilities.ts`

The system's shared language. A *capability* is what a caller asks for; a *model*
is what serves it. Nothing above this file mentions a model, and nothing below it
mentions an agent.

| Capability | Consumes | Produces |
|---|---|---|
| `text_to_text` | text | text |
| `text_to_image` | text | image |
| `image_to_image` | image, text | image |
| `text_to_3d` | text | model_3d |
| `image_to_3d` | image, text | model_3d |
| `audio_generation` | text | audio |
| `speech_to_text` | audio | text |
| `image_understanding` | image, text | text |
| `video_generation` | text, image | video |

`CAPABILITY_IO` is **documentation and defaulting, not routing policy**. The
router never reads it. It is used to fill a manifest's `inputTypes`/`outputTypes`
when the author omits them, and to render the capability snapshot the agent sees
so it can compose workflows.

Adding a capability means editing this tuple. Nothing else in the hub branches on
capability identity except the mock adapter's handler table, which is exactly the
layer that *should* know what each capability produces.

---

## 2. Model descriptor — `src/catalog/descriptor.ts`

A model is **data**. This file defines the document a human writes and the
resolution rules that turn it plus its host into a flat record.

### Two shapes, deliberately

| Shape | Written by | Meaning |
|---|---|---|
| `ModelDescriptor` | a human, in `config/models.json` | only what differs for this model |
| `ResolvedModel` | `resolveDescriptor()` | every inheritable field guaranteed present |

That split is what lets several models share one host process. A user running
ComfyUI with three checkpoints writes one host (endpoint, launch command,
resource envelope, health strategy) plus three thin model entries.

Precedence is, highest first: **descriptor → host → adapter-derived default →
built-in default**. There is exactly one implementation of that rule, so the
router and the runtime manager cannot disagree about a model's timeout or
endpoint.

### Validation collects everything

`parseModelCatalogConfig` reports *every* problem at once, with paths:

```
models.json failed validation with 3 problems:
  - models[0].id: is required
  - models[0].type: must be one of text_generation, image_generation, …
  - models[1].capabilities[0]: must be one of text_to_text, text_to_image, …
```

A catalog is edited by hand; a fix-one-rerun loop is a bad experience.

Cross-field rules that need the whole document — unique ids, resolvable hosts,
`host` XOR `runtime`, capabilities that agree with declared input/output kinds —
are checked after parsing, when every descriptor is in hand.

Unknown keys are **ignored** rather than rejected, so a stray `$comment` or a
field added for a future version does not break a working deployment. Values that
*are* read are validated strictly.

---

## 3. Catalog registry — `src/catalog/registry.ts`

Holds every model, answers capability-first lookups, and reports what the machine
can and cannot do.

- `listModels()` / `getModel()` / `requireModel()`
- `findModelsByCapability()` — enabled only, best candidate first
- `findModelsByCapabilityIncludingDisabled()` — for explaining *why* a capability is unserved
- `listCapabilities()` / `listUnservedCapabilities()`
- `checkResources()` — compares declared needs against the detected machine
- `machineProfile`

Ordering is computed once at construction, so routing order is cheap and stable.

**`listUnservedCapabilities` exists because "no model for this" has three very
different causes** — nothing configured, everything disabled, or a config error
— and an operator must be able to tell them apart. The agent gets the same
information, so it can say "video generation is unavailable here" instead of
attempting it.

**Resource checking is conservative.** When detection returned nothing (no
`nvidia-smi`, probe skipped), every model reports as supported rather than
refusing everything. Trying and failing loudly at the engine beats refusing work
the machine could actually do.

---

## 4. Model Router — `src/router/router.ts`

Answers one question deterministically: *given this request, which registered
model should serve it?*

```
resolveRequestInputs()   artifact ids → typed artifacts, with kind checks
        │
        ▼
evaluateCandidate()      per model: capability? tags? input kinds? resources?
        │                availability? → eligible with a score, or a reason
        ▼
routeRequest()           filter → sort (priority, then id) → RoutingDecision
```

The decision carries **every** candidate with its verdict, so
`explain_routing` and a failure message both come from the same computation that
actually chose the model. The explanation cannot drift from the behaviour.

`RoutingPolicy` currently offers `allowColdStarts` and `excludedModelIds` —
enough to express "route among warm models only" and "quarantine this model
without editing its descriptor".

**What is not here:** any model id, engine name, file extension, or
capability-specific rule. Enforced by a test that routes invented models.

---

## 5. Runtime Manager — `src/runtime/manager.ts`

Owns everything stateful about a model: whether a process exists, whether the hub
started it, health, in-flight calls, and when to shut down. The **only** place
that mutates lifecycle, and one of two places that spawns a process.

### State model

```
LIFECYCLE                        AVAILABILITY
not_running  could start, none known alive   available   ready to serve
starting     launch in progress              stopped     known, not running
running      hub owns a live process         starting    launch or probe in flight
stopping     shutdown in progress            unhealthy   process alive, not answering
failed       gave up                         disabled    config says no
external     alive, owned by someone else    unsupported machine cannot satisfy it
                                             error       no adapter / broken descriptor
```

`external` means **a live process the hub does not own** — never merely "not
probed". A model the hub *could* start rests at `not_running`.

### Guarantees

- **One process per model.** Every lifecycle transition is serialized per model,
  so three concurrent cold-start requests produce one process, not three. This is
  a real failure mode for engines that bind a fixed port.
- **Idempotent start.** Starting an already-healthy model is a no-op reporting
  `alreadyRunning`.
- **Honest liveness.** An existing healthy *endpoint* wins over spawning a
  duplicate; an in-process adapter's health does not count as a running process.
- **Graceful stop.** Declared shutdown budget, then a process-group kill on POSIX
  or `taskkill /T` on Windows, because inference servers routinely fork helpers
  and a bare parent kill orphans them holding the port.
- **Opt-in idle timeout.** Models declaring none are never swept. Unloading a
  model a user is about to reuse is worse than holding VRAM.
- **Fast failure.** A process that exits during startup reports its stderr
  immediately rather than waiting out the startup budget.
- **Idempotent dispose.** Every owned process is stopped; one hang does not
  prevent the others.

Health probing polls rather than sleeps, so a cold start takes as long as the
engine needs rather than as long as its declared worst case.

---

## 6. Adapter contract — `src/adapters/types.ts`

The **only** place engine-specific knowledge lives.

```ts
interface ModelAdapter {
  readonly kind: AdapterKind
  readonly displayName: string
  supports(model: ResolvedModel): { ok: true } | { ok: false; reason: string }
  health(model: ResolvedModel, signal: AbortSignal): Promise<HealthReport>
  invoke(invocation: AdapterInvocation): Promise<AdapterOutput>
}
```

`AdapterInvocation` carries everything needed — model, capability, prompt,
**resolved** input artifacts, options, the artifact store, a cancellation signal,
and a logger. Adapters are therefore stateless with respect to a model, which is
what lets one adapter instance serve many models and makes an adapter testable
without a runtime.

`supports()` is checked once at startup, turning a configuration mistake (a `cli`
model with no `modelPath`) into a clear `unsupported` status instead of a crash
mid-workflow.

`AdapterRegistry.register()` returns a disposer that restores the previous
binding — how a test substitutes a fake, and how a plugin overrides a built-in
with a real engine, without leaking the override.

### Shipped adapters

| Kind | Status | Covers |
|---|---|---|
| `mock` | ✅ implemented | Phase 1 fixtures: text, PNG images, STL meshes, WAV audio, JSON video manifests |
| `openai_compatible` | ✅ implemented | Ollama, llama.cpp, vLLM, LM Studio, KoboldCpp — `text_to_text` and `image_understanding` |
| `http_json` | contract ready | A1111/Forge, ComfyUI, and any JSON-in/JSON-out engine |
| `cli` | contract ready | Process-per-request engines reading a local checkpoint |

`http_json` and `cli` are registered as valid configuration — the catalog accepts
them, and a model declaring one is reported honestly as `error` with "no adapter
registered for kind …" rather than failing silently — but they have no
implementation yet. They are the documented Phase 3–4 work; see
[roadmap.md](roadmap.md).

---

## 7. Artifact system — `src/artifacts/`

An artifact is small, JSON-serializable, and durably addressable: *where* content
lives plus the metadata a downstream model needs to reason about it without
loading it.

```json
{
  "id": "image_mock-image-for-a-futuris_1e2ca8fae09a",
  "type": "image",
  "uri": "file:///…/files/image_mock-image-for-a-futuris_1e2ca8fae09a.png",
  "mimeType": "image/png",
  "byteLength": 67888,
  "createdAt": 1758300000000,
  "producerModelId": "mock_image_model",
  "metadata": { "width": 640, "height": 384, "format": "png", "prompt": "…" }
}
```

`uri` is a plain string so the hub is not tied to one backend: the shipped store
writes `file://`, and an HTTP or object-store backend would use `https://` /
`s3://` without changing this type or any consumer.

### The local store

- Files land as ordinary files under `artifacts/files/` with a JSON index beside
  them — an operator can open a generated image without any hub tooling.
- Writes are serialized through one queue, so two models finishing at once cannot
  interleave index writes.
- The index is written atomically (temp file + rename), so a crash cannot leave a
  truncated index.
- `resolvePath()` is the seam a local engine needs — it wants a *path*, not a
  `Uint8Array`. It re-checks containment, because the index is a file on disk and
  therefore editable: a path outside the store root is refused.

Ids are `<type>_<slug>_<hash>`: sortable by kind, readable in a tool result, and
content-derived.

---

## 8. Execution guardrails — `src/util/process.ts`

The **only** file in the hub that spawns a process, so the security posture is
reviewable in one place. The threat model: *the agent is untrusted input.*

- **No shell, ever.** Children are spawned with `shell: false` and an argv array.
  There is no string to inject into, so `; rm -rf /` in a prompt is inert data.
- **An explicit allowlist.** Only well-known inference engines launch by default.
  Names outside it are refused with an actionable message until the operator sets
  `allowAnyCommand` — an explicit, auditable decision.
- **Per-argument validation.** NUL bytes (which can truncate a C string and
  smuggle a different argument past a check) and line breaks (a strong signal
  that someone built a shell line) are refused, not sanitized.
- **A scrubbed environment.** Credential-shaped variables (`*API_KEY*`,
  `*TOKEN*`, `*SECRET*`, `AWS_*`, …) are stripped from every child. A model
  process has no business holding the harness's keys.
- **Bounded output and hard timeouts.** Captured output is capped; a child that
  overruns is killed as a tree.
- **Contained abandonment.** `withTimeout` attaches a no-op handler to the losing
  branch of its race. Without it, an abandoned operation rejecting later becomes
  an unhandled rejection and **kills the process** — turning "this model was
  slow" into "the harness died". A regression test pins this.

---

## 9. The Hub facade — `src/hub.ts`

The public API. It exposes **capability verbs only** — there is no
`generateWithStableDiffusion` here, and there never will be: the moment the agent
can name an engine, swapping that engine becomes a prompt change.

```ts
listModels()               getModel()              getModelStatus()
findModelsByCapability()   listCapabilities()      listUnservedCapabilities()
invokeModel()              tryInvokeModel()        route()
startModel()               stopModel()             restartModel()
probeModel()               probeAll()
getArtifact()              listArtifacts()
onEvent()                  dispose()
```

`invokeModel` is the whole system in one call:

```
resolve inputs → route → for each candidate (bounded, deterministic order):
                     ensure ready (start if cold) → invoke → return artifacts
                  on failure: record, emit, try the next candidate
```

`tryInvokeModel` returns a structured failure instead of throwing — the shape a
model-facing tool wants, because a clear refusal is more useful to an agent than
an exception when a capability is simply not deployed.

`HubEvent` gives observers (a log, a UI, a test) visibility into what happened
without the hub knowing about any of them, and listener failures are contained so
a broken log sink cannot fail a user's image generation.

---

## 10. Configuration loading — `src/config/load.ts`

The boundary where the hub meets the filesystem. It resolves which file to read,
reads it, and hands the raw value to the catalog's validator — it does **not**
validate, because there is exactly one validator and it is not here.

Discovery walks upward from a starting directory looking for `models.json`,
`model-catalog.json`, or `dsh-ai-model-hub.json`, including a `config/`
subdirectory at each level. Walking upward means a hub started from a nested
subdirectory — where an agent's shell happens to be — still finds the
deployment's configuration.

Which directory it starts from is the caller's decision, and it matters more than
it looks. `loadCatalogConfig` defaults to `process.cwd()`, which is the wrong
anchor for a plugin loaded by a long-running host: the host's working directory is
fixed when it is launched and has nothing to do with where a catalog lives, so a
hub anchored only there finds nothing and the agent silently gets no tools.
`loadCatalogFromAnchors` exists for callers that know better — it tries an ordered
list of anchors, first hit wins, and reports every directory it tried when none
matches, so a missing catalog is a one-step diagnosis rather than a hunt.

---

## 11. DSH plugin layer — `dsh-plugin/`

Thin by design. It loads configuration, constructs the hub, publishes it as
`ctx.modelHub`, registers generic capability tools, and binds the hub's lifetime
to the plugin fiber.

### Tools

| Tool | Purpose |
|---|---|
| `list_models` | what models exist, their capabilities, live status, resource needs |
| `list_capabilities` | what this machine can serve, and what it cannot |
| `get_model_status` | is a specific model running, healthy, available — and why not |
| `check_model_health` | probe now, with latency and failure detail |
| `invoke_model` | **run a capability**; the router picks the model |
| `explain_routing` | which model *would* be chosen, and why — read-only |
| `start_model` / `stop_model` | explicit lifecycle control |
| `list_artifacts` | find earlier output to feed into the next step |

`invoke_model`'s parameter list contains no field that could name a command, a
path, an endpoint, or a module. The only model-identifying input is `modelId`, an
optional debug pin that must match an id the catalog already knows.

Failures lead with the stable code and add a recovery hint where one exists
(`NO_COMPATIBLE_MODEL` → "call list_capabilities"; `MODEL_UNAVAILABLE` → "call
get_model_status, then start_model"). `UNSAFE_OPERATION` gets no hint, because
the correct next step is to stop.

### The capability snapshot

The plugin registers a **dynamic** runtime context listing the capabilities this
deployment can serve, generated from the live catalog on every assembly. It
cannot drift from what is deployed, because it is not written down anywhere — it
is read from the machine. Registered as runtime context rather than a
system-prompt section because it is a changing fact about the environment, not a
standing instruction.

### Catalog resolution

The plugin resolves its catalog from ordered anchors, first hit wins: `configPath`
(which, when set, is the only candidate tried), then each directory in
`searchRoots`, then the host process's working directory, then the plugin's own
installation directory — walking upward from each and accepting `models.json`,
`model-catalog.json`, or `dsh-ai-model-hub.json`, directly or in a `config/`
subdirectory. The last anchor is what makes the zero-config case work: DSH installs
profile plugins as junction links and Node resolves them to their real path, so the
plugin finds the catalog shipped by the checkout it came from.

Resolution is deliberately not per session. `apply` runs before any session
exists, so "the session's workspace" is not yet a fact; `searchRoots` and
`configPath` are the explicit way to name one. See
[roadmap.md](roadmap.md#per-session-catalog-resolution).

### Safe defaults

| Setting | Default | Why |
|---|---|---|
| `allowProcessLaunch` | `false` | A plugin must not launch an 8 GB engine unless the operator opted in |
| `allowAnyCommand` | `false` | Widening the command allowlist is an explicit decision |
| `onConfigError` | `'warn'` | A broken catalog must never make the agent unusable |
| `exposeCapabilityContext` | `true` | The agent should know what is available |

A missing or invalid catalog logs the problem, registers nothing, and lets DSH
boot. `onConfigError: 'throw'` is available for an operator actively editing a
catalog who wants to be told immediately.
