# Roadmap

Phases 1–7 are complete. The architecture is built, tested, and demonstrated end
to end against real engines; the mock models it was validated with survive only as
an internal test double — `src/adapters/mock.ts`, referenced by tests and by no
shipped catalog. What follows is what each phase actually required, measured
against what already exists rather than estimated from scratch, plus what is
genuinely still open.

The two gaps that used to be listed here are closed: a real local 3D adapter
(Phase 4), and machine probing wired into `ModelHub` so routing decides against
measured resources (Phase 5). What remains is a much shorter list, and it is
recorded honestly at the end of each phase.

---

## Phase 1 — architecture validation ✅ complete

**Goal:** prove the whole path before any real engine is involved.

Delivered:

- Capability vocabulary, model descriptor schema, and a validating catalog
- Deterministic, capability-first router with per-candidate explanations
- Runtime manager: process lifecycle, health, idle timeout, resource gating
- Adapter contract plus a deterministic in-process test double producing **real**
  artifacts — a structurally valid PNG (every chunk CRC verified), a loadable
  ASCII STL, and a playable WAV. It exists to validate the architecture and is
  never referenced by `config/models.json`
- Durable artifact store with a tamper-resistant path resolver
- DSH plugin exposing ten capability tools and a dynamic capability snapshot
- 429 tests: unit, integration, adapter, discovery, machine, and plugin — all
  runnable without an engine or a GPU, because the adapter tests run real local
  HTTP servers as stand-ins
- A vertical slice that runs the real code paths, not doubles

| Requirement | Where |
|---|---|
| Capability discovery | `list_capabilities`, `GET /model status` tools |
| Model metadata | `src/catalog/descriptor.ts` |
| Schema validation | `parseModelCatalogConfig` + published JSON Schema, drift-tested |
| Deterministic routing | `src/router/router.ts`, tested against invented models |
| Hardware/resource checks | `ModelCatalog.checkResources` + `src/machine.ts`, measured and routed against (Phase 5) |
| Lifecycle management | `src/runtime/manager.ts` |
| Typed artifacts | `src/artifacts/` — including 3D containers, sniffed from the bytes |
| Multi-step workflows | `text → image → 3D` and `image → image → 3D` in `tests/integration.test.ts` |
| Graceful failure | `tests/integration.test.ts`, `tests/plugin.test.ts` |
| Timeouts | `withTimeout`, per-model budgets, tool budgets |
| Clear logging | `HubEvent` stream + the plugin's diagnostics |
| No model-specific router logic | A test routes `zzz_completely_invented_engine` |
| No model-specific prompt logic | The prompt text is generated from the live catalog |
| Security | `src/util/process.ts`, `docs/security.md` |

---

## Phase 2 — a real local text model ✅ complete (non-streaming)

**Code required: one file, now written.** `src/adapters/openai.ts` implements the
`openai_compatible` kind, so a single adapter reaches every
`/v1/chat/completions` server: Ollama, llama.cpp's `llama-server`, vLLM, LM Studio,
KoboldCpp.

### What shipped

- `invoke()` POSTs `{ model, messages, stream: false, temperature, max_tokens, seed }`
  to `${endpoint}${runtime.path}`, reads `choices[0].message.content`, and persists
  it as a `text` artifact whose id chains into the next capability call.
- **Vision works through the same adapter:** `image_understanding` inlines input
  images as `image_url` data URLs, and text/json inputs are inlined as labelled
  context.
- Cancellation and deadlines are forwarded to the socket, and a slow engine is
  reported as `INVOCATION_TIMEOUT` rather than as an unexplained transport error.
- Engine tuning lives in `adapterConfig.extraBody` (operator-only) so `num_ctx`,
  `repeat_penalty` and friends can be set without letting an agent shape the request.
- Token usage is reported in `value` from the response's `usage` object.
- Registered in `ModelHub`'s default adapters; `config/examples/real-models.example.json`
  already carries ready-to-use Ollama and llama.cpp entries.

### What remains

- **Streaming.** `stream: true` is not implemented: responses are buffered. Worth
  adding for a long generation a user watches arrive.
- **Context enforcement.** `limits.contextTokens` is still declarative only; the
  adapter neither truncates nor refuses an over-long request, so the engine does
  whatever it does. Failing with a clear code would be better.
- **Model naming.** Ollama's `/v1` route needs the model name from
  `adapterConfig.model` (or `runtime.modelPath`); it is never inferred from the
  catalog id.

---

## Phase 3 — a real local image model

**Code required: one adapter.** Phase 5's lifecycle and resource work is already
done, which is what makes this cheap.

### What exists

- `http_json` is a valid adapter kind; A1111/Forge and ComfyUI host entries ship
  as examples.
- Resource-aware routing already picks SD 1.5 over SDXL on a small card, with no
  code change.
- `image_to_image` input resolution, aspect preservation, and artifact chaining
  are implemented and tested.

### What to build

1. **`src/adapters/a1111.ts`** for the `/sdapi/v1/txt2img` and `/img2img` routes:
   - Decode the base64 `images[0]` payload and persist it as a PNG artifact.
   - Derive `width`/`height` metadata so a downstream `image_to_3d` can reason
     about it.
   - Read `info` from the response into `value` (seed, sampler, steps) — that is
     what makes a generation reproducible.
2. **`src/adapters/comfyui.ts`** for the graph API:
   - This is genuinely more work: `/prompt` takes a node graph, not a prompt.
     Template the workflow JSON, POST it, then poll `/history/{prompt_id}` for
     outputs.
   - Keep the graph template in `adapterConfig`, so a deployment can point at its
     own workflow without a code change.
3. **`src/adapters/stable-diffusion-cpp.ts`** (optional) for `cli` engines that
   generate per invocation. This exercises the `cli` adapter kind, which stages an
   input directory and reads an output file — the one genuinely different
   execution model.

### Definition of done

- `invoke_model({ capability: 'text_to_image', prompt: '…' })` returns a real
  generated PNG, not a fixture.
- `image_to_image` accepts a produced artifact id and preserves its aspect ratio.
- A real `start_model` launches the WebUI, waits for `/sdapi/v1/sd-models`, and
  `stop_model` shuts it down — the lifecycle path is already tested, so this is
  configuration plus a smoke test.

### Risks

- **Cold start is slow.** A1111 loading SDXL can take minutes. `startupTimeoutMs`
  exists for this; set it generously and rely on the polling health check, which
  returns as soon as the engine answers.
- **VRAM is not freed promptly.** Engines cache weights aggressively. `stopModel`
  is the reliable way to reclaim VRAM, which is why the idle timeout exists.
- **Base64 payloads are large.** A 1024×1024 PNG is ~1.5 MB encoded. Cap the
  response body size rather than buffering without limit.

---

## Phase 4 — a real local 3D model ✅ shipped

**Shipped as one adapter: `three_d`.** See [three-d.md](three-d.md) for the operator
guide; this section records what the decision turned out to be and why.

This is the least standardised area. There is no dominant local text-to-3D server
with a stable API, so the adapter shapes differ from Phases 2–3.

### What the survey found

| Approach | Engine | Interface | Output |
|---|---|---|---|
| TRELLIS | `python app.py` | Gradio queue API, **two** calls (generate, then extract GLB) | `.glb` |
| Hunyuan3D-2 | Gradio app, or `api_server.py` | Gradio queue API, or a FastAPI route | `.glb`, `.obj` |
| Stable Fast 3D | Gradio app | Gradio queue API, one call | `.glb` |
| TripoSR | Gradio app | Gradio queue API, one call | `.obj` |
| Blender | `cli` adapter | Not generation — *post-processing* | — |

Two findings shaped the implementation:

1. **They are all Gradio apps.** There is no shared *product* API, but there is a
   shared *framework* one: the queue API at `/gradio_api/call/<name>`, plus a
   `/config` document that lists the names. That is a documented, stable interface
   across Gradio 3, 4, and 5 — which makes it a far better integration point than
   any single engine's routes.
2. **`text_to_3d` is not real yet.** None of these engines generates a mesh from
   text alone; text-to-3D products are a text-to-image model feeding an
   image-to-3D model. The hub models that honestly: `text_to_3d` stays in the
   vocabulary, the adapter will serve it if an engine exposes a route for it, and
   no shipped catalog claims it.

### What shipped

1. **`src/adapters/three-d.ts`** — one adapter, two transports (`gradio`,
   `http_json`), parameterised by a declarative step list in `adapterConfig`. Call
   names, argument order, argument bindings, result location, and format are all
   catalog data; the adapter knows no engine. Multi-call engines are expressed as
   several steps, with `$N[.M]` bindings carrying one step's output into the next —
   which is exactly how TRELLIS's generate-then-extract-GLB pair works.
2. **`src/artifacts/formats.ts`** — format plurality handled at the boundary rather
   than normalised away: GLB, GLTF, OBJ, STL, and PLY are recognised by *sniffing
   the bytes*, cross-checked against what the engine claimed, and a disagreement is
   recorded on the artifact as a warning. A server that answers 200 with an HTML
   error page is caught here instead of becoming a mesh the next step cannot open.
3. **`src/discovery/three-d.ts`** — a discoverer that verifies rather than trusts.
   It reads the engine's API-description document, derives capabilities from the
   *shape* of the endpoint names, and intersects them with the operator's declared
   models. An engine whose only mesh routes are exporters is refused with a message
   saying so.
4. **Metadata that matters downstream.** `vertexCount` and `triangleCount` measured
   out of the file (OBJ vertex lines, glTF accessors, STL facet count), plus
   `format`, `mimeType`, `byteLength`, `sourceArtifactId`, and a digest of the
   bytes.

### Definition of done

- `image_to_3d` accepts a generated image artifact and produces a mesh. ✅
- `text → image → 3D` runs end to end with the artifact id as the only thing that
  crosses the hop. ✅ (`npm run demo:3d`, `tests/three-d.test.ts`)
- `text_to_3d` produces a real mesh. ⚠️ **Not claimed**, because no surveyed local
  engine implements it — see finding 2 above. The capability is routable the moment
  an engine declares a route for it.
- Verified against a real engine on real hardware. ⚠️ **Not done here**: the tests
  and the demo drive a stand-in Gradio server, which proves the protocol and the
  boundary but not a specific engine's numerics. The catalog entries for TRELLIS
  and Stable Fast 3D in `config/examples/` are the ones to run first.

### Risks

- **Format plurality.** Research code emits `.obj`, `.ply`, `.glb`, or gaussian
  splats. Normalising at the adapter boundary is what keeps the artifact system
  honest — do not leak engine-specific formats upward.
- **VRAM.** Most text-to-3D models want 8–16 GB. Declare it, and let routing
  refuse the model on a machine that cannot run it rather than crashing. This is
  what the measured-headroom work in Phase 5 exists for.
- **No standard API.** Expect this adapter to need editing as the field moves.
  That is exactly why it is one file, and why the engine specifics live in the
  catalog rather than in it.

---

## Phase 5 — lifecycle and resource-aware routing ✅ complete

Delivered in Phase 1, ahead of schedule, because doing it later would have meant
retrofitting state into a stateless design:

- Start / stop / restart with graceful shutdown and process-tree kills
- Startup health polling with fast failure and captured stderr
- Per-model serialized transitions, so concurrent cold starts yield one process
- Idle-timeout sweeping that spares any model with a call in flight
- Resource gating on VRAM, RAM, and GPU presence
- Machine detection via `nvidia-smi`, with conservative behaviour when absent

### What the remaining gap was, and how it closed

The probe existed but `ModelHub` never called it: routing compared declared needs
against totals, and a machine that had never been probed passed everything. Three
changes closed that:

1. **The probe measures headroom, not just capacity.** `nvidia-smi` is asked for
   `memory.free` as well as `memory.total`, so a GPU another application has
   filled is visible. `MachineProfile` carries both pairs, and an unmeasurable
   figure is left *absent* rather than guessed at.
2. **`ModelHub` probes, publishes, and re-checks.** The constructor starts a pass
   in the background (`probeResources: true` by default), publishes the result to
   the catalog, re-validates every model's resource verdict against it, and
   re-measures when the last one is older than `resourceTtlMs`. An invocation
   invalidates the measurement outright, because a generation changes how much
   memory is free.
3. **The router asks the right question per candidate.** Capacity for a model the
   runtime can cold-start, headroom for one that is already resident — and the
   refusal names the shortfall: `needs 12 GiB VRAM but only 6.5 GiB is available
   (of 8 GiB total)`.

`hub.availableResources()` reports the probe's figures minus the declared footprint
of every resident model, which is the number to compare a model against.

### What remains

1. **Cross-model reservation.** Headroom is measured, and resident models'
   declared footprints are subtracted on request — but nothing *reserves* memory
   ahead of a start. Two models started concurrently can both be admitted against
   the same free bytes. A real reservation would need the runtime to hold a lease
   per model, and would make an idle sweep able to reclaim it.
2. **`diskGb` is advisory.** It is probed and reported but not enforced, because
   the weights it describes may not be downloaded yet.
3. **Per-process memory accounting.** RSS and VRAM per child would make the idle
   sweeper smarter than a timer. Still the right answer, still not done.

### What remains

1. **Concurrent-residency budgeting.** Today each model is checked against the
   machine in isolation. The real constraint is the *sum* of resident models.
   `resources.allowConcurrentInstances` is declared but not yet enforced as a
   global budget.
2. **LRU eviction under pressure.** When a cold start would exceed the budget,
   stop the least-recently-used model that is idle instead of refusing.
3. **Live resource sampling.** Detection is a one-shot probe at startup. Reading
   actual GPU utilization would let routing avoid a model whose card is already
   busy with something else.
4. **Memory monitoring per process.** `activeInvocations` is tracked; sampled RSS
   and VRAM per child would make the idle sweeper smarter than a timer.

---

## Phase 6 — multi-model workflows ✅ core complete

The typed artifact system makes workflows compose today:

```ts
const image = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a spaceship' })
const mesh  = await hub.invokeModel({ capability: 'image_to_3d', inputs: [image.outputs[0].id] })
```

Both `text → image → 3D` and `image → image → 3D` are covered by integration
tests against the real code paths.

### What remains

1. **Declarative workflow definitions.** A named, reusable pipeline — "concept
   art": `text → image → upscale → 3D` — as data rather than a sequence of tool
   calls, so it is reproducible and shareable. It should be a thin layer over
   `invokeModel`, not a second execution engine.
2. **Workflow-level observability.** A goal/step view for a multi-model run, so a
   failure at step 3 does not mean re-running steps 1 and 2.
3. **Intermediate caching.** Content-addressed artifact ids make memoisation
   nearly free: the same prompt and model could return an existing artifact
   rather than regenerating. Needs a cache policy, since users often *want* a
   different result from the same prompt.
4. **Parallel fan-out.** Independent steps (four variations of one prompt) should
   run concurrently. The runtime already tracks in-flight calls per model, so the
   accounting exists; what is missing is a scheduler that respects
   `allowConcurrentInstances`.

---

## Phase 7 — runtime model discovery ✅ shipped (ComfyUI auto-graph generation is a follow-up)
**Goal:** stop requiring a JSON edit before a model the machine already has can be
used.

A *host* — "ComfyUI is at `http://127.0.0.1:8188`" — is the only thing configured.
Everything about what an engine can currently do is read out of the engine's own
introspection API at runtime and synthesized into `ModelDescriptor` entries, so a
freshly pulled Ollama model, a checkpoint dropped into ComfyUI's models
directory, or a LoRA someone installed is usable without touching
`config/models.json`.

### What shipped

- `src/discovery/types.ts` — `HostDiscoverer`, a `DiscoveryRegistry` keyed by
  engine (mirroring `AdapterRegistry`), a per-host TTL cache, and the merge step.
- `src/discovery/ollama.ts` — `/api/tags` + `/api/show`: every installed model,
  with capabilities inferred from whether the engine reports a vision projector,
  `contextTokens` read from `model_info`, and resources estimated from the
  reported weight size.
- `src/discovery/a1111.ts` — `/sdapi/v1/sd-models`, `/samplers`, and `/options`:
  every checkpoint, plus the loaded one promoted by priority so routing prefers
  it (a checkpoint switch is expensive; only the loaded one serves without one).
- `src/discovery/comfyui.ts` — `/object_info`: checkpoint filenames read out of
  the loader nodes' own enumerations, capabilities from an explicit
  node-class → capability table (`CAPABILITY_SIGNALS`), and a **minimal default
  graph** generated for the common one-checkpoint shape.
- `src/discovery/three-d.ts` — the API-description document (`/gradio_api/config`
  or `/config`): the named endpoints a 3D app exposes, capabilities derived from
  their *shape*, and the operator's declared models intersected with what the
  engine actually proves. It is the one discoverer that **refuses** rather than
  reports: an engine with only mesh *exporters* gets a warning saying exactly that,
  because publishing `image_to_3d` for a mesh writer turns a clean routing refusal
  into a confusing invocation failure.
- Wired behind `ModelHubOptions.discoverModels` (default **false**) and the plugin
  setting of the same name; `ModelHub.fromConfigAndDiscovery` merges *before* the
  catalog is constructed; `ModelHub.refreshDiscovery` and the
  `refresh_model_discovery` tool are the manual refresh path.
- Four load-bearing properties, each with tests:
  1. No model identifier appears anywhere in discovery code.
  2. Output is raw descriptors, resolved only by `resolveDescriptor` — no router,
     runtime, or adapter change.
  3. An unreachable engine is a warning and an empty result, never a crash.
  4. On an id collision the **static** entry wins, by merge order and by a filter
     that runs before the catalog ever sees the list — not by priority, and not
     by relying on `ModelCatalog`'s duplicate-skipping.

### What remains

1. **ComfyUI auto-graph generation beyond the common case.** The shipped version
   synthesizes a graph only when `/object_info` contains exactly the six node
   classes a minimal `CheckpointLoaderSimple → KSampler → VAEDecode → SaveImage`
   pipeline needs, and otherwise leaves `adapterConfig.workflow` absent so the
   adapter reports "needs a workflow template". LoRA chains, controlnets,
   upscalers, second passes, and UNET+CLIP+VAE triples each need their own
   template, and guessing one is worse than saying so. The natural next step is a
   small library of *shapes* (not models) selected by which loader nodes exist.
2. **File sizes for better resource estimates.** Neither `/object_info` nor the
   A1111 listing reports a size, so ComfyUI and A1111 VRAM figures are filename
   heuristics. A companion listing (or a HEAD request per file) would replace the
   guess with a measurement; the Ollama path already has real numbers.
3. **Discovering LoRAs and UNETs as *modifiers*.** They are enumerated today but
   deliberately not published as models: a LoRA is not something this hub can
   route to on its own. Modelling "apply this LoRA to that checkpoint" needs a
   descriptor shape that does not exist yet.
4. **Per-host opt-in and per-engine budgets.** Discovery is all-or-nothing today.
   A deployment may want discovery on Ollama (cheap, local) but off for an engine
   on a slow remote endpoint.
5. **Pushing updates rather than polling.** The TTL cache re-reads on demand; a
   long-lived session would benefit from re-reading when a capability is missing
   rather than only when asked.

## Beyond Phase 6

### Directions the architecture already accommodates

| Direction | Why it fits |
|---|---|
| **HTTP daemon mode** | `ModelHub` is the whole API; `ArtifactStore` is already an interface, so an object-store backend drops in |
| **A web panel** | `HubEvent` exists to be observed; nothing in the hub would change |
| **Remote models** | A model with an `https://` endpoint and an HTTP adapter is indistinguishable from a local one to every layer above it |
| **Multiple hubs** | Catalogs are documents; nothing prevents one hub per GPU or per project |
| **Client-side UI cards** | `presentCall` / `presentResult` are available on the tool definitions and currently unused — a `diff` card for a mesh, a `web`-style card for an image |

### Per-session catalog resolution

**Status: deliberately deferred, with a known workaround.**

The plugin resolves its catalog from ordered anchors (an explicit `configPath`,
then `searchRoots`, then the host process's working directory, then the plugin's
own installation directory). It does **not** yet resolve per session, so one host
serving two workspaces that carry different `models.json` files will serve both
sessions from whichever anchor matched at boot.

The reason is timing rather than difficulty: `apply` runs when the profile boots,
before any session exists, so "the session's workspace" is not yet a fact.
Reaching it means deferring hub construction to the first tool call, where
`exec.agent.session` carries the validated workspace `cwd`, and then keeping one
hub per resolved catalog root — which turns the idle sweeper, the health prober,
and the disposer from plugin-scoped into per-root.

Until then, `searchRoots` names the workspace explicitly and `configPath` pins a
single catalog exactly.

### Distributing the plugin without a checkout

**Status: open. The plugin is installed from a checkout today, for a Node reason.**

Two separate constraints tie installation to a real checkout, and both are worth
removing.

**The plugin is TypeScript.** It is loaded through Node's type stripping, and
Node refuses to strip types for any file under `node_modules`
(`ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`). pnpm materialises every registry
and git dependency inside the profile's `node_modules`, so
`dsh plugin add github:Mhmd7-7/dsh-ai-model-hub` cannot load, however the specifier
is spelled. The junction install works precisely because the link resolves to a
directory outside `node_modules`. Lifting this means compiling `dsh-plugin/` to
JavaScript at release time: a build step and a `lib/` artifact to keep in step
with `src/`, in exchange for an install that behaves like any other package. The
hub itself would keep running TypeScript directly, since only the DSH-facing layer
needs to be a plain module.

**The plugin's peer closure is not installed for it.** `dsh plugin add` forwards
to pnpm, which warns about peer dependencies rather than installing them, so the
installer must run `npm install` inside `dsh-plugin/` first — the reason the
installer has three steps instead of one. A `pnpm` workspace with `hoist-pattern`
enabled, or a published `dsh-ai-model-hub` depended on by version instead of by
`file:..`, would remove that step and make the plugin package self-contained.
