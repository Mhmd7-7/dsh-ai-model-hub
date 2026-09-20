# Roadmap

Phase 1 is complete: the architecture is built, tested, and demonstrated end to
end with mock models. What follows is what each later phase actually requires —
measured against what already exists, not estimated from scratch.

---

## Phase 1 — architecture validation ✅ complete

**Goal:** prove the whole path before any real engine is involved.

Delivered:

- Capability vocabulary, model descriptor schema, and a validating catalog
- Deterministic, capability-first router with per-candidate explanations
- Runtime manager: process lifecycle, health, idle timeout, resource gating
- Adapter contract plus three mock models producing **real** artifacts — a
  structurally valid PNG (every chunk CRC verified), a loadable ASCII STL, and a
  playable WAV
- Durable artifact store with a tamper-resistant path resolver
- DSH plugin exposing nine capability tools and a dynamic capability snapshot
- 198 tests: unit, integration, adapter, and plugin — all runnable without an
  engine (the eight spawn-based ones need a shell that permits piped child stdio)
- A vertical slice that runs the real code paths, not doubles

| Requirement | Where |
|---|---|
| Capability discovery | `list_capabilities`, `GET /model status` tools |
| Model metadata | `src/catalog/descriptor.ts` |
| Schema validation | `parseModelCatalogConfig` + published JSON Schema, drift-tested |
| Deterministic routing | `src/router/router.ts`, tested against invented models |
| Hardware/resource checks | `ModelCatalog.checkResources` + `src/machine.ts` |
| Lifecycle management | `src/runtime/manager.ts` |
| Typed artifacts | `src/artifacts/` |
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

## Phase 4 — a real local 3D model

**Code required: one adapter, plus a decision about output format.**

This is the least standardised area. There is no dominant local text-to-3D server
with a stable API, so the adapter shapes differ from Phases 2–3.

### Candidates

| Approach | Engine | Notes |
|---|---|---|
| Shap-E / Point-E | Python script | Simple API, fast, low fidelity. Best first target |
| TripoSR / InstantMesh | Python script | Single-image → mesh, good quality, GPU-heavy |
| Stable Fast 3D | Python script, or a Gradio HTTP API | Fast, permissive licence |
| Trellis | Python server | High quality, significant VRAM |
| Blender | `cli` adapter | Not generation — *post-processing*. Decimation, UV, format conversion |

### What to build

1. **`src/adapters/python-script.ts`** — a general `cli` adapter for
   process-per-request Python inference. This is the highest-leverage piece
   because it covers Shap-E, TripoSR, Stable Fast 3D, and most future research
   code without a new adapter each time:
   - Stage inputs into a per-invocation temp directory (paths, not bytes).
   - Run with a bounded timeout and the argv guardrails.
   - Read a declared output file from the temp directory and persist it.
   - Keep the exact invocation in `adapterConfig`, so a new model is a config
     entry rather than a new adapter.
2. **Format normalisation.** Decide the canonical mesh format (GLB is the right
   answer — single file, embedded textures, universally readable) and convert in
   the adapter. The artifact `type` stays `model_3d` regardless.
3. **Metadata that matters downstream.** `vertexCount`, `triangleCount`, `format`,
   and bounding-box dimensions. A workflow that decimates or retextures a mesh
   needs these without loading it.

### Definition of done

- `text_to_3d` produces a real mesh loadable in a viewer.
- `image_to_3d` accepts a generated image artifact and produces a mesh.
- `text → image → 3D` runs end to end with real models on both hops.

### Risks

- **Format plurality.** Research code emits `.obj`, `.ply`, `.glb`, or gaussian
  splats. Normalising at the adapter boundary is what keeps the artifact system
  honest — do not leak engine-specific formats upward.
- **VRAM.** Most text-to-3D models want 8–16 GB. Declare it, and let routing
  refuse the model on a machine that cannot run it rather than crashing.
- **No standard API.** Expect this adapter to need editing as the field moves.
  That is exactly why it is one file with a `supports()` check.

---

## Phase 5 — lifecycle and resource-aware routing ✅ largely complete

Delivered in Phase 1, ahead of schedule, because doing it later would have meant
retrofitting state into a stateless design:

- Start / stop / restart with graceful shutdown and process-tree kills
- Startup health polling with fast failure and captured stderr
- Per-model serialized transitions, so concurrent cold starts yield one process
- Idle-timeout sweeping that spares any model with a call in flight
- Resource gating on VRAM, RAM, and GPU presence
- Machine detection via `nvidia-smi`, with conservative behaviour when absent

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

## Beyond Phase 6

Directions the architecture already accommodates:

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
