---
name: dsh-ai-model-hub
description: Use local AI models through the dsh-ai-model-hub plugin — text, images, and 3D meshes — by asking for a capability instead of an engine. Use when a task needs local generation, when a local engine such as Ollama, ComfyUI, or A1111 is not running and must be started, when chaining models through artifacts, when adding or fixing a model in the catalog, or when the model hub tools are missing or an invocation fails.
whenToUse: The user wants a local image/text/3D model generated, mentions the model hub, Ollama, ComfyUI, or A1111 through this plugin, needs an engine started, wants to add, inspect, or debug a model, or the hub tools such as invoke_model and list_models are absent or failing.
---

# dsh-ai-model-hub

Local AI models behind one capability interface. You ask for a **capability**; the
hub resolves a model, starts it if it is cold, calls it, and stores what comes back
as an **artifact**.

## The one rule

Never name an engine, a launch command, a path, or a model id to get work done.
Name the capability:

```
invoke_model({ capability: 'text_to_image', prompt: 'a futuristic city' })
```

Driving Ollama, ComfyUI, or A1111 directly — or launching one with a shell command —
is the hub's job, and the hub can do it: `invoke_model` starts a cold engine itself,
and `start_model` starts one explicitly. If you find yourself writing
`python main.py`, stop and use one of those instead.

## First move: find out what this machine can actually serve

Do not assume a capability exists and do not trust your memory of the catalog. Two
calls answer everything:

- `list_capabilities()` — the capability vocabulary, which ones are **served here**,
  and which are explicitly **not available**. Report the unavailable ones as
  unavailable rather than attempting them.
- `list_models({ capability? })` — every model, its capabilities, live availability,
  lifecycle state, and VRAM/RAM needs. Filter with `capability` when routing matters.

## Tools

| Tool | Use it for |
| --- | --- |
| `list_capabilities` | What this machine can and cannot do, before planning |
| `list_models` | Which models exist, what they declare, whether they are live |
| `invoke_model` | **The normal path.** Run a capability and get artifacts back |
| `explain_routing` | Read-only: why a capability would route to a given model |
| `get_model_status` | Per-model availability, lifecycle, health, and last error |
| `check_model_health` | Probe now — use after a failure or while an engine warms up |
| `list_artifacts` | Recover an artifact id from an earlier step in the conversation |
| `refresh_model_discovery` | Re-read every engine for models installed since the last check |
| `start_model` | Start an engine that is not running, and wait for it to be healthy |
| `stop_model` | Reclaim VRAM after a heavy workflow, or recover a wedged model |

### When a model you expect is missing

A model the machine already has — a freshly pulled Ollama model, a new ComfyUI
checkpoint — may not be in the catalog yet, either because the deployment
configures models by hand or because the discovery cache has not expired.
`refresh_model_discovery` re-reads every configured engine and republishes the
catalog; it reports what each engine contributed and warns about any it could not
reach. Run it before concluding a capability is unavailable, and say plainly if
it reports that runtime discovery is disabled in this deployment.

## Invoking

`invoke_model` takes the capability plus whatever that capability needs:

- `capability` (required) — e.g. `text_to_text`, `text_to_image`, `image_to_image`,
  `text_to_3d`, `image_to_3d`, `audio_generation`, `speech_to_text`,
  `image_understanding`, `video_generation`. Not every machine serves every one.
- `prompt` — the natural-language instruction: the image description, the mesh
  description, the question, or the text to process.
- `inputs` — artifacts to consume, as an id string or `{ id, type }`. Required by
  transforming capabilities (`image_to_image`, `image_to_3d`, `speech_to_text`).
- `options` — capability-specific settings passed through unchanged, e.g.
  `{ width: 1024, height: 1024, steps: 30 }`. Unknown keys are ignored.
- `modelId` — a debug pin that bypasses routing. Use only to compare models; never
  as the default.
- `requiredTags` — constrain routing, e.g. `["gpu"]` or `["low-vram"]`.
- `timeoutMs` — abandon after this long. Defaults to the deployment budget.

The result carries `capability`, `modelId` (which model actually ran), `coldStart`,
`durationMs`, and `outputs` — each with an artifact id you feed into the next step.

## Chaining models

Artifacts are the only thing that crosses a model boundary, so pipelines compose
without you knowing anything about PNG or GLB:

```
const image = await invoke_model({ capability: 'text_to_image', prompt: 'a spaceship' })
await invoke_model({ capability: 'image_to_3d', inputs: [image.outputs[0].id] })
```

Pass ids, never file paths. If an id is lost, `list_artifacts` recovers it. Setting
`inputs: [{ id, type: 'image' }]` asserts the expected kind and fails loudly on a
mismatch instead of silently feeding the wrong thing.

## If the engine is not running, start it — do not refuse the task

An engine being down is the ordinary case, not a failure: Ollama's app may be closed,
ComfyUI may not have been started since the last reboot. Handle it in four steps, in
this order:

1. **Find what serves the capability.** `list_models({ capability: 'text_to_image' })`.
   Two models for one capability is normal — the router picks by priority and health.
2. **Is it up?** `get_model_status({ modelId })` for the cached view, or
   `check_model_health({ modelId })` to probe right now. A model reported
   `stopped [not_running]` means the ENGINE is down, not that the model is broken.
3. **Start it through the hub.** `start_model({ modelId })` launches the engine's
   declared command, waits for its health check, and reports success only once the
   engine can serve. ComfyUI loading a checkpoint can take minutes; the descriptor's
   startup budget covers that, so let the call finish instead of retrying in a loop.
   Starting an engine that is already up is safe and cheap: the hub prefers the
   healthy endpoint over spawning a second copy.
4. **Then do the work.** `invoke_model({ capability, prompt, ... })`.

You rarely need step 3 on its own, because `invoke_model` starts a cold model itself
when the deployment permits it. Reach for `start_model` explicitly to surface a
startup failure separately from an invocation failure, or to preload an engine before
a multi-step chain.

Two things are never the answer: launching the engine yourself with a shell command,
and telling the user the capability is unavailable because the engine is merely
stopped.

### When a model is refused for resources

`explain_routing` and `invoke_model` name the shortfall — *"needs 12 GiB VRAM but
only 6.5 GiB is available (of 8 GiB total)"*. That is the hub measuring this machine
and refusing to start something that would die inside the engine, so it is a real
answer, not a bug: report it, and if another local model is holding the memory,
`stop_model` on it is the fix. `get_model_status` reports the machine's figures,
including what is free and which resident models are already counted against it.

A model refused for **capacity** (it does not fit even an idle machine) is
permanently `unsupported` here; a model refused for **headroom** becomes routable
again once the memory is free.

### When `start_model` answers UNSAFE_OPERATION

That is `allowProcessLaunch: false` — the deployment has forbidden the hub from
starting anything, and retrying never changes it. Say plainly that the engine is not
running and that the operator has to opt in, then name the change; do not work around
it. Two switches must agree before anything launches, and both are data:

1. **The descriptor** marks the engine startable and says how:

```json
"lifecycle": {
  "startable": true,
  "stoppable": true,
  "startupTimeoutMs": 300000,
  "awaitHealthOnStart": true,
  "start": { "command": "python.exe", "args": ["main.py"], "cwd": "C:/ComfyUI" }
}
```

2. **The deployment** opts in through the plugin row's config:
   `allowProcessLaunch: true`.

Only a live server needs `lifecycle`; a model that answers in-process needs none, and
no shipped catalog declares one. The executable is
matched by basename against the built-in allowlist of local inference runtimes
(`python`, `python.exe`, `ollama`, `comfyui`, `llama-server`, `blender`, …), so
`C:/ComfyUI/venv/Scripts/python.exe` and `ollama` need nothing extra. A launcher
outside that list needs `allowAnyCommand: true` — an explicit operator decision, so
read `docs/security.md` before asking for it.

## Adding or changing a model: data, not code

A model is a JSON entry, not a code change. Find the active catalog (the hub logs the
exact path it loaded), then add an entry:

```json
{
  "id": "a1111_sdxl",
  "name": "SDXL (WebUI)",
  "type": "image_generation",
  "host": "a1111",
  "capabilities": ["text_to_image", "image_to_image"],
  "adapterConfig": { "model": "sd_xl_base_1.0.safetensors" },
  "priority": 10,
  "tags": ["local", "gpu", "image"]
}
```

Lower `priority` wins. No router, prompt, or DSH change is needed. Engine-specific
recipes are in `docs/adding-a-model.md`, with ready-to-copy Ollama, llama.cpp,
A1111/Forge, ComfyUI, and 3D entries in `config/examples/real-models.example.json`.
ComfyUI needs an **API-format** workflow template (see `config/workflows/`), not a UI
export — a UI-format graph is rejected.

A **3D engine** follows the same rule. A local image-to-3D server is a `three_d`
model whose `adapterConfig` names the two calls its Gradio app exposes (or one, if
it is a single-call engine), and a host's `adapterConfig.models` declares what that
engine can generate — discovery verifies the declaration against the API surface
the engine actually serves before publishing it. `docs/three-d.md` has the format,
the shipped TRELLIS template, and every error message you are likely to see.
`text_to_3d` is **not** served by any of the current engines: they are image-to-3D,
so the workflow a user asking for text-to-3D actually wants is
`text_to_image` → `image_to_3d`.

## When the hub tools are missing

If `invoke_model` and `list_models` are not in the tool list at all, the plugin is
loaded but **disabled** — it registers no tools rather than failing the boot. In
order, check:

1. **Is the catalog reachable?** The plugin discovers `models.json`,
   `model-catalog.json`, or `dsh-ai-model-hub.json`, directly or in `config/`,
   walking upward from each `searchRoots` entry, then the host process's working
   directory, then its own installation directory. Fix it by creating `models.json`
   in one of those places or setting `configPath`.
2. **Is the plugin in the profile at all?** It must be listed in
   `dsh.profile.bundles`; a package sitting in the profile's `node_modules` is not
   loaded by itself. Re-run the installer and restart DeepSeek Harness.
3. **Read the boot log.** The plugin logs either
   `model hub ready: N model(s), M capability(ies) from <path>` or
   `model hub disabled — <CODE>: <message>`, and on failure it lists **every**
   directory it tried.

## Pitfalls that cost real time

- **A wrong model name can look like success.** With `openai_compatible`, a model
  name Ollama does not have answers HTTP 404; with two models declared for the
  same capability the hub then falls back to the other one. Check the `modelId` in
  the result against what you asked for — a different id means the one you wanted
  was unreachable.
- **`allowProcessLaunch` defaults to false.** Then `start_model` refuses with
  `UNSAFE_OPERATION` and only already-running engines are usable. That is the
  deployment's choice, not a bug — and not a reason to launch the engine yourself.
  Report it and ask the operator to opt in.
- **A stopped engine is not a missing capability.** `list_capabilities` reports what
  the catalog can serve, not what happens to be running right now. Check health, then
  start it, before concluding anything is unavailable.
- **Two names in `cordis.patch.yml` are easy to conflate.** `id` is the row's name in
  the composed tree; `name` is the module specifier the loader imports and must be the
  bare package name, `dsh-ai-model-hub`, whose `.` export is the plugin. Pointing it at
  a subpath (`dsh-ai-model-hub/library`) loads the wrong thing, and a subpath entry
  silently drops the Web UI half.
- **Do not hand-edit `~/.dsh/profiles/<profile>/cordis.yml`.** It is generated; the
  patch layers and `package.json` are the inputs.
- **`unavailable` beats a guess.** When `list_capabilities` reports a capability as
  not served here, say so — adding a model for it is a config edit, and pretending
  it worked is worse than reporting the gap.

## Where things live

- **Generated output** — `<workspace>/artifacts`, the *calling session's* workspace
  rather than the host's working directory. When the user asks where their image
  went, give them that path plus the file name from the artifact; do not leave them
  hunting for it. `list_artifacts` reads the same per-session location.
- `docs/architecture.md` — the design and why each boundary is where it is
- `docs/adding-a-model.md` — the common case: add a model with no code
- `docs/security.md` — read before enabling `allowProcessLaunch` or widening commands
- `config/models.json` — the active catalog
- `config/examples/` — copy-ready real-engine entries
