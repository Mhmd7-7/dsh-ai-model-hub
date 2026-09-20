# dsh-ai-model-hub

A modular local AI model ecosystem for **DeepSeek Harness (DSH)**.

DSH is the agent — it reasons, plans, calls tools, and orchestrates workflows.
It is **not** the model runtime. The specialised models (text, image, 3D, audio,
video) are external and local, reached through a capability interface. DSH never
learns how to launch Stable Diffusion, which Python file starts a mesh generator,
or how a given engine manages its process. It only asks:

> **What capabilities are available?**

Everything below that question is this project's job.

```
┌──────────────────────────────────────────────────────────────────────┐
│  DeepSeek Harness                        agent reasoning, planning, │
│  ─────────────────                       tool invocation, workflows │
│                                                                      │
│      tools: invoke_model · list_models · list_capabilities ·         │
│             get_model_status · start_model · stop_model · …          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  dsh-ai-model-hub DSH plugin
                                │  (the ONLY DSH-aware code)
┌───────────────────────────────▼──────────────────────────────────────┐
│  ModelHub facade                        invokeModel({ capability })  │
│  ┌────────────────┬──────────────────┬───────────────────────────┐   │
│  │ Capability     │ Model Router     │ Runtime Manager           │   │
│  │ Catalog        │ deterministic    │ lifecycle · health ·      │   │
│  │ registry +     │ capability-first │ idle timeout · resources  │   │
│  │ validation     │ selection        │                           │   │
│  └────────────────┴──────────────────┴───────────────────────────┘   │
│  ┌──────────────────────────────┬────────────────────────────────┐   │
│  │ Model Adapters               │ Artifact Store                 │   │
│  │ mock · http_json ·           │ durable, typed, addressable    │   │
│  │ openai_compatible · cli      │ text/image/audio/video/3D/…    │   │
│  └──────────────────────────────┴────────────────────────────────┘   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  spawn (argv, never a shell string)
┌───────────────────────────────▼──────────────────────────────────────┐
│  Local engines: Ollama · llama.cpp · A1111/Forge · ComfyUI · …       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Status

**Phase 1 complete and verified.** Three mock models validate the whole
architecture end to end — catalog → routing → runtime → adapter → artifact →
chained workflow — with 204 tests and no real engine required.

| Phase | Scope | Status |
|---|---|---|
| 1 | Mock models, full architecture, DSH plugin, tests | ✅ done |
| 2 | Real local text model (Ollama / llama.cpp / vLLM / LM Studio) | ✅ done — `openai_compatible` adapter ships |
| 3 | Real local image model (A1111 / ComfyUI) | needs the `http_json` adapter |
| 4 | Real local 3D model | needs a new adapter |
| 5 | Runtime lifecycle + resource-aware routing | ✅ done (machine probing is not yet wired into `ModelHub`) |
| 6 | Multi-model workflows | ✅ done |

Phase 2 needs no new code: the `openai_compatible` adapter is implemented, and
`config/examples/real-models.example.json` contains ready-to-copy Ollama and
llama.cpp entries. Phase 3 still needs the `http_json` adapter: its kind is
declared and validated, but not implemented, so an A1111/ComfyUI entry reports
`no adapter registered for kind "http_json"` until it is. See
[docs/adding-a-model.md](docs/adding-a-model.md).

---

## Quick start

There are two halves, and you can use either without the other.

### A. Use the hub as a library (no DSH needed)

```sh
cd dsh-ai-model-hub

npm install                  # typescript + the DSH packages the plugin links against
npm test                     # 204 tests, ~10 s, no engines required
npm run typecheck            # strict TypeScript, no errors
npm run demo                 # the vertical slice, end to end
npm run demo:workflow        # a 4-step cross-model pipeline
```

The demo prints the full path — capability discovery, routing with reasons,
invocation, a real PNG on disk, then a chained image → 3D step:

```
=== catalog: capability discovery ===
  text_to_image        text → image  via mock_image_model
  image_to_3d          image/text → model_3d  via mock_3d_model
  not available here: audio_generation, speech_to_text, image_understanding, video_generation

=== routing ===
  chose mock_image_model
  rationale: mock_image_model (score 100): cold but startable; priority 100
    [rejected] mock_text_model: does not declare capability "text_to_image"

=== artifact ===
  valid PNG: yes
  size:      66.3 KiB
```

### B. Give the capability to the DeepSeek Harness agent

```sh
# Installs into the 'web' profile (the one behind `dsh web`), then verifies.
npm run install:plugin

# ...or target another profile:
node --no-deprecation scripts/install-plugin.mjs --profile <name>
```

Then **restart DeepSeek Harness** and ask the agent:

```
List the available AI model capabilities.
```

The installer does three things, in order, and is safe to re-run:

1. `npm install` inside `dsh-plugin/` — materialises the DSH peer closure. This
   is required and cannot be skipped; see [Why the installer has three steps](#why-the-installer-has-three-steps).
2. `dsh plugin add` — installs the plugin and registers it as a profile layer,
   because the package declares `dsh.bundle.patch`. No profile file is edited.
3. `node scripts/doctor.mjs` — imports the plugin the same way DSH's loader will,
   and reports precisely what is wrong if it cannot.

Check an installation at any time:

```sh
npm run doctor                          # inspects: is it loadable, and if not, why not
node scripts/smoke.mjs --profile web    # acts: applies the plugin and lists what it registered
```

`doctor` inspects and reports; `smoke` actually runs the plugin against a real
cordis context and prints the tools it registered. Use `smoke` when you want
proof rather than a checklist:

```
Registered tools:
  check_model_health   explain_routing    get_model_status   invoke_model
  list_artifacts       list_capabilities  list_models        start_model
  stop_model
Catalog: model hub ready: 3 model(s), 5 capability(ies) from …/models.json
```

> **Note on flags.** `npm run` forwards flags to the script, but npm 12 parses
> `--profile` as its own config flag and fails. Use the direct
> `node scripts/smoke.mjs --profile <name>` form for a non-default profile, or
> `-Profile` with the PowerShell installer.

### If DSH fails to boot after installing

If the harness dies with:

```
invalid plugin, expect function or object with an "apply" method, received object
```

then the inserted row's `name` is pointing at a **library** rather than at the
plugin package. The row has two names and they are easy to conflate:
`id` is the row's name in the composed tree (used to override or disable it),
while `name` is the module specifier the loader imports — and it must be
`dsh-ai-model-hub-plugin`, which exports `name`/`inject`/`apply`. Naming
`dsh-ai-model-hub` there imports the hub library, which exports classes, and the
whole profile refuses to load. `cordis.patch.yml` documents this in place, and
`npm run doctor` now checks it.

### Where does the catalog come from?

The plugin resolves its catalog from **ordered anchors**, first hit wins:

1. **`configPath`** — an explicit file, if configured. When set it is the only
   candidate: a wrong path is an error, never a silent fallback.
2. Each directory in **`searchRoots`**, for a deployment that names its own.
3. The **working directory of the DSH host process**, walking upward.
4. **This plugin's own installation directory**, walking upward.

At every level discovery accepts `models.json`, `model-catalog.json`, or
`dsh-ai-model-hub.json`, directly or in a `config/` subdirectory.

Anchor 4 is what makes the zero-config case work. DSH installs profile plugins as
junction links and Node resolves them to their real path, so the plugin finds the
`config/models.json` shipped by the checkout it was installed from — no matter
where the host was launched. Anchor 3 covers the opposite case: a host started
inside a project that carries its own catalog.

> **Why the host's working directory and not the session's?** A plugin's `apply`
> runs when the profile boots — before any session exists — and one host serves
> many sessions with different workspaces, so there is no single "session working
> directory" to read at that moment. Anchors 1 and 2 are the explicit way to name
> a workspace. Per-session resolution is a deliberate future step, not an
> oversight; see [docs/roadmap.md](docs/roadmap.md).

If no anchor yields a catalog the plugin logs `model hub disabled` together with
**every directory it tried**, registers no tools, and lets DSH boot normally — a
missing or broken catalog never makes the agent unusable.

### Why the installer has three steps

DSH installs profile plugins as **junction links**, so Node loads the plugin from
its real path in this repository rather than from a copy inside the profile.
Module resolution therefore walks up from *here*, and the `node_modules` that
matters is `dsh-plugin/node_modules` — not the profile's. That directory must
exist, and `dsh plugin add` (which forwards to pnpm) only warns about peer
dependencies instead of installing them.

This is not theoretical: it was found by installing into a throwaway profile,
deleting this workspace's `node_modules`, and watching the plugin fail to
resolve `@deepseek-ai/schemastery`.

If all of this is unappealing, the simplest correct fix is to use `pnpm` with a
workspace that has `hoist-pattern` enabled, or to publish `dsh-ai-model-hub` and
depend on it by version instead of by path. Both are recorded in
[docs/roadmap.md](docs/roadmap.md).

### What the agent can then do

| Ask the agent | Tool it calls |
|---|---|
| "What AI models can you run?" | `list_models` |
| "What can this machine actually do?" | `list_capabilities` |
| "Create a futuristic city image." | `invoke_model` (`text_to_image`) → router picks the model |
| "Now make a 3D model from that image." | `invoke_model` (`image_to_3d`), chaining the artifact id |
| "Why did it pick that model?" | `explain_routing` — read-only |
| "Is the image model running?" | `get_model_status`, `check_model_health` |
| "Start / stop the image model." | `start_model`, `stop_model` |

The agent never names an engine, a command, or a path. That is the whole design.

### Turning on real models

Phase 2 needs no code: copy an Ollama or llama.cpp entry from
[`config/examples/real-models.example.json`](config/examples/real-models.example.json)
into `config/models.json`, point `adapterConfig.model` at a checkpoint you have
actually pulled, restart, and check with `get_model_status`. A1111/ComfyUI entries
do need code, because the `http_json` adapter is not implemented yet. See
[docs/adding-a-model.md](docs/adding-a-model.md).

Nothing is launched by default: `allowProcessLaunch` is `false`, so the hub talks
to engines you run but will not start any. See
[docs/security.md](docs/security.md) before turning that on.

---

## The five-minute mental model

**The agent asks for a capability; the router picks a model.**

```ts
// What the agent does. Note what it does NOT say: no model, no engine, no command.
await tools.invoke_model({ capability: 'text_to_image', prompt: 'a futuristic city' })
```

```ts
// What the hub does, in order:
//   1. resolve the request's artifacts
//   2. filter models by capability, input kinds, tags, resources, availability
//   3. sort deterministically (priority, then id)
//   4. start the chosen model if it is cold
//   5. call its adapter
//   6. persist the outputs as artifacts and return them
```

**A model is data, not code.** Adding one means adding a JSON entry:

```json
{
  "id": "a1111_sdxl",
  "name": "SDXL (WebUI)",
  "type": "image_generation",
  "host": "a1111",
  "capabilities": ["text_to_image", "image_to_image"],
  "adapterConfig": { "model": "sd_xl_base_1.0.safetensors", "steps": 30 },
  "priority": 10,
  "tags": ["local", "gpu", "image"]
}
```

No router change. No prompt change. No DSH change.

**Artifacts are the only thing that crosses a model boundary.** That is what
makes `text → image → 3D` compose without anything knowing about PNG or GLB:

```ts
const image = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a spaceship' })
const mesh  = await hub.invokeModel({ capability: 'image_to_3d', inputs: [image.outputs[0].id] })
```

---

## What is where

```
dsh-ai-model-hub/
├── src/                          the hub — imports NOTHING from DeepSeek Harness
│   ├── catalog/                  capability vocabulary, descriptor schema, registry,
│   │                             published JSON Schema
│   ├── router/                   deterministic capability-first selection
│   ├── runtime/                  process lifecycle, health, idle timeout
│   ├── adapters/                 adapter contract + the mock adapter, PNG/STL/WAV writers
│   ├── artifacts/                artifact contract + the local filesystem store
│   ├── config/                   catalog discovery and loading
│   ├── util/                     process guardrails, primitive validators
│   ├── machine.ts                RAM/VRAM/GPU detection
│   ├── hub.ts                    the facade: the public API
│   └── index.ts                  the public surface
├── dsh-plugin/                   the ONLY DSH-aware code (~1 file per concern)
│   ├── index.ts                  plugin entry: name / inject / apply / Config
│   ├── service.ts                the hub as a cordis service
│   ├── config.ts                 plugin configuration schema
│   ├── types.ts                  the DSH API bridge — where a breaking change lands
│   └── tools/                    discovery · lifecycle · invoke
├── config/
│   ├── models.json               active catalog (mock models)
│   ├── models.mock.json          Phase 1 fixtures
│   └── examples/                 real engines, ready to copy
├── tests/                        204 tests, all runnable without an engine
│                                 (8 spawn-based ones need an unsandboxed shell)
├── examples/vertical-slice.ts    the end-to-end demonstration
└── docs/                         architecture, components, guides
```

**The dependency rule that makes this survivable:** `src/` does not import
`@deepseek-ai/*`. Only `dsh-plugin/` does. DSH is a `0.1.x-rc` line, so breaking
changes are expected; when one lands, the blast radius is the plugin layer, and
the entire hub keeps working and keeps passing its tests.

---

## Documentation

| Document | Read it when |
|---|---|
| [docs/architecture.md](docs/architecture.md) | You want the design, the data flow, and why each boundary is where it is |
| [docs/components.md](docs/components.md) | You are working on one component and want its contract in detail |
| [docs/adding-a-model.md](docs/adding-a-model.md) | You want to add a model — the common case needs no code |
| [docs/security.md](docs/security.md) | You are about to enable process launching or widen the command allowlist |
| [docs/roadmap.md](docs/roadmap.md) | You want to know what Phases 2–6 require |

---

## Requirements

- **Node ≥ 22.6** — the hub runs TypeScript directly through Node's type
  stripping, with no build step.
- **pnpm** — only for `dsh plugin`, which forwards to it.
- No runtime dependencies. The hub is dependency-free by design: a plugin whose
  `node_modules` must resolve inside a DSH profile is a plugin that breaks on the
  next release.

## License

MIT
