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
| 3 | Real local image model (A1111 / ComfyUI) | ✅ done — `http_json` and `comfyui` adapters ship |
| 4 | Real local 3D model | needs a new adapter |
| 5 | Runtime lifecycle + resource-aware routing | ✅ done (machine probing is not yet wired into `ModelHub`) |
| 6 | Multi-model workflows | ✅ done |

Phase 2 needs no new code: the `openai_compatible` adapter is implemented, and
`config/examples/real-models.example.json` contains ready-to-copy Ollama and
llama.cpp entries. Phase 3 is now implemented too, by **two** adapters, because
the two engine families are genuinely different:

- **`http_json`** — engines that answer one request with a finished image:
  AUTOMATIC1111, Forge, and `stable-diffusion.cpp`'s `sd-server`. It POSTs the
  familiar `/sdapi/v1/txt2img` body and decodes the base64 images that come back.
- **`comfyui`** — ComfyUI takes a *node graph*, not a prompt. This adapter edits
  a workflow template (the prompt goes to the CLIPTextEncode node wired to the
  sampler's `positive` link; size and sampler settings go to the latent and
  sampler nodes), queues it on `/prompt`, polls `/history`, and downloads the
  result from `/view`.

Both are registered by default. See
[docs/adding-a-model.md](docs/adding-a-model.md) and the workflow templates in
[`config/workflows/`](config/workflows/).

---

## Install from GitHub

One line, on a machine that already has Node ≥ 22.6 and DeepSeek Harness:

**Windows (PowerShell)**

```powershell
irm https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.ps1 | iex
```

**macOS / Linux**

```sh
curl -fsSL https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.sh | sh
```

Either one checks the toolchain, clones the repository to
`~/.dsh/plugins/dsh-ai-model-hub` — the directory convention DSH's own plugin
store already uses — installs the plugin into the `web` profile, and then
verifies the result by importing the plugin the way the loader will. Re-run it to
update: it fast-forwards the clone and reinstalls.

Then **restart DeepSeek Harness** and ask:

```
List the available AI model capabilities.
```

| What you want | Windows (PowerShell) | macOS / Linux |
|---|---|---|
| Another profile | `.\install.ps1 -Profile hubtest` | `./install.sh --profile hubtest` |
| A pinned release | `.\install.ps1 -Ref v0.1.0` | `./install.sh --ref v0.1.0` |
| Another location | `.\install.ps1 -InstallDir D:\hub` | `./install.sh --dir /opt/hub` |
| From a fork | `.\install.ps1 -Repository <url>` | `./install.sh --repository <url>` |
| Install without verifying | `.\install.ps1 -SkipDoctor` | `./install.sh --skip-doctor` |

`irm | iex` and `curl | sh` cannot take arguments, so the piped form is
configured through the environment instead: `DSH_PROFILE`, `DSH_MODEL_HUB_DIR`,
`DSH_MODEL_HUB_REF` and `DSH_MODEL_HUB_REPO`, as in
`$env:DSH_PROFILE = 'hubtest'; irm … | iex`.

Run either script from inside a clone and it installs *that* clone and never
fast-forwards it, so a working tree you are editing is never touched. To remove
an installation:

```sh
dsh plugin --profile web remove dsh-ai-model-hub-plugin
rm -rf ~/.dsh/plugins/dsh-ai-model-hub     # PowerShell: Remove-Item -Recurse -Force
```

Why the GitHub route has to clone instead of installing straight from a git URL
is explained under [Why the installer has three
steps](#why-the-installer-has-three-steps).

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

### Why not `dsh plugin add github:...`?

The one-liner that works for a compiled plugin does not work here, and the
obstacle is Node's rather than DSH's:

```sh
dsh plugin --profile web add github:Mhmd7-7/dsh-ai-model-hub   # fails
```

This plugin ships TypeScript and is loaded through Node's type stripping, but
Node refuses to strip types for any file under `node_modules`:

```
ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING: Stripping types is currently
unsupported for files under node_modules
```

pnpm materialises a registry or git dependency *inside* the profile's
`node_modules` — a directory copy, or a link into `node_modules/.pnpm` — so the
loader would fail on the first `.ts` import. Installing from a git URL would not
help for a second reason either: the repository root is the hub *library* and
declares no plugin bundle, while `dsh-plugin/` reaches the library through a
`file:..` dependency, which only means something inside a real checkout.

The junction install is what sidesteps the first problem, and it is worth being
precise about why: the link points at a checkout *outside* `node_modules`, and
Node resolves the module to that real path before deciding whether type stripping
is allowed. That is the whole reason the plugin has to be installed by path.

Lifting the restriction is a structural change, not a configuration one, and
neither option is implemented: compile the plugin to JavaScript at release time
so it can be installed like any other package, or publish both packages to a
registry and depend on the plugin by version.

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

### The hub ships its own skill

The plugin also teaches the agent how to use it.
[`skills/dsh-ai-model-hub/SKILL.md`](skills/dsh-ai-model-hub/SKILL.md) is registered
as a skill provider on `ctx.skills`, the way DSH's own bundled `dsh-badge` skill is,
so:

- **nothing is copied** into `~/.dsh/skills` or a project skill root, and no profile
  file is edited — installing or updating the plugin installs the skill;
- the body stays the file in this checkout, so a `git pull` changes what the agent
  reads on its next load, with no reinstall step and no stale copy;
- registration lands in the **global** layer, which is the layer DSH merges into
  every preset's session catalog, so no preset has to know about it.

The registration is deliberately *not* a declared `inject`. It goes through
`ctx.inject(['skills'], …)`, so a composition with no skill registry — headless, or
the `sdk-minimal` profile — still gets the model tools and simply never runs the
skill callback. A missing or malformed skill file costs one warning and the skill,
never the tools beside it.

The file is an ordinary skill, so it also works the old-fashioned way: link or copy
`skills/dsh-ai-model-hub/` into a `.dsh/skills` root if you would rather not run the
plugin at all. Its frontmatter is the single source of truth for the name and
description on both paths.

### A settings page for what is on this machine

The agent is not the only one who needs to know where the engines are. The plugin
also adds a **Local models** section to the DSH settings, which answers the question
a user actually has: which engines are on this PC, where is each one installed,
is it running, and what is inside it.

| Row | Where it comes from |
|---|---|
| Installed at | The launch command's `cwd`, then a short list of well-known install directories |
| Listens on / running | A live probe of the catalog's endpoint (`Check now`), or the last known state |
| Models | The engine itself where it has an API — Ollama's `/api/tags` — otherwise the files in its model store |
| Hub can start it | The descriptor's `startable`, after the deployment's `allowProcessLaunch` |
| Catalog models / capabilities | The same live hub the tools read, so the page cannot disagree with behaviour |

Engines the catalog does **not** declare are still listed — `a1111`, `comfyui`, and
`ollama` are known by name — with the directories that were checked, so "Forge is
installed but the hub has not been told" is visible instead of silent.

Two halves, both dependency-free:

- `dsh-plugin/inventory.ts` builds the answer and serves it on
  `GET /dsh-ai-model-hub/inventory`. Every probe is bounded and failures are
  contained; the route is injected on demand, so a headless profile with no web
  server keeps every tool and simply serves no page.
- `dsh-plugin/client.js` is the browser half. DSH loads plugin clients through
  `window.__ModuleLoader__` as plain side-effect scripts — no top-level
  `import`/`export`, React handed in through the factory's `require` — so this is
  hand-written JavaScript with **no bundler and no build step**, matching the rest
  of the project. It registers one `settings.section` contribution.

### Turning on real models

Phase 2 needs no code: copy an Ollama or llama.cpp entry from
[`config/examples/real-models.example.json`](config/examples/real-models.example.json)
into `config/models.json`, point `adapterConfig.model` at a checkpoint you have
actually pulled, restart, and check with `get_model_status`.

Image generation needs no code either. For an A1111/Forge/sd.cpp server, copy the
`a1111_sdxl` or `a1111_sd15` entry. For ComfyUI, copy the `comfyui` host and the
`comfyui_z_image_turbo` model from this repository's working setup, and make sure
`adapterConfig.workflowPath` points at an API-format workflow — an example is in
[`config/workflows/z-image-turbo.api.json`](config/workflows/z-image-turbo.api.json).
That file is a saved `{client_id, prompt}` payload, which the adapter accepts
directly. See [docs/adding-a-model.md](docs/adding-a-model.md).

Nothing is launched by default: `allowProcessLaunch` is `false`, so the hub talks
to engines you run but will not start any. See
[docs/security.md](docs/security.md) before turning that on.

### Where the output goes

Artifacts land in **`<workspace>/artifacts`**, where "workspace" means the
*calling session's*, not the host process's working directory. That distinction is
the whole point: `dsh web` is one long-lived process serving many sessions with
different workspaces, and it is built — and used to resolve this path — before any
session exists. So the root is resolved per tool call, from DSH's own
`ctx.sandboxPolicy.resolve({ session }).workspaceRoot`, which is the same value the
file tools treat as the workspace boundary.

The consequence a user notices: switch workspace in the GUI and output follows,
with no configuration. `list_artifacts` reads the same per-call root, so chaining a
workflow still finds what the previous step produced.

Set `artifactRoot` in the plugin row to pin one absolute directory for every
session instead — the right choice when collecting artifacts centrally, the wrong
one when you want each conversation's images beside its code. The hub refuses a
relative root rather than guessing at a base.

Turn it on and engines start themselves. A descriptor that marks its engine
startable — `"startable": true` plus a `start` command, as the `a1111` host in
[`config/examples/real-models.example.json`](config/examples/real-models.example.json)
does — lets `start_model` bring the engine up, and lets `invoke_model` do it
implicitly for a cold model, waiting for the health check before the request is
sent. A stopped ComfyUI then costs one cold start, not a failed task:

```
invoke_model({ capability: 'text_to_image', prompt: 'a red fox in deep snow' })
  → text_to_image served by comfyui_z_image_turbo in 22620 ms (cold start).
```

Two switches have to agree — the descriptor's `startable` and the deployment's
`allowProcessLaunch` — so an 8 GB model cannot be started by a catalog file alone.
Starting an engine that is already healthy is a no-op rather than a duplicate
process, and only processes the hub started can be stopped by `stop_model`.

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
│   ├── adapters/                 adapter contract + mock, PNG/STL/WAV writers,
│   │                             openai_compatible, http_json, comfyui
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
│   ├── skills.ts                 the bundled agent skill: one provider, one file
│   ├── inventory.ts              engine discovery + the route the settings page reads
│   ├── client.js                 the browser half: the "Local models" settings page
│   └── tools/                    discovery · lifecycle · invoke
├── skills/dsh-ai-model-hub/
│   └── SKILL.md                  the agent-facing skill the plugin registers
├── install.ps1                   one-line install from GitHub (Windows/PowerShell)
├── install.sh                    one-line install from GitHub (macOS/Linux)
├── scripts/                      install-plugin · doctor · smoke — install and verification
├── config/
│   ├── models.json               active catalog (mock models)
│   ├── models.mock.json          Phase 1 fixtures
│   ├── examples/                 real engines, ready to copy
│   └── workflows/                ComfyUI API-format graph templates
├── tests/                        251 tests; the adapter ones run against real
│                                 local HTTP servers, so no engine or GPU needed
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
- **git** — only for the [install-from-GitHub](#install-from-github) scripts,
  which clone this repository. A manual clone needs nothing extra.
- No runtime dependencies. The hub is dependency-free by design: a plugin whose
  `node_modules` must resolve inside a DSH profile is a plugin that breaks on the
  next release.

## License

MIT
