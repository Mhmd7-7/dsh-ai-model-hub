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
│  │ openai_compatible · http_json│ durable, typed, addressable    │   │
│  │ comfyui · three_d ·          │ text/image/audio/video/3D/…    │   │
│  │ mock (tests only)            │                                │   │
│  └──────────────────────────────┴────────────────────────────────┘   │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  spawn (argv, never a shell string)
┌───────────────────────────────▼──────────────────────────────────────┐
│  Local engines: Ollama · llama.cpp · A1111/Forge · ComfyUI ·         │
│  TRELLIS · Hunyuan3D · Stable Fast 3D · TripoSR · …                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Status

**Live as a DSH profile plugin.** The package installs as one unit — `dsh plugin
add` mounts it, so the hub is composed on every boot — and it serves **real local
engines only**: Ollama for text, ComfyUI / A1111 / Forge for images, llama.cpp for
a second text engine, and any local image-to-3D Gradio app for meshes. There are no
mock models anywhere in the shipped catalogs and no mock fallback at runtime: a
wrong model name or a stopped engine is a loud error (or a cold start), never
fixture output.

The whole architecture — catalog → routing → runtime → adapter → artifact →
chained workflow — is covered by the test suite, which uses an in-process **mock
adapter as a test double**. That adapter is not a model: nothing ships it in a
catalog, and no deployment can reach it by accident.

| Phase | Scope | Status |
|---|---|---|
| 1 | Full architecture, DSH plugin, tests | ✅ done |
| 2 | Real local text model (Ollama / llama.cpp / vLLM / LM Studio) | ✅ done — `openai_compatible` adapter ships |
| 3 | Real local image model (A1111 / ComfyUI) | ✅ done — `http_json` and `comfyui` adapters ship |
| 4 | Real local 3D model | ✅ done — `three_d` adapter ships; see [docs/three-d.md](docs/three-d.md) |
| 5 | Runtime lifecycle + resource-aware routing | ✅ done — the machine is probed and routed against |
| 6 | Multi-model workflows | ✅ done |

Phase 2 needs no new code: the `openai_compatible` adapter is implemented, and
`config/examples/real-models.example.json` contains ready-to-copy Ollama and
llama.cpp entries. Phase 3 is implemented by **two** adapters, because the two
engine families are genuinely different:

- **`http_json`** — engines that answer one request with a finished image:
  AUTOMATIC1111, Forge, and `stable-diffusion.cpp`'s `sd-server`. It POSTs the
  familiar `/sdapi/v1/txt2img` body and decodes the base64 images that come back.
- **`comfyui`** — ComfyUI takes a *node graph*, not a prompt. This adapter edits
  a workflow template (the prompt goes to the CLIPTextEncode node wired to the
  sampler's `positive` link; size and sampler settings go to the latent and
  sampler nodes), queues it on `/prompt`, polls `/history`, and downloads the
  result from `/view`.

Phase 4 is **`three_d`**: one adapter that speaks the shape the local 3D ecosystem
actually has — a Gradio queue API or a JSON HTTP route, one or two calls, a mesh
file at the end — so TRELLIS, Hunyuan3D, Stable Fast 3D, and TripoSR are catalog
entries rather than code. It writes a typed `model_3d` artifact (GLB, GLTF, OBJ,
STL, or PLY, sniffed from the bytes), and a discoverer verifies a declared engine
against the API surface it actually exposes before publishing anything.

Phase 5's remaining gap is closed: `ModelHub` now probes RAM, VRAM, free VRAM, and
disk, re-measures on demand, and routes against measured **headroom** — so a model
that would not fit beside what is already loaded is refused with the shortfall in
the message instead of crashing inside the engine.

All adapters are registered by default. See
[docs/adding-a-model.md](docs/adding-a-model.md), [docs/three-d.md](docs/three-d.md),
and the protocol templates in [`config/workflows/`](config/workflows/).

---

## Install

One command, on a machine that already has Node ≥ 22.6 and DeepSeek Harness:

```sh
# into the profile behind `dsh web` — the normal case
dsh plugin --profile web add github:Mhmd7-7/dsh-ai-model-hub
```

This package **is** the plugin: it declares `dsh.bundle.patch`, so `dsh plugin
add` installs it *and* appends `dsh-ai-model-hub` to `dsh.profile.bundles`. The
profile composes that bundle on every boot, which is what makes the plugin
**live**, not merely installed. No profile file is edited by hand and no
`cordis.patch.yml` has to be written.

Then **restart DeepSeek Harness** and ask:

```
List the available AI model capabilities.
```

| What you want | Command |
|---|---|
| Another profile | `dsh plugin --profile hubtest add github:Mhmd7-7/dsh-ai-model-hub` |
| A pinned release | `dsh plugin --profile web add github:Mhmd7-7/dsh-ai-model-hub#v0.2.0` |
| A published copy | `dsh plugin --profile web add dsh-ai-model-hub` |
| A checkout you are editing | `node --no-deprecation scripts/install-plugin.mjs --profile web` |

The last one links the working tree instead of copying it, so the plugin loads
your sources; run `npm run build` there first if you edited anything under
`src/` or `dsh-plugin/`.

`install.ps1` and `install.sh` remain one-line wrappers for the first row, and
keep their flags (`-Profile`, `-Ref`, `-Repository`, `-SkipDoctor`):

```powershell
irm https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.ps1 | iex
```

```sh
curl -fsSL https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.sh | sh
```

To remove an installation:

```sh
dsh plugin --profile web remove dsh-ai-model-hub
```

### Why the package ships compiled JavaScript

The plugin's sources are TypeScript, and Node runs them directly through type
stripping while you work in a checkout. It refuses to do that inside
`node_modules`:

```
ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING: Stripping types is currently
unsupported for files under node_modules
```

A package installed by name — from GitHub, from a registry — lands *inside* the
profile's `node_modules`, so a `.ts` entry there could never be imported. That is
the entire reason this repository commits `lib/`: `lib/` is the same sources
compiled to JavaScript, and the package's entry points (`main`, `exports`) point
at it, so `dsh plugin add` needs no build step on your machine.

`lib/` is generated, never edited: `npm run build` regenerates it from `src/` and
`dsh-plugin/`. The one setting that makes it work is
`rewriteRelativeImportExtensions`, so the emitted JavaScript imports `./x.js`
where the source said `./x.ts`.

---

## Quick start

There are two halves, and you can use either without the other.

### A. Use the hub as a library (no DSH needed)

```sh
cd dsh-ai-model-hub

npm install                  # typescript + the DSH packages the plugin links against
npm test                     # the suite, ~20 s, no engines or GPU required
npm run typecheck            # strict TypeScript, no errors
npm run build                # compile src/ + dsh-plugin/ into lib/
npm run demo                 # the vertical slice, end to end
npm run demo:workflow        # a multi-step cross-model pipeline
npm run demo:3d              # image → 3D → a real .glb, against a stand-in engine
```

The demo prints the full path — capability discovery, routing with reasons,
invocation, and the artifact on disk:

```
=== catalog: capability discovery ===
  text_to_image        text → image  via comfyui_z_image_turbo
  text_to_text         text → text  via ollama_text_model
  not available here: image_to_3d, image_to_image, audio_generation, …

=== routing ===
  chose comfyui_z_image_turbo
  rationale: comfyui_z_image_turbo (score 1): …
    [rejected] ollama_text_model: does not declare capability "text_to_image"

=== artifact ===
  valid PNG: yes
  size:      1.2 MiB
```

It runs against `config/models.json`, so it needs the engines that catalog names:
Ollama for text, ComfyUI for images. `node examples/vertical-slice.ts <catalog>`
takes another catalog, and the last section deliberately asks for a capability
nobody serves, to show what a refusal looks like.

`npm run demo:3d` needs nothing installed. It starts a stand-in 3D engine inside
the example and runs the *whole* real path against it — discovery, health,
resource-aware routing, a two-step Gradio protocol, and a `.glb` written to disk —
then prints every HTTP request the engine saw, to make the boundary concrete. To
point it at TRELLIS instead, swap the stand-in for your catalog entry; the three
capability calls are identical. See [docs/three-d.md](docs/three-d.md).

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

### Checking an installation

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
  stop_model           refresh_model_discovery
Catalog: model hub ready: 2 model(s), 2 capability(ies) from …/config/models.json
```

> **Note on flags.** `npm run` forwards flags to the script, but npm 12 parses
> `--profile` as its own config flag and fails. Use the direct
> `node scripts/smoke.mjs --profile <name>` form for a non-default profile.

### If DSH fails to boot after installing

If the harness dies with:

```
invalid plugin, expect function or object with an "apply" method, received object
```

then the row `cordis.patch.yml` inserts is pointing at something that is not a
plugin. In this package that can only happen if the row's `name` was changed
away from the bare package name: `dsh-ai-model-hub` resolves to the package's
`.` export, which is the plugin (`lib/dsh-plugin/index.js`, exporting
`name`/`inject`/`apply`/`Config`). The hub **library** lives behind the
`dsh-ai-model-hub/library` subpath instead, precisely so the two cannot be
confused.

The row has two names, and they are easy to conflate: `id` is the row's name in
the composed tree (what a profile's own `cordis.patch.yml` uses to override or
disable it), while `name` is the module specifier the loader imports. Keep
`name` bare — a subpath such as `dsh-ai-model-hub/plugin` would load the host half
but silently drop the **Web UI half**, because DSH's client module system only
resolves a row's package manifest from a bare specifier. `cordis.patch.yml`
documents this in place, and `npm run doctor` checks it.

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

### Installing from a checkout instead of from GitHub

`dsh plugin add <path>` **links** the working tree — DSH installs profile plugins
as junction links — so Node resolves the plugin to its real path in the
repository instead of to a copy inside the profile. Peer dependencies then
resolve by walking up from the package: from `lib/dsh-plugin/` to
`<repo>/node_modules`. Run `npm install` once in the repository and that
directory exists, because `dsh plugin add` forwards to pnpm, which only *warns*
about peer dependencies instead of installing them.

This is not theoretical: it was found by installing into a throwaway profile,
deleting the workspace's `node_modules`, and watching the plugin fail to resolve
`@deepseek-ai/schemastery`.

A linked checkout also loads `lib/`, exactly like an installed copy does — so run
`npm run build` after editing `src/` or `dsh-plugin/`. If you would rather not
think about that, work on the hub as a library (`npm test`, `npm run typecheck`)
and only build when you want the plugin to pick the change up.

The historical obstacle on the GitHub route — Node refusing to strip types for
files under `node_modules` — is what committing `lib/` removes; see [Why the
package ships compiled JavaScript](#why-the-package-ships-compiled-javascript).

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

### Pointing the hub at your engines

The catalog the package ships ([`config/models.json`](config/models.json)) already
declares two real engines — Ollama for text and ComfyUI for images — so the first
run needs no editing at all; it needs those engines installed, and the model
names to match what you actually pulled (`ollama list`) and what your ComfyUI
models directory holds.

To change them, or to add more, edit that file or point the plugin at your own
with `configPath` / `searchRoots`. Everything else is copy-and-adjust from
[`config/examples/real-models.example.json`](config/examples/real-models.example.json):
an A1111/Forge/sd.cpp server uses `http_json`, llama.cpp uses
`openai_compatible`, and ComfyUI uses its own `comfyui` adapter driven by a graph
template — an example is in
[`config/workflows/z-image-turbo.api.json`](config/workflows/z-image-turbo.api.json),
a saved `{client_id, prompt}` payload the adapter accepts directly.

A catalog names those files (`workflowPath` for ComfyUI, `stepsPath` for 3D) with
paths **relative to the catalog itself**, because a catalog can live anywhere and
the process that reads it can be started from anywhere. The shipped
[`config/models.json`](config/models.json) therefore says
`workflows/z-image-turbo.api.json` — `config/` is that file's own directory —
while the same template from inside `config/examples/` is
`../workflows/z-image-turbo.api.json`. Absolute paths also work. A path that does
not resolve names itself and its base in the error rather than guessing.

A wrong model name is a loud failure — the engine answers 404 and the invocation
fails — never fixture text: there are no mock models in any shipped catalog. See
[docs/adding-a-model.md](docs/adding-a-model.md).

#### Let the engines list their own models

Editing JSON is not the only way. Set `discoverModels: true` on the plugin and
the hub also asks each configured host what it currently holds — a pulled Ollama
model, a checkpoint dropped into ComfyUI's models directory, a LoRA someone
installed — and adds those to the catalog with no edit at all:

```yaml
- insert:
    - id: dsh-ai-model-hub
      name: 'dsh-ai-model-hub'
      config:
        discoverModels: true
```

It is **off by default**, and off means no engine is ever contacted for
introspection. With it on:

- `config/models.json` still wins. A discovered model whose id a hand-written
  entry claims is dropped before the catalog is built, so the file stays the
  place to pin a checkpoint, set exact resources, attach a tuned workflow, or fix
  a priority.
- An engine that is down is a warning in the log, never a failed boot.
- Results are cached per host for `discoveryTtlMs` (60 s). The
  `refresh_model_discovery` tool bypasses the cache — that is the path right
  after `ollama pull`.
- Resource figures for discovered models are estimates (from a reported file
  size where the engine gives one, from the filename otherwise), good enough to
  keep an oversized model off a small card and not a specification.

Discovery never bypasses `resolveDescriptor()`: it produces descriptors in the
same shape as a `models.json` entry, and the router, runtime manager, and
adapters are unchanged by it. See
[docs/architecture.md](docs/architecture.md#4-host--model--runtime-discovery).

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
│   ├── adapters/                 adapter contract + the mock test double,
│   │                             PNG/STL/WAV writers, openai_compatible,
│   │                             http_json, comfyui, three_d
│   ├── artifacts/                artifact contract, the local filesystem store,
│   │                             and 3D container sniffing/measurement
│   ├── discovery/                engine introspection: ollama · comfyui · a1111 ·
│   │                             three_d, and the merge that keeps static first
│   ├── config/                   catalog discovery and loading
│   ├── util/                     process guardrails, primitive validators
│   ├── machine.ts                RAM/VRAM/GPU/disk probing, free figures included
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
├── lib/                          src/ and dsh-plugin/ compiled to JavaScript:
│                                 the entry points `dsh plugin add` loads, and
│                                 the reason a GitHub install needs no build
├── cordis.patch.yml              the bundle patch: the row the profile mounts
├── install.ps1                   one-line install from GitHub (Windows/PowerShell)
├── install.sh                    one-line install from GitHub (macOS/Linux)
├── scripts/                      install-plugin · doctor · smoke — install and verification
├── config/
│   ├── models.json               the catalog the package ships: real engines
│   ├── examples/                 more real engines, ready to copy
│   └── workflows/                ComfyUI graph templates and 3D step declarations
├── tests/                        the suite; real local HTTP servers stand in for
│                                 engines, so no GPU or model download is needed
├── examples/                     vertical-slice · workflow · three-d-slice demos
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
| [docs/three-d.md](docs/three-d.md) | You want local image-to-3D working: engine, catalog, discovery, troubleshooting |
| [docs/security.md](docs/security.md) | You are about to enable process launching or widen the command allowlist |
| [docs/roadmap.md](docs/roadmap.md) | You want to know what each phase required and what remains |

## Try it without any engine

```sh
npm test          # the whole suite; real HTTP servers stand in for engines
npm run demo:3d   # discovery → routing → a real .glb → the boundary transcript
npm run demo      # capability discovery and routing against your own catalog
```

---

## Requirements

- **Node ≥ 22.6** — the hub *sources* run directly through Node's type stripping
  while you work in a checkout. The installed package loads the compiled `lib/`
  instead, because Node refuses to strip types under `node_modules`;
  `npm run build` regenerates it.
- **pnpm** — only for `dsh plugin`, which forwards to it.
- **No runtime dependencies.** The hub is dependency-free by design: a plugin
  whose `node_modules` must resolve inside a DSH profile is a plugin that breaks
  on the next release. The DSH packages it imports are peers, supplied by the
  profile that loads it.

## License

MIT
