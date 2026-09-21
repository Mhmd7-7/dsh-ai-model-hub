# Adding a model

There are three levels of change. Most requests are level 1, which needs **no
code at all** — and in many cases not even an edit, if the engine can report its
own models.

| Level | You want to… | You touch | Code |
|---|---|---|---|
| 0 | Use a model the engine already has | `discoverModels: true` | none |
| 1 | Add a model on an already-supported engine | `config/models.json` | none |
| 2 | Add a new *kind* of engine | one adapter file + one `AdapterKind` value | ~1 file |
| 3 | Add a new capability | the vocabulary + adapter handlers | ~2 files |

The router, the DSH plugin, and DeepSeek Harness are never touched at any level.

---

## Level 0 — let the engine report its own models

If the machine can already see the model, you may not need a JSON entry at all.
Set `discoverModels: true` on the plugin (or `discoverModels: true` on
`ModelHub`), and the hub asks every configured host what it currently has:

| Engine | Where it is asked | What comes back |
|---|---|---|
| Ollama | `/api/tags`, then `/api/show` per model | every pulled model, with vision detected and the declared context window |
| ComfyUI | `/object_info` | every checkpoint file, with capabilities from the installed node packs |
| A1111 / Forge | `/sdapi/v1/sd-models`, `/samplers`, `/options` | every checkpoint, with the loaded one preferred |

Some properties worth knowing before you switch it on:

- **It is off by default**, and off means no engine is ever contacted for
  introspection. Turning it on cannot change the catalog's *static* half.
- **A hand-written entry always wins.** A discovered model whose id a
  `models.json` entry already claims is dropped before the catalog is built, so
  `config/models.json` remains the place to pin a specific checkpoint, set exact
  resources, attach a tuned workflow, or set a priority.
- **An engine that is down is a warning, not a failure.** Discovery reports it,
  contributes no models from that host, and the hub boots normally.
- **Models appear as they are installed.** The result is cached per host for
  `discoveryTtlMs` (60 s by default); the `refresh_model_discovery` tool bypasses
  the cache, which is the path right after an `ollama pull`.
- **Estimates are estimates.** A discovered model's `resources` come from a
  reported file size where the engine provides one and from a filename heuristic
  (does it say `xl`, `sdxl`, `flux`?) where it does not. They are good enough to
  filter a small card out of the running; they are not a specification.

Discovered models appear in `list_models` alongside static ones, with a note
recording that they were discovered and where from. If you want to know exactly
what discovery decided and why, run `refresh_model_discovery` — it reports what
each engine contributed, what was added or removed, and every warning.

---

## Level 1 — a model on a supported engine

### The 60-second version

1. Copy the relevant host and model entries from
   [`config/examples/real-models.example.json`](../config/examples/real-models.example.json)
   into `config/models.json`. The shipped catalog already declares an Ollama host
   and a ComfyUI host, so often the host is there and only a model entry is new.
2. Adjust `id`, `name`, `capabilities`, and `adapterConfig`.
3. Restart DSH, or reload the plugin.
4. Ask the agent: *"what capabilities are available?"*

Nothing in `config/models.json` is synthetic. Every entry names an engine that
must already be running, or that the hub launches only if the entry says
`startable: true` *and* the deployment set `allowProcessLaunch: true`.

### Why hosts exist

A **host** is a shared process. If three checkpoints sit behind one ComfyUI
instance, the endpoint, launch command, resource envelope, and health strategy
are written **once** and each model states only what differs.

```json
{
  "hosts": [
    {
      "id": "ollama",
      "name": "Ollama",
      "adapter": "openai_compatible",
      "runtime": {
        "engine": "ollama",
        "adapter": "openai_compatible",
        "endpoint": "http://127.0.0.1:11434",
        "path": "/v1/chat/completions"
      },
      "health": { "kind": "http", "path": "/api/tags", "timeoutMs": 2000 }
    }
  ],
  "models": [
    {
      "id": "ollama_llama3_8b",
      "name": "Llama 3 8B (Ollama)",
      "type": "text_generation",
      "host": "ollama",
      "capabilities": ["text_to_text"],
      "adapterConfig": { "model": "llama3:8b", "temperature": 0.7 },
      "resources": { "vramGb": 6, "ramGb": 8 },
      "limits": { "contextTokens": 8192 },
      "priority": 10,
      "tags": ["local", "gpu", "text"]
    }
  ]
}
```

A model may instead carry an inline `runtime` — exactly one of `host` or
`runtime`, never both.

### The fields that change routing

Everything else is documentation. These five decide behaviour:

| Field | Effect |
|---|---|
| `capabilities` | **the** filter — a model is only eligible for what it declares |
| `inputTypes` | what artifact kinds it accepts; defaults from the capabilities. Get this wrong and a chained workflow is refused |
| `priority` | lower wins ties; default 100 |
| `resources` | declared needs are checked against the machine; an over-large model is rejected rather than crashing at load |
| `tags` | callers can require them (`requiredTags`), which lets you steer routing without editing descriptors |

### The fields that change behaviour under failure

| Field | Effect |
|---|---|
| `health` | how liveness is decided. Defaults sensibly per adapter: HTTP adapters get an HTTP probe |
| `lifecycle.startupTimeoutMs` | how long a cold start may take before it is a failure (default 120 s) |
| `lifecycle.idleTimeoutMs` | stop after this much idle time. **Off by default** — unloading a model a user is about to reuse is worse than holding VRAM |
| `lifecycle.awaitHealthOnStart` | set `false` for an engine that accepts work before it reports ready |

### Validation tells you everything at once

```sh
node -e "import('./src/index.ts').then(m=>console.log(m.formatIssues('models.json', m.ModelCatalog.validate(JSON.parse(require('fs').readFileSync('config/models.json','utf8'))))))"
```

or simply restart and read the log. A malformed catalog reports every problem
with its path:

```
models.json failed validation with 2 problems:
  - models[2].capabilities[0]: must be one of text_to_text, text_to_image, …
  - models[2].runtime.endpoint: is required for the http_json adapter
```

Editors also get completion and inline errors from
[`model-catalog.schema.json`](../src/catalog/model-catalog.schema.json) (draft
2020-12) — point your editor's JSON schema settings at it.

### Worked example: SDXL behind Automatic1111

```json
{
  "hosts": [
    {
      "id": "a1111",
      "name": "Stable Diffusion WebUI",
      "adapter": "http_json",
      "runtime": {
        "engine": "stable-diffusion-webui",
        "adapter": "http_json",
        "endpoint": "http://127.0.0.1:7860",
        "path": "/sdapi/v1/txt2img"
      },
      "health": { "kind": "http", "path": "/sdapi/v1/sd-models", "timeoutMs": 3000 }
    }
  ],
  "models": [
    {
      "id": "a1111_sdxl",
      "name": "SDXL",
      "type": "image_generation",
      "host": "a1111",
      "capabilities": ["text_to_image", "image_to_image"],
      "adapterConfig": { "model": "sd_xl_base_1.0.safetensors", "steps": 30, "cfgScale": 7 },
      "resources": { "vramGb": 8, "ramGb": 16, "requiresGpu": true },
      "limits": { "maxWidth": 1536, "maxHeight": 1536 },
      "priority": 10,
      "tags": ["local", "gpu", "image"]
    },
    {
      "id": "a1111_sd15",
      "name": "Stable Diffusion 1.5",
      "type": "image_generation",
      "host": "a1111",
      "capabilities": ["text_to_image", "image_to_image"],
      "adapterConfig": { "model": "v1-5-pruned-emaonly.safetensors", "steps": 25 },
      "resources": { "vramGb": 4, "ramGb": 8 },
      "priority": 80,
      "tags": ["local", "gpu", "image", "low-vram"]
    }
  ]
}
```

Give SDXL `priority: 10` and SD 1.5 `priority: 80`, and routing picks SDXL on a
capable machine while automatically rejecting it — and falling through to SD
1.5 — on an 8 GB card, because SDXL's declared `vramGb` exceeds what was
detected. That is resource-aware routing working with **zero** routing code.

### Adding a model with lifecycle management

To let the hub start and stop an engine, add `lifecycle`:

```json
{
  "lifecycle": {
    "startable": true,
    "stoppable": true,
    "startupTimeoutMs": 180000,
    "shutdownTimeoutMs": 20000,
    "idleTimeoutMs": 900000,
    "start": {
      "command": "python",
      "args": ["launch.py", "--api", "--listen", "--port", "7860"],
      "cwd": "C:/tools/stable-diffusion-webui"
    }
  }
}
```

**Two things must both be true** before anything launches:

1. The descriptor sets `startable: true` with a `start` command, **and**
2. the deployment sets `allowProcessLaunch: true` in the plugin config.

The deployment-level switch means a catalog edit alone can never cause the agent
to launch a heavyweight engine. Read [security.md](security.md) before enabling
it — particularly the allowlist, which does not include shell interpreters or
launcher scripts by default.

A model with **no** `lifecycle` block is *external*: the hub talks to its
endpoint and never spawns anything. That is the right choice for engines you
manage yourself.

---

## Level 2 — a new kind of engine

Write one adapter. You still do not touch the router, the catalog, or DSH.

```ts
// src/adapters/my-engine.ts
import type { AdapterInvocation, AdapterOutput, ModelAdapter } from './types.ts'
import type { HealthReport } from '../types.ts'
import type { ResolvedModel } from '../catalog/descriptor.ts'

export function createMyEngineAdapter(): ModelAdapter {
  return {
    kind: 'my_engine',
    displayName: 'My Engine',

    supports(model: ResolvedModel) {
      // Check what this adapter requires and report a *clear* reason otherwise.
      return model.runtime.endpoint !== undefined
        ? { ok: true }
        : { ok: false, reason: 'requires a runtime.endpoint' }
    },

    async health(model, signal): Promise<HealthReport> {
      // Must never throw: an unreachable engine is a report, not an exception.
      const started = Date.now()
      try {
        const response = await fetch(new URL('/health', model.runtime.endpoint), { signal })
        return { healthy: response.ok, checkedAt: started, latencyMs: Date.now() - started }
      } catch (error) {
        return { healthy: false, checkedAt: started, detail: String(error) }
      }
    },

    async invoke(invocation: AdapterInvocation): Promise<AdapterOutput> {
      const { model, prompt, inputs, options, artifacts, signal, log } = invocation

      // An input artifact is a *reference*; resolve it to a path the engine reads.
      const source = inputs.find((artifact) => artifact.type === 'image')
      const sourcePath = source === undefined ? undefined : (await artifacts.resolvePath(source.id)).path

      log.debug('invoking my engine', { modelId: model.id, hasSource: sourcePath !== undefined })

      const response = await fetch(new URL('/generate', model.runtime.endpoint), {
        method: 'POST',
        signal,                                    // always forward cancellation
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ prompt, image: sourcePath, ...options }),
      })
      if (!response.ok) {
        // Throw ModelHubError with a stable code; the hub wraps unknown throws.
        throw new ModelHubError('INVOCATION_FAILED', `my engine returned ${response.status}`)
      }

      // Persist output through the store so it becomes a durable artifact.
      const bytes = new Uint8Array(await response.arrayBuffer())
      const artifact = await artifacts.put({
        type: 'image',
        bytes,
        mimeType: 'image/png',
        producerModelId: model.id,
        extension: '.png',
        metadata: { format: 'png', prompt },
      })
      return { outputs: [artifact], value: { format: 'png' } }
    },
  }
}
```

Then register it. **Two lines in two places:**

```ts
// src/catalog/descriptor.ts — add to the vocabulary
export const ADAPTER_KINDS = [..., 'my_engine'] as const
```

```ts
// caller — src/hub.ts defaultAdapters, or a deployment's `extraAdapters`
extraAdapters: [createMyEngineAdapter()]
```

Then a model uses it with `"adapter": "my_engine"`.

### Adapter rules that bite

- **Never throw for a negative health finding.** `health()` returns a report.
- **Always forward `signal`.** A cancelled invocation must settle promptly.
- **Resolve inputs through `artifacts.resolvePath()`** for local engines; never
  assume a file path is reachable.
- **Persist every output through `artifacts.put()`.** A bytes blob the agent
  cannot reference is not usable in a workflow.
- **Return lossless JSON in `value`.** The hub validates this and fails loudly,
  because the value rides a tool result into the session log.
- **Let `supports()` catch configuration mistakes.** A `cli` model with no
  `modelPath` should be `unsupported` at startup, not a crash at first use.

---

## Level 3 — a new capability

1. Add it to `CAPABILITIES` and `CAPABILITY_IO` in
   `src/catalog/capabilities.ts`.
2. Add a handler to the test double's `HANDLERS` table in
   `src/adapters/mock.ts`, so the capability is exercisable without an engine.
3. Add a real adapter handler.
4. Mirror it in `src/catalog/model-catalog.schema.json` — the drift test in
   `tests/schema.test.ts` will fail until you do, which is the point.
5. Add a model declaring it.

Nothing else. The router picks up the new capability automatically because it
reads the vocabulary rather than enumerating it, and the agent learns about it
from the generated capability snapshot.

---

## Level 2½ — a discoverer for an engine

Optional, and only worth it if the engine can introspect itself. One file, the
same size as an adapter, and it changes nothing above it.

```ts
// src/discovery/my-engine.ts
import type { ModelDescriptor, ModelHost } from '../catalog/descriptor.ts'
import type { HostDiscoverer } from './types.ts'
import { DISCOVERED_PRIORITY, ioForCapabilities } from './types.ts'
import { fetchJson, slugifyModelId } from './http.ts'

export function createMyEngineDiscoverer(): HostDiscoverer {
  return {
    engine: 'my_engine',          // matches host.runtime.engine
    aliases: ['my-engine'],       // other labels the same engine goes by
    async discover(host: ModelHost, signal: AbortSignal): Promise<ModelDescriptor[]> {
      const read = await fetchJson(`${host.runtime.endpoint}/models`, signal, 4000)
      if (!read.ok) throw new Error(`could not list models: ${read.reason}`)

      // 1. parse: the engine's response → your own candidate records.
      //    Keep this and the mapping below as separate, pure functions so both
      //    are testable against a canned payload with no server.
      const capabilities = ['text_to_image'] as const

      // 2. map: candidates → descriptors. Never a ResolvedModel.
      return (read.value as { files: string[] }).files.map((name) => ({
        id: slugifyModelId(name, host.runtime.engine),   // deterministic
        name,
        type: 'image_generation',
        host: host.id,
        capabilities: [...capabilities],
        ...ioForCapabilities(capabilities),              // explicit, from CAPABILITY_IO
        adapterConfig: { model: name },
        resources: { vramGb: 6, ramGb: 6, requiresGpu: true },
        priority: DISCOVERED_PRIORITY,                   // below the catalog default
        tags: ['local', 'discovered'],
      }))
    },
  }
}
```

Four rules, all of which are tested for the shipped discoverers:

1. **No model identifier in the file.** Match response *shapes*, and decide
   capabilities from structural evidence. A list of known checkpoints is wrong
   the moment someone installs one.
2. **Return descriptors, not resolved models.** `resolveDescriptor()` is the only
   thing that produces a `ResolvedModel`, and it still lives in
   `catalog/registry.ts`. Routing, runtime, and adapters stay untouched.
3. **Never throw for an unreachable engine or a surprising response.** Throw
   *only* to report "this host could not be read", which the registry turns into
   a warning and an empty result for that host. A malformed body should yield
   fewer models, not an exception.
4. **Set `inputTypes`/`outputTypes` explicitly** from `ioForCapabilities()`. They
   drive chained-workflow compatibility: a discoverer that infers capabilities
   but leaves these to the catalog's defaulting can silently break
   `text_to_image → image_to_3d` even when each capability looks right.

Register it in `defaultDiscoverers()` in `src/hub.ts` and export it from
`src/index.ts`, then test it against a real in-process HTTP server returning
canned JSON — the pattern in `tests/discovery-*.test.ts`.

---

## Verification checklist

After adding anything, confirm:

```sh
npm run typecheck    # strict TypeScript
npm test             # unit, integration, adapter, and plugin suites
npm run demo         # the vertical slice still works
```

`npm run demo` needs the engine the catalog names to be running, since the hub
never starts one on its own. Both examples take a catalog path as their first
argument, so a scratch catalog can be demonstrated without touching the shipped
one: `npm run demo -- path/to/catalog.json`.

Then, from an agent session:

```
List the available AI model capabilities.
```

and

```
Use explain_routing for text_to_image and tell me which model and why.
```

If a model does not appear, `get_model_status` names the reason — disabled,
unsupported, missing adapter, over-large for the machine — rather than leaving
you to guess.
