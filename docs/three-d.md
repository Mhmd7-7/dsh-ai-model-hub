# Local 3D generation

How to go from "turn this image into a 3D model" to a `.glb` on disk, with no key
in the agent's prompt, no Python path in the catalog, and no engine name anywhere
above the adapter.

The short version: **run a local image-to-3D Gradio app, describe its two calls in
one JSON file, and give the agent the capability `image_to_3d`.**

---

## 1. Which engine, and why

There is no dominant local 3D server with a stable REST API, so the hub does not
pretend there is one. What the published engines *do* share is a shape:

| Engine | How it runs | Interface | Output | Rough VRAM |
|---|---|---|---|---|
| [TRELLIS](https://github.com/microsoft/TRELLIS) | `python app.py` → Gradio on 8080 | Gradio queue API, two calls (generate, then extract GLB) | `.glb` | 12 GB+ |
| [Hunyuan3D-2](https://github.com/Tencent-Hunyuan3D-2) | `python api_server.py` or its Gradio app | Gradio app, or a FastAPI server with `/generate` | `.glb`, `.obj` | 12 GB+ |
| [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) | Gradio app | Gradio queue API, single call | `.glb` (UV-unwrapped) | ~6 GB |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | Gradio app | Gradio queue API, single call | `.obj` | ~6 GB |

Every one of them is a **Gradio app**, which is why the shipped adapter speaks
Gradio's queue API rather than any one engine's product. Switching between them is
a catalog edit — see §7.

**Recommendation for an 8 GB card:** Stable Fast 3D first (it fits), TRELLIS only
if you have 12 GB or more. The hub will refuse the ones that do not fit and say
why, rather than letting CUDA fail halfway through.

**`text_to_3d` is not claimed by any of them.** None of these engines generates a
mesh from text alone; a text-to-3D product is a text-to-image model feeding an
image-to-3D model. The hub models that honestly — `text_to_3d` is a capability it
can serve *if* an engine declares a route for it, and today's engines do not — so
the workflow you actually want is:

```
text_to_image  →  image artifact  →  image_to_3d  →  model_3d artifact
```

Both hops are ordinary capability calls; see §6.

---

## 2. Install the engine

The hub never installs anything. Do this once, by hand, following the engine's own
instructions. For TRELLIS, because it is the reference target:

```bash
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
python app.py --host 127.0.0.1 --port 8080
```

Then check it in a browser: `http://127.0.0.1:8080` should show the app, and its
**API** section (or `http://127.0.0.1:8080/gradio_api/config`) should list the
endpoint names — for TRELLIS, `image_to_3d` and `extract_glb`.

Those names are the only engine-specific facts you need. Everything else the hub
reads from the engine at run time.

> **A note on the first call.** Loading TRELLIS's weights takes minutes. The
> adapter's default budget for one generation is **15 minutes** for exactly this
> reason; raise it with `adapterConfig.timeoutMs` if your machine is slower.

---

## 3. Where the model files live

Wherever the engine puts them, which is why the hub does not need to know. TRELLIS
downloads `microsoft/TRELLIS-image-large` into the Hugging Face cache
(`~/.cache/huggingface/hub`, `%USERPROFILE%\.cache\huggingface\hub` on Windows) the
first time `app.py` starts. The engine loads its own weights at launch; the hub
talks to the *server*, not to the checkpoint.

You may still record the weights path in the catalog with `weightsPath` — it is
carried into `list_models` output and artifact provenance, so an operator reading a
listing can see which checkpoint produced a mesh. It is documentation, not a
launch argument.

---

## 4. How the catalog points at the engine

Two edits: a host for the process, a model for what it can generate.

```json
{
  "hosts": [
    {
      "id": "trellis",
      "name": "TRELLIS (image to 3D)",
      "adapter": "three_d",
      "runtime": {
        "engine": "trellis",
        "adapter": "three_d",
        "endpoint": "http://127.0.0.1:8080"
      },
      "adapterConfig": {
        "stepsPath": "config/workflows/three-d-trellis.gradio.json",
        "models": [
          {
            "id": "trellis_image_large",
            "name": "TRELLIS image-large",
            "capabilities": ["image_to_3d"],
            "vramGb": 12,
            "ramGb": 16,
            "requiresGpu": true
          }
        ]
      },
      "lifecycle": { "startable": false, "stoppable": false },
      "resources": { "vramGb": 12, "ramGb": 16, "requiresGpu": true }
    }
  ],
  "models": []
}
```

Three things are worth reading carefully.

**`runtime.endpoint` is the only address the hub needs.** No command, no port
number in agent-visible text, no checkpoint path.

**`adapterConfig.models` is the discovery directive.** A 3D server keeps no
inventory API — it loaded its weights at launch — so it cannot tell the hub which
models it has. The host declares them, and discovery *verifies* them against the
engine's own API surface before publishing anything (see §5). The optional
`capabilities` on an entry is intersected with what the running engine proves: a
declaration the engine cannot back is dropped, not published.

**`adapterConfig.stepsPath` describes the engine's call protocol.** The shipped
[`config/workflows/three-d-trellis.gradio.json`](../config/workflows/three-d-trellis.gradio.json)
is TRELLIS's:

```json
{
  "protocol": "gradio",
  "steps": [
    {
      "apiName": "image_to_3d",
      "bind": {
        "image": "$input",
        "multiimages": [],
        "is_multiimage": false,
        "seed": "$param:seed",
        "ss_guidance_strength": "$param:ss_guidance_strength",
        "ss_sampling_steps": "$param:ss_sampling_steps",
        "slat_guidance_strength": "$param:slat_guidance_strength",
        "slat_sampling_steps": "$param:slat_sampling_steps",
        "multiimage_algo": "stochastic"
      }
    },
    {
      "apiName": "extract_glb",
      "bind": { "state": "$0.0", "mesh_simplify": "$param:mesh_simplify", "texture_size": "$param:texture_size" },
      "resultFormat": "glb"
    }
  ],
  "generationParameters": { "seed": 0, "ss_guidance_strength": 7.5, "ss_sampling_steps": 12, "slat_guidance_strength": 3, "slat_sampling_steps": 12, "mesh_simplify": 0.95, "texture_size": 1024 }
}
```

| Field | Meaning |
|---|---|
| `protocol` | `gradio` (the queue API) or `http_json` (a FastAPI/Flask route that takes JSON) |
| `steps[].apiName` | The name the engine's own API docs list |
| `steps[].bind` | The engine's argument names, in order. A `$`-value is resolved; anything else is a literal |
| `$input` | The request's input image, encoded as a data URI (or base64, or a path — `inputMode`) |
| `$param:<key>` | A value from `generationParameters`, overridable per call by `invoke_model`'s `options` |
| `$N[.M]` | Output `M` of an earlier step `N` — this is how the second call receives the first one's state |
| `resultFormat` | Marks the step whose output *is* the mesh. That step is the primary one |

Every field of that file is validated at catalog load and again before the first
call, so a typo is a named configuration error rather than a mysterious failure ten
minutes into a generation.

---

## 5. How discovery verifies availability

`discoverModels: true` is what turns the declaration above into a routable model.
The `three_d` discoverer asks, in order:

1. **Does the engine exist and is it up?** `GET /gradio_api/config` (Gradio 5) or
   `GET /config` (Gradio 3/4). Both are tried, because which one exists depends on
   the Gradio version and on any mount prefix. A Gradio app writes that document
   only after it has finished importing its model, which makes it a far better
   liveness signal than a TCP accept.
2. **Which operations does it expose?** The named endpoints in that document.
   Capabilities are derived from their *shape* (`image_to_3d`, `img2mesh`,
   `image_to_model`, …) — never from a list of engine names.
3. **Which of the declared models are actually there?** The host's
   `adapterConfig.models`, intersected with what step 2 proved.

Then, for each published model, the runtime adds live state:

| State | What it means |
|---|---|
| `stopped` | The engine is not answering; nothing has started it |
| `available` | The API-description document answered and the surface matches |
| `unhealthy` | Something is listening but it is not a usable Gradio app |
| `unsupported` | This machine cannot satisfy the model's declared resources |
| `error` | The descriptor or its adapter is broken |

**A model is never `available` merely because it is in the catalog.** Either the
engine answered, or it is `stopped` with the reason recorded.

One deliberate refusal: an engine whose only mesh-related routes are *exporters*
(`extract_glb`, `save_obj`, `export_mesh`) is **not** published as a generator. An
exporter writes a mesh the engine was handed. Advertising that as `image_to_3d`
turns a clean routing refusal into a confusing invocation failure, and there is a
test for it.

---

## 6. How the runtime launches it, and how DSH invokes it

### The whole flow

```
user:  "turn this image into a 3D model"
  │
  ▼
DSH agent ── invoke_model({ capability: "image_to_3d", inputs: ["image_…"] })
  │
  ▼
Model Hub
  ├─ catalog      a model declares image_to_3d                       (§4)
  ├─ router       capabilities → input kinds → tags → availability → resources
  ├─ runtime      is the engine up? if not, start it (lifecycle.start)
  ├─ adapter      speak the engine's protocol                        (§4)
  └─ artifacts    store the GLB, return model_3d_…glb
  │
  ▼
DSH agent ── gets an artifact id. It never saw a port, a script, or a process.
```

### Starting the engine through the hub

By default the engine is **external** — you start it, the hub talks to it. That is
the safe default and what `config/models.json` ships with.

To let `start_model` launch it, add a command *and* opt the deployment in:

```json
"lifecycle": {
  "startable": true,
  "stoppable": true,
  "startupTimeoutMs": 600000,
  "awaitHealthOnStart": true,
  "start": {
    "command": "python",
    "args": ["app.py", "--host", "127.0.0.1", "--port", "8080"],
    "cwd": "C:/tools/TRELLIS"
  }
}
```

and set `allowProcessLaunch: true` on the plugin row. Both switches are required:
the descriptor says what *could* start, the deployment says what *may*. `python`
is already on the built-in allowlist; a launcher outside it needs
`allowAnyCommand: true`, which is an explicit operator decision — read
[docs/security.md](security.md) first.

A cold TRELLIS start takes minutes, which is what `startupTimeoutMs: 600000` is
for. `start_model` waits for the health check and reports success only once the
engine can serve, so the call is safe to make before an invocation.

### Invoking it

```ts
// The agent's entire view of 3D generation:
const image = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a futuristic robot' })
const mesh  = await hub.invokeModel({ capability: 'image_to_3d', inputs: [image.outputs[0].id] })
// mesh.outputs[0] → { type: 'model_3d', mimeType: 'model/gltf-binary', uri: 'file://…/model_3d_….glb' }
```

Through DSH it is the same two calls to `invoke_model`, with the artifact id from
the first passed as `inputs`. **Pass ids, never paths.**

Per-call knobs go in `options`, and reach the engine through the `$param:`
bindings:

```
invoke_model({ capability: 'image_to_3d', inputs: ['image_…'],
               options: { ss_sampling_steps: 24, texture_size: 2048 } })
```

A caller's `timeoutMs` is honoured: when it expires, the in-flight HTTP request is
aborted rather than left running.

### What comes back

A `model_3d` artifact — GLB, GLTF, OBJ, STL, or PLY, whichever the engine actually
produced — with metadata a downstream step can read without opening the file:

| Metadata | Source |
|---|---|
| `format` | sniffed from the leading bytes, cross-checked against the engine's claim |
| `mimeType`, `byteLength`, `createdAt` | the writer's own facts |
| `sourceArtifactId`, `sourceHash` | which image it came from, and a digest of the bytes |
| `vertexCount`, `triangleCount` | measured out of the file (OBJ vertex lines, glTF accessors, STL facet count) |
| `generationParameters` | the safe, non-secret parameters the call ran with |
| `validationWarning` | present only when the bytes do not look like the declared container |

The mesh is `outputs[0]`; a turntable preview the engine also produced is
`outputs[1]` when `adapterConfig.previewIndex` is set.

---

## 7. Adding another 3D engine

It is a catalog edit, because the adapter is protocol-driven.

**A single-call engine** (Stable Fast 3D, TripoSR) needs three fields:

```json
"adapterConfig": { "protocol": "gradio", "apiName": "image_to_3d", "inputMode": "data_uri" }
```

If its call takes more than the image, add `bind` with the argument names in order
and `generationParameters` for the defaults. If it returns a mesh somewhere other
than the first output, add `resultAt` (`"0"`, `"data.0"`, …).

**A FastAPI/Flask engine** (Hunyuan3D's `api_server.py`) uses `http_json`:
`apiName` is the route path, `method` defaults to `POST`, and the body is
`{ "data": [...] }` unless `bodyField` names another key.

**An engine in a container or on another host** needs `filePath` — the route that
serves a result file back (`/gradio_api/file=` by default for Gradio). The adapter
reads a reported path locally when it can and falls back to that route when it
cannot.

**An engine with a different call chain** gets its own steps file. Copy
`config/workflows/three-d-trellis.gradio.json`, change the `apiName`s and the
argument order to match what your engine's `/gradio_api/config` lists, and point
`stepsPath` at your copy.

Nothing in `src/`, in the router, in the DSH plugin, or in the agent's prompt
changes. That is the test `tests/three-d.test.ts` enforces: it drives the same
adapter against a fake engine whose API names, argument order, and result shape are
supplied *by the test's catalog entry*, so a passing suite is proof that the
adapter does not know TRELLIS.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no model can serve capability "image_to_3d"` | Nothing declares it | Add the host + model entries (§4) and restart |
| `is not running and is not startable by the hub` | The engine is not up and the descriptor forbids launching it | Start the app yourself, or set `startable` + `allowProcessLaunch` (§6) |
| `could not describe the 3D engine at …` | Not answering, or not a Gradio app | `curl http://127.0.0.1:8080/gradio_api/config` — if that 404s, check the port and the engine's log |
| `is reachable but exposes no image-to-3D route` | The app implements only exporters, or its API is disabled | Check the app's API docs; some apps need `--api` or `share=False` |
| `the gradio protocol needs an apiName` | The steps declaration is missing it | Copy the two names from `/gradio_api/config` |
| `binding for argument "x" names step N, which has not run yet` | A step binds forward | Bindings may only reference *earlier* steps |
| `completed without producing a 3D asset` | The engine answered, but no output looked like a mesh | Set `resultAt` to where its result really is, or `resultFormat` to the container it emits |
| `the engine's result stream … ended without a completion event` | Generation was cancelled inside the app | Look at the engine's console; usually memory |
| `needs N GiB VRAM but only M GiB is available` | Routing refused it against measured headroom | Stop whatever else holds VRAM, or pick a smaller model |
| `validationWarning: content does not carry a recognisable glb signature` | The engine returned an error page or a truncated file | Check the engine log; the artifact is stored but marked |
| `the 3D engine at … was aborted or timed out` | First call still loading weights | Raise `adapterConfig.timeoutMs`; TRELLIS needs minutes on a cold start |

Useful tools while diagnosing:

```
get_model_status({ modelId: 'trellis_image_large' })   # live state, health, pid, resources
check_model_health({ modelId: … })                     # probe right now
explain_routing({ capability: 'image_to_3d', inputArtifactIds: ['image_…'] })
refresh_model_discovery()                              # re-read the engine's API surface
```

---

## 9. Seeing it work without an engine

`npm run demo:3d` runs the whole path — discovery, health, routing, a real
two-step Gradio protocol, a real `.glb` written to disk — against a stand-in engine
started inside the example. It needs no GPU and no download, and it prints exactly
what the engine saw:

```
$ npm run demo:3d
=== discovery: what does the engine expose? ===
  three_d-local-3d-engine — Local 3D engine
    capabilities: image_to_3d
=== route: image_to_3d ===
  chose: three_d-local-3d-engine
  why:   three_d-local-3d-engine (score 500): healthy and ready; priority 500
=== invoke: image_to_3d ===
  artifact: model_3d_local-3d-engine-mesh_8bf91e066555
    format:   glb
    vertices: 3
  file on disk: …/model_3d_local-3d-engine-mesh_8bf91e066555.glb
=== the boundary ===
  the caller named a capability; the engine saw 6 HTTP request(s)
  no model id, engine name, port, script path or process id crossed it.
```

To point it at a real engine instead, swap the stand-in for your catalog entry —
[`examples/three-d-slice.ts`](../examples/three-d-slice.ts) is the same three calls
either way.
