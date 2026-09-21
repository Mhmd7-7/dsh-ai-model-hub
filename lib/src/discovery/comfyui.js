/**
 * ComfyUI discovery.
 *
 * ComfyUI has no "list your models" endpoint. What it has is far more useful:
 * `GET /object_info` returns every node class the server has loaded, and each
 * node's `input.required` fields carry the enumerations ComfyUI itself builds
 * from disk. A checkpoint loader's `ckpt_name` field *is* the list of checkpoint
 * files that exist; a UNET loader's `unet_name` field *is* the list of
 * diffusion-only weight files. Discovery reads those enumerations, and that is
 * how it enumerates checkpoints without ever naming one.
 *
 * Three things are extracted:
 *
 * 1. **What files exist** — from the file-listing enums of the loader nodes,
 *    discovered by looking for nodes whose inputs enumerate model files, not by
 *    matching a node name against a list of known loaders.
 * 2. **What capability the engine has** — from a small, explicit, documented
 *    table ({@link CAPABILITY_SIGNALS}) mapping a node class name to the
 *    capability its presence proves. That table is a *vocabulary* decision
 *    ("a node that saves a GLB means this engine can make 3D output"), which is
 *    stable, and deliberately not a model catalog, which is not.
 * 3. **Whether a working graph can be built** — see {@link buildDefaultGraph}.
 *
 * ## Why a template is still needed, and how far discovery gets without one
 *
 * A graph for an arbitrary checkpoint cannot be synthesized in general: which
 * sampler, which scheduler, how many steps, and which conditioning nodes a model
 * wants are properties of the model that no amount of introspection reveals.
 * So for the common shape — one checkpoint loader, one sampler, one decoder, one
 * image saver, all four present in `/object_info` — discovery builds a minimal
 * default API-format graph and writes it into `adapterConfig.workflow`, and for
 * anything more exotic it leaves the field absent. Absence is the honest answer:
 * the ComfyUI adapter's `supports()` then reports "needs a workflow template",
 * which is the same clear error an operator gets today, instead of a graph that
 * queues and fails deep inside the engine.
 *
 * @module dsh-ai-model-hub/discovery/comfyui
 */
import { fetchJson, isRecordLike, slugifyModelId, stableDigest } from "./http.js";
import { DISCOVERED_PRIORITY, ioForCapabilities } from "./types.js";
/** Engine labels this discoverer answers to. */
export const COMFYUI_ENGINES = ['comfyui'];
/** The primary engine label. */
export const COMFYUI_ENGINE = 'comfyui';
/** The endpoint describing every loaded node class. */
const OBJECT_INFO_PATH = '/object_info';
/** Budget for the request, in milliseconds. `object_info` can be large and slow to build. */
const REQUEST_TIMEOUT_MS = 10_000;
/** Tags every discovered model carries. */
const DISCOVERED_TAGS = ['local', 'discovered', COMFYUI_ENGINE];
/**
 * Which loader input field enumerates which kind of weight file.
 *
 * The values are the **field names** ComfyUI's own loader nodes declare, which
 * is a stable part of those nodes' contracts. Matching on the field name rather
 * than on the node class name is what keeps this free of a model list and
 * tolerant of a node pack that renames or wraps a loader: any node, from any
 * pack, whose input enumerates `ckpt_name` files is a checkpoint loader.
 */
export const WEIGHT_FIELDS = {
    checkpoint: 'ckpt_name',
    diffusionModel: 'unet_name',
    lora: 'lora_name',
    vae: 'vae_name',
};
/**
 * The node class → capability lookup table.
 *
 * This is the one place in discovery that names anything about an engine, and it
 * names *node classes*, not models. That distinction is the whole point: node
 * class names are a stable API surface with a published meaning ("a node called
 * `SaveGLB` writes a GLB file"), while model names are user data that changes
 * with every download. A capability whose node pack is not installed must not be
 * claimed, because claiming it routes work into a guaranteed failure; a
 * capability whose node pack *is* installed can be claimed for every checkpoint,
 * because the graph is what determines the output kind.
 *
 * It is deliberately small and extensible: adding a node pack means adding one
 * row, and nothing else in the codebase changes.
 */
export const CAPABILITY_SIGNALS = [
    // 3D export nodes. SaveGLB / SaveOBJ / ExportMesh and friends are the evidence
    // that this install can produce a mesh at all.
    {
        pattern: /saveglb|saveobj|saveg?ltf|exportmesh|exportglb|exportobj|meshexport|save_?mesh/i,
        capabilities: ['text_to_3d', 'image_to_3d'],
        label: 'mesh export',
    },
    // Video export nodes.
    {
        pattern: /savewebm|savevideo|videocombine|vhs_|saveanimated/i,
        capabilities: ['video_generation'],
        label: 'video export',
    },
    // Audio export nodes.
    {
        pattern: /saveaudio|saveflac|savewav|audioencode/i,
        capabilities: ['audio_generation'],
        label: 'audio export',
    },
    // Image-to-image is proven by a node that consumes an image and produces
    // latents — an encoder, or a depth/pose/edge preprocessor feeding a sampler.
    {
        pattern: /vaeencode|encode.*image|depthanything|midas|canny|openpose|lineart|zoedepth|preprocessor/i,
        capabilities: ['image_to_image'],
        label: 'image conditioning',
    },
    // Vision-language nodes are evidence the install can describe an image.
    {
        pattern: /joycaption|florence|blip|llavacaption|imagecaption|wd14|interrogat/i,
        capabilities: ['image_understanding'],
        label: 'image captioning',
    },
];
/**
 * Resource estimates, in gibibytes, selected by filename pattern.
 *
 * **Best-effort, and documented as such.** ComfyUI's `/object_info` enumerates
 * filenames but reports neither file sizes nor architectures, so there is
 * nothing exact to read. Filename keywords such as `xl` or `flux` correlate well
 * with "this is a large model" and cost nothing to check; they are a heuristic
 * on a string, never a table of known checkpoints. The router treats declared
 * resources as a filter rather than a reservation, so an approximate figure is
 * the right kind of answer, and an operator's own static entry always wins on an
 * id collision.
 */
const VRAM_ESTIMATE_PATTERNS = [
    { pattern: /(^|[^a-z])(flux|hunyuan|sd3|2k|2048)([^a-z]|$)/i, vramGb: 16 },
    { pattern: /(^|[^a-z])(xl|sdxl|1024)([^a-z]|$)/i, vramGb: 8 },
    { pattern: /(^|[^a-z])(sd15|sd1[._-]?5|v1[.-]?5|512|turbo|lightning)([^a-z]|$)/i, vramGb: 4 },
];
/** The estimate used when no pattern matches. */
const DEFAULT_VRAM_GB = 6;
/**
 * Parse `/object_info` into the few facts discovery actually needs.
 *
 * The response is an object keyed by node class name; each value describes that
 * node's inputs. Two shapes of `input.required` are traversed: the standard
 * `{ field: [typeOrOptions, opts] }` form, and a bare `{ field: [..options] }`
 * form some packs emit. Anything else is skipped. The function is total — a
 * server that answers with a novel shape yields fewer facts, never a throw —
 * which is what keeps a new ComfyUI release from breaking the whole pass.
 *
 * @param raw - the parsed `/object_info` body.
 * @returns the introspection result.
 */
export function parseComfyObjectInfo(raw) {
    const nodeClasses = [];
    const files = [];
    const seen = new Set();
    const capabilities = [];
    const signals = [];
    if (!isRecordLike(raw))
        return { nodeClasses, files, capabilities, signals };
    for (const className of Object.keys(raw)) {
        nodeClasses.push(className);
        const required = isRecordLike(raw[className]) ? raw[className]['input'] : undefined;
        const fields = isRecordLike(required) ? required['required'] : undefined;
        if (isRecordLike(fields)) {
            for (const [fieldName, fieldSpec] of Object.entries(fields)) {
                const kind = weightKindForField(fieldName);
                if (kind === undefined)
                    continue;
                for (const filename of fileOptionsIn(fieldSpec)) {
                    const key = `${kind}\u0000${filename}`;
                    if (seen.has(key))
                        continue;
                    seen.add(key);
                    files.push({ filename, kind });
                }
            }
        }
        for (const signal of CAPABILITY_SIGNALS) {
            if (!signal.pattern.test(className))
                continue;
            signals.push(`${signal.label} (${className})`);
            for (const capability of signal.capabilities) {
                if (!capabilities.includes(capability))
                    capabilities.push(capability);
            }
        }
    }
    return { nodeClasses, files, capabilities, signals };
}
/**
 * The weight kind an input field enumerates.
 * @param fieldName - the input field's name.
 * @returns the kind, or `undefined` when the field does not enumerate weights.
 */
function weightKindForField(fieldName) {
    for (const [kind, field] of Object.entries(WEIGHT_FIELDS)) {
        if (fieldName === field)
            return kind;
    }
    // LoRA enumerations are commonly namespaced (`lora_name` inside a wrapper, or
    // `lora_1` on a chained loader), so a suffix test catches those too.
    if (fieldName.endsWith('lora_name'))
        return 'lora';
    return undefined;
}
/**
 * The filename options inside one input field's spec.
 *
 * ComfyUI documents an enumerable input as `[["a.safetensors", "b.ckpt"], {...}]`
 * — a two-element array whose first element is the option list. This reads that
 * shape, and also tolerates a spec that is *only* the option list, which some
 * node packs emit.
 *
 * @param fieldSpec - the field's spec value.
 * @returns every string option found.
 */
function fileOptionsIn(fieldSpec) {
    if (!Array.isArray(fieldSpec))
        return [];
    const candidates = Array.isArray(fieldSpec[0]) ? fieldSpec[0] : fieldSpec;
    if (!Array.isArray(candidates))
        return [];
    const options = [];
    for (const option of candidates) {
        if (typeof option === 'string' && option.trim().length > 0)
            options.push(option);
    }
    return options;
}
/**
 * The capabilities a discovered ComfyUI model can serve.
 *
 * `text_to_image` is a property of having a checkpoint at all: any checkpoint
 * this install can load and any sampler is a text-to-image path. The rest come
 * from the installed node packs — see {@link CAPABILITY_SIGNALS} — and apply to
 * every checkpoint, because in ComfyUI the graph, not the checkpoint, decides
 * the output kind.
 *
 * @param introspection - what the node graph revealed.
 * @returns the capability list, never empty.
 */
export function capabilitiesForComfyModel(introspection) {
    const capabilities = ['text_to_image'];
    for (const capability of introspection.capabilities) {
        if (!capabilities.includes(capability))
            capabilities.push(capability);
    }
    return capabilities;
}
/**
 * The `type` a descriptor declares for a capability set.
 *
 * `type` is operator-facing grouping; the router filters on `capabilities`. A
 * model that can make 3D or video output is described by that, since it is the
 * most specific thing it can do.
 *
 * @param capabilities - the declared capabilities.
 * @returns the model type.
 */
function modelTypeForCapabilities(capabilities) {
    if (capabilities.includes('text_to_3d') || capabilities.includes('image_to_3d'))
        return 'three_d_generation';
    if (capabilities.includes('video_generation'))
        return 'video_generation';
    if (capabilities.includes('audio_generation'))
        return 'audio_generation';
    if (capabilities.includes('image_to_image'))
        return 'image_editing';
    return 'image_generation';
}
/**
 * Estimate a checkpoint's VRAM need from its filename.
 * @param filename - the filename ComfyUI enumerated.
 * @returns the estimate in gibibytes.
 */
export function estimateComfyVram(filename) {
    for (const rule of VRAM_ESTIMATE_PATTERNS) {
        if (rule.pattern.test(filename))
            return rule.vramGb;
    }
    return DEFAULT_VRAM_GB;
}
/**
 * The default workflow's placeholders.
 *
 * Every value here is a *starting point* an operator is expected to tune; none
 * of them names a model. They are the conventional ComfyUI defaults for a
 * generic checkpoint, expressed as data so the generated graph is readable.
 */
const DEFAULT_GRAPH_SETTINGS = {
    steps: 20,
    cfg: 7,
    sampler: 'euler',
    scheduler: 'normal',
    width: 1024,
    height: 1024,
    filenamePrefix: 'hub/discovered',
};
/** The node classes a minimal default graph needs, by the role they play. */
const GRAPH_ROLES = {
    checkpointLoader: 'CheckpointLoaderSimple',
    latent: 'EmptyLatentImage',
    sampler: 'KSampler',
    decode: 'VAEDecode',
    save: 'SaveImage',
};
/**
 * Build a minimal API-format graph for a checkpoint, when this install has every
 * node it needs.
 *
 * The graph is the smallest one that produces an image from a prompt:
 *
 * ```
 * CheckpointLoaderSimple → KSampler → VAEDecode → SaveImage
 *                          ↑
 *            EmptyLatentImage + two CLIPTextEncode nodes
 * ```
 *
 * It is returned only when all six node classes exist in `/object_info`. That
 * check is not paranoia: queueing a graph that references a node class the
 * server does not have produces a rejection from deep inside ComfyUI, whereas
 * returning `undefined` produces the adapter's own clear "needs a workflow
 * template" error at routing time — and existence is exactly the kind of thing
 * `/object_info` can answer.
 *
 * This is intentionally the *only* graph shape discovery attempts. Anything
 * requiring a LoRA chain, a controlnet, an upscaler, or a second pass varies too
 * much between models to guess at, and guessing wrong is worse than saying so.
 *
 * @param introspection - what the node graph revealed.
 * @param checkpoint - the checkpoint filename to wire into the loader.
 * @returns the graph, or `undefined` when this install cannot run the default shape.
 */
export function buildDefaultGraph(introspection, checkpoint) {
    const classes = new Set(introspection.nodeClasses);
    const missing = [
        GRAPH_ROLES.checkpointLoader,
        GRAPH_ROLES.latent,
        GRAPH_ROLES.sampler,
        GRAPH_ROLES.decode,
        GRAPH_ROLES.save,
        'CLIPTextEncode',
    ].filter((className) => !classes.has(className));
    if (missing.length > 0)
        return undefined;
    const settings = DEFAULT_GRAPH_SETTINGS;
    return {
        '1': { class_type: GRAPH_ROLES.checkpointLoader, inputs: { ckpt_name: checkpoint } },
        '2': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: '' } },
        '3': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: '' } },
        '4': {
            class_type: GRAPH_ROLES.latent,
            inputs: { width: settings.width, height: settings.height, batch_size: 1 },
        },
        '5': {
            class_type: GRAPH_ROLES.sampler,
            inputs: {
                model: ['1', 0],
                positive: ['2', 0],
                negative: ['3', 0],
                latent_image: ['4', 0],
                seed: 0,
                steps: settings.steps,
                cfg: settings.cfg,
                sampler_name: settings.sampler,
                scheduler: settings.scheduler,
                denoise: 1,
            },
        },
        '6': { class_type: GRAPH_ROLES.decode, inputs: { samples: ['5', 0], vae: ['1', 2] } },
        '7': { class_type: GRAPH_ROLES.save, inputs: { images: ['6', 0], filename_prefix: settings.filenamePrefix } },
    };
}
/**
 * Map one discovered weight file into a descriptor.
 *
 * @param file - the enumerated weight file.
 * @param host - the host it belongs to.
 * @param introspection - what the node graph revealed.
 * @returns the descriptor.
 */
export function mapComfyWeightFile(file, host, introspection) {
    const capabilities = capabilitiesForComfyModel(introspection);
    const { inputTypes, outputTypes } = ioForCapabilities(capabilities);
    const id = slugifyModelId(file.filename, host.runtime.engine || COMFYUI_ENGINE, `checkpoint-${stableDigest(file.filename)}`);
    const vramGb = estimateComfyVram(file.filename);
    const graph = file.kind === 'checkpoint' ? buildDefaultGraph(introspection, file.filename) : undefined;
    return {
        id,
        name: file.filename,
        type: modelTypeForCapabilities(capabilities),
        host: host.id,
        capabilities,
        // Explicit, from CAPABILITY_IO, so a chained workflow's compatibility check
        // sees the same contract a hand-written entry declares — which is precisely
        // what breaks silently if a discoverer infers capabilities but leaves these
        // to the catalog's defaulting.
        inputTypes,
        outputTypes,
        adapterConfig: {
            ...(graph === undefined ? {} : { workflow: graph }),
            discovery: {
                weightKind: file.kind,
                ...(introspection.signals.length === 0 ? {} : { capabilitySignals: [...introspection.signals] }),
            },
        },
        resources: { vramGb, ramGb: vramGb, requiresGpu: true },
        priority: DISCOVERED_PRIORITY,
        tags: [...DISCOVERED_TAGS],
        notes: describeComfyModel(file, introspection, graph !== undefined, vramGb),
    };
}
/**
 * A provenance note assembled from what the engine reported.
 * @param file - the weight file.
 * @param introspection - what the node graph revealed.
 * @param hasGraph - whether a default graph was synthesized.
 * @param vramGb - the estimate that was applied.
 * @returns the note.
 */
function describeComfyModel(file, introspection, hasGraph, vramGb) {
    const facts = [
        `Discovered from ComfyUI's /object_info (${file.kind} file); not listed in models.json.`,
    ];
    if (introspection.signals.length > 0)
        facts.push(`Capability signals: ${introspection.signals.join(', ')}.`);
    if (hasGraph) {
        facts.push('A default one-checkpoint graph was generated, so this model needs no template to start from.');
    }
    else {
        facts.push('No workflow template is attached and this install could not run the default one-checkpoint graph, ' +
            'so ComfyUI will report that it needs a template until one is added to adapterConfig.workflowPath.');
    }
    facts.push(`VRAM estimate ${vramGb} GiB, inferred from the filename; treat as approximate.`);
    return facts.join(' ');
}
/**
 * Build the ComfyUI discoverer.
 *
 * @param options - options for the pass.
 * @returns the discoverer.
 */
export function createComfyUiDiscoverer(options = {}) {
    const requestTimeoutMs = Math.max(1, options.requestTimeoutMs ?? REQUEST_TIMEOUT_MS);
    return {
        engine: COMFYUI_ENGINE,
        aliases: COMFYUI_ENGINES.filter((label) => label !== COMFYUI_ENGINE),
        async discover(host, signal) {
            const endpoint = host.runtime.endpoint;
            if (endpoint === undefined || endpoint.trim().length === 0)
                return [];
            const read = await fetchJson(join(endpoint, OBJECT_INFO_PATH), signal, requestTimeoutMs);
            if (!read.ok) {
                // Reported, not swallowed: the registry turns this into a per-host
                // warning and an empty result for this host. `/object_info` is also slow
                // to build on a cold ComfyUI, so a timeout here is ordinary rather than
                // exceptional.
                throw new Error(`could not read /object_info from ${endpoint}: ${read.reason}`);
            }
            const introspection = parseComfyObjectInfo(read.value);
            // One descriptor per *checkpoint*. A LoRA or a bare UNET is not a model
            // this hub can route to on its own — the ComfyUI adapter takes a graph,
            // and these enumerated files are what a graph would reference. They are
            // deliberately not published as models: doing so would advertise a
            // capability with no way to serve it.
            const seen = new Set();
            const descriptors = [];
            for (const file of introspection.files) {
                if (file.kind !== 'checkpoint')
                    continue;
                const descriptor = mapComfyWeightFile(file, host, introspection);
                if (seen.has(descriptor.id))
                    continue;
                seen.add(descriptor.id);
                descriptors.push(descriptor);
            }
            return descriptors;
        },
    };
}
/**
 * Join a base URL and a path without doubling or dropping the separator.
 * @param endpoint - the base URL.
 * @param path - the path.
 * @returns the absolute URL.
 */
function join(endpoint, path) {
    const base = endpoint.endsWith('/') ? endpoint.slice(0, -1) : endpoint;
    return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}
/** Kept linked for the provenance note's future use of the reported node count. */
export function describeIntrospection(introspection) {
    const counts = new Map();
    for (const file of introspection.files)
        counts.set(file.kind, (counts.get(file.kind) ?? 0) + 1);
    const parts = [...counts].map(([kind, count]) => `${count} ${kind}`);
    return `${introspection.nodeClasses.length} node class(es), ${parts.join(', ') || 'no weight files'}`;
}
