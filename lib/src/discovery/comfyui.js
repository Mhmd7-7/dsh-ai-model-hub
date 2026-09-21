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
    textEncoder: 'clip_name',
};
/**
 * The node class → capability lookup table.
 *
 * This is the one place in discovery that names anything about an engine, and it
 * names *node classes*, not models. That distinction is the whole point: node
 * class names are a stable API surface with a published meaning ("a node called
 * `SaveGLB` writes a GLB file"), while model names are user data that changes
 * with every download.
 *
 * **`scope` is the correction of a false positive found twice in the field.** A
 * signal answers "does this install contain the machinery?"; a descriptor answers
 * "can *this model* serve this?" — and those are different questions. Holding one
 * UNET, one text encoder, and one VAE, an install reported `text_to_3d`,
 * `image_to_3d`, `video_generation`, and `audio_generation` because ComfyUI ships
 * generator nodes for all of them, even though every one needs a separate model
 * that was not installed. Claiming a capability routes work into a guaranteed
 * failure, which is worse than omitting it, so only what the discovered weights
 * can serve is granted. The rest is still recorded as a signal, because "the
 * machinery is installed but the weights are not" is exactly what an operator
 * needs to see when a capability they expect is missing.
 *
 * It is deliberately small and extensible: adding a node pack means adding one
 * row, and nothing else in the codebase changes.
 */
export const CAPABILITY_SIGNALS = [
    // 3D generation: always a separate model pack (Hunyuan3D, Stable3D, SV3D) or a
    // cloud API node (Tripo, Meshy, Rodin). The discovered checkpoint does not
    // power it.
    {
        pattern: /hunyuan3d|triposg|triposplat|tripo.*model|meshy.*model|rodin3d|stable3d|sv3d|moge|image.*to.*3d|image.*to.*model/i,
        capabilities: ['text_to_3d', 'image_to_3d'],
        scope: 'install',
        label: '3D generator',
    },
    // Mesh *writers*. Core, and they write a mesh they were handed — which is not
    // generation either, so they grant nothing at all.
    {
        pattern: /saveglb|saveobj|saveg?ltf|save3d|exportmesh|exportglb|exportobj|meshexport|mesh.*to.*file|save_?mesh/i,
        capabilities: [],
        scope: 'install',
        label: 'mesh export',
    },
    // Video export: in practice a dedicated video model, not the discovered one.
    {
        pattern: /savewebm|savevideo|videocombine|vhs_|saveanimated/i,
        capabilities: ['video_generation'],
        scope: 'install',
        label: 'video export',
    },
    // Audio export: likewise a dedicated audio model.
    {
        pattern: /saveaudio|saveflac|savewav|audioencode/i,
        capabilities: ['audio_generation'],
        scope: 'install',
        label: 'audio export',
    },
    // Image-to-image is the *same sampler with different conditioning*, so the
    // discovered weights do serve it: an image is encoded into the latent space the
    // model already works in. This is why it is the one extra capability granted.
    {
        pattern: /vaeencode|encode.*image|depthanything|midas|canny|openpose|lineart|zoedepth|preprocessor/i,
        capabilities: ['image_to_image'],
        scope: 'model',
        label: 'image conditioning',
    },
    // Captioning is its own model (a vision-language checkpoint), not the
    // discovered one.
    {
        pattern: /joycaption|florence|blip|llavacaption|imagecaption|wd14|interrogat/i,
        capabilities: ['image_understanding'],
        scope: 'install',
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
 * The `type` declared to `CLIPLoader` for a generated diffusion-model graph.
 *
 * **This is the one value a generated graph cannot derive from the engine.** A
 * text encoder's architecture is not discoverable from `/object_info`; ComfyUI
 * makes the operator choose, and the choice is per-model. `qwen_image` is the
 * default because it is what the CLIP loader's own enum lists first on the
 * install this was developed against, and because getting it wrong fails loudly
 * with ComfyUI naming the valid values. The generated descriptor says which value
 * was used, and the operator edits it in `adapterConfig.workflow` if their model
 * wants another.
 */
const DEFAULT_CLIP_TYPE = 'qwen_image';
/** The weight precision requested from `UNETLoader`; the engine's own default. */
const DEFAULT_WEIGHT_DTYPE = 'default';
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
            // A signal scoped to the install records provenance only: it says the
            // machinery exists, not that these weights can drive it.
            if (signal.scope !== 'model')
                continue;
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
 * ComfyUI's input-type markers, which appear in an enum slot but name a *type*
 * rather than a file.
 *
 * A dynamically-typed input is declared as `["COMBO", {...}]`, and some
 * third-party nodes put an underlying scalar type in the same position. Treating
 * one of these as a filename publishes a model called after a type marker — a
 * `comfyui-combo` entry that corresponds to no file on disk — so they are
 * filtered out. The test is deliberately narrow and uppercased: a real file
 * named `INT` or `COMBO` would be pathological, and the alternative (accepting
 * them) invents models.
 */
const TYPE_MARKERS = new Set([
    'COMBO',
    'INT',
    'FLOAT',
    'STRING',
    'BOOLEAN',
    'MODEL',
    'CLIP',
    'VAE',
    'LATENT',
    'IMAGE',
    'MASK',
    'CONDITIONING',
]);
/**
 * The filename options inside one input field's spec.
 *
 * ComfyUI documents an enumerable input as `[["a.safetensors", "b.ckpt"], {...}]`
 * — a two-element array whose first element is the option list. This reads that
 * shape, and also tolerates a spec that is *only* the option list, which some
 * node packs emit. Type markers such as `COMBO` are dropped: see
 * {@link TYPE_MARKERS}.
 *
 * @param fieldSpec - the field's spec value.
 * @returns every string option found that names something rather than a type.
 */
function fileOptionsIn(fieldSpec) {
    if (!Array.isArray(fieldSpec))
        return [];
    const candidates = Array.isArray(fieldSpec[0]) ? fieldSpec[0] : fieldSpec;
    if (!Array.isArray(candidates))
        return [];
    const options = [];
    for (const option of candidates) {
        if (typeof option !== 'string')
            continue;
        const value = option.trim();
        if (value.length === 0)
            continue;
        if (TYPE_MARKERS.has(value.toUpperCase()))
            continue;
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
    const capabilities = [];
    // Always text_to_image: a descriptor is only published for a checkpoint, and a
    // checkpoint plus a sampler is a text-to-image path by construction.
    capabilities.push('text_to_image');
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
 * The node classes a diffusion-model graph needs, by the role they play.
 *
 * Deliberately separate from {@link GRAPH_ROLES}: a checkpoint loader supplies
 * model, CLIP, and VAE from one file, while a diffusion-model directory supplies
 * them from three, and the two shapes need different loaders.
 */
const DIFFUSION_GRAPH_ROLES = {
    latent: 'EmptyLatentImage',
    /** Preferred over {@link DIFFUSION_GRAPH_ROLES.latent} when present. */
    sd3Latent: 'EmptySD3LatentImage',
    sampler: 'KSampler',
    decode: 'VAEDecode',
    save: 'SaveImage',
    clipLoader: 'CLIPLoader',
    vaeLoader: 'VAELoader',
};
/**
 * Build a minimal graph for a model that exists as loose weight files rather
 * than a single checkpoint.
 *
 * ComfyUI has two ways to hold a model, and they are not interchangeable:
 *
 * - a **checkpoint** — one file the `CheckpointLoaderSimple` splits into model,
 *   CLIP, and VAE; and
 * - a **diffusion-model directory** — an `unet_name` file plus a separate
 *   `clip_name` text encoder and a separate `vae_name`, each loaded by its own
 *   node.
 *
 * Publishing only the first shape meant a perfectly usable install published
 * *nothing*: observed live, a ComfyUI holding one UNET, one text encoder, and
 * one VAE produced zero models, because the UNET is not a checkpoint. So this
 * builds the second shape when, and only when, every node it needs exists and
 * the CLIP and VAE enumerations are non-empty — an empty enum would produce a
 * graph that ComfyUI rejects, which is exactly what discovery must avoid.
 *
 * **The text-encoder `type` and the latent class are the two unguessable
 * choices.** No introspection reveals which architecture a UNET wants; ComfyUI
 * itself makes the operator pick. So `type` is taken from the CLIP loader's own
 * enum where the server offers a sensible default, and everything chosen here is
 * reported back to the caller to put in the descriptor's notes, with the
 * descriptor left editable.
 *
 * As with the checkpoint shape, anything beyond this — LoRAs, controlnets,
 * upscalers, second passes — is not guessed at.
 *
 * @param introspection - what the node graph revealed.
 * @param model - the diffusion-model (UNET) filename to load.
 * @param options - the CLIP and VAE files to use; each falls back to the first
 *   the engine enumerates, so a single-model install needs no configuration.
 * @returns the graph and what it selected, or `undefined` when this install
 *   cannot run the shape.
 */
export function buildDiffusionModelGraph(introspection, model, options = {}) {
    const classes = new Set(introspection.nodeClasses);
    const required = [
        DIFFUSION_GRAPH_ROLES.sampler,
        DIFFUSION_GRAPH_ROLES.decode,
        DIFFUSION_GRAPH_ROLES.save,
        DIFFUSION_GRAPH_ROLES.clipLoader,
        DIFFUSION_GRAPH_ROLES.vaeLoader,
        'CLIPTextEncode',
    ].filter((className) => !classes.has(className));
    const latentClass = classes.has(DIFFUSION_GRAPH_ROLES.sd3Latent)
        ? DIFFUSION_GRAPH_ROLES.sd3Latent
        : classes.has(DIFFUSION_GRAPH_ROLES.latent)
            ? DIFFUSION_GRAPH_ROLES.latent
            : undefined;
    if (required.length > 0 || latentClass === undefined)
        return undefined;
    const clipName = options.clipName ?? firstFileOfKind(introspection, 'textEncoder');
    const vaeName = options.vaeName ?? firstFileOfKind(introspection, 'vae');
    // An enum with nothing in it means the file is not installed, so the graph
    // would reference a name the server cannot load.
    if (clipName === undefined || vaeName === undefined)
        return undefined;
    const settings = DEFAULT_GRAPH_SETTINGS;
    const clipType = options.clipType ?? DEFAULT_CLIP_TYPE;
    return {
        graph: {
            '1': { class_type: 'UNETLoader', inputs: { unet_name: model, weight_dtype: DEFAULT_WEIGHT_DTYPE } },
            '2': { class_type: DIFFUSION_GRAPH_ROLES.clipLoader, inputs: { clip_name: clipName, type: clipType } },
            '3': { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 0], text: '' } },
            '4': { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 0], text: '' } },
            '5': { class_type: DIFFUSION_GRAPH_ROLES.vaeLoader, inputs: { vae_name: vaeName } },
            '6': {
                class_type: latentClass,
                inputs: { width: settings.width, height: settings.height, batch_size: 1 },
            },
            '7': {
                class_type: DIFFUSION_GRAPH_ROLES.sampler,
                inputs: {
                    model: ['1', 0],
                    positive: ['3', 0],
                    negative: ['4', 0],
                    latent_image: ['6', 0],
                    seed: 0,
                    steps: settings.steps,
                    cfg: settings.cfg,
                    sampler_name: settings.sampler,
                    scheduler: settings.scheduler,
                    denoise: 1,
                },
            },
            '8': { class_type: DIFFUSION_GRAPH_ROLES.decode, inputs: { samples: ['7', 0], vae: ['5', 0] } },
            '9': { class_type: DIFFUSION_GRAPH_ROLES.save, inputs: { images: ['8', 0], filename_prefix: settings.filenamePrefix } },
        },
        clipName,
        vaeName,
        latentClass,
    };
}
/**
 * The first filename of a kind the engine enumerated.
 * @param introspection - what the node graph revealed.
 * @param kind - the weight kind to look for.
 * @returns the filename, or `undefined` when the enum was empty.
 */
function firstFileOfKind(introspection, kind) {
    return introspection.files.find((file) => file.kind === kind)?.filename;
}
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
 * This is intentionally one of only *two* graph shapes discovery attempts — see
 * {@link buildDiffusionModelGraph} for the other. Anything requiring a LoRA
 * chain, a controlnet, an upscaler, or a second pass varies too much between
 * models to guess at, and guessing wrong is worse than saying so.
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
    const id = slugifyModelId(file.filename, host.runtime.engine || COMFYUI_ENGINE, `model-${stableDigest(file.filename)}`);
    const vramGb = estimateComfyVram(file.filename);
    // The two shapes a model can take here, each with its own loader path. A
    // checkpoint carries its own CLIP and VAE; a diffusion model does not, so the
    // encoder and VAE are chosen from what the engine enumerates and reported in
    // the descriptor's notes.
    const selection = file.kind === 'checkpoint'
        ? checkpointSelection(buildDefaultGraph(introspection, file.filename))
        : file.kind === 'diffusionModel'
            ? diffusionSelection(buildDiffusionModelGraph(introspection, file.filename))
            : undefined;
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
            ...(selection === undefined ? {} : { workflow: selection.graph }),
            discovery: {
                weightKind: file.kind,
                ...(selection?.clipName === undefined ? {} : { textEncoder: selection.clipName }),
                ...(selection?.vaeName === undefined ? {} : { vae: selection.vaeName }),
                ...(selection?.clipType === undefined ? {} : { clipType: selection.clipType }),
                ...(introspection.signals.length === 0 ? {} : { capabilitySignals: [...introspection.signals] }),
            },
        },
        resources: { vramGb, ramGb: vramGb, requiresGpu: true },
        priority: DISCOVERED_PRIORITY,
        tags: [...DISCOVERED_TAGS],
        notes: describeComfyModel(file, introspection, selection, vramGb),
    };
}
/**
 * Summarize capability signals by kind rather than listing every node class.
 *
 * A real install produced 58 firing signals, and pasting them into a descriptor's
 * `notes` — which reaches `list_models` and an agent's context — buried the two
 * lines that matter. The full list stays in `adapterConfig.discovery`, where it is
 * machine-readable and free; the note gets a count per label.
 *
 * @param signals - the signals that fired, as `label (NodeClass)`.
 * @returns a short summary such as `3D generator ×18, image conditioning ×20`.
 */
export function summarizeSignals(signals) {
    const counts = new Map();
    for (const signal of signals) {
        const label = signal.replace(/\s*\([^)]*\)$/, '').trim();
        counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    return [...counts].map(([label, count]) => `${label} ×${count}`).join(', ');
}
/** The labels of signals that fired but granted no capability to this model. */
function installScopedLabels(introspection) {
    const granted = new Set();
    for (const signal of CAPABILITY_SIGNALS) {
        if (signal.scope === 'model')
            for (const capability of signal.capabilities)
                granted.add(capability);
    }
    const labels = new Set();
    for (const signal of introspection.signals) {
        const label = signal.replace(/\s*\([^)]*\)$/, '').trim();
        const definition = CAPABILITY_SIGNALS.find((candidate) => candidate.label === label);
        if (definition === undefined || definition.scope === 'model')
            continue;
        if (definition.capabilities.every((capability) => !introspection.capabilities.includes(capability))) {
            labels.add(label);
        }
    }
    return [...labels];
}
/**
 * Narrow a checkpoint graph into a {@link GraphSelection}.
 * @param graph - the graph, or `undefined` when the shape could not be built.
 * @returns the selection, or `undefined`.
 */
function checkpointSelection(graph) {
    return graph === undefined ? undefined : { graph };
}
/**
 * Narrow a diffusion-model graph into a {@link GraphSelection}, recording the
 * two choices that could not be derived from the engine.
 * @param selection - what {@link buildDiffusionModelGraph} produced.
 * @returns the selection, or `undefined`.
 */
function diffusionSelection(selection) {
    if (selection === undefined)
        return undefined;
    return {
        graph: selection.graph,
        clipName: selection.clipName,
        vaeName: selection.vaeName,
        clipType: DEFAULT_CLIP_TYPE,
        assumption: `A separate text encoder (${selection.clipName}) and VAE (${selection.vaeName}) were wired in, because this ` +
            `model is stored as loose weight files rather than a checkpoint. The text encoder's declared type is ` +
            `"${DEFAULT_CLIP_TYPE}": that is the one value no engine introspection reveals, so check it against your model ` +
            `and change it in adapterConfig.workflow if ComfyUI rejects the graph.`,
    };
}
/**
 * A provenance note assembled from what the engine reported.
 * @param file - the weight file.
 * @param introspection - what the node graph revealed.
 * @param selection - what a generated graph wired up, or `undefined` when none was.
 * @param vramGb - the estimate that was applied.
 * @returns the note.
 */
function describeComfyModel(file, introspection, selection, vramGb) {
    const facts = [
        `Discovered from ComfyUI's /object_info (${file.kind} file); not listed in models.json.`,
    ];
    if (selection === undefined) {
        facts.push('No workflow template is attached and this install could not run either graph shape discovery knows ' +
            '(one checkpoint, or one diffusion model with a text encoder and a VAE), so ComfyUI will report that it ' +
            'needs a template until one is added to adapterConfig.workflowPath.');
    }
    else if (selection.assumption !== undefined) {
        facts.push(selection.assumption);
    }
    else {
        facts.push('A default one-checkpoint graph was generated, so this model needs no template to start from.');
    }
    const summary = summarizeSignals(introspection.signals);
    if (summary.length > 0)
        facts.push(`Installed node machinery: ${summary}.`);
    const withheld = installScopedLabels(introspection);
    if (withheld.length > 0) {
        // The single most useful line on the note: it tells an operator that the
        // capability they are looking for is one download away, instead of implying
        // the engine cannot do it.
        facts.push(`Not served by this model, because it needs separate weights that are not installed: ${withheld.join(', ')}.`);
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
            // One descriptor per *routeable model*: a checkpoint, or a diffusion model
            // the generated graph can actually load. A LoRA is not published — it is a
            // modifier, not a model, and there is no descriptor shape for "apply this
            // to that checkpoint" yet. A diffusion model is published only when a graph
            // for it could be built, so the catalog never advertises something the
            // adapter would then refuse.
            const seen = new Set();
            const descriptors = [];
            for (const file of introspection.files) {
                if (file.kind !== 'checkpoint' && file.kind !== 'diffusionModel')
                    continue;
                const descriptor = mapComfyWeightFile(file, host, introspection);
                if (descriptor.adapterConfig?.['workflow'] === undefined)
                    continue;
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
