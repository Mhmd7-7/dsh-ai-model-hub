/**
 * The `comfyui` adapter: image generation and 3D generation through a ComfyUI
 * server, with no graph-building knowledge above this file.
 *
 * Why this is not `http_json`
 * ---------------------------
 * A1111 and stable-diffusion.cpp answer a single request with the finished
 * image. ComfyUI does not: it takes a *node graph*, queues it, returns an id,
 * executes asynchronously, and only then exposes outputs by filename. There is
 * no request body that means "draw this prompt". The catalog example says as
 * much — "ComfyUI has a JSON graph API rather than a simple prompt endpoint, so
 * it needs its own adapter" — and this is that adapter.
 *
 * What it serves
 * --------------
 * Two families of work, and they are the same machinery underneath:
 *
 * - **`text_to_image` / `image_to_image`** — a prompt and settings written into a
 *   sampler graph, whose output is a PNG.
 * - **`image_to_3d` / `text_to_3d`** — an image and/or a prompt written into a 3D
 *   workflow (TRELLIS.2, Hunyuan3D, Stable Fast 3D, and whatever is installed
 *   next), whose output is a mesh file. ComfyUI's own 3D nodes write GLB, so the
 *   deliverable is a `model_3d` artifact whose container is sniffed from its bytes
 *   rather than assumed.
 *
 * The one capability-specific thing here is *where the input goes*: an image that
 * drives an image-to-3D graph belongs on a `LoadImage` node, and a prompt that
 * drives a sampler or a conditioning node belongs on a `CLIPTextEncode` node.
 * Both are located structurally — the adapter never names a node id, and never
 * names a workflow.
 *
 * How a prompt reaches a graph
 * ----------------------------
 * A graph is model-specific, so this adapter does not invent one. It takes a
 * *template* graph (an API-format workflow, either inline as `workflow` or from
 * a file named by `workflowPath`) and edits it:
 *
 *   1. the prompt text is written to the CLIPTextEncode node that feeds the
 *      sampler's `positive` input;
 *   2. width/height go to the latent node (`Empty*Latent*`), when the graph has
 *      one — a 3D graph conditioned on an image usually does not, and inventing a
 *      latent node for it would be wrong;
 *   3. seed/steps/cfg/sampler go to the sampler node;
 *   4. an input image is uploaded to `/upload/image` and its returned filename is
 *      written to the first `LoadImage` node, which is what makes an
 *      `image_to_3d` request reachable at all.
 *
 * Every target node is discovered from the graph's own wiring, so a template
 * keeps working when node ids are renumbered by the ComfyUI editor. Explicit
 * `*NodeId` settings override discovery when a graph is unusual.
 *
 * @module dsh-ai-model-hub/adapters/comfyui
 */
import { readFile } from 'node:fs/promises';
import { measureThreeD, sniffThreeDFormat, threeDFormatInfo, threeDFormatOf } from "../artifacts/formats.js";
import { isThreeDFilename } from "../discovery/comfyui-workflow.js";
import { readContract, loadComfyGraph, validateComfyGraph } from "../comfy/workflow.js";
import { ModelHubError } from "../errors.js";
import { describeAdapterPath, resolveAdapterPath } from "./paths.js";
/** Capabilities this adapter serves by writing a prompt into a graph. */
const IMAGE_CAPABILITIES = ['text_to_image', 'image_to_image'];
/**
 * Capabilities this adapter serves by running a 3D workflow.
 *
 * Kept separate from {@link IMAGE_CAPABILITIES} because the two families differ in
 * what they must inject and in what they return, and because `supports` has to be
 * able to say which family a model's capabilities fall into.
 */
const THREE_D_CAPABILITIES = ['text_to_3d', 'image_to_3d'];
/** Every capability this adapter serves, for error messages and support checks. */
const ALL_CAPABILITIES = [...IMAGE_CAPABILITIES, ...THREE_D_CAPABILITIES];
/** Default budget for one queued graph. Diffusion on a laptop GPU is slow. */
const DEFAULT_TIMEOUT_MS = 600_000;
/**
 * Default budget for a 3D workflow, in milliseconds.
 *
 * Longer than the image budget on purpose, and not padding: a TRELLIS-class run is
 * several sampling passes plus voxel decoding, mesh conversion, decimation,
 * unwrapping and texture baking, and the first call of a session additionally pays
 * for loading several gigabytes of weights. A tighter default turns "slow but
 * working" into "mysteriously times out", which is the most common way a local 3D
 * setup is reported as broken.
 */
const DEFAULT_THREE_D_TIMEOUT_MS = 1_800_000;
/** How often to ask ComfyUI whether the queued graph has finished. */
const DEFAULT_POLL_INTERVAL_MS = 750;
/** Response bodies quoted into error messages are bounded to this many characters. */
const MAX_ERROR_BODY_CHARS = 600;
/** How many bytes of a downloaded file are inspected when sniffing its container. */
const SNIFF_PREFIX_BYTES = 4096;
/** The route ComfyUI accepts an input image on. */
const UPLOAD_PATH = '/upload/image';
/**
 * The `/view` `type` a mesh download is fetched with.
 *
 * ComfyUI's output nodes write under `output` and its preview nodes under `temp`,
 * and `SaveGLB`/`Save3DAdvanced` are output nodes. A workflow that writes through
 * a custom node elsewhere overrides this with `adapterConfig.outputViewType`.
 */
const DEFAULT_VIEW_TYPE = 'output';
/** Read a finite number from options, then config. */
function numberSetting(options, config, key) {
    const fromOptions = options[key];
    if (typeof fromOptions === 'number' && Number.isFinite(fromOptions))
        return fromOptions;
    const fromConfig = config[key];
    if (typeof fromConfig === 'number' && Number.isFinite(fromConfig))
        return fromConfig;
    return undefined;
}
/** Read a non-empty string from options, then config. */
function stringSetting(options, config, key) {
    const fromOptions = options[key];
    if (typeof fromOptions === 'string' && fromOptions.length > 0)
        return fromOptions;
    const fromConfig = config[key];
    if (typeof fromConfig === 'string' && fromConfig.length > 0)
        return fromConfig;
    return undefined;
}
/**
 * Resolve every setting for one invocation.
 *
 * A prompt is **optional for a 3D capability and required for an image one**:
 * `image_to_3d` is driven by an image and may legitimately carry no text at all,
 * so demanding a prompt there would refuse a request the engine could serve. The
 * prompt is still forwarded when the caller supplied one, because a 3D workflow
 * that has a text-conditioning node can use it.
 *
 * @param invocation - the resolved request.
 * @returns the merged settings.
 * @throws ModelHubError when no prompt was supplied for a prompt-driven capability.
 */
function settingsFor(invocation) {
    const config = invocation.model.adapterConfig;
    const options = invocation.options;
    const isThreeD = THREE_D_CAPABILITIES.includes(invocation.capability);
    const prompt = invocation.prompt ?? (isThreeD ? '' : undefined);
    if (!isThreeD && (prompt === undefined || prompt.trim().length === 0)) {
        throw new ModelHubError('INVOCATION_FAILED', `capability "${invocation.capability}" requires a \`prompt\` and none was supplied`, { modelId: invocation.model.id, capability: invocation.capability });
    }
    const inline = config['workflow'];
    const inlineGraph = inline !== null && typeof inline === 'object' && !Array.isArray(inline)
        ? inline
        : undefined;
    const width = numberSetting(options, config, 'width') ?? invocation.model.limits.maxWidth;
    const height = numberSetting(options, config, 'height') ?? invocation.model.limits.maxHeight;
    return {
        prompt: prompt ?? '',
        ...(stringSetting(options, config, 'negativePrompt') === undefined
            ? {}
            : { negativePrompt: stringSetting(options, config, 'negativePrompt') }),
        ...(width === undefined ? {} : { width: Math.max(1, Math.round(width)) }),
        ...(height === undefined ? {} : { height: Math.max(1, Math.round(height)) }),
        ...(numberSetting(options, config, 'steps') === undefined
            ? {}
            : { steps: Math.max(1, Math.round(numberSetting(options, config, 'steps'))) }),
        ...(numberSetting(options, config, 'cfg') === undefined
            ? {}
            : { cfg: numberSetting(options, config, 'cfg') }),
        ...(numberSetting(options, config, 'seed') === undefined
            ? {}
            : { seed: Math.round(numberSetting(options, config, 'seed')) }),
        ...(numberSetting(options, config, 'denoise') === undefined
            ? {}
            : { denoise: numberSetting(options, config, 'denoise') }),
        ...(stringSetting(options, config, 'sampler') === undefined
            ? {}
            : { sampler: stringSetting(options, config, 'sampler') }),
        ...(stringSetting(options, config, 'scheduler') === undefined
            ? {}
            : { scheduler: stringSetting(options, config, 'scheduler') }),
        ...(stringSetting(options, config, 'filenamePrefix') === undefined
            ? {}
            : { filenamePrefix: stringSetting(options, config, 'filenamePrefix') }),
        clientId: stringSetting(options, config, 'clientId') ?? 'dsh-ai-model-hub',
        pollIntervalMs: Math.max(50, numberSetting(options, config, 'pollIntervalMs') ?? DEFAULT_POLL_INTERVAL_MS),
        timeoutMs: Math.max(1, numberSetting(options, config, 'timeoutMs') ?? (isThreeD ? DEFAULT_THREE_D_TIMEOUT_MS : DEFAULT_TIMEOUT_MS)),
        ...(stringSetting(options, config, 'workflowPath') === undefined
            ? {}
            : { workflowPath: stringSetting(options, config, 'workflowPath') }),
        ...(inlineGraph === undefined ? {} : { workflow: inlineGraph }),
        ...(stringSetting(options, config, 'promptNodeId') === undefined
            ? {}
            : { promptNodeId: stringSetting(options, config, 'promptNodeId') }),
        ...(stringSetting(options, config, 'negativePromptNodeId') === undefined
            ? {}
            : { negativePromptNodeId: stringSetting(options, config, 'negativePromptNodeId') }),
        ...(stringSetting(options, config, 'latentNodeId') === undefined
            ? {}
            : { latentNodeId: stringSetting(options, config, 'latentNodeId') }),
        ...(stringSetting(options, config, 'samplerNodeId') === undefined
            ? {}
            : { samplerNodeId: stringSetting(options, config, 'samplerNodeId') }),
        outputViewType: stringSetting(options, config, 'outputViewType') ?? DEFAULT_VIEW_TYPE,
    };
}
/**
 * Read a template graph from a file.
 *
 * Accepts both shapes found in the wild: a bare graph, and the
 * `{ client_id, prompt }` envelope ComfyUI's own API examples use.
 *
 * The path is resolved by {@link resolveAdapterPath}: a relative
 * `adapterConfig.workflowPath` is relative to the catalog that wrote it, never to
 * `process.cwd()`. That rule is shared with the `three_d` adapter's `stepsPath`,
 * because it is one convention and the two spellings drifted apart once already —
 * `config/workflows/…` read from `<pkg>/config/models.json` composed into
 * `<pkg>/config/config/workflows/…`, a path that has never existed, and the
 * invocation failed on its first call.
 *
 * @param path - the workflow file path, absolute or relative to the catalog.
 * @param catalogDir - the catalog's directory, when the hub was built from a file.
 * @returns the graph.
 * @throws ModelHubError when the file cannot be read or holds no graph.
 */
async function loadWorkflowFile(path, catalogDir) {
    const resolved = resolveAdapterPath(path, catalogDir);
    let text;
    try {
        text = await readFile(resolved.absolute, 'utf8');
    }
    catch (error) {
        throw new ModelHubError('CONFIG_ERROR', `comfyui workflow file ${describeAdapterPath(resolved)} could not be read: ${error instanceof Error ? error.message : String(error)}`, { path: resolved.absolute, base: resolved.base ?? null, origin: resolved.origin });
    }
    let parsed;
    try {
        parsed = JSON.parse(text);
    }
    catch (error) {
        throw new ModelHubError('CONFIG_ERROR', `comfyui workflow file ${describeAdapterPath(resolved)} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`, { path: resolved.absolute, base: resolved.base ?? null, origin: resolved.origin });
    }
    return unwrapGraph(parsed, resolved.absolute);
}
/**
 * Accept either a bare graph or a `{ prompt: graph }` envelope.
 * @param parsed - the parsed document.
 * @param origin - where it came from, for error messages.
 * @returns the graph.
 * @throws ModelHubError when neither shape is present.
 */
function unwrapGraph(parsed, origin) {
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new ModelHubError('CONFIG_ERROR', `comfyui workflow at ${origin} is not a JSON object`, { origin });
    }
    const record = parsed;
    const inner = record['prompt'];
    if (inner !== null && typeof inner === 'object' && !Array.isArray(inner)) {
        return inner;
    }
    return record;
}
/**
 * Find the node id whose node has one of the given class types.
 * @param graph - the graph.
 * @param classTypes - acceptable `class_type` values.
 * @returns the node id, or `undefined`.
 */
function findNodeByClass(graph, classTypes) {
    for (const [id, node] of Object.entries(graph)) {
        if (typeof node.class_type === 'string' && classTypes.includes(node.class_type))
            return id;
    }
    return undefined;
}
/**
 * Find the sampler node: explicit id, else the first KSampler-family node.
 * @param graph - the graph.
 * @param explicit - a configured node id.
 * @returns the node id, or `undefined`.
 */
function findSampler(graph, explicit) {
    if (explicit !== undefined)
        return graph[explicit] === undefined ? undefined : explicit;
    return findNodeByClass(graph, ['KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'KSamplerSelect']);
}
/**
 * Resolve the prompt node from the sampler's `positive` link.
 *
 * This is why node ids need not be pinned: whatever the editor renumbered, the
 * node that feeds `positive` is the one that carries the prompt.
 *
 * @param graph - the graph.
 * @param samplerId - the sampler node id.
 * @returns the node id, or `undefined`.
 */
function findPositiveNode(graph, samplerId) {
    const link = graph[samplerId]?.inputs?.['positive'];
    if (Array.isArray(link) && typeof link[0] === 'string')
        return link[0];
    if (typeof link === 'string')
        return link;
    return findNodeByClass(graph, ['CLIPTextEncode']);
}
/**
 * Resolve the negative prompt node from the sampler's `negative` link.
 * @param graph - the graph.
 * @param samplerId - the sampler node id.
 * @returns the node id, or `undefined`.
 */
function findNegativeNode(graph, samplerId) {
    const link = graph[samplerId]?.inputs?.['negative'];
    if (Array.isArray(link) && typeof link[0] === 'string')
        return link[0];
    if (typeof link === 'string')
        return link;
    return undefined;
}
/**
 * Find the latent node that carries width and height.
 * @param graph - the graph.
 * @param explicit - a configured node id.
 * @returns the node id, or `undefined`.
 */
function findLatent(graph, explicit) {
    if (explicit !== undefined)
        return graph[explicit] === undefined ? undefined : explicit;
    for (const [id, node] of Object.entries(graph)) {
        const kind = node.class_type;
        if (typeof kind === 'string' && kind.startsWith('Empty') && kind.includes('Latent'))
            return id;
    }
    return undefined;
}
/**
 * Apply a mutation to a node's inputs, ignoring missing nodes.
 * @param graph - the graph.
 * @param nodeId - the node id, or `undefined` to do nothing.
 * @param patch - the input fields to set.
 */
function patchInputs(graph, nodeId, patch) {
    if (nodeId === undefined)
        return;
    const node = graph[nodeId];
    if (node === undefined)
        return;
    node.inputs = { ...(node.inputs ?? {}), ...patch };
}
/**
 * Build an image-generation graph by editing a copy of the template.
 *
 * @param settings - resolved settings.
 * @param graph - the template graph.
 * @returns the edited graph.
 * @throws ModelHubError when the template has no usable sampler or prompt node.
 */
function buildImageGraph(settings, graph) {
    const draft = structuredClone(graph);
    const samplerId = findSampler(draft, settings.samplerNodeId);
    if (samplerId === undefined) {
        throw new ModelHubError('INVOCATION_FAILED', 'the comfyui workflow contains no sampler node (expected KSampler or KSamplerAdvanced)', { nodeIds: Object.keys(draft) });
    }
    const promptId = settings.promptNodeId ?? findPositiveNode(draft, samplerId);
    if (promptId === undefined || draft[promptId] === undefined) {
        throw new ModelHubError('INVOCATION_FAILED', 'could not locate the prompt node in the comfyui workflow; set `promptNodeId` explicitly', { samplerId, nodeIds: Object.keys(draft) });
    }
    patchInputs(draft, promptId, { text: settings.prompt });
    if (settings.negativePrompt !== undefined) {
        patchInputs(draft, settings.negativePromptNodeId ?? findNegativeNode(draft, samplerId), {
            text: settings.negativePrompt,
        });
    }
    const latentId = findLatent(draft, settings.latentNodeId);
    patchInputs(draft, latentId, {
        ...(settings.width === undefined ? {} : { width: settings.width }),
        ...(settings.height === undefined ? {} : { height: settings.height }),
    });
    // Seed -1 means "pick one", which is what an image generator should do unless
    // the caller asked for a reproducible render.
    const seed = settings.seed ?? Math.floor(Math.random() * 2 ** 31);
    patchInputs(draft, samplerId, {
        seed,
        ...(settings.steps === undefined ? {} : { steps: settings.steps }),
        ...(settings.cfg === undefined ? {} : { cfg: settings.cfg }),
        ...(settings.sampler === undefined ? {} : { sampler_name: settings.sampler }),
        ...(settings.scheduler === undefined ? {} : { scheduler: settings.scheduler }),
        ...(settings.denoise === undefined ? {} : { denoise: settings.denoise }),
    });
    if (settings.filenamePrefix !== undefined) {
        const saverId = findNodeByClass(draft, ['SaveImage', 'SaveImageWebsocket']);
        patchInputs(draft, saverId, { filename_prefix: settings.filenamePrefix });
    }
    return draft;
}
/** The node classes ComfyUI uses to load an image off disk. */
const IMAGE_LOADER_CLASSES = [
    'LoadImage',
    'LoadImageOutput',
    'LoadImageMask',
    'ImageLoad',
    'LoadImagesFromDirectory',
];
/**
 * The first image-loader node in the graph.
 *
 * Located by class name rather than by wiring because there is nothing to follow:
 * a `LoadImage` node has no upstream link, so no structural inference can find it.
 * The class name is ComfyUI's own contract for "this node reads an image from the
 * input directory", not a model identity, which is why naming it here does not
 * violate the rule that the adapter never hardcodes a model.
 *
 * @param graph - the graph.
 * @returns the node id, or `undefined`.
 */
function findImageLoader(graph) {
    return findNodeByClass(graph, IMAGE_LOADER_CLASSES);
}
/**
 * The node a 3D workflow's prompt should be written to.
 *
 * The sampler's `positive` link is followed first, because that is the structural
 * answer. It is not the only answer in this family: a 3D graph's conditioning often
 * comes from a clip-vision encoder rather than from text, so `positive` may point at
 * a node that has no `text` input at all, and a `CLIPTextEncode` elsewhere in the
 * graph is then the only place a prompt can go. Falling back to the sampler itself
 * — which is what the image path's helper does — would throw away a caller's prompt
 * on a workflow that could have used it, so each candidate is checked for a text
 * field before it is accepted.
 *
 * @param graph - the graph.
 * @param samplerId - the sampler node id, when there is one.
 * @returns the node id, or `undefined` when nothing in the graph takes text.
 */
function findTextNode(graph, samplerId) {
    const candidates = [];
    if (samplerId !== undefined) {
        const linked = findPositiveNode(graph, samplerId);
        if (linked !== undefined && linked !== samplerId)
            candidates.push(linked);
    }
    const encode = findNodeByClass(graph, ['CLIPTextEncode', 'CLIPTextEncodeSDXL', 'TextEncodeQwenImageEdit']);
    if (encode !== undefined)
        candidates.push(encode);
    for (const [id, node] of Object.entries(graph)) {
        if (typeof node.class_type === 'string' && /text/i.test(node.class_type))
            candidates.push(id);
    }
    for (const id of candidates) {
        if (typeof graph[id]?.inputs?.['text'] === 'string')
            return id;
    }
    return undefined;
}
/**
 * Build a 3D-generation graph by editing a copy of the template.
 *
 * A 3D workflow differs from an image one in exactly three ways, and all three are
 * structural rather than model-specific:
 *
 * 1. its input is usually an **image on a loader node**, which the adapter injects
 *    after uploading the artifact to ComfyUI, and never a latent;
 * 2. its **prompt is optional** — a graph with no text-conditioning node simply
 *    does not get one, and a caller who supplied a prompt to such a graph is told
 *    so rather than having their text silently dropped;
 * 3. its **parameters belong to whatever sampler it has**, so steps/cfg/seed are
 *    applied when a sampler is present and skipped when it is not. Width and
 *    height go to a latent node only when one exists, because in this family
 *    resolution is a property of the input image.
 *
 * @param settings - resolved settings.
 * @param graph - the template graph.
 * @param options - `imageName` is the filename the engine returned from an upload.
 * @returns the edited graph.
 * @throws ModelHubError when the capability's required input has nowhere to go.
 */
function buildThreeDGraph(settings, graph, options) {
    const draft = structuredClone(graph);
    const imageNodeId = findImageLoader(draft);
    if (options.wantedImage) {
        if (imageNodeId === undefined) {
            throw new ModelHubError('INVOCATION_FAILED', 'this 3D workflow has no image-loading node, so an image cannot be given to it. ' +
                `Its nodes are: ${describeNodeClasses(draft)}. Point the model at a workflow that takes an image, ` +
                'or set adapterConfig.workflow to one that does.', { nodeIds: Object.keys(draft) });
        }
        if (options.imageName === undefined) {
            throw new ModelHubError('INVOCATION_FAILED', 'the input image was not uploaded to ComfyUI, so the workflow cannot read it', { modelId: settings.clientId });
        }
        // `LoadImage` names the file and derives the mask from it, so one assignment
        // satisfies both of its outputs.
        patchInputs(draft, imageNodeId, { image: options.imageName });
    }
    const samplerId = findSampler(draft, settings.samplerNodeId);
    if (settings.prompt.trim().length > 0) {
        const promptId = settings.promptNodeId ?? findTextNode(draft, samplerId);
        if (promptId === undefined) {
            throw new ModelHubError('INVOCATION_FAILED', 'a prompt was supplied, but this 3D workflow has no node that accepts one, so the prompt would have ' +
                `been discarded. Its nodes are: ${describeNodeClasses(draft)}. ` +
                'Omit the prompt to run it as an image-driven workflow.', { nodeIds: Object.keys(draft) });
        }
        patchInputs(draft, promptId, { text: settings.prompt });
        if (settings.negativePrompt !== undefined) {
            patchInputs(draft, settings.negativePromptNodeId ?? findNegativeNode(draft, samplerId), {
                text: settings.negativePrompt,
            });
        }
    }
    else if (options.wantedPrompt && samplerId !== undefined) {
        // `text_to_3d` with no prompt is a caller mistake rather than a graph one.
        throw new ModelHubError('INVOCATION_FAILED', 'capability "text_to_3d" requires a `prompt`, and none was supplied', { modelId: settings.clientId });
    }
    const latentId = findLatent(draft, settings.latentNodeId);
    patchInputs(draft, latentId, {
        ...(settings.width === undefined ? {} : { width: settings.width }),
        ...(settings.height === undefined ? {} : { height: settings.height }),
    });
    if (samplerId !== undefined) {
        const seed = settings.seed ?? Math.floor(Math.random() * 2 ** 31);
        patchInputs(draft, samplerId, {
            seed,
            ...(settings.steps === undefined ? {} : { steps: settings.steps }),
            ...(settings.cfg === undefined ? {} : { cfg: settings.cfg }),
            ...(settings.sampler === undefined ? {} : { sampler_name: settings.sampler }),
            ...(settings.scheduler === undefined ? {} : { scheduler: settings.scheduler }),
            ...(settings.denoise === undefined ? {} : { denoise: settings.denoise }),
        });
    }
    if (settings.filenamePrefix !== undefined) {
        const saverId = findNodeByClass(draft, ['SaveGLB', 'Save3DAdvanced', 'SaveGaussianSplat', 'MeshToFile3D', 'SaveImage']);
        patchInputs(draft, saverId, { filename_prefix: settings.filenamePrefix });
    }
    return draft;
}
/**
 * Every node class in a graph, for an error message.
 *
 * Bounded, because the interesting failure is "which workflow did I point this at"
 * and a two-hundred-node class list would bury the answer.
 *
 * @param graph - the graph.
 * @returns a comma-separated class list.
 */
function describeNodeClasses(graph) {
    const classes = [...new Set(Object.values(graph).map((node) => String(node.class_type ?? '?')))];
    return classes.slice(0, 12).join(', ') + (classes.length > 12 ? `, … (${classes.length} total)` : '');
}
/**
 * Create the ComfyUI adapter.
 * @returns the adapter instance.
 */
export function createComfyUiAdapter() {
    return {
        kind: 'comfyui',
        displayName: 'ComfyUI graph-queue endpoint',
        supports(model) {
            if (model.runtime.endpoint === undefined) {
                return { ok: false, reason: 'the comfyui adapter requires `runtime.endpoint` (e.g. http://127.0.0.1:8188)' };
            }
            const config = model.adapterConfig;
            const hasTemplate = (typeof config['workflowPath'] === 'string' && config['workflowPath'].length > 0) ||
                (config['workflow'] !== null && typeof config['workflow'] === 'object');
            try {
                readContract(config, model.capabilities);
            }
            catch (error) {
                return { ok: false, reason: error instanceof Error ? error.message : String(error) };
            }
            if (!hasTemplate) {
                return {
                    ok: false,
                    reason: 'the comfyui adapter requires a workflow template: set `adapterConfig.workflowPath` to an ' +
                        'API-format workflow JSON — relative to the catalog that names it, e.g. `workflows/mine.api.json`, ' +
                        'or absolute — or `adapterConfig.workflow` to an inline graph',
                };
            }
            const supported = model.capabilities.filter((capability) => ALL_CAPABILITIES.includes(capability));
            if (supported.length === 0) {
                return {
                    ok: false,
                    reason: `it declares only ${model.capabilities.join(', ') || 'no capabilities'}, and this adapter serves ` +
                        `${ALL_CAPABILITIES.join(', ')}`,
                };
            }
            return { ok: true };
        },
        async health(model, signal) {
            const started = Date.now();
            const endpoint = model.runtime.endpoint;
            if (endpoint === undefined) {
                return { healthy: false, checkedAt: started, detail: 'no runtime.endpoint configured' };
            }
            let url;
            try {
                const base = endpoint.endsWith('/') ? endpoint : `${endpoint}/`;
                url = new URL((model.health.path ?? '/system_stats').replace(/^\//, ''), base).toString();
            }
            catch {
                return { healthy: false, checkedAt: started, detail: `endpoint "${endpoint}" is not a valid URL` };
            }
            // Never throws: this runs on a timer and inside the cold-start gate.
            const controller = new AbortController();
            const onAbort = () => controller.abort();
            if (signal.aborted)
                controller.abort();
            else
                signal.addEventListener('abort', onAbort, { once: true });
            const timer = setTimeout(() => controller.abort(), model.health.timeoutMs ?? 3_000);
            try {
                const response = await fetch(url, { method: 'GET', signal: controller.signal, headers: { accept: '*/*' } });
                return {
                    healthy: response.status < 500,
                    checkedAt: started,
                    latencyMs: Date.now() - started,
                    detail: `${url} responded ${response.status}`,
                };
            }
            catch (error) {
                return {
                    healthy: false,
                    checkedAt: started,
                    latencyMs: Date.now() - started,
                    detail: `${url} unreachable: ${error instanceof Error ? error.message : String(error)}`,
                };
            }
            finally {
                clearTimeout(timer);
                signal.removeEventListener('abort', onAbort);
            }
        },
        async invoke(invocation) {
            const model = invocation.model;
            const isThreeD = THREE_D_CAPABILITIES.includes(invocation.capability);
            if (!isThreeD && !IMAGE_CAPABILITIES.includes(invocation.capability)) {
                throw new ModelHubError('UNSUPPORTED_OPERATION', `the comfyui adapter does not implement capability "${invocation.capability}"`, { modelId: model.id, capability: invocation.capability, supported: [...ALL_CAPABILITIES] });
            }
            const endpoint = model.runtime.endpoint;
            if (endpoint === undefined) {
                throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" has no runtime.endpoint for the comfyui adapter`, { modelId: model.id });
            }
            const settings = settingsFor(invocation);
            const contract = readContract(model.adapterConfig, model.capabilities);
            const template = await loadComfyGraph(model.adapterConfig, invocation.catalogDir);
            validateComfyGraph(template, contract);
            const base = endpoint.endsWith('/') ? endpoint : `${endpoint}/`;
            // One deadline and one controller for the whole operation: upload, queue,
            // execute, and every file fetch share the caller's budget. The controller is
            // created before the graph is built because an `image_to_3d` upload is part
            // of the same budget.
            const controller = new AbortController();
            let timedOut = false;
            const started = Date.now();
            const onCallerAbort = () => controller.abort();
            if (invocation.signal.aborted)
                controller.abort();
            else
                invocation.signal.addEventListener('abort', onCallerAbort, { once: true });
            const timer = setTimeout(() => {
                timedOut = true;
                controller.abort();
            }, settings.timeoutMs);
            const reason = () => {
                if (timedOut) {
                    return new ModelHubError('INVOCATION_TIMEOUT', `model "${model.id}" did not finish within ${settings.timeoutMs} ms`, { modelId: model.id, timeoutMs: settings.timeoutMs });
                }
                if (invocation.signal.aborted) {
                    return new ModelHubError('INVOCATION_ABORTED', 'invocation cancelled while ComfyUI was working', {
                        modelId: model.id,
                    });
                }
                return new ModelHubError('INVOCATION_FAILED', 'ComfyUI request aborted', { modelId: model.id });
            };
            try {
                // What the adapter must inject is decided by the *contract*, not by the
                // capability name: a workflow that binds an image input needs an upload
                // whatever capability brought the caller here, and a scan-inferred
                // contract can express capabilities this adapter has no special case for.
                const wantedImage = contract.bindings['image'] !== undefined;
                if (wantedImage && !invocation.inputs.some((artifact) => artifact.type === 'image')) {
                    throw new ModelHubError('INVOCATION_FAILED', `"${invocation.capability}" needs an input artifact of type \`image\`; this request supplied ` +
                        `${invocation.inputs.length === 0 ? 'none' : invocation.inputs.map((artifact) => artifact.type).join(', ')}. ` +
                        'Pass the image artifact id in `inputs`, or generate one first with text_to_image.', { modelId: model.id, capability: invocation.capability });
                }
                // The image is uploaded first, because its returned *filename* is what the
                // graph's loader node must name — the caller's artifact path is never
                // visible to ComfyUI.
                const imageName = wantedImage ? await uploadInputImage(invocation, base, controller.signal, reason) : undefined;
                const graph = bindPublicInputs(template, contract, invocation, imageName);
                const { promptId, outputs } = await runGraph(invocation, graph, base, settings, controller.signal, reason);
                const selected = selectedOutputs(outputs, contract);
                // Which collector runs follows the declared artifact type, so a workflow
                // that produces a mesh is read as a mesh even when it also writes preview
                // renders, and one that produces images is read as images.
                const producesMesh = Object.values(contract.outputs).some((output) => output.type === 'model_3d');
                return producesMesh
                    ? await collectThreeD(invocation, selected, base, settings, promptId, controller.signal, reason, started)
                    : await collectImages(invocation, selected, base, settings, promptId, controller.signal, reason, started);
            }
            finally {
                clearTimeout(timer);
                invocation.signal.removeEventListener('abort', onCallerAbort);
            }
        },
    };
}
/**
 * The artifact kind each 3D capability is conventionally driven by.
 *
 * Documentation for the capability vocabulary. The adapter itself decides what to
 * inject from the *contract* — a scanned workflow that binds an image is driven
 * by that binding — so nothing branches on this table.
 */
export const THREE_D_SOURCE_KINDS = Object.freeze({
    image_to_3d: 'image',
    text_to_3d: 'text',
});
function bindPublicInputs(template, contract, invocation, imageName) {
    const graph = structuredClone(template);
    const supplied = { ...invocation.options, prompt: invocation.prompt, image: imageName };
    // A workflow whose prompt box the author named something else still has exactly
    // one text binding, and a caller's prompt is what belongs there. Without this a
    // request could route to the workflow and then have its prompt discarded — the
    // failure mode where an image comes back looking nothing like what was asked for.
    if (contract.bindings['prompt'] === undefined && invocation.prompt !== undefined) {
        const textual = Object.keys(contract.bindings).filter((name) => contract.inputKinds?.[name] === 'text');
        if (textual.length === 1)
            supplied[textual[0]] = invocation.prompt;
    }
    // A public binding is a typed scalar; no option can mutate a graph path, class or loader.
    // Defaults in the trusted API graph remain unchanged unless a bound value is supplied.
    for (const [name, binding] of Object.entries(contract.bindings)) {
        let value = supplied[name];
        if (name === 'negative_prompt')
            value = supplied['negativePrompt'] ?? supplied['negative_prompt'];
        if (name === 'seed' && value === undefined)
            value = Math.floor(Math.random() * 2 ** 31);
        if (value === undefined || value === null || value === '')
            continue;
        graph[binding.node].inputs[binding.input] = coerceParameter(name, contract.inputKinds?.[name], value);
    }
    // A prompt is required by the capabilities that are *driven* by one. An
    // image-driven workflow may declare a prompt binding as an optional extra —
    // many 3D graphs accept text conditioning they do not need — so demanding it
    // there would refuse a request the engine could serve.
    const promptDriven = invocation.capability !== 'image_to_image' && invocation.capability !== 'image_to_3d';
    if (promptDriven && contract.bindings['prompt'] !== undefined && !invocation.prompt?.trim()) {
        const capability = invocation.capability;
        throw new ModelHubError('INVOCATION_FAILED', `"${capability}" needs a \`prompt\`, and none was supplied`, { modelId: invocation.model.id, capability });
    }
    if (contract.bindings['image'] !== undefined && !imageName) {
        throw new ModelHubError('INVOCATION_FAILED', `capability "${invocation.capability}" requires an image artifact`);
    }
    return graph;
}
/**
 * Coerce one caller-supplied value to the kind its binding declares.
 *
 * The scan derives each input's kind from the node's declared output type —
 * `BOOLEAN`, `INT`, `FLOAT`, `STRING` — and that is what decides the literal that
 * reaches the graph. Writing `"false"` into a `BOOLEAN` input, or `4096` into a
 * `STRING` one, is a graph ComfyUI rejects for a reason that has nothing to do
 * with what the caller asked for, so the conversion happens here with a message
 * that names the parameter.
 *
 * An undeclared kind — which is what a hand-written contract has — is passed
 * through unchanged, preserving the behaviour those catalogs already rely on.
 *
 * @param name - the public parameter name, for messages.
 * @param kind - the declared kind, when one is known.
 * @param value - the caller's value.
 * @returns the literal to write into the node input.
 * @throws ModelHubError when the value cannot be read as the declared kind.
 */
function coerceParameter(name, kind, value) {
    const reject = (expected) => {
        throw new ModelHubError('INVOCATION_FAILED', `workflow parameter "${name}" must be ${expected} (received ${typeof value === 'string' ? `"${value}"` : String(value)})`, { parameter: name, declaredKind: kind ?? 'unspecified' });
    };
    switch (kind) {
        case 'text':
        case 'image':
            return typeof value === 'string' ? value : reject('a string');
        case 'number': {
            if (typeof value === 'number' && Number.isFinite(value))
                return value;
            if (typeof value === 'string' && value.trim().length > 0 && Number.isFinite(Number(value)))
                return Number(value);
            return reject('a finite number');
        }
        case 'boolean': {
            if (typeof value === 'boolean')
                return value;
            if (value === 'true' || value === 1)
                return true;
            if (value === 'false' || value === 0)
                return false;
            return reject('true or false');
        }
        default:
            return value;
    }
}
function selectedOutputs(outputs, contract) {
    const selected = {};
    for (const [name, binding] of Object.entries(contract.outputs)) {
        if (!Object.hasOwn(outputs, binding.node))
            throw new ModelHubError('INVOCATION_FAILED', `ComfyUI produced no declared output "${name}" at node "${binding.node}"`);
        selected[binding.node] = outputs[binding.node];
    }
    return selected;
}
/**
 * Queue one graph, wait for it to finish, and return what it produced.
 *
 * Split out from the capability paths because queueing, polling and error
 * reporting are identical for an image graph and a 3D one — only the download
 * differs. Sharing this is what keeps the 3D path from being a second, subtly
 * different implementation of ComfyUI's job protocol.
 *
 * @param invocation - the resolved request.
 * @param graph - the graph to queue.
 * @param base - the endpoint with a trailing slash.
 * @param settings - resolved settings.
 * @param signal - the scoped cancellation.
 * @param reason - builds the timeout/abort error for this call.
 * @returns the prompt id and the node outputs.
 * @throws ModelHubError when the engine refuses, fails, or never finishes.
 */
async function runGraph(invocation, graph, base, settings, signal, reason) {
    const model = invocation.model;
    const queueUrl = new URL('prompt', base).toString();
    invocation.log.debug('comfyui: queueing graph', { queueUrl, nodes: Object.keys(graph).length });
    let queued;
    try {
        queued = await fetch(queueUrl, {
            method: 'POST',
            signal,
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ client_id: settings.clientId, prompt: graph }),
        });
    }
    catch (error) {
        if (signal.aborted)
            throw reason();
        throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" could not reach ${queueUrl}: ${error instanceof Error ? error.message : String(error)}`, { modelId: model.id, url: queueUrl });
    }
    if (!queued.ok) {
        const detail = await queued.text().catch(() => '');
        throw new ModelHubError('INVOCATION_FAILED', `ComfyUI rejected the queued graph (${queued.status} ${queued.statusText}): ${detail.slice(0, MAX_ERROR_BODY_CHARS)}`, { modelId: model.id, status: queued.status });
    }
    const queueBody = (await queued.json());
    const promptId = queueBody['prompt_id'];
    if (typeof promptId !== 'string' || promptId.length === 0) {
        const nodeErrors = queueBody['node_errors'];
        throw new ModelHubError('INVOCATION_FAILED', `ComfyUI accepted the request but returned no prompt_id: ${JSON.stringify(nodeErrors ?? queueBody).slice(0, MAX_ERROR_BODY_CHARS)}`, { modelId: model.id });
    }
    const historyUrl = new URL(`history/${promptId}`, base).toString();
    let entry;
    while (entry === undefined) {
        if (signal.aborted)
            throw reason();
        await new Promise((resolveSleep) => setTimeout(resolveSleep, settings.pollIntervalMs));
        if (signal.aborted)
            throw reason();
        let polled;
        try {
            polled = await fetch(historyUrl, { signal, headers: { accept: 'application/json' } });
        }
        catch (error) {
            if (signal.aborted)
                throw reason();
            throw new ModelHubError('INVOCATION_FAILED', `could not poll ComfyUI history at ${historyUrl}: ${error instanceof Error ? error.message : String(error)}`, { modelId: model.id, url: historyUrl });
        }
        if (!polled.ok) {
            throw new ModelHubError('INVOCATION_FAILED', `ComfyUI history at ${historyUrl} answered ${polled.status} ${polled.statusText}`, { modelId: model.id, status: polled.status });
        }
        const history = (await polled.json());
        const candidate = history[promptId];
        if (candidate !== undefined && typeof candidate === 'object' && candidate !== null) {
            entry = candidate;
        }
    }
    // ComfyUI reports per-node execution errors alongside the outputs; a graph can
    // "complete" with nothing produced, and saying so plainly is far more useful
    // than reporting zero outputs.
    const status = entry['status'];
    const statusStr = status !== null && typeof status === 'object'
        ? status['status_str']
        : undefined;
    if (statusStr === 'error') {
        const messages = status !== null && typeof status === 'object'
            ? status['messages']
            : undefined;
        throw new ModelHubError('INVOCATION_FAILED', `ComfyUI failed to execute the graph: ${JSON.stringify(messages ?? status).slice(0, MAX_ERROR_BODY_CHARS)}`, { modelId: model.id, promptId });
    }
    const outputs = entry['outputs'];
    return { promptId, outputs: outputs !== null && typeof outputs === 'object' ? outputs : {} };
}
/**
 * Every file reference a ComfyUI `outputs` payload contains.
 *
 * ComfyUI keys an output node's files by the node's own choice of name: `images`
 * for the image savers, `3d` for `SaveGLB`, `audio` for the audio savers, and so
 * on. So discovery walks every value of every node's output object and collects
 * anything that names a file, rather than looking under one key and hoping. Items
 * are either `{ filename, subfolder, type }` objects or bare path strings — the
 * newer 3D nodes emit the latter — and both are accepted.
 *
 * @param outputs - the `outputs` object from `/history/{id}`.
 * @returns every file reference found, in engine order.
 */
export function readComfyFileRefs(outputs) {
    const refs = [];
    for (const nodeOutput of Object.values(outputs)) {
        if (nodeOutput === null || typeof nodeOutput !== 'object')
            continue;
        for (const value of Object.values(nodeOutput)) {
            if (!Array.isArray(value))
                continue;
            for (const item of value) {
                const ref = fileRefOf(item);
                if (ref !== undefined)
                    refs.push(ref);
            }
        }
    }
    return refs;
}
/**
 * Read one file reference out of an output item.
 * @param item - the item.
 * @returns the reference, or `undefined` when the item names no file.
 */
function fileRefOf(item) {
    if (typeof item === 'string') {
        const trimmed = item.trim();
        if (!looksLikeFileReference(trimmed))
            return undefined;
        // A string form may be a bare filename or a `subfolder/name` path, and may
        // carry a `[output]`-style type suffix from the newer UI payloads.
        const [path, typeSuffix] = splitTypeSuffix(trimmed);
        const slash = path.lastIndexOf('/');
        const filename = slash === -1 ? path : path.slice(slash + 1);
        const subfolder = slash === -1 ? '' : path.slice(0, slash);
        return {
            filename,
            subfolder: subfolder === 'temp' ? '' : subfolder,
            type: typeSuffix ?? (subfolder === 'temp' ? 'temp' : DEFAULT_VIEW_TYPE),
        };
    }
    if (item !== null && typeof item === 'object' && !Array.isArray(item)) {
        const record = item;
        const filename = record['filename'];
        if (typeof filename !== 'string' || filename.length === 0)
            return undefined;
        return {
            filename,
            subfolder: typeof record['subfolder'] === 'string' ? record['subfolder'] : '',
            type: typeof record['type'] === 'string' ? record['type'] : DEFAULT_VIEW_TYPE,
        };
    }
    return undefined;
}
/**
 * Split a `name.glb [output]` reference into its path and view type.
 * @param value - the reference text.
 * @returns the path and the type, when one was stated.
 */
function splitTypeSuffix(value) {
    const match = /^(.*?)\s*\[([a-z_]+)\]\s*$/i.exec(value);
    if (match === null)
        return [value, undefined];
    return [match[1], match[2]];
}
/**
 * Whether a string plausibly names a file rather than a placeholder token.
 *
 * Both branches of the extension test are anchored at the end, which matters: a
 * greedy `.*` with an unanchored `\.(png|…)` would accept `"a.png [output]"` (and
 * everything else with a dot and a known suffix anywhere), rejecting real
 * references instead of accepting them.
 *
 * @param value - the candidate reference text.
 * @returns true when it ends in something that looks like a file extension.
 */
export function looksLikeFileReference(value) {
    const trimmed = value.trim();
    if (trimmed.length === 0)
        return false;
    return /\.[a-z0-9]{2,5}$/i.test(trimmed) || /\.[a-z0-9]{2,5}\s*\[[a-z_]+\]\s*$/i.test(trimmed);
}
/**
 * Download one produced file from ComfyUI's `/view` route.
 * @param ref - the file reference.
 * @param base - the endpoint with a trailing slash.
 * @param signal - the scoped cancellation.
 * @param fallbackType - the `/view` type to use when the reference states none.
 * @param reason - builds the timeout/abort error for this call.
 * @returns the bytes.
 * @throws ModelHubError when the download fails.
 */
async function downloadFile(ref, base, signal, fallbackType, reason) {
    const view = new URL('view', base);
    view.searchParams.set('filename', ref.filename);
    view.searchParams.set('subfolder', ref.subfolder);
    view.searchParams.set('type', ref.type || fallbackType);
    let downloaded;
    try {
        downloaded = await fetch(view.toString(), { signal });
    }
    catch (error) {
        if (signal.aborted)
            throw reason();
        throw new ModelHubError('INVOCATION_FAILED', `could not download "${ref.filename}" from ${view.toString()}: ${error instanceof Error ? error.message : String(error)}`, { filename: ref.filename });
    }
    if (!downloaded.ok) {
        throw new ModelHubError('INVOCATION_FAILED', `ComfyUI view endpoint answered ${downloaded.status} for "${ref.filename}"`, { filename: ref.filename, status: downloaded.status });
    }
    return new Uint8Array(await downloaded.arrayBuffer());
}
/**
 * Store every image a graph produced, in the order ComfyUI reported them.
 *
 * @param invocation - the resolved request.
 * @param outputs - the `outputs` object.
 * @param base - the endpoint with a trailing slash.
 * @param settings - resolved settings.
 * @param promptId - the queued graph's id.
 * @param signal - the scoped cancellation.
 * @param reason - builds the timeout/abort error for this call.
 * @param started - when the invocation began, for the empty-output message.
 * @returns the invocation result.
 * @throws ModelHubError when nothing image-like was produced.
 */
async function collectImages(invocation, outputs, base, settings, promptId, signal, reason, started) {
    const model = invocation.model;
    const images = readComfyFileRefs(outputs).filter((ref) => !isThreeDFilename(ref.filename));
    if (images.length === 0) {
        throw new ModelHubError('INVOCATION_FAILED', `ComfyUI executed the graph (${Date.now() - started} ms) but produced no images. ` +
            'Check that the template ends in a SaveImage node.', { modelId: model.id, promptId });
    }
    const artifacts = [];
    for (const image of images) {
        const bytes = await downloadFile(image, base, signal, image.type, reason);
        const dimensions = readPngDimensions(bytes);
        artifacts.push(await invocation.artifacts.put({
            type: 'image',
            bytes,
            mimeType: 'image/png',
            label: baseName(image.filename),
            producerModelId: model.id,
            extension: '.png',
            metadata: {
                prompt: settings.prompt,
                ...(settings.negativePrompt === undefined ? {} : { negativePrompt: settings.negativePrompt }),
                ...(dimensions === undefined ? {} : { width: dimensions.width, height: dimensions.height }),
                seed: settings.seed ?? null,
                promptId,
                source: 'comfyui',
                comfyFilename: image.filename,
                ...(image.subfolder.length === 0 ? {} : { comfySubfolder: image.subfolder }),
            },
        }));
    }
    invocation.log.info('comfyui: generated image(s)', { count: artifacts.length, promptId });
    const first = artifacts[0];
    return {
        outputs: artifacts,
        value: {
            count: artifacts.length,
            format: 'png',
            prompt: settings.prompt,
            promptId,
            ...(first?.metadata['width'] === undefined ? {} : { width: first.metadata['width'] }),
            ...(first?.metadata['height'] === undefined ? {} : { height: first.metadata['height'] }),
        },
    };
}
/**
 * Store the 3D asset a graph produced, plus any companion render it also wrote.
 *
 * The mesh is identified by **sniffing bytes**, not by trusting an extension:
 * ComfyUI's own 3D nodes emit GLB, a custom node may emit OBJ or PLY, and a
 * server that answers `200` with an HTML error page is a real failure mode. The
 * artifact therefore carries the container the bytes actually are, and the
 * declared extension is only used to choose what to try first.
 *
 * Companion images (a turntable render, a preview) are stored after the mesh, so
 * `outputs[0]` is always the thing the caller asked for.
 *
 * @param invocation - the resolved request.
 * @param outputs - the `outputs` object.
 * @param base - the endpoint with a trailing slash.
 * @param settings - resolved settings.
 * @param promptId - the queued graph's id.
 * @param signal - the scoped cancellation.
 * @param reason - builds the timeout/abort error for this call.
 * @param started - when the invocation began, for the empty-output message.
 * @returns the invocation result.
 * @throws ModelHubError when nothing mesh-like was produced.
 */
async function collectThreeD(invocation, outputs, base, settings, promptId, signal, reason, started) {
    const model = invocation.model;
    const refs = readComfyFileRefs(outputs);
    const meshRefs = refs.filter((ref) => isThreeDFilename(ref.filename));
    if (meshRefs.length === 0) {
        const other = refs.length;
        throw new ModelHubError('INVOCATION_FAILED', `ComfyUI executed the 3D workflow (${Date.now() - started} ms) but produced no 3D file. ` +
            (other === 0
                ? 'The graph completed without writing anything: check that it ends in a SaveGLB or Save3DAdvanced node.'
                : `It wrote ${other} file(s), none of them a mesh (${refs.map((ref) => ref.filename).slice(0, 4).join(', ')}). ` +
                    'A workflow that only *previews* 3D output writes to ComfyUI\'s temp directory, which is not a deliverable.'), { modelId: model.id, promptId, files: refs.map((ref) => ref.filename).slice(0, 8) });
    }
    const artifacts = [];
    let format;
    let warning;
    for (const [index, ref] of meshRefs.entries()) {
        const bytes = await downloadFile(ref, base, signal, ref.type || settings.outputViewType, reason);
        const claimed = extensionOf(ref.filename);
        const sniff = sniffThreeDFormat(bytes.subarray(0, SNIFF_PREFIX_BYTES), claimed);
        const resolved = sniff.format ?? threeDFormatOf(claimed);
        const info = resolved === undefined ? undefined : threeDFormatInfo(resolved);
        const measured = measureThreeD(bytes, resolved);
        // Only the first mesh's sniff is reported in the structured value, because
        // that is the one the caller asked for; the rest are companions.
        if (index === 0) {
            format = resolved;
            warning = sniff.warning;
        }
        artifacts.push(await invocation.artifacts.put({
            type: 'model_3d',
            bytes,
            ...(info === undefined ? {} : { mimeType: info.mimeType }),
            label: baseName(ref.filename),
            producerModelId: model.id,
            extension: info?.extension ?? '.bin',
            metadata: {
                ...(resolved === undefined ? {} : { format: resolved }),
                ...(measured.vertexCount === undefined ? {} : { vertexCount: measured.vertexCount }),
                ...(measured.triangleCount === undefined ? {} : { triangleCount: measured.triangleCount }),
                ...(warning === undefined ? {} : { warning }),
                promptId,
                source: 'comfyui',
                comfyFilename: ref.filename,
                ...(ref.subfolder.length === 0 ? {} : { comfySubfolder: ref.subfolder }),
                ...(settings.prompt.trim().length === 0 ? {} : { prompt: settings.prompt }),
            },
        }));
    }
    // Companions: a preview render the workflow also wrote, stored after the mesh.
    for (const ref of refs) {
        if (isThreeDFilename(ref.filename))
            continue;
        if (!/\.(png|jpe?g|webp)$/i.test(ref.filename))
            continue;
        const bytes = await downloadFile(ref, base, signal, ref.type, reason);
        const dimensions = readPngDimensions(bytes);
        artifacts.push(await invocation.artifacts.put({
            type: 'image',
            bytes,
            mimeType: /\.jpe?g$/i.test(ref.filename) ? 'image/jpeg' : 'image/png',
            label: baseName(ref.filename),
            producerModelId: model.id,
            extension: /\.jpe?g$/i.test(ref.filename) ? '.jpg' : '.png',
            metadata: {
                ...(dimensions === undefined ? {} : { width: dimensions.width, height: dimensions.height }),
                promptId,
                source: 'comfyui',
                role: 'companion',
                comfyFilename: ref.filename,
            },
        }));
    }
    const primary = artifacts[0];
    invocation.log.info('comfyui: generated 3D asset(s)', {
        count: meshRefs.length,
        format: format ?? 'unknown',
        promptId,
    });
    return {
        outputs: artifacts,
        value: {
            count: meshRefs.length,
            format: format ?? 'unknown',
            ...(primary?.mimeType === undefined ? {} : { mimeType: primary.mimeType }),
            ...(primary?.byteLength === undefined ? {} : { byteLength: primary.byteLength }),
            promptId,
            ...(primary?.metadata['vertexCount'] === undefined
                ? {}
                : { vertexCount: primary.metadata['vertexCount'] }),
            ...(primary?.metadata['triangleCount'] === undefined
                ? {}
                : { triangleCount: primary.metadata['triangleCount'] }),
            ...(warning === undefined ? {} : { warning }),
        },
    };
}
/**
 * Upload the request's input image and return the filename ComfyUI stored it as.
 *
 * The caller's artifact is a file in the hub's own store, which the engine cannot
 * see; ComfyUI's `LoadImage` reads from *its* input directory. So the image is
 * posted to `/upload/image`, and the name that comes back is what the graph must
 * carry. `overwrite` is left at the server default so that two concurrent
 * invocations with the same artifact cannot race on one filename — the alternative,
 * always overwriting, would let one request's mask be another's image.
 *
 * @param invocation - the resolved request.
 * @param base - the endpoint with a trailing slash.
 * @param signal - the scoped cancellation.
 * @param reason - builds the timeout/abort error for this call.
 * @returns the stored filename.
 * @throws ModelHubError when the artifact cannot be read or the upload fails.
 */
async function uploadInputImage(invocation, base, signal, reason) {
    const image = invocation.inputs.find((artifact) => artifact.type === 'image');
    if (image === undefined) {
        throw new ModelHubError('INVOCATION_FAILED', `"${invocation.capability}" needs an input artifact of type \`image\``, { modelId: invocation.model.id });
    }
    const { path } = await invocation.artifacts.resolvePath(image.id);
    const bytes = await readFile(path);
    const form = new FormData();
    form.append('image', new Blob([new Uint8Array(bytes)], { type: image.mimeType ?? 'image/png' }), uploadNameFor(image));
    form.append('type', 'input');
    const uploadUrl = new URL(UPLOAD_PATH.replace(/^\//, ''), base).toString();
    let uploaded;
    try {
        uploaded = await fetch(uploadUrl, { method: 'POST', body: form, signal });
    }
    catch (error) {
        if (signal.aborted)
            throw reason();
        throw new ModelHubError('INVOCATION_FAILED', `could not upload the input image to ${uploadUrl}: ${error instanceof Error ? error.message : String(error)}`, { modelId: invocation.model.id, url: uploadUrl });
    }
    if (!uploaded.ok) {
        const detail = await uploaded.text().catch(() => '');
        throw new ModelHubError('INVOCATION_FAILED', `ComfyUI refused the input image upload (${uploaded.status} ${uploaded.statusText}): ${detail.slice(0, MAX_ERROR_BODY_CHARS)}`, { modelId: invocation.model.id, status: uploaded.status });
    }
    const body = (await uploaded.json().catch(() => undefined));
    const record = body !== null && typeof body === 'object' ? body : {};
    const name = record['name'];
    if (typeof name !== 'string' || name.length === 0) {
        // Older ComfyUI builds answer with just the filename in a JSON string.
        if (typeof body === 'string' && body.length > 0)
            return body;
        throw new ModelHubError('INVOCATION_FAILED', `ComfyUI accepted the input image but returned no filename, so the workflow cannot reference it: ` +
            `${JSON.stringify(body).slice(0, MAX_ERROR_BODY_CHARS)}`, { modelId: invocation.model.id });
    }
    const subfolder = record['subfolder'];
    return typeof subfolder === 'string' && subfolder.length > 0 ? `${subfolder}/${name}` : name;
}
/**
 * The upload filename to use for an input artifact.
 *
 * The artifact's own id is already a validated, filesystem-safe token, so it
 * makes a stable and collision-free name. The extension is taken from the
 * artifact's own URI — ComfyUI's loader infers the image type from it, so getting
 * it wrong turns a PNG into an unreadable file — falling back to the MIME type
 * and then to `.png`.
 *
 * @param artifact - the input image artifact.
 * @returns the multipart filename.
 */
function uploadNameFor(artifact) {
    const fromUri = /(\.[a-z0-9]{2,5})$/i.exec(artifact.uri)?.[1];
    if (fromUri !== undefined)
        return `${artifact.id}${fromUri.toLowerCase()}`;
    const extension = artifact.mimeType === 'image/jpeg' ? '.jpg' : artifact.mimeType === 'image/webp' ? '.webp' : '.png';
    return `${artifact.id}${extension}`;
}
/** The basename of a path, without its directory. */
function baseName(path) {
    const slash = path.lastIndexOf('/');
    return slash === -1 ? path : path.slice(slash + 1);
}
/** The lowercase extension of a filename, including the dot. */
function extensionOf(filename) {
    const match = /(\.[a-z0-9]+)$/i.exec(filename);
    if (match === null)
        return '';
    return (match[1] ?? '').toLowerCase();
}
/**
 * Read pixel dimensions out of a PNG's IHDR chunk.
 * @param bytes - the encoded image.
 * @returns the dimensions, or `undefined` when it is not a PNG.
 */
export function readPngDimensions(bytes) {
    const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    if (bytes.length < 24)
        return undefined;
    for (let index = 0; index < PNG_SIGNATURE.length; index += 1) {
        if (bytes[index] !== PNG_SIGNATURE[index])
            return undefined;
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const width = view.getUint32(16, false);
    const height = view.getUint32(20, false);
    if (width === 0 || height === 0)
        return undefined;
    return { width, height };
}
