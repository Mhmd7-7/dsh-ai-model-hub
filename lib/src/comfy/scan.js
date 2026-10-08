/**
 * ComfyUI workflow scanning: from a saved graph to a runnable public contract.
 *
 * A workflow an operator saved in ComfyUI is a complete executable unit, but it
 * carries no statement of what a caller may set or what it produces. This module
 * derives that statement *structurally* from the graph — which node holds the
 * prompt, which one carries the image, which one is the deliverable — and
 * classifies how ready the document is to run.
 *
 * That is the difference between the two halves of the ComfyUI integration:
 *
 * - a **configured** workflow (see `discovery/comfyui.ts`) states its contract by
 *   hand, and is the operator's deliberate decision about what is routable;
 * - a **scanned** workflow (this module) has its contract inferred, so a workflow
 *   saved in the ComfyUI editor becomes usable without hand-writing JSON.
 *
 * The inference is deliberately conservative and always answers with a *public*
 * surface: a prompt, an optional image, a seed, dimensions. It never exposes a
 * checkpoint name, a LoRA, a VAE, a node id, or any other graph internals — those
 * stay inside the workflow, which is the whole point of the boundary.
 *
 * ## Editor format versus API format
 *
 * Two serializations exist and they are not equally runnable:
 *
 * - **API format** — `{ "3": { class_type, inputs } }` — is what ComfyUI queues.
 *   A workflow in this shape is `ready`.
 * - **UI format** — `{ nodes: [...], links: [...] }` — is what the editor saves.
 *   It is converted to an API graph during the scan, but it is reported as
 *   `needs_conversion` rather than `ready`, because the conversion is a
 *   reconstruction and the operator should know which one they are looking at.
 *
 * A document that cannot be read at all is `invalid` with the parser's own reason,
 * never silently dropped: a workflow the hub cannot run is still a workflow the
 * operator wants to see, precisely so they can fix it.
 *
 * @module dsh-ai-model-hub/comfy/scan
 */
import { slugifyModelId, stableDigest } from "../discovery/http.js";
import { ComfyWorkflowError, linkSource, parseComfyWorkflow } from "../discovery/comfyui-workflow.js";
import { readContract, validateComfyGraph } from "./workflow.js";
/** Node classes that write an image a caller can keep. */
const IMAGE_WRITER_PATTERN = /^save.*image|^saveimage|image.*save|^export.*image/i;
/** Node classes that write a 3D file a caller can keep. */
const THREE_D_WRITER_PATTERN = /save.*(glb|gltf|obj|stl|ply|fbx|usdz|3d|mesh|splat)|export.*(3d|mesh|glb)|mesh.*to.*file|(glb|obj|stl).*export/i;
/** Node classes that only preview, which writes to ComfyUI's temp directory. */
const PREVIEW_PATTERN = /preview/i;
/** Node classes that load an image from disk. */
const IMAGE_LOADER_PATTERN = /load.*image|imageload|image.*load|^loadimage/i;
/** Input field names a latent/size node declares. */
const WIDTH_FIELDS = ['width'];
const HEIGHT_FIELDS = ['height'];
/**
 * Whether a value is a plain JSON object.
 * @param value - candidate value.
 * @returns true for a non-null, non-array object.
 */
function isObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}
/**
 * Every node of an API graph, in id order.
 * @param graph - the API-format graph.
 * @returns the nodes.
 */
function graphNodes(graph) {
    const nodes = [];
    for (const [id, node] of Object.entries(graph)) {
        if (!isObject(node))
            continue;
        const classType = node['class_type'];
        const inputs = node['inputs'];
        if (typeof classType !== 'string' || !isObject(inputs))
            continue;
        nodes.push({ id, classType, inputs });
    }
    return nodes;
}
/**
 * The first field of a node that carries a literal (non-link) value of a kind.
 *
 * A link is an array or a bare node-id string, so only a scalar counts as a
 * settable widget. That distinction is what keeps "this node has a width" from
 * being confused with "this node is wired to something that has a width".
 *
 * @param node - the node.
 * @param fields - candidate field names.
 * @returns the field name, or `undefined`.
 */
function literalField(node, fields) {
    for (const field of fields) {
        const value = node.inputs[field];
        if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean')
            return field;
    }
    return undefined;
}
/**
 * Whether a node has a field holding a non-empty string.
 * @param node - the node.
 * @param field - the field name.
 * @returns true when the field holds textual content.
 */
function hasText(node, field) {
    return typeof node.inputs[field] === 'string';
}
/**
 * The node a link points at, when it points at one in this graph.
 * @param nodes - every node, by id.
 * @param link - the input value.
 * @returns the node, or `undefined` when it is a literal or dangling.
 */
function linkedNode(nodes, link) {
    const source = linkSource(link);
    return source === undefined ? undefined : nodes.get(source);
}
/**
 * Locate the nodes a public contract can bind to.
 *
 * Every target is found from the graph's own wiring and declared field names, so a
 * workflow keeps working when the editor renumbers node ids. Nothing here names a
 * model: `class_type` is a stable part of ComfyUI's node contract, and even that
 * is only used to break ties the field names cannot.
 *
 * @param graph - the API-format graph.
 * @returns the targets that were found.
 */
function findTargets(graph) {
    const nodes = graphNodes(graph);
    const byId = new Map(nodes.map((node) => [node.id, node]));
    // The sampler is the node that actually generates: it consumes a model and a
    // conditioning pair and carries a seed. Requiring the seed is what separates it
    // from every other node in a diffusion graph.
    const sampler = nodes.find((node) => literalField(node, ['seed', 'noise_seed']) !== undefined && literalField(node, ['steps']) !== undefined) ??
        nodes.find((node) => /sampler/i.test(node.classType) && literalField(node, ['seed', 'noise_seed']) !== undefined);
    const samplerSeedField = sampler === undefined ? undefined : literalField(sampler, ['seed', 'noise_seed']);
    // The prompt is whatever the sampler's `positive` input reads, when that node
    // actually accepts text. Falling back to the first text-encoding node covers
    // graphs whose conditioning is arranged differently.
    const positive = sampler === undefined ? undefined : linkedNode(byId, sampler.inputs['positive']);
    const prompt = positive !== undefined && hasText(positive, 'text')
        ? positive
        : nodes.find((node) => /textencode|text_encode|prompt|cliptext/i.test(node.classType) && hasText(node, 'text'));
    const negativeLink = sampler === undefined ? undefined : linkedNode(byId, sampler.inputs['negative']);
    const negative = negativeLink !== undefined && hasText(negativeLink, 'text') ? negativeLink : undefined;
    const size = nodes.find((node) => literalField(node, WIDTH_FIELDS) !== undefined && literalField(node, HEIGHT_FIELDS) !== undefined);
    const imageLoader = nodes.find((node) => hasText(node, 'image') && (IMAGE_LOADER_PATTERN.test(node.classType) || /load|input|read|open/i.test(node.classType)));
    // The deliverable. A saver is preferred over a preview, and a 3D writer over an
    // image writer, because a graph that does both is a 3D graph with a render.
    const writers = nodes.filter((node) => THREE_D_WRITER_PATTERN.test(node.classType) || IMAGE_WRITER_PATTERN.test(node.classType));
    const threeDWriter = writers.find((node) => THREE_D_WRITER_PATTERN.test(node.classType));
    const imageWriter = writers.find((node) => !THREE_D_WRITER_PATTERN.test(node.classType));
    const writer = threeDWriter ?? imageWriter;
    const outputKind = writer === undefined ? undefined : threeDWriter !== undefined ? 'model_3d' : 'image';
    return {
        ...(sampler === undefined ? {} : { sampler }),
        ...(samplerSeedField === undefined ? {} : { samplerSeedField }),
        ...(prompt === undefined ? {} : { prompt }),
        ...(negative === undefined ? {} : { negative }),
        ...(size === undefined ? {} : { size }),
        ...(imageLoader === undefined ? {} : { imageLoader }),
        ...(writer === undefined ? {} : { writer }),
        ...(outputKind === undefined ? {} : { outputKind }),
    };
}
/**
 * The capabilities a scanned graph's targets prove.
 *
 * The rule mirrors the rest of the integration: a capability is claimed only when
 * the graph contains the machinery that serves it *and* a source that reaches it.
 * An image writer plus a prompt is `text_to_image`; add an image loader and it is
 * also `image_to_image`; a 3D writer follows the same shape.
 *
 * @param targets - the nodes that were found.
 * @returns the capabilities, in vocabulary order.
 */
function capabilitiesForTargets(targets) {
    const capabilities = [];
    const hasPrompt = targets.prompt !== undefined;
    const hasImage = targets.imageLoader !== undefined;
    if (targets.outputKind === 'image') {
        if (hasPrompt)
            capabilities.push('text_to_image');
        if (hasImage)
            capabilities.push('image_to_image');
    }
    else if (targets.outputKind === 'model_3d') {
        if (hasImage)
            capabilities.push('image_to_3d');
        if (hasPrompt)
            capabilities.push('text_to_3d');
    }
    return capabilities;
}
/**
 * Build the public contract a scanned workflow exposes.
 *
 * Only the parameters a caller genuinely needs are bound: the prompt, the
 * negative prompt when the graph has one, the input image when a capability
 * consumes one, and the seed, steps, CFG, width and height when the graph has a
 * node to receive them. Everything else — loaders, LoRAs, VAEs, ControlNets,
 * custom nodes — is left exactly as the workflow's author saved it.
 *
 * @param targets - the nodes that were found.
 * @param capabilities - the capabilities the graph proves.
 * @returns the inferred contract.
 */
function contractForTargets(targets, capabilities) {
    const writer = targets.writer;
    const outputKind = targets.outputKind;
    if (writer === undefined || outputKind === undefined)
        return undefined;
    const bindings = {};
    if (targets.prompt !== undefined)
        bindings['prompt'] = { node: targets.prompt.id, input: 'text' };
    if (targets.negative !== undefined)
        bindings['negative_prompt'] = { node: targets.negative.id, input: 'text' };
    const wantsImage = capabilities.includes('image_to_image') || capabilities.includes('image_to_3d');
    if (wantsImage && targets.imageLoader !== undefined) {
        bindings['image'] = { node: targets.imageLoader.id, input: 'image' };
    }
    if (targets.sampler !== undefined) {
        if (targets.samplerSeedField !== undefined)
            bindings['seed'] = { node: targets.sampler.id, input: targets.samplerSeedField };
        if (literalField(targets.sampler, ['steps']) !== undefined)
            bindings['steps'] = { node: targets.sampler.id, input: 'steps' };
        if (literalField(targets.sampler, ['cfg']) !== undefined)
            bindings['cfg'] = { node: targets.sampler.id, input: 'cfg' };
    }
    if (targets.size !== undefined) {
        const width = literalField(targets.size, WIDTH_FIELDS);
        const height = literalField(targets.size, HEIGHT_FIELDS);
        if (width !== undefined)
            bindings['width'] = { node: targets.size.id, input: width };
        if (height !== undefined)
            bindings['height'] = { node: targets.size.id, input: height };
    }
    const outputs = {};
    outputs[outputKind] = { node: writer.id, type: outputKind };
    const config = { bindings, outputs };
    return readContract(config, capabilities);
}
/**
 * The model `type` a scanned workflow's capabilities imply.
 *
 * `type` is operator-facing grouping; routing filters on `capabilities`. Kept in
 * step with the type a configured workflow gets, so the two halves of ComfyUI
 * discovery describe the same workflow identically.
 *
 * @param capabilities - the proven capabilities.
 * @returns the model type.
 */
export function workflowModelTypeFor(capabilities) {
    if (capabilities.includes('text_to_3d') || capabilities.includes('image_to_3d'))
        return 'three_d_generation';
    if (capabilities.includes('image_to_image'))
        return 'image_editing';
    return 'image_generation';
}
/**
 * Whether a document is a ComfyUI editor export rather than an API graph.
 *
 * Used to describe a document the parser refused: "this is an editor export whose
 * node packs are not installed here" is actionable, while "unreadable" is not.
 *
 * @param raw - the parsed document.
 * @returns true when it has the editor's node array.
 */
export function looksLikeEditorWorkflow(raw) {
    if (!isObject(raw))
        return false;
    if (Array.isArray(raw['nodes']))
        return true;
    const envelope = raw['prompt'];
    return isObject(envelope) && Array.isArray(envelope['nodes']);
}
/**
 * Scan one workflow document into either a runnable provider or a reported gap.
 *
 * @param input - the document, its name, and whatever the engine revealed.
 * @returns the scan result; never throws.
 */
export function scanWorkflowDocument(input) {
    const { raw, name, io, classes } = input;
    const editorFormat = looksLikeEditorWorkflow(raw);
    let parsed;
    try {
        parsed = parseComfyWorkflow(raw, name, io ?? {});
    }
    catch (error) {
        const reason = error instanceof ComfyWorkflowError ? error.message : String(error);
        // A document the parser could not read is reported, never dropped. An editor
        // export the hub cannot reconstruct is a *conversion* problem — the workflow
        // is valid, this install just cannot interpret it — and is labelled so.
        return {
            format: editorFormat ? 'ui' : 'api',
            readiness: editorFormat ? 'needs_conversion' : 'invalid',
            runnable: false,
            detail: editorFormat
                ? `editor (UI) format that could not be converted to an API graph: ${reason}`
                : `not a readable API workflow: ${reason}`,
            capabilities: [],
            inputs: [],
            outputs: [],
        };
    }
    // The executable graph is rebuilt from the parsed nodes rather than taken from
    // the raw document, so any non-node key an export carries (a name, an editor
    // version, a viewport) cannot reach the graph ComfyUI is asked to queue.
    const graph = {};
    for (const [id, node] of Object.entries(parsed.nodes)) {
        graph[id] = { class_type: node.classType, inputs: { ...node.inputs } };
    }
    const targets = findTargets(graph);
    const capabilities = capabilitiesForTargets(targets);
    if (capabilities.length === 0) {
        return {
            format: parsed.format,
            readiness: 'invalid',
            runnable: false,
            detail: targets.writer === undefined
                ? 'no saving node: the graph writes nothing a caller could keep (a preview writes to ComfyUI\'s temp directory)'
                : 'no prompt or image input reaches the saving node, so no capability is proven',
            capabilities: [],
            inputs: [],
            outputs: [],
        };
    }
    let contract;
    try {
        const inferred = contractForTargets(targets, capabilities);
        if (inferred === undefined) {
            return {
                format: parsed.format,
                readiness: 'invalid',
                runnable: false,
                detail: 'the graph has no output node the hub could collect',
                capabilities,
                inputs: [],
                outputs: [],
            };
        }
        validateComfyGraph(graph, inferred, classes);
        contract = inferred;
    }
    catch (error) {
        return {
            format: parsed.format,
            readiness: 'invalid',
            runnable: false,
            detail: error instanceof Error ? error.message : String(error),
            capabilities,
            inputs: [],
            outputs: [],
        };
    }
    const inputs = Object.keys(contract.bindings);
    const outputs = Object.keys(contract.outputs);
    return {
        format: parsed.format,
        readiness: parsed.format === 'ui' ? 'needs_conversion' : 'ready',
        runnable: true,
        detail: parsed.format === 'ui'
            ? 'editor (UI) format — converted to an API graph for execution; export with "Save (API Format)" to make this exact'
            : 'API format — ready to run as saved',
        capabilities,
        inputs,
        outputs,
        contract,
        graph,
    };
}
/**
 * A stable catalog id for a scanned workflow.
 *
 * The id is derived from the source location rather than a counter, so the same
 * workflow found again on a later scan resolves to the same entry and a refresh
 * *updates* it instead of adding a second copy. The digest is the fallback for a
 * name with nothing Latin in it, matching the convention the discoverers use.
 *
 * @param name - the workflow's display name.
 * @param source - where it was found, which is what actually identifies it.
 * @returns a lowercase kebab-case id.
 */
export function scannedWorkflowId(name, source) {
    return slugifyModelId(name, 'comfy-wf', `wf-${stableDigest(source)}`);
}
/**
 * A stable id for one Ollama model.
 *
 * Ollama's own tag is the identity, namespaced by the host so two Ollama
 * installations on one machine cannot collide.
 *
 * @param hostId - the host the model was read from.
 * @param name - the engine's own model tag.
 * @returns a lowercase kebab-case id.
 */
export function scannedOllamaModelId(hostId, name) {
    return slugifyModelId(name, `ollama-${slugifyModelId(hostId, 'host')}`, `model-${stableDigest(`${hostId}/${name}`)}`);
}
