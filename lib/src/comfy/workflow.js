// ComfyUI's graph is trusted operator configuration. Only these declarations are public.
import { readFile } from 'node:fs/promises';
import { CAPABILITY_IO, isCapability, isIoType } from "../catalog/capabilities.js";
import { resolveAdapterPath } from "../adapters/paths.js";
import { ModelHubError } from "../errors.js";
/** The conventional names, kept for documentation and for hand-written catalogs. */
export const PUBLIC_INPUTS = new Set(['prompt', 'negativePrompt', 'negative_prompt', 'image', 'seed', 'steps', 'cfg', 'width', 'height', 'strength', 'denoise', 'sampler', 'scheduler']);
/**
 * A public parameter name must be a plain identifier.
 *
 * This is the security boundary, and it is deliberately a shape rather than a
 * list. Naming a parameter is not the same as reaching into the graph: a name may
 * not contain a separator, a space, or punctuation, so it cannot address a node,
 * a class type, or a filesystem path. The list above is a convention, not the
 * rule — a scanned workflow needs names like `switch_to_trellis2`, and refusing
 * them would mean refusing the analysis this module exists to perform.
 */
const PARAMETER_NAME = /^[A-Za-z][A-Za-z0-9_]{0,63}$/;
/** Input fields that are graph or engine configuration rather than a caller's value. */
const forbidden = /(?:ckpt|checkpoint|lora|vae|unet|clip|controlnet|model|path|file|command|script|python)/i;
const INPUT_KINDS = ['text', 'image', 'number', 'boolean'];
function fail(message) { throw new ModelHubError('CONFIG_ERROR', `ComfyUI workflow: ${message}`); }
function record(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
export function readContract(config, capabilities) {
    if (!record(config['bindings']) || !record(config['outputs']))
        fail('declare bindings and outputs objects');
    const bindings = {};
    const outputs = {};
    const declaredKinds = record(config['inputKinds']) ? config['inputKinds'] : undefined;
    const inputKinds = {};
    for (const [key, value] of Object.entries(config['bindings'])) {
        if (!PARAMETER_NAME.test(key) || !record(value) || typeof value['node'] !== 'string' || typeof value['input'] !== 'string' || !value['node'] || !value['input'] || forbidden.test(value['input']))
            fail(`invalid or non-public binding "${key}"`);
        if (key === 'image' && !/image/i.test(value['input']))
            fail('image binding must target an image input');
        bindings[key] = { node: value['node'], input: value['input'] };
        const kind = declaredKinds?.[key];
        if (typeof kind === 'string' && INPUT_KINDS.includes(kind)) {
            inputKinds[key] = kind;
        }
    }
    for (const [key, value] of Object.entries(config['outputs'])) {
        if (!PARAMETER_NAME.test(key) || !record(value) || typeof value['node'] !== 'string' || !value['node'] || !(typeof value['type'] === 'string' && isIoType(value['type'])))
            fail(`invalid output "${key}"`);
        if (!capabilities.some((capability) => CAPABILITY_IO[capability].output.includes(value['type'])) && !(key === 'image' && capabilities.some((capability) => capability === 'image_to_3d' || capability === 'text_to_3d')))
            fail(`output "${key}" has a type not supported by its capabilities`);
        outputs[key] = { node: value['node'], type: value['type'] };
    }
    if (Object.keys(outputs).length === 0)
        fail('declare at least one output');
    for (const capability of capabilities) {
        if (!isCapability(capability))
            fail(`unsupported capability "${capability}"`);
        if (!Object.values(outputs).some((output) => CAPABILITY_IO[capability].output.includes(output.type)))
            fail(`no output for capability "${capability}"`);
        if (CAPABILITY_IO[capability].input.includes('image') && !bindings['image'])
            fail(`capability "${capability}" needs an image binding`);
        if ((capability === 'text_to_image' || capability === 'text_to_3d') && !bindings['prompt'])
            fail(`capability "${capability}" needs a prompt binding`);
    }
    return Object.keys(inputKinds).length === 0 ? { bindings, outputs } : { bindings, outputs, inputKinds };
}
export async function loadComfyGraph(config, catalogDir) {
    let raw = config['workflow'];
    if (raw === undefined) {
        const path = config['workflowPath'];
        if (typeof path !== 'string' || !path.trim())
            fail('workflowPath must name an API workflow JSON file');
        const resolved = resolveAdapterPath(path, catalogDir);
        let text;
        try {
            text = await readFile(resolved.absolute, 'utf8');
        }
        catch (error) {
            fail(`workflow file ${resolved.absolute} could not be read: ${String(error)}`);
        }
        try {
            raw = JSON.parse(text);
        }
        catch (error) {
            fail(`workflow file ${resolved.absolute} contains invalid JSON: ${String(error)}`);
        }
    }
    if (record(raw) && record(raw['prompt']))
        raw = raw['prompt'];
    if (!record(raw) || !Object.keys(raw).length)
        fail('API workflow must be a nonempty object of nodes');
    for (const [node, value] of Object.entries(raw)) {
        if (!record(value) || typeof value['class_type'] !== 'string' || !record(value['inputs']))
            fail(`node "${node}" is not an API-format node`);
    }
    return raw;
}
export function validateComfyGraph(graph, contract, classes) {
    for (const node of Object.values(graph)) {
        if (typeof node?.class_type !== 'string')
            continue;
        if (classes && !classes.has(node.class_type))
            fail('required ComfyUI node class is unavailable; check workflow dependencies');
    }
    for (const [name, binding] of Object.entries(contract.bindings)) {
        if (!Object.hasOwn(graph, binding.node))
            fail(`binding "${name}" references missing node "${binding.node}"`);
        if (!Object.hasOwn(graph[binding.node].inputs, binding.input))
            fail(`binding "${name}" references missing input "${binding.input}"`);
    }
    for (const [name, output] of Object.entries(contract.outputs)) {
        if (!Object.hasOwn(graph, output.node))
            fail(`output "${name}" references missing node "${output.node}"`);
        if (['image', 'model_3d', 'audio', 'video'].includes(name) && !/save|export|output|write|download|to_?file|to_?mesh|to_?3d/i.test(graph[output.node].class_type))
            fail(`output "${name}" must target a saving node`);
    }
}
