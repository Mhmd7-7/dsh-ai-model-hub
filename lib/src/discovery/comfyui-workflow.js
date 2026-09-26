/**
 * ComfyUI workflow discovery: what can this install actually *do*?
 *
 * ComfyUI holds two completely different kinds of thing, and the first version of
 * this discoverer only understood one of them:
 *
 * - **Weight files** — checkpoints, UNETs, text encoders, VAEs — enumerated by
 *   the file-listing enums of the loader nodes in `/object_info`. See
 *   `discovery/comfyui.ts`. A weight file proves an engine *can load* something;
 *   it says nothing about what graph that something is wired into.
 * - **Workflows** — the graphs an operator actually runs, stored by ComfyUI under
 *   its user data directory and enumerable through `GET /userdata?dir=workflows`.
 *   A workflow is the only artifact that proves a *capability*: it names the input
 *   node, the generation nodes, and the output node, and therefore what the engine
 *   can be asked to produce.
 *
 * The distinction is the whole point of this module. A ComfyUI install with
 * TRELLIS.2 nodes loaded but no 3D weights has the *machinery* for
 * `image_to_3d` and cannot serve it. An install holding a saved workflow that
 * loads an image, generates a mesh and writes a GLB can serve it even when no
 * checkpoint is involved at all. Publishing capability from weights alone
 * advertises work the engine then fails; publishing it from a workflow's own
 * structure is evidence.
 *
 * ## What is discoverable, and how
 *
 * | Question | Source | Reliability |
 * | --- | --- | --- |
 * | Is a node class installed? | `/object_info` keys | exact |
 * | What does a node take and return? | `/object_info` `input.required`, `output` | exact |
 * | Which weight files exist? | loader input enums in `/object_info` | exact |
 * | Which workflows are saved? | `GET /userdata?dir=workflows&recurse=true` | exact |
 * | What does a workflow contain? | `GET /userdata/{name}` | exact |
 * | Which models a workflow needs | file-valued inputs in the graph | exact names, no sizes |
 * | VRAM/RAM a workflow needs | **not reported by ComfyUI at all** | heuristic |
 *
 * The last row is why resource figures here are always labelled estimates. There is
 * no endpoint that reports a graph's memory demand; see {@link estimateComfyWorkflowVram}.
 *
 * ## Capability detection
 *
 * Capabilities are inferred from the *graph*, never from a filename:
 *
 * ```
 *   image input  +  a node that generates a mesh  +  a node that writes a 3D file
 *   = image_to_3d
 * ```
 *
 * Each of those three is established from evidence:
 *
 * - an **input node** is one whose class declares a file-valued input for the kind
 *   (`LoadImage.image`) or, for text, a string widget that reaches the sampler;
 * - a **3D generator** is a node that outputs a mesh kind (`MESH`, `File3D*`,
 *   `VOXEL`), consumes an image, latent, voxel or another mesh, and whose work
 *   produces *new* geometry rather than post-processing existing geometry;
 * - a **3D file writer** is an output node that consumes a 3D *file* kind and
 *   whose name says it writes rather than previews.
 *
 * Anything that cannot be proven this way is reported as **unknown** and grants
 * nothing. That is deliberate: a workflow the hub cannot classify is one an
 * operator can describe in configuration, and a fabricated capability is worse
 * than an honest gap.
 *
 * @module dsh-ai-model-hub/discovery/comfyui-workflow
 */
import { isCapability } from "../catalog/capabilities.js";
import { fetchJson, isRecordLike, readArray, readString, slugifyModelId, stableDigest } from "./http.js";
import { DISCOVERED_PRIORITY, ioForCapabilities } from "./types.js";
/** The route that lists a user's saved workflows. */
export const COMFYUI_WORKFLOW_DIR = 'workflows';
/** Query string that lists every file under the workflows directory, recursively. */
const WORKFLOW_LIST_QUERY = `dir=${COMFYUI_WORKFLOW_DIR}&recurse=true`;
/** Tags every discovered workflow carries. */
const WORKFLOW_TAGS = ['local', 'discovered', 'comfyui', 'workflow'];
/**
 * Input or output type names that prove geometry.
 *
 * ComfyUI declares a mesh-valued port as `MESH`, and a 3D *file* port as one of
 * the `File3D*` family (`File3DGLB`, `File3DOBJ`, …). Both spellings are matched
 * by shape rather than by an enumerated list, so a future `File3DUSDZ`-style
 * addition keeps working without a code change.
 */
const THREE_D_IO_PATTERN = /^(MESH|VOXEL|GAUSSIAN|SPLAT|FILE3D|FILE_3D|3D)/i;
/**
 * Whether an IO type name proves the port carries geometry.
 *
 * A **function declaration on purpose**: this is called from
 * {@link classifyComfyNode}, which is defined above it in the file, and a `const`
 * arrow function would be in its temporal dead zone at that point — the call would
 * throw at module load, and a `.some(isGeometryIo)` reference would be evaluated
 * as `undefined`. Declarations hoist; that is what makes the ordering safe.
 *
 * @param type - the declared IO type name.
 * @returns true when the type carries geometry.
 */
export function isGeometryIo(type) {
    const trimmed = type.trim().toUpperCase();
    return (trimmed === 'MESH' ||
        trimmed === 'VOXEL' ||
        trimmed === 'GAUSSIAN' ||
        trimmed === 'SPLAT' ||
        trimmed.startsWith('FILE3D') ||
        trimmed.startsWith('FILE_3D') ||
        trimmed.startsWith('3D'));
}
/** Whether an IO type name proves the port carries an image. */
function isImageIo(type) {
    return type.trim().toUpperCase() === 'IMAGE';
}
/** Whether an IO type name proves the port carries text. */
function isTextIo(type) {
    return type.trim().toUpperCase() === 'STRING';
}
/**
 * Node class names that *write* a 3D file the caller can download.
 *
 * The `Save*` family is ComfyUI's convention for an output node that persists to
 * the output directory, and `MeshToFile3D` is the converter TRELLIS.2 pipelines
 * use before a save. A `Preview3D*` node is deliberately absent: it writes to the
 * *temp* directory, which is not a deliverable.
 */
const THREE_D_WRITER_PATTERN = /save.*(glb|gltf|obj|stl|ply|fbx|usdz|3d|mesh|splat)|export.*(3d|mesh|glb)|mesh.*to.*file|(glb|obj|stl).*export/i;
/**
 * Node class names that *preview* 3D work without persisting it.
 *
 * Recognised so the difference can be reported. "This workflow previews a mesh it
 * never saves" is the most useful diagnostic an operator can get, and it is the
 * shape most downloaded workflows arrive in.
 */
const THREE_D_PREVIEW_PATTERN = /preview.*(3d|mesh|splat)|(3d|mesh|splat).*preview/i;
/**
 * Node class names that produce geometry but are *not* generation.
 *
 * Mesh post-processing (decimation, remeshing, welding, UV unwrapping, normals,
 * export conversion) consumes a mesh and returns a mesh. Treating one of those as
 * a generator would advertise `image_to_3d` for a workflow that cannot make
 * anything — the same false positive the weight-file discoverer was corrected for.
 */
const MESH_UTILITY_PATTERN = /decimate|remesh|weld|unwrap|smooth|fillholes|fill_holes|holed|remesh|postprocess|post_process|convert|tomesh$|morph|simplify|subdivide/i;
/** Node class names that load a model, a clip-vision encoder, or a VAE. */
const MODEL_LOADER_PATTERN = /^(unet|checkpoint|clip|vae|model|loadmodel|loadcheckpoint|loraloader).*loader/i;
/** File extensions ComfyUI's 3D writers can emit. */
const THREE_D_EXTENSIONS = ['.glb', '.gltf', '.obj', '.stl', '.ply', '.fbx', '.usdz', '.splat', '.spz', '.ksplat'];
/** Whether a filename looks like a 3D asset by extension. */
export function isThreeDFilename(filename) {
    const lower = filename.toLowerCase();
    return THREE_D_EXTENSIONS.some((extension) => lower.endsWith(extension));
}
/**
 * Read the IO contract of every node class in an `/object_info` document.
 *
 * Both the current `input.required` shape and the older flat `input` object are
 * accepted, because which one a ComfyUI release emits is not something the hub
 * gets to choose. An unrecognised shape yields fewer facts rather than a throw —
 * the same totality rule the weight-file reader follows.
 *
 * @param raw - the parsed `/object_info` body.
 * @returns the index, never `undefined`; an unreadable document yields `{}`.
 */
export function readComfyNodeIo(raw) {
    const index = {};
    if (!isRecordLike(raw))
        return index;
    for (const [className, spec] of Object.entries(raw)) {
        const inputs = {};
        const inputBlock = isRecordLike(spec) ? spec['input'] : undefined;
        const required = isRecordLike(inputBlock) ? inputBlock['required'] : undefined;
        const optional = isRecordLike(inputBlock) ? inputBlock['optional'] : undefined;
        for (const fields of [required, optional]) {
            if (!isRecordLike(fields))
                continue;
            for (const [fieldName, fieldSpec] of Object.entries(fields)) {
                const types = inputTypeNames(fieldSpec);
                if (types.length > 0)
                    inputs[fieldName] = types;
            }
        }
        const outputs = outputTypeNames(isRecordLike(spec) ? spec['output'] : undefined);
        index[className] = { inputs, outputs };
    }
    return index;
}
/**
 * The type names declared for one input field.
 *
 * ComfyUI writes a typed input as `["IMAGE", {...}]` and an enumerable one as
 * `[["a.safetensors", "b.ckpt"], {...}]`, and which of those a field is decides
 * whether it has a *port type* for the classifier to read. An element list whose
 * entries look like filenames is a file enumeration, not a list of type names, so
 * it is reported as `STRING` — the field is a string-valued choice. Without this
 * distinction a `SaveGLB`'s `mesh: [["MESH"], {}]`-style list would be read as a
 * filename list and the node would look like it takes no mesh at all.
 *
 * @param fieldSpec - the field's spec value.
 * @returns the declared type names; empty when the field carries only values.
 */
function inputTypeNames(fieldSpec) {
    if (!Array.isArray(fieldSpec))
        return [];
    const head = fieldSpec[0];
    if (typeof head === 'string')
        return [head];
    if (Array.isArray(head)) {
        if (head.some((entry) => typeof entry === 'string' && looksLikeFilename(entry)))
            return ['STRING'];
        const strings = head.filter((entry) => typeof entry === 'string');
        return strings.length === 0 ? [] : strings;
    }
    return [];
}
/** Whether an option-list entry names a file rather than a port type. */
function looksLikeFilename(value) {
    const trimmed = value.trim();
    if (trimmed.length === 0)
        return false;
    return /[/\\]/.test(trimmed) || /\.(safetensors|ckpt|pt|pth|bin|gguf|sft|onnx)$/i.test(trimmed) || /\.(png|jpe?g|webp|mp4|glb|obj|ply|stl)$/i.test(trimmed);
}
/**
 * The output type names declared by a node class.
 *
 * @param raw - the node's `output` field.
 * @returns the type names in declaration order.
 */
function outputTypeNames(raw) {
    if (!Array.isArray(raw))
        return [];
    const names = [];
    for (const entry of raw) {
        if (typeof entry === 'string' && entry.trim().length > 0)
            names.push(entry.trim());
    }
    return names;
}
/**
 * Classify one node class from its declared IO and its name.
 *
 * The name is used only for the handful of questions the IO cannot answer — does
 * this node *generate* or *post-process*, does it *write* or *preview* — and never
 * to identify a model. Every capability decision downstream reads this
 * classification, so a node pack's naming convention only ever affects these
 * three booleans.
 *
 * @param className - the node class name.
 * @param io - its declared IO, or `undefined` when the engine did not list it.
 * @returns the classification.
 */
export function classifyComfyNode(className, io) {
    const takes = io === undefined ? [] : unique(Object.values(io.inputs).flat());
    const returns = io === undefined ? [] : [...io.outputs];
    const hasMeshInput = takes.some(isGeometryIo);
    const hasImageInput = takes.some(isImageIo);
    const returnsGeometry = returns.some(isGeometryIo);
    const returnsGeometryFile = returns.some((type) => /^FILE3D|^FILE_3D/i.test(type.trim()));
    const isFileInput = io !== undefined && Object.keys(io.inputs).some((field) => isFileField(className, field));
    // A writer **consumes** geometry and persists it. It deliberately does *not*
    // have to return geometry: `SaveGLB` returns nothing at all, which is exactly
    // what "this wrote a file" looks like in `/object_info`. Requiring a geometry
    // return here would make every real 3D saver invisible, and the resulting
    // "this graph cannot return a mesh" refusal would be the hub's fault rather than
    // the workflow's.
    const isGeometryWriter = hasMeshInput && THREE_D_WRITER_PATTERN.test(className) && !THREE_D_PREVIEW_PATTERN.test(className);
    const isGeometryPreview = hasMeshInput && THREE_D_PREVIEW_PATTERN.test(className) && !isGeometryWriter;
    const isUtility = MESH_UTILITY_PATTERN.test(className);
    // A generator is a node that **returns geometry it built**. Every 3D-generation
    // node the hub has seen declares that shape: TRELLIS.2's shape decoder takes a
    // latent and a mesh and returns a refined mesh, Hunyuan3D's voxel-to-mesh takes a
    // voxel and returns a mesh, Pixal3D's conditioning feeds a latent the decoder
    // reads. So the evidence is the engine's own port declaration — a mesh/voxel/splat
    // on the way out — plus geometry on the way in, which is what separates a
    // generator from a writer (no geometry in) and from a pure loader (no geometry
    // out).
    //
    // The one thing it must not do is require an *image* input: a text-driven
    // generator conditions on text instead, and requiring an image would make
    // `text_to_3d` undetectable — the capability the caller asked for and the one
    // this must neither fabricate nor hide.
    const isGeometryGenerator = returnsGeometry &&
        !returnsGeometryFile &&
        !isGeometryWriter &&
        !isGeometryPreview &&
        !isUtility &&
        hasMeshInput &&
        !MODEL_LOADER_PATTERN.test(className);
    return {
        className,
        takes,
        returns,
        returnsGeometry,
        returnsGeometryFile,
        consumesMesh: hasMeshInput && !returnsGeometry,
        processesMesh: hasMeshInput && returnsGeometry,
        isGeometryGenerator,
        takesImage: hasImageInput,
        isFileInput,
        isGeometryWriter,
        isGeometryPreview,
    };
}
/**
 * Whether an input field is a file-valued choice rather than a typed port.
 *
 * ComfyUI's loaders declare their file as an enumeration of filenames, which
 * {@link inputTypeNames} has already normalized to `STRING`. The field *name* is
 * what distinguishes "this node takes a file off disk" from "this node takes a
 * string to type", and those names are a stable part of the loader contracts.
 *
 * @param className - the node class, for the image loader's own convention.
 * @param fieldName - the input field name.
 * @returns true when the field selects a file.
 */
function isFileField(className, fieldName) {
    if (/^(ckpt|unet|vae|clip|lora|model|control_net|style_model|embedding|gligen|diffusion_model).*_name$/i.test(fieldName)) {
        return true;
    }
    // A *model* file selected by a loader. Deliberately narrower than "a file": the
    // image, video, and audio a workflow loads are its *inputs*, not its model
    // files, and recording them as dependencies would tell an operator they are
    // missing a weight when they are missing an artifact.
    if (/^(model_file|mesh_file|file)$/i.test(fieldName) && /load/i.test(className))
        return true;
    if (/^(model|weights?)_(file|path|name)$/i.test(fieldName))
        return true;
    return false;
}
/**
 * Remove duplicate strings, preserving first-seen order.
 * @param values - the values to deduplicate.
 * @returns the unique values.
 */
function unique(values) {
    const seen = new Set();
    const result = [];
    for (const value of values) {
        if (seen.has(value))
            continue;
        seen.add(value);
        result.push(value);
    }
    return result;
}
/** A problem that makes a workflow unusable, as opposed to merely unclassifiable. */
export class ComfyWorkflowError extends Error {
    /**
     * @param message - what is wrong with the document.
     */
    constructor(message) {
        super(message);
        this.name = 'ComfyWorkflowError';
    }
}
/**
 * Parse a workflow document into the shape capability detection walks.
 *
 * Three serializations exist in the wild and all three are accepted:
 *
 * - **API format** — `{ "3": { class_type, inputs } }`, which is what the
 *   adapter queues. Used as-is.
 * - **UI format** — `{ nodes: [...], links: [...] }`, which is what ComfyUI's
 *   editor saves and what users therefore have. Converted, because a UI export is
 *   a perfectly good statement of what the workflow does and refusing it would
 *   make discovery useless on most real installs.
 * - **A `{ prompt: … }` envelope** — the shape ComfyUI's own API examples use.
 *
 * @param raw - the parsed document.
 * @param name - the workflow's name, for messages.
 * @param io - the node index, needed only to convert a UI export.
 * @returns the parsed workflow.
 * @throws ComfyWorkflowError when the document is not a workflow this can read.
 */
export function parseComfyWorkflow(raw, name, io = {}) {
    if (!isRecordLike(raw)) {
        throw new ComfyWorkflowError(`workflow "${name}" is not a JSON object`);
    }
    const envelope = raw['prompt'];
    const document = isRecordLike(envelope) && !Array.isArray(raw['nodes']) ? envelope : raw;
    if (Array.isArray(document['nodes'])) {
        return convertUiWorkflow(document, name, io);
    }
    const nodes = {};
    let recognised = 0;
    for (const [id, value] of Object.entries(document)) {
        if (!isRecordLike(value))
            continue;
        const classType = readString(value, 'class_type');
        if (classType === undefined)
            continue;
        recognised += 1;
        nodes[id] = { classType, inputs: isRecordLike(value['inputs']) ? value['inputs'] : {} };
    }
    if (recognised === 0) {
        throw new ComfyWorkflowError(`workflow "${name}" has no nodes: an API-format graph is an object of nodes keyed by id, each with a ` +
            'class_type. A UI export normally has a `nodes` array instead.');
    }
    return { name, nodes, graph: document, format: 'api' };
}
/**
 * Convert a UI-format export into an API-format graph.
 *
 * ComfyUI's editor stores connections in `links` (as link ids) and widget values
 * in a positional `widgets_values` array, while the API wants both inline on each
 * node. Two facts make the conversion well-defined rather than guesswork:
 *
 * 1. the node's `inputs` array says which named inputs are *connected*, by link
 *    id, so those are reconstructed exactly; and
 * 2. `/object_info` declares the remaining required inputs **in the order the
 *    editor produced `widgets_values`**, so the leftover widget values map
 *    positionally onto the leftover input names.
 *
 * Without the second fact a conversion would have to guess, so a UI export is
 * converted only when `/object_info` describes its nodes; otherwise this reports
 * that it cannot and the workflow is left unclassified rather than misread. A
 * handful of core nodes order their widgets differently from their schema
 * (samplers put `seed` first), and {@link WIDGET_ORDER_OVERRIDES} records those.
 *
 * @param document - the UI-format document.
 * @param name - the workflow's name, for messages.
 * @param io - the node index.
 * @returns the parsed workflow.
 * @throws ComfyWorkflowError when the export cannot be converted faithfully.
 */
function convertUiWorkflow(document, name, io) {
    const uiNodes = readArray(document, 'nodes');
    const links = new Map();
    for (const link of readArray(document, 'links')) {
        if (!Array.isArray(link))
            continue;
        const id = link[0];
        const originId = link[1];
        const originSlot = link[2];
        if (typeof id === 'number' && (typeof originId === 'string' || typeof originId === 'number')) {
            links.set(id, [String(originId), typeof originSlot === 'number' ? originSlot : 0]);
        }
    }
    const nodes = {};
    const missingSchema = [];
    for (const entry of uiNodes) {
        if (!isRecordLike(entry))
            continue;
        const classType = readString(entry, 'type');
        const id = entry['id'];
        if (classType === undefined || (typeof id !== 'string' && typeof id !== 'number'))
            continue;
        const spec = io[classType];
        if (spec === undefined) {
            missingSchema.push(classType);
            continue;
        }
        const inputs = {};
        const connected = [];
        for (const port of readArray(entry, 'inputs')) {
            if (!isRecordLike(port))
                continue;
            const portName = readString(port, 'name');
            const linkId = port['link'];
            if (portName === undefined)
                continue;
            connected.push(portName);
            if (typeof linkId === 'number') {
                const origin = links.get(linkId);
                if (origin !== undefined)
                    inputs[portName] = [origin[0], origin[1]];
            }
        }
        // `widgets_values` is either a positional array or, on some nodes, an object
        // keyed by input name. Both are read; the positional form is the common one.
        const widgetValues = entry['widgets_values'];
        if (isRecordLike(widgetValues)) {
            for (const [key, value] of Object.entries(widgetValues)) {
                if (inputs[key] === undefined)
                    inputs[key] = value;
            }
        }
        else if (Array.isArray(widgetValues)) {
            const order = widgetInputOrder(classType, spec, connected);
            let cursor = 0;
            for (const fieldName of order) {
                if (cursor >= widgetValues.length)
                    break;
                // A `undefined` slot consumes its value without assigning it; see
                // {@link widgetInputOrder}.
                if (fieldName !== undefined)
                    inputs[fieldName] = widgetValues[cursor];
                cursor += 1;
            }
        }
        nodes[String(id)] = { classType, inputs };
    }
    if (missingSchema.length > 0) {
        throw new ComfyWorkflowError(`workflow "${name}" is a UI export that names ${unique(missingSchema).length} node class(es) this ComfyUI ` +
            `does not have (${unique(missingSchema).slice(0, 5).join(', ')}), so its graph cannot be reconstructed`);
    }
    if (Object.keys(nodes).length === 0) {
        throw new ComfyWorkflowError(`workflow "${name}" is a UI export with no readable nodes`);
    }
    return {
        name,
        nodes,
        graph: nodes,
        format: 'ui',
        conversionNote: `This workflow was saved in ComfyUI's *UI* format and was converted to an API-format graph at discovery ` +
            `time, using each node's declared inputs from /object_info. Widget values on node classes this hub does not ` +
            `know how to order may be off; if ComfyUI rejects the graph, export the workflow with "Save (API Format)" ` +
            `or set adapterConfig.workflow explicitly.`,
    };
}
/**
 * Widget values whose editor order differs from the schema's declaration order.
 *
 * ComfyUI's editor emits `widgets_values` in the order its own widget list was
 * built, which for a few long-standing core nodes is not the order
 * `input.required` lists. These are the ones that matter for the graphs this hub
 * runs; a node pack outside this table falls back to the schema order, which is
 * right for everything that follows the framework convention.
 */
const WIDGET_ORDER_OVERRIDES = {
    KSampler: [
        'seed',
        'control_after_generate',
        'steps',
        'cfg',
        'sampler_name',
        'scheduler',
        'denoise',
    ],
    KSamplerAdvanced: [
        'add_noise',
        'noise_seed',
        'control_after_generate',
        'steps',
        'cfg',
        'sampler_name',
        'scheduler',
        'start_at_step',
        'end_at_step',
        'return_with_leftover_noise',
    ],
    CheckpointLoaderSimple: ['ckpt_name'],
    CheckpointLoader: ['config_name', 'ckpt_name'],
    UNETLoader: ['unet_name', 'weight_dtype'],
    VAELoader: ['vae_name'],
    CLIPLoader: ['clip_name', 'type', 'device'],
    DualCLIPLoader: ['clip_name1', 'clip_name2', 'type', 'device'],
    LoraLoader: ['lora_name', 'strength_model', 'strength_clip'],
    LoadImage: ['image', 'upload'],
    CLIPTextEncode: ['text'],
    EmptyLatentImage: ['width', 'height', 'batch_size'],
    EmptySD3LatentImage: ['width', 'height', 'batch_size'],
};
/**
 * The widget slot names a UI export's `widgets_values` map onto, in order.
 *
 * An entry that is `undefined` is a widget the hub does not assign: the editor
 * emits some controls that are not node inputs at all — `control_after_generate`
 * on every sampler is the common one — and *omitting* it would shift every value
 * after it by one, silently turning a step count into a CFG scale. So the slot is
 * kept and skipped rather than filtered out.
 *
 * @param className - the node class.
 * @param spec - its declared IO.
 * @param connected - the input names already satisfied by a link.
 * @returns the widget slots, in editor order.
 */
function widgetInputOrder(className, spec, connected) {
    const linked = new Set(connected);
    const declared = Object.keys(spec.inputs).filter((field) => !linked.has(field) && !spec.outputs.includes(field));
    const override = WIDGET_ORDER_OVERRIDES[className];
    if (override === undefined)
        return declared;
    const slots = override.map((field) => (declared.includes(field) ? field : undefined));
    return [...slots, ...declared.filter((field) => !override.includes(field))];
}
/**
 * The workflow names in a `GET /userdata?dir=workflows` response.
 *
 * The route answers with a flat array of paths relative to the workflows
 * directory, optionally with a `.json` extension the caller may or may not have
 * used. Directory entries are not returned by this route, so every element is a
 * file; a non-`.json` element is skipped because a workflow is JSON.
 *
 * @param raw - the parsed response.
 * @returns the workflow references, in engine order.
 */
export function parseWorkflowList(raw) {
    const entries = Array.isArray(raw) ? raw : Array.isArray(readArray(raw, 'files')) ? readArray(raw, 'files') : [];
    const refs = [];
    const seen = new Set();
    for (const entry of entries) {
        let name;
        if (typeof entry === 'string')
            name = entry;
        else if (isRecordLike(entry))
            name = readString(entry, 'path') ?? readString(entry, 'name');
        if (name === undefined)
            continue;
        const cleaned = name.replace(/^\/+/, '').trim();
        if (cleaned.length === 0 || !cleaned.toLowerCase().endsWith('.json'))
            continue;
        if (seen.has(cleaned))
            continue;
        seen.add(cleaned);
        refs.push({ name: cleaned });
    }
    return refs;
}
/**
 * Read and parse every saved workflow this ComfyUI exposes.
 *
 * A workflow that cannot be read or parsed costs one workflow, never the pass:
 * the reasons are collected in {@link ComfyWorkflowReadResult.skipped} so an
 * operator sees exactly which file was unusable and why. That containment matters
 * more here than elsewhere because workflow files are user-authored — a truncated
 * download or an experimental graph must not be able to hide the other twenty
 * workflows in the directory.
 *
 * @param endpoint - the ComfyUI base URL.
 * @param io - the node index, needed to convert UI-format exports.
 * @param signal - cancellation for the pass.
 * @param timeoutMs - budget per request.
 * @param options - `maxWorkflows` bounds how many documents are read.
 * @returns what was read, and what could not be.
 */
export async function readComfyWorkflows(endpoint, io, signal, timeoutMs, options = {}) {
    const base = endpoint.endsWith('/') ? endpoint.slice(0, -1) : endpoint;
    const listing = await fetchJson(`${base}/userdata?${WORKFLOW_LIST_QUERY}`, signal, timeoutMs);
    if (!listing.ok)
        return { workflows: [], skipped: [], listed: false };
    const refs = parseWorkflowList(listing.value);
    const limit = Math.max(0, options.maxWorkflows ?? Number.POSITIVE_INFINITY);
    const workflows = [];
    const skipped = [];
    for (const ref of refs.slice(0, limit)) {
        const encoded = ref.name
            .split('/')
            .map((segment) => encodeURIComponent(segment))
            .join('/');
        const read = await fetchJson(`${base}/userdata/${encoded}`, signal, timeoutMs);
        if (!read.ok) {
            skipped.push(`${ref.name}: ${read.reason}`);
            continue;
        }
        try {
            workflows.push(parseComfyWorkflow(read.value, ref.name, io));
        }
        catch (error) {
            skipped.push(error instanceof Error ? error.message : String(error));
        }
    }
    return { workflows, skipped, listed: true };
}
/**
 * The capabilities a workflow's own graph proves.
 *
 * The inference is deliberately three-part and deliberately strict:
 *
 * 1. **an input node** for the capability's source kind exists, and
 * 2. **a geometry generator** — a node that returns a mesh it built, from an
 *    image/latent/voxel/mesh input — exists, and
 * 3. **a writer** that persists a 3D file exists, and
 * 4. the generator is reachable from the input node and the writer from the
 *    generator, so three unrelated islands in one file do not add up to a
 *    capability.
 *
 * Failing any of those grants nothing and records why. `text_to_3d` is held to the
 * same standard, which in practice means it is only advertised for a graph whose
 * prompt genuinely feeds the generator — the common case, an image-to-3D workflow
 * that happens to contain a `CLIPTextEncode` for a texture pass, does *not* qualify,
 * and the note says so rather than the capability being invented.
 *
 * The two capabilities are **not mutually exclusive**, and a graph that conditions
 * its generator on both an image and a prompt honestly proves both. What the
 * reachability test rules out is a capability whose *source* never arrives.
 *
 * @param workflow - the parsed workflow.
 * @param io - the node index.
 * @returns the inference.
 */
export function inferWorkflowCapabilities(workflow, io) {
    const classes = new Map();
    const classificationOf = (className) => {
        const existing = classes.get(className);
        if (existing !== undefined)
            return existing;
        const created = classifyComfyNode(className, io[className]);
        classes.set(className, created);
        return created;
    };
    const imageInputs = [];
    const textInputs = [];
    const generators = [];
    const writers = [];
    const previews = [];
    for (const [id, node] of Object.entries(workflow.nodes)) {
        const info = classificationOf(node.classType);
        if (isImageInputNode(node.classType, info))
            imageInputs.push(id);
        if (isTextInputNode(node.classType, info, io))
            textInputs.push(id);
        if (info.isGeometryGenerator)
            generators.push(id);
        if (info.isGeometryWriter)
            writers.push(id);
        if (info.isGeometryPreview)
            previews.push(id);
    }
    const inputClasses = unique([
        ...imageInputs.map((id) => workflow.nodes[id]?.classType ?? ''),
        ...textInputs.map((id) => workflow.nodes[id]?.classType ?? ''),
    ]).filter((name) => name.length > 0);
    const generatorClasses = unique(generators.map((id) => workflow.nodes[id]?.classType ?? '')).filter((name) => name.length > 0);
    const evidence = [];
    const notes = [];
    const capabilities = [];
    if (generators.length === 0) {
        notes.push(previews.length > 0
            ? `the graph manipulates a mesh (${unique(previews.map((id) => workflow.nodes[id]?.classType ?? '')).join(', ')}) ` +
                'but contains no node that generates new geometry'
            : 'the graph contains no node that generates mesh geometry');
    }
    if (writers.length === 0) {
        notes.push(previews.length > 0
            ? 'the graph only previews 3D output (a preview writes to ComfyUI\'s temp directory, not a deliverable)'
            : 'the graph has no node that writes a 3D file, so nothing could be returned to a caller');
    }
    if (generators.length > 0 && writers.length > 0) {
        const reachable = reachableSet(workflow);
        const writableGenerators = generators.filter((generator) => writers.some((writer) => reaches(reachable, generator, writer)));
        if (writableGenerators.length > 0) {
            evidence.push(`3D generation: ${unique(writableGenerators.map((id) => workflow.nodes[id]?.classType ?? '')).join(', ')} ` +
                `→ ${unique(writers.map((id) => workflow.nodes[id]?.classType ?? '')).join(', ')}`);
            const reachableImages = imageInputs.filter((input) => writableGenerators.some((generator) => reaches(reachable, input, generator)));
            if (reachableImages.length > 0) {
                capabilities.push('image_to_3d');
                evidence.push(`image input: ${unique(reachableImages.map((id) => workflow.nodes[id]?.classType ?? '')).join(', ')}`);
            }
            const reachableText = textInputs.filter((input) => writableGenerators.some((generator) => reaches(reachable, input, generator)));
            if (reachableText.length > 0) {
                capabilities.push('text_to_3d');
                evidence.push(`text input: ${unique(reachableText.map((id) => workflow.nodes[id]?.classType ?? '')).join(', ')}`);
            }
            if (reachableImages.length === 0 && reachableText.length === 0) {
                notes.push(imageInputs.length > 0 || textInputs.length > 0
                    ? 'the graph loads an image or a prompt, but neither feeds the 3D generator, so no capability is proven'
                    : 'the graph has no image or text input node, so nothing can be handed to the 3D generator');
            }
            if (capabilities.includes('image_to_3d') && reachableText.length === 0 && textInputs.length > 0) {
                // Worth stating explicitly: this is the case operators expect to be
                // text_to_3d and it is not one.
                notes.push('a prompt node exists but does not reach the 3D generator, so text_to_3d is not advertised ' +
                    '(the prompt conditions something else, such as a texture pass)');
            }
        }
        else {
            notes.push('a generator and a 3D writer both exist, but the generator does not feed the writer');
        }
    }
    if (capabilities.length === 0) {
        notes.push('no capability is proven by this graph; add one to configuration if you know what it does');
    }
    return {
        capabilities: orderCapabilities(capabilities),
        evidence,
        notes,
        generatorClasses,
        inputClasses,
        hasGeometryWriter: writers.length > 0,
    };
}
/**
 * Whether a node class is an image input for a generation graph.
 *
 * A `LoadImage` node is the one port whose *name* proves it supplies an image, and
 * this is the case that matters: without it, `image_to_3d` could never be inferred,
 * because the image-loading node is the only evidence that an image can be fed in.
 * The test is on the class name because that is exactly what the name means in
 * ComfyUI — `LoadImage`, `LoadImageMask`, `LoadImageOutput` and their
 * third-party equivalents all read a file the caller supplies.
 *
 * It deliberately does **not** require the `isFileInput` heuristic, which matches
 * weight-file field names (`ckpt_name`, `vae_name`) and has nothing to say about an
 * image; requiring it would silently disable the whole image path.
 *
 * @param className - the node class.
 * @param info - its classification.
 * @returns true when this node supplies an image.
 */
function isImageInputNode(className, info) {
    if (!/image/i.test(className))
        return false;
    if (/encode|decode|to_?image|image_?to|preview|save|resize|scale|composite|blend|batch/i.test(className))
        return false;
    return /load|input|read|open|crop|mask/i.test(className);
}
/**
 * Whether a node class is a text input for a generation graph.
 *
 * A `CLIPTextEncode` whose `text` is a widget is ComfyUI's prompt node; the test
 * is on the class name because that is the framework's own contract for it. The
 * field check keeps a node that merely *returns* text out of the set.
 *
 * @param className - the node class.
 * @param info - its classification.
 * @param io - the node index.
 * @returns true when this node supplies prompt text.
 */
function isTextInputNode(className, info, io) {
    const lower = className.toLowerCase();
    if (!/textencode|text_encode|textinput|cliptext|prompt/.test(lower))
        return false;
    const inputs = io[className]?.inputs ?? {};
    return Object.keys(inputs).some((field) => /^text$/i.test(field)) || info.takes.some(isTextIo);
}
/**
 * The set of nodes reachable from anywhere, computed by reverse traversal.
 *
 * Built once per workflow rather than per query: the graph is small (tens of
 * nodes) and {@link reaches} is asked a handful of times.
 *
 * @param workflow - the parsed workflow.
 * @returns the reverse adjacency map: node id → the ids that feed it.
 */
function reachableSet(workflow) {
    const consumers = new Map();
    for (const [id, node] of Object.entries(workflow.nodes)) {
        for (const link of Object.values(node.inputs)) {
            const source = linkSource(link);
            if (source === undefined)
                continue;
            const bucket = consumers.get(source);
            if (bucket === undefined)
                consumers.set(source, [id]);
            else
                bucket.push(id);
        }
    }
    return consumers;
}
/**
 * Whether `from` reaches `to` by following links forwards.
 *
 * @param consumers - the reverse adjacency map from {@link reachableSet}.
 * @param from - the source node id.
 * @param to - the target node id.
 * @returns true when a path exists.
 */
function reaches(consumers, from, to) {
    if (from === to)
        return true;
    const seen = new Set([from]);
    const queue = [from];
    while (queue.length > 0) {
        const current = queue.shift();
        for (const next of consumers.get(current) ?? []) {
            if (next === to)
                return true;
            if (seen.has(next))
                continue;
            seen.add(next);
            queue.push(next);
        }
    }
    return false;
}
/**
 * The node id a link points at, when it points at one.
 *
 * An API-format link is `[nodeId, slot]` where the node id may be a string or a
 * number depending on who wrote the file. A bare string is accepted too, because
 * a hand-written graph commonly omits the slot.
 *
 * @param link - the input value.
 * @returns the source node id, or `undefined` for a literal value.
 */
export function linkSource(link) {
    if (typeof link === 'string')
        return link.length > 0 ? link : undefined;
    if (Array.isArray(link) && link.length > 0) {
        const head = link[0];
        if (typeof head === 'string')
            return head;
        if (typeof head === 'number' && Number.isFinite(head))
            return String(head);
    }
    return undefined;
}
/**
 * Put capabilities into the catalog's canonical vocabulary order.
 *
 * @param capabilities - the capabilities found.
 * @returns them in vocabulary order, deduplicated.
 */
function orderCapabilities(capabilities) {
    const order = ['text_to_text', 'text_to_image', 'image_to_image', 'text_to_3d', 'image_to_3d'];
    const result = [];
    for (const capability of order)
        if (capabilities.includes(capability))
            result.push(capability);
    for (const capability of capabilities)
        if (!result.includes(capability))
            result.push(capability);
    return result;
}
/**
 * The model `type` a workflow's capabilities imply.
 * @param capabilities - the proven capabilities.
 * @returns the model type.
 */
function workflowModelType(capabilities) {
    if (capabilities.includes('text_to_3d') || capabilities.includes('image_to_3d'))
        return 'three_d_generation';
    if (capabilities.includes('image_to_image'))
        return 'image_editing';
    return 'image_generation';
}
/** The workflow name without its directory and extension, for display. */
function displayNameOf(name) {
    const last = name.split('/').pop() ?? name;
    return last.replace(/\.json$/i, '');
}
// ── resource estimation ─────────────────────────────────────────────────────
/**
 * VRAM estimates, in gibibytes, for the architectures ComfyUI's 3D pipelines use.
 *
 * **A heuristic on names, and documented as one everywhere it appears.**
 * ComfyUI reports neither a workflow's memory demand nor a model file's size, so
 * there is nothing exact to read; the router treats declared resources as a filter
 * rather than a reservation, so an approximate conservative figure is the right
 * kind of answer. An operator who knows better overrides it in the catalog entry
 * that pins the workflow, which always wins.
 */
const WORKFLOW_VRAM_PATTERNS = [
    { pattern: /trellis|pixal|hunyuan3d|hunyuan_3d|stable3d|stable_3d|sf3d|triposplat|tripo.*splat/i, vramGb: 16 },
    { pattern: /hunyuan|voxel|moge|zero123|sv3d|instantmesh|wonder3d/i, vramGb: 12 },
];
/** The estimate used when nothing recognisable appears in the workflow. */
const DEFAULT_WORKFLOW_VRAM_GB = 12;
/** The estimate used when the graph shows a texture or multi-stage pass. */
const TEXTURED_WORKFLOW_VRAM_GB = 16;
/**
 * Estimate a workflow's VRAM need from the node classes and files it names.
 *
 * Two signals are combined: the architecture keywords in the graph's own node
 * classes, and whether the graph clearly runs more than one generation stage
 * (shape plus texture), which is what pushes TRELLIS-style pipelines past 12 GiB.
 *
 * @param workflow - the parsed workflow.
 * @param inference - the capability inference, for the generator classes.
 * @returns the estimate in gibibytes.
 */
export function estimateComfyWorkflowVram(workflow, inference) {
    void inference;
    const haystack = Object.values(workflow.nodes)
        .map((node) => node.classType)
        .join(' ');
    for (const rule of WORKFLOW_VRAM_PATTERNS) {
        if (rule.pattern.test(haystack)) {
            return Math.max(rule.vramGb, workflowHasTexturePass(haystack) ? TEXTURED_WORKFLOW_VRAM_GB : rule.vramGb);
        }
    }
    return workflowHasTexturePass(haystack) ? TEXTURED_WORKFLOW_VRAM_GB : DEFAULT_WORKFLOW_VRAM_GB;
}
/**
 * Whether a graph applies textures in a second pass.
 * @param haystack - every node class name, joined.
 * @returns true when a texture stage is present.
 */
function workflowHasTexturePass(haystack) {
    return /texture|paintmesh|unwrap|uvatlas|bake/i.test(haystack);
}
/**
 * Parse a host's `adapterConfig.workflows` override list.
 *
 * Validated strictly and reported as one reason, because an override is operator
 * data: a typo that reached the catalog as a schema error would take the whole
 * catalog down rather than one entry, which is a far worse failure than a refusal
 * naming the field.
 *
 * @param host - the configured host.
 * @returns the parsed overrides, or why the list is unusable.
 */
export function parseComfyWorkflowOverrides(host) {
    const source = host.adapterConfig?.['workflows'];
    if (source === undefined)
        return { ok: true, overrides: [] };
    if (!Array.isArray(source)) {
        return { ok: false, reason: '`workflows` must be an array of workflow overrides' };
    }
    const overrides = [];
    for (const [index, entry] of source.entries()) {
        if (!isRecordLike(entry))
            return { ok: false, reason: `workflows[${index}] must be an object` };
        const workflowName = readString(entry, 'workflowName');
        if (workflowName === undefined) {
            return { ok: false, reason: `workflows[${index}].workflowName is required (the name ComfyUI stores it under)` };
        }
        const declared = readCapabilities(entry['capabilities']);
        if (declared === undefined) {
            return { ok: false, reason: `workflows[${index}].capabilities must be an array of known capabilities` };
        }
        const override = { workflowName };
        const id = readString(entry, 'id');
        if (id !== undefined)
            override.id = id;
        const name = readString(entry, 'name');
        if (name !== undefined)
            override.name = name;
        // `workflowPath` is the field the adapter already reads, so an operator can
        // pin a workflow with the same spelling they would use on a hand-written
        // entry.
        const workflowPath = readString(entry, 'workflowPath');
        if (workflowPath !== undefined)
            override.workflowPath = workflowPath;
        if (declared.length > 0)
            override.capabilities = declared;
        else if (Array.isArray(entry['capabilities']) && entry['capabilities'].length === 0)
            override.capabilities = [];
        if (typeof entry['enabled'] === 'boolean')
            override.enabled = entry['enabled'];
        const priority = readNumberValue(entry['priority']);
        if (priority !== undefined)
            override.priority = priority;
        if (Array.isArray(entry['tags'])) {
            const tags = entry['tags'].filter((tag) => typeof tag === 'string');
            if (tags.length !== entry['tags'].length)
                return { ok: false, reason: `workflows[${index}].tags must be strings` };
            override.tags = tags;
        }
        const vramGb = readNumberValue(entry['vramGb']);
        if (vramGb !== undefined)
            override.vramGb = vramGb;
        const ramGb = readNumberValue(entry['ramGb']);
        if (ramGb !== undefined)
            override.ramGb = ramGb;
        if (typeof entry['requiresGpu'] === 'boolean')
            override.requiresGpu = entry['requiresGpu'];
        const notes = readString(entry, 'notes');
        if (notes !== undefined)
            override.notes = notes;
        overrides.push(override);
    }
    return { ok: true, overrides };
}
/**
 * Read an optional capability list, rejecting unknown names.
 * @param raw - the candidate value.
 * @returns the capabilities, `[]` when absent, or `undefined` when invalid.
 */
function readCapabilities(raw) {
    if (raw === undefined)
        return [];
    if (!Array.isArray(raw))
        return undefined;
    const capabilities = [];
    for (const entry of raw) {
        if (typeof entry !== 'string' || !isCapability(entry))
            return undefined;
        if (!capabilities.includes(entry))
            capabilities.push(entry);
    }
    return capabilities;
}
/**
 * Read an optional finite number.
 * @param value - the candidate value.
 * @returns the number, or `undefined`.
 */
function readNumberValue(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}
/**
 * Find the override that names a workflow, if any.
 *
 * Matching is on the stored name, its basename, and either spelling with or
 * without `.json`, case-insensitively — the three ways an operator reasonably
 * writes it. `workflowPath` is accepted as an alias so a workflow can be pinned
 * with the same field name the adapter already uses for its rendered graph.
 *
 * @param overrides - the parsed overrides.
 * @param workflowName - the workflow's stored name.
 * @returns the matching override, or `undefined`.
 */
export function matchWorkflowOverride(overrides, workflowName) {
    const normalize = (value) => value.replace(/^\/+/, '').replace(/\\/g, '/').toLowerCase();
    const candidates = new Set();
    for (const value of [workflowName, displayNameOf(workflowName)]) {
        const normalized = normalize(value);
        candidates.add(normalized);
        candidates.add(normalized.replace(/\.json$/, ''));
    }
    for (const override of overrides) {
        const normalized = normalize(override.workflowName);
        if (candidates.has(normalized) || candidates.has(normalized.replace(/\.json$/, '')))
            return override;
    }
    return undefined;
}
/**
 * Which of a workflow's dependencies this install can actually resolve.
 *
 * ComfyUI's workflow document names the weights it loads but says nothing about
 * whether they are on disk, and `/object_info` is the only place the *installed*
 * files are enumerated. Comparing the two is what closes the gap between
 * "this graph supports image_to_3d" and "this install can run it": a workflow whose
 * checkpoint is missing proves the capability and cannot serve it, and advertising
 * it would send work into a guaranteed failure.
 *
 * The comparison is deliberately one-directional and tolerant. A file the graph
 * names but the enumerations do not list may be reachable through a node pack the
 * hub does not read, or listed by a loader whose field name this does not
 * recognise, so a "missing" verdict is reported as a *reason* rather than as a hard
 * fact — an operator can override it by pinning the workflow.
 *
 * @param workflow - the parsed workflow.
 * @param io - the node index.
 * @param installedFiles - every weight filename `/object_info` enumerated.
 * @returns the dependencies that could not be resolved.
 */
export function missingWorkflowModelFiles(workflow, io, installedFiles) {
    const installed = new Set(installedFiles);
    return workflowModelFiles(workflow, io).filter((file) => !installed.has(file));
}
/**
 * Map one discovered workflow into a catalog descriptor.
 *
 * The descriptor's id is derived from the workflow's own name, so it is stable
 * across restarts and cannot collide with a weight-file model on the same host.
 * The workflow's identity is preserved in three separate places, because
 * collapsing them is how a router ends up unable to tell a workflow from the
 * checkpoint it loads:
 *
 * - `name` / `adapterConfig.workflow` — the **workflow**;
 * - `adapterConfig.discovery.workflowName` — the name ComfyUI stores it under;
 * - `adapterConfig.discovery.modelFiles` and `.nodePackages` — the weights and
 *   node packs it actually depends on.
 *
 * A descriptor is produced even for a workflow whose capabilities could not be
 * proven, and it is left **disabled**: an operator then sees exactly what was
 * found and can describe it in configuration, whereas dropping it silently would
 * hide a workflow that is genuinely installed.
 *
 * @param workflow - the parsed workflow.
 * @param host - the host it belongs to.
 * @param inference - the capability inference.
 * @param options - the matching override, the model files the graph names, and the
 *   files this install actually reports.
 * @returns the descriptor and its provenance.
 */
export function mapComfyWorkflow(workflow, host, inference, options = {}) {
    const override = options.override;
    const missing = options.missingModelFiles ?? [];
    const proven = [...inference.capabilities];
    const narrowed = override?.capabilities === undefined ? proven : proven.filter((capability) => override.capabilities?.includes(capability));
    const display = override?.name ?? displayNameOf(workflow.name);
    // `slugifyModelId` collapses the workflow's own name into the catalogue's id
    // grammar; the digest is only needed for a name with nothing Latin in it.
    const derived = slugifyModelId(display, host.runtime.engine || 'comfyui', `workflow-${stableDigest(workflow.name)}`);
    const id = override?.id ?? derived;
    const { inputTypes, outputTypes } = ioForCapabilities(narrowed.length === 0 ? ['text_to_image'] : narrowed);
    const vramGb = override?.vramGb ?? estimateComfyWorkflowVram(workflow, inference);
    const ramGb = override?.ramGb ?? vramGb;
    const notes = describeWorkflow(workflow, inference, narrowed, vramGb, override, missing);
    const resources = {
        vramGb,
        ramGb,
        requiresGpu: override?.requiresGpu ?? true,
    };
    // Publication rules, in order of specificity:
    //
    // 1. an override stating `enabled` always wins — it is the operator's decision;
    // 2. a workflow whose dependencies are missing is published **disabled**, because
    //    it would fail on the engine and the honest answer is "installed, not
    //    runnable";
    // 3. a workflow whose capability could not be proven is published **disabled**,
    //    because the router must never send work into a refusal.
    const enabled = override?.enabled ?? (narrowed.length > 0 && missing.length === 0);
    const descriptor = {
        id,
        name: display,
        type: workflowModelType(narrowed),
        host: host.id,
        capabilities: narrowed,
        // Explicit rather than defaulted: a chained request's compatibility check
        // reads these, and a workflow is the one model whose kinds are not implied by
        // its capabilities alone.
        inputTypes: narrowed.length === 0 ? [] : inputTypes,
        outputTypes: narrowed.length === 0 ? [] : outputTypes,
        adapterConfig: {
            // A pinned `workflowPath` replaces the discovered graph as the thing that
            // actually runs. The two are mutually exclusive in the adapter, and the
            // operator's explicit file is the more specific statement.
            ...(override?.workflowPath === undefined
                ? { workflow: workflow.graph }
                : { workflowPath: override.workflowPath }),
            discovery: {
                workflowName: workflow.name,
                workflowFormat: workflow.format,
                ...(inference.generatorClasses.length === 0 ? {} : { generatorNodes: [...inference.generatorClasses] }),
                ...(inference.inputClasses.length === 0 ? {} : { inputNodes: [...inference.inputClasses] }),
                ...(options.modelFiles === undefined || options.modelFiles.length === 0
                    ? {}
                    : { modelFiles: [...options.modelFiles] }),
                ...(missing.length === 0 ? {} : { missingModelFiles: [...missing] }),
                ...(inference.evidence.length === 0 ? {} : { evidence: [...inference.evidence] }),
                ...(inference.notes.length === 0 ? {} : { unproven: [...inference.notes] }),
                ...(workflow.conversionNote === undefined ? {} : { conversionNote: workflow.conversionNote }),
                ...(override === undefined ? {} : { configured: true }),
            },
        },
        resources,
        priority: override?.priority ?? DISCOVERED_PRIORITY,
        tags: unique([...WORKFLOW_TAGS, ...(override?.tags ?? [])]),
        enabled,
        notes,
    };
    return { descriptor, inference, applied: override !== undefined, missingModelFiles: missing };
}
/**
 * A provenance note assembled from what the graph proved.
 * @param workflow - the parsed workflow.
 * @param inference - the capability inference.
 * @param capabilities - the capabilities actually published.
 * @param vramGb - the VRAM estimate applied.
 * @param override - the override that applied, when one did.
 * @param missing - the dependencies this install could not resolve.
 * @returns the note.
 */
function describeWorkflow(workflow, inference, capabilities, vramGb, override, missing = []) {
    const facts = [
        `Discovered from ComfyUI's saved workflows (${workflow.name}); not listed in models.json.`,
    ];
    if (inference.evidence.length > 0)
        facts.push(`Proven from the graph — ${inference.evidence.join('; ')}.`);
    if (missing.length > 0) {
        // The single most actionable line for a 3D workflow: the capability is real
        // and the download is what is absent.
        facts.push(`Published disabled: the graph loads ${missing.join(', ')}, which this ComfyUI does not report as installed. ` +
            'Install those weights, or pin the workflow in configuration if they live somewhere the hub cannot see.');
    }
    if (capabilities.length === 0) {
        facts.push('No capability could be proven from this graph, so it is published disabled. ' +
            (inference.notes.length > 0 ? `Because ${inference.notes.join('; ')}. ` : '') +
            'Describe it in configuration (adapterConfig.workflows) to declare what it does.');
    }
    else if (inference.notes.length > 0) {
        facts.push(`Not claimed: ${inference.notes.join('; ')}.`);
    }
    if (override !== undefined) {
        facts.push(`Configuration pins this workflow${override.capabilities === undefined ? '' : ` and narrows its capabilities to ${override.capabilities.join(', ') || 'none'}`}.`);
    }
    facts.push(`VRAM estimate ${vramGb} GiB, inferred from the graph's node classes; treat as approximate.`);
    if (workflow.conversionNote !== undefined)
        facts.push(workflow.conversionNote);
    if (override?.notes !== undefined)
        facts.push(override.notes);
    return facts.join(' ');
}
/**
 * The weight files a workflow loads, read from its own loader nodes.
 *
 * This is the *underlying model* half of the identity the router must not
 * confuse with the workflow: `3d_pixal3d_trellis2_image_to_model` is a workflow,
 * and the UNET it loads is a different thing entirely. Reporting both lets an
 * operator see which download is missing when a workflow will not run.
 *
 * @param workflow - the parsed workflow.
 * @param io - the node index.
 * @returns the filenames it names, deduplicated.
 */
export function workflowModelFiles(workflow, io) {
    const files = [];
    for (const node of Object.values(workflow.nodes)) {
        const index = io[node.classType];
        if (index === undefined)
            continue;
        for (const field of Object.keys(index.inputs)) {
            if (!isFileField(node.classType, field))
                continue;
            const value = node.inputs[field];
            if (typeof value === 'string' && value.trim().length > 0)
                files.push(value);
        }
    }
    return unique(files);
}
