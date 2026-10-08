/**
 * ComfyUI workflow analysis: from a saved graph to a runnable public contract.
 *
 * A workflow an operator saved in ComfyUI is a complete executable unit, but it
 * carries no statement of what a caller may set or what it produces. This module
 * derives that statement from the graph **and from what ComfyUI says about the
 * nodes in it** — never from a filename, and never from a hardcoded list of
 * workflows.
 *
 * ## What ComfyUI is asked, and why each answer matters
 *
 * | Question | Source | Used for |
 * | --- | --- | --- |
 * | Which classes exist? | `/object_info` keys | conversion, validation |
 * | What does a node take? | `input.required`, `input.optional` | which fields are public |
 * | What is required? | `input.required` names | judging a conversion |
 * | What does a node return? | `output` type names | what a deliverable *is* |
 * | Is this a terminal node? | `output_node` | which nodes produce artifacts |
 *
 * `output_node` is the load-bearing one. It is ComfyUI's own declaration that a
 * node is a terminal output, which is a far better answer to "what does this
 * workflow produce" than pattern-matching a class name. It is necessary but not
 * sufficient — `PreviewImage` and `Preview3DAdvanced` are output nodes too, and a
 * preview writes to ComfyUI's temp directory rather than to something a caller
 * can keep — so it is combined with the declared output *types* and the preview
 * convention instead of being trusted alone.
 *
 * ## The two rules this module exists to obey
 *
 * 1. **A typed output is not automatically an artifact.** Only type families that
 *    denote a file a caller can keep count: `IMAGE`, `AUDIO`, `VIDEO`, and the
 *    `FILE_3D*` / mesh families. `INT`, `LOAD3D_MODEL_INFO`, `LOAD3D_CAMERA` and
 *    `MASK` are execution detail, and a node that merely passes a `MESH` along is
 *    not a deliverable.
 * 2. **When the metadata cannot support a mapping, say so.** Every refusal
 *    carries a diagnostic naming the specific gap, so an operator can act on it
 *    rather than wondering why a workflow is quietly missing.
 *
 * ## Inputs, generally
 *
 * Four kinds of public input are recognized, each from evidence rather than from
 * a name:
 *
 * - **text** — a node that reaches a sampler's `positive`/`negative` and holds a
 *   literal `text`;
 * - **image** — a node whose class declares a file-valued `image` input;
 * - **controls** — a node that declares a `value` input of a scalar type *and*
 *   carries an author-set title, which is how an author marks a knob they intend
 *   a caller to turn;
 * - **sampler/size parameters** — seed, steps, CFG, width and height, exposed
 *   only when the graph has exactly one node that could carry them, because with
 *   several there is no evidence for which one governs the result.
 *
 * @module dsh-ai-model-hub/comfy/scan
 */

import type { Capability, IoType, ModelType } from '../catalog/capabilities.ts';
import { slugifyModelId, stableDigest } from '../discovery/http.ts';
import type { ComfyNodeIndex } from '../discovery/comfyui-workflow.ts';
import { ComfyWorkflowError, linkSource, parseComfyWorkflow } from '../discovery/comfyui-workflow.ts';
import type { ComfyGraph, InputBinding, OutputBinding, WorkflowContract } from './workflow.ts';
import { readContract, validateComfyGraph } from './workflow.ts';

/** How a workflow document was serialized. */
export type ScannedWorkflowFormat = 'api' | 'ui';

/**
 * How ready a scanned workflow is to run.
 *
 * `unreadable` is deliberately distinct from `invalid`. "The engine listed this
 * workflow but would not give me its contents" is an engine or transport problem
 * with an HTTP status attached; "I read the contents and they do not describe a
 * runnable workflow" is a property of the document. Collapsing the two — which an
 * earlier version did, reporting a 404 as "invalid JSON" — sends an operator
 * looking for a broken file that was never broken.
 */
export type ScannedWorkflowReadiness =
  | 'ready'
  | 'needs_conversion'
  | 'unreadable'
  | 'invalid'
  | 'unavailable';

/** The value kind a public input carries. */
export type PublicInputKind = 'text' | 'image' | 'number' | 'boolean';

/** One parameter a caller may set, and where it goes. */
export interface PublicInput {
  /** The name a caller uses, e.g. `prompt`, `image`, `texture_resolution`. */
  readonly name: string;
  /** An author-facing label: the node's own title when it has one. */
  readonly label: string;
  /** What the value is. */
  readonly kind: PublicInputKind;
  /** The node the value is written to. */
  readonly node: string;
  /** The input field on that node. */
  readonly input: string;
  /** Whether the workflow needs it, as far as the metadata shows. */
  readonly required: boolean;
}

/** One artifact a workflow produces. */
export interface PublicOutput {
  /** The name a caller receives it under. */
  readonly name: string;
  /** The artifact kind, from the node's declared output type. */
  readonly type: IoType;
  /** The node whose outputs are collected. */
  readonly node: string;
  /** The node class, for diagnostics. */
  readonly nodeClass: string;
  /** Whether this node only previews (a preview is a companion, not a deliverable). */
  readonly preview: boolean;
}

/** What analysing a graph produced. */
export interface WorkflowAnalysis {
  readonly inputs: readonly PublicInput[];
  readonly outputs: readonly PublicOutput[];
  readonly capabilities: readonly Capability[];
  /** Facts established from the metadata, for the descriptor's notes. */
  readonly evidence: readonly string[];
  /** Why something was not exposed, phrased for an operator. */
  readonly diagnostics: readonly string[];
}

/** What {@link scanWorkflowDocument} needs to read one document. */
export interface ScanWorkflowInput {
  /** The parsed JSON document. */
  readonly raw: unknown;
  /** The name it was found under, used in messages and as a display fallback. */
  readonly name: string;
  /** Every node class's declared IO, when the engine was reachable. */
  readonly io?: ComfyNodeIndex;
  /** Every node class the engine has installed, when it was reachable. */
  readonly classes?: ReadonlySet<string>;
}

/** One scanned workflow, with the public surface it exposes. */
export interface ScannedWorkflow {
  readonly format: ScannedWorkflowFormat;
  readonly readiness: ScannedWorkflowReadiness;
  /** Whether the hub can queue it as it stands. */
  readonly runnable: boolean;
  /** A sentence explaining {@link readiness}, safe to show a user. */
  readonly detail: string;
  readonly capabilities: readonly Capability[];
  /** One row per public parameter, for display. */
  readonly inputs: readonly PublicInput[];
  /** One row per produced artifact, for display. */
  readonly outputs: readonly PublicOutput[];
  /** Why something could not be exposed. Empty when nothing was withheld. */
  readonly diagnostics: readonly string[];
  /** The inferred contract, present exactly when the workflow is runnable. */
  readonly contract?: WorkflowContract;
  /** The API-format graph to queue, present exactly when the workflow is runnable. */
  readonly graph?: ComfyGraph;
  /**
   * A digest of the document the analysis was read from.
   *
   * Two scans that produce the same digest produced the same analysis, which is
   * what lets a caller reuse a cached registration instead of rebuilding it, and
   * what tells it the analysis is stale once the workflow's contents change.
   */
  readonly digest: string;
}

/**
 * Output type families that denote something a caller can keep, read from a class
 * name when the engine's own metadata is unavailable.
 *
 * This is the *fallback*, used only when `/object_info` could not be read — an
 * engine that is down, or a class the index does not describe. ComfyUI's declared
 * `output` types are the primary evidence and are always preferred, because a
 * class name is a naming convention while an output type is a contract. The note
 * this produces is reported as a diagnostic so a fallback analysis is never
 * mistaken for a verified one.
 */
const CLASS_DELIVERABLE_TYPES: readonly { readonly pattern: RegExp; readonly type: IoType }[] = [
  { pattern: /save.*(glb|gltf|obj|stl|ply|fbx|usdz|3d|mesh|splat)|mesh.*to.*file|export.*(3d|mesh|glb)|(glb|obj|stl).*export/i, type: 'model_3d' },
  { pattern: /save.*(video|webm|animated|mp4)|video.*save/i, type: 'video' },
  { pattern: /save.*(audio|wav|flac|mp3)|audio.*save/i, type: 'audio' },
  { pattern: /save.*image|image.*save/i, type: 'image' },
];

/** Output type families that denote something a caller can keep. */
const DELIVERABLE_TYPES: readonly { readonly pattern: RegExp; readonly type: IoType }[] = [
  { pattern: /^IMAGE$/i, type: 'image' },
  { pattern: /^AUDIO$/i, type: 'audio' },
  { pattern: /^VIDEO$/i, type: 'video' },
  { pattern: /^(FILE_?3D|MESH|VOXEL|GAUSSIAN|SPLAT|3D)/i, type: 'model_3d' },
];

/** Node classes whose name says they persist a file, used only as a fallback. */
const SAVE_CLASS_PATTERN = /save|export|write/i;

/** Node classes whose name says they only preview, which writes to the temp directory. */
const PREVIEW_CLASS_PATTERN = /preview/i;

/** Node classes that load an image from disk. */
const IMAGE_LOADER_PATTERN = /load.*image|imageload|image.*load|^loadimage/i;

/** Scalar types a control node may carry, mapped to the value kind a caller sets. */
const CONTROL_TYPE_KINDS: readonly { readonly pattern: RegExp; readonly kind: PublicInputKind }[] = [
  { pattern: /^BOOLEAN$/i, kind: 'boolean' },
  { pattern: /^(INT|FLOAT|NUMBER)$/i, kind: 'number' },
  { pattern: /^STRING$/i, kind: 'text' },
];

/**
 * Whether a value is a plain JSON object.
 * @param value - candidate value.
 * @returns true for a non-null, non-array object.
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** A node in the graph, reduced to what the analysis reads. */
interface GraphNode {
  readonly id: string;
  readonly classType: string;
  readonly inputs: Readonly<Record<string, unknown>>;
  /** The author's label for the node, when the export carried one. */
  readonly title?: string;
}

/**
 * Every node of an API graph, in id order, with whatever titles the export kept.
 *
 * Titles live outside the graph in a UI export, so they are passed in separately
 * and merged here — they are how an author marks a control as intended for a
 * caller, and losing them would mean losing the only evidence for that.
 *
 * @param graph - the API-format graph.
 * @param titles - node id → author title.
 * @returns the nodes.
 */
function graphNodes(graph: ComfyGraph, titles: ReadonlyMap<string, string>): GraphNode[] {
  const nodes: GraphNode[] = [];
  for (const [id, node] of Object.entries(graph)) {
    if (!isObject(node)) continue;
    const classType = node['class_type'];
    const inputs = node['inputs'];
    if (typeof classType !== 'string' || !isObject(inputs)) continue;
    const title = titles.get(id);
    nodes.push({ id, classType, inputs, ...(title === undefined ? {} : { title }) });
  }
  return nodes;
}

/**
 * The first field of a node that carries a literal (non-link) value.
 *
 * A link is an array or a bare node-id string, so only a scalar counts as a
 * settable widget. That distinction is what keeps "this node has a width" from
 * being confused with "this node is wired to something that has a width".
 *
 * @param node - the node.
 * @param fields - candidate field names.
 * @returns the field name, or `undefined`.
 */
function literalField(node: GraphNode, fields: readonly string[]): string | undefined {
  for (const field of fields) {
    const value = node.inputs[field];
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return field;
  }
  return undefined;
}

/**
 * The public name for an author-labelled control.
 *
 * Usually the label, slugified. The exception is a text control the author named
 * like a user prompt: a workflow whose prompt arrives through a `PrimitiveString`
 * called "Text String (User Prompt)" would otherwise expose it under that name
 * and advertise no capability, because routing and invocation both speak in terms
 * of a `prompt`. Naming it `prompt` is what makes "generate a cat" reach it, and
 * the label is still shown as the human-facing description.
 *
 * Only the *user* prompt is renamed: a sibling "System Prompt" control keeps its
 * own name, so the two cannot be confused for one another.
 *
 * @param title - the author's label.
 * @param fallback - used when the label has nothing usable.
 * @param kind - the value kind the control carries.
 * @param existing - the inputs established so far.
 * @returns the parameter name.
 */
function controlName(
  title: string | undefined,
  fallback: string,
  kind: PublicInputKind,
  existing: readonly PublicInput[],
): string {
  const slug = parameterName(title, fallback);
  if (kind !== 'text') return slug;
  if (existing.some((input) => input.name === 'prompt')) return slug;
  const looksLikePrompt = title !== undefined && /(^|[^a-z])user[ _-]?prompt([^a-z]|$)|^prompt$/i.test(title.trim());
  return looksLikePrompt ? 'prompt' : slug;
}

/**
 * A stable parameter name derived from a node's title.
 *
 * The title is what the author called the control, so it is the best available
 * name — but it is free text with spaces and punctuation, and a binding name
 * reaches `options`, JSON keys and tool arguments. Slugifying keeps the meaning
 * while producing something safe to type.
 *
 * @param title - the author's label.
 * @param fallback - used when the title has nothing usable.
 * @returns a lowercase parameter name.
 */
function parameterName(title: string | undefined, fallback: string): string {
  if (title === undefined) return fallback;
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return slug.length === 0 ? fallback : slug;
}

/**
 * The value kind a control node carries, from its declared output type.
 * @param node - the control node.
 * @param io - the node index.
 * @returns the kind, or `undefined` when the node is not a scalar control.
 */
function controlKind(node: GraphNode, io: ComfyNodeIndex | undefined): PublicInputKind | undefined {
  const spec = io?.[node.classType];
  if (spec === undefined) return undefined;
  for (const outputType of spec.outputs) {
    for (const rule of CONTROL_TYPE_KINDS) {
      if (rule.pattern.test(outputType.trim())) return rule.kind;
    }
  }
  return undefined;
}

/**
 * The artifact kind a node produces, from its own declared output types.
 *
 * Only deliverable type families count, which is what stops an intermediate
 * `MESH` port or a `LOAD3D_MODEL_INFO` handle from being published as an artifact.
 *
 * @param node - the candidate output node.
 * @param io - the node index.
 * @returns the artifact kind, or `undefined` when the node returns no deliverable.
 */
function deliverableType(node: GraphNode, io: ComfyNodeIndex | undefined): IoType | undefined {
  const spec = io?.[node.classType];
  if (spec !== undefined) {
    for (const outputType of spec.outputs) {
      for (const rule of DELIVERABLE_TYPES) {
        if (rule.pattern.test(outputType.trim())) return rule.type;
      }
    }
    // The engine described this class and none of its outputs is a deliverable,
    // so the answer is no — a class name does not get to overrule its contract.
    return undefined;
  }
  for (const rule of CLASS_DELIVERABLE_TYPES) {
    if (rule.pattern.test(node.classType)) return rule.type;
  }
  return undefined;
}

/**
 * The nodes whose outputs a caller receives.
 *
 * ComfyUI's `output_node` flag is the primary evidence. Previews are held back
 * when anything else is available, because a preview writes to ComfyUI's temp
 * directory — it is a companion render, not the thing that was asked for. Only
 * when a class-based writer pattern is all the metadata offers (a custom node that
 * saves a file without flagging itself) does the name convention decide.
 *
 * @param nodes - every node.
 * @param io - the node index.
 * @returns the chosen output nodes, in graph order.
 */
function outputNodes(nodes: readonly GraphNode[], io: ComfyNodeIndex | undefined): GraphNode[] {
  const isPreview = (node: GraphNode): boolean => PREVIEW_CLASS_PATTERN.test(node.classType);
  const flagged = nodes.filter((node) => io?.[node.classType]?.outputNode === true);
  const declared = flagged.filter((node) => !isPreview(node) && deliverableType(node, io) !== undefined);
  if (declared.length > 0) return declared;

  const savers = nodes.filter(
    (node) => !isPreview(node) && SAVE_CLASS_PATTERN.test(node.classType) && deliverableType(node, io) !== undefined,
  );
  if (savers.length > 0) return savers;

  // Nothing but previews: publish them, because a workflow whose only output is a
  // render still produces something, and the alternative is an unexplained refusal.
  return flagged.filter((node) => deliverableType(node, io) !== undefined);
}

/**
 * The public inputs a graph's own wiring and metadata support.
 *
 * @param nodes - every node.
 * @param io - the node index.
 * @returns the inputs, their evidence, and what was withheld.
 */
function collectInputs(
  nodes: readonly GraphNode[],
  io: ComfyNodeIndex | undefined,
): { inputs: PublicInput[]; evidence: string[]; diagnostics: string[] } {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const inputs: PublicInput[] = [];
  const evidence: string[] = [];
  const diagnostics: string[] = [];
  const claimed = new Set<string>();

  const add = (input: PublicInput): void => {
    if (claimed.has(`${input.node}:${input.input}`)) return;
    claimed.add(`${input.node}:${input.input}`);
    inputs.push(input);
  };

  // ── image inputs ──────────────────────────────────────────────────────────
  const imageLoaders = nodes.filter(
    (node) =>
      typeof node.inputs['image'] === 'string' &&
      (IMAGE_LOADER_PATTERN.test(node.classType) || /load|input|read|open/i.test(node.classType)),
  );
  imageLoaders.forEach((node, index) => {
    add({
      name: index === 0 ? 'image' : `image_${index + 1}`,
      label: node.title ?? `Image (${node.classType})`,
      kind: 'image',
      node: node.id,
      input: 'image',
      required: true,
    });
  });
  if (imageLoaders.length > 0) {
    evidence.push(`image input: ${[...new Set(imageLoaders.map((node) => node.classType))].join(', ')}`);
  }

  // ── text prompts ──────────────────────────────────────────────────────────
  const samplers = nodes.filter(
    (node) => literalField(node, ['seed', 'noise_seed']) !== undefined && literalField(node, ['steps']) !== undefined,
  );
  const textNode = (id: string | undefined): GraphNode | undefined => {
    const node = id === undefined ? undefined : byId.get(id);
    return node !== undefined && typeof node.inputs['text'] === 'string' ? node : undefined;
  };
  const positive = samplers
    .map((sampler) => textNode(linkSource(sampler.inputs['positive'])))
    .find((node) => node !== undefined);
  const negative = samplers
    .map((sampler) => textNode(linkSource(sampler.inputs['negative'])))
    .find((node) => node !== undefined);
  if (positive !== undefined) {
    add({ name: 'prompt', label: 'Prompt', kind: 'text', node: positive.id, input: 'text', required: true });
    evidence.push(`prompt node: ${positive.classType}`);
  }
  if (negative !== undefined) {
    add({
      name: 'negative_prompt',
      label: 'Negative prompt',
      kind: 'text',
      node: negative.id,
      input: 'text',
      required: false,
    });
  }

  // ── author-marked controls ────────────────────────────────────────────────
  // A titled node with a scalar `value` input is how an author says "this is a
  // knob"; an untitled one is almost always an internal constant, and exposing
  // every one of those would bury the real inputs in noise.
  for (const node of nodes) {
    if (!Object.hasOwn(node.inputs, 'value')) continue;
    if (claimed.has(`${node.id}:value`)) continue;
    const kind = controlKind(node, io);
    if (kind === undefined) continue;
    if (node.title === undefined || node.title.trim().length === 0) {
      diagnostics.push(
        `${node.classType} #${node.id} is an unlabelled ${kind} control, so it was not exposed; give it a title in ` +
          'ComfyUI to publish it as an input.',
      );
      continue;
    }
    const name = controlName(node.title, `value_${node.id}`, kind, inputs);
    if (inputs.some((input) => input.name === name)) continue;
    add({ name, label: node.title, kind, node: node.id, input: 'value', required: false });
    evidence.push(`control "${node.title}" (${node.classType})`);
  }

  // ── sampler and size parameters ───────────────────────────────────────────
  // Only exposed when exactly one node could carry each, because with several
  // there is no evidence for which one governs the result, and binding a guess
  // would silently change what the caller gets.
  if (samplers.length === 1) {
    const sampler = samplers[0] as GraphNode;
    const seedField = literalField(sampler, ['seed', 'noise_seed']);
    if (seedField !== undefined) {
      add({ name: 'seed', label: 'Seed', kind: 'number', node: sampler.id, input: seedField, required: false });
    }
    for (const field of ['steps', 'cfg', 'denoise'] as const) {
      if (literalField(sampler, [field]) === undefined) continue;
      add({
        name: field,
        label: field === 'cfg' ? 'CFG' : field,
        kind: 'number',
        node: sampler.id,
        input: field,
        required: false,
      });
    }
    evidence.push(`sampler: ${sampler.classType}`);
  } else if (samplers.length > 1) {
    diagnostics.push(
      `${samplers.length} sampler nodes are present, so seed, steps and CFG were not exposed: which sampler governs ` +
        'the result cannot be established from the graph.',
    );
  }

  const sizeNodes = nodes.filter(
    (node) =>
      literalField(node, ['width']) !== undefined &&
      literalField(node, ['height']) !== undefined &&
      !claimed.has(`${node.id}:width`),
  );
  if (sizeNodes.length === 1) {
    const size = sizeNodes[0] as GraphNode;
    const width = literalField(size, ['width']);
    const height = literalField(size, ['height']);
    if (width !== undefined) {
      add({ name: 'width', label: 'Width', kind: 'number', node: size.id, input: width, required: false });
    }
    if (height !== undefined) {
      add({ name: 'height', label: 'Height', kind: 'number', node: size.id, input: height, required: false });
    }
  } else if (sizeNodes.length > 1) {
    diagnostics.push(
      `${sizeNodes.length} nodes declare a width and height, so dimensions were not exposed; which one governs the ` +
        'result cannot be established from the graph.',
    );
  }

  return { inputs, evidence, diagnostics };
}

/**
 * The capabilities an analysed graph proves.
 *
 * Derived from what the workflow *consumes* and what it *produces*, in the
 * catalog's own input/output vocabulary, rather than from any class name. A
 * capability is claimed only when both halves are present.
 *
 * @param inputs - the public inputs that were established.
 * @param outputs - the artifacts that were established.
 * @returns the capabilities, in vocabulary order.
 */
function capabilitiesFor(inputs: readonly PublicInput[], outputs: readonly PublicOutput[]): Capability[] {
  const hasText = inputs.some((input) => input.kind === 'text' && input.name === 'prompt');
  const hasImage = inputs.some((input) => input.kind === 'image');
  const produced = new Set(outputs.map((output) => output.type));
  const capabilities: Capability[] = [];

  if (produced.has('image')) {
    if (hasText) capabilities.push('text_to_image');
    if (hasImage) capabilities.push('image_to_image');
  }
  if (produced.has('model_3d')) {
    if (hasImage) capabilities.push('image_to_3d');
    if (hasText) capabilities.push('text_to_3d');
  }
  if (produced.has('audio') && hasText) capabilities.push('audio_generation');
  if (produced.has('video')) capabilities.push('video_generation');
  return capabilities;
}

/**
 * The model `type` a scanned workflow's capabilities imply.
 *
 * @param capabilities - the proven capabilities.
 * @returns the model type.
 */
export function workflowModelTypeFor(capabilities: readonly Capability[]): ModelType {
  if (capabilities.includes('text_to_3d') || capabilities.includes('image_to_3d')) return 'three_d_generation';
  if (capabilities.includes('video_generation')) return 'video_generation';
  if (capabilities.includes('audio_generation')) return 'audio_generation';
  if (capabilities.includes('image_to_image')) return 'image_editing';
  return 'image_generation';
}

/**
 * Whether a document is a ComfyUI editor export rather than an API graph.
 *
 * @param raw - the parsed document.
 * @returns true when it has the editor's node array.
 */
export function looksLikeEditorWorkflow(raw: unknown): boolean {
  if (!isObject(raw)) return false;
  if (Array.isArray(raw['nodes'])) return true;
  const envelope = raw['prompt'];
  return isObject(envelope) && Array.isArray(envelope['nodes']);
}

/**
 * The author-set titles of a UI export's nodes, by node id.
 *
 * A title is the only signal that separates a control an author meant a caller to
 * use from an internal constant, so it is read out of the export before the graph
 * is converted and carried into the analysis.
 *
 * @param raw - the raw document.
 * @returns node id → title.
 */
export function editorNodeTitles(raw: unknown): ReadonlyMap<string, string> {
  const titles = new Map<string, string>();
  if (!isObject(raw)) return titles;
  const document = isObject(raw['prompt']) && !Array.isArray(raw['nodes']) ? raw['prompt'] : raw;
  const nodes = Array.isArray(document['nodes']) ? document['nodes'] : [];
  for (const entry of nodes) {
    if (!isObject(entry)) continue;
    const id = entry['id'];
    const title = entry['title'];
    if ((typeof id === 'string' || typeof id === 'number') && typeof title === 'string' && title.trim().length > 0) {
      titles.set(String(id), title.trim());
    }
  }
  return titles;
}

/**
 * Every required input a converted graph failed to fill in.
 *
 * This is the check that makes "converted" an honest claim. Reconstructing an API
 * graph from an editor export depends on knowing which widget value maps to which
 * schema field, and for a custom node whose widget order differs from its schema
 * that reconstruction can silently miss a field. ComfyUI's own `/object_info`
 * says which fields are required, so an unfilled one is detectable rather than a
 * surprise at execution time.
 *
 * @param graph - the converted graph.
 * @param io - the node index, when known.
 * @param classes - the installed node classes, when known.
 * @returns one message per gap; empty when the conversion looks complete.
 */
export function conversionGaps(
  graph: ComfyGraph,
  io: ComfyNodeIndex | undefined,
  classes: ReadonlySet<string> | undefined,
): string[] {
  if (io === undefined) return [];
  const gaps: string[] = [];
  for (const [id, node] of Object.entries(graph)) {
    const spec = io[node.class_type];
    if (spec === undefined) {
      if (classes !== undefined && !classes.has(node.class_type)) {
        gaps.push(`node ${id} (${node.class_type}) is not installed on this ComfyUI`);
      }
      continue;
    }
    for (const field of spec.requiredInputs ?? []) {
      if (!Object.hasOwn(node.inputs, field)) {
        gaps.push(`node ${id} (${node.class_type}) has no value for required input "${field}"`);
      }
    }
  }
  return gaps;
}

/**
 * Scan one workflow document into either a runnable provider or a reported gap.
 *
 * Never throws: a document the engine would not hand over, an export that cannot
 * be reconstructed, and a graph with no usable output all come back as a result
 * with the reason attached.
 *
 * @param input - the document, its name, and whatever the engine revealed.
 * @returns the scan result.
 */
export function scanWorkflowDocument(input: ScanWorkflowInput): ScannedWorkflow {
  const { raw, name, io, classes } = input;
  const editorFormat = looksLikeEditorWorkflow(raw);
  const digest = stableDigest(JSON.stringify(raw ?? null));

  let parsed;
  try {
    parsed = parseComfyWorkflow(raw, name, io ?? {});
  } catch (error) {
    const reason = error instanceof ComfyWorkflowError ? error.message : String(error);
    return {
      format: editorFormat ? 'ui' : 'api',
      readiness: editorFormat ? 'needs_conversion' : 'invalid',
      runnable: false,
      digest,
      detail: editorFormat
        ? `editor (UI) format that could not be converted to an API graph: ${reason}`
        : `not a readable API workflow: ${reason}`,
      capabilities: [],
      inputs: [],
      outputs: [],
      diagnostics: [reason],
    };
  }

  // The executable graph is rebuilt from the parsed nodes rather than taken from
  // the raw document, so any non-node key an export carries (a name, an editor
  // version, a viewport) cannot reach the graph ComfyUI is asked to queue.
  const graph: ComfyGraph = {};
  for (const [id, node] of Object.entries(parsed.nodes)) {
    graph[id] = { class_type: node.classType, inputs: { ...node.inputs } };
  }

  // Titles reach the analysis from two places, because the two serializations put
  // them in different ones: a UI export keeps them on the raw nodes, and a
  // conversion — subgraph flattening included, which synthesizes a label for every
  // promoted input — keeps them on the parsed nodes. Neither alone is complete.
  const titles = new Map(editorNodeTitles(raw));
  for (const [id, node] of Object.entries(parsed.nodes)) {
    if (node.title !== undefined) titles.set(id, node.title);
  }
  const nodes = graphNodes(graph, titles);
  const diagnostics: string[] = [];

  // A conversion is only trusted as far as the metadata can confirm it. The
  // analysis still runs — a workflow whose reconstruction is incomplete is
  // exactly the one whose inputs an operator needs to see — but it is never
  // published as runnable.
  let conversionTrusted = true;
  if (parsed.format === 'ui') {
    const gaps = conversionGaps(graph, io, classes);
    if (gaps.length > 0) {
      conversionTrusted = false;
      diagnostics.push(...gaps);
    }
  }

  const collected = collectInputs(nodes, io);
  const inputs = collected.inputs;
  diagnostics.push(...collected.diagnostics);
  if (io === undefined) {
    diagnostics.push(
      'ComfyUI\'s node definitions were not available, so this analysis was inferred from the graph and from node ' +
        'class names. Start ComfyUI and scan again to verify it against the engine\'s own metadata.',
    );
  }

  const artifactNodes = outputNodes(nodes, io);
  const outputs: PublicOutput[] = [];
  const usedNames = new Map<IoType, number>();
  for (const node of artifactNodes) {
    const type = deliverableType(node, io);
    if (type === undefined) continue;
    const seen = usedNames.get(type) ?? 0;
    usedNames.set(type, seen + 1);
    outputs.push({
      name: seen === 0 ? type : `${type}_${seen + 1}`,
      type,
      node: node.id,
      nodeClass: node.classType,
      preview: PREVIEW_CLASS_PATTERN.test(node.classType),
    });
  }

  const capabilities = capabilitiesFor(inputs, outputs);

  if (capabilities.length === 0) {
    diagnostics.push(
      outputs.length === 0
        ? 'no node in this graph declares a deliverable output type (image, audio, video or a 3D file)'
        : 'the graph produces an artifact but consumes no prompt or image the hub can supply',
    );
    return {
      format: parsed.format,
      readiness: 'invalid',
      runnable: false,
      digest,
      detail:
        outputs.length === 0
          ? 'no deliverable output: the graph writes nothing a caller could keep (a preview writes to ComfyUI\'s ' +
            'temp directory and is not returned)'
          : 'no prompt or image input reaches the deliverable, so no capability is proven',
      capabilities: [],
      inputs,
      outputs,
      diagnostics,
    };
  }

  const bindings: Record<string, InputBinding> = {};
  const inputKinds: Record<string, PublicInputKind> = {};
  for (const input of inputs) {
    bindings[input.name] = { node: input.node, input: input.input };
    inputKinds[input.name] = input.kind;
  }
  const outputBindings: Record<string, OutputBinding> = {};
  for (const output of outputs) {
    outputBindings[output.name] = { node: output.node, type: output.type };
  }

  // The surface is reported, but the workflow is not called runnable: a
  // reconstruction that missed a required value could run and produce something
  // other than what the caller asked for, which is worse than a clear refusal.
  if (!conversionTrusted) {
    return {
      format: 'ui',
      readiness: 'needs_conversion',
      runnable: false,
      digest,
      detail:
        `editor (UI) format: converted, but ${diagnostics.length} value(s) could not be reconstructed, so running it ` +
        'could silently change the result. Re-export the workflow with "Save (API Format)" and scan again.',
      capabilities,
      inputs,
      outputs,
      diagnostics,
    };
  }

  let contract: WorkflowContract;
  try {
    contract = readContract({ bindings, outputs: outputBindings, inputKinds }, capabilities);
    validateComfyGraph(graph, contract, classes);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    diagnostics.push(reason);
    return {
      format: parsed.format,
      readiness: 'invalid',
      runnable: false,
      digest,
      detail: reason,
      capabilities,
      inputs,
      outputs,
      diagnostics,
    };
  }

  return {
    format: parsed.format,
    readiness: parsed.format === 'ui' ? 'needs_conversion' : 'ready',
    runnable: true,
    digest,
    detail:
      parsed.format === 'ui'
        ? 'editor (UI) format — every required input was reconstructed from ComfyUI\'s node definitions; export ' +
          'with "Save (API Format)" to remove the reconstruction step'
        : 'API format — ready to run as saved',
    capabilities,
    inputs,
    outputs,
    diagnostics,
    contract,
    graph,
  };
}

/**
 * A stable catalog id for a scanned workflow.
 *
 * The id is derived from the source location rather than a counter, so the same
 * workflow found again on a later scan resolves to the same entry and a refresh
 * *updates* it instead of adding a second copy.
 *
 * @param name - the workflow's display name.
 * @param source - where it was found, which is what actually identifies it.
 * @returns a lowercase kebab-case id.
 */
export function scannedWorkflowId(name: string, source: string): string {
  return slugifyModelId(name, 'comfy-wf', `wf-${stableDigest(source)}`);
}

/**
 * A stable id for one Ollama model.
 *
 * @param hostId - the host the model was read from.
 * @param name - the engine's own model tag.
 * @returns a lowercase kebab-case id.
 */
export function scannedOllamaModelId(hostId: string, name: string): string {
  return slugifyModelId(name, `ollama-${slugifyModelId(hostId, 'host')}`, `model-${stableDigest(`${hostId}/${name}`)}`);
}
