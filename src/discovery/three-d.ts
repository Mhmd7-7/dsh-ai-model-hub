/**
 * 3D-generation engine discovery.
 *
 * A 3D engine is not introspectable the way Ollama is: there is no `/api/tags`
 * that lists what a local image-to-3D server can generate, because the "models"
 * such a server has are its *loaded weights*, chosen when the process started. So
 * this discoverer answers a different, and more useful, question — **what can this
 * engine actually do right now?** — and it answers it from evidence:
 *
 * 1. **Does the engine exist and is it up?** A request to its API-description
 *    route. For a Gradio app that is `/gradio_api/config` (Gradio 5) or `/config`
 *    (Gradio 3/4), and the document is written only once the app has finished
 *    importing its model — which makes it a far better liveness signal than a TCP
 *    accept or a 200 on an HTML shell.
 * 2. **Which operations does it expose?** The named endpoints in that document.
 *    These are the engine's own words for what it does, so capability claims are
 *    derived from them by *shape* (`*_to_3d`, `image_to_model`, `extract_glb`, …)
 *    rather than from a hardcoded list of engine names or version numbers.
 * 3. **Which of the operator's declared models are present?** The host declares
 *    them, because only the operator knows which checkpoint the engine was
 *    launched with; a declared model is published, and an undeclared engine is
 *    still published once, using the capabilities the surface itself proves.
 * 4. **What does each model need?** Declared per model, with the engine's own
 *    figures preferred where it reports them.
 *
 * The one thing this file will **not** do is claim a capability the engine has no
 * route for. A hub that publishes `image_to_3d` because a catalog entry said so,
 * on an engine whose Gradio app only has a mesh *exporter*, turns a clean routing
 * refusal into a confusing invocation failure — the failure mode
 * `tests/discovery-comfyui.test.ts` already guards against for ComfyUI.
 *
 * @module dsh-ai-model-hub/discovery/three-d
 */

import type { Capability, IoType, ModelType } from '../catalog/capabilities.ts';
import { isCapability } from '../catalog/capabilities.ts';
import type { ModelDescriptor, ModelHost } from '../catalog/descriptor.ts';
import { isValidModelId } from '../catalog/descriptor.ts';
import { fetchJson, isRecordLike, readArray, readString, slugifyModelId } from './http.ts';
import { DISCOVERED_PRIORITY, ioForCapabilities } from './types.ts';
import type { HostDiscoverer } from './types.ts';

/**
 * The engine label this discoverer handles.
 *
 * `three_d` is the generic label; the aliases are the names the engines'
 * publishers actually use, so an operator who writes `"engine": "trellis"` gets
 * introspection rather than silence. Adding an engine that shares this
 * discoverer's shape needs no code change — only a name here if it is new.
 */
export const THREE_D_ENGINE = 'three_d';

/** Engine labels that reach this discoverer, beyond its primary one. */
export const THREE_D_ENGINE_ALIASES: readonly string[] = [
  '3d',
  'threed',
  'three-d',
  'trellis',
  'hunyuan3d',
  'hunyuan3d-2',
  'sf3d',
  'stable-fast-3d',
  'triposr',
  'instantmesh',
  'wonder3d',
  'zero123',
  'gradio3d',
];

/** Budget for one request, in milliseconds. */
const REQUEST_TIMEOUT_MS = 5_000;

/** The API-description routes a Gradio app may answer on, most modern first. */
const GRADIO_CONFIG_PATHS: readonly string[] = ['/gradio_api/config', '/config'];

/** Tags every discovered model carries, so routing can filter on provenance. */
const DISCOVERED_TAGS: readonly string[] = ['local', 'discovered', THREE_D_ENGINE];

/**
 * A capability implied by the shape of an endpoint name.
 *
 * The patterns are deliberately about *verbs and directions* — "image to 3D" —
 * and never about a model or a product. An engine that names its route
 * `image_to_model` and one that names it `image_to_3d` are the same capability.
 */
const ENDPOINT_CAPABILITY_PATTERNS: readonly { readonly pattern: RegExp; readonly capability: Capability }[] = [
  // `image_to_3d`, `image_to_model`, `img2mesh`, `image_to_mesh`, …
  { pattern: /(?:^|[^a-z])(?:image|img|photo|picture)[^a-z]*(?:to|2)[^a-z]*(?:3d|mesh|model|glb)/i, capability: 'image_to_3d' },
  { pattern: /(?:^|[^a-z])(?:3d|mesh|model)[^a-z]*(?:from|of)[^a-z]*(?:image|img|photo)/i, capability: 'image_to_3d' },
  // `text_to_3d`, `txt2mesh`, `text_to_model`, …
  { pattern: /(?:^|[^a-z])(?:text|txt|prompt)[^a-z]*(?:to|2)[^a-z]*(?:3d|mesh|model|glb)/i, capability: 'text_to_3d' },
];

/**
 * Endpoint names that *export* a mesh the engine already made.
 *
 * These prove a mesh can leave the engine, not that the engine can make one. They
 * are recognised so that the difference can be reported: an install whose only
 * 3D-shaped routes are exporters gets a warning naming exactly that, instead of
 * being silently advertised as a 3D generator.
 */
const EXPORT_ENDPOINT_PATTERN = /(?:extract|export|save|download|convert)[^a-z]*(?:glb|gltf|obj|stl|ply|mesh|3d)/i;

/** What the engine's API-description document revealed. */
export interface ThreeDSurface {
  /** The route the document was read from, e.g. `/gradio_api/config`. */
  readonly configPath: string;
  /** Every named endpoint the engine exposes, in document order. */
  readonly endpoints: readonly string[];
  /** The capabilities the endpoint names prove. */
  readonly capabilities: readonly Capability[];
  /** Endpoints that write a mesh the engine was handed, when any exist. */
  readonly exportOnly: readonly string[];
  /** The engine's own version string, when it reported one. */
  readonly version?: string;
}

/**
 * Read an engine's API surface out of a Gradio config document.
 *
 * Both Gradio document shapes are accepted: `named_endpoints` is the modern key,
 * and a `dependencies` array with `api_name` entries is what older versions
 * carry. Only the *names* are used — nothing else in the document is interpreted,
 * because everything else is Gradio's business and changes between versions.
 *
 * @param raw - the parsed config document.
 * @param configPath - the route it came from, carried into the result.
 * @returns the surface, or `undefined` when the document is not a Gradio config.
 */
export function parseGradioSurface(raw: unknown, configPath: string): ThreeDSurface | undefined {
  if (!isRecordLike(raw)) return undefined;
  const names = new Set<string>();

  const named = raw['named_endpoints'];
  if (isRecordLike(named)) {
    for (const key of Object.keys(named)) {
      const cleaned = key.replace(/^\//, '').trim();
      if (cleaned.length > 0) names.add(cleaned);
    }
  } else if (Array.isArray(named)) {
    for (const entry of named) {
      const name = readString(entry, 'api_name') ?? readString(entry, 'name');
      if (name !== undefined) names.add(name.replace(/^\//, '').trim());
    }
  }

  for (const dependency of readArray(raw, 'dependencies')) {
    const name = readString(dependency, 'api_name');
    if (name !== undefined && name.length > 0) names.add(name.replace(/^\//, '').trim());
  }

  const endpoints = [...names].filter((name) => name.length > 0);
  const capabilities: Capability[] = [];
  const exportOnly: string[] = [];
  const claimed = new Set<string>();

  for (const endpoint of endpoints) {
    const matched = ENDPOINT_CAPABILITY_PATTERNS.filter((entry) => entry.pattern.test(endpoint));
    if (matched.length > 0) {
      for (const entry of matched) {
        if (!claimed.has(entry.capability)) {
          claimed.add(entry.capability);
          capabilities.push(entry.capability);
        }
      }
      continue;
    }
    if (EXPORT_ENDPOINT_PATTERN.test(endpoint)) exportOnly.push(endpoint);
  }

  return {
    configPath,
    endpoints,
    capabilities,
    exportOnly,
    ...(readString(raw, 'version') === undefined ? {} : { version: readString(raw, 'version') as string }),
  };
}

/** One model entry a host declares in `adapterConfig.models`. */
export interface ThreeDDeclaredModel {
  /** The model id to publish. */
  readonly id: string;
  /** Human-facing name. */
  readonly name: string;
  /** Explicit capabilities, when the operator declared them. */
  readonly capabilities?: readonly Capability[];
  /** Where the weights live, recorded in the descriptor for provenance. */
  readonly weightsPath?: string;
  /** GPU memory the model needs, in gibibytes. */
  readonly vramGb?: number;
  /** System memory the model needs, in gibibytes. */
  readonly ramGb?: number;
  /** Whether a GPU is required rather than preferred. */
  readonly requiresGpu?: boolean;
  /** Lower is preferred; defaults to the discovered priority. */
  readonly priority?: number;
  /** Operator notes. */
  readonly notes?: string;
}

/** The result of reading a host's declared model list. */
export type ThreeDHostConfigResult =
  | { readonly ok: true; readonly models: readonly ThreeDDeclaredModel[] }
  | { readonly ok: false; readonly reason: string };

/**
 * Parse a host's declared 3D model list.
 *
 * The list is optional: an engine that never states which weights it loaded is
 * still useful, and is published once from what its API surface proves. When it
 * *is* present it is validated strictly, because a typo in a model id would
 * otherwise reach the catalog as a schema error and take the whole catalog down
 * rather than one entry.
 *
 * @param host - the configured host.
 * @returns the parsed models, or a reason the declaration is unusable.
 */
export function parseThreeDHostConfig(host: ModelHost): ThreeDHostConfigResult {
  const source = host.adapterConfig?.['models'];
  if (source === undefined) return { ok: true, models: [] };
  if (!Array.isArray(source)) {
    return { ok: false, reason: '`models` must be an array of model declarations' };
  }

  const models: ThreeDDeclaredModel[] = [];
  const seen = new Set<string>();
  for (const [index, entry] of source.entries()) {
    if (!isRecordLike(entry)) {
      return { ok: false, reason: `models[${index}] must be an object` };
    }
    const id = readString(entry, 'id');
    if (id === undefined) return { ok: false, reason: `models[${index}].id is required` };
    if (!isValidModelId(id)) {
      return { ok: false, reason: `models[${index}].id "${id}" is not lowercase kebab-case` };
    }
    if (seen.has(id)) return { ok: false, reason: `models[${index}].id "${id}" is declared twice` };
    seen.add(id);

    const declared = readCapabilityList(entry['capabilities']);
    if (declared === undefined) {
      return { ok: false, reason: `models[${index}].capabilities must be an array of known capabilities` };
    }
    const name = readString(entry, 'name') ?? id;
    const weightsPath = readString(entry, 'weightsPath') ?? readString(entry, 'modelPath');
    const vramGb = readNumberField(entry, 'vramGb');
    const ramGb = readNumberField(entry, 'ramGb');
    const priority = readNumberField(entry, 'priority');
    const notes = readString(entry, 'notes');
    const requiresGpu = typeof entry['requiresGpu'] === 'boolean' ? entry['requiresGpu'] : undefined;

    models.push({
      id,
      name,
      ...(declared.length === 0 ? {} : { capabilities: declared }),
      ...(weightsPath === undefined ? {} : { weightsPath }),
      ...(vramGb === undefined ? {} : { vramGb }),
      ...(ramGb === undefined ? {} : { ramGb }),
      ...(requiresGpu === undefined ? {} : { requiresGpu }),
      ...(priority === undefined ? {} : { priority }),
      ...(notes === undefined ? {} : { notes }),
    });
  }
  return { ok: true, models };
}

/**
 * Read an optional capability list, validating every name against the vocabulary.
 * @param raw - the candidate value.
 * @returns the capabilities, or `undefined` when the value is present but invalid.
 */
function readCapabilityList(raw: unknown): Capability[] | undefined {
  if (raw === undefined) return [];
  if (!Array.isArray(raw)) return undefined;
  const capabilities: Capability[] = [];
  for (const entry of raw) {
    if (typeof entry !== 'string' || !isCapability(entry)) return undefined;
    if (!capabilities.includes(entry)) capabilities.push(entry);
  }
  return capabilities;
}

/**
 * Read an optional finite number field.
 * @param source - the object.
 * @param key - the field name.
 * @returns the number, or `undefined`.
 */
function readNumberField(source: Record<string, unknown>, key: string): number | undefined {
  const value = source[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/**
 * Map one declared model plus the engine's surface into a descriptor.
 *
 * Capabilities are intersected, never unioned: a model claims what the operator
 * declared *and* what the engine's routes prove. Declaring more than the engine
 * can do is the mistake that produces invocation failures, so it is removed here
 * — and when the intersection is empty the caller reports that rather than
 * publishing a model that can do nothing.
 *
 * @param model - the declared model.
 * @param surface - what the engine's API surface proved.
 * @param host - the host the model belongs to.
 * @returns the descriptor, or a reason it cannot be published.
 */
export function mapThreeDModel(
  model: ThreeDDeclaredModel,
  surface: ThreeDSurface,
  host: ModelHost,
): { readonly ok: true; readonly descriptor: ModelDescriptor } | { readonly ok: false; readonly reason: string } {
  const capabilities =
    model.capabilities === undefined || model.capabilities.length === 0
      ? [...surface.capabilities]
      : model.capabilities.filter((capability) => surface.capabilities.includes(capability));

  if (capabilities.length === 0) {
    const declared = model.capabilities?.join(', ') ?? '(none declared)';
    const proven = surface.capabilities.join(', ') || 'none';
    return {
      ok: false,
      reason:
        `model "${model.id}" declares ${declared} but ${host.runtime.engine} at ${host.runtime.endpoint} ` +
        `only exposes routes for ${proven}`,
    };
  }

  const { inputTypes, outputTypes } = ioForCapabilities(capabilities);
  const resources: {
    vramGb?: number;
    ramGb?: number;
    requiresGpu?: boolean;
  } = {};
  if (model.vramGb !== undefined) resources.vramGb = model.vramGb;
  if (model.ramGb !== undefined) resources.ramGb = model.ramGb;
  if (model.requiresGpu !== undefined) resources.requiresGpu = model.requiresGpu;

  const primaryApi = primaryApiFor(capabilities, surface);
  // A host that declares the engine's call protocol shares it with every model on
  // that engine — the two-call shape of an image-to-3D app belongs to the app, not
  // to one checkpoint. So a model inherits the host's configuration and adds only
  // what is its own; a host that declares none gets the single-call shape the
  // engine's own API surface just proved.
  //
  // The `models` list itself is dropped on the way down: it is the *declaration*
  // directive that produced this descriptor, not an adapter setting, and a
  // descriptor carrying a copy of its own catalogue would be inherited by any
  // model that pointed at it.
  const hostConfig = host.adapterConfig ?? {};
  const engineConfig: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(hostConfig)) {
    if (key !== 'models') engineConfig[key] = value;
  }
  const hostDeclaresProtocol = engineConfig['steps'] !== undefined || engineConfig['stepsPath'] !== undefined;
  const adapterConfig = hostDeclaresProtocol
    ? {
        ...engineConfig,
        ...(model.weightsPath === undefined ? {} : { weightsPath: model.weightsPath }),
      }
    : {
        protocol: 'gradio',
        ...(primaryApi === undefined ? {} : { apiName: primaryApi }),
        ...(model.weightsPath === undefined ? {} : { weightsPath: model.weightsPath }),
      };

  return {
    ok: true,
    descriptor: {
      id: model.id,
      name: model.name,
      type: modelTypeForCapabilities(capabilities),
      host: host.id,
      capabilities,
      inputTypes,
      outputTypes,
      adapterConfig,
      // No `health` field on purpose: the adapter decides which API-description
      // route to probe, because which one exists depends on the Gradio version
      // the engine runs (see `adapters/three-d.ts`). Freezing one here would make
      // a discovered model look unhealthy on the other version.
      ...(Object.keys(resources).length === 0 ? {} : { resources }),
      priority: model.priority ?? DISCOVERED_PRIORITY,
      tags: [...DISCOVERED_TAGS],
      notes:
        `Discovered from the ${host.runtime.engine} API surface (${surface.endpoints.length} endpoint(s)); ` +
        `not listed in models.json.${model.notes === undefined ? '' : ` ${model.notes}`}`,
    },
  };
}

/**
 * Choose the endpoint a single-call model should be invoked through.
 *
 * The first endpoint whose name proves the model's *primary* capability wins;
 * `image_to_3d` is preferred over `text_to_3d` because it is the capability every
 * current image-to-3D engine actually implements, and a model that declares both
 * is almost always an image-to-3D engine with a text front end.
 *
 * @param capabilities - the model's proven capabilities.
 * @param surface - the engine's API surface.
 * @returns the endpoint name, or `undefined` when none matches.
 */
function primaryApiFor(capabilities: readonly Capability[], surface: ThreeDSurface): string | undefined {
  const order: readonly Capability[] = ['image_to_3d', 'text_to_3d'];
  for (const capability of order) {
    if (!capabilities.includes(capability)) continue;
    for (const endpoint of surface.endpoints) {
      const matched = ENDPOINT_CAPABILITY_PATTERNS.some(
        (entry) => entry.capability === capability && entry.pattern.test(endpoint),
      );
      if (matched) return endpoint;
    }
  }
  return undefined;
}

/**
 * The model `type` a capability set implies.
 *
 * Presentation only — the router filters on `capabilities` — but it keeps the
 * settings page and `list_models` grouping honest.
 *
 * @param capabilities - the declared capabilities.
 * @returns the model type.
 */
function modelTypeForCapabilities(capabilities: readonly Capability[]): ModelType {
  const threeD: readonly IoType[] = ['model_3d'];
  return capabilities.some((capability) => ioForCapabilities([capability]).outputTypes.some((kind) => threeD.includes(kind)))
    ? 'three_d_generation'
    : 'custom';
}

/**
 * Ask an engine what it can do, and publish what it proves.
 *
 * @param options - per-request budget override, for tests and slow machines.
 * @returns the discoverer.
 */
export function createThreeDDiscoverer(options: { readonly requestTimeoutMs?: number } = {}): HostDiscoverer {
  const requestTimeoutMs = Math.max(1, options.requestTimeoutMs ?? REQUEST_TIMEOUT_MS);

  return {
    engine: THREE_D_ENGINE,
    aliases: THREE_D_ENGINE_ALIASES,
    async discover(host: ModelHost, signal: AbortSignal): Promise<ModelDescriptor[]> {
      const endpoint = host.runtime.endpoint;
      if (endpoint === undefined || endpoint.trim().length === 0) return [];

      const surface = await readSurface(endpoint, signal, requestTimeoutMs);
      if (surface === undefined) {
        throw new Error(
          `could not describe the 3D engine at ${endpoint}. Tried ${GRADIO_CONFIG_PATHS.join(' and ')} — ` +
            'a Gradio app answers one of those with its API description once its model has finished loading.',
        );
      }

      const configured = parseThreeDHostConfig(host);
      const declarations = configured.ok ? configured.models : [];
      if (!configured.ok) {
        throw new Error(`the ${host.id} host declares an unusable model list: ${configured.reason}`);
      }

      const descriptors: ModelDescriptor[] = [];
      const skipped: string[] = [];

      if (declarations.length === 0) {
        const implied = mapThreeDModel({ id: defaultModelId(host), name: host.name }, surface, host);
        if (implied.ok) descriptors.push(implied.descriptor);
        else skipped.push(implied.reason);
      } else {
        for (const declaration of declarations) {
          const mapped = mapThreeDModel(declaration, surface, host);
          if (mapped.ok) descriptors.push(mapped.descriptor);
          else skipped.push(mapped.reason);
        }
      }

      if (descriptors.length === 0) {
        const exportNote =
          surface.exportOnly.length > 0
            ? ` Its only mesh-related routes write a mesh the engine was handed (${surface.exportOnly.join(', ')}), which is not generation.`
            : '';
        const skipNote = skipped.length === 0 ? '' : ` ${skipped.join('; ')}`;
        throw new Error(
          `${endpoint} is reachable but exposes no image-to-3D or text-to-3D route, so no 3D model can be published.` +
            exportNote +
            skipNote,
        );
      }

      return descriptors;
    },
  };
}

/**
 * Read an engine's API surface, trying each known description route in order.
 *
 * Both `fetchJson` failures and shape mismatches fall through to the next route,
 * which is what lets one discoverer serve Gradio 3, 4, and 5 without being told
 * which it is talking to.
 *
 * @param endpoint - the engine's base URL.
 * @param signal - cancellation for the pass.
 * @param timeoutMs - budget per request.
 * @returns the surface, or `undefined` when no route described a Gradio app.
 */
async function readSurface(
  endpoint: string,
  signal: AbortSignal,
  timeoutMs: number,
): Promise<ThreeDSurface | undefined> {
  for (const path of GRADIO_CONFIG_PATHS) {
    const read = await fetchJson(joinUrl(endpoint, path), signal, timeoutMs);
    if (!read.ok) continue;
    const surface = parseGradioSurface(read.value, path);
    if (surface !== undefined) return surface;
  }
  return undefined;
}

/**
 * The id to publish for an engine that declares no model list.
 *
 * Derived from the host so it is stable across runs, and prefixed with the engine
 * label so two 3D engines on one machine cannot collide.
 *
 * @param host - the configured host.
 * @returns a valid model id.
 */
function defaultModelId(host: ModelHost): string {
  return slugifyModelId(host.name.length > 0 ? host.name : host.id, host.runtime.engine || THREE_D_ENGINE, 'model');
}

/**
 * Join a base URL and a path without doubling or dropping the separator.
 * @param endpoint - the base URL.
 * @param path - the path, with a leading slash.
 * @returns the absolute URL.
 */
export function joinUrl(endpoint: string, path: string): string {
  const base = endpoint.endsWith('/') ? endpoint.slice(0, -1) : endpoint;
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}
