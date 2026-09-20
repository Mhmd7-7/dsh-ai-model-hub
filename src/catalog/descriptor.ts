/**
 * The model descriptor: the single source of truth about one model.
 *
 * The whole design rests on this file. A model is *data*, not code. Adding a
 * model means adding a JSON entry to a config file plus — only if its engine is
 * not already supported — one adapter. No router change, no agent-prompt change,
 * no DSH change.
 *
 * Two shapes exist here on purpose:
 *
 * - {@link ModelDescriptor} is what a human writes in `config/models.json`.
 * - {@link resolveDescriptor} turns it plus its {@link ModelHost} into a
 *   {@link ResolvedModel}, where every inheritable field is guaranteed present.
 *
 * That split is what lets several models share one host process: the host
 * declares the endpoint, launch command, and resource envelope once, and each
 * model descriptor only states what differs.
 *
 * @module dsh-ai-model-hub/catalog/descriptor
 */

import type { Capability, IoType, ModelType } from './capabilities.ts';
import {
  CAPABILITY_IO,
  isCapability,
  isIoType,
  isModelType,
} from './capabilities.ts';
import type { ValidationIssue } from '../util/validate.ts';
import {
  IssueCollector,
  formatIssues,
  isRecord,
  readOptionalBoolean,
  readOptionalEnum,
  readOptionalEnumArray,
  readOptionalNumber,
  readOptionalRecord,
  readOptionalString,
  readOptionalStringArray,
  readRequiredString,
} from '../util/validate.ts';
import { ModelHubError } from '../errors.ts';

/**
 * How the hub talks to a model. One adapter per kind, registered in the adapter
 * registry. Adding a new kind of engine integration means adding a value here
 * and one adapter — never changing the router.
 */
export const ADAPTER_KINDS = [
  /** In-process synthetic adapter; produces deterministic fixture content. */
  'mock',
  /** JSON over HTTP: POST a request body, read a response body. */
  'http_json',
  /** An OpenAI-compatible `/v1/chat/completions` endpoint (Ollama, llama.cpp, vLLM, LM Studio, …). */
  'openai_compatible',
  /** A ComfyUI server: queue a node graph on `/prompt`, poll `/history`, fetch `/view`. */
  'comfyui',
  /** A command-line program invoked per request with a staged input directory. */
  'cli',
] as const;

/** One supported integration kind. */
export type AdapterKind = (typeof ADAPTER_KINDS)[number];

/** Health-check strategies the runtime manager can run against a model. */
export const HEALTH_CHECK_KINDS = [
  /** Never probe; treat the model as healthy whenever it is running. */
  'none',
  /** TCP connect to `host:port`. */
  'tcp',
  /** HTTP `GET` the given path and accept any 2xx/3xx. */
  'http',
  /** Run `command` and accept exit code 0. */
  'command',
] as const;

/** One health-check strategy. */
export type HealthCheckKind = (typeof HEALTH_CHECK_KINDS)[number];

/**
 * How to determine whether a model is alive.
 *
 * Defaults are derived from the adapter kind when omitted: `http_json` and
 * `openai_compatible` get an HTTP check against their endpoint, `cli` gets
 * `none` (it is not a long-lived process), and `mock` gets `none`.
 */
export interface HealthCheckSpec {
  /** The strategy. */
  readonly kind: HealthCheckKind;
  /** Path for `http`, e.g. `/health`. Defaults to `/`. */
  readonly path?: string;
  /** Command for `command`; subject to the launch guardrails. */
  readonly command?: string;
  /** Arguments for `command`. */
  readonly args?: readonly string[];
  /** Per-attempt timeout in milliseconds. Defaults to 2000. */
  readonly timeoutMs?: number;
}

/**
 * Where and how a model actually executes.
 *
 * `endpoint` covers the "already running somewhere" case and `launch` the
 * "we know how to start it" case. A descriptor may declare both: the manager
 * then prefers an already-healthy endpoint over spawning a duplicate process.
 */
export interface RuntimeSpec {
  /** Free-form engine label for operators and logs, e.g. `llama.cpp`, `ollama`, `diffusers`, `comfyui`, `blender`. */
  readonly engine: string;
  /** Which adapter integrates with it. */
  readonly adapter: AdapterKind;
  /** Base URL of a long-lived server, e.g. `http://127.0.0.1:9001`. */
  readonly endpoint?: string;
  /**
   * The request path appended to {@link endpoint}, e.g. `/v1/chat/completions`
   * or `/sdapi/v1/txt2img`. Omitted means the adapter chooses its own default.
   */
  readonly path?: string;
  /**
   * Where the model's weight files live. Only used by CLI engines that read a
   * local checkpoint; ignored by server-backed engines.
   */
  readonly modelPath?: string;
  /** Extra environment variables exported to a launched process. */
  readonly env?: Readonly<Record<string, string>>;
  /** Extra arguments appended to the launch command's declared `args`. */
  readonly args?: readonly string[];
  /** Data directory the engine should use for caches and outputs. */
  readonly dataDir?: string;
}

/** A single command plus its arguments. Never a shell string — see the guardrails. */
export interface CommandSpec {
  /** Executable name or absolute path. Resolved on `PATH` when bare. */
  readonly command: string;
  /** Arguments passed verbatim; the hub never concatenates them into a shell line. */
  readonly args?: readonly string[];
  /** Working directory for the process. */
  readonly cwd?: string;
}

/**
 * How to start and stop a model's process.
 *
 * Omitting this entirely marks a model as external: the hub will only ever talk
 * to its `endpoint` and will never spawn anything. That is the right choice for
 * engines a user manages themselves (Ollama, a system ComfyUI).
 */
export interface LifecycleSpec {
  /** Whether the hub is allowed to start this model. */
  readonly startable: boolean;
  /** Whether the hub is allowed to stop it. */
  readonly stoppable: boolean;
  /** The command that starts a long-lived server. */
  readonly start?: CommandSpec;
  /** The command that stops it, when the engine has no HTTP shutdown route. */
  readonly stop?: CommandSpec;
  /** Extra seconds to wait for a clean start before the launch is a failure. Defaults to 120. */
  readonly startupTimeoutMs?: number;
  /** Extra seconds to wait for a clean stop before a forced kill. Defaults to 15. */
  readonly shutdownTimeoutMs?: number;
  /** Kill the process after this many idle milliseconds. Off when omitted or `0`. */
  readonly idleTimeoutMs?: number;
  /** Whether a start should wait for the health check to pass. Defaults to true. */
  readonly awaitHealthOnStart?: boolean;
}

/** The machine resources a model needs while resident. */
export interface ResourceSpec {
  /** GPU memory needed, in gibibytes. */
  readonly vramGb?: number;
  /** System memory needed, in gibibytes. */
  readonly ramGb?: number;
  /** Disk space needed for weights, in gibibytes. */
  readonly diskGb?: number;
  /** Whether a CUDA-capable GPU is required rather than merely preferred. */
  readonly requiresGpu?: boolean;
  /**
   * Whether two instances of this model can be resident at once. Defaults to
   * false, which makes the runtime manager treat it as an exclusive resource.
   */
  readonly allowConcurrentInstances?: boolean;
}

/** Model-specific size limits that routing can filter on. */
export interface LimitsSpec {
  /** Maximum context window in tokens, for text models. */
  readonly contextTokens?: number;
  /** Maximum input image width in pixels. */
  readonly maxWidth?: number;
  /** Maximum input image height in pixels. */
  readonly maxHeight?: number;
  /** Resolutions the model supports, as `WIDTHxHEIGHT` strings. */
  readonly resolutions?: readonly string[];
  /** Maximum output video/audio duration in seconds. */
  readonly maxDurationSeconds?: number;
}

/** Free-form adapter configuration. Shape is owned by the adapter, not the catalog. */
export type AdapterConfig = Readonly<Record<string, unknown>>;

/**
 * One model, as written in configuration.
 *
 * Only `id`, `name`, `type`, `capabilities`, and `adapter` are structurally
 * required; `host` is required unless the descriptor carries its own inline
 * `runtime`. The adapter needs to be stated somewhere — on the descriptor or on
 * the host — because two models behind one engine may still be reached by
 * different adapter kinds.
 */
export interface ModelDescriptor {
  /** Unique, stable, kebab-case id. Never reused for a different model. */
  readonly id: string;
  /** Human-facing display name. */
  readonly name: string;
  /** What the model is. */
  readonly type: ModelType;
  /** What the model can do. This — not `type` — is what routing filters on. */
  readonly capabilities: readonly Capability[];
  /** Artifact kinds accepted as input. Defaults from {@link CAPABILITY_IO}. */
  readonly inputTypes?: readonly IoType[];
  /** Artifact kinds produced as output. Defaults from {@link CAPABILITY_IO}. */
  readonly outputTypes?: readonly IoType[];
  /** Semantic version of *this descriptor*, e.g. `1.0.0` or `2024-06-01`. */
  readonly version?: string;
  /** Id of the {@link ModelHost} that owns the process. Omit for a standalone model. */
  readonly host?: string;
  /** Inline runtime spec; mutually exclusive with `host`, and wins if both appear. */
  readonly runtime?: RuntimeSpec;
  /** Which adapter integrates with this model. Defaults to the host's adapter. */
  readonly adapter?: AdapterKind;
  /** Free-form adapter settings. */
  readonly adapterConfig?: AdapterConfig;
  /** Machine resources needed while resident. */
  readonly resources?: ResourceSpec;
  /** Model-specific size limits. */
  readonly limits?: LimitsSpec;
  /** How to start/stop it. Defaults to the host's lifecycle, else not startable. */
  readonly lifecycle?: LifecycleSpec;
  /** How to check whether it is alive. Defaults from the adapter kind. */
  readonly health?: HealthCheckSpec;
  /** Whether routing may select this model at all. Defaults to true. */
  readonly enabled?: boolean;
  /** Lower numbers are preferred when several models tie on capability. Defaults to 100. */
  readonly priority?: number;
  /** Deployment/environment tags, e.g. `["gpu", "production"]`. Used for filtering. */
  readonly tags?: readonly string[];
  /** Free-form operator notes and provenance; never interpreted by the hub. */
  readonly notes?: string;
  /** JSON Schema convention: a comment for human readers, ignored by the hub. */
  readonly $comment?: string;
}

/**
 * A shared process/environment that several models sit behind.
 *
 * The host exists so that "how do I launch this engine" is written once. A user
 * running ComfyUI with three checkpoints writes one host plus three thin model
 * entries; a user with three unrelated engines writes three hosts (or none, using
 * inline runtimes).
 */
export interface ModelHost {
  /** Unique id referenced by {@link ModelDescriptor.host}. */
  readonly id: string;
  /** Human-facing name. */
  readonly name: string;
  /** Which adapter talks to this host. */
  readonly adapter: AdapterKind;
  /** Endpoint, launch command, and engine label. */
  readonly runtime: RuntimeSpec;
  /** Default lifecycle for every model on the host. */
  readonly lifecycle?: LifecycleSpec;
  /** Resource envelope reserved while any model on this host is running. */
  readonly resources?: ResourceSpec;
  /** Default health check for the host's models. */
  readonly health?: HealthCheckSpec;
  /** Whether the host's models may be selected. Defaults to true. */
  readonly enabled?: boolean;
  /** Free-form operator notes. */
  readonly notes?: string;
  /** JSON Schema convention: a comment for human readers, ignored by the hub. */
  readonly $comment?: string;
}

/** The root object of `config/models.json`. */
export interface ModelCatalogConfig {
  /** Schema version, so a future breaking change can migrate rather than guess. */
  readonly version?: string;
  /** Shared processes, keyed by id. */
  readonly hosts?: readonly ModelHost[];
  /** The models themselves. */
  readonly models: readonly ModelDescriptor[];
  /**
   * JSON Schema convention: a comment for human readers.
   *
   * Unknown keys on any object in the catalog are ignored rather than rejected —
   * a hand-edited configuration is a working document, and failing on a stray
   * `$comment` or a field added for a future version would be hostile. Values
   * that *are* read are validated strictly.
   */
  readonly $comment?: string;
}

/**
 * A descriptor with every inheritable field resolved against its host.
 *
 * This is what the router and runtime manager actually consume, so neither has
 * to know about inheritance or re-implement its precedence rules.
 */
export interface ResolvedModel {
  readonly id: string;
  readonly name: string;
  readonly type: ModelType;
  readonly capabilities: readonly Capability[];
  readonly inputTypes: readonly IoType[];
  readonly outputTypes: readonly IoType[];
  readonly version: string;
  readonly adapter: AdapterKind;
  readonly adapterConfig: AdapterConfig;
  readonly runtime: RuntimeSpec;
  readonly resources: Required<Pick<ResourceSpec, 'vramGb' | 'ramGb' | 'diskGb' | 'requiresGpu' | 'allowConcurrentInstances'>> &
    ResourceSpec;
  readonly limits: LimitsSpec;
  readonly lifecycle: {
    readonly startable: boolean;
    readonly stoppable: boolean;
    readonly start?: CommandSpec;
    readonly stop?: CommandSpec;
    readonly startupTimeoutMs: number;
    readonly shutdownTimeoutMs: number;
    readonly idleTimeoutMs: number;
    readonly awaitHealthOnStart: boolean;
  };
  readonly health: HealthCheckSpec;
  readonly enabled: boolean;
  readonly priority: number;
  readonly tags: readonly string[];
  /** The host this model inherited from, when any. */
  readonly hostId?: string;
  /** The descriptor exactly as written, for diagnostics and round-tripping. */
  readonly descriptor: ModelDescriptor;
}

const DEFAULTS = {
  version: '0.0.0',
  priority: 100,
  startupTimeoutMs: 120_000,
  shutdownTimeoutMs: 15_000,
  idleTimeoutMs: 0,
  healthTimeoutMs: 2_000,
} as const;

/**
 * Parse one untrusted value into a {@link ModelHost}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated host, or `undefined` when it cannot be read at all.
 */
export function parseHost(
  raw: unknown,
  path: string,
  collector: IssueCollector,
): ModelHost | undefined {
  if (!isRecord(raw)) {
    collector.add(path, 'must be an object');
    return undefined;
  }
  const id = readRequiredString(raw, 'id', path, collector);
  const name = readRequiredString(raw, 'name', path, collector);
  const adapter = readOptionalEnum(raw, 'adapter', ADAPTER_KINDS, path, collector);
  if (adapter === undefined) collector.add(`${path}.adapter`, 'is required');
  const runtime = parseRuntimeSpec(raw['runtime'], `${path}.runtime`, collector, adapter);
  if (id === undefined || name === undefined || adapter === undefined || runtime === undefined) {
    return undefined;
  }
  const host: {
    id: string;
    name: string;
    adapter: AdapterKind;
    runtime: RuntimeSpec;
    lifecycle?: LifecycleSpec;
    resources?: ResourceSpec;
    health?: HealthCheckSpec;
    enabled?: boolean;
    notes?: string;
  } = { id, name, adapter, runtime };
  const lifecycle = parseLifecycleSpec(raw['lifecycle'], `${path}.lifecycle`, collector);
  if (lifecycle !== undefined) host.lifecycle = lifecycle;
  const resources = parseResourceSpec(raw['resources'], `${path}.resources`, collector);
  if (resources !== undefined) host.resources = resources;
  const health = parseHealthSpec(raw['health'], `${path}.health`, collector);
  if (health !== undefined) host.health = health;
  const enabled = readOptionalBoolean(raw, 'enabled', path, collector);
  if (enabled !== undefined) host.enabled = enabled;
  const notes = readOptionalString(raw, 'notes', path, collector);
  if (notes !== undefined) host.notes = notes;
  return host;
}

/**
 * Parse one untrusted value into a {@link ModelDescriptor}.
 *
 * Cross-field rules the schema cannot express — a host that must exist, a
 * capability that contradicts the declared input/output types, an endpoint that
 * must be present for an HTTP adapter — are checked by
 * {@link validateCatalogConfig} once every descriptor is parsed, because they
 * need the whole document in hand.
 *
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated descriptor, or `undefined` when it cannot be read.
 */
export function parseModelDescriptor(
  raw: unknown,
  path: string,
  collector: IssueCollector,
): ModelDescriptor | undefined {
  if (!isRecord(raw)) {
    collector.add(path, 'must be an object');
    return undefined;
  }
  const id = readRequiredString(raw, 'id', path, collector);
  if (id !== undefined && !isValidModelId(id)) {
    // The id reaches artifact ids, tool output, and log lines, so its character
    // set is constrained rather than merely conventional. Enforcing it here keeps
    // the runtime validator and the published JSON Schema in agreement.
    collector.add(
      `${path}.id`,
      `must be lowercase kebab-case matching ${MODEL_ID_PATTERN.source} (got "${id}")`,
    );
  }
  const name = readRequiredString(raw, 'name', path, collector);
  const type = readOptionalEnum(raw, 'type', MODEL_TYPE_VALUES, path, collector);
  if (type === undefined) collector.add(`${path}.type`, 'is required');
  const capabilities = readOptionalEnumArray(raw, 'capabilities', CAPABILITY_VALUES, path, collector);
  if (capabilities === undefined) collector.add(`${path}.capabilities`, 'is required');
  else if (capabilities.length === 0) collector.add(`${path}.capabilities`, 'must name at least one capability');

  if (id === undefined || name === undefined || type === undefined || capabilities === undefined) {
    return undefined;
  }

  const descriptor: {
    id: string;
    name: string;
    type: ModelType;
    capabilities: readonly Capability[];
    inputTypes?: readonly IoType[];
    outputTypes?: readonly IoType[];
    version?: string;
    host?: string;
    runtime?: RuntimeSpec;
    adapter?: AdapterKind;
    adapterConfig?: AdapterConfig;
    resources?: ResourceSpec;
    limits?: LimitsSpec;
    lifecycle?: LifecycleSpec;
    health?: HealthCheckSpec;
    enabled?: boolean;
    priority?: number;
    tags?: readonly string[];
    notes?: string;
  } = { id, name, type, capabilities };

  const inputTypes = readOptionalEnumArray(raw, 'inputTypes', IO_TYPE_VALUES, path, collector);
  if (inputTypes !== undefined) descriptor.inputTypes = inputTypes;
  const outputTypes = readOptionalEnumArray(raw, 'outputTypes', IO_TYPE_VALUES, path, collector);
  if (outputTypes !== undefined) descriptor.outputTypes = outputTypes;
  const version = readOptionalString(raw, 'version', path, collector);
  if (version !== undefined) descriptor.version = version;
  const hostId = readOptionalString(raw, 'host', path, collector);
  if (hostId !== undefined) descriptor.host = hostId;
  const adapter = readOptionalEnum(raw, 'adapter', ADAPTER_KINDS, path, collector);
  if (adapter !== undefined) descriptor.adapter = adapter;
  const runtime = parseRuntimeSpec(raw['runtime'], `${path}.runtime`, collector, adapter);
  if (runtime !== undefined) descriptor.runtime = runtime;
  const adapterConfig = readOptionalRecord(raw, 'adapterConfig', path, collector);
  if (adapterConfig !== undefined) descriptor.adapterConfig = adapterConfig;
  const resources = parseResourceSpec(raw['resources'], `${path}.resources`, collector);
  if (resources !== undefined) descriptor.resources = resources;
  const limits = parseLimitsSpec(raw['limits'], `${path}.limits`, collector);
  if (limits !== undefined) descriptor.limits = limits;
  const lifecycle = parseLifecycleSpec(raw['lifecycle'], `${path}.lifecycle`, collector);
  if (lifecycle !== undefined) descriptor.lifecycle = lifecycle;
  const health = parseHealthSpec(raw['health'], `${path}.health`, collector);
  if (health !== undefined) descriptor.health = health;
  const enabled = readOptionalBoolean(raw, 'enabled', path, collector);
  if (enabled !== undefined) descriptor.enabled = enabled;
  const priority = readOptionalNumber(raw, 'priority', path, collector);
  if (priority !== undefined) descriptor.priority = priority;
  const tags = readOptionalStringArray(raw, 'tags', path, collector);
  if (tags !== undefined) descriptor.tags = tags;
  const notes = readOptionalString(raw, 'notes', path, collector);
  if (notes !== undefined) descriptor.notes = notes;
  return descriptor;
}

const MODEL_TYPE_VALUES: readonly ModelType[] = [
  'text_generation',
  'image_generation',
  'image_editing',
  'image_understanding',
  'three_d_generation',
  'audio_generation',
  'speech_recognition',
  'video_generation',
  'multimodal',
  'custom',
];

const CAPABILITY_VALUES: readonly Capability[] = [
  'text_to_text',
  'text_to_image',
  'image_to_image',
  'text_to_3d',
  'image_to_3d',
  'audio_generation',
  'speech_to_text',
  'image_understanding',
  'video_generation',
];

const IO_TYPE_VALUES: readonly IoType[] = [
  'text',
  'image',
  'audio',
  'video',
  'model_3d',
  'json',
  'file',
];

/**
 * Parse an untrusted value into a {@link RuntimeSpec}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @param fallbackAdapter - adapter declared on the same object, used to require an endpoint.
 * @returns the validated spec, or `undefined`.
 */
function parseRuntimeSpec(
  raw: unknown,
  path: string,
  collector: IssueCollector,
  fallbackAdapter?: AdapterKind,
): RuntimeSpec | undefined {
  if (raw === undefined) return undefined;
  if (!isRecord(raw)) {
    collector.add(path, 'must be an object');
    return undefined;
  }
  const engine = readRequiredString(raw, 'engine', path, collector);
  const adapter = readOptionalEnum(raw, 'adapter', ADAPTER_KINDS, path, collector) ?? fallbackAdapter;
  if (adapter === undefined) collector.add(`${path}.adapter`, 'is required');
  const endpoint = readOptionalString(raw, 'endpoint', path, collector);
  const requestPath = readOptionalString(raw, 'path', path, collector);
  const modelPath = readOptionalString(raw, 'modelPath', path, collector);
  const dataDir = readOptionalString(raw, 'dataDir', path, collector);
  const args = readOptionalStringArray(raw, 'args', path, collector);
  const env = readOptionalRecord(raw, 'env', path, collector);

  if (adapter === 'http_json' || adapter === 'openai_compatible' || adapter === 'comfyui') {
    if (endpoint === undefined && raw['endpoint'] === undefined) {
      collector.add(`${path}.endpoint`, `is required for the ${adapter} adapter`);
    }
  }

  if (engine === undefined || adapter === undefined) return undefined;

  const spec: {
    engine: string;
    adapter: AdapterKind;
    endpoint?: string;
    path?: string;
    modelPath?: string;
    env?: Record<string, string>;
    args?: readonly string[];
    dataDir?: string;
  } = { engine, adapter };
  if (endpoint !== undefined) spec.endpoint = endpoint;
  if (requestPath !== undefined) spec.path = requestPath;
  if (modelPath !== undefined) spec.modelPath = modelPath;
  if (dataDir !== undefined) spec.dataDir = dataDir;
  if (args !== undefined) spec.args = args;
  if (env !== undefined) {
    const clean: Record<string, string> = {};
    for (const [key, value] of Object.entries(env)) {
      if (typeof value === 'string') clean[key] = value;
      else collector.add(`${path}.env.${key}`, 'must be a string');
    }
    spec.env = clean;
  }
  return spec;
}

/**
 * Parse an untrusted value into a {@link ResourceSpec}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated spec, or `undefined` when absent.
 */
export function parseResourceSpec(
  raw: unknown,
  path: string,
  collector: IssueCollector,
): ResourceSpec | undefined {
  if (raw === undefined) return undefined;
  if (!isRecord(raw)) {
    collector.add(path, 'must be an object');
    return undefined;
  }
  const spec: {
    vramGb?: number;
    ramGb?: number;
    diskGb?: number;
    requiresGpu?: boolean;
    allowConcurrentInstances?: boolean;
  } = {};
  const vramGb = readOptionalNumber(raw, 'vramGb', path, collector);
  if (vramGb !== undefined) spec.vramGb = vramGb;
  const ramGb = readOptionalNumber(raw, 'ramGb', path, collector);
  if (ramGb !== undefined) spec.ramGb = ramGb;
  const diskGb = readOptionalNumber(raw, 'diskGb', path, collector);
  if (diskGb !== undefined) spec.diskGb = diskGb;
  const requiresGpu = readOptionalBoolean(raw, 'requiresGpu', path, collector);
  if (requiresGpu !== undefined) spec.requiresGpu = requiresGpu;
  const allowConcurrentInstances = readOptionalBoolean(raw, 'allowConcurrentInstances', path, collector);
  if (allowConcurrentInstances !== undefined) spec.allowConcurrentInstances = allowConcurrentInstances;
  return spec;
}

/**
 * Parse an untrusted value into a {@link LimitsSpec}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated spec, or `undefined` when absent.
 */
export function parseLimitsSpec(
  raw: unknown,
  path: string,
  collector: IssueCollector,
): LimitsSpec | undefined {
  if (raw === undefined) return undefined;
  if (!isRecord(raw)) {
    collector.add(path, 'must be an object');
    return undefined;
  }
  const spec: {
    contextTokens?: number;
    maxWidth?: number;
    maxHeight?: number;
    resolutions?: readonly string[];
    maxDurationSeconds?: number;
  } = {};
  const contextTokens = readOptionalNumber(raw, 'contextTokens', path, collector);
  if (contextTokens !== undefined) spec.contextTokens = contextTokens;
  const maxWidth = readOptionalNumber(raw, 'maxWidth', path, collector);
  if (maxWidth !== undefined) spec.maxWidth = maxWidth;
  const maxHeight = readOptionalNumber(raw, 'maxHeight', path, collector);
  if (maxHeight !== undefined) spec.maxHeight = maxHeight;
  const maxDurationSeconds = readOptionalNumber(raw, 'maxDurationSeconds', path, collector);
  if (maxDurationSeconds !== undefined) spec.maxDurationSeconds = maxDurationSeconds;
  const resolutions = readOptionalStringArray(raw, 'resolutions', path, collector);
  if (resolutions !== undefined) {
    resolutions.forEach((value, index) => {
      if (!/^\d+x\d+$/.test(value)) {
        collector.add(`${path}.resolutions[${index}]`, 'must look like `1024x1024`');
      }
    });
    spec.resolutions = resolutions;
  }
  return spec;
}

/**
 * Parse an untrusted value into a {@link LifecycleSpec}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated spec, or `undefined` when absent.
 */
export function parseLifecycleSpec(
  raw: unknown,
  path: string,
  collector: IssueCollector,
): LifecycleSpec | undefined {
  if (raw === undefined) return undefined;
  if (!isRecord(raw)) {
    collector.add(path, 'must be an object');
    return undefined;
  }
  const startable = readOptionalBoolean(raw, 'startable', path, collector) ?? false;
  const stoppable = readOptionalBoolean(raw, 'stoppable', path, collector) ?? false;
  const spec: {
    startable: boolean;
    stoppable: boolean;
    start?: CommandSpec;
    stop?: CommandSpec;
    startupTimeoutMs?: number;
    shutdownTimeoutMs?: number;
    idleTimeoutMs?: number;
    awaitHealthOnStart?: boolean;
  } = { startable, stoppable };
  const start = parseCommandSpec(raw['start'], `${path}.start`, collector);
  if (start !== undefined) spec.start = start;
  const stop = parseCommandSpec(raw['stop'], `${path}.stop`, collector);
  if (stop !== undefined) spec.stop = stop;
  const startupTimeoutMs = readOptionalNumber(raw, 'startupTimeoutMs', path, collector);
  if (startupTimeoutMs !== undefined) spec.startupTimeoutMs = startupTimeoutMs;
  const shutdownTimeoutMs = readOptionalNumber(raw, 'shutdownTimeoutMs', path, collector);
  if (shutdownTimeoutMs !== undefined) spec.shutdownTimeoutMs = shutdownTimeoutMs;
  const idleTimeoutMs = readOptionalNumber(raw, 'idleTimeoutMs', path, collector);
  if (idleTimeoutMs !== undefined) spec.idleTimeoutMs = idleTimeoutMs;
  const awaitHealthOnStart = readOptionalBoolean(raw, 'awaitHealthOnStart', path, collector);
  if (awaitHealthOnStart !== undefined) spec.awaitHealthOnStart = awaitHealthOnStart;

  if (startable && start === undefined) {
    collector.add(`${path}.start`, 'is required when `startable` is true');
  }
  if (stoppable && start === undefined && stop === undefined) {
    collector.add(`${path}.start`, 'is required when `stoppable` is true (the hub needs a process to own)');
  }
  return spec;
}

/**
 * Parse an untrusted value into a {@link CommandSpec}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated spec, or `undefined` when absent.
 */
function parseCommandSpec(
  raw: unknown,
  path: string,
  collector: IssueCollector,
): CommandSpec | undefined {
  if (raw === undefined) return undefined;
  if (!isRecord(raw)) {
    collector.add(path, 'must be an object');
    return undefined;
  }
  const command = readRequiredString(raw, 'command', path, collector);
  const args = readOptionalStringArray(raw, 'args', path, collector);
  const cwd = readOptionalString(raw, 'cwd', path, collector);
  if (command === undefined) return undefined;
  const spec: { command: string; args?: readonly string[]; cwd?: string } = { command };
  if (args !== undefined) spec.args = args;
  if (cwd !== undefined) spec.cwd = cwd;
  return spec;
}

/**
 * Parse an untrusted value into a {@link HealthCheckSpec}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated spec, or `undefined` when absent.
 */
function parseHealthSpec(
  raw: unknown,
  path: string,
  collector: IssueCollector,
): HealthCheckSpec | undefined {
  if (raw === undefined) return undefined;
  if (!isRecord(raw)) {
    collector.add(path, 'must be an object');
    return undefined;
  }
  const kind = readOptionalEnum(raw, 'kind', HEALTH_CHECK_KINDS, path, collector);
  if (kind === undefined) {
    collector.add(`${path}.kind`, 'is required');
    return undefined;
  }
  const spec: {
    kind: HealthCheckKind;
    path?: string;
    command?: string;
    args?: readonly string[];
    timeoutMs?: number;
  } = { kind };
  const healthPath = readOptionalString(raw, 'path', path, collector);
  if (healthPath !== undefined) spec.path = healthPath;
  const command = readOptionalString(raw, 'command', path, collector);
  if (command !== undefined) spec.command = command;
  const args = readOptionalStringArray(raw, 'args', path, collector);
  if (args !== undefined) spec.args = args;
  const timeoutMs = readOptionalNumber(raw, 'timeoutMs', path, collector);
  if (timeoutMs !== undefined) spec.timeoutMs = timeoutMs;
  if (kind === 'command' && command === undefined) {
    collector.add(`${path}.command`, 'is required for a command health check');
  }
  return spec;
}

/**
 * Parse a whole `models.json` document.
 *
 * Reports *every* problem it finds rather than stopping at the first, because
 * a catalog file is edited by hand and a fix-one-rerun loop is a bad experience.
 *
 * @param raw - the parsed JSON value.
 * @param label - a label for messages, usually the file path.
 * @returns a discriminated result; never throws for malformed input.
 */
export function parseModelCatalogConfig(
  raw: unknown,
  label = 'model catalog',
):
  | { readonly ok: true; readonly config: ModelCatalogConfig }
  | { readonly ok: false; readonly message: string; readonly issues: readonly ValidationIssue[] } {
  const collector = new IssueCollector();
  if (!isRecord(raw)) {
    collector.add('$', 'must be an object');
    return { ok: false, message: formatIssues(label, collector.collected), issues: collector.collected };
  }
  const version = readOptionalString(raw, 'version', '$', collector);
  const modelsRaw = raw['models'];
  if (!Array.isArray(modelsRaw)) {
    collector.add('$.models', 'must be an array');
    return { ok: false, message: formatIssues(label, collector.collected), issues: collector.collected };
  }

  const hosts: ModelHost[] = [];
  const hostsRaw = raw['hosts'];
  if (hostsRaw !== undefined) {
    if (!Array.isArray(hostsRaw)) {
      collector.add('$.hosts', 'must be an array');
    } else {
      hostsRaw.forEach((entry, index) => {
        const host = parseHost(entry, `hosts[${index}]`, collector);
        if (host !== undefined) hosts.push(host);
      });
    }
  }

  const models: ModelDescriptor[] = [];
  modelsRaw.forEach((entry, index) => {
    const descriptor = parseModelDescriptor(entry, `models[${index}]`, collector);
    if (descriptor !== undefined) models.push(descriptor);
  });

  validateCatalogCrossReferences(models, hosts, collector);

  if (collector.failed) {
    return { ok: false, message: formatIssues(label, collector.collected), issues: collector.collected };
  }
  const config: { version?: string; hosts: ModelHost[]; models: ModelDescriptor[] } = { hosts, models };
  if (version !== undefined) config.version = version;
  return { ok: true, config };
}

/**
 * Check the rules that need the whole document: unique ids, resolvable hosts,
 * and capabilities that agree with the declared input/output artifact kinds.
 *
 * @param models - every parsed descriptor.
 * @param hosts - every parsed host.
 * @param collector - issue sink.
 */
function validateCatalogCrossReferences(
  models: readonly ModelDescriptor[],
  hosts: readonly ModelHost[],
  collector: IssueCollector,
): void {
  const hostIds = new Set<string>();
  hosts.forEach((host, index) => {
    if (hostIds.has(host.id)) collector.add(`hosts[${index}].id`, `duplicate host id "${host.id}"`);
    hostIds.add(host.id);
  });

  const modelIds = new Set<string>();
  models.forEach((model, index) => {
    const path = `models[${index}]`;
    if (modelIds.has(model.id)) collector.add(`${path}.id`, `duplicate model id "${model.id}"`);
    modelIds.add(model.id);

    if (model.host !== undefined && model.runtime === undefined && !hostIds.has(model.host)) {
      collector.add(`${path}.host`, `references unknown host "${model.host}"`);
    }
    if (model.host === undefined && model.runtime === undefined) {
      collector.add(`${path}`, 'must declare either `host` or an inline `runtime`');
    }
    if (model.host !== undefined && model.runtime !== undefined) {
      collector.add(`${path}`, 'must not declare both `host` and `runtime`; put shared settings on the host');
    }

    const declaredInputs = model.inputTypes;
    if (declaredInputs !== undefined) {
      const expected = new Set<IoType>();
      for (const capability of model.capabilities) for (const io of CAPABILITY_IO[capability].input) expected.add(io);
      for (const declared of declaredInputs) {
        if (!expected.has(declared)) {
          collector.add(
            `${path}.inputTypes`,
            `"${declared}" is not an input of any declared capability (${[...expected].join(', ') || 'none'})`,
          );
        }
      }
    }
    const declaredOutputs = model.outputTypes;
    if (declaredOutputs !== undefined) {
      const expected = new Set<IoType>();
      for (const capability of model.capabilities) for (const io of CAPABILITY_IO[capability].output) expected.add(io);
      for (const declared of declaredOutputs) {
        if (!expected.has(declared)) {
          collector.add(
            `${path}.outputTypes`,
            `"${declared}" is not an output of any declared capability (${[...expected].join(', ') || 'none'})`,
          );
        }
      }
    }
  });
}

/**
 * Resolve one descriptor against its host into the flat record consumers use.
 *
 * Precedence, highest first: descriptor field, host field, adapter-derived
 * default, built-in default. There is exactly one implementation of this rule
 * and it lives here, so the router and the runtime manager cannot disagree
 * about what a model's timeout or endpoint is.
 *
 * @param descriptor - the model as written.
 * @param host - the host it names, when any.
 * @returns the resolved model record.
 * @throws ModelHubError with `INVALID_DESCRIPTOR` when the input is inconsistent.
 */
export function resolveDescriptor(
  descriptor: ModelDescriptor,
  host: ModelHost | undefined,
): ResolvedModel {
  if (descriptor.host !== undefined && descriptor.runtime === undefined && host === undefined) {
    throw new ModelHubError(
      'INVALID_DESCRIPTOR',
      `model "${descriptor.id}" references host "${descriptor.host}", which is not defined`,
      { modelId: descriptor.id, hostId: descriptor.host },
    );
  }
  if (descriptor.host !== undefined && descriptor.runtime !== undefined) {
    throw new ModelHubError(
      'INVALID_DESCRIPTOR',
      `model "${descriptor.id}" declares both \`host\` and \`runtime\`; declare exactly one`,
      { modelId: descriptor.id },
    );
  }

  const runtime = descriptor.runtime ?? host?.runtime;
  if (runtime === undefined) {
    throw new ModelHubError(
      'INVALID_DESCRIPTOR',
      `model "${descriptor.id}" has no runtime: declare \`host\` or an inline \`runtime\``,
      { modelId: descriptor.id },
    );
  }

  const adapter = descriptor.adapter ?? host?.adapter ?? runtime.adapter;

  const descriptorResources = descriptor.resources ?? {};
  const hostResources = host?.resources ?? {};
  const resources = {
    vramGb: descriptorResources.vramGb ?? hostResources.vramGb ?? 0,
    ramGb: descriptorResources.ramGb ?? hostResources.ramGb ?? 0,
    diskGb: descriptorResources.diskGb ?? hostResources.diskGb ?? 0,
    requiresGpu: descriptorResources.requiresGpu ?? hostResources.requiresGpu ?? false,
    allowConcurrentInstances:
      descriptorResources.allowConcurrentInstances ?? hostResources.allowConcurrentInstances ?? false,
  };

  const lifecycle = resolveLifecycle(descriptor, host);
  const health = descriptor.health ?? host?.health ?? defaultHealthFor(adapter);

  const inputTypes =
    descriptor.inputTypes !== undefined
      ? [...descriptor.inputTypes]
      : [...new Set(descriptor.capabilities.flatMap((capability) => CAPABILITY_IO[capability].input))];
  const outputTypes =
    descriptor.outputTypes !== undefined
      ? [...descriptor.outputTypes]
      : [...new Set(descriptor.capabilities.flatMap((capability) => CAPABILITY_IO[capability].output))];

  const resolved: {
    id: string;
    name: string;
    type: ModelType;
    capabilities: readonly Capability[];
    inputTypes: readonly IoType[];
    outputTypes: readonly IoType[];
    version: string;
    adapter: AdapterKind;
    adapterConfig: AdapterConfig;
    runtime: RuntimeSpec;
    resources: ResolvedModel['resources'];
    limits: LimitsSpec;
    lifecycle: ResolvedModel['lifecycle'];
    health: HealthCheckSpec;
    enabled: boolean;
    priority: number;
    tags: readonly string[];
    hostId?: string;
    descriptor: ModelDescriptor;
  } = {
    id: descriptor.id,
    name: descriptor.name,
    type: descriptor.type,
    capabilities: [...descriptor.capabilities],
    inputTypes,
    outputTypes,
    version: descriptor.version ?? DEFAULTS.version,
    adapter,
    adapterConfig: descriptor.adapterConfig ?? {},
    runtime,
    resources,
    limits: descriptor.limits ?? {},
    lifecycle,
    health,
    enabled: descriptor.enabled ?? host?.enabled ?? true,
    priority: descriptor.priority ?? DEFAULTS.priority,
    tags: [...(descriptor.tags ?? [])],
    descriptor,
  };
  if (descriptor.host !== undefined) resolved.hostId = descriptor.host;
  return resolved;
}

/**
 * Merge a descriptor's lifecycle with its host's, applying defaults.
 * @param descriptor - the model as written.
 * @param host - its host, when any.
 * @returns the fully-defaulted lifecycle.
 */
function resolveLifecycle(
  descriptor: ModelDescriptor,
  host: ModelHost | undefined,
): ResolvedModel['lifecycle'] {
  const own = descriptor.lifecycle;
  const inherited = host?.lifecycle;
  const startable = own?.startable ?? inherited?.startable ?? false;
  const stoppable = own?.stoppable ?? inherited?.stoppable ?? startable;
  const start = own?.start ?? inherited?.start;
  const stop = own?.stop ?? inherited?.stop;
  const lifecycle: {
    startable: boolean;
    stoppable: boolean;
    start?: CommandSpec;
    stop?: CommandSpec;
    startupTimeoutMs: number;
    shutdownTimeoutMs: number;
    idleTimeoutMs: number;
    awaitHealthOnStart: boolean;
  } = {
    startable: startable && start !== undefined,
    stoppable: stoppable && (start !== undefined || stop !== undefined),
    startupTimeoutMs: own?.startupTimeoutMs ?? inherited?.startupTimeoutMs ?? DEFAULTS.startupTimeoutMs,
    shutdownTimeoutMs: own?.shutdownTimeoutMs ?? inherited?.shutdownTimeoutMs ?? DEFAULTS.shutdownTimeoutMs,
    idleTimeoutMs: own?.idleTimeoutMs ?? inherited?.idleTimeoutMs ?? DEFAULTS.idleTimeoutMs,
    awaitHealthOnStart: own?.awaitHealthOnStart ?? inherited?.awaitHealthOnStart ?? true,
  };
  if (start !== undefined) lifecycle.start = start;
  if (stop !== undefined) lifecycle.stop = stop;
  return lifecycle;
}

/**
 * The health check implied by an adapter kind when a descriptor declares none.
 *
 * Server-backed adapters get an HTTP probe against their own endpoint, which is
 * both the cheapest and the most accurate test available; process-per-request
 * and in-process adapters have nothing to probe.
 *
 * @param adapter - the resolved adapter kind.
 * @returns the default health check.
 */
function defaultHealthFor(adapter: AdapterKind): HealthCheckSpec {
  switch (adapter) {
    case 'http_json':
    case 'openai_compatible':
      return { kind: 'http', path: '/', timeoutMs: DEFAULTS.healthTimeoutMs };
    case 'comfyui':
      // ComfyUI answers `/system_stats` cheaply and without touching a model.
      return { kind: 'http', path: '/system_stats', timeoutMs: DEFAULTS.healthTimeoutMs };
    case 'cli':
    case 'mock':
      return { kind: 'none' };
    default: {
      const exhaustive: never = adapter;
      throw new ModelHubError('INVALID_DESCRIPTOR', `unhandled adapter kind ${String(exhaustive)}`);
    }
  }
}

/**
 * The model-id grammar, exported so the published JSON Schema, this validator,
 * and the schema test all quote one pattern instead of three copies that drift.
 */
export const MODEL_ID_PATTERN = /^[a-z0-9][a-z0-9._-]*$/;

/**
 * Whether a string is a valid model id.
 * @param value - candidate id.
 * @returns true for a lowercase kebab-case id.
 */
export function isValidModelId(value: string): boolean {
  return MODEL_ID_PATTERN.test(value);
}

export { isCapability, isIoType, isModelType };
