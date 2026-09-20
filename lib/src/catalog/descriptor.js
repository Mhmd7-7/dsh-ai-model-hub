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
import { CAPABILITY_IO, isCapability, isIoType, isModelType, } from "./capabilities.js";
import { IssueCollector, formatIssues, isRecord, readOptionalBoolean, readOptionalEnum, readOptionalEnumArray, readOptionalNumber, readOptionalRecord, readOptionalString, readOptionalStringArray, readRequiredString, } from "../util/validate.js";
import { ModelHubError } from "../errors.js";
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
];
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
];
const DEFAULTS = {
    version: '0.0.0',
    priority: 100,
    startupTimeoutMs: 120_000,
    shutdownTimeoutMs: 15_000,
    idleTimeoutMs: 0,
    healthTimeoutMs: 2_000,
};
/**
 * Parse one untrusted value into a {@link ModelHost}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated host, or `undefined` when it cannot be read at all.
 */
export function parseHost(raw, path, collector) {
    if (!isRecord(raw)) {
        collector.add(path, 'must be an object');
        return undefined;
    }
    const id = readRequiredString(raw, 'id', path, collector);
    const name = readRequiredString(raw, 'name', path, collector);
    const adapter = readOptionalEnum(raw, 'adapter', ADAPTER_KINDS, path, collector);
    if (adapter === undefined)
        collector.add(`${path}.adapter`, 'is required');
    const runtime = parseRuntimeSpec(raw['runtime'], `${path}.runtime`, collector, adapter);
    if (id === undefined || name === undefined || adapter === undefined || runtime === undefined) {
        return undefined;
    }
    const host = { id, name, adapter, runtime };
    const lifecycle = parseLifecycleSpec(raw['lifecycle'], `${path}.lifecycle`, collector);
    if (lifecycle !== undefined)
        host.lifecycle = lifecycle;
    const resources = parseResourceSpec(raw['resources'], `${path}.resources`, collector);
    if (resources !== undefined)
        host.resources = resources;
    const health = parseHealthSpec(raw['health'], `${path}.health`, collector);
    if (health !== undefined)
        host.health = health;
    const enabled = readOptionalBoolean(raw, 'enabled', path, collector);
    if (enabled !== undefined)
        host.enabled = enabled;
    const notes = readOptionalString(raw, 'notes', path, collector);
    if (notes !== undefined)
        host.notes = notes;
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
export function parseModelDescriptor(raw, path, collector) {
    if (!isRecord(raw)) {
        collector.add(path, 'must be an object');
        return undefined;
    }
    const id = readRequiredString(raw, 'id', path, collector);
    if (id !== undefined && !isValidModelId(id)) {
        // The id reaches artifact ids, tool output, and log lines, so its character
        // set is constrained rather than merely conventional. Enforcing it here keeps
        // the runtime validator and the published JSON Schema in agreement.
        collector.add(`${path}.id`, `must be lowercase kebab-case matching ${MODEL_ID_PATTERN.source} (got "${id}")`);
    }
    const name = readRequiredString(raw, 'name', path, collector);
    const type = readOptionalEnum(raw, 'type', MODEL_TYPE_VALUES, path, collector);
    if (type === undefined)
        collector.add(`${path}.type`, 'is required');
    const capabilities = readOptionalEnumArray(raw, 'capabilities', CAPABILITY_VALUES, path, collector);
    if (capabilities === undefined)
        collector.add(`${path}.capabilities`, 'is required');
    else if (capabilities.length === 0)
        collector.add(`${path}.capabilities`, 'must name at least one capability');
    if (id === undefined || name === undefined || type === undefined || capabilities === undefined) {
        return undefined;
    }
    const descriptor = { id, name, type, capabilities };
    const inputTypes = readOptionalEnumArray(raw, 'inputTypes', IO_TYPE_VALUES, path, collector);
    if (inputTypes !== undefined)
        descriptor.inputTypes = inputTypes;
    const outputTypes = readOptionalEnumArray(raw, 'outputTypes', IO_TYPE_VALUES, path, collector);
    if (outputTypes !== undefined)
        descriptor.outputTypes = outputTypes;
    const version = readOptionalString(raw, 'version', path, collector);
    if (version !== undefined)
        descriptor.version = version;
    const hostId = readOptionalString(raw, 'host', path, collector);
    if (hostId !== undefined)
        descriptor.host = hostId;
    const adapter = readOptionalEnum(raw, 'adapter', ADAPTER_KINDS, path, collector);
    if (adapter !== undefined)
        descriptor.adapter = adapter;
    const runtime = parseRuntimeSpec(raw['runtime'], `${path}.runtime`, collector, adapter);
    if (runtime !== undefined)
        descriptor.runtime = runtime;
    const adapterConfig = readOptionalRecord(raw, 'adapterConfig', path, collector);
    if (adapterConfig !== undefined)
        descriptor.adapterConfig = adapterConfig;
    const resources = parseResourceSpec(raw['resources'], `${path}.resources`, collector);
    if (resources !== undefined)
        descriptor.resources = resources;
    const limits = parseLimitsSpec(raw['limits'], `${path}.limits`, collector);
    if (limits !== undefined)
        descriptor.limits = limits;
    const lifecycle = parseLifecycleSpec(raw['lifecycle'], `${path}.lifecycle`, collector);
    if (lifecycle !== undefined)
        descriptor.lifecycle = lifecycle;
    const health = parseHealthSpec(raw['health'], `${path}.health`, collector);
    if (health !== undefined)
        descriptor.health = health;
    const enabled = readOptionalBoolean(raw, 'enabled', path, collector);
    if (enabled !== undefined)
        descriptor.enabled = enabled;
    const priority = readOptionalNumber(raw, 'priority', path, collector);
    if (priority !== undefined)
        descriptor.priority = priority;
    const tags = readOptionalStringArray(raw, 'tags', path, collector);
    if (tags !== undefined)
        descriptor.tags = tags;
    const notes = readOptionalString(raw, 'notes', path, collector);
    if (notes !== undefined)
        descriptor.notes = notes;
    return descriptor;
}
const MODEL_TYPE_VALUES = [
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
const CAPABILITY_VALUES = [
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
const IO_TYPE_VALUES = [
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
function parseRuntimeSpec(raw, path, collector, fallbackAdapter) {
    if (raw === undefined)
        return undefined;
    if (!isRecord(raw)) {
        collector.add(path, 'must be an object');
        return undefined;
    }
    const engine = readRequiredString(raw, 'engine', path, collector);
    const adapter = readOptionalEnum(raw, 'adapter', ADAPTER_KINDS, path, collector) ?? fallbackAdapter;
    if (adapter === undefined)
        collector.add(`${path}.adapter`, 'is required');
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
    if (engine === undefined || adapter === undefined)
        return undefined;
    const spec = { engine, adapter };
    if (endpoint !== undefined)
        spec.endpoint = endpoint;
    if (requestPath !== undefined)
        spec.path = requestPath;
    if (modelPath !== undefined)
        spec.modelPath = modelPath;
    if (dataDir !== undefined)
        spec.dataDir = dataDir;
    if (args !== undefined)
        spec.args = args;
    if (env !== undefined) {
        const clean = {};
        for (const [key, value] of Object.entries(env)) {
            if (typeof value === 'string')
                clean[key] = value;
            else
                collector.add(`${path}.env.${key}`, 'must be a string');
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
export function parseResourceSpec(raw, path, collector) {
    if (raw === undefined)
        return undefined;
    if (!isRecord(raw)) {
        collector.add(path, 'must be an object');
        return undefined;
    }
    const spec = {};
    const vramGb = readOptionalNumber(raw, 'vramGb', path, collector);
    if (vramGb !== undefined)
        spec.vramGb = vramGb;
    const ramGb = readOptionalNumber(raw, 'ramGb', path, collector);
    if (ramGb !== undefined)
        spec.ramGb = ramGb;
    const diskGb = readOptionalNumber(raw, 'diskGb', path, collector);
    if (diskGb !== undefined)
        spec.diskGb = diskGb;
    const requiresGpu = readOptionalBoolean(raw, 'requiresGpu', path, collector);
    if (requiresGpu !== undefined)
        spec.requiresGpu = requiresGpu;
    const allowConcurrentInstances = readOptionalBoolean(raw, 'allowConcurrentInstances', path, collector);
    if (allowConcurrentInstances !== undefined)
        spec.allowConcurrentInstances = allowConcurrentInstances;
    return spec;
}
/**
 * Parse an untrusted value into a {@link LimitsSpec}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated spec, or `undefined` when absent.
 */
export function parseLimitsSpec(raw, path, collector) {
    if (raw === undefined)
        return undefined;
    if (!isRecord(raw)) {
        collector.add(path, 'must be an object');
        return undefined;
    }
    const spec = {};
    const contextTokens = readOptionalNumber(raw, 'contextTokens', path, collector);
    if (contextTokens !== undefined)
        spec.contextTokens = contextTokens;
    const maxWidth = readOptionalNumber(raw, 'maxWidth', path, collector);
    if (maxWidth !== undefined)
        spec.maxWidth = maxWidth;
    const maxHeight = readOptionalNumber(raw, 'maxHeight', path, collector);
    if (maxHeight !== undefined)
        spec.maxHeight = maxHeight;
    const maxDurationSeconds = readOptionalNumber(raw, 'maxDurationSeconds', path, collector);
    if (maxDurationSeconds !== undefined)
        spec.maxDurationSeconds = maxDurationSeconds;
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
export function parseLifecycleSpec(raw, path, collector) {
    if (raw === undefined)
        return undefined;
    if (!isRecord(raw)) {
        collector.add(path, 'must be an object');
        return undefined;
    }
    const startable = readOptionalBoolean(raw, 'startable', path, collector) ?? false;
    const stoppable = readOptionalBoolean(raw, 'stoppable', path, collector) ?? false;
    const spec = { startable, stoppable };
    const start = parseCommandSpec(raw['start'], `${path}.start`, collector);
    if (start !== undefined)
        spec.start = start;
    const stop = parseCommandSpec(raw['stop'], `${path}.stop`, collector);
    if (stop !== undefined)
        spec.stop = stop;
    const startupTimeoutMs = readOptionalNumber(raw, 'startupTimeoutMs', path, collector);
    if (startupTimeoutMs !== undefined)
        spec.startupTimeoutMs = startupTimeoutMs;
    const shutdownTimeoutMs = readOptionalNumber(raw, 'shutdownTimeoutMs', path, collector);
    if (shutdownTimeoutMs !== undefined)
        spec.shutdownTimeoutMs = shutdownTimeoutMs;
    const idleTimeoutMs = readOptionalNumber(raw, 'idleTimeoutMs', path, collector);
    if (idleTimeoutMs !== undefined)
        spec.idleTimeoutMs = idleTimeoutMs;
    const awaitHealthOnStart = readOptionalBoolean(raw, 'awaitHealthOnStart', path, collector);
    if (awaitHealthOnStart !== undefined)
        spec.awaitHealthOnStart = awaitHealthOnStart;
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
function parseCommandSpec(raw, path, collector) {
    if (raw === undefined)
        return undefined;
    if (!isRecord(raw)) {
        collector.add(path, 'must be an object');
        return undefined;
    }
    const command = readRequiredString(raw, 'command', path, collector);
    const args = readOptionalStringArray(raw, 'args', path, collector);
    const cwd = readOptionalString(raw, 'cwd', path, collector);
    if (command === undefined)
        return undefined;
    const spec = { command };
    if (args !== undefined)
        spec.args = args;
    if (cwd !== undefined)
        spec.cwd = cwd;
    return spec;
}
/**
 * Parse an untrusted value into a {@link HealthCheckSpec}.
 * @param raw - the parsed JSON value.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated spec, or `undefined` when absent.
 */
function parseHealthSpec(raw, path, collector) {
    if (raw === undefined)
        return undefined;
    if (!isRecord(raw)) {
        collector.add(path, 'must be an object');
        return undefined;
    }
    const kind = readOptionalEnum(raw, 'kind', HEALTH_CHECK_KINDS, path, collector);
    if (kind === undefined) {
        collector.add(`${path}.kind`, 'is required');
        return undefined;
    }
    const spec = { kind };
    const healthPath = readOptionalString(raw, 'path', path, collector);
    if (healthPath !== undefined)
        spec.path = healthPath;
    const command = readOptionalString(raw, 'command', path, collector);
    if (command !== undefined)
        spec.command = command;
    const args = readOptionalStringArray(raw, 'args', path, collector);
    if (args !== undefined)
        spec.args = args;
    const timeoutMs = readOptionalNumber(raw, 'timeoutMs', path, collector);
    if (timeoutMs !== undefined)
        spec.timeoutMs = timeoutMs;
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
export function parseModelCatalogConfig(raw, label = 'model catalog') {
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
    const hosts = [];
    const hostsRaw = raw['hosts'];
    if (hostsRaw !== undefined) {
        if (!Array.isArray(hostsRaw)) {
            collector.add('$.hosts', 'must be an array');
        }
        else {
            hostsRaw.forEach((entry, index) => {
                const host = parseHost(entry, `hosts[${index}]`, collector);
                if (host !== undefined)
                    hosts.push(host);
            });
        }
    }
    const models = [];
    modelsRaw.forEach((entry, index) => {
        const descriptor = parseModelDescriptor(entry, `models[${index}]`, collector);
        if (descriptor !== undefined)
            models.push(descriptor);
    });
    validateCatalogCrossReferences(models, hosts, collector);
    if (collector.failed) {
        return { ok: false, message: formatIssues(label, collector.collected), issues: collector.collected };
    }
    const config = { hosts, models };
    if (version !== undefined)
        config.version = version;
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
function validateCatalogCrossReferences(models, hosts, collector) {
    const hostIds = new Set();
    hosts.forEach((host, index) => {
        if (hostIds.has(host.id))
            collector.add(`hosts[${index}].id`, `duplicate host id "${host.id}"`);
        hostIds.add(host.id);
    });
    const modelIds = new Set();
    models.forEach((model, index) => {
        const path = `models[${index}]`;
        if (modelIds.has(model.id))
            collector.add(`${path}.id`, `duplicate model id "${model.id}"`);
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
            const expected = new Set();
            for (const capability of model.capabilities)
                for (const io of CAPABILITY_IO[capability].input)
                    expected.add(io);
            for (const declared of declaredInputs) {
                if (!expected.has(declared)) {
                    collector.add(`${path}.inputTypes`, `"${declared}" is not an input of any declared capability (${[...expected].join(', ') || 'none'})`);
                }
            }
        }
        const declaredOutputs = model.outputTypes;
        if (declaredOutputs !== undefined) {
            const expected = new Set();
            for (const capability of model.capabilities)
                for (const io of CAPABILITY_IO[capability].output)
                    expected.add(io);
            for (const declared of declaredOutputs) {
                if (!expected.has(declared)) {
                    collector.add(`${path}.outputTypes`, `"${declared}" is not an output of any declared capability (${[...expected].join(', ') || 'none'})`);
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
export function resolveDescriptor(descriptor, host) {
    if (descriptor.host !== undefined && descriptor.runtime === undefined && host === undefined) {
        throw new ModelHubError('INVALID_DESCRIPTOR', `model "${descriptor.id}" references host "${descriptor.host}", which is not defined`, { modelId: descriptor.id, hostId: descriptor.host });
    }
    if (descriptor.host !== undefined && descriptor.runtime !== undefined) {
        throw new ModelHubError('INVALID_DESCRIPTOR', `model "${descriptor.id}" declares both \`host\` and \`runtime\`; declare exactly one`, { modelId: descriptor.id });
    }
    const runtime = descriptor.runtime ?? host?.runtime;
    if (runtime === undefined) {
        throw new ModelHubError('INVALID_DESCRIPTOR', `model "${descriptor.id}" has no runtime: declare \`host\` or an inline \`runtime\``, { modelId: descriptor.id });
    }
    const adapter = descriptor.adapter ?? host?.adapter ?? runtime.adapter;
    const descriptorResources = descriptor.resources ?? {};
    const hostResources = host?.resources ?? {};
    const resources = {
        vramGb: descriptorResources.vramGb ?? hostResources.vramGb ?? 0,
        ramGb: descriptorResources.ramGb ?? hostResources.ramGb ?? 0,
        diskGb: descriptorResources.diskGb ?? hostResources.diskGb ?? 0,
        requiresGpu: descriptorResources.requiresGpu ?? hostResources.requiresGpu ?? false,
        allowConcurrentInstances: descriptorResources.allowConcurrentInstances ?? hostResources.allowConcurrentInstances ?? false,
    };
    const lifecycle = resolveLifecycle(descriptor, host);
    const health = descriptor.health ?? host?.health ?? defaultHealthFor(adapter);
    const inputTypes = descriptor.inputTypes !== undefined
        ? [...descriptor.inputTypes]
        : [...new Set(descriptor.capabilities.flatMap((capability) => CAPABILITY_IO[capability].input))];
    const outputTypes = descriptor.outputTypes !== undefined
        ? [...descriptor.outputTypes]
        : [...new Set(descriptor.capabilities.flatMap((capability) => CAPABILITY_IO[capability].output))];
    const resolved = {
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
    if (descriptor.host !== undefined)
        resolved.hostId = descriptor.host;
    return resolved;
}
/**
 * Merge a descriptor's lifecycle with its host's, applying defaults.
 * @param descriptor - the model as written.
 * @param host - its host, when any.
 * @returns the fully-defaulted lifecycle.
 */
function resolveLifecycle(descriptor, host) {
    const own = descriptor.lifecycle;
    const inherited = host?.lifecycle;
    const startable = own?.startable ?? inherited?.startable ?? false;
    const stoppable = own?.stoppable ?? inherited?.stoppable ?? startable;
    const start = own?.start ?? inherited?.start;
    const stop = own?.stop ?? inherited?.stop;
    const lifecycle = {
        startable: startable && start !== undefined,
        stoppable: stoppable && (start !== undefined || stop !== undefined),
        startupTimeoutMs: own?.startupTimeoutMs ?? inherited?.startupTimeoutMs ?? DEFAULTS.startupTimeoutMs,
        shutdownTimeoutMs: own?.shutdownTimeoutMs ?? inherited?.shutdownTimeoutMs ?? DEFAULTS.shutdownTimeoutMs,
        idleTimeoutMs: own?.idleTimeoutMs ?? inherited?.idleTimeoutMs ?? DEFAULTS.idleTimeoutMs,
        awaitHealthOnStart: own?.awaitHealthOnStart ?? inherited?.awaitHealthOnStart ?? true,
    };
    if (start !== undefined)
        lifecycle.start = start;
    if (stop !== undefined)
        lifecycle.stop = stop;
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
function defaultHealthFor(adapter) {
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
            const exhaustive = adapter;
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
export function isValidModelId(value) {
    return MODEL_ID_PATTERN.test(value);
}
export { isCapability, isIoType, isModelType };
