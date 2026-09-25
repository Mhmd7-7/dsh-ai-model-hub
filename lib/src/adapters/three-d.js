/**
 * The `three_d` adapter: real local image/text → 3D asset generation.
 *
 * What it speaks
 * --------------
 * There is no single 3D-generation protocol, and pretending otherwise is how a
 * hub ends up supporting exactly one engine. What the local 3D ecosystem *does*
 * share is a shape, and this adapter implements that shape:
 *
 * 1. the engine runs as a local HTTP server (Gradio app, FastAPI app, or a
 *    bespoke Flask demo — all three are common in published model repositories);
 * 2. a generation takes tens of seconds to minutes, so it is either synchronous
 *    or a job that is submitted and then polled;
 * 3. the result is named rather than streamed — a server-side path or URL, or
 *    occasionally inline base64;
 * 4. the thing worth keeping is a mesh file; anything else the engine returns
 *    (a turntable video, a Gaussian splat, a preview render) is a companion.
 *
 * So the adapter is parameterised by a small declarative *protocol description*
 * in `adapterConfig`, and the engine-specific knowledge lives in the catalog
 * entry rather than in this file. Two transports are supported:
 *
 * - `gradio`: the queue API that Gradio 4/5 exposes at `/gradio_api/call/<name>`
 *   and Gradio 3 at `/call/<name>`. A submit returns an event id, the result is
 *   read from `/gradio_api/call/<name>/<event_id>` as a server-sent-event stream.
 *   This covers the official TRELLIS, Hunyuan3D, Stable Fast 3D, and TripoSR
 *   demos, which are all Gradio apps.
 * - `http_json`: POST a JSON body to a path, optionally poll a status path, read
 *   a JSON result. This covers the FastAPI servers published alongside those
 *   same models.
 *
 * Multi-step engines
 * ------------------
 * Some engines deliberately split "generate" from "export": TRELLIS produces an
 * in-memory state and a preview video first, and only turns that state into a GLB
 * on a second call. That is a real workflow, not a quirk, and the first call's
 * result is a large opaque object that must be handed to the second call
 * verbatim. So a model may declare several `steps`, each of which receives the
 * accumulated results of the ones before it through `$N` bindings. The mesh is
 * whichever step declares `resultFormat`.
 *
 * What the agent sees
 * -------------------
 * None of the above. The adapter takes a capability and artifacts and returns
 * artifacts; the transport, the API names, the bindings, and the polling are all
 * invisible above this file, which is the point.
 *
 * @module dsh-ai-model-hub/adapters/three-d
 */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { extname, resolve } from 'node:path';
import { measureThreeD, sniffThreeDFormat, threeDFormatInfo, threeDFormatOf } from "../artifacts/formats.js";
import { ModelHubError } from "../errors.js";
import { isLosslessJson, isRecord } from "../util/validate.js";
import { withTimeout } from "../util/process.js";
/** Capabilities this adapter can serve. Anything else is refused up front. */
const THREE_D_CAPABILITIES = ['image_to_3d', 'text_to_3d'];
/**
 * Default budget for one generation, in milliseconds.
 *
 * Fifteen minutes is not padding. A 12-step TRELLIS run plus GLB extraction takes
 * minutes on a laptop GPU, and the first call of a session additionally pays for
 * loading several gigabytes of weights off disk. A tighter default would turn
 * "slow but working" into "mysteriously times out", which is the single most
 * common way a local 3D setup is reported as broken.
 */
const DEFAULT_TIMEOUT_MS = 900_000;
/** Default budget for one HTTP request inside a generation, in milliseconds. */
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;
/** Response bodies quoted into error messages are bounded to this many characters. */
const MAX_ERROR_BODY_CHARS = 400;
/** How many bytes of a mesh are inspected when sniffing its container. */
const SNIFF_PREFIX_BYTES = 4096;
/** The API-description routes a Gradio app may answer on, most modern first. */
const GRADIO_CONFIG_PATHS = ['/gradio_api/config', '/config'];
/**
 * Validate a model's `adapterConfig` for this adapter, synchronously.
 *
 * This is the up-front half of configuration resolution, and it exists because
 * {@link ModelAdapter.supports} is synchronous by contract: a catalog mistake must
 * become a clear `unsupported` status at inspection time, not a failure halfway
 * through a generation. It checks everything that can be checked without I/O —
 * which is everything except loading an external step-declaration file.
 *
 * @param model - the resolved model.
 * @returns the resolved configuration, with the steps file not yet applied.
 * @throws ModelHubError with `CONFIG_ERROR` naming the first problem found.
 */
function resolveConfigSync(model) {
    const config = model.adapterConfig;
    const where = `model "${model.id}" (adapter ${model.adapter})`;
    const protocol = requireProtocol(config, where);
    const endpoint = requireEndpoint(model, config, where);
    const steps = resolveSteps(config, endpoint, where, protocol);
    const inputModeRaw = optionalString(config, 'inputMode') ?? 'data_uri';
    if (inputModeRaw !== 'data_uri' && inputModeRaw !== 'base64' && inputModeRaw !== 'path') {
        throw configError(`${where}: adapterConfig.inputMode must be data_uri, base64, or path (got "${inputModeRaw}")`, {
            modelId: model.id,
            field: 'inputMode',
            value: inputModeRaw,
        });
    }
    const timeoutMs = positiveNumber(config, 'timeoutMs') ?? DEFAULT_TIMEOUT_MS;
    const requestTimeoutMs = positiveNumber(config, 'requestTimeoutMs') ?? DEFAULT_REQUEST_TIMEOUT_MS;
    // Read eagerly so a malformed value is reported here; the resolved record is
    // built from the same reads below.
    const extraBody = readJsonRecord(config['extraBody'], 'extraBody', where);
    const generationParameters = readJsonRecord(config['generationParameters'], 'generationParameters', where);
    return {
        protocol,
        endpoint,
        steps,
        inputMode: inputModeRaw,
        filePath: optionalString(config, 'filePath'),
        healthPath: optionalString(config, 'healthPath'),
        timeoutMs,
        requestTimeoutMs,
        extraBody,
        generationParameters,
        previewIndex: optionalNumber(config, 'previewIndex'),
    };
}
/**
 * Resolve a model's configuration, loading an external step declaration when the
 * catalog names one.
 *
 * The synchronous half has already run — `supports` calls it — so everything that
 * could be wrong without touching the filesystem has been reported by now, and
 * this path only adds the file. A missing or malformed file is still a
 * configuration error, and it is reported here rather than at the first
 * generation because `health` goes through this too.
 *
 * @param model - the resolved model.
 * @returns the resolved configuration.
 * @throws ModelHubError with `CONFIG_ERROR` when the step file is unusable.
 */
async function resolveConfig(model) {
    const where = `model "${model.id}" (adapter ${model.adapter})`;
    if (optionalString(model.adapterConfig, 'stepsPath') === undefined)
        return resolveConfigSync(model);
    const merged = await readStepsFile(model, model.adapterConfig, where);
    // Re-resolve through a model whose adapter config is the merged document, so
    // there is exactly one implementation of "what do these fields mean".
    const withSteps = { ...model, adapterConfig: merged };
    return resolveConfigSync(withSteps);
}
/**
 * Load a step declaration file, when the catalog points at one.
 *
 * A multi-step protocol is the most involved thing a 3D catalog entry carries,
 * and the repository's precedent for that kind of thing is a file rather than an
 * inline blob — the same reason ComfyUI entries name a workflow template. The
 * file holds *only* the step declarations; endpoint, lifecycle, and resources
 * stay in the catalog, because those are deployment facts while the steps are a
 * property of the engine's API.
 *
 * @param model - the resolved model.
 * @param config - the adapter configuration.
 * @param where - a label for error messages.
 * @returns the configuration with the file's steps merged in, or a reason the
 *   file could not be used.
 * @throws ModelHubError when the file is named but unreadable or malformed.
 */
async function readStepsFile(model, config, where) {
    const raw = optionalString(config, 'stepsPath');
    if (raw === undefined)
        return config;
    if (config['steps'] !== undefined) {
        throw configError(`${where}: declare either adapterConfig.steps or adapterConfig.stepsPath, not both`, { modelId: model.id });
    }
    const file = resolve(raw);
    let text;
    try {
        text = await readFile(file, 'utf8');
    }
    catch (error) {
        throw configError(`${where}: could not read the 3D step declaration at ${file}: ${error instanceof Error ? error.message : String(error)}`, { modelId: model.id, stepsPath: file });
    }
    let parsed;
    try {
        parsed = JSON.parse(text.charCodeAt(0) === 0xfeff ? text.slice(1) : text);
    }
    catch (error) {
        throw configError(`${where}: the 3D step declaration at ${file} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`, { modelId: model.id, stepsPath: file });
    }
    if (!isRecord(parsed)) {
        throw configError(`${where}: the 3D step declaration at ${file} must be a JSON object`, {
            modelId: model.id,
            stepsPath: file,
        });
    }
    // A `$comment`-style key is common in this repository's config files; the
    // steps are accepted either at the top level or under `steps`.
    const steps = parsed['steps'] ?? parsed;
    // Generation defaults may live in the same file — they belong with the argument
    // order they feed — but the catalog wins where both state the same key, because
    // the catalog is the per-deployment statement and the file is the shared recipe.
    const merged = { ...config, steps };
    if (config['generationParameters'] === undefined && parsed['generationParameters'] !== undefined) {
        merged['generationParameters'] = parsed['generationParameters'];
    }
    return merged;
}
/**
 * Read and validate the transport name.
 * @param config - the adapter configuration.
 * @param where - a label for error messages.
 * @returns the protocol.
 * @throws ModelHubError when absent or unknown.
 */
function requireProtocol(config, where) {
    const value = optionalString(config, 'protocol') ?? 'gradio';
    if (value !== 'gradio' && value !== 'http_json') {
        throw configError(`${where}: adapterConfig.protocol must be "gradio" or "http_json" (got "${value}"). ` +
            'Gradio app servers (TRELLIS, Hunyuan3D, Stable Fast 3D, TripoSR) use "gradio"; ' +
            'FastAPI/Flask servers that take a JSON body use "http_json".', { field: 'protocol', value });
    }
    return value;
}
/**
 * Resolve the engine's base URL.
 *
 * The endpoint may come from `adapterConfig.endpoint` or from the descriptor's
 * `runtime.endpoint`; the adapter config wins because it is the more specific
 * statement. One of them must be present — an HTTP adapter with nowhere to send a
 * request is a configuration error, not a runtime surprise.
 *
 * @param model - the resolved model.
 * @param config - the adapter configuration.
 * @param where - a label for error messages.
 * @returns the normalized base URL without a trailing slash.
 * @throws ModelHubError when neither source supplies one.
 */
function requireEndpoint(model, config, where) {
    const raw = optionalString(config, 'endpoint') ?? model.runtime.endpoint;
    if (raw === undefined || raw.trim().length === 0) {
        throw configError(`${where}: no endpoint. Set adapterConfig.endpoint or runtime.endpoint to the engine's base URL, ` +
            'e.g. "http://127.0.0.1:8080".', { modelId: model.id });
    }
    let parsed;
    try {
        parsed = new URL(raw);
    }
    catch {
        throw configError(`${where}: endpoint "${raw}" is not an absolute URL`, { modelId: model.id, endpoint: raw });
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
        throw configError(`${where}: endpoint "${raw}" must use http or https`, {
            modelId: model.id,
            endpoint: raw,
        });
    }
    return raw.replace(/\/+$/, '');
}
/**
 * Resolve the generation steps.
 *
 * A model may declare `steps` for a multi-call engine, `apiName` for the common
 * single-call case, or neither for an `http_json` server whose route is the
 * runtime's own `path`. Every one of those is a legitimate catalog entry, so all
 * three are accepted and normalized to one list.
 *
 * Every field a step can carry is validated here rather than where it is used.
 * That matters for the same reason the rest of this resolution is up-front: a
 * catalog entry with a mistyped binding must be reported as a configuration
 * problem when the model is inspected, not as a mysterious failure inside one
 * invocation — and a hub that only validates lazily reports the *second* mistake
 * on the second run.
 *
 * @param config - the adapter configuration.
 * @param endpoint - the base URL.
 * @param where - a label for error messages.
 * @param protocol - the transport, which decides whether an apiName is required.
 * @returns the steps, in execution order, with exactly one marked primary.
 * @throws ModelHubError when the step declarations are unusable.
 */
function resolveSteps(config, endpoint, where, protocol) {
    const rawSteps = config['steps'];
    const entries = rawSteps === undefined
        ? [config]
        : Array.isArray(rawSteps)
            ? rawSteps
            : (() => {
                throw configError(`${where}: adapterConfig.steps must be an array`, { field: 'steps' });
            })();
    if (entries.length === 0) {
        throw configError(`${where}: adapterConfig.steps must name at least one generation call`, { field: 'steps' });
    }
    const steps = entries.map((entry, index) => {
        const stepPath = rawSteps === undefined ? where : `${where} step ${index}`;
        if (!isRecord(entry)) {
            throw configError(`${stepPath}: each step must be an object`, { step: index });
        }
        const formatValue = optionalString(entry, 'resultFormat');
        const resultFormat = formatValue === undefined ? undefined : threeDFormatOf(formatValue);
        if (formatValue !== undefined && resultFormat === undefined) {
            throw configError(`${stepPath}: resultFormat "${formatValue}" is not a 3D format this hub knows (glb, gltf, obj, stl, ply)`, { step: index, resultFormat: formatValue });
        }
        const apiName = optionalString(entry, 'apiName');
        if (protocol === 'gradio' && apiName === undefined && config['stepsPath'] === undefined) {
            // A `stepsPath` is the documented escape from this check: the api names
            // live in that file, and `supports` must stay synchronous, so it cannot read
            // them. The file's contents are validated in full on the `health` and
            // `invoke` paths, which is where a bad one has to be caught anyway.
            throw configError(`${stepPath}: the gradio protocol needs an apiName for every step — the name Gradio shows in the app's ` +
                'API docs, e.g. "image_to_3d". Without it there is no route to submit the job to.', { step: index, field: 'apiName' });
        }
        const bind = readBindings(entry['bind'], stepPath);
        validateBindings(bind, index, stepPath);
        return {
            index,
            apiName,
            endpoint: (optionalString(entry, 'endpoint') ?? endpoint).replace(/\/+$/, ''),
            bind,
            extraArgs: readJsonArray(entry['extraArgs'], 'extraArgs', stepPath),
            primary: false,
            resultFormat,
            method: (optionalString(entry, 'method') ?? 'POST').toUpperCase(),
            resultAt: optionalString(entry, 'resultAt'),
        };
    });
    // The mesh is the step that declares a 3D format; failing that, the last one.
    // Choosing by declaration rather than by position is what lets an engine whose
    // export call comes *before* its preview call work without a special case.
    const declared = steps.findIndex((step) => step.resultFormat !== undefined);
    const primaryIndex = declared >= 0 ? declared : steps.length - 1;
    return steps.map((step) => (step.index === primaryIndex ? { ...step, primary: true } : step));
}
/**
 * Check that a step's binding expressions are well formed.
 *
 * Only the *syntax* is checked here. Whether `$1.0` actually resolves is a
 * question about a running engine, so it is answered at invocation time — but a
 * binding that could never resolve, because it names a future step or is not a
 * path at all, is a catalog mistake and is caught now.
 *
 * @param bind - the argument bindings.
 * @param stepIndex - the step's 0-based position.
 * @param where - a label for error messages.
 * @throws ModelHubError when an expression is unusable.
 */
function validateBindings(bind, stepIndex, where) {
    for (const [name, value] of Object.entries(bind)) {
        // A string that is not a `$`-expression is a literal argument value, exactly
        // as a number or an object is. Only the expressions are checked.
        if (typeof value !== 'string' || !value.startsWith('$'))
            continue;
        const expression = value;
        if (expression === '$input' || expression === '$image')
            continue;
        if (expression.startsWith('$param:')) {
            if (expression.slice('$param:'.length).trim().length === 0) {
                throw configError(`${where}: bind.${name} names a request option but does not say which one`, {
                    step: stepIndex,
                    argument: name,
                });
            }
            continue;
        }
        if (!expression.startsWith('$')) {
            throw configError(`${where}: bind.${name} must start with "$" — use $input for the request's image, $param:<key> for a ` +
                `request option, or $N[.M] for an earlier step's output. Got "${expression}".`, { step: stepIndex, argument: name, expression });
        }
        const [stepPart, ...rest] = expression.slice(1).split('.');
        const referenced = Number(stepPart);
        if (!Number.isInteger(referenced) || referenced < 0) {
            throw configError(`${where}: bind.${name} names step "${stepPart}", which is not a step index`, { step: stepIndex, argument: name, expression });
        }
        if (referenced >= stepIndex) {
            throw configError(`${where}: bind.${name} names step ${referenced}, which has not run yet (this is step ${stepIndex}). ` +
                'A step can only bind to results from earlier steps.', { step: stepIndex, argument: name, expression });
        }
        if (rest.length > 0 && !isWellFormedPath(rest.join('.'))) {
            throw configError(`${where}: bind.${name} indexes into step ${referenced}'s result with "${rest.join('.')}", ` +
                'which is not a property name or array index', { step: stepIndex, argument: name, expression });
        }
    }
}
/**
 * Read a step's argument bindings.
 *
 * A binding value is either a `$`-expression or a literal. Literals matter more
 * than they look: an engine's call signature has parameters a hub has no business
 * inventing values for — TRELLIS's `multiimages: []` and `is_multiimage: false`
 * are the app's own switches, not settings — so a step declaration states them
 * verbatim, and only the arguments worth exposing become `$param:` bindings.
 *
 * @param raw - the raw `bind` value.
 * @param where - a label for error messages.
 * @returns the bindings, keyed by argument name.
 * @throws ModelHubError when the value is not a JSON object of literal values.
 */
function readBindings(raw, where) {
    if (raw === undefined)
        return {};
    if (!isRecord(raw) || !isLosslessJson(raw)) {
        throw configError(`${where}: bind must be an object mapping argument names to sources or literal values`, {
            field: 'bind',
        });
    }
    return raw;
}
/**
 * Read a JSON-safe record setting.
 * @param raw - the raw value.
 * @param field - the field name, for messages.
 * @param where - a label for error messages.
 * @returns the record, or an empty object when absent.
 * @throws ModelHubError when the value is not a lossless-JSON object.
 */
function readJsonRecord(raw, field, where) {
    if (raw === undefined)
        return {};
    if (!isRecord(raw) || !isLosslessJson(raw)) {
        throw configError(`${where}: ${field} must be a JSON object of literal values`, { field });
    }
    return raw;
}
/**
 * Read a JSON-safe array setting.
 * @param raw - the raw value.
 * @param field - the field name, for messages.
 * @param where - a label for error messages.
 * @returns the array, or an empty array when absent.
 * @throws ModelHubError when the value is not a lossless-JSON array.
 */
function readJsonArray(raw, field, where) {
    if (raw === undefined)
        return [];
    if (!Array.isArray(raw) || !isLosslessJson(raw)) {
        throw configError(`${where}: ${field} must be an array of literal values`, { field });
    }
    return raw;
}
/**
 * Read an optional non-empty string setting.
 * @param config - the source object.
 * @param key - the field name.
 * @returns the string, or `undefined`.
 */
function optionalString(config, key) {
    const value = config[key];
    return typeof value === 'string' && value.trim().length > 0 ? value : undefined;
}
/**
 * Read an optional finite number setting.
 * @param config - the source object.
 * @param key - the field name.
 * @returns the number, or `undefined`.
 */
function optionalNumber(config, key) {
    const value = config[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}
/**
 * Read an optional strictly positive number setting.
 * @param config - the source object.
 * @param key - the field name.
 * @returns the number, or `undefined`.
 */
function positiveNumber(config, key) {
    const value = optionalNumber(config, key);
    return value !== undefined && value > 0 ? value : undefined;
}
/**
 * Build a configuration error.
 * @param message - the human-readable problem.
 * @param details - structured context.
 * @returns the error to throw.
 */
function configError(message, details = {}) {
    return new ModelHubError('CONFIG_ERROR', message, details);
}
/**
 * Apply a `$`-path to a value.
 *
 * Paths are the vocabulary the bindings and `resultAt` share: `0` and `data` read
 * a property, dots descend, and `$0.1` addresses the second result of the first
 * step. A `$` prefix means "an earlier step's results"; a bare path means "this
 * response".
 *
 * @param root - the value to descend into.
 * @param path - the path expression.
 * @returns the addressed value, or `undefined` when any segment is absent.
 */
function pathGet(root, path) {
    let current = root;
    for (const segment of path.split('.')) {
        if (segment.length === 0)
            continue;
        if (Array.isArray(current)) {
            const index = Number(segment);
            if (!Number.isInteger(index) || index < 0 || index >= current.length)
                return undefined;
            current = current[index];
            continue;
        }
        if (isRecord(current)) {
            if (!Object.prototype.hasOwnProperty.call(current, segment))
                return undefined;
            current = current[segment];
            continue;
        }
        return undefined;
    }
    return current;
}
/**
 * Validate that a path expression is well formed.
 * @param path - the expression.
 * @returns true when every segment is a property name or array index.
 */
function isWellFormedPath(path) {
    if (path.trim().length === 0)
        return false;
    return path
        .split('.')
        .filter((segment) => segment.length > 0)
        .every((segment) => /^[A-Za-z0-9_$-]+$/.test(segment));
}
/**
 * Render a value for an error message without dumping a large object.
 * @param value - the value.
 * @returns a bounded description.
 */
function describeValue(value) {
    if (value === undefined)
        return 'undefined';
    if (value === null)
        return 'null';
    if (typeof value === 'string') {
        return value.length <= 120 ? JSON.stringify(value) : `${JSON.stringify(value.slice(0, 117))}...`;
    }
    if (typeof value === 'number' || typeof value === 'boolean')
        return String(value);
    if (Array.isArray(value))
        return `array(${value.length})`;
    return 'object';
}
/**
 * The `three_d` adapter.
 * @returns the adapter, ready to register.
 */
export function createThreeDAdapter() {
    return {
        kind: 'three_d',
        displayName: 'Local 3D generation (HTTP engine)',
        supports(model) {
            try {
                resolveConfigSync(model);
                return { ok: true };
            }
            catch (error) {
                return { ok: false, reason: error instanceof Error ? error.message : String(error) };
            }
        },
        /**
         * Probe the engine's HTTP surface.
         *
         * For `gradio` the API-description document is the one thing every Gradio app
         * serves and which proves the process is not merely listening but has an API —
         * a bare TCP accept says nothing about whether the app finished loading its
         * weights. Which route it lives on depends on the Gradio version
         * (`/gradio_api/config` on Gradio 5, `/config` on 3 and 4) and on the app's own
         * root path, so both are tried rather than making every catalog entry state
         * which Gradio it runs. `adapterConfig.healthPath` overrides both for an
         * engine mounted under a prefix.
         *
         * For `http_json` the configured `healthPath` (default `/`) is used.
         *
         * @param model - the resolved model.
         * @param signal - cancellation for the probe.
         * @returns the probe outcome; never throws.
         */
        async health(model, signal) {
            const started = Date.now();
            let config;
            try {
                config = await resolveConfig(model);
            }
            catch (error) {
                return {
                    healthy: false,
                    checkedAt: started,
                    detail: error instanceof Error ? error.message : String(error),
                };
            }
            const budget = Math.max(1_000, model.health.timeoutMs ?? 2_000);
            const explicit = optionalString(model.adapterConfig, 'healthPath');
            const paths = explicit !== undefined
                ? [explicit]
                : config.protocol === 'gradio'
                    ? [...GRADIO_CONFIG_PATHS]
                    : [config.healthPath ?? '/'];
            const failures = [];
            for (const path of paths) {
                const url = joinUrl(config.endpoint, path);
                const read = await request({ url, method: 'GET', timeoutMs: budget, signal });
                if (!read.ok) {
                    failures.push(read.reason);
                    continue;
                }
                if (config.protocol !== 'gradio') {
                    return { healthy: true, checkedAt: started, latencyMs: Date.now() - started, detail: `engine reachable at ${config.endpoint}${path}` };
                }
                let parsed;
                try {
                    parsed = JSON.parse(read.body);
                }
                catch {
                    failures.push(`${url} answered with something that is not JSON; is that really a Gradio app?`);
                    continue;
                }
                if (!isRecord(parsed) || parsed['named_endpoints'] === undefined) {
                    failures.push(`${url} is reachable but does not look like a Gradio app config`);
                    continue;
                }
                return {
                    healthy: true,
                    checkedAt: started,
                    latencyMs: Date.now() - started,
                    detail: `engine reachable at ${url}`,
                };
            }
            return {
                healthy: false,
                checkedAt: started,
                latencyMs: Date.now() - started,
                detail: failures.join('; ') || `no health route answered at ${config.endpoint}`,
            };
        },
        /**
         * Generate a 3D asset.
         * @param invocation - the resolved request.
         * @returns the mesh artifact, plus any companion the engine produced.
         * @throws ModelHubError with a stable code; the hub wraps unknown throws.
         */
        async invoke(invocation) {
            const { model } = invocation;
            if (!THREE_D_CAPABILITIES.includes(invocation.capability)) {
                throw new ModelHubError('UNSUPPORTED_OPERATION', `the three_d adapter serves ${THREE_D_CAPABILITIES.join(' and ')}, not "${invocation.capability}"`, { modelId: model.id, capability: invocation.capability });
            }
            const config = await resolveConfig(model);
            const source = requireInputImage(invocation);
            const timeoutMs = invocationTimeout(invocation, config);
            const log = invocation.log;
            log.info(`three_d: ${invocation.capability} via ${config.protocol} at ${config.endpoint} (${config.steps.length} step(s))`, { modelId: model.id });
            // The `$N` namespace: index N holds the positional outputs of step N. It is
            // a list of lists rather than a flat list so that a step returning a tuple
            // keeps its shape and `$0.1` means what it looks like.
            const results = [];
            // A generation is abandoned, not merely reported as failed: when the
            // caller's budget expires or the caller cancels, the in-flight request has
            // to be torn down too. Without this the request would keep a socket — and,
            // on the engine side, a GPU — busy for its own much longer timeout, long
            // after the answer stopped mattering.
            const cancel = new AbortController();
            const onOuterAbort = () => cancel.abort();
            if (invocation.signal.aborted)
                cancel.abort();
            else
                invocation.signal.addEventListener('abort', onOuterAbort, { once: true });
            const scoped = { ...invocation, signal: cancel.signal };
            const run = async () => {
                let mesh;
                for (const step of config.steps) {
                    const args = await buildArguments(step, results, source, scoped, config);
                    log.debug(`three_d: step ${step.index} -> ${step.apiName ?? step.endpoint}`, { args: args.length });
                    const outputs = await runStep(step, args, config, scoped, log);
                    log.debug(`three_d: step ${step.index} returned ${outputs.length} output(s)`, {});
                    results.push(outputs);
                    if (step.primary) {
                        const payload = await extractMesh(step, outputs, config, scoped, log);
                        if (payload !== undefined)
                            mesh = payload;
                    }
                }
                return mesh;
            };
            let mesh;
            try {
                mesh = await withTimeout(run(), timeoutMs, `3D generation on ${model.id}`, invocation.signal);
            }
            catch (error) {
                cancel.abort();
                throw error;
            }
            finally {
                invocation.signal.removeEventListener('abort', onOuterAbort);
            }
            if (mesh === undefined) {
                throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" completed without producing a 3D asset. ` +
                    'The engine answered, but nothing in its response could be read as a mesh file — ' +
                    'check that adapterConfig.resultAt points at the right field and that the engine actually ' +
                    'generated a model.', { modelId: model.id, endpoint: config.endpoint, steps: config.steps.length });
            }
            const artifacts = [];
            const primary = await storeMesh(mesh, scoped, source, config, results);
            artifacts.push(primary);
            const preview = await storePreview(scoped, config, results);
            if (preview !== undefined)
                artifacts.push(preview);
            const format = mesh.format ?? mesh.declaredFormat;
            return {
                outputs: artifacts,
                value: {
                    format: format ?? 'unknown',
                    ...(primary.mimeType === undefined ? {} : { mimeType: primary.mimeType }),
                    ...(primary.byteLength === undefined ? {} : { byteLength: primary.byteLength }),
                    sourceArtifactId: source.id,
                    ...(primary.metadata['vertexCount'] === undefined
                        ? {}
                        : { vertexCount: primary.metadata['vertexCount'] }),
                    ...(primary.metadata['triangleCount'] === undefined
                        ? {}
                        : { triangleCount: primary.metadata['triangleCount'] }),
                    ...(mesh.warning === undefined ? {} : { warning: mesh.warning }),
                },
            };
        },
    };
}
/**
 * Require the input image an `image_to_3d` request must carry.
 *
 * Checked here as well as in the router because the router filters by *declared*
 * input kinds while this checks the actual artifact: a catalog entry that omits
 * `inputTypes` would otherwise reach an engine that cannot cope with the absence.
 *
 * @param invocation - the resolved request.
 * @returns the input image artifact.
 * @throws ModelHubError when no image was supplied.
 */
function requireInputImage(invocation) {
    const image = invocation.inputs.find((artifact) => artifact.type === 'image');
    if (image === undefined) {
        throw new ModelHubError('INVOCATION_FAILED', `"${invocation.capability}" needs an input artifact of type \`image\`; ` +
            `this request supplied ${invocation.inputs.length === 0 ? 'none' : invocation.inputs.map((artifact) => artifact.type).join(', ')}. ` +
            'Pass the image artifact id in `inputs`, or generate one first with text_to_image.', { modelId: invocation.model.id, kind: invocation.capability });
    }
    return image;
}
/**
 * Resolve the invocation's time budget.
 *
 * The caller's explicit budget wins, then the descriptor's
 * `adapterConfig.timeoutMs`, then the adapter's own generous default. A caller
 * asking for less than the declared budget is trusted: it is their generation.
 *
 * Both spellings are honoured — the request field the hub forwards as
 * {@link AdapterInvocation.timeoutMs} and the same key inside `options` — because
 * `options` is passed through to adapters unchanged, so a caller that wrote
 * `{ options: { timeoutMs: … } }` plainly meant the same thing.
 *
 * @param invocation - the resolved request.
 * @param config - the resolved adapter configuration.
 * @returns the budget in milliseconds.
 */
function invocationTimeout(invocation, config) {
    for (const candidate of [invocation.timeoutMs, invocation.options['timeoutMs']]) {
        if (typeof candidate === 'number' && Number.isFinite(candidate) && candidate > 0)
            return candidate;
    }
    return config.timeoutMs;
}
/**
 * Build one step's argument list.
 *
 * Argument order is the engine's, so it is declared rather than inferred: named
 * `bind` entries are emitted in declaration order, and a step that declares no
 * bindings falls back to `imageArg` followed by `extraArgs`. That fallback exists
 * because the overwhelmingly common single-call shape is "first argument is the
 * image, then the knobs", and making every catalog entry spell that out would be
 * noise.
 *
 * @param step - the step to build for.
 * @param results - the accumulated outputs of earlier steps.
 * @param source - the input image artifact.
 * @param invocation - the resolved request.
 * @param config - the resolved adapter configuration.
 * @returns the positional arguments, in order.
 * @throws ModelHubError when a binding cannot be resolved.
 */
async function buildArguments(step, results, source, invocation, config) {
    const args = [];
    for (const [name, bound] of Object.entries(step.bind)) {
        // Only a `$`-expression is resolved; any other value — including a plain
        // string — is a literal the declaration stated verbatim. That distinction is
        // what lets a step declare an engine's own switch (`multiimage_algo:
        // "stochastic"`) without the hub inventing a meaning for it.
        args.push(typeof bound === 'string' && bound.startsWith('$')
            ? await resolveBinding(name, bound, results, source, invocation, config)
            : bound);
    }
    if (Object.keys(step.bind).length === 0 && step.index === 0) {
        args.push(await encodeInputImage(source, invocation, config));
    }
    for (const extra of step.extraArgs)
        args.push(extra);
    if (Object.keys(step.bind).length === 0 && step.index > 0) {
        // A later step with no bindings: hand it the previous step's outputs, which
        // is what an engine that splits generate/export expects.
        const previous = results[results.length - 1] ?? [];
        for (const value of previous)
            args.push(toJsonValue(value, step.index));
    }
    return args;
}
/**
 * Resolve one binding expression to an argument value.
 *
 * @param name - the argument name, for messages.
 * @param expression - `$input` for the request's image, `$N[.M]` for an earlier
 *   step's output, or `$param:<key>` for a value from the request's options.
 * @param results - the accumulated outputs.
 * @param source - the input image artifact.
 * @param invocation - the resolved request.
 * @param config - the resolved adapter configuration.
 * @returns the argument value.
 * @throws ModelHubError when the expression is malformed or cannot be resolved.
 */
async function resolveBinding(name, expression, results, source, invocation, config) {
    if (expression === '$input' || expression === '$image') {
        return encodeInputImage(source, invocation, config);
    }
    if (expression.startsWith('$param:')) {
        const key = expression.slice('$param:'.length);
        // Generation defaults are exactly what a `$param` binding is for: the catalog
        // entry (or a shipped step declaration) states the value the engine should run
        // with, and a caller overrides it per request. Reading the resolved
        // `generationParameters` rather than `extraBody` is what makes a template's
        // own defaults usable without the caller having to supply every knob.
        const value = invocation.options[key] ?? config.generationParameters[key];
        if (value === undefined) {
            throw new ModelHubError('INVOCATION_FAILED', `argument "${name}" is bound to the request option "${key}", which was supplied neither by the caller ` +
                'nor as a default in the model\'s adapterConfig.generationParameters', { argument: name, option: key });
        }
        return toJsonValue(value, -1);
    }
    if (!expression.startsWith('$')) {
        throw configError(`binding for argument "${name}" must start with "$" ($input, $param:<key>, or $N[.M]); got "${expression}". ` +
            'A literal argument value does not need a binding at all — put it in `bind` directly.', { argument: name, expression });
    }
    const path = expression.slice(1);
    const [stepPart, ...rest] = path.split('.');
    const stepIndex = Number(stepPart);
    if (!Number.isInteger(stepIndex) || stepIndex < 0) {
        throw configError(`binding for argument "${name}" names step "${stepPart}", which is not a step index`, { argument: name, expression });
    }
    const stepResults = results[stepIndex];
    if (stepResults === undefined) {
        throw configError(`binding for argument "${name}" names step ${stepIndex}, which has not run yet ` +
            `(this step is number ${results.length})`, { argument: name, expression, availableSteps: results.length });
    }
    const value = rest.length === 0 ? stepResults[0] : pathGet(stepResults, rest.join('.'));
    if (value === undefined) {
        throw new ModelHubError('INVOCATION_FAILED', `binding for argument "${name}" resolved "${expression}" to nothing in step ${stepIndex}'s response ` +
            `(that step returned ${stepResults.length} output(s))`, { argument: name, expression, stepOutputs: stepResults.length });
    }
    return toJsonValue(value, stepIndex);
}
/**
 * Convert an engine value into something safe to send back in a request body.
 *
 * An engine's opaque state object — TRELLIS returns one — round-trips through
 * `JSON.parse(JSON.stringify(...))` intact because it arrived as JSON in the
 * first place. A value that does not survive that round trip is not sendable, and
 * saying so is better than sending `{}` and letting the engine fail obscurely.
 *
 * @param value - the engine value.
 * @param stepIndex - where it came from, for messages.
 * @returns the JSON-safe value.
 * @throws ModelHubError when the value cannot be sent back.
 */
function toJsonValue(value, stepIndex) {
    if (isLosslessJson(value))
        return value;
    throw new ModelHubError('INVOCATION_FAILED', `a value from step ${stepIndex < 0 ? 'the request' : stepIndex} cannot be passed to the engine: ` +
        'only JSON-serializable data can cross an engine boundary', { stepIndex });
}
/**
 * Encode the input image the way the engine expects to receive it.
 *
 * Three encodings cover the ecosystem: a `data:` URI (what Gradio accepts for a
 * file component and what most JSON APIs parse), bare base64, and a
 * server-visible filesystem path. The `path` mode is a deliberate escape hatch
 * for an engine that shares a filesystem with the hub, and is documented as such
 * because it means the artifact store must be readable by that engine.
 *
 * @param source - the input image artifact.
 * @param invocation - the resolved request.
 * @param config - the resolved adapter configuration.
 * @returns the encoded image.
 * @throws ModelHubError when the artifact cannot be resolved or read.
 */
async function encodeInputImage(source, invocation, config) {
    const { path } = await invocation.artifacts.resolvePath(source.id);
    if (config.inputMode === 'path')
        return path;
    const bytes = await readFile(path);
    const mime = source.mimeType ?? 'image/png';
    const base64 = bytes.toString('base64');
    return config.inputMode === 'base64' ? base64 : `data:${mime};base64,${base64}`;
}
/**
 * Execute one generation step and return its positional outputs.
 *
 * @param step - the step to run.
 * @param args - its positional arguments.
 * @param config - the resolved adapter configuration.
 * @param invocation - the resolved request.
 * @param log - the diagnostics sink.
 * @returns the step's outputs, in engine order.
 * @throws ModelHubError when the engine refuses the call or answers unusably.
 */
async function runStep(step, args, config, invocation, log) {
    if (config.protocol === 'gradio') {
        if (step.apiName === undefined) {
            throw configError(`step ${step.index}: the gradio protocol needs an apiName (the name Gradio shows in its API docs, e.g. "image_to_3d")`, { step: step.index });
        }
        return runGradioStep(step, args, config, invocation, log);
    }
    return runHttpJsonStep(step, args, config, invocation, log);
}
/**
 * Run one step against a Gradio app's queue API.
 *
 * The two-request dance is Gradio's, not ours: `POST …/call/<name>` with the
 * arguments enqueues the job and answers with an event id, and
 * `GET …/call/<name>/<event_id>` streams the result as server-sent events. Both
 * Gradio 5 (`/gradio_api/call/...`) and Gradio 3/4 (`/call/...`) spellings are
 * tried, in that order, because a deployment's Gradio version is not something the
 * hub can know in advance.
 *
 * @param step - the step to run.
 * @param args - its positional arguments.
 * @param config - the resolved adapter configuration.
 * @param invocation - the resolved request.
 * @param log - the diagnostics sink.
 * @returns the step's outputs.
 * @throws ModelHubError when the queue refuses the job or reports an error.
 */
async function runGradioStep(step, args, config, invocation, log) {
    const apiName = step.apiName ?? '';
    const prefixes = ['/gradio_api', ''];
    let lastReason = 'no attempt was made';
    for (const prefix of prefixes) {
        const submitUrl = joinUrl(step.endpoint, `${prefix}/call/${encodeURIComponent(apiName)}`);
        const body = JSON.stringify({ data: args });
        const submitted = await request({
            url: submitUrl,
            method: 'POST',
            body,
            headers: { 'content-type': 'application/json' },
            timeoutMs: config.requestTimeoutMs,
            signal: invocation.signal,
        });
        if (!submitted.ok) {
            lastReason = submitted.reason;
            // A 404 means this prefix is the wrong spelling, not that the engine is
            // broken; the next one gets a turn. Anything else is reported as-is.
            if (submitted.status === 404)
                continue;
            throw engineError(submitUrl, lastReason, invocation.model.id);
        }
        if (submitted.status === 404)
            continue;
        const eventId = extractEventId(submitted.body);
        if (eventId === undefined) {
            throw new ModelHubError('INVOCATION_FAILED', `the engine at ${submitUrl} accepted the request but returned no event id, so its result cannot be collected. ` +
                `Response began: ${bounded(submitted.body)}`, { modelId: invocation.model.id, url: submitUrl });
        }
        const resultUrl = joinUrl(step.endpoint, `${prefix}/call/${encodeURIComponent(apiName)}/${encodeURIComponent(eventId)}`);
        log.debug(`three_d: polling ${resultUrl}`, {});
        const streamed = await request({
            url: resultUrl,
            method: 'GET',
            headers: { accept: 'text/event-stream' },
            // The generation itself is bounded by the caller's budget; this request's
            // own timeout is only a backstop so a dead connection cannot hang forever.
            timeoutMs: Math.max(config.requestTimeoutMs, 1),
            signal: invocation.signal,
        });
        if (!streamed.ok)
            throw engineError(resultUrl, streamed.reason, invocation.model.id);
        const events = parseSse(streamed.body);
        if (events.error !== undefined) {
            throw new ModelHubError('INVOCATION_FAILED', `the engine reported an error while generating: ${bounded(events.error)}`, { modelId: invocation.model.id, step: step.index });
        }
        if (events.complete === undefined) {
            throw new ModelHubError('INVOCATION_FAILED', `the engine's result stream for "${apiName}" ended without a completion event`, { modelId: invocation.model.id, step: step.index, events: events.seen });
        }
        let parsed;
        try {
            parsed = JSON.parse(events.complete);
        }
        catch {
            throw new ModelHubError('INVOCATION_FAILED', `the engine's completion event for "${apiName}" was not JSON: ${bounded(events.complete)}`, { modelId: invocation.model.id, step: step.index });
        }
        const outputs = isRecord(parsed) && Array.isArray(parsed['data']) ? parsed['data'] : [parsed];
        if (step.resultAt !== undefined)
            return [pathGet(outputs, step.resultAt) ?? pathGet(parsed, step.resultAt)];
        return outputs;
    }
    throw engineError(joinUrl(step.endpoint, `/call/${apiName}`), lastReason, invocation.model.id);
}
/**
 * Run one step against a JSON HTTP route.
 *
 * The body is `{ data: [...args], ...extraBody }` — Gradio's own envelope, which
 * the FastAPI servers published with these models also accept because they were
 * written to mirror it — unless the catalog entry declares `bodyField: 'args'`,
 * in which case the arguments travel under that name.
 *
 * @param step - the step to run.
 * @param args - its positional arguments.
 * @param config - the resolved adapter configuration.
 * @param invocation - the resolved request.
 * @param log - the diagnostics sink.
 * @returns the step's outputs.
 * @throws ModelHubError when the engine refuses the call or reports an error.
 */
async function runHttpJsonStep(step, args, config, invocation, log) {
    const url = step.apiName === undefined ? step.endpoint : joinUrl(step.endpoint, step.apiName);
    const bodyField = optionalString(invocation.model.adapterConfig, 'bodyField');
    const body = JSON.stringify({
        ...(bodyField === undefined ? { data: args } : { [bodyField]: args }),
        ...config.extraBody,
    });
    const read = await request({
        url,
        method: step.method,
        body,
        headers: { 'content-type': 'application/json' },
        timeoutMs: config.requestTimeoutMs,
        signal: invocation.signal,
    });
    if (!read.ok)
        throw engineError(url, read.reason, invocation.model.id);
    log.debug(`three_d: ${step.method} ${url} -> ${read.status}`, {});
    let parsed;
    try {
        parsed = JSON.parse(read.body);
    }
    catch {
        throw new ModelHubError('INVOCATION_FAILED', `the engine at ${url} answered with something that is not JSON: ${bounded(read.body)}`, { modelId: invocation.model.id, url });
    }
    if (step.resultAt !== undefined) {
        const addressed = pathGet(parsed, step.resultAt);
        if (addressed === undefined) {
            throw new ModelHubError('INVOCATION_FAILED', `resultAt "${step.resultAt}" matched nothing in the engine's response from ${url}. ` +
                `Response keys: ${Object.keys(isRecord(parsed) ? parsed : {}).join(', ') || '(none)'}`, { modelId: invocation.model.id, url, resultAt: step.resultAt });
        }
        return [addressed];
    }
    if (isRecord(parsed) && Array.isArray(parsed['data']))
        return parsed['data'];
    return [parsed];
}
/**
 * Pull Gradio's event id out of a submit response.
 * @param body - the response body.
 * @returns the id, or `undefined`.
 */
function extractEventId(body) {
    try {
        const parsed = JSON.parse(body);
        if (isRecord(parsed) && typeof parsed['event_id'] === 'string' && parsed['event_id'].length > 0) {
            return parsed['event_id'];
        }
    }
    catch {
        /* not JSON: fall through to the plain-text form */
    }
    const match = /event_id:\s*([^\s]+)/.exec(body);
    return match?.[1];
}
/**
 * Parse a server-sent-event stream into the events that matter.
 *
 * Gradio emits `event: <type>` followed by `data: <json>` lines. Only `complete`
 * and `error` change what happens next, so the rest — heartbeats, progress,
 * `generating` — are counted and ignored. The stream also carries an `error`
 * *event type* whose payload may be `null` on success, which is why a null
 * payload is not treated as a failure.
 *
 * @param body - the raw stream text.
 * @returns the completion payload, any error message, and the event count.
 */
function parseSse(body) {
    let current;
    let complete;
    let error;
    let seen = 0;
    for (const rawLine of body.split(/\r?\n/)) {
        const line = rawLine.trimEnd();
        if (line.startsWith('event:')) {
            current = line.slice('event:'.length).trim();
            continue;
        }
        if (!line.startsWith('data:'))
            continue;
        const payload = line.slice('data:'.length).trim();
        seen += 1;
        if (current === 'complete' && complete === undefined)
            complete = payload;
        if (current === 'error' && error === undefined && payload !== 'null' && payload.length > 0)
            error = payload;
    }
    return {
        ...(complete === undefined ? {} : { complete }),
        ...(error === undefined ? {} : { error }),
        seen,
    };
}
/**
 * Turn one step's outputs into mesh bytes.
 *
 * The engine names the mesh rather than sending it, so this is where a
 * server-side path, a URL, or an inline base64 payload becomes bytes. A path that
 * does not exist locally is retried through the file endpoint, which is what makes
 * a containerized or remote engine work without a second code path.
 *
 * @param step - the primary step.
 * @param outputs - its outputs.
 * @param config - the resolved adapter configuration.
 * @param invocation - the resolved request.
 * @param log - the diagnostics sink.
 * @returns the mesh payload, or `undefined` when no output looked like one.
 * @throws ModelHubError when a mesh was identified but could not be read.
 */
async function extractMesh(step, outputs, config, invocation, log) {
    const candidates = collectFileCandidates(outputs, step.resultFormat);
    const preferred = candidates.find((candidate) => step.resultFormat !== undefined && candidate.format === step.resultFormat) ??
        candidates.find((candidate) => candidate.format !== undefined) ??
        candidates[0];
    if (preferred === undefined) {
        log.warn(`three_d: step ${step.index} produced no file-like output`, {
            outputs: outputs.length,
        });
        return undefined;
    }
    const bytes = await readCandidate(preferred, config, invocation, step);
    const claimed = preferred.format ?? step.resultFormat;
    const sniffed = sniffThreeDFormat(bytes.subarray(0, SNIFF_PREFIX_BYTES), claimed);
    return {
        bytes,
        format: sniffed.format ?? claimed,
        declaredFormat: claimed,
        warning: sniffed.warning,
    };
}
/**
 * Find everything in a response that could be a generated file.
 *
 * Engines report results as a bare path string, as `{ path }`, as
 * `{ url, name }`, or as a data URI; the reader is deliberately total because the
 * cost of missing a candidate is a failed generation while the cost of an extra
 * candidate is one `undefined` check.
 *
 * A bare string is only a candidate when it names a container the hub recognises,
 * which is what stops a Gradio video path or a status message being mistaken for
 * the mesh. `anyFile` lifts that restriction for a caller that already knows what
 * it is looking for — the preview route, where the value is known to be a file and
 * whose *format* is not the hub's to judge.
 *
 * @param outputs - the parts of the response to search.
 * @param declared - the format the step declared, used to prefer a candidate.
 * @param anyFile - accept any non-empty string as a location.
 * @returns the candidates, in encounter order.
 */
function collectFileCandidates(outputs, declared, anyFile = false) {
    const found = [];
    const visit = (value, label, depth) => {
        if (depth > 3)
            return;
        if (typeof value === 'string') {
            const candidate = describeString(value, label, declared, anyFile);
            if (candidate !== undefined)
                found.push(candidate);
            return;
        }
        if (Array.isArray(value)) {
            value.forEach((entry, index) => visit(entry, `${label}.${index}`, depth + 1));
            return;
        }
        if (!isRecord(value))
            return;
        const path = typeof value['path'] === 'string' ? value['path'] : undefined;
        const url = typeof value['url'] === 'string' ? value['url'] : undefined;
        const name = typeof value['name'] === 'string' ? value['name'] : label;
        const mime = typeof value['mime_type'] === 'string' ? value['mime_type'] : undefined;
        if (path !== undefined || url !== undefined) {
            found.push({
                label: name,
                location: path ?? url ?? '',
                format: threeDFormatOf(mime ?? name) ?? declared,
            });
        }
        for (const [key, entry] of Object.entries(value)) {
            if (key === 'path' || key === 'url' || key === 'name' || key === 'mime_type')
                continue;
            if (typeof entry === 'string' || Array.isArray(entry) || isRecord(entry))
                visit(entry, `${label}.${key}`, depth + 1);
        }
    };
    outputs.forEach((value, index) => visit(value, `${index}`, 0));
    return found;
}
/**
 * Describe one string as a file candidate, when it looks like a reference to one.
 * @param value - the string.
 * @param label - a label for messages.
 * @param declared - the format the step declared.
 * @param anyFile - accept any non-empty string, not only a recognised container.
 * @returns the candidate, or `undefined` when the string is ordinary data.
 */
function describeString(value, label, declared, anyFile = false) {
    const trimmed = value.trim();
    if (trimmed.length === 0)
        return undefined;
    if (/^data:/.test(trimmed)) {
        const comma = trimmed.indexOf(',');
        if (comma < 0)
            return undefined;
        return {
            label,
            base64: trimmed.slice(comma + 1),
            format: threeDFormatOf(trimmed.slice(5, comma)) ?? declared,
        };
    }
    if (/^https?:\/\//.test(trimmed)) {
        const format = threeDFormatOf(extname(new URL(trimmed).pathname)) ?? declared;
        return { label, location: trimmed, format };
    }
    // A bare path: only interesting when it names a 3D container, otherwise a
    // Gradio video path or a status string would be mistaken for the mesh. A caller
    // that knows better — the preview route — says so with `anyFile`.
    const format = threeDFormatOf(extname(trimmed));
    if (format === undefined) {
        return anyFile ? { label, location: trimmed, format: declared } : undefined;
    }
    return { label, location: trimmed, format };
}
/**
 * Read one candidate's bytes.
 *
 * @param candidate - the file to read.
 * @param config - the resolved adapter configuration.
 * @param invocation - the resolved request.
 * @param step - the step that produced it.
 * @returns the content.
 * @throws ModelHubError when a mesh was found but could not be read.
 */
async function readCandidate(candidate, config, invocation, step) {
    if (candidate.base64 !== undefined) {
        return new Uint8Array(Buffer.from(candidate.base64, 'base64'));
    }
    const location = candidate.location ?? '';
    if (/^https?:\/\//.test(location)) {
        const read = await request({
            url: location,
            method: 'GET',
            timeoutMs: config.requestTimeoutMs,
            signal: invocation.signal,
            binary: true,
        });
        if (!read.ok)
            throw engineError(location, read.reason, invocation.model.id);
        return new Uint8Array(Buffer.from(read.body, 'base64'));
    }
    // A server-side path. Gradio hands back absolute paths under its own temp
    // directory; when the hub shares that filesystem the read is direct, and when
    // it does not the file endpoint is the documented way to fetch the same bytes.
    try {
        const bytes = await readFile(location);
        return new Uint8Array(bytes);
    }
    catch (error) {
        const filePath = config.filePath ?? (config.protocol === 'gradio' ? '/gradio_api/file=' : undefined);
        if (filePath === undefined) {
            throw new ModelHubError('INVOCATION_FAILED', `the engine reported its result at "${location}", which this machine cannot read ` +
                `(${error instanceof Error ? error.message : String(error)}). ` +
                'If the engine runs in a container or on another host, set adapterConfig.filePath to its file-download route.', { modelId: invocation.model.id, location, step: step.index });
        }
        const url = filePath.startsWith('http')
            ? `${filePath}${encodeURIComponent(location)}`
            : joinUrl(config.endpoint, `${filePath}${encodeURIComponent(location)}`);
        const read = await request({
            url,
            method: 'GET',
            timeoutMs: config.requestTimeoutMs,
            signal: invocation.signal,
            binary: true,
        });
        if (!read.ok) {
            throw new ModelHubError('INVOCATION_FAILED', `the engine reported its result at "${location}" but it could not be read locally or fetched from ${url}: ${read.reason}`, { modelId: invocation.model.id, location, url, step: step.index });
        }
        return new Uint8Array(Buffer.from(read.body, 'base64'));
    }
}
/**
 * Persist the mesh as a `model_3d` artifact.
 *
 * @param mesh - the generated content and what was measured about it.
 * @param invocation - the resolved request.
 * @param source - the input image artifact.
 * @param config - the resolved adapter configuration.
 * @param results - the accumulated engine outputs, recorded for provenance.
 * @returns the stored artifact.
 */
async function storeMesh(mesh, invocation, source, config, results) {
    const info = mesh.format === undefined ? undefined : threeDFormatInfo(mesh.format);
    const measured = measureThreeD(mesh.bytes, mesh.format);
    const digest = createHash('sha256').update(mesh.bytes).digest('hex').slice(0, 16);
    const parameters = { ...config.generationParameters };
    if (config.previewIndex !== undefined)
        parameters['previewIndex'] = config.previewIndex;
    return invocation.artifacts.put({
        type: 'model_3d',
        bytes: mesh.bytes,
        mimeType: info?.mimeType ?? 'application/octet-stream',
        label: `${invocation.model.name} mesh`,
        producerModelId: invocation.model.id,
        extension: info?.extension ?? '.bin',
        metadata: {
            format: mesh.format ?? 'unknown',
            mimeType: info?.mimeType ?? 'application/octet-stream',
            byteLength: mesh.bytes.byteLength,
            createdAt: Date.now(),
            sourceArtifactId: source.id,
            sourceModelId: invocation.model.id,
            sourceHash: `sha256:${digest}`,
            generationParameters: parameters,
            protocol: config.protocol,
            stepCount: config.steps.length,
            ...(measured.vertexCount === undefined ? {} : { vertexCount: measured.vertexCount }),
            ...(measured.triangleCount === undefined ? {} : { triangleCount: measured.triangleCount }),
            ...(mesh.warning === undefined ? {} : { validationWarning: mesh.warning }),
            engineOutputs: results.length,
            prompt: invocation.prompt ?? '',
        },
    });
}
/**
 * Persist the engine's preview render as a companion artifact, when configured.
 *
 * A turntable video is the only practical way to look at a mesh without a 3D
 * viewer, so an engine that produces one is worth capturing — but it is a
 * companion, never the mesh, and it is stored as a `file` artifact rather than as
 * `video` so that a downstream model filtering on input kinds cannot mistake a
 * preview for something it can edit.
 *
 * A preview is a convenience, so nothing here is allowed to fail a generation that
 * already produced a usable mesh: a preview that cannot be read is logged and
 * dropped. What it is *not* allowed to do is fail silently — a deployment that
 * configured `previewIndex` and got nothing deserves the reason in the log.
 *
 * @param invocation - the resolved request.
 * @param config - the resolved adapter configuration.
 * @param results - the accumulated engine outputs.
 * @returns the preview artifact, or `undefined` when there is none.
 */
async function storePreview(invocation, config, results) {
    if (config.previewIndex === undefined)
        return undefined;
    const value = resolvePreviewValue(results, config.previewIndex);
    if (value === undefined) {
        invocation.log.warn(`three_d: previewIndex ${config.previewIndex} matched no engine output (${results.flat().length} available)`, { modelId: invocation.model.id });
        return undefined;
    }
    const candidates = collectFileCandidates([value], undefined, true);
    const candidate = candidates.find((entry) => entry.location !== undefined);
    if (candidate === undefined)
        return undefined;
    const location = candidate.location ?? '';
    try {
        const bytes = /^https?:\/\//.test(location)
            ? await (async () => {
                const read = await request({
                    url: location,
                    method: 'GET',
                    timeoutMs: config.requestTimeoutMs,
                    signal: invocation.signal,
                    binary: true,
                });
                return read.ok ? new Uint8Array(Buffer.from(read.body, 'base64')) : undefined;
            })()
            : new Uint8Array(await readFile(location));
        if (bytes === undefined) {
            invocation.log.warn(`three_d: the engine preview at ${location} could not be downloaded`, {
                modelId: invocation.model.id,
            });
            return undefined;
        }
        return await invocation.artifacts.put({
            type: 'file',
            bytes,
            mimeType: mimeForExtension(extname(location)),
            label: `${invocation.model.name} preview`,
            producerModelId: invocation.model.id,
            extension: extname(location).length > 0 ? extname(location) : '.bin',
            metadata: {
                role: 'preview',
                format: extname(location).replace(/^\./, '') || 'unknown',
                byteLength: bytes.byteLength,
                createdAt: Date.now(),
            },
        });
    }
    catch (error) {
        invocation.log.warn(`three_d: the engine preview at ${location} could not be stored (${error instanceof Error ? error.message : String(error)})`, { modelId: invocation.model.id });
        return undefined;
    }
}
/**
 * Find the value a `previewIndex` points at.
 *
 * @param results - the accumulated engine outputs.
 * @param index - the positional index into the flattened outputs.
 * @returns the value, or `undefined`.
 */
function resolvePreviewValue(results, index) {
    const flat = results.flat();
    return index >= 0 && index < flat.length ? flat[index] : undefined;
}
/**
 * A MIME type for a preview file, from its extension.
 * @param extension - the extension, with its dot.
 * @returns the MIME type, or `application/octet-stream`.
 */
function mimeForExtension(extension) {
    const table = {
        '.mp4': 'video/mp4',
        '.webm': 'video/webm',
        '.gif': 'image/gif',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.ply': 'model/ply',
    };
    return table[extension.toLowerCase()] ?? 'application/octet-stream';
}
/**
 * Join a base URL and a path without doubling or dropping the separator.
 * @param base - the base URL, without a trailing slash.
 * @param path - the path, with or without a leading slash.
 * @returns the absolute URL.
 */
function joinUrl(base, path) {
    if (/^https?:\/\//.test(path))
        return path;
    return `${base.replace(/\/+$/, '')}${path.startsWith('/') ? path : `/${path}`}`;
}
/**
 * Perform one HTTP request, containing every failure mode.
 *
 * A non-2xx is a result rather than an exception so the two Gradio URL spellings
 * can be tried in order; everything else — DNS failure, refused connection,
 * timeout, caller cancellation — is reported as a reason string. Never throws.
 *
 * @param options - the request.
 * @returns the response body, or why it could not be read.
 */
async function request(options) {
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    if (options.signal.aborted)
        controller.abort();
    else
        options.signal.addEventListener('abort', onAbort, { once: true });
    const timer = setTimeout(() => controller.abort(), options.timeoutMs);
    try {
        const response = await fetch(options.url, {
            method: options.method,
            signal: controller.signal,
            headers: { accept: options.binary === true ? '*/*' : 'application/json', ...options.headers },
            ...(options.body === undefined ? {} : { body: options.body }),
        });
        const contentType = response.headers.get('content-type') ?? undefined;
        if (options.binary === true) {
            if (!response.ok)
                return { ok: false, status: response.status, reason: `HTTP ${response.status} from ${options.url}`, aborted: false };
            const buffer = Buffer.from(await response.arrayBuffer());
            return { ok: true, status: response.status, body: buffer.toString('base64'), contentType };
        }
        const text = await response.text();
        if (!response.ok) {
            return {
                ok: false,
                status: response.status,
                reason: `HTTP ${response.status} from ${options.url}${text.trim().length === 0 ? '' : `: ${bounded(text)}`}`,
                aborted: false,
            };
        }
        return { ok: true, status: response.status, body: text, contentType };
    }
    catch (error) {
        if (controller.signal.aborted) {
            return { ok: false, status: 0, reason: `request to ${options.url} was aborted or timed out`, aborted: true };
        }
        return {
            ok: false,
            status: 0,
            reason: `could not reach ${options.url}: ${error instanceof Error ? error.message : String(error)}`,
            aborted: false,
        };
    }
    finally {
        clearTimeout(timer);
        options.signal.removeEventListener('abort', onAbort);
    }
}
/**
 * Build the error for an engine that refused or could not be reached.
 *
 * Cancellation is mapped to `INVOCATION_ABORTED` rather than to a generation
 * failure, because the difference matters to the hub: an aborted call is the
 * caller's decision and must never be retried on another model, while a refused
 * call may legitimately fall back.
 *
 * @param url - what was called.
 * @param reason - why it failed.
 * @param modelId - the model being invoked.
 * @returns the error to throw.
 */
function engineError(url, reason, modelId) {
    const code = /aborted or timed out/.test(reason) ? 'INVOCATION_TIMEOUT' : 'INVOCATION_FAILED';
    return new ModelHubError(code, `the 3D engine at ${url} could not serve the request: ${reason}`, {
        modelId,
        url,
        reason,
    });
}
/**
 * Bound a string for inclusion in a message.
 * @param value - the text.
 * @returns the text, truncated with an ellipsis when it was longer.
 */
function bounded(value) {
    const trimmed = value.trim();
    return trimmed.length <= MAX_ERROR_BODY_CHARS ? trimmed : `${trimmed.slice(0, MAX_ERROR_BODY_CHARS)}…`;
}
