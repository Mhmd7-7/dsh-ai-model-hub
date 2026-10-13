/**
 * The "Local models" scan: what this machine can actually be asked to do.
 *
 * The settings page's Scan button runs this, and it answers one question with two
 * independent halves:
 *
 * 1. **Ollama** — every model the configured instance has installed, read from
 *    its own `/api/tags`.
 * 2. **ComfyUI** — every *complete workflow* it can find, each one a single
 *    selectable unit. Never a checkpoint, a LoRA, a VAE, a ControlNet, a text
 *    encoder, or an individual node: those are implementation details of a
 *    workflow and stay inside it.
 *
 * ## Readiness is engine-specific, and status comes before parsing
 *
 * A bare `GET /` is not a health check. Ollama answers it with the plain text
 * `Ollama is running`, and ComfyUI answers it with the web UI's HTML — so a probe
 * that treats "root returned 200" as "parse it as JSON" reports both engines as
 * stopped while they are running perfectly. Each engine is therefore asked the
 * question it can actually answer: Ollama `/api/tags`, ComfyUI `/system_stats`.
 *
 * Every response is also classified before it is trusted. A connection refusal, an
 * HTTP error, and a 200 carrying something that is not JSON are three different
 * problems with three different fixes, and reporting them as one — as an earlier
 * version did, calling a 404 "invalid JSON" — sends an operator looking for a
 * broken file that was never broken.
 *
 * ## Retrieving a workflow means knowing the API's shape
 *
 * ComfyUI lists saved workflows as paths relative to a directory
 * (`GET /userdata?dir=workflows&recurse=true` → `["3d/my_model.json"]`) and serves
 * one back through a *single* path segment (`GET /userdata/{file}`). The
 * directory therefore has to be part of the address, and the whole relative path
 * has to be percent-encoded — `/userdata/workflows%2F3d%2Fmy_model.json`. Sending
 * the filename alone, or leaving the separators literal, produces a 404 for every
 * workflow in a subdirectory. This module asks the documented shape first and
 * falls back to the alternatives, so a different ComfyUI version still works.
 *
 * Scanning is observational: it never starts or stops engines. Live APIs, local
 * workflows and the active catalog are independent sources; failures retain prior
 * observations with explicit freshness and availability, scoped to this hub/config.
 *
 * @module dsh-ai-model-hub/dsh-plugin/scan
 */
import { readFile, readdir, stat } from 'node:fs/promises';
import { join, relative, basename, dirname } from 'node:path';
import { resolveAdapterPath } from "../src/adapters/paths.js";
import { homedir } from 'node:os';
import { DISCOVERED_PRIORITY, ioForCapabilities, readComfyNodeIo, scannedOllamaModelId, scannedWorkflowId, scanWorkflowDocument, stableDigest, workflowModelTypeFor, } from "../src/index.js";
/** The default request budget; a scan should not stall the settings page. */
const DEFAULT_TIMEOUT_MS = 4_000;
/** How deep the workflow directory is walked. */
const MAX_DIRECTORY_DEPTH = 6;
/** How many workflow files one directory scan will read. */
const MAX_WORKFLOW_FILES = 200;
/** The directory saved workflows live under in ComfyUI's user data. */
const WORKFLOW_DIR = 'workflows';
/**
 * The route prefix each engine answers a readiness question on.
 *
 * The bare root is deliberately absent from the useful entries: it is not a health
 * check for either engine. Ollama answers `/` with the text `Ollama is running`,
 * and ComfyUI answers `/` with the editor's HTML, so probing it and parsing JSON
 * reports a healthy engine as stopped.
 */
const ENGINE_READY_PATH = [
    { engine: /ollama/i, path: '/api/tags' },
    { engine: /comfyui/i, path: '/system_stats' },
    { engine: /a1111|forge|stable-diffusion/i, path: '/sdapi/v1/options' },
    { engine: /trellis|hunyuan|sf3d|three/i, path: '/gradio_api/config' },
];
/**
 * The readiness path for a host's engine.
 * @param engine - the engine label.
 * @returns an absolute path.
 */
export function readyPathFor(engine) {
    for (const rule of ENGINE_READY_PATH) {
        if (rule.engine.test(engine))
            return rule.path;
    }
    return '/';
}
// Weak ownership prevents both cross-hub leaks and keeping disposed hubs alive.
let scanStates = new WeakMap();
function stateFor(hub) {
    let state = scanStates.get(hub);
    if (state === undefined) {
        state = { cache: new Map(), generations: new Map(), publicationGeneration: 0 };
        scanStates.set(hub, state);
    }
    return state;
}
function beginSource(state, key) {
    const generation = (state.generations.get(key) ?? 0) + 1;
    state.generations.set(key, generation);
    return generation;
}
/** Bound every injected effect, including synchronous throws and ignored timeouts. */
async function bounded(effect, timeoutMs, label) {
    let timer;
    try {
        return await Promise.race([
            Promise.resolve().then(effect),
            new Promise((_, reject) => {
                timer = setTimeout(() => reject(new Error(label + ' timed out after ' + timeoutMs + ' ms')), timeoutMs);
            }),
        ]);
    }
    finally {
        if (timer !== undefined)
            clearTimeout(timer);
    }
}
function boundedProbes(probes, timeoutMs) {
    const fsDeadline = Date.now() + timeoutMs * 6;
    const fsEffect = (effect, label) => {
        const remaining = fsDeadline - Date.now();
        if (remaining <= 0)
            return Promise.reject(new Error(label + ' scan filesystem budget exhausted'));
        return bounded(effect, Math.min(timeoutMs, remaining), label);
    };
    return {
        request: async (url, budget) => {
            try {
                return await bounded(() => probes.request(url, budget), budget, url);
            }
            catch (error) {
                return { ok: false, kind: 'connection', detail: url + ' probe failed: ' + String(error) };
            }
        },
        isDirectory: (path) => fsEffect(() => probes.isDirectory(path), path + ' stat'),
        listDir: (path) => fsEffect(() => probes.listDir(path), path + ' listing'),
        readFile: (path) => fsEffect(() => (probes.readFile ?? DEFAULT_SCAN_PROBES.readFile)(path), path + ' read'),
    };
}
/** The default probes: a status-aware fetch and the real filesystem. */
export const DEFAULT_SCAN_PROBES = {
    request: async (url, timeoutMs) => {
        let response;
        try {
            response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
        }
        catch (error) {
            return {
                ok: false,
                kind: 'connection',
                detail: `${url} could not be reached: ${error instanceof Error ? error.message : String(error)}`,
            };
        }
        const contentType = response.headers.get('content-type') ?? undefined;
        const text = await response.text().catch(() => '');
        if (!response.ok) {
            return {
                ok: false,
                kind: 'http',
                status: response.status,
                detail: `${url} answered HTTP ${response.status} ${response.statusText}`,
                ...(contentType === undefined ? {} : { contentType }),
            };
        }
        try {
            return { ok: true, status: response.status, body: JSON.parse(text), ...(contentType === undefined ? {} : { contentType }) };
        }
        catch {
            return {
                ok: false,
                kind: 'not-json',
                status: response.status,
                detail: `${url} answered ${response.status} with ${contentType ?? 'an unknown content type'}, which is not JSON ` +
                    `(it began "${text.slice(0, 40).replace(/\s+/g, ' ')}")`,
            };
        }
    },
    isDirectory: async (path) => {
        try {
            return (await stat(path)).isDirectory();
        }
        catch {
            return false;
        }
    },
    listDir: async (path) => {
        try {
            const entries = await readdir(path, { withFileTypes: true });
            return await Promise.all(entries.map(async (entry) => {
                if (entry.isDirectory())
                    return { name: entry.name, isDirectory: true };
                try {
                    const info = await stat(join(path, entry.name));
                    return { name: entry.name, isDirectory: false, sizeBytes: info.size };
                }
                catch {
                    return { name: entry.name, isDirectory: false };
                }
            }));
        }
        catch {
            return undefined;
        }
    },
    readFile: async (path) => await readFile(path, 'utf8'),
};
/**
 * Whether a value is a plain JSON object.
 * @param value - candidate value.
 * @returns true for a non-null, non-array object.
 */
function isObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}
/**
 * Whether a host is an Ollama instance.
 *
 * Matched on the engine label rather than the host id, following the rule the
 * discoverers use: an operator may call the row anything, but `runtime.engine` is
 * the label the hub routes to.
 *
 * @param host - the declared host.
 * @returns true when this host speaks Ollama's own API.
 */
function isOllamaHost(host) {
    return /ollama/i.test(host.runtime.engine);
}
/**
 * Strip a trailing slash from an endpoint.
 * @param endpoint - the base URL.
 * @returns the base URL without a trailing slash.
 */
function baseOf(endpoint) {
    return endpoint.endsWith('/') ? endpoint.slice(0, -1) : endpoint;
}
/**
 * The display name of a workflow document.
 *
 * A workflow's own metadata is preferred when it has any, and the filename is the
 * fallback, because a workflow saved by the ComfyUI editor is named by its file.
 *
 * @param raw - the parsed document.
 * @param fallback - the name it was found under.
 * @returns the name to show.
 */
export function workflowDisplayName(raw, fallback) {
    if (isObject(raw)) {
        for (const key of ['name', 'title', 'workflowName']) {
            const value = raw[key];
            if (typeof value === 'string' && value.trim().length > 0)
                return value.trim();
        }
        for (const container of ['extra', 'metadata']) {
            const nested = raw[container];
            if (isObject(nested) && typeof nested['name'] === 'string' && nested['name'].trim().length > 0) {
                return nested['name'].trim();
            }
        }
    }
    const last = fallback.split(/[/\\]/).pop() ?? fallback;
    return last.replace(/\.json$/i, '');
}
/**
 * Whether the engine for a host is answering, using its own readiness route.
 *
 * @param probes - the effects to use.
 * @param host - the host to check.
 * @param timeoutMs - budget for the request.
 * @returns whether it answered, and a sentence describing what happened.
 */
async function engineReady(probes, host, timeoutMs) {
    const endpoint = host.runtime.endpoint;
    if (endpoint === undefined || endpoint.trim().length === 0) {
        return { running: false, detail: 'no runtime.endpoint configured' };
    }
    const path = readyPathFor(host.runtime.engine);
    const result = await probes.request(`${baseOf(endpoint)}${path}`, timeoutMs);
    if (result.ok && /comfyui/i.test(host.runtime.engine) &&
        !(isObject(result.body) && isObject(result.body['system']) && Array.isArray(result.body['devices']))) {
        return { running: false, detail: path + ' did not return valid ComfyUI system/device information' };
    }
    return result.ok
        ? { running: true, detail: `${path} answered ${result.status}` }
        : { running: false, detail: result.detail };
}
/**
 * The whole saved-workflow directory listing URL.
 * @param base - the endpoint without a trailing slash.
 * @returns the listing URL.
 */
function workflowListingUrl(base) {
    return `${base}/userdata?dir=${encodeURIComponent(WORKFLOW_DIR)}&recurse=true`;
}
/**
 * The addresses a saved workflow may be served at, best first.
 *
 * ComfyUI serves one file through a single path segment (`/userdata/{file}`), so
 * the directory has to be inside the encoded path. Older and newer builds have
 * differed on whether the directory prefix is implied, so the documented shape is
 * tried first and the alternatives after it — a workflow that answers on any of
 * them is read, and one that answers on none is reported with the status each
 * attempt actually returned.
 *
 * @param base - the endpoint without a trailing slash.
 * @param relative - the path the listing returned, relative to the workflows directory.
 * @returns candidate URLs, most likely first.
 */
export function workflowUrls(base, relative) {
    const cleaned = relative.replace(/^\/+/, '');
    const withDir = `${WORKFLOW_DIR}/${cleaned}`;
    const candidates = [
        // The documented shape: one encoded segment, directory included.
        `${base}/userdata/${encodeURIComponent(withDir)}`,
        // A build that keeps separators literal but still wants the directory.
        `${base}/userdata/${withDir.split('/').map(encodeURIComponent).join('/')}`,
        // A build where the listing is already relative to the workflows directory.
        `${base}/userdata/${encodeURIComponent(cleaned)}`,
        `${base}/userdata/${cleaned}`,
    ];
    return [...new Set(candidates)];
}
/**
 * Read every workflow document from the server's saved-workflow store.
 *
 * A single unreadable file costs that file, never the pass: each is fetched and
 * classified independently, and a file whose contents could not be retrieved is
 * carried through as *unreadable* with the status that was actually received —
 * never as a document that failed to parse.
 *
 * @param probes - the effects to use.
 * @param endpoint - the ComfyUI base URL.
 * @param timeoutMs - budget per request.
 * @returns the documents, or the reason the store could not be listed.
 */
async function readServerWorkflows(probes, endpoint, timeoutMs) {
    const base = baseOf(endpoint);
    const listing = await probes.request(workflowListingUrl(base), timeoutMs);
    if (!listing.ok) {
        return { documents: [], error: `the saved-workflow listing failed: ${listing.detail}` };
    }
    const entries = Array.isArray(listing.body)
        ? listing.body
        : isObject(listing.body) && Array.isArray(listing.body['files'])
            ? listing.body['files']
            : [];
    if (!Array.isArray(listing.body) && !(isObject(listing.body) && Array.isArray(listing.body['files']))) {
        return { documents: [], error: 'the saved-workflow listing was not an array of names' };
    }
    const results = await Promise.all(entries.slice(0, MAX_WORKFLOW_FILES).map(async (entry) => {
        const name = typeof entry === 'string'
            ? entry
            : isObject(entry) && typeof entry['path'] === 'string'
                ? entry['path']
                : isObject(entry) && typeof entry['name'] === 'string'
                    ? entry['name']
                    : undefined;
        if (name === undefined)
            return undefined;
        const cleaned = name.replace(/^\/+/, '').trim();
        if (cleaned.length === 0 || !cleaned.toLowerCase().endsWith('.json'))
            return undefined;
        const attempts = [];
        let read;
        for (const url of workflowUrls(base, cleaned)) {
            const result = await probes.request(url, timeoutMs);
            if (result.ok) {
                read = { name: cleaned, raw: result.body, source: url };
                break;
            }
            attempts.push(result.detail);
        }
        return read ?? {
            name: cleaned,
            raw: undefined,
            source: workflowUrls(base, cleaned)[0],
            unreadable: `the engine listed this workflow but would not return its contents. Tried: ${attempts.join(' | ')}`,
        };
    }));
    const documents = results.filter((document) => document !== undefined);
    return { documents };
}
/**
 * Every JSON file under a directory, walked breadth-first with depth and count caps.
 *
 * @param probes - the effects to use.
 * @param root - the directory to walk.
 * @returns the files found, as absolute paths.
 */
async function listWorkflowFiles(probes, root, accept = (name) => /\.json$/i.test(name)) {
    const errors = [];
    const found = [];
    const queue = [{ path: root, depth: 0 }];
    let directories = 0;
    while (queue.length > 0 && found.length < MAX_WORKFLOW_FILES && directories < MAX_WORKFLOW_FILES) {
        directories += 1;
        const current = queue.shift();
        let entries;
        try {
            entries = await probes.listDir(current.path);
        }
        catch (error) {
            errors.push(current.path + ': ' + String(error));
            continue;
        }
        if (entries === undefined) {
            errors.push(current.path + ' directory listing failed');
            continue;
        }
        for (const entry of entries.slice(0, MAX_WORKFLOW_FILES)) {
            if (found.length >= MAX_WORKFLOW_FILES)
                break;
            // Directory probes must not escape the configured root.
            if (!entry.name || /[/\\]/.test(entry.name) || entry.name === '.' || entry.name === '..')
                continue;
            const child = join(current.path, entry.name);
            if (entry.isDirectory) {
                if (current.depth < MAX_DIRECTORY_DEPTH && queue.length < MAX_WORKFLOW_FILES)
                    queue.push({ path: child, depth: current.depth + 1 });
                continue;
            }
            if (!accept(entry.name))
                continue;
            found.push({ path: child, ...(entry.sizeBytes === undefined ? {} : { sizeBytes: entry.sizeBytes }) });
        }
    }
    return { files: found, errors };
}
/**
 * Read every workflow document from a configured directory.
 *
 * @param probes - the effects to use.
 * @param root - the directory.
 * @returns the documents, or the reason the directory could not be read.
 */
async function readDirectoryWorkflows(probes, root) {
    if (!(await probes.isDirectory(root))) {
        return { documents: [], error: `the configured workflow directory ${root} is not a directory` };
    }
    const readText = probes.readFile ?? DEFAULT_SCAN_PROBES.readFile;
    const { files, errors } = await listWorkflowFiles(probes, root);
    const documents = await Promise.all(files.map(async (file) => {
        try {
            const text = await readText(file.path);
            return {
                name: file.path,
                raw: JSON.parse(text),
                source: file.path,
                ...(file.sizeBytes === undefined ? {} : { sizeBytes: file.sizeBytes }),
            };
        }
        catch (error) {
            // A file that could not be read or parsed is reported as unreadable rather
            // than as an invalid workflow: the difference is whether the document was
            // ever seen.
            return {
                name: file.path,
                raw: undefined,
                source: file.path,
                unreadable: `${file.path} could not be read as JSON: ${error instanceof Error ? error.message : String(error)}`,
                ...(file.sizeBytes === undefined ? {} : { sizeBytes: file.sizeBytes }),
            };
        }
    }));
    return { documents, ...(errors.length === 0 ? {} : { error: errors.join(' | ') }) };
}
/**
 * Describe one Ollama model for the list.
 *
 * @param host - the host it was read from.
 * @param endpoint - the endpoint it was read from.
 * @param entry - the `/api/tags` entry.
 * @param registeredId - the catalog id already serving it, when any.
 * @returns the resource.
 */
function ollamaResource(host, endpoint, entry, registeredId) {
    if (!isObject(entry))
        return undefined;
    const name = typeof entry['name'] === 'string' ? entry['name'] : undefined;
    if (name === undefined || name.trim().length === 0)
        return undefined;
    const details = isObject(entry['details']) ? entry['details'] : {};
    const parts = [details['parameter_size'], details['quantization_level'], details['family']].filter((part) => typeof part === 'string' && part.length > 0);
    const sizeBytes = typeof entry['size'] === 'number' ? entry['size'] : undefined;
    const id = scannedOllamaModelId(host.id, name);
    return {
        id,
        kind: 'ollama_model',
        typeLabel: 'Ollama Model',
        name,
        source: `${baseOf(endpoint)}/api/tags`,
        status: 'ready',
        detail: (parts.length === 0 ? 'installed' : `installed · ${parts.join(' · ')}`) +
            (registeredId === undefined ? ' (not registered for hub invocation)' : ''),
        runnable: registeredId !== undefined,
        ...(registeredId === undefined ? {} : { modelId: registeredId }),
        engineRunning: true,
        ...(sizeBytes === undefined ? {} : { sizeBytes }),
    };
}
/** Ollama treats an omitted tag as :latest; compare identities, not spellings. */
function ollamaTag(tag) {
    const cleaned = tag.trim();
    return cleaned.slice(cleaned.lastIndexOf('/') + 1).includes(':') ? cleaned : cleaned + ':latest';
}
/** One independent Ollama pass; offline catalog entries remain visible. */
async function scanOllamaHost(hub, host, probes, timeoutMs, state) {
    const endpoint = host.runtime.endpoint;
    const key = JSON.stringify(['ollama', host]);
    const generation = beginSource(state, key);
    const checkedAt = new Date().toISOString();
    const known = hub.catalog.listModels().filter((model) => model.hostId === host.id && /ollama/i.test(model.runtime.engine) &&
        model.runtime.endpoint === host.runtime.endpoint);
    const registered = new Map(known.flatMap((model) => typeof model.adapterConfig['model'] === 'string'
        ? [[ollamaTag(model.adapterConfig['model']), model.id]] : []));
    const tags = endpoint?.trim()
        ? await probes.request(baseOf(endpoint) + '/api/tags', timeoutMs)
        : { ok: false, kind: 'connection', detail: 'no runtime.endpoint configured' };
    const valid = tags.ok && isObject(tags.body) && Array.isArray(tags.body['models']) &&
        tags.body['models'].every((entry) => isObject(entry) && typeof entry['name'] === 'string' && entry['name'].trim().length > 0);
    const resources = [];
    if (valid && tags.ok && isObject(tags.body)) {
        for (const entry of tags.body['models']) {
            const tag = isObject(entry) && typeof entry['name'] === 'string' ? entry['name'] : undefined;
            const resource = ollamaResource(host, endpoint, entry, tag === undefined ? undefined : registered.get(ollamaTag(tag)));
            if (resource !== undefined)
                resources.push({ ...resource, availability: 'available', freshness: 'live', checkedAt });
        }
        if (state.generations.get(key) === generation)
            state.cache.set(key, { resources, descriptors: [] });
    }
    else {
        resources.push(...(state.cache.get(key)?.resources ?? []).map((resource) => {
            const { modelId: _priorModelId, ...observation } = resource;
            const modelId = registered.get(ollamaTag(resource.name));
            return { ...observation, ...(modelId === undefined ? {} : { modelId }), runnable: modelId !== undefined,
                engineRunning: false, availability: tags.ok ? 'unknown' : 'unavailable', freshness: 'cached',
                detail: resource.detail + ' (cached; engine could not be verified)' };
        }));
    }
    // Ollama's documented local manifest layout is useful even with its API off.
    // Never attribute this machine's model store to a remote Ollama endpoint.
    let localEndpoint = false;
    try {
        localEndpoint = ['localhost', '127.0.0.1', '[::1]'].includes(new URL(endpoint ?? '').hostname);
    }
    catch { }
    if (!valid && localEndpoint) {
        const configuredRoot = host.runtime.env?.['OLLAMA_MODELS']?.trim() || process.env['OLLAMA_MODELS']?.trim();
        const expandedRoot = configuredRoot?.replace(/^~(?=[/\\]|$)/, homedir())
            .replace(/%([^%]+)%/g, (token, name) => host.runtime.env?.[name] ?? process.env[name] ?? token)
            .replace(/\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)/g, (token, braced, plain) => {
            const name = braced ?? plain;
            return host.runtime.env?.[name] ?? process.env[name] ?? token;
        });
        const root = join(expandedRoot || join(homedir(), '.ollama', 'models'), 'manifests');
        try {
            if (await probes.isDirectory(root)) {
                const { files } = await bounded(() => listWorkflowFiles(probes, root, () => true), timeoutMs * 3, root + ' manifest scan');
                const manifests = await Promise.all(files.map(async (file) => {
                    try {
                        const manifest = JSON.parse(await probes.readFile(file.path));
                        if (!isObject(manifest) || !Array.isArray(manifest['layers']) || !isObject(manifest['config']))
                            return undefined;
                        const parts = relative(root, file.path).split(/[/\\]/);
                        if (parts.length < 4)
                            return undefined;
                        const namespace = parts.slice(1, -2).join('/');
                        const name = (namespace === 'library' ? '' : namespace + '/') + parts.at(-2) + ':' + basename(file.path);
                        return { id: scannedOllamaModelId(host.id, name), kind: 'ollama_model', typeLabel: 'Ollama Model', name,
                            source: file.path, status: 'ready',
                            ...(registered.get(ollamaTag(name)) === undefined ? {} : { modelId: registered.get(ollamaTag(name)) }),
                            runnable: registered.has(ollamaTag(name)), engineRunning: false, availability: 'unavailable', freshness: 'local', checkedAt,
                            detail: 'Local Ollama manifest found; engine and referenced blobs are not verified' +
                                (registered.has(ollamaTag(name)) ? '' : ' (not registered for hub invocation)') };
                    }
                    catch {
                        return undefined;
                    }
                }));
                for (const manifest of manifests) {
                    if (manifest === undefined)
                        continue;
                    const index = resources.findIndex((resource) => ollamaTag(resource.name) === ollamaTag(manifest.name));
                    if (index < 0)
                        resources.push(manifest);
                    else
                        resources[index] = manifest;
                }
            }
        }
        catch { /* The model store is optional; API/catalog results remain independent. */ }
    }
    for (const model of known) {
        const tag = model.adapterConfig['model'];
        if (typeof tag !== 'string' || resources.some((resource) => ollamaTag(resource.name) === ollamaTag(tag)))
            continue;
        resources.push({
            id: scannedOllamaModelId(host.id, tag), kind: 'ollama_model', typeLabel: 'Ollama Model', name: tag,
            source: 'catalog:' + model.id, modelId: model.id, status: 'ready', runnable: model.enabled,
            engineRunning: valid, availability: valid ? 'unavailable' : 'unknown', freshness: 'catalog', checkedAt,
            capabilities: model.capabilities,
            detail: valid ? 'Catalog model was not reported installed by /api/tags' : 'Known to the active catalog; installation and engine availability not verified',
        });
    }
    const detail = valid ? 'read installed models from ' + endpoint + '/api/tags'
        : tags.ok ? 'the model listing did not contain a valid models array with nonempty model names' : tags.detail;
    return { resources, descriptors: [], warnings: valid ? [] : [host.name + ': ' + detail],
        sources: [{ id: host.id, kind: 'ollama_model', label: host.name, ok: valid, detail, found: resources.length }] };
}
/**
 * Turn one workflow document into a list row, and into a provider when it runs.
 *
 * @param host - the ComfyUI host that will run it.
 * @param document - the document, its name, and where it came from.
 * @param io - the node index, when the engine was reachable.
 * @param classes - the installed node classes, when the engine was reachable.
 * @param engineRunning - whether the engine is answering right now.
 * @returns the row and provider, or `undefined` when the document has no name.
 */
function workflowResource(host, document, io, classes, engineRunning) {
    if (document.name.trim().length === 0)
        return undefined;
    const displayName = workflowDisplayName(document.raw, document.name);
    const id = scannedWorkflowId(displayName, document.source) + '-' + stableDigest(host.id + '|' + document.source);
    // Contents that were never retrieved are *unreadable*, with the transport
    // reason — never "invalid", which would blame a document nobody has seen.
    if (document.unreadable !== undefined) {
        return {
            resource: {
                id,
                kind: 'comfyui_workflow',
                typeLabel: 'ComfyUI Workflow',
                name: displayName,
                source: document.source,
                status: 'unreadable',
                detail: document.unreadable,
                runnable: false,
                engineRunning,
                diagnostics: [document.unreadable],
                ...(document.sizeBytes === undefined ? {} : { sizeBytes: document.sizeBytes }),
            },
        };
    }
    const scan = scanWorkflowDocument({
        raw: document.raw,
        name: document.name,
        ...(io === undefined ? {} : { io }),
        ...(classes === undefined ? {} : { classes }),
    });
    const inputs = scan.inputs.map((input) => ({
        name: input.name,
        label: input.label,
        kind: input.kind,
        node: input.node,
        input: input.input,
    }));
    const outputs = scan.outputs.map((output) => ({
        name: output.name,
        type: String(output.type),
        node: output.node,
        nodeClass: output.nodeClass,
    }));
    const resource = {
        id,
        kind: 'comfyui_workflow',
        typeLabel: 'ComfyUI Workflow',
        name: displayName,
        source: document.source,
        status: scan.readiness,
        detail: scan.detail,
        runnable: scan.runnable,
        format: scan.format,
        capabilities: [...scan.capabilities],
        inputs,
        outputs,
        engineRunning,
        ...(scan.diagnostics.length === 0 ? {} : { diagnostics: [...scan.diagnostics] }),
        ...(document.sizeBytes === undefined ? {} : { sizeBytes: document.sizeBytes }),
        ...(scan.runnable ? { modelId: id } : {}),
    };
    if (!scan.runnable || scan.contract === undefined || scan.graph === undefined)
        return { resource };
    const { inputTypes, outputTypes } = ioForCapabilities(scan.capabilities);
    const descriptor = {
        id,
        name: displayName,
        type: workflowModelTypeFor(scan.capabilities),
        host: host.id,
        providerKind: 'workflow',
        workflowId: id,
        capabilities: scan.capabilities,
        inputTypes,
        outputTypes,
        adapterConfig: {
            workflow: scan.graph,
            bindings: scan.contract.bindings,
            outputs: scan.contract.outputs,
            ...(scan.contract.inputKinds === undefined ? {} : { inputKinds: scan.contract.inputKinds }),
        },
        resources: { vramGb: 0, ramGb: 0 },
        // Below every static entry, so a hand-declared provider always wins routing
        // against a machine that happens to have a similar workflow saved.
        priority: DISCOVERED_PRIORITY,
        tags: ['local', 'discovered', 'comfyui', 'workflow', 'scanned'],
        enabled: true,
        notes: `Discovered from ${document.source}. Its public inputs and outputs were read from the graph and from ` +
            'ComfyUI\'s node definitions; its checkpoint, LoRA and VAE choices are part of the workflow and are not exposed.',
    };
    return { resource, descriptor };
}
/** Reconcile one source independently: failed files keep their last good contract. */
function reconcileSource(state, key, fresh, complete, engineRunning, availability, generation) {
    const previous = state.cache.get(key);
    const resources = [...fresh.resources];
    const descriptors = [...fresh.descriptors];
    for (const old of previous?.resources ?? []) {
        const index = resources.findIndex((resource) => resource.source === old.source);
        const current = resources[index];
        if (current !== undefined && current.status !== 'unreadable' && !(current.freshness === 'cached'))
            continue;
        if (current === undefined && complete)
            continue;
        const cached = {
            ...old, engineRunning, availability, freshness: 'cached',
            detail: old.detail + ' (cached observation; current resource could not be verified)',
            diagnostics: [...(old.diagnostics ?? []), ...(current?.diagnostics ?? []), ...fresh.warnings],
        };
        if (index < 0)
            resources.push(cached);
        else
            resources[index] = cached;
        const descriptor = previous?.descriptors.find((entry) => entry.id === old.modelId);
        if (descriptor !== undefined && !descriptors.some((entry) => entry.id === descriptor.id))
            descriptors.push(descriptor);
    }
    // Store only original observations, not repeatedly decorated cache responses.
    const storedResources = resources.map((resource) => resource.freshness === 'cached'
        ? previous?.resources.find((entry) => entry.source === resource.source) ?? resource : resource);
    if (state.generations.get(key) === generation)
        state.cache.set(key, { resources: storedResources, descriptors });
    return { ...fresh, resources, descriptors };
}
async function runHostPass(hub, host, options, state) {
    const { probes, timeoutMs } = options;
    const apiKey = JSON.stringify(['api', host]);
    const apiGeneration = beginSource(state, apiKey);
    const directoryGenerations = new Map(options.workflowDirs.map((root) => [root, beginSource(state, JSON.stringify(['directory', host, root]))]));
    const checkedAt = new Date().toISOString();
    const warnings = [];
    // Start the directory effects immediately, independently of API readiness.
    const directoryTasks = options.workflowDirs.map(async (root) => {
        try {
            return { root, result: await bounded(() => readDirectoryWorkflows(probes, root), timeoutMs * 3, root + ' workflow scan') };
        }
        catch (error) {
            return { root, result: { documents: [], error: root + ': ' + String(error) } };
        }
    });
    const ready = await engineReady(probes, host, timeoutMs);
    const engineRunning = ready.running;
    if (!engineRunning)
        warnings.push(host.name + ': not answering; scans never start engines (' + ready.detail + ')');
    const endpoint = host.runtime.endpoint;
    const sources = [{ id: host.id, kind: 'comfyui_workflow', label: host.name,
            ok: engineRunning, detail: ready.detail, found: 0 }];
    let io;
    let classes;
    let listed = { documents: [], error: ready.detail };
    let infoOk = false;
    if (engineRunning && endpoint?.trim()) {
        const [info, documents] = await Promise.all([
            probes.request(baseOf(endpoint) + '/object_info', timeoutMs), readServerWorkflows(probes, endpoint, timeoutMs),
        ]);
        listed = documents;
        if (info.ok && isObject(info.body) && Object.values(info.body).some((node) => isObject(node) && isObject(node['input']))) {
            io = readComfyNodeIo(info.body);
            classes = new Set(Object.keys(info.body));
            infoOk = true;
        }
        else
            warnings.push(host.name + ': node definitions could not be verified' + (info.ok ? '' : ' (' + info.detail + ')'));
    }
    const build = (documents, freshness) => {
        const resources = [];
        const descriptors = [];
        for (const document of documents) {
            const built = workflowResource(host, document, io, classes, engineRunning);
            if (built === undefined)
                continue;
            resources.push({ ...built.resource,
                availability: document.unreadable !== undefined ? 'unknown' : engineRunning ? 'unknown' : 'unavailable',
                detail: built.resource.detail + (engineRunning ? ' (engine answering; workflow dependencies not execution-verified)' : ' (engine not answering)'),
                freshness, checkedAt });
            if (built.descriptor !== undefined)
                descriptors.push(built.descriptor);
        }
        return { resources, descriptors, warnings: [], sources: [] };
    };
    const apiFresh = build(listed.documents, 'live');
    // Missing node definitions cannot overwrite previously validated conversions.
    if (!infoOk) {
        const previous = state.cache.get(JSON.stringify(['api', host]));
        for (let index = 0; index < apiFresh.resources.length; index += 1) {
            const resource = apiFresh.resources[index];
            if (previous?.resources.some((old) => old.source === resource.source)) {
                apiFresh.resources[index] = { ...resource, freshness: 'cached' };
                const descriptorIndex = apiFresh.descriptors.findIndex((entry) => entry.id === resource.modelId);
                if (descriptorIndex >= 0)
                    apiFresh.descriptors.splice(descriptorIndex, 1);
            }
        }
    }
    apiFresh.warnings.push(...warnings, ...(listed.error === undefined ? [] : [listed.error]));
    const api = reconcileSource(state, JSON.stringify(['api', host]), apiFresh, listed.error === undefined, engineRunning, engineRunning ? 'unknown' : 'unavailable', apiGeneration);
    const resources = [...api.resources];
    const descriptors = [...api.descriptors];
    if (engineRunning)
        sources.push({ id: host.id, kind: 'comfyui_workflow', label: host.name,
            ok: listed.error === undefined, detail: listed.error ?? 'read saved workflows from ' + endpoint + '/userdata',
            found: api.resources.length });
    if (listed.error !== undefined && engineRunning)
        warnings.push(listed.error);
    let authoritative = listed.error === undefined && infoOk && listed.documents.every((document) => document.unreadable === undefined);
    for (const { root, result } of await Promise.all(directoryTasks)) {
        authoritative = authoritative && result.error === undefined && result.documents.every((document) => document.unreadable === undefined);
        const fresh = build(result.documents, 'local');
        if (result.error !== undefined)
            fresh.warnings.push(result.error);
        if (!infoOk) {
            const previous = state.cache.get(JSON.stringify(['directory', host, root]));
            for (let index = 0; index < fresh.resources.length; index += 1) {
                const resource = fresh.resources[index];
                if (resource.format !== 'ui' || !previous?.resources.some((old) => old.source === resource.source))
                    continue;
                fresh.resources[index] = { ...resource, freshness: 'cached' };
                const descriptorIndex = fresh.descriptors.findIndex((entry) => entry.id === resource.modelId);
                if (descriptorIndex >= 0)
                    fresh.descriptors.splice(descriptorIndex, 1);
            }
        }
        const local = reconcileSource(state, JSON.stringify(['directory', host, root]), fresh, result.error === undefined, engineRunning, engineRunning ? 'unknown' : 'unavailable', directoryGenerations.get(root));
        resources.push(...local.resources);
        descriptors.push(...local.descriptors);
        sources.push({ id: 'workflow-dir', kind: 'comfyui_workflow', label: root, ok: result.error === undefined,
            detail: result.error ?? 'read workflow files from ' + root, found: local.resources.length });
        if (result.error !== undefined)
            warnings.push(result.error);
    }
    // Consult the active catalog (not a stale startup config). Read pinned paths
    // even offline; preserve the declared provider identity instead of republishing it.
    const known = hub.catalog.listModels().filter((model) => model.hostId === host.id && model.adapter === 'comfyui' &&
        model.runtime.endpoint === host.runtime.endpoint && !model.tags?.includes('scanned'));
    const catalogResources = await Promise.all(known.map(async (model) => {
        if (resources.some((resource) => resource.modelId === model.id))
            return undefined;
        const config = model.adapterConfig;
        const workflowPath = typeof config['workflowPath'] === 'string'
            ? resolveAdapterPath(config['workflowPath'], options.catalogDir).absolute : undefined;
        if (workflowPath !== undefined) {
            const duplicate = resources.findIndex((resource) => resource.source === workflowPath);
            if (duplicate >= 0) {
                const resource = resources[duplicate];
                resources[duplicate] = { ...resource, id: model.id, modelId: model.id, name: model.name,
                    runnable: model.enabled && resource.runnable };
                const descriptorIndex = descriptors.findIndex((entry) => entry.id === resource.modelId);
                if (descriptorIndex >= 0)
                    descriptors.splice(descriptorIndex, 1);
                return undefined;
            }
        }
        let raw = config['workflow'];
        let unreadable;
        if (workflowPath !== undefined) {
            try {
                raw = JSON.parse(await probes.readFile(workflowPath));
            }
            catch (error) {
                unreadable = String(error);
            }
        }
        const built = workflowResource(host, { name: model.name, raw, source: workflowPath ?? 'catalog:' + model.id,
            ...(unreadable === undefined ? {} : { unreadable }) }, io, classes, engineRunning);
        if (built === undefined)
            return undefined;
        return { ...built.resource, id: model.id, modelId: model.id, name: model.name,
            runnable: model.enabled && built.resource.runnable, checkedAt,
            freshness: workflowPath !== undefined && unreadable === undefined ? 'local' : 'catalog',
            availability: unreadable !== undefined ? 'unknown' : engineRunning ? 'unknown' : 'unavailable',
            detail: built.resource.detail + ' (known to catalog; workflow dependencies not execution-verified)' };
    }));
    resources.push(...catalogResources.filter((resource) => resource !== undefined));
    return { resources, descriptors, warnings, sources, authoritative };
}
/** Scan only observes. Neither compatibility flags nor launch policy grant lifecycle effects. */
export async function scanLocalResources(hub, options) {
    const requestedTimeout = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const timeoutMs = Number.isFinite(requestedTimeout) ? Math.max(1, Math.min(60_000, requestedTimeout)) : DEFAULT_TIMEOUT_MS;
    const probes = boundedProbes({ ...DEFAULT_SCAN_PROBES, ...(options.probes ?? {}) }, timeoutMs);
    const state = stateFor(hub);
    const publicationGeneration = ++state.publicationGeneration;
    const requestedHosts = options.hosts.filter((host) => isOllamaHost(host) || host.adapter === 'comfyui');
    const previousScanned = hub.catalog.listModels().filter((model) => model.tags?.includes('scanned'));
    const passes = await Promise.all(options.hosts.filter((host) => isOllamaHost(host) || host.adapter === 'comfyui')
        .map(async (host) => {
        try {
            return isOllamaHost(host) ? await scanOllamaHost(hub, host, probes, timeoutMs, state)
                : await runHostPass(hub, host, { probes, timeoutMs,
                    ...(options.catalogPath === undefined ? {} : { catalogDir: dirname(resolveAdapterPath(options.catalogPath, undefined).absolute) }),
                    workflowDirs: [...new Set([...(options.workflowDir?.trim() ? [options.workflowDir] : []),
                            ...(options.workflowDirsByHost?.[host.id] ?? []).filter((root) => root.trim())])],
                }, state);
        }
        catch (error) {
            const detail = host.name + ': scan failed: ' + String(error);
            return { resources: [], descriptors: [], warnings: [detail], sources: [{ id: host.id,
                        kind: isOllamaHost(host) ? 'ollama_model' : 'comfyui_workflow', label: host.name, ok: false, detail, found: 0 }] };
        }
    }));
    const resources = [...new Map(passes.flatMap((pass) => pass.resources).map((resource) => [resource.id, resource])).values()];
    const preserved = previousScanned.filter((model) => {
        const index = requestedHosts.findIndex((host) => host.id === model.hostId && host.runtime.endpoint === model.runtime.endpoint);
        return index >= 0 && passes[index]?.authoritative !== true;
    }).map((model) => model.descriptor);
    const descriptors = dedupeById([...passes.flatMap((pass) => pass.descriptors), ...preserved]);
    const trustworthyEmpty = requestedHosts.some((host) => host.adapter === 'comfyui') &&
        passes.every((pass, index) => requestedHosts[index]?.adapter !== 'comfyui' || pass.authoritative === true);
    let registered = 0;
    if (options.register !== false && (descriptors.length > 0 || trustworthyEmpty) && state.publicationGeneration === publicationGeneration) {
        registered = hub.publishScannedModels(descriptors);
        options.log?.('scan: published ' + registered + ' provider(s) in the catalog');
    }
    return { resources, sources: passes.flatMap((pass) => pass.sources), warnings: passes.flatMap((pass) => pass.warnings),
        registered, generatedAt: new Date().toISOString() };
}
/**
 * Keep the first descriptor for each id.
 * @param descriptors - the descriptors to deduplicate.
 * @returns the unique descriptors, in order.
 */
function dedupeById(descriptors) {
    const seen = new Set();
    const unique = [];
    for (const descriptor of descriptors) {
        if (seen.has(descriptor.id))
            continue;
        seen.add(descriptor.id);
        unique.push(descriptor);
    }
    return unique;
}
/** Clear the discovery cache. Exported for tests that need a clean machine. */
export function resetScanCache() {
    scanStates = new WeakMap();
}
/** The digest helper, re-exported so a caller can compare analyses cheaply. */
export { stableDigest };
