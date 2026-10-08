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
 * ## The engine's lifecycle belongs to the scan, but only the part it started
 *
 * Discovering workflows requires a running ComfyUI. If one is already answering it
 * is used and left alone. If none is, the hub's own launch path starts one, and
 * the scan owns that process — and only that process. Never a pre-existing
 * instance, never a process killed by name, and never while a graph is queued or
 * running: an engine with work in flight is left up and the reason is reported.
 *
 * @module dsh-ai-model-hub/dsh-plugin/scan
 */
import { readFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { DISCOVERED_PRIORITY, ioForCapabilities, readComfyNodeIo, scannedOllamaModelId, scannedWorkflowId, scanWorkflowDocument, stableDigest, workflowModelTypeFor, } from "../src/index.js";
/** The default request budget; a scan should not stall the settings page. */
const DEFAULT_TIMEOUT_MS = 4_000;
/** The default budget for an engine to come up, once launched. */
const DEFAULT_STARTUP_TIMEOUT_MS = 180_000;
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
/**
 * The last successful discovery per host.
 *
 * This is what makes a temporary failure non-destructive: a scan that cannot
 * reach ComfyUI reports the cached workflows with their engine marked as down,
 * rather than emptying the page and unregistering providers that were working a
 * minute ago. It is also what keeps a discovered contract usable after ComfyUI is
 * shut down, which is the point of discovering it in the first place.
 */
const hostCache = new Map();
/**
 * The scan passes currently in flight, keyed by host id.
 *
 * Two concurrent scans — the page's own button pressed twice, or a scan racing an
 * invocation's cold start — share one pass, so neither starts a second engine nor
 * shuts down one the other is using.
 */
const inFlight = new Map();
/** Engines this process started for a scan, and how many passes still need them. */
const ownedEngines = new Map();
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
    return result.ok
        ? { running: true, detail: `${path} answered ${result.status}` }
        : { running: false, detail: result.detail };
}
/**
 * Bring a host's engine up, and report whether this pass owns what it started.
 *
 * @param hub - the live hub, whose launch policy is the one that applies.
 * @param host - the host to start.
 * @param timeoutMs - how long to wait for readiness.
 * @param log - diagnostic sink.
 * @returns ownership and a description of what happened.
 */
async function startEngine(hub, host, timeoutMs, log, isReady) {
    try {
        const started = await hub.startEngineForHost(host.id);
        if (started.alreadyRunning) {
            // The hub found the endpoint already answering, so nothing was spawned and
            // this pass owns nothing to shut down.
            return { owned: false, detail: `${host.name} was already running` };
        }
        if (!started.started) {
            return { owned: false, detail: `${host.name} did not need starting` };
        }
        // Readiness is the engine answering its own readiness route, not the hub's
        // launch report: a process can be spawned a second before it binds its port,
        // and a queued graph says nothing about whether the API is up.
        const deadline = Date.now() + timeoutMs;
        let ready = false;
        while (Date.now() < deadline) {
            if (await isReady()) {
                ready = true;
                break;
            }
            await new Promise((resolve) => setTimeout(resolve, 500));
        }
        ownedEngines.set(started.modelId, { hostId: host.id, users: (ownedEngines.get(started.modelId)?.users ?? 0) + 1 });
        return {
            owned: true,
            detail: ready
                ? `started ${host.name} for this scan`
                : `started ${host.name}, which did not answer within ${timeoutMs} ms`,
        };
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        log(`scan: could not start ${host.name}: ${message}`);
        return { owned: false, detail: `could not start ${host.name}: ${message}` };
    }
}
/**
 * Shut down an engine this scan started, when it is safe to do so.
 *
 * Three conditions have to hold, and each exists because violating it would break
 * someone else's work: the engine must have been started *by a scan* (never a
 * pre-existing instance), no other scan may still be using it, and it must have no
 * graph queued or running. When the last does not hold the shutdown is deferred
 * and the reason is reported rather than the process being killed underneath a
 * running job.
 *
 * @param hub - the live hub.
 * @param modelId - the id the engine was started under.
 * @returns whether it was stopped, and what happened.
 */
async function releaseEngine(hub, modelId) {
    const owned = ownedEngines.get(modelId);
    if (owned === undefined)
        return { stopped: false, detail: 'this scan did not start the engine' };
    const users = owned.users - 1;
    if (users > 0) {
        ownedEngines.set(modelId, { hostId: owned.hostId, users });
        return { stopped: false, detail: 'another scan is still using the engine, so it was left running' };
    }
    const busy = hub.activeInvocationsFor(modelId);
    if (busy > 0) {
        ownedEngines.set(modelId, { hostId: owned.hostId, users: 0 });
        return {
            stopped: false,
            detail: `the engine has ${busy} graph(s) queued or running, so it was left up rather than interrupted`,
        };
    }
    ownedEngines.delete(modelId);
    try {
        const result = await hub.stopEngine(modelId);
        return {
            stopped: result.stopped,
            detail: result.stopped ? 'stopped the engine this scan started' : 'the engine was already stopped',
        };
    }
    catch (error) {
        return { stopped: false, detail: `could not stop the engine: ${error instanceof Error ? error.message : String(error)}` };
    }
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
    if (entries.length === 0 && !Array.isArray(listing.body) && !isObject(listing.body)) {
        return { documents: [], error: 'the saved-workflow listing was not an array of names' };
    }
    const documents = [];
    for (const entry of entries) {
        const name = typeof entry === 'string'
            ? entry
            : isObject(entry) && typeof entry['path'] === 'string'
                ? entry['path']
                : isObject(entry) && typeof entry['name'] === 'string'
                    ? entry['name']
                    : undefined;
        if (name === undefined)
            continue;
        const cleaned = name.replace(/^\/+/, '').trim();
        if (cleaned.length === 0 || !cleaned.toLowerCase().endsWith('.json'))
            continue;
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
        documents.push(read ?? {
            name: cleaned,
            raw: undefined,
            source: `${base}/userdata/${cleaned}`,
            unreadable: `the engine listed this workflow but would not return its contents. Tried: ${attempts.join(' | ')}`,
        });
    }
    return { documents };
}
/**
 * Every JSON file under a directory, walked breadth-first with depth and count caps.
 *
 * @param probes - the effects to use.
 * @param root - the directory to walk.
 * @returns the files found, as absolute paths.
 */
async function listWorkflowFiles(probes, root) {
    const found = [];
    const queue = [{ path: root, depth: 0 }];
    while (queue.length > 0 && found.length < MAX_WORKFLOW_FILES) {
        const current = queue.shift();
        const entries = (await probes.listDir(current.path)) ?? [];
        for (const entry of entries) {
            if (found.length >= MAX_WORKFLOW_FILES)
                break;
            const child = join(current.path, entry.name);
            if (entry.isDirectory) {
                if (current.depth < MAX_DIRECTORY_DEPTH)
                    queue.push({ path: child, depth: current.depth + 1 });
                continue;
            }
            if (!/\.json$/i.test(entry.name))
                continue;
            found.push({ path: child, ...(entry.sizeBytes === undefined ? {} : { sizeBytes: entry.sizeBytes }) });
        }
    }
    return found;
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
    const files = await listWorkflowFiles(probes, root);
    const documents = [];
    for (const file of files) {
        try {
            const text = await readText(file.path);
            documents.push({
                name: file.path,
                raw: JSON.parse(text),
                source: file.path,
                ...(file.sizeBytes === undefined ? {} : { sizeBytes: file.sizeBytes }),
            });
        }
        catch (error) {
            // A file that could not be read or parsed is reported as unreadable rather
            // than as an invalid workflow: the difference is whether the document was
            // ever seen.
            documents.push({
                name: file.path,
                raw: undefined,
                source: file.path,
                unreadable: `${file.path} could not be read as JSON: ${error instanceof Error ? error.message : String(error)}`,
                ...(file.sizeBytes === undefined ? {} : { sizeBytes: file.sizeBytes }),
            });
        }
    }
    return { documents };
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
        detail: parts.length === 0 ? 'installed' : `installed · ${parts.join(' · ')}`,
        runnable: true,
        modelId: registeredId ?? id,
        engineRunning: true,
        ...(sizeBytes === undefined ? {} : { sizeBytes }),
    };
}
/**
 * Scan the Ollama instances the catalog declares.
 *
 * @param hub - the live hub, read for already-registered model ids.
 * @param hosts - the declared hosts.
 * @param probes - the effects to use.
 * @param timeoutMs - budget per request.
 * @returns the resources and the per-host report.
 */
async function scanOllama(hub, hosts, probes, timeoutMs) {
    const resources = [];
    const sources = [];
    for (const host of hosts.filter(isOllamaHost)) {
        const endpoint = host.runtime.endpoint;
        if (endpoint === undefined || endpoint.trim().length === 0) {
            sources.push({
                id: host.id,
                kind: 'ollama_model',
                label: host.name,
                ok: false,
                detail: 'no runtime.endpoint configured, so its models cannot be listed',
                found: 0,
            });
            continue;
        }
        // Readiness and listing are the same request for Ollama: `/api/tags` is both
        // the health answer and the model list, which is why it is the readiness path.
        const tags = await probes.request(`${baseOf(endpoint)}/api/tags`, timeoutMs);
        if (!tags.ok) {
            sources.push({
                id: host.id,
                kind: 'ollama_model',
                label: host.name,
                ok: false,
                detail: tags.detail,
                found: 0,
            });
            continue;
        }
        const list = isObject(tags.body) && Array.isArray(tags.body['models']) ? tags.body['models'] : [];
        const registered = new Map();
        for (const model of hub.catalog.listModels()) {
            if (model.hostId !== host.id)
                continue;
            const tag = model.adapterConfig['model'];
            if (typeof tag === 'string')
                registered.set(tag, model.id);
        }
        let found = 0;
        for (const entry of list) {
            const tag = isObject(entry) && typeof entry['name'] === 'string' ? entry['name'] : undefined;
            const resource = ollamaResource(host, endpoint, entry, tag === undefined ? undefined : registered.get(tag));
            if (resource === undefined)
                continue;
            resources.push(resource);
            found += 1;
        }
        sources.push({
            id: host.id,
            kind: 'ollama_model',
            label: host.name,
            ok: true,
            detail: `read ${found} installed model(s) from ${baseOf(endpoint)}/api/tags`,
            found,
        });
    }
    return { resources, sources };
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
    const id = scannedWorkflowId(displayName, document.source);
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
/**
 * Run one host's discovery pass, starting and releasing its engine as needed.
 *
 * @param hub - the live hub.
 * @param host - the ComfyUI host.
 * @param options - probes, budgets, and the workflow directory.
 * @returns everything the pass produced.
 */
async function runHostPass(hub, host, options) {
    const warnings = [];
    const sources = [];
    const resources = [];
    const descriptors = [];
    const seen = new Set();
    const endpoint = host.runtime.endpoint;
    let engineRunning = false;
    let startedEngine = false;
    let releasedEngine = false;
    const ready = await engineReady(options.probes, host, options.timeoutMs);
    engineRunning = ready.running;
    if (!engineRunning && options.startEngine) {
        const started = await startEngine(hub, host, options.startupTimeoutMs, options.log, async () => {
            const check = await engineReady(options.probes, host, options.timeoutMs);
            return check.running;
        });
        startedEngine = started.owned;
        if (started.owned || started.detail.includes('already running')) {
            const recheck = await engineReady(options.probes, host, options.timeoutMs);
            engineRunning = recheck.running;
            if (!engineRunning)
                warnings.push(`${host.name}: ${recheck.detail}`);
        }
        else {
            warnings.push(`${host.name}: ${started.detail}`);
        }
    }
    else if (!engineRunning) {
        warnings.push(`${host.name}: not answering and this scan was not allowed to start it (${ready.detail})`);
    }
    // The host always reports itself, even when it contributed nothing. A
    // discovery source that stays silent when it fails is indistinguishable from
    // one that was never asked, and "ComfyUI is not running" is exactly the fact
    // the page exists to show.
    sources.push({
        id: host.id,
        kind: 'comfyui_workflow',
        label: host.name,
        ok: engineRunning,
        detail: engineRunning
            ? `${ready.running ? ready.detail : 'became ready after this scan started it'}`
            : `${host.name} is not answering: ${ready.detail}`,
        found: 0,
        ...(startedEngine ? { startedEngine: true } : {}),
    });
    let io;
    let classes;
    if (engineRunning && endpoint !== undefined && endpoint.trim().length > 0) {
        const base = baseOf(endpoint);
        const info = await options.probes.request(`${base}/object_info`, options.timeoutMs);
        if (info.ok) {
            io = readComfyNodeIo(info.body);
            classes = new Set(isObject(info.body) ? Object.keys(info.body) : []);
        }
        else {
            warnings.push(`${host.name}: node definitions could not be read, so conversions cannot be verified (${info.detail})`);
        }
    }
    if (engineRunning && endpoint !== undefined && endpoint.trim().length > 0) {
        const listed = await readServerWorkflows(options.probes, endpoint, options.timeoutMs);
        if (listed.error !== undefined) {
            sources.push({ id: host.id, kind: 'comfyui_workflow', label: host.name, ok: false, detail: listed.error, found: 0 });
        }
        let found = 0;
        for (const document of listed.documents) {
            const built = workflowResource(host, document, io, classes, engineRunning);
            if (built === undefined || seen.has(built.resource.id))
                continue;
            seen.add(built.resource.id);
            resources.push(built.resource);
            if (built.descriptor !== undefined)
                descriptors.push(built.descriptor);
            found += 1;
        }
        if (listed.error === undefined) {
            sources.push({
                id: host.id,
                kind: 'comfyui_workflow',
                label: host.name,
                ok: true,
                detail: `read ${found} saved workflow(s) from ${baseOf(endpoint)}/userdata`,
                found,
                ...(startedEngine ? { startedEngine: true } : {}),
            });
        }
    }
    // The configured directory is read even when the engine is down: an API-format
    // workflow needs no engine to be understood, and this is the only source that
    // survives ComfyUI being off.
    const owner = host;
    const workflowDir = options.workflowDir;
    if (workflowDir !== undefined && workflowDir.trim().length > 0) {
        const read = await readDirectoryWorkflows(options.probes, workflowDir);
        if (read.error !== undefined) {
            sources.push({ id: 'workflow-dir', kind: 'comfyui_workflow', label: workflowDir, ok: false, detail: read.error, found: 0 });
        }
        let found = 0;
        for (const document of read.documents) {
            const built = workflowResource(owner, document, io, classes, engineRunning);
            if (built === undefined || seen.has(built.resource.id))
                continue;
            seen.add(built.resource.id);
            resources.push(built.resource);
            if (built.descriptor !== undefined)
                descriptors.push(built.descriptor);
            found += 1;
        }
        if (read.error === undefined) {
            sources.push({
                id: 'workflow-dir',
                kind: 'comfyui_workflow',
                label: workflowDir,
                ok: true,
                detail: `read ${found} workflow file(s) from ${workflowDir}`,
                found,
            });
        }
    }
    const pass = {
        hostId: host.id,
        label: host.name,
        resources,
        descriptors,
        warnings,
        sources,
        reachable: engineRunning,
        engineRunning,
        startedEngine,
        releasedEngine,
    };
    // A reachable pass replaces the cache; an unreachable one must not erase it.
    if (engineRunning) {
        hostCache.set(host.id, { resources, descriptors });
    }
    return pass;
}
/**
 * Discover one host's workflows, sharing a single pass with any concurrent caller.
 *
 * @param hub - the live hub.
 * @param host - the host to scan.
 * @param options - probes, budgets, directory, and launch permission.
 * @returns the pass, and the engine id to release afterwards when one was started.
 */
async function hostPass(hub, host, options) {
    const pending = inFlight.get(host.id);
    if (pending !== undefined) {
        options.log(`scan: joining the pass already running for ${host.name}`);
        return { pass: await pending };
    }
    const engineKey = `engine-${host.id}`;
    const task = runHostPass(hub, host, options);
    inFlight.set(host.id, task);
    try {
        const pass = await task;
        const engineId = pass.startedEngine && ownedEngines.has(engineKey) ? engineKey : undefined;
        return engineId === undefined ? { pass } : { pass, engineId };
    }
    finally {
        inFlight.delete(host.id);
    }
}
/**
 * The cached resources for a host, marked as belonging to a stopped engine.
 *
 * @param hostId - the host to read.
 * @param detail - why the engine is not answering.
 * @returns the resources, or an empty array when nothing was ever discovered.
 */
function cachedResources(hostId, detail) {
    const cached = hostCache.get(hostId);
    if (cached === undefined)
        return [];
    return cached.resources.map((resource) => ({
        ...resource,
        engineRunning: false,
        runnable: resource.runnable,
        detail: resource.runnable
            ? `${resource.detail} (ComfyUI is not running; it will be started on demand when this workflow is invoked)`
            : resource.detail,
        diagnostics: [...(resource.diagnostics ?? []), detail],
    }));
}
/** The cached providers for a host, so a failed scan does not unregister them. */
function cachedDescriptors(hostId) {
    return hostCache.get(hostId)?.descriptors ?? [];
}
/**
 * Scan this machine for the local AI resources the settings page lists.
 *
 * @param hub - the live hub the runnable workflows are published to.
 * @param options - the declared hosts, the workflow directory, and the effects.
 * @returns the resources, the per-source report, and the warnings.
 */
export async function scanLocalResources(hub, options) {
    const probes = { ...DEFAULT_SCAN_PROBES, ...(options.probes ?? {}) };
    const timeoutMs = Math.max(1, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
    const startupTimeoutMs = Math.max(1_000, options.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS);
    const log = options.log ?? (() => { });
    const warnings = [];
    // The two halves never share a failure path: each is awaited on its own, and
    // each contains its own errors, so one engine being down can only cost its own
    // rows.
    const ollama = await scanOllama(hub, options.hosts, probes, timeoutMs);
    const comfyHosts = options.hosts.filter((host) => host.adapter === 'comfyui');
    const resources = [...ollama.resources];
    const sources = [...ollama.sources];
    const descriptors = [];
    const engineIdsToRelease = [];
    for (const host of comfyHosts) {
        const { pass, engineId } = await hostPass(hub, host, {
            probes,
            timeoutMs,
            startupTimeoutMs,
            ...(options.workflowDir === undefined ? {} : { workflowDir: options.workflowDir }),
            startEngine: options.startEngine !== false,
            log,
        });
        resources.push(...pass.resources);
        sources.push(...pass.sources);
        warnings.push(...pass.warnings);
        descriptors.push(...pass.descriptors);
        if (engineId !== undefined)
            engineIdsToRelease.push(engineId);
        // A host that could not be reached keeps whatever was discovered last time:
        // a temporary failure must not erase resources that were working, nor
        // unregister the providers that serve them.
        if (!pass.reachable) {
            const cached = cachedResources(host.id, pass.warnings[0] ?? `${pass.label} is not answering`);
            if (cached.length > 0) {
                warnings.push(`${pass.label} could not be reached, so ${cached.length} previously discovered workflow(s) are shown from ` +
                    'the last successful scan and remain registered.');
                resources.push(...cached.filter((entry) => !resources.some((existing) => existing.id === entry.id)));
                descriptors.push(...cachedDescriptors(host.id));
            }
        }
    }
    let registered = 0;
    if (options.register !== false && descriptors.length > 0) {
        registered = hub.publishScannedModels(dedupeById(descriptors));
        log(`scan: published ${registered} provider(s) in the catalog`);
    }
    else if (descriptors.length === 0 && comfyHosts.length > 0) {
        // Nothing fresh and nothing cached: leave the catalog untouched rather than
        // replacing a working set with an empty one.
        log('scan: no runnable workflows were discovered; the existing catalog was left as it was');
    }
    // The engines this scan started are shut down only after discovery is complete
    // and only when nothing else needs them.
    for (const engineId of engineIdsToRelease) {
        const released = await releaseEngine(hub, engineId);
        for (const source of sources) {
            if (source.id === engineId.replace(/^engine-/, '') || source.startedEngine === true) {
                source.releasedEngine = released.stopped;
            }
        }
        if (!released.stopped && released.detail.length > 0) {
            warnings.push(`${released.detail}.`);
        }
        log(`scan: engine ${engineId}: ${released.detail}`);
    }
    return {
        resources,
        sources,
        warnings,
        registered,
        generatedAt: new Date().toISOString(),
    };
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
    hostCache.clear();
    inFlight.clear();
    ownedEngines.clear();
}
/** The digest helper, re-exported so a caller can compare analyses cheaply. */
export { stableDigest };
