/**
 * Where the workflows a scan found are remembered between launches.
 *
 * ## Why this file exists
 *
 * ComfyUI lists its saved workflows over its own API, so a hub that only reads
 * the engine can only answer "what workflows does this machine have?" while the
 * engine is up. The engine is not the machine: the workflow files are on disk,
 * and a user who has spent an afternoon building graphs has not lost them because
 * a server process is not running. Losing the list on every restart — or on a
 * single failed refresh — is the bug this store removes.
 *
 * The rule the store exists to enforce is the one the scan already states about
 * *sources*: **a workflow is remembered until a scan positively disproves it.**
 * A connection refusal, a timeout, a 500 or an unreadable directory are all
 * "unknown", never "gone"; only a successful scan of the same source that no
 * longer contains the file is evidence of removal. So observations are written
 * here after every scan and merged back in as the lowest-priority source, which
 * is what keeps a workflow visible — with its readiness and its contract — while
 * ComfyUI is stopped.
 *
 * ## What it deliberately is not
 *
 * - **Not the catalog.** `models.json` is where *engines and their providers* are
 *   declared by an operator. A scan's observations are machine state that changes
 *   without anyone editing a file, so they live in their own document with their
 *   own owner and lifetime, exactly as the capability policy does.
 * - **Not a cache of the engine's answer alone.** Observations from local
 *   directories are stored the same way, which is what makes offline discovery
 *   work on a first launch with no previous successful connection.
 * - **Not authoritative.** A live engine answer and a file found on disk both
 *   outrank a stored observation; this document is consulted last, and every
 *   entry it contributes is reported to the user as a cached observation.
 *
 * ## Identity and scoping
 *
 * Entries are scoped to a host *and* its endpoint, because a workflow's provider
 * id is only meaningful against the engine that will run it: pointing the catalog
 * at a different ComfyUI must not resurrect the workflows of the old one. Within a
 * scope an entry is keyed by its **source** — the file path or the retrieval URL —
 * which is the same identity the scan uses to deduplicate, so a workflow found
 * again updates one row instead of adding a second.
 *
 * The file sits under the harness home, beside the capability policy, for the
 * same reason it does: that is the one place a third-party plugin can own durable
 * state, and it is outside both the catalog and the profile's generated config.
 *
 * @module dsh-ai-model-hub/dsh-plugin/workflow-catalog
 */
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { withoutEngineAnnotation } from "./scan.js";
/** The document's schema version, so a future change can refuse rather than guess. */
export const WORKFLOW_CATALOG_VERSION = 1;
/** How many workflows one scope may remember. */
const MAX_REMEMBERED_PER_SCOPE = 500;
/** The file's name under the plugin's state directory. */
const CATALOG_FILE = 'workflow-catalog.json';
/**
 * An environment override for the catalog's location.
 *
 * A plugin that owns durable state has to be testable without writing into the
 * real harness home, and "pass a path" is not enough when the default is reached
 * deep inside a code path under test. An explicit environment variable keeps the
 * production default and gives a test a supported way to redirect it.
 */
const CATALOG_PATH_ENV = 'DSH_AI_MODEL_HUB_WORKFLOW_CATALOG';
/**
 * The workflow catalog file's path.
 *
 * @param override - an explicit path, when the deployment names one.
 * @returns the path to read and write.
 */
export function defaultWorkflowCatalogPath(override) {
    const expanded = override?.trim();
    if (expanded !== undefined && expanded.length > 0)
        return expanded;
    const environment = process.env[CATALOG_PATH_ENV]?.trim();
    if (environment !== undefined && environment.length > 0)
        return environment;
    return join(homedir(), '.dsh', 'plugins', 'dsh-ai-model-hub', CATALOG_FILE);
}
/**
 * A scope's key: the host, plus the endpoint its observations belong to.
 *
 * The endpoint is part of the identity on purpose. A scan's cache, a workflow's
 * provider id and the engine that will run the graph all bind to one endpoint, so
 * an observation made against a different one is not a fact about this scope.
 *
 * @param hostId - the host's id in the catalog.
 * @param endpoint - the host's configured endpoint, when it has one.
 * @returns a stable key.
 */
export function workflowScopeKey(hostId, endpoint) {
    return hostId + '\u0000' + (endpoint ?? '');
}
/** Whether a value is a plain JSON object. */
function isObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}
/** Read one remembered workflow, returning `undefined` when it is unusable. */
function parseWorkflow(raw) {
    if (!isObject(raw))
        return undefined;
    const id = raw['id'];
    const name = raw['name'];
    const source = raw['source'];
    const status = raw['status'];
    if (typeof id !== 'string' || id.length === 0)
        return undefined;
    if (typeof name !== 'string' || name.length === 0)
        return undefined;
    if (typeof source !== 'string' || source.length === 0)
        return undefined;
    if (typeof status !== 'string')
        return undefined;
    return {
        id,
        name,
        source,
        status: status,
        runnable: raw['runnable'] === true,
        detail: typeof raw['detail'] === 'string' ? raw['detail'] : '',
        checkedAt: typeof raw['checkedAt'] === 'string' ? raw['checkedAt'] : new Date(0).toISOString(),
        engineRunning: raw['engineRunning'] === true,
        ...(raw['format'] === 'api' || raw['format'] === 'ui' ? { format: raw['format'] } : {}),
        ...(Array.isArray(raw['capabilities']) ? { capabilities: raw['capabilities'].filter((entry) => typeof entry === 'string') } : {}),
        ...(Array.isArray(raw['inputs']) ? { inputs: raw['inputs'] } : {}),
        ...(Array.isArray(raw['outputs']) ? { outputs: raw['outputs'] } : {}),
        ...(Array.isArray(raw['diagnostics']) ? { diagnostics: raw['diagnostics'].filter((entry) => typeof entry === 'string') } : {}),
        ...(typeof raw['sizeBytes'] === 'number' ? { sizeBytes: raw['sizeBytes'] } : {}),
        ...(raw['descriptor'] === undefined ? {} : { descriptor: raw['descriptor'] }),
    };
}
/** Read one scope, returning `undefined` when it names no host. */
function parseScope(raw) {
    if (!isObject(raw))
        return undefined;
    const hostId = raw['hostId'];
    if (typeof hostId !== 'string' || hostId.length === 0)
        return undefined;
    const workflows = Array.isArray(raw['workflows'])
        ? raw['workflows'].map(parseWorkflow).filter((entry) => entry !== undefined)
        : [];
    return {
        hostId,
        ...(typeof raw['endpoint'] === 'string' ? { endpoint: raw['endpoint'] } : {}),
        updatedAt: typeof raw['updatedAt'] === 'string' ? raw['updatedAt'] : new Date(0).toISOString(),
        workflows: workflows.slice(0, MAX_REMEMBERED_PER_SCOPE),
    };
}
/**
 * Read and validate the saved workflow catalog.
 *
 * A missing file is the normal first run, not an error. A malformed one is
 * reported and discarded whole: re-showing half of a document nobody can parse
 * would put invented workflows on the page, which is worse than showing none and
 * saying why.
 *
 * @param path - the file to read.
 * @param log - the plugin's logger.
 * @returns the document (empty when nothing usable was found) and the issues.
 */
export async function loadWorkflowCatalog(path, log) {
    let raw;
    try {
        raw = await readFile(path, 'utf8');
    }
    catch (error) {
        const code = error.code;
        if (code === 'ENOENT')
            return { document: { version: WORKFLOW_CATALOG_VERSION, scopes: [] }, path, found: false, issues: [] };
        const message = `could not read the saved workflow catalog at ${path}: ${error instanceof Error ? error.message : String(error)}`;
        log.warn(message);
        return { document: { version: WORKFLOW_CATALOG_VERSION, scopes: [] }, path, found: false, issues: [message] };
    }
    let parsed;
    try {
        parsed = JSON.parse(raw);
    }
    catch (error) {
        const message = `the saved workflow catalog at ${path} is not valid JSON (${error instanceof Error ? error.message : String(error)}); it was ignored and no workflows are cached`;
        log.warn(message);
        return { document: { version: WORKFLOW_CATALOG_VERSION, scopes: [] }, path, found: true, issues: [message] };
    }
    if (!isObject(parsed) || parsed['version'] !== WORKFLOW_CATALOG_VERSION || !Array.isArray(parsed['scopes'])) {
        const message = `the saved workflow catalog at ${path} is not a version ${WORKFLOW_CATALOG_VERSION} document; it was ignored and no workflows are cached`;
        log.warn(message);
        return { document: { version: WORKFLOW_CATALOG_VERSION, scopes: [] }, path, found: true, issues: [message] };
    }
    const scopes = parsed['scopes'].map(parseScope).filter((entry) => entry !== undefined);
    return { document: { version: WORKFLOW_CATALOG_VERSION, scopes }, path, found: true, issues: [] };
}
/**
 * Write the workflow catalog atomically.
 *
 * Temp-file-then-rename in the same directory, following the capability policy's
 * store: the rename is the only step that changes what the next load reads, so a
 * crash or a concurrent scan cannot leave a truncated document behind.
 *
 * @param path - where to write.
 * @param document - the document to persist.
 * @param log - the plugin's logger.
 * @returns the issues found; empty means it was written.
 */
export async function saveWorkflowCatalog(path, document, log) {
    const serialized = JSON.stringify(document, null, 2) + '\n';
    try {
        await mkdir(dirname(path), { recursive: true });
        const temporary = `${path}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
        try {
            await writeFile(temporary, serialized, { mode: 0o600 });
            await rename(temporary, path);
        }
        finally {
            await rm(temporary, { force: true });
        }
        return [];
    }
    catch (error) {
        const message = `could not save the workflow catalog to ${path}: ${error instanceof Error ? error.message : String(error)}. Discovered workflows are listed for this session only.`;
        log.warn(message);
        return [message];
    }
}
/**
 * Turn a remembered workflow back into a list row.
 *
 * The row is marked `cached` and keeps its original `checkedAt`: the observation
 * is real, it is simply not from this scan, and saying otherwise would make a
 * stale readiness claim look live. A stored workflow is never made runnable by
 * this function — `runnable` is taken from the observation, because a workflow
 * whose contract was never verified must not become executable by being read back
 * from disk.
 *
 * @param remembered - the remembered workflow.
 * @param stateSuffix - what to append, naming the engine's state now.
 * @returns the list row.
 */
export function rememberedResource(remembered, stateSuffix) {
    return {
        id: remembered.id,
        kind: 'comfyui_workflow',
        typeLabel: 'ComfyUI Workflow',
        name: remembered.name,
        source: remembered.source,
        status: remembered.status,
        detail: withoutEngineAnnotation(remembered.detail) + stateSuffix,
        runnable: remembered.runnable,
        engineRunning: false,
        availability: 'unavailable',
        freshness: 'cached',
        checkedAt: remembered.checkedAt,
        ...(remembered.format === undefined ? {} : { format: remembered.format }),
        ...(remembered.capabilities === undefined ? {} : { capabilities: remembered.capabilities }),
        ...(remembered.inputs === undefined ? {} : { inputs: remembered.inputs }),
        ...(remembered.outputs === undefined ? {} : { outputs: remembered.outputs }),
        ...(remembered.diagnostics === undefined ? {} : { diagnostics: remembered.diagnostics }),
        ...(remembered.sizeBytes === undefined ? {} : { sizeBytes: remembered.sizeBytes }),
        ...(remembered.runnable ? { modelId: remembered.id } : {}),
    };
}
/**
 * The remembered observations, shaped as the scan's fallback source.
 *
 * One snapshot per scope, so a host whose endpoint the catalog has since changed
 * is not handed the observations of the engine it used to point at. Each snapshot
 * carries the provider contracts of its runnable workflows as well as the rows,
 * which is what allows a workflow to be *executed* on a later launch with the
 * engine stopped: the graph, bindings and output descriptor are the contract, and
 * a contract kept on disk is as good as one just read.
 *
 * @param document - the loaded document.
 * @returns one snapshot per scope that has at least one usable workflow.
 */
export function rememberedSnapshots(document) {
    const snapshots = [];
    for (const scope of document.scopes) {
        const resources = scope.workflows.map((workflow) => rememberedResource(workflow, ''));
        if (resources.length === 0)
            continue;
        const descriptors = scope.workflows
            .filter((workflow) => workflow.runnable && workflow.descriptor !== undefined)
            .map((workflow) => workflow.descriptor);
        snapshots.push({
            hostId: scope.hostId,
            ...(scope.endpoint === undefined ? {} : { endpoint: scope.endpoint }),
            // A remembered snapshot is a fallback, never evidence: it may add a
            // workflow to the list but can never be the reason one disappears.
            authoritative: false,
            resources,
            descriptors,
        });
    }
    return snapshots;
}
/**
 * Record one scan's observations into a document.
 *
 * The document is rebuilt rather than appended to, for the reason the scan's own
 * cache is rebuilt: a scan that successfully enumerated a source is evidence about
 * *that source*, and re-showing an entry it deliberately dropped would resurrect a
 * workflow the operator deleted. Two rules follow, and both are the point of
 * persisting at all:
 *
 * 1. **A scope this scan did not reach keeps everything it had.** A host the pass
 *    never saw contributes no snapshot, and its previous entries are carried over
 *    unchanged.
 * 2. **Only a complete scan may replace a scope.** When the snapshot is not
 *    authoritative — an engine that refused, a directory that could not be listed,
 *    a file that could not be read — its observations are *merged into* what was
 *    remembered, so an outage can never erase a workflow. A complete scan replaces
 *    the scope outright, which is what makes a deletion stick.
 *
 * @param previous - the document that was loaded before the scan.
 * @param scanned - the snapshots this scan produced.
 * @param scannedAt - when the scan finished, as an ISO timestamp.
 * @returns the document to persist.
 */
export function recordScan(previous, scanned, scannedAt) {
    const byKey = new Map();
    for (const scope of previous.scopes)
        byKey.set(workflowScopeKey(scope.hostId, scope.endpoint), scope);
    for (const snapshot of scanned) {
        const key = workflowScopeKey(snapshot.hostId, snapshot.endpoint);
        // The descriptor is the only thing that makes a row executable, and it is
        // indexed by the row's provider id rather than its position in the list.
        const descriptorById = new Map();
        for (const descriptor of snapshot.descriptors)
            descriptorById.set(descriptor.id, descriptor);
        const observed = snapshot.resources
            .filter((resource) => resource.source.length > 0)
            .slice(0, MAX_REMEMBERED_PER_SCOPE)
            .map((resource) => {
            const descriptor = resource.modelId === undefined ? undefined : descriptorById.get(resource.modelId);
            return {
                id: resource.id,
                name: resource.name,
                source: resource.source,
                status: resource.status,
                runnable: resource.runnable,
                // Store the observation's own sentence, not this pass's statement about
                // the engine: readiness is re-derived on every scan, so a stored copy of
                // it would only be re-shown as fact long after it stopped being one.
                detail: withoutEngineAnnotation(resource.detail),
                checkedAt: resource.checkedAt ?? scannedAt,
                engineRunning: resource.engineRunning === true,
                ...(resource.format === undefined ? {} : { format: resource.format }),
                ...(resource.capabilities === undefined ? {} : { capabilities: [...resource.capabilities] }),
                ...(resource.inputs === undefined ? {} : { inputs: resource.inputs }),
                ...(resource.outputs === undefined ? {} : { outputs: resource.outputs }),
                ...(resource.diagnostics === undefined ? {} : { diagnostics: [...resource.diagnostics] }),
                ...(resource.sizeBytes === undefined ? {} : { sizeBytes: resource.sizeBytes }),
                ...(descriptor === undefined ? {} : { descriptor }),
            };
        });
        // An incomplete pass adds and updates; it never removes.
        const carried = snapshot.authoritative === true
            ? new Map()
            : new Map((byKey.get(key)?.workflows ?? []).map((workflow) => [workflow.source, workflow]));
        for (const workflow of observed)
            carried.set(workflow.source, workflow);
        byKey.set(key, {
            hostId: snapshot.hostId,
            ...(snapshot.endpoint === undefined ? {} : { endpoint: snapshot.endpoint }),
            updatedAt: snapshot.authoritative === true ? scannedAt : byKey.get(key)?.updatedAt ?? scannedAt,
            workflows: [...carried.values()].slice(0, MAX_REMEMBERED_PER_SCOPE),
        });
    }
    return { version: WORKFLOW_CATALOG_VERSION, scopes: [...byKey.values()] };
}
