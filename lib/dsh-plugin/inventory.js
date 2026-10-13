/**
 * The engine inventory behind the "Local models" settings page.
 *
 * The hub already knows which engines it can talk to, where they listen, and how
 * to start them — but only the model sees that, through tools. This module answers
 * the same question for the *human*: which local engines exist on this PC, where
 * each one is installed, whether it is running, and what is inside it.
 *
 * Two data sources, deliberately kept apart:
 *
 * - **The catalog** supplies what was *configured*: hosts, their endpoints, their
 *   launch commands, and the models behind them. It is authoritative.
 * - **The machine** supplies what is *actually there*: a live probe of each
 *   endpoint, the installer directory each engine's command resolves to, and the
 *   files in a model store.
 *
 * A known engine that appears in neither is still reported, with the directories
 * that were checked, because "A1111 is not installed" is exactly the answer
 * someone opening this page wants — and a silent omission is not.
 *
 * Everything here is read-only and bounded: probes carry a timeout, directory
 * listings are capped, and no failure is fatal. A page that cannot describe one
 * engine still describes the rest.
 *
 * @module dsh-ai-model-hub/dsh-plugin/inventory
 */
import { stat, readdir, readFile } from 'node:fs/promises';
import { withinBudget } from "./probe-budget.js";
import { homedir } from 'node:os';
import { join, dirname, isAbsolute, resolve } from 'node:path';
import { installRoute } from "./compat.js";
import { resolveAdapterPath } from "../src/adapters/paths.js";
import { scanLocalResources, readyPathFor } from "./scan.js";
import { resolveConfiguredPath } from "./paths.js";
import { defaultWorkflowCatalogPath, loadWorkflowCatalog, recordScan, rememberedSnapshots, saveWorkflowCatalog, } from "./workflow-catalog.js";
/** The route the settings page reads. Absolute, no trailing slash. */
export const INVENTORY_ROUTE = '/dsh-ai-model-hub/inventory';
/** Longest a live endpoint probe may take. */
const PROBE_TIMEOUT_MS = 2500;
/** Longest a single directory listing may be, so a huge model store cannot stall the page. */
const MAX_LISTED_FILES = 40;
/** The default probes: a timed fetch and the real filesystem. */
export const DEFAULT_PROBES = {
    readFile: (path) => readFile(path, 'utf8'),
    isFile: async (path) => {
        try {
            return (await stat(path)).isFile();
        }
        catch (error) {
            if (['ENOENT', 'ENOTDIR'].includes(String(error.code)))
                return false;
            throw error;
        }
    },
    fetchJson: async (url, timeoutMs) => {
        const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
        if (!response.ok) {
            // The status travels with the error, so a caller can tell "the engine said
            // no" from "the engine is not there" without parsing a message string.
            const failure = new Error(`HTTP ${response.status}`);
            failure.status = response.status;
            throw failure;
        }
        return await response.json();
    },
    isDirectory: async (path) => {
        try {
            return (await stat(path)).isDirectory();
        }
        catch (error) {
            if (['ENOENT', 'ENOTDIR'].includes(String(error.code)))
                return false;
            throw error;
        }
    },
    listDir: async (path) => {
        try {
            const entries = await readdir(path, { withFileTypes: true });
            return await Promise.all(entries.slice(0, 200).map(async (entry) => {
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
};
/**
 * Engines this page looks for whether or not the catalog declares them.
 *
 * This is a *display* list, not a behavioural one: nothing here is probed for
 * routing, and adding an engine never changes what the hub can do. It exists so a
 * user who has just installed Forge can see that the hub has not been told about
 * it yet, instead of wondering why an engine they use is missing from the page.
 */
const KNOWN_ENGINES = [
    {
        id: 'ollama',
        name: 'Ollama',
        adapter: 'openai_compatible',
        endpoint: 'http://127.0.0.1:11434',
        installCandidates: ['%LOCALAPPDATA%/Programs/Ollama', '%PROGRAMFILES%/Ollama', '%USERPROFILE%/.ollama'],
        storeCandidates: [{ label: 'Ollama model store', path: '%USERPROFILE%/.ollama/models' }],
    },
    {
        id: 'comfyui',
        name: 'ComfyUI',
        adapter: 'comfyui',
        endpoint: 'http://127.0.0.1:8188',
        installCandidates: ['C:/ComfyUI', '%USERPROFILE%/ComfyUI', '%USERPROFILE%/Documents/ComfyUI'],
        // Deliberately NOT the model directory. This page lists workflows, and a
        // ComfyUI install's `models/` tree is exactly the checkpoint/LoRA/VAE
        // inventory the hub treats as workflow-internal detail — enumerated here it
        // would read as a list of things a user can pick, which none of them are.
        storeCandidates: [{ label: 'ComfyUI saved workflows', path: '%INSTALL%/user/default/workflows' }],
    },
    {
        id: 'a1111',
        name: 'Stable Diffusion WebUI (A1111 / Forge)',
        adapter: 'http_json',
        endpoint: 'http://127.0.0.1:7860',
        installCandidates: [
            'C:/stable-diffusion-webui',
            'C:/sd.webui',
            'C:/forge',
            'C:/tools/stable-diffusion-webui',
            '%USERPROFILE%/stable-diffusion-webui',
            '%USERPROFILE%/sd.webui',
        ],
        storeCandidates: [{ label: 'Checkpoints', path: '%INSTALL%/models/Stable-diffusion', listFiles: true }],
    },
];
/**
 * Expand `%VAR%` and `~` in a configured path.
 * @param path - the path to expand.
 * @returns an absolute path.
 */
function expandPath(path) {
    let result = path.replace(/%([A-Za-z_]+)%/g, (match, name) => {
        const value = process.env[name] ?? process.env[name.toUpperCase()];
        return value ?? match;
    });
    if (result === '~')
        result = homedir();
    else if (result.startsWith('~/') || result.startsWith('~\\'))
        result = join(homedir(), result.slice(2));
    return result;
}
/**
 * Render a byte count for display.
 * @param bytes - the size.
 * @returns a short human-readable string.
 */
export function formatBytes(bytes) {
    if (bytes < 1024)
        return `${bytes} B`;
    const units = ['KiB', 'MiB', 'GiB', 'TiB'];
    let value = bytes / 1024;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
    }
    return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit] ?? 'GiB'}`;
}
/**
 * Build the complete inventory.
 *
 * @param hub - the live hub, read for models, status, and machine facts.
 * @param options - the catalog's declared hosts, the path it was read from, the
 *   deployment flags, and injectable probes.
 * @returns the inventory the settings page renders.
 */
/**
 * Spell a path the way the inventory reports it.
 *
 * Forward slashes on every platform, so a page, a route payload, a remembered
 * workflow's source and a config value someone pastes back all read the same —
 * and so a path does not change spelling between "configured" and "discovered".
 * `join` is still what composes them; only the last mile is spelled once.
 *
 * @param path - a path from the filesystem or the configuration.
 * @returns the path with forward slashes.
 */
function asReported(path) {
    return path.replace(/\\/g, '/');
}
export async function buildInventory(hub, options) {
    const requestedBudget = options.scanTimeoutMs ?? PROBE_TIMEOUT_MS;
    const budget = Number.isFinite(requestedBudget) ? Math.max(1, Math.min(60_000, requestedBudget)) : PROBE_TIMEOUT_MS;
    const logger = options.logger ?? {
        debug: (message) => options.log?.(message),
        info: (message) => options.log?.(message),
        warn: (message) => options.log?.('warn: ' + message),
        error: (message) => options.log?.('error: ' + message),
    };
    const configuredPath = (path) => {
        const expanded = expandPath(path);
        return isAbsolute(expanded) || /^[A-Za-z]:[/\\]/.test(expanded) || !/[/\\]/.test(expanded)
            ? expanded : resolve(dirname(options.catalogPath), expanded);
    };
    const rawProbes = options.probes ?? DEFAULT_PROBES;
    const probes = {
        fetchJson: (url, timeout) => withinBudget(() => rawProbes.fetchJson(url, timeout), Math.min(timeout, budget)),
        isDirectory: (path) => withinBudget(() => rawProbes.isDirectory(path), budget),
        listDir: (path) => withinBudget(() => rawProbes.listDir(path), budget),
        ...(rawProbes.isFile === undefined ? {} : { isFile: (path) => withinBudget(() => rawProbes.isFile(path), budget) }),
        ...(rawProbes.readFile === undefined ? {} : { readFile: (path) => withinBudget(() => rawProbes.readFile(path), budget) }),
    };
    const declaredHosts = options.hosts;
    const views = hub.listModels({ includeDisabled: true });
    const reported = new Set();
    const candidates = declaredHosts.map((host) => {
        const known = KNOWN_ENGINES.find((entry) => entry.id === host.id || entry.id === host.runtime.engine);
        reported.add(host.id);
        if (known !== undefined)
            reported.add(known.id);
        return { host, ...(known === undefined ? {} : { known }), configured: true };
    });
    for (const known of KNOWN_ENGINES) {
        if (reported.has(known.id))
            continue;
        candidates.push({
            host: { id: known.id, name: known.name, adapter: known.adapter,
                runtime: { engine: known.id, adapter: known.adapter, endpoint: known.endpoint } },
            known, configured: false,
        });
    }
    // Installation evidence is collected independently from endpoint liveness.
    const engines = await Promise.all(candidates.map(async ({ host, known, configured }) => {
        const lifecycle = host.lifecycle ?? views.find((entry) => entry.model.hostId === host.id)?.model.lifecycle;
        const start = lifecycle?.start;
        const configuredPaths = [start?.cwd, start?.command].filter((path) => path !== undefined).map(configuredPath);
        const errors = [];
        const install = await resolveInstall(probes, [
            ...(start?.cwd === undefined ? [] : [configuredPath(start.cwd)]),
            ...(known?.installCandidates ?? []),
        ], known?.id ?? host.runtime.engine, start?.command === undefined ? undefined : configuredPath(start.command), errors);
        const [endpoint, stores, models] = await Promise.all([
            describeEndpoint(probes, host, options.probe !== false),
            describeStores(probes, known, install.path, configuredPaths, host.runtime.env?.['OLLAMA_MODELS']).catch((error) => { errors.push(`model stores: ${error instanceof Error ? error.message : String(error)}`); return []; }),
            describeEngineModels(probes, known, host.runtime.endpoint, install.path),
        ]);
        if (endpoint.error !== undefined)
            errors.push(endpoint.error);
        if (models.error !== undefined)
            errors.push(models.error);
        const installation = install.path !== undefined ? 'installed' : errors.some((entry) => entry.startsWith('installation:')) ? 'unknown' : 'not_found';
        return {
            id: host.id, name: host.name, adapter: host.adapter, configured,
            ...(host.runtime.endpoint === undefined ? {} : { endpoint: host.runtime.endpoint }),
            running: endpoint.running, statusDetail: endpoint.statusDetail,
            state: endpoint.running ? 'running' : installation === 'installed' ? 'installed_unreachable' : configured || endpoint.unverified || installation === 'unknown' ? 'unknown' : 'not_found',
            installation, configuredPaths: configuredPaths.map(asReported), checkedAt: new Date().toISOString(), errors,
            startable: options.allowProcessLaunch && lifecycle?.startable === true,
            ...(start === undefined ? {} : { launch: { command: asReported(expandPath(start.command)), args: start.args ?? [], ...(start.cwd === undefined ? {} : { cwd: asReported(expandPath(start.cwd)) }) } }),
            installSource: install.source,
            ...(install.path === undefined ? { checkedPaths: install.checked.map(asReported) } : { installPath: asReported(install.path) }),
            storePaths: stores.map((store) => ({ ...store, path: asReported(store.path) })), models: models.models, modelsSource: models.modelsSource,
            servedModelIds: views.filter((entry) => entry.model.hostId === host.id).map((entry) => entry.model.id),
        };
    }));
    const workflowDirsByHost = {};
    for (const engine of engines) {
        if (engine.adapter === 'comfyui')
            workflowDirsByHost[engine.id] = engine.storePaths.filter((store) => store.exists).map((store) => store.path);
    }
    // What earlier scans discovered, read *before* this one runs so it can be handed
    // to the scan as a fallback source. This is the difference between "ComfyUI is
    // down, so there are no workflows" and "ComfyUI is down, and here are the
    // workflows it saved last time we looked" — including on a first launch, where
    // the file was written by a scan that never needed a live API at all.
    const catalogDir = dirname(resolveAdapterPath(options.catalogPath, undefined).absolute);
    const workflowCatalogPath = defaultWorkflowCatalogPath(options.workflowCatalogPath);
    const loaded = options.persistWorkflows === false
        ? { document: { version: 1, scopes: [] }, issues: [] }
        : await loadWorkflowCatalog(workflowCatalogPath, logger);
    const remembered = rememberedSnapshots(loaded.document);
    const scanned = [];
    const scannedHosts = new Set();
    const phases = [];
    let progress = '';
    // The scan is the page's actual job, so it always runs. It is contained by
    // construction — each engine is contacted on its own and reports its own
    // failure — so a machine with only one of the two engines still lists what it
    // has instead of showing an error page.
    //
    // `manageEngines` is the whole difference between looking and scanning: it is set
    // only by an explicit Scan, and it is what lets that action start the configured
    // engines and hold them until the read finishes. Every other read of this route
    // — opening the page, reloading it — stays observational.
    const scan = await scanLocalResources(hub, {
        hosts: declaredHosts,
        catalogPath: options.catalogPath,
        workflowDirsByHost,
        manageEngines: options.manageEngines === true,
        allowProcessLaunch: options.allowProcessLaunch,
        ...(options.engineStartupTimeoutMs === undefined ? {} : { engineStartupTimeoutMs: options.engineStartupTimeoutMs }),
        onProgress: (update) => {
            if (phases.at(-1) !== update.phase)
                phases.push(update.phase);
            progress = update.message;
            options.onProgress?.(update);
        },
        ...(options.workflowDir === undefined || options.workflowDir.trim().length === 0
            ? {}
            : { workflowDir: resolveConfiguredPath(options.workflowDir, catalogDir) }),
        ...(remembered.length === 0 ? {} : { persisted: remembered }),
        onHostScanned: (snapshot) => {
            scannedHosts.add(snapshot.hostId);
            scanned.push(snapshot);
        },
        ...(options.scanTimeoutMs === undefined ? {} : { timeoutMs: options.scanTimeoutMs }),
        // Only an injected probe set is adapted; otherwise the scan keeps its own
        // status-aware fetch, which is the one that can tell a 404 from bad JSON.
        ...(options.probes === undefined ? {} : { probes: scanProbesFrom(options.probes) }),
        ...(options.log === undefined ? {} : { log: options.log }),
    });
    // Persist after the scan: a failed read is reported, a failed write is reported,
    // and neither stops the page from showing what this scan did find. A host this
    // pass never reached keeps its previous entries, because "not scanned" is not
    // evidence that its workflows are gone.
    const reached = scanned.filter((snapshot) => scannedHosts.has(snapshot.hostId));
    if (options.persistWorkflows !== false) {
        await recordScannedWorkflows(workflowCatalogPath, loaded.document, reached, logger);
    }
    const currentViews = hub.listModels({ includeDisabled: true });
    for (const engine of engines) {
        engine.servedModelIds = currentViews.filter((view) => view.model.hostId === engine.id).map((view) => view.model.id);
        const failures = scan.sources.filter((source) => source.id === engine.id && !source.ok).map((source) => source.detail);
        engine.errors.push(...failures.filter((detail) => !engine.errors.includes(detail)));
    }
    const capabilities = hub.listCapabilities().map((entry) => ({
        capability: String(entry.capability),
        models: entry.modelIds.length,
    }));
    return {
        generatedAt: new Date().toISOString(),
        resources: scan.resources,
        scan: {
            sources: scan.sources,
            warnings: scan.warnings,
            registered: scan.registered,
            complete: scan.complete,
            engines: scan.engines,
            phases: scan.phases.length > 0 ? scan.phases : phases,
            progress,
            managedEngines: options.manageEngines === true,
        },
        catalogPath: options.catalogPath,
        ...(hub.catalog.loadDiagnostics.length === 0
            ? {}
            : {
                diagnostics: hub.catalog.loadDiagnostics.map((entry) => ({
                    severity: String(entry.severity),
                    message: entry.message,
                })),
            }),
        ...(options.artifactRoot === undefined || options.artifactRoot.length === 0
            ? {}
            : { artifactRoot: options.artifactRoot }),
        allowProcessLaunch: options.allowProcessLaunch,
        machine: {
            ramGb: hub.machineProfile.ramGb,
            vramGb: hub.machineProfile.vramGb,
            hasGpu: hub.machineProfile.hasGpu,
            notes: hub.machineProfile.notes,
        },
        engines,
        models: currentViews.map((view) => {
            const engine = view.model.hostId !== undefined
                ? engines.find((entry) => entry.id === view.model.hostId)
                : engines.find((entry) => entry.endpoint === view.model.runtime.endpoint && entry.id === view.model.runtime.engine);
            const native = view.model.adapterConfig?.model;
            const liveModelList = engine?.running === true && engine.modelsSource === 'Ollama /api/tags';
            const nativeFound = liveModelList && typeof native === 'string' && engine.models.some((entry) => entry.id === native || entry.id === `${native}:latest`);
            const availability = view.model.enabled === false ? 'disabled' : engine?.running === false && engine.endpoint !== undefined ? 'unavailable' : liveModelList && typeof native === 'string' ? nativeFound ? 'available' : 'unavailable' : 'unknown';
            return {
                id: view.model.id,
                name: view.model.name,
                type: String(view.model.type),
                capabilities: view.model.capabilities.map(String),
                engine: view.model.runtime.engine,
                ...(view.model.hostId === undefined ? {} : { hostId: view.model.hostId }),
                availability,
                freshness: liveModelList ? 'live' : 'catalog',
                ...(engine === undefined ? {} : { checkedAt: engine.checkedAt }),
                lifecycle: String(view.status.lifecycle),
                startable: view.model.lifecycle.startable,
                ...(view.status.pid === undefined ? {} : { pid: view.status.pid }),
                detail: availability === 'disabled' ? 'disabled by catalog' : engine?.running === false ? `${engine.statusDetail}; catalog knowledge only, not live model verification`
                    : liveModelList && typeof native === 'string' ? nativeFound ? 'model listed by live Ollama /api/tags' : 'configured model not listed by live Ollama /api/tags'
                        : 'known from catalog; model/workflow execution has not been live verified',
            };
        }),
        capabilities,
        unavailable: hub.listUnservedCapabilities().map((entry) => ({
            capability: String(entry.capability),
            reason: entry.reason,
        })),
    };
}
/**
 * Find where an engine is installed.
 *
 * Only the candidates are probed, in order. Nothing is *derived* from a launch
 * command's path: the executable is a file, and taking its parent would happily
 * report `C:/` as an installation directory when the real one is missing — a
 * confident wrong answer is worse than "not found", which is what the page shows.
 *
 * @param probes - the filesystem probes.
 * @param candidates - directories to try, in order; the first that exists wins.
 * @returns the path, how it was found, and what was checked when nothing was.
 */
async function resolveInstall(probes, candidates, engine, command, errors) {
    const checked = [...new Set(candidates.map((candidate) => expandPath(candidate).replace(/[/\\]+$/, '')).filter(Boolean))];
    const markers = /comfyui/i.test(engine) ? ['main.py']
        : /ollama/i.test(engine) ? ['ollama.exe', 'ollama', 'bin/ollama']
            : /a1111|forge/i.test(engine) ? ['webui.py', 'modules/launch_utils.py'] : [];
    // The directory whose *contents* are the installation: a ComfyUI whose `main.py`
    // and `server.py` live under `src/` is installed at `src`, not at the repository
    // root that happens to contain it, and the workflow store hangs off the former.
    const installDirectory = async (candidate) => {
        const verify = async (file, missing) => {
            if (probes.isFile !== undefined) {
                try {
                    return await probes.isFile(file);
                }
                catch {
                    return false;
                }
            }
            const parts = file.replace(/\\/g, '/');
            const split = parts.lastIndexOf('/');
            const entries = await probes.listDir(parts.slice(0, split));
            // A directory that cannot be listed is not evidence either way: report what
            // the probe could not read rather than guessing that the marker is there.
            if (entries === undefined)
                throw new Error(`could not list ${parts.slice(0, split)}`);
            if (!missing && !entries.some((entry) => entry.isDirectory && entry.name === parts.slice(split + 1)))
                return false;
            return entries.some((entry) => !entry.isDirectory && entry.name === parts.slice(split + 1));
        };
        if (/comfyui/i.test(engine)) {
            for (const subdir of ['', 'src', 'ComfyUI']) {
                if ((await verify(join(candidate, subdir, 'main.py'), subdir === '')) &&
                    (await verify(join(candidate, subdir, 'server.py'), subdir === ''))) {
                    return join(candidate, subdir);
                }
            }
            return undefined;
        }
        for (const marker of markers) {
            if (await verify(join(candidate, marker), false))
                return candidate;
        }
        return undefined;
    };
    const found = await Promise.all(checked.map(async (path) => {
        try {
            if (!(await probes.isDirectory(path)))
                return undefined;
            // An arbitrary existing cwd (or .ollama data folder) is not an installation.
            return await installDirectory(path);
        }
        catch (error) {
            errors.push(`installation: ${path}: ${error instanceof Error ? error.message : String(error)}`);
            return undefined;
        }
    }));
    const index = found.findIndex((path) => path !== undefined);
    if (index >= 0) {
        const path = found[index];
        // Compared with one spelling on both sides: `join` normalizes separators on
        // Windows, so a candidate that is genuinely the installation must not be
        // reported as "found somewhere else".
        const sameLocation = path.replace(/\\/g, '/').replace(/\/+$/, '') === checked[index].replace(/\\/g, '/').replace(/\/+$/, '');
        return { path, source: sameLocation ? 'found on this machine' : `found at ${path.replace(/\\/g, '/')}`, checked };
    }
    if (command !== undefined && probes.isFile !== undefined) {
        const expanded = expandPath(command);
        const executable = /[/\\]/.test(expanded) ? expanded : undefined;
        // A Python interpreter proves Python exists, not that ComfyUI/A1111 does.
        if (executable !== undefined && !/comfyui|a1111|forge/i.test(engine)) {
            try {
                if (await probes.isFile(executable))
                    return { path: executable.replace(/[/\\][^/\\]+$/, ''), source: 'verified launch executable', checked };
            }
            catch (error) {
                errors.push(`installation: ${expanded}: ${error instanceof Error ? error.message : String(error)}`);
            }
        }
    }
    return { source: errors.some((error) => error.startsWith('installation:')) ? 'installation could not be verified' : 'not found in checked locations', checked };
}
/**
 * Probe one engine's endpoint for readiness, on the route *that engine* answers.
 *
 * The bare root is not a health check and never was. Ollama replies to `/` with
 * the plain text `Ollama is running` and ComfyUI replies with the editor's HTML,
 * so a probe that fetched the root and parsed it as JSON reported both engines as
 * stopped while they were serving requests perfectly. Each engine is asked the
 * route it actually implements, and the response is checked for a status *before*
 * anything is parsed.
 *
 * @param probes - the probes.
 * @param host - the declared host, whose engine label selects the readiness route.
 * @param live - whether to probe now; when false the report says it was not probed.
 * @returns the running flag and a human-readable detail.
 */
async function describeEndpoint(probes, host, live) {
    const endpoint = host.runtime.endpoint;
    if (endpoint === undefined)
        return { running: false, statusDetail: 'no endpoint: runtime has not been checked', unverified: true };
    if (!live)
        return { running: false, statusDetail: 'not probed yet — press Scan', unverified: true };
    const path = readyPathFor(host.runtime.engine);
    const base = endpoint.endsWith('/') ? endpoint.slice(0, -1) : endpoint;
    try {
        const body = await probes.fetchJson(`${base}${path}`, PROBE_TIMEOUT_MS);
        if (body === null || typeof body !== 'object' || Array.isArray(body))
            throw new Error('invalid readiness data');
        if (/ollama/i.test(host.runtime.engine) && !Array.isArray(body.models))
            throw new Error('invalid Ollama model list');
        if (/comfyui/i.test(host.runtime.engine) && (!('system' in body) || !Array.isArray(body.devices)))
            throw new Error('invalid ComfyUI system stats');
        return { running: true, statusDetail: `${path} answered with valid readiness data` };
    }
    catch (error) {
        const status = error.status;
        if (typeof status === 'number') {
            const detail = `${path} answered HTTP ${status}`;
            return { running: false, statusDetail: detail, error: detail };
        }
        const detail = `${path} not verified: ${error instanceof Error ? error.message : String(error)}`;
        return { running: false, statusDetail: detail, error: detail };
    }
}
/**
 * Record this scan's observations, keeping every scope it did not reach.
 *
 * @param path - the workflow catalog to write.
 * @param previous - the document that was loaded before the scan.
 * @param scanned - the observations this scan produced.
 * @param log - the plugin's logger.
 * @returns the document that is now in force, written or not.
 */
async function recordScannedWorkflows(path, previous, scanned, log) {
    const document = recordScan(previous, scanned, new Date().toISOString());
    await saveWorkflowCatalog(path, document, log);
    return document;
}
/**
 * Adapt the inventory's probes to the scan's.
 *
 * The scan needs richer outcomes than `fetchJson` gives — a connection refusal,
 * an HTTP error and a 200 that is not JSON are different problems — but a test
 * that injects `fetchJson` is describing a machine, not a transport, so its
 * throws are mapped to the closest meaning rather than ignored. When no probes
 * were injected at all the scan keeps its own status-aware fetch.
 *
 * @param probes - the injected probes.
 * @returns the scan-side effects.
 */
function scanProbesFrom(probes) {
    return {
        request: async (url, timeoutMs) => {
            try {
                return { ok: true, status: 200, body: await probes.fetchJson(url, timeoutMs) };
            }
            catch (error) {
                const status = error.status;
                const detail = error instanceof Error ? error.message : String(error);
                return typeof status === 'number'
                    ? { ok: false, kind: 'http', status, detail: `${url} answered HTTP ${status}` }
                    : { ok: false, kind: 'connection', detail: `${url} could not be reached: ${detail}` };
            }
        },
        isDirectory: probes.isDirectory,
        listDir: probes.listDir,
        ...(probes.readFile === undefined ? {} : { readFile: probes.readFile }),
    };
}
/**
 * Describe an engine's model stores.
 *
 * A store whose path names the install directory is checked under the verified
 * installation **and** under every configured launch path. That second base is
 * what makes a ComfyUI whose `user/` tree sits beside its `main.py` — an install
 * whose launcher sets `cwd` to the source directory rather than to the repository
 * root — report its saved workflows at all. It is also what keeps the answer
 * correct after the engine folder moves: the configured path may be stale, so both
 * bases are tried and the readable one wins, rather than one silently replacing
 * the other.
 *
 * A candidate that exists nowhere is still reported, for each base that was
 * checked, because "we looked here and it was not there" is the fact an operator
 * opening this page needs. The composed path keeps the separator style the
 * candidate was written with, so a page or an API response reads the same on
 * Windows and on a POSIX host.
 *
 * @param probes - the filesystem probes.
 * @param known - the known-engine entry, when the engine is one this page knows.
 * @param installPath - the resolved installation directory.
 * @param configuredPaths - launch paths the catalog configured for this engine.
 * @param configuredOllamaStore - an Ollama store the host's own env names.
 * @returns one row per store, with its contents when the store lists files.
 */
async function describeStores(probes, known, installPath, configuredPaths, configuredOllamaStore) {
    if (known === undefined)
        return [];
    const stores = [];
    const candidates = [...known.storeCandidates];
    const ollamaStore = configuredOllamaStore ?? process.env['OLLAMA_MODELS'];
    if (known.id === 'ollama' && ollamaStore)
        candidates.unshift({ label: 'Configured Ollama model store', path: ollamaStore });
    const bases = [...new Set([installPath, ...configuredPaths].filter((path) => path !== undefined))];
    for (const candidate of candidates) {
        if (candidate.path.includes('%INSTALL%')) {
            if (bases.length === 0)
                continue;
            // Every directory that could hold it, in priority order, so a stale
            // configured path cannot hide a real install or the other way round. A
            // configured *file* — a launch command, typically — is not a base for a
            // directory, and reporting `python.exe/user/default/workflows` as a checked
            // store would be noise dressed as evidence.
            let checkedBase = false;
            for (const base of bases) {
                if (!(await probes.isDirectory(base)))
                    continue;
                checkedBase = true;
                stores.push(await describeStoreAt(probes, candidate, composeStorePath(candidate.path, base)));
            }
            if (!checkedBase) {
                stores.push({ label: candidate.label, path: composeStorePath(candidate.path, bases[0]), exists: false });
            }
            continue;
        }
        stores.push(await describeStoreAt(probes, candidate, expandPath(candidate.path)));
    }
    return stores;
}
/**
 * Compose a store path from its declared template and a resolved base.
 *
 * The composed path uses forward slashes, which both Windows and POSIX accept,
 * so the same store is spelled the same way in the page, in the route's JSON and
 * on every platform this plugin runs on. `%INSTALL%/...` and `%INSTALL%\...` are
 * both understood.
 *
 * @param template - the declared store path, containing `%INSTALL%`.
 * @param base - the resolved base directory.
 * @returns the composed path.
 */
function composeStorePath(template, base) {
    const normalizedBase = base.replace(/\\/g, '/');
    const joined = template.replace(/[/\\]?%INSTALL%/g, normalizedBase + '/');
    return expandPath(joined.replace(/\\/g, '/').replace(/\/+/g, '/'));
}
/**
 * Read one store directory's row.
 *
 * @param probes - the filesystem probes.
 * @param candidate - the declared store.
 * @param path - the resolved absolute path.
 * @returns the row, with its contents when the store lists files.
 */
async function describeStoreAt(probes, candidate, path) {
    const exists = await probes.isDirectory(path);
    if (!exists)
        return { label: candidate.label, path, exists: false };
    const files = candidate.listFiles === true ? await describeModelDirectory(probes, path) : undefined;
    return { label: candidate.label, path, exists: true, ...(files === undefined ? {} : { files }) };
}
/**
 * List the model files under a store, following one level of subdirectories.
 * @param probes - the filesystem probes.
 * @param root - the store directory.
 * @returns the files found, capped.
 */
async function describeModelDirectory(probes, root) {
    const files = [];
    const entries = (await probes.listDir(root)) ?? [];
    for (const entry of entries) {
        if (files.length >= MAX_LISTED_FILES)
            break;
        if (entry.isDirectory) {
            const nested = (await probes.listDir(join(root, entry.name))) ?? [];
            for (const file of nested) {
                if (files.length >= MAX_LISTED_FILES)
                    break;
                if (!file.isDirectory) {
                    files.push({ name: `${entry.name}/${file.name}`, ...(file.sizeBytes === undefined ? {} : { sizeBytes: file.sizeBytes }) });
                }
            }
        }
        else {
            files.push({ name: entry.name, ...(entry.sizeBytes === undefined ? {} : { sizeBytes: entry.sizeBytes }) });
        }
    }
    return files;
}
/**
 * Ask an engine what it holds, over its own API.
 * @param probes - the probes.
 * @param known - the known-engine entry, when the engine is one this page knows.
 * @param endpoint - the engine's endpoint.
 * @param installPath - the resolved installation directory.
 * @returns the engine's own model list plus how it was obtained.
 */
async function describeEngineModels(probes, known, endpoint, installPath) {
    if (known?.id === 'ollama' && endpoint !== undefined) {
        try {
            const body = await probes.fetchJson(`${endpoint}/api/tags`, PROBE_TIMEOUT_MS);
            const list = body.models;
            if (!Array.isArray(list) || list.some((entry) => entry === null || typeof entry !== 'object' || typeof entry.name !== 'string'))
                throw new Error('invalid Ollama model list');
            return {
                models: list.map((entry) => {
                    const model = entry;
                    const parts = [model.details?.parameter_size, model.details?.quantization_level]
                        .filter((part) => typeof part === 'string' && part.length > 0);
                    const size = typeof model.size === 'number' ? formatBytes(model.size) : undefined;
                    const detail = [...parts, ...(size === undefined ? [] : [size])].join(' · ');
                    return {
                        id: typeof model.name === 'string' ? model.name : 'unknown',
                        ...(detail.length === 0 ? {} : { detail }),
                    };
                }),
                modelsSource: 'Ollama /api/tags',
            };
        }
        catch (error) {
            const detail = `Ollama models not verified: ${error instanceof Error ? error.message : String(error)}`;
            return { models: [], modelsSource: detail, error: detail };
        }
    }
    if (known?.id === 'comfyui') {
        // ComfyUI reports no models of its own, and this page must not invent any:
        // its checkpoints, LoRAs and VAEs are workflow-internal detail. What the
        // engine *can* be asked to do is the workflow list, filled in by the scan.
        return {
            models: [],
            modelsSource: 'ComfyUI exposes workflows, not models — see the local resources list',
        };
    }
    if (installPath !== undefined && known !== undefined) {
        return { models: [], modelsSource: 'listed from the model store below' };
    }
    return { models: [], modelsSource: 'nothing to read: the engine is not installed here' };
}
/**
 * Offer the inventory over HTTP for the settings page.
 *
 * Injected on demand rather than declared, so a headless or SDK profile — which
 * has no web server at all — still gets every model tool and simply never serves
 * this page.
 *
 * @param ctx - the plugin context.
 * @param log - the plugin's logger.
 * @param options - the hub, the catalog's declared hosts and path, and the
 *   deployment's launch flag.
 */
export function registerInventoryRoute(ctx, log, options) {
    const respond = (res, inventory) => {
        const body = JSON.stringify(inventory, null, 2);
        res.writeHead(200, {
            'content-type': 'application/json; charset=utf-8',
            'content-length': Buffer.byteLength(body),
            'cache-control': 'no-store',
        });
        res.end(body);
    };
    /**
     * The page as it looks when the hub could not be built.
     *
     * Everything the page normally reads from the hub is empty, and the reason is
     * in `error`, because "the hub is not loaded" is itself the answer a user
     * opening this screen needs — not a 404 and not a blank list that implies the
     * machine has no models.
     *
     * @returns the degraded inventory.
     */
    const degraded = () => ({
        generatedAt: new Date().toISOString(),
        resources: [],
        scan: {
            sources: [],
            warnings: options.catalogError === undefined ? [] : [options.catalogError],
            registered: 0,
            complete: false,
            engines: [],
            phases: [],
            progress: options.catalogError ?? 'the model hub is not loaded',
            managedEngines: false,
        },
        catalogPath: options.catalogPath,
        ...(options.artifactRoot.length === 0 ? {} : { artifactRoot: options.artifactRoot }),
        allowProcessLaunch: options.allowProcessLaunch,
        machine: { ramGb: 0, vramGb: 0, hasGpu: false, notes: 'not probed — the model hub is not loaded' },
        engines: [],
        models: [],
        capabilities: [],
        unavailable: [],
        error: options.catalogError ?? 'the model hub is not loaded',
    });
    const handler = async (req, res) => {
        if (req.method !== 'GET') {
            res.writeHead(405, { 'content-type': 'application/json; charset=utf-8', allow: 'GET' });
            res.end(JSON.stringify({ error: 'method not allowed' }));
            return;
        }
        if (options.hub === undefined) {
            respond(res, degraded());
            return;
        }
        try {
            // `scan=1` is the explicit Scan action and the only request that may start
            // an engine. Everything else — the page opening, a reload, a retry — reads
            // the machine as it finds it, because looking is not a decision to launch a
            // process and must never be treated as one.
            const explicitScan = scanRequested(req.url);
            const inventory = await buildInventory(options.hub, {
                hosts: options.hosts ?? [],
                catalogPath: options.catalogPath,
                artifactRoot: options.artifactRoot,
                allowProcessLaunch: options.allowProcessLaunch,
                manageEngines: explicitScan,
                ...(options.engineStartupTimeoutMs === undefined ? {} : { engineStartupTimeoutMs: options.engineStartupTimeoutMs }),
                ...(options.probes === undefined ? {} : { probes: options.probes }),
                ...(options.workflowDir === undefined ? {} : { workflowDir: options.workflowDir }),
                ...(options.scanTimeoutMs === undefined ? {} : { scanTimeoutMs: options.scanTimeoutMs }),
                ...(options.workflowCatalogPath === undefined ? {} : { workflowCatalogPath: options.workflowCatalogPath }),
                // Every read re-checks availability; only the explicit Scan may launch.
                probe: true,
                logger: log,
                log: (message) => log.debug(message),
            });
            respond(res, inventory);
        }
        catch (error) {
            log.warn(`the engine inventory route failed: ${error instanceof Error ? error.message : String(error)}`);
            res.writeHead(500, { 'content-type': 'application/json; charset=utf-8' });
            res.end(JSON.stringify({ error: 'inventory failed' }));
        }
    };
    installRoute(ctx, INVENTORY_ROUTE, (req, res) => handler(req, res), 'engine inventory', log);
}
/**
 * Whether a request is the explicit Scan action.
 *
 * `?scan=1` is the contract the settings page already used to mean "the user
 * pressed Scan"; this is the first time the route acts on it, so the reading is
 * deliberately generous about the spelling and strict about the meaning.
 *
 * @param url - the request's URL, path and query included.
 * @returns whether this request asked for a scan.
 */
function scanRequested(url) {
    if (url === undefined)
        return false;
    const query = url.slice(url.indexOf('?') + 1);
    if (url.indexOf('?') < 0)
        return false;
    for (const pair of query.split('&')) {
        const [key, value] = pair.split('=');
        if (key !== 'scan')
            continue;
        // `?scan`, `?scan=1` and `?scan=true` all mean yes; `?scan=0` means no.
        return value === undefined || value === '' || value === '1' || value === 'true';
    }
    return false;
}
