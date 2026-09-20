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
import { stat, readdir } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
/** The route the settings page reads. Absolute, no trailing slash. */
export const INVENTORY_ROUTE = '/dsh-ai-model-hub/inventory';
/** Longest a live endpoint probe may take. */
const PROBE_TIMEOUT_MS = 2500;
/** Longest a single directory listing may be, so a huge model store cannot stall the page. */
const MAX_LISTED_FILES = 40;
/** The default probes: a timed fetch and the real filesystem. */
export const DEFAULT_PROBES = {
    fetchJson: async (url, timeoutMs) => {
        const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
        if (!response.ok)
            throw new Error(`HTTP ${response.status}`);
        return await response.json();
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
        storeCandidates: [{ label: 'ComfyUI models', path: '%INSTALL%/models', listFiles: true }],
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
export async function buildInventory(hub, options) {
    const probes = options.probes ?? DEFAULT_PROBES;
    const declaredHosts = options.hosts;
    const views = hub.listModels({ includeDisabled: true });
    const engines = [];
    const reported = new Set();
    for (const host of declaredHosts) {
        const view = views.find((entry) => entry.model.hostId === host.id);
        const lifecycle = view?.model.lifecycle;
        const start = lifecycle?.start;
        const known = KNOWN_ENGINES.find((entry) => entry.id === host.id || entry.id === host.runtime.engine);
        const endpoint = host.runtime.endpoint;
        const install = await resolveInstall(probes, [
            ...(start?.cwd === undefined ? [] : [expandPath(start.cwd)]),
            ...(start?.command === undefined ? [] : [expandPath(start.command)]),
            ...(known?.installCandidates ?? []),
        ]);
        engines.push({
            id: host.id,
            name: host.name,
            adapter: host.adapter,
            configured: true,
            ...(endpoint === undefined ? {} : { endpoint }),
            ...(await describeEndpoint(probes, endpoint, options.probe === true)),
            startable: lifecycle?.startable === true,
            ...(start === undefined
                ? {}
                : {
                    launch: {
                        command: expandPath(start.command),
                        args: start.args ?? [],
                        ...(start.cwd === undefined ? {} : { cwd: expandPath(start.cwd) }),
                    },
                }),
            installSource: install.source,
            ...(install.path === undefined ? {} : { installPath: install.path }),
            ...(install.path === undefined && install.checked.length > 0 ? { checkedPaths: install.checked } : {}),
            storePaths: await describeStores(probes, known, install.path),
            ...(await describeEngineModels(probes, known, endpoint, install.path)),
            servedModelIds: views.filter((entry) => entry.model.hostId === host.id).map((entry) => entry.model.id),
        });
        reported.add(host.id);
        if (known !== undefined)
            reported.add(known.id);
    }
    for (const known of KNOWN_ENGINES) {
        if (reported.has(known.id))
            continue;
        const install = await resolveInstall(probes, known.installCandidates);
        engines.push({
            id: known.id,
            name: known.name,
            adapter: known.adapter,
            configured: false,
            endpoint: known.endpoint,
            ...(await describeEndpoint(probes, known.endpoint, options.probe === true)),
            startable: false,
            installSource: install.source,
            ...(install.path === undefined ? { checkedPaths: install.checked } : { installPath: install.path }),
            storePaths: await describeStores(probes, known, install.path),
            ...(await describeEngineModels(probes, known, known.endpoint, install.path)),
            servedModelIds: [],
        });
    }
    const capabilities = hub.listCapabilities().map((entry) => ({
        capability: String(entry.capability),
        models: entry.modelIds.length,
    }));
    return {
        generatedAt: new Date().toISOString(),
        catalogPath: options.catalogPath,
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
        models: views.map((view) => ({
            id: view.model.id,
            name: view.model.name,
            type: String(view.model.type),
            capabilities: view.model.capabilities.map(String),
            engine: view.model.runtime.engine,
            ...(view.model.hostId === undefined ? {} : { hostId: view.model.hostId }),
            availability: String(view.status.availability),
            lifecycle: String(view.status.lifecycle),
            startable: view.model.lifecycle.startable,
            ...(view.status.pid === undefined ? {} : { pid: view.status.pid }),
            ...(view.status.health === undefined ? {} : { healthy: view.status.health.healthy }),
            ...(view.status.reason === undefined && view.status.health?.detail === undefined
                ? {}
                : { detail: view.status.reason ?? view.status.health?.detail ?? '' }),
        })),
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
async function resolveInstall(probes, candidates) {
    const checked = [];
    for (const candidate of candidates) {
        const path = expandPath(candidate).replace(/[/\\]+$/, '');
        if (path.length === 0 || checked.includes(path))
            continue;
        checked.push(path);
        if (await probes.isDirectory(path)) {
            return { path, source: 'found on this machine', checked };
        }
    }
    return { source: 'not found on this machine', checked };
}
/**
 * Probe one endpoint.
 * @param probes - the probes.
 * @param endpoint - the endpoint, when the engine has one.
 * @param live - whether to probe now; when false the report says it was not probed.
 * @returns the running flag and a human-readable detail.
 */
async function describeEndpoint(probes, endpoint, live) {
    if (endpoint === undefined)
        return { running: true, statusDetail: 'no endpoint: runs in process' };
    if (!live)
        return { running: false, statusDetail: 'not probed yet — press Check' };
    try {
        await probes.fetchJson(endpoint, PROBE_TIMEOUT_MS);
        return { running: true, statusDetail: 'answering' };
    }
    catch (error) {
        return { running: false, statusDetail: error instanceof Error ? error.message : String(error) };
    }
}
/**
 * Describe an engine's model stores.
 * @param probes - the filesystem probes.
 * @param known - the known-engine entry, when the engine is one this page knows.
 * @param installPath - the resolved installation directory.
 * @returns one row per store, with its contents when the store lists files.
 */
async function describeStores(probes, known, installPath) {
    if (known === undefined)
        return [];
    const stores = [];
    for (const candidate of known.storeCandidates) {
        if (candidate.path.includes('%INSTALL%')) {
            if (installPath === undefined)
                continue;
        }
        const path = expandPath(candidate.path.replace('%INSTALL%', installPath ?? ''));
        const exists = await probes.isDirectory(path);
        if (!exists) {
            stores.push({ label: candidate.label, path, exists: false });
            continue;
        }
        const files = candidate.listFiles === true ? await describeModelDirectory(probes, path) : undefined;
        stores.push({ label: candidate.label, path, exists: true, ...(files === undefined ? {} : { files }) });
    }
    return stores;
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
            if (!Array.isArray(list))
                return { models: [], modelsSource: 'Ollama answered without a model list' };
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
            return { models: [], modelsSource: `Ollama did not answer: ${error instanceof Error ? error.message : String(error)}` };
        }
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
    ctx.inject(['webServer'], (scope) => {
        const server = scope.webServer;
        if (server === undefined)
            return;
        const handler = async (req, res) => {
            if (req.method !== 'GET') {
                res.writeHead(405, { 'content-type': 'application/json; charset=utf-8', allow: 'GET' });
                res.end(JSON.stringify({ error: 'method not allowed' }));
                return;
            }
            try {
                const probe = new URL(req.url ?? '/', 'http://127.0.0.1').searchParams.get('probe') === '1';
                const inventory = await buildInventory(options.hub, {
                    hosts: options.hosts,
                    catalogPath: options.catalogPath,
                    artifactRoot: options.artifactRoot,
                    allowProcessLaunch: options.allowProcessLaunch,
                    ...(options.probes === undefined ? {} : { probes: options.probes }),
                    probe,
                });
                const body = JSON.stringify(inventory, null, 2);
                res.writeHead(200, {
                    'content-type': 'application/json; charset=utf-8',
                    'content-length': Buffer.byteLength(body),
                    'cache-control': 'no-store',
                });
                res.end(body);
            }
            catch (error) {
                log.warn(`the engine inventory route failed: ${error instanceof Error ? error.message : String(error)}`);
                res.writeHead(500, { 'content-type': 'application/json; charset=utf-8' });
                res.end(JSON.stringify({ error: 'inventory failed' }));
            }
        };
        server.register({ kind: 'exact', path: INVENTORY_ROUTE, handler });
        log.debug(`serving the engine inventory on ${INVENTORY_ROUTE}`);
    });
}
