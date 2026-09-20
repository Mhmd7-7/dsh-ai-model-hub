/**
 * Configuration loading.
 *
 * A deployment's models are data on disk, not code, so this module is the
 * boundary where the hub meets the filesystem. It resolves which file to read,
 * reads it, and hands the raw parsed value to the catalog's validator — it does
 * *not* validate, because there is exactly one validator and it is not here.
 *
 * @module dsh-ai-model-hub/config/load
 */
import { existsSync, readFileSync } from 'node:fs';
import { isAbsolute, join, resolve } from 'node:path';
import { parseModelCatalogConfig } from "../catalog/descriptor.js";
import { ModelHubError } from "../errors.js";
import { ModelHub as Hub } from "../hub.js";
/** The filenames discovery accepts, in priority order. */
export const DEFAULT_CONFIG_FILENAMES = [
    'models.json',
    'model-catalog.json',
    'dsh-ai-model-hub.json',
];
/**
 * Walk upward from a directory looking for a catalog file.
 *
 * Walking upward rather than only checking the working directory means a hub can
 * be run from a nested subdirectory — where an agent's shell happens to be — and
 * still find the deployment's configuration. The search stops at the filesystem
 * root and never escapes it.
 *
 * Ordering matters and is deliberate: at each level, every accepted filename is
 * tried **directly** before any `config/` subdirectory is considered. Doing it the
 * other way round lets a deeply nested `config/models.json` shadow a catalog the
 * user put right where they launched the harness, which is the opposite of what
 * "walk up to find the nearest catalog" should mean.
 *
 * @param options - discovery controls.
 * @returns the first matching absolute path, or `undefined`.
 */
export function findConfigDirectory(options = {}) {
    const filenames = options.filenames ?? DEFAULT_CONFIG_FILENAMES;
    const start = resolve(options.startDir ?? process.cwd());
    // An explicit file always wins, and is never searched for.
    if (options.configPath !== undefined) {
        const explicit = isAbsolute(options.configPath) ? options.configPath : resolve(start, options.configPath);
        return existsSync(explicit) ? explicit : undefined;
    }
    const searchParents = options.searchParents ?? true;
    let current = start;
    let levels = 0;
    for (;;) {
        // Nearest level, direct filenames first.
        for (const filename of filenames) {
            const candidate = join(current, filename);
            if (existsSync(candidate))
                return candidate;
        }
        // Then the conventional config/ subdirectory at this level.
        for (const filename of filenames) {
            const candidate = join(current, 'config', filename);
            if (existsSync(candidate))
                return candidate;
        }
        if (!searchParents)
            return undefined;
        const parent = resolve(current, '..');
        if (parent === current)
            return undefined;
        // A safety valve: never walk more than 20 levels.
        levels += 1;
        if (levels > 20)
            return undefined;
        current = parent;
    }
}
/**
 * Read and validate a catalog file.
 *
 * @param options - discovery and file controls.
 * @returns the loaded and validated catalog.
 * @throws ModelHubError with `CONFIG_ERROR` when no file is found or it cannot be
 *   read, and `INVALID_DESCRIPTOR` when it fails validation.
 */
export function loadCatalogConfig(options = {}) {
    const path = findConfigDirectory(options);
    if (path === undefined) {
        const searched = options.configPath ?? `one of ${(options.filenames ?? DEFAULT_CONFIG_FILENAMES).join(', ')}`;
        throw new ModelHubError('CONFIG_ERROR', `no model catalog found (searched for ${searched} from ${resolve(options.startDir ?? process.cwd())}). ` +
            'Create config/models.json, or call ModelHub with an explicit `config` object.', { configPath: options.configPath, startDir: options.startDir });
    }
    return readCatalogAt(path);
}
/**
 * Read and validate the catalog at an already-resolved path.
 *
 * Split out of {@link loadCatalogConfig} so the anchor search below can validate
 * a path it has already found without re-running discovery.
 *
 * @param path - an absolute path known to exist.
 * @returns the loaded and validated catalog.
 * @throws ModelHubError with `CONFIG_ERROR` when the file cannot be read, and
 *   `INVALID_DESCRIPTOR` when it fails validation.
 */
function readCatalogAt(path) {
    let text;
    try {
        text = readFileSync(path, 'utf8');
    }
    catch (error) {
        throw new ModelHubError('CONFIG_ERROR', `could not read model catalog at ${path}: ${String(error)}`, {
            path,
        });
    }
    // Strip a UTF-8 byte-order mark. Notepad, several editors, and PowerShell's
    // `Set-Content -Encoding utf8` all emit one, and `JSON.parse` rejects it with a
    // message about an unexpected character that gives no hint the BOM is the
    // cause. A hand-edited catalog is exactly the kind of file this happens to, so
    // tolerating it is worth more than the strictness.
    if (text.charCodeAt(0) === 0xfeff)
        text = text.slice(1);
    let raw;
    try {
        raw = JSON.parse(text);
    }
    catch (error) {
        throw new ModelHubError('CONFIG_ERROR', `model catalog at ${path} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`, { path });
    }
    const parsed = parseModelCatalogConfig(raw, path);
    if (!parsed.ok) {
        throw new ModelHubError('INVALID_DESCRIPTOR', parsed.message, {
            path,
            issues: parsed.issues.map((issue) => ({ path: issue.path, message: issue.message })),
        });
    }
    return { path, raw, config: parsed.config };
}
/**
 * Resolve a catalog by trying several anchor directories in order.
 *
 * This exists because {@link loadCatalogConfig}'s default anchor — the process
 * working directory — is the wrong one for a plugin loaded by a long-running
 * host. The host's working directory is fixed when it is launched and has no
 * relationship to where the operator keeps a model catalog, so a hub that
 * anchors only there silently finds nothing and appears to "not work" while
 * every component is healthy. Callers that *do* know a better anchor set (a
 * plugin knows its own installation directory, for instance) should pass it
 * here rather than relying on the default.
 *
 * The first anchor that yields a catalog wins; failure is reported with every
 * anchor tried, so a missing catalog is a one-step diagnosis instead of a hunt.
 *
 * @param options - the explicit path (if any) and the ordered anchors.
 * @returns the loaded catalog plus the search trail.
 * @throws ModelHubError with `CONFIG_ERROR` when no anchor yields a catalog, and
 *   with the validation codes of {@link loadCatalogConfig} for a bad file.
 */
export function loadCatalogFromAnchors(options) {
    if (options.configPath !== undefined && options.configPath.length > 0) {
        const explicit = loadCatalogConfig({
            configPath: options.configPath,
            ...(options.filenames === undefined ? {} : { filenames: options.filenames }),
        });
        return { ...explicit, anchor: explicit.path, searched: [explicit.path] };
    }
    const searched = [];
    for (const anchor of options.anchors) {
        const start = resolve(anchor);
        searched.push(start);
        const found = findConfigDirectory({
            startDir: start,
            ...(options.filenames === undefined ? {} : { filenames: options.filenames }),
        });
        if (found !== undefined)
            return { ...readCatalogAt(found), anchor: start, searched };
    }
    throw new ModelHubError('CONFIG_ERROR', `no model catalog found. Searched upward from ${searched.length} anchor(s): ${searched.join(', ')}. ` +
        'Create models.json in one of those directories, or set the catalog path explicitly.', { anchors: searched });
}
/**
 * Build a hub from the configuration on disk.
 *
 * @param options - hub construction options plus discovery controls.
 * @returns the hub and the path it was built from.
 * @throws ModelHubError with `CONFIG_ERROR` or `INVALID_DESCRIPTOR`.
 */
export function loadHubFromDisk(options = {}) {
    const { load, ...hubOptions } = options;
    const loaded = loadCatalogConfig(load ?? {});
    const hub = new Hub({ ...hubOptions, config: loaded.config });
    return { hub, configPath: loaded.path, config: loaded.config };
}
