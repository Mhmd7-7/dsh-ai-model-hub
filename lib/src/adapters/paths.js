/**
 * One rule for the file paths a catalog writes.
 *
 * Two adapters accept a path that points at a file shipped beside the catalog:
 * `comfyui` takes `adapterConfig.workflowPath` (a graph template) and `three_d`
 * takes `adapterConfig.stepsPath` (a call-protocol declaration). Both mean the
 * same thing, so both must resolve the same way, and that way is stated exactly
 * once — here.
 *
 * The rule
 * --------
 * **A relative path in a catalog is relative to the catalog that wrote it.**
 * Nothing else is knowable: a catalog is a document, it can live anywhere, and
 * the host process that reads it was launched from wherever its launcher stood.
 * Resolving against `process.cwd()` makes the same catalog behave differently
 * depending on who started the host — which is how a deployment ends up asking
 * for a file that is plainly there and being told `ENOENT`.
 *
 * The working directory survives as a base for one case only: a hub built from an
 * in-memory document (`ModelHub.fromConfig`), where no catalog file exists to be
 * relative to. That base is reported as `working-directory` rather than silently
 * passing for a catalog, so a diagnostic can say which rule applied.
 *
 * Why this module exists
 * ----------------------
 * The two adapters disagreed once. `comfyui` resolved against the catalog
 * directory while the shipped data wrote `config/workflows/...`, a spelling that
 * only makes sense from the package root: `config/` + `config/workflows/…`
 * composed into `<pkg>/config/config/workflows/…`, a path that has never
 * existed, and the failure surfaced as an unreadable template on the first
 * invocation. The convention is data that code has to agree with, so the
 * agreement lives in one function that both adapters call.
 *
 * @module dsh-ai-model-hub/adapters/paths
 */
import { isAbsolute, resolve } from 'node:path';
/**
 * Resolve a path a catalog wrote.
 *
 * @param path - the configured path, absolute or relative.
 * @param catalogDir - the directory of the catalog file this model was declared
 *   in, as `AdapterInvocation.catalogDir` carries it.
 * @returns the absolute path, the base it came from, and which base that was.
 */
export function resolveAdapterPath(path, catalogDir) {
    if (isAbsolute(path))
        return { absolute: path, origin: 'absolute' };
    const base = catalogDir ?? process.cwd();
    return {
        absolute: resolve(base, path),
        base,
        origin: catalogDir === undefined ? 'working-directory' : 'catalog',
    };
}
/**
 * Describe a resolved path for an error message.
 *
 * Shared so an unreadable template reads the same whichever adapter hit it, and
 * names everything needed to diagnose a mismatch without a debugger: the absolute
 * path that was tried, the base it came from, and which rule produced that base.
 *
 * @param resolved - the resolution to describe.
 * @returns a phrase to embed in a message, e.g.
 *   `"C:\pkg\config\config\workflows\x.json" (resolved against the catalog directory "C:\pkg\config")`.
 */
export function describeAdapterPath(resolved) {
    switch (resolved.origin) {
        case 'absolute':
            return `"${resolved.absolute}" (an absolute path, so no catalog directory was consulted)`;
        case 'catalog':
            return `"${resolved.absolute}" (resolved against the catalog directory "${resolved.base}", which is what a relative path in a catalog is relative to)`;
        case 'working-directory':
            return `"${resolved.absolute}" (resolved against the working directory "${resolved.base}", because this hub was built from an in-memory catalog and has no catalog directory)`;
    }
}
