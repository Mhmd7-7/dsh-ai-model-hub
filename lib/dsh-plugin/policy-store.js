/**
 * The capability policy's durable home.
 *
 * The policy is the operator's data — the order they arranged and the options
 * they switched on — so it lives in its own versioned document rather than being
 * inferred from the catalog, from a descriptor's `enabled` flag, or from what
 * happens to be visible in a picker. Everything about this module follows from
 * that: it validates on read, refuses a document it does not understand rather
 * than guessing, writes atomically so a crash cannot leave a half-written policy
 * in force, and never silently grants anything during a migration.
 *
 * The file's location is deliberately outside both the catalog and the profile's
 * generated config:
 *
 * - Not in the catalog: the catalog is where *engines* are declared, and rewriting
 *   it to record a UI preference would mix two documents with different owners and
 *   different lifetimes.
 * - Not in `cordis.yml` or the profile patch: those are generated from the patch
 *   layers and `package.json`, so a hand-written preference there is overwritten
 *   on the next reconcile.
 *
 * Instead it sits under the harness home, in the per-plugin directory the
 * subscription plugin already uses for its own durable settings, which makes the
 * DSH home the one place a third-party plugin can own state.
 *
 * @module dsh-ai-model-hub/dsh-plugin/policy-store
 */
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { parseCapabilityPolicy, serializeCapabilityPolicy } from "../src/index.js";
/** The policy file's path when the deployment names none. */
export function defaultPolicyPath() {
    return join(homedir(), '.dsh', 'plugins', 'dsh-ai-model-hub', 'capability-policy.json');
}
/**
 * Read and validate the saved policy.
 *
 * A missing file is not an error: it is the normal first run, and the caller's
 * defaults are the right answer. A malformed file *is* reported, loudly, and its
 * document is discarded — because enforcing half of a policy the operator wrote
 * is worse than enforcing none of it and saying so.
 *
 * @param path - the policy file to read.
 * @param log - the plugin's logger.
 * @returns the load result.
 */
export async function loadPolicyDocument(path, log) {
    let raw;
    try {
        raw = await readFile(path, 'utf8');
    }
    catch (error) {
        const code = error.code;
        if (code === 'ENOENT')
            return { document: undefined, path, found: false, issues: [] };
        const message = `could not read the capability policy at ${path}: ${error instanceof Error ? error.message : String(error)}`;
        log.warn(message);
        return { document: undefined, path, found: false, issues: [message] };
    }
    let parsed;
    try {
        parsed = JSON.parse(raw);
    }
    catch (error) {
        const message = `the capability policy at ${path} is not valid JSON (${error instanceof Error ? error.message : String(error)}); it was ignored and the defaults are in force`;
        log.warn(message);
        return { document: undefined, path, found: true, issues: [message] };
    }
    const result = parseCapabilityPolicy(parsed);
    if (!result.ok) {
        const message = `the capability policy at ${path} was rejected and the defaults are in force:\n${formatIssues(result.issues)}`;
        log.warn(message);
        return { document: undefined, path, found: true, issues: [message] };
    }
    if (result.issues.length > 0) {
        log.warn(`the capability policy at ${path} loaded with ${result.issues.length} ignored entr(ies):\n${formatIssues(result.issues)}`);
    }
    return {
        document: result.document,
        path,
        found: true,
        issues: result.issues.map((issue) => `${issue.path} ${issue.message}`),
    };
}
/**
 * Write a policy document atomically.
 *
 * Temp-file-then-rename, in the same directory, so a crash or a concurrent save
 * can never leave a truncated document in force: the rename is the only step that
 * changes what the next load reads, and it is atomic on every platform this ships
 * to.
 *
 * @param path - where to write.
 * @param document - the document to persist.
 * @param log - the plugin's logger.
 * @returns the issues found; empty means it was written.
 */
export async function savePolicyDocument(path, document, log) {
    const serialized = serializeCapabilityPolicy(document);
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
        const message = `could not save the capability policy to ${path}: ${error instanceof Error ? error.message : String(error)}. The change is in force for this session only.`;
        log.warn(message);
        return [message];
    }
}
/**
 * Render validation issues for a log line or an API response.
 * @param issues - the issues.
 * @returns one line per issue.
 */
export function formatIssues(issues) {
    return issues.map((issue) => `  - ${issue.path} ${issue.message}`).join('\n');
}
