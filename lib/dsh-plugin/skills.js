/**
 * The `dsh-ai-model-hub` skill, offered by the plugin itself.
 *
 * A plugin that ships a skill does not have to copy it into a user or project
 * skill root, and no profile file has to be edited: registering a provider on
 * `ctx.skills` makes the skill appear in the session catalog wherever this plugin
 * loads, and its body stays the file in this checkout — so a `git pull` changes
 * what the agent reads on the next load, with no reinstall step and no stale copy
 * in `~/.dsh/skills`.
 *
 * This mirrors the pattern DSH itself uses for a bundled skill
 * (`@deepseek-ai/dsh-skill-badge`): a provider that names one candidate and reads
 * its body from a file shipped beside the code.
 *
 * Two deliberate tradeoffs:
 *
 * - **The skill file is also a valid filesystem skill.** `skills/dsh-ai-model-hub/SKILL.md`
 *   carries real frontmatter, so dropping or linking that directory into
 *   `.dsh/skills` works too, and the frontmatter is the single source of truth for
 *   the name and description on both paths.
 * - **The catalog is not watched.** The body is re-read on every `get`, so editing
 *   the instructions reaches the next load immediately; editing the frontmatter's
 *   *description* only reaches a catalog collection the registry has not already
 *   cached. That is rare enough not to justify a file watcher in the plugin layer.
 *
 * Errors are contained rather than thrown: a broken skill file logs one warning
 * and offers nothing, because a bad skill must not cost the agent the model tools
 * beside it.
 *
 * @module dsh-ai-model-hub/dsh-plugin/skills
 */
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
/** Unique provider name registered on `ctx.skills`. */
export const PROVIDER_NAME = 'dsh-ai-model-hub';
/**
 * Precedence rank for a packaged or bundled skill source.
 *
 * Mirrors `BUNDLED_SKILL_RANK` from `@deepseek-ai/dsh-skill`. Importing the
 * constant would add a runtime dependency on a DSH package this plugin does not
 * otherwise need, which is exactly the coupling the plugin layer avoids; the value
 * is part of the registry's published contract, not an implementation detail.
 */
export const BUNDLED_SKILL_RANK = 600;
/**
 * The registry's public kebab-case skill-name grammar, mirroring `SKILL_NAME` in
 * `@deepseek-ai/dsh-skill`. A candidate that fails it is rejected by the registry
 * with a thrown error, which would degrade the whole catalog observation â€” so it
 * is checked here, where the failure can be a contained warning instead.
 */
const SKILL_NAME_GRAMMAR = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
/**
 * Walk up from a module's own directory to the package root.
 *
 * A fixed number of `..` segments cannot work here: the sources run from
 * `dsh-plugin/` in a checkout and from `lib/dsh-plugin/` once compiled, so
 * `new URL('../skills', import.meta.url)` is right in exactly one of the two
 * layouts — and silently wrong in the other, which costs the installed copy its
 * skill. The nearest `package.json` is right in both, and in an installed copy
 * sitting in `node_modules` as well.
 *
 * @param start - the directory this module was loaded from.
 * @returns the package root, or `start` if no manifest is found above it.
 */
function findPackageRoot(start) {
    let directory = start;
    for (;;) {
        if (existsSync(join(directory, 'package.json')))
            return directory;
        const parent = dirname(directory);
        if (parent === directory)
            return start;
        directory = parent;
    }
}
/** The directory this module was loaded from: `dsh-plugin/` or `lib/dsh-plugin/`. */
const MODULE_DIRECTORY = dirname(fileURLToPath(import.meta.url));
/** The package root — the directory that ships `skills/`, `config/` and `docs/`. */
const PACKAGE_ROOT = findPackageRoot(MODULE_DIRECTORY);
/** The shipped skill file: `skills/dsh-ai-model-hub/SKILL.md` in this package. */
export const SKILL_FILE = join(PACKAGE_ROOT, 'skills', 'dsh-ai-model-hub', 'SKILL.md');
/**
 * The package root, used as the loaded body's resource base so the paths the
 * instructions cite (`docs/adding-a-model.md`, `config/models.json`) resolve.
 * Normalized through `resolve` so it never carries a trailing separator.
 */
const REPOSITORY_ROOT = resolve(PACKAGE_ROOT);
/**
 * Offer the bundled skill in whatever composition mounts this plugin.
 *
 * Injected through `ctx.inject` rather than a declared `inject`, so a deployment
 * without a skill registry keeps its model tools: the callback simply never runs
 * there, instead of the plugin's fiber parking forever on a service that never
 * arrives. Inside the callback the service is present by construction, and cordis
 * re-runs the callback if it is ever replaced.
 *
 * @param ctx - the plugin context.
 * @param log - the plugin's logger.
 * @param skillFile - the skill file to read; overridable so the reader can be
 * pointed at another file by a test or a redeployment.
 */
export function registerModelHubSkill(ctx, log, skillFile = SKILL_FILE) {
    ctx.inject(['skills'], (scope) => {
        // Resolved by cordis's inject-aware context proxy, so it is present here; the
        // cast supplies the declaration DSH publishes only in `@deepseek-ai/dsh-skill`,
        // which this plugin deliberately does not depend on.
        const skills = scope.skills;
        if (skills === undefined)
            return;
        skills.registerProvider(() => createModelHubSkillProvider(log, skillFile));
        log.debug(`offering the bundled skill ${PROVIDER_NAME} from ${skillFile}`);
    });
}
/**
 * Build the provider that offers the shipped skill.
 *
 * @param log - the plugin's logger, used once for a contained read failure.
 * @param skillFile - the skill file to read.
 * @returns the provider to register on `ctx.skills`.
 */
export function createModelHubSkillProvider(log, skillFile = SKILL_FILE) {
    let warningLogged = false;
    const read = async () => {
        try {
            return parseSkillFile(await readFile(skillFile, 'utf8'), skillFile);
        }
        catch (error) {
            if (!warningLogged) {
                warningLogged = true;
                log.warn(`the bundled skill was not offered: ${describeError(error)}`);
            }
            return undefined;
        }
    };
    return {
        name: PROVIDER_NAME,
        async list(options) {
            if (options.signal?.aborted === true)
                return [];
            const skill = await read();
            return skill === undefined ? [] : [candidateFor(skill, skillFile)];
        },
        async get(candidate, options) {
            if (options.signal?.aborted === true)
                return undefined;
            if (candidate.locator !== skillFile)
                return undefined;
            const skill = await read();
            return skill === undefined ? undefined : definitionFor(skill, skillFile);
        },
    };
}
/**
 * Build the catalog candidate for a loaded skill file.
 * @param skill - the parsed skill.
 * @param skillFile - the file it came from, used as the opaque locator.
 * @returns the candidate the registry indexes.
 */
function candidateFor(skill, skillFile) {
    return {
        name: skill.name,
        description: skill.description,
        ...(skill.whenToUse === undefined ? {} : { whenToUse: skill.whenToUse }),
        invocation: { modelInvocable: true, userInvocable: true },
        source: 'bundled',
        provider: PROVIDER_NAME,
        resourceBase: { kind: 'directory', path: REPOSITORY_ROOT },
        rank: BUNDLED_SKILL_RANK,
        locator: skillFile,
        path: skillFile,
    };
}
/**
 * Build the loadable definition for a loaded skill file.
 *
 * Spelled out rather than spread from the candidate: a definition is not a
 * catalog row, and carrying `rank` and `locator` into it would suggest a
 * precedence and an identity the loaded skill does not have.
 *
 * @param skill - the parsed skill.
 * @param skillFile - the file it came from.
 * @returns the definition the model receives.
 */
function definitionFor(skill, skillFile) {
    return {
        name: skill.name,
        description: skill.description,
        ...(skill.whenToUse === undefined ? {} : { whenToUse: skill.whenToUse }),
        invocation: { modelInvocable: true, userInvocable: true },
        source: 'bundled',
        provider: PROVIDER_NAME,
        resourceBase: { kind: 'directory', path: REPOSITORY_ROOT },
        content: skill.content,
        path: skillFile,
    };
}
/**
 * Parse a skill file into its frontmatter fields and its body.
 *
 * The supported dialect is exactly what a skill file needs and nothing more: a
 * leading `---` block of **flat `key: value` scalars**, then the Markdown body.
 * Anything the reader cannot represent faithfully â€” a missing block, a fold or
 * block scalar, an indented continuation line â€” is an error rather than a silent
 * mis-parse, because the alternative is an agent following a body that begins with
 * a stray `---`.
 *
 * @param source - the whole file.
 * @param file - the file's path, for diagnostics.
 * @returns the parsed skill.
 * @throws when the file is not a usable skill.
 */
function parseSkillFile(source, file) {
    const match = /^---\r?\n([\s\S]*?)\r?\n---[ \t]*\r?\n?/.exec(source);
    const block = match?.[1];
    if (match === null || block === undefined) {
        throw new Error(`${file}: expected a frontmatter block starting with a --- line`);
    }
    const fields = new Map();
    for (const rawLine of block.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (line.length === 0 || line.startsWith('#'))
            continue;
        const separator = line.indexOf(':');
        if (separator <= 0) {
            throw new Error(`${file}: frontmatter supports only flat "key: value" lines, got "${line}"`);
        }
        const key = line.slice(0, separator).trim();
        const value = unquote(line.slice(separator + 1).trim());
        if (value === '>' || value === '|' || value === '>-' || value === '|-' || value === '>+' || value === '|+') {
            throw new Error(`${file}: frontmatter "${key}" uses a multi-line scalar, which is not supported`);
        }
        if (!fields.has(key))
            fields.set(key, value);
    }
    const name = fields.get('name') ?? '';
    if (!SKILL_NAME_GRAMMAR.test(name)) {
        throw new Error(`${file}: frontmatter "name" must be kebab-case, got "${name}"`);
    }
    const description = fields.get('description') ?? '';
    if (description.length === 0) {
        throw new Error(`${file}: frontmatter "description" is required`);
    }
    const content = source.slice(match[0].length).trim();
    if (content.length === 0) {
        throw new Error(`${file}: the body after the frontmatter is empty`);
    }
    const whenToUse = fields.get('whenToUse') ?? '';
    return {
        name,
        description,
        ...(whenToUse.length === 0 ? {} : { whenToUse }),
        content,
    };
}
/**
 * Strip matching single or double quotes from a frontmatter scalar.
 * @param value - the raw scalar text.
 * @returns the unquoted value.
 */
function unquote(value) {
    const first = value.at(0);
    const last = value.at(-1);
    if (value.length >= 2 && first === last && (first === '"' || first === "'")) {
        return value.slice(1, -1);
    }
    return value;
}
/**
 * Render an unknown thrown value for a log line.
 * @param error - whatever was thrown.
 * @returns a one-line description.
 */
function describeError(error) {
    return error instanceof Error ? error.message : String(error);
}
