/**
 * End-to-end smoke test for an installed profile.
 *
 * Runs the plugin exactly as DSH will — resolved from the profile, imported, and
 * applied to a real cordis context — and reports what it registered. This is the
 * check that proves the whole chain, not just that the module can be imported:
 *
 *   profile resolution → module import → apply() → catalog discovery →
 *   hub construction → tool registration
 *
 * The package name is `dsh-ai-model-hub`: the repository IS the plugin package, so
 * its root `main`/`exports["."]` (`lib/dsh-plugin/index.js`) is the entry point
 * DSH's loader imports. When the profile has no copy to resolve — nothing is
 * installed yet — this falls back to that same file in this repository, which is
 * the file a link install loads anyway.
 *
 * It is deliberately separate from `doctor.mjs` because it *does* work rather than
 * inspect: it registers ten tools, reads a model catalog, and offers the bundled
 * skill. Point it at a throwaway profile if you would rather not touch one.
 *
 * Usage:
 *
 *   node scripts/smoke.mjs --profile web
 *
 * Exits 0 when the plugin loaded and registered its tools, 1 otherwise.
 *
 * @module dsh-ai-model-hub/scripts/smoke
 */

import { createRequire } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

/**
 * Read a `--flag value` argument.
 * @param {string} flag - the flag name without dashes.
 * @param {string} fallback - the default.
 * @returns {string} the resolved value.
 */
function readArgument(flag, fallback) {
  const index = process.argv.indexOf(`--${flag}`);
  if (index >= 0 && process.argv[index + 1] !== undefined && !process.argv[index + 1].startsWith('--')) {
    return process.argv[index + 1];
  }
  return fallback;
}

/**
 * The plugin package name. The repository root IS the package, so this is both
 * the name the profile resolves and the name of the bundle layer it registers.
 */
const PLUGIN_PACKAGE = 'dsh-ai-model-hub';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');
const profile = readArgument('profile', 'web');
const dshHome = process.env['DSH_HOME'] ?? join(homedir(), '.dsh');
const profileDir = join(dshHome, 'profiles', profile);

console.log('dsh-ai-model-hub smoke test');
console.log(`  profile: ${profileDir}`);
console.log(`  cwd:     ${process.cwd()}`);

const profileRequire = createRequire(join(profileDir, 'package.json'));

/**
 * The relative entry path a package manifest declares.
 *
 * `exports["."]` is what the loader actually honours and `main` is the fallback,
 * so read them in that order rather than assuming one.
 *
 * @param {any} manifest - a parsed package.json.
 * @returns {string} the relative path of the plugin entry point.
 */
function entryFromManifest(manifest) {
  const root = manifest?.exports?.['.'];
  const exported = typeof root === 'string' ? root : root?.default;
  return exported ?? manifest?.main ?? 'lib/dsh-plugin/index.js';
}

// Resolve through the profile, exactly as the loader does. The fallback is this
// repository's own `main`/`exports["."]` — the same file the profile is given —
// so the script still works from a checkout that has not been installed, and
// either way it exercises the real entry point rather than a copy of it.
let entryPath;
let anchor;
let origin;
let fallbackReason;
try {
  const manifestPath = profileRequire.resolve(`${PLUGIN_PACKAGE}/package.json`);
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  entryPath = join(dirname(manifestPath), entryFromManifest(manifest));
  anchor = profileRequire;
  origin = `the '${profile}' profile`;
} catch (error) {
  const manifestPath = join(repoRoot, 'package.json');
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  entryPath = join(repoRoot, entryFromManifest(manifest));
  anchor = createRequire(manifestPath);
  origin = 'this repository';
  fallbackReason = error.message;
}

if (!existsSync(entryPath)) {
  console.error(`FAILED: the plugin entry point does not exist: ${entryPath}`);
  console.error('A checkout needs its build output — run `npm run build` at the repository root.');
  process.exit(1);
}

console.log(`  plugin:  ${entryPath}`);
console.log(`  source:  ${origin}`);
if (fallbackReason !== undefined) {
  console.log(`           (the profile does not resolve '${PLUGIN_PACKAGE}': ${fallbackReason})`);
}
console.log('');

let plugin;
try {
  plugin = await import(pathToFileURL(entryPath).href);
} catch (error) {
  console.error(`FAILED: the plugin module could not be imported from ${entryPath}: ${error.message}`);
  console.error('A copy under node_modules must be JavaScript: Node refuses to strip types inside');
  console.error('node_modules (ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING). That is why this');
  console.error('package ships lib/ and points main/exports["."] at it — reinstall the profile copy');
  console.error("if the file above is a .ts source, which means the profile holds an old version.");
  process.exit(1);
}

// A real cordis Context, with only the members the plugin touches replaced. The
// real Context matters: `Service` registers itself through `ctx.reflect`, so a
// plain object literal cannot host a service.
//
// cordis is resolved through whichever scope the plugin itself came from, not
// through this script's location. The plugin will hold a `Context` from whichever
// copy the loader gives it, and the service registry is keyed by that copy, so
// the smoke test has to use the same one or it is not reproducing what DSH does.
let Context;
try {
  const cordisPath = anchor.resolve('@deepseek-ai/cordis');
  ({ Context } = await import(pathToFileURL(cordisPath).href));
} catch (error) {
  console.error(`FAILED: @deepseek-ai/cordis does not resolve from ${origin}: ${error.message}`);
  process.exit(1);
}

const registered = new Map();
const promptContexts = [];
const logs = [];
const disposers = [];
const skillProviders = [];
const injectRequests = [];

// ── Stand in for the skill registry ─────────────────────────────────────────
// A real profile has `ctx.skills`, provided by `@deepseek-ai/dsh-skill`. This
// harness mounts no service lifecycle at all, so `ctx.inject(['skills'], …)`
// would never fire and the bundled skill would be reported as missing for a
// reason that has nothing to do with the plugin. The stand-in registers a
// recording registry and runs any injected callback immediately, which is what
// cordis does once the required service is available.
const skillsStandIn = {
  registerProvider: (create) => {
    skillProviders.push(create({ signal: new AbortController().signal }));
    return () => {};
  },
};

/** Assigned below; the inject stand-in needs the proxied context to hand back. */
let ctx;

const base = new Context();
const extended = base.extend({
  tools: {
    register: (definition) => {
      registered.set(definition.name, definition);
      return () => registered.delete(definition.name);
    },
  },
  logger: () => ({
    debug: (message) => logs.push(`debug ${message}`),
    info: (message) => logs.push(`info ${message}`),
    warn: (message) => logs.push(`warn ${message}`),
    error: (message) => logs.push(`error ${message}`),
  }),
  systemPrompt: {
    context: (contribution) => {
      promptContexts.push(contribution);
      return () => {};
    },
    getContextOrder: () => 110,
  },
  effect: (execute) => {
    const result = execute();
    const disposer = typeof result === 'function' ? result : () => {};
    disposers.push(disposer);
    return disposer;
  },
  skills: skillsStandIn,
  inject: (deps, callback) => {
    const names = Array.isArray(deps) ? deps : Object.keys(deps);
    injectRequests.push(names.join(', '));
    if (names.includes('skills')) callback(ctx);
    return { then: () => {} };
  },
});

// ── Enforce the `inject` contract, the way cordis does ──────────────────────
// cordis refuses to hand a plugin a service it did not declare, and the failure
// is fatal to the whole profile boot:
//
//   cannot get property "systemPrompt" without inject
//
// A permissive stand-in context cannot reproduce that, which is exactly how a
// broken `inject` list passed this script and then stopped DSH from starting.
// This proxy restores the enforcement: a read of a service that is neither a
// cordis builtin nor declared in the plugin's `inject` array throws here, the
// same way it would in production.
const injected = new Set(Array.isArray(plugin.inject) ? plugin.inject : []);
const cordisBuiltins = new Set([
  'root', 'baseUrl', 'events', 'logger', 'reflect', 'registry',
  'extend', 'isolate', 'intercept', 'on', 'once', 'emit',
  'parallel', 'serial', 'bail', 'waterfall', 'plugin', 'inject', 'effect',
  'get', 'set', 'provide', 'accessor', 'mixin', 'start', 'stop',
  'fiber', 'scope', 'then', 'symbols', 'constructor', 'prototype',
]);
const undeclaredReads = [];

ctx = new Proxy(extended, {
  get(target, property, receiver) {
    if (typeof property === 'string' && !cordisBuiltins.has(property) && !injected.has(property)) {
      // Only flag reads that resolve to nothing: property access on our own test
      // stubs and on the real Context is legitimate and unrelated to `inject`.
      if (Reflect.get(target, property, receiver) === undefined) {
        undeclaredReads.push(property);
        throw new Error(`cannot get property "${property}" without inject`);
      }
    }
    return Reflect.get(target, property, receiver);
  },
});

let failure;
try {
  plugin.apply(ctx, {});
} catch (error) {
  failure = error;
}

console.log('Registered tools:');
if (registered.size === 0) {
  console.log('  (none)');
} else {
  for (const name of [...registered.keys()].sort()) console.log(`  ${name}`);
}
console.log('');

const ready = logs.find((line) => line.includes('model hub ready'));
if (ready) console.log(`Catalog: ${ready.replace(/^info\s*/, '')}`);
const disabled = logs.find((line) => line.includes('model hub disabled'));
if (disabled) console.log(`Catalog: ${disabled.replace(/^warn\s*/, '')}`);

console.log(`Capability context registered: ${promptContexts.length > 0 ? 'yes' : 'no'}`);
console.log(`Lifetime disposers registered: ${disposers.length}`);
console.log(`Declared inject: ${JSON.stringify([...injected])}`);
if (injectRequests.length > 0) {
  console.log(`Injected on demand: ${JSON.stringify([...new Set(injectRequests)])}`);
}

// Read what the bundled skill provider actually advertises: a registered
// provider that offers nothing means the skill file could not be read or parsed,
// which would otherwise be visible only as a missing entry in a live session.
const offeredSkills = [];
for (const provider of skillProviders) {
  for (const candidate of await provider.list({})) {
    offeredSkills.push(`${candidate.name} (provider ${provider.name}, rank ${candidate.rank})`);
  }
}
if (skillProviders.length === 0) {
  console.log('Bundled skill: (no provider registered)');
} else if (offeredSkills.length === 0) {
  console.log('Bundled skill: (provider registered but offered nothing)');
} else {
  for (const offered of offeredSkills) console.log(`Bundled skill: ${offered}`);
}

if (undeclaredReads.length > 0) {
  console.log(`Undeclared service reads: ${[...new Set(undeclaredReads)].join(', ')}`);
}
console.log('');

// Clean up anything the plugin started, so this script never leaves a process.
for (const dispose of disposers) {
  try {
    await dispose();
  } catch {
    /* best effort */
  }
}

if (undeclaredReads.length > 0) {
  console.error(
    `FAILED: the plugin read ${[...new Set(undeclaredReads)].map((name) => `"${name}"`).join(', ')} ` +
      'without declaring it in `inject`.',
  );
  console.error('cordis aborts the entire profile boot when a plugin reads an undeclared');
  console.error('service, so DSH would refuse to start. Add the service to `inject` in');
  console.error('dsh-plugin/index.ts.');
  process.exit(1);
}
if (failure !== undefined) {
  console.error(`FAILED: apply() threw: ${failure.message}`);
  process.exit(1);
}
if (registered.size === 0) {
  console.error('FAILED: the plugin registered no tools.');
  console.error('The most likely cause is that no model catalog was found. The plugin');
  console.error('searches upward from the DSH working directory and then from its own');
  console.error('installation directory, logs the reason, and stays disabled rather than');
  console.error('failing the boot. Create models.json in one of those directories.');
  process.exit(1);
}
if (skillProviders.length > 0 && offeredSkills.length === 0) {
  console.error('FAILED: the skill provider registered but offers no skill.');
  console.error('Its file is unreadable or its frontmatter is invalid; the warning above');
  console.error('names the problem. Check skills/dsh-ai-model-hub/SKILL.md.');
  process.exit(1);
}

console.log(`OK: the plugin applied and registered ${registered.size} tools.`);
