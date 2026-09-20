/**
 * Installation doctor.
 *
 * Verifies that a DSH profile can actually load the dsh-ai-model-hub plugin, and
 * reports precisely what is missing when it cannot. Every failure mode this
 * checks has a confusing symptom at runtime — the plugin silently registers no
 * tools, or DSH fails to boot with an opaque module error — so the job here is to
 * turn each of them into one clear line.
 *
 * Usage:
 *
 *   node scripts/doctor.mjs --profile web
 *
 * Exits 0 when the installation is sound, 1 otherwise.
 *
 * @module dsh-ai-model-hub/scripts/doctor
 */

import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

/** The plugin package name this doctor verifies. */
const PLUGIN_PACKAGE = 'dsh-ai-model-hub-plugin';

/** Every `@deepseek-ai/*` package the plugin needs present in the profile. */
const REQUIRED_PEERS = [
  '@deepseek-ai/cordis',
  '@deepseek-ai/schemastery',
  '@deepseek-ai/dsh-tools',
  '@deepseek-ai/dsh-llm',
  '@deepseek-ai/dsh-system-prompt',
];

/** Collected results, so the report is printed in one block. */
const checks = [];

/**
 * Record a check result.
 * @param {string} name - what was checked.
 * @param {boolean} ok - whether it passed.
 * @param {string} [detail] - extra context, shown when it failed.
 */
function check(name, ok, detail) {
  checks.push({ name, ok, detail });
}

/**
 * Read the `--profile` argument.
 * @returns {string} the profile name, defaulting to `web`.
 */
function readProfileArgument() {
  const index = process.argv.indexOf('--profile');
  if (index >= 0 && process.argv[index + 1] !== undefined) return process.argv[index + 1];
  return 'web';
}

/**
 * Resolve the DSH home directory.
 * @returns {string} the absolute DSH home path.
 */
function dshHome() {
  return process.env['DSH_HOME'] ?? join(homedir(), '.dsh');
}

/**
 * Read and parse a JSON file, returning undefined when it is absent or invalid.
 * @param {string} path - the file to read.
 * @returns {any} the parsed value, or undefined.
 */
function readJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf8'));
  } catch {
    return undefined;
  }
}

/**
 * Read the `id` and `name` of the first row a bundle patch inserts.
 *
 * A hand-rolled reader rather than a YAML dependency: this script runs from the
 * repository, where `js-yaml` is not installed, and the shape it needs to read is
 * two scalar keys. Only `- insert:` bodies are considered, so the extensive
 * commented-out configuration below the row cannot be mistaken for the row.
 *
 * @param {string} path - the patch file.
 * @returns {{ id: string, name: string } | undefined} the row, or undefined.
 */
function readPatchTargetModule(path) {
  let text;
  try {
    text = readFileSync(path, 'utf8');
  } catch {
    return undefined;
  }
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);

  let inInsert = false;
  let id;
  let name;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.replace(/#.*$/, '');
    if (/^\s*-\s*insert\s*:/.test(line)) {
      inInsert = true;
      continue;
    }
    if (!inInsert) continue;
    // A new top-level list item ends the insert block.
    if (/^-\s/.test(line) && !/^\s+/.test(rawLine)) {
      inInsert = false;
      continue;
    }
    const idMatch = /^\s*-\s*id\s*:\s*['"]?([^'"\s]+)['"]?/.exec(line);
    if (idMatch && id === undefined) {
      id = idMatch[1];
      continue;
    }
    const nameMatch = /^\s*name\s*:\s*['"]?([^'"\s]+)['"]?/.exec(line);
    if (nameMatch && name === undefined) name = nameMatch[1];
    if (id !== undefined && name !== undefined) break;
  }
  return id !== undefined && name !== undefined ? { id, name } : undefined;
}

const profile = readProfileArgument();
const home = dshHome();
const profileDir = join(home, 'profiles', profile);
const profileModules = join(profileDir, 'node_modules');

// Declared here rather than beside their assignment further down: `report()` can
// run early, when the profile is unreadable, and it reads these. Declaring them
// below that call made the early-exit path die with "Cannot access 'catalogPath'
// before initialization" instead of printing which check failed.
let catalogPath;
let catalogSearched = [];
// Whether the catalog was resolved at all. False on the early-exit path, where
// printing "searched upward from:" would list nothing and read as a bug.
let catalogChecked = false;

console.log(`dsh-ai-model-hub doctor`);
console.log(`  profile: ${profileDir}`);
console.log('');

// ── The profile exists at all ───────────────────────────────────────────────
const profileManifestPath = join(profileDir, 'package.json');
const profileManifest = readJson(profileManifestPath);
check(
  `profile '${profile}' exists`,
  profileManifest !== undefined,
  `no readable package.json at ${profileManifestPath}. Run: dsh plugin --profile ${profile} add <path-to-dsh-plugin>`,
);
if (profileManifest === undefined) {
  report();
}

// ── The plugin is a declared profile layer ──────────────────────────────────
// This is what makes DSH load it: the package declares `dsh.bundle.patch`, and
// `dsh plugin add` appends it to `dsh.profile.bundles` after installing.
const bundles = profileManifest?.dsh?.profile?.bundles ?? [];
const isBundle = bundles.includes(PLUGIN_PACKAGE);
check(
  `'${PLUGIN_PACKAGE}' is in dsh.profile.bundles`,
  isBundle,
  isBundle
    ? undefined
    : `bundles are: ${bundles.join(', ') || '(none)'}. Re-run: dsh plugin --profile ${profile} add <path-to-dsh-plugin>`,
);

// ── The plugin is installed and its entry point resolves ────────────────────
const profileRequire = createRequire(join(profileDir, 'package.json'));
let pluginDir;
let entryPath;
try {
  const manifestPath = profileRequire.resolve(`${PLUGIN_PACKAGE}/package.json`);
  pluginDir = dirname(manifestPath);
  const manifest = readJson(manifestPath);
  entryPath = join(pluginDir, manifest?.main ?? 'index.ts');
  check(`'${PLUGIN_PACKAGE}' resolves`, true);
} catch (error) {
  check(`'${PLUGIN_PACKAGE}' resolves`, false, String(error.message ?? error));
}

check(
  'plugin entry point exists',
  entryPath !== undefined && existsSync(entryPath),
  entryPath === undefined ? 'the plugin is not installed' : `missing file: ${entryPath}`,
);

// ── The DSH peer closure is present ─────────────────────────────────────────
// These are the packages the plugin imports, plus what they import in turn.
// Because every plugin loads from its real path, resolution walks up from this
// repository — so a missing peer fails here rather than at DSH boot.
const missingPeers = [];
for (const peer of REQUIRED_PEERS) {
  try {
    profileRequire.resolve(`${peer}/package.json`);
  } catch {
    missingPeers.push(peer);
  }
}
check(
  'DSH peer packages are resolvable',
  missingPeers.length === 0,
  missingPeers.length === 0 ? undefined : `missing: ${missingPeers.join(', ')}`,
);

// ── The hub library resolves from the plugin's own scope ────────────────────
// The plugin imports `dsh-ai-model-hub` by package name. Under a junction install the
// real path is this repository, so it must be resolvable from there.
let hubResolved = false;
if (pluginDir !== undefined) {
  const pluginRequire = createRequire(join(pluginDir, 'package.json'));
  try {
    pluginRequire.resolve('dsh-ai-model-hub');
    hubResolved = true;
  } catch {
    hubResolved = false;
  }
}
check(
  "the 'dsh-ai-model-hub' library resolves from the plugin",
  hubResolved,
  hubResolved
    ? undefined
    : `run 'npm install' in ${pluginDir ?? '<the dsh-plugin directory>'} so its 'file:..' dependency is linked`,
);

// ── The plugin's own bundle patch names a real plugin ───────────────────────
// This is the check whose absence let a broken install pass. The patch file that
// gets inserted into the profile names a *module specifier*, and the loader
// imports it and demands a function or an object with an `apply` method. Naming
// the hub library instead of the plugin package there fails the entire profile
// boot with
//
//   invalid plugin, expect function or object with an "apply" method, received object
//
// while every other check in this script still passed, because they all looked at
// the plugin package rather than at what the patch actually points to.
if (pluginDir !== undefined) {
  const patchEntry = readPatchTargetModule(join(pluginDir, 'cordis.patch.yml'));
  check(
    'the bundle patch declares an inserted row',
    patchEntry !== undefined,
    `no \`- insert:\` entry with an id and name found in ${join(pluginDir, 'cordis.patch.yml')}`,
  );

  if (patchEntry !== undefined) {
    const { id, name: moduleName } = patchEntry;
    // Resolve from the PROFILE, not from the plugin directory: that is the
    // resolution DSH's loader performs, and the profile is where the row's module
    // must be installed. Anchoring on the plugin instead reports a false failure,
    // because a package cannot resolve itself by name without self-referencing.
    let targetPath;
    try {
      targetPath = profileRequire.resolve(moduleName);
      check(`the patch row's module '${moduleName}' resolves from the profile`, true);
    } catch (error) {
      check(
        `the patch row's module '${moduleName}' resolves from the profile`,
        false,
        `${String(error.message ?? error)} — the row's \`name\` must be the plugin package, not the hub library`,
      );
    }

    if (targetPath !== undefined) {
      try {
        const target = await import(pathToFileURL(targetPath).href);
        const isPlugin = typeof target.apply === 'function' && typeof target.name === 'string';
        check(
          `'${moduleName}' is actually a plugin (exports apply)`,
          isPlugin,
          isPlugin
            ? undefined
            : `it exports: ${Object.keys(target).sort().join(', ') || '(nothing)'}. The row's \`name\` must point ` +
              `at the package exporting name/inject/apply; naming a library such as 'dsh-ai-model-hub' fails the whole boot.`,
        );
        check('the patch row has an id usable for overrides', id.length > 0);
      } catch (error) {
        check(`'${moduleName}' is actually a plugin (exports apply)`, false, String(error.message ?? error));
      }
    }
  }
}

// ── The catalog the hub will read ───────────────────────────────────────────
// Resolved through the hub's OWN loader rather than a re-implementation, so this
// report can never disagree with what the plugin actually does at boot. The
// anchors mirror the plugin's: the host's working directory, then the plugin's
// own installation directory. The second is what makes a hub installed from a
// checkout work with no configuration — it finds the catalog that checkout ships.
const catalogAnchors = [process.cwd(), ...(entryPath === undefined ? [] : [dirname(entryPath)])];
try {
  const { loadCatalogFromAnchors } = await import(new URL('../src/index.ts', import.meta.url).href);
  const loaded = loadCatalogFromAnchors({ anchors: catalogAnchors });
  catalogPath = loaded.path;
  catalogSearched = loaded.searched;
} catch (error) {
  catalogSearched = Array.isArray(error?.details?.anchors) ? error.details.anchors : catalogAnchors;
}
catalogChecked = true;

// ── Actually import the plugin ──────────────────────────────────────────────
// The only check that proves everything above holds at once.
let pluginExports;
if (entryPath !== undefined && existsSync(entryPath) && missingPeers.length === 0) {
  try {
    pluginExports = await import(pathToFileURL(entryPath).href);
    check(
      'the plugin module imports',
      true,
      undefined,
    );
  } catch (error) {
    check('the plugin module imports', false, String(error.message ?? error));
  }
} else {
  check('the plugin module imports', false, 'skipped: an earlier check failed');
}

// ── The Loader contract ─────────────────────────────────────────────────────
if (pluginExports !== undefined) {
  const hasName = typeof pluginExports.name === 'string' && pluginExports.name.length > 0;
  const hasInject = Array.isArray(pluginExports.inject);
  const hasApply = typeof pluginExports.apply === 'function';
  const hasNoDefault = pluginExports.default === undefined;
  check('exports name / inject / apply', hasName && hasInject && hasApply,
    `name=${String(pluginExports.name)} inject=${JSON.stringify(pluginExports.inject)} apply=${typeof pluginExports.apply}`);
  check("no default export (a default would make DSH drop 'inject')", hasNoDefault);
  check('declares a Config schema', typeof pluginExports.Config === 'function');
}

report();

/**
 * Print the report and exit with the right status.
 */
function report() {
  console.log('Checks:');
  for (const entry of checks) {
    const marker = entry.ok ? '  OK  ' : ' FAIL ';
    const colour = entry.ok ? '\u001b[32m' : '\u001b[31m';
    console.log(`${colour}${marker}\u001b[0m ${entry.name}`);
    if (!entry.ok && entry.detail !== undefined) console.log(`         ${entry.detail}`);
  }

  const failed = checks.filter((entry) => !entry.ok);
  console.log('');
  if (!catalogChecked) {
    console.log('Model catalog not resolved: the profile check above failed first.');
  } else if (catalogPath !== undefined) {
    console.log(`Model catalog the plugin will read: ${catalogPath}`);
  } else {
    console.log('No model catalog found. The plugin searched upward from:');
    for (const anchor of catalogSearched) console.log(`  ${anchor}`);
    console.log('The plugin still loads: it logs "model hub disabled", registers no');
    console.log('tools, and lets DSH boot. Create models.json in one of those');
    console.log('directories (see config/models.mock.json) to enable it.');
  }
  console.log('');

  if (failed.length === 0) {
    console.log('Installation looks sound. Restart DeepSeek Harness to load it.');
    process.exit(0);
  }
  console.log(`${failed.length} check(s) failed.`);
  process.exit(1);
}
