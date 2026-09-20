/**
 * Cross-platform installer.
 *
 * Runs the same three steps as `install-plugin.ps1`, but from Node, so the
 * install works on macOS and Linux too and needs no shell of its own.
 *
 *   1. `npm install` inside `dsh-plugin/` to materialise the DSH peer closure.
 *   2. `dsh plugin add` to install the plugin and register it as a profile layer.
 *   3. Verify by importing the plugin exactly the way DSH's loader will.
 *
 * Usage:
 *
 *   node scripts/install-plugin.mjs [--profile web] [--skip-doctor]
 *
 * @module dsh-ai-model-hub/scripts/install-plugin
 */

import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');
const pluginDir = join(repoRoot, 'dsh-plugin');

/**
 * Read a `--flag value` argument.
 * @param {string} flag - the flag name without dashes.
 * @param {string} fallback - the value to use when the flag is absent.
 * @returns {string} the resolved value.
 */
function readArgument(flag, fallback) {
  const index = process.argv.indexOf(`--${flag}`);
  if (index >= 0 && process.argv[index + 1] !== undefined && !process.argv[index + 1].startsWith('--')) {
    return process.argv[index + 1];
  }
  return fallback;
}

const profile = readArgument('profile', 'web');
const skipDoctor = process.argv.includes('--skip-doctor');

const dshHome = process.env['DSH_HOME'] ?? join(homedir(), '.dsh');
const profileDir = join(dshHome, 'profiles', profile);

/**
 * Resolve a command to an executable that can be spawned without a shell.
 *
 * On Windows `npm` and `dsh` are `.cmd` shims, and spawning them requires either
 * `shell: true` — which Node deprecates for argument-concatenation reasons — or
 * the resolved `.cmd` path plus `shell: true` on that exact name. Resolving the
 * path explicitly and spawning it directly keeps the arguments as a real argv
 * array, which is the same no-injection posture the hub itself takes.
 *
 * @param {string} command - the command name.
 * @returns {string} a path or name safe to pass to spawnSync.
 */
function resolveExecutable(command) {
  if (process.platform !== 'win32') return command;
  const result = spawnSync('where', [command], { encoding: 'utf8' });
  if (result.status !== 0 || typeof result.stdout !== 'string') return command;
  const candidates = result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  // `where` lists the extensionless shim first, which Windows cannot execute
  // directly; the `.cmd` sibling is the one to spawn.
  return candidates.find((line) => /\.cmd$/i.test(line)) ?? candidates[0] ?? command;
}

/**
 * Run a command, inheriting stdio, and report whether it succeeded.
 *
 * Every argument here is either a literal or a path this script derived, never
 * model- or user-supplied text, so there is no injection surface.
 *
 * @param {string} command - the program to run.
 * @param {string[]} args - its arguments.
 * @param {string} [cwd] - working directory.
 * @returns {boolean} whether it exited zero.
 */
function run(command, args, cwd) {
  const executable = resolveExecutable(command);
  // A resolved `.cmd`/`.bat` still needs a shell on Windows to be executable.
  // That triggers Node's DEP0190, which warns that shell arguments are
  // concatenated rather than escaped. It does not apply here: every element of
  // `args` is a literal or a path this script derived, and none of it is user- or
  // model-supplied. Worth knowing about, hence this explanation rather than a
  // blanket suppression — but not worth printing over the install output.
  const needsShell = process.platform === 'win32' && /\.(cmd|bat)$/i.test(executable);
  const result = spawnSync(executable, args, {
    cwd,
    stdio: 'inherit',
    shell: needsShell,
  });
  if (result.error) {
    console.error(`  could not run ${command}: ${result.error.message}`);
    return false;
  }
  return result.status === 0;
}

console.log('dsh-ai-model-hub plugin installer');
console.log(`  plugin:  ${pluginDir}`);
console.log(`  profile: ${profileDir}`);
console.log('');

if (!existsSync(join(pluginDir, 'package.json'))) {
  console.error(`Could not find the plugin package at ${pluginDir}`);
  process.exit(1);
}

// ── 1. The plugin's dependencies ────────────────────────────────────────────
// DSH installs plugins as junction links, so Node loads the plugin from its real
// path here rather than from a copy inside the profile. Resolution therefore walks
// up from this repository, and the node_modules that matters is the plugin's own.
console.log("[1/3] Installing the plugin's DSH dependencies...");
const installed = run(
  'npm',
  ['install', '--no-audit', '--no-fund', '--cache', join(repoRoot, '.npm-cache')],
  pluginDir,
);
if (!installed) {
  console.error('  npm install failed.');
  process.exit(1);
}

// Fail early and clearly when the closure is incomplete: a missing peer here
// becomes an opaque module error much later, at DSH boot.
const required = [
  '@deepseek-ai/dsh-tools',
  '@deepseek-ai/dsh-llm',
  '@deepseek-ai/cordis',
  '@deepseek-ai/schemastery',
  '@deepseek-ai/dsh-agent',
  '@deepseek-ai/dsh-scope',
];
const missing = required.filter((name) => !existsSync(join(pluginDir, 'node_modules', name)));
if (missing.length > 0) {
  console.error(`  dependencies incomplete; missing: ${missing.join(', ')}`);
  process.exit(1);
}
console.log('  dependencies present.');

// ── 2. Install into the profile ─────────────────────────────────────────────
// `dsh plugin add` also appends the package to `dsh.profile.bundles`, because the
// plugin declares `dsh.bundle.patch`. No profile file is edited by hand.
console.log('');
console.log(`[2/3] Installing into profile '${profile}'...`);
if (!run('dsh', ['plugin', '--profile', profile, 'add', pluginDir])) {
  console.error('  dsh plugin add failed.');
  process.exit(1);
}

// ── 2b. Guarantee the bundle layer is registered ────────────────────────────
// `dsh plugin add` reconciles `dsh.profile.bundles` against installed packages by
// checking whether each dependency declares `dsh.bundle.patch`. That check reads
// the manifest through the profile's resolution, and it silently does the wrong
// thing for a junctioned local path in at least one case we hit: the dependency
// installs and links, but never joins the layer stack. Without the layer the
// plugin simply never loads — no error, no tools, nothing to debug.
//
// Registering it here is idempotent and cheap, and it is the difference between
// "installed" and "actually running".
const profileManifestPath = join(profileDir, 'package.json');
if (!existsSync(profileManifestPath)) {
  console.error(`  profile manifest missing at ${profileManifestPath}`);
  process.exit(1);
}
const pluginManifest = JSON.parse(readFileSync(join(pluginDir, 'package.json'), 'utf8'));
const pluginPackageName = pluginManifest.name;
if (typeof pluginPackageName !== 'string' || pluginPackageName.length === 0) {
  console.error(`  ${join(pluginDir, 'package.json')} declares no "name"`);
  process.exit(1);
}

const profileManifest = JSON.parse(readFileSync(profileManifestPath, 'utf8'));
profileManifest.dsh ??= {};
profileManifest.dsh.profile ??= {};
const bundles = profileManifest.dsh.profile.bundles;
if (!Array.isArray(bundles)) {
  console.error(`  ${profileManifestPath} has no dsh.profile.bundles array`);
  process.exit(1);
}
if (bundles.includes(pluginPackageName)) {
  console.log(`  bundle layer '${pluginPackageName}' already registered.`);
} else {
  bundles.push(pluginPackageName);
  writeFileSync(profileManifestPath, `${JSON.stringify(profileManifest, null, 2)}\n`, 'utf8');
  console.log(`  registered '${pluginPackageName}' as a bundle layer.`);
}

// ── 3. Verify ───────────────────────────────────────────────────────────────
if (!skipDoctor) {
  console.log('');
  console.log('[3/3] Verifying...');
  if (!run('node', [join(here, 'doctor.mjs'), '--profile', profile])) {
    console.error('');
    console.error('Verification failed; the plugin will not load until this passes.');
    process.exit(1);
  }
}

console.log('');
console.log('Done.');
console.log('');
console.log(`Restart DeepSeek Harness so the '${profile}' profile reloads, then ask the agent:`);
console.log('    List the available AI model capabilities.');
console.log('');
console.log('The plugin finds its model catalog by searching upward from the DSH working');
console.log("directory and then from this plugin's own installation directory, so the");
console.log('catalog shipped in this repository is used with no configuration. Set');
console.log('`searchRoots` or `configPath` in the plugin row to point it somewhere else.');
