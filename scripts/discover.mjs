/**
 * Show what runtime discovery would publish for one configured host, without
 * starting the hub or touching any catalog.
 *
 * This is the tool for the question "why is this engine's model not showing up?"
 * It runs the real discoverer against the real endpoint and prints exactly what
 * it built: capabilities, input/output kinds, the generated graph, and the
 * provenance notes an operator would read in `list_models`.
 *
 * Usage:
 *
 *   node scripts/discover.mjs                 # every host in config/models.json
 *   node scripts/discover.mjs comfyui         # one host, by id or engine
 *   node scripts/discover.mjs --catalog path/to/models.json
 *
 * Read-only: it issues the discovery requests a normal pass would, and nothing
 * else. An unreachable engine is reported as a warning, exactly as the hub does.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const args = process.argv.slice(2);
const catalogFlag = args.indexOf('--catalog');
const catalogPath = catalogFlag === -1 ? 'config/models.json' : args[catalogFlag + 1];
const selector = args.find((arg, index) => !arg.startsWith('--') && index !== catalogFlag + 1);

let raw;
try {
  raw = JSON.parse(readFileSync(resolve(catalogPath), 'utf8').replace(/^\uFEFF/, ''));
} catch (error) {
  console.error(`could not read ${catalogPath}: ${error.message}`);
  process.exit(1);
}

const lib = pathToFileURL(resolve('lib/src/index.js')).href;
const { createOllamaDiscoverer, createA1111Discoverer, createComfyUiDiscoverer } = await import(lib);

/** The discoverers the hub uses for engines it knows how to introspect. */
const discoverers = [createOllamaDiscoverer(), createA1111Discoverer(), createComfyUiDiscoverer()];
const byEngine = new Map();
for (const discoverer of discoverers) {
  byEngine.set(discoverer.engine, discoverer);
  for (const alias of discoverer.aliases ?? []) byEngine.set(alias, discoverer);
}

const hosts = (raw.hosts ?? []).filter((host) => selector === undefined || host.id === selector || host.runtime.engine === selector);
if (hosts.length === 0) {
  console.error(`no host matched ${selector === undefined ? '(none configured)' : `"${selector}"`}`);
  process.exit(1);
}

let total = 0;
for (const host of hosts) {
  const discoverer = byEngine.get(host.runtime.engine);
  console.log(`\n=== ${host.id} (engine ${host.runtime.engine}) ===`);
  if (discoverer === undefined) {
    console.log('  no discoverer handles this engine, so nothing is discovered for it');
    continue;
  }
  if (host.runtime.endpoint === undefined) {
    console.log('  no runtime.endpoint, so there is nothing to introspect');
    continue;
  }

  let descriptors;
  try {
    descriptors = await discoverer.discover(host, new AbortController().signal);
  } catch (error) {
    console.log(`  WARNING: ${error.message}`);
    console.log('  (this is a warning in the hub, not a failure: the host contributes no models)');
    continue;
  }

  console.log(`  published ${descriptors.length} model(s)`);
  total += descriptors.length;
  for (const descriptor of descriptors) {
    console.log(`\n  ${descriptor.id}`);
    console.log(`    name:         ${descriptor.name}`);
    console.log(`    capabilities: ${descriptor.capabilities.join(', ')}`);
    console.log(`    in/out:       ${(descriptor.inputTypes ?? []).join('|')} -> ${(descriptor.outputTypes ?? []).join('|')}`);
    console.log(`    resources:    ${descriptor.resources?.vramGb ?? '?'} GiB VRAM (estimated)`);
    const workflow = descriptor.adapterConfig?.workflow;
    console.log(`    graph:        ${workflow === undefined ? 'none — this model needs a workflow template' : `${Object.keys(workflow).length} generated node(s)`}`);
    console.log(`    notes:        ${descriptor.notes ?? ''}`);
  }
}
console.log(`\n${total} model(s) across ${hosts.length} host(s).`);
