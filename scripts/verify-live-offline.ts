/**
 * Verification against the live machine with ComfyUI **stopped**.
 *
 * The companion to `verify-live-scan.ts`, and the other half of the question: that
 * script asks a running engine what it saved, and this one asks whether the saved
 * workflows are still discoverable when the engine is not running at all. It uses
 * the real installation's workflow store and the real files, and no HTTP probe is
 * allowed to succeed, so every row it prints came from a file or from the
 * remembered catalog — never from a live API.
 *
 * It writes nothing by default. Pass a path to exercise persistence:
 *
 *   node scripts/verify-live-offline.ts
 *   node scripts/verify-live-offline.ts ./tmp-workflow-catalog.json
 *
 * The `DSH_AI_MODEL_HUB_WORKFLOW_CATALOG` environment variable redirects the
 * default location when that is what you want to inspect.
 *
 * @module dsh-ai-model-hub/scripts/verify-live-offline
 */

import { readFile } from 'node:fs/promises';

import { ModelHub } from '../src/index.ts';
import { buildInventory, DEFAULT_PROBES } from '../dsh-plugin/inventory.ts';
import type { InventoryProbes } from '../dsh-plugin/inventory.ts';
import { loadWorkflowCatalog, defaultWorkflowCatalogPath } from '../dsh-plugin/workflow-catalog.ts';
import type { PluginLogger } from '../dsh-plugin/types.ts';
import type { ModelCatalogConfig } from '../src/index.ts';
/** The catalog to describe: this machine's ComfyUI, at its usual endpoint. */
const catalog: ModelCatalogConfig = {
  version: '1',
  hosts: [
    {
      id: 'comfyui',
      name: 'ComfyUI',
      adapter: 'comfyui',
      runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint: 'http://127.0.0.1:8188', path: '/prompt' },
      lifecycle: { startable: true, stoppable: true, start: { command: 'python.exe', args: ['main.py'] } },
    },
  ],
  models: [],
};

/** Every HTTP effect refuses, exactly as it does when nothing is listening. */
const stopped: InventoryProbes = {
  ...DEFAULT_PROBES,
  fetchJson: async (url: string) => {
    throw Object.assign(new Error(`connect ECONNREFUSED (${url})`), { code: 'ECONNREFUSED' });
  },
};

const logger: PluginLogger = {
  debug: (message) => console.log('  debug', message),
  info: (message) => console.log('  info', message),
  warn: (message) => console.log('  warn', message),
  error: (message) => console.log('  error', message),
};

const catalogPath = process.argv[3] ?? 'config/models.json';
const persistPath = process.argv[2];

const hub = ModelHub.fromConfig(catalog, { manageTimers: false, probeResources: false, catalogPath, log: () => {} });
const report = await buildInventory(hub, {
  hosts: catalog.hosts ?? [],
  catalogPath,
  artifactRoot: '',
  allowProcessLaunch: true,
  probes: stopped,
  probe: true,
  logger,
  ...(persistPath === undefined ? { persistWorkflows: false } : { workflowCatalogPath: persistPath }),
});

console.log(`generated: ${report.generatedAt}`);
console.log(`registered: ${report.scan.registered}`);
console.log('\nworkflows visible with ComfyUI stopped:');
for (const resource of report.resources.filter((entry) => entry.kind === 'comfyui_workflow')) {
  console.log(`\n  ${resource.name}`);
  console.log(`    id:          ${resource.id}`);
  console.log(`    status:      ${resource.status}  runnable=${resource.runnable}  provider=${resource.modelId ?? '(none)'}`);
  console.log(`    freshness:   ${resource.freshness}  availability=${resource.availability}  engineRunning=${resource.engineRunning}`);
  console.log(`    source:      ${resource.source}`);
  console.log(`    detail:      ${resource.detail}`);
}

console.log('\nsources:');
for (const source of report.scan.sources) {
  console.log(`  [${source.ok ? 'ok  ' : 'fail'}] ${source.id}: ${source.detail} (found ${source.found})`);
}

console.log('\nengines:');
for (const engine of report.engines) {
  console.log(`  ${engine.id}: running=${engine.running} state=${engine.state} install=${engine.installPath ?? '(not found)'}`);
  for (const store of engine.storePaths) console.log(`    store: ${store.path} exists=${store.exists}`);
}

console.log('\nwarnings:');
for (const warning of report.scan.warnings) console.log(`  ! ${warning}`);

console.log(`\npersisted workflow catalog: ${persistPath ?? defaultWorkflowCatalogPath()} ${persistPath === undefined ? '(not written; pass a path to write one)' : ''}`);
if (persistPath !== undefined) {  const saved = await loadWorkflowCatalog(persistPath, logger);
  for (const scope of saved.document.scopes) {
    console.log(`  scope ${scope.hostId} @ ${scope.endpoint ?? '(no endpoint)'} updated ${scope.updatedAt}`);
    for (const workflow of scope.workflows) {
      console.log(`    ${workflow.name}: ${workflow.status} runnable=${workflow.runnable} descriptor=${workflow.descriptor === undefined ? 'no' : 'yes'} <- ${workflow.source}`);
    }
  }
  console.log(`  raw document:\n${await readFile(persistPath, 'utf8')}`);
}

await hub.dispose();
