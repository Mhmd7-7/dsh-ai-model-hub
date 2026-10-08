/**
 * Verification against the live ComfyUI on this machine.
 *
 * Prints what the scan discovers, analyses and registers for the real saved
 * workflows, using ComfyUI's own `/object_info` as the metadata source. Run it
 * with `node scripts/verify-live-scan.ts`; it needs a ComfyUI answering on
 * 127.0.0.1:8188 and does not start or stop anything.
 */

import { ModelHub } from '../src/index.ts';
import { scanLocalResources } from '../dsh-plugin/scan.ts';
import type { ModelCatalogConfig } from '../src/index.ts';

const catalog: ModelCatalogConfig = {
  version: '1',
  hosts: [
    {
      id: 'comfyui',
      name: 'ComfyUI',
      adapter: 'comfyui',
      runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint: 'http://127.0.0.1:8188', path: '/prompt' },
    },
    {
      id: 'ollama',
      name: 'Ollama',
      adapter: 'openai_compatible',
      runtime: { engine: 'ollama', adapter: 'openai_compatible', endpoint: 'http://127.0.0.1:11434' },
    },
  ],
  models: [],
};

const hub = ModelHub.fromConfig(catalog, { manageTimers: false, probeResources: false, log: () => {} });
const result = await scanLocalResources(hub, {
  hosts: catalog.hosts ?? [],
  startEngine: false,
  timeoutMs: 60_000,
});

console.log(`generated: ${result.generatedAt}`);
console.log(`registered: ${result.registered}`);
console.log(`warnings: ${result.warnings.length === 0 ? '(none)' : ''}`);
for (const warning of result.warnings) console.log(`  ! ${warning}`);
console.log('\nsources:');
for (const source of result.sources) {
  console.log(`  [${source.ok ? 'ok  ' : 'fail'}] ${source.id} (${source.kind}): ${source.detail} (found ${source.found})`);
}

console.log('\nresources:');
for (const resource of result.resources) {
  console.log(`\n  ${resource.typeLabel}: ${resource.name}`);
  console.log(`    id:     ${resource.id}`);
  console.log(`    status: ${resource.status}  runnable=${resource.runnable}  engineRunning=${resource.engineRunning}`);
  console.log(`    source: ${resource.source}`);
  console.log(`    detail: ${resource.detail}`);
  if (resource.capabilities?.length) console.log(`    capabilities: ${resource.capabilities.join(', ')}`);
  for (const input of resource.inputs ?? []) {
    console.log(`    input:  ${input.name} (${input.kind}) -> node ${input.node}.${input.input}  [${input.label}]`);
  }
  for (const output of resource.outputs ?? []) {
    console.log(`    output: ${output.name} (${output.type}) <- node ${output.node} (${output.nodeClass})`);
  }
  for (const note of resource.diagnostics ?? []) console.log(`    note:   ${note}`);
}

console.log('\ncatalog providers after the scan:');
for (const model of hub.catalog.listModels()) {
  console.log(`  ${model.id} [${model.type}] caps=${model.capabilities.join(',')} host=${model.hostId}`);
}

await hub.dispose();
