/**
 * End-to-end live generation through the Hub.
 *
 * Discovers the saved ComfyUI workflows — the editor/UI exports, including the one
 * whose graph is a subgraph instance — registers them, and then asks the Hub for
 * `text_to_image`. Nothing here reads or supplies an API-format workflow: the only
 * input is the UI file ComfyUI itself has saved.
 *
 * Run with `node scripts/verify-live-generate.ts`. It needs a ComfyUI answering on
 * 127.0.0.1:8188; it does not start or stop anything.
 */

import { readFile } from 'node:fs/promises';
import { ModelHub } from '../src/index.ts';
import { scanLocalResources } from '../dsh-plugin/scan.ts';
import type { ModelCatalogConfig } from '../src/index.ts';

const PROMPT = process.argv[2] ?? 'a ginger cat sitting on a sunlit windowsill, soft morning light, shallow depth of field';

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
  models: [
    {
      id: 'ollama_vision',
      name: 'Ollama gemma3 (vision)',
      type: 'image_understanding',
      host: 'ollama',
      capabilities: ['image_understanding'],
      adapterConfig: { model: 'gemma3:latest' },
    },
  ],
};

const hub = ModelHub.fromConfig(catalog, { manageTimers: false, probeResources: false, log: () => {} });

try {
  const scan = await scanLocalResources(hub, { hosts: catalog.hosts ?? [], startEngine: false, timeoutMs: 60_000 });
  console.log(`scan registered ${scan.registered} provider(s)`);
  for (const resource of scan.resources) {
    console.log(`  ${resource.status.padEnd(18)} ${resource.name} → ${resource.capabilities?.join(',') ?? '(none)'}`);
  }

  const candidates = hub.catalog.listModels().filter((model) => model.capabilities.includes('text_to_image'));
  console.log(`\ntext_to_image providers: ${candidates.map((model) => model.id).join(', ') || '(none)'}`);
  if (candidates.length === 0) {
    console.log('BLOCKED: no text_to_image provider was registered from the saved UI workflows.');
    process.exitCode = 1;
  } else {
    const chosen = candidates[0];
    console.log(`\ninvoking "${chosen?.id}" with prompt: ${PROMPT}`);
    const started = Date.now();
    const result = await hub.invokeModel({ capability: 'text_to_image', prompt: PROMPT, timeoutMs: 900_000 });
    console.log(`modelId=${result.modelId} coldStart=${result.coldStart} durationMs=${Date.now() - started}`);

    for (const output of result.outputs) {
      console.log(`  output ${output.id} type=${output.type} mime=${output.mimeType} bytes=${output.byteLength}`);
      const path = await hub.artifacts.resolvePath(output.id);
      const bytes = await readFile(path.path);
      const isPng = bytes[0] === 0x89 && bytes.subarray(1, 4).toString('latin1') === 'PNG';
      // PNG stores width and height big-endian at bytes 16..24 of the IHDR chunk.
      const width = bytes.readUInt32BE(16);
      const height = bytes.readUInt32BE(20);
      console.log(`  file: ${path.path}`);
      console.log(`  verified: PNG magic=${isPng} dimensions=${width}x${height} size=${bytes.length} bytes`);

      // Independent content check: hand the produced artifact straight back to a
      // different model through the same Hub. A valid PNG header only proves a
      // file was written; this proves something was actually rendered.
      if (hub.catalog.listModels().some((model) => model.capabilities.includes('image_understanding'))) {
        const described = await hub.invokeModel({
          capability: 'image_understanding',
          prompt: 'In one sentence: what does this image show? Is there a cat in it?',
          inputs: [{ id: output.id, type: 'image' }],
          timeoutMs: 180_000,
        });
        console.log(`  described by ${described.modelId}: ${JSON.stringify(described.value)}`);
      } else {
        console.log('  (no image_understanding provider registered; content not independently checked)');
      }
    }
  }
} finally {
  await hub.dispose();
}
