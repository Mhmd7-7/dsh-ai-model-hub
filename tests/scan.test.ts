/**
 * Tests for the Local models scan: Ollama models and ComfyUI workflows.
 *
 * The scan is exercised through injected probes, so no Ollama and no ComfyUI has
 * to be running, and one end-to-end case drives a real (fake) ComfyUI HTTP server
 * so the whole path — discover a saved workflow, publish it, queue it, collect the
 * output — is covered rather than only its parts.
 *
 * The assertions that matter most are the negative ones: a ComfyUI install full of
 * checkpoints, LoRAs and VAEs must contribute *workflows* and nothing else.
 *
 * @module dsh-ai-model-hub/tests/scan.test
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import { describe, it } from 'node:test';

import { ModelHub, renderMockPng } from '../src/index.ts';
import type { ModelCatalogConfig, ModelHost } from '../src/index.ts';
import { buildInventory } from '../dsh-plugin/inventory.ts';
import { scanLocalResources } from '../dsh-plugin/scan.ts';
import type { ScanProbes } from '../dsh-plugin/scan.ts';

/** The fake ComfyUI install: it has weights, and it has node classes. */
const objectInfo = {
  CheckpointLoaderSimple: { input: { required: { ckpt_name: [['A.safetensors', 'B.safetensors'], {}] } } },
  LoraLoader: { input: { required: { lora_name: [['C.safetensors'], {}] } } },
  VAELoader: { input: { required: { vae_name: [['D.safetensors'], {}] } } },
  ControlNetLoader: { input: { required: { control_net_name: [['E.safetensors'], {}] } } },
  CLIPLoader: { input: { required: { clip_name: [['F.safetensors'], {}] } } },
  CLIPTextEncode: {
    input: { required: { clip: ['CLIP', {}], text: ['STRING', {}] } },
    output: ['CONDITIONING'],
  },
  EmptyLatentImage: {
    input: { required: { width: ['INT', {}], height: ['INT', {}], batch_size: ['INT', {}] } },
    output: ['LATENT'],
  },
  KSampler: {
    input: {
      required: {
        model: ['MODEL', {}],
        positive: ['CONDITIONING', {}],
        negative: ['CONDITIONING', {}],
        latent_image: ['LATENT', {}],
        seed: ['INT', {}],
        steps: ['INT', {}],
        cfg: ['FLOAT', {}],
      },
    },
    output: ['LATENT'],
  },
  VAEDecode: { input: { required: { samples: ['LATENT', {}], vae: ['VAE', {}] } }, output: ['IMAGE'] },
  SaveImage: { input: { required: { images: ['IMAGE', {}], filename_prefix: ['STRING', {}] } }, output: [] },
  LoadImage: { input: { required: { image: [['input.png'], {}] } }, output: ['IMAGE', 'MASK'] },
};

/** A complete text-to-image workflow, in API format. */
const textToImage = {
  '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'A.safetensors' } },
  '2': { class_type: 'LoraLoader', inputs: { model: ['1', 0], clip: ['1', 1], lora_name: 'C.safetensors' } },
  '3': { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 1], text: 'a red fox' } },
  '4': { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 1], text: '' } },
  '5': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
  '6': {
    class_type: 'KSampler',
    inputs: {
      model: ['2', 0],
      positive: ['3', 0],
      negative: ['4', 0],
      latent_image: ['5', 0],
      seed: 1,
      steps: 20,
      cfg: 7,
    },
  },
  '7': { class_type: 'VAEDecode', inputs: { samples: ['6', 0], vae: ['1', 2] } },
  '8': { class_type: 'SaveImage', inputs: { images: ['7', 0], filename_prefix: 'x' } },
};

/** A graph whose only terminal node previews, so it delivers nothing. */
const previewOnly = {
  '1': { class_type: 'CLIPTextEncode', inputs: { text: 'a red fox' } },
  '2': { class_type: 'PreviewImage', inputs: { images: ['1', 0] } },
};

/** A complete workflow saved in the editor's UI format. */
const editorWorkflow = {
  nodes: [
    { id: 1, type: 'CLIPTextEncode', inputs: [], widgets_values: { text: 'a red fox' } },
    { id: 2, type: 'EmptyLatentImage', inputs: [], widgets_values: { width: 512, height: 512, batch_size: 1 } },
    {
      id: 3,
      type: 'KSampler',
      inputs: [
        { name: 'positive', link: 10 },
        { name: 'negative', link: 11 },
        { name: 'latent_image', link: 12 },
      ],
      widgets_values: { seed: 5, steps: 20, cfg: 7 },
    },
    { id: 4, type: 'VAEDecode', inputs: [{ name: 'samples', link: 13 }], widgets_values: {} },
    { id: 5, type: 'SaveImage', inputs: [{ name: 'images', link: 14 }], widgets_values: { filename_prefix: 'x' } },
  ],
  links: [
    [10, 1, 0, 3, 0, 'CONDITIONING'],
    [11, 1, 0, 3, 1, 'CONDITIONING'],
    [12, 2, 0, 3, 2, 'LATENT'],
    [13, 3, 0, 4, 0, 'LATENT'],
    [14, 4, 0, 5, 0, 'IMAGE'],
  ],
};

/** An editor export naming a node class this install does not have. */
const uninterpretableEditorWorkflow = {
  nodes: [{ id: 1, type: 'TotallyUnknownNode', inputs: [], widgets_values: {} }],
  links: [],
};

/** One Ollama instance's answer to `/api/tags`. */
const ollamaTags = {
  models: [
    { name: 'llama3:8b', size: 4_700_000_000, details: { parameter_size: '8B', quantization_level: 'Q4_K_M' } },
    { name: 'qwen2.5:3b', size: 1_900_000_000, details: { parameter_size: '3B' } },
  ],
};

/**
 * A catalog naming both engines.
 * @param endpoint - the ComfyUI endpoint.
 * @returns the catalog document.
 */
function catalog(endpoint: string): ModelCatalogConfig {
  return {
    version: '1',
    hosts: [
      {
        id: 'ollama',
        name: 'Ollama',
        adapter: 'openai_compatible',
        runtime: {
          engine: 'ollama',
          adapter: 'openai_compatible',
          endpoint: 'http://127.0.0.1:11434',
          path: '/v1/chat/completions',
        },
      },
      {
        id: 'comfyui',
        name: 'ComfyUI',
        adapter: 'comfyui',
        runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint, path: '/prompt' },
      },
    ],
    models: [],
  };
}

/**
 * The declared hosts of the fixture catalog.
 * @param endpoint - the ComfyUI endpoint.
 * @returns the hosts.
 */
function hostsOf(endpoint = 'http://127.0.0.1:8188'): readonly ModelHost[] {
  return catalog(endpoint).hosts ?? [];
}

/**
 * A hub over the fixture catalog, with no timers, no background discovery, and no
 * machine probing — the scan under test is the only thing that touches a machine.
 * @param endpoint - the ComfyUI endpoint.
 * @returns the hub.
 */
function hubFor(endpoint = 'http://127.0.0.1:8188'): ModelHub {
  return ModelHub.fromConfig(catalog(endpoint), {
    manageTimers: false,
    probeResources: false,
    log: () => {},
  });
}

/**
 * A machine description: what each source answers, before it becomes probes.
 *
 * `undefined` for a source means "refused the connection", which is how a test
 * says an engine is down without a second code path.
 */
interface ProbeSpec {
  readonly ollama?: unknown;
  readonly objectInfo?: unknown;
  readonly workflows?: Record<string, unknown>;
  readonly files?: Record<string, string>;
}

/**
 * Build the probes for a machine description.
 * @param spec - what each source answers.
 * @returns the probes.
 */
function probesFor(spec: ProbeSpec): ScanProbes {
  const files = spec.files ?? {};
  const norm = (path: string): string => path.replace(/\\/g, '/');
  const parentOf = (path: string): string => path.slice(0, path.lastIndexOf('/'));
  const baseOf = (path: string): string => path.slice(path.lastIndexOf('/') + 1);

  const dirs = new Set<string>();
  for (const file of Object.keys(files)) {
    const parts = norm(file).split('/');
    for (let index = 2; index < parts.length; index += 1) dirs.add(parts.slice(0, index).join('/'));
  }

  return {
    fetchJson: async (url) => {
      if (url.includes('/api/tags')) {
        if (spec.ollama === undefined) throw new Error('connect ECONNREFUSED');
        return spec.ollama;
      }
      if (url.endsWith('/object_info')) {
        if (spec.objectInfo === undefined) throw new Error('connect ECONNREFUSED');
        return spec.objectInfo;
      }
      if (url.includes('/userdata?')) {
        if (spec.workflows === undefined) throw new Error('connect ECONNREFUSED');
        return Object.keys(spec.workflows);
      }
      const match = /\/userdata\/(.+)$/.exec(url);
      if (match !== null) {
        if (spec.workflows === undefined) throw new Error('connect ECONNREFUSED');
        const name = decodeURIComponent(match[1] as string);
        if (!Object.hasOwn(spec.workflows, name)) throw new Error('HTTP 404');
        return spec.workflows[name];
      }
      throw new Error('connect ECONNREFUSED');
    },
    isDirectory: async (path) => dirs.has(norm(path)),
    listDir: async (path) => {
      const parent = norm(path);
      const entries: { name: string; isDirectory: boolean; sizeBytes?: number }[] = [];
      for (const dir of dirs) {
        if (parentOf(dir) === parent) entries.push({ name: baseOf(dir), isDirectory: true });
      }
      for (const [file, content] of Object.entries(files)) {
        if (parentOf(norm(file)) === parent) {
          entries.push({ name: baseOf(norm(file)), isDirectory: false, sizeBytes: content.length });
        }
      }
      return entries.length > 0 || dirs.has(parent) ? entries : undefined;
    },
    readFile: async (path) => {
      const key = norm(path);
      if (!Object.hasOwn(files, key)) throw new Error(`ENOENT: ${key}`);
      return files[key] as string;
    },
  };
}

describe('Local models scan', () => {
  it('lists installed Ollama models and complete ComfyUI workflows, and no weight files', async () => {
    const hub = hubFor();
    try {
      const result = await scanLocalResources(hub, {
        hosts: hostsOf(),
        probes: probesFor({
          ollama: ollamaTags,
          objectInfo,
          workflows: { 'saved/t2i.json': textToImage },
        }),
      });

      assert.deepEqual(
        result.resources.map((resource) => resource.kind),
        ['ollama_model', 'ollama_model', 'comfyui_workflow'],
      );

      const models = result.resources.filter((resource) => resource.kind === 'ollama_model');
      assert.deepEqual(models.map((resource) => resource.typeLabel), ['Ollama Model', 'Ollama Model']);
      assert.deepEqual(models.map((resource) => resource.name), ['llama3:8b', 'qwen2.5:3b']);
      assert.match(models[0]?.detail ?? '', /8B/);
      assert.equal(models[0]?.status, 'ready');

      const workflows = result.resources.filter((resource) => resource.kind === 'comfyui_workflow');
      assert.equal(workflows.length, 1);
      assert.equal(workflows[0]?.typeLabel, 'ComfyUI Workflow');
      assert.equal(workflows[0]?.name, 't2i');
      assert.equal(workflows[0]?.status, 'ready');
      assert.equal(workflows[0]?.runnable, true);

      // The boundary this whole feature exists for: the install's weights are
      // nowhere in the scan output, and neither is any node class.
      const serialized = JSON.stringify(result.resources);
      for (const weight of ['safetensors']) {
        assert.doesNotMatch(serialized, new RegExp(weight), `${weight} must not be listed`);
      }
      for (const nodeClass of [
        'CheckpointLoaderSimple',
        'LoraLoader',
        'VAELoader',
        'CLIPTextEncode',
        'KSampler',
        'SaveImage',
      ]) {
        assert.doesNotMatch(serialized, new RegExp(nodeClass), `${nodeClass} must not be listed`);
      }
    } finally {
      await hub.dispose();
    }
  });

  it('exposes only the public parameters a workflow needs', async () => {
    const hub = hubFor();
    try {
      const result = await scanLocalResources(hub, {
        hosts: hostsOf(),
        probes: probesFor({ ollama: { models: [] }, objectInfo, workflows: { 'saved/t2i.json': textToImage } }),
      });
      const workflow = result.resources[0];
      assert.deepEqual(workflow?.inputs, ['prompt', 'negative_prompt', 'seed', 'steps', 'cfg', 'width', 'height']);
      assert.deepEqual(workflow?.outputs, ['image']);
      assert.deepEqual(workflow?.capabilities, ['text_to_image']);
      assert.equal(workflow?.format, 'api');
      assert.equal(workflow?.modelId, workflow?.id);
    } finally {
      await hub.dispose();
    }
  });

  it('reports an editor-format workflow as needing conversion instead of dropping it', async () => {
    const hub = hubFor();
    try {
      const result = await scanLocalResources(hub, {
        hosts: hostsOf(),
        probes: probesFor({
          ollama: { models: [] },
          objectInfo,
          workflows: {
            'saved/editor.json': editorWorkflow,
            'saved/unknown-node.json': uninterpretableEditorWorkflow,
          },
        }),
      });

      assert.equal(result.resources.length, 2, 'both editor exports are listed');
      const converted = result.resources.find((resource) => resource.name === 'editor');
      assert.equal(converted?.status, 'needs_conversion', 'editor format is not reported as ready');
      assert.equal(converted?.runnable, true, 'a convertible workflow is still executable');
      assert.equal(converted?.format, 'ui');
      assert.match(converted?.detail ?? '', /editor \(UI\) format/);
      assert.deepEqual(converted?.capabilities, ['text_to_image']);

      const unconvertible = result.resources.find((resource) => resource.name === 'unknown-node');
      assert.equal(unconvertible?.status, 'needs_conversion');
      assert.equal(unconvertible?.runnable, false);
      assert.match(unconvertible?.detail ?? '', /could not be converted/);
    } finally {
      await hub.dispose();
    }
  });

  it('reports a workflow that produces nothing as invalid, with the reason', async () => {
    const hub = hubFor();
    try {
      const result = await scanLocalResources(hub, {
        hosts: hostsOf(),
        probes: probesFor({ ollama: { models: [] }, objectInfo, workflows: { 'saved/preview.json': previewOnly } }),
      });
      assert.equal(result.resources.length, 1, 'the workflow is listed, not silently dropped');
      assert.equal(result.resources[0]?.status, 'invalid');
      assert.equal(result.resources[0]?.runnable, false);
      assert.match(result.resources[0]?.detail ?? '', /saving node|preview/);
    } finally {
      await hub.dispose();
    }
  });

  it('scans a configured workflow directory, preferring the metadata name', async () => {
    const hub = hubFor();
    const named = { ...textToImage, name: 'Cinematic Text To Image' };
    try {
      const result = await scanLocalResources(hub, {
        hosts: hostsOf(),
        workflowDir: '/workflows/comfy',
        probes: probesFor({
          ollama: { models: [] },
          objectInfo,
          workflows: {},
          files: {
            '/workflows/comfy/flux-t2i.json': JSON.stringify(named),
            '/workflows/comfy/nested/broken.json': '{ not json',
            '/workflows/comfy/notes.txt': 'ignored',
          },
        }),
      });

      assert.deepEqual(
        result.resources.map((resource) => resource.name).sort(),
        ['Cinematic Text To Image', 'broken'],
      );
      const namedResource = result.resources.find((resource) => resource.name === 'Cinematic Text To Image');
      assert.equal(namedResource?.status, 'ready');
      assert.match(namedResource?.source ?? '', /flux-t2i\.json$/);
      const broken = result.resources.find((resource) => resource.name === 'broken');
      assert.equal(broken?.status, 'invalid');
      assert.ok(result.sources.some((source) => source.id === 'workflow-dir' && source.ok));
    } finally {
      await hub.dispose();
    }
  });

  it('refreshes on a second scan instead of duplicating entries', async () => {
    const hub = hubFor();
    const probes = probesFor({
      ollama: ollamaTags,
      objectInfo,
      workflows: { 'saved/t2i.json': textToImage, 'saved/editor.json': editorWorkflow },
    });
    try {
      const first = await scanLocalResources(hub, { hosts: hostsOf(), probes });
      const second = await scanLocalResources(hub, { hosts: hostsOf(), probes });

      assert.deepEqual(
        second.resources.map((resource) => resource.id),
        first.resources.map((resource) => resource.id),
      );
      assert.equal(second.resources.length, first.resources.length);
      assert.equal(new Set(second.resources.map((resource) => resource.id)).size, second.resources.length);
      // The published half is replaced wholesale, so a rescan cannot accumulate.
      assert.deepEqual([...hub.scannedProviderIds].sort(), ['comfy-wf-editor', 'comfy-wf-t2i']);
    } finally {
      await hub.dispose();
    }
  });

  it('keeps the two engines independent when one is down', async () => {
    const hub = hubFor();
    try {
      const ollamaDown = await scanLocalResources(hub, {
        hosts: hostsOf(),
        probes: probesFor({ objectInfo, workflows: { 'saved/t2i.json': textToImage } }),
      });
      assert.equal(ollamaDown.resources.filter((resource) => resource.kind === 'ollama_model').length, 0);
      assert.equal(ollamaDown.resources.filter((resource) => resource.kind === 'comfyui_workflow').length, 1);
      assert.ok(ollamaDown.sources.some((source) => source.kind === 'ollama_model' && !source.ok));

      const comfyDown = await scanLocalResources(hub, {
        hosts: hostsOf(),
        workflowDir: '/workflows/comfy',
        probes: probesFor({
          ollama: ollamaTags,
          files: { '/workflows/comfy/local.json': JSON.stringify(textToImage) },
        }),
      });
      assert.equal(comfyDown.resources.filter((resource) => resource.kind === 'ollama_model').length, 2);
      const local = comfyDown.resources.filter((resource) => resource.kind === 'comfyui_workflow');
      assert.equal(local.length, 1, 'a directory workflow is still found while ComfyUI is down');
      assert.equal(local[0]?.status, 'ready', 'an API workflow needs no engine to be understood');
      assert.ok(comfyDown.warnings.some((warning) => /object_info/.test(warning)));
    } finally {
      await hub.dispose();
    }
  });

  it('publishes runnable workflows as providers the hub can route', async () => {
    const hub = hubFor();
    try {
      const result = await scanLocalResources(hub, {
        hosts: hostsOf(),
        probes: probesFor({ ollama: ollamaTags, objectInfo, workflows: { 'saved/t2i.json': textToImage } }),
      });
      assert.equal(result.registered, 1);

      const view = hub.listModels({ includeDisabled: true }).find((entry) => entry.model.id === 'comfy-wf-t2i');
      assert.ok(view, 'the scanned workflow is a registered provider');
      assert.equal(view.model.providerKind, 'workflow');
      assert.equal(view.model.workflowId, 'comfy-wf-t2i');
      assert.deepEqual([...view.model.capabilities], ['text_to_image']);

      const decision = await hub.route({ capability: 'text_to_image', prompt: 'a fox' });
      assert.equal(decision.modelId, 'comfy-wf-t2i');
    } finally {
      await hub.dispose();
    }
  });

  it('runs a scanned workflow end to end through the existing ComfyUI harness', async () => {
    const png = renderMockPng(24, 16, 'scan');
    const requests: string[] = [];
    const http = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => {
        const url = req.url ?? '/';
        requests.push(url);
        if (url.startsWith('/view')) {
          res.writeHead(200, { 'content-type': 'image/png' });
          res.end(Buffer.from(png));
          return;
        }
        const body =
          url === '/object_info'
            ? objectInfo
            : url === '/system_stats'
              ? { system: {} }
              : url.startsWith('/userdata?')
                ? ['t2i.json']
                : url === '/userdata/t2i.json'
                  ? textToImage
                  : url === '/prompt'
                    ? { prompt_id: 'p-1' }
                    : url.startsWith('/history/')
                      ? {
                          'p-1': {
                            status: { status_str: 'success' },
                            outputs: { '8': { images: [{ filename: 'out.png', subfolder: '', type: 'output' }] } },
                          },
                        }
                      : [];
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify(body));
      });
    });
    await new Promise<void>((resolve) => http.listen(0, '127.0.0.1', resolve));
    const address = http.address();
    assert.ok(address !== null && typeof address !== 'string');
    const endpoint = `http://127.0.0.1:${address.port}`;

    const hub = hubFor(endpoint);
    try {
      // Only the HTTP effect is stubbed, and only to keep Ollama out of the test:
      // ComfyUI is reached over the real socket, through the fake server above.
      const scan = await scanLocalResources(hub, {
        hosts: hostsOf(endpoint),
        timeoutMs: 2_000,
        probes: {
          fetchJson: async (url, timeoutMs) => {
            if (url.includes('/api/tags')) throw new Error('no Ollama in this test');
            const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
          },
        },
      });
      const workflow = scan.resources.find((resource) => resource.kind === 'comfyui_workflow');
      assert.equal(workflow?.status, 'ready');
      assert.ok(workflow?.modelId);

      const result = await hub.invokeModel({
        capability: 'text_to_image',
        prompt: 'a futuristic city',
        options: { width: 256, height: 128 },
        modelId: workflow?.modelId ?? '',
      });
      assert.equal(result.modelId, workflow?.modelId);
      assert.equal(result.outputs.length, 1);
      assert.equal(result.outputs[0]?.type, 'image');
      assert.ok(requests.includes('/prompt'), 'the complete workflow was queued on ComfyUI');
    } finally {
      await hub.dispose();
      await new Promise<void>((resolve) => http.close(() => resolve()));
    }
  });

  it('surfaces both kinds through the inventory the settings page reads', async () => {
    const hub = hubFor();
    try {
      const inventory = await buildInventory(hub, {
        hosts: hostsOf(),
        catalogPath: '/catalog/models.json',
        artifactRoot: '/artifacts',
        allowProcessLaunch: false,
        probes: probesFor({ ollama: ollamaTags, objectInfo, workflows: { 'saved/t2i.json': textToImage } }),
        scanTimeoutMs: 1_000,
        log: () => {},
      });

      assert.deepEqual(
        inventory.resources.map((resource) => resource.typeLabel),
        ['Ollama Model', 'Ollama Model', 'ComfyUI Workflow'],
      );
      assert.equal(inventory.scan.registered, 1);
      assert.ok(inventory.scan.sources.length >= 2);
      assert.doesNotMatch(JSON.stringify(inventory.resources), /safetensors/);
      for (const model of inventory.models) {
        assert.doesNotMatch(model.id, /safetensors/, `${model.id} must not be a weight file`);
      }
    } finally {
      await hub.dispose();
    }
  });

  it('reports an empty machine as empty rather than as an error', async () => {
    const hub = hubFor();
    try {
      const result = await scanLocalResources(hub, {
        hosts: hostsOf(),
        probes: probesFor({ ollama: { models: [] }, objectInfo, workflows: {} }),
      });
      assert.deepEqual(result.resources, []);
      assert.equal(result.registered, 0);
      assert.ok(result.sources.every((source) => source.ok), 'both engines answered');
    } finally {
      await hub.dispose();
    }
  });
});
