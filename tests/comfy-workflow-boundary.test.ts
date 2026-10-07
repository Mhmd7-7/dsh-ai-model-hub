/** Configured ComfyUI workflow providers are the public boundary, not weight files. */
import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { ServerResponse } from 'node:http';
import { describe, it } from 'node:test';
import { ModelHub, renderMockPng } from '../src/index.ts';
import type { ModelCatalogConfig, MachineProfile } from '../src/index.ts';

const png = renderMockPng(32, 24, 'configured-workflow');
const info = {
  CheckpointLoaderSimple: { input: { required: { ckpt_name: [['A.safetensors', 'B.safetensors'], {}] } } },
  LoraLoader: { input: { required: { lora_name: [['C.safetensors'], {}] } } },
  VAELoader: { input: { required: { vae_name: [['VAE.safetensors'], {}] } } },
  CLIPTextEncode: { input: { required: { text: ['STRING', {}] } } },
  EmptyLatentImage: { input: { required: { width: ['INT', {}], height: ['INT', {}] } } },
  SaveImage: { input: { required: { images: ['IMAGE', {}] } }, output_node: true },
  LoadImage: { input: { required: { image: ['STRING', { image_upload: true }] } }, output: ['IMAGE'] },
  SaveGLB: { input: { required: { mesh: ['MESH', {}] } }, output_node: true },
};
const graph = {
  '17': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'A.safetensors' } },
  '23': { class_type: 'CLIPTextEncode', inputs: { text: 'template' } },
  '31': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512 } },
  '81': { class_type: 'SaveImage', inputs: { images: ['31', 0] } },
};
const bindings = { prompt: { node: '23', input: 'text' }, width: { node: '31', input: 'width' } };
const outputs = { image: { node: '81', type: 'image' } };
function glbBytes(): Uint8Array {
  const json = JSON.stringify({ asset: { version: '2.0' }, accessors: [{ count: 8 }] });
  const padded = json.padEnd(Math.ceil(json.length / 4) * 4, ' ');
  const bytes = new Uint8Array(20 + padded.length);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x46546c67, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, bytes.length, true);
  view.setUint32(12, padded.length, true);
  view.setUint32(16, 0x4e4f534a, true);
  for (let index = 0; index < padded.length; index += 1) bytes[20 + index] = padded.charCodeAt(index);
  return bytes;
}
const editGraph = {
  ...graph,
  '19': { class_type: 'LoadImage', inputs: { image: 'template.png' } },
};
function workflow(id: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return { id, name: `Workflow ${id}`, capabilities: ['text_to_image'], workflow: graph, bindings, outputs, ...extra };
}
function catalog(endpoint: string, workflows: Record<string, unknown>[]): ModelCatalogConfig {
  return { version: '1', models: [], hosts: [{ id: 'comfyui', name: 'ComfyUI fixture', adapter: 'comfyui', runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint, path: '/prompt' }, adapterConfig: { workflows } }] };
}
const machine: MachineProfile = { hasGpu: true, vramGb: 16, ramGb: 32, availableVramGb: 16, availableRamGb: 32, notes: 'fixture' };
async function server(respond?: (url: string, response: ServerResponse) => boolean) {
  const requests: { url: string; body: string }[] = [];
  const http = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      const url = req.url ?? '/';
      const body = Buffer.concat(chunks).toString('utf8');
      requests.push({ url, body });
      if (respond?.(url, res)) return;
      if (url === '/view?filename=done.png&subfolder=&type=output' || url.startsWith('/view?')) {
        res.writeHead(200, { 'content-type': 'image/png' }); res.end(Buffer.from(png)); return;
      }
      const value = url === '/object_info' ? info
        : url === '/system_stats' ? { system: { comfyui_version: 'test' } }
        : url === '/upload/image' ? { name: 'engine-upload.png', subfolder: '', type: 'input' }
        : url === '/prompt' ? { prompt_id: 'p-1' }
        : url.startsWith('/history/') ? { 'p-1': { status: { status_str: 'success', completed: true }, outputs: {
          '17': { images: [{ filename: 'wrong.png', subfolder: '', type: 'output' }] },
          '81': { images: [{ filename: 'done.png', subfolder: '', type: 'output' }] },
        } } } : [];
      res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(value));
    });
  });
  await new Promise<void>((resolve) => http.listen(0, '127.0.0.1', resolve));
  const address = http.address();
  assert.ok(address && typeof address !== 'string');
  return { endpoint: `http://127.0.0.1:${address.port}`, requests, close: () => new Promise<void>((resolve) => http.close(() => resolve())) };
}
async function discover(config: ModelCatalogConfig, resources: MachineProfile = machine) {
  return ModelHub.fromConfigAndDiscovery(config, { manageTimers: false, discoveryTimeoutMs: 1000, probeResources: false, machine: resources, log: () => {} });
}

describe('configured ComfyUI workflow boundary', () => {
  it('publishes only configured workflows, never checkpoints, LoRAs or VAEs', async () => {
    const fake = await server();
    try {
      const { hub, discovery } = await discover(catalog(fake.endpoint, [workflow('first')]));
      try {
        assert.deepEqual(discovery.descriptors.map((model) => model.id), ['first']);
        assert.deepEqual(hub.listModels({ includeDisabled: true }).map((entry) => entry.model.id), ['first']);
        assert.ok(!JSON.stringify(hub.listModels({ includeDisabled: true })).match(/A\.safetensors|B\.safetensors|C\.safetensors|VAE\.safetensors/));
        assert.equal(fake.requests.filter((entry) => entry.url === '/object_info').length, 1);
      } finally { await hub.dispose(); }
    } finally { await fake.close(); }
  });

  it('rejects missing files, malformed JSON, missing binding nodes/inputs, and output nodes', async () => {
    const fake = await server();
    const root = await mkdtemp(join(tmpdir(), 'comfy-boundary-'));
    try {
      const bad = join(root, 'bad.json');
      await writeFile(bad, '{ invalid', 'utf8');
      const cases: [Record<string, unknown>, RegExp][] = [
        [workflow('missing', { workflow: undefined, workflowPath: join(root, 'absent.json') }), /could not be read/],
        [workflow('malformed', { workflow: undefined, workflowPath: bad }), /invalid JSON/],
        [workflow('missing-node', { bindings: { prompt: { node: '404', input: 'text' } } }), /binding "prompt" references missing node "404"/],
        [workflow('missing-input', { bindings: { prompt: { node: '23', input: 'absent' } } }), /binding "prompt" references missing input "absent"/],
        [workflow('missing-output', { outputs: { image: { node: '404', type: 'image' } } }), /output "image" references missing node "404"/],
      ];
      for (const [entry, expected] of cases) {
        const { hub, discovery } = await discover(catalog(fake.endpoint, [entry]));
        try {
          assert.deepEqual(hub.listModels({ includeDisabled: true }), []);
          assert.ok(discovery.warnings.some((warning) => expected.test(warning.message)), `${entry['id']}: ${discovery.warnings.map((warning) => warning.message).join(' | ')}`);
        } finally { await hub.dispose(); }
      }
    } finally { await rm(root, { recursive: true, force: true }); await fake.close(); }
  });

  it('contains an unavailable host as a discovery warning and no model', async () => {
    const fake = await server();
    const endpoint = fake.endpoint;
    await fake.close();
    const { hub, discovery } = await discover(catalog(endpoint, [workflow('unreachable')]));
    try {
      assert.deepEqual(hub.listModels({ includeDisabled: true }), []);
      assert.match(discovery.warnings[0]?.message ?? '', /could not read \/object_info/);
    } finally { await hub.dispose(); }
  });

  it('routes two workflows by priority subject to available VRAM', async () => {
    const fake = await server();
    try {
      const { hub } = await discover(catalog(fake.endpoint, [workflow('heavy', { priority: 1, vramGb: 32 }), workflow('light', { priority: 20, vramGb: 4 })]));
      try {
        assert.deepEqual(hub.listModels({ includeDisabled: true }).map((entry) => entry.model.id), ['heavy', 'light']);
        assert.equal((await hub.route({ capability: 'text_to_image', prompt: 'test' })).modelId, 'light');
        const decision = await hub.route({ capability: 'text_to_image', prompt: 'test', modelId: 'light' });
        assert.equal(decision.modelId, 'light');
        assert.equal(decision.candidates.find((item) => item.modelId === 'heavy')?.eligible, false);
      } finally { await hub.dispose(); }
      const { hub: larger } = await discover(catalog(fake.endpoint, [workflow('heavy', { priority: 1, vramGb: 32 }), workflow('light', { priority: 20, vramGb: 4 })]), { ...machine, vramGb: 48, availableVramGb: 48 });
      try {
        assert.equal((await larger.route({ capability: 'text_to_image', prompt: 'test' })).modelId, 'heavy');
      } finally { await larger.dispose(); }
    } finally { await fake.close(); }
  });

  it('registers image_to_image and binds the uploaded artifact filename, not a local path', async () => {
    const fake = await server();
    try {
      const editable = workflow('editor', {
        capabilities: ['image_to_image'], workflow: editGraph,
        bindings: { ...bindings, image: { node: '19', input: 'image' } },
      });
      const { hub } = await discover(catalog(fake.endpoint, [editable]));
      try {
        assert.deepEqual(hub.catalog.findModelsByCapability('image_to_image').map((model) => model.id), ['editor']);
        const source = await hub.artifacts.put({ type: 'image', bytes: png, mimeType: 'image/png', extension: '.png' });
        const result = await hub.invokeModel({ capability: 'image_to_image', prompt: 'retouch this image', inputs: [source.id] });
        assert.equal(result.modelId, 'editor');
        assert.equal(result.outputs[0]?.type, 'image');
        const upload = fake.requests.find((entry) => entry.url === '/upload/image');
        assert.ok(upload, 'the artifact must be uploaded to ComfyUI');
        assert.match(upload.body, /filename="[^"]+\.png"/);
        const queued = fake.requests.find((entry) => entry.url === '/prompt');
        assert.ok(queued);
        const submitted = JSON.parse(queued.body) as { prompt: typeof editGraph };
        assert.equal(submitted.prompt['19'].inputs.image, 'engine-upload.png');
        assert.equal(submitted.prompt['23'].inputs.text, 'retouch this image');
        assert.equal(editGraph['19'].inputs.image, 'template.png');
      } finally { await hub.dispose(); }
    } finally { await fake.close(); }
  });

  it('returns the declared 3D mesh and companion preview but ignores undeclared node output', async () => {
    const fake = await server((url, response) => {
      if (url.startsWith('/history/')) {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ 'p-1': { status: { status_str: 'success', completed: true }, outputs: {
          '17': { images: [{ filename: 'unrelated.png', subfolder: '', type: 'output' }] },
          '81': { images: [{ filename: 'preview.png', subfolder: '', type: 'output' }] },
          '90': { '3d': [{ filename: 'mesh.glb', subfolder: '', type: 'output' }] },
        } } }));
        return true;
      }
      if (url.startsWith('/view?')) {
        const isMesh = url.includes('filename=mesh.glb');
        response.writeHead(200, { 'content-type': isMesh ? 'model/gltf-binary' : 'image/png' });
        response.end(Buffer.from(isMesh ? glbBytes() : png));
        return true;
      }
      return false;
    });
    try {
      const threeD = workflow('mesh-and-preview', {
        capabilities: ['image_to_3d'],
        workflow: { ...editGraph, '90': { class_type: 'SaveGLB', inputs: { mesh: ['19', 0] } } },
        bindings: { prompt: { node: '23', input: 'text' }, image: { node: '19', input: 'image' } },
        outputs: { model_3d: { node: '90', type: 'model_3d' }, image: { node: '81', type: 'image' } },
      });
      const { hub, discovery } = await discover(catalog(fake.endpoint, [threeD]));
      try {
        assert.deepEqual(discovery.warnings, [], `3D registration failed: ${discovery.warnings.map((item) => item.message).join(' | ')}`);
        const source = await hub.artifacts.put({ type: 'image', bytes: png, mimeType: 'image/png', extension: '.png' });
        const result = await hub.invokeModel({ capability: 'image_to_3d', inputs: [source.id] });
        assert.deepEqual(result.outputs.map((item) => item.type), ['model_3d', 'image']);
        assert.equal(result.outputs[0]?.metadata['format'], 'glb');
        assert.ok(fake.requests.some((item) => item.url.includes('filename=mesh.glb')));
        assert.ok(fake.requests.some((item) => item.url.includes('filename=preview.png')));
        assert.ok(!fake.requests.some((item) => item.url.includes('filename=unrelated.png')));
      } finally { await hub.dispose(); }
    } finally { await fake.close(); }
  });

  it('never maps class_type or weight filenames from invocation options into its graph', async () => {
    const fake = await server();
    try {
      const { hub } = await discover(catalog(fake.endpoint, [workflow('locked')]));
      try {
        const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'safe', options: {
          class_type: 'UnauthorizedNode', ckpt_name: 'B.safetensors', checkpoint: 'B.safetensors',
          '17.ckpt_name': 'B.safetensors', workflow: { '17': { class_type: 'UnauthorizedNode' } },
        } });
        assert.equal(result.outputs.length, 1);
        const queued = fake.requests.find((entry) => entry.url === '/prompt');
        assert.ok(queued);
        const submitted = JSON.parse(queued.body) as { prompt: typeof graph };
        assert.equal(submitted.prompt['17'].class_type, 'CheckpointLoaderSimple');
        assert.equal(submitted.prompt['17'].inputs.ckpt_name, 'A.safetensors');
        assert.equal(submitted.prompt['23'].inputs.text, 'safe');
        assert.ok(!JSON.stringify(submitted).includes('UnauthorizedNode'));
        assert.ok(!JSON.stringify(submitted).includes('B.safetensors'));
      } finally { await hub.dispose(); }
    } finally { await fake.close(); }
  });

  it('binds only public inputs and extracts declared output node from mocked HTTP', async () => {
    const fake = await server();
    try {
      const { hub } = await discover(catalog(fake.endpoint, [workflow('runner', { priority: 1 })]));
      try {
        const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a glowing fox', options: { width: 256 } });
        assert.equal(result.modelId, 'runner');
        assert.equal(result.outputs.length, 1);
        const queued = fake.requests.find((entry) => entry.url === '/prompt');
        assert.ok(queued);
        const submitted = JSON.parse(queued.body) as { prompt: typeof graph };
        assert.equal(submitted.prompt['23'].inputs.text, 'a glowing fox');
        assert.equal(submitted.prompt['31'].inputs.width, 256);
        assert.equal(graph['23'].inputs.text, 'template', 'shared graph remains immutable');
        assert.ok(fake.requests.some((entry) => entry.url.includes('filename=done.png')));
        assert.ok(!fake.requests.some((entry) => entry.url.includes('filename=wrong.png')));
      } finally { await hub.dispose(); }
    } finally { await fake.close(); }
  });
});
