/**
 * Tests for ComfyUI **workflow** discovery and 3D execution.
 *
 * Everything here exists to prove one property: the hub decides what ComfyUI can
 * do from the *graph*, and DSH never learns a workflow name, a node class, a
 * checkpoint filename, or a node id. So the assertions are about capabilities
 * inferred from structure, about the absence of capabilities that are not proven,
 * and about the artifact that comes back — never about a hardcoded workflow.
 *
 * The fixtures are deliberately shaped like real ComfyUI 0.34 `object_info` and
 * `/userdata` payloads, because the aspects being tested *are* shape: which port
 * types a node declares, whether a link reaches, and which output key a node
 * writes its files under. The workflows use invented filenames and a synthetic
 * generator node, so nothing here can become a production model list by accident.
 *
 * @module dsh-ai-model-hub/tests/discovery-comfyui-workflow
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { IncomingHttpHeaders, ServerResponse } from 'node:http';
import { describe, it } from 'node:test';

import { createComfyUiAdapter, readComfyFileRefs } from '../src/adapters/comfyui.ts';
import { sniffThreeDFormat } from '../src/artifacts/formats.ts';
import type { ModelCatalogConfig, ModelHost } from '../src/catalog/descriptor.ts';
import { createComfyUiDiscoverer } from '../src/discovery/comfyui.ts';
import type { MachineProfile } from '../src/types.ts';
import { ModelHub } from '../src/hub.ts';
import {
  classifyComfyNode,
  estimateComfyWorkflowVram,
  inferWorkflowCapabilities,
  isThreeDFilename,
  mapComfyWorkflow,
  parseComfyWorkflow,
  parseComfyWorkflowOverrides,
  parseWorkflowList,
  readComfyNodeIo,
} from '../src/discovery/comfyui-workflow.ts';

// ── fixtures ────────────────────────────────────────────────────────────────

/** One captured request. */
interface CapturedRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: IncomingHttpHeaders;
  readonly body: string;
}

/** A test double for a ComfyUI server. */
interface FakeComfy {
  readonly url: string;
  readonly requests: CapturedRequest[];
  setResponder(responder: (request: CapturedRequest, response: ServerResponse) => void): void;
  close(): Promise<void>;
}

/** Start a local ComfyUI double on an ephemeral port. */
async function startComfy(): Promise<FakeComfy> {
  let responder: ((request: CapturedRequest, response: ServerResponse) => void) | undefined;
  const requests: CapturedRequest[] = [];
  const server = createServer((request, response) => {
    const chunks: Buffer[] = [];
    request.on('data', (chunk: Buffer) => chunks.push(chunk));
    request.on('end', () => {
      const captured: CapturedRequest = {
        method: request.method ?? 'GET',
        url: request.url ?? '/',
        headers: request.headers,
        body: Buffer.concat(chunks).toString('utf8'),
      };
      requests.push(captured);
      if (responder === undefined) {
        response.writeHead(404, { 'content-type': 'application/json' });
        response.end('{}');
        return;
      }
      responder(captured, response);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    setResponder: (next) => {
      responder = next;
    },
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

/**
 * An enumerable file input, as ComfyUI documents it: `[[options], opts]`.
 *
 * This nesting matters, and getting it wrong is a real bug rather than a fixture
 * detail: a *typed* port is `["IMAGE", {}]` (a string first) while a file
 * enumeration is `[["a.safetensors"], {}]` (an array first), and the classifier
 * reads that difference to know whether a field carries a port type at all.
 */
function fileEnum(options: readonly string[]): unknown[] {
  return [options, {}];
}

/**
 * The nodes a real 3D install declares, reduced to what detection reads.
 *
 * The port types are the load-bearing part: a generator is a node that *returns*
 * `MESH`, and a writer is a node that *takes* a `MESH` and writes a `File3D*`.
 * Getting either wrong would make the inference below meaningless, so these mirror
 * the real schemas rather than a convenient fiction.
 */
function threeDObjectInfo(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return { ...threeDObjectInfoCore(), ...overrides };
}

/**
 * A UI-format export whose graph names a node class this install does not have.
 *
 * This is the case that cannot be converted faithfully: the editor's
 * `widgets_values` are positional, so a class whose schema is unknown has no
 * decodable argument order, and guessing would silently wire the wrong values.
 * Discovery reports it as unreadable instead.
 */
function uiExportWithMissingPack(): { nodes: unknown[]; links: unknown[] } {
  return {
    nodes: [
      { id: 1, type: 'LoadImage', inputs: [], widgets_values: ['example_input.png', 'image'] },
      { id: 2, type: 'NodeFromAPackThatIsNotInstalled', inputs: [], widgets_values: ['x'] },
      { id: 3, type: 'SaveGLB', inputs: [], widgets_values: ['3d/test'] },
    ],
    links: [],
  };
}

/**
 * The same install with no checkpoint *file*.
 *
 * Discovery publishes one descriptor per checkpoint as well as one per workflow, so
 * a fixture with a checkpoint file makes every "what was published" assertion count
 * a weight model the test is not about. The loader node itself stays, with an empty
 * enumeration: a graph that wires `CheckpointLoaderSimple` still needs the class to
 * exist, and an empty enum is exactly what a class with no weights on disk reports.
 */
function threeDObjectInfoCore(): Record<string, unknown> {
  return {
    LoadImage: {
      input: { required: { image: ['STRING', { image_upload: true }], upload: ['IMAGEUPLOAD', {}] } },
      output: ['IMAGE', 'MASK'],
      output_name: ['IMAGE', 'MASK'],
    },
    // The checkpoint the workflow graphs name. `threeDObjectInfo` replaces this row
    // with the same enumeration, so both fixtures describe one install; the name is
    // deliberately not a real model, only a fixture value.
    CheckpointLoaderSimple: { input: { required: { ckpt_name: fileEnum(['not-installed.safetensors']) } }, output: ['MODEL', 'CLIP', 'VAE'] },
    CLIPTextEncode: { input: { required: { clip: ['CLIP', {}], text: ['STRING', { multiline: true }] } }, output: ['CONDITIONING'] },
    EmptyLatentImage: { input: { required: { width: ['INT', {}], height: ['INT', {}], batch_size: ['INT', {}] } }, output: ['LATENT'] },
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
          sampler_name: [['euler'], {}],
          scheduler: [['normal'], {}],
          denoise: ['FLOAT', {}],
        },
      },
      output: ['LATENT'],
    },
    VAEDecode: { input: { required: { samples: ['LATENT', {}], vae: ['VAE', {}] } }, output: ['IMAGE'] },
    SaveImage: { input: { required: { images: ['IMAGE', {}], filename_prefix: ['STRING', {}] } }, output: [], output_node: true },
    // The hardware-free stand-in for the real generator/decoder pair — TRELLIS.2's
    // shape decoder, Hunyuan3D's voxel-to-mesh, Stable Fast 3D's transformer. Each
    // of those takes a latent/voxel/image *and* a mesh and returns a refined mesh,
    // and this declares exactly that contract, so what is tested here is the rule
    // rather than a convenient shape. It is named so that nothing about it can
    // become a production model entry.
    TestShapeDecoder: {
      input: {
        required: {
          positive: ['CONDITIONING', {}],
          latent_image: ['LATENT', {}],
          mesh: ['MESH', {}],
          image: ['IMAGE', {}],
        },
      },
      output: ['MESH'],
      output_name: ['mesh'],
    },
    // The text-driven counterpart. It takes no image port at all, which is what
    // makes a text-only workflow provable as text-only rather than as an
    // image workflow that happens to carry a prompt.
    TestTextTo3DDecoder: {
      input: { required: { positive: ['CONDITIONING', {}], latent_image: ['LATENT', {}], mesh: ['MESH', {}] } },
      output: ['MESH'],
      output_name: ['mesh'],
    },
    // Post-processing: consumes a mesh and returns one. Must never count as
    // generation, which is the false positive this classification guards.
    DecimateMesh: { input: { required: { mesh: ['MESH', {}], ratio: ['FLOAT', {}] } }, output: ['MESH'] },
    // A writer: takes a mesh and a filename prefix, persists it. This is what makes
    // 3D output a deliverable rather than an in-memory mesh.
    SaveGLB: { input: { required: { mesh: ['MESH', {}], filename_prefix: ['STRING', {}] } }, output: [], output_node: true },
    Save3DAdvanced: {
      input: { required: { model_3d: [['File3DGLB', 'File3DOBJ'], {}], filename_prefix: ['STRING', {}] } },
      output: ['File3DAny'],
      output_node: true,
    },
    MeshToFile3D: { input: { required: { mesh: ['MESH', {}] } }, output: ['File3DGLB'] },
    // A preview: writes to temp, so it is not a deliverable.
    Preview3DAdvanced: { input: { required: { model_3d: [['File3DGLB'], {}], camera_info: ['LOAD3DCAMERA', {}] } }, output: [], output_node: true },
  };
}

/** An API-format graph in the shape the ComfyUI editor exports. */
function apiWorkflow(graph: Record<string, unknown>): Record<string, unknown> {
  return graph;
}

/**
 * The two synthetic generator classes the fixtures use.
 *
 * Both declare the contract every real 3D generator declares — a latent and a mesh
 * in, a mesh out — and both are named so that nothing about them can be mistaken
 * for a production model entry.
 */
const DECODER = 'TestShapeDecoder';

/**
 * An image-driven 3D workflow.
 *
 * `LoadImage → conditioning → <decoder> → DecimateMesh → SaveGLB`, wired so every
 * path is followed rather than assumed. The checkpoint loader is optional because a
 * fixture declaring one also makes discovery publish a weight-file model, and
 * several tests deliberately count models without one.
 */
function imageTo3dWorkflow(options: { readonly withLoader?: boolean; readonly withImage?: boolean } = {}): Record<string, unknown> {
  const graph: Record<string, unknown> = {
    '2': { class_type: 'LoadImage', inputs: { image: 'example_input.png', upload: 'image' } },
  };
  if (options.withLoader !== false) {
    graph['1'] = { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'not-installed.safetensors' } };
  }
  graph['3'] = { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: 'a small blue robot' } };
  graph['4'] = {
    class_type: 'KSampler',
    inputs: {
      model: ['1', 0],
      positive: ['3', 0],
      negative: ['3', 0],
      latent_image: ['5', 0],
      seed: 0,
      steps: 12,
      cfg: 4,
      sampler_name: 'euler',
      scheduler: 'normal',
      denoise: 1,
    },
  };
  graph['5'] = { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } };
  graph['6'] = {
    class_type: DECODER,
    inputs: {
      positive: ['3', 0],
      latent_image: ['4', 0],
      mesh: ['1', 0],
      // `withImage: false` drops the image link, which is how a *text-only* graph
      // built from the same nodes proves `text_to_3d` and nothing else. The
      // `LoadImage` node stays in the graph as an island, so the difference under
      // test is the wiring and not the node's presence.
      ...(options.withImage === false ? {} : { image: ['2', 0] }),
    },
  };
  graph['7'] = { class_type: 'DecimateMesh', inputs: { mesh: ['6', 0], ratio: 0.5 } };
  graph['8'] = { class_type: 'SaveGLB', inputs: { mesh: ['7', 0], filename_prefix: '3d/test' } };
  return apiWorkflow(graph);
}

/**
 * A text-driven 3D workflow.
 *
 * No image loader at all: the prompt is the only input, and it reaches the decoder
 * through the conditioning the decoder reads. This is the shape that must advertise
 * `text_to_3d` and must *not* advertise `image_to_3d`.
 */
function textTo3dWorkflow(): Record<string, unknown> {
  return apiWorkflow({
    '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'not-installed.safetensors' } },
    '2': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: '' } },
    '3': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '4': { class_type: DECODER, inputs: { positive: ['2', 0], latent_image: ['3', 0], mesh: ['1', 0] } },
    '5': { class_type: 'SaveGLB', inputs: { mesh: ['4', 0], filename_prefix: '3d/test' } },
  });
}

/** A workflow with 3D machinery that only ever previews it. */
function previewOnlyWorkflow(): Record<string, unknown> {
  return apiWorkflow({
    '1': { class_type: 'LoadImage', inputs: { image: 'example_input.png' } },
    '2': { class_type: 'TestShapeDecoder', inputs: { positive: ['1', 0], latent_image: ['1', 0], mesh: ['1', 0] } },
    '3': { class_type: 'Preview3DAdvanced', inputs: { model_3d: ['2', 0], camera_info: '{}' } },
  });
}

/** A mesh post-processing workflow: no generator anywhere in it. */
function postProcessOnlyWorkflow(): Record<string, unknown> {
  return apiWorkflow({
    '1': { class_type: 'LoadImage', inputs: { image: 'example_input.png' } },
    '2': { class_type: 'MeshToFile3D', inputs: { mesh: ['1', 0] } },
    '3': { class_type: 'Save3DAdvanced', inputs: { model_3d: ['2', 0], filename_prefix: '3d/test' } },
  });
}

/** An image-to-image workflow with no 3D content: the control case. */
function imageWorkflow(): Record<string, unknown> {
  return apiWorkflow({
    '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'not-installed.safetensors' } },
    '2': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: '' } },
    '3': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '4': {
      class_type: 'KSampler',
      inputs: { model: ['1', 0], positive: ['2', 0], negative: ['2', 0], latent_image: ['3', 0], seed: 0, steps: 8, cfg: 1, sampler_name: 'euler', scheduler: 'normal', denoise: 1 },
    },
    '5': { class_type: 'VAEDecode', inputs: { samples: ['4', 0], vae: ['1', 2] } },
    '6': { class_type: 'SaveImage', inputs: { images: ['5', 0], filename_prefix: 'img/test' } },
  });
}

/**
 * A UI-format export of the image-to-3D workflow.
 *
 * The conversion path is what most real installs need, because ComfyUI's editor
 * saves this shape by default. Widget values are positional and the sampler's
 * `seed` is deliberately out of schema order, so a passing conversion proves the
 * override table is doing its job.
 */
function uiFormatWorkflow(): { nodes: unknown[]; links: unknown[] } {
  return {
    nodes: [
      {
        id: 1,
        type: 'CheckpointLoaderSimple',
        inputs: [],
        widgets_values: ['sd15-instruct.safetensors'],
      },
      { id: 2, type: 'LoadImage', inputs: [], widgets_values: ['example_input.png', 'image'] },
      {
        id: 3,
        type: 'CLIPTextEncode',
        inputs: [{ name: 'clip', type: 'CLIP', link: 1 }],
        widgets_values: ['a small blue robot'],
      },
      {
        id: 4,
        type: 'KSampler',
        inputs: [
          { name: 'model', type: 'MODEL', link: 2 },
          { name: 'positive', type: 'CONDITIONING', link: 3 },
          { name: 'negative', type: 'CONDITIONING', link: 3 },
          { name: 'latent_image', type: 'LATENT', link: 4 },
        ],
        widgets_values: [12345, 'fixed', 12, 4, 'euler', 'normal', 1],
      },
      { id: 5, type: 'EmptyLatentImage', inputs: [], widgets_values: [512, 512, 1] },
      {
        id: 6,
        type: 'TestShapeDecoder',
        inputs: [
          { name: 'positive', type: 'CONDITIONING', link: 3 },
          { name: 'latent_image', type: 'LATENT', link: 5 },
        ],
        widgets_values: [],
      },
      { id: 7, type: 'SaveGLB', inputs: [{ name: 'mesh', type: 'MESH', link: 6 }], widgets_values: ['3d/test'] },
    ],
    links: [
      [1, 1, 1, 3, 0, 'CLIP'],
      [2, 1, 0, 4, 0, 'MODEL'],
      [3, 3, 0, 4, 1, 'CONDITIONING'],
      [4, 5, 0, 4, 3, 'LATENT'],
      [5, 4, 0, 6, 1, 'LATENT'],
      [6, 6, 0, 7, 0, 'MESH'],
    ],
  };
}

/** Register a fake server's URL as a ComfyUI host. */
function comfyHost(endpoint: string, adapterConfig?: Record<string, unknown>): ModelHost {
  return {
    id: 'comfyui',
    name: 'ComfyUI',
    adapter: 'comfyui',
    runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint, path: '/prompt' },
    ...(adapterConfig === undefined ? {} : { adapterConfig }),
  };
}

/** A static catalog that declares only the host, so discovery supplies the models. */
function hostOnlyCatalog(endpoint: string, adapterConfig?: Record<string, unknown>): ModelCatalogConfig {
  return { version: '1', hosts: [comfyHost(endpoint, adapterConfig)], models: [] };
}

/** A machine with a named amount of VRAM, for resource-fit tests. */
function machine(vramGb: number): MachineProfile {
  return { vramGb, ramGb: 64, hasGpu: true, notes: 'test', availableVramGb: vramGb, availableRamGb: 64 };
}

/**
 * Serve an install: `/object_info`, a workflow listing, and each workflow.
 *
 * @param workflows - name → document, as ComfyUI stores them.
 * @param objectInfo - the node index to answer with.
 * @returns the fake server, already configured.
 */
async function serveInstall(
  workflows: Record<string, unknown>,
  objectInfo: Record<string, unknown>,
): Promise<FakeComfy> {
  const comfy = await startComfy();
  comfy.setResponder((request, response) => {
    const url = request.url;
    if (url.startsWith('/object_info')) {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify(objectInfo));
      return;
    }
    if (url.startsWith('/system_stats')) {
      // The liveness route a real ComfyUI serves, so a cold-start gate sees the
      // engine as up rather than "not running and not startable".
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ system: { comfyui_version: 'test' } }));
      return;
    }
    if (url.startsWith('/userdata?')) {
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify(Object.keys(workflows)));
      return;
    }
    const match = /^\/userdata\/(.+)$/.exec(url);
    if (match !== null && match[1] !== undefined) {
      const name = decodeURIComponent(match[1]);
      const document = workflows[name];
      if (document === undefined) {
        response.writeHead(404, { 'content-type': 'application/json' });
        response.end('{}');
        return;
      }
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify(document));
      return;
    }
    response.writeHead(404, { 'content-type': 'application/json' });
    response.end('{}');
  });
  return comfy;
}

/** A minimal but structurally valid GLB: the 12-byte header plus a JSON chunk. */
function glbBytes(): Uint8Array {
  const json = JSON.stringify({ asset: { version: '2.0' }, accessors: [{ count: 8 }] });
  const padded = json.padEnd(Math.ceil(json.length / 4) * 4, ' ');
  const total = 12 + 8 + padded.length;
  const bytes = new Uint8Array(total);
  const view = new DataView(bytes.buffer);
  view.setUint32(0, 0x46546c67, true); // 'glTF'
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, padded.length, true);
  view.setUint32(16, 0x4e4f534a, true); // 'JSON'
  for (let index = 0; index < padded.length; index += 1) bytes[20 + index] = padded.charCodeAt(index);
  return bytes;
}

// ── reading ─────────────────────────────────────────────────────────────────

describe('comfyui workflow discovery: reading the install', () => {
  it('reads each node class\'s declared IO from /object_info', () => {
    const io = readComfyNodeIo(threeDObjectInfo());
    assert.deepEqual(io['TestShapeDecoder']?.outputs, ['MESH']);
    assert.deepEqual(io['Save3DAdvanced']?.outputs, ['File3DAny']);
    assert.deepEqual(io['LoadImage']?.inputs['image'], ['STRING']);
    assert.deepEqual(io['KSampler']?.inputs['positive'], ['CONDITIONING']);
    // A file enumeration is not a port type, so it is recorded as STRING rather
    // than as the name of a checkpoint.
    assert.deepEqual(io['CheckpointLoaderSimple']?.inputs['ckpt_name'], ['STRING']);
  });

  it('never throws for an /object_info shape it does not recognise', () => {
    for (const raw of [undefined, null, 7, 'nope', []]) {
      assert.deepEqual(readComfyNodeIo(raw), {});
    }
    const io = readComfyNodeIo({ Weird: { input: 'nope' }, AlsoWeird: { input: { required: 'nope' } }, Fine: { input: { required: { a: ['IMAGE', {}] } }, output: ['MESH'] } });
    assert.deepEqual(io['Fine']?.outputs, ['MESH']);
    assert.deepEqual(io['Weird']?.inputs, {});
  });

  it('lists workflow names from /userdata and ignores anything that is not a workflow', () => {
    assert.deepEqual(parseWorkflowList(['a.json', '3d/b.json', 'notes.txt', '/c.json', 'a.json']), [
      { name: 'a.json' },
      { name: '3d/b.json' },
      { name: 'c.json' },
    ]);
    assert.deepEqual(parseWorkflowList({ files: ['x.json'] }), [{ name: 'x.json' }]);
    assert.deepEqual(parseWorkflowList('nope'), []);
  });

  it('accepts an API-format graph, a { prompt } envelope, and a UI export', () => {
    const api = parseComfyWorkflow(imageTo3dWorkflow(), 'a.json');
    assert.equal(api.format, 'api');
    assert.equal(api.nodes['8']?.classType, 'SaveGLB');

    const enveloped = parseComfyWorkflow({ prompt: imageTo3dWorkflow() }, 'a.json');
    assert.equal(enveloped.format, 'api');
    assert.equal(Object.keys(enveloped.nodes).length, 8);

    const io = readComfyNodeIo(threeDObjectInfo());
    const ui = parseComfyWorkflow(uiFormatWorkflow(), 'ui.json', io);
    assert.equal(ui.format, 'ui');
    assert.equal(ui.nodes['2']?.inputs['image'], 'example_input.png');
    // `seed` is out of schema order on KSampler; the override table puts it back.
    assert.equal(ui.nodes['4']?.inputs['seed'], 12345);
    assert.equal(ui.nodes['4']?.inputs['steps'], 12);
    assert.equal(ui.nodes['4']?.inputs['cfg'], 4);
    assert.equal(ui.nodes['4']?.inputs['sampler_name'], 'euler');
    // …and the connected ports came from the links, not the widget list.
    assert.deepEqual(ui.nodes['4']?.inputs['positive'], ['3', 0]);
    assert.match(ui.conversionNote ?? '', /UI\* format/i);
  });

  it('refuses a document that is not a workflow rather than guessing', () => {
    assert.throws(() => parseComfyWorkflow({ hello: 'world' }, 'x.json'), /no nodes/);
    assert.throws(() => parseComfyWorkflow('nope', 'x.json'), /not a JSON object/);
    // A UI export naming a node class the engine does not have cannot be
    // reconstructed faithfully, so it is refused rather than misread.
    assert.throws(
      () => parseComfyWorkflow(uiFormatWorkflow(), 'ui.json', {}),
      /does not have/,
    );
  });
});

// ── capability inference ────────────────────────────────────────────────────

describe('comfyui workflow discovery: what a graph proves', () => {
  const io = readComfyNodeIo(threeDObjectInfo());

  it('infers image_to_3d from an image input feeding a mesh generator that a writer persists', () => {
    const inference = inferWorkflowCapabilities(parseComfyWorkflow(imageTo3dWorkflow(), 'w.json'), io);
    // The fixture conditions its generator on both an image and a prompt, so both
    // directions are honest; the assertions below pin the image half and the
    // island test pins that neither is invented.
    assert.deepEqual(inference.capabilities, ['text_to_3d', 'image_to_3d']);
    assert.ok(inference.generatorClasses.includes('TestShapeDecoder'));
    assert.ok(inference.hasGeometryWriter);
    assert.ok(inference.evidence.some((line) => /3D generation/.test(line)));
    assert.ok(inference.evidence.some((line) => /image input/.test(line)));
  });

  it('infers text_to_3d only when the prompt actually reaches the generator', () => {
    const inference = inferWorkflowCapabilities(parseComfyWorkflow(textTo3dWorkflow(), 'w.json'), io);
    assert.deepEqual(inference.capabilities, ['text_to_3d']);
    assert.ok(!inference.capabilities.includes('image_to_3d'), 'a workflow with no image loader is not image_to_3d');
  });

  it('advertises both directions when the prompt genuinely feeds the generator', () => {
    // The image-to-3D fixture's prompt conditions the sampler the shape decoder
    // reads from, so this graph honestly supports both, in vocabulary order. The
    // negative case is the island test below, which is what stops a stray prompt
    // node from being read as text support.
    const inference = inferWorkflowCapabilities(parseComfyWorkflow(imageTo3dWorkflow(), 'w.json'), io);
    assert.deepEqual(inference.capabilities, ['text_to_3d', 'image_to_3d']);
  });

  it('does not advertise a capability when the prompt is an island', () => {
    // A texture pass with its own prompt node, wired to nothing the shape decoder
    // depends on. `text_to_3d` must not be fabricated from its presence — and the
    // image path, which *is* wired, must survive the check.
    const island = imageTo3dWorkflow();
    island['9'] = { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: 'metal' } };
    island['10'] = {
      class_type: 'KSampler',
      inputs: { model: ['1', 0], positive: ['9', 0], negative: ['9', 0], latent_image: ['5', 0], seed: 0, steps: 4, cfg: 1, sampler_name: 'euler', scheduler: 'normal', denoise: 1 },
    };
    island['11'] = { class_type: 'VAEDecode', inputs: { samples: ['10', 0], vae: ['1', 2] } };
    island['12'] = { class_type: 'SaveImage', inputs: { images: ['11', 0], filename_prefix: 'tex' } };
    // Sever the original prompt's route to the shape decoder entirely: the decoder
    // keeps only the image, so the island's prompt is the only text in the graph
    // and it drives nothing that makes 3D.
    const decoder = island['6'] as { inputs: Record<string, unknown> };
    delete decoder.inputs['positive'];
    delete decoder.inputs['latent_image'];
    const inference = inferWorkflowCapabilities(parseComfyWorkflow(island, 'w.json'), io);
    assert.deepEqual(inference.capabilities, ['image_to_3d']);
    // The note is the useful half: it names *why* text was not claimed.
    assert.ok(
      inference.notes.some((note) => /does not reach the 3D generator/.test(note)),
      `expected a note explaining the withheld capability, got: ${inference.notes.join(' | ')}`,
    );
  });

  it('refuses to call a preview a deliverable', () => {
    const inference = inferWorkflowCapabilities(parseComfyWorkflow(previewOnlyWorkflow(), 'w.json'), io);
    assert.deepEqual(inference.capabilities, []);
    assert.ok(inference.notes.some((note) => /previews 3D output/.test(note)));
  });

  it('refuses to call mesh post-processing a generator', () => {
    const inference = inferWorkflowCapabilities(parseComfyWorkflow(postProcessOnlyWorkflow(), 'w.json'), io);
    assert.deepEqual(inference.capabilities, [], 'decimating and exporting a mesh is not generating one');
    assert.ok(inference.notes.some((note) => /no node that generates mesh geometry/.test(note)));
  });

  it('proves nothing for an ordinary image workflow', () => {
    const inference = inferWorkflowCapabilities(parseComfyWorkflow(imageWorkflow(), 'w.json'), io);
    assert.deepEqual(inference.capabilities, []);
  });

  it('classifies nodes by port type and role, never by model name', () => {
    const generator = classifyComfyNode('TestShapeDecoder', io['TestShapeDecoder']);
    assert.equal(generator.isGeometryGenerator, true, 'a node that consumes geometry and returns geometry it built generates');
    assert.equal(generator.isGeometryWriter, false);
    assert.equal(generator.takesImage, true, 'and its image port is read from the declared types, not its name');

    const writer = classifyComfyNode('SaveGLB', io['SaveGLB']);
    assert.equal(writer.isGeometryWriter, true);
    assert.equal(writer.isGeometryGenerator, false);

    const preview = classifyComfyNode('Preview3DAdvanced', io['Preview3DAdvanced']);
    assert.equal(preview.isGeometryPreview, true);
    assert.equal(preview.isGeometryWriter, false);

    const post = classifyComfyNode('DecimateMesh', io['DecimateMesh']);
    assert.equal(post.isGeometryGenerator, false);
    assert.equal(post.returnsGeometry, true);
    assert.equal(post.processesMesh, true, 'a mesh-to-mesh node hands geometry back, which is post-processing');
    assert.equal(post.consumesMesh, false, 'it does not merely consume geometry — it hands one back');

    // An unknown node class is classified conservatively: no capability.
    const unknown = classifyComfyNode('SomethingNobodyHasSeen', undefined);
    assert.equal(unknown.isGeometryGenerator, false);
    assert.equal(unknown.isGeometryWriter, false);
  });
});

// ── mapping into descriptors ────────────────────────────────────────────────

describe('comfyui workflow discovery: mapping into a descriptor', () => {
  const io = readComfyNodeIo(threeDObjectInfo());
  const host = comfyHost('http://127.0.0.1:8188');

  it('keeps the workflow, its stored name, and its model files as separate identities', () => {
    const workflow = parseComfyWorkflow(imageTo3dWorkflow(), '3d/my_image_to_model.json', io);
    const inference = inferWorkflowCapabilities(workflow, io);
    const mapped = mapComfyWorkflow(workflow, host, inference, { modelFiles: ['not-installed.safetensors'] });
    assert.equal(mapped.descriptor.capabilities.includes('image_to_3d'), true);
    assert.equal(mapped.descriptor.type, 'three_d_generation');
    assert.equal(mapped.descriptor.name, 'my_image_to_model');
    assert.equal(mapped.descriptor.adapterConfig?.['workflow'] !== undefined, true);
    const discovery = mapped.descriptor.adapterConfig?.['discovery'] as Record<string, unknown>;
    assert.equal(discovery['workflowName'], '3d/my_image_to_model.json');
    assert.deepEqual(discovery['modelFiles'], ['not-installed.safetensors'], 'the checkpoint the graph loads, and nothing that is an input');
    assert.deepEqual(discovery['generatorNodes'], ['TestShapeDecoder']);
    assert.deepEqual(mapped.descriptor.inputTypes, ['text', 'image']);
    assert.deepEqual(mapped.descriptor.outputTypes, ['model_3d']);
    assert.equal(mapped.descriptor.enabled, true);
  });

  it('publishes an unprovable workflow disabled, naming why', () => {
    const workflow = parseComfyWorkflow(previewOnlyWorkflow(), 'mystery.json', io);
    const inference = inferWorkflowCapabilities(workflow, io);
    const mapped = mapComfyWorkflow(workflow, host, inference);
    assert.equal(mapped.descriptor.enabled, false);
    assert.deepEqual(mapped.descriptor.capabilities, []);
    assert.match(mapped.descriptor.notes ?? '', /published disabled/);
  });

  it('produces a stable id derived from the workflow name', () => {
    const workflow = parseComfyWorkflow(imageTo3dWorkflow(), '3d/My Model [v2].json', io);
    const inference = inferWorkflowCapabilities(workflow, io);
    const first = mapComfyWorkflow(workflow, host, inference);
    const second = mapComfyWorkflow(workflow, host, inference);
    assert.equal(first.descriptor.id, second.descriptor.id);
    assert.match(first.descriptor.id, /^[a-z0-9][a-z0-9._-]*$/);
  });

  it('lets configuration pin, narrow, re-prioritise and re-estimate a workflow', () => {
    const workflow = parseComfyWorkflow(imageTo3dWorkflow(), '3d/my_image_to_model.json', io);
    const inference = inferWorkflowCapabilities(workflow, io);
    const mapped = mapComfyWorkflow(workflow, host, inference, {
      override: {
        workflowName: 'my_image_to_model',
        id: 'pinned_3d',
        name: 'My pinned 3D workflow',
        capabilities: ['image_to_3d'],
        priority: 5,
        vramGb: 9,
        enabled: true,
        tags: ['pinned'],
      },
    });
    assert.equal(mapped.applied, true);
    assert.equal(mapped.descriptor.id, 'pinned_3d');
    assert.equal(mapped.descriptor.name, 'My pinned 3D workflow');
    assert.equal(mapped.descriptor.priority, 5);
    assert.equal(mapped.descriptor.resources?.vramGb, 9);
    assert.ok((mapped.descriptor.tags ?? []).includes('pinned'));
    assert.match(mapped.descriptor.notes ?? '', /Configuration pins this workflow/);
  });

  it('cannot be talked into advertising a capability the graph does not prove', () => {
    const workflow = parseComfyWorkflow(imageWorkflow(), 'plain.json', io);
    const inference = inferWorkflowCapabilities(workflow, io);
    const mapped = mapComfyWorkflow(workflow, host, inference, {
      override: { workflowName: 'plain', capabilities: ['image_to_3d', 'text_to_3d'], enabled: true },
    });
    assert.deepEqual(mapped.descriptor.capabilities, [], 'configuration narrows, it never manufactures evidence');
  });

  it('uses a configured workflow file instead of the discovered graph when one is named', () => {
    const workflow = parseComfyWorkflow(imageTo3dWorkflow(), '3d/my.json', io);
    const inference = inferWorkflowCapabilities(workflow, io);
    const mapped = mapComfyWorkflow(workflow, host, inference, {
      override: { workflowName: 'my', workflowPath: 'workflows/pinned.api.json' },
    });
    assert.equal(mapped.descriptor.adapterConfig?.['workflow'], undefined);
    assert.equal(mapped.descriptor.adapterConfig?.['workflowPath'], 'workflows/pinned.api.json');
  });

  it('estimates VRAM from the graph as an explicitly approximate figure', () => {
    const workflow = parseComfyWorkflow(imageTo3dWorkflow(), 'w.json', io);
    const inference = inferWorkflowCapabilities(workflow, io);
    const estimate = estimateComfyWorkflowVram(workflow, inference);
    assert.ok(estimate >= 12 && estimate <= 24, `expected a conservative estimate, got ${estimate}`);
    assert.match(mapComfyWorkflow(workflow, host, inference).descriptor.notes ?? '', /treat as approximate/);
  });

  it('parses an override list strictly, so a typo cannot take the catalog down', () => {
    const ok = parseComfyWorkflowOverrides(comfyHost('http://x', { workflows: [{ workflowName: 'a' }] }));
    assert.equal(ok.ok, true);
    const missing = parseComfyWorkflowOverrides(comfyHost('http://x', { workflows: [{ id: 'a' }] }));
    assert.equal(missing.ok, false);
    const badCapability = parseComfyWorkflowOverrides(
      comfyHost('http://x', { workflows: [{ workflowName: 'a', capabilities: ['make_me_a_sandwich'] }] }),
    );
    assert.equal(badCapability.ok, false);
    assert.deepEqual(parseComfyWorkflowOverrides(comfyHost('http://x')), { ok: true, overrides: [] });
  });
});

// ── discovery against a fake install ────────────────────────────────────────

describe('comfyui 3D execution: artifact normalisation', () => {
  it('recognises mesh filenames without mistaking a render for one', () => {
    for (const name of ['a.glb', 'b.GLTF', 'c.obj', 'd.stl', 'e.ply', 'f.fbx', 'g.usdz', 'h.splat', 'i.spz']) {
      assert.equal(isThreeDFilename(name), true, `${name} is a mesh`);
    }
    for (const name of ['a.png', 'b.jpg', 'c.webp', 'd.txt', 'e.mp4', 'noextension']) {
      assert.equal(isThreeDFilename(name), false, `${name} is not a mesh`);
    }
  });

  it('sniffs GLB, glTF, OBJ, STL and PLY rather than trusting an extension', () => {
    assert.equal(sniffThreeDFormat(glbBytes()).format, 'glb');
    assert.equal(sniffThreeDFormat(new TextEncoder().encode('{"asset":{"version":"2.0"}}')).format, 'gltf');
    assert.equal(sniffThreeDFormat(new TextEncoder().encode('v 0 0 0\nv 1 0 0\nf 1 2 3\n')).format, 'obj');
    assert.equal(sniffThreeDFormat(new TextEncoder().encode('ply\nformat ascii 1.0\n')).format, 'ply');
    // A mislabelled file is reported rather than accepted silently.
    const wrong = sniffThreeDFormat(glbBytes(), 'obj');
    assert.equal(wrong.format, 'obj');
    assert.match(wrong.warning ?? '', /looks like glb but was declared as obj/);
  });

  it('reads file references out of any output key, in both item shapes', () => {
    const refs = readComfyFileRefs({
      '5': { images: [{ filename: 'preview.png', subfolder: '', type: 'temp' }] },
      '8': { '3d': ['3d/model_00001_.glb'] },
      '9': { result: ['3d/other.glb [output]'] },
      '10': { nothing: [] },
    });
    assert.deepEqual(
      refs.map((ref) => ref.filename),
      ['preview.png', 'model_00001_.glb', 'other.glb'],
    );
    assert.equal(refs[1]?.subfolder, '3d');
    assert.equal(refs[2]?.type, 'output');
  });

  it('rejects a body that is not a mesh at all', () => {
    assert.equal(sniffThreeDFormat(new TextEncoder().encode('<html>error</html>')).format, undefined);
  });
});