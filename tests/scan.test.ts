/**
 * Tests for the Local models scan: Ollama models and ComfyUI workflows.
 *
 * The scan is driven through injected probes, so no Ollama and no ComfyUI has to
 * be running. The probes are deliberately *transport*-shaped — they answer with a
 * status, a content type and a body, or with a classified failure — because the
 * behaviour under test is precisely the difference between "the engine refused",
 * "the engine answered 404" and "the engine answered 200 with something that is
 * not JSON". A helper that only threw would be unable to express the bug.
 *
 * The ComfyUI node metadata in the fixtures is copied from what a real ComfyUI
 * reports through `/object_info`, including the fields that matter
 * (`output_node`, declared output types, required inputs), so the analysis is
 * exercised against the same evidence it sees in production.
 *
 * @module dsh-ai-model-hub/tests/scan.test
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import { beforeEach, describe, it } from 'node:test';

import { ModelHub, renderMockPng } from '../src/index.ts';
import type { ModelCatalogConfig, ModelHost } from '../src/index.ts';
import { scanLocalResources, resetScanCache, workflowUrls } from '../dsh-plugin/scan.ts';
import type { HttpProbeResult, ScanProbes } from '../dsh-plugin/scan.ts';

/**
 * Node metadata as `/object_info` really reports it.
 *
 * `output_node` and the declared output types are the two facts the analysis
 * leans on, and both are reproduced here exactly as ComfyUI emits them.
 */
const objectInfo = {
  CheckpointLoaderSimple: { input: { required: { ckpt_name: [['A.safetensors'], {}] } } },
  LoraLoader: { input: { required: { lora_name: [['C.safetensors'], {}] } } },
  VAELoader: { input: { required: { vae_name: [['D.safetensors'], {}] } } },
  CLIPTextEncode: { input: { required: { clip: ['CLIP', {}], text: ['STRING', {}] } }, output: ['CONDITIONING'] },
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
  SaveImage: {
    input: { required: { images: ['IMAGE', {}], filename_prefix: ['STRING', {}] } },
    output: ['IMAGE'],
    output_node: true,
  },
  PreviewImage: { input: { required: { images: ['IMAGE', {}] } }, output: [], output_node: true },
  LoadImage: { input: { required: { image: [['input.png'], {}] } }, output: ['IMAGE', 'MASK'] },
  // ── the 3D families, as a real install reports them ──────────────────────
  VoxelToMesh: { input: { required: { voxel: ['VOXEL', {}] } }, output: ['MESH'] },
  MeshToFile3D: { input: { required: { mesh: ['MESH', {}] } }, output: ['FILE_3D_GLB'], output_node: false },
  Save3DAdvanced: {
    input: {
      required: {
        model_3d: ['MESH', {}],
        filename_prefix: ['STRING', {}],
        viewport_state: ['LOAD3D_VIEWPORT_STATE', {}],
        width: ['INT', {}],
        height: ['INT', {}],
      },
    },
    output: ['FILE_3D', 'LOAD3D_MODEL_INFO', 'LOAD3D_CAMERA', 'INT', 'INT'],
    output_node: true,
  },
  Preview3DAdvanced: {
    input: {
      required: {
        model_3d: ['MESH', {}],
        viewport_state: ['LOAD3D_VIEWPORT_STATE', {}],
        width: ['INT', {}],
        height: ['INT', {}],
      },
    },
    output: ['FILE_3D', 'LOAD3D_MODEL_INFO', 'LOAD3D_CAMERA', 'INT', 'INT'],
    output_node: true,
  },
  // ── author controls ─────────────────────────────────────────────────────
  PrimitiveBoolean: { input: { required: { value: ['BOOLEAN', {}] } }, output: ['BOOLEAN'] },
  PrimitiveInt: { input: { required: { value: ['INT', {}] } }, output: ['INT'] },
  Note: { input: {}, output: [] },
};

/** A complete text-to-image workflow in API format. */
const textToImage = {
  '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'A.safetensors' } },
  '2': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: 'a red fox' } },
  '3': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: '' } },
  '4': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
  '5': {
    class_type: 'KSampler',
    inputs: {
      model: ['1', 0],
      positive: ['2', 0],
      negative: ['3', 0],
      latent_image: ['4', 0],
      seed: 1,
      steps: 20,
      cfg: 7,
    },
  },
  '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
  '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: 'x' } },
};

/**
 * A 3D workflow shaped like the Trellis2/Pixal3D family.
 *
 * Not a copy of any one template — a graph using the same *node contracts* that
 * family uses, so the analysis is tested against the shape rather than against a
 * filename: an image loader, a mesh-producing chain, a `MeshToFile3D` converter
 * that is not flagged as an output node, a flagged `Save3DAdvanced` deliverable,
 * a flagged preview that must not be mistaken for one, and two titled controls.
 */
const imageToThreeD = {
  '10': { class_type: 'LoadImage', inputs: { image: 'subject.png' } },
  '11': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'A.safetensors' } },
  '12': { class_type: 'VAELoader', inputs: { vae_name: 'D.safetensors' } },
  '13': { class_type: 'EmptyLatentImage', inputs: { width: 1024, height: 1024, batch_size: 1 } },
  '14': {
    class_type: 'KSampler',
    inputs: { model: ['11', 0], positive: ['10', 0], negative: ['10', 0], latent_image: ['13', 0], seed: 3, steps: 12, cfg: 1 },
  },
  '15': { class_type: 'VAEDecode', inputs: { samples: ['14', 0], vae: ['12', 0] } },
  '16': { class_type: 'VoxelToMesh', inputs: { voxel: ['15', 0] } },
  '17': { class_type: 'MeshToFile3D', inputs: { mesh: ['16', 0] } },
  '18': {
    class_type: 'Save3DAdvanced',
    inputs: { model_3d: ['16', 0], filename_prefix: 'trellis/capture', viewport_state: '{"cam":1}', width: 1024, height: 1024 },
  },
  '19': {
    class_type: 'Preview3DAdvanced',
    inputs: { model_3d: ['16', 0], viewport_state: '{}', width: 512, height: 512 },
  },
  '20': { class_type: 'PrimitiveBoolean', inputs: { value: false } },
  '21': { class_type: 'PrimitiveInt', inputs: { value: 4096 } },
};

/** A UI-format export of the same graph, with titles on the two controls. */
const editorThreeD = {
  nodes: [
    { id: 10, type: 'LoadImage', inputs: [], widgets_values: { image: 'subject.png' } },
    { id: 16, type: 'VoxelToMesh', inputs: [{ name: 'voxel', link: 1 }], widgets_values: {} },
    { id: 18, type: 'Save3DAdvanced', inputs: [{ name: 'model_3d', link: 2 }], widgets_values: { filename_prefix: 'x', viewport_state: '{}', width: 1024, height: 1024 } },
    { id: 20, type: 'PrimitiveBoolean', title: 'Switch to Trellis2', inputs: [], widgets_values: { value: false } },
    { id: 21, type: 'PrimitiveInt', title: 'Texture Resolution', inputs: [], widgets_values: { value: 4096 } },
  ],
  links: [
    [1, 10, 0, 16, 0, 'VOXEL'],
    [2, 16, 0, 18, 0, 'MESH'],
  ],
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
        runtime: { engine: 'ollama', adapter: 'openai_compatible', endpoint: 'http://127.0.0.1:11434', path: '/v1/chat/completions' },
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
 * A hub over the fixture catalog, with no timers, no discovery, no probing.
 * @param endpoint - the ComfyUI endpoint.
 * @returns the hub.
 */
function hubFor(endpoint = 'http://127.0.0.1:8188'): ModelHub {
  return ModelHub.fromConfig(catalog(endpoint), { manageTimers: false, probeResources: false, log: () => {} });
}

/** How a fake machine answers. */
interface Spec {
  /** Raw body for `/api/tags`, `'offline'` to refuse, or a function re-read per call. */
  readonly ollama?: unknown | 'offline';
  /**
   * Raw body for `/system_stats`, `'offline'` to refuse, or a function.
   *
   * A function is what makes "the engine was down and then this scan started it"
   * expressible: the same probe answers differently once the fake hub has run.
   */
  readonly systemStats?: unknown | 'offline' | (() => unknown | 'offline');
  /** Raw body for `/object_info`, or `'offline'`. */
  readonly objectInfo?: unknown | 'offline';
  /** Saved workflows, keyed by their path relative to the workflows directory. */
  readonly workflows?: Record<string, unknown>;
  /** HTTP status for the *listing* route, when it should fail. */
  readonly listingStatus?: number;
  /** HTTP status for an individual workflow *fetch*, when it should fail. */
  readonly workflowStatus?: number;
  /** Files on disk, keyed by absolute path. */
  readonly files?: Record<string, string>;
  /** Accept only the documented encoded single-segment workflow URL. */
  readonly strictWorkflowUrl?: boolean;
  /** Record every URL requested. */
  readonly seen?: string[];
}

/**
 * Build probes for a described machine.
 *
 * The router is deliberately keyed by the *route*, not by the whole URL, because
 * several of the behaviours under test are about which URL the scan chose — the
 * encoded workflow path in particular.
 *
 * @param spec - what each route answers.
 * @returns the probes.
 */
function probesFor(spec: Spec): ScanProbes {
  const files = spec.files ?? {};
  const norm = (path: string): string => path.replace(/\\/g, '/');
  const parentOf = (path: string): string => path.slice(0, path.lastIndexOf('/'));
  const baseOf = (path: string): string => path.slice(path.lastIndexOf('/') + 1);
  const dirs = new Set<string>();
  for (const file of Object.keys(files)) {
    const parts = norm(file).split('/');
    for (let index = 2; index < parts.length; index += 1) dirs.add(parts.slice(0, index).join('/'));
  }

  const offline = (url: string): HttpProbeResult => ({
    ok: false,
    kind: 'connection',
    detail: `${url} could not be reached: connect ECONNREFUSED`,
  });

  return {
    request: async (url) => {
      spec.seen?.push(url);
      if (url.includes('/api/tags')) {
        return spec.ollama === undefined || spec.ollama === 'offline'
          ? offline(url)
          : { ok: true, status: 200, body: spec.ollama, contentType: 'application/json' };
      }
      if (url.endsWith('/system_stats')) {
        const answer = typeof spec.systemStats === 'function' ? spec.systemStats() : spec.systemStats;
        return answer === undefined || answer === 'offline'
          ? offline(url)
          : { ok: true, status: 200, body: answer, contentType: 'application/json' };
      }
      if (url.endsWith('/object_info')) {
        return spec.objectInfo === undefined || spec.objectInfo === 'offline'
          ? offline(url)
          : { ok: true, status: 200, body: spec.objectInfo, contentType: 'application/json' };
      }
      if (url.includes('/userdata?')) {
        if (spec.workflows === undefined) return offline(url);
        if (spec.listingStatus !== undefined) {
          return { ok: false, kind: 'http', status: spec.listingStatus, detail: `${url} answered HTTP ${spec.listingStatus}` };
        }
        return { ok: true, status: 200, body: Object.keys(spec.workflows), contentType: 'application/json' };
      }
      const match = /\/userdata\/(.+)$/.exec(url);
      if (match !== null && spec.workflows !== undefined) {
        const encoded = match[1] as string;
        const decoded = decodeURIComponent(encoded);
        const relative = decoded.replace(/^workflows\//, '');
        // A build that only serves the documented shape: the directory inside a
        // single encoded segment. Anything else 404s, which is exactly the bug.
        if (spec.strictWorkflowUrl === true && encoded !== encodeURIComponent(`workflows/${relative}`)) {
          return { ok: false, kind: 'http', status: 404, detail: `${url} answered HTTP 404` };
        }
        if (spec.workflowStatus !== undefined) {
          return { ok: false, kind: 'http', status: spec.workflowStatus, detail: `${url} answered HTTP ${spec.workflowStatus}` };
        }
        if (!Object.hasOwn(spec.workflows, relative)) {
          return { ok: false, kind: 'http', status: 404, detail: `${url} answered HTTP 404` };
        }
        return { ok: true, status: 200, body: spec.workflows[relative], contentType: 'application/json' };
      }
      return offline(url);
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

/**
 * A hub stand-in that records engine lifecycle calls without spawning anything.
 *
 * `state.running` is what the injected probes read, so "the engine comes up when
 * this scan starts it" is modelled by the fake hub flipping the same flag the
 * readiness probe consults.
 */
function lifecycleHub(options: {
  readonly running?: boolean;
  readonly startable?: boolean;
  readonly startResult?: { started: boolean; alreadyRunning: boolean };
  readonly busy?: number;
}): { hub: ModelHub; calls: string[]; state: { running: boolean } } {
  const calls: string[] = [];
  const state = { running: options.running ?? false };
  const hub = {
    catalog: { listModels: () => [] },
    activeInvocationsFor: () => options.busy ?? 0,
    publishScannedModels: (descriptors: readonly unknown[]) => {
      calls.push(`publish:${descriptors.length}`);
      return descriptors.length;
    },
    startEngineForHost: async () => {
      calls.push('start');
      if (options.startable === false) throw new Error('UNSAFE_OPERATION: process launch is disabled');
      const started = options.startResult ?? { started: true, alreadyRunning: false };
      state.running = true;
      return { modelId: 'engine-comfyui', ...started, health: { healthy: true } };
    },
    stopEngine: async () => {
      calls.push('stop');
      state.running = false;
      return { stopped: true, wasRunning: true };
    },
  } as unknown as ModelHub;
  return { hub, calls, state };
}

describe('Local models scan', () => {
  // The discovery cache is what makes a temporary failure non-destructive, so it
  // is deliberately process-wide. That also means one test's successful scan
  // would otherwise be visible to the next, so it is cleared between them.
  beforeEach(() => {
    resetScanCache();
  });

  it('treats both engines as running, using the route each one actually answers', async () => {
    // The regression: Ollama answers `/` with the text "Ollama is running" and
    // ComfyUI answers `/` with the editor's HTML, so a probe that fetched the root
    // and parsed JSON reported both as stopped while they were serving fine.
    const seen: string[] = [];
    const { hub } = lifecycleHub({ running: true });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: false,
      probes: probesFor({
        ollama: ollamaTags,
        systemStats: { system: { comfyui_version: '1' } },
        objectInfo,
        workflows: {},
        seen,
      }),
    });

    assert.ok(seen.includes('http://127.0.0.1:11434/api/tags'), 'Ollama is asked /api/tags');
    assert.ok(seen.includes('http://127.0.0.1:8188/system_stats'), 'ComfyUI is asked /system_stats');
    assert.ok(!seen.includes('http://127.0.0.1:8188'), 'the bare ComfyUI root is never used as a health check');
    assert.ok(!seen.some((url) => /:11434\/?$/.test(url)), 'the bare Ollama root is never used as a health check');

    assert.equal(result.resources.filter((entry) => entry.kind === 'ollama_model').length, 2);
    assert.ok(result.sources.every((source) => source.ok), 'both engines answered');
  });

  it('reports a non-JSON 200 as a content problem, not as an unreachable engine', async () => {
    const probes: ScanProbes = {
      ...probesFor({ workflows: {}, objectInfo }),
      request: async (url) =>
        url.endsWith('/system_stats')
          ? { ok: false, kind: 'not-json', status: 200, detail: `${url} answered 200 with text/html, which is not JSON` }
          : url.includes('/api/tags')
            ? { ok: true, status: 200, body: ollamaTags }
            : { ok: false, kind: 'connection', detail: `${url} refused` },
    };
    const { hub } = lifecycleHub({});
    const result = await scanLocalResources(hub, { hosts: hostsOf(), startEngine: false, probes });
    const comfy = result.sources.find((source) => source.id === 'comfyui');
    assert.equal(comfy?.ok, false);
    assert.match(comfy?.detail ?? '', /not JSON/);
    assert.match(result.warnings.join(' '), /not JSON|did not answer/);
    assert.equal(result.resources.filter((entry) => entry.kind === 'ollama_model').length, 2, 'Ollama still resolves');
  });

  it('retrieves a workflow from a nested path using the URL the API actually serves', async () => {
    const seen: string[] = [];
    const { hub } = lifecycleHub({ running: true });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: false,
      probes: probesFor({
        ollama: { models: [] },
        systemStats: {},
        objectInfo,
        // A nested name with a space, so both the directory and the encoding matter.
        workflows: { '3d/image to model_trellis2.json': textToImage },
        strictWorkflowUrl: true,
        seen,
      }),
    });

    const workflow = result.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.ok(workflow, 'the nested workflow was retrieved');
    assert.equal(workflow.status, 'ready');
    assert.equal(workflow.name, 'image to model_trellis2');
    // The documented shape: the workflows directory inside ONE encoded segment.
    assert.ok(
      seen.includes('http://127.0.0.1:8188/userdata/workflows%2F3d%2Fimage%20to%20model_trellis2.json'),
      `expected the encoded single-segment URL, saw: ${seen.filter((u) => u.includes('/userdata/')).join(', ')}`,
    );
  });

  it('reports an unreadable workflow as unreadable, with the status, instead of invalid JSON', async () => {
    const { hub } = lifecycleHub({ running: true });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: false,
      probes: probesFor({
        ollama: { models: [] },
        systemStats: {},
        objectInfo,
        workflows: { '3d_image_to_model_trellis2_pixal3d.json': textToImage },
        workflowStatus: 404,
      }),
    });

    const workflow = result.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.ok(workflow, 'the workflow is still listed');
    assert.equal(workflow.status, 'unreadable', 'a fetch failure is not an invalid workflow');
    assert.equal(workflow.runnable, false);
    assert.match(workflow.detail, /would not return its contents/);
    assert.match(workflow.detail, /HTTP 404/, 'the actual status is reported');
    assert.doesNotMatch(workflow.detail, /invalid JSON/i, 'the document was never seen, so it is not called invalid');
  });

  it('exposes the image input, the titled controls and the one real deliverable of a 3D workflow', async () => {
    const { hub } = lifecycleHub({ running: true });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: false,
      probes: probesFor({
        ollama: { models: [] },
        systemStats: {},
        objectInfo,
        workflows: { 'image_to_model.json': imageToThreeD },
      }),
    });

    const workflow = result.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.ok(workflow);
    assert.equal(workflow.status, 'ready');
    assert.deepEqual(workflow.capabilities, ['image_to_3d'], 'an image-driven graph is image_to_3d');

    const inputNames = (workflow.inputs ?? []).map((input) => input.name);
    assert.ok(inputNames.includes('image'), `image input missing from ${inputNames.join(', ')}`);
    assert.deepEqual(workflow.outputs?.map((output) => output.type), ['model_3d']);

    // The deliverable is the flagged Save3DAdvanced, not the unflagged
    // MeshToFile3D converter and not the flagged Preview3DAdvanced — a preview
    // writes to ComfyUI's temp directory.
    assert.equal(workflow.outputs?.[0]?.nodeClass, 'Save3DAdvanced');

    assert.ok(
      !(workflow.diagnostics ?? []).some((note) => note.includes('MeshToFile3D')),
      'the unflagged converter is not a diagnostic',
    );
  });

  it('names author-titled controls and withholds untitled ones with an actionable note', async () => {
    const titled = { ...imageToThreeD, '20': { class_type: 'PrimitiveBoolean', inputs: { value: false } } };
    const { hub } = lifecycleHub({ running: true });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: false,
      probes: probesFor({
        ollama: { models: [] },
        systemStats: {},
        objectInfo,
        // The UI export carries the titles; the API graph alone cannot.
        workflows: { 'demo.json': editorThreeD },
      }),
    });
    void titled;

    const workflow = result.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.ok(workflow);
    const byName = new Map((workflow.inputs ?? []).map((input) => [input.name, input]));
    assert.ok(byName.has('switch_to_trellis2'), `expected the titled boolean control: ${[...byName.keys()].join(', ')}`);
    assert.ok(byName.has('texture_resolution'), 'expected the titled integer control');
    assert.equal(byName.get('switch_to_trellis2')?.kind, 'boolean');
    assert.equal(byName.get('texture_resolution')?.kind, 'number');
    assert.equal(byName.get('switch_to_trellis2')?.label, 'Switch to Trellis2');
  });

  it('does not let an unreachable Ollama hide the workflows', async () => {
    const { hub } = lifecycleHub({ running: true });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: false,
      probes: probesFor({ ollama: 'offline', systemStats: {}, objectInfo, workflows: { 'a.json': textToImage } }),
    });
    assert.equal(result.resources.filter((entry) => entry.kind === 'ollama_model').length, 0);
    assert.equal(result.resources.filter((entry) => entry.kind === 'comfyui_workflow').length, 1);
    assert.ok(result.sources.some((source) => source.kind === 'ollama_model' && !source.ok));
  });

  it('does not let an unreachable ComfyUI hide the models, and still reads a directory', async () => {
    const { hub } = lifecycleHub({ running: false });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: false,
      workflowDir: '/workflows/comfy',
      probes: probesFor({
        ollama: ollamaTags,
        systemStats: 'offline',
        objectInfo: 'offline',
        files: { '/workflows/comfy/local.json': JSON.stringify(textToImage) },
      }),
    });
    assert.equal(result.resources.filter((entry) => entry.kind === 'ollama_model').length, 2);
    const local = result.resources.filter((entry) => entry.kind === 'comfyui_workflow');
    assert.equal(local.length, 1, 'a directory workflow is found while ComfyUI is down');
    assert.equal(local[0]?.status, 'ready', 'an API workflow needs no engine to be understood');
    assert.equal(local[0]?.engineRunning, false);
  });

  it('starts a stopped engine for the scan and stops only the instance it started', async () => {
    const { hub, calls, state } = lifecycleHub({ running: false, startResult: { started: true, alreadyRunning: false } });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      probes: probesFor({
        ollama: { models: [] },
        systemStats: () => (state.running ? {} : 'offline'),
        objectInfo,
        workflows: { 'a.json': textToImage },
        seen: [],
      }),
    });

    assert.ok(calls.includes('start'), 'the engine was started through the hub');
    assert.ok(calls.includes('stop'), 'the instance this scan started was shut down again');
    assert.equal(result.registered >= 1, true, 'the workflow found while the engine was up was registered');
  });

  it('leaves an already-running engine alone', async () => {
    const { hub, calls, state } = lifecycleHub({ running: true, startResult: { started: false, alreadyRunning: true } });
    await scanLocalResources(hub, {
      hosts: hostsOf(),
      probes: probesFor({
        ollama: { models: [] },
        systemStats: () => (state.running ? {} : 'offline'),
        objectInfo,
        workflows: { 'a.json': textToImage },
      }),
    });
    assert.deepEqual(calls.filter((call) => call === 'stop'), [], 'a pre-existing engine is never stopped');
  });

  it('defers shutdown while a graph is queued or running', async () => {
    const { hub, calls, state } = lifecycleHub({ running: false, busy: 1 });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      probes: probesFor({
        ollama: { models: [] },
        systemStats: () => (state.running ? {} : 'offline'),
        objectInfo,
        workflows: { 'a.json': textToImage },
      }),
    });
    assert.ok(calls.includes('start'));
    assert.deepEqual(calls.filter((call) => call === 'stop'), [], 'a busy engine is not interrupted');
    assert.match(result.warnings.join(' '), /queued or running/);
  });

  it('refuses to start anything when the deployment forbids process launch, and says so', async () => {
    const { hub, calls, state } = lifecycleHub({ running: false, startable: false });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      probes: probesFor({
        ollama: { models: [] },
        systemStats: () => (state.running ? {} : 'offline'),
        objectInfo,
        workflows: {},
      }),
    });
    assert.deepEqual(calls.filter((call) => call === 'stop'), []);
    assert.match(result.warnings.join(' '), /process launch is disabled/);
  });

  it('shares one pass between concurrent scans so no second engine is started', async () => {
    let starts = 0;
    const state = { running: false };
    const base = probesFor({
      ollama: { models: [] },
      systemStats: () => (state.running ? {} : 'offline'),
      objectInfo,
      workflows: { 'a.json': textToImage },
    });
    const probes: ScanProbes = {
      ...base,
      request: async (url, timeoutMs) => {
        // Hold the graph request open long enough for the second scan to join.
        if (url.endsWith('/object_info')) await new Promise((resolve) => setTimeout(resolve, 40));
        return base.request(url, timeoutMs);
      },
    };
    const hub = {
      catalog: { listModels: () => [] },
      activeInvocationsFor: () => 0,
      publishScannedModels: () => 1,
      startEngineForHost: async () => {
        starts += 1;
        state.running = true;
        return { modelId: 'engine-comfyui', started: true, alreadyRunning: false, health: { healthy: true } };
      },
      stopEngine: async () => ({ stopped: true, wasRunning: true }),
    } as unknown as ModelHub;

    const [first, second] = await Promise.all([
      scanLocalResources(hub, { hosts: hostsOf(), probes }),
      scanLocalResources(hub, { hosts: hostsOf(), probes }),
    ]);
    assert.equal(starts, 1, 'two concurrent scans started one engine between them');
    assert.equal(first.resources.length, second.resources.length);
  });

  it('keeps previously discovered workflows visible after the engine goes away', async () => {
    const online = probesFor({ ollama: { models: [] }, systemStats: {}, objectInfo, workflows: { 'a.json': textToImage } });
    const first = await scanLocalResources(lifecycleHub({ running: true }).hub, {
      hosts: hostsOf(),
      startEngine: false,
      probes: online,
    });
    assert.equal(first.resources.filter((entry) => entry.kind === 'comfyui_workflow').length, 1);

    // ComfyUI stops, and is not allowed to be started for this pass.
    const offline = probesFor({ ollama: { models: [] }, systemStats: 'offline', objectInfo: 'offline' });
    const published: unknown[][] = [];
    const hub = {
      catalog: { listModels: () => [] },
      activeInvocationsFor: () => 0,
      publishScannedModels: (descriptors: readonly unknown[]) => {
        published.push([...descriptors]);
        return descriptors.length;
      },
      startEngineForHost: async () => {
        throw new Error('not startable');
      },
      stopEngine: async () => ({ stopped: false, wasRunning: false }),
    } as unknown as ModelHub;

    const second = await scanLocalResources(hub, { hosts: hostsOf(), startEngine: false, probes: offline });
    const workflow = second.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.ok(workflow, 'the workflow discovered earlier is still listed');
    assert.equal(workflow.engineRunning, false, 'and is reported as belonging to a stopped engine');
    assert.equal(workflow.runnable, true, 'it can still run by starting the engine on demand');
    assert.match(workflow.detail, /not running/);
    assert.ok(
      (workflow.diagnostics ?? []).some((note) => /not answering|could not be reached|ECONNREFUSED/i.test(note)),
      'the reason is attached',
    );
  });

  it('lists installed Ollama models and complete workflows, and never a weight file or a node class', async () => {
    const { hub } = lifecycleHub({ running: true });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: false,
      probes: probesFor({ ollama: ollamaTags, systemStats: {}, objectInfo, workflows: { 'demo.json': textToImage } }),
    });

    assert.deepEqual(
      result.resources.map((entry) => entry.typeLabel).sort(),
      ['ComfyUI Workflow', 'Ollama Model', 'Ollama Model'],
    );
    assert.deepEqual(
      result.resources.filter((entry) => entry.kind === 'ollama_model').map((entry) => entry.name),
      ['llama3:8b', 'qwen2.5:3b'],
    );

    const serialized = JSON.stringify(result.resources);
    assert.doesNotMatch(serialized, /safetensors/, 'no checkpoint, LoRA or VAE is listed');
    for (const nodeClass of ['CheckpointLoaderSimple', 'LoraLoader', 'VAELoader', 'KSampler']) {
      assert.doesNotMatch(serialized, new RegExp(nodeClass), `${nodeClass} must not be listed`);
    }
  });

  it('refreshes on a second scan instead of duplicating entries', async () => {
    const probes = probesFor({
      ollama: ollamaTags,
      systemStats: {},
      objectInfo,
      workflows: { 'a.json': textToImage, 'b.json': imageToThreeD },
    });
    const hub = lifecycleHub({ running: true }).hub;
    const first = await scanLocalResources(hub, { hosts: hostsOf(), startEngine: false, probes });
    const second = await scanLocalResources(hub, { hosts: hostsOf(), startEngine: false, probes });
    assert.deepEqual(
      second.resources.map((entry) => entry.id).sort(),
      first.resources.map((entry) => entry.id).sort(),
    );
    assert.equal(new Set(second.resources.map((entry) => entry.id)).size, second.resources.length);
  });

  it('runs a discovered workflow end to end, with the bindings the scan inferred', async () => {
    const png = renderMockPng(20, 12, 'scanned');
    const queued: string[] = [];
    const http = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => {
        const url = req.url ?? '/';
        if (url.startsWith('/view')) {
          res.writeHead(200, { 'content-type': 'image/png' });
          res.end(Buffer.from(png));
          return;
        }
        if (url === '/prompt') queued.push(Buffer.concat(chunks).toString('utf8'));
        const body =
          url === '/system_stats'
            ? { system: {} }
            : url === '/object_info'
              ? objectInfo
              : url.startsWith('/userdata?')
                ? ['t2i.json']
                : url.includes('/userdata/')
                  ? textToImage
                  : url === '/prompt'
                    ? { prompt_id: 'p-1' }
                    : url.startsWith('/history/')
                      ? {
                          'p-1': {
                            status: { status_str: 'success' },
                            outputs: { '7': { images: [{ filename: 'out.png', subfolder: '', type: 'output' }] } },
                          },
                        }
                      : {};
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
      // No injected probes: the scan talks to the fake engine over a real socket,
      // so retrieval, analysis, registration and invocation run as they would.
      const scan = await scanLocalResources(hub, { hosts: hostsOf(endpoint), startEngine: false, timeoutMs: 5_000 });
      const workflow = scan.resources.find((entry) => entry.kind === 'comfyui_workflow');
      assert.equal(workflow?.status, 'ready');
      assert.deepEqual(workflow?.inputs?.map((input) => input.name), ['prompt', 'negative_prompt', 'seed', 'steps', 'cfg', 'width', 'height']);

      const result = await hub.invokeModel({
        capability: 'text_to_image',
        prompt: 'a futuristic city',
        options: { width: 256, height: 128 },
        modelId: workflow?.modelId ?? '',
      });
      assert.equal(result.modelId, workflow?.modelId);
      assert.equal(result.outputs.length, 1);
      assert.equal(result.outputs[0]?.type, 'image');

      const sent = JSON.parse(queued[0] ?? '{}') as { prompt: Record<string, { inputs: Record<string, unknown> }> };
      assert.equal(sent.prompt['2']?.inputs['text'], 'a futuristic city', 'the prompt went to the node the scan found');
      assert.equal(sent.prompt['4']?.inputs['width'], 256, 'the width went to the size node the scan found');
      assert.equal(sent.prompt['4']?.inputs['height'], 128);
      assert.equal(sent.prompt['1']?.inputs['ckpt_name'], 'A.safetensors', 'the graph is otherwise untouched');
    } finally {
      await hub.dispose();
      await new Promise<void>((resolve) => http.close(() => resolve()));
    }
  });

  it('launches an engine for a host that carries no catalog model', async () => {
    // The regression this exists for: the scan has to start ComfyUI before it can
    // discover the first workflow, so at that moment the host has no model to hang
    // a launch on. The launch resolves an engine record instead — and every later
    // step (the endpoint probe, the health wait, the stop) looks the model up by
    // id, so an id the catalog has never heard of must still resolve, or the launch
    // fails halfway with MODEL_NOT_FOUND. A fake hub could not have caught this.
    const hub = ModelHub.fromConfig(
      {
        version: '1',
        hosts: [
          {
            id: 'comfyui',
            name: 'ComfyUI',
            adapter: 'comfyui',
            // A deliberately dead port: the endpoint is never live, so the only
            // way past this point is the launch itself.
            runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint: 'http://127.0.0.1:9' },
            lifecycle: {
              startable: true,
              stoppable: true,
              startupTimeoutMs: 3_000,
              awaitHealthOnStart: true,
              start: { command: 'python', args: ['-c', 'import time; time.sleep(0.3)'] },
            },
          },
        ],
        models: [],
      },
      { manageTimers: false, probeResources: false, log: () => {} },
    );
    try {
      let failure: { code?: string } | undefined;
      try {
        await hub.startEngineForHost('comfyui');
      } catch (error) {
        failure = error as { code?: string };
      }
      // Any failure is acceptable except "that id does not exist": the point is
      // that the launch reached the engine, not that a stub process came up.
      assert.notEqual(failure?.code, 'MODEL_NOT_FOUND', 'an engine record must resolve without a catalog entry');

      // And the id stays addressable, which is what lets a scan that started the
      // engine wait on it and shut it down again.
      assert.equal(hub.getModelStatus('engine-comfyui').modelId, 'engine-comfyui');
      assert.doesNotThrow(() => hub.activeInvocationsFor('engine-comfyui'));
      await hub.stopModel('engine-comfyui');
    } finally {
      await hub.dispose();
    }
  });

  it('updates a registration when the saved workflow changes, without adding a second one', async () => {
    // The id comes from the source location, so editing a workflow refreshes the
    // entry rather than leaving a stale contract behind or registering the changed
    // graph under a new id.
    let published: readonly { id: string; adapterConfig: Readonly<Record<string, unknown>> }[] = [];
    const hub = {
      catalog: { listModels: () => [] },
      activeInvocationsFor: () => 0,
      publishScannedModels: (descriptors: readonly never[]) => {
        published = descriptors as never;
        return descriptors.length;
      },
      startEngineForHost: async () => ({
        modelId: 'engine-comfyui',
        started: false,
        alreadyRunning: true,
        health: { healthy: true },
      }),
      stopEngine: async () => ({ stopped: false, wasRunning: false }),
    } as unknown as ModelHub;

    const before = probesFor({ ollama: { models: [] }, systemStats: {}, objectInfo, workflows: { 'a.json': textToImage } });
    const first = await scanLocalResources(hub, { hosts: hostsOf(), startEngine: false, probes: before });
    const firstWorkflow = first.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.equal(firstWorkflow?.status, 'ready');
    const firstSteps = (published[0]?.adapterConfig['workflow'] as Record<string, { inputs: Record<string, unknown> }>)?.['5']
      ?.inputs['steps'];
    assert.equal(firstSteps, 20);

    // The same file, saved again after an edit.
    const edited = JSON.parse(JSON.stringify(textToImage)) as Record<string, { inputs: Record<string, unknown> }>;
    edited['5']!.inputs['steps'] = 4;
    const after = probesFor({ ollama: { models: [] }, systemStats: {}, objectInfo, workflows: { 'a.json': edited } });
    const second = await scanLocalResources(hub, { hosts: hostsOf(), startEngine: false, probes: after });

    const secondWorkflow = second.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.equal(secondWorkflow?.id, firstWorkflow?.id, 'the edit refreshed the same registration');
    assert.equal(second.resources.filter((entry) => entry.kind === 'comfyui_workflow').length, 1, 'no duplicate was added');
    assert.equal(published.length, 1, 'one provider was published, not two');

    const graph = published[0]?.adapterConfig['workflow'] as Record<string, { inputs: Record<string, unknown> }>;
    assert.equal(graph['5']?.inputs['steps'], 4, 'the registered graph reflects the saved edit');
  });

  it('builds several candidate retrieval URLs, the documented one first', () => {
    const urls = workflowUrls('http://h:8188', '3d/model.json');
    assert.equal(urls[0], 'http://h:8188/userdata/workflows%2F3d%2Fmodel.json');
    assert.ok(urls.includes('http://h:8188/userdata/workflows/3d/model.json'));
    assert.equal(new Set(urls).size, urls.length, 'no duplicate attempts');
  });
});
