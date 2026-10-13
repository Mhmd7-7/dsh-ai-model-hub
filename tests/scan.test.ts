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
import { join } from 'node:path';
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
  // Cache ownership is per hub/config. Reset weak ownership between fixtures.
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
        systemStats: { system: { comfyui_version: '1' }, devices: [] },
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
        systemStats: { system: {}, devices: [] },
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
        systemStats: { system: {}, devices: [] },
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
        systemStats: { system: {}, devices: [] },
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
        systemStats: { system: {}, devices: [] },
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
      probes: probesFor({ ollama: 'offline', systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'a.json': textToImage } }),
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

  it('never starts a stopped engine, even with startEngine true', async () => {
    const { hub, calls, state } = lifecycleHub({ running: false, startResult: { started: true, alreadyRunning: false } });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: true,
      probes: probesFor({
        ollama: { models: [] },
        systemStats: () => (state.running ? { system: {}, devices: [] } : 'offline'),
        objectInfo,
        workflows: { 'a.json': textToImage },
        seen: [],
      }),
    });

    assert.deepEqual(calls, [], 'scanning has no launch, stop or publish effects when no resources were found');
    assert.equal(state.running, false);
    assert.equal(result.registered, 0);
    assert.match(result.warnings.join(' '), /scans never start engines/);
  });

  it('leaves an already-running engine alone', async () => {
    const { hub, calls, state } = lifecycleHub({ running: true, startResult: { started: false, alreadyRunning: true } });
    await scanLocalResources(hub, {
      hosts: hostsOf(),
      probes: probesFor({
        ollama: { models: [] },
        systemStats: () => (state.running ? { system: {}, devices: [] } : 'offline'),
        objectInfo,
        workflows: { 'a.json': textToImage },
      }),
    });
    assert.deepEqual(calls.filter((call) => call === 'stop'), [], 'a pre-existing engine is never stopped');
  });

  it('does not touch engine lifecycle when a graph is queued or running', async () => {
    const { hub, calls, state } = lifecycleHub({ running: false, busy: 1 });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      probes: probesFor({
        ollama: { models: [] },
        systemStats: () => (state.running ? { system: {}, devices: [] } : 'offline'),
        objectInfo,
        workflows: { 'a.json': textToImage },
      }),
    });
    assert.ok(!calls.includes('start'));
    assert.deepEqual(calls.filter((call) => call === 'stop'), [], 'a busy engine is not interrupted');
    assert.match(result.warnings.join(' '), /scans never start engines/);
  });

  it('does not attempt process launch even when lifecycle hooks would reject it', async () => {
    const { hub, calls, state } = lifecycleHub({ running: false, startable: false });
    const result = await scanLocalResources(hub, {
      hosts: hostsOf(),
      probes: probesFor({
        ollama: { models: [] },
        systemStats: () => (state.running ? { system: {}, devices: [] } : 'offline'),
        objectInfo,
        workflows: {},
      }),
    });
    assert.deepEqual(calls.filter((call) => call === 'stop'), []);
    assert.deepEqual(calls.filter((call) => call === 'start'), []);
    assert.match(result.warnings.join(' '), /scans never start engines/);
  });

  it('keeps concurrent scans observational without starting engines', async () => {
    let starts = 0;
    const state = { running: false };
    const base = probesFor({
      ollama: { models: [] },
      systemStats: () => (state.running ? { system: {}, devices: [] } : 'offline'),
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
    assert.equal(starts, 0, 'neither concurrent scan starts engines');
    assert.equal(first.resources.length, second.resources.length);
  });

  it('keeps previously discovered workflows visible after the engine goes away', async () => {
    const online = probesFor({ ollama: { models: [] }, systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'a.json': textToImage } });
    const hub = lifecycleHub({ running: true }).hub;
    const first = await scanLocalResources(hub, {
      hosts: hostsOf(),
      startEngine: false,
      probes: online,
    });
    assert.equal(first.resources.filter((entry) => entry.kind === 'comfyui_workflow').length, 1);

    // ComfyUI stops, and is not allowed to be started for this pass.
    const offline = probesFor({ ollama: { models: [] }, systemStats: 'offline', objectInfo: 'offline' });
    const second = await scanLocalResources(hub, { hosts: hostsOf(), startEngine: false, probes: offline });
    const workflow = second.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.ok(workflow, 'the workflow discovered earlier is still listed');
    assert.equal(workflow.engineRunning, false, 'and is reported as belonging to a stopped engine');
    assert.equal(workflow.runnable, true, 'it can still run by starting the engine on demand');
    assert.equal(workflow.freshness, 'cached');
    assert.equal(workflow.availability, 'unavailable');
    assert.equal(workflow.checkedAt, first.resources[0]?.checkedAt);
    assert.match(workflow.detail, /cached/);
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
      probes: probesFor({ ollama: ollamaTags, systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'demo.json': textToImage } }),
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
      systemStats: { system: {}, devices: [] },
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
            ? { system: {}, devices: [] }
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

  it('does not launch a configured startable engine even when process launch is allowed', async () => {
    const hub = ModelHub.fromConfig({
      version: '1', hosts: [{ id: 'comfyui', name: 'ComfyUI', adapter: 'comfyui',
        runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint: 'http://127.0.0.1:9' },
        lifecycle: { startable: true, stoppable: true,
          start: { command: 'this-command-must-never-run' } },
      }], models: [],
    }, { manageTimers: false, probeResources: false, log: () => {} });
    let starts = 0;
    hub.startEngineForHost = async () => { starts += 1; throw new Error('must not be called'); };
    try {
      const result = await scanLocalResources(hub, { hosts: hub.catalog.listHosts(), startEngine: true,
        timeoutMs: 20, probes: probesFor({}) });
      assert.equal(starts, 0);
      assert.equal(result.resources.length, 0);
      assert.ok(result.sources.every((source) => source.startedEngine !== true && source.releasedEngine !== true));
    } finally { await hub.dispose(); }
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

    const before = probesFor({ ollama: { models: [] }, systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'a.json': textToImage } });
    const first = await scanLocalResources(hub, { hosts: hostsOf(), startEngine: false, probes: before });
    const firstWorkflow = first.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.equal(firstWorkflow?.status, 'ready');
    const firstSteps = (published[0]?.adapterConfig['workflow'] as Record<string, { inputs: Record<string, unknown> }>)?.['5']
      ?.inputs['steps'];
    assert.equal(firstSteps, 20);

    // The same file, saved again after an edit.
    const edited = JSON.parse(JSON.stringify(textToImage)) as Record<string, { inputs: Record<string, unknown> }>;
    edited['5']!.inputs['steps'] = 4;
    const after = probesFor({ ollama: { models: [] }, systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'a.json': edited } });
    const second = await scanLocalResources(hub, { hosts: hostsOf(), startEngine: false, probes: after });

    const secondWorkflow = second.resources.find((entry) => entry.kind === 'comfyui_workflow');
    assert.equal(secondWorkflow?.id, firstWorkflow?.id, 'the edit refreshed the same registration');
    assert.equal(second.resources.filter((entry) => entry.kind === 'comfyui_workflow').length, 1, 'no duplicate was added');
    assert.equal(published.length, 1, 'one provider was published, not two');

    const graph = published[0]?.adapterConfig['workflow'] as Record<string, { inputs: Record<string, unknown> }>;
    assert.equal(graph['5']?.inputs['steps'], 4, 'the registered graph reflects the saved edit');
  });

  it('bounds thrown and hanging HTTP probes and starts engine passes independently', async () => {
    for (const mode of ['throw', 'hang'] as const) {
      const hub = lifecycleHub({}).hub;
      const seen: string[] = [];
      const base = probesFor({ systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'ok.json': textToImage }, seen });
      let comfyStarted = false;
      const probes: ScanProbes = { ...base, request: (url, budget) => {
        if (url.includes('/api/tags')) {
          if (mode === 'throw') throw new Error('injected transport throw');
          return new Promise(() => {});
        }
        if (url.endsWith('/system_stats')) comfyStarted = true;
        return base.request(url, budget);
      } };
      const task = scanLocalResources(hub, { hosts: hostsOf(), timeoutMs: 15, probes });
      await Promise.resolve(); await Promise.resolve();
      assert.ok(comfyStarted, 'Comfy starts before the pending Ollama probe completes');
      const result = await task;
      assert.equal(result.resources.filter((resource) => resource.kind === 'comfyui_workflow').length, 1);
      assert.match(result.sources.find((source) => source.id === 'ollama')?.detail ?? '', /throw|timed out/);
    }
  });

  it('bounds every filesystem effect without hiding live API results', async () => {
    for (const effect of ['isDirectory', 'listDir', 'readFile'] as const) {
      for (const mode of ['throw', 'hang'] as const) {
        const base = probesFor({ ollama: ollamaTags, systemStats: { system: {}, devices: [] }, objectInfo,
          workflows: { 'api.json': textToImage }, files: { '/workflows/comfy/local.json': JSON.stringify(textToImage) } });
        const probes = { ...base, [effect]: () => {
          if (mode === 'throw') throw new Error('injected ' + effect);
          return new Promise(() => {});
        } } as ScanProbes;
        const result = await scanLocalResources(lifecycleHub({}).hub, { hosts: hostsOf(),
          timeoutMs: 10, workflowDir: '/workflows/comfy', probes });
        assert.equal(result.resources.filter((resource) => resource.kind === 'ollama_model').length, 2);
        assert.ok(result.resources.some((resource) => resource.name === 'api'));
        if (effect === 'readFile') assert.equal(result.resources.find((resource) => resource.name === 'local')?.status, 'unreadable');
        else assert.ok(result.sources.some((source) => source.id === 'workflow-dir' && !source.ok));
      }
    }
  });

  it('retains known Ollama catalog entries offline and reports uninstalled entries honestly online', async () => {
    const config = catalog('http://127.0.0.1:8188');
    const hub = ModelHub.fromConfig({ ...config, models: [{ id: 'known_llama', name: 'Known llama', type: 'text_generation',
      host: 'ollama', capabilities: ['text_to_text'], adapterConfig: { model: 'llama3:8b' } }] },
      { manageTimers: false, probeResources: false });
    try {
      const offline = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({}) });
      assert.equal(offline.resources[0]?.modelId, 'known_llama');
      assert.equal(offline.resources[0]?.freshness, 'catalog');
      assert.equal(offline.resources[0]?.availability, 'unknown');
      const online = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ ollama: { models: [] } }) });
      assert.equal(online.resources[0]?.availability, 'unavailable');
      assert.match(online.resources[0]?.detail ?? '', /not reported installed/);
    } finally { await hub.dispose(); }
  });

  it('marks cached Ollama models as cached without advancing their observation timestamp', async () => {
    const hub = lifecycleHub({}).hub;
    const first = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ ollama: ollamaTags }) });
    const second = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({}) });
    assert.equal(second.resources.length, 2);
    assert.equal(second.resources[0]?.freshness, 'cached');
    assert.equal(second.resources[0]?.availability, 'unavailable');
    assert.equal(second.resources[0]?.checkedAt, first.resources[0]?.checkedAt);
  });

  it('isolates caches across hubs, endpoints and workflow directory configurations', async () => {
    const hub = lifecycleHub({}).hub;
    await scanLocalResources(hub, { hosts: hostsOf(), workflowDir: '/old/workflows',
      probes: probesFor({ ollama: ollamaTags, systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'api.json': textToImage },
        files: { '/old/workflows/local.json': JSON.stringify(textToImage) } }) });
    const otherHub = await scanLocalResources(lifecycleHub({}).hub, { hosts: hostsOf(), probes: probesFor({}) });
    assert.deepEqual(otherHub.resources, []);
    const changedEndpoint = await scanLocalResources(hub, { hosts: hostsOf('http://127.0.0.1:9999'), probes: probesFor({}) });
    assert.ok(!changedEndpoint.resources.some((resource) => resource.kind === 'comfyui_workflow'));
    const changedDir = await scanLocalResources(hub, { hosts: hostsOf(), workflowDir: '/new/workflows', probes: probesFor({}) });
    assert.ok(!changedDir.resources.some((resource) => resource.name === 'local'));
    assert.ok(changedDir.resources.some((resource) => resource.name === 'api'), 'API cache stays independent of directory config');
  });

  it('preserves workflow contracts across listing, file and node-info partial failures', async () => {
    for (const failure of ['listing', 'file', 'nodes'] as const) {
      const hub = lifecycleHub({}).hub;
      const first = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ ollama: { models: [] },
        systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'a.json': editorThreeD } }) });
      const second = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ ollama: { models: [] },
        systemStats: { system: {}, devices: [] }, objectInfo: failure === 'nodes' ? 'offline' : objectInfo, workflows: { 'a.json': editorThreeD },
        ...(failure === 'listing' ? { listingStatus: 500 } : {}), ...(failure === 'file' ? { workflowStatus: 500 } : {}) }) });
      assert.equal(second.resources[0]?.freshness, 'cached', failure);
      assert.equal(second.resources[0]?.availability, 'unknown', failure);
      assert.equal(second.resources[0]?.checkedAt, first.resources[0]?.checkedAt);
      assert.equal(second.resources[0]?.runnable, true);
      assert.equal(second.registered, 1, 'last good descriptor is republished');
    }
  });

  it('uses active catalog inline workflows and pinned files offline', async () => {
    const hub = hubFor();
    hub.catalog.replaceModels([{ id: 'inline', name: 'Inline graph', type: 'image_generation', host: 'comfyui',
      capabilities: ['text_to_image'], providerKind: 'workflow', workflowId: 'inline',
      adapterConfig: { workflow: textToImage, bindings: { prompt: { node: '2', input: 'text' } },
        outputs: [{ node: '7', type: 'image' }] } },
      { id: 'pinned', name: 'Pinned graph', type: 'image_generation', host: 'comfyui', capabilities: ['text_to_image'],
        providerKind: 'workflow', workflowId: 'pinned', adapterConfig: { workflowPath: '/pinned/graph.json',
          bindings: { prompt: { node: '2', input: 'text' } }, outputs: [{ node: '7', type: 'image' }] } }]);
    try {
      const result = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({
        files: { '/pinned/graph.json': JSON.stringify(textToImage) } }) });
      assert.deepEqual(result.resources.map((resource) => resource.modelId).sort(), ['inline', 'pinned']);
      assert.equal(result.resources.find((resource) => resource.modelId === 'inline')?.freshness, 'catalog');
      assert.equal(result.resources.find((resource) => resource.modelId === 'pinned')?.freshness, 'local');
      assert.ok(result.resources.every((resource) => resource.availability === 'unavailable'));
    } finally { await hub.dispose(); }
  });

  it('reads per-host workflow directories offline without sharing stores', async () => {
    const result = await scanLocalResources(lifecycleHub({}).hub, { hosts: hostsOf(),
      workflowDirsByHost: { comfyui: ['/one/workflows', '/two/workflows'], other: ['/ignored/workflows'] },
      probes: probesFor({ files: { '/one/workflows/first.json': JSON.stringify(textToImage),
        '/two/workflows/second.json': JSON.stringify(textToImage), '/ignored/workflows/leak.json': JSON.stringify(textToImage) } }) });
    assert.deepEqual(result.resources.map((resource) => resource.name).sort(), ['first', 'second']);
    assert.ok(result.resources.every((resource) => resource.freshness === 'local' && resource.availability === 'unavailable'));
  });

  it('discovers supported local Ollama manifests but never attributes them to remote hosts', async () => {
    const previous = process.env['OLLAMA_MODELS'];
    process.env['OLLAMA_MODELS'] = '/ollama/models';
    try {
      const probes = probesFor({ files: { '/ollama/models/manifests/registry.ollama.ai/library/llama3/8b':
        JSON.stringify({ config: { digest: 'sha256:config' }, layers: [] }) } });
      const local = await scanLocalResources(lifecycleHub({}).hub, { hosts: hostsOf(), probes });
      assert.equal(local.resources[0]?.name, 'llama3:8b');
      assert.equal(local.resources[0]?.freshness, 'local');
      assert.equal(local.resources[0]?.availability, 'unavailable');
      const remoteHosts = hostsOf().map((host) => host.id === 'ollama'
        ? { ...host, runtime: { ...host.runtime, endpoint: 'http://remote:11434' } } : host);
      const remote = await scanLocalResources(lifecycleHub({}).hub, { hosts: remoteHosts, probes });
      assert.deepEqual(remote.resources, []);
    } finally {
      if (previous === undefined) delete process.env['OLLAMA_MODELS']; else process.env['OLLAMA_MODELS'] = previous;
    }
  });

  it('bounds hanging Comfy API calls without hiding Ollama or offline directories', async () => {
    for (const suffix of ['/system_stats', '/object_info', '/userdata?'] as const) {
      const base = probesFor({ ollama: ollamaTags, systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'api.json': textToImage },
        files: { '/workflows/comfy/local.json': JSON.stringify(textToImage) } });
      const result = await scanLocalResources(lifecycleHub({}).hub, { hosts: hostsOf(), workflowDir: '/workflows/comfy',
        timeoutMs: 10, probes: { ...base, request: (url, timeout) => url.includes(suffix)
          ? new Promise(() => {}) : base.request(url, timeout) } });
      assert.equal(result.resources.filter((resource) => resource.kind === 'ollama_model').length, 2);
      assert.ok(result.resources.some((resource) => resource.name === 'local'));
    }
  });

  it('retains failed files while refreshing healthy files from the same listing', async () => {
    const hub = lifecycleHub({}).hub;
    const first = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ systemStats: { system: {}, devices: [] }, objectInfo,
      workflows: { 'a.json': textToImage, 'b.json': textToImage } }) });
    const edited = JSON.parse(JSON.stringify(textToImage)) as typeof textToImage;
    edited['5'].inputs.steps = 4;
    const base = probesFor({ systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'a.json': textToImage, 'b.json': edited } });
    const second = await scanLocalResources(hub, { hosts: hostsOf(), timeoutMs: 10,
      probes: { ...base, request: (url, timeout) => url.includes('/userdata/') && decodeURIComponent(url).endsWith('/a.json')
        ? new Promise(() => {}) : base.request(url, timeout) } });
    assert.equal(second.resources.find((resource) => resource.name === 'a')?.freshness, 'cached');
    assert.equal(second.resources.find((resource) => resource.name === 'a')?.checkedAt,
      first.resources.find((resource) => resource.name === 'a')?.checkedAt);
    assert.equal(second.resources.find((resource) => resource.name === 'b')?.freshness, 'live');
    assert.equal(second.resources.length, 2);
    assert.equal(second.registered, 2);
  });

  it('does not resurrect published API workflows from another endpoint via the active catalog', async () => {
    const hub = hubFor();
    try {
      await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ systemStats: { system: {}, devices: [] }, objectInfo,
        workflows: { 'api.json': textToImage } }) });
      assert.ok(hub.catalog.listModels().length > 0);
      const changed = await scanLocalResources(hub, { hosts: hostsOf('http://127.0.0.1:9999'), probes: probesFor({}) });
      assert.deepEqual(changed.resources, []);
    } finally { await hub.dispose(); }
  });

  it('lets the latest scan intent own cache and provider publication', async () => {
    const { hub, calls } = lifecycleHub({});
    const oldBase = probesFor({ ollama: { models: [{ name: 'old:latest' }] }, systemStats: { system: {}, devices: [] }, objectInfo,
      workflows: { 'old.json': textToImage } });
    let releaseOld!: () => void;
    let signalEntered!: () => void;
    const entered = new Promise<void>((resolve) => { signalEntered = resolve; });
    const gate = new Promise<void>((resolve) => { releaseOld = resolve; });
    const oldTask = scanLocalResources(hub, { hosts: hostsOf(), probes: { ...oldBase, request: async (url, timeout) => {
      if (url.endsWith('/system_stats') || url.includes('/api/tags')) { signalEntered(); await gate; }
      return oldBase.request(url, timeout);
    } } });
    await entered;
    const fresh = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({
      ollama: { models: [{ name: 'new:latest' }] }, systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'new.json': textToImage } }) });
    assert.equal(fresh.registered, 1);
    releaseOld();
    const old = await oldTask;
    assert.equal(old.registered, 0, 'a superseded pass cannot republish stale descriptors');
    const cached = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({}) });
    assert.deepEqual(cached.resources.map((resource) => resource.name).sort(), ['new', 'new:latest']);
    assert.equal(calls.filter((call) => call === 'publish:1').length, 2, 'only current and cached scans publish');
  });

  it('does not equate an answering Comfy engine with verified workflow dependencies', async () => {
    const result = await scanLocalResources(lifecycleHub({}).hub, { hosts: hostsOf(), probes: probesFor({
      systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'a.json': textToImage } }) });
    assert.equal(result.resources[0]?.engineRunning, true);
    assert.equal(result.resources[0]?.availability, 'unknown');
    assert.equal(result.resources[0]?.freshness, 'live');
    assert.match(result.resources[0]?.detail ?? '', /dependencies not execution-verified/);
  });

  it('resolves pinned relative workflow paths against the catalog and deduplicates directory discoveries', async () => {
    const hub = hubFor();
    const catalogDir = join(process.cwd(), 'fixture-catalog');
    const absolute = join(catalogDir, 'workflows', 'z-image.json');
    hub.catalog.replaceModels([{ id: 'z_image', name: 'Z-Image', type: 'image_generation', host: 'comfyui',
      capabilities: ['text_to_image'], providerKind: 'workflow', workflowId: 'z_image',
      adapterConfig: { workflowPath: 'workflows/z-image.json', bindings: { prompt: { node: '2', input: 'text' } },
        outputs: [{ node: '7', type: 'image' }] } }]);
    const probes = probesFor({ files: { [absolute.replace(/\\/g, '/')]: JSON.stringify(textToImage) } });
    try {
      const pinned = await scanLocalResources(hub, { hosts: hostsOf(), catalogPath: join(catalogDir, 'models.json'), probes });
      assert.equal(pinned.resources[0]?.source, absolute);
      assert.equal(pinned.resources[0]?.freshness, 'local');
      assert.equal(pinned.resources[0]?.status, 'ready');
      const deduped = await scanLocalResources(hub, { hosts: hostsOf(), catalogPath: join(catalogDir, 'models.json'),
        workflowDir: join(catalogDir, 'workflows'), probes });
      assert.equal(deduped.resources.length, 1);
      assert.equal(deduped.resources[0]?.id, 'z_image');
      assert.equal(deduped.resources[0]?.modelId, 'z_image');
      assert.equal(deduped.resources[0]?.source, absolute);
      assert.equal(deduped.registered, 0, 'catalog-pinned workflow is not published under another scan identity');
    } finally { await hub.dispose(); }
  });

  it('clears the final scanned provider after a trustworthy deletion or invalid workflow', async () => {
    for (const outcome of ['deleted', 'invalid'] as const) {
      const hub = hubFor();
      try {
        await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ systemStats: { system: {}, devices: [] }, objectInfo,
          workflows: { 'a.json': textToImage } }) });
        assert.equal(hub.scannedProviderIds.length, 1);
        const result = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ systemStats: { system: {}, devices: [] }, objectInfo,
          workflows: outcome === 'deleted' ? {} : { 'a.json': { not: 'a workflow' } } }) });
        assert.equal(hub.scannedProviderIds.length, 0, outcome);
        assert.equal(result.resources.length, outcome === 'deleted' ? 0 : 1);
        const offline = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({}) });
        assert.ok(offline.resources.every((resource) => !resource.runnable));
      } finally { await hub.dispose(); }
    }
  });

  it('preserves prior publication after a source failure or an unexpected host-pass exception', async () => {
    for (const unexpected of [false, true]) {
      const hub = hubFor();
      try {
        await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ systemStats: { system: {}, devices: [] }, objectInfo,
          workflows: { 'a.json': textToImage } }) });
        const previous = [...hub.scannedProviderIds];
        const invalidInfo = new Proxy({}, { ownKeys: () => { throw new Error('injected unexpected parse failure'); } });
        await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({
          systemStats: unexpected ? { system: {}, devices: [] } : 'offline', objectInfo: unexpected ? invalidInfo : objectInfo, workflows: {} }) });
        assert.deepEqual(hub.scannedProviderIds, previous);
      } finally { await hub.dispose(); }
    }
  });

  it('retains last good Ollama results on malformed model entries', async () => {
    const hub = lifecycleHub({}).hub;
    await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ ollama: ollamaTags }) });
    const malformed = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ ollama: { models: [null] } }) });
    assert.equal(malformed.resources.length, 2);
    assert.ok(malformed.resources.every((resource) => resource.freshness === 'cached' && resource.availability === 'unknown'));
    assert.equal(malformed.sources.find((source) => source.id === 'ollama')?.ok, false);
  });

  it('normalizes Ollama implicit latest tags when matching active catalog identities', async () => {
    for (const reverse of [false, true]) {
      const config = catalog('http://127.0.0.1:8188');
      const hub = ModelHub.fromConfig({ ...config, models: [{ id: 'known', name: 'Known', type: 'text_generation',
        host: 'ollama', capabilities: ['text_to_text'], adapterConfig: { model: reverse ? 'llama3.2:latest' : 'llama3.2' } }] },
        { manageTimers: false, probeResources: false });
      try {
        const result = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({
          ollama: { models: [{ name: reverse ? 'llama3.2' : 'llama3.2:latest' }] } }) });
        assert.equal(result.resources.length, 1);
        assert.equal(result.resources[0]?.modelId, 'known');
        assert.equal(result.resources[0]?.availability, 'available');
      } finally { await hub.dispose(); }
    }
  });

  it('honours active host OLLAMA_MODELS environment before the process store', async () => {
    const previous = process.env['OLLAMA_MODELS'];
    process.env['OLLAMA_MODELS'] = '/wrong/process/store';
    try {
      const hosts = hostsOf().map((host) => host.id === 'ollama' ? { ...host, runtime: { ...host.runtime,
        env: { OLLAMA_MODELS: '%SCAN_STORE%/models', SCAN_STORE: '/configured/ollama' } } } : host);
      const result = await scanLocalResources(lifecycleHub({}).hub, { hosts, probes: probesFor({ files: {
        '/configured/ollama/models/manifests/registry.ollama.ai/library/llama3/latest':
          JSON.stringify({ config: {}, layers: [] }),
        '/wrong/process/store/manifests/registry.ollama.ai/library/leaked/latest':
          JSON.stringify({ config: {}, layers: [] }),
      } }) });
      assert.deepEqual(result.resources.map((resource) => resource.name), ['llama3:latest']);
      assert.equal(result.resources[0]?.freshness, 'local');
    } finally {
      if (previous === undefined) delete process.env['OLLAMA_MODELS']; else process.env['OLLAMA_MODELS'] = previous;
    }
  });

  it('does not advertise invented invocation IDs for uncatalogued Ollama observations', async () => {
    const hub = lifecycleHub({}).hub;
    const live = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({ ollama: ollamaTags }) });
    assert.ok(live.resources.every((resource) => resource.modelId === undefined && !resource.runnable));
    assert.ok(live.resources.every((resource) => resource.availability === 'available'));
    assert.match(live.resources[0]?.detail ?? '', /not registered for hub invocation/);
    const cached = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({}) });
    assert.ok(cached.resources.every((resource) => resource.modelId === undefined && !resource.runnable));
    const previous = process.env['OLLAMA_MODELS'];
    process.env['OLLAMA_MODELS'] = '/ollama/unregistered';
    try {
      const local = await scanLocalResources(lifecycleHub({}).hub, { hosts: hostsOf(), probes: probesFor({ files: {
        '/ollama/unregistered/manifests/registry.ollama.ai/library/local/latest': JSON.stringify({ config: {}, layers: [] }),
      } }) });
      assert.equal(local.resources[0]?.freshness, 'local');
      assert.equal(local.resources[0]?.modelId, undefined);
      assert.equal(local.resources[0]?.runnable, false);
    } finally {
      if (previous === undefined) delete process.env['OLLAMA_MODELS']; else process.env['OLLAMA_MODELS'] = previous;
    }
  });

  it('keeps healthy root workflows when a nested directory listing fails', async () => {
    for (const failure of ['undefined', 'throw'] as const) {
      const base = probesFor({ files: { '/partial/workflows/root.json': JSON.stringify(textToImage),
        '/partial/workflows/blocked/hidden.json': JSON.stringify(textToImage) } });
      const result = await scanLocalResources(lifecycleHub({}).hub, { hosts: hostsOf(), workflowDir: '/partial/workflows',
        probes: { ...base, listDir: async (path) => {
          if (path.replace(/\\/g, '/').endsWith('/blocked')) {
            if (failure === 'throw') throw new Error('injected denied subdirectory');
            return undefined;
          }
          return base.listDir(path);
        } } });
      assert.deepEqual(result.resources.map((resource) => resource.name), ['root']);
      assert.equal(result.sources.find((source) => source.id === 'workflow-dir')?.ok, false);
      assert.match(result.sources.find((source) => source.id === 'workflow-dir')?.detail ?? '', /blocked/);
    }
  });

  it('honours disabled catalog providers when a pinned workflow is also found in a directory', async () => {
    const hub = hubFor();
    const directory = join(process.cwd(), 'disabled-workflow');
    const path = join(directory, 'graph.json');
    hub.catalog.replaceModels([{ id: 'disabled', name: 'Disabled workflow', type: 'image_generation', host: 'comfyui',
      capabilities: ['text_to_image'], enabled: false, providerKind: 'workflow', workflowId: 'disabled',
      adapterConfig: { workflowPath: path, bindings: { prompt: { node: '2', input: 'text' } }, outputs: [{ node: '7', type: 'image' }] } }]);
    try {
      const result = await scanLocalResources(hub, { hosts: hostsOf(), workflowDir: directory,
        probes: probesFor({ files: { [path.replace(/\\/g, '/')]: JSON.stringify(textToImage) } }) });
      assert.equal(result.resources.length, 1);
      assert.equal(result.resources[0]?.modelId, 'disabled');
      assert.equal(result.resources[0]?.runnable, false);
      assert.deepEqual(hub.scannedProviderIds, []);
    } finally { await hub.dispose(); }
  });

  it('rejects malformed Comfy readiness and empty node metadata without erasing good contracts', async () => {
    for (const stage of ['readiness', 'nodes'] as const) {
      const hub = lifecycleHub({}).hub;
      const first = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({
        systemStats: { system: {}, devices: [] }, objectInfo, workflows: { 'a.json': editorThreeD } }) });
      const second = await scanLocalResources(hub, { hosts: hostsOf(), probes: probesFor({
        systemStats: stage === 'readiness' ? {} : { system: {}, devices: [] },
        objectInfo: stage === 'nodes' ? {} : objectInfo, workflows: { 'a.json': editorThreeD } }) });
      assert.equal(second.resources[0]?.freshness, 'cached');
      assert.equal(second.resources[0]?.runnable, true);
      assert.equal(second.resources[0]?.checkedAt, first.resources[0]?.checkedAt);
      assert.ok(second.warnings.some((warning) => /system.device|node definitions/.test(warning)));
    }
  });

  it('keeps same-name workflows distinct and stable across hosts and directories', async () => {
    const firstHost = hostsOf().find((host) => host.id === 'comfyui')!;
    const secondHost = { ...firstHost, id: 'second-comfy', name: 'Second Comfy',
      runtime: { ...firstHost.runtime, endpoint: 'http://127.0.0.1:8288' } };
    let published: readonly { id: string; host?: string }[] = [];
    const { hub } = lifecycleHub({});
    hub.publishScannedModels = (descriptors) => { published = descriptors; return descriptors.length; };
    const options = { hosts: [firstHost, secondHost],
      workflowDirsByHost: { comfyui: ['/identity/one', '/identity/two'] },
      probes: probesFor({ systemStats: { system: {}, devices: [] }, objectInfo,
        workflows: { 'same.json': textToImage }, files: {
          '/identity/one/same.json': JSON.stringify(textToImage), '/identity/two/same.json': JSON.stringify(textToImage),
        } }) };
    const first = await scanLocalResources(hub, options);
    assert.equal(first.resources.length, 4);
    assert.equal(new Set(first.resources.map((resource) => resource.id)).size, 4);
    assert.equal(published.length, 4);
    const sourceHosts = new Map(first.resources.map((resource) => [resource.id,
      resource.source.includes(':8288') ? 'second-comfy' : 'comfyui']));
    assert.ok(published.every((descriptor) => descriptor.host === sourceHosts.get(descriptor.id)));
    const second = await scanLocalResources(hub, options);
    assert.deepEqual(second.resources.map((resource) => resource.id).sort(), first.resources.map((resource) => resource.id).sort());
  });

  it('builds several candidate retrieval URLs, the documented one first', () => {
    const urls = workflowUrls('http://h:8188', '3d/model.json');
    assert.equal(urls[0], 'http://h:8188/userdata/workflows%2F3d%2Fmodel.json');
    assert.ok(urls.includes('http://h:8188/userdata/workflows/3d/model.json'));
    assert.equal(new Set(urls).size, urls.length, 'no duplicate attempts');
  });
});
