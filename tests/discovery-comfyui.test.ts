/**
 * Tests for ComfyUI runtime discovery.
 *
 * Two things are worth testing here and neither is HTTP plumbing: that the
 * *files* come out of the node graph's own enumerations (so no checkpoint name
 * appears in the source), and that the **capability** set is decided by which
 * node packs are installed rather than by anything about a checkpoint.
 *
 * The auto-generated default graph is asserted on its *shape* — that it wires a
 * checkpoint loader to a sampler to a decoder to a saver, and that the prompt
 * nodes sit on the sampler's positive/negative links — because that structure,
 * not the specific node ids, is what the adapter's graph surgery depends on.
 *
 * @module dsh-ai-model-hub/tests/discovery-comfyui
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { IncomingHttpHeaders, ServerResponse } from 'node:http';
import { describe, it } from 'node:test';

import type { ModelCatalogConfig, ModelHost } from '../src/index.ts';
import {
  CAPABILITY_SIGNALS,
  DISCOVERED_PRIORITY,
  ModelCatalog,
  ModelHub,
  buildDefaultGraph,
  capabilitiesForComfyModel,
  createComfyUiDiscoverer,
  describeIntrospection,
  estimateComfyVram,
  mapComfyWeightFile,
  mergeCatalogConfig,
  parseComfyObjectInfo,
} from '../src/index.ts';

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
 * One node's entry in `/object_info`.
 *
 * The shape mirrors what ComfyUI really answers with: an `input.required` map
 * whose entries are `[typeOrOptions, opts]` pairs.
 *
 * @param required - the `input.required` map.
 * @returns the node's object-info entry.
 */
function node(required: Record<string, unknown>): { input: { required: Record<string, unknown> } } {
  return { input: { required } };
}

/** An enumerable file input, exactly as ComfyUI documents it. */
function fileEnum(options: readonly string[], extra: Record<string, unknown> = {}): unknown[] {
  return [options, extra];
}

/** The loader nodes every ComfyUI install has. */
function loaderNodes(checkpoints: readonly string[], unets: readonly string[] = [], loras: readonly string[] = []): Record<string, unknown> {
  return {
    CheckpointLoaderSimple: node({ ckpt_name: fileEnum(checkpoints) }),
    UNETLoader: node({ unet_name: fileEnum(unets), weight_dtype: [['default', 'fp8_e4m3fn'], {}] }),
    LoraLoader: node({ lora_name: fileEnum(loras), strength_model: ['FLOAT', {}] }),
  };
}

/** The nodes a minimal text-to-image graph needs. */
const CORE_GRAPH_NODES: Record<string, unknown> = {
  EmptyLatentImage: node({ width: ['INT', {}], height: ['INT', {}], batch_size: ['INT', {}] }),
  CLIPTextEncode: node({ clip: ['CLIP', {}], text: ['STRING', { multiline: true }] }),
  KSampler: node({
    model: ['MODEL', {}],
    positive: ['CONDITIONING', {}],
    negative: ['CONDITIONING', {}],
    latent_image: ['LATENT', {}],
    seed: ['INT', {}],
    steps: ['INT', {}],
    cfg: ['FLOAT', {}],
    sampler_name: [['euler', 'dpmpp_2m'], {}],
    scheduler: [['normal', 'karras'], {}],
    denoise: ['FLOAT', {}],
  }),
  VAEDecode: node({ samples: ['LATENT', {}], vae: ['VAE', {}] }),
  SaveImage: node({ images: ['IMAGE', {}], filename_prefix: ['STRING', {}] }),
};

/** A typical install: loaders, the core graph, and a couple of node packs. */
function typicalObjectInfo(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    ...loaderNodes(['alpha-xl.safetensors', 'beta-15.ckpt'], ['gamma-unet.safetensors'], ['style.safetensors']),
    ...CORE_GRAPH_NODES,
    VAEEncode: node({ pixels: ['IMAGE', {}], vae: ['VAE', {}] }),
    SaveGLB: node({ mesh: ['MESH', {}], filename_prefix: ['STRING', {}] }),
    // A real 3D *generator*, which is what makes the 3D capabilities genuine.
    // `SaveGLB` alone only proves a mesh could be written out.
    Hunyuan3Dv2Conditioning: node({}),
    ...overrides,
  };
}

/** Register a fake server's URL as a ComfyUI host. */
function comfyHost(endpoint: string, id = 'comfyui'): ModelHost {
  return {
    id,
    name: 'ComfyUI',
    adapter: 'comfyui',
    runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint, path: '/prompt' },
  };
}

/** A static catalog with one text model and no image models. */
function staticCatalog(endpoint: string): ModelCatalogConfig {
  return {
    version: '1',
    hosts: [comfyHost(endpoint)],
    models: [
      {
        id: 'static_text',
        name: 'Static text model',
        type: 'text_generation',
        capabilities: ['text_to_text'],
        host: 'comfyui',
        adapterConfig: { model: 'statically-configured' },
      },
    ],
  };
}

describe('comfyui discovery: reading the node graph', () => {
  it('enumerates checkpoint files from the loader input enums', () => {
    const introspection = parseComfyObjectInfo(typicalObjectInfo());
    const checkpoints = introspection.files.filter((file) => file.kind === 'checkpoint').map((file) => file.filename);
    assert.deepEqual(checkpoints, ['alpha-xl.safetensors', 'beta-15.ckpt']);
    // The diffusion-only and LoRA enumerations are found too, by field name
    // rather than by node class, so a wrapper node is still understood.
    assert.deepEqual(
      introspection.files.filter((file) => file.kind === 'diffusionModel').map((file) => file.filename),
      ['gamma-unet.safetensors'],
    );
    assert.deepEqual(
      introspection.files.filter((file) => file.kind === 'lora').map((file) => file.filename),
      ['style.safetensors'],
    );
  });

  it('finds loader nodes it has never heard of, by looking at their input fields', () => {
    const introspection = parseComfyObjectInfo({
      SomeThirdPartyLoaderNode: node({ ckpt_name: fileEnum(['found-me.safetensors']) }),
      NestedLoaderWrapper: node({ lora_name: fileEnum(['nested-lora.safetensors']) }),
    });
    assert.deepEqual(introspection.files.map((file) => file.filename), ['found-me.safetensors', 'nested-lora.safetensors']);
  });

  it('deduplicates a file two nodes both enumerate', () => {
    const introspection = parseComfyObjectInfo({
      CheckpointLoaderSimple: node({ ckpt_name: fileEnum(['shared.safetensors']) }),
      CheckpointLoaderA: node({ ckpt_name: fileEnum(['shared.safetensors']) }),
    });
    assert.equal(introspection.files.length, 1);
  });

  it('never throws for an unexpected shape', () => {
    for (const raw of [undefined, null, 42, 'nope', [], '{}']) {
      const introspection = parseComfyObjectInfo(raw);
      assert.deepEqual(introspection.files, []);
      assert.deepEqual(introspection.nodeClasses, []);
    }
    // Nodes with no inputs, or inputs of the wrong shape, are skipped.
    const introspection = parseComfyObjectInfo({
      NoInputs: {},
      BadInputs: { input: 'nope' },
      BadRequired: { input: { required: 'nope' } },
      BadField: node({ ckpt_name: 'not an array' }),
      Good: node({ ckpt_name: fileEnum(['good.safetensors']) }),
    });
    assert.deepEqual(introspection.files.map((file) => file.filename), ['good.safetensors']);
  });

  it('never invents a model from a type marker in an enum slot', () => {
    // A dynamically-typed input is declared as `["COMBO", {...}]`. Reading that
    // as a filename publishes a model named after a type, which corresponds to no
    // file on disk — observed live as a phantom `comfyui-combo` checkpoint.
    const introspection = parseComfyObjectInfo({
      LTXVAudioVAELoader: node({ ckpt_name: fileEnum(['COMBO']) }),
      SomeLoader: node({ unet_name: fileEnum(['COMBO', 'real-model.safetensors']) }),
      ScalarLoader: node({ lora_name: fileEnum(['INT', 'FLOAT', 'STRING', 'actual-lora.safetensors']) }),
    });
    assert.deepEqual(introspection.files.map((file) => file.filename), ['real-model.safetensors', 'actual-lora.safetensors']);
  });

  it('tolerates a spec that is only the option list', () => {
    const introspection = parseComfyObjectInfo({ Bare: node({ ckpt_name: ['bare.safetensors'] }) });
    assert.deepEqual(introspection.files.map((file) => file.filename), ['bare.safetensors']);
  });
});

describe('comfyui discovery: capability signals', () => {
  it('always claims text_to_image, because a checkpoint is a text-to-image path', () => {
    const capabilities = capabilitiesForComfyModel(parseComfyObjectInfo(typicalObjectInfo()));
    assert.equal(capabilities[0], 'text_to_image');
  });

  it('does not turn a mesh writer into a generation capability', () => {
    // The false positive this guards against, found on a real install:
    // ComfyUI's mesh *writers* are core (SaveGLB, Save3DAdvanced, MeshToFile3D),
    // but every image-to-model *generator* is an optional pack or a cloud API
    // node. Keyed on the writer, discovery advertised image_to_3d on a machine
    // with no way to serve it — and because a claimed capability routes work into
    // a guaranteed failure, that is worse than omitting it.
    const writersOnly = parseComfyObjectInfo({
      ...loaderNodes(['a.safetensors']),
      SaveGLB: node({ mesh: ['MESH', {}] }),
      Save3DAdvanced: node({ mesh: ['MESH', {}] }),
      MeshToFile3D: node({ mesh: ['MESH', {}] }),
      DecimateMesh: node({ mesh: ['MESH', {}] }),
      Load3D: node({ model_file: ['STRING', {}] }),
    });
    assert.deepEqual(
      capabilitiesForComfyModel(writersOnly),
      ['text_to_image'],
      'writing a mesh is not generating one',
    );
    // The writers are still recorded, so an operator can see why nothing is claimed.
    assert.ok(writersOnly.signals.some((signal) => /mesh export/.test(signal)));
    assert.deepEqual(writersOnly.capabilities, []);
  });

  it('claims 3D when a generator is actually installed', () => {
    const generators = [
      'Hunyuan3Dv2Conditioning',
      'TripoImageToModelNode',
      'MeshyImageToModelNode',
      'Rodin3D_Gen25_Image',
    ];
    for (const generator of generators) {
      const introspection = parseComfyObjectInfo({ ...loaderNodes(['a.safetensors']), [generator]: node({}) });
      assert.ok(
        introspection.capabilities.includes('image_to_3d') && introspection.capabilities.includes('text_to_3d'),
        `${generator} should grant 3D generation`,
      );
    }
  });

  it('claims a capability only when the node pack that proves it is installed', () => {
    const without = capabilitiesForComfyModel(parseComfyObjectInfo(loaderNodes(['a.safetensors'])));
    assert.deepEqual(without, ['text_to_image'], 'no 3D node pack, so no 3D claim');

    const with3d = capabilitiesForComfyModel(parseComfyObjectInfo(typicalObjectInfo()));
    assert.ok(with3d.includes('text_to_3d'), 'Hunyuan3Dv2 is a 3D generator');
    assert.ok(with3d.includes('image_to_3d'));
    assert.ok(with3d.includes('image_to_image'), 'VAEEncode is evidence an image can be ingested');
  });

  it('recognises video and audio packs by their export nodes', () => {
    const video = capabilitiesForComfyModel(
      parseComfyObjectInfo({ ...loaderNodes(['a.safetensors']), SaveWEBM: node({ images: ['IMAGE', {}] }) }),
    );
    assert.ok(video.includes('video_generation'));
    const audio = capabilitiesForComfyModel(
      parseComfyObjectInfo({ ...loaderNodes(['a.safetensors']), SaveAudio: node({ audio: ['AUDIO', {}] }) }),
    );
    assert.ok(audio.includes('audio_generation'));
  });

  it('names the signals that fired so an operator can see why', () => {
    const introspection = parseComfyObjectInfo(typicalObjectInfo());
    assert.ok(introspection.signals.some((signal) => /mesh export/.test(signal)));
    assert.match(describeIntrospection(introspection), /node class\(es\)/);
  });

  it('keeps the signal table to node classes, never model names', () => {
    // A guard on the constraint itself: every pattern must be about a node
    // class, so nothing here can become a model list by accident. A signal that
    // grants nothing must be explicitly marked as provenance-only, so that
    // "records something" and "advertises a capability" cannot be confused again.
    for (const signal of CAPABILITY_SIGNALS) {
      assert.ok(signal.label.length > 0);
      assert.ok(signal.capabilities.length > 0 || signal.grants === false, `${signal.label} grants nothing but is not marked as such`);
      assert.ok(!/safetensors|\.ckpt|\.pt\b/i.test(signal.pattern.source), 'a checkpoint filename must never appear in the signal table');
    }
  });
});

describe('comfyui discovery: the default graph', () => {
  it('builds a one-checkpoint graph when every node it needs exists', () => {
    const introspection = parseComfyObjectInfo(typicalObjectInfo());
    const graph = buildDefaultGraph(introspection, 'alpha-xl.safetensors');
    assert.ok(graph, 'the typical install can run the default shape');

    const nodes = graph as Record<string, { class_type: string; inputs: Record<string, unknown> }>;
    const byClass = (className: string): [string, { class_type: string; inputs: Record<string, unknown> }] => {
      const found = Object.entries(nodes).find(([, value]) => value.class_type === className);
      assert.ok(found, `expected a ${className} node`);
      return found;
    };

    const [loaderId, loader] = byClass('CheckpointLoaderSimple');
    assert.equal(loader.inputs['ckpt_name'], 'alpha-xl.safetensors');

    // The adapter's graph surgery finds the prompt node from the sampler's
    // `positive` link, so these two links are the contract that matters.
    const [, sampler] = byClass('KSampler');
    assert.equal((sampler.inputs['model'] as unknown[])[0], loaderId);
    const positiveId = (sampler.inputs['positive'] as unknown[])[0];
    const negativeId = (sampler.inputs['negative'] as unknown[])[0];
    assert.equal(nodes[String(positiveId)]?.class_type, 'CLIPTextEncode');
    assert.equal(nodes[String(negativeId)]?.class_type, 'CLIPTextEncode');

    // And the graph ends in something that writes an image.
    const [, save] = byClass('SaveImage');
    assert.equal((save.inputs['images'] as unknown[])[0], byClass('VAEDecode')[0]);
  });

  it('returns nothing rather than a graph this install cannot run', () => {
    const missingSampler = { ...typicalObjectInfo() };
    delete missingSampler['KSampler'];
    assert.equal(
      buildDefaultGraph(parseComfyObjectInfo(missingSampler), 'alpha-xl.safetensors'),
      undefined,
      'queueing a graph with an unknown node class fails deep inside ComfyUI; saying so up front is better',
    );

    const missingPrompt = { ...typicalObjectInfo() };
    delete missingPrompt['CLIPTextEncode'];
    assert.equal(buildDefaultGraph(parseComfyObjectInfo(missingPrompt), 'x'), undefined);
  });
});

describe('comfyui discovery: mapping into a descriptor', () => {
  const introspection = parseComfyObjectInfo(typicalObjectInfo());

  it('sets input/output types from CAPABILITY_IO for the chained case', () => {
    const descriptor = mapComfyWeightFile({ filename: 'alpha-xl.safetensors', kind: 'checkpoint' }, comfyHost('http://127.0.0.1:8188'), introspection);
    assert.deepEqual(descriptor.inputTypes, ['text', 'image']);
    assert.deepEqual(descriptor.outputTypes, ['image', 'model_3d']);
    assert.equal(descriptor.type, 'three_d_generation');
    // The two properties a chained workflow checks: a text_to_image output is
    // an `image`, and this model accepts one; a 3D consumer wants `model_3d`,
    // and this model produces one.
    assert.ok(descriptor.inputTypes?.includes('image'));
    assert.ok(descriptor.outputTypes?.includes('model_3d'));
  });

  it('attaches the generated graph so the model can be invoked without a template', () => {
    const descriptor = mapComfyWeightFile({ filename: 'alpha-xl.safetensors', kind: 'checkpoint' }, comfyHost('http://127.0.0.1:8188'), introspection);
    assert.ok(descriptor.adapterConfig?.['workflow'], 'a default graph is present');
    assert.match(descriptor.notes ?? '', /default one-checkpoint graph/);
  });

  it('falls back cleanly to "needs a workflow template" when the graph is exotic', () => {
    const minimal = parseComfyObjectInfo(loaderNodes(['plain.safetensors']));
    const descriptor = mapComfyWeightFile({ filename: 'plain.safetensors', kind: 'checkpoint' }, comfyHost('http://127.0.0.1:8188'), minimal);
    assert.equal(descriptor.adapterConfig?.['workflow'], undefined);
    assert.match(descriptor.notes ?? '', /needs a template|needs a workflow template/i);
    assert.deepEqual(descriptor.capabilities, ['text_to_image']);
  });

  it('estimates VRAM from the filename, as a documented heuristic', () => {
    assert.equal(estimateComfyVram('flux1-dev.safetensors'), 16);
    assert.equal(estimateComfyVram('some-xl-model.safetensors'), 8);
    assert.equal(estimateComfyVram('v1-5-pruned.ckpt'), 4);
    assert.equal(estimateComfyVram('unknown.safetensors'), 6);
    const descriptor = mapComfyWeightFile({ filename: 'some-xl-model.safetensors', kind: 'checkpoint' }, comfyHost('http://127.0.0.1:8188'), introspection);
    assert.equal(descriptor.resources?.vramGb, 8);
    assert.equal(descriptor.resources?.requiresGpu, true);
  });

  it('derives a deterministic id and sits below the catalog default priority', () => {
    const host = comfyHost('http://127.0.0.1:8188');
    const first = mapComfyWeightFile({ filename: 'A Model [v1].safetensors', kind: 'checkpoint' }, host, introspection);
    const second = mapComfyWeightFile({ filename: 'A Model [v1].safetensors', kind: 'checkpoint' }, host, introspection);
    assert.equal(first.id, second.id);
    assert.match(first.id, /^[a-z0-9][a-z0-9._-]*$/);
    assert.equal(first.priority, DISCOVERED_PRIORITY);
    assert.ok(DISCOVERED_PRIORITY > 100);
  });
});

describe('comfyui discovery: against a live server', () => {
  it('publishes one descriptor per checkpoint and consults /object_info once', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((_request, response) => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify(typicalObjectInfo()));
      });
      const descriptors = await createComfyUiDiscoverer({ requestTimeoutMs: 3000 }).discover(
        comfyHost(comfy.url),
        new AbortController().signal,
      );
      assert.deepEqual(
        descriptors.map((descriptor) => descriptor.name),
        ['alpha-xl.safetensors', 'beta-15.ckpt'],
        'one model per checkpoint, in enumeration order',
      );
      assert.equal(comfy.requests.filter((request) => request.url === '/object_info').length, 1);
    } finally {
      await comfy.close();
    }
  });

  it('does not publish a LoRA or a bare UNET as a routable model', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((_request, response) => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify(typicalObjectInfo()));
      });
      const descriptors = await createComfyUiDiscoverer({ requestTimeoutMs: 3000 }).discover(
        comfyHost(comfy.url),
        new AbortController().signal,
      );
      const names = descriptors.map((descriptor) => descriptor.name);
      assert.ok(!names.includes('style.safetensors'), 'a LoRA is not a model this hub can route to on its own');
      assert.ok(!names.includes('gamma-unet.safetensors'), 'a bare UNET has no graph the adapter could run');
    } finally {
      await comfy.close();
    }
  });

  it('reports an unreachable engine as a failure the registry contains', async () => {
    const comfy = await startComfy();
    const url = comfy.url;
    await comfy.close();
    await assert.rejects(
      () => createComfyUiDiscoverer({ requestTimeoutMs: 1000 }).discover(comfyHost(url), new AbortController().signal),
      (error: Error) => {
        assert.match(error.message, /could not read \/object_info/);
        return true;
      },
    );
  });

  it('reports a non-JSON body as zero models rather than a crash', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((_request, response) => {
        response.writeHead(200, { 'content-type': 'text/html' });
        response.end('<html>ComfyUI</html>');
      });
      await assert.rejects(
        () => createComfyUiDiscoverer({ requestTimeoutMs: 1500 }).discover(comfyHost(comfy.url), new AbortController().signal),
        (error: Error) => {
          assert.match(error.message, /could not read \/object_info/);
          return true;
        },
      );
    } finally {
      await comfy.close();
    }
  });
});

describe('comfyui discovery: catalog integration', () => {
  it('publishes discovered checkpoints the router can select, next to a static text model', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((_request, response) => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify(typicalObjectInfo()));
      });
      const { hub } = await ModelHub.fromConfigAndDiscovery(staticCatalog(comfy.url), {
        manageTimers: false,
        discoveryTimeoutMs: 3000,
        log: () => {},
      });
      const ids = hub.catalog.listModelIds();
      assert.deepEqual(ids, ['static_text', 'comfyui-alpha-xl.safetensors', 'comfyui-beta-15.ckpt']);
      assert.equal(hub.catalog.findModelsByCapability('image_to_3d').length, 2);
      assert.equal(hub.catalog.findModelsByCapability('text_to_3d').length, 2);
      // The adapter can actually run the generated graph.
      const model = hub.catalog.requireModel('comfyui-alpha-xl.safetensors');
      assert.equal(hub.adapters.require('comfyui').supports(model).ok, true);
      await hub.dispose();
    } finally {
      await comfy.close();
    }
  });

  it('keeps a checkpoint the install cannot run as a routable-but-unsupported model', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((_request, response) => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify(loaderNodes(['plain.safetensors'])));
      });
      const { hub } = await ModelHub.fromConfigAndDiscovery(staticCatalog(comfy.url), {
        manageTimers: false,
        discoveryTimeoutMs: 3000,
        log: () => {},
      });
      const model = hub.catalog.requireModel('comfyui-plain.safetensors');
      const support = hub.adapters.require('comfyui').supports(model);
      assert.equal(support.ok, false);
      assert.match(support.ok === false ? support.reason : '', /workflow template/);
      await hub.dispose();
    } finally {
      await comfy.close();
    }
  });

  it('lets a static descriptor win over the same discovered checkpoint', () => {
    const host = comfyHost('http://127.0.0.1:8188');
    const introspection = parseComfyObjectInfo(typicalObjectInfo());
    const discovered = mapComfyWeightFile({ filename: 'alpha-xl.safetensors', kind: 'checkpoint' }, host, introspection);
    const staticConfig: ModelCatalogConfig = {
      version: '1',
      hosts: [host],
      models: [
        {
          id: discovered.id,
          name: 'Tuned checkpoint',
          type: 'image_generation',
          capabilities: ['text_to_image'],
          host: host.id,
          adapterConfig: { workflowPath: 'config/workflows/mine.api.json', steps: 8 },
          priority: 10,
        },
      ],
    };
    const catalog = new ModelCatalog(mergeCatalogConfig(staticConfig, [discovered]));
    const resolved = catalog.requireModel(discovered.id);
    assert.equal(resolved.name, 'Tuned checkpoint');
    assert.equal(resolved.adapterConfig['workflowPath'], 'config/workflows/mine.api.json');
    assert.equal(resolved.priority, 10);
    assert.deepEqual(resolved.capabilities, ['text_to_image']);
    assert.deepEqual(catalog.loadDiagnostics, []);
  });
});
