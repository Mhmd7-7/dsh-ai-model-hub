/**
 * Tests for A1111 / Forge runtime discovery.
 *
 * The interesting behaviour is not HTTP plumbing: it is (a) that the *loaded*
 * checkpoint is preferred without any new routing mechanism, and (b) that the
 * two engine-wide endpoints are best-effort, so a build that does not expose
 * `/samplers` still yields models. Both are asserted against a real local server
 * answering in the WebUI's own shapes.
 *
 * @module dsh-ai-model-hub/tests/discovery-a1111
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { IncomingHttpHeaders, ServerResponse } from 'node:http';
import { describe, it } from 'node:test';

import type { ModelCatalogConfig, ModelHost } from '../src/index.ts';
import {
  DISCOVERED_PRIORITY,
  LOADED_PRIORITY,
  ModelCatalog,
  ModelHub,
  createA1111Discoverer,
  estimateA1111Vram,
  mapA1111Model,
  mergeCatalogConfig,
  parseA1111Models,
  parseA1111Options,
  parseA1111Samplers,
} from '../src/index.ts';

/** One captured request. */
interface CapturedRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: IncomingHttpHeaders;
  readonly body: string;
}

/** A test double for an A1111/Forge server. */
interface FakeWebUi {
  readonly url: string;
  readonly requests: CapturedRequest[];
  setResponder(responder: (request: CapturedRequest, response: ServerResponse) => void): void;
  close(): Promise<void>;
}

/** Start a local WebUI double on an ephemeral port. */
async function startWebUi(): Promise<FakeWebUi> {
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

/** The default responder: the three endpoints the WebUI is asked for. */
function responderFor(
  models: unknown,
  options: unknown = {},
  samplers: unknown = [{ name: 'Euler' }, { name: 'DPM++ 2M' }, { name: 'DDIM' }],
  samplerStatus = 200,
): (request: CapturedRequest, response: ServerResponse) => void {
  return (request, response) => {
    const send = (value: unknown, status = 200): void => {
      response.writeHead(status, { 'content-type': 'application/json' });
      response.end(JSON.stringify(value));
    };
    if (request.url === '/sdapi/v1/sd-models') {
      send(models);
      return;
    }
    if (request.url === '/sdapi/v1/samplers') {
      send(samplers, samplerStatus);
      return;
    }
    if (request.url === '/sdapi/v1/options') {
      send(options);
      return;
    }
    response.writeHead(404);
    response.end('{}');
  };
}

/** Register a fake server's URL as an A1111 host. */
function a1111Host(endpoint: string, engine = 'a1111'): ModelHost {
  return {
    id: 'a1111',
    name: 'A1111',
    adapter: 'http_json',
    runtime: { engine, adapter: 'http_json', endpoint, path: '/sdapi/v1/txt2img' },
  };
}

/** A static catalog with one text model and no image models. */
function staticCatalog(endpoint: string): ModelCatalogConfig {
  return {
    version: '1',
    hosts: [a1111Host(endpoint)],
    models: [
      {
        id: 'static_text',
        name: 'Static text model',
        type: 'text_generation',
        capabilities: ['text_to_text'],
        host: 'a1111',
        adapterConfig: { model: 'statically-configured' },
      },
    ],
  };
}

describe('a1111 discovery: parsing an engine response', () => {
  it('reads the checkpoint array', () => {
    const models = parseA1111Models([
      { title: 'sd_xl_base_1.0', model_name: 'sd_xl_base_1.0', hash: 'abc123', filename: '/models/sd_xl_base_1.0.safetensors' },
      { title: 'dreamshaper', model_name: 'dreamshaper', hash: null, filename: '/models/dreamshaper.safetensors' },
    ]);
    assert.equal(models.length, 2);
    assert.equal(models[0]?.title, 'sd_xl_base_1.0');
    assert.equal(models[0]?.filename, '/models/sd_xl_base_1.0.safetensors');
    assert.equal(models[1]?.hash, undefined, 'a null hash is not a string');
  });

  it('never throws for an unexpected shape', () => {
    for (const raw of [undefined, null, 42, 'nope', {}, { models: [] }]) {
      assert.deepEqual(parseA1111Models(raw), []);
    }
    assert.deepEqual(parseA1111Models([null, 7, {}, { title: 'ok' }]).map((entry) => entry.title), ['ok']);
  });

  it('falls back to model_name when the build does not report a title', () => {
    assert.equal(parseA1111Models([{ model_name: 'only-name' }])[0]?.title, 'only-name');
  });

  it('reads sampler names, accepting either field the builds use', () => {
    assert.deepEqual(parseA1111Samplers([{ name: 'Euler' }, { name: 'DPM++ 2M' }, { aliases: 'DDIM' }]), [
      'Euler',
      'DPM++ 2M',
      'DDIM',
    ]);
    assert.deepEqual(parseA1111Samplers([{ name: 'Euler' }, { name: 'Euler' }]), ['Euler'], 'deduplicated');
    assert.deepEqual(parseA1111Samplers({ not: 'an array' }), []);
  });

  it('reads the loaded checkpoint out of the options document', () => {
    assert.equal(parseA1111Options({ sd_model_checkpoint: 'sd_xl_base_1.0', other: 1 }), 'sd_xl_base_1.0');
    assert.equal(parseA1111Options({}), undefined);
    assert.equal(parseA1111Options('nope'), undefined);
  });
});

describe('a1111 discovery: mapping into a descriptor', () => {
  const facts = { samplers: ['Euler', 'DPM++ 2M'] };

  it('claims both image capabilities and declares their IO explicitly', () => {
    const descriptor = mapA1111Model({ title: 'anything' }, a1111Host('http://127.0.0.1:7860'), facts);
    assert.deepEqual(descriptor.capabilities, ['text_to_image', 'image_to_image']);
    // The union of both capabilities' canonical IO. `image_to_image` accepting
    // an `image` is what lets a previously generated artifact chain into it;
    // leaving these to the catalog's defaulting is what silently breaks that.
    assert.deepEqual(descriptor.inputTypes, ['text', 'image']);
    assert.deepEqual(descriptor.outputTypes, ['image']);
    assert.equal(descriptor.type, 'image_editing');
  });

  it('puts the checkpoint title where the http_json adapter reads it', () => {
    const descriptor = mapA1111Model({ title: 'some-checkpoint', hash: 'deadbeef' }, a1111Host('http://127.0.0.1:7860'), facts);
    assert.equal(descriptor.adapterConfig?.['model'], 'some-checkpoint');
    assert.equal(descriptor.adapterConfig?.['checkpointHash'], 'deadbeef');
  });

  it('records the engine sampler list rather than a hardcoded one', () => {
    const descriptor = mapA1111Model({ title: 'x' }, a1111Host('http://127.0.0.1:7860'), facts);
    const discovery = descriptor.adapterConfig?.['discovery'] as { availableSamplers?: string[] } | undefined;
    assert.deepEqual(discovery?.availableSamplers, ['Euler', 'DPM++ 2M']);
  });

  it('prefers the loaded checkpoint over the others', () => {
    const loaded = mapA1111Model(
      { title: 'loaded-one' },
      a1111Host('http://127.0.0.1:7860'),
      { ...facts, loadedCheckpoint: 'loaded-one' },
    );
    const other = mapA1111Model({ title: 'other-one' }, a1111Host('http://127.0.0.1:7860'), {
      ...facts,
      loadedCheckpoint: 'loaded-one',
    });
    assert.equal(loaded.priority, LOADED_PRIORITY);
    assert.equal(other.priority, DISCOVERED_PRIORITY);
    assert.ok(LOADED_PRIORITY < DISCOVERED_PRIORITY, 'the loaded model sorts first among discovered candidates');
    assert.ok(DISCOVERED_PRIORITY > 100, 'and both sort after a static model relying on the catalog default');
    assert.ok(loaded.tags?.includes('loaded'));
  });

  it('matches the loaded checkpoint even when the engine decorates the name', () => {
    // A title the engine decorated with a hash still matches the option value.
    const decorated = mapA1111Model({ title: 'some-model' }, a1111Host('http://127.0.0.1:7860'), {
      ...facts,
      loadedCheckpoint: 'some-model [deadbeef]',
    });
    assert.equal(decorated.priority, LOADED_PRIORITY);

    // An unrelated title does not, however similar it looks.
    const other = mapA1111Model(
      { title: 'another-model', filename: '/m/another-model.safetensors' },
      a1111Host('http://127.0.0.1:7860'),
      { ...facts, loadedCheckpoint: 'some-model [deadbeef]' },
    );
    assert.equal(other.priority, DISCOVERED_PRIORITY);

    // And a filename reported instead of a title is compared too.
    const byFilename = mapA1111Model(
      { title: 'unrelated', filename: 'some-model.safetensors' },
      a1111Host('http://127.0.0.1:7860'),
      { ...facts, loadedCheckpoint: 'some-model.safetensors' },
    );
    assert.equal(byFilename.priority, LOADED_PRIORITY);
  });

  it('estimates VRAM from the filename, as a documented heuristic', () => {
    assert.equal(estimateA1111Vram(['/models/sd_xl_base_1.0.safetensors']), 6);
    assert.equal(estimateA1111Vram(['/models/flux1-dev.safetensors']), 12);
    assert.equal(estimateA1111Vram(['/models/v1-5-pruned.ckpt']), 2);
    assert.equal(estimateA1111Vram(['/models/something-unknown.safetensors']), 4);
    assert.equal(estimateA1111Vram([undefined, undefined]), 4);
    // The estimate reaches the descriptor's resources, and a GPU is required.
    const descriptor = mapA1111Model({ title: 'sd_xl_base_1.0' }, a1111Host('http://127.0.0.1:7860'), facts);
    assert.equal(descriptor.resources?.vramGb, 6);
    assert.equal(descriptor.resources?.requiresGpu, true);
  });

  it('derives a deterministic id and keeps the title verbatim', () => {
    const host = a1111Host('http://127.0.0.1:7860');
    const first = mapA1111Model({ title: 'SD XL Base [1.0]' }, host, facts);
    const second = mapA1111Model({ title: 'SD XL Base [1.0]' }, host, facts);
    assert.equal(first.id, second.id);
    assert.match(first.id, /^[a-z0-9][a-z0-9._-]*$/);
    assert.equal(first.adapterConfig?.['model'], 'SD XL Base [1.0]');
  });
});

describe('a1111 discovery: against a live server', () => {
  it('enumerates checkpoints and marks the loaded one', async () => {
    const engine = await startWebUi();
    try {
      engine.setResponder(
        responderFor(
          [
            { title: 'alpha-xl', model_name: 'alpha-xl', hash: 'aaa', filename: '/m/alpha-xl.safetensors' },
            { title: 'beta-15', model_name: 'beta-15', hash: 'bbb', filename: '/m/beta-15.safetensors' },
          ],
          { sd_model_checkpoint: 'beta-15' },
        ),
      );
      const descriptors = await createA1111Discoverer({ requestTimeoutMs: 2000 }).discover(
        a1111Host(engine.url),
        new AbortController().signal,
      );
      assert.equal(descriptors.length, 2);
      const loaded = descriptors.filter((descriptor) => descriptor.priority === LOADED_PRIORITY);
      assert.equal(loaded.length, 1);
      assert.equal(loaded[0]?.name, 'beta-15');
      assert.deepEqual(
        descriptors.map((descriptor) => descriptor.id),
        ['a1111-alpha-xl', 'a1111-beta-15'],
        'input order is preserved',
      );
      // Every endpoint was consulted.
      for (const path of ['/sdapi/v1/sd-models', '/sdapi/v1/samplers', '/sdapi/v1/options']) {
        assert.ok(engine.requests.some((request) => request.url === path), `expected a GET to ${path}`);
      }
    } finally {
      await engine.close();
    }
  });

  it('still discovers models when the optional endpoints are missing', async () => {
    const engine = await startWebUi();
    try {
      engine.setResponder(
        responderFor([{ title: 'only-one', filename: '/m/only-one.safetensors' }], {}, [], 404),
      );
      const descriptors = await createA1111Discoverer({ requestTimeoutMs: 2000 }).discover(
        a1111Host(engine.url),
        new AbortController().signal,
      );
      assert.equal(descriptors.length, 1);
      const discovery = descriptors[0]?.adapterConfig?.['discovery'] as { availableSamplers?: string[] } | undefined;
      assert.equal(discovery?.availableSamplers, undefined, 'no sampler list is not a failure');
      assert.equal(descriptors[0]?.priority, DISCOVERED_PRIORITY, 'with no options answer, nothing is known to be loaded');
    } finally {
      await engine.close();
    }
  });

  it('reports an unreachable engine as a failure the registry contains', async () => {
    const engine = await startWebUi();
    const url = engine.url;
    await engine.close();
    await assert.rejects(
      () => createA1111Discoverer({ requestTimeoutMs: 1000 }).discover(a1111Host(url), new AbortController().signal),
      (error: Error) => {
        assert.match(error.message, /could not list checkpoints/);
        return true;
      },
    );
  });

  it('reports a malformed body as zero models rather than a crash', async () => {
    const engine = await startWebUi();
    try {
      engine.setResponder(responderFor({ error: 'not an array' }));
      const descriptors = await createA1111Discoverer({ requestTimeoutMs: 2000 }).discover(
        a1111Host(engine.url),
        new AbortController().signal,
      );
      assert.deepEqual(descriptors, []);
    } finally {
      await engine.close();
    }
  });

  it('answers to the engine aliases one WebUI family goes by', async () => {
    const engine = await startWebUi();
    try {
      engine.setResponder(responderFor([{ title: 'via-forge', filename: '/m/via-forge.safetensors' }]));
      // `runtime.engine` is an operator label; Forge is the same REST surface
      // under a different one. The alias is what makes the host discoverable at
      // all, and the host's own label is what prefixes the ids.
      const { hub } = await ModelHub.fromConfigAndDiscovery(
        { version: '1', hosts: [a1111Host(engine.url, 'forge')], models: [] },
        { manageTimers: false, discoveryTimeoutMs: 2000, log: () => {} },
      );
      assert.deepEqual(hub.catalog.listModelIds(), ['forge-via-forge']);
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });
});

describe('a1111 discovery: chaining and catalog integration', () => {
  it('publishes a discovered checkpoint as an image_to_image candidate that accepts an image', async () => {
    const engine = await startWebUi();
    try {
      engine.setResponder(responderFor([{ title: 'chained-xl', filename: '/m/chained-xl.safetensors' }], {}));
      const { hub } = await ModelHub.fromConfigAndDiscovery(staticCatalog(engine.url), {
        manageTimers: false,
        discoveryTimeoutMs: 2000,
        log: () => {},
      });
      const model = hub.catalog.requireModel('a1111-chained-xl');
      assert.deepEqual(model.inputTypes, ['text', 'image']);
      // This is the property a chained workflow depends on: the output of a
      // text_to_image model is an `image`, and this model accepts one.
      assert.ok(model.inputTypes.includes('image'));
      assert.equal(hub.catalog.findModelsByCapability('image_to_image').length, 1);
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('lets a static checkpoint descriptor win over the same discovered model', () => {
    const host = a1111Host('http://127.0.0.1:7860');
    const discovered = mapA1111Model({ title: 'pinned', filename: '/m/pinned.safetensors' }, host, { samplers: [] });
    const staticConfig: ModelCatalogConfig = {
      version: '1',
      hosts: [host],
      models: [
        {
          id: discovered.id,
          name: 'Operator-pinned checkpoint',
          type: 'image_generation',
          capabilities: ['text_to_image'],
          host: host.id,
          adapterConfig: { model: 'pinned', steps: 8, cfg: 1 },
          limits: { resolutions: ['1024x1024'] },
        },
      ],
    };
    const catalog = new ModelCatalog(mergeCatalogConfig(staticConfig, [discovered]));
    const resolved = catalog.requireModel(discovered.id);
    assert.equal(resolved.name, 'Operator-pinned checkpoint');
    assert.equal(resolved.adapterConfig['steps'], 8);
    assert.deepEqual(resolved.inputTypes, ['text'], 'the static IO declaration survives, not the discovered one');
    assert.deepEqual(catalog.loadDiagnostics, []);
  });
});
