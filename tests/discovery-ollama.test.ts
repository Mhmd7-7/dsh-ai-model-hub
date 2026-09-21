/**
 * Tests for Ollama runtime discovery.
 *
 * Every test here runs against a **real local HTTP server** returning canned
 * JSON in the shape Ollama actually answers with, rather than a mocked `fetch`,
 * for the same reason the adapter tests do: the things most likely to be wrong
 * are the request shape (`/api/show` takes `{ name }` in the body), whether a
 * non-2xx is reported as unreachable rather than parsed as JSON, and whether
 * cancellation reaches the socket. A mock would paper over all three.
 *
 * The two halves are tested separately as well, because they are deliberately
 * separate: `parse*` reads an engine response, `map*` decides what a descriptor
 * looks like. The parsing tests need no server at all.
 *
 * @module dsh-ai-model-hub/tests/discovery-ollama
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { IncomingHttpHeaders, ServerResponse } from 'node:http';
import { describe, it } from 'node:test';

import type { ModelCatalogConfig, ModelDescriptor, ModelHost } from '../src/index.ts';
import {
  DISCOVERED_PRIORITY,
  ModelCatalog,
  ModelHub,
  createOllamaDiscoverer,
  mapOllamaModel,
  mergeCatalogConfig,
  parseOllamaShow,
  parseOllamaTags,
  slugifyModelId,
} from '../src/index.ts';

/** One captured request, so assertions can be made about what was actually sent. */
interface CapturedRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: IncomingHttpHeaders;
  readonly body: string;
}

/** A test double for an Ollama server. */
interface FakeOllama {
  readonly url: string;
  readonly requests: CapturedRequest[];
  setResponder(responder: (request: CapturedRequest, response: ServerResponse) => void): void;
  close(): Promise<void>;
}

/** Start a local Ollama double on an ephemeral port. */
async function startOllama(): Promise<FakeOllama> {
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
      if (responder !== undefined) {
        responder(captured, response);
        return;
      }
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end('{}');
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

/** A reply helper: JSON with a 200. */
function sendJson(response: ServerResponse, value: unknown, status = 200): void {
  response.writeHead(status, { 'content-type': 'application/json' });
  response.end(JSON.stringify(value));
}

/** The descriptors a `/api/tags` + `/api/show` pair produces. */
function responderFor(
  tags: unknown,
  shows: Readonly<Record<string, unknown>>,
  onMissing?: 'empty' | 'error',
): (request: CapturedRequest, response: ServerResponse) => void {
  return (request, response) => {
    if (request.url === '/api/tags') {
      sendJson(response, tags);
      return;
    }
    if (request.url === '/api/show') {
      const name = (JSON.parse(request.body || '{}') as { name?: string }).name ?? '';
      const show = shows[name];
      if (show === undefined) {
        if (onMissing === 'error') sendJson(response, { error: 'model not found' }, 404);
        else sendJson(response, {});
        return;
      }
      sendJson(response, show);
      return;
    }
    response.writeHead(404);
    response.end('{}');
  };
}

/** Register a fake server's URL as an Ollama host. */
function ollamaHost(endpoint: string, id = 'ollama'): ModelHost {
  return {
    id,
    name: 'Ollama',
    adapter: 'openai_compatible',
    runtime: { engine: 'ollama', adapter: 'openai_compatible', endpoint, path: '/v1/chat/completions' },
  };
}

/** A minimal static catalog with one text model on the Ollama host. */
function staticCatalog(models: readonly ModelDescriptor[] = [], endpoint = 'http://127.0.0.1:1'): ModelCatalogConfig {
  return {
    version: '1',
    hosts: [ollamaHost(endpoint)],
    models: [
      {
        id: 'static_text',
        name: 'Static text model',
        type: 'text_generation',
        capabilities: ['text_to_text'],
        host: 'ollama',
        adapterConfig: { model: 'statically-configured' },
        limits: { contextTokens: 4096 },
      },
      ...models,
    ],
  };
}

const GIB = 1024 ** 3;

describe('ollama discovery: parsing an engine response', () => {
  it('reads the models array out of /api/tags', () => {
    const summaries = parseOllamaTags({
      models: [
        {
          name: 'alpha:latest',
          model: 'alpha:latest',
          size: 2 * GIB,
          digest: 'sha256:abc',
          details: { family: 'llama', parameter_size: '8.0B', quantization_level: 'Q4_K_M' },
        },
      ],
    });
    assert.equal(summaries.length, 1);
    assert.equal(summaries[0]?.name, 'alpha:latest');
    assert.equal(summaries[0]?.sizeBytes, 2 * GIB);
    assert.equal(summaries[0]?.family, 'llama');
    assert.equal(summaries[0]?.parameterSize, '8.0B');
    assert.equal(summaries[0]?.quantizationLevel, 'Q4_K_M');
  });

  it('accepts `model` when `name` is absent', () => {
    const summaries = parseOllamaTags({ models: [{ model: 'beta:7b', size: GIB }] });
    assert.equal(summaries[0]?.name, 'beta:7b');
  });

  it('never throws for an unexpected shape, it just finds nothing', () => {
    for (const raw of [undefined, null, '', 42, [], { models: 'nope' }, { models: [{ size: 1 }] }]) {
      assert.deepEqual(parseOllamaTags(raw), [], `raw=${JSON.stringify(raw)}`);
    }
    assert.deepEqual(parseOllamaTags({ models: [null, 7, {}, { name: 'ok' }] }).map((s) => s.name), ['ok']);
  });

  it('reads a context window from model_info without knowing the family name', () => {
    const details = parseOllamaShow({
      template: '{{ .Prompt }}',
      parameters: 'stop "<|start_header_id|>"',
      model_info: { 'any_architecture.context_length': 131072, 'any_architecture.embedding_length': 4096 },
    });
    assert.equal(details.contextTokens, 131072);
    assert.equal(details.vision, false);
  });

  it('omits the context window when the engine did not report one', () => {
    const details = parseOllamaShow({ model_info: { 'x.embedding_length': 4096 } });
    assert.equal(details.contextTokens, undefined);
    assert.equal('contextTokens' in details, false, 'never guess a number the engine did not state');
  });

  it('detects vision from a projector or clip signal anywhere the engine puts it', () => {
    assert.equal(parseOllamaShow({ projector_info: { 'clip.vision.image_size': 336 } }).vision, true);
    assert.equal(parseOllamaShow({ model_info: { 'x.clip_vision_encoder': 'present' } }).vision, true);
    assert.equal(parseOllamaShow({ model_info: { 'x.attention.head_count': 32 } }).vision, false);
    assert.equal(parseOllamaShow(undefined).vision, false);
  });
});

describe('ollama discovery: mapping into a descriptor', () => {
  it('always serves text_to_text and sets input/output types explicitly', () => {
    const descriptor = mapOllamaModel({ summary: { name: 'plain:8b', sizeBytes: GIB }, detailSource: 'read' }, ollamaHost('http://127.0.0.1:11434'));
    assert.deepEqual(descriptor.capabilities, ['text_to_text']);
    assert.deepEqual(descriptor.inputTypes, ['text']);
    assert.deepEqual(descriptor.outputTypes, ['text']);
    assert.equal(descriptor.type, 'text_generation');
  });

  it('accepts an image as well as text when vision is present', () => {
    const descriptor = mapOllamaModel(
      {
        summary: { name: 'sight:7b', sizeBytes: 4 * GIB },
        details: { name: 'sight:7b', vision: true, contextTokens: 8192 },
        detailSource: 'read',
      },
      ollamaHost('http://127.0.0.1:11434'),
    );
    assert.deepEqual(descriptor.capabilities, ['text_to_text', 'image_understanding']);
    // The union of the two capabilities' canonical IO. Declaring these matters:
    // `image_understanding` accepts an image, and a chained workflow checks for
    // exactly that before handing a generated PNG to this model.
    assert.deepEqual(descriptor.inputTypes, ['text', 'image']);
    assert.deepEqual(descriptor.outputTypes, ['text']);
    assert.equal(descriptor.type, 'multimodal');
  });

  it('derives a deterministic id and keeps the engine name verbatim', () => {
    const host = ollamaHost('http://127.0.0.1:11434');
    const candidate = { summary: { name: 'Llama-3.2:3B-Instruct', sizeBytes: GIB }, detailSource: 'read' as const };
    const first = mapOllamaModel(candidate, host);
    const second = mapOllamaModel(candidate, host);
    assert.equal(first.id, second.id, 'the same install must map to the same id every run');
    assert.match(first.id, /^[a-z0-9][a-z0-9._-]*$/);
    assert.equal(first.id, slugifyModelId('Llama-3.2:3B-Instruct', 'ollama'));
    // The name is the engine's primary key, so it reaches adapterConfig untouched.
    assert.equal(first.adapterConfig?.['model'], 'Llama-3.2:3B-Instruct');
  });

  it('estimates resources conservatively from the reported size', () => {
    const small = mapOllamaModel({ summary: { name: 'a', sizeBytes: 100 }, detailSource: 'read' }, ollamaHost('http://127.0.0.1:1'));
    assert.equal(small.resources?.vramGb, 1, 'anything non-zero costs at least 1 GiB');
    assert.equal(small.resources?.ramGb, 1);
    assert.equal(small.resources?.requiresGpu, false, 'Ollama runs on CPU, so a GPU is not required');

    const big = mapOllamaModel({ summary: { name: 'b', sizeBytes: 9.2 * GIB }, detailSource: 'read' }, ollamaHost('http://127.0.0.1:1'));
    assert.equal(big.resources?.vramGb, 10, 'rounded up, because the router uses this as a filter');
  });

  it('omits resources entirely rather than inventing a size', () => {
    const descriptor = mapOllamaModel({ summary: { name: 'c' }, detailSource: 'read' }, ollamaHost('http://127.0.0.1:1'));
    assert.equal(descriptor.resources?.vramGb, undefined);
    assert.equal(descriptor.resources?.ramGb, undefined);
    assert.equal('vramGb' in (descriptor.resources ?? {}), false, 'never guess a size the engine did not report');
  });

  it('carries the reported context window into limits and nothing when absent', () => {
    const withContext = mapOllamaModel(
      { summary: { name: 'd' }, details: { name: 'd', vision: false, contextTokens: 32768 }, detailSource: 'read' },
      ollamaHost('http://127.0.0.1:1'),
    );
    assert.equal(withContext.limits?.contextTokens, 32768);
    const without = mapOllamaModel({ summary: { name: 'e' }, detailSource: 'read' }, ollamaHost('http://127.0.0.1:1'));
    assert.equal(without.limits?.contextTokens, undefined);
  });

  it('sits below the catalog default priority so a static model wins a tie', () => {
    const descriptor = mapOllamaModel({ summary: { name: 'f' }, detailSource: 'read' }, ollamaHost('http://127.0.0.1:1'));
    assert.equal(descriptor.priority, DISCOVERED_PRIORITY);
    assert.ok(DISCOVERED_PRIORITY > 100, 'the descriptor default is 100; discovered must sort after it');
  });

  it('references the host rather than an inline runtime', () => {
    const descriptor = mapOllamaModel({ summary: { name: 'g' }, detailSource: 'read' }, ollamaHost('http://127.0.0.1:1', 'my-ollama'));
    assert.equal(descriptor.host, 'my-ollama');
    assert.equal(descriptor.runtime, undefined, 'a discovered model inherits its runtime from the host');
  });
});

describe('ollama discovery: against a live server', () => {
  it('enumerates every installed model and asks /api/show about each', async () => {
    const engine = await startOllama();
    try {
      engine.setResponder(
        responderFor(
          {
            models: [
              { name: 'alpha:latest', size: 3 * GIB, details: { family: 'llama', parameter_size: '8.0B', quantization_level: 'Q4_K_M' } },
              { name: 'sight:7b', size: 5 * GIB, details: { family: 'qwen2vl' } },
            ],
          },
          {
            'alpha:latest': { model_info: { 'llama.context_length': 8192 } },
            'sight:7b': { projector_info: { 'clip.vision.image_size': 336 }, model_info: { 'qwen2vl.context_length': 32768 } },
          },
        ),
      );
      const descriptors = await createOllamaDiscoverer({ requestTimeoutMs: 2000 }).discover(ollamaHost(engine.url), new AbortController().signal);

      assert.equal(descriptors.length, 2);
      const vision = descriptors.find((descriptor) => descriptor.capabilities.includes('image_understanding'));
      const textOnly = descriptors.find((descriptor) => !descriptor.capabilities.includes('image_understanding'));
      assert.ok(vision, 'the model with a projector must be discovered as vision-capable');
      assert.ok(textOnly);
      assert.deepEqual(vision.inputTypes, ['text', 'image']);
      assert.equal(vision.limits?.contextTokens, 32768);
      assert.equal(textOnly.limits?.contextTokens, 8192);

      const showRequests = engine.requests.filter((request) => request.url === '/api/show');
      assert.equal(showRequests.length, 2);
      assert.equal(showRequests[0]?.method, 'POST');
      assert.match(showRequests[0]?.body ?? '', /"name":"alpha:latest"/);
    } finally {
      await engine.close();
    }
  });

  it('still yields a model when /api/show cannot be read', async () => {
    const engine = await startOllama();
    try {
      engine.setResponder(responderFor({ models: [{ name: 'mystery:1b', size: GIB }] }, {}, 'error'));
      const descriptors = await createOllamaDiscoverer({ requestTimeoutMs: 2000 }).discover(ollamaHost(engine.url), new AbortController().signal);
      assert.equal(descriptors.length, 1);
      assert.deepEqual(descriptors[0]?.capabilities, ['text_to_text'], 'no evidence of vision, so no vision claimed');
      assert.match(descriptors[0]?.notes ?? '', /could not be read/i);
    } finally {
      await engine.close();
    }
  });

  it('reports an unreachable engine as a failure the registry contains', async () => {
    const engine = await startOllama();
    const url = engine.url;
    await engine.close();
    await assert.rejects(
      () => createOllamaDiscoverer({ requestTimeoutMs: 1000 }).discover(ollamaHost(url), new AbortController().signal),
      (error: Error) => {
        assert.match(error.message, /could not list models/);
        return true;
      },
    );
  });

  it('reports a malformed body as a failure rather than a crash', async () => {
    const engine = await startOllama();
    try {
      engine.setResponder((_request, response) => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end('{"models": "not an array"}');
      });
      // A shape the parser tolerates yields zero models, not a throw: a new
      // engine version should degrade, not take discovery down.
      const descriptors = await createOllamaDiscoverer({ requestTimeoutMs: 2000 }).discover(ollamaHost(engine.url), new AbortController().signal);
      assert.deepEqual(descriptors, []);
    } finally {
      await engine.close();
    }
  });
});

describe('discovery: the merge step', () => {
  /** A discovered descriptor with a chosen id and a distinguishing field. */
  function discovered(id: string, name = 'Discovered'): ModelDescriptor {
    return {
      id,
      name,
      type: 'text_generation',
      capabilities: ['text_to_text'],
      host: 'ollama',
      inputTypes: ['text'],
      outputTypes: ['text'],
      adapterConfig: { model: `discovered:${id}` },
      limits: { contextTokens: 111111 },
    };
  }

  it('places every static model before every discovered one', () => {
    const staticConfig = staticCatalog();
    const merged = mergeCatalogConfig(staticConfig, [discovered('a_discovered'), discovered('b_discovered')]);
    assert.deepEqual(
      merged.models.map((model) => model.id),
      ['static_text', 'a_discovered', 'b_discovered'],
    );
  });

  it('drops a discovered model whose id a static entry already claims', () => {
    const staticConfig = staticCatalog();
    const merged = mergeCatalogConfig(staticConfig, [discovered('static_text', 'Impostor'), discovered('fresh')]);
    assert.equal(merged.models.length, 2);
    assert.equal(merged.models[0]?.name, 'Static text model');
    assert.equal(merged.models[0]?.limits?.contextTokens, 4096);
    assert.equal(merged.models[1]?.id, 'fresh');
  });

  it('does not mutate the configuration it was given', () => {
    const staticConfig = staticCatalog();
    const before = staticConfig.models.length;
    mergeCatalogConfig(staticConfig, [discovered('new')]);
    assert.equal(staticConfig.models.length, before);
  });

  it('keeps hosts and the version through the merge', () => {
    const staticConfig = staticCatalog();
    const merged = mergeCatalogConfig(staticConfig, [discovered('new')]);
    assert.equal(merged.version, '1');
    assert.equal(merged.hosts?.length, 1);
    assert.equal(merged.hosts?.[0]?.id, 'ollama');
  });

  it('collapses a duplicate reported twice in one pass', () => {
    const merged = mergeCatalogConfig(staticCatalog(), [discovered('dup'), discovered('dup')]);
    assert.equal(merged.models.filter((model) => model.id === 'dup').length, 1);
  });

  it('makes the resolved catalog contain the static fields, not the discovered ones', () => {
    // The point of constraint 4: static must win on the *fields*, not merely on
    // arrival order. If the merge were relying on ModelCatalog's own duplicate
    // handling, the constructor would keep the first (static) one too — but it
    // would also log a spurious `duplicate model id` error and report a
    // diagnostic for a collision that was resolved by design.
    const catalog = new ModelCatalog(mergeCatalogConfig(staticCatalog(), [discovered('static_text', 'Impostor')]));
    const resolved = catalog.requireModel('static_text');
    assert.equal(resolved.name, 'Static text model');
    assert.equal(resolved.limits.contextTokens, 4096);
    assert.deepEqual(catalog.loadDiagnostics, [], 'a resolved collision must not be reported as a load error');
  });

  it('publishes discovered models through the catalog without a resolveDescriptor bypass', () => {
    const catalog = new ModelCatalog(mergeCatalogConfig(staticCatalog(), [discovered('a_discovered')]));
    const resolved = catalog.requireModel('a_discovered');
    // Inherited from the host, exactly as a static descriptor's would be.
    assert.equal(resolved.adapter, 'openai_compatible');
    assert.equal(resolved.runtime.endpoint, 'http://127.0.0.1:1');
    assert.equal(resolved.priority, 100, 'the descriptor declares no priority, so the catalog default applies');
    assert.deepEqual(resolved.tags, []);
  });
});

describe('discovery: the cache', () => {
  it('serves a second pass from cache within the TTL and refetches after it', async () => {
    const engine = await startOllama();
    try {
      engine.setResponder(responderFor({ models: [{ name: 'cached:1b', size: GIB }] }, { 'cached:1b': {} }));
      const { hub } = await ModelHub.fromConfigAndDiscovery(staticCatalog([], engine.url), {
        discoverModels: true,
        discoveryTtlMs: 60_000,
        discoveryTimeoutMs: 2000,
        manageTimers: false,
        log: () => {},
      });
      const afterFirst = engine.requests.filter((request) => request.url === '/api/tags').length;
      assert.equal(afterFirst, 1);

      const second = await hub.refreshDiscovery();
      assert.equal(second.cached, true);
      assert.equal(engine.requests.filter((request) => request.url === '/api/tags').length, 1, 'within the TTL, no refetch');

      const forced = await hub.refreshDiscovery({ force: true });
      assert.equal(forced.cached, false);
      assert.equal(engine.requests.filter((request) => request.url === '/api/tags').length, 2, 'force bypasses the TTL');

      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('refetches once the TTL has elapsed', async () => {
    const engine = await startOllama();
    try {
      engine.setResponder(responderFor({ models: [{ name: 'ttl:1b', size: GIB }] }, { 'ttl:1b': {} }));
      const { hub } = await ModelHub.fromConfigAndDiscovery(staticCatalog([], engine.url), {
        discoverModels: true,
        discoveryTtlMs: 20,
        discoveryTimeoutMs: 2000,
        manageTimers: false,
        log: () => {},
      });
      assert.equal(engine.requests.filter((request) => request.url === '/api/tags').length, 1);
      await new Promise((resolve) => setTimeout(resolve, 40));
      const result = await hub.refreshDiscovery();
      assert.equal(result.cached, false, 'a pass after the TTL must hit the engine again');
      assert.equal(engine.requests.filter((request) => request.url === '/api/tags').length, 2);
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('answers with an empty, cached result when runtime discovery is disabled', async () => {
    const hub = ModelHub.fromConfig(staticCatalog());
    const result = await hub.refreshDiscovery({ force: true });
    assert.deepEqual(result.descriptors, []);
    assert.equal(result.cached, true);
    assert.equal(hub.discovery, undefined, 'discovery off means no registry exists at all');
    await hub.dispose();
  });
});

describe('discovery: end to end through the hub catalog', () => {
  it('publishes a discovered Ollama model and routes to it by capability', async () => {
    const engine = await startOllama();
    try {
      engine.setResponder(
        responderFor(
          { models: [{ name: 'live:8b', size: 6 * GIB, details: { family: 'llama', parameter_size: '8.0B', quantization_level: 'Q4_K_M' } }] },
          { 'live:8b': { model_info: { 'llama.context_length': 4096 } } },
        ),
      );
      const { hub } = await ModelHub.fromConfigAndDiscovery(staticCatalog([], engine.url), {
        manageTimers: false,
        discoveryTimeoutMs: 2000,
        log: () => {},
      });

      const id = slugifyModelId('live:8b', 'ollama');
      const model = hub.catalog.requireModel(id);
      assert.equal(model.name, 'live:8b');
      assert.deepEqual(model.capabilities, ['text_to_text']);
      assert.deepEqual(model.inputTypes, ['text']);
      assert.equal(model.adapterConfig['model'], 'live:8b');
      assert.equal(model.limits.contextTokens, 4096);

      const candidates = hub.catalog.findModelsByCapability('text_to_text').map((candidate) => candidate.id);
      assert.deepEqual(candidates, ['static_text', id], 'the static model is preferred on priority');

      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('gives a model discovered after construction a full runtime status', async () => {
    // Regression. The runtime manager seeds its per-model state when it is
    // constructed; discovery adds a model later. Before `syncCatalog()`, a
    // discovered model had no state entry, so `getModelStatus` threw — and since
    // `listModels()` reads a status for every catalog model, one discovered model
    // broke the model listing, the settings page's inventory, and
    // `explain_routing` at once.
    const engine = await startOllama();
    try {
      engine.setResponder(responderFor({ models: [{ name: 'late:1b', size: GIB }] }, { 'late:1b': {} }));
      const hub = ModelHub.fromConfig(staticCatalog([], engine.url), {
        discoverModels: true,
        discoveryTimeoutMs: 2000,
        manageTimers: false,
        log: () => {},
      });
      // The constructor's pre-warm is in flight; await it rather than sleeping.
      await hub.refreshDiscovery();

      const id = slugifyModelId('late:1b', 'ollama');
      assert.ok(hub.catalog.listModelIds().includes(id), 'the discovered model is in the catalog');

      // The three surfaces that broke, all of which read a status per model.
      const views = hub.listModels();
      assert.equal(views.length, 2);
      const discoveredView = views.find((view) => view.model.id === id);
      assert.ok(discoveredView, 'every catalog model must have a status view');
      assert.equal(discoveredView.status.modelId, id);
      assert.ok(
        ['stopped', 'available', 'starting', 'unhealthy', 'unsupported', 'error', 'disabled'].includes(
          discoveredView.status.availability,
        ),
        `unexpected availability ${discoveredView.status.availability}`,
      );
      assert.doesNotThrow(() => hub.getModelStatus(id));
      assert.doesNotThrow(() => hub.catalog.listCapabilities());

      // A static model's state must survive a discovery pass untouched.
      assert.equal(hub.getModelStatus('static_text').modelId, 'static_text');

      // Discovery is off the moment the engine is gone, and the model leaves with
      // it — while the static one and its runtime state remain.
      engine.setResponder(responderFor({ models: [] }, {}));
      await hub.refreshDiscovery({ force: true });
      assert.ok(!hub.catalog.listModelIds().includes(id), 'a model the engine no longer reports stops being routable');
      assert.deepEqual(hub.listModels().map((view) => view.model.id), ['static_text']);

      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('keeps working when the engine is unreachable', async () => {
    const engine = await startOllama();
    const url = engine.url;
    await engine.close();
    const warnings: string[] = [];
    const { hub, discovery } = await ModelHub.fromConfigAndDiscovery(staticCatalog([], url), {
      manageTimers: false,
      discoveryTimeoutMs: 1000,
      log: (message) => warnings.push(message),
    });
    assert.deepEqual(discovery.descriptors, []);
    assert.equal(discovery.warnings.length, 1);
    assert.match(discovery.warnings[0]?.message ?? '', /could not list models/);
    // The hub is fully usable with only its static models.
    assert.deepEqual(hub.catalog.listModelIds(), ['static_text']);
    await hub.dispose();
  });

  it('drops a discovered model that a static entry claims by id, end to end', async () => {
    const engine = await startOllama();
    try {
      // A discovered id is `<engine>-<slugified name>`, so the collision is
      // engineered by pinning a static entry to exactly the id this install
      // will produce. That is the real-world case: an operator who wants one
      // specific model configured by hand, on a machine where the engine also
      // reports it.
      const collidingId = slugifyModelId('static-text', 'ollama');
      const staticConfig: ModelCatalogConfig = {
        version: '1',
        hosts: [ollamaHost(engine.url)],
        models: [
          {
            id: collidingId,
            name: 'Pinned static model',
            type: 'text_generation',
            capabilities: ['text_to_text'],
            host: 'ollama',
            adapterConfig: { model: 'statically-configured', temperature: 0.1 },
            limits: { contextTokens: 4096 },
          },
        ],
      };
      engine.setResponder(responderFor({ models: [{ name: 'static-text', size: 2 * GIB }] }, { 'static-text': {} }));
      const { hub, discovery } = await ModelHub.fromConfigAndDiscovery(staticConfig, {
        manageTimers: false,
        discoveryTimeoutMs: 2000,
        log: () => {},
      });

      assert.equal(discovery.descriptors.length, 1, 'the engine did report a model');
      assert.equal(discovery.descriptors[0]?.id, collidingId);
      assert.deepEqual(hub.catalog.listModelIds(), [collidingId], 'the discovered duplicate was dropped');
      const resolved = hub.catalog.requireModel(collidingId);
      assert.equal(resolved.name, 'Pinned static model', 'the static descriptor fields survive the collision');
      assert.equal(resolved.adapterConfig['model'], 'statically-configured');
      assert.equal(resolved.adapterConfig['temperature'], 0.1);
      assert.deepEqual(hub.catalog.loadDiagnostics, []);
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });
});
