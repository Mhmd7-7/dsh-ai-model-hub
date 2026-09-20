/**
 * Tests for the `http_json` image adapter.
 *
 * Like the OpenAI adapter's tests, these run against a **real local HTTP
 * server** rather than a mocked `fetch`. The failure modes that matter here are
 * all wire-level: whether the request body really carries the A1111 fields,
 * whether a timeout is reported as a timeout instead of a transport error,
 * whether an engine answering 200 with chat JSON is diagnosed clearly instead of
 * producing an empty artifact, and whether cancellation reaches the socket.
 *
 * A mock would paper over every one of those.
 *
 * @module dsh-ai-model-hub/tests/http-json
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { IncomingHttpHeaders, ServerResponse } from 'node:http';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, describe, it } from 'node:test';

import type { ModelHub, ResolvedModel } from '../src/index.ts';
import {
  ModelHub as Hub,
  extractImages,
  readPngSize,
  renderMockPng,
} from '../src/index.ts';
import { parseModelCatalogConfig } from '../src/catalog/descriptor.ts';

/** One captured request, so assertions can be made about what was actually sent. */
interface CapturedRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: IncomingHttpHeaders;
  readonly body: string;
}

/** A test double for a Stable Diffusion HTTP engine. */
interface FakeEngine {
  /** Base URL to put in `runtime.endpoint`. */
  readonly url: string;
  /** Every request the engine received, oldest first. */
  readonly requests: CapturedRequest[];
  /**
   * Replace the responder.
   * @param responder - the new responder.
   */
  setResponder(responder: (request: CapturedRequest, response: ServerResponse) => void): void;
  /** Shut the server down. */
  close(): Promise<void>;
}

/** A real, valid 128x64 PNG, base64-encoded — what a working engine returns. */
const SAMPLE_PNG_B64 = Buffer.from(renderMockPng(128, 64, 'fixture', { gridStep: 16 })).toString('base64');

/** The default responder: a healthy model listing and a one-image generation. */
function defaultResponder(request: CapturedRequest, response: ServerResponse): void {
  if (request.method === 'GET') {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify([{ title: 'sd_xl_base_1.0.safetensors' }]));
    return;
  }
  response.writeHead(200, { 'content-type': 'application/json' });
  response.end(JSON.stringify({ images: [SAMPLE_PNG_B64], parameters: {}, info: '{}' }));
}

/**
 * Start a local engine double on an ephemeral port.
 * @returns the fixture.
 */
async function startEngine(): Promise<FakeEngine> {
  let responder = defaultResponder;
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
 * Find a port that is definitely not listening, by opening and closing one.
 * @returns a URL that refuses connections.
 */
async function closedPortUrl(): Promise<string> {
  const engine = await startEngine();
  const url = engine.url;
  await engine.close();
  return url;
}

/** Every temporary artifact root created here, cleaned up once at the end. */
const roots: string[] = [];

after(async () => {
  await Promise.all(roots.map((root) => rm(root, { recursive: true, force: true })));
});

/**
 * Build a hub containing a real image engine plus a mock fallback.
 * @param endpoint - the endpoint the real model points at.
 * @param overrides - descriptor fields to merge into the real image model.
 * @returns the hub and its resolved image model.
 */
async function hubWithEngine(
  endpoint: string,
  overrides: Record<string, unknown> = {},
): Promise<{ hub: ModelHub; model: ResolvedModel }> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-httpjson-'));
  roots.push(root);
  const hub = Hub.fromConfig(
    {
      version: '1',
      models: [
        {
          id: 'sd_test',
          name: 'Test image model',
          type: 'image_generation',
          capabilities: ['text_to_image', 'image_to_image'],
          adapter: 'http_json',
          runtime: { engine: 'stable-diffusion-webui', adapter: 'http_json', endpoint, path: '/sdapi/v1/txt2img' },
          adapterConfig: { model: 'sd_xl_base_1.0.safetensors', steps: 30, cfgScale: 7, sampler: 'DPM++ 2M Karras' },
          limits: { maxWidth: 512, maxHeight: 512 },
          priority: 10,
          tags: ['local'],
          ...overrides,
        },
        {
          id: 'mock_image_model',
          name: 'Mock Image',
          type: 'image_generation',
          capabilities: ['text_to_image', 'image_to_image'],
          adapter: 'mock',
          runtime: { engine: 'in_process_mock', adapter: 'mock' },
          priority: 100,
        },
      ],
    },
    { artifactRoot: root, manageTimers: false, log: () => {} },
  );
  return { hub, model: hub.catalog.requireModel('sd_test') };
}

/**
 * Invoke the adapter directly, bypassing routing, so each assertion is about the
 * adapter's own behaviour.
 * @param hub - the hub that owns the store.
 * @param model - the model to invoke.
 * @param request - capability, prompt, inputs, and options.
 * @returns the raw adapter output.
 */
async function invokeDirectly(
  hub: ModelHub,
  model: ResolvedModel,
  request: {
    capability?: 'text_to_image' | 'image_to_image';
    prompt?: string;
    inputs?: readonly { readonly id: string }[];
    options?: Readonly<Record<string, unknown>>;
    signal?: AbortSignal;
  } = {},
): Promise<Awaited<ReturnType<ReturnType<ModelHub['adapters']['require']>['invoke']>>> {
  const adapter = hub.adapters.require('http_json');
  const inputs = await Promise.all(
    (request.inputs ?? []).map(async (reference) => {
      const artifact = await hub.artifacts.get(reference.id);
      assert.ok(artifact, `expected input artifact ${reference.id} to exist`);
      return artifact;
    }),
  );
  return adapter.invoke({
    model,
    capability: request.capability ?? 'text_to_image',
    ...(request.prompt === undefined ? {} : { prompt: request.prompt }),
    inputs,
    options: request.options ?? {},
    artifacts: hub.artifacts,
    signal: request.signal ?? new AbortController().signal,
    log: { debug: () => {}, info: () => {}, warn: () => {} },
  });
}

describe('http_json adapter: response parsing', () => {
  it('reads the A1111 { images: [...] } shape', () => {
    assert.deepEqual(extractImages({ images: ['aaa', 'bbb'] }), ['aaa', 'bbb']);
  });

  it('reads a bare { image: "..." } shape', () => {
    assert.deepEqual(extractImages({ image: 'aaa' }), ['aaa']);
  });

  it('reads an OpenAI-style { data: [{ b64_json }] } shape', () => {
    assert.deepEqual(extractImages({ data: [{ b64_json: 'aaa' }, { b64_json: 'bbb' }] }), ['aaa', 'bbb']);
  });

  it('returns nothing for a chat-completion body rather than throwing', () => {
    assert.deepEqual(extractImages({ choices: [{ message: { content: 'hi' } }] }), []);
  });

  it('returns nothing for null and for non-objects', () => {
    assert.deepEqual(extractImages(null), []);
    assert.deepEqual(extractImages('nope'), []);
    assert.deepEqual(extractImages(42), []);
  });

  it('ignores non-string and empty entries', () => {
    assert.deepEqual(extractImages({ images: ['ok', '', 7, null] }), ['ok']);
  });
});

describe('http_json adapter: PNG dimension parsing', () => {
  it('reads real dimensions out of a PNG header', () => {
    const size = readPngSize(renderMockPng(128, 64, 'fixture', { gridStep: 16 }));
    assert.deepEqual(size, { width: 128, height: 64 });
  });

  it('returns undefined for a non-PNG payload', () => {
    assert.equal(readPngSize(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])), undefined);
  });

  it('returns undefined for a payload too short to hold a header', () => {
    assert.equal(readPngSize(new Uint8Array([0x89, 0x50, 0x4e, 0x47])), undefined);
  });
});

describe('http_json adapter: request shape', () => {
  it('POSTs the A1111 generation fields to the configured route', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      await invokeDirectly(hub, model, { prompt: 'a man', options: { width: 256, height: 256 } });

      assert.equal(engine.requests.length, 1);
      const request = engine.requests[0];
      assert.ok(request);
      assert.equal(request.method, 'POST');
      assert.equal(request.url, '/sdapi/v1/txt2img');
      const body = JSON.parse(request.body) as Record<string, unknown>;
      assert.equal(body['prompt'], 'a man');
      assert.equal(body['steps'], 30);
      assert.equal(body['cfg_scale'], 7);
      assert.equal(body['sampler_name'], 'DPM++ 2M Karras');
      assert.equal(body['width'], 256);
      assert.equal(body['height'], 256);
    } finally {
      await engine.close();
    }
  });

  it('forwards a negative prompt and a seed when supplied', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      await invokeDirectly(hub, model, {
        prompt: 'a man',
        options: { negativePrompt: 'blurry', seed: 1234 },
      });
      const request = engine.requests[0];
      assert.ok(request);
      const body = JSON.parse(request.body) as Record<string, unknown>;
      assert.equal(body['negative_prompt'], 'blurry');
      assert.equal(body['seed'], 1234);
    } finally {
      await engine.close();
    }
  });

  it('posts image_to_image to the img2img sibling route with the source inline', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      const source = await hub.artifacts.put({
        type: 'image',
        bytes: renderMockPng(64, 64, 'source', { gridStep: 16 }),
        mimeType: 'image/png',
        extension: '.png',
        metadata: { width: 64, height: 64 },
      });

      await invokeDirectly(hub, model, {
        capability: 'image_to_image',
        prompt: 'make it night',
        inputs: [{ id: source.id }],
      });

      const request = engine.requests[0];
      assert.ok(request);
      // The model declared only the txt2img route; img2img must be derived from it.
      assert.equal(request.url, '/sdapi/v1/img2img');
      const body = JSON.parse(request.body) as Record<string, unknown>;
      const initImages = body['init_images'];
      assert.ok(Array.isArray(initImages), 'expected init_images to be an array');
      assert.equal(initImages.length, 1);
      // It must be the source image's real bytes, not a placeholder.
      const decoded = Buffer.from(initImages[0] as string, 'base64');
      const size = readPngSize(new Uint8Array(decoded));
      assert.deepEqual(size, { width: 64, height: 64 });
    } finally {
      await engine.close();
    }
  });
});

describe('http_json adapter: artifacts', () => {
  it('stores a real PNG whose metadata matches the bytes, not the request', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      // Ask for 512x512 but have the engine answer with a 128x64 image, which is
      // exactly what a silently-clamping engine does.
      const output = await invokeDirectly(hub, model, {
        prompt: 'a man',
        options: { width: 512, height: 512 },
      });

      assert.equal(output.outputs.length, 1);
      const artifact = output.outputs[0];
      assert.ok(artifact);
      assert.equal(artifact.type, 'image');
      assert.equal(artifact.mimeType, 'image/png');
      assert.equal(artifact.metadata['width'], 128);
      assert.equal(artifact.metadata['height'], 64);
      assert.equal(artifact.metadata['requestedWidth'], 512);
      assert.equal(artifact.metadata['prompt'], 'a man');

      const read = await hub.artifacts.read(artifact.id);
      assert.deepEqual(readPngSize(read.bytes), { width: 128, height: 64 });
    } finally {
      await engine.close();
    }
  });

  it('stores one artifact per returned image', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder((_request, response) => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ images: [SAMPLE_PNG_B64, SAMPLE_PNG_B64, SAMPLE_PNG_B64] }));
      });
      const { hub, model } = await hubWithEngine(engine.url);
      const output = await invokeDirectly(hub, model, { prompt: 'a man' });
      assert.equal(output.outputs.length, 3);
      assert.equal(output.value?.['count'], 3);
    } finally {
      await engine.close();
    }
  });
});

describe('http_json adapter: failures are actionable', () => {
  it('reports a non-2xx response with the status and the engine body', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder((_request, response) => {
        response.writeHead(500, { 'content-type': 'text/plain' });
        response.end('CUDA out of memory');
      });
      const { hub, model } = await hubWithEngine(engine.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /500/);
          assert.match(error.message, /CUDA out of memory/);
          return true;
        },
      );
    } finally {
      await engine.close();
    }
  });

  it('explains that a chat endpoint is not an image engine', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder((_request, response) => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'hello' } }] }));
      });
      const { hub, model } = await hubWithEngine(engine.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /no images/);
          assert.match(error.message, /chat completions/);
          return true;
        },
      );
    } finally {
      await engine.close();
    }
  });

  it('reports a non-JSON body as such', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder((_request, response) => {
        response.writeHead(200, { 'content-type': 'text/html' });
        response.end('<html>not json</html>');
      });
      const { hub, model } = await hubWithEngine(engine.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /not JSON/);
          return true;
        },
      );
    } finally {
      await engine.close();
    }
  });

  it('reports an unreachable engine with the URL', async () => {
    const url = await closedPortUrl();
    const { hub, model } = await hubWithEngine(url);
    await assert.rejects(
      () => invokeDirectly(hub, model, { prompt: 'a man' }),
      (error: Error) => {
        assert.match(error.message, /could not reach/);
        assert.match(error.message, new RegExp(url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
        return true;
      },
    );
  });

  it('reports a slow engine as a timeout, not a transport failure', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder(() => {
        // Never answer: the adapter's own budget must fire.
      });
      const { hub, model } = await hubWithEngine(engine.url, { adapterConfig: { timeoutMs: 150 } });
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /did not answer/);
          assert.match(error.message, /150 ms/);
          return true;
        },
      );
    } finally {
      await engine.close();
    }
  });

  it('reports cancellation as ABORTED rather than as an engine failure', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder(() => {
        // Hold the request open until the test aborts it.
      });
      const { hub, model } = await hubWithEngine(engine.url, { adapterConfig: { timeoutMs: 30_000 } });
      const controller = new AbortController();
      const pending = invokeDirectly(hub, model, { prompt: 'a man', signal: controller.signal });
      setTimeout(() => controller.abort(), 50);
      await assert.rejects(pending, (error: Error & { code?: string }) => {
        assert.equal(error.code, 'INVOCATION_ABORTED');
        return true;
      });
    } finally {
      await engine.close();
    }
  });

  it('requires a prompt', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, {}),
        (error: Error) => {
          assert.match(error.message, /requires a `prompt`/);
          return true;
        },
      );
    } finally {
      await engine.close();
    }
  });
});

describe('http_json adapter: supports and health', () => {
  it('rejects an http_json model with no endpoint at catalog-validation time', () => {
    const result = parseModelCatalogConfig(
      {
        version: '1',
        models: [
          {
            id: 'no_endpoint',
            name: 'No endpoint',
            type: 'image_generation',
            capabilities: ['text_to_image'],
            adapter: 'http_json',
            runtime: { engine: 'stable-diffusion-webui', adapter: 'http_json' },
          },
        ],
      },
      'models.json',
    );
    assert.equal(result.ok, false);
    assert.match(result.ok === false ? result.message : '', /endpoint/);
  });

  it('still refuses an endpointless model at the adapter boundary', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      // The catalog already forbids this shape, so it cannot arise from config.
      // `supports` must nevertheless be total — it is the last line of defence
      // when an adapter is driven directly or a model is built programmatically.
      const stripped: ResolvedModel = {
        ...model,
        runtime: { engine: model.runtime.engine, adapter: model.runtime.adapter },
      };
      const support = hub.adapters.require('http_json').supports(stripped);
      assert.equal(support.ok, false);
      assert.match(support.ok === false ? support.reason : '', /requires `runtime\.endpoint`/);
    } finally {
      await engine.close();
    }
  });

  it('refuses a model that declares only non-image capabilities', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url, { capabilities: ['text_to_text'] });
      const support = hub.adapters.require('http_json').supports(model);
      assert.equal(support.ok, false);
      assert.match(support.ok === false ? support.reason : '', /text_to_text/);
    } finally {
      await engine.close();
    }
  });

  it('reports healthy when the engine answers below 500', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      const report = await hub.adapters.require('http_json').health(model, new AbortController().signal);
      assert.equal(report.healthy, true);
      assert.match(report.detail ?? '', /responded 200/);
    } finally {
      await engine.close();
    }
  });

  it('reports unhealthy when the engine is unreachable, without throwing', async () => {
    const url = await closedPortUrl();
    const { hub, model } = await hubWithEngine(url);
    const report = await hub.adapters.require('http_json').health(model, new AbortController().signal);
    assert.equal(report.healthy, false);
    assert.match(report.detail ?? '', /unreachable/);
  });
});

describe('http_json adapter: routing integration', () => {
  it('routes text_to_image to the real engine over the mock, and falls back when it fails', async () => {
    const engine = await startEngine();
    try {
      const { hub } = await hubWithEngine(engine.url);
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a man' });
      // Priority 10 beats the mock's 100, so the real engine must win.
      assert.equal(result.modelId, 'sd_test');
      assert.equal(result.outputs.length, 1);

      // Now break the engine: the mock fallback must still serve the request.
      engine.setResponder((_request, response) => {
        response.writeHead(500, { 'content-type': 'text/plain' });
        response.end('boom');
      });
      const fellBack = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a man' });
      assert.equal(fellBack.modelId, 'mock_image_model');
      assert.equal(fellBack.outputs.length, 1);
    } finally {
      await engine.close();
    }
  });
});
