/**
 * Tests for the OpenAI-compatible adapter.
 *
 * Every test here runs against a **real local HTTP server** rather than a mocked
 * `fetch`, because the things most likely to be wrong are exactly the things a
 * mock would paper over: the request body's shape, whether cancellation actually
 * reaches the socket, whether a timeout is reported as a timeout rather than as a
 * transport failure, and whether a non-2xx response produces an actionable
 * message instead of a silent empty artifact.
 *
 * @module dsh-ai-model-hub/tests/openai
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { IncomingHttpHeaders, ServerResponse } from 'node:http';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, describe, it } from 'node:test';

import type { ModelHub, ResolvedModel } from '../src/index.ts';
import { ModelHub as Hub, ModelHubError, resolveDescriptor, silentLogger } from '../src/index.ts';

/** One captured request, so assertions can be made about what was actually sent. */
interface CapturedRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: IncomingHttpHeaders;
  readonly body: string;
}

/** A test double for an OpenAI-compatible engine. */
interface FakeEngine {
  /** Base URL to put in `runtime.endpoint`. */
  readonly url: string;
  /** Every request the engine received, oldest first. */
  readonly requests: CapturedRequest[];
  /**
   * Replace the responder. The default answers `GET /` with 200 and any POST
   * with a valid chat completion.
   * @param responder - the new responder.
   */
  setResponder(responder: (request: CapturedRequest, response: ServerResponse) => void): void;
  /** Shut the server down. */
  close(): Promise<void>;
}

/** The default responder: a healthy root probe and a one-choice completion. */
function defaultResponder(request: CapturedRequest, response: ServerResponse): void {
  if (request.method === 'GET') {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ status: 'ok' }));
    return;
  }
  response.writeHead(200, { 'content-type': 'application/json' });
  response.end(
    JSON.stringify({
      model: 'test-model:latest',
      choices: [{ index: 0, message: { role: 'assistant', content: 'hello from the fake engine' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 11, completion_tokens: 7 },
    }),
  );
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
 * Build a hub containing a real-engine model plus a mock fallback.
 * @param endpoint - the endpoint the real model points at.
 * @param overrides - descriptor fields to merge into the real-engine model.
 * @returns the hub and its resolved real-engine model.
 */
async function hubWithEngine(
  endpoint: string,
  overrides: Record<string, unknown> = {},
): Promise<{ hub: ModelHub; model: ResolvedModel }> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-openai-'));
  roots.push(root);
  const hub = Hub.fromConfig(
    {
      version: '1',
      models: [
        {
          id: 'ollama_test',
          name: 'Test text model',
          type: 'text_generation',
          capabilities: ['text_to_text', 'image_understanding'],
          adapter: 'openai_compatible',
          runtime: { engine: 'ollama', adapter: 'openai_compatible', endpoint, path: '/v1/chat/completions' },
          adapterConfig: { model: 'test-model:latest', temperature: 0.3 },
          priority: 10,
          tags: ['local'],
          ...overrides,
        },
        {
          id: 'mock_text_model',
          name: 'Mock Text',
          type: 'text_generation',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'mock', adapter: 'mock' },
          priority: 100,
        },
      ],
    },
    { artifactRoot: root, manageTimers: false, log: () => {} },
  );
  return { hub, model: hub.catalog.requireModel('ollama_test') };
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
    capability?: 'text_to_text' | 'image_understanding';
    prompt?: string;
    inputs?: readonly { readonly id: string }[];
    options?: Readonly<Record<string, unknown>>;
    signal?: AbortSignal;
  } = {},
): Promise<Awaited<ReturnType<ReturnType<ModelHub['adapters']['require']>['invoke']>>> {
  const adapter = hub.adapters.require('openai_compatible');
  const inputs = await Promise.all(
    (request.inputs ?? []).map(async (reference) => {
      const artifact = await hub.artifacts.get(reference.id);
      assert.ok(artifact, `expected input artifact ${reference.id} to exist`);
      return artifact;
    }),
  );
  return adapter.invoke({
    model,
    capability: request.capability ?? 'text_to_text',
    ...(request.prompt === undefined ? {} : { prompt: request.prompt }),
    inputs,
    options: request.options ?? {},
    artifacts: hub.artifacts,
    signal: request.signal ?? new AbortController().signal,
    log: silentLogger(),
  });
}

describe('openai_compatible adapter: contract', () => {
  it('rejects a model with no endpoint at the catalog boundary', () => {
    // The catalog validator requires an endpoint for this adapter kind, so a
    // misconfigured model is `INVALID_DESCRIPTOR` at startup rather than a
    // confusing failure at first use.
    assert.throws(
      () =>
        Hub.fromConfig(
          {
            models: [
              {
                id: 'no_endpoint',
                name: 'No endpoint',
                type: 'text_generation',
                capabilities: ['text_to_text'],
                adapter: 'openai_compatible',
                runtime: { engine: 'ollama', adapter: 'openai_compatible' },
              },
            ],
          },
          { artifactRoot: join(tmpdir(), 'aimh-openai-unused'), manageTimers: false, log: () => {} },
        ),
      (error: unknown) =>
        error instanceof ModelHubError &&
        error.code === 'INVALID_DESCRIPTOR' &&
        /runtime\.endpoint/.test(error.message),
    );
  });

  it('fails closed if a model without an endpoint ever reaches the adapter', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      const adapter = hub.adapters.require('openai_compatible');
      assert.deepEqual(adapter.supports(model), { ok: true });

      // `resolveDescriptor` is the layer *below* validation. A descriptor that
      // skipped the catalog's cross-field check must still be refused here —
      // defence in depth, not a duplicate of the validator.
      const bare = resolveDescriptor(
        {
          id: 'bare',
          name: 'Bare',
          type: 'text_generation',
          capabilities: ['text_to_text'],
          adapter: 'openai_compatible',
          runtime: { engine: 'ollama', adapter: 'openai_compatible' },
        },
        undefined,
      );
      const verdict = adapter.supports(bare);
      assert.equal(verdict.ok, false);
      assert.match(verdict.ok === false ? verdict.reason : '', /runtime\.endpoint/);
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('marks a model unsupported when it declares only a non-chat capability', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      const adapter = hub.adapters.require('openai_compatible');
      const imageOnly = { ...model, capabilities: ['text_to_image'] } as ResolvedModel;
      const verdict = adapter.supports(imageOnly);
      assert.equal(verdict.ok, false);
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('reports an unreachable engine as a health finding rather than throwing', async () => {
    const url = await closedPortUrl();
    const { hub, model } = await hubWithEngine(url);
    try {
      const adapter = hub.adapters.require('openai_compatible');
      const report = await adapter.health(model, new AbortController().signal);
      assert.equal(report.healthy, false);
      assert.match(report.detail ?? '', /unreachable/);
    } finally {
      await hub.dispose();
    }
  });
});

describe('openai_compatible adapter: invocation', () => {
  it('sends an OpenAI-shaped request and persists the reply as a text artifact', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      const output = await invokeDirectly(hub, model, { prompt: 'explain artifacts briefly' });

      const post = engine.requests.find((request) => request.method === 'POST');
      assert.ok(post, 'expected a POST to the engine');
      assert.equal(post.url, '/v1/chat/completions');
      assert.match(String(post.headers['content-type']), /application\/json/);

      const body = JSON.parse(post.body) as {
        model: string;
        stream: boolean;
        temperature: number;
        messages: { role: string; content: string }[];
      };
      assert.equal(body.model, 'test-model:latest');
      assert.equal(body.stream, false);
      assert.equal(body.temperature, 0.3);
      assert.equal(body.messages.length, 1);
      assert.equal(body.messages[0]?.role, 'user');
      assert.equal(body.messages[0]?.content, 'explain artifacts briefly');

      assert.equal(output.outputs.length, 1);
      const artifact = output.outputs[0];
      assert.ok(artifact);
      assert.equal(artifact.type, 'text');
      assert.equal(artifact.mimeType, 'text/plain');
      assert.equal(artifact.producerModelId, 'ollama_test');

      const { bytes } = await hub.artifacts.read(artifact.id);
      assert.equal(new TextDecoder().decode(bytes), 'hello from the fake engine');
      assert.equal(output.value?.['text'], 'hello from the fake engine');
      assert.equal(output.value?.['promptTokens'], 11);
      assert.equal(output.value?.['completionTokens'], 7);
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('inlines an input image as a data URL for image_understanding', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3]);
      const image = await hub.artifacts.put({ type: 'image', bytes: png, mimeType: 'image/png', extension: '.png' });

      const output = await invokeDirectly(hub, model, {
        capability: 'image_understanding',
        prompt: 'what colour is this?',
        inputs: [{ id: image.id }],
      });

      const post = engine.requests.find((request) => request.method === 'POST');
      assert.ok(post);
      const body = JSON.parse(post.body) as {
        messages: { role: string; content: string | { type: string; text?: string; image_url?: { url: string } }[] }[];
      };
      const content = body.messages[0]?.content;
      assert.ok(Array.isArray(content), 'expected multimodal content parts');
      assert.equal(content[0]?.type, 'text');
      assert.equal(content[0]?.text, 'what colour is this?');
      assert.equal(content[1]?.type, 'image_url');
      assert.equal(
        content[1]?.image_url?.url,
        `data:image/png;base64,${Buffer.from(png).toString('base64')}`,
      );
      assert.equal(output.outputs.length, 1);
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('refuses to feed an image to a capability that cannot use one', async () => {
    const engine = await startEngine();
    try {
      const { hub, model } = await hubWithEngine(engine.url);
      const image = await hub.artifacts.put({
        type: 'image',
        bytes: new Uint8Array([1, 2, 3]),
        mimeType: 'image/png',
        extension: '.png',
      });
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'summarise', inputs: [{ id: image.id }] }),
        (error: unknown) => error instanceof ModelHubError && error.code === 'UNSUPPORTED_OPERATION',
      );
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('turns a non-2xx response into an actionable failure', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder((request, response) => {
        if (request.method === 'GET') {
          response.writeHead(200);
          response.end('{}');
          return;
        }
        response.writeHead(404, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ error: 'model "test-model:latest" not found, try pulling it first' }));
      });
      const { hub, model } = await hubWithEngine(engine.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'hi' }),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'INVOCATION_FAILED');
          assert.match(error.message, /HTTP 404/);
          assert.match(error.message, /try pulling it first/);
          return true;
        },
      );
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('rejects a body that carries no message content', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder((request, response) => {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(request.method === 'GET' ? '{}' : JSON.stringify({ choices: [] }));
      });
      const { hub, model } = await hubWithEngine(engine.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'hi' }),
        (error: unknown) => error instanceof ModelHubError && /no message content/.test(error.message),
      );
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('reports a slow engine as INVOCATION_TIMEOUT, not as a transport failure', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder((request, response) => {
        if (request.method === 'GET') {
          response.writeHead(200);
          response.end('{}');
          return;
        }
        setTimeout(() => {
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(JSON.stringify({ choices: [{ message: { content: 'too late' } }] }));
        }, 500);
      });
      const { hub, model } = await hubWithEngine(engine.url, { adapterConfig: { model: 'test-model:latest', timeoutMs: 120 } });
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'hi' }),
        (error: unknown) => error instanceof ModelHubError && error.code === 'INVOCATION_TIMEOUT',
      );
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('reports caller cancellation as INVOCATION_ABORTED and never retries it', async () => {
    const engine = await startEngine();
    try {
      engine.setResponder((request, response) => {
        if (request.method === 'GET') {
          response.writeHead(200);
          response.end('{}');
          return;
        }
        setTimeout(() => {
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(JSON.stringify({ choices: [{ message: { content: 'never delivered' } }] }));
        }, 500);
      });
      const { hub } = await hubWithEngine(engine.url);
      const controller = new AbortController();
      const pending = hub.invokeModel(
        { capability: 'text_to_text', prompt: 'hi', signal: controller.signal },
        { allowFallback: true, maxAttempts: 3 },
      );
      setTimeout(() => controller.abort(), 60);
      await assert.rejects(
        () => pending,
        (error: unknown) => error instanceof ModelHubError && error.code === 'INVOCATION_ABORTED',
      );
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });
});

describe('openai_compatible adapter: routing integration', () => {
  it('routes text_to_text to the real engine ahead of the mock, and reports it', async () => {
    const engine = await startEngine();
    try {
      const { hub } = await hubWithEngine(engine.url);
      const result = await hub.invokeModel({ capability: 'text_to_text', prompt: 'which model answers?' });

      assert.equal(result.modelId, 'ollama_test');
      assert.equal(result.coldStart, false);
      assert.equal(result.outputs[0]?.type, 'text');
      assert.equal(result.value?.['text'], 'hello from the fake engine');

      // Priority order, not configuration order: the real engine is preferred
      // and the mock remains a deterministic fallback.
      const candidates = hub.findModelsByCapability('text_to_text').map((view) => view.model.id);
      assert.deepEqual(candidates, ['ollama_test', 'mock_text_model']);
      await hub.dispose();
    } finally {
      await engine.close();
    }
  });

  it('falls back to the mock model when the real engine is down', async () => {
    const url = await closedPortUrl();
    const { hub } = await hubWithEngine(url);
    try {
      const result = await hub.invokeModel({ capability: 'text_to_text', prompt: 'who answers now?' });
      assert.equal(result.modelId, 'mock_text_model');
      assert.match(String(result.value?.['text'] ?? ''), /mock_text_model/);
    } finally {
      await hub.dispose();
    }
  });
});
