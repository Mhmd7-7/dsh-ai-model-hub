/**
 * Tests for the `comfyui` adapter.
 *
 * The interesting behaviour here is not HTTP plumbing — it is *graph surgery*:
 * finding the prompt node from the sampler's `positive` link, finding the latent
 * node by class type, and leaving everything else untouched. So these tests
 * assert on the graph ComfyUI actually received, and deliberately renumber node
 * ids to prove discovery is structural rather than id-based.
 *
 * @module dsh-ai-model-hub/tests/comfyui
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { IncomingHttpHeaders, ServerResponse } from 'node:http';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, describe, it } from 'node:test';

import type { ModelHub, ResolvedModel } from '../src/index.ts';
import { ModelHub as Hub, renderMockPng } from '../src/index.ts';
import { readPngDimensions } from '../src/adapters/comfyui.ts';

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

/** The PNG the fake server serves from `/view`. */
const SERVED_PNG = renderMockPng(320, 192, 'comfy-fixture', { gridStep: 32 });

/**
 * A minimal but structurally real API-format graph.
 *
 * Node ids are deliberately NOT 1..n and the prompt node id (42) is not adjacent
 * to the sampler (99), so a passing test proves the adapter followed the
 * `positive` link rather than guessing numbers.
 */
function templateGraph(): Record<string, unknown> {
  return {
    '1': { class_type: 'UNETLoader', inputs: { unet_name: 'model.safetensors', weight_dtype: 'default' } },
    '2': { class_type: 'CLIPLoader', inputs: { clip_name: 'encoder.safetensors', type: 'qwen_image' } },
    '3': { class_type: 'VAELoader', inputs: { vae_name: 'vae.safetensors' } },
    '7': { class_type: 'EmptySD3LatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
    '42': { class_type: 'CLIPTextEncode', inputs: { clip: ['2', 0], text: 'TEMPLATE PROMPT' } },
    '43': { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['42', 0] } },
    '99': {
      class_type: 'KSampler',
      inputs: {
        model: ['1', 0],
        positive: ['42', 0],
        negative: ['43', 0],
        latent_image: ['7', 0],
        seed: 111,
        steps: 20,
        cfg: 7,
        sampler_name: 'euler',
        scheduler: 'normal',
        denoise: 1,
      },
    },
    '88': { class_type: 'VAEDecode', inputs: { samples: ['99', 0], vae: ['3', 0] } },
    '5': { class_type: 'SaveImage', inputs: { images: ['88', 0], filename_prefix: 'template' } },
  };
}

/** Start a local ComfyUI double on an ephemeral port. */
async function startComfy(): Promise<FakeComfy> {
  let responder:
    | ((request: CapturedRequest, response: ServerResponse) => void)
    | undefined;
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
      // Default: healthy stats, immediate completion, one image.
      if (captured.url === '/system_stats') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ system: { comfyui_version: 'test' } }));
        return;
      }
      if (captured.url === '/prompt') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ prompt_id: 'pid-1', number: 1 }));
        return;
      }
      if (captured.url.startsWith('/history/')) {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(
          JSON.stringify({
            'pid-1': {
              status: { status_str: 'success', completed: true },
              outputs: { '5': { images: [{ filename: 'out_00001_.png', subfolder: '', type: 'output' }] } },
            },
          }),
        );
        return;
      }
      if (captured.url.startsWith('/view')) {
        response.writeHead(200, { 'content-type': 'image/png' });
        response.end(Buffer.from(SERVED_PNG));
        return;
      }
      response.writeHead(404);
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

/** Temporary artifact roots, cleaned up once at the end. */
const roots: string[] = [];

after(async () => {
  await Promise.all(roots.map((root) => rm(root, { recursive: true, force: true })));
});

/** Build a hub containing a comfyui model plus a mock fallback. */
async function hubWithComfy(
  endpoint: string,
  adapterConfig: Record<string, unknown> = {},
): Promise<{ hub: ModelHub; model: ResolvedModel }> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-comfyui-'));
  roots.push(root);
  const hub = Hub.fromConfig(
    {
      version: '1',
      models: [
        {
          id: 'comfy_test',
          name: 'Test ComfyUI',
          type: 'image_generation',
          capabilities: ['text_to_image'],
          adapter: 'comfyui',
          runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint, path: '/prompt' },
          adapterConfig: { workflow: templateGraph(), pollIntervalMs: 50, ...adapterConfig },
          limits: { maxWidth: 512, maxHeight: 512 },
          priority: 10,
        },
        {
          id: 'mock_image_model',
          name: 'Mock Image',
          type: 'image_generation',
          capabilities: ['text_to_image'],
          adapter: 'mock',
          runtime: { engine: 'in_process_mock', adapter: 'mock' },
          priority: 100,
        },
      ],
    },
    { artifactRoot: root, manageTimers: false, log: () => {} },
  );
  return { hub, model: hub.catalog.requireModel('comfy_test') };
}

/** Invoke the comfyui adapter directly. */
async function invokeDirectly(
  hub: ModelHub,
  model: ResolvedModel,
  request: { prompt?: string; options?: Readonly<Record<string, unknown>>; signal?: AbortSignal } = {},
): Promise<Awaited<ReturnType<ReturnType<ModelHub['adapters']['require']>['invoke']>>> {
  const adapter = hub.adapters.require('comfyui');
  return adapter.invoke({
    model,
    capability: 'text_to_image',
    ...(request.prompt === undefined ? {} : { prompt: request.prompt }),
    inputs: [],
    options: request.options ?? {},
    artifacts: hub.artifacts,
    signal: request.signal ?? new AbortController().signal,
    log: { debug: () => {}, info: () => {}, warn: () => {} },
  });
}

/** Pull the graph out of the queued `/prompt` request. */
function queuedGraph(comfy: FakeComfy): Record<string, { class_type?: unknown; inputs?: Record<string, unknown> }> {
  const queue = comfy.requests.find((entry) => entry.url === '/prompt');
  assert.ok(queue, 'expected a POST to /prompt');
  const body = JSON.parse(queue.body) as Record<string, unknown>;
  return body['prompt'] as Record<string, { class_type?: unknown; inputs?: Record<string, unknown> }>;
}

describe('comfyui adapter: graph surgery', () => {
  it('writes the prompt into the node feeding the sampler positive link, not a guessed id', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url);
      await invokeDirectly(hub, model, { prompt: 'a man' });
      const graph = queuedGraph(comfy);
      // Node 42 is the one wired to KSampler.positive — discovered structurally.
      assert.equal(graph['42']?.inputs?.['text'], 'a man');
      // The template's own unrelated text node must be untouched.
      assert.notEqual(graph['43']?.inputs?.['conditioning'], undefined);
    } finally {
      await comfy.close();
    }
  });

  it('sets width and height on the Empty*Latent* node', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url);
      await invokeDirectly(hub, model, { prompt: 'a man', options: { width: 384, height: 256 } });
      const graph = queuedGraph(comfy);
      assert.equal(graph['7']?.inputs?.['width'], 384);
      assert.equal(graph['7']?.inputs?.['height'], 256);
    } finally {
      await comfy.close();
    }
  });

  it('applies steps, cfg, sampler and seed to the sampler node', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url);
      await invokeDirectly(hub, model, {
        prompt: 'a man',
        options: { steps: 8, cfg: 1, sampler: 'res_multistep', scheduler: 'simple', seed: 4242 },
      });
      const graph = queuedGraph(comfy);
      assert.equal(graph['99']?.inputs?.['steps'], 8);
      assert.equal(graph['99']?.inputs?.['cfg'], 1);
      assert.equal(graph['99']?.inputs?.['sampler_name'], 'res_multistep');
      assert.equal(graph['99']?.inputs?.['scheduler'], 'simple');
      assert.equal(graph['99']?.inputs?.['seed'], 4242);
    } finally {
      await comfy.close();
    }
  });

  it('does not mutate the shared template between invocations', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url);
      await invokeDirectly(hub, model, { prompt: 'first' });
      await invokeDirectly(hub, model, { prompt: 'second' });
      const config = model.adapterConfig['workflow'] as Record<string, { inputs?: Record<string, unknown> }>;
      // The configured template must still hold its placeholder text.
      assert.equal(config['42']?.inputs?.['text'], 'TEMPLATE PROMPT');
    } finally {
      await comfy.close();
    }
  });

  it('sends client_id alongside the graph', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url, { clientId: 'my-client' });
      await invokeDirectly(hub, model, { prompt: 'a man' });
      const queue = comfy.requests.find((entry) => entry.url === '/prompt');
      assert.ok(queue);
      assert.equal((JSON.parse(queue.body) as Record<string, unknown>)['client_id'], 'my-client');
    } finally {
      await comfy.close();
    }
  });
});

describe('comfyui adapter: artifacts', () => {
  it('downloads the image and records its real dimensions', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url);
      const output = await invokeDirectly(hub, model, { prompt: 'a man' });
      assert.equal(output.outputs.length, 1);
      const artifact = output.outputs[0];
      assert.ok(artifact);
      assert.equal(artifact.mimeType, 'image/png');
      assert.equal(artifact.metadata['width'], 320);
      assert.equal(artifact.metadata['height'], 192);
      assert.equal(artifact.metadata['promptId'], 'pid-1');
      const read = await hub.artifacts.read(artifact.id);
      assert.deepEqual(readPngDimensions(read.bytes), { width: 320, height: 192 });
    } finally {
      await comfy.close();
    }
  });

  it('fetches /view with the filename, subfolder and type ComfyUI reported', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((request, response) => {
        if (request.url === '/prompt') {
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(JSON.stringify({ prompt_id: 'pid-9' }));
          return;
        }
        if (request.url.startsWith('/history/')) {
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(
            JSON.stringify({
              'pid-9': {
                status: { status_str: 'success' },
                outputs: { '5': { images: [{ filename: 'x_1_.png', subfolder: 'sub', type: 'temp' }] } },
              },
            }),
          );
          return;
        }
        if (request.url.startsWith('/view')) {
          response.writeHead(200, { 'content-type': 'image/png' });
          response.end(Buffer.from(SERVED_PNG));
          return;
        }
        response.writeHead(404);
        response.end();
      });
      const { hub, model } = await hubWithComfy(comfy.url);
      await invokeDirectly(hub, model, { prompt: 'a man' });
      const view = comfy.requests.find((entry) => entry.url.startsWith('/view'));
      assert.ok(view, 'expected a GET to /view');
      assert.match(view.url, /filename=x_1_\.png/);
      assert.match(view.url, /subfolder=sub/);
      assert.match(view.url, /type=temp/);
    } finally {
      await comfy.close();
    }
  });
});

describe('comfyui adapter: failures are actionable', () => {
  it('reports node_errors when ComfyUI returns no prompt_id', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((_request, response) => {
        response.writeHead(400, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ error: 'invalid prompt', node_errors: { '1': 'bad model' } }));
      });
      const { hub, model } = await hubWithComfy(comfy.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /rejected the queued graph/);
          assert.match(error.message, /bad model/);
          return true;
        },
      );
    } finally {
      await comfy.close();
    }
  });

  it('surfaces an execution error instead of reporting zero images', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((request, response) => {
        if (request.url === '/prompt') {
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(JSON.stringify({ prompt_id: 'pid-err' }));
          return;
        }
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(
          JSON.stringify({
            'pid-err': {
              status: { status_str: 'error', messages: [['execution_error', { exception_message: 'CUDA out of memory' }]] },
              outputs: {},
            },
          }),
        );
      });
      const { hub, model } = await hubWithComfy(comfy.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /failed to execute the graph/);
          assert.match(error.message, /CUDA out of memory/);
          return true;
        },
      );
    } finally {
      await comfy.close();
    }
  });

  it('says a graph produced no images when it succeeded but saved nothing', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((request, response) => {
        if (request.url === '/prompt') {
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(JSON.stringify({ prompt_id: 'pid-empty' }));
          return;
        }
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ 'pid-empty': { status: { status_str: 'success' }, outputs: {} } }));
      });
      const { hub, model } = await hubWithComfy(comfy.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /produced no images/);
          assert.match(error.message, /SaveImage/);
          return true;
        },
      );
    } finally {
      await comfy.close();
    }
  });

  it('reports a slow graph as a timeout', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((request, response) => {
        if (request.url === '/prompt') {
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(JSON.stringify({ prompt_id: 'pid-slow' }));
          return;
        }
        // Never reach history completion.
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end('{}');
      });
      const { hub, model } = await hubWithComfy(comfy.url, { timeoutMs: 250, pollIntervalMs: 50 });
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /did not finish within 250 ms/);
          return true;
        },
      );
    } finally {
      await comfy.close();
    }
  });

  it('reports cancellation as ABORTED', async () => {
    const comfy = await startComfy();
    try {
      comfy.setResponder((request, response) => {
        if (request.url === '/prompt') {
          response.writeHead(200, { 'content-type': 'application/json' });
          response.end(JSON.stringify({ prompt_id: 'pid-abort' }));
          return;
        }
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end('{}');
      });
      const { hub, model } = await hubWithComfy(comfy.url, { timeoutMs: 30_000, pollIntervalMs: 50 });
      const controller = new AbortController();
      const pending = invokeDirectly(hub, model, { prompt: 'a man', signal: controller.signal });
      setTimeout(() => controller.abort(), 80);
      await assert.rejects(pending, (error: Error & { code?: string }) => {
        assert.equal(error.code, 'INVOCATION_ABORTED');
        return true;
      });
    } finally {
      await comfy.close();
    }
  });

  it('requires a prompt', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url);
      await assert.rejects(
        () => invokeDirectly(hub, model, {}),
        (error: Error) => {
          assert.match(error.message, /requires a `prompt`/);
          return true;
        },
      );
    } finally {
      await comfy.close();
    }
  });
});

describe('comfyui adapter: workflow templates from disk', () => {
  it('loads a bare graph and a {client_id, prompt} envelope', async () => {
    const comfy = await startComfy();
    const root = await mkdtemp(join(tmpdir(), 'aimh-comfy-wf-'));
    roots.push(root);
    try {
      const bare = join(root, 'bare.json');
      await writeFile(bare, JSON.stringify(templateGraph()), 'utf8');
      const enveloped = join(root, 'enveloped.json');
      await writeFile(enveloped, JSON.stringify({ client_id: 'x', prompt: templateGraph() }), 'utf8');

      for (const path of [bare, enveloped]) {
        const { hub, model } = await hubWithComfy(comfy.url, { workflow: undefined, workflowPath: path });
        await invokeDirectly(hub, model, { prompt: `via ${path === bare ? 'bare' : 'envelope'}` });
        const graph = queuedGraph(comfy);
        assert.equal(graph['42']?.inputs?.['text'], `via ${path === bare ? 'bare' : 'envelope'}`);
        comfy.requests.length = 0;
      }
    } finally {
      await rm(root, { recursive: true, force: true });
      await comfy.close();
    }
  });

  it('reports an unreadable workflow file clearly', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url, {
        workflow: undefined,
        workflowPath: join(tmpdir(), 'definitely-not-here-12345.json'),
      });
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /could not be read/);
          return true;
        },
      );
    } finally {
      await comfy.close();
    }
  });

  it('reports invalid JSON in a workflow file clearly', async () => {
    const comfy = await startComfy();
    const root = await mkdtemp(join(tmpdir(), 'aimh-comfy-bad-'));
    roots.push(root);
    try {
      const bad = join(root, 'bad.json');
      await writeFile(bad, '{ not json', 'utf8');
      const { hub, model } = await hubWithComfy(comfy.url, { workflow: undefined, workflowPath: bad });
      await assert.rejects(
        () => invokeDirectly(hub, model, { prompt: 'a man' }),
        (error: Error) => {
          assert.match(error.message, /not valid JSON/);
          return true;
        },
      );
    } finally {
      await comfy.close();
    }
  });
});

describe('comfyui adapter: supports and health', () => {
  it('refuses a model with no workflow template', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url, { workflow: undefined });
      const support = hub.adapters.require('comfyui').supports(model);
      assert.equal(support.ok, false);
      assert.match(support.ok === false ? support.reason : '', /workflow template/);
    } finally {
      await comfy.close();
    }
  });

  it('refuses a model with no endpoint', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url);
      const stripped: ResolvedModel = {
        ...model,
        runtime: { engine: model.runtime.engine, adapter: model.runtime.adapter },
      };
      const support = hub.adapters.require('comfyui').supports(stripped);
      assert.equal(support.ok, false);
      assert.match(support.ok === false ? support.reason : '', /requires `runtime\.endpoint`/);
    } finally {
      await comfy.close();
    }
  });

  it('reports health from /system_stats', async () => {
    const comfy = await startComfy();
    try {
      const { hub, model } = await hubWithComfy(comfy.url);
      const report = await hub.adapters.require('comfyui').health(model, new AbortController().signal);
      assert.equal(report.healthy, true);
      assert.match(report.detail ?? '', /system_stats/);
    } finally {
      await comfy.close();
    }
  });
});

describe('comfyui adapter: routing', () => {
  it('prefers the real engine and falls back to the mock when it fails', async () => {
    const comfy = await startComfy();
    try {
      const { hub } = await hubWithComfy(comfy.url);
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a man' });
      assert.equal(result.modelId, 'comfy_test');
      assert.equal(result.outputs.length, 1);

      comfy.setResponder((_request, response) => {
        response.writeHead(500, { 'content-type': 'application/json' });
        response.end('{"error":"broken"}');
      });
      const fellBack = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a man' });
      assert.equal(fellBack.modelId, 'mock_image_model');
    } finally {
      await comfy.close();
    }
  });
});
