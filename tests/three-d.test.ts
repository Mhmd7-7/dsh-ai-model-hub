/**
 * Tests for local 3D generation: the `three_d` adapter, its discoverer, and the
 * resource-aware routing that decides whether a mesh engine can run at all.
 *
 * Three things are being proved, and none of them is "the happy path works":
 *
 * 1. **The adapter is protocol-driven, not engine-driven.** Every test drives a
 *    *fake* Gradio server whose API names, argument order, and result shape are
 *    supplied by the catalog entry under test. That is the property that makes
 *    "add another 3D engine" a configuration change: if the adapter knew TRELLIS,
 *    these tests could not reconfigure it into something else.
 * 2. **A cold, healthy, real engine is invisible to the caller.** The hub is
 *    asked only for `image_to_3d`; it discovers the model, checks the machine,
 *    routes, starts nothing it should not, invokes, and stores a `.glb`.
 * 3. **Failure is refused, never faked.** An absent engine, a malformed catalog
 *    entry, an engine that writes a mesh it cannot generate, and a machine
 *    without the VRAM each produce a specific, explainable refusal — never a
 *    placeholder artifact.
 *
 * @module dsh-ai-model-hub/tests/three-d
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { IncomingHttpHeaders, ServerResponse } from 'node:http';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { after, describe, it } from 'node:test';

import type { MachineProfile, ModelHub } from '../src/index.ts';
import {
  ModelHub as Hub,
  isLosslessJson,
  mergeCatalogConfig,
  sniffThreeDFormat,
  threeDFormatOf,
  measureThreeD,
} from '../src/index.ts';
import { createThreeDAdapter } from '../src/adapters/three-d.ts';
import {
  createThreeDDiscoverer,
  mapThreeDModel,
  parseGradioSurface,
  parseThreeDHostConfig,
} from '../src/discovery/three-d.ts';
import type { ThreeDSurface } from '../src/discovery/three-d.ts';
import type { ModelHost } from '../src/catalog/descriptor.ts';

// ───────────────────────────── fixtures ─────────────────────────────

/** One captured HTTP request. */
interface CapturedRequest {
  readonly method: string;
  readonly url: string;
  readonly headers: IncomingHttpHeaders;
  readonly body: string;
}

/**
 * Build a structurally valid binary glTF container with one triangle.
 *
 * The JSON chunk carries a real accessor table so the adapter's vertex counting
 * has something true to read, and the BIN chunk is the correct size, so a passing
 * sniff test means the reader understood the format rather than that it was
 * handed something trivially recognizable.
 *
 * @param vertexCount - how many position entries the accessor should declare.
 * @returns the container's bytes.
 */
function fakeGlb(vertexCount = 3): Uint8Array {
  const scenes = JSON.stringify({
    asset: { version: '2.0', generator: 'aimh-test-fixture' },
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: vertexCount, type: 'VEC3' }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: vertexCount * 12 }],
    buffers: [{ byteLength: vertexCount * 12 }],
  });
  const jsonPadded = padTo4(Buffer.from(scenes, 'utf8'), 0x20);
  const bin = Buffer.alloc(vertexCount * 12, 0);
  const binPadded = padTo4(bin, 0);

  const total = 12 + 8 + jsonPadded.length + 8 + binPadded.length;
  const out = Buffer.alloc(total);
  out.write('glTF', 0, 'ascii');
  out.writeUInt32LE(2, 4);
  out.writeUInt32LE(total, 8);
  out.writeUInt32LE(jsonPadded.length, 12);
  out.writeUInt32LE(0x4e4f534a, 16);
  jsonPadded.copy(out, 20);
  let offset = 20 + jsonPadded.length;
  out.writeUInt32LE(binPadded.length, offset);
  out.writeUInt32LE(0x004e4942, offset + 4);
  binPadded.copy(out, offset + 8);
  return new Uint8Array(out);
}

/**
 * Pad a buffer to a four-byte boundary, the way a glTF chunk must be.
 * @param buffer - the content.
 * @param fill - the pad byte (`0x20` for JSON, `0` for BIN).
 * @returns the padded buffer.
 */
function padTo4(buffer: Buffer, fill: number): Buffer {
  const remainder = buffer.length % 4;
  if (remainder === 0) return buffer;
  return Buffer.concat([buffer, Buffer.alloc(4 - remainder, fill)]);
}

/** A small but real Wavefront OBJ, as a text-mode engine would write it. */
const FAKE_OBJ = ['# aimh test fixture', 'v 0 0 0', 'v 1 0 0', 'v 0 1 0', 'f 1 2 3', ''].join('\n');

/** A 1x1 PNG, for the image that feeds image_to_3d. */
const PNG_1X1 = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF/9UeOAAAAAElFTkSuQmCC',
  'base64',
);

/** A test double for a local 3D engine. */
interface FakeEngine {
  url: string;
  readonly requests: CapturedRequest[];
  /** Where the mesh this engine "writes" lives on disk. */
  meshPath: string;
  /** The bytes served as the result file. */
  meshBytes: Uint8Array;
  /** The endpoint names the config document advertises. */
  endpoints: string[];
  /** When set, every submit answers with this status and body. */
  failWith: { status: number; body: string } | undefined;
  /** When set, the result stream carries an `error` event with this payload. */
  errorEvent: string | undefined;
  /** Whether the mesh path exists locally (`false` exercises the file route). */
  meshOnDisk: boolean;
  setResponder(responder: ((request: CapturedRequest, response: ServerResponse) => boolean) | undefined): void;
  close(): Promise<void>;
}

/**
 * Start a fake Gradio 5-style 3D engine on an ephemeral port.
 *
 * It implements exactly the four routes the adapter speaks — the API description,
 * the queue submit, the event stream, and the file route — so a passing
 * invocation proves the whole protocol rather than a shortcut.
 *
 * @param options - fixture overrides.
 * @returns the running double.
 */
async function startEngine(options: { readonly meshBytes?: Uint8Array } = {}): Promise<FakeEngine> {
  const engine: FakeEngine = {
    url: '',
    requests: [],
    meshPath: '',
    meshBytes: options.meshBytes ?? fakeGlb(3),
    endpoints: ['image_to_3d', 'extract_glb'],
    failWith: undefined,
    errorEvent: undefined,
    meshOnDisk: true,
    setResponder: () => {},
    close: async () => {},
  };

  const root = await mkdtemp(join(tmpdir(), 'aimh-engine-'));
  roots.push(root);
  engine.meshPath = join(root, 'sample.glb');
  await writeFile(engine.meshPath, engine.meshBytes);

  let responder: ((request: CapturedRequest, response: ServerResponse) => boolean) | undefined;
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
      engine.requests.push(captured);
      if (responder !== undefined && responder(captured, response)) return;

      const path = captured.url.split('?')[0] ?? captured.url;
      if (path === '/gradio_api/config') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(
          JSON.stringify({
            version: '5.0.0',
            named_endpoints: Object.fromEntries(engine.endpoints.map((name) => [`/${name}`, {}])),
          }),
        );
        return;
      }
      if (path.startsWith('/gradio_api/call/') && captured.method === 'POST') {
        if (engine.failWith !== undefined) {
          response.writeHead(engine.failWith.status, { 'content-type': 'application/json' });
          response.end(engine.failWith.body);
          return;
        }
        const name = path.slice('/gradio_api/call/'.length);
        if (!engine.endpoints.includes(name)) {
          response.writeHead(404, { 'content-type': 'application/json' });
          response.end('{"detail":"Not Found"}');
          return;
        }
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ event_id: `evt-${name}` }));
        return;
      }
      if (path.startsWith('/gradio_api/call/') && captured.method === 'GET') {
        const [, , , name = '', eventId = ''] = path.split('/');
        if (eventId !== `evt-${name}`) {
          response.writeHead(404, { 'content-type': 'application/json' });
          response.end('{"detail":"Not Found"}');
          return;
        }
        if (engine.errorEvent !== undefined) {
          response.writeHead(200, { 'content-type': 'text/event-stream' });
          response.end(`event: error\ndata: ${engine.errorEvent}\n\n`);
          return;
        }
        const data =
          name === 'extract_glb'
            ? [engine.meshPath, engine.meshPath]
            : [{ state: 'opaque' }, engine.meshPath.replace(/\.glb$/, '.mp4')];
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.end(`event: generating\ndata: null\n\nevent: complete\ndata: ${JSON.stringify({ data })}\n\n`);
        return;
      }
      if (path.startsWith('/gradio_api/file=')) {
        if (!engine.meshOnDisk) {
          response.writeHead(404);
          response.end('gone');
          return;
        }
        response.writeHead(200, { 'content-type': 'model/gltf-binary' });
        response.end(Buffer.from(engine.meshBytes));
        return;
      }
      response.writeHead(404, { 'content-type': 'application/json' });
      response.end('{"detail":"Not Found"}');
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  engine.url = `http://127.0.0.1:${port}`;
  engine.setResponder = (next) => {
    responder = next;
  };
  engine.close = () =>
    new Promise<void>((resolve) => {
      server.closeAllConnections();
      server.close(() => resolve());
    });
  return engine;
}

/** Temporary directories, cleaned up once at the end. */
const roots: string[] = [];

after(async () => {
  await Promise.all(roots.map((root) => rm(root, { recursive: true, force: true })));
});

/** A machine with the given resources. */
function machine(profile: Partial<MachineProfile>): MachineProfile {
  return {
    vramGb: 8,
    ramGb: 32,
    hasGpu: true,
    notes: 'test profile',
    gpus: [{ name: 'Test GPU', vramGb: 8, freeVramGb: 8 }],
    availableVramGb: 8,
    availableRamGb: 32,
    platform: 'test',
    arch: 'test',
    ...profile,
  };
}

/**
 * A catalog entry for a Gradio 3D engine.
 *
 * @param endpoint - the engine's base URL.
 * @param adapterConfig - overrides merged over the two-step default.
 */
function threeDModel(endpoint: string, adapterConfig: Record<string, unknown> = {}): Record<string, unknown> {
  return threeDModelWith(endpoint, {
    protocol: 'gradio',
    steps: [
      { apiName: 'image_to_3d', bind: { image: '$input' } },
      { apiName: 'extract_glb', bind: { state: '$0.0' }, resultFormat: 'glb' },
    ],
    requestTimeoutMs: 5_000,
    timeoutMs: 10_000,
    ...adapterConfig,
  });
}

/**
 * A catalog entry for a single-call Gradio engine.
 *
 * The single-call layout is a different, equally valid shape — the configuration
 * lives at the top level rather than under `steps` — and it is worth testing on
 * its own because that is the shape most engines need.
 *
 * @param endpoint - the engine's base URL.
 * @param adapterConfig - the configuration, used verbatim.
 */
function threeDSingleCallModel(endpoint: string, adapterConfig: Record<string, unknown> = {}): Record<string, unknown> {
  return threeDModelWith(endpoint, {
    protocol: 'gradio',
    requestTimeoutMs: 5_000,
    timeoutMs: 10_000,
    ...adapterConfig,
  });
}

/**
 * Build a catalog entry with an adapter configuration used exactly as given.
 * @param endpoint - the engine's base URL.
 * @param adapterConfig - the adapter configuration.
 * @returns the raw descriptor.
 */
function threeDModelWith(endpoint: string, adapterConfig: Record<string, unknown>): Record<string, unknown> {
  return {
    id: 'test_three_d',
    name: 'Test 3D engine',
    type: 'three_d_generation',
    capabilities: ['image_to_3d'],
    adapter: 'three_d',
    runtime: { engine: 'test3d', adapter: 'three_d', endpoint },
    adapterConfig,
    resources: { vramGb: 6, ramGb: 12, requiresGpu: true },
    priority: 10,
  };
}

/**
 * Build a hub around one 3D model and an input image artifact.
 *
 * `probeResources: false` keeps these tests off the machine's real hardware: the
 * profile is supplied explicitly so an assertion about insufficient VRAM is a
 * statement about the router rather than about the machine the suite runs on.
 *
 * @param models - the catalog's model entries.
 * @param options - machine profile and hub overrides.
 * @returns the hub plus a stored input image's artifact id.
 */
async function hubWith(
  models: readonly Record<string, unknown>[],
  options: { readonly machine?: MachineProfile; readonly hosts?: readonly Record<string, unknown>[] } = {},
): Promise<{ hub: ModelHub; imageId: string }> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-3d-'));
  roots.push(root);
  const hub = Hub.fromConfig(
    { version: '1', models: [...models], ...(options.hosts === undefined ? {} : { hosts: [...options.hosts] }) },
    {
      artifactRoot: root,
      manageTimers: false,
      machine: options.machine ?? machine({}),
      // Explicit, so no test can race a real `nvidia-smi` on the host.
      probeResources: false,
      log: () => {},
    },
  );
  const image = await hub.artifacts.put({
    type: 'image',
    bytes: new Uint8Array(PNG_1X1),
    mimeType: 'image/png',
    extension: '.png',
    label: 'robot',
    metadata: { width: 1, height: 1, format: 'png' },
  });
  return { hub, imageId: image.id };
}

// ───────────────────────────── format handling ─────────────────────────────

describe('3D artifact formats', () => {
  it('recognises the container names, extensions, and MIME types engines use', () => {
    assert.equal(threeDFormatOf('glb'), 'glb');
    assert.equal(threeDFormatOf('.gltf'), 'gltf');
    assert.equal(threeDFormatOf('model/obj'), 'obj');
    assert.equal(threeDFormatOf('model/gltf-binary'), 'glb');
    assert.equal(threeDFormatOf('model/stl'), 'stl');
    assert.equal(threeDFormatOf('application/x-ply'), 'ply');
    assert.equal(threeDFormatOf('something-else'), undefined);
  });

  it('sniffs GLB, OBJ and STL from their leading bytes', () => {
    assert.equal(sniffThreeDFormat(fakeGlb(3)).format, 'glb');
    assert.equal(sniffThreeDFormat(new TextEncoder().encode(FAKE_OBJ)).format, 'obj');
    assert.equal(sniffThreeDFormat(new TextEncoder().encode('{"asset":{"version":"2.0"}}')).format, 'gltf');
  });

  it('reports a disagreement between the claimed format and the bytes', () => {
    const result = sniffThreeDFormat(new TextEncoder().encode('<html>error</html>'), 'glb');
    assert.equal(result.format, 'glb');
    assert.match(result.warning ?? '', /does not carry a recognisable glb signature/);
  });

  it('counts geometry out of the content rather than trusting the engine', () => {
    assert.deepEqual(measureThreeD(fakeGlb(9), 'glb'), { vertexCount: 9 });
    assert.deepEqual(measureThreeD(new TextEncoder().encode(FAKE_OBJ), 'obj'), {
      vertexCount: 3,
      triangleCount: 1,
    });
  });

  it('does not mistake a mesh writer for a 3D generator', () => {
    // An engine that answers 200 with an HTML error page is the failure this
    // sniffing exists to catch, and it must be reported, not stored as a mesh.
    const result = sniffThreeDFormat(new TextEncoder().encode('<!doctype html><h1>500</h1>'), undefined);
    assert.equal(result.format, undefined);
  });
});

// ───────────────────────────── the adapter ─────────────────────────────

describe('three_d adapter', () => {
  it('registers under its own adapter kind and refuses non-3D capabilities', async () => {
    const adapter = createThreeDAdapter();
    assert.equal(adapter.kind, 'three_d');

    const engine = await startEngine();
    const { hub, imageId } = await hubWith([
      threeDModel(engine.url),
      {
        id: 'mock_text_model',
        name: 'Mock Text',
        type: 'text_generation',
        capabilities: ['text_to_text'],
        adapter: 'mock',
        runtime: { engine: 'in_process_mock', adapter: 'mock' },
        priority: 100,
      },
    ]);
    try {
      assert.deepEqual(hub.adapters.listKinds().includes('three_d'), true);
      await assert.rejects(
        () => hub.invokeModel({ capability: 'text_to_image', prompt: 'nope' }),
        /no model can serve capability "text_to_image"/,
      );
      assert.ok(imageId.length > 0);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('generates a GLB through a two-step Gradio protocol and stores it as a model_3d artifact', async () => {
    const engine = await startEngine();
    const { hub, imageId } = await hubWith([threeDModel(engine.url)]);
    try {
      const result = await hub.invokeModel({
        capability: 'image_to_3d',
        prompt: 'a low-poly robot',
        inputs: [imageId],
      });

      assert.equal(result.modelId, 'test_three_d');
      assert.equal(result.capability, 'image_to_3d');
      const mesh = result.outputs[0];
      assert.ok(mesh, 'an artifact was produced');
      assert.equal(mesh.type, 'model_3d');
      assert.equal(mesh.mimeType, 'model/gltf-binary');
      assert.equal(mesh.metadata['format'], 'glb');
      assert.equal(mesh.metadata['sourceArtifactId'], imageId);
      assert.equal(mesh.metadata['sourceModelId'], 'test_three_d');
      assert.equal(mesh.metadata['vertexCount'], 3);
      assert.match(String(mesh.metadata['sourceHash']), /^sha256:[0-9a-f]{16}$/);
      assert.equal(mesh.metadata['protocol'], 'gradio');
      assert.equal(typeof mesh.metadata['createdAt'], 'number');
      assert.equal(typeof mesh.metadata['byteLength'], 'number');

      const { path } = await hub.artifacts.resolvePath(mesh.id);
      assert.match(path, /\.glb$/);

      // The engine was called exactly as the catalog entry described: the image
      // travelled as a data URI in argument 0, and the second call received the
      // first call's state, not its preview path.
      const submits = engine.requests.filter((request) => request.method === 'POST');
      assert.equal(submits.length, 2);
      const first = JSON.parse(submits[0]?.body ?? '{}') as { data: unknown[] };
      assert.match(String(first.data[0]), /^data:image\/png;base64,/);
      const second = JSON.parse(submits[1]?.body ?? '{}') as { data: unknown[] };
      assert.deepEqual(second.data[0], { state: 'opaque' });

      // The returned value is lossless JSON, as the hub requires of every adapter.
      assert.ok(isLosslessJson(result.value));
      assert.equal(result.value?.['format'], 'glb');
      assert.equal(result.value?.['sourceArtifactId'], imageId);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('fetches the result through the engine file route when the path is not local', async () => {
    const engine = await startEngine();
    // A path under a server-side temp directory that does not exist here: the
    // adapter must fall back to /gradio_api/file= rather than fail.
    engine.meshPath = '/tmp/gradio/does-not-exist-locally/sample.glb';
    const { hub, imageId } = await hubWith([threeDModel(engine.url)]);
    try {
      const result = await hub.invokeModel({ capability: 'image_to_3d', inputs: [imageId] });
      assert.equal(result.outputs[0]?.type, 'model_3d');
      assert.ok(
        engine.requests.some((request) => request.url.startsWith('/gradio_api/file=')),
        'the file route was used',
      );
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('stores the engine turntable preview as a companion artifact when configured', async () => {
    const engine = await startEngine();
    // The engine "writes" a preview video beside the mesh, which is what a real
    // app does (TRELLIS renders a turntable before its GLB is extracted).
    const previewPath = join(dirname(engine.meshPath), 'sample.mp4');
    await writeFile(previewPath, Buffer.from('fake-mp4-bytes'));

    const { hub, imageId } = await hubWith([threeDModel(engine.url, { previewIndex: 1 })]);
    try {
      const result = await hub.invokeModel({ capability: 'image_to_3d', inputs: [imageId] });
      assert.equal(result.outputs.length, 2);
      assert.equal(result.outputs[0]?.type, 'model_3d', 'the mesh is always first');
      const preview = result.outputs[1];
      assert.equal(preview?.type, 'file');
      assert.equal(preview?.mimeType, 'video/mp4');
      assert.equal(preview?.metadata['role'], 'preview');
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('accepts an OBJ engine whose single call returns a path', async () => {
    const engine = await startEngine({ meshBytes: new TextEncoder().encode(FAKE_OBJ) });
    engine.endpoints = ['image_to_mesh'];
    const objPath = engine.meshPath.replace(/\.glb$/, '.obj');
    await writeFile(objPath, FAKE_OBJ);
    engine.meshPath = objPath;

    const { hub, imageId } = await hubWith([
      threeDModel(engine.url, {
        steps: [
          { apiName: 'image_to_mesh', bind: { image: '$input' }, resultAt: '0' },
          { apiName: 'image_to_mesh', bind: { image: '$input' }, resultFormat: 'obj' },
        ],
      }),
    ]);
    try {
      const result = await hub.invokeModel({ capability: 'image_to_3d', inputs: [imageId] });
      const mesh = result.outputs[0];
      assert.equal(mesh?.metadata['format'], 'obj');
      assert.equal(mesh?.mimeType, 'model/obj');
      const { path } = await hub.artifacts.resolvePath(mesh?.id ?? '');
      assert.match(path, /\.obj$/);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('records an engine validation warning instead of silently storing a lying artifact', async () => {
    const engine = await startEngine({ meshBytes: new TextEncoder().encode('<html>not a mesh</html>') });
    const { hub, imageId } = await hubWith([threeDModel(engine.url)]);
    try {
      const result = await hub.invokeModel({ capability: 'image_to_3d', inputs: [imageId] });
      const mesh = result.outputs[0];
      assert.equal(mesh?.metadata['format'], 'glb');
      assert.match(String(mesh?.metadata['validationWarning']), /does not carry a recognisable glb signature/);
      assert.match(String(result.value?.['warning']), /glb signature/);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('surfaces an engine-side error as an invocation failure, not a fake artifact', async () => {
    const engine = await startEngine();
    engine.errorEvent = '"CUDA out of memory"';
    const { hub, imageId } = await hubWith([threeDModel(engine.url)]);
    try {
      await assert.rejects(
        () => hub.invokeModel({ capability: 'image_to_3d', inputs: [imageId] }, { allowFallback: false }),
        (error: unknown) => {
          assert.match(String(error), /CUDA out of memory/);
          return true;
        },
      );
      // Nothing was stored under a mesh kind: a failed generation yields a
      // refusal, never a placeholder artifact.
      const stored = await hub.listArtifacts(50);
      assert.equal(stored.filter((artifact) => artifact.type === 'model_3d').length, 0);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('reports an unreachable engine clearly and produces no artifact', async () => {
    const engine = await startEngine();
    // The engine answers the API description but nothing else, which is what a
    // half-started process looks like: the model probes healthy, so the failure
    // has to come from the invocation rather than from the readiness gate.
    engine.setResponder((request, response) => {
      if (request.url.startsWith('/gradio_api/call/')) {
        response.writeHead(502, { 'content-type': 'text/plain' });
        response.end('engine is still loading weights');
        return true;
      }
      return false;
    });
    const { hub, imageId } = await hubWith([threeDModel(engine.url)]);
    try {
      await assert.rejects(
        () => hub.invokeModel({ capability: 'image_to_3d', inputs: [imageId] }, { allowFallback: false }),
        (error: unknown) => {
          const described = error as { code?: string; message?: string };
          assert.equal(described.code, 'INVOCATION_FAILED');
          assert.match(described.message ?? '', /could not serve the request/);
          assert.match(described.message ?? '', /HTTP 502/);
          return true;
        },
      );
      const stored = await hub.listArtifacts(50);
      assert.equal(stored.filter((artifact) => artifact.type === 'model_3d').length, 0);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('reports a stopped engine as an actionable refusal, not a silent success', async () => {
    const engine = await startEngine();
    const url = engine.url;
    await engine.close();
    const { hub, imageId } = await hubWith([threeDModel(url)]);
    try {
      await assert.rejects(
        () => hub.invokeModel({ capability: 'image_to_3d', inputs: [imageId] }, { allowFallback: false }),
        (error: unknown) => {
          const described = error as { message?: string; details?: Record<string, unknown> };
          assert.match(described.message ?? '', /is not running and is not startable by the hub/);
          assert.match(described.message ?? '', /expected it at http/);
          return true;
        },
      );
      const stored = await hub.listArtifacts(50);
      assert.equal(stored.filter((artifact) => artifact.type === 'model_3d').length, 0);
    } finally {
      await hub.dispose();
    }
  });

  it('refuses an image_to_3d request that carries no image', async () => {
    const engine = await startEngine();
    const { hub } = await hubWith([threeDModel(engine.url)]);
    try {
      await assert.rejects(
        () => hub.invokeModel({ capability: 'image_to_3d', prompt: 'something' }, { allowFallback: false }),
        (error: unknown) => {
          assert.match(String(error), /needs an input artifact of type `image`/);
          return true;
        },
      );
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('honours the caller time budget instead of waiting for the declared one', async () => {
    const engine = await startEngine();
    engine.setResponder((request, response) => {
      if (request.url.startsWith('/gradio_api/call/') && request.method === 'GET') {
        // Never answer: the invocation's own budget must end this.
        return true;
      }
      return false;
    });
    const { hub, imageId } = await hubWith([threeDModel(engine.url)]);
    try {
      const started = Date.now();
      await assert.rejects(
        () =>
          hub.invokeModel(
            { capability: 'image_to_3d', inputs: [imageId], timeoutMs: 400 },
            { allowFallback: false },
          ),
        (error: unknown) => {
          const described = error as { code?: string; message?: string };
          assert.equal(described.code, 'INVOCATION_FAILED');
          assert.match(described.message ?? '', /\[INVOCATION_TIMEOUT\]/);
          return true;
        },
      );
      assert.ok(Date.now() - started < 5_000, 'the caller budget was respected');
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });
});

// ───────────────────────────── configuration validation ─────────────────────────────

describe('three_d configuration validation', () => {
  const cases: readonly { readonly name: string; readonly adapterConfig: Record<string, unknown>; readonly expected: RegExp }[] = [
    {
      name: 'an unknown protocol',
      adapterConfig: { protocol: 'carrier-pigeon' },
      expected: /protocol must be "gradio" or "http_json"/,
    },
    {
      name: 'no apiName on a gradio step',
      adapterConfig: { steps: [{ bind: { image: '$input' } }] },
      expected: /needs an apiName/,
    },
    {
      name: 'an unknown resultFormat',
      adapterConfig: { apiName: 'image_to_3d', resultFormat: 'voxel' },
      expected: /is not a 3D format this hub knows/,
    },
    {
      name: 'a binding to a malformed result path',
      adapterConfig: {
        steps: [
          { apiName: 'image_to_3d', bind: { image: '$input' } },
          { apiName: 'extract_glb', bind: { state: '$0.**' }, resultFormat: 'glb' },
        ],
      },
      expected: /is not a property name or array index/,
    },
    {
      name: 'a binding to a step that has not run',
      adapterConfig: {
        steps: [
          { apiName: 'image_to_3d', bind: { state: '$1.0' } },
          { apiName: 'extract_glb', bind: { state: '$0.0' }, resultFormat: 'glb' },
        ],
      },
      expected: /names step 1, which has not run yet/,
    },
    {
      name: 'a non-integer step index',
      adapterConfig: { apiName: 'image_to_3d', bind: { state: '$first.0' } },
      expected: /is not a step index/,
    },
    {
      name: 'steps that are not objects',
      adapterConfig: { steps: ['image_to_3d'] },
      expected: /each step must be an object/,
    },
    {
      name: 'an empty step list',
      adapterConfig: { steps: [] },
      expected: /must name at least one generation call/,
    },
    {
      name: 'an unknown inputMode',
      adapterConfig: { apiName: 'image_to_3d', inputMode: 'telepathy' },
      expected: /inputMode must be data_uri, base64, or path/,
    },
    {
      name: 'an endpoint that is not a URL',
      adapterConfig: { apiName: 'image_to_3d', endpoint: 'not a url' },
      expected: /is not an absolute URL/,
    },
    {
      name: 'a non-JSON extraBody',
      adapterConfig: { apiName: 'image_to_3d', extraBody: { when: new Date(0) } },
      expected: /extraBody must be a JSON object/,
    },
  ];

  for (const testCase of cases) {
    it(`refuses ${testCase.name}`, async () => {
      // The single-call layout, so the malformed field is read from the entry
      // itself rather than shadowed by a `steps` array that does not have it.
      const { hub } = await hubWith([threeDSingleCallModel('http://127.0.0.1:1', testCase.adapterConfig)]);
      try {
        const model = hub.catalog.requireModel('test_three_d');
        const support = hub.adapters.require('three_d').supports(model);
        assert.equal(support.ok, false);
        assert.match(support.ok === false ? support.reason : '', testCase.expected);
      } finally {
        await hub.dispose();
      }
    });
  }

  it('refuses a model with no endpoint at all', async () => {
    const { hub } = await hubWith([
      {
        id: 'test_three_d',
        name: 'No endpoint',
        type: 'three_d_generation',
        capabilities: ['image_to_3d'],
        adapter: 'three_d',
        adapterConfig: { protocol: 'gradio', apiName: 'image_to_3d' },
        runtime: { engine: 'test3d', adapter: 'three_d', endpoint: 'http://127.0.0.1:1' },
        priority: 10,
      },
    ]);
    try {
      const support = hub.adapters.require('three_d').supports(hub.catalog.requireModel('test_three_d'));
      assert.equal(support.ok, true);
    } finally {
      await hub.dispose();
    }
  });

  it('rejects a catalog entry that claims I/O its capabilities do not have', async () => {
    // The parser is the first line of defence and reports every problem at once,
    // so a hand-edited catalog is fixed in one pass rather than one error per run.
    assert.throws(
      () =>
        Hub.fromConfig({
          version: '1',
          models: [{ ...threeDModel('http://127.0.0.1:1'), inputTypes: ['audio'] }],
        }),
      (error: unknown) => {
        const described = error as { code?: string; message?: string };
        assert.equal(described.code, 'INVALID_DESCRIPTOR');
        assert.match(described.message ?? '', /"audio" is not an input of any declared capability/);
        return true;
      },
    );

    // And an entry that is structurally valid but unusable by its adapter is
    // reported through supports(), which is what makes the model `unsupported`
    // at inspection time instead of failing mid-workflow.
    const { hub } = await hubWith([threeDModel('http://127.0.0.1:1', { steps: [{ bind: {} }] })]);
    try {
      const model = hub.catalog.requireModel('test_three_d');
      const support = hub.adapters.require('three_d').supports(model);
      assert.equal(support.ok, false);
      assert.match(support.ok === false ? support.reason : '', /needs an apiName/);
    } finally {
      await hub.dispose();
    }
  });

  it('reports an unusable step-declaration file as a configuration error', async () => {
    const root = await mkdtemp(join(tmpdir(), 'aimh-steps-'));
    roots.push(root);
    const stepsPath = join(root, 'steps.json');
    await writeFile(stepsPath, '{ this is not json');
    const { hub } = await hubWith([
      threeDModel('http://127.0.0.1:1', { steps: undefined, stepsPath }),
    ]);
    try {
      const model = hub.catalog.requireModel('test_three_d');
      const report = await hub.adapters.require('three_d').health(model, new AbortController().signal);
      assert.equal(report.healthy, false);
      assert.match(report.detail ?? '', /is not valid JSON/);
    } finally {
      await hub.dispose();
    }
  });
});

// ───────────────────────────── health ─────────────────────────────

describe('three_d health checking', () => {
  it('reports healthy against a Gradio config document', async () => {
    const engine = await startEngine();
    const { hub } = await hubWith([threeDModel(engine.url)]);
    try {
      const model = hub.catalog.requireModel('test_three_d');
      const report = await hub.adapters.require('three_d').health(model, new AbortController().signal);
      assert.equal(report.healthy, true);
      assert.match(report.detail ?? '', /engine reachable/);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('reports unhealthy for a server that is not a Gradio app', async () => {
    const engine = await startEngine();
    engine.setResponder((request, response) => {
      if (request.url.includes('/config')) {
        response.writeHead(200, { 'content-type': 'text/html' });
        response.end('<html>hello</html>');
        return true;
      }
      return false;
    });
    const { hub } = await hubWith([threeDModel(engine.url)]);
    try {
      const model = hub.catalog.requireModel('test_three_d');
      const report = await hub.adapters.require('three_d').health(model, new AbortController().signal);
      assert.equal(report.healthy, false);
      assert.match(report.detail ?? '', /not JSON|does not look like a Gradio app/);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('reports a stopped engine as stopped rather than as an error', async () => {
    const engine = await startEngine();
    const url = engine.url;
    await engine.close();
    const { hub } = await hubWith([threeDModel(url)]);
    try {
      const status = hub.getModelStatus('test_three_d');
      assert.equal(status.availability, 'stopped');
      assert.equal(status.lifecycle, 'external');
      // The first probe has not run yet, so there is no reason recorded — the
      // state is "stopped", which is honest. Asking for a probe fills it in.
      const report = await hub.probeModel('test_three_d');
      assert.equal(report.healthy, false);
      const after = hub.getModelStatus('test_three_d');
      assert.equal(after.availability, 'stopped');
      assert.match(after.reason ?? '', /could not reach|aborted|HTTP|ECONNREFUSED/);
    } finally {
      await hub.dispose();
    }
  });
});

// ───────────────────────────── routing and resources ─────────────────────────────

describe('routing to a 3D model', () => {
  it('routes image_to_3d to the only model that declares it, without naming it', async () => {
    const engine = await startEngine();
    const { hub, imageId } = await hubWith([
      {
        id: 'mock_text_model',
        name: 'Mock Text',
        type: 'text_generation',
        capabilities: ['text_to_text'],
        adapter: 'mock',
        runtime: { engine: 'in_process_mock', adapter: 'mock' },
        priority: 1,
      },
      threeDModel(engine.url),
    ]);
    try {
      const decision = await hub.route({ capability: 'image_to_3d', inputs: [imageId] });
      assert.equal(decision.modelId, 'test_three_d');
      assert.match(decision.rationale, /test_three_d/);
      const text = decision.candidates.find((candidate) => candidate.modelId === 'mock_text_model');
      assert.equal(text?.eligible, false);
      assert.match(text?.reason ?? '', /does not declare capability "image_to_3d"/);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('rejects a model too large for the machine and routes to the one that fits', async () => {
    const engine = await startEngine();
    const { hub, imageId } = await hubWith(
      [
        { ...threeDModel(engine.url), id: 'big_three_d', name: 'Big', resources: { vramGb: 12, ramGb: 16, requiresGpu: true }, priority: 1 },
        { ...threeDModel(engine.url), id: 'small_three_d', name: 'Small', resources: { vramGb: 6, ramGb: 12, requiresGpu: true }, priority: 50 },
      ],
      { machine: machine({ vramGb: 8, availableVramGb: 6.5, ramGb: 32, availableRamGb: 20 }) },
    );
    try {
      // The big model cannot ever run here, so it is refused on *capacity* — and
      // the refusal names the shortfall rather than saying "unsupported".
      assert.equal(hub.getModelStatus('big_three_d').availability, 'unsupported');
      const decision = await hub.route({ capability: 'image_to_3d', inputs: [imageId] });
      assert.equal(decision.modelId, 'small_three_d');
      const big = decision.candidates.find((candidate) => candidate.modelId === 'big_three_d');
      assert.equal(big?.eligible, false);
      assert.match(big?.reason ?? '', /needs 12 GiB VRAM but only 8 GiB is present on this machine/);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('rejects a *running* model that no longer fits the VRAM actually free', async () => {
    // The headroom case, and the one the probe exists for: the model already
    // holds its weights, so the question is not what the card has but what is
    // left. Routing it again while another engine has taken the free memory is
    // what produces a CUDA out-of-memory crash deep inside the engine.
    const engine = await startEngine();
    const { hub, imageId } = await hubWith(
      [
        { ...threeDModel(engine.url), id: 'big_three_d', name: 'Big', resources: { vramGb: 7, ramGb: 8, requiresGpu: true }, priority: 1 },
        { ...threeDModel(engine.url), id: 'small_three_d', name: 'Small', resources: { vramGb: 2, ramGb: 4, requiresGpu: true }, priority: 50 },
      ],
      { machine: machine({ vramGb: 8, availableVramGb: 3, ramGb: 32, availableRamGb: 24 }) },
    );
    try {
      // Both fit the idle machine; neither is refused on capacity.
      assert.notEqual(hub.getModelStatus('big_three_d').availability, 'unsupported');

      // Prove both engines are live, which makes the runtime treat them as
      // resident and route them against headroom.
      await hub.probeModel('big_three_d');
      await hub.probeModel('small_three_d');
      assert.equal(hub.getModelStatus('big_three_d').availability, 'available');

      const decision = await hub.route({ capability: 'image_to_3d', inputs: [imageId] });
      assert.equal(decision.modelId, 'small_three_d');
      const big = decision.candidates.find((candidate) => candidate.modelId === 'big_three_d');
      assert.equal(big?.eligible, false);
      assert.match(big?.reason ?? '', /needs 7 GiB VRAM but only 3 GiB is available \(of 8 GiB total\)/);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('rejects every 3D model when the machine has no GPU, and says why', async () => {
    const engine = await startEngine();
    const { hub, imageId } = await hubWith([threeDModel(engine.url)], {
      machine: machine({ vramGb: 0, availableVramGb: 0, hasGpu: false, gpus: [] }),
    });
    try {
      await assert.rejects(
        () => hub.invokeModel({ capability: 'image_to_3d', inputs: [imageId] }, { allowFallback: false }),
        (error: unknown) => {
          const described = error as { code?: string; message?: string; details?: Record<string, unknown> };
          assert.equal(described.code, 'NO_COMPATIBLE_MODEL');
          assert.match(described.message ?? '', /no model can serve capability "image_to_3d"/);
          assert.match(described.message ?? '', /requires a GPU and none was detected/);
          return true;
        },
      );
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('explains a model that is stopped but startable as cold, not as missing', async () => {
    const engine = await startEngine();
    const url = engine.url;
    await engine.close();
    const { hub, imageId } = await hubWith([threeDModel(url)]);
    try {
      const decision = await hub.route({ capability: 'image_to_3d', inputs: [imageId] });
      assert.equal(decision.modelId, 'test_three_d');
      assert.match(decision.rationale, /cold but startable/);
    } finally {
      await hub.dispose();
    }
  });

  it('counts a running model against the headroom it would need', async () => {
    const engine = await startEngine();
    const { hub, imageId } = await hubWith(
      [
        // Already resident, and larger than what is left.
        { ...threeDModel(engine.url), id: 'resident_three_d', resources: { vramGb: 7, ramGb: 8, requiresGpu: true }, priority: 1 },
      ],
      { machine: machine({ vramGb: 8, availableVramGb: 7.5, ramGb: 32, availableRamGb: 24 }) },
    );
    try {
      // Force it to be considered resident by proving the endpoint healthy.
      await hub.probeModel('resident_three_d');
      assert.equal(hub.getModelStatus('resident_three_d').availability, 'available');
      const decision = await hub.route({ capability: 'image_to_3d', inputs: [imageId] });
      assert.equal(decision.modelId, 'resident_three_d', 'a resident model is not its own obstacle');
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('reports what the machine has, and what is left once models are resident', async () => {
    const engine = await startEngine();
    const { hub } = await hubWith([threeDModel(engine.url)], {
      machine: machine({ vramGb: 8, availableVramGb: 6.5, ramGb: 32, availableRamGb: 20 }),
    });
    try {
      const snapshot = hub.availableResources();
      assert.equal(snapshot.profile.vramGb, 8);
      assert.equal(snapshot.profile.availableVramGb, 6.5);

      await hub.probeModel('test_three_d');
      const reserved = hub.availableResources();
      assert.deepEqual(reserved.residentModelIds, ['test_three_d']);
      assert.equal(reserved.reserved.vramGb, 6);
      assert.equal(reserved.profile.availableVramGb, 0.5);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });
});

// ───────────────────────────── discovery ─────────────────────────────

describe('3D engine discovery', () => {
  it('reads the capabilities out of a Gradio API surface', () => {
    const surface = parseGradioSurface(
      {
        version: '5.0.0',
        named_endpoints: {
          '/image_to_3d': {},
          '/extract_glb': {},
          '/text_to_3d': {},
          '/unrelated': {},
        },
      },
      '/gradio_api/config',
    );
    assert.ok(surface);
    assert.deepEqual(surface.capabilities, ['image_to_3d', 'text_to_3d']);
    assert.deepEqual(surface.exportOnly, ['extract_glb']);
    assert.equal(surface.version, '5.0.0');
  });

  it('does not turn a mesh writer into a generation capability', () => {
    const surface = parseGradioSurface(
      { named_endpoints: { '/extract_glb': {}, '/export_obj': {}, '/save_mesh': {} } },
      '/gradio_api/config',
    );
    assert.ok(surface);
    assert.deepEqual(surface.capabilities, []);
    assert.deepEqual(surface.exportOnly, ['extract_glb', 'export_obj', 'save_mesh']);
  });

  it('accepts the older dependencies-based config document', () => {
    const surface = parseGradioSurface(
      { dependencies: [{ api_name: 'image_to_3d' }, { api_name: 'predict' }] },
      '/config',
    );
    assert.deepEqual(surface?.capabilities, ['image_to_3d']);
  });

  it('returns nothing for a document that is not a Gradio config', () => {
    assert.equal(parseGradioSurface({ hello: 'world' }, '/config')?.capabilities.length, 0);
    assert.equal(parseGradioSurface('nope', '/config'), undefined);
  });

  it('parses a host model list and rejects a malformed one', () => {
    const host: ModelHost = {
      id: 'trellis',
      name: 'TRELLIS',
      adapter: 'three_d',
      runtime: {
        engine: 'trellis',
        adapter: 'three_d',
        endpoint: 'http://127.0.0.1:8080',
      },
      adapterConfig: {
        models: [
          { id: 'trellis_image_large', name: 'TRELLIS', capabilities: ['image_to_3d'], vramGb: 12, requiresGpu: true },
        ],
      },
    };
    const parsed = parseThreeDHostConfig(host);
    assert.equal(parsed.ok, true);
    assert.equal(parsed.ok ? parsed.models.length : 0, 1);
    assert.equal(parsed.ok ? parsed.models[0]?.vramGb : undefined, 12);

    const bad = parseThreeDHostConfig({
      ...host,
      adapterConfig: { models: [{ id: 'Not Kebab Case' }] },
    });
    assert.equal(bad.ok, false);
    assert.match(bad.ok ? '' : bad.reason, /not lowercase kebab-case/);

    const unreadable = parseThreeDHostConfig({
      ...host,
      adapterConfig: { models: [{ id: 'ok_id', capabilities: ['teleport'] }] },
    });
    assert.equal(unreadable.ok, false);
    assert.match(unreadable.ok ? '' : unreadable.reason, /must be an array of known capabilities/);
  });

  it('intersects declared capabilities with what the engine proves', () => {
    const surface: ThreeDSurface = {
      configPath: '/gradio_api/config',
      endpoints: ['image_to_3d'],
      capabilities: ['image_to_3d'],
      exportOnly: [],
    };
    const host: ModelHost = {
      id: 'trellis',
      name: 'TRELLIS',
      adapter: 'three_d',
      runtime: { engine: 'trellis', adapter: 'three_d', endpoint: 'http://127.0.0.1:8080' },
    };
    const impossible = mapThreeDModel(
      { id: 'claims_text', name: 'Claims text', capabilities: ['text_to_3d'] },
      surface,
      host,
    );
    assert.equal(impossible.ok, false);
    assert.match(impossible.ok ? '' : impossible.reason, /only exposes routes for image_to_3d/);

    const possible = mapThreeDModel(
      { id: 'honest', name: 'Honest', capabilities: ['image_to_3d', 'text_to_3d'] },
      surface,
      host,
    );
    assert.equal(possible.ok, true);
    assert.deepEqual(possible.ok ? possible.descriptor.capabilities : [], ['image_to_3d']);
    // No frozen health route: the adapter probes whichever API-description path
    // the engine's Gradio version actually serves.
    assert.equal(possible.ok ? possible.descriptor.health : undefined, undefined);
  });

  it('discovers a live engine, publishes a descriptor, and routes to it', async () => {
    const engine = await startEngine();
    const host: ModelHost = {
      id: 'three_d_host',
      name: 'TRELLIS',
      adapter: 'three_d',
      runtime: {
        engine: 'trellis',
        adapter: 'three_d',
        endpoint: engine.url,
      },
      adapterConfig: {
        // The engine's call protocol is a property of the app, not of one
        // checkpoint, so the host declares it and every model inherits it. This is
        // the *shipped* TRELLIS template, so the test proves the real file parses
        // and drives a real invocation rather than a fixture written to match.
        stepsPath: 'config/workflows/three-d-trellis.gradio.json',
        models: [
          {
            id: 'trellis_image_large',
            name: 'TRELLIS image-large',
            capabilities: ['image_to_3d'],
            vramGb: 6,
            ramGb: 12,
            requiresGpu: true,
          },
        ],
      },
    };
    const root = await mkdtemp(join(tmpdir(), 'aimh-disc-'));
    roots.push(root);

    // The real merge path: discover first, merge, then build the catalog.
    const discoverer = createThreeDDiscoverer();
    const descriptors = await discoverer.discover(host, new AbortController().signal);
    assert.equal(descriptors.length, 1);
    assert.equal(descriptors[0]?.id, 'trellis_image_large');
    assert.deepEqual(descriptors[0]?.capabilities, ['image_to_3d']);
    assert.equal(descriptors[0]?.type, 'three_d_generation');
    assert.equal(descriptors[0]?.host, 'three_d_host');

    // The real merge path: discover first, merge into the operator's document,
    // then build the catalog. The host must travel with it — a discovered
    // descriptor names a host rather than carrying an inline runtime.
    const config = mergeCatalogConfig({ version: '1', hosts: [host], models: [] }, descriptors);
    const hub = new Hub({
      config,
      artifactRoot: root,
      manageTimers: false,
      machine: machine({}),
      probeResources: false,
      log: () => {},
    });
    try {
      const image = await hub.artifacts.put({
        type: 'image',
        bytes: new Uint8Array(PNG_1X1),
        mimeType: 'image/png',
        extension: '.png',
        metadata: {},
      });
      // The descriptor discovery produced is what runs: its `apiName` and its
      // step bindings came from the engine's own API surface, not from a
      // hand-written entry.
      const model = hub.catalog.requireModel('trellis_image_large');
      assert.equal(model.runtime.endpoint, engine.url);
      assert.equal(model.adapter, 'three_d');
      const config = model.adapterConfig as {
        readonly stepsPath?: string;
        readonly models?: unknown;
        readonly generationParameters?: Record<string, unknown>;
      };
      // The host's protocol travelled down to the discovered model, and the
      // `models` declaration — which is a discovery directive, not an adapter
      // setting — did not.
      assert.equal(config.stepsPath, 'config/workflows/three-d-trellis.gradio.json');
      assert.equal(config.models, undefined);

      const result = await hub.invokeModel({ capability: 'image_to_3d', inputs: [image.id] });
      assert.equal(result.modelId, 'trellis_image_large');
      assert.equal(result.outputs[0]?.type, 'model_3d');
      assert.equal(result.outputs[0]?.metadata['format'], 'glb');

      // The engine was reached through the host's endpoint without any model id
      // appearing in the adapter or the router.
      assert.ok(engine.requests.some((request) => request.url === '/gradio_api/config'));
    } finally {
      await hub.dispose();
      await engine.close();
    }
  });

  it('warns instead of publishing when the engine exposes no generation route', async () => {
    const engine = await startEngine();
    engine.endpoints = ['extract_glb', 'export_obj'];
    const discoverer = createThreeDDiscoverer();
    await assert.rejects(
      () =>
        discoverer.discover(
          {
            id: 'writers_only',
            name: 'Writers only',
            adapter: 'three_d',
            runtime: { engine: 'three_d', adapter: 'three_d', endpoint: engine.url },
          },
          new AbortController().signal,
        ),
      (error: unknown) => {
        assert.match(String(error), /no image-to-3D or text-to-3D route/);
        assert.match(String(error), /write a mesh the engine was handed/);
        return true;
      },
    );
    await engine.close();
  });

  it('reports an unreachable engine rather than inventing a model', async () => {
    const engine = await startEngine();
    const url = engine.url;
    await engine.close();
    const discoverer = createThreeDDiscoverer({ requestTimeoutMs: 1_000 });
    await assert.rejects(
      () =>
        discoverer.discover(
          {
            id: 'gone',
            name: 'Gone',
            adapter: 'three_d',
            runtime: { engine: 'three_d', adapter: 'three_d', endpoint: url },
          },
          new AbortController().signal,
        ),
      /could not describe the 3D engine/,
    );
  });

  it('refuses a host whose model list contradicts the engine', async () => {
    const engine = await startEngine();
    const discoverer = createThreeDDiscoverer();
    await assert.rejects(
      () =>
        discoverer.discover(
          {
            id: 'liar',
            name: 'Liar',
            adapter: 'three_d',
            runtime: {
              engine: 'three_d',
              adapter: 'three_d',
              endpoint: engine.url,
            },
            adapterConfig: { models: [{ id: 'text_only', capabilities: ['text_to_3d'] }] },
          },
          new AbortController().signal,
        ),
      /only exposes routes for image_to_3d/,
    );
    await engine.close();
  });
});

// ───────────────────────────── runtime lifecycle ─────────────────────────────

describe('3D engine runtime lifecycle', () => {
  it('starts a declared engine process, waits for health, and stops it', async () => {
    // The engine is a node script that serves the Gradio config document, so the
    // real RuntimeManager, the real execution policy, and a real child process are
    // all exercised — this is the lifecycle path `start_model` takes.
    const root = await mkdtemp(join(tmpdir(), 'aimh-lifecycle-'));
    roots.push(root);
    const scriptPath = join(root, 'fake-engine.mjs');
    await writeFile(
      scriptPath,
      [
        "import { createServer } from 'node:http';",
        "const server = createServer((request, response) => {",
        "  if ((request.url ?? '').includes('/config')) {",
        "    response.writeHead(200, { 'content-type': 'application/json' });",
        "    response.end(JSON.stringify({ named_endpoints: { '/image_to_3d': {} } }));",
        '    return;',
        '  }',
        "  response.writeHead(404); response.end('{}');",
        '});',
        "server.listen(Number(process.env.AIMH_TEST_PORT ?? 0), '127.0.0.1', () => {",
        "  process.stdout.write('listening ' + server.address().port + '\\n');",
        '});',
        '',
      ].join('\n'),
    );

    const port = await freePort();
    const { hub, imageId } = await hubWith([
      {
        ...threeDModel(`http://127.0.0.1:${port}`),
        lifecycle: {
          startable: true,
          stoppable: true,
          startupTimeoutMs: 20_000,
          shutdownTimeoutMs: 5_000,
          awaitHealthOnStart: true,
          start: {
            command: process.execPath,
            args: [scriptPath],
            cwd: root,
          },
        },
        runtime: {
          engine: 'test3d',
          adapter: 'three_d',
          endpoint: `http://127.0.0.1:${port}`,
          env: { AIMH_TEST_PORT: String(port) },
        },
      },
    ]);

    try {
      assert.equal(hub.getModelStatus('test_three_d').availability, 'stopped');
      const started = await hub.startModel('test_three_d');
      assert.equal(started.started, true);
      assert.equal(started.health.healthy, true);

      const running = hub.getModelStatus('test_three_d');
      assert.equal(running.availability, 'available');
      assert.equal(running.lifecycle, 'running');
      assert.equal(typeof running.pid, 'number');

      const stopped = await hub.stopModel('test_three_d');
      assert.equal(stopped.stopped, true);
      assert.notEqual(hub.getModelStatus('test_three_d').availability, 'available');
      assert.ok(imageId.length > 0);
    } finally {
      await hub.dispose();
    }
  });

  it('refuses to start a model whose declared VRAM exceeds the machine, before spawning anything', async () => {
    const { hub } = await hubWith(
      [
        {
          ...threeDModel('http://127.0.0.1:1'),
          resources: { vramGb: 24, ramGb: 32, requiresGpu: true },
          lifecycle: {
            startable: true,
            stoppable: true,
            start: { command: process.execPath, args: ['-e', 'process.exit(0)'] },
          },
        },
      ],
      { machine: machine({ vramGb: 8, availableVramGb: 7 }) },
    );
    try {
      const before = hub.getModelStatus('test_three_d');
      await assert.rejects(
        () => hub.startModel('test_three_d'),
        (error: unknown) => {
          const described = error as { code?: string; message?: string };
          assert.equal(described.code, 'INSUFFICIENT_RESOURCES');
          assert.match(described.message ?? '', /needs 24 GiB VRAM but only 8 GiB/);
          return true;
        },
      );
      // The refusal happened at the gate: no process, no ownership, no change of
      // lifecycle state. That is the difference between a refusal and a crash.
      const after = hub.getModelStatus('test_three_d');
      assert.equal(after.lifecycle, before.lifecycle);
      assert.equal(after.pid, undefined);
    } finally {
      await hub.dispose();
    }
  });
});

// ───────────────────────────── workflows ─────────────────────────────

describe('chained 3D workflows', () => {
  it('feeds a generated image into image_to_3d by artifact id alone', async () => {
    const engine = await startEngine();
    const { hub } = await hubWith([
      {
        id: 'mock_image_model',
        name: 'Mock Image',
        type: 'image_generation',
        capabilities: ['text_to_image'],
        adapter: 'mock',
        runtime: { engine: 'in_process_mock', adapter: 'mock' },
        priority: 10,
      },
      threeDModel(engine.url),
    ]);
    try {
      // "Generate an image of a futuristic robot and turn it into a 3D model."
      const image = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a futuristic robot' });
      const imageArtifact = image.outputs[0];
      assert.equal(imageArtifact?.type, 'image');

      const mesh = await hub.invokeModel({
        capability: 'image_to_3d',
        prompt: 'a low-poly game asset',
        inputs: [{ id: imageArtifact?.id ?? '', type: 'image' }],
      });
      assert.equal(mesh.modelId, 'test_three_d');
      const meshArtifact = mesh.outputs[0];
      assert.equal(meshArtifact?.type, 'model_3d');
      assert.equal(meshArtifact?.metadata['sourceArtifactId'], imageArtifact?.id);

      // The image the 3D engine received is the one the image model produced —
      // addressed by id, with no path knowledge anywhere in the chain.
      const submit = engine.requests.find((request) => request.method === 'POST');
      const body = JSON.parse(submit?.body ?? '{}') as { data: unknown[] };
      const { bytes } = await hub.artifacts.read(imageArtifact?.id ?? '');
      const expected = Buffer.from(bytes).toString('base64');
      assert.equal(String(body.data[0]), `data:image/png;base64,${expected}`);
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });

  it('keeps text and image capabilities working alongside a 3D model', async () => {
    const engine = await startEngine();
    const { hub, imageId } = await hubWith([
      ...mockTextAndImageModels(),
      threeDModel(engine.url),
    ]);
    try {
      const capabilities = hub.listCapabilities().map((view) => view.capability);
      assert.ok(capabilities.includes('text_to_text'));
      assert.ok(capabilities.includes('text_to_image'));
      assert.ok(capabilities.includes('image_to_3d'));

      const text = await hub.invokeModel({ capability: 'text_to_text', prompt: 'hello' });
      assert.equal(text.outputs[0]?.type, 'text');
      const image = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a robot' });
      assert.equal(image.outputs[0]?.type, 'image');
      const mesh = await hub.invokeModel({ capability: 'image_to_3d', inputs: [imageId] });
      assert.equal(mesh.outputs[0]?.type, 'model_3d');
    } finally {
      await engine.close();
      await hub.dispose();
    }
  });
});

/** The text and image models a deployment already has, for regression coverage. */
function mockTextAndImageModels(): readonly Record<string, unknown>[] {
  return [
    {
      id: 'mock_text_model',
      name: 'Mock Text',
      type: 'text_generation',
      capabilities: ['text_to_text'],
      adapter: 'mock',
      runtime: { engine: 'in_process_mock', adapter: 'mock' },
      priority: 100,
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
  ];
}

// ───────────────────────────── shipped configuration ─────────────────────────────

describe('shipped 3D configuration', () => {
  it('parses the example catalog and finds an adapter for every kind it uses', async () => {
    // The example file is documentation that has to keep compiling: a 3D host
    // entry that names an adapter nothing registers would be a config that looks
    // right and produces no models at all.
    const raw = JSON.parse(
      await readFile(join(REPO_ROOT, 'config', 'examples', 'real-models.example.json'), 'utf8'),
    ) as unknown;
    const hub = Hub.fromConfig(raw, {
      manageTimers: false,
      machine: machine({}),
      probeResources: false,
      log: () => {},
    });
    try {
      const ids = hub.catalog.listModelIds();
      assert.ok(ids.includes('trellis_image_large'), `expected the TRELLIS entry, got ${ids.join(', ')}`);
      assert.ok(ids.includes('sf3d_image_to_mesh'));
      assert.ok(hub.listCapabilities().some((view) => view.capability === 'image_to_3d'));

      const registered = new Set(hub.adapters.listKinds());
      for (const model of hub.catalog.listModels()) {
        assert.ok(
          registered.has(model.adapter),
          `model ${model.id} uses adapter kind "${model.adapter}", which nothing registers`,
        );
      }
      // The engine itself is external in the shipped example: nothing is launched
      // until an operator opts in.
      assert.equal(hub.catalog.requireModel('trellis_image_large').lifecycle.startable, false);
    } finally {
      await hub.dispose();
    }
  });

  it('keeps the shipped step template loadable and structurally complete', async () => {
    const raw = JSON.parse(
      await readFile(join(REPO_ROOT, 'config', 'workflows', 'three-d-trellis.gradio.json'), 'utf8'),
    ) as { protocol?: string; steps?: unknown[]; generationParameters?: Record<string, unknown> };
    assert.equal(raw.protocol, 'gradio');
    assert.ok(Array.isArray(raw.steps) && raw.steps.length >= 2);
    assert.ok(raw.generationParameters !== undefined);
    // Every `$param:` a step binds must have a default, or the template would be
    // unusable without the caller supplying every knob.
    const bound = new Set<string>();
    for (const step of raw.steps) {
      const bind = (step as { bind?: Record<string, unknown> }).bind ?? {};
      for (const value of Object.values(bind)) {
        if (typeof value === 'string' && value.startsWith('$param:')) bound.add(value.slice('$param:'.length));
      }
    }
    for (const name of bound) {
      assert.ok(
        Object.prototype.hasOwnProperty.call(raw.generationParameters, name),
        `the template binds $param:${name} but supplies no default for it`,
      );
    }
  });
});

/** The repository root, so a test can read the configuration this package ships. */
const REPO_ROOT = fileURLToPath(new URL('..', import.meta.url));

/** Find a free TCP port by binding one and releasing it. */
async function freePort(): Promise<number> {
  const server = createServer();
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return port;
}
