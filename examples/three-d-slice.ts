/**
 * The 3D slice: image → `image_to_3d` → a `.glb` artifact, end to end.
 *
 * This is the demonstration the acceptance criteria describe — "turn this image
 * into a 3D model" — with one honest substitution: the engine at the other end is
 * the twenty-line HTTP server in this file rather than TRELLIS. Everything else is
 * real. The real catalog parser, the real discovery pass against a live server,
 * the real router making a resource-aware decision, the real runtime gate, the real
 * `three_d` adapter speaking the real Gradio queue protocol, and the real artifact
 * store writing a real GLB to disk.
 *
 * Why a stand-in engine rather than a skipped demo:
 *
 * - A test that silently skips when an engine is absent proves nothing, and a demo
 *   that needs a 12 GB checkpoint and ten minutes of downloads is not run.
 * - The point being demonstrated is the *boundary*, and the boundary is identical
 *   for the stand-in and for TRELLIS. That is the whole claim of the design: the
 *   agent names `image_to_3d` and the hub does the rest, whichever engine answers.
 *
 * To run it against a real engine instead, start one (see `docs/three-d.md`) and
 * run the same three capability calls from `examples/vertical-slice.ts` against a
 * catalog entry for it.
 *
 * Run it with:
 *
 *   node examples/three-d-slice.ts
 *
 * @module dsh-ai-model-hub/examples/three-d-slice
 */

import { createServer } from 'node:http';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { createHash } from 'node:crypto';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ModelHub, createThreeDDiscoverer, mergeCatalogConfig } from '../src/index.ts';
import type { ModelHost } from '../src/index.ts';

/** The port the stand-in engine listens on. `0` lets the OS choose one. */
const ENGINE_PORT = 0;

/**
 * A minimal but structurally valid binary glTF with one triangle.
 *
 * Real enough that the adapter's format sniffing, vertex counting, and MIME
 * selection all take their real paths — a file of zero bytes would prove far less.
 *
 * @returns the container's bytes.
 */
function sampleGlb(): Uint8Array {
  const json = JSON.stringify({
    asset: { version: '2.0', generator: 'dsh-ai-model-hub example' },
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 } }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 36 }],
    buffers: [{ byteLength: 36 }],
  });
  const pad = (buffer: Buffer, fill: number): Buffer =>
    buffer.length % 4 === 0 ? buffer : Buffer.concat([buffer, Buffer.alloc(4 - (buffer.length % 4), fill)]);
  const jsonChunk = pad(Buffer.from(json, 'utf8'), 0x20);
  const binChunk = pad(Buffer.alloc(36, 0), 0);
  const total = 12 + 8 + jsonChunk.length + 8 + binChunk.length;
  const out = Buffer.alloc(total);
  out.write('glTF', 0, 'ascii');
  out.writeUInt32LE(2, 4);
  out.writeUInt32LE(total, 8);
  out.writeUInt32LE(jsonChunk.length, 12);
  out.writeUInt32LE(0x4e4f534a, 16);
  jsonChunk.copy(out, 20);
  const offset = 20 + jsonChunk.length;
  out.writeUInt32LE(binChunk.length, offset);
  out.writeUInt32LE(0x004e4942, offset + 4);
  binChunk.copy(out, offset + 8);
  return new Uint8Array(out);
}

/** What the stand-in engine can answer. */
interface FakeEngine {
  /** Its base URL. */
  readonly url: string;
  /** The GLB it hands back. */
  readonly glb: Uint8Array;
  /** Requests it received, for the transcript. */
  readonly requests: string[];
  /** Stop it. */
  close(): Promise<void>;
}

/**
 * Start an engine that speaks Gradio's queue API and returns a GLB.
 *
 * It implements four routes, which is all the `three_d` adapter ever uses:
 * `/gradio_api/config` (the API description discovery and health read), the queue
 * submit, the server-sent-event result stream, and the file route a Gradio server
 * exposes for results it wrote to its own disk.
 *
 * @returns the running engine.
 */
async function startEngine(): Promise<FakeEngine> {
  const glb = sampleGlb();
  const requests: string[] = [];
  const meshPath = `/tmp/gradio/${createHash('sha256').update(glb).digest('hex').slice(0, 8)}/sample.glb`;

  const server = createServer((request: IncomingMessage, response: ServerResponse) => {
    const url = request.url ?? '/';
    request.on('data', () => {});
    request.on('end', () => {
      requests.push(`${request.method ?? 'GET'} ${url}`);
      const path = url.split('?')[0] ?? url;

      if (path === '/gradio_api/config') {
        sendJson(response, {
          version: '5.0.0',
          named_endpoints: { '/image_to_3d': {}, '/extract_glb': {} },
        });
        return;
      }
      if (path.startsWith('/gradio_api/call/') && request.method === 'POST') {
        // Gradio enqueues the job and answers with an event id to poll.
        sendJson(response, { event_id: 'evt-1' });
        return;
      }
      if (path.startsWith('/gradio_api/call/') && request.method === 'GET') {
        // The result arrives as server-sent events; `complete` carries the output
        // list, and a path here is what the adapter turns into an artifact.
        const data = path.endsWith('extract_glb') ? [meshPath, meshPath] : [{ state: 'opaque' }, meshPath];
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        response.end(`event: complete\ndata: ${JSON.stringify({ data })}\n\n`);
        return;
      }
      if (path.startsWith('/gradio_api/file=')) {
        // Gradio's way of serving a file it wrote: the adapter asks for this when
        // the path the engine reported is not readable locally.
        response.writeHead(200, { 'content-type': 'model/gltf-binary' });
        response.end(Buffer.from(glb));
        return;
      }
      response.writeHead(404, { 'content-type': 'application/json' });
      response.end('{"detail":"Not Found"}');
    });
  });

  await new Promise<void>((resolve) => server.listen(ENGINE_PORT, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  return {
    url: `http://127.0.0.1:${port}`,
    glb,
    requests,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

/**
 * Write a JSON response.
 * @param response - the response to write.
 * @param body - the value to serialize.
 */
function sendJson(response: ServerResponse, body: unknown): void {
  response.writeHead(200, { 'content-type': 'application/json' });
  response.end(JSON.stringify(body));
}

/**
 * Run the slice.
 */
async function main(): Promise<void> {
  const artifactRoot = await mkdtemp(join(tmpdir(), 'dsh-ai-model-hub-3d-'));
  const engine = await startEngine();

  // ── 1. Configuration: one host, one model, no code ────────────────────────
  //
  // The host says how to reach the engine and what it can generate. Nothing here
  // names a Python script, a port the agent must know, or a checkpoint path — and
  // the `three_d` adapter is described entirely by the two calls the engine's own
  // API docs list.
  const host: ModelHost = {
    id: 'local_3d',
    name: 'Local 3D engine',
    adapter: 'three_d',
    runtime: { engine: 'three_d', adapter: 'three_d', endpoint: engine.url },
    adapterConfig: {
      steps: [
        { apiName: 'image_to_3d', bind: { image: '$input' } },
        { apiName: 'extract_glb', bind: { state: '$0.0' }, resultFormat: 'glb' },
      ],
    },
  };

  console.log(`catalog: a host pointing at ${engine.url}`);
  console.log('         engine label "three_d", adapter "three_d", no model id anywhere');

  try {
    // ── 2. Discovery: ask the engine what it can actually do ────────────────
    console.log('\n=== discovery: what does the engine expose? ===');
    const discovered = await createThreeDDiscoverer().discover(host, new AbortController().signal);
    for (const descriptor of discovered) {
      console.log(`  ${descriptor.id} — ${descriptor.name}`);
      console.log(`    capabilities: ${(descriptor.capabilities ?? []).join(', ')}`);
    }

    // ── 3. Build the hub from static config + discovery, as a CLI would ─────
    const catalog = mergeCatalogConfig(
      {
        version: '1',
        hosts: [host],
        models: [
          // The text and image models a machine would already have, so the chained
          // workflow below is served by the same router that serves the mesh.
          {
            id: 'demo_image_model',
            name: 'Demo image model',
            type: 'image_generation',
            capabilities: ['text_to_image'],
            adapter: 'mock',
            runtime: { engine: 'in_process_mock', adapter: 'mock' },
            priority: 10,
          },
        ],
      },
      discovered,
    );

    const hub = new ModelHub({
      config: catalog,
      artifactRoot,
      manageTimers: false,
      // The machine probe is genuinely part of routing, so it is left on: the
      // transcript below reports this machine's real VRAM and shows the resource
      // fit that produced the decision.
      log: (message) => {
        const oneLine = message.replace(/\s+/g, ' ');
        console.log(`  [hub] ${oneLine.length > 170 ? `${oneLine.slice(0, 170)}…` : oneLine}`);
      },
    });

    try {
      // The constructor kicks the probe off in the background, because it cannot
      // await a subprocess; a script that is about to print the figures waits for
      // the first one instead of racing it.
      const machine = await hub.refreshResources();
      console.log('\n=== machine ===');
      console.log(
        `  ${machine.vramGb} GiB VRAM` +
          `${machine.availableVramGb === undefined ? '' : ` (${machine.availableVramGb} GiB free)`}, ` +
          `${machine.ramGb} GiB RAM` +
          `${machine.availableRamGb === undefined ? '' : ` (${machine.availableRamGb} GiB free)`}`,
      );
      for (const line of hub.resourceSnapshot()?.evidence ?? []) console.log(`  ${line}`);

      console.log('\n=== catalog: capability discovery ===');
      for (const capability of hub.listCapabilities()) {
        console.log(
          `  ${capability.capability.padEnd(20)} ${capability.inputTypes.join('|')} → ${capability.outputTypes.join('|')}  via ${capability.modelIds.join(', ')}`,
        );
      }

      // ── 4. Health, by capability rather than by engine ────────────────────
      console.log('\n=== health ===');
      const threeDModels = hub.findModelsByCapability('image_to_3d');
      for (const view of threeDModels) {
        const report = await hub.probeModel(view.model.id);
        console.log(`  ${view.model.id}: ${report.healthy ? 'healthy' : 'unhealthy'} — ${report.detail ?? ''}`);
      }

      // ── 5. Generate an image, then turn it into a mesh ───────────────────
      console.log('\n=== text_to_image ===');
      const image = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a futuristic robot' });
      const imageArtifact = image.outputs[0];
      console.log(`  ${image.modelId} → ${imageArtifact?.id} (${imageArtifact?.mimeType ?? 'unknown'})`);

      console.log('\n=== route: image_to_3d ===');
      const decision = await hub.route({
        capability: 'image_to_3d',
        inputs: [{ id: imageArtifact?.id ?? '', type: 'image' }],
      });
      console.log(`  chose: ${decision.modelId}`);
      console.log(`  why:   ${decision.rationale}`);
      for (const candidate of decision.candidates) {
        console.log(`    [${candidate.eligible ? 'eligible' : 'rejected'}] ${candidate.modelId}: ${candidate.reason}`);
      }

      console.log('\n=== invoke: image_to_3d ===');
      const mesh = await hub.invokeModel({
        capability: 'image_to_3d',
        prompt: 'a low-poly game asset',
        inputs: [{ id: imageArtifact?.id ?? '', type: 'image' }],
      });
      const meshArtifact = mesh.outputs[0];
      console.log(`  ${mesh.modelId} served it in ${mesh.durationMs} ms`);
      console.log(`  artifact: ${meshArtifact?.id}`);
      console.log(`    type:     ${meshArtifact?.type}`);
      console.log(`    mime:     ${meshArtifact?.mimeType}`);
      console.log(`    format:   ${String(meshArtifact?.metadata['format'])}`);
      console.log(`    bytes:    ${String(meshArtifact?.metadata['byteLength'])}`);
      console.log(`    vertices: ${String(meshArtifact?.metadata['vertexCount'])}`);
      console.log(`    from:     ${String(meshArtifact?.metadata['sourceArtifactId'])}`);

      const { path } = await hub.artifacts.resolvePath(meshArtifact?.id ?? '');
      console.log(`  file on disk: ${path}`);

      // ── 6. What the agent was never told ─────────────────────────────────
      console.log('\n=== the boundary ===');
      console.log(`  the caller named a capability; the engine saw ${engine.requests.length} HTTP request(s):`);
      for (const request of engine.requests.slice(0, 6)) console.log(`    ${request}`);
      console.log('  no model id, engine name, port, script path or process id crossed it.');
    } finally {
      await hub.dispose();
    }
  } finally {
    await engine.close();
    if (process.exitCode === 1) await rm(artifactRoot, { recursive: true, force: true });
    else console.log(`\nartifacts on disk at: ${artifactRoot}`);
  }
}

await main();
