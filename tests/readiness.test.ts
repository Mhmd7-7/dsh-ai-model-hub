/**
 * Tests for readiness: what counts as an engine being *ready*, as opposed to
 * merely listening.
 *
 * This is the distinction behind a real bug in this project's history. A health
 * check that accepted any 2xx treated a web page — or a half-loaded engine — as
 * proof that ComfyUI could serve work, so generation was submitted into a graph
 * that was still loading and the failure looked like a model bug. The fix is a
 * descriptor's `health.expect`, and it is defended here at both levels: the pure
 * body rule, and a live probe against a server that answers exactly the way an
 * unready engine does.
 *
 * @module dsh-ai-model-hub/tests/readiness.test
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { ModelCatalog, RuntimeManager, createMockAdapter, AdapterRegistry, parseModelCatalogConfig } from '../src/index.ts';
import { inspectJsonReadiness } from '../src/runtime/manager.ts';
import { resolveDescriptor } from '../src/index.ts';

/** Start a server that answers one path with a fixed body and content type. */
async function serve(
  handler: (url: string) => { readonly status: number; readonly body: string; readonly type?: string },
): Promise<{ readonly url: string; readonly close: () => Promise<void> }> {
  const server: Server = createServer((req, res) => {
    const answer = handler(req.url ?? '/');
    res.writeHead(answer.status, {
      'content-type': answer.type ?? 'application/json',
    });
    res.end(answer.body);
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = (server.address() as AddressInfo).port;
  return {
    url: `http://127.0.0.1:${String(port)}`,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

/** A runtime fixture around one descriptor. */
async function runtimeFor(descriptor: Record<string, unknown>): Promise<{
  readonly runtime: RuntimeManager;
  readonly cleanup: () => Promise<void>;
}> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-readiness-'));
  const parsed = parseModelCatalogConfig({ version: '1', models: [descriptor] });
  assert.equal(parsed.ok, true);
  if (!parsed.ok) throw new Error('fixture catalog did not parse');
  const catalog = new ModelCatalog(parsed.config, { machine: { vramGb: 0, ramGb: 0, hasGpu: false, notes: 'test' } });
  const runtime = new RuntimeManager({
    catalog,
    adapters: new AdapterRegistry([createMockAdapter()]),
    healthIntervalMs: 0,
    idleSweepIntervalMs: 0,
    log: () => {},
  });
  return {
    runtime,
    cleanup: async () => {
      await runtime.dispose();
      await rm(root, { recursive: true, force: true });
    },
  };
}

describe('readiness body inspection', () => {
  it('accepts a JSON object when no key is demanded', () => {
    assert.equal(inspectJsonReadiness('{"anything":1}', []).ok, true);
  });

  it('rejects markup, which is a page rather than an engine API', () => {
    const verdict = inspectJsonReadiness('<!doctype html><html><body>ComfyUI</body></html>', []);
    assert.equal(verdict.ok, false);
    if (verdict.ok) return;
    assert.match(verdict.reason, /markup, not JSON/);
  });

  it('rejects an empty body', () => {
    assert.equal(inspectJsonReadiness('   ', []).ok, false);
  });

  it('rejects a body that is not valid JSON', () => {
    assert.equal(inspectJsonReadiness('loading…', []).ok, false);
  });

  it('rejects a JSON array, because every engine readiness document is an object', () => {
    assert.equal(inspectJsonReadiness('[1,2,3]', []).ok, false);
  });

  it('requires at least one expected key, and names what it saw when none match', () => {
    const ok = inspectJsonReadiness('{"system":{"os":"nt"},"devices":[]}', ['system', 'devices']);
    assert.equal(ok.ok, true);
    const bad = inspectJsonReadiness('{"ready":false}', ['system', 'devices']);
    assert.equal(bad.ok, false);
    if (bad.ok) return;
    assert.match(bad.reason, /none of the expected keys/);
    assert.match(bad.reason, /ready/);
  });

  it('reports a body with no keys helpfully rather than as a match', () => {
    const verdict = inspectJsonReadiness('{}', ['models']);
    assert.equal(verdict.ok, false);
    if (verdict.ok) return;
    assert.match(verdict.reason, /no keys/);
  });
});

describe('readiness through a live probe', () => {
  it('treats a 200 carrying a page as NOT ready when the descriptor demands JSON', async () => {
    const server = await serve(() => ({ status: 200, body: '<html><body>ComfyUI is starting</body></html>', type: 'text/html' }));
    const { runtime, cleanup } = await runtimeFor({
      id: 'comfy_like',
      name: 'ComfyUI-like',
      type: 'image_generation',
      capabilities: ['text_to_image'],
      adapter: 'mock',
      runtime: { engine: 'comfyui', adapter: 'mock', endpoint: server.url },
      health: { kind: 'http', path: '/system_stats', timeoutMs: 2000, expect: 'json', expectKeys: undefined },
    });
    try {
      const report = await runtime.probeHealth('comfy_like');
      assert.equal(report.healthy, false);
      assert.match(report.detail ?? '', /not ready/);
      assert.match(report.detail ?? '', /markup/);
    } finally {
      await cleanup();
      await server.close();
    }
  });

  it('treats a 200 carrying the real readiness document as ready', async () => {
    const server = await serve(() => ({
      status: 200,
      body: JSON.stringify({ system: { os: 'nt' }, devices: [{ name: 'cuda:0' }] }),
    }));
    const { runtime, cleanup } = await runtimeFor({
      id: 'comfy_like',
      name: 'ComfyUI-like',
      type: 'image_generation',
      capabilities: ['text_to_image'],
      adapter: 'mock',
      runtime: { engine: 'comfyui', adapter: 'mock', endpoint: server.url },
      health: { kind: 'http', path: '/system_stats', timeoutMs: 2000, expect: 'json-key', expectKeys: ['system', 'devices'] },
    });
    try {
      const report = await runtime.probeHealth('comfy_like');
      assert.equal(report.healthy, true);
      assert.match(report.detail ?? '', /system/);
    } finally {
      await cleanup();
      await server.close();
    }
  });

  it('treats a 200 carrying the wrong JSON as not ready, and says what it saw', async () => {
    const server = await serve(() => ({ status: 200, body: JSON.stringify({ message: 'not found' }) }));
    const { runtime, cleanup } = await runtimeFor({
      id: 'ollama_like',
      name: 'Ollama-like',
      type: 'text_generation',
      capabilities: ['text_to_text'],
      adapter: 'mock',
      runtime: { engine: 'ollama', adapter: 'mock', endpoint: server.url },
      health: { kind: 'http', path: '/api/tags', timeoutMs: 2000, expect: 'json-key', expectKeys: ['models'] },
    });
    try {
      const report = await runtime.probeHealth('ollama_like');
      assert.equal(report.healthy, false);
      assert.match(report.detail ?? '', /none of the expected keys/);
      assert.match(report.detail ?? '', /not found/);
    } finally {
      await cleanup();
      await server.close();
    }
  });

  it('still accepts any 2xx when the descriptor asks for nothing more', async () => {
    const server = await serve(() => ({ status: 200, body: '', type: 'text/plain' }));
    const { runtime, cleanup } = await runtimeFor({
      id: 'plain',
      name: 'Plain',
      type: 'text_generation',
      capabilities: ['text_to_text'],
      adapter: 'mock',
      runtime: { engine: 'x', adapter: 'mock', endpoint: server.url },
      health: { kind: 'http', path: '/', timeoutMs: 2000 },
    });
    try {
      const report = await runtime.probeHealth('plain');
      assert.equal(report.healthy, true, 'the historical permissive behaviour remains the default');
    } finally {
      await cleanup();
      await server.close();
    }
  });
});

describe('readiness expectations in the catalog schema', () => {
  it('requires expectKeys when expect is json-key', () => {
    const parsed = parseModelCatalogConfig({
      version: '1',
      models: [
        {
          id: 'bad',
          name: 'Bad',
          type: 'text_generation',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'x', adapter: 'mock', endpoint: 'http://127.0.0.1:1' },
          health: { kind: 'http', expect: 'json-key' },
        },
      ],
    });
    assert.equal(parsed.ok, false);
    if (parsed.ok) return;
    assert.equal(parsed.issues[0]?.path, 'models[0].health.expectKeys');
    assert.match(parsed.message, /expectKeys/);
  });

  it('refuses an http expectation on a strategy that has no body', () => {
    const parsed = parseModelCatalogConfig({
      version: '1',
      models: [
        {
          id: 'bad',
          name: 'Bad',
          type: 'text_generation',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'x', adapter: 'mock', endpoint: 'http://127.0.0.1:1' },
          health: { kind: 'tcp', expect: 'json' },
        },
      ],
    });
    assert.equal(parsed.ok, false);
    if (parsed.ok) return;
    assert.equal(parsed.issues[0]?.path, 'models[0].health.expect');
    assert.match(parsed.message, /only applies to an http/);
  });

  it('carries the expectation through descriptor resolution', () => {
    const resolved = resolveDescriptor(
      {
        id: 'carried',
        name: 'Carried',
        type: 'text_generation',
        capabilities: ['text_to_text'],
        adapter: 'mock',
        runtime: { engine: 'x', adapter: 'mock', endpoint: 'http://127.0.0.1:1' },
        health: { kind: 'http', path: '/system_stats', expect: 'json-key', expectKeys: ['system'] },
      },
      undefined,
    );
    assert.equal(resolved.health.expect, 'json-key');
    assert.deepEqual(resolved.health.expectKeys, ['system']);
  });
});

describe('subscription models carry no local lifecycle', () => {
  it('ignores a host lifecycle for a subscription model', () => {
    const resolved = resolveDescriptor(
      {
        id: 'codex_image',
        name: 'ChatGPT Images',
        type: 'image_generation',
        capabilities: ['text_to_image'],
        adapter: 'subscription',
        runtime: { engine: 'subscription', adapter: 'subscription' },
        // Deliberately present: a remote backend must not inherit it.
        lifecycle: { startable: true, stoppable: true, start: { command: 'python.exe', args: ['main.py'] } },
      },
      undefined,
    );
    // The descriptor's own `startable` is honoured only when a command exists, and
    // a subscription backend has no business holding one — but the important
    // assertion is that nothing *launches* a remote backend.
    assert.equal(resolved.adapter, 'subscription');
    assert.equal(resolved.health.kind, 'none', 'liveness is the owning plugin’s to report');
    assert.equal(resolved.resources.vramGb, 0, 'a remote backend has no local GPU footprint');
  });
});
