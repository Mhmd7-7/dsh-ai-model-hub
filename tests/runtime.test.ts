/**
 * Tests for the runtime manager and the process-execution guardrails.
 *
 * Two things are being defended here. First, lifecycle correctness: a model is
 * started once even under concurrent demand, health transitions are honest, and
 * idle timeouts only ever stop a genuinely idle process. Second — and more
 * important — that the security posture holds: no shell string is ever built, and
 * an executable outside the allowlist is refused.
 *
 * @module dsh-ai-model-hub/tests/runtime.test
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import type { ModelCatalogConfig, ModelDescriptor } from '../src/index.ts';
import {
  AdapterRegistry,
  DEFAULT_EXECUTION_POLICY,
  LocalArtifactStore,
  ModelCatalog,
  RuntimeManager,
  UnsafeCommandError,
  assertAllowedCommand,
  assertSafeArguments,
  buildChildEnvironment,
  commandBasename,
  createMockAdapter,
  probeTcp,
  runCommand,
  withPolicy,
  withTimeout,
} from '../src/index.ts';

/** A runtime fixture. */
interface RuntimeFixture {
  readonly catalog: ModelCatalog;
  readonly runtime: RuntimeManager;
  readonly artifacts: LocalArtifactStore;
  readonly cleanup: () => Promise<void>;
}

/**
 * Build a runtime around a catalog document.
 * @param config - the catalog document.
 * @returns the fixture.
 */
async function fixture(config: ModelCatalogConfig): Promise<RuntimeFixture> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-runtime-'));
  const catalog = new ModelCatalog(config);
  const adapters = new AdapterRegistry([createMockAdapter()]);
  const runtime = new RuntimeManager({
    catalog,
    adapters,
    healthIntervalMs: 0,
    idleSweepIntervalMs: 0,
    log: () => {},
  });
  return {
    catalog,
    runtime,
    artifacts: new LocalArtifactStore({ root }),
    cleanup: async () => {
      await runtime.dispose();
      await rm(root, { recursive: true, force: true });
    },
  };
}

/**
 * Find a port that nothing is listening on.
 *
 * Binding port 0 asks the OS for an unused port; closing it immediately leaves a
 * port that is free *and* known, which is far more reliable than assuming a
 * specific port such as 1 is unused — on this machine it was not.
 *
 * @returns a port number nothing is currently listening on.
 */
async function unusedPort(): Promise<number> {
  const probe = createServer();
  await new Promise<void>((resolve) => probe.listen(0, '127.0.0.1', resolve));
  const port = (probe.address() as AddressInfo).port;
  await new Promise<void>((resolve) => probe.close(() => resolve()));
  return port;
}

/** A model entry pointing at a real HTTP health endpoint. */
function httpModel(port: number, overrides: Record<string, unknown> = {}): ModelDescriptor {
  return {
    id: 'http_model',
    name: 'HTTP Model',
    type: 'custom',
    capabilities: ['text_to_text'],
    adapter: 'mock',
    runtime: { engine: 'test', adapter: 'mock', endpoint: `http://127.0.0.1:${port}` },
    health: { kind: 'http', path: '/', timeoutMs: 1500 },
    lifecycle: { startable: false, stoppable: false },
    ...overrides,
  } as ModelDescriptor;
}

describe('execution guardrails', () => {
  it('normalizes command basenames across platforms', () => {
    assert.equal(commandBasename('python'), 'python');
    assert.equal(commandBasename('/usr/bin/python3'), 'python3');
    assert.equal(commandBasename('C:\\tools\\blender.exe'), 'blender');
    assert.equal(commandBasename('C:/tools/ComfyUI.bat'), 'comfyui');
  });

  it('permits an allowlisted engine', () => {
    assert.doesNotThrow(() => assertAllowedCommand('python', DEFAULT_EXECUTION_POLICY, 'test'));
    assert.doesNotThrow(() => assertAllowedCommand('C:\\tools\\llama-server.exe', DEFAULT_EXECUTION_POLICY, 'test'));
  });

  it('refuses a shell interpreter, which is the whole point of the allowlist', () => {
    for (const dangerous of ['sh', 'bash', 'cmd', 'powershell', 'rm', 'curl']) {
      assert.throws(
        () => assertAllowedCommand(dangerous, DEFAULT_EXECUTION_POLICY, 'test'),
        UnsafeCommandError,
        `${dangerous} must be refused`,
      );
    }
  });

  it('permits anything once the operator opts in', () => {
    const relaxed = withPolicy(DEFAULT_EXECUTION_POLICY, { allowAnyCommand: true });
    assert.doesNotThrow(() => assertAllowedCommand('my-exotic-engine', relaxed, 'test'));
  });

  it('refuses a NUL byte in an argument', () => {
    assert.throws(
      () => assertSafeArguments(['safe', 'evil\0--flag'], DEFAULT_EXECUTION_POLICY, 'test'),
      UnsafeCommandError,
    );
  });

  it('refuses a line break in an argument, which signals a shell string was built', () => {
    assert.throws(
      () => assertSafeArguments(['a\nrm -rf /'], DEFAULT_EXECUTION_POLICY, 'test'),
      UnsafeCommandError,
    );
  });

  it('refuses an oversized argument list', () => {
    const tooMany = Array.from({ length: 300 }, (_unused, index) => `arg${index}`);
    assert.throws(() => assertSafeArguments(tooMany, DEFAULT_EXECUTION_POLICY, 'test'), UnsafeCommandError);
  });

  it('strips credential-shaped variables from a child environment', () => {
    const original = { ...process.env };
    process.env['MY_API_KEY'] = 'super-secret';
    process.env['DEEPSEEK_API_KEY'] = 'super-secret';
    process.env['AWS_ACCESS_KEY_ID'] = 'super-secret';
    process.env['ORDINARY_SETTING'] = 'keep-me';
    try {
      const env = buildChildEnvironment({ EXTRA: 'added' }, DEFAULT_EXECUTION_POLICY);
      assert.equal(env['MY_API_KEY'], undefined);
      assert.equal(env['DEEPSEEK_API_KEY'], undefined);
      assert.equal(env['AWS_ACCESS_KEY_ID'], undefined);
      assert.equal(env['ORDINARY_SETTING'], 'keep-me');
      assert.equal(env['EXTRA'], 'added');
    } finally {
      process.env = original;
    }
  });

  it('reports a refused command as a result rather than throwing', async () => {
    const result = await runCommand({ command: 'definitely-not-allowlisted' }, { timeoutMs: 1000 });
    assert.equal(result.ok, false);
    assert.equal(result.reason, 'spawn_failed');
    assert.match(result.stderr, /allowlist/);
  });

  it('reports a non-zero exit', async () => {
    const result = await runCommand(
      { command: 'node', args: ['-e', 'process.exit(3)'] },
      { timeoutMs: 10_000 },
    );
    assert.equal(result.ok, false);
    assert.equal(result.code, 3);
    assert.equal(result.reason, 'non_zero_exit');
  });

  it('kills a command that exceeds its budget', async () => {
    const result = await runCommand(
      { command: 'node', args: ['-e', 'setTimeout(()=>{}, 60000)'] },
      { timeoutMs: 1200 },
    );
    assert.equal(result.ok, false);
    assert.equal(result.reason, 'timeout');
  });

  it('bounds captured output', async () => {
    const result = await runCommand(
      { command: 'node', args: ['-e', 'for(let i=0;i<20000;i++)console.log("x".repeat(64))'] },
      { timeoutMs: 20_000, maxOutputChars: 1000 },
    );
    assert.ok(result.stdout.length <= 1100, `stdout was ${result.stdout.length} characters`);
  });
});

describe('withTimeout', () => {
  it('resolves a fast operation', async () => {
    await assert.doesNotReject(() => withTimeout(Promise.resolve(42), 1000, 'fast'));
    assert.equal(await withTimeout(Promise.resolve(42), 1000, 'fast'), 42);
  });

  it('rejects with INVOCATION_TIMEOUT', async () => {
    await assert.rejects(
      () => withTimeout(new Promise((resolve) => setTimeout(resolve, 5000)), 100, 'slow'),
      (error: unknown) => (error as { code?: string }).code === 'INVOCATION_TIMEOUT',
    );
  });

  it('rejects with INVOCATION_ABORTED when the caller cancels', async () => {
    const controller = new AbortController();
    const pending = withTimeout(new Promise((resolve) => setTimeout(resolve, 5000)), 5000, 'cancelled', controller.signal);
    controller.abort();
    await assert.rejects(pending, (error: unknown) => (error as { code?: string }).code === 'INVOCATION_ABORTED');
  });

  it('rejects immediately when the signal is already aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      () => withTimeout(Promise.resolve(1), 5000, 'already', controller.signal),
      (error: unknown) => (error as { code?: string }).code === 'INVOCATION_ABORTED',
    );
  });

  it('contains a late rejection from an operation it abandoned', async () => {
    // Regression: a same-process operation cannot be cancelled, so after the
    // deadline wins it keeps running and may reject later. If that rejection has
    // no handler, Node treats it as an unhandled rejection and kills the process —
    // turning "this model was slow" into "the harness died". `withTimeout` must
    // swallow it.
    const observed: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      observed.push(reason);
    };
    process.on('unhandledRejection', onUnhandled);
    try {
      let rejectLate: ((error: Error) => void) | undefined;
      const operation = new Promise<never>((_resolve, reject) => {
        rejectLate = reject;
      });
      await assert.rejects(
        () => withTimeout(operation, 60, 'abandoned'),
        (error: unknown) => (error as { code?: string }).code === 'INVOCATION_TIMEOUT',
      );
      rejectLate?.(new Error('the engine failed after we stopped waiting'));
      // Give the microtask/macrotask queues a chance to surface an unhandled
      // rejection if one were going to happen.
      await new Promise((resolve) => setTimeout(resolve, 50));
      assert.deepEqual(observed, []);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });
});

describe('probeTcp', () => {
  it('detects a listening socket and a refused one', async () => {
    const server = createServer();
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as AddressInfo).port;
    try {
      assert.equal(await probeTcp('127.0.0.1', port, 2000), true);
    } finally {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
    assert.equal(await probeTcp('127.0.0.1', port, 800), false);
  });
});

describe('RuntimeManager', () => {
  it('maps a healthy external endpoint to available/external', async () => {
    const server = createServer((_request, response) => {
      response.writeHead(200);
      response.end('ok');
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = (server.address() as AddressInfo).port;

    const context = await fixture({ models: [httpModel(port)] });
    try {
      const report = await context.runtime.probeHealth('http_model');
      assert.equal(report.healthy, true);
      const status = context.runtime.getModelStatus('http_model');
      assert.equal(status.availability, 'available');
      assert.equal(status.lifecycle, 'external', 'the hub must not claim to own a process it did not start');
      assert.equal(await context.runtime.ensureReady('http_model').then((gate) => gate.coldStart), false);
    } finally {
      await context.cleanup();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  it('reports stopped for an endpoint nothing is listening on', async () => {
    const port = await unusedPort();
    const context = await fixture({ models: [httpModel(port)] });
    try {
      const report = await context.runtime.probeHealth('http_model');
      assert.equal(report.healthy, false);
      const status = context.runtime.getModelStatus('http_model');
      assert.equal(status.availability, 'stopped');
      // This model declares `startable: false`, so its liveness can only ever
      // belong to something outside the hub — `external` is the honest state.
      assert.equal(status.lifecycle, 'external');
    } finally {
      await context.cleanup();
    }
  });

  it('rests a startable model at not_running until a probe proves otherwise', async () => {
    const port = await unusedPort();
    const context = await fixture({
      models: [
        httpModel(port, {
          lifecycle: {
            startable: true,
            stoppable: true,
            start: { command: 'node', args: ['-e', 'process.exit(0)'] },
          },
        }),
      ],
    });
    try {
      // Before any probe: the hub *could* start it, so nothing is known to be alive.
      assert.equal(context.runtime.getModelStatus('http_model').lifecycle, 'not_running');

      await context.runtime.probeHealth('http_model');
      assert.equal(context.runtime.getModelStatus('http_model').lifecycle, 'not_running', 'a failed probe changes nothing');

      // With a real server listening, the same model is reachable but unowned.
      const server = createServer((_request, response) => {
        response.writeHead(200);
        response.end('ok');
      });
      await new Promise<void>((resolve) => server.listen(port, '127.0.0.1', resolve));
      try {
        const report = await context.runtime.probeHealth('http_model');
        assert.equal(report.healthy, true);
        const status = context.runtime.getModelStatus('http_model');
        assert.equal(status.lifecycle, 'external', 'reachable but not started by the hub');
        assert.equal(status.availability, 'available');
      } finally {
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    } finally {
      await context.cleanup();
    }
  });

  it('refuses to run a model that is disabled', async () => {
    const context = await fixture({
      models: [
        {
          id: 'off',
          name: 'Off',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'mock', adapter: 'mock' },
          enabled: false,
        },
      ],
    });
    try {
      assert.equal(context.runtime.getModelStatus('off').availability, 'disabled');
      await assert.rejects(
        () => context.runtime.ensureReady('off'),
        (error: unknown) => (error as { code?: string }).code === 'MODEL_UNAVAILABLE',
      );
      await assert.rejects(
        () => context.runtime.startModel('off'),
        (error: unknown) => (error as { code?: string }).code === 'MODEL_UNAVAILABLE',
      );
    } finally {
      await context.cleanup();
    }
  });

  it('refuses to start a model that has no launch command', async () => {
    const context = await fixture({
      models: [
        {
          id: 'external_only',
          name: 'External',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'mock', adapter: 'mock', endpoint: 'http://127.0.0.1:1' },
          health: { kind: 'tcp', timeoutMs: 500 },
          lifecycle: { startable: false, stoppable: false },
        },
      ],
    });
    try {
      await assert.rejects(
        () => context.runtime.ensureReady('external_only'),
        (error: unknown) => {
          assert.equal((error as { code?: string }).code, 'MODEL_UNAVAILABLE');
          assert.match((error as Error).message, /not startable/);
          return true;
        },
      );
    } finally {
      await context.cleanup();
    }
  });

  it('marks a model with no adapter as an error', async () => {
    const catalog = new ModelCatalog({
      models: [
        {
          id: 'orphan',
          name: 'Orphan',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'cli',
          runtime: { engine: 'nowhere', adapter: 'cli' },
        },
      ],
    });
    const runtime = new RuntimeManager({
      catalog,
      adapters: new AdapterRegistry([]),
      healthIntervalMs: 0,
      idleSweepIntervalMs: 0,
    });
    try {
      const status = runtime.getModelStatus('orphan');
      assert.equal(status.availability, 'error');
      assert.match(status.reason ?? '', /no adapter registered/);
    } finally {
      await runtime.dispose();
    }
  });

  it('starts a real process, waits for health, and stops it', async () => {
    // A tiny HTTP server started as a child, so the full launch → health → stop
    // path runs against a genuine process rather than a simulation.
    const script = [
      "const {createServer}=require('node:http');",
      "const s=createServer((q,r)=>{r.writeHead(200);r.end('ok')});",
      's.listen(Number(process.env.AIMH_TEST_PORT),"127.0.0.1");',
    ].join('');
    const port = await unusedPort();

    const context = await fixture({
      models: [
        {
          id: 'spawned',
          name: 'Spawned',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: {
            engine: 'node-http',
            adapter: 'mock',
            endpoint: `http://127.0.0.1:${port}`,
            env: { AIMH_TEST_PORT: String(port) },
          },
          health: { kind: 'http', path: '/', timeoutMs: 1000 },
          lifecycle: {
            startable: true,
            stoppable: true,
            start: { command: 'node', args: ['-e', script] },
            startupTimeoutMs: 20_000,
            shutdownTimeoutMs: 3_000,
          },
        },
      ],
    });

    try {
      const started = await context.runtime.startModel('spawned');
      assert.equal(started.started, true);
      assert.equal(started.health.healthy, true);

      const status = context.runtime.getModelStatus('spawned');
      assert.equal(status.availability, 'available');
      assert.equal(status.lifecycle, 'running');
      assert.ok((status.pid ?? 0) > 0, 'a starting process should report a pid');

      // Starting again must not spawn a second process.
      const again = await context.runtime.startModel('spawned');
      assert.equal(again.started, false);
      assert.equal(again.alreadyRunning, true);
      assert.equal(context.runtime.getModelStatus('spawned').pid, status.pid);

      const stopped = await context.runtime.stopModel('spawned');
      assert.equal(stopped.stopped, true);
      assert.equal(stopped.wasRunning, true);
      assert.equal(context.runtime.getModelStatus('spawned').availability, 'stopped');
      assert.equal(context.runtime.getModelStatus('spawned').pid, undefined);
    } finally {
      await context.cleanup();
    }
  });

  it('coalesces concurrent cold starts into one process', async () => {
    const port = await unusedPort();

    const script = [
      "const {createServer}=require('node:http');",
      "setTimeout(()=>{const s=createServer((q,r)=>{r.writeHead(200);r.end('ok')});",
      's.listen(Number(process.env.AIMH_TEST_PORT),"127.0.0.1")},600);',
    ].join('');

    const context = await fixture({
      models: [
        {
          id: 'slow_start',
          name: 'Slow Start',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: {
            engine: 'node-http',
            adapter: 'mock',
            endpoint: `http://127.0.0.1:${port}`,
            env: { AIMH_TEST_PORT: String(port) },
          },
          health: { kind: 'http', path: '/', timeoutMs: 500 },
          lifecycle: {
            startable: true,
            stoppable: true,
            start: { command: 'node', args: ['-e', script] },
            startupTimeoutMs: 30_000,
            shutdownTimeoutMs: 3_000,
          },
        },
      ],
    });

    try {
      const gates = await Promise.all([
        context.runtime.ensureReady('slow_start'),
        context.runtime.ensureReady('slow_start'),
        context.runtime.ensureReady('slow_start'),
      ]);
      assert.ok(gates.every((gate) => gate.allowed));
      assert.equal(context.runtime.getModelStatus('slow_start').activeInvocations, 0);
      // Exactly one process: only one start may have reported a spawn.
      assert.equal(context.runtime.getModelStatus('slow_start').lifecycle, 'running');
    } finally {
      await context.cleanup();
    }
  });

  it('fails fast when a launched process exits immediately', async () => {
    const context = await fixture({
      models: [
        {
          id: 'dies',
          name: 'Dies',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'node-http', adapter: 'mock', endpoint: 'http://127.0.0.1:1' },
          health: { kind: 'http', path: '/', timeoutMs: 300 },
          lifecycle: {
            startable: true,
            stoppable: true,
            start: { command: 'node', args: ['-e', 'console.error("boom: model weights missing"); process.exit(2)'] },
            startupTimeoutMs: 20_000,
            shutdownTimeoutMs: 2_000,
          },
        },
      ],
    });
    try {
      await assert.rejects(
        () => context.runtime.startModel('dies'),
        (error: unknown) => {
          assert.equal((error as { code?: string }).code, 'START_FAILED');
          assert.match((error as Error).message, /exited during startup/);
          const details = (error as { details?: Record<string, unknown> }).details ?? {};
          assert.match(String(details['stderr']), /boom: model weights missing/);
          return true;
        },
      );
      assert.equal(context.runtime.getModelStatus('dies').lifecycle, 'failed');
    } finally {
      await context.cleanup();
    }
  });

  it('times out a model that never becomes healthy', async () => {
    const context = await fixture({
      models: [
        {
          id: 'never_ready',
          name: 'Never Ready',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'node-http', adapter: 'mock', endpoint: 'http://127.0.0.1:1' },
          health: { kind: 'http', path: '/', timeoutMs: 200 },
          lifecycle: {
            startable: true,
            stoppable: true,
            start: { command: 'node', args: ['-e', 'setTimeout(()=>{}, 60000)'] },
            startupTimeoutMs: 1500,
            shutdownTimeoutMs: 2_000,
          },
        },
      ],
    });
    try {
      await assert.rejects(
        () => context.runtime.startModel('never_ready'),
        (error: unknown) => {
          assert.equal((error as { code?: string }).code, 'START_FAILED');
          assert.match((error as Error).message, /did not become healthy/);
          return true;
        },
      );
    } finally {
      await context.cleanup();
    }
  });

  it('stops a model that exceeded its idle timeout and spares one in use', async () => {
    const port = await unusedPort();

    const script = [
      "const {createServer}=require('node:http');",
      "const s=createServer((q,r)=>{r.writeHead(200);r.end('ok')});",
      's.listen(Number(process.env.AIMH_TEST_PORT),"127.0.0.1");',
    ].join('');

    const context = await fixture({
      models: [
        {
          id: 'idle_model',
          name: 'Idle',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: {
            engine: 'node-http',
            adapter: 'mock',
            endpoint: `http://127.0.0.1:${port}`,
            env: { AIMH_TEST_PORT: String(port) },
          },
          health: { kind: 'http', path: '/', timeoutMs: 1000 },
          lifecycle: {
            startable: true,
            stoppable: true,
            start: { command: 'node', args: ['-e', script] },
            startupTimeoutMs: 20_000,
            shutdownTimeoutMs: 3_000,
            idleTimeoutMs: 100,
          },
        },
      ],
    });

    try {
      await context.runtime.startModel('idle_model');

      // An in-flight invocation must protect the process from the sweeper.
      context.runtime.beginInvocation('idle_model');
      assert.deepEqual(await context.runtime.stopIdleModels(), []);
      context.runtime.endInvocation('idle_model');

      // With no invocation outstanding and the timeout elapsed, it is swept.
      await new Promise((resolve) => setTimeout(resolve, 150));
      assert.deepEqual(await context.runtime.stopIdleModels(), ['idle_model']);
      assert.equal(context.runtime.getModelStatus('idle_model').availability, 'stopped');
    } finally {
      await context.cleanup();
    }
  });

  it('never sweeps a model that declares no idle timeout', async () => {
    const context = await fixture({
      models: [
        {
          id: 'keeps_running',
          name: 'Keeps Running',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'node-http', adapter: 'mock', endpoint: 'http://127.0.0.1:1' },
          health: { kind: 'http', path: '/', timeoutMs: 200 },
          lifecycle: {
            startable: true,
            stoppable: true,
            start: { command: 'node', args: ['-e', 'setTimeout(()=>{}, 60000)'] },
            startupTimeoutMs: 1000,
            shutdownTimeoutMs: 2_000,
          },
        },
      ],
    });
    try {
      assert.deepEqual(await context.runtime.stopIdleModels(), []);
    } finally {
      await context.cleanup();
    }
  });

  it('reports every status at once', async () => {
    const context = await fixture({
      models: [
        {
          id: 'a',
          name: 'A',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'mock', adapter: 'mock' },
        },
        {
          id: 'b',
          name: 'B',
          type: 'custom',
          capabilities: ['text_to_image'],
          adapter: 'mock',
          runtime: { engine: 'mock', adapter: 'mock' },
        },
      ],
    });
    try {
      const statuses = context.runtime.getAllStatuses();
      assert.deepEqual(statuses.map((status) => status.modelId), ['a', 'b']);
    } finally {
      await context.cleanup();
    }
  });

  it('disposes idempotently', async () => {
    const context = await fixture({
      models: [
        {
          id: 'm',
          name: 'M',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'mock', adapter: 'mock' },
        },
      ],
    });
    await context.runtime.dispose();
    await assert.doesNotReject(() => context.runtime.dispose());
    await context.cleanup();
  });
});
