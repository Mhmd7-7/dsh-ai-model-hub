/**
 * Tests for the local-engine startup lifecycle.
 *
 * The scenario this exists for: an engine is stopped, a request needs it, and the
 * hub is supposed to bring it up and carry the request through — not report the
 * engine as missing and let something else answer.
 *
 * The engine is a real child process (`tests/helpers/engine.mjs`) because a fake
 * would not exercise the parts that actually break. The fixture binds its port
 * immediately while its readiness route stays `503` for a while, counts what was
 * submitted before it was ready, and counts how many times it was started — so
 * "it worked" and "it worked once, after readiness, on a single process" are
 * distinguishable claims rather than the same green tick.
 *
 * @module dsh-ai-model-hub/tests/startup.test
 */

import { strict as assert } from 'node:assert';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

import { DEFAULT_EXECUTION_POLICY, ModelHub, ModelHubError } from '../src/index.ts';
import type { ExecutionPolicy, ModelHubOptions } from '../src/index.ts';

/** The fixture engine, launched by the runtime manager as a real process. */
const ENGINE = fileURLToPath(new URL('./helpers/engine.mjs', import.meta.url));

/**
 * An execution policy that permits the fixture engine.
 *
 * The shipping allowlist names local inference runtimes — `python`, `ollama`,
 * `llama-server` — and has no opinion about a Node test harness. Widening it here
 * keeps the *shipping* policy untouched and the security tests honest.
 */
const TEST_POLICY: ExecutionPolicy = { ...DEFAULT_EXECUTION_POLICY, allowAnyCommand: true };

/** A port nothing is listening on. */
async function unusedPort(): Promise<number> {
  const probe = createServer();
  await new Promise<void>((resolve) => probe.listen(0, '127.0.0.1', resolve));
  const port = (probe.address() as AddressInfo).port;
  await new Promise<void>((resolve) => probe.close(() => resolve()));
  return port;
}

/** What the fixture engine should do. */
interface EngineBehaviour {
  /** Milliseconds `/ready` answers 503 before 200. */
  readonly readyAfterMs?: number;
  /** Extra command-line flags for the engine. */
  readonly flags?: readonly string[];
}

/** What the fixture engine counted. */
interface EngineStats {
  readonly starts: number;
  readonly generations: number;
  readonly rejected: number;
}

/** The mutable shape of a test catalog, so a launch command can be swapped. */
interface MutableCatalog {
  version: string;
  hosts: {
    id: string;
    name: string;
    adapter: string;
    runtime: Record<string, unknown>;
    health: Record<string, unknown>;
    lifecycle: Record<string, unknown>;
    resources: Record<string, unknown>;
  }[];
  models: Record<string, unknown>[];
}

/**
 * Point a fixture host's launch command at the engine with the given flags.
 *
 * The port travels as an argument because the hub injects no environment of its
 * own — a descriptor declares `runtime.env` or it declares nothing — so the
 * fixture is launched the way a real engine is: with its own CLI.
 *
 * @param config - the catalog to modify in place.
 * @param port - the port the engine must bind.
 * @param args - extra engine arguments.
 * @returns the catalog, for chaining.
 */
function launchWith(config: MutableCatalog, port: number, args: readonly string[] = []): MutableCatalog {
  const host = config.hosts[0];
  assert.ok(host !== undefined);
  host.lifecycle['start'] = { command: process.execPath, args: [ENGINE, `--port=${String(port)}`, ...args] };
  return config;
}

/** A catalog whose one model is served by the fixture engine. */
function engineCatalog(
  port: number,
  options: {
    readonly id?: string;
    readonly startupTimeoutMs?: number;
    /** Drop the launch command, to exercise LIFECYCLE_UNSUPPORTED. */
    readonly withoutCommand?: boolean;
    /** Do not demand a JSON readiness body, to exercise the permissive default. */
    readonly permissiveHealth?: boolean;
    readonly priority?: number;
  } = {},
): { version: string; hosts: Record<string, unknown>[]; models: Record<string, unknown>[] } {
  const id = options.id ?? 'fixture_image';
  return {
    version: '1',
    hosts: [
      {
        id: 'engine_host',
        name: 'Fixture engine',
        adapter: 'http_json',
        runtime: {
          engine: 'fixture',
          adapter: 'http_json',
          endpoint: `http://127.0.0.1:${String(port)}`,
          path: '/generate',
        },
        health: {
          kind: 'http',
          path: '/ready',
          timeoutMs: 1_000,
          ...(options.permissiveHealth === true ? {} : { expect: 'json-key', expectKeys: ['ready'] }),
        },
        lifecycle: {
          startable: true,
          stoppable: true,
          startupTimeoutMs: options.startupTimeoutMs ?? 20_000,
          awaitHealthOnStart: true,
          ...(options.withoutCommand === true
            ? {}
            : {
                start: {
                  command: process.execPath,
                  // The port travels as an argument: the hub injects no environment
                  // of its own, so the fixture is launched the way a real engine
                  // would be — with its own CLI.
                  args: [ENGINE, `--port=${String(port)}`],
                },
              }),
        },
        resources: { vramGb: 0, ramGb: 0 },
      },
    ],
    models: [
      {
        id,
        name: 'Fixture image model',
        type: 'image_generation',
        capabilities: ['text_to_image'],
        host: 'engine_host',
        adapterConfig: {},
        priority: options.priority ?? 10,
        tags: ['local', 'gpu', 'image'],
      },
    ],
  };
}

/** A hub over one fixture model, with process launching permitted. */
async function hubFor(
  config: unknown,
  options: Partial<ModelHubOptions> = {},
): Promise<{
  readonly hub: ModelHub;
  readonly cleanup: () => Promise<void>;
  readonly stats: () => Promise<EngineStats>;
}> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-startup-'));
  const hub = ModelHub.fromConfig(config, {
    artifactRoot: root,
    manageTimers: false,
    probeResources: false,
    log: () => {},
    executionPolicy: TEST_POLICY,
    ...options,
  });
  return {
    hub,
    stats: async () => {
      const endpoint = hub.catalog.listHosts()[0]?.runtime.endpoint;
      assert.ok(endpoint !== undefined);
      return (await (await fetch(`${endpoint}/stats`)).json()) as EngineStats;
    },
    cleanup: async () => {
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    },
  };
}

/** Launch the fixture engine outside the hub and wait until it is genuinely ready. */
async function externalEngine(port: number): Promise<{ readonly stop: () => void }> {
  const child = spawn(process.execPath, [ENGINE], {
    env: { ...process.env, AIMH_TEST_PORT: String(port) },
    stdio: 'ignore',
  });
  const deadline = Date.now() + 15_000;
  for (;;) {
    try {
      const response = await fetch(`http://127.0.0.1:${String(port)}/ready`);
      if (response.ok) break;
    } catch {
      // not up yet
    }
    if (Date.now() > deadline) {
      child.kill('SIGKILL');
      throw new Error('the fixture engine never came up');
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return { stop: () => child.kill('SIGKILL') };
}

describe('local engine startup', () => {
  it('starts a stopped enabled engine, waits for readiness, and generates once', async () => {
    const port = await unusedPort();
    const config = engineCatalog(port, { id: 'slow_engine_image' }) as MutableCatalog;
    const host = config.hosts[0];
    assert.ok(host !== undefined);
    launchWith(config, port, ['--ready-after=1200']);
    const { hub, cleanup, stats } = await hubFor(config);
    try {
      assert.equal(hub.getModelStatus('slow_engine_image').availability, 'stopped', 'the engine is down before the call');

      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'a test image' });
      assert.equal(result.modelId, 'slow_engine_image');
      assert.equal(result.coldStart, true, 'the call is reported as a cold start');
      assert.equal(result.outputs.length, 1);
      assert.equal(result.outputs[0]?.type, 'image');
      assert.equal(hub.getModelStatus('slow_engine_image').lifecycle, 'running', 'the hub owns what it started');

      const counted = await stats();
      assert.equal(counted.generations, 1);
      assert.equal(counted.rejected, 0, 'nothing may be submitted before the readiness check passes');
    } finally {
      await cleanup();
    }
  });

  it('does not submit generation while the engine is only listening', async () => {
    const port = await unusedPort();
    const config = engineCatalog(port, { id: 'listening_image' }) as MutableCatalog;
    const host = config.hosts[0];
    assert.ok(host !== undefined);
    launchWith(config, port, ['--ready-after=1500']);
    const { hub, cleanup, stats } = await hubFor(config);
    try {
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      assert.equal(result.outputs.length, 1);
      const counted = await stats();
      assert.equal(counted.rejected, 0, 'an open port is not readiness');
      assert.equal(counted.generations, 1);
    } finally {
      await cleanup();
    }
  });

  it('treats a 200 carrying a page as not ready, and reports why', async () => {
    const port = await unusedPort();
    const config = engineCatalog(port, { id: 'markup_image' }) as MutableCatalog;
    const host = config.hosts[0];
    assert.ok(host !== undefined);
    launchWith(config, port, ['--ready-as-markup']);
    const { hub, cleanup } = await hubFor(config);
    try {
      await assert.rejects(
        () => hub.startModel('markup_image'),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'START_FAILED');
          assert.match(String(error.details['lastHealth'] ?? ''), /markup/);
          return true;
        },
      );
    } finally {
      await cleanup();
    }
  });

  it('coalesces concurrent cold starts into a single process', async () => {
    const port = await unusedPort();
    const config = engineCatalog(port, { id: 'shared_image' }) as MutableCatalog;
    const host = config.hosts[0];
    assert.ok(host !== undefined);
    launchWith(config, port, ['--ready-after=900']);
    const { hub, cleanup, stats } = await hubFor(config);
    try {
      const results = await Promise.all([
        hub.invokeModel({ capability: 'text_to_image', prompt: 'one' }),
        hub.invokeModel({ capability: 'text_to_image', prompt: 'two' }),
        hub.invokeModel({ capability: 'text_to_image', prompt: 'three' }),
      ]);
      for (const result of results) assert.equal(result.modelId, 'shared_image');
      const counted = await stats();
      assert.equal(counted.starts, 1, 'three concurrent requests must produce one engine, not three');
      assert.equal(counted.generations, 3, 'and all three requests must still be served');
    } finally {
      await cleanup();
    }
  });

  it('reuses an engine started outside the hub instead of restarting it', async () => {
    const port = await unusedPort();
    const external = await externalEngine(port);
    const { hub, cleanup, stats } = await hubFor(engineCatalog(port, { id: 'external_image' }));
    try {
      // The hub has never probed anything: it is constructed and then asked to
      // invoke. Reaching a live endpoint has to be discovered by the invocation
      // path itself, which is the case that matters for a service the operator
      // started by hand.
      assert.equal(hub.getModelStatus('external_image').availability, 'stopped');
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      assert.equal(result.coldStart, false, 'an engine already running is not a cold start');
      const counted = await stats();
      assert.equal(counted.starts, 1, 'the hub must not start a second instance');
      assert.equal(hub.getModelStatus('external_image').lifecycle, 'external');
    } finally {
      await cleanup();
      external.stop();
    }
  });

  it('refuses a model the policy disabled without starting anything', async () => {
    const port = await unusedPort();
    const { hub, cleanup } = await hubFor(engineCatalog(port, { id: 'disabled_image' }));
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: { text_to_image: { order: [{ id: 'disabled_image', enabled: false }] } },
      });
      await assert.rejects(
        () => hub.startModel('disabled_image'),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'POLICY_DISABLED');
          assert.match(error.message, /refusing to start/);
          return true;
        },
      );
      const reachable = await fetch(`http://127.0.0.1:${String(port)}/stats`).then(
        () => true,
        () => false,
      );
      assert.equal(reachable, false, 'a disabled option must not be launched, ever');
    } finally {
      await cleanup();
    }
  });

  it('reports launch permission that is off, rather than starting anyway', async () => {
    const port = await unusedPort();
    // `allowProcessLaunch: false` is a *deployment* switch, expressed in this
    // project by collapsing every descriptor's startable flag before the hub is
    // built — which is what the plugin's `buildHub` does. The manager must then
    // say the engine is not startable by it, never that it is merely stopped.
    const config = engineCatalog(port, { id: 'blocked_image' });
    const collapsed = {
      ...config,
      hosts: config.hosts.map((host) => ({
        ...host,
        lifecycle: { startable: false, stoppable: false },
      })),
    };
    const { hub, cleanup } = await hubFor(collapsed);
    try {
      assert.equal(hub.catalog.requireModel('blocked_image').lifecycle.startable, false);
      await assert.rejects(
        () => hub.startModel('blocked_image'),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'LIFECYCLE_UNSUPPORTED');
          assert.match(error.message, /not startable/);
          return true;
        },
      );
      const reachable = await fetch(`http://127.0.0.1:${String(port)}/stats`).then(
        () => true,
        () => false,
      );
      assert.equal(reachable, false);
    } finally {
      await cleanup();
    }
  });

  it('reports an engine nobody configured a launch command for as LIFECYCLE_UNSUPPORTED', async () => {
    // The realistic catalog: an engine a user manages by hand. It declares no
    // lifecycle at all, so the hub must say it has no command to run rather than
    // implying the engine is broken or missing.
    const port = await unusedPort();
    const config = engineCatalog(port, { id: 'no_command_image', withoutCommand: true }) as MutableCatalog;
    const host = config.hosts[0];
    assert.ok(host !== undefined);
    host.lifecycle = { startable: false };
    const { hub, cleanup } = await hubFor(config);
    try {
      assert.equal(hub.catalog.requireModel('no_command_image').lifecycle.startable, false);
      await assert.rejects(
        () => hub.startModel('no_command_image'),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'LIFECYCLE_UNSUPPORTED');
          assert.match(error.message, /no launch command configured/);
          return true;
        },
      );
      // And an invocation reports the same thing rather than hanging or claiming
      // every option failed for a mysterious reason.
      await assert.rejects(
        () => hub.invokeModel({ capability: 'text_to_image', prompt: 'x' }),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.match(error.message, /no launch command configured|no permitted option|not running/);
          return true;
        },
      );
    } finally {
      await cleanup();
    }
  });

  it('reports a process that dies during startup as START_FAILED with its output', async () => {
    const port = await unusedPort();
    const config = engineCatalog(port, { id: 'dying_image' }) as MutableCatalog;
    const host = config.hosts[0];
    assert.ok(host !== undefined);
    launchWith(config, port, ['--exit-immediately']);
    const { hub, cleanup } = await hubFor(config);
    try {
      await assert.rejects(
        () => hub.startModel('dying_image'),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'START_FAILED');
          assert.match(error.message, /exited during startup|did not become healthy/);
          return true;
        },
      );
      const status = hub.getModelStatus('dying_image');
      assert.equal(status.availability, 'error');
      assert.ok((status.reason ?? '').length > 0, 'the failure carries a reason');
    } finally {
      await cleanup();
    }
  });

  it('reports a readiness timeout as START_FAILED, naming the budget', async () => {
    const port = await unusedPort();
    const config = engineCatalog(port, { id: 'never_ready_image', startupTimeoutMs: 900 }) as MutableCatalog;
    const host = config.hosts[0];
    assert.ok(host !== undefined);
    launchWith(config, port, ['--never-ready']);
    const { hub, cleanup } = await hubFor(config);
    try {
      await assert.rejects(
        () => hub.startModel('never_ready_image'),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'START_FAILED');
          assert.equal(error.details['timeoutMs'], 900);
          assert.match(error.message, /did not become healthy in time/);
          return true;
        },
      );
    } finally {
      await cleanup();
    }
  });

  it('falls past a startup failure to the next permitted option', async () => {
    const deadPort = await unusedPort();
    const livePort = await unusedPort();
    const config = {
      version: '1',
      hosts: [
        {
          ...(engineCatalog(deadPort, { id: 'dead_image' }).hosts[0] as Record<string, unknown>),
          id: 'dead_host',
          lifecycle: {
            startable: true,
            stoppable: true,
            startupTimeoutMs: 800,
            awaitHealthOnStart: true,
            start: { command: process.execPath, args: [ENGINE, '--never-ready'] },
          },
        },
        {
          ...(engineCatalog(livePort, { id: 'live_image' }).hosts[0] as Record<string, unknown>),
          id: 'live_host',
        },
      ],
      models: [
        { ...(engineCatalog(deadPort, { id: 'dead_image' }).models[0] as Record<string, unknown>), host: 'dead_host', priority: 1 },
        { ...(engineCatalog(livePort, { id: 'live_image' }).models[0] as Record<string, unknown>), host: 'live_host', priority: 2 },
      ],
    };
    const { hub, cleanup } = await hubFor(config);
    try {
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      assert.equal(result.modelId, 'live_image', 'a startup failure advances to the next permitted option');
      const attempts = result.attempts ?? [];
      assert.equal(attempts.length, 2);
      assert.equal(attempts[0]?.optionId, 'dead_image');
      assert.equal(attempts[0]?.outcome, 'failed');
      assert.equal(attempts[0]?.stage, 'readiness', 'and the stage says the engine, not the model');
      assert.equal(attempts[1]?.optionId, 'live_image');
      assert.equal(attempts[1]?.outcome, 'succeeded');
    } finally {
      await cleanup();
    }
  });

  it('stops a cold-started engine on dispose', async () => {
    const port = await unusedPort();
    const { hub, cleanup } = await hubFor(engineCatalog(port, { id: 'owned_image' }));
    await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
    assert.equal(hub.getModelStatus('owned_image').lifecycle, 'running');
    await cleanup();
    await new Promise((resolve) => setTimeout(resolve, 300));
    const stillUp = await fetch(`http://127.0.0.1:${String(port)}/stats`).then(
      () => true,
      () => false,
    );
    assert.equal(stillUp, false, 'disposing the hub stops the engine it started');
  });

  it('starts a host that carries no model, which is how a Scan brings an engine up', async () => {
    // The regression this exists for: a ComfyUI host whose catalog entries were all
    // ignored has *no* servable model, so the launch has to be tracked as an engine
    // record — and the policy gate must be told that id is a management action
    // rather than an unknown option. Without that, the hub refused to start its own
    // engine with `"...is not in the catalog"`, and a Scan on a real machine could
    // never bring ComfyUI up.
    const port = await unusedPort();
    const config = engineCatalog(port, { id: 'ignored_image' }) as MutableCatalog;
    config.models = [];
    const { hub, cleanup } = await hubFor(config);
    try {
      assert.deepEqual(hub.catalog.listModels(), [], 'the host genuinely carries no model');

      const result = await hub.startEngineForHost('engine_host');
      assert.equal(result.modelId, 'engine-engine_host');
      assert.equal(result.started, true, 'the engine was started even though nothing routable is declared on it');
      assert.equal(result.health.healthy, true);

      // It is a real process the hub owns: probeable and stoppable by that id.
      assert.equal(hub.getModelStatus('engine-engine_host').lifecycle, 'running');
      const stopped = await hub.stopModel('engine-engine_host');
      assert.equal(stopped.stopped, true);
      assert.equal((await fetch(`http://127.0.0.1:${String(port)}/stats`).then(() => true, () => false)), false);
    } finally {
      await cleanup();
    }
  });
});
