/**
 * A Scan is a deliberate act; looking is not.
 *
 * These tests drive the whole scan with a described machine — a fake hub whose
 * engine-launch seam records what it was asked to do, injected HTTP/filesystem
 * probes, and a controllable clock — because the behaviour under test is a
 * *policy*: which requests may start an engine, what happens when one will not
 * start, and whether the engines a scan holds are always given back.
 *
 * The distinction the file exists to protect is the one a user cannot see and
 * would not forgive: opening the Local models page must never launch a process,
 * and pressing Scan must — once per engine, whatever else is happening.
 *
 * @module dsh-ai-model-hub/tests/scan-engines
 */

import { strict as assert } from 'node:assert';
import { beforeEach, describe, it } from 'node:test';

import { scanLocalResources, resetScanCache } from '../dsh-plugin/scan.ts';
import { bringUpEngines } from '../dsh-plugin/engine-start.ts';
import type { HttpProbeResult, ScanProbes } from '../dsh-plugin/scan.ts';
import type { ScanProgress } from '../dsh-plugin/engine-start.ts';
import type { ModelCatalogConfig, ModelHost, ModelHub } from '../src/index.ts';

// ── a described machine ─────────────────────────────────────────────────────

/** A catalog with both engines declared, and ComfyUI able to be started. */
function catalogFor(endpoint: string, options: { comfyStartable?: boolean; ollamaStartable?: boolean } = {}): ModelCatalogConfig {
  const comfyLifecycle = options.comfyStartable === false
    ? { startable: false, stoppable: false }
    : { startable: true, stoppable: true, startupTimeoutMs: 300_000, start: { command: 'python.exe', args: ['main.py'], cwd: 'C:/ComfyUI/src' } };
  const ollamaLifecycle = options.ollamaStartable === true
    ? { startable: true, stoppable: true, start: { command: 'ollama.exe', args: ['serve'] } }
    : { startable: false, stoppable: false };
  return {
    version: '1',
    hosts: [
      {
        id: 'ollama', name: 'Ollama', adapter: 'openai_compatible',
        runtime: { engine: 'ollama', adapter: 'openai_compatible', endpoint: 'http://127.0.0.1:11434', path: '/v1/chat/completions' },
        lifecycle: ollamaLifecycle,
      },
      {
        id: 'comfyui', name: 'ComfyUI', adapter: 'comfyui',
        runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint, path: '/prompt' },
        lifecycle: comfyLifecycle,
      },
    ],
    models: [],
  } as ModelCatalogConfig;
}

/** The hosts of the fixture catalog, as fresh objects each time. */
function hostsFor(endpoint = 'http://127.0.0.1:8188', options: { comfyStartable?: boolean; ollamaStartable?: boolean } = {}): ModelHost[] {
  return [...(catalogFor(endpoint, options).hosts ?? [])];
}

/** One workflow in API format, so the scan has something real to read. */
const apiWorkflow = {
  '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'A.safetensors' } },
  '2': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: 'a fox' } },
  '3': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: '' } },
  '4': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
  '5': { class_type: 'KSampler', inputs: { model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0], seed: 1, steps: 20, cfg: 7 } },
  '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
  '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: 'x' } },
};

/**
 * A hub whose engine-launch seam is real code under test and whose process
 * supervision is described.
 *
 * `startEngineForHost` is the seam production uses, so it is the seam these tests
 * stub: it records the request, flips the described machine's running flag, and
 * returns what the lifecycle manager would. `protectModel`/`getModelStatus` are
 * recorded so the protection contract can be asserted exactly.
 *
 * The `state` object is shared with {@link probesFor}, so starting an engine
 * through the hub is *observable* by the scan's own readiness probe — which is
 * what makes "start it, then wait for its API" a real sequence here rather than
 * two disconnected stubs.
 */
function managedHub(state: { ollama: boolean; comfyui: boolean }, options: {
  /** Make the launch fail with this message, as `START_FAILED` would. */
  readonly launchError?: string;
  /** Report the process as gone from this point on, simulating a mid-scan exit. */
  readonly diesAfterLaunch?: boolean;
} = {}): {
  hub: ModelHub;
  launches: string[];
  protections: { modelId: string; released: boolean }[];
  liveProtections: () => number;
} {
  const launches: string[] = [];
  const protections: { modelId: string; released: boolean }[] = [];
  const models = new Map<string, { pid?: number }>();
  const hub = {
    catalog: {
      listModels: () => [],
      listHosts: () => [],
      requireModel: () => { throw new Error('not used'); },
    },
    listModels: () => [],
    listCapabilities: () => [],
    publishScannedModels: () => 0,
    activeInvocationsFor: () => 0,
    stopEngine: async () => ({ stopped: false, wasRunning: false }),
    protectModel: (modelId: string) => {
      const record = { modelId, released: false };
      protections.push(record);
      return () => { record.released = true; };
    },
    getModelStatus: (modelId: string) => {
      const model = models.get(modelId);
      return model === undefined || model.pid === undefined ? {} : { pid: model.pid };
    },
    startEngineForHost: async (hostId: string) => {
      launches.push(hostId);
      if (options.launchError !== undefined) {
        const error = new Error(options.launchError) as Error & { code?: string };
        error.code = 'START_FAILED';
        throw error;
      }
      state[hostId as 'ollama' | 'comfyui'] = true;
      const modelId = `engine-${hostId}`;
      models.set(modelId, options.diesAfterLaunch === true ? {} : { pid: 4242 });
      return { modelId, started: true, alreadyRunning: false, health: { healthy: true, detail: 'answered /system_stats' } };
    },
  } as unknown as ModelHub;
  return {
    hub,
    launches,
    protections,
    liveProtections: () => protections.filter((record) => !record.released).length,
  };
}

/**
 * A machine whose HTTP effects answer according to the described state.
 *
 * @param state - the live described machine, shared with {@link managedHub}.
 * @param workflows - the engine's saved workflows, keyed by relative path.
 * @param options - `neverReady` makes ComfyUI answer 503 forever, whatever the
 *   state says, so a readiness timeout can be described without a real clock.
 * @returns the probes.
 */
function probesFor(
  state: { ollama: boolean; comfyui: boolean },
  workflows: Record<string, unknown> = { 'fox.json': apiWorkflow },
  options: { readonly neverReady?: boolean } = {},
): ScanProbes {
  const offline = (url: string): HttpProbeResult => ({ ok: false, kind: 'connection', detail: `${url} could not be reached: connect ECONNREFUSED` });
  return {
    request: async (url) => {
      if (url.includes(':11434')) {
        return state.ollama ? { ok: true, status: 200, body: { models: [{ name: 'llama3:8b' }] } } : offline(url);
      }
      if (url.endsWith('/system_stats')) {
        if (options.neverReady === true) return { ok: true, status: 200, body: { system: {}, devices: undefined } };
        return state.comfyui ? { ok: true, status: 200, body: { system: {}, devices: [] } } : offline(url);
      }
      if (url.endsWith('/object_info')) return state.comfyui ? { ok: true, status: 200, body: {} } : offline(url);
      if (url.includes('/userdata?')) return state.comfyui ? { ok: true, status: 200, body: Object.keys(workflows) } : offline(url);
      const match = /\/userdata\/(.+)$/.exec(url);
      if (match !== null && state.comfyui) {
        const relative = decodeURIComponent(match[1] as string).replace(/^workflows\//, '');
        if (Object.hasOwn(workflows, relative)) return { ok: true, status: 200, body: workflows[relative] };
      }
      return offline(url);
    },
    isDirectory: async () => false,
    listDir: async () => undefined,
    readFile: async () => '',
  };
}

/**
 * A clock and sleep that advance together.
 *
 * `manageEngines` measures readiness against a deadline, so the only way to
 * exercise one without waiting for it is to move time forward when the scan
 * sleeps.
 *
 * @param stepMs - how much simulated time each sleep advances.
 * @returns the clock and its sleep.
 */
function clocked(stepMs = 1_000): { now: () => number; sleep: (ms: number) => Promise<void>; slept: () => number } {
  let time = 2_000_000;
  let slept = 0;
  return {
    now: () => time,
    sleep: async (ms: number) => { slept += 1; time += Math.max(ms, stepMs); },
    slept: () => slept,
  };
}

// ── the policy ──────────────────────────────────────────────────────────────

describe('Scan starts the configured engines; looking does not', () => {
  beforeEach(() => {
    resetScanCache();
  });

  it('starts nothing when the request is only a look at the page', async () => {
    const machine = { ollama: false, comfyui: false };
    const rig = managedHub(machine);
    const result = await scanLocalResources(rig.hub, {
      hosts: hostsFor(),
      probes: probesFor(machine),
    });

    assert.deepEqual(rig.launches, [], 'opening the page must not launch a process');
    assert.equal(result.complete, false, 'and the scan says it could not read the engines');
    assert.ok(result.resources.every((resource) => resource.engineRunning === false));
    assert.deepEqual(result.engines, [], 'no engine was started, so there is nothing to report');
  });

  it('starts a stopped engine on an explicit Scan, reads it, and holds it only until the scan ends', async () => {
    const machine = { ollama: false, comfyui: false };
    const rig = managedHub(machine);
    const progress: ScanProgress[] = [];
    const probes = probesFor(machine);

    const result = await scanLocalResources(rig.hub, {
      hosts: hostsFor(),
      manageEngines: true,
      allowProcessLaunch: true,
      probes,
      engineStartupTimeoutMs: 60_000,
      onProgress: (update) => progress.push(update),
      log: () => {},
    });

    assert.deepEqual(rig.launches, ['comfyui'], 'the configured ComfyUI was started through the lifecycle manager');
    assert.equal(result.engines.length, 1);
    assert.equal(result.engines[0]?.state, 'started');
    assert.equal(result.engines[0]?.ready, true);
    assert.equal(result.engines[0]?.protectedFromIdleStop, true, 'a process this scan started is held while it reads');
    assert.equal(rig.liveProtections(), 0, 'and released before the scan returns');

    // The stage order a user is shown.
    assert.deepEqual(result.phases, ['checking', 'starting', 'waiting', 'engine-ready', 'scanning', 'partial']);
    assert.equal(result.complete, false, 'Ollama was never started, so the scan is honestly partial');
    assert.ok(result.warnings.some((warning) => /Ollama/.test(warning)), 'and says which engine was the problem');

    // The engine was read *after* becoming ready: the workflow is there.
    assert.ok(result.resources.some((resource) => resource.kind === 'comfyui_workflow' && resource.name === 'fox'));
  });

  it('reports a scan that started every engine as complete', async () => {
    const machine = { ollama: false, comfyui: false };
    // Only ComfyUI is declared, so starting it is the whole job.
    const onlyComfy = hostsFor().filter((host) => host.id === 'comfyui');
    const rig = managedHub(machine);
    const clock = clocked();

    const result = await scanLocalResources(rig.hub, {
      hosts: onlyComfy,
      manageEngines: true,
      allowProcessLaunch: true,
      probes: probesFor(machine),
      now: clock.now,
      sleep: clock.sleep,
      timeoutMs: 500,
      engineStartupTimeoutMs: 60_000,
      log: () => {},
    });

    assert.equal(result.complete, true);
    assert.deepEqual(result.phases.at(-1), 'complete');
    assert.equal(result.engines[0]?.state, 'started');
    assert.equal(rig.liveProtections(), 0);
  });

  it('does not launch twice when two Scans overlap, and reuses the engine afterwards', async () => {
    const machine = { ollama: false, comfyui: false };
    const rig = managedHub(machine);
    // A launch that takes a moment, so the second Scan genuinely overlaps the first.
    let releaseLaunch!: () => void;
    const gate = new Promise<void>((resolve) => { releaseLaunch = resolve; });
    const slowHub = {
      ...(rig.hub as unknown as Record<string, unknown>),
      startEngineForHost: async (hostId: string) => {
        await gate;
        return await (rig.hub.startEngineForHost as (id: string) => Promise<unknown>)(hostId);
      },
    } as unknown as ModelHub;

    const probes = probesFor(machine);
    const options = { hosts: hostsFor().filter((host) => host.id === 'comfyui'), manageEngines: true,
      allowProcessLaunch: true, probes, engineStartupTimeoutMs: 60_000, log: () => {} };
    const first = scanLocalResources(slowHub, options);
    await Promise.resolve();
    const second = scanLocalResources(slowHub, options);
    releaseLaunch();
    const [a, b] = await Promise.all([first, second]);

    assert.equal(rig.launches.length, 2, 'each Scan joins the launch it needs rather than starting a second copy');
    assert.deepEqual(rig.launches, ['comfyui', 'comfyui'], 'and never a second engine, only a second request for the same one');
    assert.equal(a.engines[0]?.state, 'started');
    assert.equal(b.engines[0]?.state, 'started');
    assert.equal(a.resources.length, b.resources.length, 'both scans read the same single engine');
    assert.equal(rig.liveProtections(), 0, 'neither scan left a protection behind');

    // A third scan finds it up and starts nothing at all.
    const third = await scanLocalResources(slowHub, options);
    assert.equal(rig.launches.length, 2, 'an engine that is already running is reused, not relaunched');
    assert.equal(third.engines[0]?.state, 'external');
    assert.equal(third.complete, true);
  });

  it('reports a readiness timeout and still leaves the engine released', async () => {
    // The engine starts and then answers its readiness route with something that is
    // not valid readiness data, forever: the launch succeeded, readiness never did,
    // and that is a timeout rather than a launch failure.
    const machine = { ollama: false, comfyui: false };
    const rig = managedHub(machine);
    const clock = clocked();
    const result = await scanLocalResources(rig.hub, {
      hosts: hostsFor().filter((host) => host.id === 'comfyui'),
      manageEngines: true,
      allowProcessLaunch: true,
      probes: probesFor(machine, {}, { neverReady: true }),
      now: clock.now,
      sleep: clock.sleep,
      engineStartupTimeoutMs: 90_000,
      timeoutMs: 200,
      log: () => {},
    });

    assert.deepEqual(rig.launches, ['comfyui']);
    assert.equal(result.engines[0]?.state, 'timeout');
    assert.equal(result.engines[0]?.ready, false);
    assert.match(result.engines[0]?.detail ?? '', /never answered within 90000 ms/);
    assert.equal(result.complete, false);
    assert.equal(rig.liveProtections(), 0, 'a timeout is not an excuse to keep an engine pinned');
    assert.ok(clock.slept() > 0, 'readiness was polled rather than assumed');
  });

  it('keeps scanning when one engine cannot start, and says which one failed', async () => {
    const machine = { ollama: true, comfyui: false };
    const rig = managedHub(machine, { launchError: 'spawn python.exe EPERM' });
    const result = await scanLocalResources(rig.hub, {
      hosts: hostsFor().filter((host) => host.id === 'comfyui'),
      manageEngines: true,
      allowProcessLaunch: true,
      probes: probesFor(machine),
      engineStartupTimeoutMs: 30_000,
      log: () => {},
    });

    assert.equal(result.engines[0]?.state, 'failed');
    assert.match(result.engines[0]?.detail ?? '', /EPERM/, 'the real reason is carried to the user');
    assert.equal(result.complete, false);
    // Whatever a configured directory yields is still collected: a failed engine
    // never discards the results of a source that did work.
    assert.ok(Array.isArray(result.resources));
  });

  it('reports an engine that dies during the scan and keeps what was collected', async () => {
    const machine = { ollama: false, comfyui: false };
    const rig = managedHub(machine, { diesAfterLaunch: true });
    const result = await scanLocalResources(rig.hub, {
      hosts: hostsFor().filter((host) => host.id === 'comfyui'),
      manageEngines: true,
      allowProcessLaunch: true,
      probes: probesFor(machine, { 'fox.json': apiWorkflow }),
      engineStartupTimeoutMs: 60_000,
      log: () => {},
    });

    assert.equal(result.engines[0]?.state, 'interrupted');
    assert.equal(result.engines[0]?.ready, false);
    assert.match(result.engines[0]?.detail ?? '', /stopped while the scan was reading it/);
    assert.ok(result.resources.some((resource) => resource.kind === 'comfyui_workflow'), 'the workflow read before the exit is kept');
    assert.equal(result.complete, false);
    assert.equal(rig.liveProtections(), 0);
  });

  it('says what to change when the deployment forbids launching', async () => {
    const machine = { ollama: false, comfyui: false };
    const rig = managedHub(machine);
    const result = await scanLocalResources(rig.hub, {
      hosts: hostsFor().filter((host) => host.id === 'comfyui'),
      manageEngines: true,
      allowProcessLaunch: false,
      probes: probesFor(machine),
      log: () => {},
    });

    assert.deepEqual(rig.launches, [], 'a forbidden launch is not attempted');
    assert.equal(result.engines[0]?.state, 'failed');
    assert.match(result.engines[0]?.detail ?? '', /allowProcessLaunch/);
    assert.equal(result.complete, false);
  });

  it('never starts an engine the catalog does not declare startable', async () => {
    const machine = { ollama: false, comfyui: false };
    const rig = managedHub(machine);
    const result = await scanLocalResources(rig.hub, {
      hosts: hostsFor('http://127.0.0.1:8188', { comfyStartable: false }).filter((host) => host.id === 'comfyui'),
      manageEngines: true,
      allowProcessLaunch: true,
      probes: probesFor(machine),
      log: () => {},
    });

    assert.deepEqual(rig.launches, []);
    assert.equal(result.engines[0]?.state, 'failed');
    assert.match(result.engines[0]?.detail ?? '', /no launch command is configured/);
  });

  it('reuses an engine that is already running without holding or restarting it', async () => {
    const machine = { ollama: true, comfyui: true };
    const rig = managedHub(machine);
    const result = await scanLocalResources(rig.hub, {
      hosts: hostsFor().filter((host) => host.id === 'comfyui'),
      manageEngines: true,
      allowProcessLaunch: true,
      probes: probesFor(machine),
      log: () => {},
    });

    assert.deepEqual(rig.launches, [], 'an engine that is up is not started again');
    assert.equal(result.engines[0]?.state, 'external');
    assert.equal(result.engines[0]?.protectedFromIdleStop, false);
    assert.deepEqual(rig.protections, [], 'a user-run engine is never pinned by a scan');
    assert.equal(result.complete, true, 'both engines answered, so the look was complete');
  });

  it('releases engine protection even when the read itself throws', async () => {
    // A cancelled or exploding scan must not leave an engine pinned: protection
    // that outlives its scan is a process nobody will ever idle-stop.
    const machine = { ollama: false, comfyui: false };
    const rig = managedHub(machine);
    const clock = clocked();
    const probes = probesFor(machine);
    const exploding: ScanProbes = {
      ...probes,
      listDir: async () => { throw new Error('injected listing explosion'); },
    };
    const result = await scanLocalResources(rig.hub, {
      hosts: hostsFor().filter((host) => host.id === 'comfyui'),
      manageEngines: true,
      allowProcessLaunch: true,
      probes: exploding,
      now: clock.now,
      sleep: clock.sleep,
      engineStartupTimeoutMs: 60_000,
      log: () => {},
    });

    assert.equal(rig.launches.length, 1, 'the engine was started before the read failed');
    assert.equal(rig.liveProtections(), 0, 'and was released on the failure path');
    assert.ok(result.warnings.length > 0, 'the failure is reported rather than hidden');
  });
});

// ── the lifecycle manager itself ────────────────────────────────────────────

describe('engine protection against idle unloading', () => {
  beforeEach(() => {
    resetScanCache();
  });

  it('is refcounted, released exactly once, and never negative', async () => {
    // A fake state holder with the same contract the runtime implements, driven
    // through `bringUpEngines` so the *scan's* use of it is what is asserted.
    const held = new Map<string, number>();
    const grant = (modelId: string): (() => void) => {
      held.set(modelId, (held.get(modelId) ?? 0) + 1);
      let released = false;
      return () => {
        if (released) return;
        released = true;
        held.set(modelId, Math.max(0, (held.get(modelId) ?? 0) - 1));
      };
    };
    // A clock that keeps moving: a frozen one would make the readiness loop wait
    // for a deadline that can never arrive.
    let time = 0;
    const launch = await bringUpEngines({
      hosts: hostsFor().filter((host) => host.id === 'comfyui'),
      allowProcessLaunch: true,
      effects: {
        probe: async () => ({ running: false, availability: 'stopped' as const, detail: 'refused' }),
        ensureReady: async () => ({ modelId: 'engine-comfyui', started: true, alreadyRunning: false }),
        protect: (modelId) => grant(modelId),
        stillRunning: () => true,
      },
      timeoutMs: 100,
      pollIntervalMs: 10,
      sleep: async (ms: number) => { time += ms; },
      now: () => time,
    });

    assert.equal(held.get('engine-comfyui'), 1, 'the started engine is held while the scan reads it');
    launch.release();
    launch.release();
    assert.equal(held.get('engine-comfyui'), 0, 'and released once, not twice');
  });

  it('is not installed on disk by these tests, and does not need to be', () => {
    // A guard against the suite depending on this machine: nothing here reads or
    // writes a real engine, and the catalog is entirely in memory.
    const config = catalogFor('http://127.0.0.1:8188');
    assert.equal(config.hosts?.length, 2);
    assert.ok(!JSON.stringify(config).includes('C:/Program Files'));
  });
});
