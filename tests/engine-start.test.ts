/**
 * Tests for starting the engines a Scan is about to read.
 *
 * These drive `bringUpEngines` directly, with injected effects, because that is
 * where every decision this behaviour makes actually lives: whether to launch,
 * what to do when a launch is impossible, how long to wait, what to say when
 * readiness never arrives, and — the one that is easy to get wrong and expensive
 * to get wrong — exactly when engine protection is taken and given back.
 *
 * No process is spawned and no clock is waited on: `sleep` is injected, so a test
 * that exercises a 300-second readiness budget spends no time on it.
 *
 * @module dsh-ai-model-hub/tests/engine-start
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import {
  bringUpEngines,
  launchBlocker,
  markInterruptedEngines,
  startupBudgetFor,
} from '../dsh-plugin/engine-start.ts';
import type { EngineStartupEffects, ScanProgress } from '../dsh-plugin/engine-start.ts';
import type { ModelHost } from '../src/index.ts';

/** A host the scan manages: ComfyUI is the engine this behaviour exists for. */
function comfyHost(overrides: Record<string, unknown> = {}): ModelHost {
  return {
    id: 'comfyui',
    name: 'ComfyUI',
    adapter: 'comfyui',
    runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint: 'http://127.0.0.1:8188', path: '/prompt' },
    lifecycle: {
      startable: true,
      stoppable: true,
      start: { command: 'python.exe', args: ['main.py'], cwd: 'C:/ComfyUI/src' },
    },
    ...overrides,
  } as ModelHost;
}

/** An engine that is not a managed one, to prove it is left alone. */
function ollamaHost(): ModelHost {
  return {
    id: 'ollama',
    name: 'Ollama',
    adapter: 'openai_compatible',
    runtime: { engine: 'ollama', adapter: 'openai_compatible', endpoint: 'http://127.0.0.1:11434', path: '/v1/chat/completions' },
  } as ModelHost;
}

/**
 * A described machine: what the engine answers, what starting it does, and what
 * the bookkeeping records.
 */
function machine(options: {
  /** Whether the engine is running when the scan begins. */
  readonly initiallyRunning?: boolean;
  /** How many readiness probes answer "not yet" after a launch before it answers. */
  readonly readyAfterProbes?: number;
  /** Make `ensureReady` reject with this message instead of starting anything. */
  readonly launchError?: string;
  /** Make the engine answer but never become ready. */
  readonly neverReady?: boolean;
  /** Whether the tracked process is still alive when the scan finishes. */
  readonly diesMidScan?: boolean;
} = {}): {
  effects: EngineStartupEffects;
  probes: number;
  starts: number;
  protections: string[];
  releases: number;
  sleeps: number;
} {
  const state = {
    running: options.initiallyRunning === true,
    remaining: options.readyAfterProbes ?? 0,
    probes: 0,
    starts: 0,
    protections: [] as string[],
    releases: 0,
    sleeps: 0,
  };
  let protectedCount = 0;
  const effects: EngineStartupEffects = {
    probe: async () => {
      state.probes += 1;
      if (!state.running) return { running: false, availability: 'stopped' as const, detail: 'connect ECONNREFUSED' };
      if (options.neverReady === true) return { running: false, availability: 'error' as const, detail: 'answered 503' };
      if (state.remaining > 0) { state.remaining -= 1; return { running: false, availability: 'unreachable' as const, detail: 'still loading' }; }
      return { running: true, availability: 'answering' as const, detail: 'system_stats answered 200' };
    },
    ensureReady: async () => {
      if (options.launchError !== undefined) throw new Error(options.launchError);
      state.starts += 1;
      state.running = true;
      state.remaining = options.readyAfterProbes ?? 0;
      return { modelId: 'engine-comfyui', started: true, alreadyRunning: false };
    },
    protect: (modelId) => {
      state.protections.push(modelId);
      protectedCount += 1;
      return () => { protectedCount -= 1; state.releases += 1; };
    },
    stillRunning: () => options.diesMidScan !== true,
  };
  return {
    effects,
    get probes() { return state.probes; },
    get starts() { return state.starts; },
    get protections() { return state.protections; },
    get releases() { return state.releases; },
    get sleeps() { return state.sleeps; },
  } as never;
}

/**
 * A clock and a sleep that move together, so a deadline is exercised without
 * being waited on.
 *
 * @param stepMs - how much simulated time each sleep advances.
 * @returns the clock, the sleep, and a counter of how many sleeps happened.
 */
function fakeClock(stepMs: number): { now: () => number; sleep: (ms: number) => Promise<void>; slept: () => number } {
  let time = 1_000_000;
  let slept = 0;
  return {
    now: () => time,
    sleep: async (ms: number) => { slept += 1; time += Math.max(ms, stepMs); },
    slept: () => slept,
  };
}

describe('bringing engines up for a scan', () => {
  it('reuses an engine that is already running, and never holds or starts it', async () => {
    const clock = fakeClock(25);
    const rig = machine({ initiallyRunning: true });
    const result = await bringUpEngines({
      hosts: [comfyHost()],
      allowProcessLaunch: true,
      effects: rig.effects,
      sleep: clock.sleep,
      now: clock.now,
    });

    assert.equal(result.reports.length, 1);
    assert.equal(result.reports[0]?.state, 'external');
    assert.equal(result.reports[0]?.started, false, 'the scan did not start it');
    assert.equal(result.reports[0]?.ready, true);
    assert.equal(result.reports[0]?.protectedFromIdleStop, false, 'an engine the user runs is not ours to hold');
    assert.deepEqual(rig.protections, [], 'nothing was protected, because nothing was launched');
    assert.deepEqual(result.readyHostIds, ['comfyui']);
    result.release();
    assert.equal(rig.releases, 0, 'there was nothing to release');
  });

  it('starts a stopped engine, waits for its API, and holds it until released', async () => {
    const clock = fakeClock(25);
    const rig = machine({ initiallyRunning: false, readyAfterProbes: 2 });
    const progress: ScanProgress[] = [];
    const result = await bringUpEngines({
      hosts: [comfyHost()],
      allowProcessLaunch: true,
      effects: rig.effects,
      onProgress: (update) => progress.push(update),
      pollIntervalMs: 25,
      sleep: clock.sleep,
      now: clock.now,
    });

    assert.equal(rig.starts, 1, 'the lifecycle manager was asked exactly once');
    assert.equal(result.reports[0]?.state, 'started');
    assert.equal(result.reports[0]?.started, true);
    assert.equal(result.reports[0]?.ready, true);
    assert.equal(result.reports[0]?.protectedFromIdleStop, true);
    assert.deepEqual(rig.protections, ['engine-comfyui'], 'the process it started is the one it holds');
    assert.ok(clock.slept() >= 1, 'readiness was polled rather than assumed');

    // The stages a user is shown, in order.
    const phases = progress.map((update) => update.phase);
    assert.deepEqual(phases, ['checking', 'starting', 'waiting', 'engine-ready']);

    result.release();
    assert.equal(rig.releases, 1);
    result.release();
    assert.equal(rig.releases, 1, 'releasing twice releases once');
  });

  it('starts nothing for an engine the catalog gives no launch command', async () => {
    const clock = fakeClock(25);
    const rig = machine();
    const host = comfyHost({ lifecycle: { startable: false } });
    const result = await bringUpEngines({
      hosts: [host],
      allowProcessLaunch: true,
      effects: rig.effects,
      sleep: clock.sleep,
      now: clock.now,
    });

    assert.equal(rig.starts, 0);
    assert.equal(result.reports[0]?.state, 'failed');
    assert.equal(result.reports[0]?.ready, false);
    assert.match(result.reports[0]?.detail ?? '', /no launch command is configured/);
    assert.match(result.reports[0]?.detail ?? '', /lifecycle/, 'the message names the fix');
    assert.deepEqual(result.readyHostIds, []);
    result.release();
  });

  it('starts nothing when the deployment forbids launching, and says which switch', async () => {
    const clock = fakeClock(25);
    const rig = machine();
    const result = await bringUpEngines({
      hosts: [comfyHost()],
      allowProcessLaunch: false,
      effects: rig.effects,
      sleep: clock.sleep,
      now: clock.now,
    });

    assert.equal(rig.starts, 0, 'a policy refusal is not an attempt');
    assert.equal(result.reports[0]?.state, 'failed');
    assert.match(result.reports[0]?.detail ?? '', /not allowed to launch processes/);
    assert.match(result.reports[0]?.detail ?? '', /allowProcessLaunch/, 'the message names the switch to flip');
    result.release();
  });

  it('reports a launch that fails, without holding anything', async () => {
    const clock = fakeClock(25);
    const rig = machine({ launchError: 'spawn python.exe ENOENT' });
    const result = await bringUpEngines({
      hosts: [comfyHost()],
      allowProcessLaunch: true,
      effects: rig.effects,
      sleep: clock.sleep,
      now: clock.now,
    });

    assert.equal(result.reports[0]?.state, 'failed');
    assert.match(result.reports[0]?.detail ?? '', /could not be started/);
    assert.match(result.reports[0]?.detail ?? '', /ENOENT/, 'the real reason is carried through');
    assert.deepEqual(rig.protections, [], 'a failed launch protects nothing');
    result.release();
  });

  it('times out when a started engine answers but never becomes ready', async () => {
    const clock = fakeClock(25);
    const rig = machine({ initiallyRunning: false, neverReady: true });
    const options = {
      hosts: [comfyHost()],
      allowProcessLaunch: true,
      effects: rig.effects,
      timeoutMs: 1_500,
      pollIntervalMs: 10,
      sleep: clock.sleep,
      now: clock.now,
    };
    const result = await bringUpEngines(options);

    assert.equal(rig.starts, 1, 'it was started');
    assert.equal(result.reports[0]?.state, 'timeout');
    assert.equal(result.reports[0]?.ready, false);
    assert.match(result.reports[0]?.detail ?? '', /never answered within 1500 ms/);
    assert.deepEqual(result.readyHostIds, [], 'an engine that is not ready is not readable');
    result.release();
    assert.equal(rig.releases, 1, 'the holder of an unready process still gives it back');
  });

  it('honours the host\'s declared startup budget over the defaults', () => {
    const host = comfyHost({ lifecycle: { startable: true, startupTimeoutMs: 42_000, start: { command: 'x' } } });
    assert.equal(startupBudgetFor(host), 42_000, 'the catalog describes this machine');
    assert.equal(startupBudgetFor(host, 900), 900, 'an explicit budget wins over the catalog');
    assert.equal(startupBudgetFor(comfyHost({ lifecycle: undefined })), 300_000, 'ComfyUI cold-starts are slow');
    assert.equal(startupBudgetFor(ollamaHost()), 120_000, 'and an engine with no budget gets the general one');
    assert.equal(startupBudgetFor(comfyHost({ lifecycle: undefined }), 0), 300_000, 'zero is "no override", not "no budget"');
  });

  it('describes each way a launch is impossible, separately', () => {
    assert.equal(launchBlocker(comfyHost(), true), undefined, 'a startable host with a command can be launched');
    assert.match(launchBlocker(comfyHost({ lifecycle: { startable: false } }), true) ?? '', /no launch command/);
    assert.match(launchBlocker(comfyHost({ lifecycle: { startable: true } }), true) ?? '', /no launch command/);
    assert.match(launchBlocker(comfyHost(), false) ?? '', /allowProcessLaunch/);
  });

  it('continues to the other engine when one cannot be started', async () => {
    const clock = fakeClock(25);
    const refused = machine({ launchError: 'spawn python.exe EPERM' });
    const fine = machine({ initiallyRunning: true });
    const reports: string[] = [];
    // One effects object per host, keyed by id, so each describes its own engine.
    const effects: EngineStartupEffects = {
      probe: async (host) => (host.id === 'comfyui' ? await refused.effects.probe(host) : await fine.effects.probe(host)),
      ensureReady: async (host) => {
        reports.push(host.id);
        return host.id === 'comfyui' ? await refused.effects.ensureReady(host) : await fine.effects.ensureReady(host);
      },
      protect: (modelId) => fine.effects.protect(modelId),
      stillRunning: () => true,
    };
    const result = await bringUpEngines({
      hosts: [comfyHost(), { ...ollamaHost(), id: 'ollama', adapter: 'comfyui' } as ModelHost],
      allowProcessLaunch: true,
      effects,
      sleep: clock.sleep,
      now: clock.now,
    });

    assert.equal(result.reports.length, 2, 'both engines are reported on');
    assert.equal(result.reports[0]?.state, 'failed');
    assert.equal(result.reports[1]?.state, 'external', 'the other engine was still evaluated');
    assert.deepEqual(result.readyHostIds, ['ollama'], 'only the readable engine is offered to the scan');
    result.release();
  });

  it('releases every protection when the caller never gets a report', async () => {
    // An unexpected throw inside the loop must not leave an engine pinned: a
    // process nobody can idle-stop is a leak that outlives the scan.
    const clock = fakeClock(25);
    const rig = machine({ initiallyRunning: false });
    let throws = 0;
    const effects: EngineStartupEffects = {
      ...rig.effects,
      probe: async (host) => {
        throws += 1;
        if (throws > 1) throw new Error('injected probe explosion');
        return await rig.effects.probe(host);
      },
    };
    await assert.rejects(
      async () => await bringUpEngines({
        hosts: [comfyHost(), { ...comfyHost(), id: 'second', name: 'Second' } as ModelHost],
        allowProcessLaunch: true,
        effects,
        sleep: clock.sleep,
      now: clock.now,
      }),
      /injected probe explosion/,
    );
    assert.equal(rig.releases, 1, 'the first engine was protected, then released by the failure path');
  });

  it('reports an engine that died mid-scan, without discarding what was collected', async () => {
    const clock = fakeClock(25);
    const rig = machine({ initiallyRunning: false, diesMidScan: true });
    const result = await bringUpEngines({
      hosts: [comfyHost()],
      allowProcessLaunch: true,
      effects: rig.effects,
      sleep: clock.sleep,
      now: clock.now,
    });
    assert.equal(result.reports[0]?.ready, true, 'it was ready when the read began');

    const reports = [...result.reports];
    const interrupted = markInterruptedEngines(reports, () => false, result.modelIds);
    assert.deepEqual(interrupted, ['comfyui']);
    assert.equal(reports[0]?.state, 'interrupted');
    assert.equal(reports[0]?.ready, false);
    assert.match(reports[0]?.detail ?? '', /stopped while the scan was reading it/);
    assert.match(reports[0]?.detail ?? '', /kept/, 'and says the collected results survive');

    // An engine that is still alive is not marked.
    const alive = [...result.reports];
    assert.deepEqual(markInterruptedEngines(alive, () => true, result.modelIds), []);
    assert.equal(alive[0]?.state, 'started');
    result.release();
  });
});
