/**
 * Starting the local engines a scan is about to read.
 *
 * ## Why this exists
 *
 * Reading an engine's store and having the engine *running* are different
 * problems, and a scan wants both: Ollama lists its models over `/api/tags`, and
 * ComfyUI serves its saved workflows over `/userdata`, so an engine that is
 * installed but stopped yields nothing to list. The honest answer for a user who
 * pressed **Scan** is therefore to start the configured instance, wait until its
 * API actually answers, read it, and then hand the engine back to whatever policy
 * was in force — not to report "not answering" about an engine this machine owns
 * and could have started.
 *
 * ## What it deliberately reuses
 *
 * - **The lifecycle manager decides everything about launching.** Which command,
 *   which arguments, which working directory, whether an already-live endpoint
 *   should be adopted instead of spawning a second instance, and how the
 *   readiness budget is spent are all the runtime's answers, not this module's.
 *   All this module adds is the decision *to* start, an engine-level deadline, and
 *   what to say when it does not work.
 * - **`ensureReady` is the gate.** It is the same path a cold invocation takes, so
 *   a scan and a generation racing for the same engine join one launch instead of
 *   starting two. That is also why repeated Scan clicks cannot double-start: the
 *   runtime serializes transitions per model.
 * - **Protection is the runtime's.** The scan asks for it and releases it in a
 *   `finally`; the sweeper is the only thing it suppresses.
 *
 * ## What it does not do
 *
 * It never installs anything, never guesses a launch command, and never starts an
 * engine the catalog has not declared startable. If the machine does not have the
 * engine or the deployment has not been given a way to launch it, the answer is an
 * actionable error naming the missing piece — `allowProcessLaunch`, a `lifecycle`
 * block, a `start` command — because each of those has a different fix.
 *
 * @module dsh-ai-model-hub/dsh-plugin/engine-start
 */

import type { ModelHost } from '../src/index.ts';

/** Why an engine is not answering, at the level a user can act on. */
export type EngineAvailability = 'answering' | 'stopped' | 'unreachable' | 'unconfigured' | 'not-comfyui' | 'error';

/**
 * What happened to one engine this scan manages, from the moment it was looked at.
 *
 * The states are ordered stages of one engine's life during a scan, not categories
 * of error: `ready` and `started` both mean "readable", `external` means "already
 * up and not ours to touch", and everything else means "we could not read it, and
 * here is why".
 */
export type EngineStartState =
  /** Already answering when the scan began; nothing was launched and nothing is held. */
  | 'external'
  /** Was stopped, the hub started it, and its API answered. */
  | 'started'
  /** The hub was asked to start it and its API did not answer inside the deadline. */
  | 'timeout'
  /** The launch failed: no command, a refused spawn, or a process that died. */
  | 'failed'
  /** Existed and answered, started by the hub, and then stopped before the read finished. */
  | 'interrupted';

/**
 * The user-visible stage of a scan, in order.
 *
 * `engine-ready` and `engine-failed` are per-engine ticks: one engine finishing
 * says nothing about the scan as a whole, so they are deliberately distinct from
 * `complete` and `partial`, which only the scan itself may emit.
 */
export type ScanPhase =
  | 'checking'
  | 'starting'
  | 'waiting'
  | 'engine-ready'
  | 'engine-failed'
  | 'scanning'
  | 'complete'
  | 'partial';

/** One progress tick, so a UI can explain a long scan rather than spin. */
export interface ScanProgress {
  readonly phase: ScanPhase;
  /** What is happening, in one sentence. */
  readonly message: string;
  /** The engine this tick is about, when it is about one. */
  readonly hostId?: string;
}

/** What one engine contributed to the scan's ability to read it. */
export interface EngineStartupReport {
  readonly hostId: string;
  readonly hostName: string;
  readonly state: EngineStartState;
  /** Whether this scan is the reason the engine is running. */
  readonly started: boolean;
  /** Whether the engine answered its readiness route by the end. */
  readonly ready: boolean;
  /** How long the bring-up took, in milliseconds. */
  readonly durationMs: number;
  /** A sentence naming what happened, or what to change. */
  readonly detail: string;
  /** True until this scan's `finally` releases it. */
  readonly protectedFromIdleStop: boolean;
}

/**
 * The effects bring-up needs, isolated so a test can describe an engine without
 * owning one.
 *
 * `ensureReady` is deliberately the *runtime's* shape rather than a generic "start
 * it" hook: the whole point is that a scan takes the same gate a cold invocation
 * takes, so that two callers cannot spawn two processes.
 */
export interface EngineStartupEffects {
  /** Check readiness without starting anything. */
  readonly probe: (host: ModelHost) => Promise<{ readonly running: boolean; readonly availability: EngineAvailability; readonly detail: string }>;
  /** Make an engine ready for use, starting it when it is stopped. */
  readonly ensureReady: (host: ModelHost) => Promise<{ readonly modelId: string; readonly started: boolean; readonly alreadyRunning: boolean; readonly detail?: string }>;
  /** Suppress automatic idle shutdown of a process for the duration of a scan. */
  readonly protect: (modelId: string) => () => void;
  /** Whether a process that was alive at the start of the scan is still alive. */
  readonly stillRunning: (modelId: string) => boolean;
  /** Diagnostic sink. */
  readonly log?: (message: string) => void;
}

/** Options for {@link bringUpEngines}. */
export interface EngineStartupOptions {
  /** The hosts to consider. Hosts that are not engines this module knows are ignored. */
  readonly hosts: readonly ModelHost[];
  /** Whether the deployment permits the hub to launch processes at all. */
  readonly allowProcessLaunch: boolean;
  /** How long one engine may take to answer after a launch, in milliseconds. */
  readonly timeoutMs?: number;
  /** Injectable effects; omitted members fall back to real ones supplied by the caller. */
  readonly effects: EngineStartupEffects;
  readonly onProgress?: (progress: ScanProgress) => void;
  /** How often to re-check readiness while waiting. Defaults to 500 ms. */
  readonly pollIntervalMs?: number;
  /** Injectable sleep, so a test does not spend the deadline it is testing. */
  readonly sleep?: (ms: number) => Promise<void>;
  /**
   * Injectable clock, in milliseconds.
   *
   * Readiness is a *deadline*, and a deadline needs a clock: an injected sleep
   * that returns immediately would otherwise leave the poller looping until real
   * time caught up. Tests supply a clock that advances with the sleep.
   */
  readonly now?: () => number;
}

/** The default per-engine readiness budget. */
const DEFAULT_STARTUP_TIMEOUT_MS = 120_000;

/** ComfyUI loads checkpoints and custom nodes before it answers, so it gets longer. */
const COMFYUI_STARTUP_TIMEOUT_MS = 300_000;

/** How often to re-check readiness while an engine comes up. */
const DEFAULT_POLL_INTERVAL_MS = 500;

/** Whether a host is an engine this module brings up for a scan. */
function isManagedEngine(host: ModelHost): boolean {
  return host.adapter === 'comfyui';
}

/**
 * The readiness budget for one engine, when the deployment does not set one.
 *
 * A catalog that declares its own startup budget is describing this machine, so it
 * wins; the defaults exist for a catalog that declares none.
 *
 * @param host - the engine's host.
 * @param override - the deployment's own budget, when it set one.
 * @returns the budget in milliseconds.
 */
export function startupBudgetFor(host: ModelHost, override?: number): number {
  if (override !== undefined && Number.isFinite(override) && override > 0) return override;
  const declared = host.lifecycle?.startupTimeoutMs;
  if (typeof declared === 'number' && Number.isFinite(declared) && declared > 0) return declared;
  return /comfyui/i.test(host.runtime.engine) ? COMFYUI_STARTUP_TIMEOUT_MS : DEFAULT_STARTUP_TIMEOUT_MS;
}

/**
 * Whether the catalog has given the hub a way to launch this engine.
 *
 * Three different configurations produce "I cannot start this", and they have
 * three different fixes, so they are reported separately rather than collapsed
 * into "not startable":
 *
 * - the deployment forbids launching at all (`allowProcessLaunch: false`);
 * - the host declares no `lifecycle` or marks itself not startable;
 * - the host is startable but has no `start` command to run.
 *
 * @param host - the engine's host.
 * @param allowProcessLaunch - whether the deployment permits launching.
 * @returns the reason it cannot be launched, or `undefined` when it can.
 */
export function launchBlocker(host: ModelHost, allowProcessLaunch: boolean): string | undefined {
  const lifecycle = host.lifecycle;
  if (lifecycle?.start === undefined || lifecycle.startable !== true) {
    return `no launch command is configured for "${host.name}". Add a \`lifecycle\` block with \`startable: true\` and a ` +
      '`start` command to its host entry in the catalog, and the hub can start it for a scan.';
  }
  if (!allowProcessLaunch) {
    return `the hub is not allowed to launch processes here, so "${host.name}" was not started. Set ` +
      '`allowProcessLaunch: true` for this plugin in the profile configuration, then scan again.';
  }
  return undefined;
}

/**
 * Bring every managed engine up for a scan and hold it there.
 *
 * Each engine is handled independently: one that fails to launch, times out, or
 * has no launch command does not stop the others from being read, and the
 * per-engine reports say which is which. The returned `release` function must be
 * called in a `finally` — it gives back every protection this call took, so an
 * engine the scan started is subject to the normal idle policy again.
 *
 * @param options - the hosts, the deployment's launch policy, and the effects.
 * @returns the per-engine reports, the engines to read now, the model id each
 *   engine is tracked under, and the release hook.
 */
export async function bringUpEngines(options: EngineStartupOptions): Promise<{
  readonly reports: readonly EngineStartupReport[];
  readonly readyHostIds: readonly string[];
  readonly modelIds: ReadonlyMap<string, string>;
  readonly release: () => void;
}> {
  const { effects, onProgress } = options;
  const managed = options.hosts.filter(isManagedEngine);
  const reports: EngineStartupReport[] = [];
  const releases: (() => void)[] = [];
  const modelIds = new Map<string, string>();
  const sleep = options.sleep ?? (async (ms: number) => { await new Promise((resolve) => setTimeout(resolve, ms)); });
  const now = options.now ?? ((): number => Date.now());
  const pollIntervalMs = options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;

  const release = (): void => {
    for (const give of releases.splice(0)) {
      try { give(); }
      catch (error) { effects.log?.('engine scan release failed: ' + String(error)); }
    }
  };

  try {
    for (const host of managed) {
      const startedAt = now();
      onProgress?.({ phase: 'checking', message: `Checking whether ${host.name} is running`, hostId: host.id });
      const running = await effects.probe(host);
      if (running.running) {
        // Already up: nothing was started, so nothing is protected either. An
        // engine the user is running is not this scan's to hold or to stop.
        reports.push(reportOf(host, 'external', { started: false, ready: true, durationMs: now() - startedAt,
          detail: `${host.name} was already answering (${running.detail}); this scan did not start or hold it.` }));
        onProgress?.({ phase: 'engine-ready', message: `${host.name} is already running`, hostId: host.id });
        continue;
      }

      const blocker = launchBlocker(host, options.allowProcessLaunch);
      if (blocker !== undefined) {
        // Not installed, or not configured to be launched. Either way the fix is
        // configuration, so the message names the exact piece that is missing.
        reports.push(reportOf(host, 'failed', { started: false, ready: false, durationMs: now() - startedAt,
          detail: `${host.name} is not answering (${running.detail}) and ${blocker}` }));
        onProgress?.({ phase: 'engine-failed', message: `${host.name} could not be started`, hostId: host.id });
        continue;
      }

      const budget = startupBudgetFor(host, options.timeoutMs);
      onProgress?.({ phase: 'starting', message: `Starting ${host.name}`, hostId: host.id });

      // Protection is taken *after* a launch succeeds, because a release must
      // never unprotect a process this scan did not create. See below.
      let ready: { started: boolean; alreadyRunning: boolean; detail?: string; modelId: string };
      try {
        ready = await withTimeout(effects.ensureReady(host), budget);
      } catch (error) {
        const timedOut = (error as { code?: string }).code === 'SCAN_STARTUP_TIMEOUT';
        const message = error instanceof Error ? error.message : String(error);
        reports.push(reportOf(host, timedOut ? 'timeout' : 'failed', { started: false, ready: false,
          durationMs: now() - startedAt,
          detail: timedOut
            ? `${host.name} did not become ready within ${budget} ms; its API never answered. ${message}`
            : `${host.name} could not be started: ${message}` }));
        onProgress?.({ phase: 'engine-failed', message: `${host.name} could not be started`, hostId: host.id });
        continue;
      }

      // The hub started it, or adopted a process that appeared while it was
      // starting. Either way the process exists now and is ours to hold until the
      // scan finishes reading it.
      modelIds.set(host.id, ready.modelId);
      try {
        releases.push(effects.protect(ready.modelId));
      } catch (error) {
        effects.log?.(`engine scan: could not protect ${ready.modelId}: ${String(error)}`);
      }

      onProgress?.({ phase: 'waiting', message: `Waiting for ${host.name} to become ready`, hostId: host.id });
      const confirmed = await waitForReady(host, effects, budget, pollIntervalMs, sleep, now(), now);
      if (!confirmed.running) {
        reports.push(reportOf(host, 'timeout', { started: ready.started, ready: false, durationMs: now() - startedAt,
          detail: `${host.name} was started but its readiness route never answered within ${budget} ms (${confirmed.detail})`,
          protectedFromIdleStop: true }));
        onProgress?.({ phase: 'engine-failed', message: `${host.name} started but is not ready`, hostId: host.id });
        continue;
      }

      reports.push(reportOf(host, 'started', { started: ready.started, ready: true, durationMs: now() - startedAt,
        detail: ready.started
          ? `started ${host.name} and confirmed readiness (${confirmed.detail})`
          : `adopted a running ${host.name} and confirmed readiness (${confirmed.detail})`,
        protectedFromIdleStop: true }));
      onProgress?.({ phase: 'engine-ready', message: `${host.name} is ready`, hostId: host.id });
    }
  } catch (error) {
    // An unexpected throw must not leak protection: release everything, then say
    // what happened. The caller's own `finally` will call `release` again, which
    // is a no-op.
    release();
    throw error;
  }

  // The engines this scan may read: the ones that answered, whether this scan
  // started them or found them already up. A failed engine is simply absent, and
  // the pass for its host then reports "not answering" with its own evidence.
  const readyHostIds = reports.filter((report) => report.ready).map((report) => report.hostId);
  return { reports, readyHostIds, modelIds, release };
}

/**
 * Note that an engine stopped between being started and being read.
 *
 * A process that dies mid-scan is not the same failure as one that never started:
 * the read may already have collected models or workflows, and those results are
 * kept. The report is amended in place, because the interruption is discovered
 * after the read, and it changes that engine's outcome rather than adding one.
 *
 * @param reports - the reports from {@link bringUpEngines}.
 * @param stillRunning - whether a tracked process is still alive.
 * @param modelIds - the model id each engine was tracked under.
 * @returns the host ids whose engine died mid-scan.
 */
export function markInterruptedEngines(
  reports: EngineStartupReport[],
  stillRunning: (modelId: string) => boolean,
  modelIds: ReadonlyMap<string, string>,
): string[] {
  const interrupted: string[] = [];
  for (let index = 0; index < reports.length; index += 1) {
    const report = reports[index]!;
    if (!report.started || !report.ready) continue;
    const modelId = modelIds.get(report.hostId);
    if (modelId === undefined) continue;
    if (stillRunning(modelId)) continue;
    interrupted.push(report.hostId);
    reports[index] = {
      ...report,
      state: 'interrupted',
      ready: false,
      detail: `${report.hostName} stopped while the scan was reading it: it was started and confirmed ready, then its ` +
        'process exited before the scan finished. Whatever was collected before that point is kept, and the catalog is not ' +
        'replaced with an empty list.',
    };
  }
  return interrupted;
}

/**
 * Poll readiness until it answers or the deadline passes.
 *
 * @param host - the engine.
 * @param effects - the probe.
 * @param budgetMs - the deadline, measured from `startedAt`.
 * @param intervalMs - how long to wait between probes.
 * @param sleep - the sleep to use.
 * @param startedAt - when the bring-up began.
 * @param now - the clock the deadline is measured against.
 * @returns the last probe result.
 */
async function waitForReady(
  host: ModelHost,
  effects: EngineStartupEffects,
  budgetMs: number,
  intervalMs: number,
  sleep: (ms: number) => Promise<void>,
  startedAt: number,
  now: () => number,
): Promise<{ running: boolean; detail: string }> {
  let last = await effects.probe(host);
  while (!last.running && now() - startedAt < budgetMs) {
    await sleep(intervalMs);
    last = await effects.probe(host);
  }
  return { running: last.running, detail: last.detail };
}

/**
 * Bound one effect so a hung engine cannot hang the scan.
 *
 * @param operation - the effect.
 * @param timeoutMs - the budget.
 * @returns the operation's result.
 * @throws an Error carrying `code: 'SCAN_STARTUP_TIMEOUT'` when the budget expires.
 */
async function withTimeout<T>(operation: Promise<T>, timeoutMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          const error = new Error(`the readiness budget of ${timeoutMs} ms expired`) as Error & { code?: string };
          error.code = 'SCAN_STARTUP_TIMEOUT';
          reject(error);
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

/** Build one engine report, filling the fields a caller does not name. */
function reportOf(
  host: ModelHost,
  state: EngineStartState,
  fields: {
    readonly started: boolean;
    readonly ready: boolean;
    readonly durationMs: number;
    readonly detail: string;
    readonly protectedFromIdleStop?: boolean;
  },
): EngineStartupReport {
  return {
    hostId: host.id,
    hostName: host.name,
    state,
    started: fields.started,
    ready: fields.ready,
    durationMs: fields.durationMs,
    detail: fields.detail,
    protectedFromIdleStop: fields.protectedFromIdleStop === true,
  };
}
