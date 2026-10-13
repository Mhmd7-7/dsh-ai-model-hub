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
/** The default per-engine readiness budget. */
const DEFAULT_STARTUP_TIMEOUT_MS = 120_000;
/** ComfyUI loads checkpoints and custom nodes before it answers, so it gets longer. */
const COMFYUI_STARTUP_TIMEOUT_MS = 300_000;
/** How often to re-check readiness while an engine comes up. */
const DEFAULT_POLL_INTERVAL_MS = 500;
/** Whether a host is an engine this module brings up for a scan. */
function isManagedEngine(host) {
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
export function startupBudgetFor(host, override) {
    if (override !== undefined && Number.isFinite(override) && override > 0)
        return override;
    const declared = host.lifecycle?.startupTimeoutMs;
    if (typeof declared === 'number' && Number.isFinite(declared) && declared > 0)
        return declared;
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
export function launchBlocker(host, allowProcessLaunch) {
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
export async function bringUpEngines(options) {
    const { effects, onProgress } = options;
    const managed = options.hosts.filter(isManagedEngine);
    const reports = [];
    const releases = [];
    const modelIds = new Map();
    const sleep = options.sleep ?? (async (ms) => { await new Promise((resolve) => setTimeout(resolve, ms)); });
    const now = options.now ?? (() => Date.now());
    const pollIntervalMs = options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
    const release = () => {
        for (const give of releases.splice(0)) {
            try {
                give();
            }
            catch (error) {
                effects.log?.('engine scan release failed: ' + String(error));
            }
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
            let ready;
            try {
                ready = await withTimeout(effects.ensureReady(host), budget);
            }
            catch (error) {
                const timedOut = error.code === 'SCAN_STARTUP_TIMEOUT';
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
            }
            catch (error) {
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
    }
    catch (error) {
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
export function markInterruptedEngines(reports, stillRunning, modelIds) {
    const interrupted = [];
    for (let index = 0; index < reports.length; index += 1) {
        const report = reports[index];
        if (!report.started || !report.ready)
            continue;
        const modelId = modelIds.get(report.hostId);
        if (modelId === undefined)
            continue;
        if (stillRunning(modelId))
            continue;
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
async function waitForReady(host, effects, budgetMs, intervalMs, sleep, startedAt, now) {
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
async function withTimeout(operation, timeoutMs) {
    let timer;
    try {
        return await Promise.race([
            operation,
            new Promise((_, reject) => {
                timer = setTimeout(() => {
                    const error = new Error(`the readiness budget of ${timeoutMs} ms expired`);
                    error.code = 'SCAN_STARTUP_TIMEOUT';
                    reject(error);
                }, timeoutMs);
            }),
        ]);
    }
    finally {
        if (timer !== undefined)
            clearTimeout(timer);
    }
}
/** Build one engine report, filling the fields a caller does not name. */
function reportOf(host, state, fields) {
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
