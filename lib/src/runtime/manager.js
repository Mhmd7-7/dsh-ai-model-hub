/**
 * The Model Runtime Manager.
 *
 * This component owns everything *stateful* about a model: whether a process
 * exists, whether the hub started it, whether it is healthy, how many calls are
 * in flight, and when to shut it down. It is the only place that mutates
 * lifecycle, and the only place (besides health probes) that spawns processes.
 *
 * It knows nothing about capabilities. It will start "model X" because it was
 * asked to; deciding *that* X is the right model is the router's job. Keeping
 * that line clean is what lets you swap routing policy without touching process
 * supervision, and vice versa.
 *
 * @module dsh-ai-model-hub/runtime/manager
 */
import { Socket } from 'node:net';
import { AdapterRegistry, silentLogger } from "../adapters/types.js";
import { ModelHubError, toHubError } from "../errors.js";
import { DEFAULT_EXECUTION_POLICY, delay, runCommand, spawnProcess, withTimeout, } from "../util/process.js";
/**
 * Supervises model processes and answers "may I use this right now?".
 *
 * Every state transition is serialized per model. Two agents asking for the same
 * cold model concurrently must produce one process, not two — a real failure mode
 * for engines that bind a fixed port.
 */
export class RuntimeManager {
    catalog;
    adapters;
    policy;
    log;
    healthIntervalMs;
    idleSweepIntervalMs;
    /** The catalog's directory, handed to adapters that resolve a catalog-relative file. */
    catalogDir;
    states = new Map();
    healthTimer;
    idleTimer;
    disposed = false;
    /**
     * @param options - catalog, adapters, policy, and probe cadence.
     */
    constructor(options) {
        this.catalog = options.catalog;
        this.adapters = options.adapters;
        this.policy = options.policy ?? DEFAULT_EXECUTION_POLICY;
        this.log = options.log ?? (() => { });
        this.healthIntervalMs = options.healthIntervalMs ?? 30_000;
        this.idleSweepIntervalMs = options.idleSweepIntervalMs ?? 15_000;
        this.catalogDir = options.catalogDir;
        this.syncCatalog();
    }
    /**
     * Bring the runtime's per-model state into agreement with the catalog.
     *
     * Called from the constructor and again whenever the catalog gains or loses
     * models. That second case is not hypothetical: runtime model discovery
     * republishes the catalog after the hub is already constructed, so a model can
     * appear that this manager has never seen. Without a state entry for it,
     * `getModelStatus` throws — and because `ModelHub.listModels()` reads a status
     * for every catalog model, one discovered model would break the whole listing,
     * the settings page, and `explain_routing` at once.
     *
     * Everything here is idempotent and additive:
     *
     * - A model with no state gets the same initial state and the same one-time
     *   adapter/resource validation the constructor performs.
     * - A model that already has state is left completely alone, so a running
     *   process, a health report, or an in-flight invocation is never disturbed by
     *   a discovery pass.
     * - State for a model that has left the catalog is kept if the hub owns a
     *   process for it, so the process is still stoppable and still gets shut down
     *   on dispose; otherwise it is dropped.
     */
    syncCatalog() {
        for (const model of this.catalog.listModels()) {
            if (this.states.has(model.id))
                continue;
            this.states.set(model.id, {
                // A model the hub *could* start rests at `not_running`: nothing is known
                // to be alive yet. A model it can never start is `external` from the
                // outset, because any liveness it has belongs to something else. A probe
                // that succeeds later moves a `not_running` model to `external` too — so
                // the state means "a live process the hub does not own", never merely
                // "not probed".
                lifecycle: model.lifecycle.startable ? 'not_running' : 'external',
                availability: model.enabled ? 'stopped' : 'disabled',
                health: undefined,
                process: undefined,
                startedAt: undefined,
                lastUsedAt: undefined,
                activeInvocations: 0,
                reason: model.enabled ? undefined : 'disabled in configuration',
                transition: Promise.resolve(),
            });
            this.validateModel(model.id);
        }
        // A model that has left the catalog stops being routable immediately —
        // `requireModel` in the catalog already refuses it — but a process the hub
        // owns must stay stoppable, so its state is retained until it is gone.
        const present = new Set(this.catalog.listModelIds());
        for (const [modelId, state] of [...this.states]) {
            if (present.has(modelId))
                continue;
            if (state.process !== undefined)
                continue;
            this.states.delete(modelId);
        }
    }
    /**
     * Check one model's adapter support and resource fit once, recording the
     * verdict as its initial availability.
     *
     * Separate from {@link syncCatalog} only so the constructor path and the
     * discovery path cannot disagree about what a healthy starting state is.
     *
     * @param modelId - the model to validate; it must already have a state entry.
     */
    validateModel(modelId) {
        const model = this.catalog.getModel(modelId);
        if (model === undefined || !model.enabled)
            return;
        const state = this.states.get(modelId);
        if (state === undefined)
            return;
        const adapter = this.adapters.get(model.adapter);
        if (adapter === undefined) {
            state.availability = 'error';
            state.reason = `no adapter registered for kind "${model.adapter}"`;
            this.log(`runtime: model ${model.id} has no adapter for kind ${model.adapter}`, { modelId: model.id });
            return;
        }
        const support = adapter.supports(model);
        if (!support.ok) {
            state.availability = 'unsupported';
            state.reason = support.reason;
            this.log(`runtime: model ${model.id} is not supported by its adapter: ${support.reason}`, {
                modelId: model.id,
            });
            return;
        }
        const resources = this.catalog.checkResources(model, { detail: true });
        if (!resources.supported) {
            state.availability = 'unsupported';
            state.reason = resources.reason;
            this.log(`runtime: model ${model.id} exceeds this machine: ${resources.reason}`, { modelId: model.id });
        }
    }
    /**
     * Start periodic health probing and idle sweeping.
     *
     * Separate from the constructor so tests can construct a manager without
     * timers, and so a CLI can run one-shot without a background loop.
     */
    start() {
        if (this.disposed)
            return;
        if (this.healthIntervalMs > 0 && this.healthTimer === undefined) {
            this.healthTimer = setInterval(() => {
                void this.probeAll();
            }, this.healthIntervalMs);
            this.healthTimer.unref?.();
        }
        if (this.idleSweepIntervalMs > 0 && this.idleTimer === undefined) {
            this.idleTimer = setInterval(() => {
                void this.stopIdleModels();
            }, this.idleSweepIntervalMs);
            this.idleTimer.unref?.();
        }
    }
    /**
     * Stop probing, stop the idle sweeper, and shut down every process the hub owns.
     *
     * Graceful by construction: each process gets its declared shutdown budget, and
     * shutdown is attempted for every model even if one hangs.
     */
    async dispose() {
        this.disposed = true;
        if (this.healthTimer !== undefined)
            clearInterval(this.healthTimer);
        if (this.idleTimer !== undefined)
            clearInterval(this.idleTimer);
        this.healthTimer = undefined;
        this.idleTimer = undefined;
        const owned = [...this.states.entries()].filter(([, state]) => state.process !== undefined);
        await Promise.all(owned.map(async ([modelId]) => {
            try {
                await this.stopModel(modelId, { force: false });
            }
            catch (error) {
                this.log(`runtime: shutdown of ${modelId} failed: ${String(error)}`, { modelId });
            }
        }));
    }
    /**
     * Re-check one model's adapter support and resource fit, and correct its state.
     *
     * Called by the hub after a machine re-probe, because resource fit is the one
     * verdict that goes *stale* rather than merely being unknown: a catalog built
     * before the first probe has no measurements, so `checkResources` passes
     * everything through; the probe then lands and several models become
     * unrunnable. Without this re-check they would keep the verdict from before the
     * machine was measured, and a `start_model` would be refused at the gate with a
     * diagnosis ("unsupported") that contradicts what routing says.
     *
     * Only a model resting in a resource-derived state is revisited: a model that
     * is disabled, in an error state, or unsupported by its *adapter* has a reason
     * the machine has nothing to do with, and a live or owned process is never
     * disturbed.
     *
     * @param modelId - the model to re-check.
     */
    revalidateResources(modelId) {
        const model = this.catalog.getModel(modelId);
        const state = this.states.get(modelId);
        if (model === undefined || state === undefined || !model.enabled)
            return;
        if (state.process !== undefined)
            return;
        if (state.availability === 'disabled' || state.availability === 'error')
            return;
        if (state.availability === 'available' || state.availability === 'starting')
            return;
        const resources = this.catalog.checkResources(model, { detail: true });
        if (resources.supported) {
            // A model that was `unsupported` only because nothing had been measured yet
            // becomes startable again. Its previous reason belonged to that stale
            // verdict, so it goes with it.
            if (state.availability === 'unsupported') {
                state.availability = 'stopped';
                state.reason = undefined;
            }
            return;
        }
        state.availability = 'unsupported';
        state.reason = resources.reason;
    }
    /**
     * Report the live state of one model.
     * @param modelId - the model id.
     * @returns the status.
     * @throws ModelHubError with `MODEL_NOT_FOUND` when unknown.
     */
    getModelStatus(modelId) {
        this.catalog.requireModel(modelId);
        const state = this.states.get(modelId);
        if (state === undefined) {
            throw new ModelHubError('MODEL_NOT_FOUND', `no runtime state for model "${modelId}"`, { modelId });
        }
        const status = {
            modelId,
            availability: state.availability,
            lifecycle: state.lifecycle,
            activeInvocations: state.activeInvocations,
        };
        if (state.health !== undefined)
            status.health = state.health;
        const pid = state.process?.pid;
        if (pid !== undefined)
            status.pid = pid;
        if (state.startedAt !== undefined)
            status.startedAt = state.startedAt;
        if (state.reason !== undefined)
            status.reason = state.reason;
        return status;
    }
    /**
     * Report the live state of every model.
     * @returns one status per configured model.
     */
    getAllStatuses() {
        return this.catalog.listModelIds().map((modelId) => this.getModelStatus(modelId));
    }
    /**
     * Whether a model is usable right now, without changing anything.
     * @param modelId - the model id.
     * @returns the availability state.
     */
    checkAvailability(modelId) {
        return this.getModelStatus(modelId).availability;
    }
    /**
     * Run a health probe against one model and record the result.
     *
     * Never throws: an unreachable engine is a report. The status transitions are
     * the same whether the caller is the periodic sweeper or a cold start, so
     * there is one code path for "is it alive" and one place it can be wrong.
     *
     * @param modelId - the model id.
     * @returns the probe result.
     */
    async probeHealth(modelId) {
        const model = this.catalog.requireModel(modelId);
        const state = this.states.get(modelId);
        if (state === undefined) {
            throw new ModelHubError('MODEL_NOT_FOUND', `no runtime state for model "${modelId}"`, { modelId });
        }
        if (!model.enabled) {
            const report = { healthy: false, checkedAt: Date.now(), detail: 'disabled' };
            state.health = report;
            state.availability = 'disabled';
            return report;
        }
        const adapter = this.adapters.get(model.adapter);
        if (adapter === undefined) {
            const report = {
                healthy: false,
                checkedAt: Date.now(),
                detail: `no adapter registered for kind "${model.adapter}"`,
            };
            state.health = report;
            state.availability = 'error';
            state.reason = report.detail;
            return report;
        }
        const controller = new AbortController();
        const budget = model.health.timeoutMs ?? 2_000;
        let report;
        try {
            report = await withTimeout(this.runHealthCheck(model, controller.signal), budget, `health check for ${modelId}`);
        }
        catch (error) {
            const described = toHubError(error, 'HEALTH_CHECK_FAILED', { modelId });
            report = {
                healthy: false,
                checkedAt: Date.now(),
                detail: described.message,
            };
            controller.abort();
        }
        state.health = report;
        // Endpoint liveness is probed only when the hub owns no process: if it does,
        // that process already settles the question, and probing could mistake the
        // model's own (still loading) server for an externally managed one.
        const endpointLive = state.process === undefined ? await this.probeEndpointLiveness(model) : false;
        this.resolveAvailability(model, state, report, endpointLive);
        return report;
    }
    /**
     * Ask whether anything is actually listening for this model's endpoint.
     *
     * Returns false immediately for a model with no endpoint, and for a `none`
     * health strategy, because neither can establish liveness. This is deliberately
     * independent of the adapter: an adapter can report that its own logic works
     * when there is no process at all (the in-process mock adapter does exactly
     * that), and treating that as "already running" would make the hub refuse to
     * launch a model it should launch.
     *
     * @param model - the resolved model.
     * @returns whether a live endpoint answered.
     */
    async probeEndpointLiveness(model) {
        if (model.runtime.endpoint === undefined)
            return false;
        const target = parseHostPort(model);
        if (target === undefined)
            return false;
        return probeTcp(target.host, target.port, model.health.timeoutMs ?? 2_000);
    }
    /**
     * Execute the model's declared health strategy.
     *
     * The adapter's own `health` is used for adapters that have a meaningful
     * in-process answer (the mock adapter always does), while a declared
     * `tcp`/`http`/`command` strategy is executed here because it is about the
     * *process or endpoint*, not the adapter's logic.
     *
     * @param model - the resolved model.
     * @param signal - cancellation for the probe.
     * @returns the probe result.
     */
    async runHealthCheck(model, signal) {
        const started = Date.now();
        switch (model.health.kind) {
            case 'none': {
                const adapter = this.adapters.get(model.adapter);
                if (adapter === undefined) {
                    return { healthy: false, checkedAt: started, detail: `no adapter for "${model.adapter}"` };
                }
                // The catalog directory travels with the probe: an adapter that loads a
                // file the catalog named must resolve it the same way here as during an
                // invocation, or a correctly configured model probes as broken.
                return adapter.health(model, signal, ...(this.catalogDir === undefined ? [] : [{ catalogDir: this.catalogDir }]));
            }
            case 'tcp': {
                const target = parseHostPort(model);
                if (target === undefined) {
                    return {
                        healthy: false,
                        checkedAt: started,
                        detail: 'a tcp health check requires an endpoint with a host and port',
                    };
                }
                const ok = await probeTcp(target.host, target.port, model.health.timeoutMs ?? 2_000, signal);
                return {
                    healthy: ok,
                    checkedAt: started,
                    latencyMs: Date.now() - started,
                    detail: ok ? `tcp ${target.host}:${target.port} accepted a connection` : `tcp ${target.host}:${target.port} refused`,
                };
            }
            case 'http': {
                return probeHttp(model, signal, started);
            }
            case 'command': {
                const command = model.health.command;
                if (command === undefined) {
                    return { healthy: false, checkedAt: started, detail: 'a command health check requires `command`' };
                }
                const result = await runCommand({ command, args: [...(model.health.args ?? [])] }, {
                    timeoutMs: model.health.timeoutMs ?? 5_000,
                    policy: this.policy,
                    signal,
                    label: `health(${model.id})`,
                });
                return {
                    healthy: result.ok,
                    checkedAt: started,
                    latencyMs: Date.now() - started,
                    detail: result.ok ? `command exited 0` : `command failed (${result.reason ?? 'unknown'}): ${result.stderr.trim().slice(0, 300)}`,
                };
            }
            default: {
                const exhaustive = model.health.kind;
                return { healthy: false, checkedAt: started, detail: `unhandled health kind ${String(exhaustive)}` };
            }
        }
    }
    /**
     * Derive availability from a health report, honoring the lifecycle.
     *
     * Two distinct facts are being combined, and conflating them was a real bug:
     *
     * - *Health* answers "is the model's engine responsive?" For an in-process
     *   adapter the answer is always yes, because there is no engine to be
     *   unresponsive.
     * - *Liveness* answers "is there a process behind this model?" That is decided
     *   by whether the hub owns a process, or — when it does not — by whether a
     *   live *endpoint* answered. An adapter-only probe can never establish it.
     *
     * So a healthy in-process model reports `available` while remaining
     * `not_running`, which is why starting one still launches a real process rather
     * than concluding a process already exists.
     *
     * @param model - the resolved model.
     * @param state - its mutable state.
     * @param report - the fresh probe.
     * @param endpointLive - whether a live endpoint answered for this model.
     */
    resolveAvailability(model, state, report, endpointLive) {
        // Capacity, not headroom: a model that is not resident is not competing with
        // anything, and the definitive headroom question for a *running* model is
        // asked at routing time and again before a launch. Asking it here would make
        // a busy machine mark a perfectly startable model `unsupported`, which is a
        // state nothing would ever clear.
        const resources = this.catalog.checkResources(model, { detail: true });
        if (!resources.supported) {
            state.availability = 'unsupported';
            // The reason is kept, not just the verdict: routing refuses an unsupported
            // model with this sentence, and "not supported on this machine" without the
            // shortfall is exactly the unhelpful answer an operator is trying to avoid.
            state.reason = resources.reason;
            return;
        }
        if (!model.enabled) {
            state.availability = 'disabled';
            state.reason = 'disabled in configuration';
            return;
        }
        if (state.process !== undefined) {
            // The hub owns the process, so its liveness is unambiguous.
            state.lifecycle = 'running';
            state.availability = report.healthy ? 'available' : 'unhealthy';
            state.reason = report.healthy ? undefined : report.detail;
            state.startedAt ??= Date.now();
            return;
        }
        if (!report.healthy) {
            state.reason = report.detail;
            state.availability = state.lifecycle === 'starting' ? 'starting' : 'stopped';
            if (state.lifecycle !== 'failed' && state.lifecycle !== 'starting') {
                state.lifecycle = model.lifecycle.startable ? 'not_running' : 'external';
            }
            return;
        }
        // Healthy with no owned process. Something is serving this model, and the
        // hub did not start it; the only question is whether that is knowable.
        state.availability = 'available';
        state.reason = undefined;
        if (endpointLive) {
            state.lifecycle = 'external';
            state.startedAt ??= Date.now();
        }
        else if (state.lifecycle !== 'running' && state.lifecycle !== 'starting') {
            // In-process or process-per-request: usable, and no process of ours exists.
            state.lifecycle = model.lifecycle.startable ? 'not_running' : 'external';
        }
    }
    /**
     * Probe every enabled model once.
     *
     * Run concurrently but bounded: each probe has its own timeout, and results are
     * recorded independently, so one dead engine cannot stall the survey.
     */
    async probeAll() {
        if (this.disposed)
            return;
        const targets = this.catalog.listModels().filter((model) => model.enabled);
        await Promise.all(targets.map(async (model) => {
            try {
                await this.probeHealth(model.id);
            }
            catch (error) {
                this.log(`runtime: health probe for ${model.id} failed: ${String(error)}`, { modelId: model.id });
            }
        }));
    }
    /**
     * Start a model's process and, unless configured otherwise, wait for health.
     *
     * Idempotent: starting an already-healthy model is a no-op that reports
     * `alreadyRunning`, which is what makes it safe to call from the invocation
     * gate on every request.
     *
     * @param modelId - the model id.
     * @returns a summary of what happened.
     * @throws ModelHubError for unknown, disabled, non-startable, or failing models.
     */
    async startModel(modelId) {
        const model = this.catalog.requireModel(modelId);
        const state = this.requireState(modelId);
        return this.serializeTransition(state, async () => {
            if (!model.enabled) {
                throw new ModelHubError('MODEL_UNAVAILABLE', `model "${modelId}" is disabled in configuration`, {
                    modelId,
                });
            }
            const support = this.adapters.require(model.adapter).supports(model);
            if (!support.ok) {
                throw new ModelHubError('UNSUPPORTED_OPERATION', `model "${modelId}" cannot run: ${support.reason}`, {
                    modelId,
                });
            }
            const resources = this.catalog.checkResources(model, {
                // A model the hub is about to launch gets the whole machine, so the
                // question is capacity, not headroom. The exception is a model whose
                // engine is somehow already resident — then its own footprint is real
                // and must be counted against what is free.
                live: state.availability === 'available',
                detail: true,
            });
            if (!resources.supported) {
                throw new ModelHubError('INSUFFICIENT_RESOURCES', `model "${modelId}" cannot run: ${resources.reason}`, {
                    modelId,
                    machine: this.catalog.machineProfile,
                });
            }
            // An already-live endpoint wins over spawning a duplicate. Only an
            // *endpoint* can satisfy this: a healthy in-process adapter says nothing
            // about whether a process exists, and a process the hub already owns is
            // handled below.
            if (state.process === undefined) {
                const endpointLive = await this.probeEndpointLiveness(model);
                if (endpointLive) {
                    const existing = await this.probeHealth(modelId);
                    if (existing.healthy) {
                        this.log(`runtime: ${modelId} is already served by a live endpoint; not starting a second instance`, {
                            modelId,
                        });
                        return { started: false, alreadyRunning: true, health: existing };
                    }
                }
            }
            if (!model.lifecycle.startable || model.lifecycle.start === undefined) {
                throw new ModelHubError('LIFECYCLE_UNSUPPORTED', `model "${modelId}" is not startable (no launch command configured). ` +
                    'Start its engine yourself and it will be detected as available.', { modelId, endpoint: model.runtime.endpoint });
            }
            if (state.process !== undefined) {
                // A process exists. If it is answering, there is nothing to do; otherwise
                // it must be replaced, since leaving it would hold the port.
                const owned = await this.probeHealth(modelId);
                if (owned.healthy) {
                    return { started: false, alreadyRunning: true, health: owned };
                }
                this.log(`runtime: ${modelId} has an unhealthy process; replacing it`, { modelId, pid: state.process.pid });
                await this.stopProcess(model, state, { force: true });
            }
            this.log(`runtime: starting ${modelId} via ${model.lifecycle.start.command}`, {
                modelId,
                command: model.lifecycle.start.command,
            });
            state.lifecycle = 'starting';
            state.availability = 'starting';
            state.reason = undefined;
            let handle;
            try {
                handle = spawnProcess({
                    command: model.lifecycle.start.command,
                    args: [...(model.lifecycle.start.args ?? []), ...(model.runtime.args ?? [])],
                    ...(model.lifecycle.start.cwd === undefined ? {} : { cwd: model.lifecycle.start.cwd }),
                }, {
                    policy: this.policy,
                    ...(model.runtime.env === undefined ? {} : { env: model.runtime.env }),
                    label: `model:${modelId}`,
                });
            }
            catch (error) {
                const described = toHubError(error, 'START_FAILED', { modelId });
                state.lifecycle = 'failed';
                state.availability = 'error';
                state.reason = described.message;
                throw described;
            }
            state.process = handle;
            state.startedAt = Date.now();
            if (!model.lifecycle.awaitHealthOnStart) {
                state.lifecycle = 'running';
                state.availability = 'available';
                state.reason = undefined;
                return {
                    started: true,
                    alreadyRunning: false,
                    health: { healthy: true, checkedAt: Date.now(), detail: 'health check deferred by configuration' },
                };
            }
            const health = await this.awaitHealthy(model, state, handle);
            return { started: true, alreadyRunning: false, health };
        });
    }
    /**
     * Poll health until the model answers or the startup budget expires.
     *
     * Polling rather than sleeping is what makes a cold start as fast as the engine
     * allows instead of as slow as its declared worst case.
     *
     * @param model - the resolved model.
     * @param state - its mutable state.
     * @param handle - the process that was just started.
     * @returns the first successful health report.
     * @throws ModelHubError with `START_FAILED` when the budget expires or the process dies.
     */
    async awaitHealthy(model, state, handle) {
        const deadline = Date.now() + model.lifecycle.startupTimeoutMs;
        const interval = Math.min(1_000, Math.max(100, Math.round(model.health.timeoutMs ?? 2_000) / 2));
        let last = { healthy: false, checkedAt: Date.now(), detail: 'not probed yet' };
        while (Date.now() < deadline) {
            if (!handle.running) {
                const detail = handle.stderrSnapshot().trim().slice(-500);
                state.lifecycle = 'failed';
                state.availability = 'error';
                state.reason = `process exited during startup (${detail.length > 0 ? detail : 'no output'})`;
                throw new ModelHubError('START_FAILED', `model "${model.id}" exited during startup`, {
                    modelId: model.id,
                    stderr: detail,
                    stdout: handle.stdoutSnapshot().trim().slice(-500),
                });
            }
            last = await this.probeHealth(model.id);
            if (last.healthy) {
                state.lifecycle = 'running';
                state.availability = 'available';
                state.reason = undefined;
                this.log(`runtime: ${model.id} is healthy after ${Date.now() - (state.startedAt ?? Date.now())} ms`, {
                    modelId: model.id,
                    pid: handle.pid,
                });
                return last;
            }
            await delay(interval);
        }
        state.lifecycle = 'failed';
        state.availability = 'unhealthy';
        state.reason = `did not become healthy within ${model.lifecycle.startupTimeoutMs} ms`;
        throw new ModelHubError('START_FAILED', `model "${model.id}" did not become healthy in time`, {
            modelId: model.id,
            timeoutMs: model.lifecycle.startupTimeoutMs,
            lastHealth: last.detail,
            stderr: handle.stderrSnapshot().trim().slice(-500),
        });
    }
    /**
     * Stop a model's process.
     * @param modelId - the model id.
     * @param options - `force` skips the declared graceful budget.
     * @returns what happened.
     * @throws ModelHubError when the model is unknown or has no process to stop.
     */
    async stopModel(modelId, options = {}) {
        const model = this.catalog.requireModel(modelId);
        const state = this.requireState(modelId);
        return this.serializeTransition(state, async () => {
            if (state.process === undefined) {
                // Nothing owned. Reset to a coherent resting state so the status is honest.
                state.lifecycle = model.lifecycle.startable ? 'not_running' : 'external';
                if (state.availability !== 'disabled' && state.availability !== 'unsupported' && state.availability !== 'error') {
                    state.availability = 'stopped';
                }
                return { stopped: false, wasRunning: false };
            }
            const wasRunning = state.process.running;
            await this.stopProcess(model, state, { force: options.force ?? false });
            return { stopped: true, wasRunning };
        });
    }
    /**
     * Terminate a process and update state, without taking the transition lock.
     * @param model - the resolved model.
     * @param state - its mutable state.
     * @param options - `force` skips the graceful budget.
     */
    async stopProcess(model, state, options) {
        const handle = state.process;
        if (handle === undefined)
            return;
        state.lifecycle = 'stopping';
        state.availability = 'starting';
        state.reason = undefined;
        const grace = options.force ? 1_000 : model.lifecycle.shutdownTimeoutMs;
        try {
            await handle.stop(grace);
        }
        catch (error) {
            this.log(`runtime: error stopping ${model.id}: ${String(error)}`, { modelId: model.id });
        }
        // Give the engine a moment to release its port before anything else binds it.
        if (model.lifecycle.stop !== undefined) {
            const result = await runCommand({ command: model.lifecycle.stop.command, args: [...(model.lifecycle.stop.args ?? [])] }, { timeoutMs: Math.max(grace, 5_000), policy: this.policy, label: `stop:${model.id}` });
            if (!result.ok) {
                this.log(`runtime: stop command for ${model.id} failed: ${result.stderr.trim().slice(0, 200)}`, {
                    modelId: model.id,
                });
            }
        }
        await delay(150);
        state.process = undefined;
        state.startedAt = undefined;
        state.lifecycle = model.lifecycle.startable ? 'not_running' : 'external';
        state.availability = 'stopped';
        state.health = { healthy: false, checkedAt: Date.now(), detail: 'stopped by the hub' };
        this.log(`runtime: stopped ${model.id}`, { modelId: model.id });
    }
    /**
     * Restart a model, stopping first when a process is owned.
     * @param modelId - the model id.
     * @returns what happened.
     */
    async restartModel(modelId) {
        this.catalog.requireModel(modelId);
        const state = this.requireState(modelId);
        if (state.process !== undefined) {
            await this.stopModel(modelId, { force: false });
        }
        return this.startModel(modelId);
    }
    /**
     * Stop every hub-owned process that has been idle past its declared timeout.
     *
     * Idle means both "no invocation in flight" and "no invocation for longer than
     * `idleTimeoutMs`". Models that declare no timeout are never swept — an idle
     * timeout is opt-in because unloading a model the user is about to reuse is a
     * far worse outcome than holding VRAM.
     */
    async stopIdleModels() {
        if (this.disposed)
            return [];
        const stopped = [];
        const now = Date.now();
        for (const model of this.catalog.listModels()) {
            if (model.lifecycle.idleTimeoutMs <= 0)
                continue;
            const state = this.states.get(model.id);
            if (state === undefined || state.process === undefined)
                continue;
            if (state.activeInvocations > 0)
                continue;
            const since = state.lastUsedAt ?? state.startedAt;
            if (since === undefined)
                continue;
            if (now - since < model.lifecycle.idleTimeoutMs)
                continue;
            this.log(`runtime: stopping ${model.id} after ${model.lifecycle.idleTimeoutMs} ms idle`, {
                modelId: model.id,
            });
            try {
                const result = await this.stopModel(model.id, { force: false });
                if (result.stopped)
                    stopped.push(model.id);
            }
            catch (error) {
                this.log(`runtime: idle stop of ${model.id} failed: ${String(error)}`, { modelId: model.id });
            }
        }
        return stopped;
    }
    /**
     * Ensure a model is ready for an invocation, starting it when necessary.
     *
     * This is the single gate every routed call passes through, which is why
     * concurrent cold starts are coalesced by the per-model transition lock and why
     * the returned `coldStart` flag is trustworthy enough to report to the user.
     *
     * @param modelId - the model id.
     * @returns the gate decision.
     * @throws ModelHubError when the model cannot be made ready.
     */
    async ensureReady(modelId) {
        const model = this.catalog.requireModel(modelId);
        const status = this.getModelStatus(modelId);
        if (status.availability === 'disabled') {
            throw new ModelHubError('MODEL_UNAVAILABLE', `model "${modelId}" is disabled in configuration`, { modelId });
        }
        if (status.availability === 'unsupported') {
            throw new ModelHubError('INSUFFICIENT_RESOURCES', `model "${modelId}" cannot run here: ${status.reason ?? 'unsupported'}`, { modelId, reason: status.reason });
        }
        if (status.availability === 'error') {
            throw new ModelHubError('MODEL_UNAVAILABLE', `model "${modelId}" is in an error state: ${status.reason ?? 'unknown'}`, {
                modelId,
                reason: status.reason,
            });
        }
        if (status.availability === 'available') {
            return { allowed: true, coldStart: false };
        }
        if (!model.lifecycle.startable) {
            // Not available and not startable: probe once so an externally-started
            // engine that appeared since the last sweep is picked up immediately.
            const probe = await this.probeHealth(modelId);
            if (probe.healthy)
                return { allowed: true, coldStart: false };
            throw new ModelHubError('MODEL_UNAVAILABLE', `model "${modelId}" is not running and is not startable by the hub` +
                (model.runtime.endpoint === undefined ? '' : ` (expected it at ${model.runtime.endpoint})`), { modelId, health: probe.detail, endpoint: model.runtime.endpoint });
        }
        if (status.availability === 'starting') {
            // Another caller is already starting it; wait on the same transition.
            const state = this.requireState(modelId);
            await state.transition.catch(() => undefined);
            const after = await this.probeHealth(modelId);
            if (after.healthy)
                return { allowed: true, coldStart: true };
            throw new ModelHubError('START_FAILED', `model "${modelId}" failed to become available`, {
                modelId,
                health: after.detail,
            });
        }
        await this.startModel(modelId);
        return { allowed: true, coldStart: true };
    }
    /**
     * Record that an invocation started, for idle accounting.
     * @param modelId - the model id.
     */
    beginInvocation(modelId) {
        const state = this.requireState(modelId);
        state.activeInvocations += 1;
        state.lastUsedAt = Date.now();
    }
    /**
     * Record that an invocation settled, for idle accounting.
     * @param modelId - the model id.
     */
    endInvocation(modelId) {
        const state = this.states.get(modelId);
        if (state === undefined)
            return;
        state.activeInvocations = Math.max(0, state.activeInvocations - 1);
        state.lastUsedAt = Date.now();
    }
    /**
     * Build an adapter logger bound to one model, so adapter diagnostics carry the
     * model id without the adapter knowing it has one.
     *
     * @param modelId - the model id.
     * @returns a logger.
     */
    adapterLogger(modelId) {
        return {
            debug: (message, fields) => this.log(`adapter[${modelId}] ${message}`, fields),
            info: (message, fields) => this.log(`adapter[${modelId}] ${message}`, fields),
            warn: (message, fields) => this.log(`adapter[${modelId}] ${message}`, fields),
        };
    }
    /**
     * Fetch a model's state, failing loudly when it is absent.
     * @param modelId - the model id.
     * @returns the mutable state.
     */
    requireState(modelId) {
        const state = this.states.get(modelId);
        if (state === undefined) {
            throw new ModelHubError('MODEL_NOT_FOUND', `no runtime state for model "${modelId}"`, { modelId });
        }
        return state;
    }
    /**
     * Run a lifecycle transition with exclusive access to one model.
     * @param state - the model's state.
     * @param operation - the transition.
     * @returns the transition's result.
     */
    serializeTransition(state, operation) {
        const run = state.transition.then(operation, operation);
        state.transition = run.then(() => undefined, () => undefined);
        return run;
    }
}
/**
 * Extract host and port from a model's endpoint.
 * @param model - the resolved model.
 * @returns the host and port, or `undefined` when the endpoint has no usable port.
 */
export function parseHostPort(model) {
    const endpoint = model.runtime.endpoint;
    if (endpoint === undefined)
        return undefined;
    try {
        const url = new URL(endpoint);
        const port = url.port.length > 0 ? Number(url.port) : url.protocol === 'https:' ? 443 : 80;
        if (!Number.isInteger(port) || port <= 0)
            return undefined;
        return { host: url.hostname, port };
    }
    catch {
        return undefined;
    }
}
/**
 * Attempt a TCP connection with a deadline.
 * @param host - target host.
 * @param port - target port.
 * @param timeoutMs - connect budget.
 * @param signal - cancellation.
 * @returns whether the connection succeeded.
 */
export function probeTcp(host, port, timeoutMs, signal) {
    return new Promise((resolve) => {
        let settled = false;
        const finish = (value) => {
            if (settled)
                return;
            settled = true;
            socket.destroy();
            signal?.removeEventListener('abort', onAbort);
            resolve(value);
        };
        const socket = new Socket();
        socket.setTimeout(timeoutMs);
        const onAbort = () => finish(false);
        socket.once('connect', () => finish(true));
        socket.once('timeout', () => finish(false));
        socket.once('error', () => finish(false));
        if (signal !== undefined) {
            if (signal.aborted) {
                finish(false);
                return;
            }
            signal.addEventListener('abort', onAbort, { once: true });
        }
        socket.connect(port, host);
    });
}
/**
 * Probe a model's HTTP endpoint.
 *
 * Any 2xx/3xx counts as healthy: engines disagree about whether `/` returns 200,
 * 404, or a redirect, and all three mean "something is listening and speaking
 * HTTP", which is the question being asked.
 *
 * @param model - the resolved model.
 * @param signal - cancellation.
 * @param started - when the probe began, for latency reporting.
 * @returns the probe result.
 */
async function probeHttp(model, signal, started) {
    const endpoint = model.runtime.endpoint;
    if (endpoint === undefined) {
        return { healthy: false, checkedAt: started, detail: 'an http health check requires an endpoint' };
    }
    const relative = model.health.path ?? '/';
    let url;
    try {
        url = new URL(relative, endpoint.endsWith('/') ? endpoint : `${endpoint}/`).toString();
    }
    catch {
        return { healthy: false, checkedAt: started, detail: `endpoint "${endpoint}" is not a valid URL` };
    }
    const timeoutMs = model.health.timeoutMs ?? 2_000;
    const controller = new AbortController();
    const onAbort = () => controller.abort();
    signal.addEventListener('abort', onAbort, { once: true });
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(url, {
            method: 'GET',
            signal: controller.signal,
            redirect: 'manual',
            headers: { accept: '*/*' },
        });
        const healthy = response.status >= 200 && response.status < 400;
        return {
            healthy,
            checkedAt: started,
            latencyMs: Date.now() - started,
            detail: `${url} responded ${response.status}`,
        };
    }
    catch (error) {
        return {
            healthy: false,
            checkedAt: started,
            latencyMs: Date.now() - started,
            detail: `${url} unreachable: ${error instanceof Error ? error.message : String(error)}`,
        };
    }
    finally {
        clearTimeout(timer);
        signal.removeEventListener('abort', onAbort);
    }
}
/** A logger that discards everything, for runtimes constructed without one. */
export { silentLogger };
