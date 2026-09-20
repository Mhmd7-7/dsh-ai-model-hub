/**
 * The hub facade and its public API.
 *
 * This is the surface the DSH plugin layer binds to, and the surface a CLI or an
 * HTTP daemon would bind to as well. It exposes *capability* verbs only —
 * `invokeModel({ capability, … })` — and never a model-specific one. There is no
 * `generateWithStableDiffusion` here, and there never will be: the agent is not
 * permitted to know which engine serves a request, because the moment it knows,
 * swapping the engine becomes an agent-prompt change.
 *
 * Every method named in the original requirements lives here:
 * `listModels`, `getModel`, `getModelStatus`, `findModelsByCapability`,
 * `invokeModel`, `startModel`, `stopModel`.
 *
 * @module dsh-ai-model-hub/hub
 */
import { LocalArtifactStore } from "./artifacts/local-store.js";
import { AdapterRegistry } from "./adapters/types.js";
import { silentLogger } from "./adapters/types.js";
import { createMockAdapter } from "./adapters/mock.js";
import { createOpenAiCompatibleAdapter } from "./adapters/openai.js";
import { createHttpJsonAdapter } from "./adapters/http-json.js";
import { createComfyUiAdapter } from "./adapters/comfyui.js";
import { isCapability } from "./catalog/capabilities.js";
import { parseModelCatalogConfig } from "./catalog/descriptor.js";
import { ModelCatalog } from "./catalog/registry.js";
import { ModelHubError, toHubError } from "./errors.js";
import { probeMachine } from "./machine.js";
import { DEFAULT_EXECUTION_POLICY } from "./util/process.js";
import { RuntimeManager } from "./runtime/manager.js";
import { DEFAULT_ROUTING_POLICY, explainDecision, resolveRequestInputs, routeRequest } from "./router/router.js";
import { isLosslessJson } from "./util/validate.js";
/**
 * The hub: catalog + router + runtime + adapters + artifacts behind one API.
 *
 * Construct it once per process and share it. Its only mutable state is the
 * runtime manager's model lifecycle, which is internally serialized.
 */
export class ModelHub {
    /** The capability catalog. */
    catalog;
    /** The artifact store. */
    artifacts;
    /** The runtime manager. */
    runtime;
    /** The adapter registry, so a plugin can register an engine adapter later. */
    adapters;
    /** The routing policy in force. */
    routingPolicy;
    /** The execution policy in force, for diagnostics. */
    executionPolicy;
    listeners = new Set();
    log;
    ownsArtifacts;
    disposed = false;
    /**
     * Stores for the per-call roots {@link InvokeOptions.artifactRoot} names, keyed
     * by the root as given. The hub's own {@link artifacts} store is not in here.
     */
    callStores = new Map();
    /**
     * @param options - configuration, adapters, and policies.
     */
    constructor(options) {
        this.log = options.log ?? (() => { });
        this.routingPolicy = options.routingPolicy ?? DEFAULT_ROUTING_POLICY;
        this.executionPolicy = options.executionPolicy ?? DEFAULT_EXECUTION_POLICY;
        this.catalog = new ModelCatalog(options.config, {
            ...(options.machine === undefined ? {} : { machine: options.machine }),
            log: (message) => this.log(message),
        });
        const store = new LocalArtifactStore({
            root: options.artifactRoot ?? defaultArtifactRoot(),
            log: (message) => this.log(message),
        });
        this.artifacts = store;
        this.ownsArtifacts = true;
        // The shipping set: the Phase 1 fixture adapter, the real-engine adapter
        // that reaches every `/v1/chat/completions` server (Ollama, llama.cpp, vLLM,
        // LM Studio, …), the real-image adapter that reaches every
        // `/sdapi/v1/txt2img` server (A1111, Forge, stable-diffusion.cpp), and the
        // graph-queue adapter for ComfyUI. A deployment adds more through
        // `extraAdapters`, or replaces the whole set with `adapters`.
        const defaultAdapters = options.adapters ?? [
            createMockAdapter(),
            createOpenAiCompatibleAdapter(),
            createHttpJsonAdapter(),
            createComfyUiAdapter(),
        ];
        this.adapters = new AdapterRegistry([...defaultAdapters, ...(options.extraAdapters ?? [])]);
        this.runtime = new RuntimeManager({
            catalog: this.catalog,
            adapters: this.adapters,
            ...(options.executionPolicy === undefined ? {} : { policy: options.executionPolicy }),
            log: this.log,
            ...(options.healthIntervalMs === undefined ? {} : { healthIntervalMs: options.healthIntervalMs }),
            ...(options.idleSweepIntervalMs === undefined ? {} : { idleSweepIntervalMs: options.idleSweepIntervalMs }),
        });
        if (options.manageTimers !== false)
            this.runtime.start();
    }
    /**
     * The artifact store one call reads and writes through.
     *
     * Omitting the root returns the hub's own store — the deployment's configured
     * root, or the process-working-directory default — so a caller that says
     * nothing keeps the single-store behaviour. A named root gets its own store,
     * created once and reused, which is how a per-session workspace costs one map
     * lookup per call rather than a directory listing.
     *
     * @param root - the per-call artifact root, when the caller named one.
     * @returns the store to use.
     * @throws ModelHubError when the named root is not absolute.
     */
    storeFor(root) {
        if (root === undefined)
            return this.artifacts;
        const existing = this.callStores.get(root);
        if (existing !== undefined)
            return existing;
        const created = new LocalArtifactStore({ root, log: (message) => this.log(message) });
        this.callStores.set(root, created);
        return created;
    }
    /**
     * Build a hub from an untrusted configuration document.
     * @param raw - parsed JSON, typically from `config/models.json`.
     * @param options - everything except `config`.
     * @returns the hub.
     * @throws ModelHubError with `INVALID_DESCRIPTOR` when the document is malformed.
     */
    static fromConfig(raw, options = {}) {
        const parsed = parseModelCatalogConfig(raw, 'models.json');
        if (!parsed.ok) {
            throw new ModelHubError('INVALID_DESCRIPTOR', parsed.message, {
                issues: parsed.issues.map((issue) => ({ path: issue.path, message: issue.message })),
            });
        }
        return new ModelHub({ ...options, config: parsed.config });
    }
    /**
     * Add a listener for hub events.
     * @param listener - the listener.
     * @returns a disposer that removes it.
     */
    onEvent(listener) {
        this.listeners.add(listener);
        return () => {
            this.listeners.delete(listener);
        };
    }
    /**
     * Emit one event to every listener, containing listener failures.
     *
     * A listener that throws must not fail the invocation it was observing — the
     * hub's job is to serve work, and a broken log sink is not a reason to fail a
     * user's image generation.
     *
     * @param event - the event to emit.
     */
    emit(event) {
        for (const listener of this.listeners) {
            try {
                listener(event);
            }
            catch (error) {
                this.log(`hub: event listener threw: ${String(error)}`, { event: event.type });
            }
        }
    }
    // ───────────────────────────── discovery ─────────────────────────────
    /**
     * Every configured model with its live status.
     * @param options - `includeDisabled` lists models excluded from routing too.
     * @returns one view per model.
     */
    listModels(options = {}) {
        const includeDisabled = options.includeDisabled ?? true;
        return this.catalog
            .listModels()
            .filter((model) => includeDisabled || model.enabled)
            .map((model) => ({ model, status: this.runtime.getModelStatus(model.id) }));
    }
    /**
     * Look up one model with its live status.
     * @param modelId - the model id.
     * @returns the view.
     * @throws ModelHubError with `MODEL_NOT_FOUND`.
     */
    getModel(modelId) {
        return { model: this.catalog.requireModel(modelId), status: this.runtime.getModelStatus(modelId) };
    }
    /**
     * The live status of one model.
     * @param modelId - the model id.
     * @returns the status.
     * @throws ModelHubError with `MODEL_NOT_FOUND`.
     */
    getModelStatus(modelId) {
        return this.runtime.getModelStatus(modelId);
    }
    /**
     * Models that declare a capability, best candidate first.
     *
     * @param capability - the capability to search for.
     * @returns views of the matching models, including disabled ones so an operator
     *   can see why a capability is unserved.
     * @throws ModelHubError with `UNKNOWN_CAPABILITY` for an unknown name.
     */
    findModelsByCapability(capability) {
        if (!isCapability(capability)) {
            throw new ModelHubError('UNKNOWN_CAPABILITY', `"${capability}" is not a known capability. Known capabilities are listed by listCapabilities().`, { capability });
        }
        return this.catalog
            .findModelsByCapabilityIncludingDisabled(capability)
            .map((model) => ({ model, status: this.runtime.getModelStatus(model.id) }));
    }
    /**
     * Every capability at least one enabled model declares.
     * @returns capability views in vocabulary order.
     */
    listCapabilities() {
        return this.catalog.listCapabilities();
    }
    /**
     * Capabilities no enabled model declares, with the reason.
     * @returns one entry per unserved capability.
     */
    listUnservedCapabilities() {
        return this.catalog.listUnservedCapabilities();
    }
    /** The machine resources routing decisions are made against. */
    get machineProfile() {
        return this.catalog.machineProfile;
    }
    // ───────────────────────────── lifecycle ─────────────────────────────
    /**
     * Start a model.
     * @param modelId - the model id.
     * @returns whether a process was started, and the resulting health.
     * @throws ModelHubError with `START_FAILED`, `INSUFFICIENT_RESOURCES`, or `MODEL_NOT_FOUND`.
     */
    async startModel(modelId) {
        const started = Date.now();
        const result = await this.runtime.startModel(modelId);
        if (result.started) {
            this.emit({
                type: 'model/started',
                modelId,
                pid: this.runtime.getModelStatus(modelId).pid,
                coldStartMs: Date.now() - started,
            });
        }
        return {
            started: result.started,
            alreadyRunning: result.alreadyRunning,
            health: {
                healthy: result.health.healthy,
                ...(result.health.detail === undefined ? {} : { detail: result.health.detail }),
            },
        };
    }
    /**
     * Stop a model's process.
     * @param modelId - the model id.
     * @param options - `force` skips the graceful shutdown budget.
     * @returns whether anything was stopped.
     */
    async stopModel(modelId, options = {}) {
        const result = await this.runtime.stopModel(modelId, options);
        if (result.stopped)
            this.emit({ type: 'model/stopped', modelId });
        return result;
    }
    /**
     * Restart a model.
     * @param modelId - the model id.
     * @returns whether a process was started, and the resulting health.
     */
    async restartModel(modelId) {
        const started = Date.now();
        const result = await this.runtime.restartModel(modelId);
        if (result.started) {
            this.emit({
                type: 'model/started',
                modelId,
                pid: this.runtime.getModelStatus(modelId).pid,
                coldStartMs: Date.now() - started,
            });
        }
        return {
            started: result.started,
            alreadyRunning: result.alreadyRunning,
            health: {
                healthy: result.health.healthy,
                ...(result.health.detail === undefined ? {} : { detail: result.health.detail }),
            },
        };
    }
    /**
     * Run a health probe against one model and record the result.
     * @param modelId - the model id.
     * @returns the probe report.
     */
    async probeModel(modelId) {
        const report = await this.runtime.probeHealth(modelId);
        this.emit({
            type: 'model/health',
            modelId,
            healthy: report.healthy,
            ...(report.detail === undefined ? {} : { detail: report.detail }),
        });
        return {
            healthy: report.healthy,
            ...(report.detail === undefined ? {} : { detail: report.detail }),
            ...(report.latencyMs === undefined ? {} : { latencyMs: report.latencyMs }),
        };
    }
    /**
     * Probe every enabled model.
     * @returns how many are healthy.
     */
    async probeAll() {
        await this.runtime.probeAll();
        const statuses = this.catalog
            .listModels()
            .filter((model) => model.enabled)
            .map((model) => this.runtime.getModelStatus(model.id));
        return { healthy: statuses.filter((status) => status.availability === 'available').length, total: statuses.length };
    }
    // ───────────────────────────── invocation ────────────────────────────
    /**
     * Route a request without executing it.
     *
     * Exposed because "which model would you pick for this?" is a genuinely useful
     * question — for the agent to explain its plan, and for an operator to debug a
     * surprising choice — and because the answer must come from the same code path
     * that actually executes.
     *
     * @param request - the invocation request.
     * @returns the routing decision.
     */
    async route(request, options = {}) {
        this.assertUsableCapability(request.capability);
        const artifacts = this.storeFor(options.artifactRoot);
        const resolved = await resolveRequestInputs({ artifacts }, request);
        const decision = routeRequest({ catalog: this.catalog, runtime: this.runtime, artifacts }, resolved, this.routingPolicy);
        this.emit({
            type: 'routing/decided',
            capability: request.capability,
            modelId: decision.modelId,
            rationale: decision.rationale,
        });
        return decision;
    }
    /**
     * Invoke a capability. The router chooses the model; the caller never names one
     * (except to pin one explicitly for debugging).
     *
     * The sequence is: resolve inputs → route → (start if cold) → invoke → return
     * artifacts. On failure with fallback enabled, the next eligible candidate from
     * the *same* routing decision is tried, so a fallback is as reproducible as the
     * primary choice.
     *
     * @param request - the invocation request.
     * @param options - fallback controls.
     * @returns the invocation result.
     * @throws ModelHubError with a stable code when every candidate fails.
     */
    async invokeModel(request, options = {}) {
        this.assertUsableCapability(request.capability);
        const artifacts = this.storeFor(options.artifactRoot);
        const resolved = await resolveRequestInputs({ artifacts }, request);
        const decision = routeRequest({ catalog: this.catalog, runtime: this.runtime, artifacts }, resolved, this.routingPolicy);
        this.emit({
            type: 'routing/decided',
            capability: request.capability,
            modelId: decision.modelId,
            rationale: decision.rationale,
        });
        const ordering = [
            decision.modelId,
            ...decision.candidates.filter((candidate) => candidate.eligible && candidate.modelId !== decision.modelId).map((candidate) => candidate.modelId),
        ];
        const allowFallback = options.allowFallback ?? true;
        const maxAttempts = Math.max(1, options.maxAttempts ?? 3);
        const attempts = allowFallback ? ordering.slice(0, maxAttempts) : ordering.slice(0, 1);
        const failures = [];
        for (let index = 0; index < attempts.length; index += 1) {
            const modelId = attempts[index];
            if (modelId === undefined)
                continue;
            try {
                const result = await this.invokeOn(request, resolved, modelId, decision, artifacts);
                if (index > 0) {
                    const previous = attempts[index - 1];
                    this.emit({
                        type: 'invocation/fellback',
                        fromModelId: previous ?? 'unknown',
                        toModelId: modelId,
                        reason: failures[failures.length - 1]?.message ?? 'previous model failed',
                    });
                }
                return { ...result, decision };
            }
            catch (error) {
                const described = toHubError(error, 'INVOCATION_FAILED', { modelId });
                // A cancellation is the caller's decision, not a model failure: retrying
                // it on another model would be actively wrong.
                if (described.code === 'INVOCATION_ABORTED')
                    throw described;
                failures.push({ modelId, code: described.code, message: described.message });
                this.emit({
                    type: 'invocation/failed',
                    modelId,
                    capability: request.capability,
                    code: described.code,
                    message: described.message,
                });
                this.log(`hub: ${modelId} failed (${described.code}): ${described.message}`, {
                    modelId,
                    capability: request.capability,
                });
            }
        }
        const detail = failures.map((failure) => `  - ${failure.modelId}: [${failure.code}] ${failure.message}`).join('\n');
        throw new ModelHubError('INVOCATION_FAILED', `every candidate model failed for capability "${request.capability}" (${failures.length} attempt(s)):\n${detail}`, {
            capability: request.capability,
            failures: failures,
            routing: explainDecision(decision, true),
        });
    }
    /**
     * Execute one request against one specific model.
     *
     * Split out from {@link invokeModel} because it is also the path `startModel`
     * callers and adapter tests use: it does the gate → begin → invoke → end
     * accounting, and nothing else.
     *
     * @param request - the original request.
     * @param resolved - the resolved inputs.
     * @param modelId - the model to run.
     * @param decision - the routing decision, for logging.
     * @param artifacts - the store this call reads and writes through.
     * @returns the invocation result.
     */
    async invokeOn(request, resolved, modelId, decision, artifacts) {
        const model = this.catalog.requireModel(modelId);
        const adapter = this.adapters.get(model.adapter);
        if (adapter === undefined) {
            throw new ModelHubError('UNSUPPORTED_OPERATION', `model "${modelId}" uses adapter kind "${model.adapter}", which is not registered`, { modelId, adapter: model.adapter });
        }
        const support = adapter.supports(model);
        if (!support.ok) {
            throw new ModelHubError('UNSUPPORTED_OPERATION', `model "${modelId}" cannot run: ${support.reason}`, {
                modelId,
                reason: support.reason,
            });
        }
        const gate = await this.runtime.ensureReady(modelId);
        const started = Date.now();
        this.runtime.beginInvocation(modelId);
        this.emit({
            type: 'invocation/started',
            modelId,
            capability: request.capability,
            coldStart: gate.coldStart,
        });
        try {
            const output = await adapter.invoke({
                model,
                capability: request.capability,
                ...(request.prompt === undefined ? {} : { prompt: request.prompt }),
                inputs: resolved.inputs,
                options: request.options ?? {},
                artifacts,
                signal: request.signal ?? new AbortController().signal,
                log: this.runtime.adapterLogger(modelId),
            });
            const durationMs = Date.now() - started;
            // The structured value is declared as lossless JSON so it can ride a tool
            // result into the session log. An adapter that returns a Date, a function,
            // or a cyclic object is broken, and failing here says so at the adapter
            // boundary instead of letting the registry drop fields later.
            const structured = output.value;
            if (structured !== undefined && !isLosslessJson(structured)) {
                throw new ModelHubError('INVOCATION_FAILED', `adapter for model "${modelId}" returned a structured value that is not lossless JSON`, { modelId, adapter: model.adapter });
            }
            // The guard above establishes the runtime property the declared type
            // asserts; the cast records that check rather than dodging it.
            const value = structured;
            const result = {
                modelId,
                capability: request.capability,
                outputs: output.outputs,
                ...(value === undefined ? {} : { value }),
                durationMs,
                coldStart: gate.coldStart,
            };
            this.emit({
                type: 'invocation/succeeded',
                modelId,
                capability: request.capability,
                durationMs,
                outputCount: output.outputs.length,
            });
            this.log(`hub: ${request.capability} served by ${modelId} in ${durationMs} ms ` +
                `(${output.outputs.length} output(s)${gate.coldStart ? ', cold start' : ''})`, { modelId, capability: request.capability, rationale: decision.rationale });
            return result;
        }
        finally {
            this.runtime.endInvocation(modelId);
        }
    }
    /**
     * Serve a capability, or explain why it cannot be served, without throwing.
     *
     * This is the shape a model-facing tool wants: a tool that throws and a tool
     * that returns a failure are different to the agent, and for a capability that
     * is simply not deployed, a clear refusal is more useful than an exception.
     *
     * @param request - the invocation request.
     * @param options - fallback controls.
     * @returns either the result or a structured failure.
     */
    async tryInvokeModel(request, options = {}) {
        try {
            const result = await this.invokeModel(request, options);
            const { decision, ...invocation } = result;
            return { ok: true, result: invocation, decision };
        }
        catch (error) {
            const hubError = toHubError(error, 'INVOCATION_FAILED');
            return { ok: false, error: hubError };
        }
    }
    // ───────────────────────────── artifacts ─────────────────────────────
    /**
     * Look up an artifact by id.
     * @param artifactId - the artifact id.
     * @returns the artifact, or `undefined`.
     */
    async getArtifact(artifactId, artifactRoot) {
        return this.storeFor(artifactRoot).get(artifactId);
    }
    /**
     * List stored artifacts, newest first.
     * @param limit - maximum number to return. Defaults to 20.
     * @returns artifact references.
     */
    async listArtifacts(limit = 20, artifactRoot) {
        return this.storeFor(artifactRoot).list(limit);
    }
    // ───────────────────────────── lifecycle of the hub ──────────────────
    /**
     * Stop background timers and shut down every process the hub owns.
     *
     * Safe to call more than once. Every owned process is given its declared
     * shutdown budget; a failure on one does not prevent stopping the others.
     */
    async dispose() {
        if (this.disposed)
            return;
        this.disposed = true;
        await this.runtime.dispose();
        this.listeners.clear();
    }
    /**
     * Reject an unknown capability before any work happens.
     * @param capability - the requested capability.
     * @throws ModelHubError with `UNKNOWN_CAPABILITY`.
     */
    assertUsableCapability(capability) {
        if (!isCapability(capability)) {
            throw new ModelHubError('UNKNOWN_CAPABILITY', `"${capability}" is not a capability this hub knows. ` +
                'Call listCapabilities() for the vocabulary this deployment supports.', { capability, known: this.catalog.listCapabilities().map((view) => view.capability) });
        }
    }
}
/**
 * The hub library's fallback artifact root: `artifacts/` under the current
 * working directory.
 *
 * This is what a hub built *without* `artifactRoot` and called *without*
 * {@link InvokeOptions.artifactRoot} uses. For a process that owns its working
 * directory — a CLI, a test, a script — that is exactly right.
 *
 * For a long-lived host it is the wrong question: `dsh web` is one process
 * serving many sessions, its working directory is whatever it was launched from,
 * and a caller's notion of "here" lives on the session. DSH's plugin layer
 * therefore resolves the calling session's workspace per call and passes it as
 * {@link InvokeOptions.artifactRoot}; this remains the last-resort default for
 * every other caller.
 *
 * @returns an absolute path.
 */
export function defaultArtifactRoot() {
    const fromEnv = process.env['AIMH_ARTIFACT_ROOT'];
    if (fromEnv !== undefined && fromEnv.trim().length > 0)
        return fromEnv;
    return `${process.cwd()}/artifacts`;
}
/** Re-export so callers can construct a hub with a silent logger without another import. */
export { silentLogger };
