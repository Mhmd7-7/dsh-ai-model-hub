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
import { dirname } from 'node:path';
import { LocalArtifactStore } from "./artifacts/local-store.js";
import { AdapterRegistry } from "./adapters/types.js";
import { silentLogger } from "./adapters/types.js";
import { createMockAdapter } from "./adapters/mock.js";
import { createOpenAiCompatibleAdapter } from "./adapters/openai.js";
import { createHttpJsonAdapter } from "./adapters/http-json.js";
import { createComfyUiAdapter } from "./adapters/comfyui.js";
import { createThreeDAdapter } from "./adapters/three-d.js";
import { isCapability } from "./catalog/capabilities.js";
import { parseModelCatalogConfig } from "./catalog/descriptor.js";
import { ModelCatalog } from "./catalog/registry.js";
import { createComfyUiDiscoverer } from "./discovery/comfyui.js";
import { createA1111Discoverer } from "./discovery/a1111.js";
import { createOllamaDiscoverer } from "./discovery/ollama.js";
import { createThreeDDiscoverer } from "./discovery/three-d.js";
import { DiscoveryRegistry, mergeCatalogConfig } from "./discovery/types.js";
import { reserveResources } from "./types.js";
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
    /**
     * The engine→discoverer map, or `undefined` when runtime discovery is off.
     *
     * `undefined` is the honest representation of "off", not an empty registry:
     * with discovery disabled the hub never constructs one, so no code path can
     * accidentally reach an engine.
     */
    discovery;
    /**
     * The catalog document exactly as it was supplied, *before* any discovered
     * model was merged in.
     *
     * Kept because a refresh must re-merge from the static half rather than append
     * to the merged half: otherwise a checkpoint deleted from disk would live on in
     * the catalog forever, and repeated refreshes would grow it without bound.
     */
    staticConfig;
    /**
     * The in-flight or most recent discovery pass.
     *
     * Held so concurrent refreshes share one round of network traffic and so a
     * caller can await the pre-warm the constructor started.
     */
    discoveryPass;
    listeners = new Set();
    log;
    ownsArtifacts;
    probeOptions;
    resourceTtlMs;
    /**
     * The directory the catalog was read from, when it was read from a file.
     *
     * Held resolved, because every use is "resolve a catalog-relative path against
     * this", and the adapters that need it should not each re-derive it from
     * {@link ModelHubOptions.catalogPath}.
     */
    catalogDir;
    disposed = false;
    /**
     * The most recent probe of this machine, when one has run.
     *
     * Held here rather than read back out of the catalog because the two answer
     * different questions at different times: the catalog holds the figures routing
     * currently uses, while this holds the *measurement*, with its evidence and its
     * timestamp, whether or not it was allowed to change routing.
     */
    resources;
    /** The in-flight probe, so concurrent callers share one `nvidia-smi` run. */
    resourcePass;
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
        this.staticConfig = options.config;
        this.probeOptions = resolveProbeOptions(options);
        this.resourceTtlMs = options.resourceTtlMs ?? DEFAULT_RESOURCE_TTL_MS;
        this.catalogDir = options.catalogPath === undefined ? undefined : dirname(options.catalogPath);
        // Discovery is built before the catalog so the very first pass can be
        // started below, but its results are *not* folded in here: the constructor
        // cannot await an engine. `mergeModelsWithConfig` is exported so a caller
        // that can await (a CLI, a script, a test) can merge first and construct the
        // catalog with the discovered models already in `config.models` — which is
        // the path `ModelHub.fromConfigAndDiscovery` takes.
        this.discovery =
            options.discoveryRegistry ??
                (options.discoverModels === true
                    ? new DiscoveryRegistry([...(options.discoverers ?? defaultDiscoverers()), ...(options.extraDiscoverers ?? [])], {
                        ...(options.discoveryTtlMs === undefined ? {} : { ttlMs: options.discoveryTtlMs }),
                        ...(options.discoveryTimeoutMs === undefined ? {} : { timeoutMs: options.discoveryTimeoutMs }),
                        log: (message, fields) => this.log(message, fields),
                    })
                    : undefined);
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
        // `/sdapi/v1/txt2img` server (A1111, Forge, stable-diffusion.cpp), the
        // graph-queue adapter for ComfyUI, and the 3D adapter that reaches every local
        // 3D-generation server (TRELLIS, Hunyuan3D, Stable Fast 3D, TripoSR, …) over
        // its Gradio queue API or a JSON route. A deployment adds more through
        // `extraAdapters`, or replaces the whole set with `adapters`.
        const defaultAdapters = options.adapters ?? [
            createMockAdapter(),
            createOpenAiCompatibleAdapter(),
            createHttpJsonAdapter(),
            createComfyUiAdapter(),
            createThreeDAdapter(),
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
        // Probe this machine's resources in the background, for the same reason
        // discovery is pre-warmed: the constructor cannot await a subprocess. Until
        // it lands the catalog reports "not probed", whose resource check passes
        // everything through — deliberately permissive, because refusing every model
        // for the first few hundred milliseconds of a process's life would be worse
        // than the OOM the probe exists to prevent.
        if (this.probeOptions !== undefined) {
            void this.refreshResources().catch((error) => {
                this.log(`hub: resource probe failed: ${error instanceof Error ? error.message : String(error)}`);
            });
        }
        // Pre-warm discovery in the background. The constructor is synchronous by
        // contract — a plugin's `apply` cannot await it — so the catalog starts with
        // the static models and grows a moment later. A caller that needs discovered
        // models present before the first invocation should either await
        // `refreshDiscovery()` or use `ModelHub.fromConfigAndDiscovery`, which merges
        // before the catalog exists at all.
        if (this.discovery !== undefined) {
            void this.refreshDiscovery().catch((error) => {
                this.log(`hub: discovery pre-warm failed: ${error instanceof Error ? error.message : String(error)}`);
            });
        }
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
     * Build a hub whose catalog already contains what the configured engines
     * report — discovered models merged in *before* the catalog exists.
     *
     * This is the "merge step before `new ModelCatalog(config)`" path, in its
     * strongest form: a caller that can await gets a hub whose very first routing
     * decision can select a discovered model, with no background pass to race. The
     * plugin cannot use it, because `apply` is synchronous; a CLI, a script, or a
     * test can and should.
     *
     * @param raw - parsed JSON, typically from `config/models.json`.
     * @param options - everything except `config`; `discoverModels` is implied.
     * @returns the hub, plus the discovery result for diagnostics.
     * @throws ModelHubError with `INVALID_DESCRIPTOR` when the document is malformed.
     */
    static async fromConfigAndDiscovery(raw, options = {}) {
        const parsed = parseModelCatalogConfig(raw, 'models.json');
        if (!parsed.ok) {
            throw new ModelHubError('INVALID_DESCRIPTOR', parsed.message, {
                issues: parsed.issues.map((issue) => ({ path: issue.path, message: issue.message })),
            });
        }
        const registry = new DiscoveryRegistry([...(options.discoverers ?? defaultDiscoverers()), ...(options.extraDiscoverers ?? [])], {
            ...(options.discoveryTtlMs === undefined ? {} : { ttlMs: options.discoveryTtlMs }),
            ...(options.discoveryTimeoutMs === undefined ? {} : { timeoutMs: options.discoveryTimeoutMs }),
            ...(options.log === undefined ? {} : { log: (message, fields) => options.log?.(message, fields) }),
        });
        const discovery = await registry.generate(parsed.config.hosts ?? []);
        const hub = new ModelHub({
            ...options,
            config: mergeCatalogConfig(parsed.config, discovery.descriptors),
            // The merged catalog is already in `config`, and the registry has already
            // run its first pass. Handing the registry over rather than re-creating it
            // keeps its cache warm for a later `refreshDiscovery()` and stops a second
            // pass from refetching what this one just read.
            discoveryRegistry: registry,
        });
        return { hub, discovery };
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
     * Re-read every configured engine and republish the catalog.
     *
     * This is the manual refresh path: an operator who just pulled an Ollama model
     * calls it instead of waiting out the cache TTL. It is safe to call
     * concurrently — passes are deduplicated per host — and it is a no-op that
     * reports `cached: true` when runtime discovery is disabled or when nothing is
     * stale and `force` is not set.
     *
     * The merged model list is always `static models + this pass's discovered
     * models`, so a checkpoint that has been deleted stops being routable rather
     * than lingering. Hosts and the machine profile are untouched.
     *
     * @param options - `force` bypasses the cache even when a pass is fresh.
     * @returns the discovery result, including per-host warnings.
     */
    async refreshDiscovery(options = {}) {
        const registry = this.discovery;
        if (registry === undefined) {
            return { descriptors: [], warnings: [], cached: true, durationMs: 0, hostIds: [] };
        }
        const pass = (async () => {
            const result = await registry.generate(this.staticConfig.hosts ?? [], options.force === true ? { refresh: true } : {});
            if (result.cached)
                return result;
            for (const warning of result.warnings) {
                this.log(`hub: discovery warning: ${warning.hostId} (${warning.engine}): ${warning.message}`, {
                    hostId: warning.hostId,
                });
            }
            const before = new Set(this.catalog.listModelIds());
            this.catalog.replaceModels(mergeCatalogConfig(this.staticConfig, result.descriptors).models);
            // The runtime manager seeds its per-model state when it is constructed, and
            // discovery can add a model long after that. Without this, a discovered
            // model has no runtime state and `getModelStatus` throws for it — which
            // takes `listModels`, the settings page, and `explain_routing` down with
            // it, over a reason that has nothing to do with any of them.
            this.runtime.syncCatalog();
            const after = new Set(this.catalog.listModelIds());
            const added = [...after].filter((id) => !before.has(id));
            const removed = [...before].filter((id) => !after.has(id));
            if (added.length > 0 || removed.length > 0) {
                this.log(`hub: discovery updated the catalog (${added.length} added, ${removed.length} removed)`, { added, removed, hosts: result.hostIds });
            }
            return result;
        })();
        this.discoveryPass = pass;
        try {
            return await pass;
        }
        finally {
            if (this.discoveryPass === pass)
                this.discoveryPass = undefined;
        }
    }
    /**
     * The most recent discovery pass, when one is in flight or just finished.
     *
     * Exposed so a caller can await the background pre-warm the constructor
     * started without guessing at a delay.
     */
    get pendingDiscovery() {
        return this.discoveryPass;
    }
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
    // ───────────────────────────── machine resources ─────────────────────
    /**
     * The machine resources routing decisions are made against.
     *
     * This is the profile the router reads, so it includes the resource headroom
     * figure when one has been probed. It is what `explain_routing` and
     * `get_model_status` show an operator who is asking "why was that model
     * rejected?".
     */
    get machineProfile() {
        return this.catalog.machineProfile;
    }
    /**
     * The last measurement of this machine, with the evidence behind it.
     *
     * Distinct from {@link machineProfile}: that is what routing currently uses,
     * while this is what was *measured* — including the case where the two differ
     * because the deployment supplied a profile by hand or turned probing off. The
     * evidence lines are what make a surprising resource decision diagnosable
     * ("only 5.2 GiB free") instead of mysterious.
     *
     * @returns the snapshot, or `undefined` when nothing has been probed.
     */
    resourceSnapshot() {
        return this.resources;
    }
    /**
     * What the machine has left once the running models are accounted for.
     *
     * Reported figures are the probe's, minus the declared footprint of every
     * model whose engine is currently resident. This is the number an operator
     * should compare a model's requirements against, and it is the number the
     * router uses for a model that is already running.
     *
     * @returns the current profile with its available figures reduced, plus what
     *   was subtracted.
     */
    availableResources() {
        const reserved = { vramGb: 0, ramGb: 0 };
        const residentModelIds = [];
        for (const model of this.catalog.listModels()) {
            const availability = this.runtime.checkAvailability(model.id);
            if (availability !== 'available' && availability !== 'unhealthy' && availability !== 'starting')
                continue;
            residentModelIds.push(model.id);
            reserved.vramGb += model.resources.vramGb;
            reserved.ramGb += model.resources.ramGb;
        }
        return {
            profile: reserveResources(this.catalog.machineProfile, reserved),
            reserved,
            residentModelIds,
        };
    }
    /**
     * Re-measure this machine and republish the catalog's profile.
     *
     * Safe to call at any time and safe to call concurrently — passes are shared.
     * A failed probe leaves the previous profile in place and reports the problem,
     * because losing a measurement must never cost the ability to route.
     *
     * @returns the profile now in force.
     */
    async refreshResources() {
        const existing = this.resourcePass;
        if (existing !== undefined)
            return existing;
        const pass = (async () => {
            const probed = await probeMachine({
                ...(this.probeOptions ?? {}),
                // The deployment's own policy travels down so a narrowed allowlist is
                // respected — the probe widens it by exactly one diagnostic binary.
                policy: this.executionPolicy,
            });
            this.resources = { profile: probed.profile, evidence: probed.evidence };
            // The catalog owns the profile the router reads, so publishing is a
            // mutation of the catalog rather than a second source of truth. With
            // probing disabled the catalog keeps whatever it was constructed with.
            if (this.probeOptions !== undefined) {
                this.catalog.replaceMachineProfile(probed.profile);
                // The catalog was built before these numbers existed, so every model's
                // resource verdict was made against "not probed" — which passes
                // everything. Re-checking here is what turns the measurement into a
                // decision instead of a fact nothing acts on.
                for (const model of this.catalog.listModels())
                    this.runtime.revalidateResources(model.id);
            }
            this.log(`hub: machine resources — ${probed.profile.vramGb} GiB VRAM` +
                `${probed.profile.availableVramGb === undefined ? '' : ` (${probed.profile.availableVramGb} GiB free)`}, ` +
                `${probed.profile.ramGb} GiB RAM` +
                `${probed.profile.availableRamGb === undefined ? '' : ` (${probed.profile.availableRamGb} GiB free)`}`, { evidence: probed.evidence });
            return probed.profile;
        })();
        this.resourcePass = pass;
        try {
            return await pass;
        }
        finally {
            if (this.resourcePass === pass)
                this.resourcePass = undefined;
        }
    }
    /**
     * Re-measure if this deployment probes and the last measurement is stale.
     *
     * Called at the top of every routing decision, which is what makes the probe a
     * *routing* input rather than a startup report. It is deliberately quiet about
     * failure: a machine whose GPU query times out must still route, using the last
     * numbers it had, because "I could not measure the GPU" is not a reason to
     * refuse every model on the machine.
     */
    async freshenResources() {
        if (this.probeOptions === undefined)
            return;
        try {
            await this.ensureResourcesFresh(this.resourceTtlMs);
        }
        catch (error) {
            this.log(`hub: resource probe failed, routing against the last measurement: ${String(error)}`);
        }
    }
    /**
     * Re-measure the machine if the last measurement is older than `maxAgeMs`.
     *
     * The seam a long-lived host uses: resources change as other applications come
     * and go, and a routing decision made against a probe from yesterday is barely
     * better than no probe at all. Cheap when the answer is "still fresh" — it is
     * one timestamp comparison.
     *
     * @param maxAgeMs - how stale a measurement may be. Defaults to 60000.
     * @returns the profile now in force.
     */
    async ensureResourcesFresh(maxAgeMs = 60_000) {
        const probedAt = this.resources?.profile.probedAt;
        if (probedAt !== undefined && Date.now() - probedAt < Math.max(0, maxAgeMs)) {
            return this.catalog.machineProfile;
        }
        return this.refreshResources();
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
        await this.freshenResources();
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
        await this.freshenResources();
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
                const result = await this.invokeWithResourceRelease(artifacts, modelId, () => this.invokeOn(request, resolved, modelId, decision, artifacts));
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
                // The caller's budget is forwarded as its own field rather than folded
                // into `options`, because it is a property of *this call* and not a
                // capability setting: an adapter uses it to abort its own in-flight work,
                // which an option bag cannot express.
                ...(request.timeoutMs === undefined ? {} : { timeoutMs: request.timeoutMs }),
                // Where the catalog lives, so an adapter can resolve a path the catalog
                // wrote relative to itself. Not `process.cwd()`: a long-lived host's
                // working directory is wherever its launcher stood, not where the
                // deployment keeps its files.
                ...(this.catalogDir === undefined ? {} : { catalogDir: this.catalogDir }),
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
     * Run one model attempt, releasing this machine's resource measurement after it.
     *
     * A generation is the only thing here that genuinely changes how much memory is
     * free: a 3D run loads several gigabytes that were not resident a moment ago,
     * and a big text model's KV cache grows while it answers. That means the
     * measurement taken before a generation is the *wrong* basis for the next
     * routing decision — it describes the machine as it was, not as it is now.
     *
     * So the cache is invalidated here rather than merely aged. It is not re-probed
     * inline, because a probe is a subprocess and an invocation's own latencies are
     * the user's; the next decision that needs resources calls
     * {@link ensureResourcesFresh} and pays for it then.
     *
     * Failure is irrelevant to this bookkeeping: whether the model succeeded, timed
     * out, or crashed, it has still moved the machine's memory, so the release
     * happens on both paths.
     *
     * @param artifacts - the store this attempt writes through (unused, kept for
     *   call-site symmetry with {@link invokeOn}).
     * @param modelId - the model being attempted, for diagnostics.
     * @param run - the attempt.
     * @returns the attempt's result.
     */
    async invokeWithResourceRelease(artifacts, modelId, run) {
        void artifacts;
        try {
            return await run();
        }
        finally {
            this.invalidateResources(modelId);
        }
    }
    /**
     * Mark the machine measurement as taken before the last generation.
     *
     * Implemented by rewinding `probedAt` rather than by clearing the snapshot, so
     * {@link availableResources} keeps reporting the last *known* figures — an
     * operator asking "what is free?" is better served by a stale number with a
     * timestamp than by no answer — while {@link ensureResourcesFresh} knows to
     * re-measure before it decides anything.
     *
     * @param modelId - why the measurement was invalidated, for diagnostics.
     */
    invalidateResources(modelId) {
        const current = this.resources;
        if (current === undefined)
            return;
        this.resources = { profile: { ...current.profile, probedAt: 0 }, evidence: current.evidence };
        this.log(`hub: machine resources will be re-probed before the next decision (after ${modelId})`, { modelId });
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
 * How long a machine measurement stays fresh enough to route against.
 *
 * Thirty seconds is a compromise between two failure modes: measuring before
 * every decision adds an `nvidia-smi` subprocess to each one, while measuring too
 * rarely means a model another application loaded a minute ago is invisible and
 * routing walks straight into an out-of-memory crash.
 */
const DEFAULT_RESOURCE_TTL_MS = 30_000;
/**
 * Decide whether this hub probes the machine, and how.
 *
 * Three inputs, in precedence order:
 *
 * 1. An explicit `probeResources: false` turns probing off outright.
 * 2. A hand-supplied `machine` profile also turns it off: an operator or a test
 *    that states what this machine has must not be second-guessed by a
 *    subprocess, or a deterministic test would race a real `nvidia-smi`.
 * 3. Otherwise the default is on, with any `MachineProbeOptions` the deployment
 *    passed passed through to the probe itself.
 *
 * @param options - the hub's construction options.
 * @returns the probe options, or `undefined` when nothing should be probed.
 */
function resolveProbeOptions(options) {
    if (options.probeResources === false)
        return undefined;
    if (options.machine !== undefined)
        return undefined;
    if (options.probeResources === true || options.probeResources === undefined)
        return {};
    return options.probeResources;
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
/**
 * The discoverers a hub gets when runtime discovery is enabled and the caller
 * names none.
 *
 * This is the exact analogue of the default adapter set in the constructor: one
 * entry per engine family the hub knows how to introspect, and nothing about any
 * particular model. Listing them here means a new engine becomes usable by
 * adding one file, exactly as a new adapter kind does.
 *
 * @returns the built-in discoverers.
 */
function defaultDiscoverers() {
    return [createOllamaDiscoverer(), createComfyUiDiscoverer(), createA1111Discoverer(), createThreeDDiscoverer()];
}
/** Re-export so callers can construct a hub with a silent logger without another import. */
export { silentLogger };
