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

import type { Artifact, ArtifactStore } from './artifacts/types.ts';
import { LocalArtifactStore } from './artifacts/local-store.ts';
import { AdapterRegistry } from './adapters/types.ts';
import { silentLogger } from './adapters/types.ts';
import type { ModelAdapter } from './adapters/types.ts';
import { createMockAdapter } from './adapters/mock.ts';
import { createOpenAiCompatibleAdapter } from './adapters/openai.ts';
import { createHttpJsonAdapter } from './adapters/http-json.ts';
import { createComfyUiAdapter } from './adapters/comfyui.ts';
import type { Capability } from './catalog/capabilities.ts';
import { isCapability } from './catalog/capabilities.ts';
import type { ModelCatalogConfig } from './catalog/descriptor.ts';
import { parseModelCatalogConfig } from './catalog/descriptor.ts';
import type { CapabilityView } from './catalog/registry.ts';
import { ModelCatalog } from './catalog/registry.ts';
import { createComfyUiDiscoverer } from './discovery/comfyui.ts';
import { createA1111Discoverer } from './discovery/a1111.ts';
import { createOllamaDiscoverer } from './discovery/ollama.ts';
import { DiscoveryRegistry, mergeCatalogConfig } from './discovery/types.ts';
import type { DiscoveryResult, HostDiscoverer } from './discovery/types.ts';
import type { MachineProfile } from './types.ts';
import type { AvailabilityState, InvocationRequest, InvocationResult, ModelRuntimeStatus, ModelView, RoutingDecision } from './types.ts';
import { ModelHubError, toHubError } from './errors.ts';
import { probeMachine } from './machine.ts';
import type { ExecutionPolicy } from './util/process.ts';
import { DEFAULT_EXECUTION_POLICY } from './util/process.ts';
import { RuntimeManager } from './runtime/manager.ts';
import type { RoutingPolicy } from './router/router.ts';
import { DEFAULT_ROUTING_POLICY, explainDecision, resolveRequestInputs, routeRequest } from './router/router.ts';
import { isLosslessJson } from './util/validate.ts';
import type { JsonValue } from './util/validate.ts';

/**
 * A hub lifecycle event.
 *
 * Events exist so a UI, a log file, or a test can observe what the hub did
 * *without* the hub knowing about any of them. The DSH plugin uses them to write
 * a single coherent diagnostic line per invocation; a future web panel would use
 * them to render model activity. Adding a listener never changes control flow.
 */
export type HubEvent =
  | { readonly type: 'model/started'; readonly modelId: string; readonly pid: number | undefined; readonly coldStartMs: number }
  | { readonly type: 'model/stopped'; readonly modelId: string }
  | { readonly type: 'model/health'; readonly modelId: string; readonly healthy: boolean; readonly detail?: string }
  | { readonly type: 'routing/decided'; readonly capability: Capability; readonly modelId: string; readonly rationale: string }
  | { readonly type: 'invocation/started'; readonly modelId: string; readonly capability: Capability; readonly coldStart: boolean }
  | { readonly type: 'invocation/succeeded'; readonly modelId: string; readonly capability: Capability; readonly durationMs: number; readonly outputCount: number }
  | { readonly type: 'invocation/failed'; readonly modelId: string; readonly capability: Capability; readonly code: string; readonly message: string }
  | { readonly type: 'invocation/fellback'; readonly fromModelId: string; readonly toModelId: string; readonly reason: string };

/** A listener for {@link HubEvent}. */
export type HubEventListener = (event: HubEvent) => void;

/** Construction options for {@link ModelHub}. */
export interface ModelHubOptions {
  /** The validated catalog document. */
  readonly config: ModelCatalogConfig;
  /** Where artifacts are stored. Defaults to a subdirectory of the working directory. */
  readonly artifactRoot?: string;
  /** Adapters to register. Defaults to the built-in mock adapter only. */
  readonly adapters?: readonly ModelAdapter[];
  /** Extra adapters appended to the defaults. */
  readonly extraAdapters?: readonly ModelAdapter[];
  /** Process-execution policy. Defaults to the shipping allowlist policy. */
  readonly executionPolicy?: ExecutionPolicy;
  /** Routing policy. Defaults to {@link DEFAULT_ROUTING_POLICY}. */
  readonly routingPolicy?: RoutingPolicy;
  /** Machine profile override, for tests and for operators who know better. */
  readonly machine?: MachineProfile;
  /** Diagnostic sink. Defaults to silence. */
  readonly log?: (message: string, fields?: Readonly<Record<string, unknown>>) => void;
  /** How often the runtime re-probes health. `0` disables probing. Defaults to 30000. */
  readonly healthIntervalMs?: number;
  /** How often the runtime sweeps idle timeouts. `0` disables sweeping. Defaults to 15000. */
  readonly idleSweepIntervalMs?: number;
  /**
   * Whether to start the runtime's background timers. Defaults to true in
   * production usage; tests and one-shot CLI runs set false.
   */
  readonly manageTimers?: boolean;
  /**
   * Whether to augment the catalog with models discovered from the configured
   * engines at runtime.
   *
   * **Off by default, and deliberately so.** With it off, discovery does not run
   * at all: no HTTP request is made to any engine, and the catalog is exactly the
   * document on disk. Turning it on adds descriptors for what each engine
   * reports it currently has — a pulled Ollama model, a checkpoint dropped into
   * ComfyUI's models directory — without a JSON edit.
   *
   * Static configuration always wins: a discovered descriptor whose id a static
   * entry already claims is dropped by {@link mergeModelsWithConfig} before the
   * catalog is built, and every static entry precedes every discovered one. See
   * `src/discovery/types.ts` for why that ordering is enforced by the merge
   * rather than by the catalog.
   *
   * When enabled, discovery is kicked off in the background at construction —
   * the constructor itself is synchronous and cannot await an engine — and the
   * catalog is rebuilt once the first pass lands. {@link ModelHub.refreshDiscovery}
   * forces a pass for a caller that will not wait out the cache TTL.
   */
  readonly discoverModels?: boolean;
  /** Discoverers to use instead of the built-in engine set. */
  readonly discoverers?: readonly HostDiscoverer[];
  /** Extra discoverers appended to the built-in set. */
  readonly extraDiscoverers?: readonly HostDiscoverer[];
  /**
   * A registry to use instead of building one.
   *
   * Supplied by {@link ModelHub.fromConfigAndDiscovery}, which has already run
   * the first pass to merge its results into `config.models`. A caller that
   * passes one owns it; providing both this and {@link discoverers} is a
   * programming error rather than a merge, because there is no sensible
   * precedence between "the registry I built" and "the discoverers I named".
   */
  readonly discoveryRegistry?: DiscoveryRegistry;
  /** How long one discovery pass stays cached, in milliseconds. Defaults to 60000. */
  readonly discoveryTtlMs?: number;
  /** Budget for one engine's discovery pass, in milliseconds. Defaults to 5000. */
  readonly discoveryTimeoutMs?: number;
}

/** Optional per-call controls for {@link ModelHub.invokeModel}. */
export interface InvokeOptions {
  /**
   * Whether a failed invocation may be retried on the next-best eligible model.
   *
   * Defaults to true. This is what turns "the 3D model is installed but broken"
   * into a successful workflow on a fallback model instead of a dead end — and
   * because the candidate order comes from the same deterministic routing
   * decision, the fallback is reproducible too.
   */
  readonly allowFallback?: boolean;
  /** Maximum number of models to attempt, including the first. Defaults to 3. */
  readonly maxAttempts?: number;
  /**
   * Read and write this call's artifacts under `root` instead of the hub's own
   * store, which stays the default for every call that omits it.
   *
   * This is what lets a long-lived host give each session its own artifact
   * directory: the hub is constructed once, before any session exists, so the
   * workspace is only knowable per call. The store for a root is created on first
   * use and reused afterwards, so passing the same root on every call from one
   * session costs a map lookup.
   */
  readonly artifactRoot?: string;
}

/**
 * The hub: catalog + router + runtime + adapters + artifacts behind one API.
 *
 * Construct it once per process and share it. Its only mutable state is the
 * runtime manager's model lifecycle, which is internally serialized.
 */
export class ModelHub {
  /** The capability catalog. */
  readonly catalog: ModelCatalog;
  /** The artifact store. */
  readonly artifacts: ArtifactStore;
  /** The runtime manager. */
  readonly runtime: RuntimeManager;
  /** The adapter registry, so a plugin can register an engine adapter later. */
  readonly adapters: AdapterRegistry;
  /** The routing policy in force. */
  readonly routingPolicy: RoutingPolicy;
  /** The execution policy in force, for diagnostics. */
  readonly executionPolicy: ExecutionPolicy;

  /**
   * The engine→discoverer map, or `undefined` when runtime discovery is off.
   *
   * `undefined` is the honest representation of "off", not an empty registry:
   * with discovery disabled the hub never constructs one, so no code path can
   * accidentally reach an engine.
   */
  readonly discovery: DiscoveryRegistry | undefined;

  /**
   * The catalog document exactly as it was supplied, *before* any discovered
   * model was merged in.
   *
   * Kept because a refresh must re-merge from the static half rather than append
   * to the merged half: otherwise a checkpoint deleted from disk would live on in
   * the catalog forever, and repeated refreshes would grow it without bound.
   */
  private readonly staticConfig: ModelCatalogConfig;

  /**
   * The in-flight or most recent discovery pass.
   *
   * Held so concurrent refreshes share one round of network traffic and so a
   * caller can await the pre-warm the constructor started.
   */
  private discoveryPass: Promise<DiscoveryResult> | undefined;

  private readonly listeners = new Set<HubEventListener>();
  private readonly log: (message: string, fields?: Readonly<Record<string, unknown>>) => void;
  private readonly ownsArtifacts: boolean;
  private disposed = false;

  /**
   * Stores for the per-call roots {@link InvokeOptions.artifactRoot} names, keyed
   * by the root as given. The hub's own {@link artifacts} store is not in here.
   */
  private readonly callStores = new Map<string, ArtifactStore>();

  /**
   * @param options - configuration, adapters, and policies.
   */
  constructor(options: ModelHubOptions) {
    this.log = options.log ?? ((): void => {});
    this.routingPolicy = options.routingPolicy ?? DEFAULT_ROUTING_POLICY;
    this.executionPolicy = options.executionPolicy ?? DEFAULT_EXECUTION_POLICY;
    this.staticConfig = options.config;

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
    // `/sdapi/v1/txt2img` server (A1111, Forge, stable-diffusion.cpp), and the
    // graph-queue adapter for ComfyUI. A deployment adds more through
    // `extraAdapters`, or replaces the whole set with `adapters`.
    const defaultAdapters: readonly ModelAdapter[] =
      options.adapters ?? [
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

    if (options.manageTimers !== false) this.runtime.start();

    // Pre-warm discovery in the background. The constructor is synchronous by
    // contract — a plugin's `apply` cannot await it — so the catalog starts with
    // the static models and grows a moment later. A caller that needs discovered
    // models present before the first invocation should either await
    // `refreshDiscovery()` or use `ModelHub.fromConfigAndDiscovery`, which merges
    // before the catalog exists at all.
    if (this.discovery !== undefined) {
      void this.refreshDiscovery().catch((error: unknown) => {
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
  private storeFor(root: string | undefined): ArtifactStore {
    if (root === undefined) return this.artifacts;
    const existing = this.callStores.get(root);
    if (existing !== undefined) return existing;
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
  static fromConfig(
    raw: unknown,
    options: Omit<ModelHubOptions, 'config'> = {},
  ): ModelHub {
    const parsed = parseModelCatalogConfig(raw, 'models.json');
    if (!parsed.ok) {
      throw new ModelHubError('INVALID_DESCRIPTOR', parsed.message, {
        issues: parsed.issues.map((issue) => ({ path: issue.path, message: issue.message })) as unknown as Record<string, unknown>[],
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
  static async fromConfigAndDiscovery(
    raw: unknown,
    options: Omit<ModelHubOptions, 'config'> = {},
  ): Promise<{ readonly hub: ModelHub; readonly discovery: DiscoveryResult }> {
    const parsed = parseModelCatalogConfig(raw, 'models.json');
    if (!parsed.ok) {
      throw new ModelHubError('INVALID_DESCRIPTOR', parsed.message, {
        issues: parsed.issues.map((issue) => ({ path: issue.path, message: issue.message })) as unknown as Record<string, unknown>[],
      });
    }
    const registry = new DiscoveryRegistry(
      [...(options.discoverers ?? defaultDiscoverers()), ...(options.extraDiscoverers ?? [])],
      {
        ...(options.discoveryTtlMs === undefined ? {} : { ttlMs: options.discoveryTtlMs }),
        ...(options.discoveryTimeoutMs === undefined ? {} : { timeoutMs: options.discoveryTimeoutMs }),
        ...(options.log === undefined ? {} : { log: (message, fields) => options.log?.(message, fields) }),
      },
    );
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
  onEvent(listener: HubEventListener): () => void {
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
  private emit(event: HubEvent): void {
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (error) {
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
  async refreshDiscovery(options: { readonly force?: boolean } = {}): Promise<DiscoveryResult> {
    const registry = this.discovery;
    if (registry === undefined) {
      return { descriptors: [], warnings: [], cached: true, durationMs: 0, hostIds: [] };
    }

    const pass = (async (): Promise<DiscoveryResult> => {
      const result = await registry.generate(this.staticConfig.hosts ?? [], options.force === true ? { refresh: true } : {});
      if (result.cached) return result;

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
        this.log(
          `hub: discovery updated the catalog (${added.length} added, ${removed.length} removed)`,
          { added, removed, hosts: result.hostIds },
        );
      }
      return result;
    })();

    this.discoveryPass = pass;
    try {
      return await pass;
    } finally {
      if (this.discoveryPass === pass) this.discoveryPass = undefined;
    }
  }

  /**
   * The most recent discovery pass, when one is in flight or just finished.
   *
   * Exposed so a caller can await the background pre-warm the constructor
   * started without guessing at a delay.
   */
  get pendingDiscovery(): Promise<DiscoveryResult> | undefined {
    return this.discoveryPass;
  }

  /**
   * Every configured model with its live status.
   * @param options - `includeDisabled` lists models excluded from routing too.
   * @returns one view per model.
   */
  listModels(options: { readonly includeDisabled?: boolean } = {}): readonly ModelView[] {
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
  getModel(modelId: string): ModelView {
    return { model: this.catalog.requireModel(modelId), status: this.runtime.getModelStatus(modelId) };
  }

  /**
   * The live status of one model.
   * @param modelId - the model id.
   * @returns the status.
   * @throws ModelHubError with `MODEL_NOT_FOUND`.
   */
  getModelStatus(modelId: string): ModelRuntimeStatus {
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
  findModelsByCapability(capability: string): readonly ModelView[] {
    if (!isCapability(capability)) {
      throw new ModelHubError(
        'UNKNOWN_CAPABILITY',
        `"${capability}" is not a known capability. Known capabilities are listed by listCapabilities().`,
        { capability },
      );
    }
    return this.catalog
      .findModelsByCapabilityIncludingDisabled(capability)
      .map((model) => ({ model, status: this.runtime.getModelStatus(model.id) }));
  }

  /**
   * Every capability at least one enabled model declares.
   * @returns capability views in vocabulary order.
   */
  listCapabilities(): readonly CapabilityView[] {
    return this.catalog.listCapabilities();
  }

  /**
   * Capabilities no enabled model declares, with the reason.
   * @returns one entry per unserved capability.
   */
  listUnservedCapabilities(): readonly { capability: Capability; reason: string }[] {
    return this.catalog.listUnservedCapabilities();
  }

  /** The machine resources routing decisions are made against. */
  get machineProfile(): MachineProfile {
    return this.catalog.machineProfile;
  }

  // ───────────────────────────── lifecycle ─────────────────────────────

  /**
   * Start a model.
   * @param modelId - the model id.
   * @returns whether a process was started, and the resulting health.
   * @throws ModelHubError with `START_FAILED`, `INSUFFICIENT_RESOURCES`, or `MODEL_NOT_FOUND`.
   */
  async startModel(
    modelId: string,
  ): Promise<{ started: boolean; alreadyRunning: boolean; health: { healthy: boolean; detail?: string } }> {
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
  async stopModel(
    modelId: string,
    options: { readonly force?: boolean } = {},
  ): Promise<{ stopped: boolean; wasRunning: boolean }> {
    const result = await this.runtime.stopModel(modelId, options);
    if (result.stopped) this.emit({ type: 'model/stopped', modelId });
    return result;
  }

  /**
   * Restart a model.
   * @param modelId - the model id.
   * @returns whether a process was started, and the resulting health.
   */
  async restartModel(
    modelId: string,
  ): Promise<{ started: boolean; alreadyRunning: boolean; health: { healthy: boolean; detail?: string } }> {
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
  async probeModel(modelId: string): Promise<{ healthy: boolean; detail?: string; latencyMs?: number }> {
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
  async probeAll(): Promise<{ healthy: number; total: number }> {
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
  async route(request: InvocationRequest, options: InvokeOptions = {}): Promise<RoutingDecision> {
    this.assertUsableCapability(request.capability);
    const artifacts = this.storeFor(options.artifactRoot);
    const resolved = await resolveRequestInputs({ artifacts }, request);
    const decision = routeRequest(
      { catalog: this.catalog, runtime: this.runtime, artifacts },
      resolved,
      this.routingPolicy,
    );
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
  async invokeModel(
    request: InvocationRequest,
    options: InvokeOptions = {},
  ): Promise<InvocationResult & { readonly decision: RoutingDecision }> {
    this.assertUsableCapability(request.capability);
    const artifacts = this.storeFor(options.artifactRoot);
    const resolved = await resolveRequestInputs({ artifacts }, request);
    const decision = routeRequest(
      { catalog: this.catalog, runtime: this.runtime, artifacts },
      resolved,
      this.routingPolicy,
    );
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

    const failures: { modelId: string; code: string; message: string }[] = [];

    for (let index = 0; index < attempts.length; index += 1) {
      const modelId = attempts[index];
      if (modelId === undefined) continue;
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
      } catch (error) {
        const described = toHubError(error, 'INVOCATION_FAILED', { modelId });
        // A cancellation is the caller's decision, not a model failure: retrying
        // it on another model would be actively wrong.
        if (described.code === 'INVOCATION_ABORTED') throw described;
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
    throw new ModelHubError(
      'INVOCATION_FAILED',
      `every candidate model failed for capability "${request.capability}" (${failures.length} attempt(s)):\n${detail}`,
      {
        capability: request.capability,
        failures: failures as unknown as Record<string, unknown>[],
        routing: explainDecision(decision, true),
      },
    );
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
  private async invokeOn(
    request: InvocationRequest,
    resolved: Awaited<ReturnType<typeof resolveRequestInputs>>,
    modelId: string,
    decision: RoutingDecision,
    artifacts: ArtifactStore,
  ): Promise<InvocationResult> {
    const model = this.catalog.requireModel(modelId);
    const adapter = this.adapters.get(model.adapter);
    if (adapter === undefined) {
      throw new ModelHubError(
        'UNSUPPORTED_OPERATION',
        `model "${modelId}" uses adapter kind "${model.adapter}", which is not registered`,
        { modelId, adapter: model.adapter },
      );
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
        throw new ModelHubError(
          'INVOCATION_FAILED',
          `adapter for model "${modelId}" returned a structured value that is not lossless JSON`,
          { modelId, adapter: model.adapter },
        );
      }
      // The guard above establishes the runtime property the declared type
      // asserts; the cast records that check rather than dodging it.
      const value = structured as Readonly<Record<string, JsonValue>> | undefined;

      const result: InvocationResult = {
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
      this.log(
        `hub: ${request.capability} served by ${modelId} in ${durationMs} ms ` +
          `(${output.outputs.length} output(s)${gate.coldStart ? ', cold start' : ''})`,
        { modelId, capability: request.capability, rationale: decision.rationale },
      );
      return result;
    } finally {
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
  async tryInvokeModel(
    request: InvocationRequest,
    options: InvokeOptions = {},
  ): Promise<
    | { readonly ok: true; readonly result: InvocationResult; readonly decision: RoutingDecision }
    | { readonly ok: false; readonly error: ModelHubError; readonly decision?: RoutingDecision }
  > {
    try {
      const result = await this.invokeModel(request, options);
      const { decision, ...invocation } = result;
      return { ok: true, result: invocation, decision };
    } catch (error) {
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
  async getArtifact(artifactId: string, artifactRoot?: string): Promise<Artifact | undefined> {
    return this.storeFor(artifactRoot).get(artifactId);
  }

  /**
   * List stored artifacts, newest first.
   * @param limit - maximum number to return. Defaults to 20.
   * @returns artifact references.
   */
  async listArtifacts(limit = 20, artifactRoot?: string): Promise<readonly Artifact[]> {
    return this.storeFor(artifactRoot).list(limit);
  }

  // ───────────────────────────── lifecycle of the hub ──────────────────

  /**
   * Stop background timers and shut down every process the hub owns.
   *
   * Safe to call more than once. Every owned process is given its declared
   * shutdown budget; a failure on one does not prevent stopping the others.
   */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.runtime.dispose();
    this.listeners.clear();
  }

  /**
   * Reject an unknown capability before any work happens.
   * @param capability - the requested capability.
   * @throws ModelHubError with `UNKNOWN_CAPABILITY`.
   */
  private assertUsableCapability(capability: string): void {
    if (!isCapability(capability)) {
      throw new ModelHubError(
        'UNKNOWN_CAPABILITY',
        `"${capability}" is not a capability this hub knows. ` +
          'Call listCapabilities() for the vocabulary this deployment supports.',
        { capability, known: this.catalog.listCapabilities().map((view) => view.capability) },
      );
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
export function defaultArtifactRoot(): string {
  const fromEnv = process.env['AIMH_ARTIFACT_ROOT'];
  if (fromEnv !== undefined && fromEnv.trim().length > 0) return fromEnv;
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
function defaultDiscoverers(): readonly HostDiscoverer[] {
  return [createOllamaDiscoverer(), createComfyUiDiscoverer(), createA1111Discoverer()];
}

/** Re-export so callers can construct a hub with a silent logger without another import. */
export { silentLogger };
export type { AvailabilityState };
