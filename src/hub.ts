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

import type { Artifact, ArtifactStore } from './artifacts/types.ts';
import { LocalArtifactStore } from './artifacts/local-store.ts';
import { AdapterRegistry } from './adapters/types.ts';
import { silentLogger } from './adapters/types.ts';
import type { ModelAdapter } from './adapters/types.ts';
import { createMockAdapter } from './adapters/mock.ts';
import { createOpenAiCompatibleAdapter } from './adapters/openai.ts';
import { createHttpJsonAdapter } from './adapters/http-json.ts';
import { createComfyUiAdapter } from './adapters/comfyui.ts';
import { createThreeDAdapter } from './adapters/three-d.ts';
import type { Capability } from './catalog/capabilities.ts';
import { CAPABILITIES, isCapability } from './catalog/capabilities.ts';
import type { ModelCatalogConfig, ModelDescriptor, ModelHost } from './catalog/descriptor.ts';
import { parseModelCatalogConfig, resolveDescriptor } from './catalog/descriptor.ts';
import type { CapabilityView } from './catalog/registry.ts';
import { ModelCatalog } from './catalog/registry.ts';
import { createComfyUiDiscoverer } from './discovery/comfyui.ts';
import { createA1111Discoverer } from './discovery/a1111.ts';
import { createOllamaDiscoverer } from './discovery/ollama.ts';
import { createThreeDDiscoverer } from './discovery/three-d.ts';
import { DiscoveryRegistry, mergeCatalogConfig } from './discovery/types.ts';
import type { DiscoveryResult, HostDiscoverer } from './discovery/types.ts';
import type { MachineProfile, MachineResourceUse } from './types.ts';
import { reserveResources } from './types.ts';
import type { AvailabilityState, InvocationAttempt, InvocationRequest, InvocationResult, ModelRuntimeStatus, ModelView, RoutingDecision, UnresolvedOutcome } from './types.ts';
import { ModelHubError, toHubError } from './errors.ts';
import { probeMachine } from './machine.ts';
import type { MachineProbeOptions } from './machine.ts';
import type { ExecutionPolicy } from './util/process.ts';
import { DEFAULT_EXECUTION_POLICY } from './util/process.ts';
import { RuntimeManager } from './runtime/manager.ts';
import type { RoutingPolicy } from './router/router.ts';
import { DEFAULT_ROUTING_POLICY, explainDecision, resolveRequestInputs, routeRequest } from './router/router.ts';
import type { CapabilityPlan, OptionLiveState, PolicyOptionFacts, ResolvedPolicyOption } from './policy/capability-policy.ts';
import { CapabilityPolicy, isUsableState, modelOptionFacts } from './policy/capability-policy.ts';
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
  | { readonly type: 'invocation/fellback'; readonly fromModelId: string; readonly toModelId: string; readonly reason: string }
  /** An option the plan refused to use, and the switch or fact that refused it. */
  | { readonly type: 'capability/skipped'; readonly capability: Capability; readonly optionId: string; readonly reason: string }
  /** The chosen option could not be made ready; the plan moves to the next one. */
  | { readonly type: 'capability/readiness-failed'; readonly capability: Capability; readonly optionId: string; readonly reason: string }
  /** A permitted option's engine is about to be started, by this call or another. */
  | { readonly type: 'capability/launch'; readonly capability: Capability; readonly optionId: string; readonly detail: string }
  /** The operator's policy changed while the hub was running. */
  | { readonly type: 'policy/changed'; readonly revision: number; readonly globalEnabled: boolean }
  /** The outcome after submission is unknown, so nothing was retried. */
  | { readonly type: 'invocation/unresolved'; readonly modelId: string; readonly capability: Capability; readonly reason: string };

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
  /**
   * Catalog entries that were read and then deliberately not published.
   *
   * Passed through to the catalog so {@link ModelHub.loadDiagnostics} and the
   * settings page can report a superseded entry — a ComfyUI weight-file provider
   * written by an older discovery pass, say — without the entry taking the whole
   * deployment down with it.
   */
  readonly catalogWarnings?: readonly string[];
  /**
   * Probe this machine's resources before routing, rather than trusting totals.
   *
   * **On by default.** The probe is what turns "this model needs 8 GiB VRAM and
   * the card has 8 GiB" into a correct decision when another engine is already
   * holding 5 of them: without it, routing compares against capacity and happily
   * selects a model that will die with a CUDA out-of-memory error.
   *
   * Two switches modify it:
   *
   * - {@link machineProbe}.`skipGpuProbe` reports only OS memory. Cheap, and the
   *   right choice in a container with no GPU visibility.
   * - Setting the whole object to `false` disables the probe. The catalog is then
   *   built from `undefined`, i.e. its own "not probed" placeholder, and resource
   *   checks pass everything through — which is exactly what a test that wants a
   *   deterministic catalog, or an operator who has supplied `machine` by hand,
   *   needs.
   *
   * The probe is asynchronous by nature (`nvidia-smi` is a subprocess) while the
   * constructor is synchronous by contract, so it runs in the background and the
   * catalog is updated when it lands. Routing then re-measures on demand once the
   * previous measurement has gone stale — see
   * {@link ModelHubOptions.resourceTtlMs} — so a long-lived host decides against
   * the machine as it is now rather than as it was at boot.
   */
  readonly probeResources?: boolean | MachineProbeOptions;
  /**
   * How long a machine measurement stays fresh enough to route against.
   *
   * Defaults to 30000. A routing decision re-measures when the last measurement is
   * older than this, and always after an invocation, because a generation is the
   * one thing here that genuinely changes how much memory is free.
   *
   * The cost is one `nvidia-smi` subprocess; the benefit is that a model started
   * by another application ten minutes ago is visible to the next decision instead
   * of being routed into an out-of-memory crash. Set it to `Infinity` to measure
   * once at startup and never again, or to `0` to measure before every decision.
   */
  readonly resourceTtlMs?: number;
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
  /**
   * The catalog file this hub was built from, when it was built from one.
   *
   * It exists because a catalog is a document whose relative paths are relative
   * to *itself*: `adapterConfig.workflowPath` names a file next to the catalog
   * that declares it, and nothing about the process's working directory says
   * where that is. A long-lived host is launched once, from wherever its launcher
   * happened to be, and then serves catalogs in directories it has never stood
   * in — so a configured relative path resolved against `process.cwd()` points
   * somewhere unrelated and fails with a bare `ENOENT`. The hub knows the exact
   * path it loaded, so it hands the directory to adapters as
   * {@link AdapterInvocation.catalogDir}.
   *
   * Omitted when the catalog came from an in-memory document
   * ({@link ModelHub.fromConfig}), in which case an adapter falls back to its
   * previous behaviour.
   */
  readonly catalogPath?: string;
  /**
   * The saved capability policy: per-capability order and enable/disable.
   *
   * Omitted means the empty policy with the deny-by-default rule, under which
   * every catalog option is permitted exactly as its descriptor says (so an
   * existing deployment behaves unchanged) and every non-catalog option — a
   * subscription backend — is refused until an operator enables it. That
   * asymmetry is the point: presence in a list is not authorization.
   */
  readonly capabilityPolicy?: CapabilityPolicy;
  /**
   * Extra capability options that are not catalog models.
   *
   * A subscription backend is one: it has no process, no endpoint, and no place
   * in `models.json`, but it *is* an option that can serve a capability, so it
   * belongs in the same ordered plan as everything else. Each entry's `invoke`
   * is what actually runs the work, which is how the hub stays free of any
   * knowledge about subscriptions, credentials, or sessions.
   */
  readonly capabilityOptions?: readonly CapabilityOption[];
  /**
   * The deployment's default limit on how many options one request may attempt.
   *
   * Omitted — the shipping default — means the whole permitted list is walked.
   * Setting it is an explicit admission that some permitted options will go
   * untried, and the failure message says so rather than implying they failed.
   */
  readonly attemptBudget?: number;
}

/**
 * One capability option the hub cannot execute itself.
 *
 * The owning plugin provides the option's facts and the function that performs
 * the work; the hub provides ordering, enablement, planning, diagnostics, and
 * the promise that a disabled option is never called. An option whose backend
 * cannot serve a request throws a `ModelHubError`, and the plan treats it exactly
 * like a failed local model — which is what makes local engines, subscription
 * backends, and everything in between fall back in one saved order.
 */
export interface CapabilityOption {
  /** The option's facts, including its stable id and declared capabilities. */
  readonly facts: PolicyOptionFacts;
  /**
   * Live usability, asked once per plan.
   *
   * This is where "no usable login" is reported, so a subscription backend with
   * nobody logged in is skipped with a reason instead of failing mid-request.
   *
   * @returns the state and the reason, in words an operator can act on.
   */
  readonly status?: () => Promise<OptionLiveState> | OptionLiveState;
  /**
   * Perform the work.
   *
   * @param request - the original request.
   * @param context - resolved inputs and the store the produced artifacts must
   *   be written into, so the option's output joins the same artifact system the
   *   local adapters use and later steps can consume it.
   * @returns the artifact ids produced, and any lossless-JSON value.
   */
  readonly invoke: (
    request: InvocationRequest,
    context: CapabilityOptionInvocation,
  ) => Promise<{ readonly artifactIds: readonly string[]; readonly value?: Readonly<Record<string, JsonValue>> }>;
}

/** What a {@link CapabilityOption} is handed when it runs. */
export interface CapabilityOptionInvocation {
  /** The capability being served. */
  readonly capability: Capability;
  /** The prompt, when the request carried one. */
  readonly prompt?: string;
  /** The resolved input artifacts, in caller order. */
  readonly inputs: readonly Artifact[];
  /** Capability-specific settings. */
  readonly options: Readonly<Record<string, unknown>>;
  /** The store produced artifacts must be written into. */
  readonly artifacts: ArtifactStore;
  /** Cancellation. A remote backend must abandon its request when this fires. */
  readonly signal: AbortSignal;
  /**
   * Report that the request has been submitted to the backend.
   *
   * Called by the option the moment the work can no longer be called off — after
   * an HTTP request is sent, after a job is queued. From that point a timeout or
   * a disconnect produces an *unresolved* outcome rather than a retry, because
   * resubmitting could double-charge or double-render.
   *
   * @param jobIds - identifiers that can be used to look the work up later.
   */
  readonly onSubmit?: (jobIds?: Readonly<Record<string, string>>) => void;
  /** Diagnostics sink. */
  readonly log: (message: string, fields?: Readonly<Record<string, unknown>>) => void;
}

/** Optional per-call controls for {@link ModelHub.invokeModel}. */
export interface InvokeOptions {
  /**
   * Whether a failed invocation may continue to the next permitted option.
   *
   * Defaults to true. This is what turns "the 3D model is installed but broken"
   * into a successful workflow on a fallback model instead of a dead end — and
   * because the order comes from the same saved policy the operator arranged, the
   * fallback is reproducible too.
   */
  readonly allowFallback?: boolean;
  /**
   * The most options to attempt, including the first.
   *
   * **Unset by default, and deliberately so.** The permitted list is walked in
   * full: a saved order of ten options means ten attempts, because stopping
   * silently at three is how a working fallback gets reported as "every option
   * failed". A deployment that genuinely needs a smaller budget sets one here or
   * through `attemptBudget` in the plugin config, and the failure message then
   * says how many options went untried rather than pretending they were tried.
   */
  readonly maxAttempts?: number;
  /**
   * An overall deadline for the whole invocation, in milliseconds.
   *
   * Applies across *all* attempts, not to each one: an option that hangs must not
   * be able to consume the budget of the options behind it. When it expires, the
   * remaining options are reported as untried — a limitation of the plan, not a
   * model failure.
   */
  readonly deadlineMs?: number;
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

/** One option in an assembled execution plan, and how it will be executed. */
export interface PlannedOption {
  /** The permitted, compatible option. */
  readonly option: ResolvedPolicyOption;
  /** Whether a local adapter runs it or a registered remote option does. */
  readonly kind: 'model' | 'option';
}

/** A request's complete execution plan, assembled before anything runs. */
export interface PlannedInvocation {
  /** The policy's resolved plan for the capability. */
  readonly plan: CapabilityPlan;
  /** The request's resolved inputs. */
  readonly resolved: Awaited<ReturnType<typeof resolveRequestInputs>>;
  /** The options execution will walk, in order. */
  readonly chain: readonly PlannedOption[];
  /** Every option left out, and why. */
  readonly reasons: readonly { readonly id: string; readonly reason: string }[];
  /** The routing decision, annotated with the policy that produced it. */
  readonly decision: RoutingDecision;
}

/** One capability with every option that can serve it, for the settings page. */
export interface CapabilityOptionListEntry {
  /** The capability. */
  readonly capability: Capability;
  /** Its resolved, ordered plan. */
  readonly plan: CapabilityPlan;
}

/** The outcome of one option's dispatch. */
type DispatchOutcome =
  | { readonly kind: 'result'; readonly result: InvocationResult }
  | {
      readonly kind: 'unresolved';
      readonly result: InvocationResult;
      readonly unresolved: UnresolvedOutcome;
    };

/**
 * The router's candidate order, winner first.
 *
 * This is the fix for a real ordering bug: {@link ModelHub.invokeModel} used to
 * build its fallback list by filtering the decision's `candidates` array, which
 * the router fills in *catalog iteration* order rather than in preference order.
 * The winner was correct and every fallback after it was arbitrary — a
 * higher-priority model could sit behind a lower-priority one purely because of
 * where it appeared in `models.json`.
 *
 * @param decision - the routing decision.
 * @returns the eligible candidates, best first.
 */
function routerOrder(decision: RoutingDecision): readonly string[] {
  const eligible = decision.candidates.filter((candidate) => candidate.eligible);
  return [
    decision.modelId,
    ...eligible
      .filter((candidate) => candidate.modelId !== decision.modelId)
      .sort((left, right) => {
        const leftScore = left.score ?? Number.POSITIVE_INFINITY;
        const rightScore = right.score ?? Number.POSITIVE_INFINITY;
        if (leftScore !== rightScore) return leftScore - rightScore;
        return left.modelId.localeCompare(right.modelId);
      })
      .map((candidate) => candidate.modelId),
  ];
}

/**
 * Which stage a failure code belongs to, for the attempt log.
 *
 * "Launch forbidden", "stopped", "startup failed", and "generation failed" are
 * four different incidents and an operator fixes each differently, so they are
 * reported as four different stages rather than as one failure list.
 *
 * @param code - the stable failure code.
 * @returns the stage.
 */
function stageForCode(code: string): InvocationAttempt['stage'] {
  switch (code) {
    case 'POLICY_DISABLED':
    case 'LIFECYCLE_UNSUPPORTED':
    case 'UNSAFE_OPERATION':
      return 'launch';
    case 'START_FAILED':
    case 'HEALTH_CHECK_FAILED':
    case 'INSUFFICIENT_RESOURCES':
    case 'MODEL_UNAVAILABLE':
      return 'readiness';
    default:
      return 'dispatch';
  }
}

/**
 * Build the cancellation error.
 * @param capability - the capability that was being served.
 * @returns the error to throw.
 */
function abortedError(capability: Capability): ModelHubError {
  return new ModelHubError('INVOCATION_ABORTED', `the ${capability} request was cancelled before dispatch`, {
    capability,
  });
}

/**
 * The reason a specific option is forbidden for a specific capability.
 * @param capability - the capability.
 * @param modelId - the option.
 * @returns the reason sentence.
 */
function disabledWith(capability: Capability, modelId: string): string {
  return `"${modelId}" is disabled for capability "${capability}" by the capability policy`;
}

/**
 * Whether a failure *after submission* leaves the outcome genuinely unknown.
 *
 * The distinction this draws is the difference between a retry and a duplicate:
 *
 * - A timeout or a cancellation after the work was handed over means the engine
 *   may well have finished. Retrying could produce a second paid image or a
 *   second multi-minute render, so it must not happen.
 * - A refusal the backend stated — a quota rejection, an authentication failure,
 *   a 4xx — is a definite "no", and the next option may be tried with a clear
 *   conscience.
 *
 * A cancellation is included deliberately: the caller's own abort is *also* an
 * answer the hub cannot complete, and reporting it as an unresolved outcome is
 * more honest than claiming nothing ran.
 *
 * @param error - the failure.
 * @returns whether the outcome is unknown.
 */
function isAmbiguousFailure(error: unknown): boolean {
  const described = toHubError(error, 'INVOCATION_FAILED');
  switch (described.code) {
    case 'INVOCATION_TIMEOUT':
    case 'INVOCATION_ABORTED':
      return true;
    default:
      return /aborted|timeout|timed out|socket hang up|ECONNRESET|EPIPE|premature close/i.test(
        `${described.message} ${String(described.details['systemCode'] ?? '')}`,
      );
  }
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
   * The capability policy: saved order and enable/disable, per capability.
   *
   * Mutable by design — an operator flipping a switch must take effect on the
   * next request rather than the next restart — and monotonic in its revision, so
   * a request can say which policy produced it.
   */
  readonly policy: CapabilityPolicy;

  /**
   * Options the hub cannot execute itself, keyed by option id.
   *
   * Subscription backends live here. They are registered rather than configured
   * because their transport belongs to the plugin that owns the credential: the
   * hub's part is ordering them, refusing them when disabled, and reporting what
   * happened.
   */
  private readonly capabilityOptions = new Map<string, CapabilityOption>();

  /**
   * Option ids that describe a *process* rather than a catalog entry.
   *
   * A scan can launch an engine the catalog carries no model for. Such an id has
   * no capabilities, so it has no policy entry, and the launch gate must not
   * refuse it for that — the gate exists to stop a *routing* choice from
   * overriding the operator, not to block an explicit management action.
   */
  private readonly engineRecordIds = new Set<string>();

  /**
   * The deployment's per-request attempt budget, when it configured one.
   *
   * Undefined is the shipping default and means "walk the whole permitted list".
   */
  private readonly defaultAttemptBudget: number | undefined;

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

  /**
   * The models the last completed discovery pass reported.
   *
   * Kept because {@link publishScannedModels} republishes the catalog outside a
   * discovery pass, and a scan must not quietly erase what a discoverer found.
   */
  private discoveredModels: readonly ModelDescriptor[] = [];

  /**
   * The models an explicit scan registered (the "Local models" page).
   *
   * A scan is a user action, not a background pass, so its results are held here
   * and merged ahead of discovery results: an operator who just scanned a
   * workflow has said that workflow should be usable now.
   */
  private scannedModels: readonly ModelDescriptor[] = [];

  private readonly listeners = new Set<HubEventListener>();
  private readonly log: (message: string, fields?: Readonly<Record<string, unknown>>) => void;
  private readonly ownsArtifacts: boolean;
  private readonly probeOptions: MachineProbeOptions | undefined;
  private readonly resourceTtlMs: number;
  /**
   * The directory the catalog was read from, when it was read from a file.
   *
   * Held resolved, because every use is "resolve a catalog-relative path against
   * this", and the adapters that need it should not each re-derive it from
   * {@link ModelHubOptions.catalogPath}.
   */
  private readonly catalogDir: string | undefined;
  private disposed = false;

  /**
   * The most recent probe of this machine, when one has run.
   *
   * Held here rather than read back out of the catalog because the two answer
   * different questions at different times: the catalog holds the figures routing
   * currently uses, while this holds the *measurement*, with its evidence and its
   * timestamp, whether or not it was allowed to change routing.
   */
  private resources: { readonly profile: MachineProfile; readonly evidence: readonly string[] } | undefined;

  /** The in-flight probe, so concurrent callers share one `nvidia-smi` run. */
  private resourcePass: Promise<MachineProfile> | undefined;

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
    this.policy = options.capabilityPolicy ?? new CapabilityPolicy();
    for (const option of options.capabilityOptions ?? []) this.capabilityOptions.set(option.facts.id, option);
    this.defaultAttemptBudget = options.attemptBudget;
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
      ...(options.catalogWarnings === undefined ? {} : { warnings: options.catalogWarnings }),
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
    const defaultAdapters: readonly ModelAdapter[] =
      options.adapters ?? [
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
      // Probes go through the same catalog-relative rule as invocations, so a
      // model whose file lives beside its catalog is not reported unhealthy.
      ...(this.catalogDir === undefined ? {} : { catalogDir: this.catalogDir }),
      // The launch gate. Consulted immediately before a process is spawned, so a
      // toggle flipped during a three-minute cold start wins: two concurrent
      // requests join one transition, and a check performed only at routing time
      // would let a since-disabled option ride someone else's launch.
      isPermitted: (modelId) => this.permissionFor(modelId),
    });

    if (options.manageTimers !== false) this.runtime.start();

    // Probe this machine's resources in the background, for the same reason
    // discovery is pre-warmed: the constructor cannot await a subprocess. Until
    // it lands the catalog reports "not probed", whose resource check passes
    // everything through — deliberately permissive, because refusing every model
    // for the first few hundred milliseconds of a process's life would be worse
    // than the OOM the probe exists to prevent.
    if (this.probeOptions !== undefined) {
      void this.refreshResources().catch((error: unknown) => {
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
    return new ModelHub({ ...options, config: parsed.config, catalogWarnings: parsed.warnings });
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
    const discovery = await registry.generate(withComfyCatalogDir(parsed.config.hosts ?? [], options.catalogPath === undefined ? undefined : dirname(options.catalogPath)));
    const hub = new ModelHub({
      ...options,
      config: mergeCatalogConfig(parsed.config, discovery.descriptors),
      catalogWarnings: parsed.warnings,
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
      const result = await registry.generate(withComfyCatalogDir(this.staticConfig.hosts ?? [], this.catalogDir), options.force === true ? { refresh: true } : {});
      if (result.cached) return result;

      for (const warning of result.warnings) {
        this.log(`hub: discovery warning: ${warning.hostId} (${warning.engine}): ${warning.message}`, {
          hostId: warning.hostId,
        });
      }

      const before = new Set(this.catalog.listModelIds());
      this.discoveredModels = result.descriptors;
      this.catalog.replaceModels(this.mergedModelList());
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
   * The catalog's model list, assembled from its three sources in precedence
   * order: static configuration, then an explicit scan, then runtime discovery.
   *
   * `mergeCatalogConfig` is what applies that precedence — static entries come
   * first and a later duplicate id is dropped — so this method only has to state
   * the order once, and every publisher of the catalog agrees on it.
   *
   * @returns the merged descriptors.
   */
  private mergedModelList(): readonly ModelDescriptor[] {
    return mergeCatalogConfig(this.staticConfig, [...this.scannedModels, ...this.discoveredModels]).models;
  }

  /**
   * Publish the providers an explicit scan found, alongside everything already
   * registered.
   *
   * This is the seam the "Local models" page uses. A scan is a deliberate user
   * action — "find the workflows on this machine and let me run them" — so its
   * results are added to the running catalog rather than requiring a JSON edit
   * and a restart. It is the same `replaceModels` rebuild runtime discovery uses,
   * so the router, the runtime manager, and the tools all see the new providers
   * immediately, and calling it again with a fresh scan *replaces* the scanned
   * half rather than growing it: a workflow the operator deleted stops being
   * routable, and a repeated scan cannot accumulate duplicates.
   *
   * Static configuration still wins: a scanned provider whose id a `models.json`
   * entry already claims is dropped by the merge, so a hand-declared workflow is
   * never shadowed by a discovered one.
   *
   * @param descriptors - the providers the scan produced, in publication order.
   * @returns how many models the catalog now holds.
   */
  publishScannedModels(descriptors: readonly ModelDescriptor[]): number {
    this.scannedModels = descriptors;
    this.catalog.replaceModels(this.mergedModelList());
    this.runtime.syncCatalog();
    this.log(`hub: scan published ${descriptors.length} provider(s)`, {
      total: this.catalog.listModels().length,
    });
    return this.catalog.listModels().length;
  }

  /** The providers an explicit scan registered, in publication order. */
  get scannedProviderIds(): readonly string[] {
    return this.scannedModels.map((model) => model.id);
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
      .map((model) => ({ model: publicModelView(model), status: this.runtime.getModelStatus(model.id) }));
  }

  /**
   * Look up one model with its live status.
   * @param modelId - the model id.
   * @returns the view.
   * @throws ModelHubError with `MODEL_NOT_FOUND`.
   */
  getModel(modelId: string): ModelView {
    return { model: publicModelView(this.catalog.requireModel(modelId)), status: this.runtime.getModelStatus(modelId) };
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
      .map((model) => ({ model: publicModelView(model), status: this.runtime.getModelStatus(model.id) }));
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
   * Start the engine behind a declared host, whether or not it serves a model.
   *
   * The scan needs this: it has to have ComfyUI answering before it can ask which
   * workflows exist, and a host with no configured workflow yet carries no model
   * to launch through. When the host *does* have a model, that model's id is used
   * so the process is tracked, stopped and disposed exactly as any other; when it
   * does not, a resolved engine record supplies the runtime and lifecycle the
   * launch needs, and is never added to the catalog, so it cannot become a
   * routing candidate for a capability nothing serves.
   *
   * Starting an engine that is already answering is a no-op that reports
   * `alreadyRunning`, which is what keeps a scan from spawning a second ComfyUI
   * beside the one the user is running.
   *
   * @param hostId - the declared host.
   * @returns the id the process is tracked under, and what happened.
   * @throws ModelHubError when the host is unknown or cannot be started.
   */
  async startEngineForHost(hostId: string): Promise<{
    readonly modelId: string;
    readonly started: boolean;
    readonly alreadyRunning: boolean;
    readonly health: { healthy: boolean; detail?: string };
  }> {
    const host = this.catalog.listHosts().find((candidate) => candidate.id === hostId);
    if (host === undefined) {
      throw new ModelHubError('MODEL_NOT_FOUND', `no host with id "${hostId}" is declared`, { hostId });
    }
    const existing = this.catalog.listModels().find((model) => model.hostId === hostId);
    const result =
      existing !== undefined
        ? await this.runtime.startModel(existing.id)
        : await this.runtime.startEngineRecord(
            resolveDescriptor(
              {
                id: `engine-${host.id}`,
                name: host.name,
                type: 'custom',
                capabilities: ['text_to_text'],
                host: host.id,
              },
              host,
            ),
          );
    const modelId = existing?.id ?? `engine-${host.id}`;
    if (result.started) {
      this.emit({ type: 'model/started', modelId, pid: this.runtime.getModelStatus(modelId).pid, coldStartMs: 0 });
    }
    return {
      modelId,
      started: result.started,
      alreadyRunning: result.alreadyRunning,
      health: {
        healthy: result.health.healthy,
        ...(result.health.detail === undefined ? {} : { detail: result.health.detail }),
      },
    };
  }

  /**
   * Stop a process the hub started for a host.
   *
   * Refuses to touch anything the hub does not own: `stopModel` is a no-op for a
   * model whose state holds no process, so a ComfyUI the user launched by hand is
   * never terminated by a scan finishing.
   *
   * @param modelId - the id returned by {@link startEngineForHost}.
   * @returns whether a process was stopped.
   */
  async stopEngine(modelId: string): Promise<{ stopped: boolean; wasRunning: boolean }> {
    return this.stopModel(modelId);
  }

  /**
   * Whether a model (or engine record) currently has work in flight.
   *
   * Used to decide whether an engine this scan started may be shut down: a
   * queued or running graph is never interrupted to tidy up after discovery.
   *
   * @param modelId - the id to inspect.
   * @returns the in-flight invocation count, or `0` when the id is unknown.
   */
  activeInvocationsFor(modelId: string): number {
    try {
      return this.runtime.getModelStatus(modelId).activeInvocations;
    } catch {
      return 0;
    }
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

  // ───────────────────────────── machine resources ─────────────────────

  /**
   * The machine resources routing decisions are made against.
   *
   * This is the profile the router reads, so it includes the resource headroom
   * figure when one has been probed. It is what `explain_routing` and
   * `get_model_status` show an operator who is asking "why was that model
   * rejected?".
   */
  get machineProfile(): MachineProfile {
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
  resourceSnapshot(): { readonly profile: MachineProfile; readonly evidence: readonly string[] } | undefined {
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
  availableResources(): { readonly profile: MachineProfile; readonly reserved: MachineResourceUse; readonly residentModelIds: readonly string[] } {
    const reserved: { vramGb: number; ramGb: number } = { vramGb: 0, ramGb: 0 };
    const residentModelIds: string[] = [];
    for (const model of this.catalog.listModels()) {
      const availability = this.runtime.checkAvailability(model.id);
      if (availability !== 'available' && availability !== 'unhealthy' && availability !== 'starting') continue;
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
  async refreshResources(): Promise<MachineProfile> {
    const existing = this.resourcePass;
    if (existing !== undefined) return existing;

    const pass = (async (): Promise<MachineProfile> => {
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
        for (const model of this.catalog.listModels()) this.runtime.revalidateResources(model.id);
      }
      this.log(
        `hub: machine resources — ${probed.profile.vramGb} GiB VRAM` +
          `${probed.profile.availableVramGb === undefined ? '' : ` (${probed.profile.availableVramGb} GiB free)`}, ` +
          `${probed.profile.ramGb} GiB RAM` +
          `${probed.profile.availableRamGb === undefined ? '' : ` (${probed.profile.availableRamGb} GiB free)`}`,
        { evidence: probed.evidence },
      );
      return probed.profile;
    })();

    this.resourcePass = pass;
    try {
      return await pass;
    } finally {
      if (this.resourcePass === pass) this.resourcePass = undefined;
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
  private async freshenResources(): Promise<void> {
    if (this.probeOptions === undefined) return;
    try {
      await this.ensureResourcesFresh(this.resourceTtlMs);
    } catch (error) {
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
  async ensureResourcesFresh(maxAgeMs = 60_000): Promise<MachineProfile> {
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
  async route(request: InvocationRequest, options: InvokeOptions = {}): Promise<RoutingDecision> {
    this.assertUsableCapability(request.capability);
    const artifacts = this.storeFor(options.artifactRoot);
    const plan = await this.planFor(request, { artifacts });
    this.emit({
      type: 'routing/decided',
      capability: request.capability,
      modelId: plan.decision.modelId,
      rationale: plan.decision.rationale,
    });
    return plan.decision;
  }

  /**
   * Resolve the full ordered plan for a request, without executing anything.
   *
   * This is the one place the answer to "what would happen?" is computed, which
   * is why `route`, `invoke_model`, `explain_routing`, and the settings page all
   * agree: they read the same plan.
   *
   * The catalog half is still routed by {@link routeRequest}, deliberately — the
   * router owns tag filtering, resource fit, and the reasons it records for each
   * rejection, and re-deriving those here would be a second implementation that
   * could contradict the first. The router's *result* is then filtered and
   * ordered by the operator's policy, and the non-catalog options are folded in
   * around it.
   *
   * @param request - the request to plan for.
   * @param context - the artifact store this call reads through.
   * @returns the plan and the routing decision built from it.
   */
  async planFor(
    request: InvocationRequest,
    context: { readonly artifacts: ArtifactStore },
  ): Promise<PlannedInvocation> {
    await this.freshenResources();
    const resolved = await resolveRequestInputs({ artifacts: context.artifacts }, request);
    const decision = routeRequest(
      { catalog: this.catalog, runtime: this.runtime, artifacts: context.artifacts },
      resolved,
      this.routingPolicy,
    );
    const plan = this.policy.plan(request.capability, await this.planInputs(request.capability, routerOrder(decision)));
    return this.assemblePlan(request, resolved, decision, plan);
  }

  /**
   * The policy's inputs for one capability: the catalog, the runtime, and every
   * registered remote option with its live state.
   *
   * @param capability - the capability being planned for.
   * @returns the plan inputs.
   */
  private async planInputs(capability: Capability, preferredOrder: readonly string[] = []): Promise<{
    readonly catalog: ModelCatalog;
    readonly runtime: { describe(modelId: string): OptionLiveState };
    readonly extraOptions: readonly PolicyOptionFacts[];
    readonly extraRuntime: Readonly<Record<string, OptionLiveState>>;
    readonly preferredOrder: readonly string[];
  }> {
    const extraRuntime: Record<string, OptionLiveState> = {};
    const extraOptions: PolicyOptionFacts[] = [];
    for (const [id, option] of this.capabilityOptions) {
      if (!option.facts.capabilities.includes(capability)) continue;
      extraOptions.push(option.facts);
      if (!this.policy.permits(capability, option.facts).enabled) continue;
      if (option.status === undefined) {
        extraRuntime[id] = { state: 'ready', reason: 'Ready: this backend reports no separate readiness gate.' };
        continue;
      }
      try {
        extraRuntime[id] = await option.status();
      } catch (error) {
        const described = toHubError(error, 'MODEL_UNAVAILABLE', { modelId: id });
        extraRuntime[id] = { state: 'unavailable', reason: `Unavailable: ${described.message}` };
        this.log(`hub: readiness of option ${id} could not be determined: ${described.message}`, { modelId: id });
      }
    }
    return {
      catalog: this.catalog,
      runtime: { describe: (modelId) => this.runtime.describeOption(modelId, capability) },
      extraOptions,
      extraRuntime,
      preferredOrder,
    };
  }

  /**
   * Turn a policy plan and a routing decision into the ordered chain execution
   * walks.
   *
   * @param request - the original request.
   * @param resolved - its resolved inputs.
   * @param decision - the router's decision for the catalog half.
   * @param plan - the policy's plan.
   * @returns the assembled plan.
   */
  private assemblePlan(
    request: InvocationRequest,
    resolved: Awaited<ReturnType<typeof resolveRequestInputs>>,
    decision: RoutingDecision,
    plan: CapabilityPlan,
  ): PlannedInvocation {
    const byId = new Map(plan.options.map((option) => [option.option.id, option]));
    const chain: PlannedOption[] = [];
    const reasons: { id: string; reason: string }[] = [];
    const pinned = request.modelId !== undefined && request.modelId.trim().length > 0 ? request.modelId : undefined;

    if (pinned !== undefined) {
      const chosen = this.restrictToPin(pinned, plan, request.capability);
      if (chosen !== undefined) chain.push(chosen);
    }

    // The effective order: everything the policy knows about, in the policy's own
    // saved order. One loop for local models and remote backends alike, because
    // they *are* the same kind of thing to a plan — a way to serve a capability —
    // and two loops would silently append every remote option after every local
    // one, which is exactly the ordering bug this walk exists to avoid.
    //
    // A catalog option the policy has never seen is appended by the policy's own
    // deterministic tie-break, so the router's preference survives for options
    // nobody has arranged yet.
    for (const option of plan.options) {
      if (pinned !== undefined && option.option.id !== pinned) continue;
      const remote = option.option.source === 'subscription';
      if (!this.catalog.getModel(option.option.id) && !remote) continue;
      if (this.capabilityOptions.has(option.option.id) !== remote) continue;
      if (!option.enabled) {
        reasons.push({ id: option.option.id, reason: option.reason });
        continue;
      }
      if (option.state === 'disabled' || option.state === 'unsupported' || option.state === 'unavailable') {
        reasons.push({ id: option.option.id, reason: option.reason });
        continue;
      }
      if (remote && (request.requiredTags ?? []).length > 0) {
        // A remote backend declares no tag vocabulary of its own, so it cannot
        // claim to satisfy a tag filter it was never asked about.
        reasons.push({
          id: option.option.id,
          reason: `requiredTags [${(request.requiredTags ?? []).join(', ')}] cannot be satisfied by a remote backend`,
        });
        continue;
      }
      if (!this.isOptionCompatible(option, resolved, request)) {
        reasons.push({ id: option.option.id, reason: 'it is not compatible with this request' });
        continue;
      }
      chain.push({ option, kind: remote ? 'option' : 'model' });
    }

    // Catalog models the router rejected keep the router's own explanation: it is
    // the component that knows whether a tag, a resource shortfall, or a bad
    // descriptor was responsible.
    for (const candidate of decision.candidates) {
      if (candidate.eligible) continue;
      if (chain.some((entry) => entry.option.option.id === candidate.modelId)) continue;
      if (reasons.some((entry) => entry.id === candidate.modelId)) continue;
      if (byId.get(candidate.modelId)?.enabled === false) continue;
      reasons.push({ id: candidate.modelId, reason: candidate.reason });
    }

    // Anything the policy left out but the loops above did not reach — an option
    // that exists only in the saved document, for instance — is reported too: an
    // operator asking "why was my local workflow not used?" needs the switch that
    // said no, not the absence of a row.
    for (const option of plan.options) {
      if (chain.some((entry) => entry.option.option.id === option.option.id)) continue;
      if (reasons.some((entry) => entry.id === option.option.id)) continue;
      reasons.push({ id: option.option.id, reason: option.reason });
    }

    const head = chain[0];
    // The router's own sentence is preserved and the policy context is appended
    // to it. The router's words are the ones that say *why this option* — "cold
    // but startable", "healthy and ready" — and a policy summary that replaced
    // them would trade a specific answer for a general one.
    const rationale =
      head === undefined
        ? `no permitted option can serve "${request.capability}"`
        : `${decision.rationale} [policy revision ${plan.revision}; ${chain.length} permitted option(s) in order: ${chain.map((entry) => entry.option.option.id).join(', ')}]`;

    return {
      plan,
      resolved,
      chain,
      reasons,
      decision: {
        ...decision,
        modelId: head?.option.option.id ?? decision.modelId,
        rationale,
        policyRevision: plan.revision,
        permittedOptions: chain.map((entry) => entry.option.option.id),
        skippedOptions: reasons,
      },
    };
  }

  /**
   * Build a {@link ModelHubError} for "nothing permitted can serve this".
   *
   * The message leads with the *policy* answer, because that is the difference
   * between "your machine cannot do this" and "you told me not to": the second is
   * fixed by a switch, and the error says which.
   *
   * @param capability - the requested capability.
   * @param plan - the plan that came out empty.
   * @returns the error to throw.
   */
  private noPermittedOption(capability: Capability, plan: PlannedInvocation): ModelHubError {
    if (plan.plan.options.length === 0) {
      return new ModelHubError(
        'NO_COMPATIBLE_MODEL',
        `no model or backend declares capability "${capability}" in this deployment`,
        {
          capability,
          policyRevision: plan.plan.revision,
          routing: explainDecision(plan.decision, true),
        },
      );
    }
    const disabled = plan.plan.options.filter((option) => !option.enabled);
    const usable = plan.plan.options.filter((option) => option.enabled && isUsableState(option.state));
    const detail = plan.reasons.map((entry) => `  - ${entry.id}: ${entry.reason}`).join('\n');
    if (usable.length === 0 && disabled.length > 0) {
      return new ModelHubError(
        'POLICY_DISABLED',
        `no option may serve "${capability}": ${disabled.length} of ${plan.plan.options.length} are switched off.\n${detail}\n` +
          'Enable one in Settings → Local models → Capability priorities, or turn the global switch back on.',
        {
          capability,
          policyRevision: plan.plan.revision,
          disabledOptions: disabled.map((option) => option.option.id) as unknown as Record<string, unknown>[],
          skippedOptions: plan.reasons as unknown as Record<string, unknown>[],
        },
      );
    }
    return new ModelHubError(
      'NO_COMPATIBLE_MODEL',
      `no permitted option can serve capability "${capability}" for this request:\n${detail}`,
      {
        capability,
        policyRevision: plan.plan.revision,
        skippedOptions: plan.reasons as unknown as Record<string, unknown>[],
        routing: explainDecision(plan.decision, true),
      },
    );
  }

  /**
   * Narrow a plan to one explicitly pinned option.
   *
   * An explicit request for a disabled option is answered with
   * `POLICY_DISABLED`, not with a silent substitution and not with a generic
   * "no compatible model": the agent asked for a specific thing and the honest
   * answer is that a switch forbids it.
   *
   * @param pinned - the requested option id.
   * @param plan - the policy's plan.
   * @param capability - the requested capability.
   * @returns the pinned option, or `undefined` when the router already rejected it.
   */
  private restrictToPin(
    pinned: string,
    plan: CapabilityPlan,
    capability: Capability,
  ): PlannedOption | undefined {
    const option = plan.options.find((entry) => entry.option.id === pinned);
    if (option === undefined) {
      // Not anything the policy knows about. If the catalog knows it, let the
      // router's own rejection carry the explanation; if nothing does, say so.
      if (this.catalog.getModel(pinned) !== undefined) return undefined;
      if (this.capabilityOptions.has(pinned)) {
        throw new ModelHubError('UNSUPPORTED_OPERATION', `option "${pinned}" cannot serve this capability`, {
          modelId: pinned,
          capability,
        });
      }
      throw new ModelHubError('MODEL_NOT_FOUND', `no model or backend with id "${pinned}" is registered`, {
        modelId: pinned,
        capability,
      });
    }
    if (!option.enabled) {
      throw new ModelHubError(
        'POLICY_DISABLED',
        `"${pinned}" is disabled for capability "${capability}" and will not run. ${option.reason}`,
        { modelId: pinned, capability, reason: option.reason },
      );
    }
    return { option, kind: option.option.source === 'subscription' ? 'option' : 'model' };
  }

  /**
   * Whether a permitted option can serve *this* request.
   *
   * The router already did this for catalog candidates; this repeats the check
   * for remote options, which the router has never heard of, and keeps the rule
   * in one place so an option cannot pass one filter and fail the other.
   *
   * @param option - the permitted option.
   * @param resolved - the resolved request.
   * @param request - the original request.
   * @returns whether it can serve.
   */
  private isOptionCompatible(
    option: ResolvedPolicyOption,
    resolved: Awaited<ReturnType<typeof resolveRequestInputs>>,
    request: InvocationRequest,
  ): boolean {
    if (!option.option.capabilities.includes(request.capability)) return false;
    for (const kind of resolved.kindSet) {
      if (!option.option.inputTypes.includes(kind)) return false;
    }
    // A prompt that no declared input can carry is dropped silently by some
    // engines and rejected by others; requiring `text` among the inputs makes the
    // difference visible at planning time instead of at generation time.
    if (resolved.hasPrompt && !option.option.inputTypes.includes('text')) return false;
    return true;
  }

  /**
   * Whether a catalog model may run at all, asked of the policy.
   * @param modelId - the catalog model id.
   * @returns whether it may run, with a reason when it may not.
   */
  private permissionFor(modelId: string): { readonly enabled: boolean; readonly reason: string } {
    const model = this.catalog.getModel(modelId);
    if (model === undefined) {
      // An engine record a scan registered has no catalog entry and therefore no
      // capability policy; a launch for one is a management action, not routing.
      return this.engineRecordIds.has(modelId)
        ? { enabled: true, reason: '' }
        : { enabled: false, reason: `"${modelId}" is not in the catalog` };
    }
    return this.policy.permitsModel(model);
  }

  /**
   * Whether a model or remote option may serve one specific capability now.
   * @param modelId - the catalog model id or option id.
   * @param capability - the capability.
   * @returns whether it is permitted, with a reason when it is not.
   */
  private permissionForCapability(
    modelId: string,
    capability: Capability,
  ): { readonly enabled: boolean; readonly reason: string } {
    const model = this.catalog.getModel(modelId);
    const facts = model !== undefined ? modelOptionFacts(model) : this.capabilityOptions.get(modelId)?.facts;
    if (facts === undefined) return { enabled: true, reason: '' };
    const verdict = this.policy.permits(capability, facts);
    return {
      enabled: verdict.enabled,
      reason: verdict.enabled ? '' : `"${modelId}" is disabled for capability "${capability}"`,
    };
  }

  /**
   * Register a capability option the hub cannot execute itself.
   *
   * Idempotent on the option id, so a plugin reload replaces rather than
   * duplicates. The returned disposer removes it, which keeps a plugin's unload
   * honest: a backend nobody serves must stop appearing in plans.
   *
   * @param option - the option to register.
   * @returns a disposer.
   */
  registerCapabilityOption(option: CapabilityOption): () => void {
    const previous = this.capabilityOptions.get(option.facts.id);
    this.capabilityOptions.set(option.facts.id, option);
    return () => {
      if (previous === undefined) this.capabilityOptions.delete(option.facts.id);
      else this.capabilityOptions.set(option.facts.id, previous);
    };
  }

  /** Every registered non-catalog option, in registration order. */
  listCapabilityOptions(): readonly PolicyOptionFacts[] {
    return [...this.capabilityOptions.values()].map((option) => option.facts);
  }

  /**
   * Every capability option — catalog and remote — with its saved policy and live
   * state. This is the data behind the "Capability priorities" section.
   *
   * The order shown is the order that would actually be used, so an option nobody
   * has arranged yet appears where the router would put it rather than in
   * alphabetical limbo. That means the page and the runtime cannot disagree about
   * which option is first — the one thing a priorities page must never get wrong.
   *
   * @param options - which capabilities to describe, and whether empty ones count.
   * @returns one entry per capability that has at least one option.
   */
  describeCapabilityOptions(
    options: { readonly capabilities?: readonly Capability[]; readonly includeEmpty?: boolean } = {},
  ): readonly CapabilityOptionListEntry[] {
    const entries: CapabilityOptionListEntry[] = [];
    for (const capability of options.capabilities ?? CAPABILITIES) {
      const plan = this.policy.plan(capability, {
        catalog: this.catalog,
        runtime: { describe: (modelId) => this.runtime.describeOption(modelId, capability) },
        extraOptions: [...this.capabilityOptions.values()].map((option) => option.facts),
        preferredOrder: this.routerOrderFor(capability),
      });
      if (plan.options.length === 0 && options.includeEmpty !== true) continue;
      entries.push({ capability, plan });
    }
    return entries;
  }

  /**
   * The router's ranking for one capability, as a list of ids, best first.
   *
   * Asked without a request, so the answer is the ranking for "this capability,
   * no particular inputs" — which is exactly what a priorities list needs and is
   * what the router would choose for the simplest call. A capability the router
   * cannot rank (nothing declares it, or every candidate is rejected) yields an
   * empty list, and the policy falls back to its own deterministic tie-break.
   *
   * @param capability - the capability to rank.
   * @returns candidate ids, best first; empty when nothing can be ranked.
   */
  private routerOrderFor(capability: Capability): readonly string[] {
    try {
      const decision = routeRequest(
        { catalog: this.catalog, runtime: this.runtime, artifacts: this.artifacts },
        {
          request: { capability },
          inputs: [],
          kindSet: new Set(),
          hasPrompt: false,
        },
        this.routingPolicy,
      );
      return routerOrder(decision);
    } catch {
      // Nothing eligible, or nothing declares it: an ordering hint is not worth
      // failing a page render over.
      return [];
    }
  }

  /**
   * Invoke a capability. The caller names a capability, never an engine.
   *
   * The sequence, in the order it actually happens:
   *
   * 1. Resolve and validate the request's inputs against the artifact store.
   * 2. Build the capability plan: global and per-capability enablement, then the
   *    saved order, then compatibility and live readiness — with a stopped but
   *    startable local option kept *in place* rather than demoted.
   * 3. Walk that plan in order. For each option: re-check permission, make its
   *    engine ready (reusing a warm one, joining a startup already in progress,
   *    or starting it), re-check permission once more, then dispatch to that
   *    exact option — and only that one.
   * 4. On a definitive failure, continue to the next permitted option. On a
   *    cancellation, stop. On an ambiguous post-submission outcome, stop and
   *    report it as unresolved rather than re-generating.
   *
   * There is no attempt cap by default: a saved order of ten options is tried in
   * full, because silently stopping at three is how a working fallback gets
   * reported as "every candidate failed".
   *
   * @param request - the invocation request.
   * @param options - artifact root, fallback control, deadline, and attempt budget.
   * @returns the invocation result, with the routing decision and the attempt log.
   * @throws ModelHubError with a stable code when nothing can serve the request.
   */
  async invokeModel(
    request: InvocationRequest,
    options: InvokeOptions = {},
  ): Promise<InvocationResult & { readonly decision: RoutingDecision }> {
    this.assertUsableCapability(request.capability);
    const artifacts = this.storeFor(options.artifactRoot);
    const plan = await this.planFor(request, { artifacts });

    if (plan.chain.length === 0) {
      throw this.noPermittedOption(request.capability, plan);
    }

    const allowFallback = options.allowFallback ?? true;
    const queue = allowFallback ? plan.chain : plan.chain.slice(0, 1);
    const requestedBudget = options.maxAttempts ?? this.defaultAttemptBudget;
    const budget = requestedBudget === undefined ? queue.length : Math.max(1, requestedBudget);
    const attemptsAllowed = Math.min(queue.length, budget);
    const deadline = options.deadlineMs === undefined ? undefined : Date.now() + Math.max(0, options.deadlineMs);

    const attemptLog: InvocationAttempt[] = [];
    const failures: { modelId: string; code: string; message: string }[] = [];

    for (let index = 0; index < attemptsAllowed; index += 1) {
      const planned = queue[index];
      if (planned === undefined) continue;
      const modelId = planned.option.option.id;

      // Cancellation is checked before anything is launched or dispatched: a
      // caller who has already given up must not cause a process to start.
      if (request.signal?.aborted === true) throw abortedError(request.capability);

      // Re-checked here, not merely at plan time: a plan built a moment ago can
      // be stale, and the launch that follows is irreversible.
      const permission = this.permissionForCapability(modelId, request.capability);
      if (!permission.enabled) {
        const reason = permission.reason.length > 0 ? permission.reason : 'it is disabled by the capability policy';
        this.emit({ type: 'capability/skipped', capability: request.capability, optionId: modelId, reason });
        failures.push({ modelId, code: 'POLICY_DISABLED', message: reason });
        attemptLog.push({
          optionId: modelId,
          order: planned.option.order,
          outcome: 'failed',
          stage: 'dispatch',
          coldStart: false,
          durationMs: 0,
          code: 'POLICY_DISABLED',
          reason,
        });
        continue;
      }

      if (deadline !== undefined && Date.now() >= deadline) {
        // An exhausted deadline is reported as a limitation of the *plan*, not as
        // a model failure: the remaining options were never tried, and saying
        // otherwise would misattribute the cause.
        failures.push({
          modelId,
          code: 'INVOCATION_TIMEOUT',
          message: `the invocation deadline expired before this option was tried (${queue.length - index} option(s) untried)`,
        });
        break;
      }

      const startedAt = Date.now();
      try {
        const outcome = await this.invokeWithResourceRelease(artifacts, modelId, () =>
          this.dispatchOption(request, plan, planned, artifacts),
        );
        if (outcome.kind === 'unresolved') {
          this.emit({
            type: 'invocation/unresolved',
            modelId,
            capability: request.capability,
            reason: outcome.unresolved.reason,
          });
          attemptLog.push({
            optionId: modelId,
            order: planned.option.order,
            outcome: 'unresolved',
            stage: 'dispatch',
            coldStart: outcome.result.coldStart,
            durationMs: outcome.result.durationMs,
            reason: outcome.unresolved.reason,
          });
          return { ...outcome.result, decision: plan.decision, attempts: attemptLog, unresolved: outcome.unresolved };
        }
        if (index > 0) {
          this.emit({
            type: 'invocation/fellback',
            fromModelId: queue[index - 1]?.option.option.id ?? 'unknown',
            toModelId: modelId,
            reason: failures[failures.length - 1]?.message ?? 'previous option failed',
          });
        }
        attemptLog.push({
          optionId: modelId,
          order: planned.option.order,
          outcome: 'succeeded',
          stage: 'dispatch',
          coldStart: outcome.result.coldStart,
          durationMs: outcome.result.durationMs,
          reason: 'served the request',
        });
        return { ...outcome.result, decision: plan.decision, attempts: attemptLog };
      } catch (error) {
        const described = toHubError(error, 'INVOCATION_FAILED', { modelId });
        // A cancellation is the caller's decision, not a model failure: retrying
        // it on another option would be actively wrong.
        if (described.code === 'INVOCATION_ABORTED') throw described;
        failures.push({ modelId, code: described.code, message: described.message });
        attemptLog.push({
          optionId: modelId,
          order: planned.option.order,
          outcome: 'failed',
          stage: stageForCode(described.code),
          coldStart: false,
          durationMs: Date.now() - startedAt,
          code: described.code,
          reason: described.message,
        });
        this.emit({
          type: 'invocation/failed',
          modelId,
          capability: request.capability,
          code: described.code,
          message: described.message,
        });
        if (described.code === 'START_FAILED' || described.code === 'LIFECYCLE_UNSUPPORTED') {
          this.emit({
            type: 'capability/readiness-failed',
            capability: request.capability,
            optionId: modelId,
            reason: described.message,
          });
        }
        this.log(`hub: ${modelId} failed (${described.code}): ${described.message}`, {
          modelId,
          capability: request.capability,
        });
      }
    }

    const detail = failures.map((failure) => `  - ${failure.modelId}: [${failure.code}] ${failure.message}`).join('\n');
    const untried = queue.length - attemptsAllowed;
    throw new ModelHubError(
      'INVOCATION_FAILED',
      `every candidate model failed for capability "${request.capability}" (${failures.length} attempt(s); ` +
        `${attemptsAllowed} of ${queue.length} permitted option(s) were allowed by the attempt budget):\n${detail}` +
        // A budget that stopped the walk early is reported as a limitation of the
        // plan rather than as a model failure: those options failed nothing,
        // because nothing was asked of them.
        (untried > 0
          ? `\n${untried} permitted option(s) went untried because of the configured attempt budget of ${attemptsAllowed}. ` +
            'Raise it, or clear `attemptBudget`, to let the whole permitted order run.'
          : ''),
      {
        capability: request.capability,
        failures: failures as unknown as Record<string, unknown>[],
        routing: explainDecision(plan.decision, true),
        policyRevision: plan.decision.policyRevision ?? plan.plan.revision,
        permittedOptions: queue.map((entry) => entry.option.option.id),
        skippedOptions: plan.plan.skipped as unknown as Record<string, unknown>[],
        attemptBudget: attemptsAllowed,
        untriedOptions: untried,
      },
    );
  }

  /**
   * Execute one request against exactly one planned option.
   *
   * Two execution paths, chosen by the option's own kind and never by its name:
   * a catalog model goes through its adapter and the runtime gate, and a
   * registered remote option goes through the function its owning plugin
   * supplied. Everything around them — the permission checks, the cancellation
   * checks, the artifact store, the attempt accounting — is shared, which is what
   * makes a subscription backend and a local ComfyUI workflow fall back in one
   * saved order instead of two.
   *
   * @param request - the original request.
   * @param plan - the assembled plan, for the decision being logged.
   * @param planned - the option to run.
   * @param artifacts - the store this call reads and writes through.
   * @returns the result, or an unresolved outcome.
   */
  private async dispatchOption(
    request: InvocationRequest,
    plan: PlannedInvocation,
    planned: PlannedOption,
    artifacts: ArtifactStore,
  ): Promise<DispatchOutcome> {
    // The plan was assembled before this option's turn, and a cold start for an
    // earlier option can take minutes. Re-checked here, immediately before
    // dispatch, so the answer is about *now*: an operator who disabled this
    // option while the previous engine was starting gets what they asked for.
    // A cancellation is checked for the same reason — no work should be
    // submitted on behalf of a caller who has already given up.
    if (request.signal?.aborted === true) throw abortedError(request.capability);
    const permission = this.permissionForCapability(planned.option.option.id, request.capability);
    if (!permission.enabled) {
      throw new ModelHubError(
        'POLICY_DISABLED',
        `${permission.reason.length > 0 ? permission.reason : `"${planned.option.option.id}" is disabled`}. ` +
          'It was permitted when this request was planned and is not any more, so nothing was submitted.',
        { modelId: planned.option.option.id, capability: request.capability },
      );
    }

    return planned.kind === 'option'
      ? this.dispatchRemoteOption(request, plan, planned.option, artifacts)
      : this.dispatchModel(request, plan, planned.option, artifacts);
  }

  /**
   * Run one catalog model: gate its engine, then invoke its adapter.
   *
   * @param request - the original request.
   * @param plan - the assembled plan.
   * @param planned - the permitted option.
   * @param artifacts - the store this call reads and writes through.
   * @returns the result.
   */
  private async dispatchModel(
    request: InvocationRequest,
    plan: PlannedInvocation,
    planned: ResolvedPolicyOption,
    artifacts: ArtifactStore,
  ): Promise<DispatchOutcome> {
    const modelId = planned.option.id;
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

    if (planned.state === 'stopped' && planned.startable) {
      this.emit({
        type: 'capability/launch',
        capability: request.capability,
        optionId: modelId,
        detail: 'stopped and startable; making it ready before dispatch',
      });
    }

    // The cold-start gate. `ensureReady` reuses a warm engine, joins a startup
    // already in progress rather than spawning a second process, and waits for the
    // engine's own readiness check to pass — process creation or an open port is
    // not enough, which is why the descriptors now declare what a real answer
    // looks like.
    const gate = await this.runtime.ensureReady(modelId);
    const started = Date.now();
    this.runtime.beginInvocation(modelId);
    this.emit({
      type: 'invocation/started',
      modelId,
      capability: request.capability,
      coldStart: gate.coldStart,
    });

    let submitted = false;
    const jobIds: Record<string, string> = {};
    try {
      const output = await adapter.invoke({
        model,
        capability: request.capability,
        ...(request.prompt === undefined ? {} : { prompt: request.prompt }),
        inputs: plan.resolved.inputs,
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
        // From here the work may be beyond recall: an image queued in ComfyUI, a
        // request already on the wire. A timeout after this point produces an
        // unresolved outcome instead of a retry, because resubmitting could
        // double-render or double-charge.
        onSubmit: (ids) => {
          submitted = true;
          if (ids !== undefined) Object.assign(jobIds, ids);
        },
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
        { modelId, capability: request.capability, rationale: plan.decision.rationale },
      );
      return { kind: 'result', result };
    } catch (error) {
      if (submitted && isAmbiguousFailure(error)) {
        return this.unresolvedOutcome(request, modelId, error, jobIds, Date.now() - started, gate.coldStart);
      }
      throw error;
    } finally {
      this.runtime.endInvocation(modelId);
    }
  }

  /**
   * Run one registered remote option — a subscription backend, say.
   *
   * The hub contributes the artifact contract and the failure classification; the
   * option contributes the transport. Outputs come back as artifact ids because
   * that is the only currency a later workflow step can consume, so the option is
   * required to have written them into the store it was handed rather than
   * returning a path nobody can chain from.
   *
   * @param request - the original request.
   * @param plan - the assembled plan.
   * @param option - the permitted, permitted-again option.
   * @param artifacts - the store this call reads and writes through.
   * @returns the result.
   */
  private async dispatchRemoteOption(
    request: InvocationRequest,
    plan: PlannedInvocation,
    option: ResolvedPolicyOption,
    artifacts: ArtifactStore,
  ): Promise<DispatchOutcome> {
    const entry = this.capabilityOptions.get(option.option.id);
    if (entry === undefined) {
      throw new ModelHubError(
        'UNSUPPORTED_OPERATION',
        `option "${option.option.id}" is no longer registered, so nothing can execute it`,
        { modelId: option.option.id, capability: request.capability },
      );
    }
    const started = Date.now();
    let submitted = false;
    const jobIds: Record<string, string> = {};
    this.emit({
      type: 'invocation/started',
      modelId: option.option.id,
      capability: request.capability,
      coldStart: false,
    });
    try {
      const output = await entry.invoke(request, {
        capability: request.capability,
        ...(request.prompt === undefined ? {} : { prompt: request.prompt }),
        inputs: plan.resolved.inputs,
        options: request.options ?? {},
        artifacts,
        signal: request.signal ?? new AbortController().signal,
        onSubmit: (ids) => {
          submitted = true;
          if (ids !== undefined) Object.assign(jobIds, ids);
        },
        log: (message, fields) => this.log(`option[${option.option.id}] ${message}`, fields),
      });
      const durationMs = Date.now() - started;
      const outputs: Artifact[] = [];
      for (const id of output.artifactIds) {
        const artifact = await artifacts.get(id);
        if (artifact === undefined) {
          throw new ModelHubError(
            'ARTIFACT_ERROR',
            `option "${option.option.id}" reported artifact "${id}", which is not in this call's artifact store, ` +
              'so no later step could consume it',
            { modelId: option.option.id, artifactId: id },
          );
        }
        outputs.push(artifact);
      }
      const result: InvocationResult = {
        modelId: option.option.id,
        capability: request.capability,
        outputs,
        ...(output.value === undefined ? {} : { value: output.value }),
        durationMs,
        coldStart: false,
      };
      this.emit({
        type: 'invocation/succeeded',
        modelId: option.option.id,
        capability: request.capability,
        durationMs,
        outputCount: outputs.length,
      });
      this.log(`hub: ${request.capability} served by ${option.option.id} in ${durationMs} ms (${outputs.length} output(s))`, {
        modelId: option.option.id,
        capability: request.capability,
        rationale: plan.decision.rationale,
      });
      return { kind: 'result', result };
    } catch (error) {
      if (submitted && isAmbiguousFailure(error)) {
        return this.unresolvedOutcome(request, option.option.id, error, jobIds, Date.now() - started, false);
      }
      throw error;
    }
  }

  /**
   * Package a post-submission ambiguity as an explicit unresolved outcome.
   *
   * The alternative — reporting a failure and letting the plan try the next
   * option — is how a duplicate gets produced: the first backend may well have
   * completed the work, and the caller would be paying for it twice without
   * knowing. So the plan stops here, and the result carries whatever identifiers
   * can be used to look the work up.
   *
   * @param request - the original request.
   * @param modelId - the backend that may or may not have produced something.
   * @param error - the ambiguous failure.
   * @param jobIds - identifiers reported before the failure.
   * @param durationMs - how long the attempt ran.
   * @param coldStart - whether an engine was started for it.
   * @returns the unresolved outcome.
   */
  private unresolvedOutcome(
    request: InvocationRequest,
    modelId: string,
    error: unknown,
    jobIds: Readonly<Record<string, string>>,
    durationMs: number,
    coldStart: boolean,
  ): DispatchOutcome {
    const described = toHubError(error, 'INVOCATION_FAILED', { modelId });
    const unresolved: UnresolvedOutcome = {
      reason: described.message,
      modelId,
      submitted: true,
      ...(Object.keys(jobIds).length === 0 ? {} : { jobIds: { ...jobIds } }),
      guidance:
        'The request reached the backend and its outcome could not be determined, so nothing was retried and no ' +
        'duplicate was created. Check the backend for the work described above (by the identifiers listed, when ' +
        'they are present) before asking for it again.',
    };
    this.log(`hub: ${request.capability} outcome for ${modelId} is unresolved: ${described.message}`, {
      modelId,
      capability: request.capability,
    });
    return {
      kind: 'unresolved',
      unresolved,
      result: {
        modelId,
        capability: request.capability,
        outputs: [],
        durationMs,
        coldStart,
      },
    };
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
  private async invokeWithResourceRelease<T>(
    artifacts: ArtifactStore,
    modelId: string,
    run: () => Promise<T>,
  ): Promise<T> {
    void artifacts;
    try {
      return await run();
    } finally {
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
  private invalidateResources(modelId: string): void {
    const current = this.resources;
    if (current === undefined) return;
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
function resolveProbeOptions(options: ModelHubOptions): MachineProbeOptions | undefined {
  if (options.probeResources === false) return undefined;
  if (options.machine !== undefined) return undefined;
  if (options.probeResources === true || options.probeResources === undefined) return {};
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
// Public model views never serialize the trusted workflow graph.
function publicModelView<T extends { readonly adapter: string; readonly adapterConfig: Readonly<Record<string, unknown>>; readonly descriptor: ModelCatalogConfig['models'][number] }>(model: T): T {
  if (model.adapter !== 'comfyui') return model;
  return {
    ...model,
    adapterConfig: {},
    descriptor: { ...model.descriptor, adapterConfig: {} },
  };
}

function withComfyCatalogDir(hosts: readonly ModelHost[], catalogDir: string | undefined): readonly ModelHost[] {
  if (catalogDir === undefined) return hosts;
  return hosts.map((host) => host.adapter !== 'comfyui' ? host : {
    ...host, adapterConfig: { ...(host.adapterConfig ?? {}), catalogDir },
  });
}

function defaultDiscoverers(): readonly HostDiscoverer[] {
  return [createOllamaDiscoverer(), createComfyUiDiscoverer(), createA1111Discoverer(), createThreeDDiscoverer()];
}

/** Re-export so callers can construct a hub with a silent logger without another import. */
export { silentLogger };
export type { AvailabilityState };
