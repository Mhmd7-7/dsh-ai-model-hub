/**
 * The hub's shared runtime vocabulary.
 *
 * These types describe the *host* side of a model — is it configured, is it
 * running, is it healthy, may the router pick it — as opposed to the *descriptor*
 * side, which is static configuration.
 *
 * @module dsh-ai-model-hub/types
 */

import type { Artifact } from './artifacts/types.ts';
import type { Capability, IoType } from './catalog/capabilities.ts';
import type { ResolvedModel } from './catalog/descriptor.ts';
import type { JsonValue } from './util/validate.ts';

/**
 * Whether a model can serve work right now.
 *
 * Kept distinct from {@link LifecycleState} because they answer different
 * questions: `availability` is the routing input ("may I choose this?"), while
 * `lifecycle` is the management input ("is there a process, and do I own it?").
 * A model with an external endpoint the hub never started is `available` with
 * `lifecycle: 'external'`.
 */
export const AVAILABILITY_STATES = [
  /** Ready to serve. */
  'available',
  /** Known and configured, but its process is not running. Startable models live here. */
  'stopped',
  /** A start or health probe is in flight. */
  'starting',
  /** Running but its health check fails. */
  'unhealthy',
  /** Disabled in configuration; never selected. */
  'disabled',
  /** This machine cannot satisfy the model's declared resource needs. */
  'unsupported',
  /** The descriptor is broken or its adapter is missing. */
  'error',
] as const;

/** One availability state. */
export type AvailabilityState = (typeof AVAILABILITY_STATES)[number];

/** Which process, if any, is behind a model. */
export const LIFECYCLE_STATES = [
  /** No process; nothing known about liveness. */
  'not_running',
  /** A process owned by the hub is starting. */
  'starting',
  /** A process owned by the hub is running. */
  'running',
  /** A process is stopping. */
  'stopping',
  /** A process failed and the hub gave up on it. */
  'failed',
  /** Reached over the network; the hub did not start it and does not manage it. */
  'external',
] as const;

/** One lifecycle state. */
export type LifecycleState = (typeof LIFECYCLE_STATES)[number];

/** The outcome of the most recent health probe. */
export interface HealthReport {
  /** Whether the probe passed. */
  readonly healthy: boolean;
  /** When the probe ran, as Unix epoch milliseconds. */
  readonly checkedAt: number;
  /** How long the probe took, in milliseconds. */
  readonly latencyMs?: number;
  /** Why it failed, when it did. */
  readonly detail?: string;
}

/** What the runtime manager knows about one model right now. */
export interface ModelRuntimeStatus {
  /** The model this describes. */
  readonly modelId: string;
  /** Whether the router may select it. */
  readonly availability: AvailabilityState;
  /** Which process, if any, is behind it. */
  readonly lifecycle: LifecycleState;
  /** The last health probe's result, when one has run. */
  readonly health?: HealthReport;
  /** OS process id of a hub-owned process. */
  readonly pid?: number;
  /** When the current process started, as Unix epoch milliseconds. */
  readonly startedAt?: number;
  /** Why the model is not `available`, when it is not. */
  readonly reason?: string;
  /** In-flight invocations, for observability. */
  readonly activeInvocations: number;
}

/** A model as reported to callers: static facts plus live state. */
export interface ModelView {
  /** The resolved descriptor. */
  readonly model: ResolvedModel;
  /** Live runtime state. */
  readonly status: ModelRuntimeStatus;
}

/** What the machine can offer, as far as the hub can tell. */
export interface MachineProfile {
  /** Total GPU VRAM in gibibytes, summed over detected devices. */
  readonly vramGb: number;
  /** Total system RAM in gibibytes. */
  readonly ramGb: number;
  /** Whether any CUDA-capable GPU was detected. */
  readonly hasGpu: boolean;
  /** Human-readable description of what was detected, for diagnostics. */
  readonly notes: string;
}

/**
 * How a caller asks for work to be done.
 *
 * Note what is *absent*: no model id, no endpoint, no launch command. The caller
 * states a capability and content; the router decides everything else. That
 * absence is the design's point.
 */
export interface InvocationRequest {
  /** The capability the caller needs. */
  readonly capability: Capability;
  /** Natural-language instruction, when the capability takes one. */
  readonly prompt?: string;
  /**
   * Artifacts to consume, as ids returned by a previous invocation or
   * `getArtifact`. A bare id is accepted alongside the explicit
   * {@link ArtifactInput} form so the common case stays terse.
   */
  readonly inputs?: readonly (ArtifactInput | string)[];
  /** Capability-specific settings, e.g. `{ width: 1024, steps: 30 }`. */
  readonly options?: Readonly<Record<string, unknown>>;
  /** Pin execution to one model. Bypasses routing; intended for debugging and tests. */
  readonly modelId?: string;
  /** Restrict routing to models carrying all of these tags. */
  readonly requiredTags?: readonly string[];
  /** Milliseconds before the invocation is abandoned. Defaults to the model's own budget. */
  readonly timeoutMs?: number;
  /** Caller cancellation. */
  readonly signal?: AbortSignal;
}

/**
 * One artifact offered as input, given either as a bare id or as id plus an
 * expected kind.
 *
 * `type` is a plain string rather than the {@link IoType} union on purpose: this
 * shape is reachable from model-generated tool arguments, so the declared kind is
 * untrusted input. The router validates it against the artifact vocabulary and
 * reports an unknown kind as a clear error, which is better than a type assertion
 * that would let an invalid kind reach an adapter.
 */
export interface ArtifactInput {
  /** The artifact id. */
  readonly id: string;
  /** The kind the caller believes this is; validated when present. */
  readonly type?: string;
}

/** What an invocation produced. */
export interface InvocationResult {
  /** The model that actually ran. */
  readonly modelId: string;
  /** The capability that was served. */
  readonly capability: Capability;
  /** Artifacts produced, in the order the adapter reported them. */
  readonly outputs: readonly Artifact[];
  /**
   * Machine-readable structured outcome the adapter returned, when any.
   *
   * Typed as lossless JSON because this value rides a tool result into the
   * durable session log; the hub validates the adapter's return against that
   * guarantee rather than trusting it.
   */
  readonly value?: Readonly<Record<string, JsonValue>>;
  /** Wall-clock duration of the invocation, in milliseconds. */
  readonly durationMs: number;
  /** Whether the model was started as part of this invocation. */
  readonly coldStart: boolean;
}

/**
 * What the router decided and why.
 *
 * Returned alongside every routed invocation. The agent gets the outcome; an
 * operator (or a test) gets the reasoning, and the two can never disagree
 * because they come from one decision.
 */
export interface RoutingDecision {
  /** The chosen model id. */
  readonly modelId: string;
  /** Every candidate considered, with the verdict for each. */
  readonly candidates: readonly RoutingCandidate[];
  /** Why the chosen model won, in one line. */
  readonly rationale: string;
}

/** One model's fate during routing, kept for explainability. */
export interface RoutingCandidate {
  /** The candidate model id. */
  readonly modelId: string;
  /** Whether it survived filtering. */
  readonly eligible: boolean;
  /** Why it was rejected, or why it was a weaker choice than the winner. */
  readonly reason: string;
  /** The candidate's sort key, present only when eligible. */
  readonly score?: number;
}

/** Options accepted by the catalog constructor. */
export interface CatalogOptions {
  /** Override the machine profile, for tests and for operators who know better. */
  readonly machine?: MachineProfile;
  /** Emit diagnostic lines. Defaults to silence. */
  readonly log?: (message: string) => void;
}
