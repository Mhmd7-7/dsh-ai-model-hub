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

/** One accelerators's memory, as the probe reported it. */
export interface GpuInfo {
  /** Device name as the driver reports it, e.g. `NVIDIA GeForce RTX 5060 Laptop GPU`. */
  readonly name: string;
  /** Total device memory in gibibytes. */
  readonly vramGb: number;
  /** Free device memory in gibibytes at probe time, when the probe could tell. */
  readonly freeVramGb?: number;
}

/**
 * A model's share of the machine, in gibibytes.
 *
 * Used by {@link reserveResources} to subtract what is already running from what
 * the machine has, so the router can answer "does this fit *now*" rather than
 * "would this fit on an idle machine".
 */
export interface MachineResourceUse {
  /** GPU memory the resident model holds. */
  readonly vramGb?: number;
  /** System memory the resident model holds. */
  readonly ramGb?: number;
}

/**
 * What the machine can offer, as far as the hub can tell.
 *
 * Two pairs of numbers exist on purpose, and conflating them is the mistake this
 * type is shaped to prevent:
 *
 * - {@link vramGb} / {@link ramGb} are *capacity*: what the machine has in total.
 *   They answer "is this model ever runnable here?".
 * - {@link availableVramGb} / {@link availableRamGb} are *headroom*: what is free
 *   right now. They answer "can this model run here without evicting something
 *   else?".
 *
 * Only the probe can know the second pair; a hand-supplied profile may omit it,
 * in which case resource checks fall back to capacity and say so.
 */
export interface MachineProfile {
  /** Total GPU VRAM in gibibytes, summed over detected devices. */
  readonly vramGb: number;
  /** Total system RAM in gibibytes. */
  readonly ramGb: number;
  /** Whether any CUDA-capable GPU was detected. */
  readonly hasGpu: boolean;
  /** Human-readable description of what was detected, for diagnostics. */
  readonly notes: string;
  /** Free GPU VRAM in gibibytes at probe time, when the probe could tell. */
  readonly availableVramGb?: number;
  /** Free system RAM in gibibytes at probe time, when the probe could tell. */
  readonly availableRamGb?: number;
  /** Free disk space where the hub writes, in gibibytes, when probed. */
  readonly availableDiskGb?: number;
  /** Every detected accelerator, in driver order. */
  readonly gpus?: readonly GpuInfo[];
  /** `process.platform` at probe time, e.g. `win32`, `darwin`, `linux`. */
  readonly platform?: string;
  /** `process.arch` at probe time, e.g. `x64`, `arm64`. */
  readonly arch?: string;
  /** When the probe ran, as Unix epoch milliseconds. */
  readonly probedAt?: number;
}

/**
 * Subtract what is already resident from what the machine has.
 *
 * Only the *available* pair is reduced. Capacity is a property of the hardware
 * and does not change because a model is loaded — rewriting it would be the
 * "claiming more than the machine offers" failure the probe is written to avoid,
 * in reverse.
 *
 * An undefined available value stays undefined: "we never measured it" is not
 * the same fact as "there is none left", and the caller must be able to tell.
 *
 * @param profile - the probed profile.
 * @param use - what the resident models hold.
 * @returns a profile whose available figures exclude `use`.
 */
export function reserveResources(profile: MachineProfile, use: MachineResourceUse): MachineProfile {
  const reserved: {
    availableVramGb?: number;
    availableRamGb?: number;
  } = {};
  if (profile.availableVramGb !== undefined) {
    reserved.availableVramGb = Math.max(0, round1(profile.availableVramGb - (use.vramGb ?? 0)));
  }
  if (profile.availableRamGb !== undefined) {
    reserved.availableRamGb = Math.max(0, round1(profile.availableRamGb - (use.ramGb ?? 0)));
  }
  return { ...profile, ...reserved };
}

/**
 * Round to one decimal place, the precision every resource figure is reported at.
 * @param value - the number to round.
 * @returns the rounded value.
 */
function round1(value: number): number {
  return Math.round(value * 10) / 10;
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
