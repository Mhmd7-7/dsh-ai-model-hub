/**
 * The Model Router.
 *
 * The router's job is to answer one question deterministically: *given this
 * request, which registered model should serve it?* It filters by capability,
 * then by declared input kinds, then by tags, then by live availability and
 * resource fit, and finally sorts what remains by a total order.
 *
 * The rule this file exists to obey: **no model-specific logic, ever.** There is
 * no `if (modelId === 'sdxl')` here, no engine name, no capability-specific
 * branch beyond a generic input-kind check. Every decision reads a declared fact
 * off a descriptor. That is what makes "add a model without changing the router"
 * literally true rather than aspirational — the router cannot mention a model it
 * has never heard of.
 *
 * Determinism is a requirement, not a nicety. Given the same catalog and the same
 * request, the router returns the same model, so a routing surprise is
 * reproducible and explainable rather than a flaky heisenbug.
 *
 * @module dsh-ai-model-hub/router/router
 */

import type { Artifact, ArtifactStore } from '../artifacts/types.ts';
import { describeArtifact } from '../artifacts/types.ts';
import type { Capability, IoType } from '../catalog/capabilities.ts';
import { IO_TYPES, isIoType } from '../catalog/capabilities.ts';
import type { ResolvedModel } from '../catalog/descriptor.ts';
import type { ModelCatalog } from '../catalog/registry.ts';
import type { AvailabilityState, ArtifactInput, InvocationRequest, RoutingCandidate, RoutingDecision } from '../types.ts';
import { ModelHubError } from '../errors.ts';
import type { RuntimeManager } from '../runtime/manager.ts';

/** Everything the router needs to resolve a request's inputs. */
export interface RoutingContext {
  /** The catalog of declared models. */
  readonly catalog: ModelCatalog;
  /** Live state, for availability filtering. */
  readonly runtime: RuntimeManager;
  /** Where input artifacts are resolved from. */
  readonly artifacts: ArtifactStore;
}

/** A request with its inputs resolved and its declared input kinds derived. */
export interface ResolvedRequest {
  /** The original request. */
  readonly request: InvocationRequest;
  /** The concrete input artifacts, in caller order. */
  readonly inputs: readonly Artifact[];
  /** The distinct artifact kinds the request supplies. */
  readonly kindSet: ReadonlySet<IoType>;
  /** Whether the request carries non-empty prompt text. */
  readonly hasPrompt: boolean;
}

/** Tunable routing behaviour. Defaults are the shipping policy. */
export interface RoutingPolicy {
  /**
   * Whether a model that is merely `stopped` is a candidate.
   *
   * True by default: a cold model that the runtime can start is a perfectly good
   * choice, and refusing it would make the hub useless on a laptop where nothing
   * runs until asked. Set false to route only among already-warm models.
   */
  readonly allowColdStarts: boolean;
  /**
   * Model ids to exclude. Lets an operator quarantine a model without editing
   * its descriptor — the difference between "this model is broken" and "this
   * model is not right now".
   */
  readonly excludedModelIds: readonly string[];
}

/** The shipping routing policy. */
export const DEFAULT_ROUTING_POLICY: RoutingPolicy = {
  allowColdStarts: true,
  excludedModelIds: [],
};

/**
 * Resolve a request's declared inputs against the artifact store.
 *
 * This is where "the caller passed an artifact id" becomes "these are images".
 * Type-checking here rather than inside an adapter means every adapter gets the
 * same guarantee, and a mismatch is reported in the vocabulary the caller used
 * (artifact kinds) instead of as an engine error.
 *
 * @param context - catalog, runtime, and artifact store.
 * @param request - the caller's request.
 * @returns the request with its inputs resolved.
 * @throws ModelHubError with `ARTIFACT_ERROR` for unknown or mistyped inputs.
 */
export async function resolveRequestInputs(
  context: Pick<RoutingContext, 'artifacts'>,
  request: InvocationRequest,
): Promise<ResolvedRequest> {
  const inputs: Artifact[] = [];
  const kindSet = new Set<IoType>();
  const declared = request.inputs ?? [];

  for (const entry of declared) {
    const reference = normalizeArtifactInput(entry);
    const artifact = await context.artifacts.get(reference.id);
    if (artifact === undefined) {
      throw new ModelHubError(
        'ARTIFACT_ERROR',
        `input artifact "${reference.id}" does not exist. ` +
          'Artifact ids come from a previous invocation result or from list_artifacts.',
        { artifactId: reference.id, capability: request.capability },
      );
    }
    if (reference.type !== undefined && reference.type.length > 0) {
      // The declared kind is model-supplied, so it is checked against the
      // vocabulary before it is compared — an unknown kind is a caller mistake
      // worth naming precisely, not a silent non-match.
      if (!isIoType(reference.type)) {
        throw new ModelHubError(
          'ARTIFACT_ERROR',
          `input artifact "${artifact.id}" was declared as type "${reference.type}", which is not an artifact kind`,
          { artifactId: artifact.id, declaredType: reference.type, knownKinds: IO_TYPES },
        );
      }
      if (reference.type !== artifact.type) {
        throw new ModelHubError(
          'ARTIFACT_ERROR',
          `input artifact "${artifact.id}" is of type "${artifact.type}" but was declared as "${reference.type}"`,
          { artifactId: artifact.id, declaredType: reference.type, actualType: artifact.type },
        );
      }
      kindSet.add(reference.type);
    } else {
      kindSet.add(artifact.type);
    }
    inputs.push(artifact);
  }

  const hasPrompt = request.prompt !== undefined && request.prompt.trim().length > 0;
  return { request, inputs, kindSet, hasPrompt };
}

/**
 * Accept either a bare artifact id or an `{ id, type }` object as an input.
 * @param entry - the caller-supplied input.
 * @returns the normalized reference.
 */
function normalizeArtifactInput(entry: ArtifactInput | string): ArtifactInput {
  if (typeof entry === 'string') {
    return { id: entry };
  }
  return entry;
}

/**
 * Why a candidate was rejected, or how strongly it was preferred.
 *
 * A numeric score is used rather than a comparator chain so the reason attached
 * to each candidate can explain the exact contribution that decided the outcome.
 * Lower is better.
 */
interface Scored {
  readonly model: ResolvedModel;
  readonly score: number;
  readonly notes: string[];
}

/**
 * Choose the model that should serve a request.
 *
 * @param context - catalog, runtime, and artifact store.
 * @param request - the resolved request (see {@link resolveRequestInputs}).
 * @param policy - routing policy overrides.
 * @returns the decision, including every candidate considered and why.
 * @throws ModelHubError with `NO_COMPATIBLE_MODEL` when nothing can serve it.
 */
export function routeRequest(
  context: RoutingContext,
  request: ResolvedRequest,
  policy: RoutingPolicy = DEFAULT_ROUTING_POLICY,
): RoutingDecision {
  const { capability } = request.request;
  const excluded = new Set(policy.excludedModelIds);
  const pinned = request.request.modelId;
  const requiredTags = request.request.requiredTags ?? [];

  const considered: RoutingCandidate[] = [];
  const eligible: Scored[] = [];

  for (const model of context.catalog.listModels()) {
    const verdict = evaluateCandidate(model, request, {
      policy,
      excluded,
      pinned,
      requiredTags,
      availability: context.runtime.checkAvailability(model.id),
      resourceFit: context.catalog.checkResources(model),
    });
    if (verdict.eligible) {
      eligible.push(verdict.scored);
      considered.push({
        modelId: model.id,
        eligible: true,
        reason: verdict.scored.notes.join('; '),
        score: verdict.scored.score,
      });
    } else {
      considered.push({ modelId: model.id, eligible: false, reason: verdict.reason });
    }
  }

  if (eligible.length === 0) {
    const offered = context.catalog.findModelsByCapabilityIncludingDisabled(capability);
    throw new ModelHubError(
      'NO_COMPATIBLE_MODEL',
      buildNoCandidateMessage(capability, considered, offered.length, request),
      {
        capability,
        requiredInputKinds: [...request.kindSet],
        requiresPrompt: request.hasPrompt,
        candidates: considered.map((candidate) => ({
          modelId: candidate.modelId,
          reason: candidate.reason,
        })) as unknown as Record<string, unknown>[],
        modelsDeclaringCapability: offered.map((model) => model.id),
      },
    );
  }

  eligible.sort((left, right) => {
    if (left.score !== right.score) return left.score - right.score;
    return left.model.id.localeCompare(right.model.id);
  });

  const winner = eligible[0];
  if (winner === undefined) {
    throw new ModelHubError('NO_COMPATIBLE_MODEL', `no model could serve "${capability}"`, { capability });
  }

  return {
    modelId: winner.model.id,
    candidates: considered,
    rationale: `${winner.model.id} (score ${winner.score}): ${winner.notes.join('; ')}`,
  };
}

/** Inputs to one candidate's evaluation, bundled to keep the call site readable. */
interface EvaluationInputs {
  readonly policy: RoutingPolicy;
  readonly excluded: ReadonlySet<string>;
  readonly pinned: string | undefined;
  readonly requiredTags: readonly string[];
  readonly availability: AvailabilityState;
  readonly resourceFit: { supported: boolean; reason?: string };
}

/**
 * Decide whether one model can serve the request, and how well.
 *
 * Every check reads a declared fact. The only behaviour that is not a pure
 * descriptor read is the availability check, and that deliberately treats
 * startable-but-stopped models as candidates so a cold machine still routes.
 *
 * @param model - the candidate.
 * @param request - the resolved request.
 * @param inputs - policy, pin, and live state.
 * @returns either an eligible score or a rejection reason.
 */
function evaluateCandidate(
  model: ResolvedModel,
  request: ResolvedRequest,
  inputs: EvaluationInputs,
): { eligible: true; scored: Scored } | { eligible: false; reason: string } {
  const capability = request.request.capability;
  const notes: string[] = [];

  if (inputs.pinned !== undefined && model.id !== inputs.pinned) {
    return { eligible: false, reason: `not the pinned model (${inputs.pinned})` };
  }

  if (!model.enabled) {
    return { eligible: false, reason: 'disabled in configuration' };
  }

  if (inputs.excluded.has(model.id)) {
    return { eligible: false, reason: 'excluded by routing policy' };
  }

  if (!model.capabilities.includes(capability)) {
    return { eligible: false, reason: `does not declare capability "${capability}"` };
  }

  for (const tag of inputs.requiredTags) {
    if (!model.tags.includes(tag)) {
      return { eligible: false, reason: `missing required tag "${tag}"` };
    }
  }

  // Declared input kinds must cover everything the request supplies.
  for (const kind of request.kindSet) {
    if (!model.inputTypes.includes(kind)) {
      return {
        eligible: false,
        reason: `accepts ${model.inputTypes.join(', ')} but the request supplies a "${kind}"`,
      };
    }
  }

  if (!inputs.resourceFit.supported) {
    return { eligible: false, reason: inputs.resourceFit.reason ?? 'exceeds this machine' };
  }

  switch (inputs.availability) {
    case 'available':
      notes.push('healthy and ready');
      break;
    case 'stopped':
      if (!inputs.policy.allowColdStarts) {
        return { eligible: false, reason: 'not running and cold starts are disabled by policy' };
      }
      notes.push('cold but startable');
      break;
    case 'starting':
      if (!inputs.policy.allowColdStarts) {
        return { eligible: false, reason: 'currently starting and cold starts are disabled by policy' };
      }
      notes.push('currently starting');
      break;
    case 'unhealthy':
      // Deliberately rejected: routing into a known-broken process turns a clean
      // "no model available" into a confusing timeout.
      return { eligible: false, reason: 'running but failing its health check' };
    case 'disabled':
      return { eligible: false, reason: 'disabled in configuration' };
    case 'unsupported':
      return { eligible: false, reason: 'not supported on this machine' };
    case 'error':
      return { eligible: false, reason: 'in an error state' };
    default: {
      const exhaustive: never = inputs.availability;
      return { eligible: false, reason: `unhandled availability ${String(exhaustive)}` };
    }
  }

  // Ordering contributions. Smaller is better; the weights are policy, not
  // model knowledge, so they apply identically to every model ever registered.
  const score = model.priority;
  notes.push(`priority ${model.priority}`);

  return { eligible: true, scored: { model, score, notes } };
}

/**
 * Compose an actionable message when nothing can serve the request.
 *
 * The failure modes here are the ones an operator actually hits — the model is
 * disabled, the artifact kind is wrong, the machine is too small — so the message
 * leads with the specific near-misses rather than the full rejection table.
 *
 * @param capability - the requested capability.
 * @param candidates - every considered model with its verdict.
 * @param declaringCount - how many models declare the capability at all.
 * @param request - the resolved request.
 * @returns a multi-line explanation.
 */
function buildNoCandidateMessage(
  capability: Capability,
  candidates: readonly RoutingCandidate[],
  declaringCount: number,
  request: ResolvedRequest,
): string {
  const lines = [`no model can serve capability "${capability}" for this request.`];
  if (declaringCount === 0) {
    lines.push(
      `No configured model declares "${capability}" at all. Add one to config/models.json, ` +
        'or check `list_models` for the capabilities this deployment does offer.',
    );
    return lines.join(' ');
  }
  lines.push(`${declaringCount} model(s) declare it, but none passed filtering:`);
  for (const candidate of candidates) {
    if (candidate.eligible) continue;
    lines.push(`  - ${candidate.modelId}: ${candidate.reason}`);
  }
  if (request.kindSet.size > 0) {
    lines.push(`This request supplied input kinds: ${[...request.kindSet].join(', ')}.`);
  }
  return lines.join('\n');
}

/**
 * Describe the request's content for a log line or a tool result.
 *
 * @param request - the resolved request.
 * @returns a one-line summary.
 */
export function describeResolvedRequest(request: ResolvedRequest): string {
  const parts = [`capability=${request.request.capability}`];
  if (request.hasPrompt) {
    const prompt = request.request.prompt ?? '';
    parts.push(`prompt="${prompt.length > 60 ? `${prompt.slice(0, 57)}...` : prompt}"`);
  }
  if (request.inputs.length > 0) {
    parts.push(`inputs=[${request.inputs.map((artifact) => artifact.id).join(', ')}]`);
  }
  return parts.join(' ');
}

/**
 * Render a routing decision for humans and for the agent.
 *
 * @param decision - the decision to explain.
 * @param verbose - include rejected candidates in full.
 * @returns a multi-line explanation.
 */
export function explainDecision(decision: RoutingDecision, verbose = false): string {
  const lines = [`chose ${decision.modelId}: ${decision.rationale}`];
  if (verbose) {
    for (const candidate of decision.candidates) {
      const marker = candidate.eligible ? 'eligible' : 'rejected';
      lines.push(`  [${marker}] ${candidate.modelId}: ${candidate.reason}`);
    }
  }
  return lines.join('\n');
}

/**
 * Render one artifact for inclusion in an invocation result.
 * @param artifact - the artifact to render.
 * @returns a one-line handle.
 */
export function artifactHandle(artifact: Artifact): string {
  return describeArtifact(artifact);
}
