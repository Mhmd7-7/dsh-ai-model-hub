/**
 * The Model Capability Catalog.
 *
 * The catalog answers one question well: *what can this machine do?* It holds
 * every model's static identity, validates it, and exposes capability-first
 * lookups. It deliberately knows nothing about processes (that is the runtime
 * manager) and nothing about choosing between candidates (that is the router).
 *
 * Keeping those three apart is what lets you replace the scheduler, or the
 * process supervisor, without touching model definitions — and add a model
 * without touching any of the three.
 *
 * @module dsh-ai-model-hub/catalog/registry
 */

import type { Capability, IoType, ModelType } from './capabilities.ts';
import { CAPABILITY_IO, CAPABILITIES, isCapability } from './capabilities.ts';
import type {
  ModelCatalogConfig,
  ModelDescriptor,
  ModelHost,
  ResolvedModel,
} from './descriptor.ts';
import { resolveDescriptor } from './descriptor.ts';
import type { CatalogOptions, MachineProfile } from '../types.ts';
import { ModelHubError } from '../errors.ts';
import { formatIssues, type ValidationIssue } from '../util/validate.ts';
import { parseModelCatalogConfig } from './descriptor.ts';

/** One capability and which models can serve it. */
export interface CapabilityView {
  /** The capability name. */
  readonly capability: Capability;
  /** What it consumes by convention. */
  readonly inputTypes: readonly IoType[];
  /** What it produces by convention. */
  readonly outputTypes: readonly IoType[];
  /** Ids of every *enabled* model declaring it, best-priority first. */
  readonly modelIds: readonly string[];
}

/**
 * The in-memory registry of models.
 *
 * Instances are cheap; the hub builds one at startup and can rebuild it on
 * configuration change. All lookups are synchronous because publication order
 * matters to the router: it needs a stable, ordered candidate list per decision,
 * not a stream.
 */
export class ModelCatalog {
  private readonly hosts = new Map<string, ModelHost>();
  private readonly models = new Map<string, ResolvedModel>();
  private readonly byCapability = new Map<Capability, string[]>();
  private readonly machine: MachineProfile;
  private readonly log: (message: string) => void;
  private readonly diagnostics: { severity: 'warning' | 'error'; message: string }[] = [];

  /**
   * @param config - the validated catalog document.
   * @param options - machine profile override and diagnostics sink.
   */
  constructor(config: ModelCatalogConfig, options: CatalogOptions = {}) {
    this.machine = options.machine ?? { vramGb: 0, ramGb: 0, hasGpu: false, notes: 'not probed' };
    this.log = options.log ?? ((): void => {});

    for (const host of config.hosts ?? []) this.hosts.set(host.id, host);

    for (const descriptor of config.models) {
      let resolved: ResolvedModel;
      try {
        resolved = resolveDescriptor(descriptor, descriptor.host === undefined ? undefined : this.hosts.get(descriptor.host));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        this.diagnostics.push({ severity: 'error', message });
        this.log(`catalog: skipping model: ${message}`);
        continue;
      }
      if (this.models.has(resolved.id)) {
        const message = `duplicate model id "${resolved.id}"`;
        this.diagnostics.push({ severity: 'error', message });
        continue;
      }
      this.models.set(resolved.id, resolved);
      for (const capability of resolved.capabilities) {
        const bucket = this.byCapability.get(capability);
        if (bucket === undefined) this.byCapability.set(capability, [resolved.id]);
        else bucket.push(resolved.id);
      }
    }

    // Sort each capability bucket once, so routing order is stable and cheap.
    for (const [capability, ids] of this.byCapability) {
      ids.sort((left, right) => this.compareForRouting(left, right, capability));
    }

    this.log(
      `catalog: ${this.models.size} model(s), ${this.byCapability.size} capability(ies) from ${this.hosts.size} host(s)`,
    );
  }

  /**
   * Build a catalog from an untrusted configuration document.
   * @param raw - parsed JSON, typically from `config/models.json`.
   * @param label - a label for validation messages, usually the file path.
   * @param options - machine profile and diagnostics sink.
   * @returns the catalog.
   * @throws ModelHubError with `INVALID_DESCRIPTOR` listing every problem found.
   */
  static fromConfig(
    raw: unknown,
    label = 'model catalog',
    options: CatalogOptions = {},
  ): ModelCatalog {
    const parsed = parseModelCatalogConfig(raw, label);
    if (!parsed.ok) {
      throw new ModelHubError('INVALID_DESCRIPTOR', parsed.message, {
        issues: parsed.issues.map((issue) => ({ path: issue.path, message: issue.message })) as unknown as Record<
          string,
          unknown
        >[],
      });
    }
    return new ModelCatalog(parsed.config, options);
  }

  /**
   * Validate a configuration document without building a catalog.
   * @param raw - parsed JSON.
   * @param label - a label for validation messages.
   * @returns every problem found; an empty array means the document is valid.
   */
  static validate(raw: unknown, label = 'model catalog'): readonly ValidationIssue[] {
    const parsed = parseModelCatalogConfig(raw, label);
    return parsed.ok ? [] : parsed.issues;
  }

  /** Every configured model, in configuration order. */
  listModels(): readonly ResolvedModel[] {
    return [...this.models.values()];
  }

  /** Ids of every configured model, in configuration order. */
  listModelIds(): readonly string[] {
    return [...this.models.keys()];
  }

  /**
   * Look up one model by id.
   * @param modelId - the model id.
   * @returns the resolved model, or `undefined`.
   */
  getModel(modelId: string): ResolvedModel | undefined {
    return this.models.get(modelId);
  }

  /**
   * Look up one model by id, failing loudly when it is absent.
   * @param modelId - the model id.
   * @returns the resolved model.
   * @throws ModelHubError with `MODEL_NOT_FOUND`.
   */
  requireModel(modelId: string): ResolvedModel {
    const model = this.models.get(modelId);
    if (model === undefined) {
      throw new ModelHubError('MODEL_NOT_FOUND', `no model with id "${modelId}" is registered`, {
        modelId,
        knownIds: this.listModelIds(),
      });
    }
    return model;
  }

  /**
   * Every *enabled* model declaring a capability, best candidate first.
   *
   * Ordering is deterministic — priority ascending, then id — so a given catalog
   * always produces the same candidate list. Determinism is a requirement, not a
   * nicety: without it, "the router picked a different model" is unreproducible.
   *
   * @param capability - the capability to search for.
   * @returns matching resolved models.
   */
  findModelsByCapability(capability: Capability): readonly ResolvedModel[] {
    const ids = this.byCapability.get(capability) ?? [];
    const result: ResolvedModel[] = [];
    for (const id of ids) {
      const model = this.models.get(id);
      if (model !== undefined && model.enabled) result.push(model);
    }
    return result;
  }

  /**
   * Every model declaring a capability, including disabled ones.
   *
   * Used by management surfaces, which must be able to explain *why* a
   * capability is unavailable rather than silently omitting a model.
   *
   * @param capability - the capability to search for.
   * @returns matching resolved models, enabled first.
   */
  findModelsByCapabilityIncludingDisabled(capability: Capability): readonly ResolvedModel[] {
    const ids = this.byCapability.get(capability) ?? [];
    const models: ResolvedModel[] = [];
    for (const id of ids) {
      const model = this.models.get(id);
      if (model !== undefined) models.push(model);
    }
    return models.sort((left, right) => {
      if (left.enabled !== right.enabled) return left.enabled ? -1 : 1;
      return this.compareForRouting(left.id, right.id, capability);
    });
  }

  /**
   * Every capability at least one enabled model declares.
   * @returns capability views, in the canonical vocabulary order.
   */
  listCapabilities(): readonly CapabilityView[] {
    const views: CapabilityView[] = [];
    for (const capability of CAPABILITIES) {
      const modelIds = this.findModelsByCapability(capability).map((model) => model.id);
      if (modelIds.length === 0) continue;
      const io = CAPABILITY_IO[capability];
      views.push({
        capability,
        inputTypes: [...io.input],
        outputTypes: [...io.output],
        modelIds,
      });
    }
    return views;
  }

  /**
   * Capabilities that no enabled model declares, mapped to why.
   *
   * A capability with zero models has three very different causes — nothing
   * configured, everything disabled, or a configuration error — and an operator
   * needs to tell them apart.
   *
   * @returns one entry per unserved capability.
   */
  listUnservedCapabilities(): readonly { capability: Capability; reason: string }[] {
    const result: { capability: Capability; reason: string }[] = [];
    for (const capability of CAPABILITIES) {
      if (this.findModelsByCapability(capability).length > 0) continue;
      const all = this.findModelsByCapabilityIncludingDisabled(capability);
      if (all.length === 0) {
        result.push({ capability, reason: 'no configured model declares this capability' });
      } else {
        result.push({
          capability,
          reason: `every model declaring it is disabled (${all.map((model) => model.id).join(', ')})`,
        });
      }
    }
    return result;
  }

  /**
   * Every model with a given type.
   * @param type - the model type.
   * @returns matching models.
   */
  findModelsByType(type: ModelType): readonly ResolvedModel[] {
    return this.listModels().filter((model) => model.type === type);
  }

  /**
   * Every model carrying every one of the given tags.
   * @param tags - required tags.
   * @returns matching models.
   */
  findModelsByTags(tags: readonly string[]): readonly ResolvedModel[] {
    if (tags.length === 0) return this.listModels();
    return this.listModels().filter((model) => tags.every((tag) => model.tags.includes(tag)));
  }

  /** The detected machine resources routing decisions are made against. */
  get machineProfile(): MachineProfile {
    return this.machine;
  }

  /** Configuration problems found while loading, if any. */
  get loadDiagnostics(): readonly { severity: 'warning' | 'error'; message: string }[] {
    return this.diagnostics;
  }

  /**
   * Whether a model's declared resource needs fit this machine.
   *
   * The check is intentionally simple and conservative: declared needs are
   * compared against detected totals with no overcommit. Real schedulers
   * multiplex, but a hub that lets a 24 GB model onto an 8 GB card produces a
   * confusing engine crash instead of a clear routing rejection.
   *
   * @param model - the model to test.
   * @returns `{ supported: true }`, or `supported: false` with the shortfall.
   */
  checkResources(model: ResolvedModel): { supported: boolean; reason?: string } {
    const machine = this.machine;
    if (machine.ramGb === 0 && machine.vramGb === 0 && !machine.hasGpu) {
      // Detection failed or was skipped: refusing everything would be worse than
      // trying. Report support and let the engine fail loudly if it must.
      return { supported: true };
    }
    if (model.resources.requiresGpu && !machine.hasGpu) {
      return { supported: false, reason: 'requires a GPU and none was detected' };
    }
    if (!machine.hasGpu && model.resources.vramGb > 0) {
      return {
        supported: false,
        reason: `declares ${model.resources.vramGb} GiB VRAM but no GPU was detected`,
      };
    }
    if (machine.hasGpu && model.resources.vramGb > machine.vramGb) {
      return {
        supported: false,
        reason: `needs ${model.resources.vramGb} GiB VRAM but only ${machine.vramGb} GiB was detected`,
      };
    }
    if (machine.ramGb > 0 && model.resources.ramGb > machine.ramGb) {
      return {
        supported: false,
        reason: `needs ${model.resources.ramGb} GiB RAM but only ${machine.ramGb} GiB was detected`,
      };
    }
    return { supported: true };
  }

  /**
   * Compare two models for a given capability.
   *
   * The order is total and deterministic: enabled before disabled, then lower
   * `priority` first, then more specific tag matches are irrelevant here, then
   * lexicographic id. Nothing about a specific model or capability appears in
   * this function — that is the "no hard-coded model logic in the router" rule
   * showing up as a sort comparator that cannot name a model.
   *
   * @param leftId - first model id.
   * @param rightId - second model id.
   * @param capability - the capability being served.
   * @returns negative when `leftId` should be preferred.
   */
  private compareForRouting(leftId: string, rightId: string, _capability: Capability): number {
    const left = this.models.get(leftId);
    const right = this.models.get(rightId);
    if (left === undefined) return 1;
    if (right === undefined) return -1;
    if (left.enabled !== right.enabled) return left.enabled ? -1 : 1;
    if (left.priority !== right.priority) return left.priority - right.priority;
    return left.id.localeCompare(right.id);
  }
}

/**
 * Whether a string is a capability this hub knows.
 * @param value - candidate string.
 * @returns true for a known capability.
 */
export { isCapability };

/**
 * Build a catalog from a configuration file's parsed contents, reporting
 * validation problems as a single message.
 * @param raw - parsed JSON.
 * @param label - a label for messages.
 * @param options - machine profile and diagnostics sink.
 * @returns the catalog.
 */
export function catalogFromConfig(
  raw: unknown,
  label?: string,
  options?: CatalogOptions,
): ModelCatalog {
  return ModelCatalog.fromConfig(raw, label, options);
}

/**
 * Render one descriptor to a compact multi-line summary.
 *
 * Shared by the `list_models` tool and the CLI so both describe a model the
 * same way — the agent's view and the operator's view never drift.
 *
 * @param model - the resolved model.
 * @param indent - leading whitespace for nested rendering.
 * @returns a summary such as `mock_image_model — Mock Image Model [image_generation]`.
 */
export function summarizeModel(model: ResolvedModel, indent = ''): string {
  const host = model.hostId === undefined ? '' : ` via host ${model.hostId}`;
  const lines = [
    `${indent}${model.id} — ${model.name} [${model.type}]${host}`,
    `${indent}  capabilities: ${model.capabilities.join(', ')}`,
    `${indent}  in/out: ${model.inputTypes.join('|')} → ${model.outputTypes.join('|')}`,
    `${indent}  runtime: ${model.runtime.engine} (${model.adapter})${model.runtime.endpoint === undefined ? '' : ` @ ${model.runtime.endpoint}`}`,
    `${indent}  resources: ${model.resources.vramGb} GiB VRAM, ${model.resources.ramGb} GiB RAM${model.resources.requiresGpu ? ', GPU required' : ''}`,
  ];
  if (model.limits.contextTokens !== undefined) {
    lines.push(`${indent}  context: ${model.limits.contextTokens} tokens`);
  }
  if (model.limits.resolutions !== undefined && model.limits.resolutions.length > 0) {
    lines.push(`${indent}  resolutions: ${model.limits.resolutions.join(', ')}`);
  }
  lines.push(
    `${indent}  lifecycle: ${model.lifecycle.startable ? 'startable' : 'not startable'}, ${model.lifecycle.stoppable ? 'stoppable' : 'not stoppable'}`,
  );
  return lines.join('\n');
}

/**
 * Validate that a descriptor declares a sensible capability set, used by the
 * catalog's own tests and by tooling that edits configuration.
 * @param descriptor - the descriptor to check.
 * @returns problems found; empty means coherent.
 */
export function checkDescriptorCoherence(descriptor: ModelDescriptor): readonly string[] {
  const problems: string[] = [];
  if (descriptor.capabilities.length === 0) problems.push('declares no capabilities');
  const seen = new Set<string>();
  for (const capability of descriptor.capabilities) {
    if (!isCapability(capability)) {
      problems.push(`declares unknown capability "${capability}" (known: ${CAPABILITIES.join(', ')})`);
      continue;
    }
    if (seen.has(capability)) problems.push(`declares capability "${capability}" twice`);
    seen.add(capability);
  }
  return problems;
}

/** Convenience re-export so callers can validate a document without a catalog. */
export { formatIssues };
