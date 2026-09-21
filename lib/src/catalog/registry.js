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
import { CAPABILITY_IO, CAPABILITIES, isCapability } from "./capabilities.js";
import { resolveDescriptor } from "./descriptor.js";
import { ModelHubError } from "../errors.js";
import { formatIssues } from "../util/validate.js";
import { parseModelCatalogConfig } from "./descriptor.js";
/**
 * The in-memory registry of models.
 *
 * Instances are cheap; the hub builds one at startup and can rebuild it on
 * configuration change. All lookups are synchronous because publication order
 * matters to the router: it needs a stable, ordered candidate list per decision,
 * not a stream.
 */
export class ModelCatalog {
    hosts = new Map();
    models = new Map();
    byCapability = new Map();
    machine;
    log;
    diagnostics = [];
    /**
     * @param config - the validated catalog document.
     * @param options - machine profile override and diagnostics sink.
     */
    constructor(config, options = {}) {
        this.machine = options.machine ?? { vramGb: 0, ramGb: 0, hasGpu: false, notes: 'not probed' };
        this.log = options.log ?? (() => { });
        for (const host of config.hosts ?? [])
            this.hosts.set(host.id, host);
        this.build(config.models);
        this.log(`catalog: ${this.models.size} model(s), ${this.byCapability.size} capability(ies) from ${this.hosts.size} host(s)`);
    }
    /**
     * Resolve a model list into the lookup tables, replacing whatever was there.
     *
     * The hosts and the machine profile survive: a rebuild is what runtime
     * discovery uses to fold newly found models in, and neither "which endpoints
     * exist" nor "what this machine has" changes when a checkpoint does. Behavior
     * on a bad descriptor is identical to the constructor's — a per-model
     * diagnostic and a skip, never a throw — because a discovered model that fails
     * validation must cost one model, not the catalog.
     *
     * @param descriptors - the models to publish, in publication order.
     */
    build(descriptors) {
        this.diagnostics.length = 0;
        this.models.clear();
        this.byCapability.clear();
        for (const descriptor of descriptors) {
            let resolved;
            try {
                resolved = resolveDescriptor(descriptor, descriptor.host === undefined ? undefined : this.hosts.get(descriptor.host));
            }
            catch (error) {
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
                if (bucket === undefined)
                    this.byCapability.set(capability, [resolved.id]);
                else
                    bucket.push(resolved.id);
            }
        }
        // Sort each capability bucket once, so routing order is stable and cheap.
        for (const [capability, ids] of this.byCapability) {
            ids.sort((left, right) => this.compareForRouting(left, right, capability));
        }
    }
    /**
     * Publish a different model list on the same hosts.
     *
     * This is the seam runtime discovery uses: a discovery pass produces
     * descriptors, they are merged with the static ones *before* they get here,
     * and the merged list replaces the catalog's contents in one step. Every
     * consumer — the router, the runtime manager, the plugin tools — holds this
     * same object and therefore sees the change without being rebuilt.
     *
     * It does not touch host definitions or the machine profile, and it does not
     * stop any process: a model that disappears from the list simply stops being
     * routable.
     *
     * @param descriptors - the complete model list to publish, static entries first.
     * @returns how many models are now registered.
     */
    replaceModels(descriptors) {
        this.build(descriptors);
        this.log(`catalog: rebuilt with ${this.models.size} model(s), ${this.byCapability.size} capability(ies)`);
        return this.models.size;
    }
    /**
     * Build a catalog from an untrusted configuration document.
     * @param raw - parsed JSON, typically from `config/models.json`.
     * @param label - a label for validation messages, usually the file path.
     * @param options - machine profile and diagnostics sink.
     * @returns the catalog.
     * @throws ModelHubError with `INVALID_DESCRIPTOR` listing every problem found.
     */
    static fromConfig(raw, label = 'model catalog', options = {}) {
        const parsed = parseModelCatalogConfig(raw, label);
        if (!parsed.ok) {
            throw new ModelHubError('INVALID_DESCRIPTOR', parsed.message, {
                issues: parsed.issues.map((issue) => ({ path: issue.path, message: issue.message })),
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
    static validate(raw, label = 'model catalog') {
        const parsed = parseModelCatalogConfig(raw, label);
        return parsed.ok ? [] : parsed.issues;
    }
    /** Every configured model, in configuration order. */
    listModels() {
        return [...this.models.values()];
    }
    /** Ids of every configured model, in configuration order. */
    listModelIds() {
        return [...this.models.keys()];
    }
    /**
     * Look up one model by id.
     * @param modelId - the model id.
     * @returns the resolved model, or `undefined`.
     */
    getModel(modelId) {
        return this.models.get(modelId);
    }
    /**
     * Look up one model by id, failing loudly when it is absent.
     * @param modelId - the model id.
     * @returns the resolved model.
     * @throws ModelHubError with `MODEL_NOT_FOUND`.
     */
    requireModel(modelId) {
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
    findModelsByCapability(capability) {
        const ids = this.byCapability.get(capability) ?? [];
        const result = [];
        for (const id of ids) {
            const model = this.models.get(id);
            if (model !== undefined && model.enabled)
                result.push(model);
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
    findModelsByCapabilityIncludingDisabled(capability) {
        const ids = this.byCapability.get(capability) ?? [];
        const models = [];
        for (const id of ids) {
            const model = this.models.get(id);
            if (model !== undefined)
                models.push(model);
        }
        return models.sort((left, right) => {
            if (left.enabled !== right.enabled)
                return left.enabled ? -1 : 1;
            return this.compareForRouting(left.id, right.id, capability);
        });
    }
    /**
     * Every capability at least one enabled model declares.
     * @returns capability views, in the canonical vocabulary order.
     */
    listCapabilities() {
        const views = [];
        for (const capability of CAPABILITIES) {
            const modelIds = this.findModelsByCapability(capability).map((model) => model.id);
            if (modelIds.length === 0)
                continue;
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
    listUnservedCapabilities() {
        const result = [];
        for (const capability of CAPABILITIES) {
            if (this.findModelsByCapability(capability).length > 0)
                continue;
            const all = this.findModelsByCapabilityIncludingDisabled(capability);
            if (all.length === 0) {
                result.push({ capability, reason: 'no configured model declares this capability' });
            }
            else {
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
    findModelsByType(type) {
        return this.listModels().filter((model) => model.type === type);
    }
    /**
     * Every model carrying every one of the given tags.
     * @param tags - required tags.
     * @returns matching models.
     */
    findModelsByTags(tags) {
        if (tags.length === 0)
            return this.listModels();
        return this.listModels().filter((model) => tags.every((tag) => model.tags.includes(tag)));
    }
    /** The detected machine resources routing decisions are made against. */
    get machineProfile() {
        return this.machine;
    }
    /** Configuration problems found while loading, if any. */
    get loadDiagnostics() {
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
    checkResources(model) {
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
    compareForRouting(leftId, rightId, _capability) {
        const left = this.models.get(leftId);
        const right = this.models.get(rightId);
        if (left === undefined)
            return 1;
        if (right === undefined)
            return -1;
        if (left.enabled !== right.enabled)
            return left.enabled ? -1 : 1;
        if (left.priority !== right.priority)
            return left.priority - right.priority;
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
export function catalogFromConfig(raw, label, options) {
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
 * @returns a summary such as `comfyui_z_image_turbo — Z-Image Turbo (ComfyUI) [image_generation]`.
 */
export function summarizeModel(model, indent = '') {
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
    lines.push(`${indent}  lifecycle: ${model.lifecycle.startable ? 'startable' : 'not startable'}, ${model.lifecycle.stoppable ? 'stoppable' : 'not stoppable'}`);
    return lines.join('\n');
}
/**
 * Validate that a descriptor declares a sensible capability set, used by the
 * catalog's own tests and by tooling that edits configuration.
 * @param descriptor - the descriptor to check.
 * @returns problems found; empty means coherent.
 */
export function checkDescriptorCoherence(descriptor) {
    const problems = [];
    if (descriptor.capabilities.length === 0)
        problems.push('declares no capabilities');
    const seen = new Set();
    for (const capability of descriptor.capabilities) {
        if (!isCapability(capability)) {
            problems.push(`declares unknown capability "${capability}" (known: ${CAPABILITIES.join(', ')})`);
            continue;
        }
        if (seen.has(capability))
            problems.push(`declares capability "${capability}" twice`);
        seen.add(capability);
    }
    return problems;
}
/** Convenience re-export so callers can validate a document without a catalog. */
export { formatIssues };
