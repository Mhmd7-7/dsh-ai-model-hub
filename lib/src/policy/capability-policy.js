/**
 * The Capability Policy: saved order and enablement, per capability.
 *
 * Three different questions get asked about every model, workflow, or remote
 * backend, and conflating them is the bug this module exists to prevent:
 *
 * | Question                  | Answer lives in                    |
 * | ------------------------- | ---------------------------------- |
 * | May I use this at all?    | this policy (`enabled`)            |
 * | When should I prefer it?  | this policy (`order`)              |
 * | Is it usable *right now*? | the runtime manager (availability) |
 *
 * So "disabled" is not "stopped", and "lower priority" is not "unavailable". A
 * disabled option stays in the list, keeps its position, and is never launched,
 * probed into service, or used as a fallback — the only thing that can change it
 * is the operator enabling it. A lower-priority enabled option is tried only
 * after every higher-ranked permitted option has definitively failed.
 *
 * The policy is *data*, and it is the operator's data: it is written to its own
 * versioned document, validated on read, and never inferred from what a picker
 * happens to show. A newly discovered option is disabled until someone says
 * otherwise.
 *
 * @module dsh-ai-model-hub/policy/capability-policy
 */
import { CAPABILITIES } from "../catalog/capabilities.js";
import { isCapability } from "../catalog/capabilities.js";
/** Current document revision. Bumped only by a change that needs migrating. */
export const CAPABILITY_POLICY_VERSION = 1;
/** A fresh policy: nothing saved, nothing globally disabled. */
export function emptyPolicyDocument() {
    return { version: CAPABILITY_POLICY_VERSION, globalEnabled: true, capabilities: {} };
}
/**
 * Validate and normalize an untrusted policy document.
 *
 * A policy is an operator's file, so it is read like one: every problem is
 * reported rather than the first, an unknown capability or a duplicate option is
 * dropped with an issue instead of taking the document down, and a document that
 * cannot be understood at all is rejected so the caller can fall back to the
 * defaults rather than silently enforcing nothing.
 *
 * @param raw - parsed JSON, typically from the policy file.
 * @returns the validated document, or every issue found.
 */
export function parseCapabilityPolicy(raw) {
    const issues = [];
    if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
        return { ok: false, issues: [{ path: '$', message: 'must be an object' }] };
    }
    const record = raw;
    const version = record['version'];
    if (version === undefined) {
        issues.push({ path: '$.version', message: 'is required' });
        return { ok: false, issues };
    }
    if (!Number.isSafeInteger(version) || version < 1) {
        issues.push({ path: '$.version', message: 'must be a positive integer' });
        return { ok: false, issues };
    }
    if (version > CAPABILITY_POLICY_VERSION) {
        // A newer document belongs to a newer deployment. Refusing it is the honest
        // answer: enforcing a policy this build only half-understands could permit
        // something the operator disabled with a field this build ignores.
        return {
            ok: false,
            issues: [
                {
                    path: '$.version',
                    message: `is ${String(version)}, but this build understands at most ${String(CAPABILITY_POLICY_VERSION)}`,
                },
            ],
        };
    }
    const globalEnabledRaw = record['globalEnabled'];
    if (globalEnabledRaw !== undefined && typeof globalEnabledRaw !== 'boolean') {
        issues.push({ path: '$.globalEnabled', message: 'must be a boolean' });
    }
    // An absent global switch means "not globally disabled": the safe reading of a
    // missing field is the non-destructive one, and every option's own `enabled`
    // flag still has to say yes.
    const globalEnabled = typeof globalEnabledRaw === 'boolean' ? globalEnabledRaw : true;
    const capabilities = {};
    const capabilitiesRaw = record['capabilities'];
    if (capabilitiesRaw !== undefined) {
        if (capabilitiesRaw === null || typeof capabilitiesRaw !== 'object' || Array.isArray(capabilitiesRaw)) {
            issues.push({ path: '$.capabilities', message: 'must be an object' });
        }
        else {
            for (const [capability, entryRaw] of Object.entries(capabilitiesRaw)) {
                if (!isCapability(capability)) {
                    issues.push({ path: `$.capabilities.${capability}`, message: 'is not a known capability' });
                    continue;
                }
                if (entryRaw === null || typeof entryRaw !== 'object' || Array.isArray(entryRaw)) {
                    issues.push({ path: `$.capabilities.${capability}`, message: 'must be an object' });
                    continue;
                }
                const orderRaw = entryRaw['order'];
                if (!Array.isArray(orderRaw)) {
                    issues.push({ path: `$.capabilities.${capability}.order`, message: 'must be an array' });
                    continue;
                }
                const order = [];
                const seen = new Set();
                for (const [index, itemRaw] of orderRaw.entries()) {
                    const at = `$.capabilities.${capability}.order[${index}]`;
                    if (itemRaw === null || typeof itemRaw !== 'object' || Array.isArray(itemRaw)) {
                        issues.push({ path: at, message: 'must be an object with `id` and `enabled`' });
                        continue;
                    }
                    const item = itemRaw;
                    const id = item['id'];
                    if (typeof id !== 'string' || id.trim().length === 0) {
                        issues.push({ path: `${at}.id`, message: 'must be a non-empty string' });
                        continue;
                    }
                    if (typeof item['enabled'] !== 'boolean') {
                        issues.push({ path: `${at}.enabled`, message: 'must be a boolean' });
                        continue;
                    }
                    if (seen.has(id)) {
                        issues.push({ path: `${at}.id`, message: `duplicates option "${id}"` });
                        continue;
                    }
                    seen.add(id);
                    order.push({ id, enabled: item['enabled'] });
                }
                capabilities[capability] = { order };
            }
        }
    }
    return { ok: true, document: { version: CAPABILITY_POLICY_VERSION, globalEnabled, capabilities }, issues };
}
/**
 * Serialize a document for persistence, with a stable key order so a diff is
 * readable and an unchanged policy writes identical bytes.
 *
 * @param document - the document to write.
 * @returns pretty-printed JSON with a trailing newline.
 */
export function serializeCapabilityPolicy(document) {
    const capabilities = {};
    for (const capability of CAPABILITIES) {
        const entry = document.capabilities[capability];
        if (entry === undefined)
            continue;
        capabilities[capability] = { order: entry.order.map((item) => ({ id: item.id, enabled: item.enabled })) };
    }
    return `${JSON.stringify({ version: CAPABILITY_POLICY_VERSION, globalEnabled: document.globalEnabled, capabilities }, null, 2)}\n`;
}
/** The conservative default: nothing activates itself. */
export const DENY_BY_DEFAULT = {
    defaultEnabled: () => false,
};
/**
 * The library default: an unsaved option follows the catalog's own `enabled`
 * flag, so a deployment's behaviour is unchanged until an operator saves a
 * policy.
 */
export const CATALOG_DEFAULTS = {
    defaultEnabled: (_capability, facts) => facts.catalogEnabled,
};
/**
 * The deployment default: only options that predate the policy may inherit the
 * catalog's flag; everything else starts disabled.
 *
 * The baseline is a snapshot, not a live view, and that is the point. A workflow
 * discovered a minute ago, a checkpoint dropped into ComfyUI, a subscription
 * backend a plugin just registered — none of them were there when the operator's
 * configuration was written, so none of them may activate themselves. Their rows
 * appear immediately, marked Disabled, and stay that way until someone says
 * otherwise.
 */
export class CatalogBaselineDefaults {
    baseline;
    /**
     * @param baseline - option ids that existed before the policy applied.
     */
    constructor(baseline) {
        this.baseline = new Set(baseline);
    }
    /** The ids this rule treats as pre-existing. */
    get ids() {
        return this.baseline;
    }
    /**
     * Whether an unsaved option is pre-existing and catalog-enabled.
     * @param capability - the capability the option would serve.
     * @param facts - the option's facts.
     * @returns whether it defaults to permitted.
     */
    defaultEnabled(capability, facts) {
        void capability;
        if (!this.baseline.has(facts.id))
            return false;
        return facts.catalogEnabled;
    }
}
/**
 * The policy in force: saved order, saved enablement, and the resolution rules.
 *
 * Construct one from a document (or nothing, for the empty defaults), replace
 * documents as the operator saves, and ask it for a {@link CapabilityPlan} per
 * request. Resolution is pure and synchronous — it reads the catalog and the
 * runtime's already-computed status and makes no network calls — so it is cheap
 * enough to run per request and identical in tests and production.
 */
export class CapabilityPolicy {
    document;
    revision;
    defaults;
    /**
     * @param document - the saved document; the empty defaults when omitted.
     * @param options - revision start and the default-enablement rule.
     */
    constructor(document = emptyPolicyDocument(), options = {}) {
        this.document = document;
        this.revision = options.revision ?? 1;
        this.defaults = options.defaults ?? CATALOG_DEFAULTS;
    }
    /** The document in force. */
    get current() {
        return this.document;
    }
    /** The revision counter, for diagnostics: "which policy decided this?". */
    get currentRevision() {
        return this.revision;
    }
    /** Whether the global switch permits anything at all. */
    get globallyEnabled() {
        return this.document.globalEnabled;
    }
    /**
     * Replace the document, bumping the revision.
     *
     * The revision is what lets a tool result say *which* policy produced it, so a
     * call made under yesterday's order is distinguishable from one made under
     * today's without replaying the file.
     *
     * @param document - the new document.
     * @returns the new revision.
     */
    replace(document) {
        this.document = document;
        this.revision += 1;
        return this.revision;
    }
    /**
     * The saved entry for one capability, when there is one.
     * @param capability - the capability.
     * @returns the entry, or `undefined` when the operator has never saved one.
     */
    entryFor(capability) {
        return this.document.capabilities[capability];
    }
    /**
     * Whether one option may serve one capability.
     *
     * The single source of truth for the enable/disable question, so an explicit
     * `invoke_model` pin, automatic routing, a fallback attempt, a direct
     * subscription tool call, and the settings page cannot disagree.
     *
     * @param capability - the capability.
     * @param facts - the option's facts.
     * @returns whether it is permitted, and which switch decided.
     */
    permits(capability, facts) {
        if (!this.document.globalEnabled)
            return { enabled: false, enabledBy: 'global-disable' };
        const saved = this.document.capabilities[capability]?.order.find((item) => item.id === facts.id);
        if (saved !== undefined)
            return { enabled: saved.enabled, enabledBy: 'saved' };
        return { enabled: this.defaults.defaultEnabled(capability, facts), enabledBy: 'default' };
    }
    /**
     * Whether a catalog model may run at all, expressed as one question.
     *
     * This is the launch gate's question — "may a process for this model exist?" —
     * as opposed to the plan's per-capability one. A model is launchable when *any*
     * capability it declares permits it, because refusing on the strength of a
     * capability nobody is exercising would forbid an engine for the wrong reason;
     * the per-capability decision is then made again, per request, by
     * {@link plan}.
     *
     * @param model - the resolved model.
     * @returns whether it may run, with a reason when it may not.
     */
    permitsModel(model) {
        if (!this.document.globalEnabled) {
            return {
                enabled: false,
                reason: 'every capability is switched off globally (global disable wins over per-capability settings)',
            };
        }
        if (model.capabilities.length === 0) {
            return { enabled: false, reason: 'it declares no capability, so there is nothing for it to serve' };
        }
        const facts = modelOptionFacts(model);
        const allowed = model.capabilities.filter((capability) => this.permits(capability, facts).enabled);
        if (allowed.length > 0)
            return { enabled: true, reason: '' };
        return {
            enabled: false,
            reason: `it is disabled for every capability it declares (${model.capabilities.join(', ')})`,
        };
    }
    /**
     * Whether one option may serve *any* capability it declares.
     *
     * Used where a caller has an option in hand but not the capability — a direct
     * tool call that can reach two capabilities, say.
     *
     * @param facts - the option's facts.
     * @returns whether it is permitted anywhere, with the capabilities that permit it.
     */
    permitsAny(facts) {
        const allowed = facts.capabilities.filter((capability) => this.permits(capability, facts).enabled);
        return { enabled: allowed.length > 0, capabilities: allowed };
    }
    /**
     * The document with an entry seeded for every option the operator has never
     * ruled on, so a newly discovered option is present, visible, and (by the
     * default rule) disabled.
     *
     * This is the migration path, and it is deliberately non-granting: a saved
     * decision is never touched, and an option that was absent from the list is
     * added rather than promoted. Live options that are *not* given here keep
     * their saved settings, which is what makes "temporarily undiscovered" survive
     * a catalog refresh.
     *
     * @param seeds - the live options, with their facts.
     * @returns the document to persist, and whether anything changed.
     */
    seed(seeds) {
        let changed = false;
        const capabilities = {};
        for (const [capability, entry] of Object.entries(this.document.capabilities)) {
            capabilities[capability] = { order: [...entry.order] };
        }
        for (const capability of CAPABILITIES) {
            const relevant = seeds.filter((seed) => seed.capabilities.includes(capability));
            if (relevant.length === 0)
                continue;
            const entry = capabilities[capability] ?? { order: [] };
            const known = new Set(entry.order.map((item) => item.id));
            const order = [...entry.order];
            // New options are appended in a deterministic order rather than inserted at
            // the top: a saved preference is never silently demoted by a discovery pass.
            for (const seed of [...relevant].sort((left, right) => left.id.localeCompare(right.id))) {
                if (known.has(seed.id))
                    continue;
                order.push({ id: seed.id, enabled: this.defaults.defaultEnabled(capability, seed) });
                known.add(seed.id);
                changed = true;
            }
            capabilities[capability] = { order };
        }
        if (!changed)
            return { document: this.document, changed: false };
        return { document: { ...this.document, capabilities }, changed: true };
    }
    /**
     * Resolve the full, ordered plan for one capability.
     *
     * Ordering is the saved order first, then anything unsaved in a stable
     * deterministic order, so a catalog refresh can never permute a list the
     * operator arranged. Each option's *state* is decided by live facts, and its
     * *permission* by this policy — the two never leak into one another.
     *
     * @param capability - the capability to plan for.
     * @param inputs - catalog, live runtime view, and any extra (non-catalog) options.
     * @returns the plan.
     */
    plan(capability, inputs) {
        const facts = collectOptionFacts(capability, inputs);
        const entry = this.document.capabilities[capability];
        const savedIndex = new Map();
        entry?.order.forEach((item, index) => savedIndex.set(item.id, index));
        const preferredIndex = new Map();
        inputs.preferredOrder?.forEach((id, index) => preferredIndex.set(id, index));
        const ordered = [...facts].sort((left, right) => {
            const leftIndex = savedIndex.get(left.id);
            const rightIndex = savedIndex.get(right.id);
            if (leftIndex !== undefined && rightIndex !== undefined)
                return leftIndex - rightIndex;
            if (leftIndex !== undefined)
                return -1;
            if (rightIndex !== undefined)
                return 1;
            // Neither is arranged. Fall back to the router's ranking when it offered
            // one, then to id, so the FIRST render of a brand-new option is the choice
            // the deployment would have made before this feature existed — and is
            // stable rather than incidental either way.
            const leftPreferred = preferredIndex.get(left.id);
            const rightPreferred = preferredIndex.get(right.id);
            if (leftPreferred !== undefined && rightPreferred !== undefined)
                return leftPreferred - rightPreferred;
            if (leftPreferred !== undefined)
                return -1;
            if (rightPreferred !== undefined)
                return 1;
            return left.id.localeCompare(right.id);
        });
        const options = [];
        const permitted = [];
        const skipped = [];
        ordered.forEach((option, index) => {
            const permission = this.permits(capability, option);
            const live = liveStateFor(option, inputs);
            const resolved = resolveOption(index + 1, option, permission, live, inputs.runtime !== undefined);
            options.push(resolved);
            if (resolved.enabled && resolved.state !== 'unsupported' && resolved.state !== 'unavailable') {
                permitted.push(resolved);
            }
            else {
                skipped.push({ id: option.id, reason: resolved.reason });
            }
        });
        return { revision: this.revision, capability, options, permitted, skipped };
    }
    /**
     * The order and enablement to persist after an operator edit.
     *
     * Validation is deliberately strict about identity: an id that is not a live
     * option *and* not already saved is rejected rather than silently added, so a
     * typo in an API call cannot invent policy. A saved-but-absent id is kept, so
     * editing a list while an engine is down does not delete its settings.
     *
     * @param capability - the capability being edited.
     * @param order - the full ordered list the operator submitted.
     * @param knownIds - ids that exist right now.
     * @returns the new document and the issues found.
     */
    applyEdit(capability, order, knownIds) {
        const issues = [];
        const existing = this.document.capabilities[capability];
        const allowed = new Set([...knownIds, ...(existing?.order.map((item) => item.id) ?? [])]);
        const seen = new Set();
        const next = [];
        for (const [index, item] of order.entries()) {
            if (!allowed.has(item.id)) {
                issues.push({ path: `order[${index}].id`, message: `"${item.id}" is not an option for ${capability}` });
                continue;
            }
            if (seen.has(item.id)) {
                issues.push({ path: `order[${index}].id`, message: `duplicates option "${item.id}"` });
                continue;
            }
            seen.add(item.id);
            next.push({ id: item.id, enabled: item.enabled });
        }
        if (issues.length > 0)
            return { document: this.document, issues };
        // Anything the caller did not mention keeps its saved place at the end, so a
        // partial submission (a single toggle, say) cannot drop options.
        for (const item of existing?.order ?? []) {
            if (seen.has(item.id))
                continue;
            next.push({ id: item.id, enabled: item.enabled });
        }
        const capabilities = {};
        for (const [name, value] of Object.entries(this.document.capabilities))
            capabilities[name] = value;
        capabilities[capability] = { order: next };
        return { document: { ...this.document, capabilities }, issues: [] };
    }
    /**
     * The document with the global switch set.
     * @param enabled - whether anything may run at all.
     * @returns the new document.
     */
    withGlobalEnabled(enabled) {
        return { ...this.document, globalEnabled: enabled };
    }
}
/**
 * Why an option is disabled, phrased so the cases stay distinguishable.
 * @param enabledBy - which switch decided it.
 * @returns the reason sentence.
 */
export function disabledReason(enabledBy) {
    switch (enabledBy) {
        case 'global-disable':
            return 'Disabled: every capability is switched off globally (global disable wins over per-capability settings).';
        case 'saved':
            return 'Disabled: you switched this option off for this capability. It is never used — not as a first choice, and not as a fallback.';
        case 'default':
            return 'Disabled: not authorized yet. Enable it explicitly to let it serve this capability.';
        default: {
            const exhaustive = enabledBy;
            return `Disabled (${String(exhaustive)}).`;
        }
    }
}
/**
 * Resolve one option into its reported shape.
 *
 * @param order - 1-based position.
 * @param option - the option's facts.
 * @param permission - the policy verdict.
 * @param live - live state, when anything reported one.
 * @param hasRuntime - whether the caller supplied a runtime view at all.
 * @returns the resolved option.
 */
function resolveOption(order, option, permission, live, hasRuntime) {
    if (!permission.enabled) {
        return {
            order,
            enabled: false,
            enabledBy: permission.enabledBy,
            state: 'disabled',
            reason: disabledReason(permission.enabledBy),
            startable: false,
            option,
        };
    }
    if (!option.executable) {
        const reason = `Unsupported: ${option.unsupportedReason ?? 'this deployment cannot execute it'}`;
        return { order, enabled: true, enabledBy: permission.enabledBy, state: 'unsupported', reason, startable: false, option };
    }
    if (live === undefined) {
        const reason = option.source === 'subscription'
            ? 'Unavailable: this backend reported no usable session or login.'
            : hasRuntime
                ? 'Unavailable: the runtime could not report a live state for it.'
                : 'Unavailable: no runtime was attached to this policy plan.';
        return {
            order,
            enabled: true,
            enabledBy: permission.enabledBy,
            state: 'unavailable',
            reason,
            startable: false,
            option,
        };
    }
    return {
        order,
        enabled: true,
        enabledBy: permission.enabledBy,
        state: live.state,
        reason: live.reason,
        startable: live.startable ?? live.state === 'stopped',
        option,
    };
}
/**
 * Fetch one option's live state from whichever source owns it.
 * @param option - the option's facts.
 * @param inputs - the plan inputs.
 * @returns the live state, or `undefined` when nothing reported one.
 */
function liveStateFor(option, inputs) {
    if (option.source === 'subscription')
        return inputs.extraRuntime?.[option.id];
    return inputs.runtime?.describe(option.id);
}
/**
 * Gather the facts of every option that declares a capability.
 *
 * @param capability - the capability.
 * @param inputs - catalog, runtime view, and extra options.
 * @returns the option facts, catalog options first.
 */
function collectOptionFacts(capability, inputs) {
    const facts = [];
    for (const model of inputs.catalog.findModelsByCapabilityIncludingDisabled(capability)) {
        facts.push(modelOptionFacts(model));
    }
    for (const extra of inputs.extraOptions ?? []) {
        if (!extra.capabilities.includes(capability))
            continue;
        if (facts.some((fact) => fact.id === extra.id))
            continue;
        facts.push(extra);
    }
    return facts;
}
/**
 * Build the facts for a catalog model, so a caller that is not the planner can
 * describe an option identically.
 *
 * @param model - the resolved model.
 * @returns the facts.
 */
export function modelOptionFacts(model) {
    return {
        id: model.id,
        name: model.name,
        type: model.type,
        source: 'model',
        provider: model.runtime.engine,
        providerLabel: `local · ${model.runtime.engine}`,
        capabilities: model.capabilities,
        inputTypes: model.inputTypes,
        outputTypes: model.outputTypes,
        executable: true,
        catalogEnabled: model.enabled,
    };
}
/**
 * Whether a live state means "this option can actually serve work right now".
 *
 * `stopped` counts: a stopped engine the hub may start is a valid choice, and
 * demoting it is exactly the bug where a ready remote backend wins over a
 * higher-priority local engine that merely needs starting.
 *
 * @param state - the state to test.
 * @returns whether it is usable.
 */
export function isUsableState(state) {
    return state === 'ready' || state === 'stopped' || state === 'starting';
}
