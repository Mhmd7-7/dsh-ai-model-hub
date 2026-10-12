/**
 * The execution boundary for subscription tools.
 *
 * A capability policy that only governs `invoke_model` is not a policy. The
 * subscription plugin registers `image_generate`, `video_generate`, and
 * `x_search` as tools in their own right, and an agent that finds its routed
 * request refused can call one of those directly — at which point the operator's
 * switch has been bypassed by the very component that was supposed to honour it.
 * Prompt instructions do not close that hole; a check inside the dispatch
 * pipeline does.
 *
 * So this module registers a **tool guard**. Guards are the registry's monotonic
 * denial layer: they run after every extensible pre-execute listener and
 * immediately before the tool body, and nothing a listener returns can turn a
 * guard's denial back into permission. That is the property the requirement needs
 * — "an agent must not bypass a rejection by calling a subscription tool
 * directly" is a statement about ordering and about who can overrule whom, and
 * this is the only place in DSH's pipeline where neither can be undone.
 *
 * Three consequences shape the rules below:
 *
 * - **A call must name its backend.** `image_generate` defaults to ChatGPT and
 *   *falls back to Grok when ChatGPT is logged out*. That fallback is inside the
 *   tool, past this guard, so a call that does not name a provider cannot be
 *   checked at all. Such calls are refused: with a policy in force, "let the tool
 *   decide" is exactly the behaviour the operator switched off.
 * - **The option is derived from what the call will actually do.** A call with
 *   reference images runs the *edit* option, so it is checked against
 *   `image_to_image`'s policy rather than `text_to_image`'s. Two independent
 *   switches stay independent.
 * - **Refusals are stated, not implied.** The denial says which switch is off and
 *   how to turn it on, because a guard's message is all the agent will ever see.
 *
 * @module dsh-ai-model-hub/dsh-plugin/subscription/enforcement
 */
import { CapabilityPolicy } from "../../src/index.js";
import { backendFacts, canonicalToolName, optionIdForToolCall, subscriptionToolNames } from "./backends.js";
/**
 * Decide whether a direct subscription tool call may proceed.
 *
 * Pure, total, and synchronous: it reads the policy and the call's arguments and
 * returns a verdict. Keeping it a plain function is what makes the bypass rule
 * testable without a running harness, which matters for a rule whose whole job is
 * to be impossible to get around.
 *
 * @param policy - the capability policy in force.
 * @param call - the tool call being guarded.
 * @returns whether it may proceed, with a reason when it may not.
 */
export function checkSubscriptionToolCall(policy, call) {
    const canonical = canonicalToolName(call.name);
    if (canonical === undefined)
        return { allowed: true };
    if (canonical !== 'image_generate' && canonical !== 'video_generate') {
        // `x_search` is a search tool, not a generative one this hub routes, and the
        // hub holds no capability policy over it. Left alone deliberately: a guard
        // that denied tools it has no opinion about would be worse than no guard.
        return { allowed: true };
    }
    const args = asRecord(call.args);
    const provider = readProvider(args);
    const edits = canonical === 'image_generate' && Array.isArray(args['referenceImages']) && args['referenceImages'].length > 0;
    const capability = canonical === 'video_generate' ? 'video_generation' : edits ? 'image_to_image' : 'text_to_image';
    const optionId = canonical === 'video_generate' ? 'grok_video' : optionIdForToolCall(canonical, provider, edits);
    if (optionId === undefined) {
        return {
            allowed: false,
            reason: `${call.name} was refused: it names no provider. The AI Hub's Capability priorities decide which subscription ` +
                'backend may serve this capability, and a call that lets the tool choose could run a backend that is switched ' +
                `off. Pass \`provider\` explicitly (\`gpt\` for ChatGPT, \`grok\` for Grok), or use invoke_model with the ` +
                `capability "${capability}" so the hub routes it.`,
        };
    }
    const facts = backendFacts(optionId);
    if (facts === undefined)
        return { allowed: true };
    const permission = policy.permits(capability, facts);
    if (!permission.enabled) {
        const which = permission.enabledBy === 'global-disable'
            ? 'every capability is switched off globally'
            : `"${optionId}" is switched off for "${capability}"`;
        return {
            allowed: false,
            reason: `${call.name} was refused: ${which} in the AI Hub's Capability priorities, so this backend must not run — ` +
                `not as a first choice and not as a fallback. Enable ${optionId} for ${capability} in Settings → Local ` +
                'models → Capability priorities, or choose a different approach.',
        };
    }
    return { allowed: true };
}
/**
 * Register the bypass guard on the tool registry.
 *
 * Registered on the plugin's own fiber through `ctx.inject(['tools'])`, so it is
 * installed only where a tool registry exists and is removed with the plugin. The
 * guard closes over the *live* policy object rather than a snapshot, so a switch
 * flipped one second ago is honoured by the next call with no reload.
 *
 * @param ctx - the plugin context.
 * @param policy - the live capability policy.
 * @param log - the plugin's logger.
 */
export function registerSubscriptionEnforcement(ctx, policy, log) {
    ctx.inject(['tools'], (scope) => {
        const registry = scope.get('tools');
        if (registry === undefined || typeof registry.guard !== 'function') {
            log.warn('the tool registry exposes no guard API, so direct subscription tool calls cannot be checked against the ' +
                'capability policy; routed invocations are still enforced');
            return;
        }
        const dispose = registry.guard((execution) => {
            const verdict = checkSubscriptionToolCall(policy, { name: execution.name, args: execution.args });
            if (verdict.allowed)
                return undefined;
            log.warn(`blocked a direct subscription tool call: ${execution.name}`);
            return verdict.reason;
        });
        log.debug(`capability policy enforced on direct calls to: ${subscriptionToolNames().join(', ')}`);
        scope.effect(() => () => dispose(), 'dsh-ai-model-hub: subscription tool policy guard');
    });
}
/**
 * Read the tool's resolved provider argument, tolerating its own vocabulary.
 *
 * The subscription plugin's tool spells the ChatGPT provider `gpt`, which is the
 * provider's *API* name rather than its subscription route name. Both spellings
 * are accepted here because a caller reading the tool's own schema will use the
 * documented one, and refusing `codex` would be a confusing rejection of a
 * correct intent.
 *
 * @param args - the call's arguments.
 * @returns the provider, or `undefined` when the call named none.
 */
export function readProvider(args) {
    const value = args['provider'];
    if (value === undefined)
        return undefined;
    if (typeof value !== 'string')
        return undefined;
    switch (value.trim().toLowerCase()) {
        case 'gpt':
        case 'codex':
        case 'chatgpt':
            return 'codex';
        case 'grok':
            return 'grok';
        default:
            return undefined;
    }
}
/**
 * Narrow an unknown argument bag to a record.
 * @param value - the candidate.
 * @returns the record, or an empty one.
 */
function asRecord(value) {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
        ? value
        : {};
}
