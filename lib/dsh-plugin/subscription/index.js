/**
 * Subscription integrations, wired into the capability plan.
 *
 * This module is the seam between two plugins that must not know each other's
 * internals. The subscription plugin owns credentials, accounts, sessions, and
 * the provider transport; the AI Hub owns ordering, enablement, planning,
 * fallback, and artifacts. Neither imports the other: this module reaches the
 * subscription plugin's tools through the *tool registry*, which is DSH's
 * supported same-process dispatch surface, and hands the hub a plain
 * {@link CapabilityOption}.
 *
 * What that buys, concretely, is that a subscription backend stops being a
 * separate path:
 *
 * - It appears in the same ordered list as a local workflow, with the same
 *   enable/disable control, and is skipped with a reason when it is off.
 * - It participates in the same fallback walk, so "ComfyUI failed, ChatGPT
 *   Images is next" is one saved order rather than a chain of prompt
 *   instructions.
 * - Its output is archived as a hub artifact, so a later `image_to_3d` step can
 *   consume it by id.
 * - It is subject to the same launch/readiness accounting, reporting "no usable
 *   login" as a definitive pre-submission failure the plan can fall past.
 *
 * It also does not do the things the requirements forbid: no token is read, no
 * second login flow is created, no paid API key replaces the subscription, and no
 * account scheduling is reimplemented. Session restrictions and tool policies
 * stay exactly where they were — enforced at the tool boundary, which this module
 * calls through rather than around.
 *
 * @module dsh-ai-model-hub/dsh-plugin/subscription
 */
import { ModelHub, ModelHubError } from "../../src/index.js";
import { SUBSCRIPTION_BACKENDS, buildSubscriptionBackendOptions, canonicalToolName, providerArgument, } from "./backends.js";
import { registerSubscriptionEnforcement } from "./enforcement.js";
/**
 * Register the subscription backends as hub capability options and enforce the
 * policy on direct calls to their tools.
 *
 * Called once per plugin load. Every sub-step is contained: a profile without the
 * subscription plugin registers nothing and says so, rather than failing the
 * model hub for a missing optional integration.
 *
 * @param ctx - the plugin context.
 * @param hub - the live hub, so options can write artifacts through the calling
 *   session's store.
 * @param log - the plugin's logger.
 * @returns what was registered, or `undefined` when there is no tool registry.
 */
export function registerSubscriptionIntegrations(ctx, hub, log) {
    // The bypass guard goes on first, and on the *global* registry layer, so it is
    // in place even for a call that arrives before the options below are
    // registered.
    registerSubscriptionEnforcement(ctx, hub.policy, log);
    const tools = readToolRuntime(ctx);
    if (tools === undefined) {
        log.info('no tool registry is mounted, so subscription backends are not offered as capability options; ' +
            'the hub still routes every local model');
        return { registered: [], unavailable: [], tools: {} };
    }
    // The tool registry is read through the scope so a same-process dispatch runs
    // the call through the same pipeline a model-direct call would: pre-execute
    // policy, guards (including this plugin's own), timeout wrappers, and result
    // materialization. Reaching into the subscription plugin's internals instead
    // would skip all of it.
    const access = {
        resolveTool: (canonical) => resolveToolName(tools, canonical),
        callTool: async (name, args, signal) => {
            const outcome = await tools.execute({
                callId: toolCallId(),
                name,
                arguments: args,
                signal,
            });
            if (outcome.isError) {
                throw new ModelHubError('INVOCATION_FAILED', outcome.error?.message ?? `${name} failed without a message`, { tool: name });
            }
            return outcome.value;
        },
    };
    const built = buildSubscriptionBackendOptions(access, log);
    const registered = [];
    const unavailable = [];
    const disposal = [];
    for (const entry of built) {
        if (!entry.executable) {
            unavailable.push({ id: entry.facts.id, reason: entry.reason ?? 'its tool is not registered' });
            continue;
        }
        disposal.push(hub.registerCapabilityOption(entry.option));
        registered.push(entry.facts.id);
    }
    if (disposal.length > 0) {
        ctx.effect(() => () => {
            for (const dispose of disposal)
                dispose();
        }, 'dsh-ai-model-hub: unregister subscription capability options');
    }
    const resolvedTools = {};
    for (const backend of SUBSCRIPTION_BACKENDS) {
        const name = access.resolveTool(backend.tool);
        if (name !== undefined)
            resolvedTools[backend.tool] = name;
    }
    if (registered.length === 0) {
        log.info('the subscription plugin registers no image or video tool in this profile, so no subscription backend is ' +
            'offered as a capability option; nothing was invented to fill the gap');
    }
    else {
        log.info(`subscription backends offered as capability options (disabled until enabled in Capability priorities): ${registered.join(', ')}`);
    }
    return { registered, unavailable, tools: resolvedTools };
}
/**
 * Resolve one subscription tool's registered name, honouring the plugin's alias
 * fallback.
 *
 * The subscription plugin registers `image_generate` under its canonical name and
 * falls back to `dsh_subscriptions_image_generate` when another plugin already
 * owns that name — a real configuration, not a hypothetical one, since any
 * image-generation plugin would collide. Trying both names and reporting which
 * one answered is the difference between working in that deployment and silently
 * calling a different plugin's tool under the same name.
 *
 * @param tools - the tool registry.
 * @param canonical - the canonical name.
 * @returns the registered name, or `undefined`.
 */
export function resolveToolName(tools, canonical) {
    if (tools.get(canonical) !== undefined)
        return canonical;
    const alias = `dsh_subscriptions_${canonical}`;
    return tools.get(alias) !== undefined ? alias : undefined;
}
/**
 * Read the tool runtime from the plugin context, tolerating its absence.
 *
 * `ctx.get` rather than a declared `inject`: the hub must keep loading in a
 * composition with no tool registry at all, because a missing *optional*
 * integration must cost one feature rather than every model tool.
 *
 * @param ctx - the plugin context.
 * @returns the registry, or `undefined`.
 */
function readToolRuntime(ctx) {
    const tools = ctx.get('tools');
    if (tools === undefined)
        return undefined;
    if (typeof tools.get !== 'function' || typeof tools.execute !== 'function')
        return undefined;
    return tools;
}
/**
 * Mint a tool-call id for a same-process dispatch.
 *
 * The registry requires one so the call can be correlated, and the brand exists
 * precisely so an id cannot be confused with any other string. A namespaced,
 * unique value keeps hub-dispatched calls distinguishable in the session log from
 * a model's own call.
 *
 * @returns a fresh call id.
 */
function toolCallId() {
    return `aihub-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
/** Re-exported for tests and for the settings page's own description of a backend. */
export { SUBSCRIPTION_BACKENDS, canonicalToolName, providerArgument };
