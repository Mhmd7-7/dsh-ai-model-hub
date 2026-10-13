/**
 * The DSH compatibility layer: the one module that knows what a host looks like.
 *
 * DeepSeek Harness is an `0.x-rc` line, so its plugin surface moves. The rest of
 * this project is insulated from that by construction — the catalog, router,
 * runtime, adapters, artifact store, and every model-facing tool import nothing
 * from DSH — but *something* has to call the host, and that something was
 * previously spread across eleven files and fourteen call sites. When DSH moved a
 * member, the failure was a `TypeError: ctx.tools is undefined` at boot, pointing
 * at whichever file happened to call it first, with no statement of what the host
 * actually offered.
 *
 * This module replaces that with one rule:
 *
 * > **A host capability is something to detect, not something to assume.**
 *
 * Every DSH service, member, and method the plugin uses is resolved through a
 * function here, checked for the shape the plugin needs, and reported on when it
 * is absent. Nothing throws merely because a host is different; a plugin that
 * cannot do a thing says which thing and carries on with the rest.
 *
 * ## What is deliberately NOT wrapped
 *
 * `defineTool` is a pure function from a closed input to a closed output. It
 * compiles the author-facing schema and installs argument validation, and
 * `@deepseek-ai/dsh-tools` has shipped it with the same signature across the
 * whole `0.1.5-rc` → `0.2.0-rc` line. It is the *interfaces the plugin consumes*
 * that move, so those are what this module fences. Function identity is fine to
 * depend on; ambient registry shape is not.
 *
 * The static `import type` lines elsewhere are also not a compatibility risk:
 * a type-only import is erased at compile time and never resolved at runtime, so
 * a host that renames a type cannot break a running plugin.
 *
 * ## Two halves
 *
 * 1. {@link resolveCompatHost} — reads the host's {@link Context} and describes
 *    what it actually has. Pure, cheap, and safe to call before anything else.
 * 2. The install helpers ({@link installTool}, {@link installEffect},
 *    {@link installPromptContext}, {@link withService}) — each one takes the
 *    probe's verdict and either does the work or contains the absence to a single
 *    diagnostic line.
 *
 * @module dsh-ai-model-hub/dsh-plugin/compat
 */
/**
 * The runtime members this plugin uses, and how each one is treated.
 *
 * `required` entries are the plugin's reason to exist — without a tool registry
 * there is nothing to register, and a hub the agent cannot reach is not a
 * degraded plugin but an absent one, so its absence is reported as a hard load
 * failure rather than silently continuing. Everything else is optional by
 * design, because every other member costs one feature and never the plugin:
 * no `systemPrompt` costs the capability snapshot, no `webServer` costs the
 * settings pages, no `skills` costs the bundled skill.
 */
export const REQUIRED_MEMBERS = ['tools'];
/** The context members the plugin reads, in the order the plugin first needs them. */
const MEMBER_PURPOSES = [
    { member: 'tools', purpose: 'every model-facing tool this plugin offers', needs: 'a callable `register`' },
    { member: 'systemPrompt', purpose: 'the live capability snapshot in the prompt', needs: 'a callable `context`' },
    { member: 'logger', purpose: 'diagnostics', needs: 'to be callable with a scope name' },
    { member: 'effect', purpose: 'tying the hub and its processes to the plugin lifetime', needs: 'a callable function' },
    { member: 'inject', purpose: 'optional integrations (web routes, skills)', needs: 'a callable function' },
    { member: 'get', purpose: 'reading optional services without parking the fiber', needs: 'a callable function' },
    { member: 'webServer', purpose: 'the Local models and Priorities pages', needs: 'a callable `register`' },
    { member: 'sandboxPolicy', purpose: 'placing artifacts in the calling session workspace', needs: 'a callable `resolve`' },
    { member: 'skills', purpose: 'offering the bundled skill', needs: 'a callable `registerProvider`' },
];
/**
 * Whether a value is a non-null object or function, the only two host shapes.
 * @param value - the candidate.
 * @returns true when the value can carry members.
 */
function isObjectLike(value) {
    return (typeof value === 'object' && value !== null) || typeof value === 'function';
}
/**
 * Read one host member by its literal name.
 *
 * Spelled out as a switch rather than as `(ctx as Record<string, unknown>)[name]`
 * on purpose. Two reasons, and the second is the load-bearing one:
 *
 * 1. It is what actually happens — every member this plugin reads has a fixed
 *    name that is known at build time, so a dynamic index was never needed.
 * 2. `tests/plugin.test.ts` audits the plugin's sources for `ctx.<name>` reads and
 *    requires each one to be either a declared injection or a cordis builtin,
 *    because cordis aborts the *whole profile boot* for an undeclared read. A
 *    dynamic index is invisible to that audit, so writing one would have quietly
 *    punched a hole in a check that exists because this plugin has already
 *    stopped a real harness from starting once.
 *
 * @param ctx - the host context.
 * @param member - the member name, from the closed set below.
 * @returns the member's value, or `undefined`.
 */
function readMember(ctx, member) {
    switch (member) {
        case 'tools':
            return ctx.tools;
        case 'systemPrompt':
            return ctx.systemPrompt;
        case 'logger':
            return ctx.logger;
        case 'effect':
            return ctx.effect;
        case 'inject':
            return ctx.inject;
        case 'get':
            return ctx.get;
        case 'webServer':
            return ctx.webServer;
        case 'sandboxPolicy':
            return ctx.sandboxPolicy;
        case 'skills':
            return ctx.skills;
        default:
            return undefined;
    }
}
/**
 * Read one member and judge it.
 *
 * The `needs` clause is evaluated with `typeof … === 'function'` rather than
 * truthiness, because a member that exists but is a plain value (a `webServer`
 * record, an `inject` array) would pass a truthiness check and then fail at the
 * call site — which is precisely the failure mode this module exists to remove.
 *
 * @param ctx - the host context.
 * @param spec - the member to look for and what it must be able to do.
 * @returns the verdict.
 */
function judgeMember(ctx, spec) {
    const value = readMember(ctx, spec.member);
    const present = value !== undefined && value !== null;
    const needsMember = /`([A-Za-z]+)`/.exec(spec.needs)?.[1];
    const usable = present &&
        (needsMember === undefined
            ? isObjectLike(value) || typeof value === 'string'
            : isObjectLike(value) && typeof value[needsMember] === 'function');
    const judgement = {
        member: spec.member,
        present,
        usable,
        purpose: spec.purpose,
    };
    if (usable)
        return judgement;
    // Built by concatenation rather than with a template literal, and deliberately
    // not spelled as a backticked member reference. `tests/plugin.test.ts` scans the
    // plugin's sources for `ctx.<name>` reads and requires each one to be a declared
    // injection or a known optional integration, so a diagnostic that *looks* like a
    // read would be reported as an undeclared one. Concatenation keeps the prose out
    // of that audit's way without special-casing the audit.
    const qualified = 'ctx.' + spec.member;
    return {
        ...judgement,
        consequence: present
            ? `${qualified} is present but is not usable for ${spec.purpose} (needs ${spec.needs}), which is a host API change rather than a configuration problem`
            : `${qualified} is absent, so this deployment has no ${spec.purpose}`,
    };
}
/**
 * Describe what the host offers, without changing anything.
 *
 * Called before the first registration so that a version mismatch is reported as
 * one honest line at boot rather than as a crash inside an unrelated call. The
 * result is pure data: a test can assert against it without mounting a host.
 *
 * @param ctx - the host context.
 * @returns the observation, including one diagnostic line per unusable member.
 */
export function resolveCompatHost(ctx) {
    const members = MEMBER_PURPOSES.map((spec) => judgeMember(ctx, spec));
    const byName = new Map(members.map((verdict) => [verdict.member, verdict]));
    const complete = REQUIRED_MEMBERS.every((member) => byName.get(member)?.usable === true);
    const diagnostics = members
        .filter((verdict) => !verdict.usable)
        .map((verdict) => `${verdict.member}: ${verdict.consequence ?? 'unavailable'}`);
    return { ctx, members, complete, diagnostics };
}
/**
 * Look one verdict up by member name.
 * @param host - the observation.
 * @param member - the member to find.
 * @returns the verdict, or `undefined` when the member is not tracked.
 */
export function memberVerdict(host, member) {
    return host.members.find((verdict) => verdict.member === member);
}
/**
 * Log the host observation at the level each line deserves.
 *
 * Missing optional members are `info`, not `warn`: a headless profile with no web
 * server is a correct composition, not a problem, and warning about it trains an
 * operator to ignore warnings. An unusable member *is* a warning, because it
 * means the host changed under the plugin.
 *
 * @param host - the observation.
 * @param log - the plugin's logger.
 * @returns the number of members reported, for a caller that wants to assert.
 */
export function reportCompatHost(host, log) {
    for (const verdict of host.members) {
        if (verdict.usable)
            continue;
        const line = `${verdict.member}: ${verdict.consequence ?? 'unavailable'}`;
        if (verdict.present)
            log.warn(line);
        else
            log.debug(line);
    }
    return host.diagnostics.length;
}
/**
 * Register one tool definition on the host's tool registry.
 *
 * Contained on purpose. A tool registry that rejects a definition — because a
 * schema keyword this DSH version does not know, because the name collides with
 * another plugin's, because `register` threw — must cost that one tool rather
 * than abort `apply` and take the other twelve with it. The caller gets a
 * boolean and the reason lands in the log.
 *
 * @param ctx - the host context.
 * @param definition - the tool definition to register.
 * @param log - the plugin's logger.
 * @returns true when the tool is registered.
 */
export function installTool(ctx, definition, log) {
    const registry = ctx.tools;
    if (registry === undefined || typeof registry.register !== 'function') {
        log.warn(`the tool ${definition.name} was not registered: this host exposes no tool registry`);
        return false;
    }
    try {
        registry.register(definition);
        return true;
    }
    catch (error) {
        log.warn(`the tool ${definition.name} was refused by this host's tool registry: ${error instanceof Error ? error.message : String(error)}`);
        return false;
    }
}
/**
 * Define a tool and register it in one step.
 *
 * Every tool in this plugin is declared with this rather than with `defineTool`
 * plus `ctx.tools.register`, so a change to *either* half is one edit here. The
 * definition is still compiled by DSH's own `defineTool`, so the parameter and
 * output schemas are validated exactly as the host validates them — this is a
 * containment boundary, not a reimplementation.
 *
 * **Why `defineTool` is called by the caller and not in here.** Schemastery's
 * `DefineToolOptions<S, O>` is a deeply conditional type, and re-parameterising it
 * through this function's generics makes the compiler give up with
 * `TS2321: Excessive stack depth comparing types 'InferObject<S, ?>'`. Calling
 * `defineTool` at the declaration site keeps its own inference — including the
 * contextual typing of every `render`, `presentCall`, and `execute` callback
 * parameter, which is what makes a schema mistake a compile error — and leaves
 * this function with one job: registering the finished definition without letting
 * a hostile registry abort `apply`.
 *
 * @param ctx - a context with a tool registry.
 * @param definition - the definition `defineTool` returned.
 * @param log - the plugin's logger.
 * @returns true when the tool is registered.
 */
export function installDefinedTool(ctx, definition, log) {
    return installTool(ctx, definition, log);
}
/**
 * Install a cleanup-aware effect on the plugin's fiber, when the host has one.
 *
 * An effect is how the hub's lifetime is bound to the plugin: unloading the
 * plugin stops every process the hub started. Without the member there is no
 * binding, so the consequence is stated once — and the *body is not run*, because
 * running a disposer factory with no one to call the disposer would leak exactly
 * what the effect exists to release.
 *
 * @param ctx - the host context.
 * @param execute - produces the disposer to run on teardown.
 * @param label - effect label shown in fiber diagnostics.
 * @param log - the plugin's logger.
 * @returns the disposer the host returned, when it did.
 */
export function installEffect(ctx, execute, label, log) {
    const effect = ctx.effect;
    if (typeof effect !== 'function') {
        log.warn(`${label}: this host exposes no \`ctx.effect\`, so this cleanup is not bound to the plugin lifetime and will not ` +
            'run when the plugin unloads');
        return undefined;
    }
    try {
        return effect(execute, label);
    }
    catch (error) {
        log.warn(`${label}: could not be installed (${error instanceof Error ? error.message : String(error)})`);
        return undefined;
    }
}
/**
 * The order used when the host will not name one.
 *
 * `SANDBOX_POLICY + 1` is the plugin's intended position: after the facts about
 * the sandbox, before anything a user wrote. When `getContextOrder` is missing or
 * throws, this stands in rather than aborting the registration — an environment
 * fact placed at a slightly different position in the prompt is worth more than
 * an environment fact the model never sees.
 */
export const DEFAULT_CONTEXT_ORDER = 110;
/**
 * Ask the host where a named context sits in the prompt assembly order.
 *
 * @param prompt - the host's `systemPrompt` service.
 * @param name - the named order slot, e.g. `SANDBOX_POLICY`.
 * @param log - the plugin's logger.
 * @returns the order, falling back to {@link DEFAULT_CONTEXT_ORDER}.
 */
export function contextOrderFor(prompt, name, log) {
    if (prompt === undefined || typeof prompt.getContextOrder !== 'function') {
        log.debug(`this host names no context order, so the capability snapshot uses order ${DEFAULT_CONTEXT_ORDER}`);
        return DEFAULT_CONTEXT_ORDER;
    }
    try {
        const order = prompt.getContextOrder(name);
        if (typeof order === 'number' && Number.isFinite(order))
            return order + 1;
        log.debug(`the host reported no order for "${name}", so the capability snapshot uses ${DEFAULT_CONTEXT_ORDER}`);
        return DEFAULT_CONTEXT_ORDER;
    }
    catch (error) {
        log.debug(`the host rejected the order query for "${name}" (${error instanceof Error ? error.message : String(error)}), so the capability snapshot uses ${DEFAULT_CONTEXT_ORDER}`);
        return DEFAULT_CONTEXT_ORDER;
    }
}
/**
 * Register the capability snapshot as model-visible runtime context.
 *
 * The whole registration is optional and contained: a host without the member, or
 * with a `context` that refuses the contribution, loses the snapshot and keeps
 * every tool. That ordering is deliberate — the tools are how the model *acts*,
 * and the snapshot is how it *knows*; losing the second must never cost the first.
 *
 * @param ctx - the host context.
 * @param name - the contribution name.
 * @param orderName - the named order slot to sit after.
 * @param text - produces the snapshot at assembly time.
 * @param log - the plugin's logger.
 * @returns true when the contribution is registered.
 */
export function installPromptContext(ctx, name, orderName, text, log) {
    const prompt = ctx.systemPrompt;
    if (prompt === undefined || typeof prompt.context !== 'function') {
        log.warn('the capability snapshot was not registered: this host exposes no `ctx.systemPrompt.context`, so the model will ' +
            'not be told which local capabilities exist until it calls list_capabilities');
        return false;
    }
    try {
        prompt.context({ name, order: contextOrderFor(prompt, orderName, log), text });
        return true;
    }
    catch (error) {
        log.warn(`the capability snapshot was refused by this host's prompt service: ${error instanceof Error ? error.message : String(error)}`);
        return false;
    }
}
/**
 * Run a callback in a scope where an optional service is present.
 *
 * This is the plugin's alternative to declaring a service in `inject`: a declared
 * dependency parks the plugin's fiber until the service arrives, so a deployment
 * without it — a headless or SDK profile — would lose every model tool waiting
 * for a skill registry it never wanted. Injecting instead means the callback
 * simply never runs there.
 *
 * When the host has no `ctx.inject` at all, the callback is run once against the
 * plugin's own context: `ctx.get(name)` then returns `undefined` and the caller's
 * own guard handles it. That is strictly better than skipping, because a host
 * that provides the service by another route still gets the integration.
 *
 * @param ctx - the host context.
 * @param service - the service name to scope on.
 * @param callback - the work to do inside the scope.
 * @param log - the plugin's logger.
 * @returns true when the callback was scheduled or run.
 */
export function withService(ctx, service, callback, log) {
    // Already present: run it now rather than scheduling. Cordis's inject is an
    // *awaited* dependency — it fires only when the service is provided, which for a
    // service already on this context is already true, and for a host whose inject
    // never fires would mean the integration silently never attaches. Running the
    // callback immediately is both faster and the more honest reading of "the
    // service is here".
    if (readMember(ctx, service) !== undefined) {
        try {
            callback(ctx);
            return true;
        }
        catch (error) {
            log.warn(`the ${service} integration could not be attached: ${error instanceof Error ? error.message : String(error)}`);
            return false;
        }
    }
    const inject = ctx.inject;
    if (typeof inject !== 'function') {
        log.debug(`this host exposes no \`ctx.inject\`, so the ${service} integration runs against the plugin context itself`);
        try {
            callback(ctx);
            return true;
        }
        catch (error) {
            log.warn(`the ${service} integration could not be attached: ${error instanceof Error ? error.message : String(error)}`);
            return false;
        }
    }
    try {
        inject.call(ctx, [service], callback);
        return true;
    }
    catch (error) {
        log.warn(`the ${service} integration could not be attached: ${error instanceof Error ? error.message : String(error)}`);
        return false;
    }
}
/**
 * Read an optional service without requiring it.
 *
 * `ctx.get` is the member that changed name most often in this project's history,
 * so the read is centralised: a host with `get`, with a `get` that throws for an
 * unregistered name, or with no `get` at all produces `undefined` rather than an
 * exception, and the caller's guard decides what that costs.
 *
 * @param ctx - the host context.
 * @param service - the service name.
 * @returns the service, or `undefined` when the host cannot provide it.
 */
export function optionalService(ctx, service) {
    const get = ctx.get;
    if (typeof get !== 'function')
        return undefined;
    try {
        return get.call(ctx, service);
    }
    catch {
        return undefined;
    }
}
/**
 * A logger that writes somewhere even when the host has none.
 *
 * The plugin logs from `apply` before anything else can fail, so a host whose
 * `logger` is missing must still produce those lines — otherwise a version
 * mismatch is silent, which is the one outcome this module exists to prevent. The
 * fallback writes to the console under a prefixed scope, matching the level.
 *
 * @param ctx - the host context.
 * @param scope - the logger scope name.
 * @returns a logger, from the host when it has one.
 */
export function pluginLogger(ctx, scope) {
    const logger = ctx.logger;
    if (typeof logger === 'function') {
        try {
            const created = logger.call(ctx, scope);
            if (created !== undefined && created !== null && typeof created.info === 'function') {
                return {
                    debug: (message) => created.debug?.(message),
                    info: (message) => created.info?.(message),
                    warn: (message) => created.warn?.(message),
                    error: (message) => created.error?.(message),
                };
            }
        }
        catch {
            // Fall through to the console logger below: a host logger that throws on
            // construction must not cost the diagnostics that would explain why.
        }
    }
    const write = (level) => (message) => {
        const line = `[${scope}] ${message}`;
        if (level === 'error')
            console.error(line);
        else if (level === 'warn')
            console.warn(line);
        else
            console.log(line);
    };
    return { debug: write('debug'), info: write('info'), warn: write('warn'), error: write('error') };
}
/**
 * Attach one HTTP route to the host's web server, if it has one.
 *
 * Both settings pages — the engine inventory and the capability priorities editor
 * — are registered this way. The service is reached through injection rather than
 * a declared `inject` for the same reason the skill registry is: a headless or
 * SDK profile has no web server, and declaring the dependency would park the
 * plugin's fiber waiting for a service that profile will never mount, costing the
 * agent every model tool in order to serve a page nobody can open.
 *
 * The `register` shape is checked, not assumed. A host whose web server gained a
 * different route-registration API loses its pages and keeps its tools, and says
 * so once.
 *
 * @param ctx - the host context.
 * @param path - the exact route path.
 * @param handler - the request handler.
 * @param what - a human description of the page, for the diagnostic.
 * @param log - the plugin's logger.
 * @returns true when the route is registered.
 */
export function installRoute(ctx, path, handler, what, log) {
    let installed = false;
    withService(ctx, 'webServer', (scope) => {
        // `serviceInScope`, not `optionalService`: the inject-aware proxy resolves an
        // injected name as a *member*, and `ctx.get` only answers for services the
        // cordis registry holds. Reading only through `get` would silently skip every
        // route on a host that resolves the proxy but keeps no registry entry.
        const server = serviceInScope(scope, 'webServer');
        if (server === undefined || typeof server.register !== 'function') {
            log.debug(`this host exposes no usable web server, so ${what} is not served on ${path}; every model tool still works`);
            return;
        }
        try {
            server.register({ kind: 'exact', path, handler: handler });
            installed = true;
            log.debug(`serving ${what} on ${path}`);
        }
        catch (error) {
            log.warn(`${what} could not be registered on ${path}: ${error instanceof Error ? error.message : String(error)}`);
        }
    }, log);
    return installed;
}
/**
 * Read a service out of a scope that `ctx.inject` produced.
 *
 * Cordis's inject-aware context proxy resolves the injected name as a property, so
 * `scope.webServer` and `scope.get('webServer')` are both correct on the versions
 * observed. This tries both, property first, so a host that dropped the `get`
 * mixin keeps working and a host that dropped the property accessor does too.
 *
 * @param scope - the injected scope.
 * @param name - the service name.
 * @returns the service, or `undefined`.
 */
export function serviceInScope(scope, name) {
    // Through the same named read the host observation uses, so this module has
    // exactly one place that touches a `ctx` member and the source audit in
    // `tests/plugin.test.ts` sees all of them.
    const direct = readMember(scope, name);
    if (direct !== undefined && direct !== null)
        return direct;
    return optionalService(scope, name);
}
/**
 * Read the calling session out of a tool run context, defensively.
 *
 * `exec.agent.session.cwd` is the documented path, but a run context is host data
 * crossing into plugin code, so every link is checked: a renamed field costs the
 * session-scoped artifact root (the hub falls back to its own store) and never a
 * thrown error inside a tool call.
 *
 * @param runContext - the tool call's run context.
 * @returns the session cwd, when the host supplies an absolute one.
 */
export function sessionCwdOf(runContext) {
    const agent = runContext?.agent;
    if (!isObjectLike(agent))
        return undefined;
    const session = agent.session;
    if (!isObjectLike(session))
        return undefined;
    const cwd = session.cwd;
    return typeof cwd === 'string' && cwd.length > 0 ? cwd : undefined;
}
