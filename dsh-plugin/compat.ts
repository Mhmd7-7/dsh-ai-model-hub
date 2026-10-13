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

import type { Context, Disposable } from '@deepseek-ai/cordis';
import type { ToolDefinition } from '@deepseek-ai/dsh-tools';
import type { PluginLogger, ToolCallScope } from './types.ts';

/**
 * The only thing this plugin needs from a host context to register a tool.
 *
 * Declared structurally instead of as cordis's `Context` so that a caller which
 * has already proved it holds a registry — and so a test with a three-line
 * stand-in — can call the helpers below without a cast.
 */
export interface ToolHostContext {
  /** The host's tool registry. */
  readonly tools: {
    /**
     * Register one tool definition.
     * @param definition - the definition to register.
     * @returns a disposer in every DSH version observed; ignored here, because the
     *   tool's lifetime is the plugin fiber's.
     */
    register(definition: ToolDefinition): unknown;
  };
}

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
export const REQUIRED_MEMBERS = ['tools'] as const;

/**
 * A named verdict about one host member.
 *
 * `present` means the member was found, and `usable` means it has the shape the
 * plugin needs — a member can be present and unusable (a `webServer` with no
 * `register`, a `tools` with no `register`), which is a different sentence in the
 * log and a different fix for whoever reads it.
 */
export interface MemberVerdict {
  /** The `ctx` member name, e.g. `tools`. */
  readonly member: string;
  /** Whether the member exists on the host context at all. */
  readonly present: boolean;
  /** Whether it has the shape the plugin needs. */
  readonly usable: boolean;
  /** What the plugin does with it. */
  readonly purpose: string;
  /** A sentence naming the consequence when it is not usable. */
  readonly consequence?: string;
}

/** The context members the plugin reads, in the order the plugin first needs them. */
const MEMBER_PURPOSES: readonly { readonly member: string; readonly purpose: string; readonly needs: string }[] = [
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
 * What the host actually offers.
 *
 * Read once at the top of `apply` and passed down, so the plugin's decisions
 * about what to register are all made against one consistent observation instead
 * of nine independent guesses.
 */
export interface CompatHost {
  /** The raw context, for the members that need no wrapping. */
  readonly ctx: Context;
  /** One verdict per member in {@link MEMBER_PURPOSES}. */
  readonly members: readonly MemberVerdict[];
  /** Whether every member the plugin cannot work without is usable. */
  readonly complete: boolean;
  /**
   * Human-readable lines describing anything missing or unusable, ready to log.
   *
   * Empty when the host has everything. Each line names the member, what it is
   * for, and what is lost without it, so an operator reading one warning knows
   * both the consequence and the fix.
   */
  readonly diagnostics: readonly string[];
}

/**
 * Whether a value is a non-null object or function, the only two host shapes.
 * @param value - the candidate.
 * @returns true when the value can carry members.
 */
function isObjectLike(value: unknown): value is Record<string, unknown> {
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
function readMember(ctx: Context, member: string): unknown {
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
function judgeMember(
  ctx: Context,
  spec: { readonly member: string; readonly purpose: string; readonly needs: string },
): MemberVerdict {
  const value = readMember(ctx, spec.member);
  const present = value !== undefined && value !== null;
  const needsMember = /`([A-Za-z]+)`/.exec(spec.needs)?.[1];
  const usable =
    present &&
    (needsMember === undefined
      ? isObjectLike(value) || typeof value === 'string'
      : isObjectLike(value) && typeof (value as Record<string, unknown>)[needsMember] === 'function');
  const judgement: MemberVerdict = {
    member: spec.member,
    present,
    usable,
    purpose: spec.purpose,
  };
  if (usable) return judgement;
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
export function resolveCompatHost(ctx: Context): CompatHost {
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
export function memberVerdict(host: CompatHost, member: string): MemberVerdict | undefined {
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
export function reportCompatHost(host: CompatHost, log: PluginLogger): number {
  for (const verdict of host.members) {
    if (verdict.usable) continue;
    const line = `${verdict.member}: ${verdict.consequence ?? 'unavailable'}`;
    if (verdict.present) log.warn(line);
    else log.debug(line);
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
export function installTool(ctx: ToolHostContext, definition: ToolDefinition, log: PluginLogger): boolean {
  const registry = (ctx as { readonly tools?: { register?(definition: ToolDefinition): unknown } }).tools;
  if (registry === undefined || typeof registry.register !== 'function') {
    log.warn(`the tool ${definition.name} was not registered: this host exposes no tool registry`);
    return false;
  }
  try {
    registry.register(definition);
    return true;
  } catch (error) {
    log.warn(
      `the tool ${definition.name} was refused by this host's tool registry: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
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
export function installDefinedTool(
  ctx: ToolHostContext,
  definition: ToolDefinition,
  log: PluginLogger,
): boolean {
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
export function installEffect(
  ctx: Context,
  execute: () => unknown,
  label: string,
  log: PluginLogger,
): Disposable<Promise<void>> | undefined {
  const effect = ctx.effect as unknown;
  if (typeof effect !== 'function') {
    log.warn(
      `${label}: this host exposes no \`ctx.effect\`, so this cleanup is not bound to the plugin lifetime and will not ` +
        'run when the plugin unloads',
    );
    return undefined;
  }
  try {
    return effect(execute, label);
  } catch (error) {
    log.warn(`${label}: could not be installed (${error instanceof Error ? error.message : String(error)})`);
    return undefined;
  }
}

/** The prompt-context contribution this plugin registers. */
export interface PromptContextContribution {
  /** Contribution name, shown in prompt assembly diagnostics. */
  readonly name: string;
  /** Order among runtime contexts; lower sorts earlier. */
  readonly order: number;
  /** The text, or a function producing it at assembly time. */
  readonly text: () => string;
}

/** The `systemPrompt` subset this plugin uses. */
interface SystemPromptLike {
  context?(contribution: { name: string; order: number; text: () => string }): unknown;
  getContextOrder?(name: string): number;
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
export function contextOrderFor(prompt: SystemPromptLike | undefined, name: string, log: PluginLogger): number {
  if (prompt === undefined || typeof prompt.getContextOrder !== 'function') {
    log.debug(`this host names no context order, so the capability snapshot uses order ${DEFAULT_CONTEXT_ORDER}`);
    return DEFAULT_CONTEXT_ORDER;
  }
  try {
    const order = prompt.getContextOrder(name);
    if (typeof order === 'number' && Number.isFinite(order)) return order + 1;
    log.debug(`the host reported no order for "${name}", so the capability snapshot uses ${DEFAULT_CONTEXT_ORDER}`);
    return DEFAULT_CONTEXT_ORDER;
  } catch (error) {
    log.debug(
      `the host rejected the order query for "${name}" (${
        error instanceof Error ? error.message : String(error)
      }), so the capability snapshot uses ${DEFAULT_CONTEXT_ORDER}`,
    );
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
export function installPromptContext(
  ctx: Context,
  name: string,
  orderName: string,
  text: () => string,
  log: PluginLogger,
): boolean {
  const prompt = ctx.systemPrompt as SystemPromptLike | undefined;
  if (prompt === undefined || typeof prompt.context !== 'function') {
    log.warn(
      'the capability snapshot was not registered: this host exposes no `ctx.systemPrompt.context`, so the model will ' +
        'not be told which local capabilities exist until it calls list_capabilities',
    );
    return false;
  }
  try {
    prompt.context({ name, order: contextOrderFor(prompt, orderName, log), text });
    return true;
  } catch (error) {
    log.warn(
      `the capability snapshot was refused by this host's prompt service: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    return false;
  }
}

/** The shape `ctx.inject` has, in every DSH version observed. */
type InjectLike = (deps: readonly string[], callback: (scope: Context) => void) => unknown;

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
export function withService(
  ctx: Context,
  service: string,
  callback: (scope: Context) => void,
  log: PluginLogger,
): boolean {
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
    } catch (error) {
      log.warn(`the ${service} integration could not be attached: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  const inject = ctx.inject as unknown as InjectLike | undefined;
  if (typeof inject !== 'function') {
    log.debug(
      `this host exposes no \`ctx.inject\`, so the ${service} integration runs against the plugin context itself`,
    );
    try {
      callback(ctx);
      return true;
    } catch (error) {
      log.warn(`the ${service} integration could not be attached: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }
  try {
    inject.call(ctx, [service], callback);
    return true;
  } catch (error) {
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
export function optionalService<T>(ctx: Context, service: string): T | undefined {
  const get = ctx.get as unknown as ((name: string) => unknown) | undefined;
  if (typeof get !== 'function') return undefined;
  try {
    return get.call(ctx, service) as T | undefined;
  } catch {
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
export function pluginLogger(ctx: Context, scope: string): PluginLogger {
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
    } catch {
      // Fall through to the console logger below: a host logger that throws on
      // construction must not cost the diagnostics that would explain why.
    }
  }
  const write = (level: 'debug' | 'info' | 'warn' | 'error') => (message: string) => {
    const line = `[${scope}] ${message}`;
    if (level === 'error') console.error(line);
    else if (level === 'warn') console.warn(line);
    else console.log(line);
  };
  return { debug: write('debug'), info: write('info'), warn: write('warn'), error: write('error') };
}

/**
 * A host route handler, as the web server calls it.
 *
 * Declared with `unknown` request and response types on purpose: this plugin only
 * ever reads a URL and writes a status, headers, and a body, and typing those
 * through a DSH-specific HTTP type would make a host that vendored a different
 * `node:http` shape unassignable. The concrete types are recovered at the two call
 * sites with a narrow structural interface.
 */
export interface RouteHandler {
  /**
   * Answer one request.
   * @param request - the incoming request.
   * @param response - the response to write.
   * @returns nothing; the handler writes the response itself.
   */
  (request: never, response: never): void | Promise<void>;
}

/** The `webServer` subset this plugin uses. */
export interface WebServerLike {
  /**
   * Register one exact route.
   * @param route - the path, its kind, and the handler.
   * @returns the disposer that unregisters it.
   */
  register(route: { kind: 'exact'; path: string; handler: RouteHandler }): () => void;
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
export function installRoute(
  ctx: Context,
  path: string,
  handler: (request: unknown, response: unknown) => void | Promise<void>,
  what: string,
  log: PluginLogger,
): boolean {
  let installed = false;
  withService(
    ctx,
    'webServer',
    (scope) => {
      // `serviceInScope`, not `optionalService`: the inject-aware proxy resolves an
      // injected name as a *member*, and `ctx.get` only answers for services the
      // cordis registry holds. Reading only through `get` would silently skip every
      // route on a host that resolves the proxy but keeps no registry entry.
      const server = serviceInScope<WebServerLike>(scope, 'webServer');
      if (server === undefined || typeof server.register !== 'function') {
        log.debug(
          `this host exposes no usable web server, so ${what} is not served on ${path}; every model tool still works`,
        );
        return;
      }
      try {
        server.register({ kind: 'exact', path, handler: handler as RouteHandler });
        installed = true;
        log.debug(`serving ${what} on ${path}`);
      } catch (error) {
        log.warn(
          `${what} could not be registered on ${path}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    },
    log,
  );
  return installed;
}

/**
 * The `tools` registry subset the subscription guard uses.
 *
 * A different member of the same service the tools are registered on: reading it
 * separately means a host that has `register` but not `guard` still gets every
 * tool, with one candid warning about the guard it cannot install.
 */
export interface ToolGuardRegistry {
  /**
   * Install a pre-dispatch guard.
   * @param guard - returns a refusal reason, or `undefined` to allow the call.
   * @returns the disposer that uninstalls the guard.
   */
  guard(check: (execution: { readonly name: string; readonly args: Readonly<Record<string, unknown>> }) => string | undefined): () => void;
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
export function serviceInScope<T>(scope: Context, name: string): T | undefined {
  // Through the same named read the host observation uses, so this module has
  // exactly one place that touches a `ctx` member and the source audit in
  // `tests/plugin.test.ts` sees all of them.
  const direct = readMember(scope, name);
  if (direct !== undefined && direct !== null) return direct as T;
  return optionalService<T>(scope, name);
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
export function sessionCwdOf(runContext: ToolCallScope | undefined): string | undefined {
  const agent = (runContext as { readonly agent?: unknown } | undefined)?.agent;
  if (!isObjectLike(agent)) return undefined;
  const session = (agent as { readonly session?: unknown }).session;
  if (!isObjectLike(session)) return undefined;
  const cwd = (session as { readonly cwd?: unknown }).cwd;
  return typeof cwd === 'string' && cwd.length > 0 ? cwd : undefined;
}

