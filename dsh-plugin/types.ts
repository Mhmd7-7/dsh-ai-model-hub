/**
 * Type bridges for the DeepSeek Harness plugin API.
 *
 * The DSH plugin layer pins its dependency on DSH to this one file plus the
 * handful of `@deepseek-ai/*` imports in the modules beside it. That is a
 * deliberate survivability decision: DSH is a `0.1.x-rc` line, so breaking
 * changes are expected. When one lands, the blast radius is here — not in the
 * catalog, router, runtime, or adapters, which import nothing from DSH at all and
 * are covered by tests that run without DSH installed.
 *
 * Everything added to the `Context` interface below comes from cordis's runtime
 * mixins. They exist at runtime but are absent from the published `Context`
 * declaration, so the augmentation records the real, verified behaviour rather
 * than guessing at an API. Each is documented with the source that establishes it.
 *
 * @module dsh-ai-model-hub/dsh-plugin/types
 */

import type { Disposable, Effect } from '@deepseek-ai/cordis';

declare module '@deepseek-ai/cordis' {
  interface Context {
    /**
     * Register a cleanup-aware effect on the plugin's fiber.
     *
     * Mixed onto `Context` by `RegistryService`, whose methods "are mixed onto
     * `ctx`". The effect body runs immediately; whatever disposer it returns (or
     * yields, or resolves to) runs when the returned disposer is called or when
     * the fiber unloads, whichever comes first.
     *
     * This is how the hub's lifetime is bound to the plugin: unloading the plugin
     * stops every model process the hub started, including a runaway engine.
     *
     * @param execute - produces the disposer(s) to run on teardown.
     * @param label - effect label shown in fiber diagnostics.
     * @returns a disposer that tears the effect down and settles once done.
     */
    effect(execute: () => Effect, label?: string): Disposable<Promise<void>>;
  }
}

/**
 * The subset of DSH's sandbox-policy service this plugin borrows.
 *
 * Read through `ctx.get('sandboxPolicy')`, never through a declared `inject`: the
 * hub must keep loading in a composition without it (a headless or SDK profile),
 * and it is only ever consulted to place artifacts, so its absence degrades one
 * default rather than the plugin.
 *
 * `resolve` is verified against `SandboxPolicyService` in
 * `@deepseek-ai/dsh-sandbox-policy`: an optional session and approved mode in,
 * the fully resolved `{ mode, workspaceRoot }` out, where a session's immutable
 * `cwd` becomes `workspaceRoot` and the deployment's configured root is only the
 * fallback for calls that carry no session.
 */
export interface SandboxPolicyService {
  /**
   * Resolve the policy for one capability call.
   * @param request - the calling session, when there is one.
   * @returns the per-call mode and absolute workspace root.
   */
  resolve(request?: { readonly session?: unknown }): { readonly mode: string; readonly workspaceRoot: string };
}

/**
 * The shape of a tool call this plugin reads its session from.
 *
 * Declared structurally rather than imported, for the same reason as everything
 * else in this file: the plugin pins its DSH coupling to the few facts it
 * verifies, and `ToolRunContext.agent.session.cwd` is one of them.
 */
export interface ToolCallScope {
  /** The calling agent, when the outer call has one. */
  readonly agent?: { readonly session?: { readonly cwd?: string } };
}

/**
 * The subset of the DSH skill registry this plugin borrows.
 *
 * `skills` is read through `ctx.get('skills')`, never through a declared
 * `inject`, because the hub must keep loading in a deployment that has no skill
 * registry at all — a headless or SDK profile, or any composition that omits
 * `@deepseek-ai/dsh-skill`. Declaring it in `inject` would park the plugin's
 * fiber forever waiting for a service that never arrives, which costs the agent
 * every model tool rather than one skill.
 *
 * `registerProvider` is verified against `SkillRegistry` in
 * `@deepseek-ai/dsh-skill`: it takes a synchronous factory receiving this
 * registration's lifecycle control, registers into the calling context's layer,
 * and returns the disposer that unregisters it.
 */
export interface SkillsService {
  /**
   * Register a skill provider owned by the current fiber.
   * @param create - factory receiving this registration's control handles.
   * @returns the disposer that unregisters the provider.
   */
  registerProvider(create: (control: SkillProviderControl) => SkillProvider): () => void;
}

/**
 * Registration-scoped lifecycle handles handed to a provider factory.
 *
 * Only `signal` is used: it aborts when this exact registration is disposed, so a
 * provider can stop in-flight work instead of touching a registry it no longer
 * belongs to.
 */
export interface SkillProviderControl {
  /** Aborts when this exact provider registration is disposed. */
  readonly signal: AbortSignal;
}

/** Invocation controls shared by skill discovery consumers. */
export interface SkillInvocationPolicy {
  /** Whether model-facing catalogs and loaders include this skill. */
  readonly modelInvocable: boolean;
  /** Whether human-facing command catalogs and loaders include this skill. */
  readonly userInvocable: boolean;
}

/** Skill metadata a provider returns from `list`, before the body is loaded. */
export interface SkillCandidate {
  /** Kebab-case identifier used to address the skill. */
  readonly name: string;
  /** Short routing description shown by discovery consumers. */
  readonly description: string;
  /** Optional extra routing guidance. */
  readonly whenToUse?: string;
  /** Resolved model and user invocation controls. */
  readonly invocation: SkillInvocationPolicy;
  /** Discovery source bucket; prompt-visible metadata, not precedence. */
  readonly source: string;
  /** Provider that owns this skill body. */
  readonly provider: string;
  /** Provider-specific base used by the loaded body to resolve relative resources. */
  readonly resourceBase?: SkillResourceBase;
  /** Lower ranks win duplicate skill names before provider registration order. */
  readonly rank: number;
  /** Opaque provider-owned handle passed back to `get`. */
  readonly locator: unknown;
  /** Absolute file path when the provider has one. */
  readonly path?: string;
}

/** A complete skill, including the body the registry hands to the model. */
export interface SkillDefinition {
  /** Kebab-case identifier used to address the skill. */
  readonly name: string;
  /** Short routing description shown by discovery consumers. */
  readonly description: string;
  /** Optional extra routing guidance. */
  readonly whenToUse?: string;
  /** Resolved model and user invocation controls. */
  readonly invocation: SkillInvocationPolicy;
  /** Discovery source bucket; prompt-visible metadata, not precedence. */
  readonly source: string;
  /** Provider that owns this skill body. */
  readonly provider: string;
  /** Provider-specific base used by the body to resolve relative resources. */
  readonly resourceBase?: SkillResourceBase;
  /** Markdown instruction body, with provider metadata already removed. */
  readonly content: string;
  /** Absolute file path when the skill came from disk. */
  readonly path?: string;
}

/** Provider-specific base for resolving a loaded body's relative resources. */
export type SkillResourceBase = { readonly kind: 'directory'; readonly path: string };

/**
 * One source of skills, as consumed by the registry.
 *
 * `list` may return an incomplete observation instead of an array; this provider
 * never needs that, because one file on disk either reads or does not.
 */
export interface SkillProvider {
  /** Unique provider name in the `ctx.skills` registry. */
  readonly name: string;
  /**
   * List the candidates this provider currently offers.
   * @param options - lookup options; only `signal` is consulted here.
   * @returns the provider's candidates.
   */
  list(options: { readonly signal?: AbortSignal | undefined }): Promise<readonly SkillCandidate[]>;
  /**
   * Load one candidate's full body.
   * @param candidate - a candidate previously returned by {@link list}.
   * @param options - lookup options; only `signal` is consulted here.
   * @returns the loaded skill, or `undefined` when it is no longer loadable.
   */
  get(
    candidate: SkillCandidate,
    options: { readonly signal?: AbortSignal | undefined },
  ): Promise<SkillDefinition | undefined>;
}

/**
 * The subset of a cordis logger the plugin uses.
 *
 * Mirrors `Logger` from `@deepseek-ai/cordis` without importing it, so the tool
 * modules do not each take a dependency on cordis just for a log signature.
 */
export interface PluginLogger {
  /**
   * Log a diagnostic detail.
   * @param message - what happened.
   */
  debug(message: string): void;
  /**
   * Log a normal lifecycle event.
   * @param message - what happened.
   */
  info(message: string): void;
  /**
   * Log a recoverable problem.
   * @param message - what happened.
   */
  warn(message: string): void;
  /**
   * Log a failure.
   * @param message - what happened.
   */
  error(message: string): void;
}
