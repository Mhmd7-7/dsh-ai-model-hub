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
