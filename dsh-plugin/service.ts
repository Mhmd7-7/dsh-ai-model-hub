/**
 * The `modelHub` service: the hub exposed as a cordis service.
 *
 * Registering the hub on the context does three things at once:
 *
 * 1. Other DSH plugins can consume it with `inject: ['modelHub']` instead of
 *    re-reading configuration and constructing their own.
 * 2. Its lifetime is bound to the plugin fiber, so unloading the plugin disposes
 *    the hub â€” which stops every process the hub started.
 * 3. The tool layer resolves the hub through the context rather than through a
 *    module-level singleton, which is what lets a test mount the whole plugin
 *    against a hub it built itself.
 *
 * The service adds no behaviour: it is the hub plus cordis lifetime management.
 * A service that wrapped the hub's API would be a second API to keep in sync, and
 * the entire point is that there is one.
 *
 * @module dsh-ai-model-hub/dsh-plugin/service
 */

import { Service } from '@deepseek-ai/cordis';
import type { Context } from '@deepseek-ai/cordis';
import type { ModelHub } from '../src/hub.ts';
import { ModelHub as Hub } from '../src/hub.ts';
import type { ModelCatalogConfig } from '../src/index.ts';

/** The cordis service name other plugins inject. */
export const MODEL_HUB_SERVICE = 'modelHub';

/** Construction options for {@link ModelHubService}. */
export interface ModelHubServiceOptions {
  /**
   * A hub to publish instead of constructing one.
   *
   * Tests use this to mount the plugin against a hub with an injected machine
   * profile or a substituted adapter, without a second configuration file.
   */
  readonly hub?: ModelHub;
  /** Catalog document used when {@link hub} is not supplied. */
  readonly config?: ModelCatalogConfig;
  /** Artifact root used when {@link hub} is not supplied. */
  readonly artifactRoot?: string;
  /** Diagnostic sink forwarded to a hub constructed here. */
  readonly log?: (message: string) => void;
}

/**
 * The hub, published under {@link MODEL_HUB_SERVICE}.
 *
 * Disposal is the plugin's responsibility (see `index.ts`) rather than the
 * service's, because the hub outlives an individual service instance whenever
 * another plugin owns it.
 */
export class ModelHubService extends Service<never> {
  /** The hub instance this service publishes. */
  readonly hub: ModelHub;

  /**
   * @param ctx - the registrant context.
   * @param hub - the hub to publish.
   */
  constructor(ctx: Context, hub: ModelHub) {
    super(ctx, MODEL_HUB_SERVICE);
    this.hub = hub;
  }

  /**
   * Build a hub from a catalog document and publish it.
   *
   * The synchronous constructor cannot await machine detection, so a hub built
   * this way reports an unprobed machine and therefore treats every model's
   * declared resources as satisfiable. That is the correct conservative default:
   * refusing everything because detection had not run yet would be worse.
   *
   * @param ctx - the registrant context.
   * @param options - an existing hub, or a catalog document to build one from.
   * @returns the service.
   * @throws ModelHubError when neither a hub nor a config is supplied.
   */
  static from(ctx: Context, options: ModelHubServiceOptions): ModelHubService {
    if (options.hub !== undefined) return new ModelHubService(ctx, options.hub);
    if (options.config === undefined) {
      throw new TypeError('ModelHubService.from requires either `hub` or `config`');
    }
    const hub = new Hub({
      config: options.config,
      ...(options.artifactRoot === undefined ? {} : { artifactRoot: options.artifactRoot }),
      ...(options.log === undefined ? {} : { log: (message) => options.log?.(message) }),
    });
    return new ModelHubService(ctx, hub);
  }
}

declare module '@deepseek-ai/cordis' {
  interface Context {
    /** The AI model hub, when the `dsh-ai-model-hub` plugin is mounted. */
    modelHub: ModelHubService;
  }
}
