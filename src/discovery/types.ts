/**
 * The runtime model-discovery layer.
 *
 * A *host* is the only thing an operator configures: "ComfyUI is at
 * http://127.0.0.1:8188", "Ollama is at http://127.0.0.1:11434". Everything
 * about what that engine can actually do *right now* — which checkpoints are on
 * disk, which of them has a vision projector, which node packs are installed —
 * is read out of the engine's own introspection API at runtime and synthesized
 * into {@link ModelDescriptor} entries.
 *
 * Three boundaries hold this together, and each one exists for a reason:
 *
 * 1. **Discovery produces descriptors, never {@link ResolvedModel}.** A
 *    discovered entry is exactly the same *shape* as a hand-written
 *    `config/models.json` entry. `resolveDescriptor` is still the only thing
 *    that turns a descriptor into a resolved model, and it still lives in
 *    `catalog/registry.ts`. Discovery never calls it, so inheritance, defaults,
 *    and validation cannot diverge between static and discovered models.
 * 2. **Parsing is separate from mapping.** Every discoverer exposes a pure
 *    `parse*` function (engine response → candidate records) and a pure `map*`
 *    function (candidate records → descriptors). Each is unit-testable without
 *    a server, and the response shape is the only thing that knows anything
 *    engine-specific.
 * 3. **Static configuration wins.** {@link mergeCatalogConfig} places every
 *    static entry before every discovered one and drops a discovered entry whose
 *    id a static entry already claims. It runs *before* `new ModelCatalog(…)`,
 *    so `ModelCatalog`'s constructor never sees a duplicate id and its
 *    "skip the later one, log a diagnostic" behaviour is never load-bearing.
 *
 * Nothing in this module names a model, a checkpoint, or a capability. The
 * engine vocabulary lives in one place per engine (`src/discovery/<engine>.ts`),
 * and the capability vocabulary is the catalog's own (`CAPABILITY_IO`).
 *
 * @module dsh-ai-model-hub/discovery/types
 */

import type { Capability, IoType } from '../catalog/capabilities.ts';
import { CAPABILITY_IO } from '../catalog/capabilities.ts';
import type { ModelCatalogConfig, ModelDescriptor, ModelHost } from '../catalog/descriptor.ts';

/**
 * The default {@link ModelDescriptor.priority} for a discovered model.
 *
 * `DEFAULTS.priority` in `catalog/descriptor.ts` is 100 for a descriptor that
 * states none. Discovered entries sit well below that, so a static model of the
 * same capability is preferred by the router whenever both remain eligible.
 *
 * This is a routing tie-break and nothing more: it is *not* what makes a static
 * model win on an id collision. That is {@link mergeCatalogConfig}'s job, and
 * relying on priority instead would be wrong — two models with the same id can
 * never both be in the catalog in the first place.
 */
export const DISCOVERED_PRIORITY = 500;

/** The default discovery cache lifetime, in milliseconds. */
export const DEFAULT_DISCOVERY_TTL_MS = 60_000;

/** The default budget for one engine's whole discovery pass, in milliseconds. */
export const DEFAULT_DISCOVERY_TIMEOUT_MS = 5_000;

/**
 * One engine's introspection client.
 *
 * A discoverer is keyed by `engine`, which must equal the host's
 * `runtime.engine` — the free-form label already carried on every host. It is
 * the *host* that decides which engines exist; a discoverer only decides what to
 * ask one.
 *
 * Implementations must be fail-soft: an unreachable engine, a timeout, or a
 * response in an unexpected shape is a warning and an empty (or partial) result,
 * never a throw that reaches the caller. {@link DiscoveredDescriptor} records
 * that contract, and every shipped discoverer honours it — but
 * {@link DiscoveryRegistry#generate} contains a throwing discoverer anyway,
 * because a third-party one will eventually forget.
 */
export interface HostDiscoverer {
  /** Matches {@link ModelHost.runtime.engine}. */
  readonly engine: string;
  /**
   * Additional engine labels this discoverer answers to.
   *
   * `runtime.engine` is a free-form operator label, and one engine family
   * genuinely has several names in the wild — the automatic1111 WebUI, its Forge
   * fork, and a stable-diffusion.cpp server all serve the same `/sdapi/v1` REST
   * shape under different labels. Listing the aliases here keeps that plurality
   * in one place instead of forcing an operator to guess the label this code
   * happens to expect.
   *
   * The primary {@link engine} is always implied; aliases only add.
   */
  readonly aliases?: readonly string[];
  /**
   * Ask one engine what it can currently serve.
   *
   * @param host - the configured host to introspect; `host.runtime.endpoint` is
   *   the base URL to reach it at, and `host.runtime.engine` identifies the
   *   engine family this discoverer handles.
   * @param signal - cancellation for the whole pass, including its own deadline.
   *   Forward it to every request.
   * @returns descriptors for what the engine reported. Never throws for a
   *   transport or shape problem.
   */
  discover(host: ModelHost, signal: AbortSignal): Promise<ModelDescriptor[]>;
}

/** A warning emitted while discovering one host. */
export interface DiscoveryWarning {
  /** The host the warning is about. */
  readonly hostId: string;
  /** Which engine family it came from. */
  readonly engine: string;
  /** What went wrong, phrased for an operator. */
  readonly message: string;
}

/** What one discovery pass produced. */
export interface DiscoveryResult {
  /** Every discovered descriptor, in host order then engine order. */
  readonly descriptors: readonly ModelDescriptor[];
  /** Non-fatal problems, one per host that could not be read. */
  readonly warnings: readonly DiscoveryWarning[];
  /** Whether the result came from the cache rather than a fresh pass. */
  readonly cached: boolean;
  /** How long the pass took, in milliseconds. `0` for a cache hit. */
  readonly durationMs: number;
  /** The hosts that were asked, in order. */
  readonly hostIds: readonly string[];
}

/** Options for {@link DiscoveryRegistry}. */
export interface DiscoveryOptions {
  /** How long a per-host result stays fresh, in milliseconds. */
  readonly ttlMs?: number;
  /** Budget for one host's discovery pass, in milliseconds. */
  readonly timeoutMs?: number;
  /** Diagnostic sink; defaults to silence. */
  readonly log?: (message: string, fields?: Readonly<Record<string, unknown>>) => void;
}

/** One host's cached discovery result. */
interface CacheEntry {
  /** When the pass that filled this entry finished, as epoch milliseconds. */
  readonly at: number;
  /** The descriptors it produced. */
  readonly descriptors: readonly ModelDescriptor[];
  /** The warnings it produced. */
  readonly warnings: readonly DiscoveryWarning[];
  /** How long the pass took. */
  readonly durationMs: number;
}

/**
 * The engine → discoverer map, mirroring `AdapterRegistry`.
 *
 * Lookup is by the host's own declared `runtime.engine`, never by host id, model
 * id, or name — the same rule the adapter registry follows. Registering a second
 * discoverer for one engine replaces the first and returns a disposer that
 * restores it, which is how a test substitutes a fixture and how a deployment
 * overrides a built-in.
 */
export class DiscoveryRegistry {
  private readonly discoverers = new Map<string, HostDiscoverer>();
  private readonly ttlMs: number;
  private readonly timeoutMs: number;
  private readonly log: (message: string, fields?: Readonly<Record<string, unknown>>) => void;
  private readonly cache = new Map<string, CacheEntry>();
  private readonly inFlight = new Map<string, Promise<CacheEntry>>();

  /**
   * @param discoverers - discoverers to register immediately.
   * @param options - cache lifetime, per-host budget, and diagnostics sink.
   */
  constructor(discoverers: readonly HostDiscoverer[] = [], options: DiscoveryOptions = {}) {
    this.ttlMs = Math.max(0, options.ttlMs ?? DEFAULT_DISCOVERY_TTL_MS);
    this.timeoutMs = Math.max(1, options.timeoutMs ?? DEFAULT_DISCOVERY_TIMEOUT_MS);
    this.log = options.log ?? ((): void => {});
    for (const discoverer of discoverers) this.register(discoverer);
  }

  /**
   * Register a discoverer, replacing any previous one for the same engine.
   *
   * One discoverer is stored under its primary engine *and* under every alias it
   * declares, so a host labelled `forge` reaches the same instance as one
   * labelled `a1111`.
   *
   * @param discoverer - the discoverer.
   * @returns a disposer that restores whatever was registered before.
   */
  register(discoverer: HostDiscoverer): () => void {
    const keys = [discoverer.engine, ...(discoverer.aliases ?? [])];
    const restore = keys.map((key) => {
      const previous = this.discoverers.get(key);
      this.discoverers.set(key, discoverer);
      return () => {
        if (previous === undefined) this.discoverers.delete(key);
        else this.discoverers.set(key, previous);
      };
    });
    this.cache.clear();
    return () => {
      for (const undo of restore.reverse()) undo();
      this.cache.clear();
    };
  }

  /**
   * The discoverer for an engine.
   * @param engine - the engine label.
   * @returns the discoverer, or `undefined` when this engine has none.
   */
  get(engine: string): HostDiscoverer | undefined {
    return this.discoverers.get(engine);
  }

  /** Every engine a discoverer is registered for, in registration order. */
  listEngines(): readonly string[] {
    return [...this.discoverers.keys()];
  }

  /**
   * Forget every cached result, so the next pass hits the engines again.
   *
   * This is the "I just pulled a new model" path: it is what makes a manual
   * refresh possible without waiting out the TTL.
   */
  clear(): void {
    this.cache.clear();
  }

  /** When the cache entry for a host was filled, or `undefined` when absent. */
  cacheAge(hostId: string, now = Date.now()): number | undefined {
    const entry = this.cache.get(hostId);
    return entry === undefined ? undefined : now - entry.at;
  }

  /**
   * Discover every host that (a) is enabled, (b) names an engine a discoverer
   * handles, and (c) declares an endpoint to reach.
   *
   * Hosts are asked concurrently — they are independent services, and a slow
   * engine must not serialize a fast one behind it.
   *
   * @param hosts - the configured hosts, in configuration order.
   * @param options - `refresh` forces a pass even when a fresh cache entry exists.
   * @returns the merged descriptors plus per-host warnings.
   */
  async generate(
    hosts: readonly ModelHost[],
    options: { readonly refresh?: boolean; readonly signal?: AbortSignal } = {},
  ): Promise<DiscoveryResult> {
    const started = Date.now();
    const eligible = hosts.filter(
      (host) => host.enabled !== false && this.discoverers.has(host.runtime.engine) && hasEndpoint(host),
    );

    if (eligible.length === 0) {
      return { descriptors: [], warnings: [], cached: false, durationMs: 0, hostIds: [] };
    }

    if (options.refresh !== true) {
      // A cache hit requires *every* eligible host to have a fresh entry. A
      // partial hit would otherwise hide the hosts that just came online, and an
      // entry older than the TTL is not a hit at all — that check is the whole
      // point of having a TTL.
      const now = Date.now();
      const allFresh = eligible.every((host) => {
        const entry = this.cache.get(host.id);
        return entry !== undefined && now - entry.at < this.ttlMs;
      });
      if (allFresh) {
        const descriptors: ModelDescriptor[] = [];
        const warnings: DiscoveryWarning[] = [];
        for (const host of eligible) {
          const entry = this.cache.get(host.id);
          if (entry === undefined) continue;
          descriptors.push(...entry.descriptors);
          warnings.push(...entry.warnings);
        }
        return { descriptors, warnings, cached: true, durationMs: 0, hostIds: eligible.map((host) => host.id) };
      }
    }

    const perHost = await Promise.all(
      eligible.map(async (host): Promise<CacheEntry> => {
        const entry = await this.discoverHost(host, options.signal);
        this.cache.set(host.id, entry);
        return entry;
      }),
    );

    const descriptors: ModelDescriptor[] = [];
    const warnings: DiscoveryWarning[] = [];
    for (const entry of perHost) {
      descriptors.push(...entry.descriptors);
      warnings.push(...entry.warnings);
      for (const warning of entry.warnings) this.log(`discovery: ${warning.hostId}: ${warning.message}`, { engine: warning.engine });
    }

    return {
      descriptors,
      warnings,
      cached: false,
      durationMs: Date.now() - started,
      hostIds: eligible.map((host) => host.id),
    };
  }

  /**
   * Ask one host, containing every failure mode into a warning.
   *
   * A pass is deduplicated by host id: two concurrent callers (a pre-warm and a
   * first tool call, say) share one round of network traffic rather than racing.
   *
   * @param host - the host to introspect.
   * @param outer - cancellation supplied by the caller, if any.
   * @returns the cache entry to store.
   */
  private discoverHost(host: ModelHost, outer: AbortSignal | undefined): Promise<CacheEntry> {
    const pending = this.inFlight.get(host.id);
    if (pending !== undefined) return pending;

    const task = this.runDiscoverer(host, outer).finally(() => {
      this.inFlight.delete(host.id);
    });
    this.inFlight.set(host.id, task);
    return task;
  }

  /**
   * Run one discoverer under a deadline, converting every throw into a warning.
   * @param host - the host to introspect.
   * @param outer - cancellation supplied by the caller, if any.
   * @returns the cache entry to store.
   */
  private async runDiscoverer(host: ModelHost, outer: AbortSignal | undefined): Promise<CacheEntry> {
    const engine = host.runtime.engine;
    const discoverer = this.discoverers.get(engine);
    const started = Date.now();
    const warnings: DiscoveryWarning[] = [];
    if (discoverer === undefined) {
      return { at: started, descriptors: [], warnings, durationMs: 0 };
    }

    const controller = new AbortController();
    const onOuterAbort = (): void => controller.abort();
    if (outer !== undefined) {
      if (outer.aborted) controller.abort();
      else outer.addEventListener('abort', onOuterAbort, { once: true });
    }
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const descriptors = await discoverer.discover(host, controller.signal);
      return { at: Date.now(), descriptors, warnings, durationMs: Date.now() - started };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      warnings.push({ hostId: host.id, engine, message: `discovery failed: ${message}` });
      return { at: Date.now(), descriptors: [], warnings, durationMs: Date.now() - started };
    } finally {
      clearTimeout(timer);
      if (outer !== undefined) outer.removeEventListener('abort', onOuterAbort);
    }
  }
}

/**
 * Whether a host declares an endpoint a discoverer could reach.
 * @param host - the host.
 * @returns true when `runtime.endpoint` is a usable base URL.
 */
function hasEndpoint(host: ModelHost): boolean {
  return typeof host.runtime.endpoint === 'string' && host.runtime.endpoint.trim().length > 0;
}

/**
 * The input and output artifact kinds implied by a set of capabilities.
 *
 * Derived from {@link CAPABILITY_IO} rather than invented, because that table is
 * also what `catalog/descriptor.ts` validates a descriptor's declared
 * `inputTypes`/`outputTypes` against. A discovered model that derived its kinds
 * any other way could be rejected by the catalog, or — worse — accepted and then
 * silently break a chained workflow (`text_to_image` producing something
 * `image_to_3d` will not accept).
 *
 * @param capabilities - the capabilities the model declares.
 * @returns the union of their canonical inputs and outputs, deduplicated in
 *   vocabulary order.
 */
export function ioForCapabilities(capabilities: readonly Capability[]): {
  readonly inputTypes: readonly IoType[];
  readonly outputTypes: readonly IoType[];
} {
  const inputs: IoType[] = [];
  const outputs: IoType[] = [];
  for (const capability of capabilities) {
    const io = CAPABILITY_IO[capability];
    for (const kind of io.input) if (!inputs.includes(kind)) inputs.push(kind);
    for (const kind of io.output) if (!outputs.includes(kind)) outputs.push(kind);
  }
  return { inputTypes: inputs, outputTypes: outputs };
}

/**
 * The static models a merged catalog already claims by id.
 *
 * A discovered descriptor whose id appears here is dropped rather than
 * appended, so the static entry's *fields* survive — not merely its position.
 *
 * @param config - the static configuration.
 * @returns the claimed ids.
 */
export function staticModelIds(config: ModelCatalogConfig): ReadonlySet<string> {
  return new Set(config.models.map((model) => model.id));
}

/**
 * Merge discovered descriptors into a catalog document, with static winning.
 *
 * The result is `[...static, ...discovered.filter(id not claimed)]`. Both halves
 * of that expression matter and neither is decoration:
 *
 * - **Order.** `ModelCatalog`'s constructor iterates `config.models` in order and
 *   silently skips (logging a diagnostic only) any later descriptor whose id it
 *   has already seen. There is no priority logic and no "static wins" logic
 *   anywhere in this codebase, so the merge — not the constructor — has to place
 *   static first.
 * - **Filter.** Order alone would leave a duplicate for the constructor to
 *   discard, which turns a deliberate precedence rule into a logged accident and
 *   pollutes `loadDiagnostics` with errors that are not errors. Filtering here
 *   means the constructor only ever sees unique ids.
 *
 * Collisions are resolved against the static ids *and* against ids already
 * accepted in this merge, so a discovery pass that reports the same model twice
 * (two loader nodes listing one file, say) yields one entry rather than two.
 *
 * Because the merge is always `static + this pass's discovered`, calling it
 * again with a fresh pass *replaces* the discovered half rather than growing it:
 * a checkpoint the operator deleted stops being routable on the next refresh.
 *
 * @param config - the static configuration, already validated.
 * @param discovered - descriptors produced by discovery.
 * @returns a new configuration object; the input is not mutated.
 */
export function mergeCatalogConfig(
  config: ModelCatalogConfig,
  discovered: readonly ModelDescriptor[],
): ModelCatalogConfig {
  const claimed = new Set<string>(staticModelIds(config));
  const models: ModelDescriptor[] = [...config.models];
  for (const descriptor of discovered) {
    if (claimed.has(descriptor.id)) continue;
    claimed.add(descriptor.id);
    models.push(descriptor);
  }
  return {
    ...config,
    ...(config.hosts === undefined ? {} : { hosts: config.hosts }),
    models,
  };
}
