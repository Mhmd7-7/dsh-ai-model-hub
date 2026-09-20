/**
 * The adapter contract.
 *
 * An adapter is the *only* place engine-specific knowledge lives. It knows how
 * to launch Stable Diffusion, or how to POST to a ComfyUI graph, or which flag
 * makes `llama-server` bind a port. Nothing above it does.
 *
 * The trade is explicit: to add a model whose engine is already supported, you
 * write a config entry. To add a model on a *new* kind of engine, you write one
 * adapter that implements this interface — and you still never touch the router,
 * the catalog, the runtime manager, or DSH.
 *
 * @module dsh-ai-model-hub/adapters/types
 */

import type { Artifact, ArtifactStore } from '../artifacts/types.ts';
import type { Capability } from '../catalog/capabilities.ts';
import type { AdapterKind, ResolvedModel } from '../catalog/descriptor.ts';
import type { HealthReport } from '../types.ts';

/** A diagnostic sink adapters write to. */
export interface AdapterLogger {
  /**
   * Record a diagnostic line.
   * @param message - what happened.
   * @param fields - structured context appended to the line.
   */
  debug(message: string, fields?: Readonly<Record<string, unknown>>): void;
  /**
   * Record a notable but non-fatal event.
   * @param message - what happened.
   * @param fields - structured context appended to the line.
   */
  info(message: string, fields?: Readonly<Record<string, unknown>>): void;
  /**
   * Record a failure the caller will also surface.
   * @param message - what happened.
   * @param fields - structured context appended to the line.
   */
  warn(message: string, fields?: Readonly<Record<string, unknown>>): void;
}

/** Everything an adapter needs to serve one request. */
export interface AdapterInvocation {
  /** The resolved model being invoked; carries runtime, limits, and adapter config. */
  readonly model: ResolvedModel;
  /** The capability the caller asked for. */
  readonly capability: Capability;
  /** The natural-language instruction, when the capability takes one. */
  readonly prompt?: string;
  /** Input artifacts, already resolved and type-checked. */
  readonly inputs: readonly Artifact[];
  /** Capability-specific settings from the caller. */
  readonly options: Readonly<Record<string, unknown>>;
  /** The store to write produced artifacts into. */
  readonly artifacts: ArtifactStore;
  /** Cancellation. Adapters must stop work and settle promptly when this fires. */
  readonly signal: AbortSignal;
  /** Diagnostics sink. */
  readonly log: AdapterLogger;
}

/** What an adapter returns on success. */
export interface AdapterOutput {
  /** Artifacts the adapter persisted via {@link AdapterInvocation.artifacts}. */
  readonly outputs: readonly Artifact[];
  /** Machine-readable structured outcome, merged into the invocation result. */
  readonly value?: Readonly<Record<string, unknown>>;
}

/**
 * One engine integration.
 *
 * Adapters are stateless with respect to a model: everything they need arrives
 * in {@link AdapterInvocation}. That is what lets one adapter instance serve many
 * models, and what makes an adapter trivially testable without a runtime.
 */
export interface ModelAdapter {
  /** Which adapter kind this implements; used to bind models to adapters. */
  readonly kind: AdapterKind;
  /** Human-facing name shown in diagnostics and `getModelStatus`. */
  readonly displayName: string;
  /**
   * Whether this adapter can serve the given model at all.
   *
   * Checked once at startup so a configuration mistake — a `cli` model with no
   * `modelPath`, an `http_json` model with no endpoint — becomes a clear
   * `unsupported` status instead of a runtime crash mid-workflow.
   *
   * @param model - the resolved model.
   * @returns `{ ok: true }`, or `ok: false` with the reason.
   */
  supports(model: ResolvedModel): { ok: true } | { ok: false; reason: string };

  /**
   * Probe the model's liveness.
   *
   * Runs on a background timer and before a cold start completes, so it must be
   * cheap and must never throw: an unreachable engine is a report, not an
   * exception.
   *
   * @param model - the resolved model.
   * @param signal - cancellation for the probe itself.
   * @returns the probe outcome.
   */
  health(model: ResolvedModel, signal: AbortSignal): Promise<HealthReport>;

  /**
   * Serve one request.
   *
   * @param invocation - the fully-resolved request.
   * @returns the produced artifacts and structured value.
   * @throws ModelHubError with a stable code; the hub wraps unknown throws.
   */
  invoke(invocation: AdapterInvocation): Promise<AdapterOutput>;
}

/**
 * A registry of adapters keyed by kind.
 *
 * The runtime manager resolves `model.adapter` through this registry. Because
 * lookup is by the model's own declared kind, an adapter is never chosen by
 * model id or name anywhere in the codebase.
 */
export class AdapterRegistry {
  private readonly adapters = new Map<AdapterKind, ModelAdapter>();

  /**
   * @param adapters - adapters to register immediately.
   */
  constructor(adapters: readonly ModelAdapter[] = []) {
    for (const adapter of adapters) this.register(adapter);
  }

  /**
   * Register an adapter, replacing any previous one of the same kind.
   *
   * Replacement is allowed deliberately: it is how a test substitutes a fake, and
   * how a plugin overrides a built-in with a real engine. The returned disposer
   * restores the previous binding, which keeps that override scoped.
   *
   * @param adapter - the adapter to register.
   * @returns a disposer that restores whatever was registered before.
   */
  register(adapter: ModelAdapter): () => void {
    const previous = this.adapters.get(adapter.kind);
    this.adapters.set(adapter.kind, adapter);
    return () => {
      if (previous === undefined) this.adapters.delete(adapter.kind);
      else this.adapters.set(adapter.kind, previous);
    };
  }

  /**
   * Find the adapter for a kind.
   * @param kind - the adapter kind.
   * @returns the adapter, or `undefined` when none is registered.
   */
  get(kind: AdapterKind): ModelAdapter | undefined {
    return this.adapters.get(kind);
  }

  /**
   * Find the adapter for a kind, failing loudly when absent.
   * @param kind - the adapter kind.
   * @returns the adapter.
   * @throws ModelHubError with `UNSUPPORTED_OPERATION`.
   */
  require(kind: AdapterKind): ModelAdapter {
    const adapter = this.adapters.get(kind);
    if (adapter === undefined) {
      throw new Error(`no adapter registered for kind "${kind}"`);
    }
    return adapter;
  }

  /** Every registered adapter kind, in registration order. */
  listKinds(): readonly AdapterKind[] {
    return [...this.adapters.keys()];
  }
}

/**
 * A logger that discards everything, used as the default.
 * @returns a silent logger.
 */
export function silentLogger(): AdapterLogger {
  return {
    debug: (): void => {},
    info: (): void => {},
    warn: (): void => {},
  };
}

/**
 * Build a logger that forwards to a single line-oriented sink.
 * @param sink - receives fully-formatted lines.
 * @param prefix - prepended to every line.
 * @returns the logger.
 */
export function lineLogger(sink: (line: string) => void, prefix: string): AdapterLogger {
  const emit =
    (level: string) =>
    (message: string, fields?: Readonly<Record<string, unknown>>): void => {
      const suffix =
        fields === undefined || Object.keys(fields).length === 0
          ? ''
          : ` ${JSON.stringify(fields)}`;
      sink(`${prefix} ${level} ${message}${suffix}`);
    };
  return { debug: emit('debug'), info: emit('info'), warn: emit('warn') };
}
