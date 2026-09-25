/**
 * DSH plugin configuration.
 *
 * Kept in its own module so the plugin entry can be read as pure wiring. Every
 * field here is a deployment decision, and every one of them has a default that
 * makes the zero-config case work: `dsh plugin add` followed by a restart is
 * enough to get the catalog the plugin ships behind the agent.
 *
 * @module dsh-ai-model-hub/dsh-plugin/config
 */

import z from '@deepseek-ai/schemastery';

/**
 * Plugin configuration as written under the plugin's row in `cordis.patch.yml`.
 *
 * Every field is optional in the file; the Loader's schema fills defaults before
 * `apply` runs, and {@link resolvePluginConfig} does the same for direct callers.
 */
export interface PluginConfig {
  /**
   * Explicit path to the catalog file.
   *
   * When set it is the ONLY thing tried: an operator who names a file should be
   * told that file is wrong, not quietly given a different catalog instead.
   *
   * Omitted (the default) means discovery: `models.json`,
   * `model-catalog.json`, or `dsh-ai-model-hub.json`, searched upward from each of
   * {@link searchRoots}, then the process working directory, then this plugin's
   * own installation directory — including a `config/` subdirectory at each
   * level. See {@link searchRoots} for why the plugin's own directory is in that
   * list.
   */
  configPath?: string;
  /**
   * Extra directories to start catalog discovery from, in priority order.
   *
   * Discovery's built-in anchors are the process working directory and then this
   * plugin's own installation directory. The first of those is a poor anchor for
   * a host that runs as a long-lived server: it is fixed when the host is
   * launched and has nothing to do with where a catalog lives, so a hub anchored
   * only there finds nothing and the agent silently gets no model tools. The
   * second makes a hub installed from a checkout find its own shipped catalog.
   *
   * Name a root here to point the hub somewhere else — typically the workspace
   * the deployment actually serves — without pinning an exact file.
   */
  searchRoots?: string[];
  /**
   * Where produced artifacts are stored.
   *
   * **Unset is the normal case, and the right one.** Every tool call then writes
   * under the *calling session's* workspace — `<workspace>/artifacts`, resolved
   * per call from DSH's sandbox policy — so each conversation's images land beside
   * that conversation's code, and switching workspace in the GUI switches where
   * output goes without any configuration.
   *
   * Setting it pins one absolute directory for every session, which is what a
   * deployment that collects artifacts centrally wants. There is no relative
   * form: the hub refuses a non-absolute root rather than guessing at a base.
   */
  artifactRoot?: string;
  /**
   * Whether the plugin owns the hub's lifetime.
   *
   * True (the default) means the plugin constructs the hub on load and disposes
   * it — stopping every process it started — when the plugin unloads. Set false
   * only when another plugin in the same profile already provides the hub, in
   * which case this plugin attaches to it instead of building a second one.
   */
  manageHub?: boolean;
  /**
   * Whether live/one-shot models may be started by the hub.
   *
   * False (the default) makes every descriptor's `startable` flag collapse to
   * false, so a plugin cannot launch a heavyweight engine unless the operator
   * opted in here. This is the deployment-level kill switch: it makes "the agent
   * silently started an 8 GB model" impossible even if the catalog permits it.
   */
  allowProcessLaunch?: boolean;
  /**
   * Permit launch commands outside the built-in engine allowlist.
   *
   * False by default. Enabling it is an explicit, auditable decision to let the
   * catalog name arbitrary executables; see `docs/security.md` for what that
   * means.
   */
  allowAnyCommand?: boolean;
  /**
   * Register the capability catalog as a model-visible runtime context.
   *
   * True (the default) gives the agent a dynamic snapshot of which capabilities
   * this deployment can serve. The text is generated from the catalog, never
   * hard-coded, so it cannot drift from what is actually available.
   */
  exposeCapabilityContext?: boolean;
  /**
   * What to do when the catalog is missing or invalid.
   *
   * `warn` (the default) logs the problem, registers no tools, and lets DSH boot
   * normally — a broken model catalog must never make the agent unusable.
   * `throw` fails the plugin load, which an operator wants while they are
   * actively editing the catalog and want to be told immediately.
   */
  onConfigError?: 'warn' | 'throw';
  /** How often the hub re-probes model health, in milliseconds. `0` disables it. */
  healthIntervalMs?: number;
  /** How often the hub stops models that exceeded their idle timeout. `0` disables it. */
  idleSweepIntervalMs?: number;
  /**
   * Default timeout for a single `invoke_model` call, in milliseconds.
   *
   * The tool declares this as its cooperative budget as well, so a hung engine
   * cancels the tool call instead of stalling the turn. Omitted means no default
   * deadline beyond the model's own.
   */
  invocationTimeoutMs?: number;
  /**
   * Whether the hub probes this machine's resources, and how.
   *
   * True (the default) means routing decisions are made against measured RAM,
   * VRAM, and free space rather than against declared totals — which is what lets
   * the router refuse a model that will not fit *now* instead of starting it and
   * letting the engine die with an out-of-memory error.
   *
   * Set it to `false` for a deployment where spawning `nvidia-smi` is undesirable;
   * resource checks then fall back to whatever the catalog declares, and no engine
   * is ever refused for resource reasons.
   */
  probeResources?: boolean;
  /**
   * How long a machine measurement stays fresh enough to route against.
   *
   * Defaults to 30000. The probe runs again when the last measurement is older
   * than this, and always after an invocation. Raising it reduces subprocess
   * churn on a busy deployment; lowering it makes another application's memory use
   * visible sooner.
   */
  resourceTtlMs?: number;
  /**
   * Whether to augment the catalog with models discovered from the configured
   * engines at runtime.
   *
   * False (the default) means the catalog is exactly the document on disk and no
   * engine is ever contacted for introspection. True means the hub asks every
   * configured host what it currently has — a pulled Ollama model, a checkpoint
   * dropped into ComfyUI's models directory — and republishes the catalog with
   * those added, so no JSON edit is needed before the agent can use them.
   *
   * Static configuration always wins: a discovered model whose id a
   * `models.json` entry already claims is dropped before the catalog is built,
   * and every static entry precedes every discovered one. See
   * `src/discovery/types.ts`.
   *
   * Discovery is fail-soft by design: an engine that is down is a warning and no
   * models from that host, never a failed boot.
   */
  discoverModels?: boolean;
  /**
   * How long a discovery pass stays cached, in milliseconds. Defaults to 60000.
   *
   * list_models and invoke_model calls in quick succession must not re-hit every
   * engine. The `refresh_model_discovery` tool bypasses this, which is the path
   * for "I just pulled a new model and do not want to wait".
   */
  discoveryTtlMs?: number;
  /**
   * Budget for one engine's discovery pass, in milliseconds. Defaults to 5000.
   *
   * ComfyUI answers `/object_info` slowly on a cold start, so a deployment whose
   * discovery times out should raise this rather than disable discovery.
   */
  discoveryTimeoutMs?: number;
}

/** Schemastery schema for {@link PluginConfig}. */
export const Config = z.object({
  configPath: z.string().default(''),
  searchRoots: z.array(z.string()).default([]),
  artifactRoot: z.string().default(''),
  manageHub: z.boolean().default(true),
  allowProcessLaunch: z.boolean().default(false),
  allowAnyCommand: z.boolean().default(false),
  exposeCapabilityContext: z.boolean().default(true),
  onConfigError: z.union([z.const('warn'), z.const('throw')]).default('warn'),
  healthIntervalMs: z.number().step(1).min(0).default(30_000),
  idleSweepIntervalMs: z.number().step(1).min(0).default(15_000),
  invocationTimeoutMs: z.number().step(1).min(0).default(600_000),
  probeResources: z.boolean().default(true),
  resourceTtlMs: z.number().step(1).min(0).default(30_000),
  discoverModels: z.boolean().default(false),
  discoveryTtlMs: z.number().step(1).min(0).default(60_000),
  discoveryTimeoutMs: z.number().step(1).min(1).default(5_000),
});

/** Apply defaults for direct callers that bypass Loader validation. */
export function resolvePluginConfig(config: Partial<PluginConfig> | undefined): ResolvedPluginConfig {
  return {
    configPath: config?.configPath ?? '',
    searchRoots: config?.searchRoots ?? [],
    artifactRoot: config?.artifactRoot ?? '',
    manageHub: config?.manageHub ?? true,
    allowProcessLaunch: config?.allowProcessLaunch ?? false,
    allowAnyCommand: config?.allowAnyCommand ?? false,
    exposeCapabilityContext: config?.exposeCapabilityContext ?? true,
    onConfigError: config?.onConfigError ?? 'warn',
    healthIntervalMs: config?.healthIntervalMs ?? 30_000,
    idleSweepIntervalMs: config?.idleSweepIntervalMs ?? 15_000,
    invocationTimeoutMs: config?.invocationTimeoutMs ?? 600_000,
    probeResources: config?.probeResources ?? true,
    resourceTtlMs: config?.resourceTtlMs ?? 30_000,
    discoverModels: config?.discoverModels ?? false,
    discoveryTtlMs: config?.discoveryTtlMs ?? 60_000,
    discoveryTimeoutMs: config?.discoveryTimeoutMs ?? 5_000,
  };
}

/**
 * Configuration with every default applied.
 *
 * `apply` receives this shape: the Loader validates against {@link Config}, whose
 * schemastery defaults fill every field, so no field is `undefined` by the time
 * the plugin body runs.
 */
export type ResolvedPluginConfig = Required<PluginConfig>;
