/**
 * DSH plugin configuration.
 *
 * Kept in its own module so the plugin entry can be read as pure wiring. Every
 * field here is a deployment decision, and every one of them has a default that
 * makes the zero-config case work: `dsh plugin add` followed by a restart is
 * enough to get the mock models behind the agent.
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
   * Defaults to `<working directory>/artifacts`, so a conversation's generated
   * images land beside its code rather than in a shared global directory.
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
