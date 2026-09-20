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
export function resolvePluginConfig(config) {
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
