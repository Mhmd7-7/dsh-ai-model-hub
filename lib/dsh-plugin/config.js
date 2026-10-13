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
    probeResources: z.boolean().default(true),
    resourceTtlMs: z.number().step(1).min(0).default(30_000),
    discoverModels: z.boolean().default(false),
    discoveryTtlMs: z.number().step(1).min(0).default(60_000),
    discoveryTimeoutMs: z.number().step(1).min(1).default(5_000),
    comfyuiWorkflowDir: z.string().default(''),
    scanTimeoutMs: z.number().step(1).min(1).default(4_000),
    // 0 is "no override": fall back to the host's declared startup budget.
    engineStartupTimeoutMs: z.number().step(1).min(0).default(0),
    workflowCatalogPath: z.string().default(''),
    policyPath: z.string().default(''),
    // 0 is the unlimited sentinel; the default must itself pass Loader validation.
    attemptBudget: z.number().step(1).min(0).default(0),
    integrateSubscriptions: z.boolean().default(true),
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
        probeResources: config?.probeResources ?? true,
        resourceTtlMs: config?.resourceTtlMs ?? 30_000,
        discoverModels: config?.discoverModels ?? false,
        discoveryTtlMs: config?.discoveryTtlMs ?? 60_000,
        discoveryTimeoutMs: config?.discoveryTimeoutMs ?? 5_000,
        comfyuiWorkflowDir: config?.comfyuiWorkflowDir ?? '',
        scanTimeoutMs: config?.scanTimeoutMs ?? 4_000,
        // Normalized away like `attemptBudget`: absent means "no override", so the
        // catalog's declared startup budget is what the scan honors.
        ...(config?.engineStartupTimeoutMs === undefined || config.engineStartupTimeoutMs <= 0
            ? {} : { engineStartupTimeoutMs: config.engineStartupTimeoutMs }),
        workflowCatalogPath: config?.workflowCatalogPath ?? '',
        policyPath: config?.policyPath ?? '',
        // `0` is the schema's "not configured" spelling for an absent budget, since
        // schemastery's numeric default cannot be `undefined`. It is normalized away
        // here so the hub sees the honest shape: absent means "no budget".
        ...(config?.attemptBudget === undefined || config.attemptBudget <= 0 ? {} : { attemptBudget: config.attemptBudget }),
        integrateSubscriptions: config?.integrateSubscriptions ?? true,
    };
}
