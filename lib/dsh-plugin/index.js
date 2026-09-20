/**
 * The DeepSeek Harness plugin entry point.
 *
 * This is the only DSH-aware module in the project, and it is deliberately thin:
 * it loads configuration, constructs the hub, publishes it as a service,
 * registers **generic capability tools**, and binds the hub's lifetime to the
 * plugin fiber. It contains no model knowledge whatsoever Ã¢â‚¬â€ no model id, no
 * engine name, no launch command, no capability-specific branch.
 *
 * The whole DSH coupling is:
 *
 * | DSH API                          | Used for                                    |
 * | -------------------------------- | ------------------------------------------- |
 * | `name` / `inject` / `apply` / `Config` | plugin registration (Loader contract)  |
 * | `ctx.tools.register`             | exposing capability tools to the model      |
 * | `ctx.systemPrompt.context`       | a dynamic snapshot of available capabilities |
 * | `ctx.inject` / `ctx.skills`      | offering the bundled skill                  |
 * | `ctx.logger`                     | diagnostics                                 |
 * | `ctx.effect`                     | disposing the hub with the plugin           |
 *
 * That list is the entire compatibility surface. If DSH changes one of these,
 * this file changes and nothing else does Ã¢â‚¬â€ the catalog, router, runtime,
 * adapters, and artifact store import nothing from DSH and are tested without it.
 *
 * The plugin is a **function plugin with named exports and no default export**:
 * a stray `export default` would make the Loader's `unwrapExports` collapse the
 * module and drop `inject`, which is a documented DSH failure mode.
 *
 * @module dsh-ai-model-hub/dsh-plugin
 */
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Config as SchemaConfig, resolvePluginConfig } from "./config.js";
import { ModelHubService } from "./service.js";
import { registerInventoryRoute } from "./inventory.js";
import { registerModelHubSkill } from "./skills.js";
import { createArtifactRootResolver } from "./workspace.js";
import { registerDiscoveryTools, registerRoutingTool } from "./tools/discovery.js";
import { registerLifecycleTools } from "./tools/lifecycle.js";
import { registerInvokeTool } from "./tools/invoke.js";
import { ModelHub, ModelHubError, loadCatalogFromAnchors, toHubError } from "../src/index.js";
import { DEFAULT_EXECUTION_POLICY } from "../src/index.js";
/**
 * This plugin's own installation directory.
 *
 * Used as the last catalog-discovery anchor, and it is the anchor that makes the
 * zero-config case actually work. DSH installs profile plugins as junction links
 * and Node resolves them to their real path, so this is the hub checkout the
 * plugin was installed from — the one that ships `config/models.json`. Without
 * it, the only anchor is the host process's working directory, which for a
 * long-running `dsh web` has nothing to do with where the catalog lives: the
 * plugin then finds no catalog, logs a warning, registers no tools, and the
 * agent silently has no model capability at all.
 */
const PLUGIN_DIRECTORY = dirname(fileURLToPath(import.meta.url));
/** Stable Loader identity. */
export const name = 'dsh-ai-model-hub';
/**
 * Services this plugin requires.
 *
 * Both are required, not optional.
 *
 * `tools` is obvious: without the tool registry there is nothing for this plugin
 * to do, and failing to load is more honest than loading a hub the agent cannot
 * reach.
 *
 * `systemPrompt` is required because the capability snapshot is registered as
 * runtime context. cordis enforces this: reading a service that is not declared
 * in `inject` throws `cannot get property "systemPrompt" without inject` and
 * aborts the whole profile boot. That is exactly what happened the first time
 * this plugin was loaded for real â€” the declared list is a contract, not a hint.
 *
 * The catalog is read from disk by the hub itself rather than through a DSH
 * service, so no filesystem service is injected; that keeps the hub usable in a
 * headless or SDK deployment where `ctx.fs` may be absent.
 */
export const inject = ['tools', 'systemPrompt'];
/** Re-exported so the Loader validates plugin config against this schema. */
export const Config = SchemaConfig;
/**
 * Register the model hub and its tools.
 *
 * A missing or invalid catalog is contained: by default the plugin logs the
 * problem, registers nothing, and lets DSH boot. An unusable model catalog must
 * never make the agent unusable Ã¢â‚¬â€ the agent can still do everything that does not
 * need a local model.
 *
 * @param ctx - the plugin's context.
 * @param rawConfig - deployment configuration, already validated by the Loader.
 */
export function apply(ctx, rawConfig) {
    const config = resolvePluginConfig(rawConfig);
    const log = ctx.logger('dsh-ai-model-hub');
    // Offer the hub's own instructions as a skill before anything can fail: the
    // skill is most valuable precisely when the catalog did not load, because it
    // tells the agent how to check and where to look. Its own failure is contained
    // to one warning and never affects the tools below.
    registerModelHubSkill(ctx, log);
    let hub;
    // The catalog's own facts, carried out of the try block for the settings page:
    // the hub deliberately does not republish them, because a hub built from a live
    // service is not the same object as the document it was built from.
    let catalogPath = '';
    let catalogHosts = [];
    try {
        const loaded = loadCatalogFromAnchors({
            ...(config.configPath.length === 0 ? {} : { configPath: config.configPath }),
            anchors: catalogAnchors(config),
        });
        hub = buildHub(loaded.config, config, log);
        catalogPath = loaded.path;
        catalogHosts = loaded.config.hosts ?? [];
        log.info(`model hub ready: ${hub.catalog.listModels().length} model(s), ${hub.catalog.listCapabilities().length} capability(ies) from ${loaded.path}`);
        if (loaded.searched.length > 1) {
            log.debug(`catalog anchors tried, in order: ${loaded.searched.join(', ')}`);
        }
    }
    catch (error) {
        const hubError = toHubError(error, 'CONFIG_ERROR');
        const message = `model hub disabled Ã¢â‚¬â€ ${hubError.code}: ${hubError.message}`;
        if (config.onConfigError === 'throw')
            throw new ModelHubError(hubError.code, message, hubError.details);
        log.warn(message);
        log.warn('No model tools were registered. Fix the catalog and restart, or reload the plugin after editing it.');
        return;
    }
    const service = new ModelHubService(ctx, hub);
    // Bind the hub's lifetime to the plugin fiber. cordis runs disposers in reverse
    // registration order when the fiber unloads, and awaits async ones Ã¢â‚¬â€ so a
    // plugin reload genuinely stops every model process before the new instance
    // starts, rather than racing it for the same port.
    ctx.effect(() => () => {
        void hub.dispose();
    }, 'dsh-ai-model-hub: dispose the model hub and every process it started');
    // Where a call's artifacts go: the calling session's workspace by default, since
    // the hub is built before any session exists and the host's working directory
    // says nothing about the conversation's.
    const artifactRootFor = createArtifactRootResolver(ctx, config.artifactRoot, log);
    registerDiscoveryTools(ctx, service, { artifactRootFor });
    registerRoutingTool(ctx, service, { artifactRootFor });
    registerInvokeTool(ctx, service, {
        invocationTimeoutMs: config.invocationTimeoutMs,
        artifactRootFor,
    });
    registerLifecycleTools(ctx, service, { allowProcessLaunch: config.allowProcessLaunch });
    if (config.exposeCapabilityContext) {
        registerCapabilityContext(ctx, service);
    }
    // The same facts, for the human: the "Local models" settings page reads a live
    // inventory of the engines on this machine. Injected on demand, so a profile with
    // no web server (headless, sdk-minimal) keeps every tool and serves no page.
    registerInventoryRoute(ctx, log, {
        hub,
        hosts: catalogHosts,
        catalogPath,
        artifactRoot: config.artifactRoot,
        allowProcessLaunch: config.allowProcessLaunch,
    });
    // One diagnostic line per invocation, so an operator can see what ran without
    // the hub knowing about logging. This is the only event subscriber.
    ctx.effect(() => hub.onEvent((event) => {
        if (event.type === 'invocation/fellback') {
            log.warn(`fell back from ${event.fromModelId} to ${event.toModelId}: ${event.reason}`);
        }
        if (event.type === 'model/started') {
            log.info(`started ${event.modelId} (pid ${event.pid ?? 'unknown'}) in ${event.coldStartMs} ms`);
        }
    }), 'dsh-ai-model-hub: hub event diagnostics');
    const unserved = hub.listUnservedCapabilities();
    if (unserved.length > 0) {
        log.info(`capabilities with no usable model: ${unserved.map((entry) => `${entry.capability} (${entry.reason})`).join('; ')}`);
    }
}
/**
 * The directories catalog discovery starts from, in priority order.
 *
 * Operator-specified roots win, so a deployment that names where its catalog
 * lives is never second-guessed. The working directory comes next, preserving
 * the useful case of a host started inside a project that carries its own
 * catalog. The plugin's own directory comes last, which is what makes a hub
 * installed from a checkout find the catalog that checkout ships.
 *
 * @param config - resolved plugin configuration.
 * @returns the ordered anchors.
 */
function catalogAnchors(config) {
    return [...config.searchRoots, process.cwd(), PLUGIN_DIRECTORY];
}
/**
 * Construct the hub from a validated catalog.
 *
 * The `allowProcessLaunch: false` default is applied here, where it belongs: the
 * descriptor files say what a model *could* do, and the deployment says what this
 * plugin is *permitted* to do. Collapsing every `startable`/`stoppable` to false
 * is how a cautious deployment keeps the model-management tools registered and
 * honest while making process launches impossible.
 *
 * @param catalogConfig - the validated catalog document.
 * @param config - resolved plugin configuration.
 * @param log - the plugin's logger.
 * @returns the hub.
 */
function buildHub(catalogConfig, config, log) {
    const effectiveConfig = config.allowProcessLaunch
        ? catalogConfig
        : {
            ...catalogConfig,
            models: catalogConfig.models.map((model) => ({
                ...model,
                lifecycle: { ...(model.lifecycle ?? { startable: false, stoppable: false }), startable: false, stoppable: false },
            })),
            ...(catalogConfig.hosts === undefined
                ? {}
                : {
                    hosts: catalogConfig.hosts.map((host) => ({
                        ...host,
                        lifecycle: { ...(host.lifecycle ?? { startable: false, stoppable: false }), startable: false, stoppable: false },
                    })),
                }),
        };
    return new ModelHub({
        config: effectiveConfig,
        ...(config.artifactRoot.length === 0 ? {} : { artifactRoot: config.artifactRoot }),
        healthIntervalMs: config.healthIntervalMs,
        idleSweepIntervalMs: config.idleSweepIntervalMs,
        executionPolicy: {
            // Spreading the shipped policy and widening only `allowAnyCommand` keeps
            // the credential scrub list and argument bounds in force.
            ...DEFAULT_EXECUTION_POLICY,
            allowAnyCommand: config.allowAnyCommand,
        },
        log: (message) => log.info(message),
    });
}
/**
 * Register the capability catalog as model-visible runtime context.
 *
 * The text is generated from the live catalog on every assembly, so it cannot
 * drift from what is actually deployed Ã¢â‚¬â€ and, crucially, it is *not* a
 * hand-written list of capabilities in a prompt. The model learns what this
 * machine offers from the machine.
 *
 * Registered as runtime context rather than as a system-prompt section because it
 * is a fact about the environment that changes, not a standing instruction.
 *
 * @param ctx - the plugin context.
 * @param service - the hub service.
 */
function registerCapabilityContext(ctx, service) {
    ctx.systemPrompt.context({
        name: 'dsh-ai-model-hub:capabilities',
        order: ctx.systemPrompt.getContextOrder('SANDBOX_POLICY') + 1,
        text: () => {
            const capabilities = service.hub.listCapabilities();
            if (capabilities.length === 0) {
                return 'Local AI models: none available in this deployment.';
            }
            const lines = [
                'Local AI capabilities available on this machine, served through the invoke_model tool:',
            ];
            for (const entry of capabilities) {
                lines.push(`  - ${entry.capability}: accepts ${entry.inputTypes.join('/')}, produces ${entry.outputTypes.join('/')} (${entry.modelIds.length} model(s))`);
            }
            const unserved = service.hub.listUnservedCapabilities();
            if (unserved.length > 0) {
                lines.push(`Not available here: ${unserved.map((entry) => entry.capability).join(', ')}. Do not attempt these; say they are unavailable instead.`);
            }
            lines.push('These models run locally. Use invoke_model with a capability name Ã¢â‚¬â€ do not try to launch model software yourself.');
            return lines.join('\n');
        },
    });
}
