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

import type { Context } from '@deepseek-ai/cordis';
import { dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Config as SchemaConfig, resolvePluginConfig } from './config.ts';
import type { PluginConfig, ResolvedPluginConfig } from './config.ts';
import { ModelHubService } from './service.ts';
import { registerInventoryRoute } from './inventory.ts';
import { registerPrioritiesRoute } from './priorities-route.ts';
import { defaultPolicyPath, loadPolicyDocument, savePolicyDocument } from './policy-store.ts';
import { registerSubscriptionIntegrations } from './subscription/index.ts';
import { registerModelHubSkill } from './skills.ts';
import { createArtifactRootResolver } from './workspace.ts';
import { registerDiscoveryTools, registerRoutingTool } from './tools/discovery.ts';
import { registerLifecycleTools } from './tools/lifecycle.ts';
import { registerInvokeTool } from './tools/invoke.ts';
import { ModelHub, ModelHubError, loadCatalogFromAnchors, toHubError } from '../src/index.ts';
import { DEFAULT_EXECUTION_POLICY } from '../src/index.ts';
import { CapabilityPolicy, CatalogBaselineDefaults, modelOptionFacts } from '../src/index.ts';
import type { ModelCatalogConfig, PolicyOptionFacts } from '../src/index.ts';
import type { ModelHost } from '../src/index.ts';

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

/** Re-exported type, so a TypeScript consumer can configure the plugin safely. */
export type Config = PluginConfig;

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
export function apply(ctx: Context, rawConfig: PluginConfig): void {
  const config: ResolvedPluginConfig = resolvePluginConfig(rawConfig);
  const log = ctx.logger('dsh-ai-model-hub');

  // Offer the hub's own instructions as a skill before anything can fail: the
  // skill is most valuable precisely when the catalog did not load, because it
  // tells the agent how to check and where to look. Its own failure is contained
  // to one warning and never affects the tools below.
  registerModelHubSkill(ctx, log);

  let hub: ModelHub | undefined;
  // The catalog's own facts, carried out of the try block for the settings page:
  // the hub deliberately does not republish them, because a hub built from a live
  // service is not the same object as the document it was built from.
  let catalogPath = '';
  let catalogHosts: readonly ModelHost[] = [];
  const policyPath = config.policyPath.length === 0 ? defaultPolicyPath() : config.policyPath;
  let policy: CapabilityPolicy | undefined;
  try {
    const loaded = loadCatalogFromAnchors({
      ...(config.configPath.length === 0 ? {} : { configPath: config.configPath }),
      anchors: catalogAnchors(config),
    });
    // The baseline is captured *here*, before any discovery pass can run, and it
    // is the whole point of the migration rule: the options this deployment
    // already had keep working exactly as they did, and anything that appears
    // afterwards — a pulled Ollama model, a workflow saved in ComfyUI, a
    // subscription backend — starts disabled until the operator enables it.
    policy = buildPolicy(loaded.config);
    hub = buildHub(loaded.config, config, log, loaded.path, loaded.warnings, policy);
    catalogPath = loaded.path;
    catalogHosts = loaded.config.hosts ?? [];
    log.info(
      `model hub ready: ${hub.catalog.listModels().length} model(s), ${hub.catalog.listCapabilities().length} capability(ies) from ${loaded.path}`,
    );
    for (const warning of loaded.warnings) log.warn(`catalog: ${warning}`);
    if (loaded.searched.length > 1) {
      log.debug(`catalog anchors tried, in order: ${loaded.searched.join(', ')}`);
    }
    log.info(
      `capability policy: ${policy.globallyEnabled ? 'enabled' : 'globally disabled'}, ` +
        `${Object.keys(policy.current.capabilities).length} capability entr(ies) at revision ${policy.currentRevision} from ${policyPath}`,
    );
  } catch (error) {
    const hubError = toHubError(error, 'CONFIG_ERROR');
    const message = `model hub disabled Ã¢â‚¬â€ ${hubError.code}: ${hubError.message}`;
    if (config.onConfigError === 'throw') throw new ModelHubError(hubError.code, message, hubError.details);
    log.warn(message);
    log.warn(
      'No model tools were registered. The Local models page is still served, and reports this error, so the ' +
        'problem is visible in the settings UI rather than as a missing route.',
    );
    // The settings page is registered on the failure path too. It is the surface
    // whose whole job is to explain the state of this machine, so leaving it
    // unregistered would turn "your catalog has a bad entry" into an unexplained
    // HTTP 404 — which is exactly the confusion this branch exists to remove.
    registerInventoryRoute(ctx, log, {
      catalogPath,
      artifactRoot: config.artifactRoot,
      allowProcessLaunch: config.allowProcessLaunch,
      catalogError: message,
    });
    return;
  }

  const service = new ModelHubService(ctx, hub);

  // The optional integration, before anything is published: a subscription
  // backend has to be a registered capability option before the first request is
  // planned, and registering it here means the settings page's very first read
  // already lists it (disabled).
  if (config.integrateSubscriptions) {
    try {
      registerSubscriptionIntegrations(ctx, hub, log);
    } catch (error) {
      const described = toHubError(error, 'CONFIG_ERROR');
      log.warn(`subscription integration unavailable — ${described.code}: ${described.message}`);
    }
  }
  // Seed and persist once the full option list is known, so a newly discovered
  // option or a newly registered backend is written into the document as
  // *disabled* rather than waiting to be noticed. A failure here is reported and
  // never blocks the hub: an unwritable policy file costs the settings page, not
  // the models.
  void seedPolicy(hub, policyPath, log);

  // Bind the hub's lifetime to the plugin fiber. cordis runs disposers in reverse
  // registration order when the fiber unloads, and awaits async ones Ã¢â‚¬â€ so a
  // plugin reload genuinely stops every model process before the new instance
  // starts, rather than racing it for the same port.
  ctx.effect(
    () => () => {
      void hub.dispose();
    },
    'dsh-ai-model-hub: dispose the model hub and every process it started',
  );

  // Where a call's artifacts go: the calling session's workspace by default, since
  // the hub is built before any session exists and the host's working directory
  // says nothing about the conversation's.
  const artifactRootFor = createArtifactRootResolver(ctx, config.artifactRoot, log);

  registerDiscoveryTools(ctx, service, { artifactRootFor });
  registerRoutingTool(ctx, service, { artifactRootFor });
  registerInvokeTool(ctx, service, {
    invocationTimeoutMs: config.invocationTimeoutMs,
    artifactRootFor,
    ...(config.attemptBudget === undefined ? {} : { attemptBudget: config.attemptBudget }),
  });
  registerLifecycleTools(ctx, service, { allowProcessLaunch: config.allowProcessLaunch });

  // The human-facing half of the same policy the tools enforce.
  if (policy !== undefined) {
    registerPrioritiesRoute(ctx, log, { hub, policy, policyPath });
  }

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
    ...(config.comfyuiWorkflowDir.length === 0 ? {} : { workflowDir: config.comfyuiWorkflowDir }),
    scanTimeoutMs: config.scanTimeoutMs,
  });

  // One diagnostic line per invocation, so an operator can see what ran without
  // the hub knowing about logging. This is the only event subscriber.
  ctx.effect(
    () =>
      hub.onEvent((event) => {
        if (event.type === 'invocation/fellback') {
          log.warn(`fell back from ${event.fromModelId} to ${event.toModelId}: ${event.reason}`);
        }
        if (event.type === 'model/started') {
          log.info(`started ${event.modelId} (pid ${event.pid ?? 'unknown'}) in ${event.coldStartMs} ms`);
        }
      }),
    'dsh-ai-model-hub: hub event diagnostics',
  );

  const unserved = hub.listUnservedCapabilities();
  if (unserved.length > 0) {
    log.info(
      `capabilities with no usable model: ${unserved.map((entry) => `${entry.capability} (${entry.reason})`).join('; ')}`,
    );
  }
}

/**
 * Build the capability policy from the saved document and this catalog.
 *
 * Synchronous, and deliberately so: the Loader's `apply` cannot await, and the
 * policy has to be in force before the first request can be planned. The saved
 * document is read from disk on a background pass — see {@link seedPolicy} —
 * which also writes back the entries a newly seen option needs.
 *
 * The baseline passed here is the set of option ids the *catalog* declares right
 * now, captured before discovery can add any. That is the migration rule stated
 * as data: an option that was already configured keeps the catalog's own
 * `enabled` flag as its default, and an option that appears later does not.
 *
 * @param catalogConfig - the catalog document that was just loaded.
 * @returns the policy to enforce.
 */
function buildPolicy(catalogConfig: ModelCatalogConfig): CapabilityPolicy {
  const baseline = new Set<string>();
  for (const model of catalogConfig.models) baseline.add(model.id);
  for (const host of catalogConfig.hosts ?? []) {
    for (const workflow of readHostWorkflowIds(host)) baseline.add(workflow);
  }
  return new CapabilityPolicy(undefined, { defaults: new CatalogBaselineDefaults(baseline) });
}

/**
 * Read a host's configured ComfyUI workflow ids, when it has any.
 *
 * A workflow declared on the host is as much a configured option as a `models[]`
 * entry — it is what a user actually runs — so it belongs in the baseline for the
 * same reason: it existed before the policy did.
 *
 * @param host - the host declaration.
 * @returns the workflow ids it declares.
 */
function readHostWorkflowIds(host: ModelHost): readonly string[] {
  const config = host.adapterConfig as { readonly workflows?: readonly { readonly id?: unknown }[] } | undefined;
  const workflows = config?.workflows;
  if (!Array.isArray(workflows)) return [];
  return workflows
    .map((workflow) => workflow?.id)
    .filter((id): id is string => typeof id === 'string' && id.length > 0);
}

/**
 * Load the saved policy, seed it with every live option, and persist if anything
 * changed.
 *
 * Runs after the options are registered, so a catalog refresh and a subscription
 * backend alike are written into the document as **disabled** the first time they
 * are seen. A failure at any step is reported and contained: the in-memory policy
 * the hub is already using stays in force, which is the safe half of the
 * operation.
 *
 * @param hub - the live hub, for the option list.
 * @param policyPath - where the document lives.
 * @param log - the plugin's logger.
 */
async function seedPolicy(hub: ModelHub, policyPath: string, log: { info: (m: string) => void; warn: (m: string) => void }): Promise<void> {
  try {
    const loaded = await loadPolicyDocument(policyPath, log as never);
    if (loaded.document !== undefined) hub.policy.replace(loaded.document);
    const facts: PolicyOptionFacts[] = [
      ...hub.catalog.listModels().map((model) => modelOptionFacts(model)),
      ...hub.listCapabilityOptions(),
    ];
    const seeded = hub.policy.seed(facts);
    if (!seeded.changed) return;
    hub.policy.replace(seeded.document);
    const issues = await savePolicyDocument(policyPath, seeded.document, log as never);
    if (issues.length === 0) {
      log.info(
        `capability policy: ${facts.length} option(s) seeded, new entries start disabled, saved to ${policyPath} ` +
          `at revision ${hub.policy.currentRevision}`,
      );
    }
  } catch (error) {
    log.warn(
      `the capability policy could not be loaded or seeded (${error instanceof Error ? error.message : String(error)}); ` +
        'the in-memory defaults are in force and nothing was activated',
    );
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
function catalogAnchors(config: ResolvedPluginConfig): string[] {
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
 * @param catalogPath - the file the catalog was read from, so an adapter can
 *   resolve a path the catalog wrote relative to itself rather than against the
 *   host process's working directory.
 * @returns the hub.
 */
function buildHub(
  catalogConfig: ModelCatalogConfig,
  config: ResolvedPluginConfig,
  log: { info: (message: string) => void; warn: (message: string) => void },
  catalogPath: string,
  catalogWarnings: readonly string[] = [],
  policy?: CapabilityPolicy,
): ModelHub {
  const effectiveConfig: ModelCatalogConfig = config.allowProcessLaunch
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
    // Where this catalog lives. A catalog's relative paths are relative to the
    // catalog, and a long-lived host's working directory says nothing about that.
    catalogPath,
    catalogWarnings,
    ...(config.artifactRoot.length === 0 ? {} : { artifactRoot: config.artifactRoot }),
    healthIntervalMs: config.healthIntervalMs,
    idleSweepIntervalMs: config.idleSweepIntervalMs,
    // The machine is measured, not assumed: routing compares a model's declared
    // needs against what this machine actually has free right now, and re-measures
    // after every generation. A deployment can turn it off, in which case resource
    // checks fall back to the declared profile and nothing is refused for
    // resources — see docs/adding-a-model.md.
    probeResources: config.probeResources,
    resourceTtlMs: config.resourceTtlMs,
    // Runtime discovery is opt-in. Off, no engine is contacted for
    // introspection and the catalog is exactly the document on disk; on, the
    // catalog grows in the background with what each engine reports it has.
    // Either way the static entries win an id collision — see
    // `src/discovery/types.ts`.
    discoverModels: config.discoverModels,
    discoveryTtlMs: config.discoveryTtlMs,
    discoveryTimeoutMs: config.discoveryTimeoutMs,
    // The operator's saved order and enablement. Passed in rather than built here
    // so the same object is shared with the settings route and the execution
    // boundary: one switch, one place it lives, every consumer sees the flip.
    ...(policy === undefined ? {} : { capabilityPolicy: policy }),
    // Absent means "walk the whole permitted list", which is the shipping
    // behaviour: the old silent cap of three attempts is deliberately gone.
    ...(config.attemptBudget === undefined ? {} : { attemptBudget: config.attemptBudget }),
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
 * drift from what is actually deployed — and, crucially, it is *not* a
 * hand-written list of capabilities in a prompt. The model learns what this
 * machine offers from the machine.
 *
 * It also reports what the catalog *dropped*. That matters more than it sounds:
 * a capability that is missing because an entry failed validation looks exactly
 * like a capability this machine cannot do, and the two need different answers.
 * When a `text_to_image` provider is silently unregistered, the honest outcome is
 * for the agent to say "the image workflow in your catalog was rejected, here is
 * why" instead of "image generation is unavailable here" — or, worse, for it to
 * quietly reach for a different tool the user never asked it to spend.
 *
 * Registered as runtime context rather than as a system-prompt section because it
 * is a fact about the environment that changes, not a standing instruction.
 *
 * @param ctx - the plugin context.
 * @param service - the hub service.
 */
function registerCapabilityContext(ctx: Context, service: ModelHubService): void {
  ctx.systemPrompt.context({
    name: 'dsh-ai-model-hub:capabilities',
    order: ctx.systemPrompt.getContextOrder('SANDBOX_POLICY') + 1,
    text: () => {
      const capabilities = service.hub.listCapabilities();
      const lines: string[] = [];
      if (capabilities.length === 0) {
        lines.push('Local AI models: none available in this deployment.');
      } else {
        lines.push('Local AI capabilities available on this machine, served through the invoke_model tool:');
        for (const entry of capabilities) {
          lines.push(
            `  - ${entry.capability}: accepts ${entry.inputTypes.join('/')}, produces ${entry.outputTypes.join('/')} (${entry.modelIds.length} model(s))`,
          );
        }
      }
      const unserved = service.hub.listUnservedCapabilities();
      if (unserved.length > 0) {
        lines.push(
          `Not available here: ${unserved.map((entry) => entry.capability).join(', ')}. Do not attempt these; say they are unavailable instead.`,
        );
      }
      const dropped = service.hub.catalog.loadDiagnostics.filter((issue) => issue.severity === 'error');
      if (dropped.length > 0) {
        lines.push(
          'Catalog entries that were read and then REJECTED, so they serve nothing (report these to the user rather ' +
            'than working around them):',
        );
        for (const issue of dropped.slice(0, 8)) lines.push(`  - ${issue.message}`);
        if (dropped.length > 8) lines.push(`  - …and ${String(dropped.length - 8)} more`);
      }
      lines.push(
        'These models run locally. Use invoke_model with a capability name — do not try to launch model software yourself.',
      );
      return lines.join('\n');
    },
  });
}
