/**
 * The Capability priorities route.
 *
 * The settings page owns no data of its own: it reads `GET` here and writes
 * `PUT`/`POST` here, and this module is the only thing that touches the saved
 * policy. That division is what keeps the page honest — the numbers it shows are
 * the numbers the hub would actually use, computed by the hub's own planner,
 * rather than a second opinion assembled in the browser.
 *
 * The route is served from the host half and reaches the browser through the same
 * `webServer` registration the inventory route uses, so a profile with no web
 * server (headless, sdk-minimal) keeps every tool and serves no page.
 *
 * @module dsh-ai-model-hub/dsh-plugin/priorities-route
 */

import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Context } from '@deepseek-ai/cordis';
import type { Capability, CapabilityPolicy, PolicyEntry } from '../src/index.ts';
import { CAPABILITIES, isCapability } from '../src/index.ts';
import type { ModelHub } from '../src/index.ts';
import { savePolicyDocument } from './policy-store.ts';
import type { PluginLogger } from './types.ts';

/** The route the settings page reads and writes. Absolute, no trailing slash. */
export const PRIORITIES_ROUTE = '/dsh-ai-model-hub/priorities';

/** The subset of `webServer` this route needs. */
interface WebServerLike {
  register(route: {
    kind: 'exact';
    path: string;
    handler: (req: IncomingMessage, res: ServerResponse) => void | Promise<void>;
  }): () => void;
}

/** What the route needs to serve and persist the policy. */
export interface PrioritiesRouteOptions {
  /** The hub, for its live plan per capability. */
  readonly hub: ModelHub;
  /** The live policy being edited. */
  readonly policy: CapabilityPolicy;
  /** Where the document is persisted. */
  readonly policyPath: string;
  /** Called after a successful save, so the host can log or re-seed. */
  readonly onSaved?: (revision: number) => void;
}

/** One option as the page renders it. */
interface OptionPayload {
  /** Stable id, the only identifier the page may send back. */
  readonly id: string;
  /** Display name. */
  readonly name: string;
  /** Model/workflow/backend type. */
  readonly type: string;
  /** `model` or `subscription`. */
  readonly source: string;
  /** Engine or provider. */
  readonly provider: string;
  /** Human-facing provider description. */
  readonly providerLabel: string;
  /** Position in the effective order, 1-based. */
  readonly order: number;
  /** Whether the operator permits it. */
  readonly enabled: boolean;
  /** Which switch decided that. */
  readonly enabledBy: string;
  /** Live usability. */
  readonly state: string;
  /** Why it is in that state. */
  readonly reason: string;
  /** Whether the hub could still start it. */
  readonly startable: boolean;
  /** Declared capabilities. */
  readonly capabilities: readonly string[];
  /** Declared input kinds, so task variants stay distinguishable. */
  readonly inputTypes: readonly string[];
  /** Declared output kinds. */
  readonly outputTypes: readonly string[];
  /** Whether it can execute at all. */
  readonly executable: boolean;
  /** Whether it is absent from the current catalog but kept for its saved settings. */
  readonly missing?: boolean;
}

/**
 * Register the priorities route.
 *
 * @param ctx - the plugin context.
 * @param log - the plugin's logger.
 * @param options - the hub, policy, and persistence path.
 */
export function registerPrioritiesRoute(
  ctx: Context,
  log: PluginLogger,
  options: PrioritiesRouteOptions,
): void {
  ctx.inject(['webServer'], (scope) => {
    const webServer = scope.get('webServer') as unknown as WebServerLike | undefined;
    if (webServer === undefined || typeof webServer.register !== 'function') {
      log.debug('no web server is mounted, so the Capability priorities route is not served');
      return;
    }

    const handler = async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
      try {
        if (req.method === 'GET' || req.method === 'HEAD') {
          sendJson(res, 200, describe(options));
          return;
        }
        if (req.method === 'PUT' || req.method === 'POST') {
          const body = await readJsonBody(req);
          await applyEdit(options, body, res, log);
          return;
        }
        res.writeHead(405, { 'content-type': 'application/json; charset=utf-8', allow: 'GET, PUT, POST' });
        res.end(JSON.stringify({ error: 'method not allowed' }));
      } catch (error) {
        log.warn(`the Capability priorities route failed: ${error instanceof Error ? error.message : String(error)}`);
        sendJson(res, 500, { error: error instanceof Error ? error.message : String(error) });
      }
    };

    webServer.register({ kind: 'exact', path: PRIORITIES_ROUTE, handler });
    log.debug(`serving Capability priorities on ${PRIORITIES_ROUTE}`);
  });
}

/**
 * Render the whole policy as the page needs it.
 *
 * @param options - the hub, policy, and path.
 * @returns the payload.
 */
function describe(options: PrioritiesRouteOptions): Record<string, unknown> {
  const { hub, policy, policyPath } = options;
  const plans = hub.describeCapabilityOptions({ includeEmpty: true });
  const byCapability = new Map(plans.map((entry) => [entry.capability, entry.plan]));

  const capabilities = CAPABILITIES.map((capability) => {
    const plan = byCapability.get(capability);
    const options_: OptionPayload[] = (plan?.options ?? []).map((option) => ({
      id: option.option.id,
      name: option.option.name,
      type: option.option.type,
      source: option.option.source,
      provider: option.option.provider,
      providerLabel: option.option.providerLabel,
      order: option.order,
      enabled: option.enabled,
      enabledBy: option.enabledBy,
      state: option.state,
      reason: option.reason,
      startable: option.startable,
      capabilities: option.option.capabilities,
      inputTypes: option.option.inputTypes,
      outputTypes: option.option.outputTypes,
      executable: option.option.executable,
    }));
    return {
      capability,
      inputTypes: plan?.options[0]?.option.inputTypes ?? [],
      outputTypes: plan?.options[0]?.option.outputTypes ?? [],
      options: options_,
      permitted: plan?.permitted.map((option) => option.option.id) ?? [],
      skipped: plan?.skipped ?? [],
    };
  }).filter((entry) => entry.options.length > 0 || entry.capability !== undefined);

  return {
    policyPath,
    revision: policy.currentRevision,
    globalEnabled: policy.globallyEnabled,
    savedAt: new Date().toISOString(),
    capabilities,
    documentation:
      'Enabled permits an option to serve this capability. Disabled forbids it: it is never used, not even as a ' +
      'fallback, and never started. Lower priority means it runs only after every higher-ranked permitted option ' +
      'has definitively failed. Stopped means its engine is not running; it may still be enabled and startable.',
  };
}

/**
 * Apply one edit and persist it.
 *
 * The whole document is rebuilt from the submitted order so a drag is saved
 * exactly as it was arranged, and every id is validated against what actually
 * exists — plus whatever the policy already remembers, so editing a list while an
 * engine is down cannot delete its settings.
 *
 * @param options - the hub, policy, and path.
 * @param body - the request body.
 * @param res - the response.
 * @param log - the plugin's logger.
 */
async function applyEdit(
  options: PrioritiesRouteOptions,
  body: unknown,
  res: ServerResponse,
  log: PluginLogger,
): Promise<void> {
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    sendJson(res, 400, { error: 'the request body must be an object' });
    return;
  }
  const record = body as Record<string, unknown>;

  if (typeof record['globalEnabled'] === 'boolean') {
    const next = options.policy.withGlobalEnabled(record['globalEnabled']);
    options.policy.replace(next);
    const issues = await savePolicyDocument(options.policyPath, next, log);
    options.onSaved?.(options.policy.currentRevision);
    log.info(`capability policy: global switch set to ${record['globalEnabled'] ? 'enabled' : 'disabled'}`);
    sendJson(res, 200, { ...describe(options), saveIssues: issues });
    return;
  }

  const capability = record['capability'];
  if (!isCapability(String(capability))) {
    sendJson(res, 400, {
      error: `"${String(capability)}" is not a known capability`,
      known: CAPABILITIES,
    });
    return;
  }
  const orderRaw = record['order'];
  if (!Array.isArray(orderRaw)) {
    sendJson(res, 400, { error: 'order must be an array of { id, enabled }' });
    return;
  }
  const order: PolicyEntry[] = [];
  for (const [index, item] of orderRaw.entries()) {
    if (item === null || typeof item !== 'object' || Array.isArray(item)) {
      sendJson(res, 400, { error: `order[${index}] must be an object with id and enabled` });
      return;
    }
    const entry = item as Record<string, unknown>;
    if (typeof entry['id'] !== 'string' || typeof entry['enabled'] !== 'boolean') {
      sendJson(res, 400, { error: `order[${index}] needs a string id and a boolean enabled` });
      return;
    }
    order.push({ id: entry['id'], enabled: entry['enabled'] });
  }

  const known = options.hub
    .describeCapabilityOptions({ capabilities: [capability as Capability] })
    .flatMap((entry) => entry.plan.options.map((option) => option.option.id));
  const outcome = options.policy.applyEdit(capability as Capability, order, known);
  if (outcome.issues.length > 0) {
    sendJson(res, 400, {
      error: 'the submitted order was rejected',
      issues: outcome.issues.map((issue) => `${issue.path} ${issue.message}`),
    });
    return;
  }
  options.policy.replace(outcome.document);
  const issues = await savePolicyDocument(options.policyPath, outcome.document, log);
  options.onSaved?.(options.policy.currentRevision);
  log.info(
    `capability policy: ${capability} order saved (${order.length} option(s), ` +
      `${order.filter((entry) => entry.enabled).length} enabled) at revision ${options.policy.currentRevision}`,
  );
  sendJson(res, 200, { ...describe(options), saveIssues: issues });
}

/**
 * Read and parse a bounded JSON request body.
 *
 * Bounded because this route is reachable from a browser: a policy document is a
 * few kilobytes, so a megabyte cap is generous and anything past it is a mistake
 * or an attack rather than a legitimate edit.
 *
 * @param req - the request.
 * @returns the parsed body.
 * @throws Error when the body is too large or not valid JSON.
 */
async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const limit = 1_000_000;
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = typeof chunk === 'string' ? Buffer.from(chunk) : (chunk as Buffer);
    size += buffer.byteLength;
    if (size > limit) throw new Error(`the request body exceeds ${String(limit)} bytes`);
    chunks.push(buffer);
  }
  const text = Buffer.concat(chunks).toString('utf8').trim();
  if (text.length === 0) return {};
  return JSON.parse(text);
}

/**
 * Send a JSON response.
 * @param res - the response.
 * @param status - the HTTP status.
 * @param payload - the JSON body.
 */
function sendJson(res: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': Buffer.byteLength(body),
  });
  res.end(body);
}
