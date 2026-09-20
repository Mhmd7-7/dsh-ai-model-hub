/**
 * The `comfyui` adapter: real image generation through a ComfyUI server.
 *
 * Why this is not `http_json`
 * ---------------------------
 * A1111 and stable-diffusion.cpp answer a single request with the finished
 * image. ComfyUI does not: it takes a *node graph*, queues it, returns an id,
 * executes asynchronously, and only then exposes outputs by filename. There is
 * no request body that means "draw this prompt". The catalog example says as
 * much — "ComfyUI has a JSON graph API rather than a simple prompt endpoint, so
 * it needs its own adapter" — and this is that adapter.
 *
 * How a prompt reaches a graph
 * ----------------------------
 * A graph is model-specific, so this adapter does not invent one. It takes a
 * *template* graph (an API-format workflow, either inline as `workflow` or from
 * a file named by `workflowPath`) and edits it:
 *
 *   1. the prompt text is written to the CLIPTextEncode node that feeds the
 *      sampler's `positive` input;
 *   2. width/height go to the latent node (`Empty*Latent*`);
 *   3. seed/steps/cfg/sampler go to the sampler node.
 *
 * Every target node is discovered from the graph's own wiring, so a template
 * keeps working when node ids are renumbered by the ComfyUI editor. Explicit
 * `*NodeId` settings override discovery when a graph is unusual.
 *
 * @module dsh-ai-model-hub/adapters/comfyui
 */

import { readFile } from 'node:fs/promises';
import { isAbsolute, resolve } from 'node:path';

import type { Capability } from '../catalog/capabilities.ts';
import type { ResolvedModel } from '../catalog/descriptor.ts';
import type { HealthReport } from '../types.ts';
import { ModelHubError } from '../errors.ts';
import type { AdapterInvocation, AdapterOutput, ModelAdapter } from './types.ts';

/** Capabilities this adapter can serve. */
const IMAGE_CAPABILITIES: readonly Capability[] = ['text_to_image', 'image_to_image'];

/** Default budget for one queued graph. Diffusion on a laptop GPU is slow. */
const DEFAULT_TIMEOUT_MS = 600_000;

/** How often to ask ComfyUI whether the queued graph has finished. */
const DEFAULT_POLL_INTERVAL_MS = 750;

/** A node in an API-format ComfyUI graph. */
interface ComfyNode {
  readonly class_type?: unknown;
  /** Mutable: the adapter rewrites inputs to inject the prompt and settings. */
  inputs?: Record<string, unknown>;
}

/** An API-format ComfyUI graph: node id to node. */
type ComfyGraph = Record<string, ComfyNode>;

/** One queued image output as ComfyUI reports it. */
interface ComfyImageRef {
  readonly filename: string;
  readonly subfolder: string;
  readonly type: string;
}

/** Settings resolved from adapter config and per-call options. */
interface ComfySettings {
  readonly prompt: string;
  readonly negativePrompt?: string;
  readonly width?: number;
  readonly height?: number;
  readonly steps?: number;
  readonly cfg?: number;
  readonly seed?: number;
  readonly sampler?: string;
  readonly scheduler?: string;
  readonly denoise?: number;
  readonly filenamePrefix?: string;
  readonly clientId: string;
  readonly pollIntervalMs: number;
  readonly timeoutMs: number;
  readonly workflowPath?: string;
  readonly workflow?: ComfyGraph;
  readonly promptNodeId?: string;
  readonly negativePromptNodeId?: string;
  readonly latentNodeId?: string;
  readonly samplerNodeId?: string;
}

/** Read a finite number from options, then config. */
function numberSetting(
  options: Readonly<Record<string, unknown>>,
  config: Readonly<Record<string, unknown>>,
  key: string,
): number | undefined {
  const fromOptions = options[key];
  if (typeof fromOptions === 'number' && Number.isFinite(fromOptions)) return fromOptions;
  const fromConfig = config[key];
  if (typeof fromConfig === 'number' && Number.isFinite(fromConfig)) return fromConfig;
  return undefined;
}

/** Read a non-empty string from options, then config. */
function stringSetting(
  options: Readonly<Record<string, unknown>>,
  config: Readonly<Record<string, unknown>>,
  key: string,
): string | undefined {
  const fromOptions = options[key];
  if (typeof fromOptions === 'string' && fromOptions.length > 0) return fromOptions;
  const fromConfig = config[key];
  if (typeof fromConfig === 'string' && fromConfig.length > 0) return fromConfig;
  return undefined;
}

/**
 * Resolve every setting for one invocation.
 * @param invocation - the resolved request.
 * @returns the merged settings.
 * @throws ModelHubError when no prompt was supplied.
 */
function settingsFor(invocation: AdapterInvocation): ComfySettings {
  const config = invocation.model.adapterConfig;
  const options = invocation.options;

  const prompt = invocation.prompt;
  if (prompt === undefined || prompt.trim().length === 0) {
    throw new ModelHubError(
      'INVOCATION_FAILED',
      `capability "${invocation.capability}" requires a \`prompt\` and none was supplied`,
      { modelId: invocation.model.id, capability: invocation.capability },
    );
  }

  const inline = config['workflow'];
  const inlineGraph =
    inline !== null && typeof inline === 'object' && !Array.isArray(inline)
      ? (inline as ComfyGraph)
      : undefined;

  const width = numberSetting(options, config, 'width') ?? invocation.model.limits.maxWidth;
  const height = numberSetting(options, config, 'height') ?? invocation.model.limits.maxHeight;

  return {
    prompt,
    ...(stringSetting(options, config, 'negativePrompt') === undefined
      ? {}
      : { negativePrompt: stringSetting(options, config, 'negativePrompt') as string }),
    ...(width === undefined ? {} : { width: Math.max(1, Math.round(width)) }),
    ...(height === undefined ? {} : { height: Math.max(1, Math.round(height)) }),
    ...(numberSetting(options, config, 'steps') === undefined
      ? {}
      : { steps: Math.max(1, Math.round(numberSetting(options, config, 'steps') as number)) }),
    ...(numberSetting(options, config, 'cfg') === undefined
      ? {}
      : { cfg: numberSetting(options, config, 'cfg') as number }),
    ...(numberSetting(options, config, 'seed') === undefined
      ? {}
      : { seed: Math.round(numberSetting(options, config, 'seed') as number) }),
    ...(numberSetting(options, config, 'denoise') === undefined
      ? {}
      : { denoise: numberSetting(options, config, 'denoise') as number }),
    ...(stringSetting(options, config, 'sampler') === undefined
      ? {}
      : { sampler: stringSetting(options, config, 'sampler') as string }),
    ...(stringSetting(options, config, 'scheduler') === undefined
      ? {}
      : { scheduler: stringSetting(options, config, 'scheduler') as string }),
    ...(stringSetting(options, config, 'filenamePrefix') === undefined
      ? {}
      : { filenamePrefix: stringSetting(options, config, 'filenamePrefix') as string }),
    clientId: stringSetting(options, config, 'clientId') ?? 'dsh-ai-model-hub',
    pollIntervalMs: Math.max(50, numberSetting(options, config, 'pollIntervalMs') ?? DEFAULT_POLL_INTERVAL_MS),
    timeoutMs: Math.max(1, numberSetting(options, config, 'timeoutMs') ?? DEFAULT_TIMEOUT_MS),
    ...(stringSetting(options, config, 'workflowPath') === undefined
      ? {}
      : { workflowPath: stringSetting(options, config, 'workflowPath') as string }),
    ...(inlineGraph === undefined ? {} : { workflow: inlineGraph }),
    ...(stringSetting(options, config, 'promptNodeId') === undefined
      ? {}
      : { promptNodeId: stringSetting(options, config, 'promptNodeId') as string }),
    ...(stringSetting(options, config, 'negativePromptNodeId') === undefined
      ? {}
      : { negativePromptNodeId: stringSetting(options, config, 'negativePromptNodeId') as string }),
    ...(stringSetting(options, config, 'latentNodeId') === undefined
      ? {}
      : { latentNodeId: stringSetting(options, config, 'latentNodeId') as string }),
    ...(stringSetting(options, config, 'samplerNodeId') === undefined
      ? {}
      : { samplerNodeId: stringSetting(options, config, 'samplerNodeId') as string }),
  };
}

/**
 * Read a template graph from a file.
 *
 * Accepts both shapes found in the wild: a bare graph, and the
 * `{ client_id, prompt }` envelope ComfyUI's own API examples use.
 *
 * @param path - the workflow file path, absolute or relative to the cwd.
 * @returns the graph.
 * @throws ModelHubError when the file cannot be read or holds no graph.
 */
async function loadWorkflowFile(path: string): Promise<ComfyGraph> {
  const absolute = isAbsolute(path) ? path : resolve(process.cwd(), path);
  let text: string;
  try {
    text = await readFile(absolute, 'utf8');
  } catch (error) {
    throw new ModelHubError(
      'CONFIG_ERROR',
      `comfyui workflow file "${absolute}" could not be read: ${error instanceof Error ? error.message : String(error)}`,
      { path: absolute },
    );
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new ModelHubError(
      'CONFIG_ERROR',
      `comfyui workflow file "${absolute}" is not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
      { path: absolute },
    );
  }
  return unwrapGraph(parsed, absolute);
}

/**
 * Accept either a bare graph or a `{ prompt: graph }` envelope.
 * @param parsed - the parsed document.
 * @param origin - where it came from, for error messages.
 * @returns the graph.
 * @throws ModelHubError when neither shape is present.
 */
function unwrapGraph(parsed: unknown, origin: string): ComfyGraph {
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new ModelHubError('CONFIG_ERROR', `comfyui workflow at ${origin} is not a JSON object`, { origin });
  }
  const record = parsed as Record<string, unknown>;
  const inner = record['prompt'];
  if (inner !== null && typeof inner === 'object' && !Array.isArray(inner)) {
    return inner as ComfyGraph;
  }
  return record as ComfyGraph;
}

/**
 * Find the node id whose node has one of the given class types.
 * @param graph - the graph.
 * @param classTypes - acceptable `class_type` values.
 * @returns the node id, or `undefined`.
 */
function findNodeByClass(graph: ComfyGraph, classTypes: readonly string[]): string | undefined {
  for (const [id, node] of Object.entries(graph)) {
    if (typeof node.class_type === 'string' && classTypes.includes(node.class_type)) return id;
  }
  return undefined;
}

/**
 * Find the sampler node: explicit id, else the first KSampler-family node.
 * @param graph - the graph.
 * @param explicit - a configured node id.
 * @returns the node id, or `undefined`.
 */
function findSampler(graph: ComfyGraph, explicit: string | undefined): string | undefined {
  if (explicit !== undefined) return graph[explicit] === undefined ? undefined : explicit;
  return findNodeByClass(graph, ['KSampler', 'KSamplerAdvanced', 'SamplerCustom', 'KSamplerSelect']);
}

/**
 * Resolve the prompt node from the sampler's `positive` link.
 *
 * This is why node ids need not be pinned: whatever the editor renumbered, the
 * node that feeds `positive` is the one that carries the prompt.
 *
 * @param graph - the graph.
 * @param samplerId - the sampler node id.
 * @returns the node id, or `undefined`.
 */
function findPositiveNode(graph: ComfyGraph, samplerId: string): string | undefined {
  const link = graph[samplerId]?.inputs?.['positive'];
  if (Array.isArray(link) && typeof link[0] === 'string') return link[0];
  if (typeof link === 'string') return link;
  return findNodeByClass(graph, ['CLIPTextEncode']);
}

/**
 * Resolve the negative prompt node from the sampler's `negative` link.
 * @param graph - the graph.
 * @param samplerId - the sampler node id.
 * @returns the node id, or `undefined`.
 */
function findNegativeNode(graph: ComfyGraph, samplerId: string): string | undefined {
  const link = graph[samplerId]?.inputs?.['negative'];
  if (Array.isArray(link) && typeof link[0] === 'string') return link[0];
  if (typeof link === 'string') return link;
  return undefined;
}

/**
 * Find the latent node that carries width and height.
 * @param graph - the graph.
 * @param explicit - a configured node id.
 * @returns the node id, or `undefined`.
 */
function findLatent(graph: ComfyGraph, explicit: string | undefined): string | undefined {
  if (explicit !== undefined) return graph[explicit] === undefined ? undefined : explicit;
  for (const [id, node] of Object.entries(graph)) {
    const kind = node.class_type;
    if (typeof kind === 'string' && kind.startsWith('Empty') && kind.includes('Latent')) return id;
  }
  return undefined;
}

/**
 * Apply a mutation to a node's inputs, ignoring missing nodes.
 * @param graph - the graph.
 * @param nodeId - the node id, or `undefined` to do nothing.
 * @param patch - the input fields to set.
 */
function patchInputs(graph: ComfyGraph, nodeId: string | undefined, patch: Record<string, unknown>): void {
  if (nodeId === undefined) return;
  const node = graph[nodeId];
  if (node === undefined) return;
  node.inputs = { ...(node.inputs ?? {}), ...patch };
}

/**
 * Build the graph for one invocation by editing a copy of the template.
 *
 * @param settings - resolved settings.
 * @param graph - the template graph.
 * @returns the edited graph.
 * @throws ModelHubError when the template has no usable sampler or prompt node.
 */
function buildGraph(settings: ComfySettings, graph: ComfyGraph): ComfyGraph {
  const draft: ComfyGraph = structuredClone(graph);

  const samplerId = findSampler(draft, settings.samplerNodeId);
  if (samplerId === undefined) {
    throw new ModelHubError(
      'INVOCATION_FAILED',
      'the comfyui workflow contains no sampler node (expected KSampler or KSamplerAdvanced)',
      { nodeIds: Object.keys(draft) },
    );
  }

  const promptId = settings.promptNodeId ?? findPositiveNode(draft, samplerId);
  if (promptId === undefined || draft[promptId] === undefined) {
    throw new ModelHubError(
      'INVOCATION_FAILED',
      'could not locate the prompt node in the comfyui workflow; set `promptNodeId` explicitly',
      { samplerId, nodeIds: Object.keys(draft) },
    );
  }
  patchInputs(draft, promptId, { text: settings.prompt });

  if (settings.negativePrompt !== undefined) {
    patchInputs(draft, settings.negativePromptNodeId ?? findNegativeNode(draft, samplerId), {
      text: settings.negativePrompt,
    });
  }

  const latentId = findLatent(draft, settings.latentNodeId);
  patchInputs(draft, latentId, {
    ...(settings.width === undefined ? {} : { width: settings.width }),
    ...(settings.height === undefined ? {} : { height: settings.height }),
  });

  // Seed -1 means "pick one", which is what an image generator should do unless
  // the caller asked for a reproducible render.
  const seed = settings.seed ?? Math.floor(Math.random() * 2 ** 31);
  patchInputs(draft, samplerId, {
    seed,
    ...(settings.steps === undefined ? {} : { steps: settings.steps }),
    ...(settings.cfg === undefined ? {} : { cfg: settings.cfg }),
    ...(settings.sampler === undefined ? {} : { sampler_name: settings.sampler }),
    ...(settings.scheduler === undefined ? {} : { scheduler: settings.scheduler }),
    ...(settings.denoise === undefined ? {} : { denoise: settings.denoise }),
  });

  if (settings.filenamePrefix !== undefined) {
    const saverId = findNodeByClass(draft, ['SaveImage', 'SaveImageWebsocket']);
    patchInputs(draft, saverId, { filename_prefix: settings.filenamePrefix });
  }

  return draft;
}

/**
 * Create the ComfyUI adapter.
 * @returns the adapter instance.
 */
export function createComfyUiAdapter(): ModelAdapter {
  return {
    kind: 'comfyui',
    displayName: 'ComfyUI graph-queue endpoint',

    supports(model: ResolvedModel): { ok: true } | { ok: false; reason: string } {
      if (model.runtime.endpoint === undefined) {
        return { ok: false, reason: 'the comfyui adapter requires `runtime.endpoint` (e.g. http://127.0.0.1:8188)' };
      }
      const config = model.adapterConfig;
      const hasTemplate =
        (typeof config['workflowPath'] === 'string' && config['workflowPath'].length > 0) ||
        (config['workflow'] !== null && typeof config['workflow'] === 'object');
      if (!hasTemplate) {
        return {
          ok: false,
          reason:
            'the comfyui adapter requires a workflow template: set `adapterConfig.workflowPath` to an ' +
            'API-format workflow JSON, or `adapterConfig.workflow` to an inline graph',
        };
      }
      const supported = model.capabilities.filter((capability) => IMAGE_CAPABILITIES.includes(capability));
      if (supported.length === 0) {
        return {
          ok: false,
          reason:
            `it declares only ${model.capabilities.join(', ') || 'no capabilities'}, and this adapter serves ` +
            `${IMAGE_CAPABILITIES.join(', ')}`,
        };
      }
      return { ok: true };
    },

    async health(model: ResolvedModel, signal: AbortSignal): Promise<HealthReport> {
      const started = Date.now();
      const endpoint = model.runtime.endpoint;
      if (endpoint === undefined) {
        return { healthy: false, checkedAt: started, detail: 'no runtime.endpoint configured' };
      }
      let url: string;
      try {
        const base = endpoint.endsWith('/') ? endpoint : `${endpoint}/`;
        url = new URL((model.health.path ?? '/system_stats').replace(/^\//, ''), base).toString();
      } catch {
        return { healthy: false, checkedAt: started, detail: `endpoint "${endpoint}" is not a valid URL` };
      }

      // Never throws: this runs on a timer and inside the cold-start gate.
      const controller = new AbortController();
      const onAbort = (): void => controller.abort();
      if (signal.aborted) controller.abort();
      else signal.addEventListener('abort', onAbort, { once: true });
      const timer = setTimeout(() => controller.abort(), model.health.timeoutMs ?? 3_000);
      try {
        const response = await fetch(url, { method: 'GET', signal: controller.signal, headers: { accept: '*/*' } });
        return {
          healthy: response.status < 500,
          checkedAt: started,
          latencyMs: Date.now() - started,
          detail: `${url} responded ${response.status}`,
        };
      } catch (error) {
        return {
          healthy: false,
          checkedAt: started,
          latencyMs: Date.now() - started,
          detail: `${url} unreachable: ${error instanceof Error ? error.message : String(error)}`,
        };
      } finally {
        clearTimeout(timer);
        signal.removeEventListener('abort', onAbort);
      }
    },

    async invoke(invocation: AdapterInvocation): Promise<AdapterOutput> {
      const model = invocation.model;
      if (!IMAGE_CAPABILITIES.includes(invocation.capability)) {
        throw new ModelHubError(
          'UNSUPPORTED_OPERATION',
          `the comfyui adapter does not implement capability "${invocation.capability}"`,
          { modelId: model.id, capability: invocation.capability, supported: [...IMAGE_CAPABILITIES] },
        );
      }

      const endpoint = model.runtime.endpoint;
      if (endpoint === undefined) {
        throw new ModelHubError(
          'INVOCATION_FAILED',
          `model "${model.id}" has no runtime.endpoint for the comfyui adapter`,
          { modelId: model.id },
        );
      }

      const settings = settingsFor(invocation);

      const template =
        settings.workflow ??
        (settings.workflowPath === undefined ? undefined : await loadWorkflowFile(settings.workflowPath));
      if (template === undefined) {
        throw new ModelHubError(
          'INVOCATION_FAILED',
          `model "${model.id}" has no comfyui workflow template configured`,
          { modelId: model.id },
        );
      }

      const graph = buildGraph(settings, template);
      const base = endpoint.endsWith('/') ? endpoint : `${endpoint}/`;
      const queueUrl = new URL('prompt', base).toString();

      // One deadline and one controller for the whole operation: queue, execute,
      // and every image fetch share the caller's budget.
      const controller = new AbortController();
      let timedOut = false;
      const started = Date.now();
      const onCallerAbort = (): void => controller.abort();
      if (invocation.signal.aborted) controller.abort();
      else invocation.signal.addEventListener('abort', onCallerAbort, { once: true });
      const timer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, settings.timeoutMs);

      const reason = (): ModelHubError => {
        if (timedOut) {
          return new ModelHubError(
            'INVOCATION_TIMEOUT',
            `model "${model.id}" did not finish within ${settings.timeoutMs} ms`,
            { modelId: model.id, timeoutMs: settings.timeoutMs },
          );
        }
        if (invocation.signal.aborted) {
          return new ModelHubError('INVOCATION_ABORTED', 'invocation cancelled while ComfyUI was working', {
            modelId: model.id,
          });
        }
        return new ModelHubError('INVOCATION_FAILED', 'ComfyUI request aborted', { modelId: model.id });
      };

      try {
        invocation.log.debug('comfyui: queueing graph', { queueUrl, nodes: Object.keys(graph).length });

        let queued: Response;
        try {
          queued = await fetch(queueUrl, {
            method: 'POST',
            signal: controller.signal,
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ client_id: settings.clientId, prompt: graph }),
          });
        } catch (error) {
          if (controller.signal.aborted) throw reason();
          throw new ModelHubError(
            'INVOCATION_FAILED',
            `model "${model.id}" could not reach ${queueUrl}: ${error instanceof Error ? error.message : String(error)}`,
            { modelId: model.id, url: queueUrl },
          );
        }

        if (!queued.ok) {
          let detail = '';
          try {
            detail = (await queued.text()).slice(0, 800);
          } catch {
            detail = '(body unreadable)';
          }
          throw new ModelHubError(
            'INVOCATION_FAILED',
            `ComfyUI rejected the queued graph (${queued.status} ${queued.statusText}): ${detail}`,
            { modelId: model.id, status: queued.status },
          );
        }

        const queueBody = (await queued.json()) as Record<string, unknown>;
        const promptId = queueBody['prompt_id'];
        if (typeof promptId !== 'string' || promptId.length === 0) {
          const nodeErrors = queueBody['node_errors'];
          throw new ModelHubError(
            'INVOCATION_FAILED',
            `ComfyUI accepted the request but returned no prompt_id: ${JSON.stringify(nodeErrors ?? queueBody).slice(0, 800)}`,
            { modelId: model.id },
          );
        }

        const historyUrl = new URL(`history/${promptId}`, base).toString();
        let entry: Record<string, unknown> | undefined;
        while (entry === undefined) {
          if (controller.signal.aborted) throw reason();
          await new Promise<void>((resolveSleep) => setTimeout(resolveSleep, settings.pollIntervalMs));
          if (controller.signal.aborted) throw reason();

          let polled: Response;
          try {
            polled = await fetch(historyUrl, { signal: controller.signal, headers: { accept: 'application/json' } });
          } catch (error) {
            if (controller.signal.aborted) throw reason();
            throw new ModelHubError(
              'INVOCATION_FAILED',
              `could not poll ComfyUI history at ${historyUrl}: ${error instanceof Error ? error.message : String(error)}`,
              { modelId: model.id, url: historyUrl },
            );
          }
          if (!polled.ok) {
            throw new ModelHubError(
              'INVOCATION_FAILED',
              `ComfyUI history at ${historyUrl} answered ${polled.status} ${polled.statusText}`,
              { modelId: model.id, status: polled.status },
            );
          }
          const history = (await polled.json()) as Record<string, unknown>;
          const candidate = history[promptId];
          if (candidate !== undefined && typeof candidate === 'object' && candidate !== null) {
            entry = candidate as Record<string, unknown>;
          }
        }

        // ComfyUI reports per-node execution errors alongside the outputs; a
        // graph can "complete" with nothing produced, and saying so plainly is
        // far more useful than reporting zero images.
        const status = entry['status'];
        const statusStr =
          status !== null && typeof status === 'object'
            ? (status as Record<string, unknown>)['status_str']
            : undefined;
        if (statusStr === 'error') {
          const messages =
            status !== null && typeof status === 'object'
              ? (status as Record<string, unknown>)['messages']
              : undefined;
          throw new ModelHubError(
            'INVOCATION_FAILED',
            `ComfyUI failed to execute the graph: ${JSON.stringify(messages ?? status).slice(0, 800)}`,
            { modelId: model.id, promptId },
          );
        }

        const outputs = entry['outputs'];
        const images: ComfyImageRef[] = [];
        if (outputs !== null && typeof outputs === 'object') {
          for (const nodeOutput of Object.values(outputs as Record<string, unknown>)) {
            if (nodeOutput === null || typeof nodeOutput !== 'object') continue;
            const list = (nodeOutput as Record<string, unknown>)['images'];
            if (!Array.isArray(list)) continue;
            for (const item of list) {
              if (item === null || typeof item !== 'object') continue;
              const record = item as Record<string, unknown>;
              const filename = record['filename'];
              if (typeof filename !== 'string' || filename.length === 0) continue;
              images.push({
                filename,
                subfolder: typeof record['subfolder'] === 'string' ? record['subfolder'] : '',
                type: typeof record['type'] === 'string' ? record['type'] : 'output',
              });
            }
          }
        }

        if (images.length === 0) {
          throw new ModelHubError(
            'INVOCATION_FAILED',
            `ComfyUI executed the graph (${Date.now() - started} ms) but produced no images. ` +
              'Check that the template ends in a SaveImage node.',
            { modelId: model.id, promptId },
          );
        }

        const artifacts = [];
        for (const image of images) {
          const view = new URL('view', base);
          view.searchParams.set('filename', image.filename);
          view.searchParams.set('subfolder', image.subfolder);
          view.searchParams.set('type', image.type);

          let downloaded: Response;
          try {
            downloaded = await fetch(view.toString(), { signal: controller.signal });
          } catch (error) {
            if (controller.signal.aborted) throw reason();
            throw new ModelHubError(
              'INVOCATION_FAILED',
              `could not download "${image.filename}" from ${view.toString()}: ${error instanceof Error ? error.message : String(error)}`,
              { modelId: model.id, filename: image.filename },
            );
          }
          if (!downloaded.ok) {
            throw new ModelHubError(
              'INVOCATION_FAILED',
              `ComfyUI view endpoint answered ${downloaded.status} for "${image.filename}"`,
              { modelId: model.id, filename: image.filename, status: downloaded.status },
            );
          }

          const bytes = new Uint8Array(await downloaded.arrayBuffer());
          const dimensions = readPngDimensions(bytes);
          const artifact = await invocation.artifacts.put({
            type: 'image',
            bytes,
            mimeType: 'image/png',
            label: image.filename.replace(/\.[^.]+$/, ''),
            producerModelId: model.id,
            extension: '.png',
            metadata: {
              prompt: settings.prompt,
              ...(settings.negativePrompt === undefined ? {} : { negativePrompt: settings.negativePrompt }),
              ...(dimensions === undefined ? {} : { width: dimensions.width, height: dimensions.height }),
              seed: settings.seed ?? null,
              promptId,
              source: 'comfyui',
              comfyFilename: image.filename,
            },
          });
          artifacts.push(artifact);
        }

        invocation.log.info('comfyui: generated image(s)', { count: artifacts.length, promptId });

        const first = artifacts[0];
        return {
          outputs: artifacts,
          value: {
            count: artifacts.length,
            format: 'png',
            prompt: settings.prompt,
            promptId,
            ...(first?.metadata['width'] === undefined ? {} : { width: first.metadata['width'] as number }),
            ...(first?.metadata['height'] === undefined ? {} : { height: first.metadata['height'] as number }),
          },
        };
      } finally {
        clearTimeout(timer);
        invocation.signal.removeEventListener('abort', onCallerAbort);
      }
    },
  };
}

/**
 * Read pixel dimensions out of a PNG's IHDR chunk.
 * @param bytes - the encoded image.
 * @returns the dimensions, or `undefined` when it is not a PNG.
 */
export function readPngDimensions(bytes: Uint8Array): { width: number; height: number } | undefined {
  const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 24) return undefined;
  for (let index = 0; index < PNG_SIGNATURE.length; index += 1) {
    if (bytes[index] !== PNG_SIGNATURE[index]) return undefined;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const width = view.getUint32(16, false);
  const height = view.getUint32(20, false);
  if (width === 0 || height === 0) return undefined;
  return { width, height };
}
