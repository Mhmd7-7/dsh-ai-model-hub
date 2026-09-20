/**
 * Lifecycle tools: start, stop, and health-check a model.
 *
 * These exist because *the hub* knows how to manage a process and the agent does
 * not. The agent never learns a launch command, an endpoint, or a Python
 * invocation â€” it names a model id it discovered through `list_models`, and the
 * runtime manager consults the descriptor.
 *
 * They are also the tools a deployment can restrict: `allowProcessLaunch: false`
 * on the plugin leaves `start_model` registered but failing with a clear reason,
 * so the agent gets an honest answer instead of a tool that vanished.
 *
 * @module dsh-ai-model-hub/dsh-plugin/tools/lifecycle
 */

import { defineTool } from '@deepseek-ai/dsh-tools';
import type { Context } from '@deepseek-ai/cordis';
import type { ModelHubService } from '../service.ts';
import { toHubError } from 'dsh-ai-model-hub/index.ts';
import { textBlock, toToolError } from './support.ts';

/**
 * Register `start_model`, `stop_model`, and `check_model_health`.
 *
 * @param ctx - the context whose `tools` registry receives them.
 * @param service - the hub service.
 * @param options - deployment controls.
 */
export function registerLifecycleTools(
  ctx: Context,
  service: ModelHubService,
  options: { readonly allowProcessLaunch: boolean },
): void {
  const hub = service.hub;

  ctx.tools.register(
    defineTool({
      name: 'start_model',
      description:
        'Start a local AI model so it can serve requests. ' +
        'Usually unnecessary: invoke_model starts a model automatically when needed and it is startable. ' +
        'Use this to preload a model before a multi-step workflow, or to surface a startup failure separately from an invocation failure. ' +
        'Report the model id from list_models. You do not need to know how the model launches.',
      parameters: {
        modelId: {
          type: 'string',
          required: true,
          description: 'The id of the model to start, as reported by list_models.',
        },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            modelId: { type: 'string', required: true },
            started: { type: 'boolean', required: true },
            alreadyRunning: { type: 'boolean', required: true },
            healthy: { type: 'boolean', required: true },
            detail: { type: 'string' },
          },
        },
        render: (_args, value) => {
          if (value.alreadyRunning) {
            return textBlock(`${value.modelId} was already running and healthy.`);
          }
          const lines = [`Started ${value.modelId}.`];
          lines.push(value.healthy ? 'Health check passing; it can serve requests now.' : `Warning: it started but is not healthy yet â€” ${value.detail ?? 'no detail'}`);
          return textBlock(lines.join('\n'));
        },
      },
      execute: async (args) => {
        if (!options.allowProcessLaunch) {
          throw new Error(
            'Error [UNSAFE_OPERATION]: this deployment has disabled hub-managed process launching ' +
              '(allowProcessLaunch is false). Start the engine yourself, or ask the operator to enable it. ' +
              'Models that are already running are still usable.',
          );
        }
        try {
          const result = await hub.startModel(args.modelId);
          return {
            modelId: args.modelId,
            started: result.started,
            alreadyRunning: result.alreadyRunning,
            healthy: result.health.healthy,
            ...(result.health.detail === undefined ? {} : { detail: result.health.detail }),
          };
        } catch (error) {
          throw toToolError(error, 'START_FAILED', { modelId: args.modelId });
        }
      },
    }),
  );

  ctx.tools.register(
    defineTool({
      name: 'stop_model',
      description:
        'Stop a model the hub started, freeing its memory and GPU. ' +
        'Only works on processes the hub launched; a model you started yourself is left alone. ' +
        'Use this to reclaim VRAM after a heavy workflow, or to recover a model stuck in a bad state.',
      parameters: {
        modelId: {
          type: 'string',
          required: true,
          description: 'The id of the model to stop.',
        },
        force: {
          type: 'boolean',
          description: 'Skip the graceful shutdown budget and terminate immediately. Defaults to false.',
        },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            modelId: { type: 'string', required: true },
            stopped: { type: 'boolean', required: true },
            wasRunning: { type: 'boolean', required: true },
          },
        },
        render: (_args, value) => {
          if (!value.wasRunning && !value.stopped) {
            return textBlock(`${value.modelId} has no process managed by the hub; nothing to stop.`);
          }
          return textBlock(`Stopped ${value.modelId}.`);
        },
      },
      execute: async (args) => {
        try {
          const result = await hub.stopModel(args.modelId, { force: args.force ?? false });
          return { modelId: args.modelId, stopped: result.stopped, wasRunning: result.wasRunning };
        } catch (error) {
          throw toToolError(error, 'LIFECYCLE_UNSUPPORTED', { modelId: args.modelId });
        }
      },
    }),
  );

  ctx.tools.register(
    defineTool({
      name: 'check_model_health',
      description:
        'Probe a model right now and report whether it is responding, how long the probe took, and why it failed if it did. ' +
        'Use this to diagnose an invocation failure or to wait for a slow engine to finish loading. ' +
        'Omit `modelId` to probe every model at once.',
      parameters: {
        modelId: {
          type: 'string',
          description: 'The model to probe. Omit to probe every registered model.',
        },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            results: {
              type: 'array',
              required: true,
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  modelId: { type: 'string', required: true },
                  healthy: { type: 'boolean', required: true },
                  detail: { type: 'string' },
                  latencyMs: { type: 'number' },
                  availability: { type: 'string', required: true },
                },
              },
            },
            healthyCount: { type: 'integer', required: true },
          },
        },
        render: (_args, value) => {
          const lines = [`${value.healthyCount} of ${value.results.length} model(s) healthy.`];
          for (const result of value.results) {
            const latency = result.latencyMs === undefined ? '' : ` in ${result.latencyMs} ms`;
            lines.push(
              `  ${result.modelId}: ${result.healthy ? `healthy${latency}` : `unhealthy â€” ${result.detail ?? 'no detail'}`} (now ${result.availability})`,
            );
          }
          return textBlock(lines.join('\n'));
        },
      },
      execute: async (args) => {
        try {
          const ids =
            args.modelId === undefined || args.modelId.trim().length === 0
              ? hub.listModels({ includeDisabled: true }).map((view) => view.model.id)
              : [args.modelId];
          const results: {
            modelId: string;
            healthy: boolean;
            availability: string;
            detail?: string;
            latencyMs?: number;
          }[] = [];
          for (const modelId of ids) {
            try {
              const probe = await hub.probeModel(modelId);
              results.push({
                modelId,
                healthy: probe.healthy,
                availability: hub.getModelStatus(modelId).availability,
                ...(probe.detail === undefined ? {} : { detail: probe.detail }),
                ...(probe.latencyMs === undefined ? {} : { latencyMs: probe.latencyMs }),
              });
            } catch (error) {
              // An unreachable engine is a *result*, not a tool failure: the whole
              // point of a health check is to report a negative finding.
              const hubError = toHubError(error, 'HEALTH_CHECK_FAILED', { modelId });
              results.push({
                modelId,
                healthy: false,
                availability: hub.getModelStatus(modelId).availability,
                detail: hubError.message,
              });
            }
          }
          return { results, healthyCount: results.filter((result) => result.healthy).length };
        } catch (error) {
          throw toToolError(error, 'MODEL_NOT_FOUND', { requestedModelId: args.modelId });
        }
      },
    }),
  );
}
