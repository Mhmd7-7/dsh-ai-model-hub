/**
 * The invocation tool â€” `invoke_model`.
 *
 * This is the seam the entire architecture exists to create. The agent states a
 * capability and content; the router picks the model; the runtime starts it if
 * needed; the adapter runs it; the artifacts come back. The agent never learns
 * which engine answered, and that is the point: swapping Stable Diffusion for
 * ComfyUI is a catalog edit, not a prompt edit.
 *
 * The tool's parameter list contains no field that could name a command, a path,
 * an endpoint, or a Python module. There is nothing here for a hostile prompt to
 * reach: the only model-identifying input is `modelId`, an *optional debug pin*
 * that must match an id the catalog already knows, and that is ignored for
 * routing purposes beyond candidate selection.
 *
 * @module dsh-ai-model-hub/dsh-plugin/tools/invoke
 */

import { defineTool } from '@deepseek-ai/dsh-tools';
import type { Context } from '@deepseek-ai/cordis';
import type { ModelHubService } from '../service.ts';
import type { Capability } from '../../src/index.ts';
import { toHubError } from '../../src/index.ts';
import { withTimeout } from '../../src/index.ts';
import { formatArtifact, formatFailure, textBlock } from './support.ts';
import type { ToolRunContext } from '@deepseek-ai/dsh-tools';
import type { ArtifactRootResolver } from '../workspace.ts';
import type { ToolCallScope } from '../types.ts';

/** How the tool asks for work. Mirrors `InvocationRequest` minus the parts the agent must not set. */
interface InvokeArgs {
  readonly capability: string;
  readonly prompt?: string;
  readonly inputs?: readonly (string | { readonly id: string; readonly type?: string })[];
  readonly options?: Readonly<Record<string, unknown>>;
  readonly modelId?: string;
  readonly requiredTags?: readonly string[];
  readonly timeoutMs?: number;
}

/**
 * Normalize the tool's input forms into the hub's request shape.
 *
 * The tool accepts a bare id string as well as `{ id, type }` because the terse
 * form is what the agent reaches for in a chained workflow. The declared kind is
 * carried across as an unchecked string: the hub validates it against the
 * artifact vocabulary and reports an unknown kind as an `ARTIFACT_ERROR`, which is
 * the right place for that judgement â€” narrowing here would produce a TypeScript
 * cast that lies about a value the model chose.
 *
 * @param inputs - the model-supplied input list.
 * @returns references the hub accepts, or `undefined` when none were supplied.
 */
function normalizeInputs(
  inputs: InvokeArgs['inputs'],
): readonly { readonly id: string; readonly type?: string }[] | undefined {
  if (inputs === undefined) return undefined;
  return inputs.map((entry) => (typeof entry === 'string' ? { id: entry } : entry));
}

/**
 * Register `invoke_model`.
 *
 * @param ctx - the context whose `tools` registry receives it.
 * @param service - the hub service.
 * @param options - deployment controls, notably the default timeout budget.
 */
export function registerInvokeTool(
  ctx: Context,
  service: ModelHubService,
  options: {
    readonly invocationTimeoutMs: number;
    /** Where this call's artifacts go: the calling session's workspace by default. */
    readonly artifactRootFor: ArtifactRootResolver;
  },
): void {
  const hub = service.hub;

  ctx.tools.register(
    defineTool({
      name: 'invoke_model',
      description:
        'Run a local AI capability and get the produced artifacts back. ' +
        'This is how you generate an image, build a 3D asset, transcribe audio, or run a local text model. ' +
        'You name the CAPABILITY you need (e.g. text_to_image); the system selects, starts, and calls a suitable model for you. ' +
        'You never need to know which model or engine handles it, and you should not try to run model software yourself. ' +
        'Feed one produced artifact into the next step by passing its artifact id in `inputs` â€” for example text_to_image then image_to_3d to go from a prompt to a mesh. ' +
        'Call list_capabilities first to see what this machine can serve.',
      parameters: {
        capability: {
          type: 'string',
          required: true,
          description:
            'The capability to run, e.g. text_to_text, text_to_image, image_to_image, text_to_3d, image_to_3d, audio_generation, speech_to_text, image_understanding, or video_generation.',
        },
        prompt: {
          type: 'string',
          description: 'The natural-language instruction: the image description, the mesh description, the question, or the text to process.',
        },
        inputs: {
          type: 'array',
          description:
            'Artifacts to consume, each either an artifact id string or { id, type }. Get ids from invoke_model results or list_artifacts. Required by capabilities that transform existing content, such as image_to_image, image_to_3d, and speech_to_text.',
          items: {
            // The DSL forbids `type` beside `oneOf`: an exact-one union IS the
            // node's type, so declaring both is a schema authoring error the
            // compiler rejects rather than silently ignoring.
            oneOf: [
              { type: 'string', description: 'An artifact id.' },
              {
                type: 'object',
                additionalProperties: false,
                properties: {
                  id: { type: 'string', required: true, description: 'The artifact id.' },
                  type: {
                    type: 'string',
                    description:
                      'The expected artifact kind, e.g. image. Checked against the artifact and reported clearly when it does not match.',
                  },
                },
              },
            ],
          },
        },
        options: {
          type: 'object',
          additionalProperties: true,
          description:
            'Capability-specific settings passed to the model unchanged, e.g. { "width": 1024, "height": 1024, "steps": 30 } for image generation. Unknown keys are ignored by models that do not use them.',
        },
        modelId: {
          type: 'string',
          description:
            'Optional debug pin: force one specific model from list_models instead of letting the system choose. Use only to compare models; omitting it is the normal path.',
        },
        requiredTags: {
          type: 'array',
          description: 'Restrict the choice to models carrying all of these tags, e.g. ["gpu"] or ["low-vram"].',
          items: { type: 'string' },
        },
        timeoutMs: {
          type: 'integer',
          description: 'Abandon the request after this many milliseconds. Defaults to the deployment budget.',
        },
      },
      timeoutMs: options.invocationTimeoutMs,
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            capability: { type: 'string', required: true },
            modelId: { type: 'string', required: true },
            coldStart: { type: 'boolean', required: true },
            durationMs: { type: 'integer', required: true },
            outputs: {
              type: 'array',
              required: true,
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  id: { type: 'string', required: true },
                  type: { type: 'string', required: true },
                  description: { type: 'string', required: true },
                },
              },
            },
            value: { type: 'object', additionalProperties: true },
          },
        },
        render: (args, value) => {
          const header = `${args.capability} served by ${value.modelId} in ${value.durationMs} ms${value.coldStart ? ' (cold start)' : ''}.`;
          if (value.outputs.length === 0) {
            return textBlock(`${header}\nThe model returned no artifacts.`);
          }
          const lines = [header, '', `Produced ${value.outputs.length} artifact(s):`];
          for (const artifact of value.outputs) {
            lines.push(`  ${artifact.type} â€” ${artifact.description}`);
            lines.push(`    artifact id: ${artifact.id}`);
          }
          lines.push('');
          lines.push(
            'Pass these artifact ids in the `inputs` parameter of a later invoke_model call to build on this output.',
          );
          return textBlock(lines.join('\n'));
        },
      },
      execute: async (args: InvokeArgs, exec: ToolRunContext) => {
        const budget = args.timeoutMs ?? options.invocationTimeoutMs;

        // Compose a single controller: it aborts on caller cancellation (the
        // tool call being interrupted) or on the deadline, and the hub sees one
        // signal either way.
        const controller = new AbortController();
        const onAbort = (): void => controller.abort();
        if (exec.signal.aborted) controller.abort();
        else exec.signal.addEventListener('abort', onAbort, { once: true });

        try {
          const normalizedInputs = normalizeInputs(args.inputs);
          // Resolved from the calling session, so a long-lived host writes each
          // conversation's images beside that conversation's code.
          const artifactRoot = options.artifactRootFor(exec as unknown as ToolCallScope);
          const invocation = hub.invokeModel(
            {
              capability: args.capability as Capability,
              ...(args.prompt === undefined ? {} : { prompt: args.prompt }),
              ...(normalizedInputs === undefined ? {} : { inputs: normalizedInputs }),
              ...(args.options === undefined ? {} : { options: args.options }),
              ...(args.modelId === undefined || args.modelId.trim().length === 0
                ? {}
                : { modelId: args.modelId }),
              ...(args.requiredTags === undefined ? {} : { requiredTags: args.requiredTags }),
              signal: controller.signal,
            },
            {
              allowFallback: true,
              maxAttempts: 3,
              ...(artifactRoot === undefined ? {} : { artifactRoot }),
            },
          );

          const settled = await withTimeout(
            invocation,
            budget,
            `invoke_model(${args.capability})`,
            exec.signal,
          );

          return {
            capability: settled.capability,
            modelId: settled.modelId,
            coldStart: settled.coldStart,
            durationMs: settled.durationMs,
            outputs: settled.outputs.map((artifact) => ({
              id: artifact.id,
              type: artifact.type,
              description: formatArtifact(artifact, '')
                .split('\n')[0]
                ?.replace(/^\s+/, '') ?? artifact.id,
            })),
            ...(settled.value === undefined ? {} : { value: { ...settled.value } }),
          };
        } catch (error) {
          // A tool failure is what the model needs to see: the stable code plus a
          // message that says what to do differently, not a stack trace.
          const hubError = toHubError(error, 'INVOCATION_FAILED', { capability: args.capability });
          const hint = hintFor(hubError.code);
          throw new Error(
            `${formatFailure(hubError.code, hubError.message, hubError.details)}${hint === '' ? '' : `\n${hint}`}`,
          );
        } finally {
          exec.signal.removeEventListener('abort', onAbort);
        }
      },
    }),
  );
}

/**
 * Add an actionable next step for the failure codes an agent can act on.
 *
 * A failure the agent can recover from should say how. A failure it cannot â€”
 * `UNSAFE_OPERATION` â€” gets no hint, because the correct next step is to stop.
 *
 * @param code - the stable failure code.
 * @returns a suggestion line, or an empty string.
 */
function hintFor(code: string): string {
  switch (code) {
    case 'NO_COMPATIBLE_MODEL':
      return 'Hint: call list_capabilities to see which capabilities this machine can actually serve, and choose a different approach if this one is not available.';
    case 'UNKNOWN_CAPABILITY':
      return 'Hint: call list_capabilities for the exact capability names this deployment supports.';
    case 'MODEL_UNAVAILABLE':
      return 'Hint: call get_model_status for the model, then start_model if it is startable and stopped.';
    case 'INSUFFICIENT_RESOURCES':
      return 'Hint: this model needs more memory than the machine has. Call list_models and choose a model with smaller requirements, or drop the requiredTags filter.';
    case 'ARTIFACT_ERROR':
      return 'Hint: check the artifact id with list_artifacts, and that its kind matches what this capability consumes.';
    case 'INVOCATION_TIMEOUT':
      return 'Hint: the model exceeded its budget. Retry with a larger timeoutMs, or choose a smaller/faster model.';
    case 'INVOCATION_FAILED':
      return 'Hint: call check_model_health for the models involved; every candidate failed, so this is likely an engine problem rather than a request problem.';
    case 'START_FAILED':
      return 'Hint: call get_model_status â€” the engine probably failed to load. Its output is included in the failure details.';
    default:
      return '';
  }
}
