/**
 * Shared formatting for model-facing tool output.
 *
 * Every tool renders through these helpers so the agent sees one consistent
 * vocabulary for availability, resources, and artifacts, whether it called
 * `list_models`, `get_model_status`, or `invoke_model`.
 *
 * The rendering is chosen for a model reader, not a human one: short labelled
 * lines, no decorative boxes, and every field that changes a decision (is it
 * usable? what does it cost? what did it produce?) present and stable.
 *
 * @module dsh-ai-model-hub/dsh-plugin/tools/support
 */

import type { ContentBlock } from '@deepseek-ai/dsh-llm';
import type { Artifact, ModelView, ModelRuntimeStatus } from '../../src/index.ts';
import { describeArtifact, toHubError } from '../../src/index.ts';

/**
 * Wrap text as a single model-facing content block.
 * @param text - the rendered text.
 * @returns one content block.
 */
export function textBlock(text: string): ContentBlock[] {
  return [{ type: 'text', text }];
}

/**
 * Render an availability state the way the agent should read it.
 * @param status - the runtime status.
 * @returns a labelled line.
 */
export function formatAvailability(status: ModelRuntimeStatus): string {
  const health =
    status.health === undefined
      ? ''
      : status.health.healthy
        ? ' (health check passing)'
        : ` (health check failing: ${status.health.detail ?? 'no detail'})`;
  const pid = status.pid === undefined ? '' : `, pid ${status.pid}`;
  const reason = status.reason === undefined ? '' : ` â€” ${status.reason}`;
  return `${status.availability} [${status.lifecycle}${pid}]${health}${reason}`;
}

/**
 * Render one model as a compact block.
 * @param view - the model and its live status.
 * @param indent - leading whitespace.
 * @returns a multi-line summary.
 */
export function formatModel(view: ModelView, indent = ''): string {
  const { model, status } = view;
  const lines = [
    `${indent}${model.id} â€” ${model.name}`,
    `${indent}  type: ${model.type}${model.hostId === undefined ? '' : ` | host: ${model.hostId}`}`,
    `${indent}  capabilities: ${model.capabilities.length === 0 ? '(none)' : model.capabilities.join(', ')}`,
    `${indent}  accepts: ${model.inputTypes.join(', ') || '(nothing)'} â†’ produces: ${model.outputTypes.join(', ') || '(nothing)'}`,
    `${indent}  status: ${formatAvailability(status)}`,
    `${indent}  runtime: ${model.runtime.engine} via ${model.adapter}${model.runtime.endpoint === undefined ? '' : ` at ${model.runtime.endpoint}`}`,
    `${indent}  resources: ${formatResources(model.resources.vramGb, model.resources.ramGb, model.resources.requiresGpu)}`,
    `${indent}  lifecycle control: ${formatLifecycleControl(model.lifecycle.startable, model.lifecycle.stoppable)}`,
  ];
  if (model.limits.contextTokens !== undefined) {
    lines.push(`${indent}  context window: ${model.limits.contextTokens} tokens`);
  }
  if (model.limits.resolutions !== undefined && model.limits.resolutions.length > 0) {
    lines.push(`${indent}  supported resolutions: ${model.limits.resolutions.join(', ')}`);
  }
  if (model.tags.length > 0) {
    lines.push(`${indent}  tags: ${model.tags.join(', ')}`);
  }
  if (model.version !== '0.0.0') {
    lines.push(`${indent}  version: ${model.version}`);
  }
  return lines.join('\n');
}

/**
 * Render a resource envelope.
 * @param vramGb - requested VRAM.
 * @param ramGb - requested RAM.
 * @param requiresGpu - whether a GPU is mandatory.
 * @returns a compact phrase.
 */
export function formatResources(vramGb: number, ramGb: number, requiresGpu: boolean): string {
  const parts: string[] = [];
  if (vramGb > 0) parts.push(`${vramGb} GiB VRAM`);
  if (ramGb > 0) parts.push(`${ramGb} GiB RAM`);
  if (requiresGpu) parts.push('GPU required');
  return parts.length === 0 ? 'negligible' : parts.join(', ');
}

/**
 * Render whether the hub may start and stop a model.
 * @param startable - whether a launch command exists.
 * @param stoppable - whether the hub owns the process.
 * @returns a compact phrase with the reason when not startable.
 */
export function formatLifecycleControl(startable: boolean, stoppable: boolean): string {
  if (!startable) return 'external â€” the hub will not start or stop it';
  return stoppable ? 'the hub can start, stop, and restart it' : 'the hub can start it but not stop it';
}

/**
 * Render one artifact for a tool result.
 * @param artifact - the artifact.
 * @param indent - leading whitespace.
 * @returns a multi-line block naming the artifact, its content, and its handle.
 */
export function formatArtifact(artifact: Artifact, indent = '  '): string {
  const lines = [`${indent}${describeArtifact(artifact)}`, `${indent}  artifact id: ${artifact.id}`];
  if (artifact.uri.startsWith('file://')) {
    lines.push(`${indent}  path: ${artifact.uri.slice('file://'.length)}`);
  } else {
    lines.push(`${indent}  uri: ${artifact.uri}`);
  }
  const metadata = Object.entries(artifact.metadata).filter(([key]) => key !== 'prompt');
  if (metadata.length > 0) {
    lines.push(
      `${indent}  metadata: ${metadata.map(([key, value]) => `${key}=${String(value)}`).join(', ')}`,
    );
  }
  if (artifact.producerModelId !== undefined) {
    lines.push(`${indent}  produced by: ${artifact.producerModelId}`);
  }
  return lines.join('\n');
}

/**
 * Render an error for a tool result.
 *
 * The code is included because it is the machine-readable part: an agent can
 * distinguish `NO_COMPATIBLE_MODEL` (plan differently) from `MODEL_UNAVAILABLE`
 * (start it) from `UNSAFE_OPERATION` (stop; this is a bug).
 *
 * @param code - the stable failure code.
 * @param message - the human-readable message.
 * @param details - structured context, rendered when small enough to be useful.
 * @returns a multi-line explanation.
 */
export function formatFailure(
  code: string,
  message: string,
  details?: Readonly<Record<string, unknown>>,
): string {
  const lines = [`Error [${code}]: ${message}`];
  if (details !== undefined) {
    const rendered = JSON.stringify(details);
    if (rendered.length <= 1500) lines.push(`details: ${rendered}`);
  }
  return lines.join('\n');
}

/**
 * Turn any thrown value into the tool failure the model should see.
 *
 * Every tool funnels its errors through here so the stable code is always
 * present and is always in the same place. Without that consistency an agent
 * cannot branch on failure kind, and a recovery hint attached by one tool would
 * go missing in another.
 *
 * @param error - the caught value.
 * @param fallbackCode - code to use when the value carries none.
 * @param context - extra structured context merged into the details.
 * @returns an `Error` carrying the formatted, code-led message.
 */
export function toToolError(
  error: unknown,
  fallbackCode: Parameters<typeof toHubError>[1] = 'INVOCATION_FAILED',
  context: Readonly<Record<string, unknown>> = {},
): Error {
  const hubError = toHubError(error, fallbackCode, context);
  return new Error(formatFailure(hubError.code, hubError.message, hubError.details));
}
