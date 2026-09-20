/**
 * The mock adapter (Phase 1).
 *
 * Three mock models — `mock_text_model`, `mock_image_model`, `mock_3d_model` —
 * exist to validate the *architecture* end to end before any real engine is
 * involved. They are not stubs: they honor the same adapter contract, produce
 * real artifacts in real formats (a genuine PNG, a loadable STL, real text), and
 * fail the same way a real engine does. Swapping one for a real adapter later
 * changes nothing above this file.
 *
 * Everything here is deterministic. The same prompt always yields byte-identical
 * output, which is what makes the integration tests meaningful rather than flaky.
 *
 * @module dsh-ai-model-hub/adapters/mock
 */

import type { Capability } from '../catalog/capabilities.ts';
import type { ResolvedModel } from '../catalog/descriptor.ts';
import type { HealthReport } from '../types.ts';
import { ModelHubError } from '../errors.ts';
import { delay } from '../util/process.ts';
import { readArtifactConventions } from '../artifacts/types.ts';
import type { AdapterInvocation, AdapterOutput, ModelAdapter } from './types.ts';
import { MOCK_STL_VERTEX_COUNT, renderMockPng, renderMockStl } from './png.ts';

/** Options the mock adapter honours, read from `adapterConfig` and per-call `options`. */
interface MockSettings {
  /** Milliseconds of simulated work, so timeouts and cancellation are exercisable. */
  readonly latencyMs: number;
  /** Default image width in pixels. */
  readonly width: number;
  /** Default image height in pixels. */
  readonly height: number;
  /** Grid spacing in the generated image, in pixels. */
  readonly gridStep: number;
}

const MOCK_DEFAULTS: MockSettings = {
  latencyMs: 0,
  width: 512,
  height: 512,
  gridStep: 64,
};

/**
 * Read a numeric option from per-call options, then adapter config, then default.
 * @param options - per-call options.
 * @param config - the model's adapter configuration.
 * @param key - the option name.
 * @param fallback - the value to use when neither source supplies one.
 * @returns the resolved number.
 */
function numberSetting(
  options: Readonly<Record<string, unknown>>,
  config: Readonly<Record<string, unknown>>,
  key: string,
  fallback: number,
): number {
  const fromOptions = options[key];
  if (typeof fromOptions === 'number' && Number.isFinite(fromOptions)) return fromOptions;
  const fromConfig = config[key];
  if (typeof fromConfig === 'number' && Number.isFinite(fromConfig)) return fromConfig;
  return fallback;
}

/**
 * Resolve the mock adapter's settings for one invocation.
 * @param invocation - the resolved request.
 * @returns merged settings.
 */
function settingsFor(invocation: AdapterInvocation): MockSettings {
  const config = invocation.model.adapterConfig;
  const options = invocation.options;
  return {
    latencyMs: Math.max(0, numberSetting(options, config, 'latencyMs', MOCK_DEFAULTS.latencyMs)),
    width: Math.max(1, Math.round(numberSetting(options, config, 'width', invocation.model.limits.maxWidth ?? MOCK_DEFAULTS.width))),
    height: Math.max(1, Math.round(numberSetting(options, config, 'height', invocation.model.limits.maxHeight ?? MOCK_DEFAULTS.height))),
    gridStep: Math.max(2, Math.round(numberSetting(options, config, 'gridStep', MOCK_DEFAULTS.gridStep))),
  };
}

/**
 * Simulate work in a cancellable way.
 *
 * A real engine takes time; a mock that returns instantly cannot exercise
 * cancellation, timeouts, or concurrent-call accounting. Sleeping the declared
 * latency, and rejecting when the signal fires, makes those paths testable.
 *
 * @param ms - how long to simulate.
 * @param signal - the caller's cancellation.
 * @throws ModelHubError with `INVOCATION_ABORTED` when cancelled.
 */
async function simulateWork(ms: number, signal: AbortSignal): Promise<void> {
  if (ms <= 0) {
    if (signal.aborted) throw new ModelHubError('INVOCATION_ABORTED', 'invocation cancelled before it started');
    return;
  }
  if (signal.aborted) throw new ModelHubError('INVOCATION_ABORTED', 'invocation cancelled before it started');
  const cancelled = new Promise<never>((_resolve, reject) => {
    signal.addEventListener(
      'abort',
      () => reject(new ModelHubError('INVOCATION_ABORTED', 'invocation cancelled during simulated work')),
      { once: true },
    );
  });
  await Promise.race([delay(ms), cancelled]);
}

/**
 * The prompt, or a deterministic fallback when the capability allows it to be absent.
 * @param invocation - the resolved request.
 * @param required - whether a missing prompt is an error.
 * @returns the prompt text.
 * @throws ModelHubError with `INVOCATION_FAILED` when required and absent.
 */
function requirePrompt(invocation: AdapterInvocation, required: boolean): string {
  const prompt = invocation.prompt;
  if (prompt !== undefined && prompt.trim().length > 0) return prompt;
  if (required) {
    throw new ModelHubError(
      'INVOCATION_FAILED',
      `capability "${invocation.capability}" requires a \`prompt\` and none was supplied`,
      { modelId: invocation.model.id, capability: invocation.capability },
    );
  }
  return '';
}

/** A file-name-safe truncation of a seed string. */
function slug(seed: string, maxLength = 32): string {
  const cleaned = seed
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned.length === 0 ? 'untitled' : cleaned.slice(0, maxLength);
}

/**
 * Create the mock adapter.
 *
 * One adapter serves every mock model. It dispatches on the *capability* the
 * caller requested, not on the model id — the same rule the router follows, for
 * the same reason: model identity must never drive behaviour.
 *
 * @returns the adapter instance.
 */
export function createMockAdapter(): ModelAdapter {
  return {
    kind: 'mock',
    displayName: 'Mock adapter (Phase 1 fixtures)',

    supports(model: ResolvedModel): { ok: true } | { ok: false; reason: string } {
      if (model.capabilities.length === 0) {
        return { ok: false, reason: 'declares no capabilities' };
      }
      return { ok: true };
    },

    async health(model: ResolvedModel, signal: AbortSignal): Promise<HealthReport> {
      const checkedAt = Date.now();
      if (signal.aborted) {
        return { healthy: false, checkedAt, detail: 'probe cancelled' };
      }
      return {
        healthy: true,
        checkedAt,
        latencyMs: 0,
        detail: `in-process mock (${model.adapterConfig['fixture'] === undefined ? 'default' : String(model.adapterConfig['fixture'])})`,
      };
    },

    async invoke(invocation: AdapterInvocation): Promise<AdapterOutput> {
      const settings = settingsFor(invocation);
      await simulateWork(settings.latencyMs, invocation.signal);
      const handler = HANDLERS[invocation.capability];
      if (handler === undefined) {
        throw new ModelHubError(
          'UNSUPPORTED_OPERATION',
          `the mock adapter does not implement capability "${invocation.capability}"`,
          { capability: invocation.capability, modelId: invocation.model.id },
        );
      }
      return handler(invocation, settings);
    },
  };
}

/** Per-capability fixture builders. */
const HANDLERS: Readonly<
  Record<Capability, (invocation: AdapterInvocation, settings: MockSettings) => Promise<AdapterOutput>>
> = {
  text_to_text: async (invocation) => {
    const prompt = requirePrompt(invocation, true);
    const contextNote =
      invocation.inputs.length === 0
        ? ''
        : `\n\n(Read ${invocation.inputs.length} input artifact(s): ${invocation.inputs.map((artifact) => artifact.id).join(', ')})`;
    const text = [
      `[mock_text_model] deterministic fixture response`,
      `prompt: ${prompt}`,
      `characters: ${prompt.length}`,
      `words: ${prompt.split(/\s+/).filter((word) => word.length > 0).length}${contextNote}`,
    ].join('\n');
    const artifact = await invocation.artifacts.put({
      type: 'text',
      text,
      mimeType: 'text/plain',
      label: `mock text for "${slug(prompt, 24)}"`,
      producerModelId: invocation.model.id,
      extension: '.txt',
      metadata: { prompt, fixture: 'text_to_text' },
    });
    return { outputs: [artifact], value: { text, characters: prompt.length } };
  },

  text_to_image: async (invocation, settings) => {
    const prompt = requirePrompt(invocation, true);
    const bytes = renderMockPng(settings.width, settings.height, prompt, { gridStep: settings.gridStep });
    const artifact = await invocation.artifacts.put({
      type: 'image',
      bytes,
      mimeType: 'image/png',
      label: `mock image for "${slug(prompt, 24)}"`,
      producerModelId: invocation.model.id,
      extension: '.png',
      metadata: { width: settings.width, height: settings.height, prompt, format: 'png', fixture: 'text_to_image' },
    });
    return {
      outputs: [artifact],
      value: { width: settings.width, height: settings.height, format: 'png', prompt },
    };
  },

  image_to_image: async (invocation, settings) => {
    const source = invocation.inputs.find((artifact) => artifact.type === 'image');
    if (source === undefined) {
      throw new ModelHubError(
        'INVOCATION_FAILED',
        'image_to_image requires an input artifact of type `image`',
        { modelId: invocation.model.id, providedTypes: invocation.inputs.map((artifact) => artifact.type) },
      );
    }
    const conventions = readArtifactConventions(source);
    // Preserve the source's aspect when the caller did not override dimensions.
    const width = Math.round(numberSetting(invocation.options, invocation.model.adapterConfig, 'width', conventions.width ?? settings.width));
    const height = Math.round(numberSetting(invocation.options, invocation.model.adapterConfig, 'height', conventions.height ?? settings.height));
    const instruction = invocation.prompt ?? 'restyle';
    const bytes = renderMockPng(width, height, `${instruction}:${source.id}`, { gridStep: settings.gridStep });
    const artifact = await invocation.artifacts.put({
      type: 'image',
      bytes,
      mimeType: 'image/png',
      label: `mock edit of ${source.id}`,
      producerModelId: invocation.model.id,
      extension: '.png',
      metadata: {
        width,
        height,
        format: 'png',
        prompt: instruction,
        sourceArtifactId: source.id,
        fixture: 'image_to_image',
      },
    });
    return {
      outputs: [artifact],
      value: { width, height, format: 'png', sourceArtifactId: source.id },
    };
  },

  text_to_3d: async (invocation) => {
    const prompt = requirePrompt(invocation, true);
    const stl = renderMockStl(prompt, { scale: 1 });
    const artifact = await invocation.artifacts.put({
      type: 'model_3d',
      text: stl,
      mimeType: 'model/stl',
      label: `mock mesh for "${slug(prompt, 24)}"`,
      producerModelId: invocation.model.id,
      extension: '.stl',
      metadata: {
        format: 'stl',
        vertexCount: MOCK_STL_VERTEX_COUNT,
        triangleCount: MOCK_STL_VERTEX_COUNT / 3,
        prompt,
        fixture: 'text_to_3d',
      },
    });
    return {
      outputs: [artifact],
      value: {
        format: 'stl',
        vertexCount: MOCK_STL_VERTEX_COUNT,
        triangleCount: MOCK_STL_VERTEX_COUNT / 3,
      },
    };
  },

  image_to_3d: async (invocation) => {
    const source = invocation.inputs.find((artifact) => artifact.type === 'image');
    if (source === undefined) {
      throw new ModelHubError(
        'INVOCATION_FAILED',
        'image_to_3d requires an input artifact of type `image`',
        { modelId: invocation.model.id, providedTypes: invocation.inputs.map((artifact) => artifact.type) },
      );
    }
    const conventions = readArtifactConventions(source);
    const seed = `${invocation.prompt ?? 'mesh'}:${source.id}:${conventions.width ?? 0}x${conventions.height ?? 0}`;
    const stl = renderMockStl(seed, { scale: 1 });
    const artifact = await invocation.artifacts.put({
      type: 'model_3d',
      text: stl,
      mimeType: 'model/stl',
      label: `mock mesh from ${source.id}`,
      producerModelId: invocation.model.id,
      extension: '.stl',
      metadata: {
        format: 'stl',
        vertexCount: MOCK_STL_VERTEX_COUNT,
        triangleCount: MOCK_STL_VERTEX_COUNT / 3,
        sourceArtifactId: source.id,
        prompt: invocation.prompt ?? '',
        fixture: 'image_to_3d',
      },
    });
    return {
      outputs: [artifact],
      value: {
        format: 'stl',
        vertexCount: MOCK_STL_VERTEX_COUNT,
        triangleCount: MOCK_STL_VERTEX_COUNT / 3,
        sourceArtifactId: source.id,
      },
    };
  },

  image_understanding: async (invocation) => {
    const source = invocation.inputs.find((artifact) => artifact.type === 'image');
    if (source === undefined) {
      throw new ModelHubError(
        'INVOCATION_FAILED',
        'image_understanding requires an input artifact of type `image`',
        { modelId: invocation.model.id, providedTypes: invocation.inputs.map((artifact) => artifact.type) },
      );
    }
    const conventions = readArtifactConventions(source);
    const text = [
      `[mock image understanding] deterministic fixture description`,
      `artifact: ${source.id}`,
      `dimensions: ${conventions.width ?? 'unknown'}x${conventions.height ?? 'unknown'}`,
      `question: ${invocation.prompt ?? '(none)'}`,
    ].join('\n');
    const artifact = await invocation.artifacts.put({
      type: 'text',
      text,
      mimeType: 'text/plain',
      label: `mock description of ${source.id}`,
      producerModelId: invocation.model.id,
      extension: '.txt',
      metadata: { sourceArtifactId: source.id, fixture: 'image_understanding' },
    });
    return { outputs: [artifact], value: { text, sourceArtifactId: source.id } };
  },

  audio_generation: async (invocation) => {
    const prompt = requirePrompt(invocation, false);
    // A 44-byte canonical WAV header followed by deterministically-generated
    // PCM silence-with-tone, so the output is a real, playable file.
    const seconds = Math.min(
      numberSetting(invocation.options, invocation.model.adapterConfig, 'seconds', 1),
      invocation.model.limits.maxDurationSeconds ?? Number.POSITIVE_INFINITY,
    );
    const sampleRate = 16_000;
    const sampleCount = Math.max(1, Math.round(seconds * sampleRate));
    const bytes = renderMockWav(sampleCount, sampleRate, prompt);
    const artifact = await invocation.artifacts.put({
      type: 'audio',
      bytes,
      mimeType: 'audio/wav',
      label: `mock audio for "${slug(prompt, 24)}"`,
      producerModelId: invocation.model.id,
      extension: '.wav',
      metadata: { durationSeconds: seconds, sampleRate, format: 'wav', prompt, fixture: 'audio_generation' },
    });
    return { outputs: [artifact], value: { durationSeconds: seconds, sampleRate, format: 'wav' } };
  },

  speech_to_text: async (invocation) => {
    const source = invocation.inputs.find((artifact) => artifact.type === 'audio');
    if (source === undefined) {
      throw new ModelHubError(
        'INVOCATION_FAILED',
        'speech_to_text requires an input artifact of type `audio`',
        { modelId: invocation.model.id, providedTypes: invocation.inputs.map((artifact) => artifact.type) },
      );
    }
    const text = `[mock transcription] deterministic fixture for ${source.id}`;
    const artifact = await invocation.artifacts.put({
      type: 'text',
      text,
      mimeType: 'text/plain',
      label: `mock transcript of ${source.id}`,
      producerModelId: invocation.model.id,
      extension: '.txt',
      metadata: { sourceArtifactId: source.id, fixture: 'speech_to_text' },
    });
    return { outputs: [artifact], value: { text, sourceArtifactId: source.id } };
  },

  video_generation: async (invocation, settings) => {
    const prompt = requirePrompt(invocation, true);
    // A minimal, valid MP4 is far more machinery than a fixture justifies, so the
    // mock emits a self-describing JSON manifest instead and says so in metadata.
    // A downstream consumer that needs real video must use a real adapter.
    const seconds = Math.min(
      numberSetting(invocation.options, invocation.model.adapterConfig, 'seconds', 2),
      invocation.model.limits.maxDurationSeconds ?? Number.POSITIVE_INFINITY,
    );
    const manifest = JSON.stringify(
      { fixture: 'video_generation', prompt, seconds, width: settings.width, height: settings.height, frames: Math.round(seconds * 8) },
      null,
      2,
    );
    const artifact = await invocation.artifacts.put({
      type: 'video',
      text: manifest,
      mimeType: 'application/json',
      label: `mock video manifest for "${slug(prompt, 24)}"`,
      producerModelId: invocation.model.id,
      extension: '.video.json',
      metadata: { durationSeconds: seconds, format: 'mock-manifest', prompt, fixture: 'video_generation' },
    });
    return {
      outputs: [artifact],
      value: { durationSeconds: seconds, format: 'mock-manifest' },
    };
  },
};

/**
 * Render a valid mono 16-bit PCM WAV containing a short deterministic tone.
 * @param sampleCount - number of samples to emit.
 * @param sampleRate - samples per second.
 * @param seed - string the tone frequency is derived from.
 * @returns the encoded WAV bytes.
 */
export function renderMockWav(sampleCount: number, sampleRate: number, seed: string): Uint8Array {
  const dataBytes = sampleCount * 2;
  const buffer = new Uint8Array(44 + dataBytes);
  const view = new DataView(buffer.buffer);

  const ascii = (offset: number, text: string): void => {
    for (let index = 0; index < text.length; index += 1) {
      buffer[offset + index] = text.charCodeAt(index);
    }
  };

  let hash = 0x811c9dc5;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  const frequency = 220 + (hash % 440);

  ascii(0, 'RIFF');
  view.setUint32(4, 36 + dataBytes, true);
  ascii(8, 'WAVE');
  ascii(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // channels: mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  ascii(36, 'data');
  view.setUint32(40, dataBytes, true);

  for (let index = 0; index < sampleCount; index += 1) {
    const t = index / sampleRate;
    const envelope = 0.25 * Math.min(1, index / (sampleRate * 0.05), (sampleCount - index) / (sampleRate * 0.05));
    const sample = Math.sin(2 * Math.PI * frequency * t) * envelope;
    view.setInt16(44 + index * 2, Math.round(sample * 32_767), true);
  }
  return buffer;
}
