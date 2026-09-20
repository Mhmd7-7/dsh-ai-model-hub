/**
 * The capability vocabulary.
 *
 * A *capability* is the single currency of this system: the agent asks for a
 * capability, never for a model. Everything above this file (router, DSH plugin,
 * agent prompt) speaks capabilities; everything below it (adapters, runtimes,
 * launch commands) speaks models.
 *
 * Adding a capability is a vocabulary change plus, usually, an adapter and a
 * mock. It is NEVER a router change — the router contains no capability-specific
 * branches.
 *
 * @module dsh-ai-model-hub/catalog/capabilities
 */

/** Every capability this hub understands, as a frozen literal tuple. */
export const CAPABILITIES = [
  'text_to_text',
  'text_to_image',
  'image_to_image',
  'text_to_3d',
  'image_to_3d',
  'audio_generation',
  'speech_to_text',
  'image_understanding',
  'video_generation',
] as const;

/** One capability name. */
export type Capability = (typeof CAPABILITIES)[number];

/**
 * What a model fundamentally *is*, as opposed to what it can do.
 *
 * `type` drives operator-facing grouping and adapter selection by default; the
 * router itself filters on `capabilities`, not on `type`, so a model may declare
 * a `type` that sits awkwardly beside one of its capabilities without breaking
 * routing.
 */
export const MODEL_TYPES = [
  'text_generation',
  'image_generation',
  'image_editing',
  'image_understanding',
  'three_d_generation',
  'audio_generation',
  'speech_recognition',
  'video_generation',
  'multimodal',
  'custom',
] as const;

/** One model type. */
export type ModelType = (typeof MODEL_TYPES)[number];

/** The artifact kinds a model may consume or produce. */
export const IO_TYPES = [
  'text',
  'image',
  'audio',
  'video',
  'model_3d',
  'json',
  'file',
] as const;

/** One artifact kind. */
export type IoType = (typeof IO_TYPES)[number];

/**
 * The canonical input/output contract of each capability.
 *
 * This table is *documentation and defaulting*, not routing policy: the router
 * never reads it. The catalog uses it to fill a manifest's `inputTypes` /
 * `outputTypes` when the manifest author omits them, and the DSH prompt section
 * renders it so the model can see which capabilities compose into which
 * workflows (e.g. that `text_to_image` yields an `image`, which is exactly what
 * `image_to_3d` accepts).
 */
export const CAPABILITY_IO: Readonly<
  Record<Capability, { readonly input: readonly IoType[]; readonly output: readonly IoType[] }>
> = Object.freeze({
  text_to_text: { input: ['text'], output: ['text'] },
  text_to_image: { input: ['text'], output: ['image'] },
  image_to_image: { input: ['image', 'text'], output: ['image'] },
  text_to_3d: { input: ['text'], output: ['model_3d'] },
  image_to_3d: { input: ['image', 'text'], output: ['model_3d'] },
  audio_generation: { input: ['text'], output: ['audio'] },
  speech_to_text: { input: ['audio'], output: ['text'] },
  image_understanding: { input: ['image', 'text'], output: ['text'] },
  video_generation: { input: ['text', 'image'], output: ['video'] },
});

const CAPABILITY_SET: ReadonlySet<string> = new Set(CAPABILITIES);
const MODEL_TYPE_SET: ReadonlySet<string> = new Set(MODEL_TYPES);
const IO_TYPE_SET: ReadonlySet<string> = new Set(IO_TYPES);

/**
 * Narrow an arbitrary string to a known capability.
 * @param value - candidate string.
 * @returns true when the value is a known capability.
 */
export function isCapability(value: string): value is Capability {
  return CAPABILITY_SET.has(value);
}

/**
 * Narrow an arbitrary string to a known model type.
 * @param value - candidate string.
 * @returns true when the value is a known model type.
 */
export function isModelType(value: string): value is ModelType {
  return MODEL_TYPE_SET.has(value);
}

/**
 * Narrow an arbitrary string to a known artifact kind.
 * @param value - candidate string.
 * @returns true when the value is a known artifact kind.
 */
export function isIoType(value: string): value is IoType {
  return IO_TYPE_SET.has(value);
}

/**
 * The default artifact kinds a capability consumes and produces.
 * @param capability - the capability to describe.
 * @returns its canonical input and output kinds; `json` only for unknown input.
 */
export function defaultIoFor(capability: Capability): { input: IoType[]; output: IoType[] } {
  const io = CAPABILITY_IO[capability];
  return { input: [...io.input], output: [...io.output] };
}
