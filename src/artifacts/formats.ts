/**
 * 3D asset formats.
 *
 * A 3D artifact is not one format. Engines disagree about which container they
 * emit, the same way image engines disagree about PNG and WebP, and a hub that
 * hardcoded one would turn "this engine writes OBJ" into "this engine does not
 * work". So the format is a *fact about the bytes*, established by sniffing them,
 * and this module is the only place that knows the vocabulary.
 *
 * Two rules shape it:
 *
 * - **Sniff, do not trust.** A server that answers `200` with an HTML error page
 *   is a real failure mode of every HTTP engine. Validating the leading bytes
 *   turns that into "the engine returned something that is not a mesh" instead of
 *   an artifact the next model in the chain cannot open.
 * - **A sniff is evidence, not a gate.** OBJ and binary STL have no magic number
 *   worth the name, so a failed sniff is recorded as a warning on the artifact
 *   rather than raised as an error. Refusing to store a mesh because its header
 *   was unusual would be worse than storing it with an honest caveat.
 *
 * @module dsh-ai-model-hub/artifacts/formats
 */

import type { IoType } from '../catalog/capabilities.ts';

/** The 3D container formats the hub recognises by name. */
export const THREE_D_FORMATS = ['glb', 'gltf', 'obj', 'stl', 'ply'] as const;

/** One recognised 3D container format. */
export type ThreeDFormat = (typeof THREE_D_FORMATS)[number];

/** What is known about one format. */
export interface ThreeDFormatInfo {
  /** The canonical lowercase name, also used as the artifact's `format` metadata. */
  readonly format: ThreeDFormat;
  /** File extension including the dot. */
  readonly extension: string;
  /** MIME type registered for the container. */
  readonly mimeType: string;
  /** The artifact kind a file of this format is stored as. */
  readonly ioType: IoType;
}

/** The registry, keyed by format name. */
const FORMATS: Readonly<Record<ThreeDFormat, ThreeDFormatInfo>> = Object.freeze({
  glb: { format: 'glb', extension: '.glb', mimeType: 'model/gltf-binary', ioType: 'model_3d' },
  gltf: { format: 'gltf', extension: '.gltf', mimeType: 'model/gltf+json', ioType: 'model_3d' },
  obj: { format: 'obj', extension: '.obj', mimeType: 'model/obj', ioType: 'model_3d' },
  stl: { format: 'stl', extension: '.stl', mimeType: 'model/stl', ioType: 'model_3d' },
  ply: { format: 'ply', extension: '.ply', mimeType: 'model/ply', ioType: 'model_3d' },
});

/**
 * MIME type used for a 3D artifact whose container could not be identified.
 *
 * Deliberately the generic octet-stream rather than a guess at `model/gltf-binary`:
 * a downstream consumer that trusts a wrong MIME type fails while parsing, which
 * is precisely the confusion this module exists to prevent.
 */
export const UNKNOWN_THREE_D_MIME = 'application/octet-stream';

/** Extension used for a 3D artifact whose container could not be identified. */
export const UNKNOWN_THREE_D_EXTENSION = '.bin';

/** The outcome of examining a mesh's leading bytes. */
export interface ThreeDSniffResult {
  /** The detected format, or `undefined` when the bytes matched nothing known. */
  readonly format?: ThreeDFormat;
  /** A short explanation when the bytes do not look like the format claimed. */
  readonly warning?: string;
}

/**
 * Normalize a format name, extension, or MIME type to a known format.
 *
 * Accepts the three spellings engines and configuration actually use —
 * `glb`, `.glb`, `model/gltf-binary` — because an operator writing a catalog entry
 * should not have to know which one this particular field wants.
 *
 * @param value - the candidate name, extension, or MIME type.
 * @returns the format, or `undefined` when it is not recognised.
 */
export function threeDFormatOf(value: string): ThreeDFormat | undefined {
  const normalized = value.trim().toLowerCase();
  if (normalized.length === 0) return undefined;
  const bare = normalized.startsWith('.') ? normalized.slice(1) : normalized;
  for (const format of THREE_D_FORMATS) {
    const info = FORMATS[format];
    if (bare === format || bare === info.mimeType || normalized === info.mimeType) return format;
  }
  // `model/x.stl-binary`, `application/x-ply`, `model/gltf-binary; charset=…`
  // and friends: strip any vendor prefix and parameters, then match on the tail,
  // so a parameterised or non-standard MIME type still lands on its format.
  const subtype = normalized.includes('/') ? normalized.slice(normalized.indexOf('/') + 1) : normalized;
  const tail = subtype.split(';')[0]?.split('+').pop()?.trim() ?? subtype;
  for (const format of THREE_D_FORMATS) {
    const info = FORMATS[format];
    if (tail === format || tail.endsWith(`.${format}`) || tail.includes(format)) return format;
    if (info.mimeType.endsWith(`/${tail}`)) return format;
  }
  return undefined;
}

/**
 * The registry entry for a format.
 * @param format - the format name.
 * @returns its extension, MIME type, and artifact kind.
 */
export function threeDFormatInfo(format: ThreeDFormat): ThreeDFormatInfo {
  return FORMATS[format];
}

/**
 * Examine a mesh's leading bytes and report what container they look like.
 *
 * The checks are the ones a reader would make, in the order that makes them
 * unambiguous:
 *
 * - GLB begins with the ASCII magic `glTF`.
 * - glTF is JSON, so it begins with `{` (after an optional BOM and whitespace).
 * - PLY begins with `ply`.
 * - Binary STL has no magic, but its 80-byte header is followed by a 32-bit
 *   triangle count whose implied file length must match the actual length — a
 *   strong enough signal to be worth stating when it holds.
 * - OBJ is line-oriented text whose geometry lines start with `v `, `f `, or
 *   `o `; a file with none of those is reported as suspect.
 *
 * @param bytes - the content, or at least its first few hundred bytes.
 * @param claimed - the format the engine or configuration claimed, if any.
 * @returns the detected format and, when it disagrees with `claimed`, a warning.
 */
export function sniffThreeDFormat(bytes: Uint8Array, claimed?: string): ThreeDSniffResult {
  const detected = detectFormat(bytes);
  const claimedFormat = claimed === undefined ? undefined : threeDFormatOf(claimed);

  if (claimedFormat !== undefined && detected !== undefined && detected !== claimedFormat) {
    return {
      format: claimedFormat,
      warning: `content looks like ${detected} but was declared as ${claimedFormat}`,
    };
  }
  if (claimedFormat !== undefined && detected === undefined) {
    return {
      format: claimedFormat,
      warning: `content does not carry a recognisable ${claimedFormat} signature`,
    };
  }
  return detected === undefined ? {} : { format: detected };
}

/**
 * Identify a container from its leading bytes.
 * @param bytes - the content.
 * @returns the detected format, or `undefined`.
 */
function detectFormat(bytes: Uint8Array): ThreeDFormat | undefined {
  if (bytes.byteLength === 0) return undefined;
  const head = decodeHead(bytes, 512);

  if (head.startsWith('glTF')) return 'glb';
  if (head.startsWith('ply')) return 'ply';

  const text = stripBom(head).trimStart();
  if (text.startsWith('{')) return 'gltf';
  if (looksLikeObj(text)) return 'obj';
  if (text.startsWith('solid') && text.includes('facet')) return 'stl';
  if (looksLikeBinaryStl(bytes)) return 'stl';
  return undefined;
}

/**
 * Whether text carries OBJ geometry.
 * @param text - the leading text of the file.
 * @returns true when a vertex, face, or object line is present.
 */
function looksLikeObj(text: string): boolean {
  return /^(v|vn|vt|f|o|g|usemtl|mtllib)\s/m.test(text);
}

/**
 * Whether the bytes are consistent with a binary STL.
 *
 * The 80-byte header is unconstrained, so the only real evidence is arithmetic:
 * 50 bytes per triangle after the header and the count. A mismatch means the
 * file is something else that merely happens to be long enough.
 *
 * @param bytes - the content.
 * @returns true when the declared triangle count matches the actual length.
 */
function looksLikeBinaryStl(bytes: Uint8Array): boolean {
  if (bytes.byteLength < 84) return false;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const triangles = view.getUint32(80, true);
  if (triangles === 0) return false;
  return 84 + triangles * 50 === bytes.byteLength;
}

/**
 * Remove a UTF-8 byte-order mark from the head of a decoded string.
 * @param value - the decoded text.
 * @returns the text without a leading BOM.
 */
function stripBom(value: string): string {
  return value.charCodeAt(0) === 0xfeff ? value.slice(1) : value;
}

/**
 * Decode the leading bytes as UTF-8, stopping early on binary content.
 *
 * A GLB's header is ASCII but its body is not, and `TextDecoder` with
 * `fatal: false` replaces invalid sequences rather than throwing, so decoding a
 * bounded prefix is safe and cheap.
 *
 * @param bytes - the content.
 * @param maxBytes - how many bytes to decode at most.
 * @returns the decoded prefix.
 */
function decodeHead(bytes: Uint8Array, maxBytes: number): string {
  const slice = bytes.subarray(0, Math.min(maxBytes, bytes.byteLength));
  return new TextDecoder('utf-8', { fatal: false }).decode(slice);
}

/**
 * Count vertices in a mesh, from the content itself.
 *
 * Reported as artifact metadata so a downstream model or the agent can judge a
 * mesh's weight without opening it. The count is derived per format from the
 * file's own structure — OBJ vertex lines, STL's declared triangle count, glTF's
 * accessor table, GLB's embedded JSON chunk — and is simply absent when the
 * format does not make it cheap to obtain.
 *
 * @param bytes - the content.
 * @param format - the detected or claimed format.
 * @returns the vertex count and triangle count, when either is knowable.
 */
export function measureThreeD(
  bytes: Uint8Array,
  format: ThreeDFormat | undefined,
): { readonly vertexCount?: number; readonly triangleCount?: number } {
  const result: { vertexCount?: number; triangleCount?: number } = {};
  if (format === 'obj') {
    const text = decodeHead(bytes, bytes.byteLength);
    let verts = 0;
    let faces = 0;
    for (const line of text.split(/\r?\n/)) {
      if (line.startsWith('v ')) verts += 1;
      else if (line.startsWith('f ')) faces += 1;
    }
    if (verts > 0) result.vertexCount = verts;
    if (faces > 0) result.triangleCount = faces;
    return result;
  }
  if (format === 'stl') {
    if (looksLikeBinaryStl(bytes)) {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const triangles = view.getUint32(80, true);
      result.triangleCount = triangles;
      result.vertexCount = triangles * 3;
      return result;
    }
    const text = decodeHead(bytes, bytes.byteLength);
    const facets = text.match(/facet\s+normal/g)?.length ?? 0;
    if (facets > 0) {
      result.triangleCount = facets;
      result.vertexCount = facets * 3;
    }
    return result;
  }
  if (format === 'glb' || format === 'gltf') {
    const accessors = readGltfAccessors(bytes, format);
    if (accessors !== undefined) {
      // Position accessors of a mesh primitive carry one entry per vertex. Where
      // several primitives share an accessor the count is still an upper bound,
      // which is the honest thing to report for a fact read out of a file.
      let positions = 0;
      for (const accessor of accessors) {
        if (typeof accessor.count === 'number' && Number.isFinite(accessor.count)) positions += accessor.count;
      }
      if (positions > 0) result.vertexCount = positions;
    }
    return result;
  }
  return result;
}

/**
 * Read the accessor list out of a glTF document, embedded or standalone.
 * @param bytes - the content.
 * @param format - `glb` or `gltf`.
 * @returns the accessors, or `undefined` when the JSON cannot be read.
 */
function readGltfAccessors(
  bytes: Uint8Array,
  format: 'glb' | 'gltf',
): readonly { readonly count?: unknown }[] | undefined {
  const json = format === 'gltf' ? decodeHead(bytes, bytes.byteLength) : readGlbJsonChunk(bytes);
  if (json === undefined) return undefined;
  try {
    const parsed = JSON.parse(json) as unknown;
    if (parsed === null || typeof parsed !== 'object') return undefined;
    const accessors = (parsed as { accessors?: unknown }).accessors;
    return Array.isArray(accessors) ? (accessors as readonly { readonly count?: unknown }[]) : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Extract the JSON chunk from a binary glTF container.
 *
 * The header is 12 bytes (magic, version, total length), then chunks of
 * `[length, type, data]`. Chunk type `JSON` (0x4E4F534A) is the one wanted; its
 * data is padded with spaces, which `JSON.parse` tolerates.
 *
 * @param bytes - the container's bytes.
 * @returns the JSON text, or `undefined` when the container is malformed.
 */
function readGlbJsonChunk(bytes: Uint8Array): string | undefined {
  if (bytes.byteLength < 20) return undefined;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== 0x46546c67) return undefined; // 'glTF'
  let offset = 12;
  while (offset + 8 <= bytes.byteLength) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (length === 0 || start + length > bytes.byteLength) return undefined;
    if (type === 0x4e4f534a) {
      return new TextDecoder('utf-8', { fatal: false }).decode(bytes.subarray(start, start + length));
    }
    offset = start + length;
  }
  return undefined;
}
