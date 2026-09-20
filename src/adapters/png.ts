/**
 * A tiny, dependency-free PNG writer.
 *
 * The mock image adapter must produce a file a human can actually open, or the
 * vertical slice proves nothing: "it returned an artifact" is not evidence that
 * the artifact pipeline works. Composing the bytes here keeps the hub free of an
 * image library while still emitting a valid PNG.
 *
 * @module dsh-ai-model-hub/adapters/png
 */

import { deflateSync } from 'node:zlib';

/** The eight-byte PNG signature. */
const PNG_SIGNATURE = Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/** Build the CRC-32 lookup table once. */
const CRC_TABLE: Uint32Array = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) === 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
})();

/**
 * Compute the CRC-32 of a byte range, as PNG requires for each chunk.
 * @param bytes - the chunk type plus data.
 * @returns the unsigned 32-bit checksum.
 */
function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc = (CRC_TABLE[(crc ^ byte) & 0xff] ?? 0) ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Assemble one PNG chunk: length, type, data, CRC.
 * @param type - the four-byte chunk type, e.g. `IHDR`.
 * @param data - the chunk payload.
 * @returns the complete chunk bytes.
 */
function chunk(type: string, data: Uint8Array): Uint8Array {
  const typeBytes = new TextEncoder().encode(type);
  const body = new Uint8Array(typeBytes.length + data.length);
  body.set(typeBytes, 0);
  body.set(data, typeBytes.length);

  const out = new Uint8Array(4 + body.length + 4);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  out.set(body, 4);
  view.setUint32(4 + body.length, crc32(body));
  return out;
}

/** A minimal RGB colour. */
export interface Rgb {
  readonly r: number;
  readonly g: number;
  readonly b: number;
}

/**
 * Deterministically derive a colour from a string.
 *
 * The test double's models are fixtures, so their output should be stable across
 * runs: the same prompt must always yield the same picture. Hashing the seed
 * gives that property without any global state.
 *
 * @param seed - the string to colourise, typically the prompt.
 * @returns an RGB triple.
 */
export function colorFromSeed(seed: string): Rgb {
  let hash = 0x811c9dc5;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return {
    r: 40 + (hash & 0x7f),
    g: 40 + ((hash >>> 8) & 0x7f),
    b: 60 + ((hash >>> 16) & 0x7f),
  };
}

/**
 * Render a deterministic gradient-with-grid image and encode it as a PNG.
 *
 * The visual is intentionally synthetic — this is a fixture, and it must be
 * obvious at a glance that no real model produced it. The grid makes scaling and
 * aspect-ratio changes visible, which is exactly what a pipeline test needs to
 * see.
 *
 * @param width - image width in pixels; must be positive.
 * @param height - image height in pixels; must be positive.
 * @param seed - the string that determines the colour, typically the prompt.
 * @param options - grid and accent controls.
 * @returns the encoded PNG bytes.
 * @throws RangeError when the dimensions are not positive integers.
 */
export function renderMockPng(
  width: number,
  height: number,
  seed: string,
  options: { readonly gridStep?: number; readonly accent?: Rgb } = {},
): Uint8Array {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`renderMockPng requires positive integer dimensions, got ${width}x${height}`);
  }

  const base = colorFromSeed(seed);
  const accent = options.accent ?? { r: 235, g: 240, b: 250 };
  const gridStep = options.gridStep ?? 64;

  // One filter byte (0 = None) per scanline, then RGB triples.
  const stride = width * 3 + 1;
  const raw = new Uint8Array(stride * height);

  for (let y = 0; y < height; y += 1) {
    const rowStart = y * stride;
    raw[rowStart] = 0;
    for (let x = 0; x < width; x += 1) {
      const offset = rowStart + 1 + x * 3;
      const gx = x / Math.max(1, width - 1);
      const gy = y / Math.max(1, height - 1);
      const onGrid = x % gridStep === 0 || y % gridStep === 0;

      if (onGrid) {
        raw[offset] = accent.r;
        raw[offset + 1] = accent.g;
        raw[offset + 2] = accent.b;
        continue;
      }

      // Horizontal gradient toward the accent, vertically shaded.
      const shade = 0.35 + 0.65 * (1 - gy);
      raw[offset] = clampByte(Math.round((base.r + (accent.r - base.r) * gx * 0.45) * shade));
      raw[offset + 1] = clampByte(Math.round((base.g + (accent.g - base.g) * gx * 0.45) * shade));
      raw[offset + 2] = clampByte(Math.round((base.b + (accent.b - base.b) * gx * 0.45) * shade));
    }
  }

  const ihdr = new Uint8Array(13);
  const ihdrView = new DataView(ihdr.buffer);
  ihdrView.setUint32(0, width);
  ihdrView.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type: truecolour RGB
  ihdr[10] = 0; // compression: deflate
  ihdr[11] = 0; // filter method
  ihdr[12] = 0; // interlace: none

  const parts = [
    PNG_SIGNATURE,
    chunk('IHDR', ihdr),
    chunk('IDAT', new Uint8Array(deflateSync(raw, { level: 6 }))),
    chunk('IEND', new Uint8Array(0)),
  ];

  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const png = new Uint8Array(total);
  let cursor = 0;
  for (const part of parts) {
    png.set(part, cursor);
    cursor += part.length;
  }
  return png;
}

/**
 * Clamp a numeric channel value into the 0–255 byte range.
 * @param value - the candidate channel value.
 * @returns the clamped byte.
 */
function clampByte(value: number): number {
  if (value < 0) return 0;
  if (value > 255) return 255;
  return value;
}

/**
 * Encode an ASCII/UTF-8 STL solid containing one deterministic box.
 *
 * STL is the lowest common denominator for 3D meshes and is text-encodable, so
 * the mock 3D adapter can emit a file any viewer or downstream pipeline reads
 * without the hub depending on a mesh library. The box is scaled by the seed so
 * different prompts produce visibly different geometry.
 *
 * @param seed - the string the geometry is derived from, typically the prompt.
 * @param options - scale controls.
 * @returns the encoded STL document as UTF-8 text.
 */
export function renderMockStl(seed: string, options: { readonly scale?: number } = {}): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  const scale = options.scale ?? 1;
  const sx = (0.5 + ((hash >>> 0) & 0x7) / 10) * scale;
  const sy = (0.5 + ((hash >>> 3) & 0x7) / 10) * scale;
  const sz = (0.5 + ((hash >>> 6) & 0x7) / 10) * scale;

  const v: readonly (readonly [number, number, number])[] = [
    [0, 0, 0],
    [sx, 0, 0],
    [sx, sy, 0],
    [0, sy, 0],
    [0, 0, sz],
    [sx, 0, sz],
    [sx, sy, sz],
    [0, sy, sz],
  ];
  // Each face is two triangles wound counter-clockwise when seen from outside.
  const faces: readonly (readonly [number, number, number, number])[] = [
    [0, 3, 2, 1], // bottom, normal -Z
    [4, 5, 6, 7], // top, normal +Z
    [0, 1, 5, 4], // front, normal -Y
    [2, 3, 7, 6], // back, normal +Y
    [1, 2, 6, 5], // right, normal +X
    [0, 4, 7, 3], // left, normal -X
  ];
  const normals: readonly (readonly [number, number, number])[] = [
    [0, 0, -1],
    [0, 0, 1],
    [0, -1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [-1, 0, 0],
  ];

  const lines: string[] = [`solid ai_model_hub_mock_${seed.replace(/[^\w-]/g, '_').slice(0, 40)}`];
  faces.forEach((face, faceIndex) => {
    const normal = normals[faceIndex];
    if (normal === undefined) return;
    const triangles: readonly (readonly [number, number, number])[] = [
      [face[0], face[1], face[2]],
      [face[0], face[2], face[3]],
    ];
    for (const triangle of triangles) {
      lines.push(`  facet normal ${normal[0]} ${normal[1]} ${normal[2]}`);
      lines.push('    outer loop');
      for (const vertexIndex of triangle) {
        const vertex = v[vertexIndex];
        if (vertex === undefined) continue;
        lines.push(`      vertex ${vertex[0]} ${vertex[1]} ${vertex[2]}`);
      }
      lines.push('    endloop');
      lines.push('  endfacet');
    }
  });
  lines.push(`endsolid ai_model_hub_mock_${seed.replace(/[^\w-]/g, '_').slice(0, 40)}`);
  return `${lines.join('\n')}\n`;
}

/** Vertex count of {@link renderMockStl}: 6 quad faces, 2 triangles each, 3 vertices each. */
export const MOCK_STL_VERTEX_COUNT = 36;
