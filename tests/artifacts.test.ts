/**
 * Tests for the artifact system.
 *
 * The interesting properties here are structural: that produced content is a
 * *real* file of the declared format, that the index survives a reload, that a
 * concurrent write cannot corrupt it, and that the store refuses to expose a path
 * outside its root even if the on-disk index is edited.
 *
 * @module dsh-ai-model-hub/tests/artifacts.test
 */

import { strict as assert } from 'node:assert';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, it } from 'node:test';
import { inflateSync } from 'node:zlib';

import { LocalArtifactStore, fileUriToPath, pathToFileUri, readArtifactConventions } from '../src/index.ts';
import { MOCK_STL_VERTEX_COUNT, renderMockPng, renderMockStl } from '../src/index.ts';

/** Build a temporary store and hand it to a test, cleaning up afterwards. */
async function withStore(run: (store: LocalArtifactStore, root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-artifacts-'));
  try {
    await run(new LocalArtifactStore({ root }), root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

/** One decoded PNG chunk. */
interface PngChunk {
  readonly type: string;
  readonly data: Uint8Array;
  readonly declaredCrc: number;
  readonly computedCrc: number;
}

/**
 * Walk a PNG's chunk list, recomputing every CRC-32.
 *
 * This is what makes "the mock produced a valid PNG" a claim with evidence: a
 * correct signature and IHDR are cheap to fake, whereas every chunk's CRC being
 * right means the bytes were laid out correctly.
 *
 * @param bytes - the encoded PNG.
 * @returns the chunks with their declared and computed checksums.
 */
function parsePngChunks(bytes: Uint8Array): PngChunk[] {
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let index = 0; index < signature.length; index += 1) {
    assert.equal(bytes[index], signature[index], 'PNG signature byte mismatch');
  }

  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) === 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  const crc32 = (input: Uint8Array): number => {
    let crc = 0xffffffff;
    for (const byte of input) crc = (table[(crc ^ byte) & 0xff] ?? 0) ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
  };

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const chunks: PngChunk[] = [];
  let cursor = 8;
  while (cursor < bytes.length) {
    const length = view.getUint32(cursor);
    const type = String.fromCharCode(...bytes.subarray(cursor + 4, cursor + 8));
    const data = bytes.subarray(cursor + 8, cursor + 8 + length);
    const declaredCrc = view.getUint32(cursor + 8 + length);
    const body = bytes.subarray(cursor + 4, cursor + 8 + length);
    chunks.push({ type, data, declaredCrc, computedCrc: crc32(body) });
    cursor += 12 + length;
  }
  return chunks;
}

describe('file URI conversion', () => {
  it('round-trips Windows paths', () => {
    const uri = pathToFileUri('C:\\Users\\test\\artifacts\\a.png');
    assert.equal(uri, 'file:///C:/Users/test/artifacts/a.png');
    assert.equal(fileUriToPath(uri), 'C:\\Users\\test\\artifacts\\a.png');
  });

  it('round-trips POSIX paths', { skip: process.platform === 'win32' }, () => {
    const uri = pathToFileUri('/home/test/artifacts/a.png');
    assert.equal(uri, 'file:///home/test/artifacts/a.png');
    assert.equal(fileUriToPath(uri), '/home/test/artifacts/a.png');
  });

  it('normalizes separators for the running platform', () => {
    // The stored URI is always forward-slashed; the returned path always uses the
    // platform separator, which is what another local process expects.
    const absolute = resolve('tmp', 'artifacts', 'a.png');
    const uri = pathToFileUri(absolute);
    assert.ok(!uri.includes('\\'), `URI must not contain a backslash: ${uri}`);
    assert.equal(fileUriToPath(uri), absolute);

    // A POSIX-style URI written on one platform is still interpreted sensibly.
    if (process.platform === 'win32') {
      assert.equal(fileUriToPath('file:///C:/tmp/a.png'), 'C:\\tmp\\a.png');
    }
  });

  it('rejects a non-file URI', () => {
    assert.equal(fileUriToPath('https://example.com/a.png'), undefined);
  });
});

describe('LocalArtifactStore', () => {
  it('stores bytes and reports a durable reference', async () => {
    await withStore(async (store) => {
      const artifact = await store.put({
        type: 'image',
        bytes: new Uint8Array([1, 2, 3, 4]),
        mimeType: 'image/png',
        label: 'Test Image',
        producerModelId: 'mock_image_model',
        metadata: { width: 1, height: 1 },
      });

      assert.equal(artifact.type, 'image');
      assert.equal(artifact.mimeType, 'image/png');
      assert.equal(artifact.byteLength, 4);
      assert.equal(artifact.label, 'Test Image');
      assert.equal(artifact.producerModelId, 'mock_image_model');
      assert.ok(artifact.id.startsWith('image_'), `unexpected id ${artifact.id}`);
      assert.ok(artifact.createdAt > 0);

      const reread = await store.get(artifact.id);
      assert.deepEqual(reread, artifact);

      const { bytes } = await store.read(artifact.id);
      assert.deepEqual([...bytes], [1, 2, 3, 4]);
    });
  });

  it('stores text and picks a matching extension', async () => {
    await withStore(async (store) => {
      const artifact = await store.put({ type: 'text', text: 'hello world' });
      assert.ok(artifact.uri.endsWith('.txt'), artifact.uri);
      const { path } = await store.resolvePath(artifact.id);
      assert.equal(await readFile(path, 'utf8'), 'hello world');
      assert.equal(artifact.mimeType, 'text/plain');
    });
  });

  it('refuses a write with both bytes and text, or neither', async () => {
    await withStore(async (store) => {
      await assert.rejects(
        () => store.put({ type: 'text', text: 'a', bytes: new Uint8Array([1]) }),
        (error: unknown) => (error as { code?: string }).code === 'ARTIFACT_ERROR',
      );
      await assert.rejects(
        () => store.put({ type: 'text' }),
        (error: unknown) => (error as { code?: string }).code === 'ARTIFACT_ERROR',
      );
    });
  });

  it('resolves a missing artifact to ARTIFACT_ERROR', async () => {
    await withStore(async (store) => {
      await assert.rejects(
        () => store.read('nope'),
        (error: unknown) => (error as { code?: string }).code === 'ARTIFACT_ERROR',
      );
    });
  });

  it('survives a reload from disk', async () => {
    await withStore(async (store, root) => {
      const artifact = await store.put({ type: 'text', text: 'persisted' });
      const reopened = new LocalArtifactStore({ root });
      const found = await reopened.get(artifact.id);
      assert.equal(found?.id, artifact.id);
      const { bytes } = await reopened.read(artifact.id);
      assert.equal(new TextDecoder().decode(bytes), 'persisted');
    });
  });

  it('serializes concurrent writes without losing index entries', async () => {
    await withStore(async (store, root) => {
      const writes = Array.from({ length: 25 }, (_unused, index) =>
        store.put({ type: 'text', text: `payload ${index}`, label: `entry ${index}` }),
      );
      const artifacts = await Promise.all(writes);
      assert.equal(new Set(artifacts.map((artifact) => artifact.id)).size, 25);

      const reopened = new LocalArtifactStore({ root });
      assert.equal(await reopened.size(), 25);
      for (const artifact of artifacts) {
        assert.ok(await reopened.get(artifact.id), `${artifact.id} missing after reload`);
      }
    });
  });

  it('lists newest first and honours the limit', async () => {
    await withStore(async (store) => {
      const first = await store.put({ type: 'text', text: 'one' });
      await new Promise((resolve) => setTimeout(resolve, 5));
      const second = await store.put({ type: 'text', text: 'two' });
      const listed = await store.list(10);
      assert.equal(listed.length, 2);
      assert.equal(listed[0]?.id, second.id);
      assert.equal(listed[1]?.id, first.id);
      assert.equal((await store.list(1)).length, 1);
    });
  });

  it('refuses to expose content outside its root', async () => {
    await withStore(async (store, root) => {
      const artifact = await store.put({ type: 'text', text: 'legit' });
      // Simulate an edited (or hostile) on-disk index pointing at an arbitrary file.
      const outside = join(root, '..', 'outside.txt');
      await writeFile(outside, 'secret', 'utf8');
      try {
        const index = JSON.parse(await readFile(store.indexPath, 'utf8')) as {
          artifacts: { id: string; uri: string }[];
        };
        const entry = index.artifacts.find((candidate) => candidate.id === artifact.id);
        if (entry !== undefined) entry.uri = pathToFileUri(outside);
        await writeFile(store.indexPath, JSON.stringify(index), 'utf8');

        const reopened = new LocalArtifactStore({ root });
        await assert.rejects(
          () => reopened.read(artifact.id),
          (error: unknown) => (error as { code?: string }).code === 'ARTIFACT_ERROR',
        );
      } finally {
        await rm(outside, { force: true });
      }
    });
  });

  it('reports orphans and unreferenced files', async () => {
    await withStore(async (store) => {
      const kept = await store.put({ type: 'text', text: 'kept' });
      const removed = await store.put({ type: 'text', text: 'removed' });
      const { path } = await store.resolvePath(removed.id);
      await rm(path);

      assert.deepEqual(await store.findOrphans(), [removed.id]);
      assert.deepEqual(await store.findOrphans().then((ids) => ids.includes(kept.id)), false);

      await writeFile(join(store.contentDir, 'stray.txt'), 'not indexed', 'utf8');
      const unreferenced = await store.findUnreferencedFiles();
      assert.ok(unreferenced.some((file) => file.endsWith('stray.txt')));
    });
  });

  it('deletes an artifact and its content', async () => {
    await withStore(async (store) => {
      const artifact = await store.put({ type: 'text', text: 'bye' });
      const { path } = await store.resolvePath(artifact.id);
      assert.equal(await store.delete(artifact.id), true);
      assert.equal(await store.get(artifact.id), undefined);
      await assert.rejects(() => stat(path));
      assert.equal(await store.delete(artifact.id), false);
    });
  });
});

describe('mock fixture encoders', () => {
  it('produces a structurally valid PNG', () => {
    const png = renderMockPng(64, 32, 'a test prompt');
    const chunks = parsePngChunks(png);

    assert.deepEqual(
      chunks.map((chunk) => chunk.type),
      ['IHDR', 'IDAT', 'IEND'],
    );
    for (const chunk of chunks) {
      assert.equal(
        chunk.computedCrc,
        chunk.declaredCrc,
        `${chunk.type} chunk has a bad CRC (declared ${chunk.declaredCrc}, computed ${chunk.computedCrc})`,
      );
    }

    const ihdr = chunks[0];
    assert.ok(ihdr);
    const view = new DataView(ihdr.data.buffer, ihdr.data.byteOffset, ihdr.data.byteLength);
    assert.equal(view.getUint32(0), 64, 'IHDR width');
    assert.equal(view.getUint32(4), 32, 'IHDR height');
    assert.equal(ihdr.data[8], 8, 'bit depth');
    assert.equal(ihdr.data[9], 2, 'colour type must be truecolour RGB');

    // The IDAT payload must inflate to exactly one filter byte plus RGB per pixel.
    const idat = chunks[1];
    assert.ok(idat);
    const raw = inflateSync(idat.data);
    assert.equal(raw.length, (64 * 3 + 1) * 32, 'inflated IDAT length must match the scanline layout');
  });

  it('is deterministic for the same prompt and different for another', () => {
    const one = renderMockPng(16, 16, 'same prompt');
    const two = renderMockPng(16, 16, 'same prompt');
    const other = renderMockPng(16, 16, 'different prompt');
    assert.deepEqual([...one], [...two]);
    assert.notDeepEqual([...one], [...other]);
  });

  it('rejects non-positive dimensions', () => {
    assert.throws(() => renderMockPng(0, 10, 'x'), RangeError);
    assert.throws(() => renderMockPng(10, -1, 'x'), RangeError);
    assert.throws(() => renderMockPng(10.5, 10, 'x'), RangeError);
  });

  it('produces a loadable ASCII STL', () => {
    const stl = renderMockStl('a low-poly spaceship');
    const lines = stl.trim().split('\n');
    assert.ok(lines[0]?.startsWith('solid '), 'must open with a solid name');
    assert.ok(lines[lines.length - 1]?.startsWith('endsolid'), 'must close with endsolid');

    const facets = lines.filter((line) => line.trim().startsWith('facet normal')).length;
    const vertices = lines.filter((line) => line.trim().startsWith('vertex')).length;
    assert.equal(facets, 12, 'a box is 12 triangles');
    assert.equal(vertices, MOCK_STL_VERTEX_COUNT);

    // Every facet must declare the same normal for its three vertices, and the
    // file must balance its open/close markers.
    assert.equal(lines.filter((line) => line.trim() === 'outer loop').length, 12);
    assert.equal(lines.filter((line) => line.trim() === 'endloop').length, 12);
    assert.equal(lines.filter((line) => line.trim() === 'endfacet').length, 12);
  });

  it('derives different geometry from different seeds', () => {
    assert.notEqual(renderMockStl('spaceship'), renderMockStl('castle'));
  });
});

describe('readArtifactConventions', () => {
  it('reads the conventional fields and ignores malformed ones', () => {
    const conventions = readArtifactConventions({
      id: 'a',
      type: 'image',
      uri: 'file:///a.png',
      createdAt: 0,
      metadata: { width: 640, height: 'tall', vertexCount: 36, format: 'png', extra: true },
    });
    assert.equal(conventions.width, 640);
    assert.equal(conventions.height, undefined, 'a non-numeric height must be ignored');
    assert.equal(conventions.vertexCount, 36);
    assert.equal(conventions.format, 'png');
  });
});
