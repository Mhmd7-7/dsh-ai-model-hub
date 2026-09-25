/**
 * Tests for the one rule that resolves a file a catalog names.
 *
 * `comfyui` reads `adapterConfig.workflowPath` and `three_d` reads
 * `adapterConfig.stepsPath`, and both go through `resolveAdapterPath()`. The rule
 * is small and has exactly one interesting property — the base is the catalog,
 * not the process — so it is pinned here, where a change to it fails with the
 * function's own name rather than as a confusing invocation failure three layers
 * up.
 *
 * The regression this guards is concrete: shipped data wrote
 * `config/workflows/…` while the code resolved against the catalog directory
 * inside `config/`, so the two composed into `config/config/workflows/…` and the
 * first image request failed with `ENOENT`.
 *
 * @module dsh-ai-model-hub/tests/paths
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import { isAbsolute, join } from 'node:path';

import { describeAdapterPath, resolveAdapterPath } from '../src/index.ts';

/**
 * An absolute directory to resolve against.
 *
 * Built from the working directory rather than from a leading separator, because
 * `\pkg\config` is *drive-relative* on Windows: `path.resolve` gives it the
 * current drive, which would make these assertions about the machine rather than
 * about the rule.
 */
const PKG = join(process.cwd(), 'aimh-pkg');

describe('catalog path resolution', () => {
  it('resolves a relative path against the catalog directory, never the working directory', () => {
    const catalogDir = join(PKG, 'config');
    const resolved = resolveAdapterPath('workflows/z-image-turbo.api.json', catalogDir);

    assert.equal(resolved.absolute, join(catalogDir, 'workflows', 'z-image-turbo.api.json'));
    assert.equal(resolved.base, catalogDir);
    assert.equal(resolved.origin, 'catalog');
    assert.ok(isAbsolute(resolved.absolute));

    // The composition that caused the bug: `config/` written into a catalog that
    // already lives in `config/` lands one directory too deep, and that is what
    // the shipped data used to say.
    assert.equal(
      resolveAdapterPath('config/workflows/x.json', catalogDir).absolute,
      join(catalogDir, 'config', 'workflows', 'x.json'),
    );
    // Which is a different file from the one the author meant.
    assert.notEqual(
      resolveAdapterPath('config/workflows/x.json', catalogDir).absolute,
      resolveAdapterPath('workflows/x.json', catalogDir).absolute,
    );
  });

  it('walks up out of the catalog directory when the path says so', () => {
    // The example catalog lives one level below the shipped templates, so the
    // same file is `../workflows/…` from there — the reason one convention cannot
    // be a single spelling for every catalog.
    const resolved = resolveAdapterPath('../workflows/x.json', join(PKG, 'config', 'examples'));
    assert.equal(resolved.absolute, join(PKG, 'config', 'workflows', 'x.json'));
  });

  it('uses an absolute path as written', () => {
    const absolute = join(PKG, 'elsewhere', 'template.api.json');
    const resolved = resolveAdapterPath(absolute, join(PKG, 'config'));

    assert.equal(resolved.absolute, absolute);
    assert.equal(resolved.base, undefined);
    assert.equal(resolved.origin, 'absolute');
  });

  it('falls back to the working directory only when there is no catalog file', () => {
    // An in-memory catalog (`ModelHub.fromConfig`) has no file to be relative to,
    // and saying so is the point: the fallback is reported rather than passing for
    // a catalog.
    const resolved = resolveAdapterPath('workflows/x.json', undefined);

    assert.equal(resolved.absolute, join(process.cwd(), 'workflows', 'x.json'));
    assert.equal(resolved.base, process.cwd());
    assert.equal(resolved.origin, 'working-directory');
  });

  it('describes both the absolute path it tried and the base it used', () => {
    const catalogDir = join(PKG, 'config');
    const described = describeAdapterPath(resolveAdapterPath('config/workflows/x.json', catalogDir));

    assert.ok(described.includes(join(catalogDir, 'config', 'workflows', 'x.json')), described);
    assert.ok(described.includes(catalogDir), described);
    // The wording is shared by both adapters, so their failures stay comparable.
    assert.match(described, /resolved against the catalog directory/);

    assert.match(describeAdapterPath(resolveAdapterPath(join(PKG, 'x.json'), undefined)), /absolute path/);
    assert.match(describeAdapterPath(resolveAdapterPath('x.json', undefined)), /working directory/);
  });
});
