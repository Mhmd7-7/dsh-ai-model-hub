/**
 * Tests for the manifest gate: the check that decides whether DSH loads this
 * plugin at all.
 *
 * ## The bug this file exists for
 *
 * DSH reads a plugin's `peerDependencies` and refuses the row when a
 * `@deepseek-ai/dsh*` range does not match the running harness — *before* the
 * plugin's module is imported, so before `apply` runs and before any of the
 * runtime defensiveness in `compat.ts` can help.
 *
 * The plugin used to declare `"@deepseek-ai/dsh-tools": "^0.1.5-rc.2"`. That looks
 * permissive and is not: **for a `0.x` version, caret pins the MINOR**, so
 * `^0.1.5-rc.2` means `>=0.1.5-rc.2 <0.2.0` and rejects every `0.2.x` harness.
 * The entire `compat.ts` layer would have been dead code on a `0.2.0-rc.2` host,
 * because the plugin would never have been loaded to run it.
 *
 * ## What is asserted
 *
 * 1. The ranges this project actually ships accept every DSH version known to be
 *    in the supported line, and reject the next major line.
 * 2. The bug class itself: the old caret form is shown rejecting `0.2.0-rc.2`,
 *    so a future edit that reintroduces it fails a test instead of a boot.
 * 3. The real gate, when the harness that implements it is resolvable — the
 *    strongest possible check, and one that is skipped rather than faked when the
 *    package is not installed.
 *
 * ## On the local range matcher
 *
 * `matchesRange` is a small, deliberate subset of semver: the operators and clause
 * forms these manifests use, with npm's `0.x` caret rule and the standard
 * prerelease rule. It is not a general semver implementation and does not try to
 * be. It is a test double for one specific gate, written out so the test states
 * the semantics it relies on instead of inheriting them from a dependency — and
 * assertion (3) is what keeps the double honest.
 *
 * @module dsh-ai-model-hub/tests/peer-range.test
 */

import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

/** This package's root, derived from this file's own location. */
const PACKAGE_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

/** The manifest as it ships. */
const MANIFEST = JSON.parse(readFileSync(join(PACKAGE_ROOT, 'package.json'), 'utf8')) as {
  readonly name: string;
  readonly version: string;
  readonly peerDependencies?: Readonly<Record<string, string>>;
  readonly peerDependenciesMeta?: Readonly<Record<string, { readonly optional?: boolean }>>;
};

/**
 * The package names DSH's compatibility gate inspects.
 *
 * The gate looks only at peers named `@deepseek-ai/dsh` or matching
 * `@deepseek-ai/dsh-*`. Everything else in `peerDependencies` — cordis,
 * schemastery, react — is invisible to it, which is why a wrong range there is a
 * runtime problem rather than a load-time one.
 */
const GATED_PREFIXES = ['@deepseek-ai/dsh'];

/**
 * One parsed version, with its prerelease identifiers.
 *
 * @param text - a version string such as `0.2.0-rc.2`.
 * @returns the major, minor, patch, and prerelease parts.
 */
function parseVersion(text: string): { major: number; minor: number; patch: number; pre: (string | number)[] } {
  const match = /^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$/.exec(text.trim().replace(/^v/, ''));
  if (match === null) throw new Error(`not a version: ${text}`);
  const pre = (match[4] ?? '').split('.').filter((part) => part.length > 0);
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    pre: pre.map((part) => (/^\d+$/.test(part) ? Number(part) : part)),
  };
}

/**
 * Compare two parsed versions the way semver does, including prerelease ordering.
 *
 * @param a - the left version.
 * @param b - the right version.
 * @returns negative, zero, or positive.
 */
function compare(a: ReturnType<typeof parseVersion>, b: ReturnType<typeof parseVersion>): number {
  for (const key of ['major', 'minor', 'patch'] as const) {
    if (a[key] !== b[key]) return a[key] < b[key] ? -1 : 1;
  }
  // A version with a prerelease sorts below the same version without one.
  if (a.pre.length === 0 && b.pre.length === 0) return 0;
  if (a.pre.length === 0) return 1;
  if (b.pre.length === 0) return -1;
  for (let index = 0; index < Math.max(a.pre.length, b.pre.length); index += 1) {
    const left = a.pre[index];
    const right = b.pre[index];
    if (left === undefined) return -1;
    if (right === undefined) return 1;
    if (left === right) continue;
    if (typeof left === 'number' && typeof right === 'string') return -1;
    if (typeof left === 'string' && typeof right === 'number') return 1;
    return left < right ? -1 : 1;
  }
  return 0;
}

/**
 * Whether a version satisfies one comparator, e.g. `>=0.1.5-rc.2`.
 *
 * @param version - the version under test.
 * @param comparator - the comparator text.
 * @returns whether it matches.
 */
function matchesComparator(version: string, comparator: string): boolean {
  const match = /^(>=|<=|>|<|=|\^|~)?\s*(.+)$/.exec(comparator.trim());
  if (match === null) throw new Error(`not a comparator: ${comparator}`);
  const operator = match[1] ?? '=';
  const target = parseVersion(match[2] as string);
  const actual = parseVersion(version);
  const ordering = compare(actual, target);

  switch (operator) {
    case '>=':
      return ordering >= 0;
    case '<=':
      return ordering <= 0;
    case '>':
      return ordering > 0;
    case '<':
      return ordering < 0;
    case '=':
      return ordering === 0;
    case '~':
      // `~0.1.5` allows patch-level changes only.
      return ordering >= 0 && actual.major === target.major && actual.minor === target.minor;
    case '^': {
      if (ordering < 0) return false;
      if (target.major > 0) return actual.major === target.major;
      if (target.minor > 0) return actual.major === 0 && actual.minor === target.minor;
      // `^0.0.x` pins the patch, because there is no smaller component left.
      return actual.major === 0 && actual.minor === 0 && actual.patch === target.patch;
    }
    default:
      throw new Error(`unsupported operator: ${operator}`);
  }
}

/**
 * Whether a version satisfies a whole range, including `||` clauses and spaces.
 *
 * The prerelease rule lives here rather than in {@link matchesComparator}, and
 * that placement is the whole subtlety. npm applies it **once per clause**: a
 * clause is skipped entirely when the version is a prerelease and no comparator
 * *in that clause* carries one. It is not a per-comparator filter.
 *
 * The difference is visible in the range this project actually ships. Under a
 * per-comparator reading, `0.1.5-rc.2` would fail `>=0.1.5-rc.2 <0.3.0` — the
 * upper bound has no prerelease and a different version tuple — and the manifest
 * would look as though it refused the very version the package is developed
 * against. Running the harness's own gate settles it the other way: the clause
 * contains a prerelease comparator, so the rule does not apply, and both bounds
 * are evaluated normally.
 *
 * @param version - the version under test.
 * @param range - the range text.
 * @returns whether it matches.
 */
function matchesRange(version: string, range: string): boolean {
  const actual = parseVersion(version);
  const isPrerelease = actual.pre.length > 0;

  return range
    .split('||')
    .some((clause) => {
      const comparators = clause
        .trim()
        .split(/\s+/)
        .filter((part) => part.length > 0);

      if (isPrerelease && !comparators.some((comparator) => comparatorCarriesPrerelease(comparator))) {
        return false;
      }
      return comparators.every((comparator) => matchesComparator(version, comparator));
    });
}

/**
 * Whether one comparator's own version text carries a prerelease.
 *
 * Shorthand operators are expanded first, because npm expands them before the
 * rule is applied: `^0.1.5-rc.2` becomes `>=0.1.5-rc.2 <0.2.0-0`, so the clause
 * carries a prerelease even when a reader might not think of a caret as doing so.
 *
 * @param comparator - the raw comparator text.
 * @returns true when the expanded clause contains a prerelease.
 */
function comparatorCarriesPrerelease(comparator: string): boolean {
  const match = /^(>=|<=|>|<|=|\^|~)?\s*(.+)$/.exec(comparator.trim());
  if (match === null) return false;
  const operator = match[1] ?? '=';
  const text = (match[2] as string).trim();
  // `-0` is npm's own marker for "the lowest prerelease of this version", which is
  // exactly what an upper bound needs in order to include prereleases below it.
  const expand = (part: string): string[] => {
    switch (operator) {
      case '^': {
        const v = parseVersion(part);
        const upper = v.major > 0
          ? `${v.major + 1}.0.0-0`
          : v.minor > 0
            ? `0.${v.minor + 1}.0-0`
            : `0.0.${v.patch + 1}-0`;
        return [`>=${part}`, `<${upper}`];
      }
      case '~': {
        const v = parseVersion(part);
        return [`>=${part}`, `<${v.major}.${v.minor + 1}.0-0`];
      }
      default:
        return [text];
    }
  };
  return expand(text).some((part) => /-/.test(part.replace(/^[<>=]+/, '')));
}

/** The gated peers this manifest declares. */
function gatedPeers(): { readonly name: string; readonly range: string }[] {
  const peers = MANIFEST.peerDependencies ?? {};
  return Object.entries(peers)
    .filter(([name]) => GATED_PREFIXES.some((prefix) => name === prefix || name.startsWith(`${prefix}-`)))
    .map(([name, range]) => ({ name, range }));
}

describe('the local range matcher itself', () => {
  // A test double is only worth trusting if its own semantics are tested. These
  // are the npm behaviours the assertions below depend on.
  it('applies npm\'s 0.x caret rule, which is the whole point of this file', () => {
    assert.equal(matchesRange('0.2.0-rc.2', '^0.1.5-rc.2'), false, '0.x caret pins the minor');
    assert.equal(matchesRange('0.1.9', '^0.1.5-rc.2'), true, 'within 0.1 it is allowed to move forward');
    assert.equal(matchesRange('0.2.0', '^0.1.5'), false, 'and 0.2 is outside it');
  });

  it('treats a prerelease as below its release, and out of a range without one', () => {
    assert.equal(matchesRange('0.2.0-rc.2', '>=0.2.0'), false, 'a prerelease sorts below the release');
    assert.equal(matchesRange('0.2.0', '>=0.2.0-rc.2'), true);
    // The rule has to let the range that *names* a prerelease accept it, or a
    // manifest could not support the release candidate it was built against — which
    // is exactly the trap the old caret form fell into.
    assert.equal(matchesRange('0.1.5-rc.2', '>=0.1.5-rc.2'), true, 'a comparator naming the same tuple qualifies it');
    assert.equal(matchesRange('0.2.0-rc.2', '>=0.1.5-rc.2 <0.3.0'), true);
    // And the rule still has to *refuse*, or a prerelease build of a version the
    // manifest never mentions could be loaded onto the harness.
    assert.equal(matchesRange('0.3.0-beta.1', '>=0.2.0'), false, 'a prerelease of a higher version is still out');
    assert.equal(matchesRange('0.2.1-rc.1', '>=0.2.0'), false, 'and a prerelease above the floor is out too');
  });

  it('handles the other operators and clause forms the manifests use', () => {
    assert.equal(matchesRange('0.1.5-rc.2', '~0.1.5-rc.2'), true);
    assert.equal(matchesRange('0.1.9', '~0.1.5'), true, '~0.1.5 allows patch-level changes, and 0.1.9 is one');
    assert.equal(matchesRange('3.18.4', '^3.18.2'), true);
    assert.equal(matchesRange('0.2.0-rc.2', '^0.1.5-rc.2 || ^0.2.0-rc.1'), true, 'an OR clause is satisfied by either side');
    assert.equal(matchesRange('0.2.5', '>=0.1.5-rc.2 <0.3.0'), true, 'a space-joined range is a conjunction');
    assert.equal(matchesRange('0.3.0', '>=0.1.5-rc.2 <0.3.0'), false);
  });
});

describe('the peers DSH gates this plugin on', () => {
  it('declares at least one, or the gate would not be exercised at all', () => {
    assert.ok(gatedPeers().length > 0, 'the plugin declares gated peers');
  });

  it('accepts every DSH version in the supported line, on both sides of the old cliff', () => {
    // The two versions this project has actually been developed and tested
    // against: the one in its own node_modules and the one installed as the host.
    const supported = ['0.1.5-rc.2', '0.1.6-alpha.2', '0.2.0-rc.1', '0.2.0-rc.2', '0.2.1-alpha.2'];
    for (const peer of gatedPeers()) {
      for (const version of supported) {
        assert.equal(
          matchesRange(version, peer.range),
          true,
          `${peer.name} ${peer.range} accepts ${version}`,
        );
      }
    }
  });

  it('refuses the next major line, so the acceptance is a range and not a wildcard', () => {
    for (const peer of gatedPeers()) {
      assert.equal(matchesRange('0.3.0', peer.range), false, `${peer.name} ${peer.range} refuses 0.3.0`);
      assert.equal(matchesRange('1.0.0', peer.range), false, `${peer.name} ${peer.range} refuses 1.0.0`);
    }
  });

  it('is not written in the caret form that caused the original refusal', () => {
    // `^0.1.5-rc.2` is the exact string that made a 0.2.0-rc.2 harness reject the
    // plugin. It must not come back, in any peer the gate reads.
    for (const peer of gatedPeers()) {
      assert.equal(
        matchesRange('0.2.0-rc.2', peer.range),
        true,
        `${peer.name} must not pin the 0.x minor: ${peer.range}`,
      );
    }
  });

  it('marks the peers a composition may legitimately omit as optional', () => {
    // `dsh-tools` is deliberately NOT optional: without it there are no tools, and
    // the plugin says so. The rest are read through `ctx` or are type-only, so a
    // profile without them must not be told its dependencies are unmet.
    const meta = MANIFEST.peerDependenciesMeta ?? {};
    for (const name of ['@deepseek-ai/cordis', '@deepseek-ai/dsh-system-prompt', '@deepseek-ai/schemastery', 'react']) {
      assert.equal(meta[name]?.optional, true, `${name} is optional`);
    }
    assert.equal(meta['@deepseek-ai/dsh-tools']?.optional, undefined, 'dsh-tools stays required');
  });
});

/**
 * The harness version the manifest is checked against.
 *
 * Passed explicitly rather than left to default. `evaluatePluginCompatibility`
 * defaults to *its own* package version, which means the same assertion would
 * answer differently depending on which copy of `dsh-app-boot` happened to be
 * resolvable — and a test whose meaning moves with the environment is not a test.
 * `0.2.0-rc.2` is the harness this project is actually run on, and it is the
 * version the old caret ranges were refusing.
 */
const RUNTIME_UNDER_TEST = '0.2.0-rc.2';

describe('the harness\'s own compatibility gate, when it is reachable', () => {
  /**
   * Resolve the harness package that implements the gate.
   *
   * Tried from this package first — which is what a checkout or a CI run has —
   * and then from the installed DSH host, which is what a profile plugin is
   * loaded by. Either is the genuine article; neither is simulated.
   *
   * @returns the module, or `undefined` when it is not installed here.
   */
  async function loadAppBoot(): Promise<
    | {
        evaluatePluginCompatibility?: (
          manifest: object,
          exemptions?: Readonly<Record<string, readonly string[]>>,
          runtimeVersion?: string,
        ) => unknown;
      }
    | undefined
  > {
    const roots = [import.meta.url];
    const home = process.env['DSH_HOME'];
    if (home !== undefined) {
      roots.push(pathToFileURL(join(home, '..', 'AppData', 'Roaming', 'npm', 'node_modules', '@deepseek-ai', 'dsh', 'lib', 'index.js')).href);
    }
    for (const root of roots) {
      try {
        const require = createRequire(root);
        return (await import(pathToFileURL(require.resolve('@deepseek-ai/dsh-app-boot')).href)) as {
          evaluatePluginCompatibility?: (
            manifest: object,
            exemptions?: Readonly<Record<string, readonly string[]>>,
            runtimeVersion?: string,
          ) => unknown;
        };
      } catch {
        // Not resolvable from this root; try the next.
      }
    }
    return undefined;
  }

  it('accepts this plugin\'s manifest', async (context) => {
    const appBoot = await loadAppBoot();
    if (appBoot?.evaluatePluginCompatibility === undefined) {
      // Skipped rather than faked. The matcher assertions above still hold the
      // line; this check is the one that would catch a divergence between them.
      context.skip('@deepseek-ai/dsh-app-boot is not installed here, so the real gate cannot be run');
      return;
    }
    const verdict = appBoot.evaluatePluginCompatibility(MANIFEST, undefined, RUNTIME_UNDER_TEST);
    assert.equal(
      verdict,
      undefined,
      `the harness's own gate accepts this manifest; it reported ${JSON.stringify(verdict)}`,
    );
  });

  it('would have refused the manifest this project used to ship', async (context) => {
    const appBoot = await loadAppBoot();
    if (appBoot?.evaluatePluginCompatibility === undefined) {
      context.skip('@deepseek-ai/dsh-app-boot is not installed here, so the regression cannot be demonstrated');
      return;
    }
    // The exact prior declaration, so the test proves the gate is the thing that
    // was failing rather than merely that a range looks different.
    const before = {
      ...MANIFEST,
      peerDependencies: {
        ...MANIFEST.peerDependencies,
        '@deepseek-ai/dsh-llm': '^0.1.5-rc.2',
        '@deepseek-ai/dsh-system-prompt': '^0.1.5-rc.2',
        '@deepseek-ai/dsh-tools': '^0.1.5-rc.2',
      },
    };
    const verdict = appBoot.evaluatePluginCompatibility(before, undefined, RUNTIME_UNDER_TEST) as
      | { readonly peers?: Record<string, string> }
      | undefined;
    assert.notEqual(verdict, undefined, 'the old caret ranges are refused by the real gate');
    assert.deepEqual(
      Object.keys(verdict?.peers ?? {}).sort(),
      ['@deepseek-ai/dsh-llm', '@deepseek-ai/dsh-system-prompt', '@deepseek-ai/dsh-tools'],
      'and the refusal names exactly those three peers',
    );
  });
});
