/**
 * Tests for the browser half of the settings page.
 *
 * `dsh-plugin/client.js` is a side-effect script with no build step, so nothing in
 * the normal toolchain would notice if it stopped loading: a broken bundle does
 * not fail a build, it fails silently in front of a user as a settings section
 * that never mounts. So this suite loads it the way DSH's module loader does —
 * under `window.__ModuleLoader__.load`, with React handed to the factory's
 * `require` — and asserts the things that must be true of the result.
 *
 * It deliberately stops at the loader contract and the registration, and checks
 * the rendered structure from the source rather than by rendering: React is a
 * *runtime* dependency handed in by the harness at load time, not a package this
 * project depends on, so rendering here would mean adding a second React to a
 * project that exists partly to have none. The registration is exercised for real;
 * the markup is checked by shape.
 *
 * @module dsh-ai-model-hub/tests/plugin-ui.test
 */

import { strict as assert } from 'node:assert';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

const CLIENT = fileURLToPath(new URL('../dsh-plugin/client.js', import.meta.url));

/** The module exports the loader captured. */
interface ClientExports {
  readonly name?: string;
  readonly inject?: readonly string[];
  readonly apply?: (ctx: unknown) => void;
}

/**
 * Load the client bundle exactly as DSH does.
 *
 * The factory is real: it receives a `require` and must return the module. React
 * is stubbed, because the bundle only *calls* `createElement` — the interesting
 * failure mode being defended here is a bundle that loads and registers, not one
 * that renders identically to a browser.
 *
 * @returns the captured module and the id it registered under.
 */
async function loadClient(): Promise<{ readonly id: string | undefined; readonly exports: ClientExports }> {
  const source = await readFile(CLIENT, 'utf8');
  let id: string | undefined;
  let captured: ClientExports | undefined;

  /** A jsx-runtime stand-in that records the element tree it is asked to build. */
  const jsx = (type: unknown, props: Record<string, unknown> | null, key?: unknown): unknown => ({
    type,
    props: props ?? {},
    key,
  });
  const react = {
    createElement: jsx,
    useCallback: (fn: unknown) => fn,
    useEffect: () => {},
    useState: (initial: unknown) => [initial, () => {}],
  };
  const moduleShim = { exports: {} as ClientExports };
  const requireShim = (specifier: string): unknown => {
    if (specifier === 'react') return react;
    if (specifier === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: 'Fragment' };
    throw new Error(`the client bundle required an unexpected module: ${specifier}`);
  };
  const window = {
    __ModuleLoader__: {
      load: (entry: { id: string; factory: (req: (s: string) => unknown) => ClientExports }) => {
        id = entry.id;
        captured = entry.factory(requireShim);
      },
    },
  };

  // A script, not a module: it registers itself on `window` and exports nothing.
  const run = new Function('window', 'require', 'module', 'exports', source);
  run(window, requireShim, moduleShim, moduleShim.exports);
  assert.ok(captured !== undefined, 'the bundle must register itself with __ModuleLoader__.load');
  return { id, exports: captured };
}

/** Run the plugin's `apply` against a slots stub and return what it registered. */
function registrations(
  exports: ClientExports,
): { readonly sections: { name: string; id: string; label: () => string; component: unknown }[]; readonly slots: string[] } {
  const sections: { name: string; id: string; label: () => string; component: unknown }[] = [];
  const slots: string[] = [];
  const ctx = {
    slots: {
      inject: (slot: string, generator: () => Generator<unknown>) => {
        slots.push(slot);
        for (const _ of generator()) void _;
      },
      register: (options: { name: string; id: string; label: () => string }, component: unknown) => {
        sections.push({ ...options, component });
        return () => {};
      },
    },
  };
  exports.apply?.(ctx);
  return { sections, slots };
}

describe('client bundle contract', () => {
  it('registers under the package name, which is what the loader keys on', async () => {
    const client = await loadClient();
    assert.equal(client.id, 'dsh-ai-model-hub', 'a bundle registered under another id never mounts');
    assert.equal(client.exports.name, 'dsh-ai-model-hub');
  });

  it('declares only the slots service it uses', async () => {
    const client = await loadClient();
    assert.deepEqual(client.exports.inject, ['slots']);
    assert.equal(typeof client.exports.apply, 'function');
  });

  it('has no top-level import or export, which the loader contract forbids', async () => {
    const source = await readFile(CLIENT, 'utf8');
    assert.doesNotMatch(source, /^\s*import\s/m, 'no top-level import');
    assert.doesNotMatch(source, /^\s*export\s/m, 'no top-level export');
  });

  it('requires nothing but React and its jsx runtime', async () => {
    // The stub throws on any other specifier, so a successful load proves it.
    await assert.doesNotReject(() => loadClient());
  });

  it('mounts exactly one settings section, labelled for a human', async () => {
    const client = await loadClient();
    const { sections, slots } = registrations(client.exports);
    assert.deepEqual(slots, ['settings.section']);
    assert.equal(sections.length, 1);
    assert.equal(sections[0]?.name, 'settings.section');
    assert.equal(sections[0]?.id, 'dsh-ai-model-hub');
    assert.equal(sections[0]?.label(), 'Local models');
    assert.equal(typeof sections[0]?.component, 'function');
  });
});

describe('the priorities editor is present and honest', () => {
  /** The client source, for the structural assertions. */
  async function source(): Promise<string> {
    return readFile(CLIENT, 'utf8');
  }

  it('has a Capability priorities section with the three concepts explained', async () => {
    const text = await source();
    assert.match(text, /Capability priorities/);
    assert.match(text, /Enabled permits an option to serve this capability/);
    assert.match(text, /it is never run, not even as a '\s*\+\s*'fallback/);
    assert.match(text, /Lower priority means it is tried only after/);
    assert.match(text, /Stopped means the engine is not running/);
  });

  it('offers a global switch that says it wins', async () => {
    const text = await source();
    assert.match(text, /Global switch/);
    assert.match(text, /Global disable wins over every per-capability setting/);
  });

  it('ships real drag-and-drop plus keyboard-reachable move controls', async () => {
    const text = await source();
    assert.match(text, /draggable:/, 'rows must be draggable');
    assert.match(text, /onDragStart:/);
    assert.match(text, /onDrop:/);
    assert.match(text, /Drag to reorder/);
    for (const label of [
      /Move ' \+ row\.name \+ ' up/,
      /Move ' \+ row\.name \+ ' down/,
      /to the top/,
      /to the bottom/,
    ]) {
      assert.match(text, label, `missing a keyboard reorder control: ${String(label)}`);
    }
    assert.match(text, /aria-pressed/, 'the enable toggle must expose its state');
    assert.match(text, /aria-label/, 'and the move controls must be nameable');
  });

  it('renders disabled rows visibly, without hiding them', async () => {
    const text = await source();
    assert.match(text, /children: row\.enabled \? 'Enabled' : 'Disabled'/);
    // A disabled row keeps its handle and its move buttons: it must stay
    // reorderable so re-enabling it restores the intended position.
    assert.doesNotMatch(text, /draggable: row\.enabled/);
  });

  it('shows the source, the state, and the reason on every row', async () => {
    const text = await source();
    assert.match(text, /label: 'source'/);
    assert.match(text, /label: 'reason'/);
    assert.match(text, /row\.providerLabel \+ ' · ' \+ row\.source/);
  });

  it('reads and writes the priorities route, and keeps edits until Save', async () => {
    const text = await source();
    assert.match(text, /'\/dsh-ai-model-hub\/priorities'/);
    assert.match(text, /method: 'PUT'/);
    assert.match(text, /'Saving…' : 'Save'/);
    assert.match(text, /button\('Cancel'/);
    assert.match(text, /unsaved changes/);
  });

  it('keeps the capability policy editor separate from the rescanning inventory', async () => {
    const text = await source();
    // A Scan must not throw away a half-finished reorder, which is why the two
    // live in different components.
    assert.match(text, /function CapabilityPriorities\(\)/);
    assert.match(text, /function LocalModelsSection\(\)/);
    assert.match(text, /function LocalModelsPage\(\)/);
    assert.match(text, /children: h\(LocalModelsSection, \{\}\)/);
    assert.match(text, /children: h\(CapabilityPriorities, \{\}\)/);
    assert.match(text, /hidden: tab !== 'inventory'/);
    assert.match(text, /hidden: tab !== 'priorities'/);
    assert.match(text, /ariaLabel: 'Edit priorities'/);
  });
});
