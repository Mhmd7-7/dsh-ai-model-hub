/**
 * Behavioral tests for the public settings component registered by client.js.
 *
 * React is supplied by the host, not a test dependency. This small renderer
 * resolves function components recursively and keeps hook state per keyed
 * instance/path, including instances inside hidden panels. Removing an instance
 * really does discard its state, so tab-remount regressions cannot pass by
 * accident. Fetch responses settle through microtasks, without timer polling.
 *
 * @module dsh-ai-model-hub/tests/client-interactions.test
 */

import { strict as assert } from 'node:assert';
import { readFile } from 'node:fs/promises';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';
import { runInNewContext } from 'node:vm';

const CLIENT = fileURLToPath(new URL('../dsh-plugin/client.js', import.meta.url));
const CAPABILITY = 'text_to_text';
const ORION = 'local:orion';
const QUASAR = 'subscription:quasar';
const SEARCH_LABEL = 'Search capabilities and providers';

type Props = Record<string, unknown>;
type Component = (props: Props) => unknown;

interface Element {
  readonly type: unknown;
  readonly props: Props;
  readonly key: unknown;
}

interface Hook {
  readonly kind: 'state' | 'memo' | 'effect' | 'ref';
  value: unknown;
  deps: readonly unknown[] | undefined;
  cleanup: (() => void) | undefined;
}

interface Instance {
  readonly type: Component;
  readonly hooks: Hook[];
  cursor: number;
  mounted: boolean;
}

interface Hit {
  readonly node: Element;
  readonly ancestors: readonly Element[];
  readonly hidden: boolean;
}

function isElement(value: unknown): value is Element {
  return typeof value === 'object' && value !== null && 'type' in value && 'props' in value;
}

function textContent(value: unknown): string {
  if (value === null || value === undefined || typeof value === 'boolean') return '';
  if (Array.isArray(value)) return value.map(textContent).join(' ');
  if (isElement(value)) return textContent(value.props.children);
  return String(value);
}

function sameDeps(before: readonly unknown[] | undefined, after: readonly unknown[] | undefined): boolean {
  return before !== undefined && after !== undefined && before.length === after.length &&
    before.every((value, index) => Object.is(value, after[index]));
}

/** Just the React surface the side-effect bundle needs; no DOM or React package. */
class HookRenderer {
  private readonly instances = new Map<string, Instance>();
  private active: Instance | undefined;
  private seen = new Set<string>();
  private effects: (() => void)[] = [];
  private dirty = true;
  private root: Component | undefined;
  tree: unknown;

  readonly jsx = (type: unknown, props: Props | null, key?: unknown): Element => ({
    type, props: props ?? {}, key: key ?? props?.key,
  });

  readonly react = {
    createElement: (type: unknown, props: Props | null, ...children: unknown[]) =>
      this.jsx(type, children.length > 0 ? { ...props, children } : props),
    Fragment: 'Fragment',
    useState: (initial: unknown): [unknown, (update: unknown) => void] => {
      const instance = this.current();
      const hook = this.slot('state', () => typeof initial === 'function' ? initial() : initial);
      return [hook.value, (update: unknown) => {
        if (!instance.mounted) return;
        const next = typeof update === 'function' ? update(hook.value) : update;
        if (!Object.is(hook.value, next)) {
          hook.value = next;
          this.dirty = true;
        }
      }];
    },
    useCallback: (callback: unknown, deps?: readonly unknown[]) => this.memo(() => callback, deps),
    useMemo: (factory: () => unknown, deps?: readonly unknown[]) => this.memo(factory, deps),
    useRef: (initial: unknown) => this.slot('ref', () => ({ current: initial })).value,
    useEffect: (effect: () => unknown, deps?: readonly unknown[]) => {
      const hook = this.slot('effect', () => undefined);
      if (hook.value !== true || !sameDeps(hook.deps, deps)) {
        hook.value = true;
        hook.deps = deps;
        this.effects.push(() => {
          hook.cleanup?.();
          const cleanup = effect();
          hook.cleanup = typeof cleanup === 'function' ? cleanup as () => void : undefined;
        });
      }
    },
  };

  private current(): Instance {
    assert.ok(this.active, 'hooks must execute inside a component instance');
    return this.active;
  }

  private slot(kind: Hook['kind'], initialize: () => unknown): Hook {
    const instance = this.current();
    const index = instance.cursor++;
    let hook = instance.hooks[index];
    if (!hook) {
      hook = { kind, value: initialize(), deps: undefined, cleanup: undefined };
      instance.hooks[index] = hook;
    }
    assert.equal(hook.kind, kind, 'hook order must remain stable within an instance');
    return hook;
  }

  private memo(factory: () => unknown, deps: readonly unknown[] | undefined): unknown {
    const hook = this.slot('memo', () => ({ initialized: false, result: undefined as unknown }));
    const memory = hook.value as { initialized: boolean; result: unknown };
    if (!memory.initialized || !sameDeps(hook.deps, deps)) {
      memory.initialized = true;
      memory.result = factory();
      hook.deps = deps;
    }
    return memory.result;
  }

  mount(root: Component): void {
    this.root = root;
    this.dirty = true;
    this.render();
  }

  private unmount(instance: Instance): void {
    instance.mounted = false;
    for (const hook of instance.hooks) hook.cleanup?.();
  }

  private resolve(value: unknown, path: string): unknown {
    if (Array.isArray(value)) {
      return value.map((child, index) => this.resolve(child,
        path + '/' + (isElement(child) && child.key !== undefined ? 'key:' + String(child.key) : index)));
    }
    if (!isElement(value)) return value;
    if (typeof value.type === 'function') {
      const type = value.type as Component;
      let instance = this.instances.get(path);
      if (!instance || instance.type !== type) {
        if (instance) this.unmount(instance);
        instance = { type, hooks: [], cursor: 0, mounted: true };
        this.instances.set(path, instance);
      }
      this.seen.add(path);
      instance.cursor = 0;
      const previous = this.active;
      this.active = instance;
      let output: unknown;
      try {
        output = type(value.props);
      } finally {
        this.active = previous;
      }
      return this.resolve(output, path + '/render');
    }
    return { ...value, props: { ...value.props, children: this.resolve(value.props.children, path + '/children') } };
  }

  render(): void {
    let passes = 0;
    while (this.dirty) {
      assert.ok(++passes < 50, 'component did not settle after synchronous updates');
      assert.ok(this.root, 'a registered component must be mounted first');
      this.dirty = false;
      this.seen = new Set();
      this.effects = [];
      this.tree = this.resolve(this.jsx(this.root, {}), 'root');
      for (const [path, instance] of this.instances) {
        if (!this.seen.has(path)) {
          this.unmount(instance);
          this.instances.delete(path);
        }
      }
      for (const effect of this.effects) effect();
    }
  }

  /** Bounded promise turns drain the mock fetch/json chains, not browser timers. */
  async settle(): Promise<void> {
    for (let turn = 0; turn < 24; turn++) {
      this.render();
      await Promise.resolve();
    }
    this.render();
  }

  hits(): Hit[] {
    const result: Hit[] = [];
    const visit = (value: unknown, ancestors: readonly Element[], hidden: boolean): void => {
      if (Array.isArray(value)) {
        for (const child of value) visit(child, ancestors, hidden);
      } else if (isElement(value)) {
        const nextHidden = hidden || value.props.hidden === true || value.props.hidden === '';
        result.push({ node: value, ancestors, hidden: nextHidden });
        visit(value.props.children, [...ancestors, value], nextHidden);
      }
    };
    visit(this.tree, [], false);
    return result;
  }

  button(name: string, includeHidden = false): Hit {
    const matches = this.hits().filter((hit) => hit.node.type === 'button' &&
      (includeHidden || !hit.hidden) &&
      (hit.node.props['aria-label'] ?? textContent(hit.node.props.children).trim()) === name);
    assert.equal(matches.length, 1, `expected exactly one ${includeHidden ? 'mounted' : 'visible'} button named ${JSON.stringify(name)}`);
    return matches[0]!;
  }

  panelFor(name: string): Element {
    const hit = this.button(name, true);
    const panel = [...hit.ancestors].reverse().find((node) => Object.hasOwn(node.props, 'hidden'));
    assert.ok(panel, `${name} must remain inside a mounted panel with an explicit hidden prop`);
    return panel;
  }

  async click(name: string): Promise<void> {
    const { node } = this.button(name);
    assert.notEqual(node.props.disabled, true, `${name} must be enabled before clicking it`);
    assert.equal(typeof node.props.onClick, 'function', `${name} must have a click handler`);
    await (node.props.onClick as (event: unknown) => unknown)({
      target: node, currentTarget: node, preventDefault() {}, stopPropagation() {},
    });
    await this.settle();
  }

  async search(value: string): Promise<void> {
    const matches = this.hits().filter((hit) => !hit.hidden && hit.node.type === 'input' &&
      hit.node.props['aria-label'] === SEARCH_LABEL);
    assert.equal(matches.length, 1, 'the priorities editor must expose its named search input');
    const input = matches[0]!.node;
    assert.equal(typeof input.props.onChange, 'function');
    (input.props.onChange as (event: unknown) => unknown)({ target: { value }, currentTarget: { value } });
    await this.settle();
  }

  rowNames(includeHidden = false): string[] {
    return this.hits().filter((hit) => hit.node.type === 'button' && (includeHidden || !hit.hidden))
      .map((hit) => String(hit.node.props['aria-label'] ?? '').match(/^\w+ (.+) for text_to_text$/)?.[1])
      .filter((name): name is string => name !== undefined);
  }

  visibleText(): string {
    return this.hits().filter((hit) => !hit.hidden && !isElement(hit.node.props.children) &&
      !Array.isArray(hit.node.props.children)).map((hit) => textContent(hit.node.props.children)).join(' ');
  }

  dispose(): void {
    for (const instance of this.instances.values()) this.unmount(instance);
    this.instances.clear();
  }
}

interface Option {
  id: string;
  name: string;
  type: string;
  source: string;
  provider: string;
  providerLabel: string;
  order: number;
  enabled: boolean;
  enabledBy: string;
  state: string;
  reason: string;
  startable: boolean;
  executable: boolean;
  capabilities: string[];
  inputTypes: string[];
  outputTypes: string[];
}

interface PolicyPayload {
  policyPath: string;
  revision: number;
  globalEnabled: boolean;
  savedAt: string;
  capabilities: {
    capability: string;
    inputTypes: string[];
    outputTypes: string[];
    options: Option[];
    permitted: string[];
    skipped: unknown[];
  }[];
}

function initialPolicy(): PolicyPayload {
  return {
    policyPath: 'C:\\fixture\\capability-policy.json', revision: 1, globalEnabled: true,
    savedAt: '2026-01-01T00:00:00.000Z',
    capabilities: [{
      capability: CAPABILITY, inputTypes: ['text'], outputTypes: ['text'], permitted: [ORION], skipped: [],
      options: [{
        id: ORION, name: 'Orion', type: 'ollama_model', source: 'model', provider: 'ollama',
        providerLabel: 'local · Ollama', order: 1, enabled: true, enabledBy: 'capability-policy',
        state: 'stopped', reason: 'The engine is stopped; the hub can start it on demand.',
        startable: true, executable: true, capabilities: [CAPABILITY], inputTypes: ['text'], outputTypes: ['text'],
      }, {
        id: QUASAR, name: 'Quasar', type: 'subscription', source: 'subscription', provider: 'grok',
        providerLabel: 'subscription · Grok (X Premium)', order: 2, enabled: false, enabledBy: 'capability-policy',
        state: 'disabled', reason: 'Disabled for this capability by the saved policy.',
        startable: false, executable: true, capabilities: [CAPABILITY], inputTypes: ['text'], outputTypes: ['text'],
      }],
    }],
  };
}

interface FetchCall {
  readonly url: string;
  readonly method: string;
  readonly body: unknown;
}

interface Scenario {
  readonly globalEnabled?: boolean;
  readonly failNextPut?: boolean;
  readonly manualTimers?: boolean;
}

/** The server returns the complete updated document after each PUT, like the real route. */
class RouteServer {
  readonly calls: FetchCall[] = [];
  readonly unexpected: string[] = [];
  private policy = initialPolicy();
  private failNextPut: boolean;
  holdInventory = false;
  readonly pendingInventory: { signal: AbortSignal | undefined; resolve: (response: unknown) => void; reject: (error: Error) => void }[] = [];
  readonly timers = new Map<number, () => void>();
  private nextTimer = 0;
  readonly setTimer = (callback: () => void): number => { const id = ++this.nextTimer; this.timers.set(id, callback); return id; };
  readonly clearTimer = (id: number): void => { this.timers.delete(id); };
  timeoutRequests(): void { for (const callback of [...this.timers.values()]) callback(); }

  constructor(scenario: Scenario) {
    this.policy.globalEnabled = scenario.globalEnabled ?? true;
    this.failNextPut = scenario.failNextPut ?? false;
  }

  /** Older hosts report effective flags, not saved choices, when the global switch is off. */
  private policyResponse(): PolicyPayload {
    const payload = structuredClone(this.policy);
    if (!payload.globalEnabled) {
      for (const entry of payload.capabilities) {
        entry.permitted = [];
        entry.options = entry.options.map((option) => ({
          ...option, enabled: false, enabledBy: 'global-switch', state: 'disabled',
          reason: 'The global capability switch is off.',
        }));
      }
    }
    return payload;
  }

  readonly fetch = async (input: unknown, init?: { method?: string; body?: unknown; signal?: AbortSignal }): Promise<unknown> => {
    const url = new URL(String(input), 'http://localhost');
    const method = init?.method ?? 'GET';
    const body: unknown = typeof init?.body === 'string' ? JSON.parse(init.body) : undefined;
    this.calls.push({ url: url.pathname + url.search, method, body });
    let payload: unknown;
    if (url.pathname === '/dsh-ai-model-hub/inventory' && method === 'GET') {
      if (this.holdInventory) return new Promise((resolve, reject) => {
        // Intentionally ignores abort so latest-wins is independently exercised.
        this.pendingInventory.push({ signal: init?.signal, resolve, reject });
      });
      payload = {
        catalogPath: 'C:\\fixture\\catalog.yaml', artifactRoot: null, allowProcessLaunch: true,
        generatedAt: 'inventory scan ' + this.inventoryReads(), machine: { vramGb: 8, ramGb: 32, hasGpu: true },
        diagnostics: [], engines: [], models: [], resources: [], scan: { sources: [], warnings: [] },
        capabilities: [{ capability: CAPABILITY, models: 2 }], unavailable: [],
      };
    } else if (url.pathname === '/dsh-ai-model-hub/priorities' && method === 'GET') {
      payload = this.policyResponse();
    } else if (url.pathname === '/dsh-ai-model-hub/priorities' && method === 'PUT') {
      if (this.failNextPut) {
        this.failNextPut = false;
        return { ok: false, status: 503, json: async () => ({ error: 'Temporary policy write failure' }) };
      }
      assert.ok(body && typeof body === 'object', 'a policy edit must have a JSON object body');
      const edit = body as { globalEnabled?: boolean; capability?: string; order?: { id: string; enabled: boolean }[] };
      if (typeof edit.globalEnabled === 'boolean') this.policy.globalEnabled = edit.globalEnabled;
      if (edit.capability !== undefined) {
        const entry = this.policy.capabilities.find((item) => item.capability === edit.capability);
        assert.ok(entry, 'PUT must name a real capability');
        assert.ok(Array.isArray(edit.order), 'PUT must include an ordered list');
        const byId = new Map(entry.options.map((option) => [option.id, option]));
        entry.options = edit.order.map((row, index) => {
          const option = byId.get(row.id);
          assert.ok(option, 'PUT must retain known option ids');
          assert.equal(typeof row.enabled, 'boolean');
          return { ...option, order: index + 1, enabled: row.enabled,
            state: row.enabled ? option.startable ? 'stopped' : 'ready' : 'disabled' };
        });
      }
      for (const entry of this.policy.capabilities) {
        entry.permitted = entry.options.filter((option) => this.policy.globalEnabled && option.enabled).map((option) => option.id);
      }
      this.policy.revision++;
      payload = this.policyResponse();
    } else {
      this.unexpected.push(method + ' ' + url.pathname + url.search);
      throw new Error('Unexpected mock route: ' + this.unexpected.at(-1));
    }
    const snapshot = structuredClone(payload);
    return { ok: true, status: 200, json: async () => structuredClone(snapshot) };
  };

  completeInventory(index: number, marker: string, extra: Record<string, unknown> = {}): void {
    this.pendingInventory[index]!.resolve({ ok: true, status: 200, json: async () => ({
      generatedAt: marker, catalogPath: marker, allowProcessLaunch: false,
      machine: { ramGb: 1, vramGb: 0, hasGpu: false }, resources: [], scan: { sources: [], warnings: [] },
      engines: [], models: [], capabilities: [], unavailable: [], ...extra,
    }) });
  }

  inventoryReads(): number {
    return this.calls.filter((call) => call.method === 'GET' && call.url.startsWith('/dsh-ai-model-hub/inventory')).length;
  }

  priorityReads(): number {
    return this.calls.filter((call) => call.method === 'GET' && call.url === '/dsh-ai-model-hub/priorities').length;
  }

  writes(): unknown[] {
    return this.calls.filter((call) => call.method === 'PUT').map((call) => call.body);
  }
}

async function withClient(
  run: (ui: HookRenderer, server: RouteServer) => Promise<void>,
  scenario: Scenario = {},
): Promise<void> {
  const source = await readFile(CLIENT, 'utf8');
  const ui = new HookRenderer();
  const server = new RouteServer(scenario);
  const previousFetch = Object.getOwnPropertyDescriptor(globalThis, 'fetch');
  Object.defineProperty(globalThis, 'fetch', { configurable: true, writable: true, value: server.fetch });
  try {
    let captured: { apply?: (ctx: unknown) => void } | undefined;
    const requireShim = (specifier: string): unknown => {
      if (specifier === 'react') return ui.react;
      if (specifier === 'react/jsx-runtime') return { jsx: ui.jsx, jsxs: ui.jsx, Fragment: 'Fragment' };
      throw new Error('Unexpected client require: ' + specifier);
    };
    const window = { __ModuleLoader__: {
      load: (entry: { id: string; factory: (require: (specifier: string) => unknown) => typeof captured }) => {
        assert.equal(entry.id, 'dsh-ai-model-hub');
        captured = entry.factory(requireShim);
      },
    } };
    const module = { exports: {} };
    runInNewContext(source, { window, require: requireShim, module, exports: module.exports,
      fetch: server.fetch, AbortController,
      setTimeout: scenario.manualTimers ? server.setTimer : setTimeout,
      clearTimeout: scenario.manualTimers ? server.clearTimer : clearTimeout }, { filename: CLIENT });
    assert.equal(typeof captured?.apply, 'function', 'the bundle must publish its registration');
    const components: Component[] = [];
    captured!.apply!({ slots: {
      inject: (slot: string, generator: () => Generator<unknown>) => {
        assert.equal(slot, 'settings.section');
        for (const _ of generator()) void _;
      },
      register: (options: { name: string; id: string }, component: Component) => {
        assert.equal(options.name, 'settings.section');
        assert.equal(options.id, 'dsh-ai-model-hub');
        components.push(component);
        return () => {};
      },
    } });
    assert.equal(components.length, 1, 'exercise the single public registered settings page');
    ui.mount(components[0]!);
    await ui.settle();
    await run(ui, server);
    assert.deepEqual(server.unexpected, [], 'all client fetches must use mocked public routes');
  } finally {
    ui.dispose();
    if (previousFetch) Object.defineProperty(globalThis, 'fetch', previousFetch);
    else Reflect.deleteProperty(globalThis, 'fetch');
  }
}

function assertUnsaved(ui: HookRenderer): void {
  assert.match(ui.visibleText(), /unsaved changes/i);
  assert.notEqual(ui.button('Save').node.props.disabled, true, 'dirty drafts must be saveable');
  assert.notEqual(ui.button('Cancel').node.props.disabled, true, 'dirty drafts must be cancellable');
}

function edit(order: { id: string; enabled: boolean }[]): unknown {
  return { capability: CAPABILITY, order };
}

describe('registered settings page interactions', { concurrency: false }, () => {
  it('repeat and out-of-order Refresh/Scan clicks keep only latest response and abort superseded work', async () => {
    await withClient(async (ui, server) => {
      server.holdInventory = true;
      await ui.click('Refresh');
      await ui.click('Scan');
      assert.equal(server.pendingInventory[0]!.signal!.aborted, true);
      assert.equal(server.pendingInventory[1]!.signal!.aborted, false);
      server.completeInventory(1, 'newest-result');
      await ui.settle();
      assert.match(ui.visibleText(), /newest-result/);
      server.completeInventory(0, 'obsolete-result');
      await ui.settle();
      assert.doesNotMatch(ui.visibleText(), /obsolete-result/);
      assert.match(ui.visibleText(), /newest-result/);
      assert.ok(ui.hits().filter((hit) => hit.node.props['aria-busy'] !== undefined)
        .every((hit) => hit.node.props['aria-busy'] === false));
    });
  });

  it('failed refresh keeps stale data, clears loading and Retry restores a fresh snapshot', async () => {
    await withClient(async (ui, server) => {
      server.holdInventory = true;
      await ui.click('Refresh');
      server.pendingInventory[0]!.reject(new Error('fixture disconnected'));
      await ui.settle();
      assert.match(textContent(ui.panelFor('Refresh')), /fixture disconnected/);
      assert.match(textContent(ui.panelFor('Refresh')), /stale/i);
      assert.match(ui.visibleText(), /inventory scan 1/);
      await ui.click('Retry');
      assert.match(textContent(ui.panelFor('Refresh')), /stale — retrying/);
      assert.match(textContent(ui.panelFor('Refresh')), /inventory scan 1/);
      server.completeInventory(1, 'recovered-result');
      await ui.settle();
      assert.match(ui.visibleText(), /recovered-result/);
      assert.doesNotMatch(ui.visibleText(), /fixture disconnected|stale — refresh failed/);
    });
  });

  it('invalid successful HTTP bodies fail safely without replacing the last inventory snapshot', async () => {
    await withClient(async (ui, server) => {
      server.holdInventory = true;
      for (const body of [null, {}, { machine: {}, engines: null }]) {
        const index = server.pendingInventory.length;
        await ui.click('Refresh');
        server.pendingInventory[index]!.resolve({ ok: true, status: 200, json: async () => body });
        await ui.settle();
        assert.match(textContent(ui.panelFor('Refresh')), /Invalid inventory response from the scan service/);
        assert.match(textContent(ui.panelFor('Refresh')), /stale/);
        assert.match(ui.visibleText(), /inventory scan 1/);
        assert.equal(ui.hits().find((hit) => hit.node.type === 'section' && hit.node.props['aria-busy'] === true), undefined);
      }
      await ui.click('Retry');
      assert.match(textContent(ui.panelFor('Refresh')), /stale — retrying/);
      server.completeInventory(3, 'valid-after-invalid');
      await ui.settle();
      assert.match(ui.visibleText(), /valid-after-invalid/);
      assert.doesNotMatch(textContent(ui.panelFor('Refresh')), /Invalid inventory response|stale —/);
    });
  });

  it('engine cards distinguish configured paths from verified installation and expose per-engine failures and freshness', async () => {
    await withClient(async (ui, server) => {
      server.holdInventory = true;
      await ui.click('Refresh');
      server.completeInventory(0, 'card-fixture', { engines: [{
        id: 'comfy', name: 'Fixture ComfyUI', adapter: 'comfyui', configured: true, running: false,
        state: 'installed_unreachable', installation: 'installed', installPath: 'C:/verified/ComfyUI',
        configuredPaths: ['C:/configured/ComfyUI'], checkedPaths: ['C:/verified/ComfyUI'],
        installSource: 'main.py marker', checkedAt: 'fixture-check-time', errors: ['fixture endpoint timed out'],
        statusDetail: 'Endpoint unavailable', models: [], storePaths: [], modelsSource: 'not read', servedModelIds: [],
      }], models: [{ id: 'workflow', availability: 'unknown', lifecycle: 'external', capabilities: [], freshness: 'catalog' }] });
      await ui.settle();
      const content = textContent(ui.panelFor('Refresh'));
      assert.match(content, /installed · unreachable/);
      assert.match(content, /configured paths C:\/configured\/ComfyUI/);
      assert.match(content, /verified install C:\/verified\/ComfyUI/);
      assert.match(content, /fixture endpoint timed out/);
      assert.match(content, /fixture-check-time/);
      assert.match(content, /catalog state — not live verified/);
      assert.match(content, /last scan snapshot/);
    });
  });

  it('request timeout aborts the transport, clears busy state and timers, preserves stale data and permits Retry', async () => {
    await withClient(async (ui, server) => {
      server.holdInventory = true;
      await ui.click('Refresh');
      assert.equal(server.timers.size, 1);
      server.timeoutRequests();
      await ui.settle();
      assert.equal(server.pendingInventory[0]!.signal!.aborted, true);
      assert.equal(server.timers.size, 0);
      assert.match(textContent(ui.panelFor('Refresh')), /timed out after 30 seconds/);
      assert.match(textContent(ui.panelFor('Refresh')), /stale/);
      assert.equal(ui.hits().find((hit) => hit.node.type === 'section' && hit.node.props['aria-busy'] === true), undefined);
      await ui.click('Retry');
      server.completeInventory(1, 'after-timeout');
      await ui.settle();
      assert.equal(server.timers.size, 0);
      assert.match(ui.visibleText(), /after-timeout/);
    }, { manualTimers: true });
  });

  it('unmount aborts pending inventory work and settling it cannot remount the page', async () => {
    await withClient(async (ui, server) => {
      server.holdInventory = true;
      await ui.click('Scan');
      ui.dispose();
      assert.equal(server.pendingInventory[0]!.signal!.aborted, true);
      server.completeInventory(0, 'after-unmount');
      await Promise.resolve();
      await Promise.resolve();
    });
  });

  it('a legacy globally paused payload cannot erase unknown saved per-option choices', async () => {
    await withClient(async (ui, server) => {
      await ui.click('Edit priorities');
      assert.match(ui.visibleText(), /Paused/);
      assert.deepEqual(ui.rowNames(), [], 'unknown saved permissions must not be presented as editable disabled choices');
      assert.equal(ui.button('Move Orion down').node.props.disabled, true);
      assert.equal(ui.button('Move Quasar up').node.props.disabled, true);
      assert.equal(ui.button('Save').node.props.disabled, true);
      assert.deepEqual(server.writes(), [], 'opening a paused policy must not write guessed option flags');

      await ui.click('Inventory');
      await ui.click('Priorities');
      assert.equal(ui.button('Move Orion down').node.props.disabled, true);
      await ui.click('Enable capabilities');
      assertUnsaved(ui);
      assert.equal(ui.button('Move Orion down').node.props.disabled, true,
        'draft enabling alone cannot reveal choices hidden by the older server');
      assert.deepEqual(server.writes(), []);
      await ui.click('Save');
      assert.deepEqual(server.writes(), [{ globalEnabled: true }],
        'a global resume must never overwrite an older host\'s saved capability choices');
      assert.deepEqual(ui.rowNames(), ['Orion', 'Quasar']);
      assert.equal(ui.button('Disable Orion for text_to_text').node.props['aria-pressed'], true);
      assert.equal(ui.button('Enable Quasar for text_to_text').node.props['aria-pressed'], false);
      assert.notEqual(ui.button('Move Orion down').node.props.disabled, true);
      assert.equal(ui.button('Save').node.props.disabled, true);
    }, { globalEnabled: false });
  });

  it('a rejected PUT retains the unsaved draft and permits a successful retry', async () => {
    await withClient(async (ui, server) => {
      await ui.click('Edit priorities');
      await ui.click('Enable Quasar for text_to_text');
      await ui.click('Move Orion down');
      await ui.click('Save');
      const expected = edit([{ id: QUASAR, enabled: true }, { id: ORION, enabled: true }]);
      assert.deepEqual(server.writes(), [expected]);
      assert.match(ui.visibleText(), /Not saved: HTTP 503 — Temporary policy write failure/);
      assertUnsaved(ui);
      assert.deepEqual(ui.rowNames(), ['Quasar', 'Orion']);
      assert.equal(ui.button('Disable Quasar for text_to_text').node.props['aria-pressed'], true);
      await ui.click('Save');
      assert.deepEqual(server.writes(), [expected, expected], 'retry must send exactly the retained draft');
      assert.equal(ui.button('Save').node.props.disabled, true);
      assert.equal(ui.button('Cancel').node.props.disabled, true);
      assert.deepEqual(ui.rowNames(), ['Quasar', 'Orion']);
    }, { failNextPut: true });
  });

  it('Edit priorities routes to its tab, and tab switches hide rather than unmount either panel', async () => {
    await withClient(async (ui, server) => {
      assert.equal(ui.button('Inventory').node.props['aria-selected'], true);
      assert.equal(ui.panelFor('Refresh').props.hidden, false);
      assert.equal(ui.panelFor('Enable Quasar for text_to_text').props.hidden, true);
      assert.equal(server.inventoryReads(), 1);
      assert.equal(server.priorityReads(), 1, 'the hidden priorities panel must already be mounted');

      await ui.click('Edit priorities');
      assert.equal(ui.button('Priorities').node.props['aria-selected'], true);
      assert.equal(ui.panelFor('Refresh').props.hidden, true);
      assert.equal(ui.panelFor('Enable Quasar for text_to_text').props.hidden, false);
      assert.deepEqual(ui.rowNames(), ['Orion', 'Quasar']);

      await ui.click('Inventory');
      assert.equal(ui.button('Inventory').node.props['aria-selected'], true);
      assert.equal(ui.panelFor('Refresh').props.hidden, false);
      assert.equal(ui.panelFor('Enable Quasar for text_to_text').props.hidden, true);
      await ui.click('Priorities');
      assert.equal(ui.panelFor('Refresh').props.hidden, true);
      assert.equal(ui.panelFor('Enable Quasar for text_to_text').props.hidden, false);
      assert.equal(server.inventoryReads(), 1, 'switching tabs must not remount or rescan inventory');
      assert.equal(server.priorityReads(), 1, 'switching tabs must not remount or reload the editor');
    });
  });

  it('enabled edits stay unsaved until Save PUTs the full order, and Cancel restores the saved draft', async () => {
    await withClient(async (ui, server) => {
      await ui.click('Edit priorities');
      assert.equal(ui.button('Enable Quasar for text_to_text').node.props['aria-pressed'], false);
      assert.equal(ui.button('Save').node.props.disabled, true);
      await ui.click('Enable Quasar for text_to_text');
      assert.equal(ui.button('Disable Quasar for text_to_text').node.props['aria-pressed'], true);
      assertUnsaved(ui);
      assert.deepEqual(server.writes(), [], 'toggling is a draft edit, not an immediate write');

      await ui.click('Save');
      assert.deepEqual(server.writes(), [edit([{ id: ORION, enabled: true }, { id: QUASAR, enabled: true }])]);
      assert.equal(ui.button('Save').node.props.disabled, true);
      assert.equal(ui.button('Cancel').node.props.disabled, true);

      await ui.click('Disable Quasar for text_to_text');
      assert.equal(ui.button('Enable Quasar for text_to_text').node.props['aria-pressed'], false);
      assertUnsaved(ui);
      await ui.click('Cancel');
      assert.equal(ui.button('Disable Quasar for text_to_text').node.props['aria-pressed'], true,
        'Cancel must restore the last saved policy, not the original mount snapshot');
      assert.deepEqual(ui.rowNames(), ['Orion', 'Quasar']);
      assert.equal(ui.button('Save').node.props.disabled, true);
      assert.equal(server.writes().length, 1, 'Cancel must not persist another policy edit');
    });
  });

  it('Move Orion down changes the visible ordering and PUT order without enabling a disabled row', async () => {
    await withClient(async (ui, server) => {
      await ui.click('Edit priorities');
      await ui.click('Move Orion down');
      assert.deepEqual(ui.rowNames(), ['Quasar', 'Orion']);
      assert.equal(ui.button('Move Orion down').node.props.disabled, true, 'the last option cannot move farther down');
      assert.equal(ui.button('Enable Quasar for text_to_text').node.props['aria-pressed'], false);
      assertUnsaved(ui);
      assert.deepEqual(server.writes(), []);
      await ui.click('Save');
      assert.deepEqual(server.writes(), [edit([{ id: QUASAR, enabled: false }, { id: ORION, enabled: true }])]);
      assert.deepEqual(ui.rowNames(), ['Quasar', 'Orion']);
      assert.equal(ui.button('Save').node.props.disabled, true);
    });
  });

  it('inventory Refresh and Scan preserve a hidden unsaved priorities draft', async () => {
    await withClient(async (ui, server) => {
      await ui.click('Edit priorities');
      await ui.click('Enable Quasar for text_to_text');
      await ui.click('Move Orion down');
      assertUnsaved(ui);
      const inventoryReads = server.inventoryReads();
      const priorityReads = server.priorityReads();

      await ui.click('Inventory');
      await ui.click('Refresh');
      await ui.click('Scan');
      assert.equal(server.inventoryReads(), inventoryReads + 2);
      assert.equal(server.priorityReads(), priorityReads, 'inventory reload must not fetch or replace the policy draft');
      assert.equal(ui.panelFor('Disable Quasar for text_to_text').props.hidden, true);
      assert.deepEqual(ui.rowNames(true), ['Quasar', 'Orion'], 'the hidden panel must still own its edited order');
      await ui.click('Priorities');
      assertUnsaved(ui);
      assert.equal(ui.button('Disable Quasar for text_to_text').node.props['aria-pressed'], true);
      assert.deepEqual(ui.rowNames(), ['Quasar', 'Orion']);
      assert.deepEqual(server.writes(), []);
      await ui.click('Save');
      assert.deepEqual(server.writes(), [edit([{ id: QUASAR, enabled: true }, { id: ORION, enabled: true }])]);
    });
  });

  it('search matches model/provider names, hides unmatched capabilities, and retains their full draft', async () => {
    await withClient(async (ui, server) => {
      await ui.click('Edit priorities');
      await ui.click('Enable Quasar for text_to_text');
      await ui.click('Move Orion down');
      await ui.search('Orion');
      // Search filters whole capability cards, not individual rows: the complete
      // ordering context must remain truthful when just one option matches.
      assert.deepEqual(ui.rowNames(), ['Quasar', 'Orion'], 'a model-name match must keep its capability visible');
      await ui.search('gRoK');
      assert.deepEqual(ui.rowNames(), ['Quasar', 'Orion'], 'provider matching must be case insensitive');
      await ui.search('__no_such_provider__');
      assert.deepEqual(ui.rowNames(), [], 'a nonmatching capability must be filtered out');
      assertUnsaved(ui);
      assert.deepEqual(server.writes(), [], 'filtering must never persist or delete hidden draft options');
      await ui.search('OLLAMA');
      assert.deepEqual(ui.rowNames(), ['Quasar', 'Orion']);
      assert.equal(ui.button('Disable Quasar for text_to_text').node.props['aria-pressed'], true,
        'restoring a hidden card must retain its enablement edit');
      await ui.search('');
      assert.deepEqual(ui.rowNames(), ['Quasar', 'Orion']);
      await ui.click('Save');
      assert.deepEqual(server.writes(), [edit([{ id: QUASAR, enabled: true }, { id: ORION, enabled: true }])]);
      assert.equal(server.priorityReads(), 1, 'search must not reload the saved policy over the draft');
    });
  });
});
