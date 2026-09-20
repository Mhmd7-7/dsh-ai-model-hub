/**
 * Tests for the settings page: the host-side inventory and the browser-side file.
 *
 * Both halves are exercised without a browser and without a server. The inventory
 * runs against a real `ModelHub` and a *stub machine* (the probes are injectable
 * precisely so a test can describe an installation without owning one), and the
 * route is driven through a captured handler with fake request/response objects.
 *
 * The client file is the interesting case: it is plain JavaScript that runs as a
 * side effect against `window.__ModuleLoader__`, so the test stands in for the
 * browser — a recording module loader, a minimal React with working hooks, and a
 * stubbed fetch — and then renders the section far enough to assert that the paths
 * a user came for are actually on the screen. That is the only way to catch a typo
 * in a file no type checker reads.
 *
 * @module dsh-ai-model-hub/tests/inventory.test
 */

import { strict as assert } from 'node:assert';
import { readFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { describe, it } from 'node:test';

import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Context } from '@deepseek-ai/cordis';

import { ModelHub } from '../src/index.ts';
import { INVENTORY_ROUTE, buildInventory, registerInventoryRoute } from '../dsh-plugin/inventory.ts';
import type { Inventory, InventoryProbes } from '../dsh-plugin/inventory.ts';
import type { PluginLogger } from '../dsh-plugin/types.ts';

/** A catalog with two real engines: one reachable, one that is not installed. */
function catalog(): Record<string, unknown> {
  return {
    version: '1',
    hosts: [
      {
        id: 'ollama',
        name: 'Ollama',
        adapter: 'openai_compatible',
        runtime: { engine: 'ollama', adapter: 'openai_compatible', endpoint: 'http://127.0.0.1:11434', path: '/v1/chat/completions' },
        health: { kind: 'http', path: '/api/tags' },
        lifecycle: { startable: true, stoppable: true, start: { command: 'ollama.exe', args: ['serve'] } },
      },
      {
        id: 'comfyui',
        name: 'ComfyUI',
        adapter: 'comfyui',
        runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint: 'http://127.0.0.1:8188', path: '/prompt' },
        lifecycle: {
          startable: true,
          stoppable: true,
          start: { command: 'C:/ComfyUI/venv/Scripts/python.exe', args: ['C:/ComfyUI/src/main.py'], cwd: 'C:/ComfyUI' },
        },
      },
    ],
    models: [
      { id: 'ollama_text', name: 'Local text', type: 'text_generation', host: 'ollama', capabilities: ['text_to_text'] },
      { id: 'comfyui_image', name: 'Local image', type: 'image_generation', host: 'comfyui', capabilities: ['text_to_image'] },
    ],
  };
}

/** A machine where ComfyUI is installed and listening, and nothing else is. */
function machine(options: { ollamaUp?: boolean; probeOllamaTags?: boolean } = {}): InventoryProbes {
  // The inventory joins paths with the platform separator, so the stub normalizes
  // before matching rather than pretending Windows uses forward slashes.
  const norm = (path: string): string => path.replace(/\\/g, '/');
  return {
    fetchJson: async (url) => {
      if (url === 'http://127.0.0.1:8188' || url === 'http://127.0.0.1:8188/system_stats') return { ok: true };
      if (url === 'http://127.0.0.1:11434' || url === 'http://127.0.0.1:11434/api/tags') {
        if (options.ollamaUp === false) throw new Error('connect ECONNREFUSED');
        if (url === 'http://127.0.0.1:11434/api/tags' && options.probeOllamaTags !== false) {
          return { models: [{ name: 'llama3:8b', size: 4_700_000_000, details: { parameter_size: '8B', quantization_level: 'Q4_K_M' } }] };
        }
        return { models: [] };
      }
      throw new Error('connect ECONNREFUSED');
    },
    isDirectory: async (path) => norm(path) === 'C:/ComfyUI' || norm(path) === 'C:/ComfyUI/models',
    listDir: async (path) => {
      if (norm(path) === 'C:/ComfyUI/models') return [{ name: 'checkpoints', isDirectory: true }];
      if (norm(path) === 'C:/ComfyUI/models/checkpoints') {
        return [{ name: 'z_image_turbo.safetensors', isDirectory: false, sizeBytes: 2_048_000 }];
      }
      return undefined;
    },
  };
}

/** A hub over the fixture catalog. */
function hub(): ModelHub {
  return ModelHub.fromConfig(catalog(), { manageTimers: false, log: () => {} });
}

/** Build an inventory over the fixture machine. */
async function inventory(options: { probes?: InventoryProbes; probe?: boolean } = {}): Promise<Inventory> {
  const instance = hub();
  try {
    return await buildInventory(instance, {
      hosts: (catalog()['hosts'] ?? []) as never,
      catalogPath: 'C:/catalog/models.json',
      artifactRoot: 'C:/workspace/artifacts',
      allowProcessLaunch: true,
      probes: options.probes ?? machine(),
      probe: options.probe ?? true,
    });
  } finally {
    await instance.dispose();
  }
}

/** A logger that records instead of printing. */
function recordingLogger(): { lines: string[]; logger: PluginLogger } {
  const lines: string[] = [];
  return {
    lines,
    logger: {
      debug: (message) => lines.push(`debug ${message}`),
      info: (message) => lines.push(`info ${message}`),
      warn: (message) => lines.push(`warn ${message}`),
      error: (message) => lines.push(`error ${message}`),
    },
  };
}

describe('engine inventory (host half)', () => {
  it('reports a configured engine with where it is installed and how it starts', async () => {
    const report = await inventory();
    const comfyui = report.engines.find((engine) => engine.id === 'comfyui');
    assert.notEqual(comfyui, undefined);

    assert.equal(comfyui?.configured, true);
    assert.equal(comfyui?.installPath, 'C:/ComfyUI');
    assert.equal(comfyui?.installSource, 'found on this machine');
    assert.equal(comfyui?.running, true);
    assert.equal(comfyui?.startable, true);
    assert.equal(comfyui?.launch?.command, 'C:/ComfyUI/venv/Scripts/python.exe');
    assert.deepEqual(comfyui?.launch?.args, ['C:/ComfyUI/src/main.py']);
    assert.equal(comfyui?.launch?.cwd, 'C:/ComfyUI');
    assert.deepEqual(
      comfyui?.servedModelIds,
      ['comfyui_image'],
      'the catalog models behind a host are grouped under it',
    );
  });

  it('takes the engine\'s own model list from the engine, with its sizes', async () => {
    const report = await inventory();
    const ollama = report.engines.find((engine) => engine.id === 'ollama');

    assert.equal(ollama?.modelsSource, 'Ollama /api/tags');
    assert.equal(ollama?.models.length, 1);
    assert.equal(ollama?.models[0]?.id, 'llama3:8b');
    assert.match(ollama?.models[0]?.detail ?? '', /8B/);
    assert.match(ollama?.models[0]?.detail ?? '', /Q4_K_M/);
    assert.match(
      ollama?.models[0]?.detail ?? '',
      /GiB/,
      'the reported size is rendered, not shipped raw',
    );
  });

  it('lists the model store and the files inside it', async () => {
    const report = await inventory();
    const comfyui = report.engines.find((engine) => engine.id === 'comfyui');
    const store = comfyui?.storePaths.find((entry) => entry.exists === true);

    assert.equal(store?.path, 'C:/ComfyUI/models');
    assert.deepEqual(store?.files, [{ name: 'checkpoints/z_image_turbo.safetensors', sizeBytes: 2_048_000 }]);
  });

  it('still reports an engine that is neither installed nor configured', async () => {
    const report = await inventory();
    const a1111 = report.engines.find((engine) => engine.id === 'a1111');

    assert.notEqual(a1111, undefined, 'a known engine is never silently omitted');
    assert.equal(a1111?.configured, false);
    assert.equal(a1111?.installPath, undefined);
    assert.equal(a1111?.running, false);
    assert.equal(a1111?.startable, false);
    assert.ok(
      (a1111?.checkedPaths ?? []).some((path) => path.includes('stable-diffusion-webui')),
      'the page can tell the user which directories were looked at',
    );
  });

  it('does not claim an endpoint is up unless it was probed', async () => {
    const report = await inventory({ probe: false });
    for (const engine of report.engines) {
      assert.equal(engine.running, false);
      assert.match(engine.statusDetail, /not probed/);
    }
  });

  it('says an engine is stopped when its endpoint refuses the connection', async () => {
    const report = await inventory({ probes: machine({ ollamaUp: false }) });
    const ollama = report.engines.find((engine) => engine.id === 'ollama');
    assert.equal(ollama?.running, false);
    assert.match(ollama?.statusDetail ?? '', /ECONNREFUSED/);
  });

  it('carries the catalog models, capabilities, and what is unavailable', async () => {
    const report = await inventory();
    assert.equal(report.catalogPath, 'C:/catalog/models.json');
    assert.equal(report.artifactRoot, 'C:/workspace/artifacts');
    assert.equal(report.allowProcessLaunch, true);
    assert.deepEqual(
      report.models.map((model) => model.id).sort(),
      ['comfyui_image', 'ollama_text'],
    );
    assert.ok(report.capabilities.some((entry) => entry.capability === 'text_to_image'));
    assert.ok(
      report.unavailable.some((entry) => entry.capability === 'video_generation'),
      'capabilities nothing serves are reported, not hidden',
    );
  });
});

describe('the inventory route (host half)', () => {
  /** Run `registerInventoryRoute` against a stand-in server and return the route. */
  function captureRoute(hubLike: unknown): {
    handler: (req: IncomingMessage, res: ServerResponse) => void | Promise<void>;
    lines: string[];
  } {
    const { lines, logger } = recordingLogger();
    let handler: ((req: IncomingMessage, res: ServerResponse) => void | Promise<void>) | undefined;
    const ctx = {
      inject: (_deps: readonly string[], callback: (scope: unknown) => void) => {
        callback({
          webServer: {
            register(route: { handler: (req: IncomingMessage, res: ServerResponse) => void | Promise<void> }) {
              handler = route.handler;
              return () => {};
            },
          },
        });
      },
    } as unknown as Context;

    registerInventoryRoute(ctx, logger, {
      hub: hubLike as never,
      hosts: (catalog()['hosts'] ?? []) as never,
      catalogPath: 'C:/catalog/models.json',
      artifactRoot: '',
      allowProcessLaunch: false,
      probes: machine(),
    });
    assert.notEqual(handler, undefined, 'the route was registered');
    return { handler: handler as never, lines };
  }

  /** A response object that records what was written. */
  function fakeResponse(): { captured: { status?: number; body?: string | undefined }; res: ServerResponse } {
    const captured: { status?: number; body?: string | undefined } = {};
    return {
      captured,
      res: {
        writeHead: (status: number) => {
          captured.status = status;
        },
        end: (body?: string) => {
          captured.body = body;
        },
      } as unknown as ServerResponse,
    };
  }

  it('answers GET with the inventory as JSON', async () => {
    const instance = hub();
    try {
      const { handler } = captureRoute(instance);
      const { captured, res } = fakeResponse();
      await handler({ method: 'GET', url: INVENTORY_ROUTE } as IncomingMessage, res);

      assert.equal(captured.status, 200);
      const parsed = JSON.parse(captured.body ?? '{}') as Inventory;
      assert.equal(parsed.catalogPath, 'C:/catalog/models.json');
      assert.ok(parsed.engines.length >= 3, 'the response carries every engine');
      assert.equal(parsed.allowProcessLaunch, false);
    } finally {
      await instance.dispose();
    }
  });

  it('reads the live-probe switch from the query string', async () => {
    const instance = hub();
    try {
      const { handler } = captureRoute(instance);
      const withoutProbe = fakeResponse();
      await handler({ method: 'GET', url: INVENTORY_ROUTE } as IncomingMessage, withoutProbe.res);
      const lazy = JSON.parse(withoutProbe.captured.body ?? '{}') as Inventory;
      assert.equal(lazy.engines.find((engine) => engine.id === 'comfyui')?.running, false);
      assert.match(lazy.engines.find((engine) => engine.id === 'comfyui')?.statusDetail ?? '', /not probed/);

      const withProbe = fakeResponse();
      await handler({ method: 'GET', url: `${INVENTORY_ROUTE}?probe=1` } as IncomingMessage, withProbe.res);
      const live = JSON.parse(withProbe.captured.body ?? '{}') as Inventory;
      assert.equal(live.engines.find((engine) => engine.id === 'comfyui')?.running, true);
    } finally {
      await instance.dispose();
    }
  });

  it('refuses anything but GET', async () => {
    const instance = hub();
    try {
      const { handler } = captureRoute(instance);
      const { captured, res } = fakeResponse();
      await handler({ method: 'POST', url: INVENTORY_ROUTE } as IncomingMessage, res);
      assert.equal(captured.status, 405);
    } finally {
      await instance.dispose();
    }
  });

  it('answers 500 and logs rather than throwing when the inventory fails', async () => {
    const broken = {
      listModels: () => {
        throw new Error('catalog exploded');
      },
      machineProfile: { ramGb: 0, vramGb: 0, hasGpu: false, notes: '' },
    };
    const { handler, lines } = captureRoute(broken);
    const { captured, res } = fakeResponse();
    await handler({ method: 'GET', url: INVENTORY_ROUTE } as IncomingMessage, res);

    assert.equal(captured.status, 500);
    assert.match(lines.join('\n'), /engine inventory route failed: catalog exploded/);
  });
});

describe('the settings page (browser half)', () => {
  /**
   * Collect every string in a rendered jsx tree.
   *
   * Function components are invoked with their props, which is exactly what React
   * would do; the page's own component is the only one that uses hooks, and it is
   * the root, so nothing here needs a scheduler.
   */
  function flatten(node: unknown, out: string[] = []): string[] {
    if (node === null || node === undefined || typeof node === 'boolean') return out;
    if (typeof node === 'string' || typeof node === 'number') {
      out.push(String(node));
      return out;
    }
    if (Array.isArray(node)) {
      for (const child of node) flatten(child, out);
      return out;
    }
    const element = node as { type?: unknown; props?: { children?: unknown } };
    if (typeof element.type === 'function' && element.props !== undefined) {
      flatten((element.type as (props: unknown) => unknown)(element.props), out);
    }
    if (element.props !== undefined) flatten(element.props.children, out);
    return out;
  }

  /**
   * The module loader's captured output, filled on the first import.
   *
   * A module is imported once per process, so the second test reuses the factory
   * the first one captured rather than expecting `load` to fire again.
   */
  let capturedId: string | undefined;
  let capturedFactory: ((require: (name: string) => unknown) => {
    name: string;
    inject: unknown;
    apply: (ctx: unknown) => void;
  }) | undefined;

  /** Load `dsh-plugin/client.js` the way the browser does, and return its module. */
  async function loadClient(inventoryJson: unknown): Promise<{
    module: { name: string; inject: unknown; apply: (ctx: unknown) => void };
    registered: { spec?: Record<string, unknown>; component?: () => unknown };
    render: () => unknown;
  }> {
    const clientPath = fileURLToPath(new URL('../dsh-plugin/client.js', import.meta.url));
    const source = await readFile(clientPath, 'utf8');

    // The contract DSH relies on: a side-effect script with no module syntax of its
    // own. A stray `import` here would be a silent double-load in a real browser.
    assert.equal(/^\s*(import|export)\s/m.test(source), false, 'the client must not use import/export');

    const globals = globalThis as { window?: unknown; fetch?: unknown };
    const savedWindow = globals.window;
    const savedFetch = globals.fetch;
    if (capturedFactory === undefined) {
      globals.window = {
        __ModuleLoader__: {
          load(info: { id: string; factory: typeof capturedFactory }) {
            capturedId = info.id;
            capturedFactory = info.factory;
          },
        },
      };
    }
    globals.fetch = (async () => ({ ok: true, status: 200, json: async () => inventoryJson })) as unknown;

    try {
      if (capturedFactory === undefined) {
        await import(pathToFileURL(clientPath).href);
      }

      assert.equal(capturedId, 'dsh-ai-model-hub', 'the module id matches the package name');
      assert.notEqual(capturedFactory, undefined);

      // A minimal React with hooks that actually hold state, so the page's
      // loading -> ready transition can be replayed.
      let slotIndex = 0;
      const slots: { value: unknown }[] = [];
      let effectsRun = false;
      const react = {
        useState: (initial: unknown) => {
          const index = slotIndex;
          slotIndex += 1;
          if (slots.length <= index) {
            slots.push({ value: typeof initial === 'function' ? (initial as () => unknown)() : initial });
          }
          const slot = slots[index] as { value: unknown };
          return [
            slot.value,
            (next: unknown) => {
              slot.value = typeof next === 'function' ? (next as (previous: unknown) => unknown)(slot.value) : next;
            },
          ];
        },
        useEffect: (effect: () => void) => {
          if (!effectsRun) {
            effectsRun = true;
            effect();
          }
        },
        useCallback: (fn: unknown) => fn,
      };
      const h = (type: unknown, props: { children?: unknown }) => ({ type, props: props ?? {} });
      const requireStub = (name: string): unknown => {
        if (name === 'react') return react;
        if (name === 'react/jsx-runtime') return { jsx: h };
        throw new Error(`the client required an unexpected module: ${name}`);
      };

      const module = (capturedFactory as NonNullable<typeof capturedFactory>)(requireStub);
      const registered: { spec?: Record<string, unknown>; component?: () => unknown } = {};
      const ctx = {
        slots: {
          inject: (slotName: string, generator: () => Iterator<unknown>) => {
            assert.equal(slotName, 'settings.section');
            generator().next();
          },
          register: (spec: Record<string, unknown>, component: () => unknown) => {
            registered.spec = spec;
            registered.component = component;
            return () => {};
          },
        },
      };
      module.apply(ctx);

      // Render once (loading), let the fetch promise settle, render again (ready).
      const render = (): unknown => {
        slotIndex = 0;
        return registered.component?.();
      };
      render();
      await new Promise((resolve) => setTimeout(resolve, 0));

      return { module, registered, render };
    } finally {
      globals.window = savedWindow;
      globals.fetch = savedFetch;
    }
  }

  it('registers a settings section named after the plugin', async () => {
    const { module, registered } = await loadClient({
      engines: [],
      models: [],
      capabilities: [],
      unavailable: [],
      machine: {},
      generatedAt: '',
    });
    assert.equal(module.name, 'dsh-ai-model-hub');
    assert.deepEqual(module.inject, ['slots']);
    assert.equal(registered.spec?.['name'], 'settings.section');
    assert.equal(registered.spec?.['id'], 'dsh-ai-model-hub');
    assert.equal(typeof registered.spec?.['label'], 'function');
    assert.equal((registered.spec?.['label'] as () => string)(), 'Local models');
  });

  it('renders where each engine is, whether it runs, and what is inside it', async () => {
    const report = await inventory();
    const { render } = await loadClient(report);

    // The loader rendered once while loading; this is the same component rendered
    // again with the state its effect produced, i.e. the settled page.
    const page = flatten(render()).join('\n');

    assert.match(page, /Local models/);
    assert.match(page, /Ollama/);
    assert.match(page, /ComfyUI/);
    assert.match(page, /C:\/ComfyUI/, 'the installation directory is on the page');
    assert.match(page, /C:\/catalog\/models\.json/, 'the active catalog is on the page');
    assert.match(page, /llama3:8b/, "the engine's own models are on the page");
    assert.match(page, /checkpoints\/z_image_turbo\.safetensors/, 'the model store is on the page');
    assert.match(page, /Stable Diffusion WebUI/, 'an engine that is not installed is still listed');
    assert.match(page, /video_generation/, 'what cannot be served is reported');
  });
});
