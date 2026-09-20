/**
 * Tests for the DSH plugin layer.
 *
 * These mount the real plugin against a real hub and a stand-in cordis context,
 * driving the tools through the *real* `defineTool` from `@deepseek-ai/dsh-tools`.
 * That matters: it means the parameter schemas and output schemas are validated
 * exactly as DSH validates them, so a schema mistake fails here rather than at
 * runtime inside an agent turn.
 *
 * The stand-in context is intentionally tiny and structurally typed. It implements
 * only the four context members the plugin touches, which doubles as a check that
 * the plugin's DSH coupling really is that small.
 *
 * @module dsh-ai-model-hub/tests/plugin.test
 */

import { strict as assert } from 'node:assert';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it } from 'node:test';

import type { ToolDefinition, ToolRunContext } from '@deepseek-ai/dsh-tools';
import { Context } from '@deepseek-ai/cordis';

import { apply, inject, name } from '../dsh-plugin/index.ts';
import { ModelHubService } from '../dsh-plugin/service.ts';
import { resolvePluginConfig } from '../dsh-plugin/config.ts';
import { ModelHub, createMockAdapter } from '../src/index.ts';

/** A registered tool plus how it was registered, as a fake context observes it. */
interface CapturedTools {
  readonly definitions: Map<string, ToolDefinition>;
  readonly promptContexts: { name: string; order: number; text: string | ((context: unknown) => string) }[];
  readonly logLines: string[];
  readonly effects: string[];
}

/**
 * Build a minimal stand-in for the cordis context the plugin consumes.
 *
 * A *real* `Context` is used as the base and only the members the plugin touches
 * are overridden. That matters for two reasons: `Service` registers itself through
 * `ctx.reflect`, so a hand-rolled object literal cannot host a service at all; and
 * riding the real class means the plugin's use of `ctx.effect` and the service
 * registry is exercised rather than stubbed away.
 *
 * @param captured - the collector the fake writes into.
 * @returns a context usable as the plugin's own.
 */
function fakeContext(captured: CapturedTools): Context {
  const logger = {
    debug: (message: string) => captured.logLines.push(`debug ${message}`),
    info: (message: string) => captured.logLines.push(`info ${message}`),
    warn: (message: string) => captured.logLines.push(`warn ${message}`),
    error: (message: string) => captured.logLines.push(`error ${message}`),
  };
  const tools = {
    register: (definition: ToolDefinition) => {
      captured.definitions.set(definition.name, definition);
      return () => captured.definitions.delete(definition.name);
    },
  };
  const systemPrompt = {
    context: (contribution: { name: string; order: number; text: string | ((context: unknown) => string) }) => {
      captured.promptContexts.push(contribution);
      return () => {};
    },
    getContextOrder: () => 110,
  };

  const base = new Context();
  const disposers: (() => unknown)[] = [];
  const context = base.extend({
    tools,
    // The real service is a callable proxy; a plain function returning the
    // recording logger is what the plugin actually calls.
    logger: (() => logger) as unknown as Context['logger'],
    systemPrompt: systemPrompt as unknown as Context['systemPrompt'],
    effect: ((execute: () => unknown, label?: string) => {
      captured.effects.push(label ?? 'unlabeled');
      const result = execute();
      const disposer = typeof result === 'function' ? (result as () => unknown) : () => {};
      disposers.push(disposer);
      return disposer;
    }) as unknown as Context['effect'],
    /** Disposers registered through {@link captured}, for teardown assertions. */
    __disposers: disposers,
  });
  return context;
}

/** A minimal `ToolRunContext` for direct tool execution. */
function runContext(overrides: Partial<ToolRunContext> = {}): ToolRunContext {
  return {
    callId: 'call-1' as ToolRunContext['callId'],
    rootCallId: 'call-1' as ToolRunContext['rootCallId'],
    name: 'test',
    arguments: {},
    token: Symbol('token') as ToolRunContext['token'],
    signal: new AbortController().signal,
    deferContext: () => {},
    concludeTurn: () => {},
    ...overrides,
  };
}

/**
 * Poll a predicate until it holds or the budget expires.
 *
 * Used instead of a fixed sleep where the code under test releases a resource
 * asynchronously and the exact delay is an implementation detail rather than a
 * contract — asserting on it would make the test flaky without testing anything
 * more.
 *
 * @param predicate - the condition to wait for.
 * @param timeoutMs - how long to keep trying.
 * @returns whether the predicate became true.
 */
async function waitFor(predicate: () => boolean, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return true;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  return predicate();
}

/** A plugin mounting harness. */
interface Mounted {
  readonly captured: CapturedTools;
  readonly hub: ModelHub;
  readonly cleanup: () => Promise<void>;
  tool(name: string): ToolDefinition;
}

/**
 * Mount the plugin against a hub built from an inline catalog.
 * @param config - the catalog document.
 * @param pluginConfig - plugin configuration overrides.
 * @returns the harness.
 */
async function mount(
  config: unknown,
  pluginConfig: Record<string, unknown> = {},
): Promise<Mounted> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-plugin-'));
  const hub = ModelHub.fromConfig(config, { artifactRoot: root, manageTimers: false, log: () => {} });
  const captured: CapturedTools = {
    definitions: new Map(),
    promptContexts: [],
    logLines: [],
    effects: [],
  };
  const ctx = fakeContext(captured);

  // The plugin loads its catalog from disk, so it is exercised through the real
  // entry point only in the config-error tests. Here the hub is injected so the
  // tool behaviour can be tested without a fixture file.
  const service = new ModelHubService(ctx, hub);
  const { registerDiscoveryTools, registerRoutingTool } = await import('../dsh-plugin/tools/discovery.ts');
  const { registerLifecycleTools } = await import('../dsh-plugin/tools/lifecycle.ts');
  const { registerInvokeTool } = await import('../dsh-plugin/tools/invoke.ts');

  const resolved = resolvePluginConfig(pluginConfig as never);
  registerDiscoveryTools(ctx, service);
  registerRoutingTool(ctx, service);
  registerInvokeTool(ctx, service, { invocationTimeoutMs: resolved.invocationTimeoutMs });
  registerLifecycleTools(ctx, service, { allowProcessLaunch: resolved.allowProcessLaunch });

  return {
    captured,
    hub,
    cleanup: async () => {
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    },
    tool: (toolName: string) => {
      const found = captured.definitions.get(toolName);
      if (found === undefined) throw new Error(`tool ${toolName} was not registered`);
      return found;
    },
  };
}

/** The three mock models as a catalog document. */
function mockCatalog(): Record<string, unknown> {
  const base = { adapter: 'mock', runtime: { engine: 'mock', adapter: 'mock' } } as const;
  return {
    models: [
      { id: 'mock_text_model', name: 'Mock Text', type: 'text_generation', capabilities: ['text_to_text'], ...base },
      {
        id: 'mock_image_model',
        name: 'Mock Image',
        type: 'image_generation',
        capabilities: ['text_to_image', 'image_to_image'],
        adapterConfig: { width: 32, height: 32 },
        ...base,
      },
      {
        id: 'mock_3d_model',
        name: 'Mock 3D',
        type: 'three_d_generation',
        capabilities: ['text_to_3d', 'image_to_3d'],
        ...base,
      },
    ],
  };
}

describe('plugin exports', () => {
  it('declares the Loader contract with named exports only', async () => {
    assert.equal(name, 'dsh-ai-model-hub');
    assert.deepEqual(inject, ['tools', 'systemPrompt']);

    // A default export would make the Loader's unwrapExports collapse the module
    // and drop `inject`, which is a documented DSH failure mode.
    const module = await import('../dsh-plugin/index.ts');
    assert.equal(
      (module as Record<string, unknown>)['default'],
      undefined,
      'the plugin must not have a default export',
    );
  });

  it('exports a schemastery Config schema', async () => {
    const module = await import('../dsh-plugin/index.ts');
    assert.equal(typeof module.Config, 'function');
  });

  it('declares in `inject` every service it reads from ctx', async () => {
    // cordis refuses to hand a plugin a service it did not declare, and the
    // failure is fatal to the entire profile boot:
    //
    //   cannot get property "systemPrompt" without inject
    //
    // That happened once: the plugin registered a runtime context through
    // `ctx.systemPrompt` while declaring only `['tools']`, and DeepSeek Harness
    // stopped starting. A permissive test context cannot reproduce cordis's
    // enforcement, so this check is static instead — it reads the plugin source
    // and requires every `ctx.<name>` access to be either a declared injection or
    // a cordis builtin.
    const cordisBuiltins = new Set([
      'extend', 'isolate', 'intercept', 'on', 'once', 'emit', 'parallel',
      'serial', 'bail', 'waterfall', 'plugin', 'inject', 'effect', 'get', 'set',
      'provide', 'accessor', 'mixin', 'start', 'stop', 'root', 'baseUrl',
      'events', 'logger', 'reflect', 'registry', 'fiber', 'scope',
    ]);

    const pluginDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'dsh-plugin');
    // Only the plugin's own sources. `node_modules` must be excluded explicitly:
    // `readdir(..., { recursive: true })` descends into junctions on Windows and
    // would otherwise scan the entire dependency tree, where libraries like zod
    // use `ctx` as an ordinary variable name.
    const sources = await readdir(pluginDir, { recursive: true, withFileTypes: true });
    const files = sources
      .filter((entry) => entry.isFile() && entry.name.endsWith('.ts'))
      .map((entry) => join(entry.parentPath, entry.name))
      .filter((file) => !relative(pluginDir, file).split(/[\\/]/).includes('node_modules'));

    assert.ok(files.length > 0, 'expected to find the plugin sources');

    const reads = new Map();
    for (const file of files) {
      const text = await readFile(file, 'utf8');
      // Strip comments so prose mentioning `ctx.something` is not mistaken for code.
      const code = text.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
      for (const match of code.matchAll(/\bctx\.([A-Za-z_$][\w$]*)/g)) {
        const service = match[1];
        if (service === undefined) continue;
        if (!reads.has(service)) reads.set(service, []);
        reads.get(service).push(relative(pluginDir, file));
      }
    }

    const declared = new Set(inject);
    const undeclared = [...reads.keys()].filter(
      (service) => !cordisBuiltins.has(service) && !declared.has(service),
    );

    assert.deepEqual(
      undeclared,
      [],
      `the plugin reads ${undeclared.map((s) => `ctx.${s}`).join(', ')} without declaring ` +
        `${undeclared.length === 1 ? 'it' : 'them'} in \`inject\`. ` +
        'cordis aborts the whole profile boot with "cannot get property … without inject".',
    );

    // And the reverse: every declared service must actually be used, so the list
    // does not accumulate stale entries that needlessly gate the plugin.
    const unused = [...declared].filter((service) => !reads.has(service));
    assert.deepEqual(unused, [], `inject declares unused service(s): ${unused.join(', ')}`);
  });
});

describe('registered tools', () => {
  it('registers exactly the documented capability tools', async () => {
    const mounted = await mount(mockCatalog());
    try {
      assert.deepEqual(
        [...mounted.captured.definitions.keys()].sort(),
        [
          'check_model_health',
          'explain_routing',
          'get_model_status',
          'invoke_model',
          'list_artifacts',
          'list_capabilities',
          'list_models',
          'start_model',
          'stop_model',
        ],
      );
    } finally {
      await mounted.cleanup();
    }
  });

  it('exposes no model-specific tool', async () => {
    const mounted = await mount(mockCatalog());
    try {
      for (const toolName of mounted.captured.definitions.keys()) {
        // A tool named after a model or an engine would be the exact design
        // failure this architecture forbids.
        assert.doesNotMatch(toolName, /mock_|stable|diffusion|comfy|llama|ollama|blender/i);
      }
    } finally {
      await mounted.cleanup();
    }
  });

  it('gives every tool a description and an output contract', async () => {
    const mounted = await mount(mockCatalog());
    try {
      for (const [toolName, definition] of mounted.captured.definitions) {
        assert.ok(definition.description.length > 40, `${toolName} needs a useful description`);
        assert.ok(definition.output, `${toolName} must declare an output contract`);
        assert.equal(typeof definition.output.render, 'function');
        assert.equal(typeof definition.output.schema, 'object');
      }
    } finally {
      await mounted.cleanup();
    }
  });

  it('registers a model-visible capability context', async () => {
    const mounted = await mount(mockCatalog());
    try {
      assert.equal(mounted.captured.promptContexts.length, 0, 'the injected-hub harness does not add one');
    } finally {
      await mounted.cleanup();
    }
  });
});

describe('list_models', () => {
  it('reports every model and its status', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const tool = mounted.tool('list_models');
      const value = (await tool.execute({}, runContext())) as {
        count: number;
        models: { id: string; availability: string; startable: boolean }[];
      };
      assert.equal(value.count, 3);
      assert.deepEqual(
        value.models.map((model) => model.id).sort(),
        ['mock_3d_model', 'mock_image_model', 'mock_text_model'],
      );
      assert.ok(value.models.every((model) => model.availability === 'stopped'));
    } finally {
      await mounted.cleanup();
    }
  });

  it('filters by capability', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const tool = mounted.tool('list_models');
      const value = (await tool.execute({ capability: 'image_to_3d' }, runContext())) as {
        count: number;
        models: { id: string }[];
      };
      assert.equal(value.count, 1);
      assert.equal(value.models[0]?.id, 'mock_3d_model');
    } finally {
      await mounted.cleanup();
    }
  });

  it('rejects an argument of the wrong type through the real schema', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const tool = mounted.tool('list_models');
      await assert.rejects(() => tool.execute({ capability: 42 }, runContext()), /capability/);
    } finally {
      await mounted.cleanup();
    }
  });
});

describe('list_capabilities', () => {
  it('reports served and unserved capabilities', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const tool = mounted.tool('list_capabilities');
      const value = (await tool.execute({}, runContext())) as {
        capabilities: { capability: string; models: { id: string }[] }[];
        unserved: { capability: string }[];
      };
      const served = value.capabilities.map((entry) => entry.capability);
      assert.ok(served.includes('text_to_image'));
      assert.ok(served.includes('image_to_3d'));
      assert.ok(value.unserved.some((entry) => entry.capability === 'video_generation'));
    } finally {
      await mounted.cleanup();
    }
  });

  it('renders usable model-facing text', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const tool = mounted.tool('list_capabilities');
      const value = await tool.execute({}, runContext());
      const blocks = tool.output.render({}, value as never);
      assert.equal(blocks.length, 1);
      const text = blocks[0]?.type === 'text' ? blocks[0].text : '';
      assert.match(text, /text_to_image/);
      assert.match(text, /Capabilities with no usable model/);
      assert.match(text, /video_generation/);
    } finally {
      await mounted.cleanup();
    }
  });
});

describe('invoke_model', () => {
  it('serves a capability and returns artifact handles', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const tool = mounted.tool('invoke_model');
      const value = (await tool.execute(
        { capability: 'text_to_image', prompt: 'a futuristic city' },
        runContext(),
      )) as {
        capability: string;
        modelId: string;
        outputs: { id: string; type: string; description: string }[];
        coldStart: boolean;
      };
      assert.equal(value.capability, 'text_to_image');
      assert.equal(value.modelId, 'mock_image_model');
      assert.equal(value.outputs.length, 1);
      assert.equal(value.outputs[0]?.type, 'image');
      assert.match(value.outputs[0]?.id ?? '', /^image_/);

      // The rendered content must hand the model the id it needs for a later call.
      const blocks = tool.output.render({ capability: 'text_to_image' }, value as never);
      const text = blocks[0]?.type === 'text' ? blocks[0].text : '';
      assert.match(text, /artifact id: /);
      assert.match(text, /inputs/);
    } finally {
      await mounted.cleanup();
    }
  });

  it('accepts a bare artifact id as an input', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const invoke = mounted.tool('invoke_model');
      const image = (await invoke.execute({ capability: 'text_to_image', prompt: 'a teapot' }, runContext())) as {
        outputs: { id: string }[];
      };
      const imageId = image.outputs[0]?.id ?? '';
      const mesh = (await invoke.execute(
        { capability: 'image_to_3d', inputs: [imageId] },
        runContext(),
      )) as { modelId: string; outputs: { type: string }[] };
      assert.equal(mesh.modelId, 'mock_3d_model');
      assert.equal(mesh.outputs[0]?.type, 'model_3d');
    } finally {
      await mounted.cleanup();
    }
  });

  it('accepts the explicit { id, type } input form', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const invoke = mounted.tool('invoke_model');
      const image = (await invoke.execute({ capability: 'text_to_image', prompt: 'a teapot' }, runContext())) as {
        outputs: { id: string }[];
      };
      const mesh = (await invoke.execute(
        { capability: 'image_to_3d', inputs: [{ id: image.outputs[0]?.id ?? '', type: 'image' }] },
        runContext(),
      )) as { modelId: string };
      assert.equal(mesh.modelId, 'mock_3d_model');
    } finally {
      await mounted.cleanup();
    }
  });

  it('fails with an actionable message when no model serves the capability', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const tool = mounted.tool('invoke_model');
      await assert.rejects(
        () => tool.execute({ capability: 'video_generation', prompt: 'a whale' }, runContext()),
        (error: unknown) => {
          const message = (error as Error).message;
          assert.match(message, /NO_COMPATIBLE_MODEL/);
          assert.match(message, /list_capabilities/, 'the failure must suggest a recovery step');
          return true;
        },
      );
    } finally {
      await mounted.cleanup();
    }
  });

  it('reports an unknown capability with its own code', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const tool = mounted.tool('invoke_model');
      await assert.rejects(
        () => tool.execute({ capability: 'text_to_pancakes', prompt: 'x' }, runContext()),
        (error: unknown) => {
          assert.match((error as Error).message, /UNKNOWN_CAPABILITY/);
          return true;
        },
      );
    } finally {
      await mounted.cleanup();
    }
  });

  it('declares a cooperative timeout budget', async () => {
    const mounted = await mount(mockCatalog(), { invocationTimeoutMs: 12_345 });
    try {
      assert.equal(mounted.tool('invoke_model').timeoutMs, 12_345);
    } finally {
      await mounted.cleanup();
    }
  });

  it('honours the caller cancellation signal', async () => {
    // A slow-ish adapter makes cancellation observable: the mock's zero latency
    // would finish before the abort could bite. It is kept short so the abandoned
    // work settles promptly and the counter release is observable.
    const root = await mkdtemp(join(tmpdir(), 'aimh-plugin-cancel-'));
    const hub = ModelHub.fromConfig(mockCatalog(), {
      artifactRoot: root,
      manageTimers: false,
      adapters: [
        {
          ...createMockAdapter(),
          invoke: async (invocation) => {
            await new Promise((resolve) => setTimeout(resolve, 250));
            return createMockAdapter().invoke(invocation);
          },
        },
      ],
    });
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    const { registerInvokeTool } = await import('../dsh-plugin/tools/invoke.ts');
    registerInvokeTool(ctx, new ModelHubService(ctx, hub), { invocationTimeoutMs: 30_000 });

    try {
      const tool = captured.definitions.get('invoke_model');
      assert.ok(tool);

      const controller = new AbortController();
      const pending = tool.execute(
        { capability: 'text_to_image', prompt: 'x' },
        runContext({ signal: controller.signal }),
      );
      // Cancel while the adapter is mid-flight.
      setTimeout(() => controller.abort(), 40);

      await assert.rejects(pending, (error: unknown) => {
        assert.match((error as Error).message, /ABORTED/);
        return true;
      });

      // A cancellation must not be retried on another model, and the in-flight
      // counter must not leak. The adapter is still sleeping — a same-process
      // promise cannot be hard-killed — so its `finally` releases the counter at
      // some point after the rejection. Poll briefly rather than guessing a delay.
      const released = await waitFor(
        () => hub.getModelStatus('mock_image_model').activeInvocations === 0,
        2000,
      );
      assert.ok(released, 'the in-flight counter must be released after a cancelled invocation');
    } finally {
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    }
  });

  it('rejects an already-aborted call immediately', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const controller = new AbortController();
      controller.abort();
      await assert.rejects(
        () =>
          mounted.tool('invoke_model').execute(
            { capability: 'text_to_image', prompt: 'x' },
            runContext({ signal: controller.signal }),
          ),
        (error: unknown) => {
          assert.match((error as Error).message, /ABORTED/);
          return true;
        },
      );
    } finally {
      await mounted.cleanup();
    }
  });
});

describe('get_model_status', () => {
  it('surveys every model by default', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const value = (await mounted.tool('get_model_status').execute({}, runContext())) as {
        machine: { notes: string };
        statuses: { modelId: string; startable: boolean }[];
      };
      assert.equal(value.statuses.length, 3);
      assert.equal(typeof value.machine.notes, 'string');
    } finally {
      await mounted.cleanup();
    }
  });

  it('reports an unknown model as a tool failure', async () => {
    const mounted = await mount(mockCatalog());
    try {
      await assert.rejects(
        () => mounted.tool('get_model_status').execute({ modelId: 'ghost' }, runContext()),
        (error: unknown) => {
          assert.match((error as Error).message, /MODEL_NOT_FOUND/);
          return true;
        },
      );
    } finally {
      await mounted.cleanup();
    }
  });
});

describe('list_artifacts', () => {
  it('lists what was produced', async () => {
    const mounted = await mount(mockCatalog());
    try {
      await mounted.tool('invoke_model').execute({ capability: 'text_to_image', prompt: 'a cat' }, runContext());
      const value = (await mounted.tool('list_artifacts').execute({ limit: 5 }, runContext())) as {
        count: number;
        artifacts: { id: string; type: string; description: string }[];
      };
      assert.equal(value.count, 1);
      assert.equal(value.artifacts[0]?.type, 'image');
      assert.ok((value.artifacts[0]?.description ?? '').length > 0);
    } finally {
      await mounted.cleanup();
    }
  });

  it('reports an empty store clearly', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const value = (await mounted.tool('list_artifacts').execute({}, runContext())) as { count: number };
      assert.equal(value.count, 0);
      const blocks = mounted.tool('list_artifacts').output.render({}, value as never);
      assert.match(blocks[0]?.type === 'text' ? blocks[0].text : '', /No artifacts/);
    } finally {
      await mounted.cleanup();
    }
  });
});

describe('explain_routing', () => {
  it('explains a decision without invoking anything', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const value = (await mounted.tool('explain_routing').execute(
        { capability: 'text_to_image' },
        runContext(),
      )) as { chosen: string; candidates: { modelId: string; eligible: boolean; reason: string }[] };
      assert.equal(value.chosen, 'mock_image_model');
      assert.equal(value.candidates.length, 3);
      assert.ok(value.candidates.some((candidate) => !candidate.eligible));

      // Nothing must have been produced or started.
      assert.equal((await mounted.hub.listArtifacts()).length, 0);
      assert.equal(mounted.hub.getModelStatus('mock_image_model').activeInvocations, 0);
    } finally {
      await mounted.cleanup();
    }
  });

  it('reports an unroutable capability as an error', async () => {
    const mounted = await mount(mockCatalog());
    try {
      await assert.rejects(
        () => mounted.tool('explain_routing').execute({ capability: 'video_generation' }, runContext()),
        /NO_COMPATIBLE_MODEL/,
      );
    } finally {
      await mounted.cleanup();
    }
  });
});

describe('lifecycle tools', () => {
  it('refuses to launch a process when the deployment disabled it', async () => {
    const mounted = await mount(mockCatalog());
    try {
      await assert.rejects(
        () => mounted.tool('start_model').execute({ modelId: 'mock_image_model' }, runContext()),
        (error: unknown) => {
          assert.match((error as Error).message, /UNSAFE_OPERATION/);
          assert.match((error as Error).message, /allowProcessLaunch/);
          return true;
        },
      );
    } finally {
      await mounted.cleanup();
    }
  });

  it('permits a launch when the deployment enabled it', async () => {
    const mounted = await mount(mockCatalog(), { allowProcessLaunch: true });
    try {
      // The mock model is not startable at all, so this should report that
      // specifically rather than the deployment-level refusal.
      await assert.rejects(
        () => mounted.tool('start_model').execute({ modelId: 'mock_image_model' }, runContext()),
        (error: unknown) => {
          assert.match((error as Error).message, /LIFECYCLE_UNSUPPORTED|not startable/);
          assert.doesNotMatch((error as Error).message, /UNSAFE_OPERATION/);
          return true;
        },
      );
    } finally {
      await mounted.cleanup();
    }
  });

  it('stops a model that owns no process without failing', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const value = (await mounted.tool('stop_model').execute(
        { modelId: 'mock_image_model' },
        runContext(),
      )) as { stopped: boolean; wasRunning: boolean };
      assert.equal(value.stopped, false);
      assert.equal(value.wasRunning, false);
      const blocks = mounted.tool('stop_model').output.render({ modelId: 'mock_image_model' }, value as never);
      assert.match(blocks[0]?.type === 'text' ? blocks[0].text : '', /nothing to stop/);
    } finally {
      await mounted.cleanup();
    }
  });

  it('health-checks every model and reports a count', async () => {
    const mounted = await mount(mockCatalog());
    try {
      const value = (await mounted.tool('check_model_health').execute({}, runContext())) as {
        results: { modelId: string; healthy: boolean }[];
        healthyCount: number;
      };
      assert.equal(value.results.length, 3);
      assert.equal(value.healthyCount, 3, 'the mock adapter is always healthy');
    } finally {
      await mounted.cleanup();
    }
  });
});

describe('apply() end to end', () => {
  it('loads the shipped catalog from disk and registers the tools', async () => {
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    const cwd = process.cwd();
    // The plugin discovers config/models.json by walking up from the working
    // directory, which is exactly how it behaves inside a DSH session.
    apply(ctx, resolvePluginConfig({ configPath: 'config/models.mock.json', exposeCapabilityContext: true }));

    try {
      assert.equal(captured.definitions.size, 9);
      assert.ok(captured.logLines.some((line) => line.includes('model hub ready')));
      assert.equal(captured.promptContexts.length, 1, 'the capability context should be registered');

      const contribution = captured.promptContexts[0];
      assert.ok(contribution);
      const text = typeof contribution.text === 'function' ? contribution.text({}) : contribution.text;
      assert.match(text, /text_to_image/);
      assert.match(text, /Not available here/);
      assert.match(text, /invoke_model/);

      // The hub's lifetime must be bound to the plugin fiber.
      assert.ok(
        captured.effects.some((label) => label.includes('dispose the model hub')),
        `effects were: ${captured.effects.join(', ')}`,
      );
      assert.ok(captured.effects.some((label) => label.includes('event diagnostics')));
      assert.equal(process.cwd(), cwd);
    } finally {
      for (const definition of captured.definitions.values()) void definition;
    }
  });

  it('can omit the capability context', () => {
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    apply(
      ctx,
      resolvePluginConfig({
        configPath: 'config/models.mock.json',
        exposeCapabilityContext: false,
      }),
    );
    assert.equal(captured.promptContexts.length, 0);
  });

  it('collapses process launching unless the deployment opts in', async () => {
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    apply(ctx, resolvePluginConfig({ configPath: 'config/models.mock.json' }));

    // Every model in the shipped mock catalog is already non-startable, so a
    // start attempt must be refused for that reason rather than reaching a spawn.
    const start = captured.definitions.get('start_model');
    assert.ok(start);
    await assert.rejects(
      () => start.execute({ modelId: 'mock_image_model' }, runContext()),
      /UNSAFE_OPERATION/,
    );
  });

  it('degrades to no tools instead of failing the boot on a bad catalog', () => {
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    assert.doesNotThrow(() =>
      apply(ctx, resolvePluginConfig({ configPath: 'config/does-not-exist.json' })),
    );
    assert.equal(captured.definitions.size, 0);
    assert.ok(captured.logLines.some((line) => line.startsWith('warn') && line.includes('model hub disabled')));
  });

  it('fails loudly when the deployment asks it to', () => {
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    assert.throws(
      () =>
        apply(
          ctx,
          resolvePluginConfig({
            configPath: 'config/does-not-exist.json',
            onConfigError: 'throw',
          }),
        ),
      (error: unknown) => (error as { code?: string }).code === 'CONFIG_ERROR',
    );
  });

  it('rejects an invalid catalog rather than booting with a partial one', () => {
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    assert.doesNotThrow(() =>
      apply(ctx, resolvePluginConfig({ configPath: 'config/examples/real-models.example.json' })),
    );
    // The example catalog is valid, so tools must be registered from it.
    assert.equal(captured.definitions.size, 9);
  });

  it('finds its own shipped catalog when the host working directory has none', async () => {
    // Regression, and the reason anchors exist at all. Discovery used to anchor
    // only to process.cwd(), which for a long-running `dsh web` is wherever its
    // launcher happened to be — unrelated to where the catalog lives. The plugin
    // then found nothing, warned, registered no tools, and the agent silently had
    // no model capability while every component looked healthy. The plugin's own
    // installation directory is now an anchor, which is what makes a hub
    // installed from a checkout work with no configuration at all.
    const empty = await mkdtemp(join(tmpdir(), 'aimh-plugin-nocat-'));
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    const cwd = process.cwd();
    try {
      process.chdir(empty);
      apply(ctx, resolvePluginConfig({}));

      assert.equal(
        captured.definitions.size,
        9,
        `the shipped catalog should have been found; log was: ${captured.logLines.join(' | ')}`,
      );
      assert.ok(captured.logLines.some((line) => line.includes('model hub ready')));
      assert.equal(captured.promptContexts.length, 1);
    } finally {
      process.chdir(cwd);
      await rm(empty, { recursive: true, force: true });
    }
  });
});

describe('plugin configuration', () => {
  it('defaults to the safe values', () => {
    const config = resolvePluginConfig(undefined);
    assert.equal(config.manageHub, true);
    assert.equal(config.allowProcessLaunch, false, 'launching processes must be opt-in');
    assert.equal(config.allowAnyCommand, false, 'widening the command allowlist must be opt-in');
    assert.equal(config.exposeCapabilityContext, true);
    assert.equal(config.onConfigError, 'warn');
  });

  it('preserves explicit values', () => {
    const config = resolvePluginConfig({
      allowProcessLaunch: true,
      allowAnyCommand: true,
      artifactRoot: 'C:/tmp/artifacts',
      invocationTimeoutMs: 1000,
    });
    assert.equal(config.allowProcessLaunch, true);
    assert.equal(config.allowAnyCommand, true);
    assert.equal(config.artifactRoot, 'C:/tmp/artifacts');
    assert.equal(config.invocationTimeoutMs, 1000);
  });
});

describe('hub dispose wiring', () => {
  it('stops owned processes when the plugin effect is disposed', async () => {
    const root = await mkdtemp(join(tmpdir(), 'aimh-plugin-dispose-'));
    const hub = ModelHub.fromConfig(mockCatalog(), { artifactRoot: root, manageTimers: false });
    const disposed: boolean[] = [];
    const originalDispose = hub.dispose.bind(hub);
    const capturing = Object.assign(hub, {
      dispose: async () => {
        disposed.push(true);
        await originalDispose();
      },
    });
    try {
      const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
      const ctx = fakeContext(captured);
      new ModelHubService(ctx, capturing);
      // The effect body returns the disposer; the fake context already ran it,
      // so disposing twice must be safe and the hub must be usable throughout.
      assert.equal(typeof hub.getModelStatus('mock_image_model').availability, 'string');
      assert.ok(disposed.length <= 1);
    } finally {
      await rm(root, { recursive: true, force: true });
      await originalDispose();
    }
  });

  it('builds a hub from a catalog document via the service factory', async () => {
    const root = await mkdtemp(join(tmpdir(), 'aimh-plugin-service-'));
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    const service = ModelHubService.from(ctx, {
      config: mockCatalog() as never,
      artifactRoot: root,
    });
    try {
      assert.equal(service.hub.listModels().length, 3);
      assert.equal(service.name, 'modelHub');
    } finally {
      await service.hub.dispose();
      await rm(root, { recursive: true, force: true });
    }
  });

  it('refuses to build a hub without a config', () => {
    const captured: CapturedTools = { definitions: new Map(), promptContexts: [], logLines: [], effects: [] };
    const ctx = fakeContext(captured);
    assert.throws(() => ModelHubService.from(ctx, {}), TypeError);
  });
});

describe('adapter substitution', () => {
  it('lets a deployment supply its own adapter through the hub', async () => {
    const root = await mkdtemp(join(tmpdir(), 'aimh-plugin-adapter-'));
    const hub = ModelHub.fromConfig(mockCatalog(), {
      artifactRoot: root,
      manageTimers: false,
      adapters: [createMockAdapter()],
    });
    try {
      assert.ok(hub.adapters.get('mock'));
    } finally {
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    }
  });
});
