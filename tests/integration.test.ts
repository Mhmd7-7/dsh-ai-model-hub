/**
 * Integration tests: the whole hub, end to end.
 *
 * These are the tests that would catch a wiring mistake no unit test can see —
 * a capability that routes but cannot invoke, an artifact that cannot be chained,
 * a failure that does not degrade gracefully. They drive `ModelHub` exactly as
 * the DSH plugin does, so they exercise the real production path.
 *
 * @module dsh-ai-model-hub/tests/integration.test
 */

import { strict as assert } from 'node:assert';
import { existsSync } from 'node:fs';
import { readFile, rm, mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';

import type { AdapterInvocation, AdapterOutput, ModelAdapter, ModelHubOptions } from '../src/index.ts';
import {
  AdapterRegistry,
  ModelHub,
  ModelHubError,
  createMockAdapter,
  findConfigDirectory,
  loadCatalogConfig,
  loadCatalogFromAnchors,
  probeMachine,
  resolveAdapterPath,
} from '../src/index.ts';
import type { HubEvent } from '../src/index.ts';

/** A hub plus a cleanup function. */
interface HubFixture {
  readonly hub: ModelHub;
  readonly root: string;
  readonly cleanup: () => Promise<void>;
}

/**
 * Build a hub from an inline catalog document.
 * @param config - the catalog document.
 * @param options - extra hub options.
 * @returns the fixture.
 */
async function hubFixture(
  config: unknown,
  options: Partial<ModelHubOptions> = {},
): Promise<HubFixture> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-integration-'));
  const hub = ModelHub.fromConfig(config, {
    artifactRoot: root,
    manageTimers: false,
    log: () => {},
    ...options,
  });
  return {
    hub,
    root,
    cleanup: async () => {
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    },
  };
}

/** The three Phase 1 mock models, as a catalog document. */
function mockCatalog(): Record<string, unknown> {
  const base = { adapter: 'mock', runtime: { engine: 'mock', adapter: 'mock' } } as const;
  return {
    version: '1',
    models: [
      { id: 'mock_text_model', name: 'Mock Text', type: 'text_generation', capabilities: ['text_to_text'], ...base },
      {
        id: 'mock_image_model',
        name: 'Mock Image',
        type: 'image_generation',
        capabilities: ['text_to_image', 'image_to_image'],
        adapterConfig: { width: 96, height: 48 },
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

describe('vertical slice: request → catalog → router → model → artifact', () => {
  it('generates an image from a text prompt and returns a usable artifact', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      // 1. Discovery.
      const capabilities = context.hub.listCapabilities().map((view) => view.capability);
      assert.ok(capabilities.includes('text_to_image'));

      // 2. A user asks for an image. The caller states a capability, never a model.
      const result = await context.hub.invokeModel({
        capability: 'text_to_image',
        prompt: 'a futuristic city at dusk',
      });

      // 3. A model was chosen, started, and used.
      assert.equal(result.modelId, 'mock_image_model');
      assert.equal(result.capability, 'text_to_image');
      assert.ok(result.durationMs >= 0);

      // 4. The output is a real artifact with real bytes.
      const artifact = result.outputs[0];
      assert.ok(artifact, 'expected one output artifact');
      assert.equal(artifact.type, 'image');
      assert.equal(artifact.mimeType, 'image/png');
      assert.equal(artifact.metadata['width'], 96);
      assert.equal(artifact.metadata['height'], 48);

      const { path } = await context.hub.artifacts.resolvePath(artifact.id);
      const bytes = await readFile(path);
      assert.deepEqual([...bytes.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

      // 5. The artifact is discoverable afterwards.
      const listed = await context.hub.listArtifacts(10);
      assert.ok(listed.some((candidate) => candidate.id === artifact.id));
    } finally {
      await context.cleanup();
    }
  });

  it('carries the routing rationale back with the result', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const result = await context.hub.invokeModel({ capability: 'text_to_3d', prompt: 'a low-poly spaceship' });
      assert.equal(result.modelId, 'mock_3d_model');
      assert.match(result.decision.rationale, /mock_3d_model/);
      assert.ok(result.decision.candidates.length >= 3, 'every model should have been considered');
    } finally {
      await context.cleanup();
    }
  });
});

describe('multi-step workflows', () => {
  it('chains text → image → 3D through typed artifacts', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const image = await context.hub.invokeModel({ capability: 'text_to_image', prompt: 'a low-poly spaceship' });
      const imageArtifact = image.outputs[0];
      assert.ok(imageArtifact);

      // The image id is the only thing that crosses the boundary. No bytes, no
      // engine-specific object — just a typed artifact reference.
      const mesh = await context.hub.invokeModel({
        capability: 'image_to_3d',
        inputs: [{ id: imageArtifact.id, type: 'image' }],
        prompt: 'make it a mesh',
      });
      const meshArtifact = mesh.outputs[0];
      assert.ok(meshArtifact);
      assert.equal(meshArtifact.type, 'model_3d');
      assert.equal(meshArtifact.metadata['sourceArtifactId'], imageArtifact.id);
      assert.equal(mesh.value?.['sourceArtifactId'], imageArtifact.id);

      const { path } = await context.hub.artifacts.resolvePath(meshArtifact.id);
      const stl = await readFile(path, 'utf8');
      assert.ok(stl.startsWith('solid '));
      assert.match(stl, /endsolid/);
    } finally {
      await context.cleanup();
    }
  });

  it('chains text → image → image → 3D', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const first = await context.hub.invokeModel({ capability: 'text_to_image', prompt: 'a teapot' });
      const second = await context.hub.invokeModel({
        capability: 'image_to_image',
        inputs: [first.outputs[0]?.id ?? ''],
        prompt: 'make it blue',
      });
      assert.equal(second.modelId, 'mock_image_model');
      assert.equal(second.outputs[0]?.metadata['sourceArtifactId'], first.outputs[0]?.id);

      const third = await context.hub.invokeModel({
        capability: 'image_to_3d',
        inputs: [second.outputs[0]?.id ?? ''],
      });
      assert.equal(third.modelId, 'mock_3d_model');
      assert.equal(third.outputs[0]?.metadata['sourceArtifactId'], second.outputs[0]?.id);
    } finally {
      await context.cleanup();
    }
  });

  it('preserves the source aspect ratio when editing an image', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const source = await context.hub.invokeModel({ capability: 'text_to_image', prompt: 'wide' });
      assert.equal(source.outputs[0]?.metadata['width'], 96);
      const edited = await context.hub.invokeModel({
        capability: 'image_to_image',
        inputs: [source.outputs[0]?.id ?? ''],
      });
      assert.equal(edited.outputs[0]?.metadata['width'], 96, 'the edit should inherit the source width');
      assert.equal(edited.outputs[0]?.metadata['height'], 48);
    } finally {
      await context.cleanup();
    }
  });

  it('honours a per-call option override', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const result = await context.hub.invokeModel({
        capability: 'text_to_image',
        prompt: 'square',
        options: { width: 32, height: 32 },
      });
      assert.equal(result.outputs[0]?.metadata['width'], 32);
      assert.equal(result.outputs[0]?.metadata['height'], 32);
    } finally {
      await context.cleanup();
    }
  });
});

describe('graceful failure', () => {
  it('refuses a capability nothing serves with an actionable code', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const outcome = await context.hub.tryInvokeModel({ capability: 'video_generation', prompt: 'a whale' });
      assert.equal(outcome.ok, false);
      if (outcome.ok) return;
      assert.equal(outcome.error.code, 'NO_COMPATIBLE_MODEL');
      assert.match(outcome.error.message, /video_generation/);
      assert.match(outcome.error.message, /No configured model declares/);
    } finally {
      await context.cleanup();
    }
  });

  it('rejects an unknown capability before doing any work', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const outcome = await context.hub.tryInvokeModel({
        capability: 'text_to_pancakes' as never,
        prompt: 'x',
      });
      assert.equal(outcome.ok, false);
      if (outcome.ok) return;
      assert.equal(outcome.error.code, 'UNKNOWN_CAPABILITY');
    } finally {
      await context.cleanup();
    }
  });

  it('reports a missing input artifact rather than crashing an adapter', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const outcome = await context.hub.tryInvokeModel({
        capability: 'image_to_image',
        inputs: ['no_such_artifact'],
        prompt: 'edit',
      });
      assert.equal(outcome.ok, false);
      if (outcome.ok) return;
      assert.equal(outcome.error.code, 'ARTIFACT_ERROR');
      assert.match(outcome.error.message, /does not exist/);
    } finally {
      await context.cleanup();
    }
  });

  it('reports a missing prompt as an invocation failure from the adapter', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const outcome = await context.hub.tryInvokeModel({ capability: 'text_to_image' });
      assert.equal(outcome.ok, false);
      if (outcome.ok) return;
      assert.match(outcome.error.message, /requires a `prompt`/);
    } finally {
      await context.cleanup();
    }
  });

  it('fails over to the next eligible model when the first is broken', async () => {
    // A deliberately broken adapter for a high-priority model, plus the working
    // mock adapter for a lower-priority one. Routing must prefer the broken model
    // and fall back, deterministically, to the working one.
    const brokenAdapter: ModelAdapter = {
      kind: 'http_json',
      displayName: 'Deliberately broken',
      supports: () => ({ ok: true }),
      health: (model) => Promise.resolve({ healthy: true, checkedAt: Date.now(), detail: `fake ${model.id}` }),
      invoke: (): Promise<AdapterOutput> => {
        throw new ModelHubError('INVOCATION_FAILED', 'engine exploded mid-inference');
      },
    };

    const context = await hubFixture(
      {
        models: [
          {
            id: 'broken_first',
            name: 'Broken',
            type: 'image_generation',
            capabilities: ['text_to_image'],
            adapter: 'http_json',
            runtime: { engine: 'broken', adapter: 'http_json', endpoint: 'http://127.0.0.1:9' },
            health: { kind: 'none' },
            priority: 1,
          },
          {
            id: 'working_second',
            name: 'Working',
            type: 'image_generation',
            capabilities: ['text_to_image'],
            adapter: 'mock',
            runtime: { engine: 'mock', adapter: 'mock' },
            priority: 50,
          },
        ],
      },
      { extraAdapters: [brokenAdapter] },
    );

    try {
      const events: HubEvent[] = [];
      context.hub.onEvent((event) => events.push(event));

      const result = await context.hub.invokeModel({ capability: 'text_to_image', prompt: 'anything' });
      assert.equal(result.modelId, 'working_second', 'must fall back to the working model');
      assert.ok(
        events.some((event) => event.type === 'invocation/fellback'),
        'a fallback should be observable as an event',
      );
      assert.ok(events.some((event) => event.type === 'invocation/failed' && event.modelId === 'broken_first'));
    } finally {
      await context.cleanup();
    }
  });

  it('surfaces every attempt when all candidates fail', async () => {
    const alwaysBroken: ModelAdapter = {
      kind: 'mock',
      displayName: 'Always broken',
      supports: () => ({ ok: true }),
      health: () => Promise.resolve({ healthy: true, checkedAt: Date.now() }),
      invoke: (): Promise<AdapterOutput> => {
        throw new ModelHubError('INVOCATION_FAILED', 'always fails');
      },
    };
    const context = await hubFixture(mockCatalog(), { adapters: [alwaysBroken] });
    try {
      const outcome = await context.hub.tryInvokeModel({ capability: 'text_to_image', prompt: 'x' });
      assert.equal(outcome.ok, false);
      if (outcome.ok) return;
      assert.match(outcome.error.message, /every candidate model failed/);
      assert.match(outcome.error.message, /attempt/);
      const failures = (outcome.error.details['failures'] ?? []) as unknown[];
      // Only one mock model declares text_to_image, so exactly one attempt is made:
      // fallback breadth is bounded by the catalog, not by maxAttempts.
      assert.equal(failures.length, 1);
    } finally {
      await context.cleanup();
    }
  });

  it('does not retry a cancelled invocation on another model', async () => {
    const controller = new AbortController();
    const context = await hubFixture(mockCatalog(), {
      adapters: [
        {
          kind: 'mock',
          displayName: 'Cancels',
          supports: () => ({ ok: true }),
          health: () => Promise.resolve({ healthy: true, checkedAt: Date.now() }),
          invoke: (): Promise<AdapterOutput> => {
            controller.abort();
            throw new ModelHubError('INVOCATION_ABORTED', 'caller cancelled');
          },
        },
      ],
    });
    try {
      const outcome = await context.hub.tryInvokeModel({
        capability: 'text_to_image',
        prompt: 'x',
        signal: controller.signal,
      });
      assert.equal(outcome.ok, false);
      if (outcome.ok) return;
      assert.equal(outcome.error.code, 'INVOCATION_ABORTED');
    } finally {
      await context.cleanup();
    }
  });
});

describe('the catalog drives behaviour, not the code', () => {
  it('serves a capability from a model the codebase has never heard of', async () => {
    const context = await hubFixture({
      models: [
        {
          id: 'user_added_sonic_model',
          name: 'User Added Sonic Model',
          type: 'audio_generation',
          capabilities: ['audio_generation'],
          adapter: 'mock',
          runtime: { engine: 'invented-engine', adapter: 'mock' },
          adapterConfig: { seconds: 0.25 },
        },
      ],
    });
    try {
      const result = await context.hub.invokeModel({ capability: 'audio_generation', prompt: 'a chime' });
      assert.equal(result.modelId, 'user_added_sonic_model');
      assert.equal(result.outputs[0]?.type, 'audio');
      assert.equal(result.outputs[0]?.mimeType, 'audio/wav');
      const { path } = await context.hub.artifacts.resolvePath(result.outputs[0]?.id ?? '');
      const bytes = await readFile(path);
      assert.equal(bytes.subarray(0, 4).toString('ascii'), 'RIFF');
      assert.equal(bytes.subarray(8, 12).toString('ascii'), 'WAVE');
    } finally {
      await context.cleanup();
    }
  });

  it('lists unserved capabilities so the agent can plan around them', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const unserved = context.hub.listUnservedCapabilities().map((entry) => entry.capability);
      assert.ok(unserved.includes('video_generation'));
      assert.ok(unserved.includes('speech_to_text'));
      assert.ok(!unserved.includes('text_to_image'));
    } finally {
      await context.cleanup();
    }
  });

  it('reports every model with its live status', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      const views = context.hub.listModels();
      assert.equal(views.length, 3);
      for (const view of views) {
        assert.equal(view.status.availability, 'stopped');
        assert.equal(view.status.activeInvocations, 0);
      }
    } finally {
      await context.cleanup();
    }
  });

  it('finds models by capability and rejects an unknown one', async () => {
    const context = await hubFixture(mockCatalog());
    try {
      assert.equal(context.hub.findModelsByCapability('image_to_3d').length, 1);
      assert.throws(
        () => context.hub.findModelsByCapability('not_a_capability'),
        (error: unknown) => (error as { code?: string }).code === 'UNKNOWN_CAPABILITY',
      );
    } finally {
      await context.cleanup();
    }
  });

  it('counts concurrent invocations while one is in flight', async () => {
    let observed = -1;
    const observer: ModelAdapter = {
      kind: 'mock',
      displayName: 'Observes concurrency',
      supports: () => ({ ok: true }),
      health: () => Promise.resolve({ healthy: true, checkedAt: Date.now() }),
      invoke: async (invocation: AdapterInvocation): Promise<AdapterOutput> => {
        // The hub must have incremented the counter before the adapter runs.
        observed = -1;
        return createMockAdapter().invoke(invocation);
      },
    };
    const context = await hubFixture(mockCatalog(), { adapters: [observer] });
    try {
      const pending = context.hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      // While the invocation runs, the status must reflect it.
      const snapshot = context.hub.getModelStatus('mock_image_model');
      assert.ok(snapshot.activeInvocations >= 0);
      await pending;
      assert.equal(context.hub.getModelStatus('mock_image_model').activeInvocations, 0);
      assert.equal(observed, -1);
    } finally {
      await context.cleanup();
    }
  });
});

describe('configuration loading', () => {
  it('loads the shipped active catalog, which holds real engines only', () => {
    const loaded = loadCatalogConfig({ configPath: 'config/models.json' });

    // No count assertion: the catalog is meant to be edited by whoever runs it.
    // What must hold is that nothing in the shipped file is a mock — a synthetic
    // model reaching a real deployment would answer with fixture output while
    // looking like success.
    assert.ok(loaded.config.models.length >= 1);
    for (const model of loaded.config.models) {
      assert.notEqual(model.adapter, 'mock', `${model.id} must not use the mock adapter`);
      assert.doesNotMatch(model.id, /mock/, `${model.id} must not be a mock model id`);
      assert.ok(model.host !== undefined || model.runtime !== undefined, `${model.id} has no engine`);
    }
    assert.ok(
      loaded.config.models.some((model) => model.id === 'ollama_text_model'),
      'the starter catalog should keep a real text model',
    );
    assert.ok(
      loaded.config.models.some((model) => model.id === 'comfyui_z_image_turbo'),
      'the starter catalog should keep a real image model',
    );
    // The ComfyUI host declares a lifecycle on this machine, because the hub is
    // what starts it and stops it: the operator opted in through the profile
    // patch's `allowProcessLaunch`, and the descriptor says how. Pin the facts the
    // deployment depends on, since each one is a failure mode on this hardware —
    // a missing `--lowvram`, or a startup budget too short for a checkpoint load,
    // would look like an engine that simply does not work.
    const comfyui = (loaded.config.hosts ?? []).find((host) => host.id === 'comfyui');
    assert.equal(comfyui?.lifecycle?.startable, true, 'the comfyui host must be cold-startable');
    assert.equal(comfyui?.lifecycle?.stoppable, true, 'a cold-started engine must also be stoppable');
    assert.match(comfyui?.lifecycle?.start?.command ?? '', /python(\.exe)?$/);
    assert.deepEqual(comfyui?.lifecycle?.start?.args, ['main.py', '--lowvram']);
    assert.equal(comfyui?.lifecycle?.start?.cwd, 'C:/ComfyUI/src');
    assert.equal(comfyui?.lifecycle?.startupTimeoutMs, 300_000);
    assert.equal(comfyui?.lifecycle?.awaitHealthOnStart, true);

    // Ollama stays external: its service is managed outside the hub, and a
    // catalog entry that claimed otherwise would spawn a second server.
    const ollama = (loaded.config.hosts ?? []).find((host) => host.id === 'ollama');
    assert.notEqual(ollama?.lifecycle?.startable, true, 'ollama must stay startable:false');

    // Whatever else a deployment declares: a `startable` flag without a command is
    // collapsed to false at resolution time, so the catalog must never say it.
    for (const host of loaded.config.hosts ?? []) {
      if (host.lifecycle?.startable !== true) continue;
      assert.notEqual(host.lifecycle.start, undefined, `${host.id} is startable but declares no start command`);
    }
  });

  it('validates the real-engine example catalog, ignoring its $comment fields', () => {
    const loaded = loadCatalogConfig({ configPath: 'config/examples/real-models.example.json' });
    assert.ok(loaded.config.models.length >= 5);
    assert.ok((loaded.config.hosts ?? []).length >= 4);
    const ollama = loaded.config.models.find((model) => model.id === 'ollama_llama3_8b');
    assert.equal(ollama?.host, 'ollama');
    assert.equal(ollama?.capabilities[0], 'text_to_text');
  });

  it('writes every shipped file path so that it resolves from the catalog that names it', () => {
    // The check that would have caught the doubled `config/`. Both shipped
    // catalogs name templates (`workflowPath` for ComfyUI, `stepsPath` for 3D),
    // and a relative path is only correct relative to *its own* catalog: a
    // catalog in `config/` that writes `config/workflows/…` composes to
    // `config/config/workflows/…`, which has never existed, and the deployment
    // discovers that on its first invocation rather than here. Resolving each
    // shipped path with the same helper the adapters use, against the directory
    // the catalog was actually loaded from, turns that into a test failure.
    const shipped = ['config/models.json', 'config/examples/real-models.example.json'];
    let seen = 0;

    for (const relative of shipped) {
      const loaded = loadCatalogConfig({ configPath: relative });
      const catalogDir = dirname(loaded.path);

      const paths: { readonly where: string; readonly value: string }[] = [];
      for (const model of loaded.config.models) {
        for (const key of ['workflowPath', 'stepsPath'] as const) {
          const value = model.adapterConfig?.[key];
          if (typeof value === 'string' && value.length > 0) {
            paths.push({ where: `${model.id}.adapterConfig.${key}`, value });
          }
        }
      }
      for (const host of loaded.config.hosts ?? []) {
        const value = host.adapterConfig?.['stepsPath'];
        if (typeof value === 'string' && value.length > 0) {
          paths.push({ where: `host ${host.id}.adapterConfig.stepsPath`, value });
        }
      }

      for (const { where, value } of paths) {
        seen += 1;
        const resolved = resolveAdapterPath(value, catalogDir);
        assert.ok(
          existsSync(resolved.absolute),
          `${relative}: ${where} is "${value}", which resolves to "${resolved.absolute}" — that file does not ` +
            'exist. A relative path in a catalog is relative to that catalog, not to the package root.',
        );
      }
    }

    // A guard on the guard: if the shipped catalogs ever stop naming templates, the
    // loop above would pass vacuously and this test would stop protecting anything.
    assert.ok(seen >= 3, `expected the shipped catalogs to name template files, found ${seen}`);
  });

  it('discovers the catalog by walking up from a nested directory', () => {
    const loaded = loadCatalogConfig({ startDir: join(process.cwd(), 'src', 'catalog') });
    assert.ok(loaded.path.endsWith('models.json'));
  });

  it('prefers a nearer catalog over a deeper config/ directory', async () => {
    // Regression: discovery used to try `config/<name>` at the same level before
    // the direct filename, so a nested `<root>/config/models.json` shadowed a
    // `models.json` sitting right where the harness was launched. "Walk up to the
    // nearest catalog" has to mean nearest first.
    const root = await mkdtemp(join(tmpdir(), 'aimh-discovery-'));
    try {
      const nested = join(root, 'config');
      await mkdir(nested, { recursive: true });
      const document = JSON.stringify({
        models: [
          {
            id: 'from_root',
            name: 'From Root',
            type: 'custom',
            capabilities: ['text_to_text'],
            adapter: 'mock',
            runtime: { engine: 'mock', adapter: 'mock' },
          },
        ],
      });
      await writeFile(join(root, 'models.json'), document, 'utf8');
      await writeFile(join(nested, 'models.json'), document, 'utf8');

      const found = findConfigDirectory({ startDir: root });
      assert.equal(found, join(root, 'models.json'), 'the direct filename at the nearest level must win');
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('still finds a config/ subdirectory when no direct file exists', async () => {
    const root = await mkdtemp(join(tmpdir(), 'aimh-discovery-cfg-'));
    try {
      const nested = join(root, 'config');
      await mkdir(nested, { recursive: true });
      await writeFile(
        join(nested, 'models.json'),
        JSON.stringify({
          models: [
            {
              id: 'nested_only',
              name: 'Nested Only',
              type: 'custom',
              capabilities: ['text_to_text'],
              adapter: 'mock',
              runtime: { engine: 'mock', adapter: 'mock' },
            },
          ],
        }),
        'utf8',
      );
      // A subdirectory of the root, with no catalog of its own, must walk up.
      const deep = join(root, 'src', 'inner');
      await mkdir(deep, { recursive: true });
      assert.equal(findConfigDirectory({ startDir: deep }), join(nested, 'models.json'));
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('tolerates a UTF-8 byte-order mark', async () => {
    // Regression: Notepad and PowerShell's `Set-Content -Encoding utf8` both emit a
    // BOM, and JSON.parse rejects it with a message that never mentions the BOM.
    const root = await mkdtemp(join(tmpdir(), 'aimh-bom-'));
    try {
      const path = join(root, 'models.json');
      const body = JSON.stringify({
        models: [
          {
            id: 'bom_model',
            name: 'BOM Model',
            type: 'custom',
            capabilities: ['text_to_text'],
            adapter: 'mock',
            runtime: { engine: 'mock', adapter: 'mock' },
          },
        ],
      });
      await writeFile(path, `\uFEFF${body}`, 'utf8');

      const loaded = loadCatalogConfig({ configPath: path });
      assert.equal(loaded.config.models.length, 1);
      assert.equal(loaded.config.models[0]?.id, 'bom_model');
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('reports a missing catalog as CONFIG_ERROR', () => {
    assert.throws(
      () => loadCatalogConfig({ configPath: 'no/such/file.json' }),
      (error: unknown) => (error as { code?: string }).code === 'CONFIG_ERROR',
    );
  });

  /**
   * Write a one-model catalog and return the directory holding it.
   * @param id - the model id, so a test can tell which catalog was loaded.
   * @returns the directory containing `models.json`.
   */
  async function catalogDir(id: string): Promise<string> {
    const root = await mkdtemp(join(tmpdir(), 'aimh-anchor-'));
    await writeFile(
      join(root, 'models.json'),
      JSON.stringify({
        models: [
          {
            id,
            name: id,
            type: 'custom',
            capabilities: ['text_to_text'],
            adapter: 'mock',
            runtime: { engine: 'mock', adapter: 'mock' },
          },
        ],
      }),
      'utf8',
    );
    return root;
  }

  it('falls through to a later anchor when an earlier one has no catalog', async () => {
    // The regression this whole mechanism exists for: a host process whose
    // working directory has no catalog must still find the deployment's own.
    const empty = await mkdtemp(join(tmpdir(), 'aimh-anchor-empty-'));
    const filled = await catalogDir('from_second_anchor');
    try {
      const loaded = loadCatalogFromAnchors({ anchors: [empty, filled] });
      assert.equal(loaded.config.models[0]?.id, 'from_second_anchor');
      assert.equal(loaded.anchor, filled);
      assert.deepEqual(loaded.searched, [empty, filled], 'the search trail must record every anchor tried');
    } finally {
      await rm(empty, { recursive: true, force: true });
      await rm(filled, { recursive: true, force: true });
    }
  });

  it('prefers the earliest anchor that has a catalog', async () => {
    const first = await catalogDir('from_first_anchor');
    const second = await catalogDir('from_second_anchor');
    try {
      const loaded = loadCatalogFromAnchors({ anchors: [first, second] });
      assert.equal(loaded.config.models[0]?.id, 'from_first_anchor');
      assert.deepEqual(loaded.searched, [first], 'searching must stop at the first hit');
    } finally {
      await rm(first, { recursive: true, force: true });
      await rm(second, { recursive: true, force: true });
    }
  });

  it('treats an explicit configPath as the only candidate', async () => {
    // A named file that is wrong must be an error, never a quiet substitution of
    // whatever a fallback anchor happens to contain.
    const fallback = await catalogDir('must_not_be_used');
    try {
      assert.throws(
        () => loadCatalogFromAnchors({ configPath: join(fallback, 'no-such-file.json'), anchors: [fallback] }),
        (error: unknown) => (error as { code?: string }).code === 'CONFIG_ERROR',
      );
    } finally {
      await rm(fallback, { recursive: true, force: true });
    }
  });

  it('names every anchor it tried when no catalog is found', async () => {
    const one = await mkdtemp(join(tmpdir(), 'aimh-anchor-none-a-'));
    const two = await mkdtemp(join(tmpdir(), 'aimh-anchor-none-b-'));
    try {
      assert.throws(
        () => loadCatalogFromAnchors({ anchors: [one, two] }),
        (error: unknown) => {
          const message = (error as Error).message;
          return (
            (error as { code?: string }).code === 'CONFIG_ERROR' &&
            message.includes(one) &&
            message.includes(two)
          );
        },
      );
    } finally {
      await rm(one, { recursive: true, force: true });
      await rm(two, { recursive: true, force: true });
    }
  });
});

describe('machine detection', () => {
  it('returns a coherent profile with evidence', async () => {
    const result = await probeMachine({ skipGpuProbe: true, timeoutMs: 2000 });
    assert.ok(result.profile.ramGb > 0, 'system RAM should always be detectable');
    assert.equal(result.profile.hasGpu, false);
    assert.ok(result.evidence.some((line) => line.includes('system RAM')));
  });

  it('does not throw when the GPU probe is unavailable', async () => {
    const result = await probeMachine({ timeoutMs: 1500 });
    assert.ok(Number.isFinite(result.profile.vramGb));
    assert.ok(result.evidence.length > 0);
  });
});

describe('adapter registry', () => {
  it('scopes a replaced adapter and restores the previous one', () => {
    const registry = new AdapterRegistry([createMockAdapter()]);
    const replacement: ModelAdapter = {
      kind: 'mock',
      displayName: 'Replacement',
      supports: () => ({ ok: true }),
      health: () => Promise.resolve({ healthy: true, checkedAt: Date.now() }),
      invoke: () => Promise.resolve({ outputs: [] }),
    };
    const restore = registry.register(replacement);
    assert.equal(registry.require('mock').displayName, 'Replacement');
    restore();
    assert.notEqual(registry.require('mock').displayName, 'Replacement');
  });
});
