/**
 * Tests for the Model Router.
 *
 * The router's contract has two halves, and both are tested here: it must pick
 * the *right* model, and it must never contain knowledge of a *particular* model.
 * The second half is checked by running the router against catalogs it has never
 * seen, including ones whose model ids and capabilities are invented for the test.
 *
 * @module dsh-ai-model-hub/tests/router.test
 */

import { strict as assert } from 'node:assert';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import type { InvocationRequest, ModelCatalogConfig, ModelDescriptor } from '../src/index.ts';
import {
  AdapterRegistry,
  LocalArtifactStore,
  ModelCatalog,
  ModelHubError,
  RuntimeManager,
  createMockAdapter,
  resolveRequestInputs,
  routeRequest,
} from '../src/index.ts';

/** A ready-to-use routing fixture. */
interface Fixture {
  readonly catalog: ModelCatalog;
  readonly runtime: RuntimeManager;
  readonly artifacts: LocalArtifactStore;
  readonly cleanup: () => Promise<void>;
}

/**
 * Build a routing context around a catalog document.
 * @param config - the catalog document.
 * @param machine - optional machine profile.
 * @returns the fixture and a cleanup function.
 */
async function fixture(
  config: ModelCatalogConfig,
  machine?: { vramGb: number; ramGb: number; hasGpu: boolean; notes: string },
): Promise<Fixture> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-router-'));
  const catalog = new ModelCatalog(config, machine === undefined ? {} : { machine });
  const adapters = new AdapterRegistry([createMockAdapter()]);
  const runtime = new RuntimeManager({ catalog, adapters, healthIntervalMs: 0, idleSweepIntervalMs: 0 });
  const artifacts = new LocalArtifactStore({ root });
  return {
    catalog,
    runtime,
    artifacts,
    cleanup: async () => {
      await runtime.dispose();
      await rm(root, { recursive: true, force: true });
    },
  };
}

/**
 * One inline model entry with sensible defaults.
 *
 * Returns a `ModelDescriptor` so the fixture's catalog document is type-checked
 * against the real descriptor contract rather than being an untyped blob.
 *
 * @param overrides - fields to override on the default entry.
 * @returns a descriptor.
 */
function model(overrides: Record<string, unknown>): ModelDescriptor {
  return {
    id: 'm',
    name: 'M',
    type: 'custom',
    capabilities: ['text_to_text'],
    adapter: 'mock',
    runtime: { engine: 'mock', adapter: 'mock' },
    lifecycle: { startable: false, stoppable: false },
    ...overrides,
  } as ModelDescriptor;
}

/** Route a request against a fixture. */
async function route(
  context: Fixture,
  request: InvocationRequest,
): Promise<ReturnType<typeof routeRequest>> {
  const resolved = await resolveRequestInputs({ artifacts: context.artifacts }, request);
  return routeRequest(
    { catalog: context.catalog, runtime: context.runtime, artifacts: context.artifacts },
    resolved,
  );
}

describe('routeRequest', () => {
  it('selects a model declaring the capability', async () => {
    const context = await fixture({
      models: [
        model({ id: 'text_only', capabilities: ['text_to_text'] }),
        model({ id: 'image_maker', type: 'image_generation', capabilities: ['text_to_image'] }),
      ],
    });
    try {
      const decision = await route(context, { capability: 'text_to_image', prompt: 'a cat' });
      assert.equal(decision.modelId, 'image_maker');
    } finally {
      await context.cleanup();
    }
  });

  it('prefers the lower priority number', async () => {
    const context = await fixture({
      models: [
        model({ id: 'worse', capabilities: ['text_to_text'], priority: 90 }),
        model({ id: 'better', capabilities: ['text_to_text'], priority: 10 }),
      ],
    });
    try {
      const decision = await route(context, { capability: 'text_to_text', prompt: 'hi' });
      assert.equal(decision.modelId, 'better');
    } finally {
      await context.cleanup();
    }
  });

  it('breaks a priority tie by id, deterministically', async () => {
    const context = await fixture({
      models: [
        model({ id: 'zulu', capabilities: ['text_to_text'], priority: 5 }),
        model({ id: 'alpha', capabilities: ['text_to_text'], priority: 5 }),
      ],
    });
    try {
      for (let attempt = 0; attempt < 5; attempt += 1) {
        const decision = await route(context, { capability: 'text_to_text', prompt: 'hi' });
        assert.equal(decision.modelId, 'alpha');
      }
    } finally {
      await context.cleanup();
    }
  });

  it('accepts a stopped but startable model as a candidate', async () => {
    const context = await fixture({
      models: [model({ id: 'sick', capabilities: ['text_to_text'] })],
    });
    try {
      // A model that is merely cold is a legitimate routing choice: the runtime
      // starts it on demand. Rejecting cold models would make the hub useless on a
      // machine where nothing runs until asked.
      const status = context.runtime.getModelStatus('sick');
      assert.equal(status.availability, 'stopped');
      const decision = await route(context, { capability: 'text_to_text', prompt: 'hi' });
      assert.equal(decision.modelId, 'sick');
      assert.match(decision.rationale, /cold but startable/);
    } finally {
      await context.cleanup();
    }
  });

  it('rejects a model whose declared inputs do not cover the request', async () => {
    const context = await fixture({
      models: [
        model({
          id: 'text_only_inputs',
          type: 'image_generation',
          capabilities: ['text_to_image'],
          inputTypes: ['text'],
        }),
      ],
    });
    try {
      const stored = await context.artifacts.put({ type: 'image', bytes: new Uint8Array([1]) });
      await assert.rejects(
        () => route(context, { capability: 'text_to_image', inputs: [{ id: stored.id }] }),
        (error: unknown) => (error as { code?: string }).code === 'NO_COMPATIBLE_MODEL',
      );
    } finally {
      await context.cleanup();
    }
  });

  it('accepts a request whose kinds the model declares', async () => {
    const context = await fixture({
      models: [
        model({
          id: 'editor',
          type: 'image_editing',
          capabilities: ['image_to_image'],
          inputTypes: ['image', 'text'],
          outputTypes: ['image'],
        }),
      ],
    });
    try {
      const stored = await context.artifacts.put({ type: 'image', bytes: new Uint8Array([1]) });
      const decision = await route(context, {
        capability: 'image_to_image',
        inputs: [{ id: stored.id }],
        prompt: 'make it blue',
      });
      assert.equal(decision.modelId, 'editor');
    } finally {
      await context.cleanup();
    }
  });

  it('honours required tags', async () => {
    const context = await fixture({
      models: [
        model({ id: 'cpu', capabilities: ['text_to_text'], tags: ['cpu'] }),
        model({ id: 'gpu', capabilities: ['text_to_text'], tags: ['gpu'] }),
      ],
    });
    try {
      const decision = await route(context, {
        capability: 'text_to_text',
        prompt: 'hi',
        requiredTags: ['gpu'],
      });
      assert.equal(decision.modelId, 'gpu');

      await assert.rejects(
        () => route(context, { capability: 'text_to_text', prompt: 'hi', requiredTags: ['tpu'] }),
        (error: unknown) => (error as { code?: string }).code === 'NO_COMPATIBLE_MODEL',
      );
    } finally {
      await context.cleanup();
    }
  });

  it('honours an explicit model pin', async () => {
    const context = await fixture({
      models: [
        model({ id: 'default', capabilities: ['text_to_text'], priority: 1 }),
        model({ id: 'pinned', capabilities: ['text_to_text'], priority: 99 }),
      ],
    });
    try {
      const decision = await route(context, { capability: 'text_to_text', prompt: 'hi', modelId: 'pinned' });
      assert.equal(decision.modelId, 'pinned');
    } finally {
      await context.cleanup();
    }
  });

  it('rejects a model that exceeds the machine', async () => {
    const context = await fixture(
      {
        models: [
          model({
            id: 'enormous',
            type: 'image_generation',
            capabilities: ['text_to_image'],
            resources: { vramGb: 48, ramGb: 128, requiresGpu: true },
          }),
        ],
      },
      { vramGb: 8, ramGb: 32, hasGpu: true, notes: 'test' },
    );
    try {
      await assert.rejects(
        () => route(context, { capability: 'text_to_image', prompt: 'a cat' }),
        (error: unknown) => (error as { code?: string }).code === 'NO_COMPATIBLE_MODEL',
      );
    } finally {
      await context.cleanup();
    }
  });

  it('explains every rejection when nothing is compatible', async () => {
    const context = await fixture({
      models: [
        model({ id: 'text_only', capabilities: ['text_to_text'] }),
        model({ id: 'disabled_image', type: 'image_generation', capabilities: ['text_to_image'], enabled: false }),
      ],
    });
    try {
      await assert.rejects(
        () => route(context, { capability: 'text_to_image', prompt: 'a cat' }),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'NO_COMPATIBLE_MODEL');
          const candidates = (error.details['candidates'] ?? []) as { modelId: string; reason: string }[];
          assert.equal(candidates.length, 2);
          assert.ok(candidates.some((candidate) => candidate.reason.includes('does not declare capability')));
          assert.ok(candidates.some((candidate) => candidate.reason.includes('disabled')));
          return true;
        },
      );
    } finally {
      await context.cleanup();
    }
  });

  it('names the unserved capability when no model declares it', async () => {
    const context = await fixture({ models: [model({ id: 'text_only', capabilities: ['text_to_text'] })] });
    try {
      await assert.rejects(
        () => route(context, { capability: 'video_generation', prompt: 'a whale' }),
        (error: unknown) => {
          assert.match((error as Error).message, /No configured model declares "video_generation"/);
          return true;
        },
      );
    } finally {
      await context.cleanup();
    }
  });

  it('treats an unknown declared input kind as an artifact error', async () => {
    const context = await fixture({ models: [model({ id: 'm', capabilities: ['text_to_text'] })] });
    try {
      const stored = await context.artifacts.put({ type: 'text', text: 'hi' });
      await assert.rejects(
        () => route(context, { capability: 'text_to_text', prompt: 'x', inputs: [{ id: stored.id, type: 'hologram' }] }),
        (error: unknown) => (error as { code?: string }).code === 'ARTIFACT_ERROR',
      );
    } finally {
      await context.cleanup();
    }
  });

  it('reports a declared-kind mismatch precisely', async () => {
    const context = await fixture({ models: [model({ id: 'm', capabilities: ['text_to_text'] })] });
    try {
      const stored = await context.artifacts.put({ type: 'text', text: 'hi' });
      await assert.rejects(
        () => route(context, { capability: 'text_to_text', prompt: 'x', inputs: [{ id: stored.id, type: 'image' }] }),
        (error: unknown) => {
          assert.equal((error as { code?: string }).code, 'ARTIFACT_ERROR');
          assert.match((error as Error).message, /is of type "text" but was declared as "image"/);
          return true;
        },
      );
    } finally {
      await context.cleanup();
    }
  });

  it('reports a missing artifact id clearly', async () => {
    const context = await fixture({ models: [model({ id: 'm', capabilities: ['text_to_text'] })] });
    try {
      await assert.rejects(
        () => route(context, { capability: 'text_to_text', prompt: 'x', inputs: ['artifact_does_not_exist'] }),
        (error: unknown) => {
          assert.equal((error as { code?: string }).code, 'ARTIFACT_ERROR');
          assert.match((error as Error).message, /does not exist/);
          return true;
        },
      );
    } finally {
      await context.cleanup();
    }
  });

  it('routes on capabilities the router was never told about', async () => {
    // The router has no knowledge of model ids or engines, so a catalog using
    // invented names must route exactly as well as the shipped one. This is the
    // "no hard-coded model logic in the router" property, tested rather than
    // asserted in a comment.
    const context = await fixture({
      models: [
        model({
          id: 'zzz_completely_invented_engine',
          name: 'Invented',
          type: 'multimodal',
          capabilities: ['text_to_3d', 'image_to_3d'],
          priority: 3,
        }),
      ],
    });
    try {
      const decision = await route(context, { capability: 'text_to_3d', prompt: 'a teapot' });
      assert.equal(decision.modelId, 'zzz_completely_invented_engine');
    } finally {
      await context.cleanup();
    }
  });

  it('produces a stable decision across repeated calls', async () => {
    const context = await fixture({
      models: [
        model({ id: 'one', capabilities: ['text_to_text'], priority: 1 }),
        model({ id: 'two', capabilities: ['text_to_text'], priority: 2 }),
        model({ id: 'three', capabilities: ['text_to_text'], priority: 3 }),
      ],
    });
    try {
      const decisions = await Promise.all(
        Array.from({ length: 10 }, () => route(context, { capability: 'text_to_text', prompt: 'hi' })),
      );
      assert.deepEqual(new Set(decisions.map((decision) => decision.modelId)), new Set(['one']));
    } finally {
      await context.cleanup();
    }
  });
});

describe('resolveRequestInputs', () => {
  it('accepts a bare id and an explicit reference interchangeably', async () => {
    const context = await fixture({ models: [model({ id: 'm' })] });
    try {
      const stored = await context.artifacts.put({ type: 'image', bytes: new Uint8Array([1]) });
      const bare = await resolveRequestInputs({ artifacts: context.artifacts }, {
        capability: 'image_to_image',
        inputs: [stored.id],
      });
      const explicit = await resolveRequestInputs({ artifacts: context.artifacts }, {
        capability: 'image_to_image',
        inputs: [{ id: stored.id, type: 'image' }],
      });
      assert.deepEqual([...bare.kindSet], ['image']);
      assert.deepEqual([...explicit.kindSet], ['image']);
      assert.equal(bare.inputs[0]?.id, stored.id);
    } finally {
      await context.cleanup();
    }
  });

  it('treats a whitespace-only prompt as absent', async () => {
    const context = await fixture({ models: [model({ id: 'm' })] });
    try {
      const resolved = await resolveRequestInputs({ artifacts: context.artifacts }, {
        capability: 'text_to_text',
        prompt: '   ',
      });
      assert.equal(resolved.hasPrompt, false);
    } finally {
      await context.cleanup();
    }
  });
});
