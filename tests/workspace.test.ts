/**
 * Tests for where artifacts land.
 *
 * Two halves, tested separately because they fail differently. The hub keeps a
 * store per named root, so its tests run a real catalog through the mock adapter
 * and assert on *where the bytes went* — the claim is about the filesystem, not
 * about a return value. The resolver decides which root a call names, so its tests
 * drive it with a stand-in sandbox policy, including the failure it must survive.
 *
 * The regression these guard is specific: `defaultArtifactRoot()` is
 * `<process.cwd()>/artifacts`, and for a long-lived `dsh web` the host's working
 * directory has nothing to do with the conversation's — which is how a session's
 * images end up somewhere the user never looks.
 *
 * @module dsh-ai-model-hub/tests/workspace.test
 */

import { strict as assert } from 'node:assert';
import { mkdtemp, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { Context } from '@deepseek-ai/cordis';

import { ModelHub } from '../src/index.ts';
import { ARTIFACTS_DIRECTORY, createArtifactRootResolver, sessionCwdOf } from '../dsh-plugin/workspace.ts';
import type { PluginLogger } from '../dsh-plugin/types.ts';

/** A catalog whose one model produces a real PNG with no engine at all. */
function mockCatalog(): Record<string, unknown> {
  return {
    version: '1',
    models: [
      {
        id: 'mock_image_model',
        name: 'Mock Image',
        type: 'image_generation',
        capabilities: ['text_to_image'],
        adapter: 'mock',
        runtime: { engine: 'in_process_mock', adapter: 'mock' },
        adapterConfig: { fixture: 'deterministic-png', width: 32, height: 32 },
      },
    ],
  };
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

describe('per-call artifact roots (hub)', () => {
  it('writes a call\'s artifacts under the root that call names', async () => {
    const defaultRoot = await mkdtemp(join(tmpdir(), 'aimh-default-'));
    const sessionRoot = await mkdtemp(join(tmpdir(), 'aimh-session-'));
    const hub = ModelHub.fromConfig(mockCatalog(), {
      artifactRoot: defaultRoot,
      manageTimers: false,
      log: () => {},
    });

    try {
      const result = await hub.invokeModel(
        { capability: 'text_to_image', prompt: 'a fox' },
        { artifactRoot: join(sessionRoot, ARTIFACTS_DIRECTORY) },
      );
      assert.equal(result.outputs.length, 1);

      const store = join(sessionRoot, ARTIFACTS_DIRECTORY);
      assert.ok(existsSync(join(store, 'index.json')), 'the session root received the index');
      assert.ok(existsSync(join(store, 'files')), 'the session root received the content directory');
      assert.equal(
        existsSync(join(defaultRoot, 'index.json')),
        false,
        'the default store was left untouched',
      );

      // The listing must follow the same root, or chaining a workflow would not
      // find the artifact it just produced.
      const listed = await hub.listArtifacts(10, store);
      assert.equal(listed.length, 1);
      assert.equal(listed[0]?.id, result.outputs[0]?.id);
      assert.equal((await hub.listArtifacts(10)).length, 0, 'the default store still lists nothing');
      assert.notEqual(await hub.getArtifact(result.outputs[0]?.id ?? '', store), undefined);
      assert.equal(await hub.getArtifact(result.outputs[0]?.id ?? ''), undefined);
    } finally {
      await hub.dispose();
      await rm(defaultRoot, { recursive: true, force: true });
      await rm(sessionRoot, { recursive: true, force: true });
    }
  });

  it('reuses one store per root instead of rebuilding it', async () => {
    const defaultRoot = await mkdtemp(join(tmpdir(), 'aimh-default-'));
    const hub = ModelHub.fromConfig(mockCatalog(), {
      artifactRoot: defaultRoot,
      manageTimers: false,
      log: () => {},
    });

    try {
      const store = join(defaultRoot, 'shared');
      const first = await hub.invokeModel({ capability: 'text_to_image', prompt: 'one' }, { artifactRoot: store });
      const second = await hub.invokeModel({ capability: 'text_to_image', prompt: 'two' }, { artifactRoot: store });
      assert.equal(first.outputs.length, 1);
      assert.equal(second.outputs.length, 1);
      assert.equal(
        (await hub.listArtifacts(10, store)).length,
        2,
        'both calls accumulated in the same store',
      );
    } finally {
      await hub.dispose();
      await rm(defaultRoot, { recursive: true, force: true });
    }
  });
});

describe('resolving the artifact root for a call (plugin)', () => {
  /** A context whose `get` answers for `sandboxPolicy`. */
  function contextWith(policy: unknown): Context {
    return { get: (name: string) => (name === 'sandboxPolicy' ? policy : undefined) } as unknown as Context;
  }

  it('lets an explicit configured root win over every session', () => {
    const { logger } = recordingLogger();
    const resolver = createArtifactRootResolver(contextWith({ resolve: () => ({ workspaceRoot: 'C:/ws' }) }), 'C:/pinned', logger);
    assert.equal(resolver({ agent: { session: { cwd: 'C:/ws' } } }), undefined);
  });

  it('sends a call to the calling session\'s workspace', () => {
    const { logger } = recordingLogger();
    const resolver = createArtifactRootResolver(
      contextWith({ resolve: () => ({ mode: 'workspace-write', workspaceRoot: 'C:/ws' }) }),
      '',
      logger,
    );
    assert.equal(resolver({ agent: { session: { cwd: 'C:/ignored' } } }), join('C:/ws', ARTIFACTS_DIRECTORY));
  });

  it('falls back to the session cwd when no sandbox policy is mounted', () => {
    const { logger } = recordingLogger();
    const resolver = createArtifactRootResolver(contextWith(undefined), '', logger);
    assert.equal(resolver({ agent: { session: { cwd: 'C:/only-cwd' } } }), join('C:/only-cwd', ARTIFACTS_DIRECTORY));
  });

  it('leaves the hub\'s own store alone for a call with no session', () => {
    const { logger } = recordingLogger();
    const resolver = createArtifactRootResolver(
      contextWith({ resolve: () => ({ workspaceRoot: 'C:/ws' }) }),
      '',
      logger,
    );
    assert.equal(resolver(undefined), undefined);
    assert.equal(resolver({}), undefined);
  });

  it('survives a policy that throws, warning once and using the session cwd', () => {
    const { lines, logger } = recordingLogger();
    const resolver = createArtifactRootResolver(
      contextWith({
        resolve: () => {
          throw new Error('no cwd recorded');
        },
      }),
      '',
      logger,
    );
    assert.equal(resolver({ agent: { session: { cwd: 'C:/ws' } } }), join('C:/ws', ARTIFACTS_DIRECTORY));
    assert.equal(resolver({ agent: { session: { cwd: 'C:/ws' } } }), join('C:/ws', ARTIFACTS_DIRECTORY));
    assert.equal(lines.length, 1, 'one warning for any number of failing calls');
    assert.match(lines[0] ?? '', /could not resolve the sandbox workspace/);
  });

  it('reads the session cwd defensively', () => {
    assert.equal(sessionCwdOf(undefined), undefined);
    assert.equal(sessionCwdOf({}), undefined);
    assert.equal(sessionCwdOf({ agent: {} }), undefined);
    assert.equal(sessionCwdOf({ agent: { session: {} } }), undefined);
    assert.equal(sessionCwdOf({ agent: { session: { cwd: '' } } }), undefined);
    assert.equal(sessionCwdOf({ agent: { session: { cwd: 'C:/ws' } } }), 'C:/ws');
  });

  it('is the resolver invoke_model actually uses, per call', async () => {
    // The wiring, not the parts: a real registrar, a real tool call, and the
    // question the whole change exists to answer — did the image land in the
    // calling session's workspace?
    const { registerInvokeTool } = await import('../dsh-plugin/tools/invoke.ts');
    const { ModelHubService } = await import('../dsh-plugin/service.ts');

    const bootRoot = await mkdtemp(join(tmpdir(), 'aimh-boot-'));
    const workspace = await mkdtemp(join(tmpdir(), 'aimh-ws-'));
    const hub = ModelHub.fromConfig(mockCatalog(), { artifactRoot: bootRoot, manageTimers: false, log: () => {} });

    const registered = new Map<string, { execute: (args: unknown, exec: unknown) => Promise<unknown> }>();
    // A real Context is required, not an object literal: `ModelHubService` is a
    // cordis Service, which registers itself through `ctx.reflect`.
    const context = new Context().extend({
      tools: {
        register: (definition: { name: string }) => {
          registered.set(definition.name, definition as never);
          return () => {};
        },
      },
    });

    const { lines, logger } = recordingLogger();
    const resolver = createArtifactRootResolver(
      contextWith({ resolve: () => ({ mode: 'workspace-write', workspaceRoot: workspace }) }),
      '',
      logger,
    );
    registerInvokeTool(context, new ModelHubService(context, hub), {
      invocationTimeoutMs: 30_000,
      artifactRootFor: resolver,
    });

    try {
      const tool = registered.get('invoke_model');
      assert.notEqual(tool, undefined, 'invoke_model was registered');

      await tool?.execute(
        { capability: 'text_to_image', prompt: 'a fox' },
        { signal: new AbortController().signal, agent: { session: { cwd: workspace } } },
      );

      assert.ok(
        existsSync(join(workspace, ARTIFACTS_DIRECTORY, 'index.json')),
        "the call wrote into the session's workspace",
      );
      assert.equal(existsSync(join(bootRoot, 'index.json')), false, 'the boot-time root stayed empty');
      assert.deepEqual(lines, [], 'a healthy resolve warns about nothing');
    } finally {
      await hub.dispose();
      await rm(bootRoot, { recursive: true, force: true });
      await rm(workspace, { recursive: true, force: true });
    }
  });
});
