/**
 * Tests for the Capability priorities route.
 *
 * This is the only thing that writes the policy, so what is defended here is that
 * it writes *exactly* what was submitted and nothing else: an edit that names an
 * option the capability does not have is rejected rather than invented, a partial
 * submission cannot delete saved settings, and the document that lands on disk is
 * the one the next load will enforce.
 *
 * @module dsh-ai-model-hub/tests/priorities-route.test
 */

import { strict as assert } from 'node:assert';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import type { Context } from '@deepseek-ai/cordis';
import { CapabilityPolicy, ModelHub, parseCapabilityPolicy } from '../src/index.ts';
import { PRIORITIES_ROUTE, registerPrioritiesRoute } from '../dsh-plugin/priorities-route.ts';
import type { PluginLogger } from '../dsh-plugin/types.ts';

/** A catalog with two image options and one text option. */
function catalog(): unknown {
  return {
    version: '1',
    models: [
      {
        id: 'z_image_turbo',
        name: 'Z-Image Turbo (ComfyUI)',
        type: 'image_generation',
        capabilities: ['text_to_image'],
        adapter: 'mock',
        runtime: { engine: 'comfyui', adapter: 'mock' },
        priority: 10,
      },
      {
        id: 'krea_turbo',
        name: 'Krea Turbo (ComfyUI)',
        type: 'image_generation',
        capabilities: ['text_to_image'],
        adapter: 'mock',
        runtime: { engine: 'comfyui', adapter: 'mock' },
        priority: 20,
      },
      {
        id: 'ollama_text',
        name: 'Ollama text',
        type: 'text_generation',
        capabilities: ['text_to_text'],
        adapter: 'mock',
        runtime: { engine: 'ollama', adapter: 'mock' },
      },
    ],
  };
}

/** A logger that records what it was told. */
function recordingLogger(): { readonly logger: PluginLogger; readonly lines: string[] } {
  const lines: string[] = [];
  const push = (level: string) => (message: string) => {
    lines.push(`${level} ${message}`);
  };
  return {
    lines,
    logger: {
      debug: push('debug'),
      info: push('info'),
      warn: push('warn'),
      error: push('error'),
    },
  };
}

/** A response object that records what was written. */
function fakeResponse(): { captured: { status?: number | undefined; body?: string | undefined }; res: ServerResponse } {
  const captured: { status?: number | undefined; body?: string | undefined } = {};
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

/** A request whose body is the given JSON. */
function requestWith(method: string, body?: unknown): IncomingMessage {
  const text = body === undefined ? '' : JSON.stringify(body);
  return {
    method,
    url: PRIORITIES_ROUTE,
    async *[Symbol.asyncIterator]() {
      if (text.length > 0) yield Buffer.from(text, 'utf8');
    },
  } as unknown as IncomingMessage;
}

/** A registered route plus its hub and policy file. */
interface Fixture {
  readonly hub: ModelHub;
  readonly policy: CapabilityPolicy;
  readonly policyPath: string;
  readonly handler: (req: IncomingMessage, res: ServerResponse) => void | Promise<void>;
  readonly lines: string[];
  readonly cleanup: () => Promise<void>;
}

/** Build the route over a fresh hub and a temp policy file. */
async function fixture(): Promise<Fixture> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-priorities-'));
  const policyPath = join(root, 'capability-policy.json');
  const policy = new CapabilityPolicy();
  const hub = ModelHub.fromConfig(catalog(), {
    artifactRoot: root,
    manageTimers: false,
    probeResources: false,
    log: () => {},
    capabilityPolicy: policy,
  });
  const { logger, lines } = recordingLogger();
  let handler: ((req: IncomingMessage, res: ServerResponse) => void | Promise<void>) | undefined;
  const register = (route: {
    handler: (req: IncomingMessage, res: ServerResponse) => void | Promise<void>;
  }): (() => void) => {
    handler = route.handler;
    return () => {};
  };
  const routeDisposers: (() => void)[] = [];
  const ctx = {
    inject: (_deps: readonly string[], callback: (scope: unknown) => void) => {
      // The route reads the server through `scope.get`, which is how a cordis
      // scope exposes a service it has resolved.
      callback({ effect: (body: () => () => void) => { routeDisposers.push(body()); },
        get: (name: string) => (name === 'webServer' ? { register } : undefined) });
    },
  } as unknown as Context;
  registerPrioritiesRoute(ctx, logger, { hub, policy, policyPath });
  assert.notEqual(handler, undefined, 'the route must be registered');
  return {
    hub,
    policy,
    policyPath,
    handler: handler as never,
    lines,
    cleanup: async () => {
      for (const dispose of routeDisposers) dispose();
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    },
  };
}

/** Parse a response body. */
function json(body: string | undefined): Record<string, unknown> {
  return JSON.parse(body ?? '{}') as Record<string, unknown>;
}

describe('the Capability priorities route', () => {
  it('answers GET with every capability, its options, and their states', async () => {
    const context = await fixture();
    try {
      const { captured, res } = fakeResponse();
      await context.handler(requestWith('GET'), res);
      assert.equal(captured.status, 200);
      const payload = json(captured.body);
      assert.equal(payload['revision'], 1);
      assert.equal(payload['globalEnabled'], true);
      assert.equal(payload['policyPath'], context.policyPath);

      const capabilities = payload['capabilities'] as {
        capability: string;
        options: { id: string; name: string; providerLabel: string; state: string; enabled: boolean; reason: string }[];
        permitted: string[];
      }[];
      const image = capabilities.find((entry) => entry.capability === 'text_to_image');
      assert.ok(image !== undefined);
      assert.deepEqual(
        image.options.map((option) => option.id),
        ['z_image_turbo', 'krea_turbo'],
        'the catalog order decides until the operator does',
      );
      // Every row carries what the page must show: identity, source, live state
      // and the reason for it.
      for (const option of image.options) {
        assert.ok(option.name.length > 0);
        assert.match(option.providerLabel, /local · comfyui/);
        assert.ok(option.reason.length > 0);
        assert.equal(typeof option.enabled, 'boolean');
      }
      assert.deepEqual(image.permitted, ['z_image_turbo', 'krea_turbo']);
    } finally {
      await context.cleanup();
    }
  });

  it('saves a submitted order exactly as arranged, disabled rows included', async () => {
    const context = await fixture();
    try {
      const { captured, res } = fakeResponse();
      await context.handler(
        requestWith('PUT', {
          capability: 'text_to_image',
          order: [
            { id: 'krea_turbo', enabled: true },
            { id: 'z_image_turbo', enabled: false },
          ],
        }),
        res,
      );
      assert.equal(captured.status, 200);
      const payload = json(captured.body);
      const image = (payload['capabilities'] as { capability: string; options: { id: string; enabled: boolean; state: string; order: number }[] }[]).find(
        (entry) => entry.capability === 'text_to_image',
      );
      assert.ok(image !== undefined);
      assert.deepEqual(
        image.options.map((option) => [option.id, option.enabled, option.state]),
        [
          // An enabled mock model has no engine to be running, so it is `stopped`
          // — not `ready`, and emphatically not `disabled`. The two must not be
          // conflated: `stopped` is permitted and may still serve.
          ['krea_turbo', true, 'stopped'],
          ['z_image_turbo', false, 'disabled'],
        ],
      );
      assert.equal(payload['revision'], 2, 'saving bumps the revision');

      // And the file on disk is what the next load will enforce.
      const reloaded = parseCapabilityPolicy(JSON.parse(await readFile(context.policyPath, 'utf8')));
      assert.equal(reloaded.ok, true);
      if (!reloaded.ok) return;
      assert.deepEqual(reloaded.document.capabilities['text_to_image']?.order, [
        { id: 'krea_turbo', enabled: true },
        { id: 'z_image_turbo', enabled: false },
      ]);
    } finally {
      await context.cleanup();
    }
  });

  it('keeps options the submission did not mention', async () => {
    const context = await fixture();
    try {
      // Save a full order first, then submit only one row.
      await context.handler(
        requestWith('PUT', {
          capability: 'text_to_image',
          order: [
            { id: 'krea_turbo', enabled: true },
            { id: 'z_image_turbo', enabled: true },
          ],
        }),
        fakeResponse().res,
      );
      const { captured, res } = fakeResponse();
      await context.handler(
        requestWith('PUT', { capability: 'text_to_image', order: [{ id: 'z_image_turbo', enabled: false }] }),
        res,
      );
      assert.equal(captured.status, 200);
      const saved = context.policy.entryFor('text_to_image');
      assert.deepEqual(saved?.order, [
        { id: 'z_image_turbo', enabled: false },
        { id: 'krea_turbo', enabled: true },
      ]);
    } finally {
      await context.cleanup();
    }
  });

  it('rejects an option the capability does not have, without writing anything', async () => {
    const context = await fixture();
    try {
      const { captured, res } = fakeResponse();
      await context.handler(
        requestWith('PUT', { capability: 'text_to_image', order: [{ id: 'invented', enabled: true }] }),
        res,
      );
      assert.equal(captured.status, 400);
      assert.match(String(json(captured.body)['error']), /rejected/);
      assert.equal(context.policy.entryFor('text_to_image'), undefined, 'nothing was written');
    } finally {
      await context.cleanup();
    }
  });

  it('rejects an unknown capability', async () => {
    const context = await fixture();
    try {
      const { captured, res } = fakeResponse();
      await context.handler(requestWith('PUT', { capability: 'generate_poetry', order: [] }), res);
      assert.equal(captured.status, 400);
      assert.match(String(json(captured.body)['error']), /not a known capability/);
    } finally {
      await context.cleanup();
    }
  });

  it('rejects a malformed order entry', async () => {
    const context = await fixture();
    try {
      for (const order of [[{ id: 'z_image_turbo' }], [{ enabled: true }], [{ id: 7, enabled: true }], ['x']]) {
        const { captured, res } = fakeResponse();
        await context.handler(requestWith('PUT', { capability: 'text_to_image', order }), res);
        assert.equal(captured.status, 400, `order ${JSON.stringify(order)} must be refused`);
      }
    } finally {
      await context.cleanup();
    }
  });

  it('sets the global switch on its own, and says so in the payload', async () => {
    const context = await fixture();
    try {
      const { captured, res } = fakeResponse();
      await context.handler(requestWith('PUT', { globalEnabled: false }), res);
      assert.equal(captured.status, 200);
      assert.equal(json(captured.body)['globalEnabled'], false);
      assert.equal(context.policy.globallyEnabled, false);

      // And it survives a reload.
      const reloaded = parseCapabilityPolicy(JSON.parse(await readFile(context.policyPath, 'utf8')));
      assert.equal(reloaded.ok, true);
      if (!reloaded.ok) return;
      assert.equal(reloaded.document.globalEnabled, false);
    } finally {
      await context.cleanup();
    }
  });

  it('refuses any method but GET and the two writes', async () => {
    const context = await fixture();
    try {
      const { captured, res } = fakeResponse();
      await context.handler(requestWith('DELETE'), res);
      assert.equal(captured.status, 405);
    } finally {
      await context.cleanup();
    }
  });

  it('applies a change it could not persist, and says the write did not land', async () => {
    // A policy file that cannot be written is not a reason to lose the operator's
    // decision for this session — but it *is* a reason to say so, because the
    // change will not survive a reload. The response carries the edit's effect AND
    // the failure.
    const root = await mkdtemp(join(tmpdir(), 'aimh-priorities-unwritable-'));
    try {
      const policy = new CapabilityPolicy();
      const hub = ModelHub.fromConfig(catalog(), {
        artifactRoot: root,
        manageTimers: false,
        probeResources: false,
        log: () => {},
        capabilityPolicy: policy,
      });
      const { logger, lines } = recordingLogger();
      let handler: ((req: IncomingMessage, res: ServerResponse) => void | Promise<void>) | undefined;
      const ctx = {
        inject: (_deps: readonly string[], callback: (scope: unknown) => void) => {
          callback({
            effect: (body: () => unknown) => body(),
            get: (name: string) =>
              name === 'webServer'
                ? {
                    register(route: {
                      handler: (req: IncomingMessage, res: ServerResponse) => void | Promise<void>;
                    }) {
                      handler = route.handler;
                      return () => {};
                    },
                  }
                : undefined,
          });
        },
      } as unknown as Context;
      // A path whose own parent is a *file*, so `mkdir` fails with ENOTDIR.
      const blocker = join(root, 'not-a-directory');
      await writeFile(blocker, 'x', 'utf8');
      registerPrioritiesRoute(ctx, logger, {
        hub,
        policy,
        policyPath: join(blocker, 'capability-policy.json'),
      });
      assert.notEqual(handler, undefined);
      try {
        const { captured, res } = fakeResponse();
        await (handler as never as (req: IncomingMessage, res: ServerResponse) => Promise<void>)(
          requestWith('PUT', { capability: 'text_to_image', order: [{ id: 'krea_turbo', enabled: true }] }),
          res,
        );
        assert.equal(captured.status, 200, 'the edit applies for this session');
        const payload = json(captured.body);
        assert.ok(Array.isArray(payload['saveIssues']));
        assert.equal((payload['saveIssues'] as string[]).length, 1);
        assert.match((payload['saveIssues'] as string[])[0] ?? '', /could not save/);
        assert.deepEqual(policy.entryFor('text_to_image')?.order, [{ id: 'krea_turbo', enabled: true }]);
        assert.ok(lines.some((line) => line.startsWith('warn') && line.includes('could not save')));
      } finally {
        await hub.dispose();
      }
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
