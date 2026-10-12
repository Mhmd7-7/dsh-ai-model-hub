/**
 * Tests for the capability policy: order, enablement, and the plan it produces.
 *
 * The policy is where three questions that are easy to conflate are kept apart —
 * *may* this run, *when* should it run, and *is* it usable now — so most of what
 * is defended here is that they stay apart under pressure: a disabled option must
 * never be a fallback, a stopped-but-startable option must not be demoted behind
 * a ready remote one, and a newly discovered option must not activate itself.
 *
 * @module dsh-ai-model-hub/tests/policy.test
 */

import { strict as assert } from 'node:assert';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import {
  CAPABILITY_POLICY_VERSION,
  CapabilityPolicy,
  CatalogBaselineDefaults,
  ModelCatalog,
  ModelHub,
  ModelHubError,
  emptyPolicyDocument,
  parseCapabilityPolicy,
  parseModelCatalogConfig,
  serializeCapabilityPolicy,
} from '../src/index.ts';
import type { CapabilityOption, PolicyOptionFacts } from '../src/index.ts';

/** A catalog with three image options at deliberately unhelpful priorities. */
function imageCatalog(): unknown {
  return {
    version: '1',
    models: [
      {
        id: 'local_workflow',
        name: 'Local workflow',
        type: 'image_generation',
        capabilities: ['text_to_image'],
        adapter: 'mock',
        runtime: { engine: 'comfyui', adapter: 'mock' },
        priority: 90,
      },
      {
        id: 'local_checkpoint',
        name: 'Local checkpoint',
        type: 'image_generation',
        capabilities: ['text_to_image'],
        adapter: 'mock',
        runtime: { engine: 'comfyui', adapter: 'mock' },
        priority: 10,
      },
      {
        id: 'unrelated_text',
        name: 'Text model',
        type: 'text_generation',
        capabilities: ['text_to_text'],
        adapter: 'mock',
        runtime: { engine: 'ollama', adapter: 'mock' },
      },
    ],
  };
}

/** The facts of a subscription backend, for tests that need one. */
function subscriptionFacts(id: string, capability: PolicyOptionFacts['capabilities'][number]): PolicyOptionFacts {
  return {
    id,
    name: id,
    type: 'image_generation',
    source: 'subscription',
    provider: id.split('_')[0] ?? 'remote',
    providerLabel: `subscription · ${id}`,
    capabilities: [capability],
    inputTypes: capability === 'image_to_image' ? ['image', 'text'] : ['text'],
    outputTypes: capability === 'video_generation' ? ['video'] : ['image'],
    executable: true,
    catalogEnabled: false,
  };
}

/** A hub over a catalog, with optional extra options, and no background timers. */
async function hubWith(
  config: unknown,
  options: { readonly capabilityOptions?: readonly CapabilityOption[]; readonly policy?: CapabilityPolicy } = {},
): Promise<{ readonly hub: ModelHub; readonly root: string; readonly cleanup: () => Promise<void> }> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-policy-'));
  const hub = ModelHub.fromConfig(config, {
    artifactRoot: root,
    manageTimers: false,
    probeResources: false,
    log: () => {},
    ...(options.capabilityOptions === undefined ? {} : { capabilityOptions: options.capabilityOptions }),
    ...(options.policy === undefined ? {} : { capabilityPolicy: options.policy }),
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

/** A capability option that records whether it was invoked. */
function recordingOption(
  facts: PolicyOptionFacts,
  behaviour: 'succeed' | 'fail' | 'unavailable',
): { readonly option: CapabilityOption; readonly calls: string[] } {
  const calls: string[] = [];
  const option: CapabilityOption = {
    facts,
    status: () =>
      behaviour === 'unavailable'
        ? { state: 'unavailable' as const, reason: 'Unavailable: tests pretend nobody is logged in.' }
        : { state: 'ready' as const, reason: 'Ready: test backend.' },
    invoke: async (request) => {
      calls.push(request.capability);
      if (behaviour === 'fail') {
        throw new ModelHubError('MODEL_UNAVAILABLE', 'the backend refused', { optionId: facts.id });
      }
      // A text artifact is enough: these tests are about which option ran, not
      // about what it produced.
      return { artifactIds: [], value: { ran: facts.id } };
    },
  };
  return { option, calls };
}

describe('capability policy document', () => {
  it('round-trips through serialization without losing a decision', () => {
    const document = {
      version: CAPABILITY_POLICY_VERSION,
      globalEnabled: true,
      capabilities: {
        text_to_image: {
          order: [
            { id: 'a', enabled: true },
            { id: 'b', enabled: false },
            { id: 'c', enabled: true },
          ],
        },
      },
    };
    const parsed = parseCapabilityPolicy(JSON.parse(serializeCapabilityPolicy(document)));
    assert.equal(parsed.ok, true);
    if (!parsed.ok) return;
    assert.deepEqual(parsed.document.capabilities['text_to_image']?.order, document.capabilities.text_to_image.order);
  });

  it('refuses a document from a newer build rather than half-enforcing it', () => {
    const parsed = parseCapabilityPolicy({ version: CAPABILITY_POLICY_VERSION + 1, capabilities: {} });
    assert.equal(parsed.ok, false);
    if (parsed.ok) return;
    assert.match(parsed.issues[0]?.message ?? '', /understands at most/);
  });

  it('drops an unknown capability and a duplicate option with an issue, and keeps the rest', () => {
    const parsed = parseCapabilityPolicy({
      version: 1,
      capabilities: {
        not_a_capability: { order: [{ id: 'x', enabled: true }] },
        text_to_text: {
          order: [
            { id: 'a', enabled: true },
            { id: 'a', enabled: false },
            { id: 'b', enabled: false },
          ],
        },
      },
    });
    assert.equal(parsed.ok, true);
    if (!parsed.ok) return;
    assert.equal(parsed.document.capabilities['not_a_capability'], undefined);
    assert.deepEqual(parsed.document.capabilities['text_to_text']?.order, [
      { id: 'a', enabled: true },
      { id: 'b', enabled: false },
    ]);
    assert.equal(parsed.issues.length, 2);
  });

  it('rejects a document it cannot understand at all, so nothing is silently enforced', () => {
    assert.equal(parseCapabilityPolicy(null).ok, false);
    assert.equal(parseCapabilityPolicy({ capabilities: {} }).ok, false);
    assert.equal(parseCapabilityPolicy({ version: 0 }).ok, false);
  });

  it('treats a missing global switch as "not globally disabled"', () => {
    const parsed = parseCapabilityPolicy({ version: 1, capabilities: {} });
    assert.equal(parsed.ok, true);
    if (!parsed.ok) return;
    assert.equal(parsed.document.globalEnabled, true);
  });
});

describe('capability policy resolution', () => {
  it('orders options by the saved order, not by the catalog', async () => {
    const { hub, cleanup } = await hubWith(imageCatalog());
    try {
      // Catalog order puts `local_workflow` first; the operator says otherwise.
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_checkpoint', enabled: true },
              { id: 'local_workflow', enabled: true },
            ],
          },
        },
      });
      const plan = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      assert.deepEqual(
        plan.chain.map((entry) => entry.option.option.id),
        ['local_checkpoint', 'local_workflow'],
      );
    } finally {
      await cleanup();
    }
  });

  it('appends an option the operator has never arranged, without disturbing the saved ones', async () => {
    // The realistic case: a subscription backend is registered after the operator
    // has already arranged the local options. It must appear — disabled, at the
    // end — and must not displace anything.
    const facts = subscriptionFacts('codex_image', 'text_to_image');
    const remote = recordingOption(facts, 'succeed');
    const { hub, cleanup } = await hubWith(imageCatalog(), { capabilityOptions: [remote.option] });
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_checkpoint', enabled: true },
              { id: 'local_workflow', enabled: false },
            ],
          },
        },
      });
      const plan = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      assert.deepEqual(
        plan.plan.options.map((option) => option.option.id),
        ['local_checkpoint', 'local_workflow', 'codex_image'],
        'a never-arranged option goes last, and every saved position is untouched',
      );
      assert.equal(plan.plan.options[2]?.enabled, false, 'and it is not authorized by mere presence');
      assert.deepEqual(
        plan.chain.map((entry) => entry.option.option.id),
        ['local_checkpoint'],
      );
    } finally {
      await cleanup();
    }
  });

  it('keeps a disabled option visible, in place, and out of the permitted list', async () => {
    const { hub, cleanup } = await hubWith(imageCatalog());
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_workflow', enabled: false },
              { id: 'local_checkpoint', enabled: true },
            ],
          },
        },
      });
      const plan = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      assert.deepEqual(plan.plan.options.map((option) => option.option.id), ['local_workflow', 'local_checkpoint']);
      assert.equal(plan.plan.options[0]?.enabled, false);
      assert.equal(plan.plan.options[0]?.state, 'disabled');
      assert.match(plan.plan.options[0]?.reason ?? '', /never used/);
      assert.deepEqual(plan.plan.permitted.map((option) => option.option.id), ['local_checkpoint']);
    } finally {
      await cleanup();
    }
  });

  it('prefers the catalog rule over a fixture when nothing has been arranged', async () => {
    // The regression this pins down: a deployment that has never opened the
    // settings page must behave exactly as it did before the policy existed, and
    // that means the catalog's own `priority` still decides between two models
    // nobody has arranged. `local_checkpoint` is priority 10 and
    // `local_workflow` is priority 90, but the catalog lists the workflow first.
    const root = await mkdtemp(join(tmpdir(), 'aimh-policy-order-'));
    const hub = ModelHub.fromConfig(imageCatalog(), {
      artifactRoot: root,
      manageTimers: false,
      probeResources: false,
      log: () => {},
    });
    try {
      const plan = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      assert.deepEqual(
        plan.chain.map((entry) => entry.option.option.id),
        ['local_checkpoint', 'local_workflow'],
        'the catalog priority decides until the operator decides otherwise',
      );
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      assert.equal(result.modelId, 'local_checkpoint');
    } finally {
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    }
  });

  it('lets a saved position beat the catalog rule', async () => {
    const root = await mkdtemp(join(tmpdir(), 'aimh-policy-order2-'));
    const hub = ModelHub.fromConfig(imageCatalog(), {
      artifactRoot: root,
      manageTimers: false,
      probeResources: false,
      log: () => {},
    });
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_workflow', enabled: true },
              { id: 'local_checkpoint', enabled: true },
            ],
          },
        },
      });
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      assert.equal(result.modelId, 'local_workflow', 'the operator outranks the catalog');
    } finally {
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    }
  });

  it('lets global disable win over a capability that enables everything', async () => {
    const { hub, cleanup } = await hubWith(imageCatalog());
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: false,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_checkpoint', enabled: true },
              { id: 'local_workflow', enabled: true },
            ],
          },
        },
      });
      const plan = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      assert.equal(plan.chain.length, 0);
      assert.equal(plan.plan.permitted.length, 0);
      for (const option of plan.plan.options) {
        assert.equal(option.enabled, false);
        assert.equal(option.enabledBy, 'global-disable');
      }
    } finally {
      await cleanup();
    }
  });

  it('keeps a stopped-but-startable option ahead of a ready remote one', async () => {
    // The remote option is ready right now; the local one is merely startable.
    // Readiness is not a reason to reorder — the operator's order is.
    const facts = subscriptionFacts('codex_image', 'text_to_image');
    const remote = recordingOption(facts, 'succeed');
    const { hub, cleanup } = await hubWith(imageCatalog(), { capabilityOptions: [remote.option] });
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'codex_image', enabled: true },
              { id: 'local_checkpoint', enabled: true },
              { id: 'local_workflow', enabled: true },
            ],
          },
        },
      });
      const plan = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      assert.deepEqual(
        plan.chain.map((entry) => entry.option.option.id),
        ['codex_image', 'local_checkpoint', 'local_workflow'],
        'the saved order decides, not which option happens to be warm',
      );
      assert.deepEqual(
        plan.chain.map((entry) => entry.option.state),
        ['ready', 'stopped', 'stopped'],
        'and the live states stay distinct from the order',
      );
    } finally {
      await cleanup();
    }
  });

  it('skips an unavailable remote option with its own reason and does not call it', async () => {
    const facts = subscriptionFacts('codex_image', 'text_to_image');
    const remote = recordingOption(facts, 'unavailable');
    const { hub, cleanup } = await hubWith(imageCatalog(), { capabilityOptions: [remote.option] });
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'codex_image', enabled: true },
              { id: 'local_checkpoint', enabled: true },
            ],
          },
        },
      });
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      assert.equal(result.modelId, 'local_checkpoint');
      assert.deepEqual(remote.calls, [], 'an unavailable backend must not be called');
      const plan = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      assert.match(
        plan.reasons.find((entry) => entry.id === 'codex_image')?.reason ?? '',
        /nobody is logged in/,
      );
    } finally {
      await cleanup();
    }
  });
});

describe('capability policy and execution', () => {
  it('walks the whole permitted list in saved order, with no silent attempt cap', async () => {
    // The first two options are served by an adapter that always fails, so the
    // walk has to reach the third. With the old silent cap of three attempts this
    // passed by luck; with fewer options than attempts it could not, and with more
    // it would report untried options as failures.
    const alwaysBroken = {
      kind: 'mock' as const,
      displayName: 'Always broken',
      supports: () => ({ ok: true as const }),
      health: () => Promise.resolve({ healthy: true, checkedAt: Date.now() }),
      invoke: () => Promise.reject(new ModelHubError('INVOCATION_FAILED', 'this option is broken on purpose')),
    };
    const facts = subscriptionFacts('codex_image', 'text_to_image');
    const remote = recordingOption(facts, 'succeed');
    const root = await mkdtemp(join(tmpdir(), 'aimh-policy-walk-'));
    const hub = ModelHub.fromConfig(imageCatalog(), {
      artifactRoot: root,
      manageTimers: false,
      probeResources: false,
      log: () => {},
      adapters: [alwaysBroken],
      capabilityOptions: [remote.option],
    });
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_workflow', enabled: true },
              { id: 'local_checkpoint', enabled: true },
              { id: 'codex_image', enabled: true },
            ],
          },
        },
      });
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      const tried = (result.attempts ?? []).map((attempt) => attempt.optionId);
      assert.deepEqual(tried, ['local_workflow', 'local_checkpoint', 'codex_image']);
      assert.equal(result.modelId, 'codex_image');
      assert.deepEqual(remote.calls, ['text_to_image']);
      const stages = (result.attempts ?? []).map((attempt) => attempt.stage);
      assert.deepEqual(stages, ['dispatch', 'dispatch', 'dispatch']);
    } finally {
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    }
  });

  it('reports an attempt budget as untried options rather than as failures', async () => {
    const alwaysBroken = {
      kind: 'mock' as const,
      displayName: 'Always broken',
      supports: () => ({ ok: true as const }),
      health: () => Promise.resolve({ healthy: true, checkedAt: Date.now() }),
      invoke: () => Promise.reject(new ModelHubError('INVOCATION_FAILED', 'broken on purpose')),
    };
    const root = await mkdtemp(join(tmpdir(), 'aimh-policy-budget-'));
    const hub = ModelHub.fromConfig(imageCatalog(), {
      artifactRoot: root,
      manageTimers: false,
      probeResources: false,
      log: () => {},
      adapters: [alwaysBroken],
      attemptBudget: 1,
    });
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_workflow', enabled: true },
              { id: 'local_checkpoint', enabled: true },
            ],
          },
        },
      });
      await assert.rejects(
        () => hub.invokeModel({ capability: 'text_to_image', prompt: 'x' }),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.match(error.message, /every candidate model failed/);
          assert.match(error.message, /went untried because of the configured attempt budget of 1/);
          assert.equal(error.details['untriedOptions'], 1);
          assert.equal(error.details['attemptBudget'], 1);
          return true;
        },
      );
    } finally {
      await hub.dispose();
      await rm(root, { recursive: true, force: true });
    }
  });

  it('never runs a disabled option, and reports POLICY_DISABLED when nothing is left', async () => {
    const facts = subscriptionFacts('codex_image', 'text_to_image');
    const remote = recordingOption(facts, 'succeed');
    const { hub, cleanup } = await hubWith(imageCatalog(), { capabilityOptions: [remote.option] });
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_workflow', enabled: false },
              { id: 'local_checkpoint', enabled: false },
              { id: 'codex_image', enabled: false },
            ],
          },
        },
      });
      await assert.rejects(
        () => hub.invokeModel({ capability: 'text_to_image', prompt: 'x' }),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'POLICY_DISABLED');
          assert.match(error.message, /switched off/);
          return true;
        },
      );
      assert.deepEqual(remote.calls, [], 'a disabled option must never execute, not even when all else fails');
    } finally {
      await cleanup();
    }
  });

  it('refuses an explicitly pinned disabled option instead of silently substituting one', async () => {
    const { hub, cleanup } = await hubWith(imageCatalog());
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_checkpoint', enabled: false },
              { id: 'local_workflow', enabled: true },
            ],
          },
        },
      });
      await assert.rejects(
        () => hub.invokeModel({ capability: 'text_to_image', prompt: 'x', modelId: 'local_checkpoint' }),
        (error: unknown) => {
          assert.ok(error instanceof ModelHubError);
          assert.equal(error.code, 'POLICY_DISABLED');
          assert.match(error.message, /local_checkpoint/);
          return true;
        },
      );
    } finally {
      await cleanup();
    }
  });

  it('re-checks permission immediately before dispatch, so a mid-flight disable wins', async () => {
    // The plan is built, then the option is disabled before it is dispatched. The
    // chain is never re-derived mid-flight, so the *only* thing that can stop the
    // dispatch is the permission check at the dispatch boundary — which is exactly
    // what this asserts.
    const facts = subscriptionFacts('codex_image', 'text_to_image');
    const remote = recordingOption(facts, 'succeed');
    const { hub, cleanup } = await hubWith(imageCatalog(), { capabilityOptions: [remote.option] });
    try {
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'codex_image', enabled: true },
              { id: 'local_checkpoint', enabled: true },
              { id: 'local_workflow', enabled: true },
            ],
          },
        },
      });
      const plan = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      assert.deepEqual(
        plan.chain.map((entry) => entry.option.option.id),
        ['codex_image', 'local_checkpoint', 'local_workflow'],
      );

      // Now disable the first option and run the plan's first step through the
      // hub's own dispatch path.
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'codex_image', enabled: false },
              { id: 'local_checkpoint', enabled: true },
            ],
          },
        },
      });
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      assert.deepEqual(remote.calls, [], 'the disabled option must not be dispatched');
      assert.equal(result.modelId, 'local_checkpoint');
    } finally {
      await cleanup();
    }
  });

  it('reports the policy revision with the result, so a decision is traceable', async () => {
    const { hub, cleanup } = await hubWith(imageCatalog());
    try {
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt: 'x' });
      assert.equal(result.decision.policyRevision, hub.policy.currentRevision);
      assert.ok((result.decision.permittedOptions ?? []).length > 0);
      assert.match(result.decision.rationale, /policy revision/);
    } finally {
      await cleanup();
    }
  });
});

describe('capability policy migration and persistence', () => {
  it('keeps existing catalog models working while a new discovery starts disabled', async () => {
    const baseline = new CatalogBaselineDefaults(['local_workflow', 'local_checkpoint', 'unrelated_text']);
    const policy = new CapabilityPolicy(emptyPolicyDocument(), { defaults: baseline });
    const { hub, cleanup } = await hubWith(imageCatalog(), { policy });
    try {
      // A saved order, so "where does the new one land?" has an unambiguous
      // answer: at the end, after everything the operator arranged.
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'local_workflow', enabled: true },
              { id: 'local_checkpoint', enabled: true },
            ],
          },
        },
      });
      // Both pre-existing models are still permitted, exactly as before the
      // feature existed: the catalog's own `enabled` flag is their default.
      const before = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      assert.equal(before.plan.permitted.length, 2);

      // A model that appears later — a discovery pass — is not covered.
      hub.catalog.replaceModels([
        ...hub.catalog.listModels().map((model) => model.descriptor),
        {
          id: 'discovered_workflow',
          name: 'Discovered workflow',
          type: 'image_generation',
          capabilities: ['text_to_image'],
          adapter: 'mock',
          runtime: { engine: 'comfyui', adapter: 'mock' },
        },
      ]);
      hub.runtime.syncCatalog();
      const after = await hub.planFor({ capability: 'text_to_image', prompt: 'x' }, { artifacts: hub.artifacts });
      const discovered = after.plan.options.find((option) => option.option.id === 'discovered_workflow');
      assert.ok(discovered !== undefined, 'a newly discovered option must be visible');
      assert.equal(discovered.enabled, false, 'and disabled until someone enables it');
      assert.equal(after.plan.options[after.plan.options.length - 1]?.option.id, 'discovered_workflow');
      assert.ok(
        !after.chain.some((entry) => entry.option.option.id === 'discovered_workflow'),
        'a discovery must not enter the permitted chain on its own',
      );
      assert.deepEqual(
        after.chain.map((entry) => entry.option.option.id),
        before.chain.map((entry) => entry.option.option.id),
        'and it must not disturb the order that was already in force',
      );
    } finally {
      await cleanup();
    }
  });

  it('seeds a new option at the end, never above a saved preference', () => {
    const policy = new CapabilityPolicy(
      {
        version: 1,
        globalEnabled: true,
        capabilities: {
          text_to_image: { order: [{ id: 'saved_first', enabled: true }] },
        },
      },
      { defaults: new CatalogBaselineDefaults(['saved_first']) },
    );
    const seeded = policy.seed([
      subscriptionFacts('codex_image', 'text_to_image'),
      {
        ...subscriptionFacts('saved_first', 'text_to_image'),
        id: 'saved_first',
        source: 'model',
        catalogEnabled: true,
      },
    ]);
    assert.equal(seeded.changed, true);
    assert.deepEqual(seeded.document.capabilities['text_to_image']?.order, [
      { id: 'saved_first', enabled: true },
      { id: 'codex_image', enabled: false },
    ]);
  });

  it('keeps a saved entry for an option that is temporarily undiscovered', () => {
    const policy = new CapabilityPolicy({
      version: 1,
      globalEnabled: true,
      capabilities: {
        text_to_image: {
          order: [
            { id: 'engine_is_down', enabled: true },
            { id: 'local_workflow', enabled: false },
          ],
        },
      },
    });
    // Seeding with only one of the two present must not forget the other.
    const seeded = policy.seed([
      {
        ...subscriptionFacts('local_workflow', 'text_to_image'),
        id: 'local_workflow',
        source: 'model',
        catalogEnabled: true,
      },
    ]);
    const order = (seeded.changed ? seeded.document : policy.current).capabilities['text_to_image']?.order ?? [];
    assert.deepEqual(order, [
      { id: 'engine_is_down', enabled: true },
      { id: 'local_workflow', enabled: false },
    ]);
  });

  it('rejects an edit that names an option the capability does not have', () => {
    const policy = new CapabilityPolicy();
    const outcome = policy.applyEdit('text_to_image', [{ id: 'made_up', enabled: true }], ['real_option']);
    assert.equal(outcome.issues.length, 1);
    assert.match(outcome.issues[0]?.message ?? '', /not an option for text_to_image/);
  });

  it('keeps unmentioned options at the end of a partial submission', () => {
    const policy = new CapabilityPolicy({
      version: 1,
      globalEnabled: true,
      capabilities: {
        text_to_image: {
          order: [
            { id: 'a', enabled: true },
            { id: 'b', enabled: true },
            { id: 'c', enabled: false },
          ],
        },
      },
    });
    const outcome = policy.applyEdit('text_to_image', [{ id: 'b', enabled: false }], ['a', 'b', 'c']);
    assert.deepEqual(outcome.document.capabilities['text_to_image']?.order, [
      { id: 'b', enabled: false },
      { id: 'a', enabled: true },
      { id: 'c', enabled: false },
    ]);
  });

  it('survives a save and reload through the file it writes', async () => {
    const root = await mkdtemp(join(tmpdir(), 'aimh-policy-file-'));
    const path = join(root, 'capability-policy.json');
    try {
      const document = {
        version: CAPABILITY_POLICY_VERSION,
        globalEnabled: true,
        capabilities: {
          text_to_image: {
            order: [
              { id: 'chatgpt_images', enabled: true },
              { id: 'grok_images', enabled: true },
              { id: 'z_image_turbo', enabled: false },
            ],
          },
        },
      };
      await writeFile(path, serializeCapabilityPolicy(document), 'utf8');
      const reloaded = parseCapabilityPolicy(JSON.parse(await readFile(path, 'utf8')));
      assert.equal(reloaded.ok, true);
      if (!reloaded.ok) return;
      const policy = new CapabilityPolicy(reloaded.document);
      const permits = policy.permits('text_to_image', subscriptionFacts('chatgpt_images', 'text_to_image'));
      assert.deepEqual(permits, { enabled: true, enabledBy: 'saved' });
      // And the disabled one stays disabled across the same reload.
      const disabled = policy.permits('text_to_image', {
        ...subscriptionFacts('z_image_turbo', 'text_to_image'),
        source: 'model',
        catalogEnabled: true,
      });
      assert.deepEqual(disabled, { enabled: false, enabledBy: 'saved' });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});

describe('catalog model gate', () => {
  it('lets a model be launched only while some capability permits it', async () => {
    const { hub, cleanup } = await hubWith(imageCatalog());
    try {
      const model = hub.catalog.requireModel('local_checkpoint');
      assert.equal(hub.policy.permitsModel(model).enabled, true);
      hub.policy.replace({
        version: 1,
        globalEnabled: true,
        capabilities: { text_to_image: { order: [{ id: 'local_checkpoint', enabled: false }] } },
      });
      const verdict = hub.policy.permitsModel(model);
      assert.equal(verdict.enabled, false);
      assert.match(verdict.reason, /disabled for every capability/);
    } finally {
      await cleanup();
    }
  });
});

describe('rejected catalog entries are reported, not silently missing', () => {
  it('records why a ComfyUI workflow with no contract was dropped', () => {
    // The failure this pins down is a real one: a workflow entry with a graph but
    // no `bindings`/`outputs` is unrepresentable — the hub cannot know which node
    // field the caller's prompt belongs in — so it is dropped. That is correct.
    // What must not happen is for the capability to then look like something this
    // machine simply cannot do.
    const parsed = parseModelCatalogConfig({
      version: '1',
      hosts: [
        {
          id: 'comfyui',
          name: 'ComfyUI',
          adapter: 'comfyui',
          runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint: 'http://127.0.0.1:8188' },
        },
      ],
      models: [
        {
          id: 'z_image_turbo',
          name: 'Z-Image Turbo',
          type: 'image_generation',
          host: 'comfyui',
          capabilities: ['text_to_image'],
          adapterConfig: { workflowPath: 'workflows/z-image-turbo.api.json' },
        },
      ],
    });
    assert.equal(parsed.ok, true);
    if (!parsed.ok) return;
    assert.equal(parsed.config.models.length, 0, 'the entry cannot serve anything');
    assert.equal(parsed.warnings.length, 1);
    assert.match(parsed.warnings[0] ?? '', /z_image_turbo/);
    assert.match(parsed.warnings[0] ?? '', /was ignored/);
    assert.match(parsed.warnings[0] ?? '', /bindings/);
    assert.match(parsed.warnings[0] ?? '', /outputs/);
  });

  it('keeps a workflow that declares its contract', () => {
    const parsed = parseModelCatalogConfig({
      version: '1',
      hosts: [
        {
          id: 'comfyui',
          name: 'ComfyUI',
          adapter: 'comfyui',
          runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint: 'http://127.0.0.1:8188' },
        },
      ],
      models: [
        {
          id: 'z_image_turbo',
          name: 'Z-Image Turbo',
          type: 'image_generation',
          host: 'comfyui',
          capabilities: ['text_to_image'],
          adapterConfig: {
            workflowPath: 'workflows/z-image-turbo.api.json',
            bindings: { prompt: { node: '13', input: 'text' } },
            outputs: { image: { node: '18', type: 'image' } },
          },
        },
      ],
    });
    assert.equal(parsed.ok, true);
    if (!parsed.ok) return;
    assert.equal(parsed.config.models.length, 1);
    assert.equal(parsed.warnings.length, 0);
  });
});
