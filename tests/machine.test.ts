/**
 * Tests for machine resource probing and its wiring into the hub.
 *
 * The probe answers two different questions and the tests keep them apart:
 * *capacity* ("could this model ever run here?") and *headroom* ("will it fit
 * right now?"). Conflating them is the failure mode with real consequences — a
 * second model started onto a GPU the first one has already filled dies with an
 * engine-level out-of-memory instead of a routing refusal.
 *
 * The probe itself shells out to `nvidia-smi`, which is exactly the kind of thing
 * that makes a test suite machine-dependent. So the parsing and the arithmetic
 * are tested through {@link reserveResources} and the profile consumers, and the
 * one test that does call `probeMachine` asserts only what must be true on *any*
 * machine.
 *
 * @module dsh-ai-model-hub/tests/machine
 */

import { strict as assert } from 'node:assert';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, describe, it } from 'node:test';

import type { MachineProfile } from '../src/index.ts';
import { ModelHub, DEFAULT_EXECUTION_POLICY, reserveResources } from '../src/index.ts';
import { probeMachine } from '../src/machine.ts';

/** Temporary artifact roots, cleaned up once at the end. */
const roots: string[] = [];

after(async () => {
  await Promise.all(roots.map((root) => rm(root, { recursive: true, force: true })));
});

/** A machine profile with everything measurable filled in. */
function profile(overrides: Partial<MachineProfile> = {}): MachineProfile {
  return {
    vramGb: 8,
    ramGb: 32,
    hasGpu: true,
    notes: 'test profile',
    gpus: [{ name: 'Test GPU', vramGb: 8, freeVramGb: 6 }],
    availableVramGb: 6,
    availableRamGb: 24,
    availableDiskGb: 100,
    platform: 'test',
    arch: 'test',
    probedAt: Date.now(),
    ...overrides,
  };
}

describe('machine resource arithmetic', () => {
  it('subtracts resident use from headroom, never from capacity', () => {
    const reserved = reserveResources(profile(), { vramGb: 2.5, ramGb: 4 });
    assert.equal(reserved.vramGb, 8, 'capacity is a property of the hardware');
    assert.equal(reserved.ramGb, 32);
    assert.equal(reserved.availableVramGb, 3.5);
    assert.equal(reserved.availableRamGb, 20);
  });

  it('floors at zero rather than reporting negative memory', () => {
    const reserved = reserveResources(profile({ availableVramGb: 1 }), { vramGb: 4 });
    assert.equal(reserved.availableVramGb, 0);
  });

  it('keeps an unmeasured figure unmeasured', () => {
    // "We never measured it" is not the same fact as "there is none left", and a
    // consumer that cannot tell them apart would refuse every model.
    const reserved = reserveResources(
      { vramGb: 8, ramGb: 32, hasGpu: true, notes: 'no probe' },
      { vramGb: 4, ramGb: 4 },
    );
    assert.equal(reserved.availableVramGb, undefined);
    assert.equal(reserved.availableRamGb, undefined);
  });
});

describe('machine probing', () => {
  it('always reports OS memory and a platform, whatever the hardware', async () => {
    const { profile: probed, evidence } = await probeMachine({ skipGpuProbe: true, skipDiskProbe: true });
    assert.ok(probed.ramGb > 0, 'system RAM is always measurable');
    assert.equal(probed.hasGpu, false, 'the GPU probe was skipped');
    assert.equal(probed.vramGb, 0);
    assert.equal(probed.platform, process.platform);
    assert.equal(probed.arch, process.arch);
    assert.equal(typeof probed.probedAt, 'number');
    assert.ok(evidence.length >= 2);
    assert.ok(evidence.some((line) => /GPU probe skipped/.test(line)));
  });

  it('reports free space on the filesystem it is pointed at', async () => {
    const root = await mkdtemp(join(tmpdir(), 'aimh-machine-'));
    roots.push(root);
    const { profile: probed } = await probeMachine({
      skipGpuProbe: true,
      diskPath: root,
    });
    assert.ok((probed.availableDiskGb ?? 0) > 0, 'a real filesystem has free space');
  });

  it('runs nvidia-smi even under a narrowed command policy', async () => {
    // The probe widens the policy it is handed by exactly one diagnostic binary;
    // without that, a deployment whose allowlist excludes `nvidia-smi` silently
    // reports no GPU, and every GPU model is then disqualified.
    const { evidence } = await probeMachine({
      skipDiskProbe: true,
      policy: { ...DEFAULT_EXECUTION_POLICY, allowlist: ['python'], allowAnyCommand: false },
    });
    assert.ok(
      !evidence.some((line) => /not in the command allowlist/.test(line)),
      `the probe must not be blocked by the model-command allowlist: ${evidence.join('; ')}`,
    );
  });
});

describe('hub resource wiring', () => {
  /** Build a hub with the given model and machine. */
  async function hubWith(
    model: Record<string, unknown>,
    options: { readonly machine?: MachineProfile; readonly probeResources?: boolean } = {},
  ): Promise<ModelHub> {
    const root = await mkdtemp(join(tmpdir(), 'aimh-machine-hub-'));
    roots.push(root);
    return ModelHub.fromConfig(
      {
        version: '1',
        models: [
          {
            id: 'test_model',
            name: 'Test model',
            type: 'text_generation',
            capabilities: ['text_to_text'],
            adapter: 'mock',
            runtime: { engine: 'in_process_mock', adapter: 'mock' },
            priority: 10,
            ...model,
          },
        ],
      },
      {
        artifactRoot: root,
        manageTimers: false,
        probeResources: options.probeResources ?? false,
        ...(options.machine === undefined ? {} : { machine: options.machine }),
        log: () => {},
      },
    );
  }

  it('routes against the profile it was given, and rejects what does not fit', async () => {
    const hub = await hubWith(
      { resources: { vramGb: 12, ramGb: 8, requiresGpu: true } },
      { machine: profile({ vramGb: 8, availableVramGb: 5 }) },
    );
    try {
      const decision = await hub.route({ capability: 'text_to_text', prompt: 'hi' });
      assert.fail(`expected a refusal, got ${decision.modelId}`);
    } catch (error) {
      const described = error as { code?: string; message?: string };
      assert.equal(described.code, 'NO_COMPATIBLE_MODEL');
      assert.match(described.message ?? '', /needs 12 GiB VRAM but only 8 GiB is present on this machine/);
    } finally {
      await hub.dispose();
    }
  });

  it('refuses to start a model that needs a GPU this machine does not have', async () => {
    const noGpu: MachineProfile = {
      vramGb: 0,
      ramGb: 32,
      hasGpu: false,
      notes: 'no accelerator detected',
      gpus: [],
    };
    const hub = await hubWith({ resources: { vramGb: 6, requiresGpu: true } }, { machine: noGpu });
    try {
      await assert.rejects(
        () => hub.startModel('test_model'),
        (error: unknown) => {
          const described = error as { code?: string; message?: string };
          // Not startable by the hub either way, but the *resource* refusal is
          // what must win: it is the actionable one.
          assert.ok(
            described.code === 'INSUFFICIENT_RESOURCES' || described.code === 'LIFECYCLE_UNSUPPORTED',
            `unexpected code ${String(described.code)}`,
          );
          return true;
        },
      );
      assert.equal(hub.getModelStatus('test_model').availability, 'unsupported');
      assert.match(hub.getModelStatus('test_model').reason ?? '', /requires a GPU and none was detected/);
    } finally {
      await hub.dispose();
    }
  });

  it('publishes a probed profile and re-checks models against it', async () => {
    // Probing is on and no hand-supplied profile competes with it: this is the
    // production wiring, and it must leave the catalog with a *measured* profile.
    const hub = await hubWith(
      { resources: { vramGb: 2, ramGb: 1, requiresGpu: false } },
      { probeResources: true },
    );
    try {
      const probed = await hub.ensureResourcesFresh(0);
      assert.equal(typeof probed.probedAt, 'number');
      assert.ok(probed.ramGb > 0);
      const snapshot = hub.resourceSnapshot();
      assert.ok(snapshot !== undefined);
      assert.equal(snapshot?.profile.ramGb, probed.ramGb);
      // An unstartable mock model stays routable; the point is that measuring the
      // machine did not accidentally disqualify something that fits it.
      assert.notEqual(hub.getModelStatus('test_model').availability, 'unsupported');
      const decision = await hub.route({ capability: 'text_to_text', prompt: 'hi' });
      assert.equal(decision.modelId, 'test_model');
    } finally {
      await hub.dispose();
    }
  });

  it('invalidates a stale measurement after an invocation, so the next decision re-probes', async () => {
    const hub = await hubWith({ resources: { vramGb: 1, ramGb: 1 } }, { probeResources: true });
    try {
      await hub.ensureResourcesFresh(0);
      const before = hub.resourceSnapshot();
      assert.ok((before?.profile.probedAt ?? 0) > 0);

      await hub.invokeModel({ capability: 'text_to_text', prompt: 'hello' });
      const after = hub.resourceSnapshot();
      // The measurement is marked as taken before the generation rather than
      // discarded, so `availableResources` still answers — but it is no longer
      // treated as fresh.
      assert.equal(after?.profile.probedAt, 0);

      const refreshed = await hub.ensureResourcesFresh(60_000);
      assert.ok((refreshed.probedAt ?? 0) > 0, 'a stale measurement is re-taken on demand');
    } finally {
      await hub.dispose();
    }
  });

  it('reports what is left once the running models are accounted for', async () => {
    const hub = await hubWith(
      { resources: { vramGb: 4, ramGb: 8, requiresGpu: true } },
      { machine: profile({ availableVramGb: 6, availableRamGb: 24 }) },
    );
    try {
      await hub.probeModel('test_model');
      const available = hub.availableResources();
      assert.deepEqual(available.residentModelIds, ['test_model']);
      assert.equal(available.reserved.vramGb, 4);
      assert.equal(available.reserved.ramGb, 8);
      assert.equal(available.profile.availableVramGb, 2);
      assert.equal(available.profile.availableRamGb, 16);
      assert.equal(available.profile.vramGb, 8, 'capacity is untouched');
    } finally {
      await hub.dispose();
    }
  });
});
