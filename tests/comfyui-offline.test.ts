/**
 * Offline workflow discovery: saved workflows must stay visible with ComfyUI down.
 *
 * The bug these tests exist for is the one every "live-only" reader has: the list
 * of saved workflows and the *health* of the engine that saved them are two
 * different facts, and collapsing them means a stopped server takes the user's
 * work off the screen. Each test therefore describes a machine — ComfyUI up, down,
 * answering with an error, or holding a directory the process cannot read — and
 * asserts on what the *user* sees, not on which internal source produced it.
 *
 * Nothing here spawns a process. ComfyUI is described through injected filesystem
 * and HTTP probes for the scan half, and through a real `ModelHub` plus a stubbed
 * runtime start for the execution half, because the behaviour under test is "the
 * existing lifecycle manager is asked to start the engine", not "a child process
 * appears". Where a test needs the engine to answer over a real socket it starts a
 * local HTTP server, and those two are the only tests in this file that speak the
 * real protocol.
 *
 * @module dsh-ai-model-hub/tests/comfyui-offline
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, beforeEach, describe, it } from 'node:test';

import { ModelHub } from '../src/index.ts';
import type { ModelCatalogConfig, ModelHost } from '../src/index.ts';
import { buildInventory, DEFAULT_PROBES } from '../dsh-plugin/inventory.ts';
import type { Inventory, InventoryProbes } from '../dsh-plugin/inventory.ts';
import { scanLocalResources, resetScanCache } from '../dsh-plugin/scan.ts';
import type { LocalResource, ScannedHostSnapshot } from '../dsh-plugin/scan.ts';
import {
  loadWorkflowCatalog,
  recordScan,
  rememberedSnapshots,
  saveWorkflowCatalog,
  defaultWorkflowCatalogPath,
  workflowScopeKey,
  WORKFLOW_CATALOG_VERSION,
} from '../dsh-plugin/workflow-catalog.ts';
import { resolveConfiguredPath } from '../dsh-plugin/paths.ts';
import type { PluginLogger } from '../dsh-plugin/types.ts';

// ── fixtures ────────────────────────────────────────────────────────────────

/** A complete text-to-image workflow in API format. */
const apiWorkflow = {
  '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'A.safetensors' } },
  '2': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: 'a red fox' } },
  '3': { class_type: 'CLIPTextEncode', inputs: { clip: ['1', 1], text: '' } },
  '4': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 512, batch_size: 1 } },
  '5': {
    class_type: 'KSampler',
    inputs: { model: ['1', 0], positive: ['2', 0], negative: ['3', 0], latent_image: ['4', 0], seed: 1, steps: 20, cfg: 7 },
  },
  '6': { class_type: 'VAEDecode', inputs: { samples: ['5', 0], vae: ['1', 2] } },
  '7': { class_type: 'SaveImage', inputs: { images: ['6', 0], filename_prefix: 'x' } },
};

/** The node metadata a real ComfyUI reports through `/object_info`. */
const objectInfo = {
  CheckpointLoaderSimple: { input: { required: { ckpt_name: [['A.safetensors'], {}] } } },
  CLIPTextEncode: { input: { required: { clip: ['CLIP', {}], text: ['STRING', {}] } }, output: ['CONDITIONING'] },
  EmptyLatentImage: { input: { required: { width: ['INT', {}], height: ['INT', {}], batch_size: ['INT', {}] } }, output: ['LATENT'] },
  KSampler: {
    input: { required: { model: ['MODEL', {}], positive: ['CONDITIONING', {}], negative: ['CONDITIONING', {}], latent_image: ['LATENT', {}], seed: ['INT', {}], steps: ['INT', {}], cfg: ['FLOAT', {}] } },
    output: ['LATENT'],
  },
  VAEDecode: { input: { required: { samples: ['LATENT', {}], vae: ['VAE', {}] } }, output: ['IMAGE'] },
  SaveImage: { input: { required: { images: ['IMAGE', {}], filename_prefix: ['STRING', {}] } }, output: ['IMAGE'], output_node: true },
};

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

/** The catalog this file's machines declare: one ComfyUI host, one endpoint. */
function catalogFor(endpoint: string): ModelCatalogConfig {
  return {
    version: '1',
    hosts: [
      {
        id: 'comfyui',
        name: 'ComfyUI',
        adapter: 'comfyui',
        runtime: { engine: 'comfyui', adapter: 'comfyui', endpoint, path: '/prompt' },
        // Startable, so the execution half can be tested without a real install:
        // the lifecycle manager is what starts it, and the test stubs exactly that.
        lifecycle: { startable: true, stoppable: true, startupTimeoutMs: 5_000, start: { command: 'python.exe', args: ['main.py'] } },
      },
    ],
    models: [],
  };
}

/** The hosts of the fixture catalog, as fresh objects each time. */
function hostsFor(endpoint: string): ModelHost[] {
  return [...(catalogFor(endpoint).hosts ?? [])];
}

/**
 * A hub over the fixture catalog.
 *
 * @param endpoint - the ComfyUI endpoint to declare.
 * @returns the hub.
 */
function hubFor(endpoint: string): ModelHub {
  return ModelHub.fromConfig(catalogFor(endpoint), { manageTimers: false, probeResources: false, log: () => {} });
}

/**
 * Keep the suite out of the real harness home.
 *
 * `buildInventory` reaches the default workflow-catalog location on every scan
 * that persists, which is correct in production and would otherwise mean a test
 * run writing into `~/.dsh` — and, worse, reading whatever a previous run or a
 * real session left there, which is how a test starts failing for a reason that
 * has nothing to do with the code.
 *
 * The redirect happens at **module load**, not in a `before` hook. A hook only
 * runs once a test does, so a run whose name filter matches nothing — or any code
 * that reaches the store earlier — would have written to the real home first. A
 * module-level `mkdtemp` costs one temporary directory and removes the whole class
 * of accident, including the one this suite caused.
 */
const isolationRoot = await mkdtemp(join(tmpdir(), 'hub-offline-suite-'));
const previousCatalogPath = process.env['DSH_AI_MODEL_HUB_WORKFLOW_CATALOG'];
process.env['DSH_AI_MODEL_HUB_WORKFLOW_CATALOG'] = join(isolationRoot, 'suite-catalog.json');
after(() => {
  if (previousCatalogPath === undefined) delete process.env['DSH_AI_MODEL_HUB_WORKFLOW_CATALOG'];
  else process.env['DSH_AI_MODEL_HUB_WORKFLOW_CATALOG'] = previousCatalogPath;
});

/**
 * The roots this suite's machine reports as existing directories.
 *
 * The inventory legitimately looks for a ComfyUI installation in well-known
 * places, so on a developer's machine it finds the real one and lists the real
 * saved workflows — which is exactly right in production and exactly wrong in a
 * test, because the expected row count then depends on what the person running
 * the suite happens to have installed. Registering the temporary root the test
 * created, and nothing else, keeps every fixture machine local to its own test.
 */
const knownRoots = new Set<string>();

/** Normalize a path for cross-platform comparison inside the stub machine. */
function normalizePath(path: string): string {
  return path.replace(/\\/g, '/').replace(/\/+$/, '');
}

/** Register a directory tree as existing, so the stub machine can find it. */
function registerRoot(path: string): void {
  knownRoots.add(normalizePath(path));
}

/**
 * Filesystem probes limited to the roots the current test created.
 *
 * Reading and listing a real directory inside those roots is deliberate: the
 * fixtures are real files, written by the test, so the parser and the directory
 * walker are exercised for real rather than against a mock of a directory.
 */
const localFilesystem: Pick<InventoryProbes, 'isDirectory' | 'listDir' | 'readFile'> = {
  isDirectory: async (path: string) => knownRoots.has(normalizePath(path)),
  listDir: async (path: string) => {
    if (!knownRoots.has(normalizePath(path))) return undefined;
    try {
      return await DEFAULT_PROBES.listDir(path);
    } catch {
      return undefined;
    }
  },
  readFile: async (path: string) => await DEFAULT_PROBES.readFile!(path),
};

/**
 * A machine where ComfyUI is stopped, described without touching the real one.
 *
 * @param overrides - probe members to replace.
 * @returns the probes.
 */
function offlineMachine(overrides: Partial<InventoryProbes> = {}): InventoryProbes {
  return {
    ...DEFAULT_PROBES,
    ...localFilesystem,
    fetchJson: async (url: string) => {
      throw Object.assign(new Error(`connect ECONNREFUSED (${url})`), { code: 'ECONNREFUSED' });
    },
    ...overrides,
  };
}

/** A machine where ComfyUI answers, serving the given saved workflows and nodes. */
function onlineMachine(options: {
  workflows?: Record<string, unknown>;
  objectInfo?: unknown;
  systemStats?: unknown;
  seen?: string[];
} = {}): InventoryProbes {
  const workflows = options.workflows ?? {};
  return {
    ...DEFAULT_PROBES,
    ...localFilesystem,
    fetchJson: async (url: string) => {
      options.seen?.push(url);
      if (url.endsWith('/system_stats') && !url.includes('/userdata')) {
        return options.systemStats ?? { system: { comfyui_version: '1' }, devices: [] };
      }
      if (url.endsWith('/object_info')) return options.objectInfo ?? objectInfo;
      if (url.includes('/userdata?')) return Object.keys(workflows);
      const match = /\/userdata\/(.+)$/.exec(url);
      if (match !== null) {
        const relative = decodeURIComponent(match[1] as string).replace(/^workflows\//, '');
        if (Object.hasOwn(workflows, relative)) return workflows[relative];
      }
      throw Object.assign(new Error(`connect ECONNREFUSED (${url})`), { code: 'ECONNREFUSED' });
    },
  };
}

/**
 * Create a temporary directory tree for one test.
 *
 * The root, its immediate children and the `workflows` directory are registered
 * with the stub machine, which is what lets the fixture's own files be read for
 * real while the developer's actual ComfyUI install stays out of the test.
 *
 * @returns the directory, and a helper that writes a file inside it.
 */
async function workspace(): Promise<{ root: string; write: (relative: string, contents: string) => Promise<string> }> {
  const root = await mkdtemp(join(tmpdir(), 'hub-offline-'));
  registerRoot(root);
  registerRoot(join(root, 'workflows'));
  registerRoot(join(root, 'workflows', 'nested'));
  registerRoot(join(root, 'state'));
  registerRoot(join(root, 'first'));
  registerRoot(join(root, 'first', 'workflows'));
  return {
    root,
    write: async (relative: string, contents: string) => {
      const path = join(root, relative);
      await mkdir(join(path, '..'), { recursive: true });
      await writeFile(path, contents, 'utf8');
      return path;
    },
  };
}

/** Build an inventory against a described machine. */
async function inventory(
  hub: ModelHub,
  options: {
    hosts: readonly ModelHost[];
    catalogPath: string;
    workflowDir?: string;
    workflowCatalogPath: string;
    probes: InventoryProbes;
    persistWorkflows?: boolean;
  },
): Promise<Inventory> {
  const { logger } = recordingLogger();
  return await buildInventory(hub, {
    hosts: options.hosts,
    catalogPath: options.catalogPath,
    artifactRoot: '',
    allowProcessLaunch: true,
    probes: options.probes,
    probe: true,
    workflowCatalogPath: options.workflowCatalogPath,
    logger,
    ...(options.workflowDir === undefined ? {} : { workflowDir: options.workflowDir }),
    ...(options.persistWorkflows === undefined ? {} : { persistWorkflows: options.persistWorkflows }),
  });
}

/** The workflow rows of an inventory, in list order. */
function workflowsOf(report: Inventory): LocalResource[] {
  return report.resources.filter((resource) => resource.kind === 'comfyui_workflow');
}

/** The advertised-status detail of one workflow row. */
function statusOf(report: Inventory, name: string): LocalResource {
  const found = workflowsOf(report).find((resource) => resource.name === name);
  assert.ok(found, `expected a workflow named "${name}", saw: ${workflowsOf(report).map((entry) => entry.name).join(', ')}`);
  return found;
}

describe('ComfyUI offline: saved workflows stay discoverable', () => {
  beforeEach(() => {
    resetScanCache();
    knownRoots.clear();
    registerRoot(isolationRoot);
  });

  it('discovers local workflow files on a first launch with no cache and no live API', async () => {
    // Requirement: offline discovery must work on first launch when valid local
    // workflow files exist, even with no previous successful API connection.
    const space = await workspace();
    await space.write('workflows/fox.json', JSON.stringify(apiWorkflow));
    const catalogPath = join(space.root, 'models.json');
    await writeFile(catalogPath, '{}', 'utf8');
    const workflowCatalogPath = join(space.root, 'state', 'workflow-catalog.json');

    const hub = hubFor('http://127.0.0.1:59999');
    try {
      const report = await inventory(hub, {
        hosts: hostsFor('http://127.0.0.1:59999'),
        catalogPath,
        workflowDir: join(space.root, 'workflows'),
        workflowCatalogPath,
        probes: offlineMachine(),
      });

      const fox = statusOf(report, 'fox');
      assert.equal(fox.status, 'ready', 'an API-format graph is understood without the engine');
      assert.equal(fox.runnable, true, 'and is runnable once the engine is started');
      assert.equal(fox.engineRunning, false, 'the row says the engine is not answering');
      assert.equal(fox.availability, 'unavailable');
      assert.match(fox.detail, /ComfyUI (stopped|unreachable)/);

      const comfy = report.scan.sources.find((source) => source.id === 'comfyui');
      assert.equal(comfy?.ok, false, 'the engine itself is still reported as unreachable');
      assert.match(comfy?.detail ?? '', /ComfyUI (stopped|unreachable)/, 'and the source says which state belongs to this evidence');
      assert.ok(report.scan.sources.some((source) => source.id === 'workflow-dir' && source.ok), 'the directory is a source of its own');
      assert.ok(report.scan.sources.every((source) => source.startedEngine !== true), 'nothing started ComfyUI');

      // The provider is registered, so invoke_model can pin it.
      assert.ok(hub.catalog.listModels().some((model) => model.id === fox.modelId), 'the workflow is a routable provider');

      // And it is now remembered, so the next launch needs no live source at all.
      const saved = await loadWorkflowCatalog(workflowCatalogPath, recordingLogger().logger);
      assert.equal(saved.found, true);
      assert.equal(saved.document.scopes.length, 1);
      assert.equal(saved.document.scopes[0]?.workflows[0]?.name, 'fox');
      assert.ok(saved.document.scopes[0]?.workflows[0]?.descriptor, 'its contract is remembered, not just its name');
    } finally {
      await hub.dispose();
    }
  });

  it('re-shows remembered workflows on a later launch while ComfyUI is stopped', async () => {
    // Requirement: the persisted catalog is a source, so a restart does not lose
    // the workflows an earlier scan found.
    const space = await workspace();
    const firstDir = join(space.root, 'first', 'workflows');
    await mkdir(firstDir, { recursive: true });
    await writeFile(join(firstDir, 'fox.json'), JSON.stringify(apiWorkflow), 'utf8');
    const catalogPath = join(space.root, 'models.json');
    await writeFile(catalogPath, '{}', 'utf8');
    const workflowCatalogPath = join(space.root, 'state', 'workflow-catalog.json');

    // Launch one: ComfyUI answers, and the workflow comes from its own store.
    const online = hubFor('http://127.0.0.1:8188');
    try {
      const report = await inventory(online, {
        hosts: hostsFor('http://127.0.0.1:8188'),
        catalogPath,
        workflowCatalogPath,
        probes: onlineMachine({ workflows: { 'fox.json': apiWorkflow } }),
      });
      assert.equal(statusOf(report, 'fox').freshness, 'live');
    } finally {
      await online.dispose();
    }

    // Launch two: a fresh hub, no in-memory cache, ComfyUI stopped, and the
    // directory that held the file gone with it.
    resetScanCache();
    const offline = hubFor('http://127.0.0.1:8188');
    try {
      const report = await inventory(offline, {
        hosts: hostsFor('http://127.0.0.1:8188'),
        catalogPath,
        workflowCatalogPath,
        probes: offlineMachine(),
      });

      const fox = statusOf(report, 'fox');
      assert.equal(fox.freshness, 'cached', 'the row is identified as a cached observation');
      assert.equal(fox.status, 'ready', 'its readiness is preserved');
      assert.equal(fox.runnable, true, 'and it is still runnable by starting the engine');
      assert.equal(fox.engineRunning, false);
      assert.match(fox.detail, /cached/);
      assert.match(fox.detail, /ComfyUI (stopped|unreachable)/);
      assert.doesNotMatch(fox.detail, /engine answering/, 'no stale sentence from when it was stored');
      assert.ok(report.scan.sources.some((source) => source.id === 'remembered' && source.found === 1), 'the remembered source reports itself');
      assert.ok(offline.catalog.listModels().some((model) => model.id === fox.modelId), 'the provider is published again, so it stays executable');
    } finally {
      await offline.dispose();
    }
  });

  it('keeps workflows listed after ComfyUI goes away mid-session, without duplicating them', async () => {
    // Requirement: a connection failure after a successful discovery is a status
    // change, not a deletion.
    const endpoint = 'http://127.0.0.1:8188';
    const space = await workspace();
    const catalogPath = join(space.root, 'models.json');
    await writeFile(catalogPath, '{}', 'utf8');
    const workflowCatalogPath = join(space.root, 'state', 'workflow-catalog.json');
    const hub = hubFor(endpoint);
    try {
      const first = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowCatalogPath,
        probes: onlineMachine({ workflows: { 'fox.json': apiWorkflow } }),
      });
      assert.equal(workflowsOf(first).length, 1);

      const second = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowCatalogPath,
        probes: offlineMachine(),
      });
      const rows = workflowsOf(second);
      assert.equal(rows.length, 1, 'still exactly one row, not one live plus one cached');
      assert.equal(new Set(rows.map((row) => row.id)).size, rows.length);
      assert.equal(rows[0]?.id, workflowsOf(first)[0]?.id, 'the identity is stable across the outage');
      assert.equal(rows[0]?.engineRunning, false);
      assert.equal(rows[0]?.runnable, true);
      assert.match(second.scan.warnings.join(' '), /not answering|scans never start engines/);
    } finally {
      await hub.dispose();
    }
  });

  it('preserves the visible list when a refresh fails outright', async () => {
    // Requirement: connection failures, timeouts and failed refreshes must not
    // replace the catalog with an empty list — and the failure must be reported.
    const endpoint = 'http://127.0.0.1:8188';
    const space = await workspace();
    const catalogPath = join(space.root, 'models.json');
    await writeFile(catalogPath, '{}', 'utf8');
    const workflowCatalogPath = join(space.root, 'state', 'workflow-catalog.json');
    const hub = hubFor(endpoint);
    try {
      const first = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowCatalogPath,
        probes: onlineMachine({ workflows: { 'fox.json': apiWorkflow, 'cat.json': apiWorkflow } }),
      });
      assert.equal(workflowsOf(first).length, 2);
      const before = workflowsOf(first).map((row) => row.id).sort();

      // The engine answers with HTTP 500 rather than refusing: a *failed* refresh
      // is a different failure from an unreachable one, and must behave the same.
      const failing = offlineMachine({
        fetchJson: async (url: string) => {
          throw Object.assign(new Error(`HTTP 500 from ${url}`), { status: 500 });
        },
      });
      const second = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowCatalogPath, probes: failing,
      });
      assert.deepEqual(workflowsOf(second).map((row) => row.id).sort(), before, 'the same two workflows are still listed');
      assert.ok(workflowsOf(second).every((row) => row.freshness === 'cached'));
      assert.ok(second.scan.warnings.length > 0, 'the failure is reported, not swallowed');
      assert.ok(second.scan.sources.some((source) => source.id === 'comfyui' && !source.ok));

      // A timeout behaves the same way — and the engine's own store is gone, so
      // the list can only come from what this scan remembered.
      resetScanCache();
      const third = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowCatalogPath,
        probes: offlineMachine({ fetchJson: async () => new Promise(() => {}) }),
        persistWorkflows: true,
      });
      assert.deepEqual(workflowsOf(third).map((row) => row.id).sort(), before, 'a timeout also keeps the list');
      assert.ok(third.scan.sources.some((source) => source.id === 'remembered'), 'from the remembered catalog');
    } finally {
      await hub.dispose();
    }
  });

  it('shows the valid workflows when one file is malformed and one directory is unreadable', async () => {
    // Requirement: an unreadable directory or malformed file must not prevent the
    // other valid workflows from appearing — and must not be reported as a
    // deletion either.
    const endpoint = 'http://127.0.0.1:59999';
    const space = await workspace();
    const workflowDir = join(space.root, 'workflows');
    await space.write('workflows/good.json', JSON.stringify(apiWorkflow));
    await space.write('workflows/broken.json', '{ this is not json');
    await space.write('workflows/nested/also-good.json', JSON.stringify(apiWorkflow));
    const catalogPath = join(space.root, 'models.json');
    await writeFile(catalogPath, '{}', 'utf8');
    const workflowCatalogPath = join(space.root, 'state', 'workflow-catalog.json');

    const hub = hubFor(endpoint);
    try {
      const report = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowDir, workflowCatalogPath,
        probes: offlineMachine(),
      });
      const names = workflowsOf(report).map((row) => row.name).sort();
      assert.deepEqual(names, ['also-good', 'broken', 'good'], 'every file is accounted for');

      const broken = statusOf(report, 'broken');
      assert.equal(broken.status, 'unreadable', 'a file that never parsed is unreadable, not invalid');
      assert.equal(broken.runnable, false);
      assert.match(broken.detail, /could not be read as JSON/);

      assert.equal(statusOf(report, 'good').status, 'ready');
      assert.equal(statusOf(report, 'also-good').status, 'ready', 'a nested valid file still appears');

      // Now the whole directory becomes unreadable. The previously valid
      // workflows must survive it: an unreadable directory is not a deletion.
      resetScanCache();
      const denied = offlineMachine({
        listDir: async (path: string) => {
          if (path.replace(/\\/g, '/').endsWith('/workflows')) {
            throw Object.assign(new Error('EACCES: permission denied'), { code: 'EACCES' });
          }
          return await DEFAULT_PROBES.listDir(path);
        },
      });
      const second = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowDir, workflowCatalogPath, probes: denied,
      });
      assert.deepEqual(
        workflowsOf(second).map((row) => row.name).sort(),
        ['also-good', 'broken', 'good'],
        'an unreadable directory keeps what was already known',
      );
      assert.ok(second.scan.sources.some((source) => source.id === 'workflow-dir' && !source.ok), 'and reports the denial');
    } finally {
      await hub.dispose();
    }
  });

  it('comes back online without duplicating entries, updating readiness in place', async () => {
    // Requirement: readiness is refreshed when ComfyUI returns, and the list does
    // not grow a second copy of a workflow that was already known. The offline
    // workflow lives in a configured directory; the engine's own store holds a
    // different one, so "coming online" genuinely adds a source.
    const endpoint = 'http://127.0.0.1:8188';
    const space = await workspace();
    const workflowDir = join(space.root, 'workflows');
    await space.write('workflows/local-only.json', JSON.stringify(apiWorkflow));
    const catalogPath = join(space.root, 'models.json');
    await writeFile(catalogPath, '{}', 'utf8');
    const workflowCatalogPath = join(space.root, 'state', 'workflow-catalog.json');
    const hub = hubFor(endpoint);
    try {
      const offline = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowDir, workflowCatalogPath,
        probes: offlineMachine(),
      });
      const local = statusOf(offline, 'local-only');
      assert.equal(local.engineRunning, false);
      assert.equal(local.freshness, 'local', 'found on disk, so not a cached engine answer');
      const offlineCount = workflowsOf(offline).length;

      const online = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowDir, workflowCatalogPath,
        probes: onlineMachine({ workflows: { 'server-only.json': apiWorkflow } }),
      });
      const rows = workflowsOf(online);
      assert.equal(new Set(rows.map((row) => row.id)).size, rows.length, 'no duplicate ids');
      assert.equal(rows.filter((row) => row.name === 'local-only').length, 1, 'the local workflow is one row, not two');
      assert.equal(rows.length, offlineCount + 1, 'coming online adds the workflow only the engine knew about');

      const refreshed = statusOf(online, 'local-only');
      assert.equal(refreshed.engineRunning, true, 'readiness is updated on the row that already existed');
      assert.equal(refreshed.id, local.id, 'under the same identity');
      assert.equal(online.scan.registered > 0, true, 'providers are published after coming online');
    } finally {
      await hub.dispose();
    }
  });

  it('runs a workflow while ComfyUI is stopped by starting the engine once', async () => {
    // Requirement: execution goes through the existing lifecycle manager. The
    // engine is stubbed at the runtime seam, so what is asserted is that the real
    // start path was taken exactly once — not that a process appeared.
    const png = Buffer.from(
      '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c6300010000050001' +
        '0d0a2db40000000049454e44ae426082',
      'hex',
    );
    let systemStats = 0;
    let prompts = 0;
    const server = createServer((req, res) => {
      const url = req.url ?? '/';
      if (url.startsWith('/view')) {
        res.writeHead(200, { 'content-type': 'image/png' });
        res.end(png);
        return;
      }
      if (url === '/prompt') prompts += 1;
      const body =
        url === '/system_stats' ? (systemStats += 1, { system: {}, devices: [] })
          : url === '/object_info' ? objectInfo
            : url === '/prompt' ? { prompt_id: 'p-1' }
              : url.startsWith('/history/')
                ? { 'p-1': { status: { status_str: 'success' }, outputs: { '7': { images: [{ filename: 'out.png', subfolder: '', type: 'output' }] } } } }
                : url.startsWith('/userdata?') ? ['fox.json']
                  : url.includes('/userdata/') ? apiWorkflow
                    : {};
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify(body));
    });
    await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
    const { port } = server.address() as AddressInfo;
    const endpoint = `http://127.0.0.1:${port}`;

    const space = await workspace();
    const catalogPath = join(space.root, 'models.json');
    await writeFile(catalogPath, '{}', 'utf8');
    const hub = hubFor(endpoint);
    const starts: string[] = [];
    const started = hub.runtime.startModel.bind(hub.runtime);
    // The engine is "stopped" until the lifecycle manager is asked to start it.
    hub.runtime.startModel = async (modelId: string) => {
      starts.push(modelId);
      return await started(modelId);
    };

    try {
      const report = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath,
        workflowCatalogPath: join(space.root, 'state', 'workflow-catalog.json'),
        probes: DEFAULT_PROBES,
      });
      const workflow = statusOf(report, 'fox');
      assert.equal(workflow.status, 'ready');

      const result = await hub.invokeModel({
        capability: 'text_to_image', prompt: 'a red fox', modelId: workflow.modelId ?? '', timeoutMs: 15_000,
      });
      assert.equal(result.modelId, workflow.modelId);
      assert.equal(result.outputs.length, 1);
      assert.equal(result.outputs[0]?.type, 'image');
      assert.equal(starts.length, 1, 'the lifecycle manager started the engine exactly once');
      assert.equal(prompts, 1, 'and the graph was queued once, after the engine answered');

      // A second call finds the engine up and must not start it again.
      await hub.invokeModel({ capability: 'text_to_image', prompt: 'again', modelId: workflow.modelId ?? '', timeoutMs: 15_000 });
      assert.equal(starts.length, 1, 'no duplicate startup attempt');
      assert.ok(systemStats > 0, 'readiness was probed rather than assumed');
    } finally {
      await hub.dispose();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  it('keeps a stopped ComfyUI workflow visible when startup is refused', async () => {
    // Requirement: if startup is unavailable, the workflow stays visible and the
    // failure is explained. Nothing silently falls through to another provider.
    const endpoint = 'http://127.0.0.1:59999';
    const space = await workspace();
    const workflowDir = join(space.root, 'workflows');
    await space.write('workflows/fox.json', JSON.stringify(apiWorkflow));
    const catalogPath = join(space.root, 'models.json');
    await writeFile(catalogPath, '{}', 'utf8');
    const hub = hubFor(endpoint);
    hub.runtime.startModel = async () => {
      throw new Error('UNSAFE_OPERATION: process launch is disabled');
    };
    try {
      const report = await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowDir,
        workflowCatalogPath: join(space.root, 'state', 'workflow-catalog.json'),
        probes: offlineMachine(),
      });
      const fox = statusOf(report, 'fox');
      assert.equal(fox.runnable, true, 'the row stays visible and runnable-looking while the engine is off');

      await assert.rejects(
        async () => await hub.invokeModel({ capability: 'text_to_image', prompt: 'x', modelId: fox.modelId ?? '', timeoutMs: 5_000 }),
        (error: Error) => {
          assert.match(String(error), /fox|START_FAILED|UNSAFE_OPERATION|failed/i);
          return true;
        },
      );
      // The row is still there afterwards: a failed start does not hide anything.
      const after = workflowsOf(await inventory(hub, {
        hosts: hostsFor(endpoint), catalogPath, workflowDir,
        workflowCatalogPath: join(space.root, 'state', 'workflow-catalog.json'),
        probes: offlineMachine(),
      }));
      assert.equal(after.length, 1);
    } finally {
      await hub.dispose();
    }
  });
});

describe('workflow catalog document', () => {
  beforeEach(() => {
    resetScanCache();
  });

  it('scopes entries to a host and endpoint, and never crosses endpoints', () => {
    assert.notEqual(workflowScopeKey('comfyui', 'http://a:1'), workflowScopeKey('comfyui', 'http://a:2'));
    assert.notEqual(workflowScopeKey('comfyui', undefined), workflowScopeKey('other', undefined));
    assert.equal(workflowScopeKey('comfyui', 'http://a:1'), workflowScopeKey('comfyui', 'http://a:1'));
  });

  it('records a scan and keeps a scope the scan did not reach', () => {
    const remembered = {
      id: 'wf-a', name: 'a', source: '/one/a.json', status: 'ready' as const, runnable: true,
      detail: 'x', checkedAt: '2020-01-01T00:00:00.000Z', engineRunning: false,
    };
    const previous = {
      version: WORKFLOW_CATALOG_VERSION,
      scopes: [{ hostId: 'comfyui', endpoint: 'http://a:1', updatedAt: '2020-01-01T00:00:00.000Z', workflows: [remembered] }],
    };
    const untouched = recordScan(previous, [], '2021-01-01T00:00:00.000Z');
    assert.equal(untouched.scopes.length, 1, 'a scan that reached nothing keeps everything');

    // A complete scan that no longer sees the workflow is evidence of removal.
    const cleaned = recordScan(previous, [{ hostId: 'comfyui', endpoint: 'http://a:1', authoritative: true, resources: [], descriptors: [] }], '2021-01-01T00:00:00.000Z');
    assert.deepEqual(cleaned.scopes[0]?.workflows, [], 'a complete scan replaces the scope — deletion sticks');

    // An incomplete one is not: an outage must never erase a workflow.
    const duringOutage = recordScan(previous, [{ hostId: 'comfyui', endpoint: 'http://a:1', authoritative: false, resources: [], descriptors: [] }], '2021-01-01T00:00:00.000Z');
    assert.equal(duringOutage.scopes[0]?.workflows.length, 1, 'an incomplete scan keeps what it could not disprove');
    assert.equal(duringOutage.scopes[0]?.updatedAt, '2020-01-01T00:00:00.000Z', 'and does not claim a fresh observation time');

    const other = recordScan(previous, [{ hostId: 'comfyui', endpoint: 'http://a:2', authoritative: true, resources: [], descriptors: [] }], '2021-01-01T00:00:00.000Z');
    assert.equal(other.scopes.length, 2, 'a different endpoint is a different scope');
    assert.equal(other.scopes.find((scope) => scope.endpoint === 'http://a:1')?.workflows.length, 1);
  });

  it('refuses a document it does not understand instead of guessing', async () => {
    const space = await workspace();
    const path = join(space.root, 'workflow-catalog.json');
    await writeFile(path, JSON.stringify({ version: 99, scopes: [{ hostId: 'comfyui', workflows: [] }] }), 'utf8');
    const { lines, logger } = recordingLogger();
    const loaded = await loadWorkflowCatalog(path, logger);
    assert.deepEqual(loaded.document.scopes, [], 'a future version is not half-read');
    assert.ok(lines.some((line) => line.includes('warn')), 'and the reason is logged');

    // A malformed file is the same answer, and a missing file is not an error.
    await writeFile(path, '{ not json', 'utf8');
    const broken = await loadWorkflowCatalog(path, recordingLogger().logger);
    assert.deepEqual(broken.document.scopes, []);
    assert.equal((await loadWorkflowCatalog(join(space.root, 'missing.json'), recordingLogger().logger)).found, false);
  });

  it('round-trips a snapshot, including the descriptor that makes it executable', async () => {
    const space = await workspace();
    const path = join(space.root, 'state', 'workflow-catalog.json');
    const descriptor = { id: 'wf-a', name: 'A', type: 'image_generation' } as never;
    const resource: LocalResource = {
      id: 'wf-a', kind: 'comfyui_workflow', typeLabel: 'ComfyUI Workflow', name: 'A', source: '/one/a.json',
      status: 'ready', detail: 'd', runnable: true, modelId: 'wf-a', engineRunning: true,
      availability: 'unknown', freshness: 'local', checkedAt: '2024-01-01T00:00:00.000Z',
      capabilities: ['text_to_image'], inputs: [], outputs: [],
    };
    const document = recordScan(
      { version: WORKFLOW_CATALOG_VERSION, scopes: [] },
      [{ hostId: 'comfyui', endpoint: 'http://a:1', authoritative: true, resources: [resource], descriptors: [descriptor] }],
      '2024-01-01T00:00:00.000Z',
    );
    const issues = await saveWorkflowCatalog(path, document, recordingLogger().logger);
    assert.deepEqual(issues, []);

    const loaded = await loadWorkflowCatalog(path, recordingLogger().logger);
    const snapshots = rememberedSnapshots(loaded.document);
    assert.equal(snapshots.length, 1);
    assert.equal(snapshots[0]?.hostId, 'comfyui');
    assert.equal(snapshots[0]?.endpoint, 'http://a:1');
    assert.deepEqual(snapshots[0]?.descriptors.map((entry) => entry.id), ['wf-a']);
    const row = snapshots[0]?.resources[0];
    assert.equal(row?.checkedAt, '2024-01-01T00:00:00.000Z', 'the observation time survives');
    assert.equal(row?.freshness, 'cached');
    assert.equal(row?.engineRunning, false);
  });

  it('says "stopped" only when the evidence supports it', async () => {
    // A refused connection is the signature of a process that is not running. A
    // timeout is not, and claiming "stopped" for it would send someone looking for
    // a process that may be running perfectly well.
    const space = await workspace();
    const workflowDir = join(space.root, 'workflows');
    await space.write('workflows/fox.json', JSON.stringify(apiWorkflow));
    const catalogPath = join(space.root, 'models.json');
    await writeFile(catalogPath, '{}', 'utf8');
    const workflowCatalogPath = join(space.root, 'state', 'workflow-catalog.json');

    const refused = hubFor('http://127.0.0.1:8188');
    try {
      const report = await inventory(refused, {
        hosts: hostsFor('http://127.0.0.1:8188'), catalogPath, workflowDir, workflowCatalogPath,
        probes: offlineMachine({
          fetchJson: async (url: string) => { throw new Error(`connect ECONNREFUSED 127.0.0.1:8188 (${url})`); },
        }),
      });
      assert.match(statusOf(report, 'fox').detail, /ComfyUI stopped/);
    } finally {
      await refused.dispose();
    }

    resetScanCache();
    const timedOut = hubFor('http://127.0.0.1:8188');
    try {
      // The scan's own per-effect bound is what produces the timeout here: the
      // injected probe simply never answers, which is exactly what a hung socket
      // looks like from this layer.
      const result = await scanLocalResources(timedOut, {
        hosts: hostsFor('http://127.0.0.1:8188'),
        workflowDir,
        timeoutMs: 100,
        probes: {
          ...DEFAULT_PROBES,
          request: async () => new Promise(() => {}),
          isDirectory: DEFAULT_PROBES.isDirectory,
          listDir: DEFAULT_PROBES.listDir,
          readFile: async (path: string) => await DEFAULT_PROBES.readFile!(path),
        },
      });
      const fox = result.resources.find((resource) => resource.name === 'fox');
      assert.ok(fox, 'the local workflow is still listed');
      assert.match(fox.detail, /ComfyUI unreachable/);
      assert.doesNotMatch(fox.detail, /stopped/, 'a timeout is not evidence that the process stopped');
    } finally {
      await timedOut.dispose();
    }
  });

  it('scans independently of the API, so a stopped engine still yields local rows', async () => {    // The scan half asked directly, with no inventory in front of it.
    const space = await workspace();
    await space.write('workflows/fox.json', JSON.stringify(apiWorkflow));
    const hub = hubFor('http://127.0.0.1:59999');
    try {
      const result = await scanLocalResources(hub, {
        hosts: hostsFor('http://127.0.0.1:59999'),
        workflowDir: join(space.root, 'workflows'),
        probes: {
          ...localFilesystem,
          request: async (url: string) => ({ ok: false as const, kind: 'connection' as const, detail: `${url} refused` }),
        },
      });
      assert.equal(result.resources.filter((row) => row.kind === 'comfyui_workflow').length, 1);
      assert.equal(result.registered, 1, 'the provider is published even with the engine down');
    } finally {
      await hub.dispose();
    }
  });

  it('falls back to remembered workflows when the API source itself throws', async () => {
    // An unexpected exception inside a host pass must not cost the remembered
    // list: the caller still gets a usable snapshot to persist and show.
    const hub = hubFor('http://127.0.0.1:59999');
    const snapshot: ScannedHostSnapshot = {
      hostId: 'comfyui',
      endpoint: 'http://127.0.0.1:59999',
      authoritative: false,
      resources: [{
        id: 'wf-a', kind: 'comfyui_workflow', typeLabel: 'ComfyUI Workflow', name: 'A', source: '/one/a.json',
        status: 'ready', detail: 'd', runnable: false, engineRunning: false, availability: 'unavailable',
        freshness: 'cached', checkedAt: '2024-01-01T00:00:00.000Z',
      }],
      descriptors: [],
    };
    try {
      const result = await scanLocalResources(hub, {
        hosts: hostsFor('http://127.0.0.1:59999'),
        persisted: [snapshot],
        probes: {
          request: async () => {
            throw new Error('injected transport explosion');
          },
          isDirectory: async () => false,
          listDir: async () => undefined,
          readFile: async () => '',
        },
      });
      const rows = result.resources.filter((row) => row.kind === 'comfyui_workflow');
      assert.equal(rows.length, 1, 'the remembered workflow survives a throwing API source');
      assert.equal(rows[0]?.name, 'A');
      assert.equal(rows[0]?.freshness, 'cached');
    } finally {
      await hub.dispose();
    }
  });

  it('does not offer an invented default path outside the harness home', () => {
    // Kept as its own assertion in the document suite too: the location is the one
    // place a third-party plugin may own state.
    assert.match(defaultWorkflowCatalogPath('C:/x/y.json'), /y\.json$/);
  });

  it('resolves a configured workflow directory the way the catalog rule says', () => {
    // A configured directory is resolved against the catalog that configured it,
    // with Windows' `%VAR%`, `~` and both separators understood. Nothing here
    // depends on the host's own platform: the spellings are the point.
    const relative = resolveConfiguredPath('workflows/comfy', 'C:/catalog').replace(/\\/g, '/');
    assert.equal(relative, 'C:/catalog/workflows/comfy');

    const absolute = resolveConfiguredPath('C:/workflows/comfy', 'C:/catalog').replace(/\\/g, '/');
    assert.equal(absolute, 'C:/workflows/comfy');

    const expanded = resolveConfiguredPath('%USERPROFILE%/ComfyUI/workflows', 'C:/catalog').replace(/\\/g, '/');
    assert.doesNotMatch(expanded, /%USERPROFILE%/, 'a variable is expanded, not left literal');
    assert.match(expanded, /\/ComfyUI\/workflows$/);

    const home = resolveConfiguredPath('~/ComfyUI/workflows', 'C:/catalog').replace(/\\/g, '/');
    assert.doesNotMatch(home, /^~\//);
    assert.match(home, /\/ComfyUI\/workflows$/);
  });
});

describe('workflow catalog state on the real machine', () => {
  it('resolves the default location without creating or writing anything', () => {
    // The default path is derived, never hardcoded to a machine: it lives under
    // the harness home, and the redirect a test uses is honoured.
    assert.match(defaultWorkflowCatalogPath('C:/explicit/workflow-catalog.json').replace(/\\/g, '/'), /workflow-catalog\.json$/);
    assert.equal(defaultWorkflowCatalogPath(isolationRoot + '/x.json').replace(/\\/g, '/'), isolationRoot.replace(/\\/g, '/') + '/x.json');
    // With the redirect in force, the derived default is the redirect, not the home.
    assert.equal(defaultWorkflowCatalogPath(), join(isolationRoot, 'suite-catalog.json'));
    // And with it cleared, the harness home is the answer — never a machine path.
    delete process.env['DSH_AI_MODEL_HUB_WORKFLOW_CATALOG'];
    try {
      assert.match(defaultWorkflowCatalogPath().replace(/\\/g, '/'), /\.dsh\/plugins\/dsh-ai-model-hub\/workflow-catalog\.json$/);
    } finally {
      process.env['DSH_AI_MODEL_HUB_WORKFLOW_CATALOG'] = join(isolationRoot, 'suite-catalog.json');
    }
  });
});
