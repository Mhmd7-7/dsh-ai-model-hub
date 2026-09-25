/**
 * The first vertical slice, end to end.
 *
 * This is the proof that the architecture works against real engines:
 *
 *   request → catalog → capability lookup → router → runtime gate
 *          → adapter → artifact → response
 *
 * Nothing here is a test double: it loads a catalog from disk and drives the
 * *same* code paths the DSH plugin uses — `ModelHub`, not a stand-in — so if this
 * runs, the plugin's wiring is exercised. Which capabilities it can demonstrate
 * is whatever the catalog declares; the shipped `config/models.json` serves
 * `text_to_image` through ComfyUI, so that is the step it insists on. Run it with:
 *
 *   node examples/vertical-slice.ts [path/to/catalog.json]
 *
 * The catalog path defaults to `config/models.json`. Any engine the catalog
 * names must already be running — this script never launches one — and the
 * failure it prints when one is not reachable is meant to be read, not worked
 * around.
 *
 * @module dsh-ai-model-hub/examples/vertical-slice
 */

import { mkdtemp, rm, readFile, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ModelHub, loadCatalogConfig } from '../src/index.ts';

/**
 * Format a byte count for the console.
 * @param bytes - the count.
 * @returns a compact string.
 */
function humanBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KiB', 'MiB', 'GiB'];
  let value = bytes / 1024;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(1)} ${units[index] ?? 'KiB'}`;
}

/**
 * Run the slice.
 */
async function main(): Promise<void> {
  const artifactRoot = await mkdtemp(join(tmpdir(), 'dsh-ai-model-hub-demo-'));
  const checkpoint: string[] = [];

  try {
    // ── 1. Configuration is data ────────────────────────────────────────────
    const catalogPath = process.argv[2] ?? 'config/models.json';
    const loaded = loadCatalogConfig({ configPath: catalogPath });
    console.log(`catalog: ${loaded.path}`);

    const hub = new ModelHub({
      config: loaded.config,
      artifactRoot,
      // No background timers: this is a one-shot script.
      manageTimers: false,
      // No machine probe either. Routing is real either way, but this demo is
      // meant to be readable and repeatable: a live VRAM measurement would make
      // the transcript differ between two runs on the same machine, and the
      // numbers it prints are the declared requirements rather than the free
      // memory. `npm run demo:3d` shows the probed figures.
      probeResources: false,
      // A real engine's rejection can carry an entire response body. The hub keeps
      // all of it in the error; a terminal wants the first line of it.
      log: (message) => {
        const oneLine = message.replace(/\s+/g, ' ');
        console.log(`  [hub] ${oneLine.length > 200 ? `${oneLine.slice(0, 200)}…` : oneLine}`);
      },
    });

    try {
      // ── 2. What can this machine do? ──────────────────────────────────────
      console.log('\n=== catalog: capability discovery ===');
      const capabilities = hub.listCapabilities();
      for (const capability of capabilities) {
        console.log(
          `  ${capability.capability.padEnd(20)} ${capability.inputTypes.join('/')} → ${capability.outputTypes.join('/')}  via ${capability.modelIds.join(', ')}`,
        );
      }
      const unserved = hub.listUnservedCapabilities();
      if (unserved.length > 0) {
        console.log(`  not available here: ${unserved.map((entry) => entry.capability).join(', ')}`);
      }
      checkpoint.push(`discovered ${capabilities.length} capabilities`);

      // ── 3. The user's request: "create a futuristic city image" ───────────
      const prompt = 'a futuristic city at dusk, neon reflections on wet streets';
      console.log(`\n=== request: text_to_image ===\n  prompt: "${prompt}"`);

      // Refusing clearly beats failing halfway: an image engine has to be in the
      // catalog and running before any of the steps below mean anything.
      if (hub.findModelsByCapability('text_to_image').length === 0) {
        console.error(
          `\nNo model in ${loaded.path} serves text_to_image.\n` +
            'Add an image engine to that catalog — copy-ready ComfyUI and A1111 entries\n' +
            'are in config/examples/real-models.example.json, and docs/adding-a-model.md\n' +
            'walks through the edit. No code changes are involved.',
        );
        process.exitCode = 1;
        return;
      }

      // ── 4. The router decides, and explains itself ────────────────────────
      const decision = await hub.route({ capability: 'text_to_image', prompt });
      console.log(`\n=== routing ===\n  chose ${decision.modelId}`);
      console.log(`  rationale: ${decision.rationale}`);
      for (const candidate of decision.candidates) {
        console.log(`    [${candidate.eligible ? 'eligible' : 'rejected'}] ${candidate.modelId}: ${candidate.reason}`);
      }
      checkpoint.push(`routed to ${decision.modelId}`);

      // ── 5. Invoke. The caller never names a model. ────────────────────────
      const outcome = await hub.tryInvokeModel({ capability: 'text_to_image', prompt });
      if (!outcome.ok) {
        // The first line of the hub's own message, not the whole chain: the full
        // detail is a diagnostic, and a wall of JSON helps nobody at a terminal.
        const reason = outcome.error.message.split('\n')[0] ?? outcome.error.message;
        console.error(
          `\n${decision.modelId} could not serve text_to_image: ${outcome.error.code}\n  ${reason}\n\n` +
            'The endpoint answered, so the engine is up — check its own console for what it\n' +
            'rejected. A ComfyUI graph, for instance, must name checkpoints, text encoders and\n' +
            'VAEs that exist in that installation; see the workflowPath entry in the catalog.',
        );
        process.exitCode = 1;
        return;
      }
      const result = outcome.result;
      console.log(
        `\n=== invocation ===\n  ${result.capability} served by ${result.modelId} in ${result.durationMs} ms${result.coldStart ? ' (cold start)' : ''}`,
      );
      checkpoint.push(`invoked ${result.modelId}`);

      // ── 6. Inspect the artifact for real ──────────────────────────────────
      console.log('\n=== artifact ===');
      const image = result.outputs[0];
      if (image === undefined) throw new Error(`${result.modelId} reported success but produced no output`);

      const { path } = await hub.artifacts.resolvePath(image.id);
      const bytes = await readFile(path);
      const info = await stat(path);
      const isPng = bytes[0] === 0x89 && bytes.subarray(1, 4).toString('ascii') === 'PNG';
      console.log(`  id:        ${image.id}`);
      console.log(`  type:      ${image.type}`);
      console.log(`  uri:       ${image.uri}`);
      console.log(`  path:      ${path}`);
      console.log(`  mime:      ${image.mimeType}`);
      console.log(`  size:      ${humanBytes(info.size)}`);
      console.log(`  metadata:  ${JSON.stringify(image.metadata)}`);
      console.log(`  valid PNG: ${isPng ? 'yes' : 'NO'}`);
      if (!isPng) throw new Error('the produced artifact is not a valid PNG');
      checkpoint.push(`artifact ${image.id} is a valid PNG`);

      // ── 7. Graceful failure: a capability nobody serves ───────────────────
      // Chaining image → 3D is the natural next step, and it is exactly what a
      // catalog without a 3D engine cannot do. Asking anyway is the point: the
      // refusal names the gap instead of producing something invented.
      console.log('\n=== graceful failure: capability nobody serves ===');
      const failure = await hub.tryInvokeModel({
        capability: 'image_to_3d',
        inputs: [{ id: image.id, type: 'image' }],
        prompt: 'a low-poly spaceship',
      });
      if (failure.ok) {
        console.log(`  unexpected success: ${failure.result.modelId} produced ${failure.result.outputs[0]?.id}`);
      } else {
        console.log(`  refused with ${failure.error.code}`);
        console.log(`  ${failure.error.message.split('\n').slice(0, 3).join('\n  ')}`);
        checkpoint.push(`refused image_to_3d with ${failure.error.code}`);
      }

      // ── 8. Artifact listing, as the agent would see it ────────────────────
      console.log('\n=== artifacts produced in this run ===');
      for (const artifact of await hub.listArtifacts(10)) {
        console.log(`  ${artifact.type.padEnd(9)} ${artifact.id}`);
      }

      console.log(`\n=== vertical slice complete ===`);
      for (const step of checkpoint) console.log(`  ✓ ${step}`);
      console.log(`\nartifacts on disk at: ${artifactRoot}`);
    } finally {
      await hub.dispose();
    }
  } catch (error) {
    console.error(`\nvertical slice FAILED: ${error instanceof Error ? error.stack ?? error.message : String(error)}`);
    process.exitCode = 1;
  } finally {
    // Keep the artifacts for a human to look at, but only when the run worked.
    if (process.exitCode === 1) await rm(artifactRoot, { recursive: true, force: true });
  }
}

await main();
