/**
 * The first vertical slice, end to end.
 *
 * This is the proof that the architecture works before any real engine exists:
 *
 *   request → catalog → capability lookup → router → runtime gate
 *          → mock image model → artifact → response
 *
 * It uses the *same* code paths the DSH plugin uses — `ModelHub`, not a test
 * double — so if this runs, the plugin's wiring is exercised. Run it with:
 *
 *   node examples/vertical-slice.ts
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
    const loaded = loadCatalogConfig({ configPath: 'config/models.mock.json' });
    console.log(`catalog: ${loaded.path}`);

    const hub = new ModelHub({
      config: loaded.config,
      artifactRoot,
      // No background timers: this is a one-shot script.
      manageTimers: false,
      log: (message) => console.log(`  [hub] ${message}`),
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

      // ── 4. The router decides, and explains itself ────────────────────────
      const decision = await hub.route({ capability: 'text_to_image', prompt });
      console.log(`\n=== routing ===\n  chose ${decision.modelId}`);
      console.log(`  rationale: ${decision.rationale}`);
      for (const candidate of decision.candidates) {
        console.log(`    [${candidate.eligible ? 'eligible' : 'rejected'}] ${candidate.modelId}: ${candidate.reason}`);
      }
      checkpoint.push(`routed to ${decision.modelId}`);

      // ── 5. Invoke. The caller never names a model. ────────────────────────
      const result = await hub.invokeModel({ capability: 'text_to_image', prompt });
      console.log(
        `\n=== invocation ===\n  ${result.capability} served by ${result.modelId} in ${result.durationMs} ms${result.coldStart ? ' (cold start)' : ''}`,
      );
      checkpoint.push(`invoked ${result.modelId}`);

      // ── 6. Inspect the artifact for real ──────────────────────────────────
      console.log('\n=== artifact ===');
      const image = result.outputs[0];
      if (image === undefined) throw new Error('the mock image model produced no output');

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

      // ── 7. Chained workflow: the image feeds a 3D model ───────────────────
      console.log('\n=== chained workflow: image → 3D ===');
      console.log('  passing the produced image id straight into image_to_3d');
      const mesh = await hub.invokeModel({
        capability: 'image_to_3d',
        inputs: [{ id: image.id, type: 'image' }],
        prompt: 'a low-poly spaceship',
      });
      const meshArtifact = mesh.outputs[0];
      if (meshArtifact === undefined) throw new Error('the mock 3D model produced no output');
      const meshPath = (await hub.artifacts.resolvePath(meshArtifact.id)).path;
      const meshText = await readFile(meshPath, 'utf8');
      console.log(`  ${mesh.capability} served by ${mesh.modelId}`);
      console.log(`  artifact:  ${meshArtifact.id} (${meshArtifact.mimeType})`);
      console.log(`  valid STL: ${meshText.startsWith('solid ') && meshText.trimEnd().startsWith('endsolid', meshText.lastIndexOf('endsolid')) ? 'yes' : 'NO'}`);
      console.log(`  vertices:  ${String(meshArtifact.metadata['vertexCount'])}`);
      checkpoint.push(`chained image → 3D producing ${meshArtifact.id}`);

      // ── 8. Graceful failure: an impossible request ────────────────────────
      console.log('\n=== graceful failure: capability nobody serves ===');
      const failure = await hub.tryInvokeModel({ capability: 'video_generation', prompt: 'a flying whale' });
      if (failure.ok) {
        console.log('  unexpected success');
      } else {
        console.log(`  refused with ${failure.error.code}`);
        console.log(`  ${failure.error.message.split('\n').slice(0, 3).join('\n  ')}`);
        checkpoint.push(`refused video_generation with ${failure.error.code}`);
      }

      // ── 9. Artifact listing, as the agent would see it ────────────────────
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
