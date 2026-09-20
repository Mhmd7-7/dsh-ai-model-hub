/**
 * A multi-step, cross-model workflow — Phase 6 in miniature.
 *
 * The point of this example is what it does *not* contain: no engine name, no
 * launch command, no model-specific branching. It states a sequence of
 * capabilities and lets the router satisfy each step. Swap SDXL for ComfyUI, or
 * the mock mesh generator for TripoSR, and this file does not change.
 *
 * Run it with:
 *
 *   node examples/workflow.ts
 *
 * @module dsh-ai-model-hub/examples/workflow
 */

import { mkdtemp, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { ModelHub, loadCatalogConfig } from '../src/index.ts';
import type { Artifact, Capability } from '../src/index.ts';

/** One declared step of the workflow. */
interface WorkflowStep {
  /** A label for progress output. */
  readonly label: string;
  /** The capability this step needs. */
  readonly capability: Capability;
  /** The prompt for this step, given the artifacts produced so far. */
  readonly prompt: string | ((previous: readonly Artifact[]) => string);
  /** Whether this step consumes the previous step's first output. */
  readonly consumesPrevious: boolean;
}

/**
 * The workflow, as data.
 *
 * Declaring it this way — rather than as a hard-coded sequence of calls — is what
 * would let a user save and share a pipeline. The runner below is 30 lines
 * because the hard part (typed artifacts carrying content between models) is
 * already the hub's job.
 */
const CONCEPT_ART_PIPELINE: readonly WorkflowStep[] = [
  {
    label: 'concept image',
    capability: 'text_to_image',
    prompt: 'a low-poly spaceship landing on a desert plateau, isometric',
    consumesPrevious: false,
  },
  {
    label: 'restyled image',
    capability: 'image_to_image',
    prompt: 'shift the palette to teal and amber, keep the composition',
    consumesPrevious: true,
  },
  {
    label: 'game mesh',
    capability: 'image_to_3d',
    prompt: 'produce a low-poly mesh suitable for a real-time game',
    consumesPrevious: true,
  },
  {
    // `consumesPrevious: false` is not an oversight. A text model declares that it
    // accepts `text`, so handing it a `model_3d` artifact is correctly refused:
    // "accepts text but the request supplies a model_3d". Declaring the input kind
    // is what makes that check possible, and the router refuses to guess.
    label: 'asset notes',
    capability: 'text_to_text',
    prompt: 'Summarise this asset pipeline run in one sentence.',
    consumesPrevious: false,
  },
];

/**
 * Run one step, feeding the previous step's output in when it asks for it.
 * @param hub - the hub.
 * @param step - the declared step.
 * @param previous - artifacts produced so far.
 * @returns the artifacts this step produced.
 */
async function runStep(
  hub: ModelHub,
  step: WorkflowStep,
  previous: readonly Artifact[],
): Promise<readonly Artifact[]> {
  const prompt = typeof step.prompt === 'function' ? step.prompt(previous) : step.prompt;
  const inputs =
    step.consumesPrevious && previous[0] !== undefined ? [{ id: previous[0].id }] : undefined;

  // Route first so the choice can be reported before the work happens — the same
  // decision the invocation will use, because it comes from the same code path.
  const decision = await hub.route({
    capability: step.capability,
    prompt,
    ...(inputs === undefined ? {} : { inputs }),
  });

  const result = await hub.invokeModel({
    capability: step.capability,
    prompt,
    ...(inputs === undefined ? {} : { inputs }),
  });

  const size = await Promise.all(
    result.outputs.map(async (artifact) => {
      const { path } = await hub.artifacts.resolvePath(artifact.id);
      return (await stat(path)).size;
    }),
  );

  console.log(`\n[${step.label}]`);
  console.log(`  capability:  ${step.capability}`);
  console.log(`  routed to:   ${decision.modelId}`);
  console.log(`  why:         ${decision.rationale}`);
  if (inputs !== undefined) console.log(`  consumed:    ${inputs[0]?.id}`);
  console.log(`  duration:    ${result.durationMs} ms${result.coldStart ? ' (cold start)' : ''}`);
  for (const [index, artifact] of result.outputs.entries()) {
    console.log(`  produced:    ${artifact.type} ${artifact.id} (${size[index] ?? 0} bytes)`);
  }

  return result.outputs;
}

/**
 * Run the pipeline.
 */
async function main(): Promise<void> {
  const artifactRoot = await mkdtemp(join(tmpdir(), 'dsh-ai-model-hub-workflow-'));
  const loaded = loadCatalogConfig({ configPath: 'config/models.mock.json' });
  const hub = new ModelHub({
    config: loaded.config,
    artifactRoot,
    manageTimers: false,
    log: () => {},
  });

  try {
    console.log(`catalog:  ${loaded.path}`);
    console.log(`workflow: ${CONCEPT_ART_PIPELINE.length} steps`);
    console.log(
      `steps:    ${CONCEPT_ART_PIPELINE.map((step) => step.capability).join(' → ')}`,
    );

    // Fail fast on a capability this deployment cannot serve, rather than
    // discovering it after two expensive steps have already run.
    const unavailable = CONCEPT_ART_PIPELINE.filter(
      (step) => hub.findModelsByCapability(step.capability).length === 0,
    );
    if (unavailable.length > 0) {
      console.error(
        `\nThis deployment cannot run the pipeline. Missing capabilities: ${unavailable
          .map((step) => step.capability)
          .join(', ')}`,
      );
      console.error('Run with config/models.mock.json, which serves all four.');
      process.exitCode = 1;
      return;
    }

    let produced: readonly Artifact[] = [];
    const trace: string[] = [];

    for (const step of CONCEPT_ART_PIPELINE) {
      produced = await runStep(hub, step, produced);
      const first = produced[0];
      if (first === undefined) throw new Error(`step "${step.label}" produced no artifact`);
      trace.push(`${step.capability} → ${first.type} (${first.id})`);
    }

    console.log('\n=== pipeline complete ===');
    for (const line of trace) console.log(`  ${line}`);

    console.log('\n=== final artifact ===');
    const final = produced[0];
    if (final !== undefined) {
      console.log(`  ${final.type} ${final.id}`);
      console.log(`  produced by: ${final.producerModelId ?? 'unknown'}`);
      console.log(`  metadata:    ${JSON.stringify(final.metadata)}`);
    }

    const all = await hub.listArtifacts(20);
    console.log(`\n${all.length} artifact(s) on disk at ${artifactRoot}`);
  } catch (error) {
    console.error(`\nworkflow FAILED: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  } finally {
    await hub.dispose();
    if (process.exitCode === 1) await rm(artifactRoot, { recursive: true, force: true });
  }
}

await main();
