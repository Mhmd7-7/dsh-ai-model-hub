/**
 * Ollama discovery.
 *
 * Ollama is the simplest engine to introspect, and that makes it the right first
 * discoverer: one `GET /api/tags` lists every installed model with its size, and
 * one `POST /api/show` per model adds the template, the parameters, and the
 * `model_info` block a context window can be read out of. Two endpoints are
 * enough to synthesize a complete descriptor.
 *
 * **No model name appears in this file.** The discoverer learns that
 * `some-checkpoint:7b` exists, how big it is, whether it carries a vision
 * projector, and what context window it declares — all from the engine's own
 * answer. Capabilities are decided by *shape* (does `/api/show` report vision
 * information?) rather than by matching a name, because a name list would be
 * wrong the moment someone pulls a checkpoint that did not exist when it was
 * written.
 *
 * The one deliberate narrowing: this reports `text_to_text` and, when vision
 * information is present, `image_understanding`. Ollama's own surface for
 * embeddings, re-ranking, and raw completion is not reachable through this hub's
 * capability vocabulary, so claiming those would be a lie routing could act on.
 *
 * @module dsh-ai-model-hub/discovery/ollama
 */

import type { Capability, ModelType } from '../catalog/capabilities.ts';
import type { ModelDescriptor, ModelHost } from '../catalog/descriptor.ts';
import {
  bytesToGib,
  fetchJson,
  hasKeyMatching,
  isRecordLike,
  readArray,
  readNumber,
  readNumberBySuffix,
  readString,
  slugifyModelId,
  stableDigest,
} from './http.ts';
import { DISCOVERED_PRIORITY, ioForCapabilities } from './types.ts';
import type { HostDiscoverer } from './types.ts';

/** The engine label this discoverer handles, matched against `host.runtime.engine`. */
export const OLLAMA_ENGINE = 'ollama';

/** The endpoint listing every installed model. */
const TAGS_PATH = '/api/tags';

/** The endpoint reporting one model's metadata. */
const SHOW_PATH = '/api/show';

/** Budget for one request, in milliseconds. */
const REQUEST_TIMEOUT_MS = 4_000;

/** How many `/api/show` requests are allowed to be open at once. */
const SHOW_CONCURRENCY = 4;

/** Tags every discovered model carries, so routing can filter on provenance. */
const DISCOVERED_TAGS: readonly string[] = ['local', 'discovered', OLLAMA_ENGINE];

/**
 * One installed model as reported by `GET /api/tags`.
 *
 * This is the *parsed* half of discovery, deliberately separate from the
 * descriptor mapping so it can be tested against a canned payload without a
 * server and without any opinion about capabilities.
 */
export interface OllamaModelSummary {
  /** The name to pass back to the engine, e.g. `llama3.2:latest`. */
  readonly name: string;
  /** Weight size in bytes, when the engine reported one. */
  readonly sizeBytes?: number;
  /** The engine's own family label, when reported. */
  readonly family?: string;
  /** The engine's parameter-count label, kept as text because `8.0B` is not a number. */
  readonly parameterSize?: string;
  /** The engine's quantization label, e.g. `Q4_K_M`. */
  readonly quantizationLevel?: string;
}

/** What `POST /api/show` added about one model. */
export interface OllamaModelDetails {
  /** The model name the details belong to. */
  readonly name: string;
  /** Whether the engine reported vision/projector information. */
  readonly vision: boolean;
  /** The declared context window in tokens, when one could be read. */
  readonly contextTokens?: number;
}

/** Where an {@link OllamaCandidate} came from, for diagnostics. */
export type OllamaDetailSource = 'read' | 'unreadable';

/** One model plus whatever `/api/show` could say about it. */
export interface OllamaCandidate {
  /** What `/api/tags` reported. */
  readonly summary: OllamaModelSummary;
  /** What `/api/show` reported, or `undefined` when it could not be read. */
  readonly details?: OllamaModelDetails;
  /** Whether `/api/show` was reached. Never fail-soft in the other direction. */
  readonly detailSource: OllamaDetailSource;
}

/**
 * Parse the body of `GET /api/tags`.
 *
 * Totally tolerant: entries that are not objects, or that carry no usable name,
 * are skipped rather than throwing. An engine that answers 200 with something
 * unexpected yields fewer models, never a failed pass — the whole point of
 * discovery being fail-soft is that a new or patched engine version degrades
 * instead of taking the catalog down.
 *
 * @param raw - the parsed JSON body.
 * @returns one summary per usable entry.
 */
export function parseOllamaTags(raw: unknown): OllamaModelSummary[] {
  const models: OllamaModelSummary[] = [];
  for (const entry of readArray(raw, 'models')) {
    const name = readDisplayName(entry);
    if (name === undefined) continue;
    const sizeBytes = readNumber(entry, 'size');
    const details = isRecordLike(entry) ? entry['details'] : undefined;
    const family = readString(details, 'family');
    const parameterSize = readString(details, 'parameter_size');
    const quantizationLevel = readString(details, 'quantization_level');
    models.push({
      name,
      ...(sizeBytes === undefined ? {} : { sizeBytes }),
      ...(family === undefined ? {} : { family }),
      ...(parameterSize === undefined ? {} : { parameterSize }),
      ...(quantizationLevel === undefined ? {} : { quantizationLevel }),
    });
  }
  return models;
}

/**
 * The name to address a model by.
 *
 * `name` is what Ollama's own endpoints accept, so it is preferred; `model` is
 * the fallback for the older response shape. Either way the value is passed back
 * verbatim — discovery never rewrites a name, because the name is the engine's
 * primary key.
 *
 * @param entry - one entry from `/api/tags`.
 * @returns the name, or `undefined` when the entry carries none.
 */
function readDisplayName(entry: unknown): string | undefined {
  return readString(entry, 'name') ?? readString(entry, 'model');
}

/**
 * Parse the body of `POST /api/show`.
 *
 * Two signals are read, both generically:
 *
 * - **Vision.** Ollama reports a vision model's projector in a `projector_info`
 *   block (and, on some versions, alongside it in `model_info`). Rather than
 *   hardcode which key, this looks for *any* top-level key or `model_info`
 *   sub-key whose name mentions a projector or clips — a shape test, not a name
 *   test.
 * - **Context window.** Reported inside `model_info` under
 *   `<architecture>.context_length`. The architecture is the server's business,
 *   so the key *suffix* is what is matched.
 *
 * @param raw - the parsed JSON body.
 * @param name - the model the response is about, carried into the result.
 * @returns the details; `contextTokens` and `vision` are absent/false when the
 *   engine reported nothing usable.
 */
export function parseOllamaShow(raw: unknown, name = ''): OllamaModelDetails {
  const modelInfo = isRecordLike(raw) ? raw['model_info'] : undefined;
  // A model with a projector is a model that can look at an image. Both a
  // top-level `projector_info` and a nested `model_info.*projector*` are
  // accepted, because the exact location has moved between Ollama releases and
  // discovery should not have to track that.
  const vision =
    hasKeyMatching(raw, (key) => /projector|vision|clip/i.test(key)) ||
    hasKeyMatching(modelInfo, (key) => /projector|vision|clip/i.test(key));
  const contextTokens = readNumberBySuffix(modelInfo, '.context_length');
  return {
    name,
    vision,
    ...(contextTokens === undefined ? {} : { contextTokens }),
  };
}

/**
 * The capabilities a candidate model can serve.
 *
 * Always `text_to_text`: every model this engine installs is a language model
 * first. `image_understanding` is added only on positive evidence, because a
 * capability claimed without the ability behind it turns into a confusing engine
 * error at invocation time rather than a clean routing rejection.
 *
 * @param candidate - the parsed candidate.
 * @returns the capability list.
 */
export function capabilitiesForOllamaModel(candidate: OllamaCandidate): Capability[] {
  const capabilities: Capability[] = ['text_to_text'];
  if (candidate.details?.vision === true) capabilities.push('image_understanding');
  return capabilities;
}

/**
 * Map one candidate into a descriptor.
 *
 * Every field is either read from the engine or derived by a rule stated in the
 * code — nothing is transcribed from a model list.
 *
 * - **id** — {@link slugifyModelId} over the engine's own name, with the engine
 *   label as a prefix. Deterministic, so the same install maps to the same id on
 *   every run.
 * - **inputTypes / outputTypes** — the union of the declared capabilities'
 *   canonical IO, taken from `CAPABILITY_IO`. Set explicitly rather than left to
 *   the catalog's defaulting, so a chained workflow's compatibility check sees
 *   the same contract a hand-written entry would declare. This matters most for
 *   the vision case, where the model *accepts* an image as well as text.
 * - **resources** — the weight size, rounded up to whole gibibytes, applied to
 *   both RAM and VRAM. Loading the weights is the dominant cost and it is
 *   roughly the file size; the hub's router treats these as a filter, not a
 *   reservation, so a conservative estimate is the right kind of wrong.
 * - **priority** — {@link DISCOVERED_PRIORITY}, below the catalog's own default,
 *   so a hand-written model wins a tie between two otherwise-equal candidates.
 *   This is a tie-break only; "static wins" on an id collision is enforced in
 *   the merge step, not here.
 *
 * @param candidate - the parsed candidate.
 * @param host - the host it belongs to; its `runtime.engine` is the id prefix.
 * @returns the descriptor.
 */
export function mapOllamaModel(candidate: OllamaCandidate, host: ModelHost): ModelDescriptor {
  const capabilities = capabilitiesForOllamaModel(candidate);
  const { inputTypes, outputTypes } = ioForCapabilities(capabilities);
  const id = slugifyModelId(candidate.summary.name, host.runtime.engine || OLLAMA_ENGINE, `model-${stableDigest(candidate.summary.name)}`);
  const sizeBytes = candidate.summary.sizeBytes;
  // The weight size is the one resource fact this engine reports, so it drives
  // both figures. Loading the weights is the dominant cost and is roughly the
  // file size; the router treats declared resources as a filter rather than a
  // reservation, so a conservative estimate is the right kind of wrong.
  const gib = sizeBytes === undefined ? undefined : bytesToGib(sizeBytes);
  const contextTokens = candidate.details?.contextTokens;

  return {
    id,
    name: candidate.summary.name,
    type: modelTypeForCapabilities(capabilities),
    host: host.id,
    capabilities,
    // Explicit, never defaulted: see the method docs above.
    inputTypes,
    outputTypes,
    adapterConfig: { model: candidate.summary.name },
    resources: {
      ...(gib === undefined ? {} : { ramGb: gib, vramGb: gib }),
      requiresGpu: false,
    },
    ...(contextTokens === undefined ? {} : { limits: { contextTokens } }),
    priority: DISCOVERED_PRIORITY,
    tags: [...DISCOVERED_TAGS],
    notes: describeOllamaModel(candidate),
  };
}

/**
 * The `type` a descriptor declares for a capability set.
 *
 * `type` drives operator-facing grouping only — the router filters on
 * `capabilities`, never on `type` — so this is a presentation choice. A model
 * that can read an image *and* write text is `multimodal`; a text-only one is
 * `text_generation`.
 *
 * @param capabilities - the declared capabilities.
 * @returns the model type.
 */
function modelTypeForCapabilities(capabilities: readonly Capability[]): ModelType {
  return capabilities.includes('image_understanding') ? 'multimodal' : 'text_generation';
}

/**
 * A one-line provenance note, assembled from what the engine actually said.
 * @param candidate - the parsed candidate.
 * @returns the note, never empty.
 */
function describeOllamaModel(candidate: OllamaCandidate): string {
  const facts: string[] = ['Discovered from the Ollama API; not listed in models.json.'];
  const { family, parameterSize, quantizationLevel } = candidate.summary;
  if (family !== undefined) facts.push(`family ${family}.`);
  if (parameterSize !== undefined) facts.push(`${parameterSize} parameters.`);
  if (quantizationLevel !== undefined) facts.push(`quantized ${quantizationLevel}.`);
  if (candidate.detailSource === 'unreadable') {
    facts.push('Metadata could not be read, so capabilities were assumed to be text-only.');
  }
  return facts.join(' ');
}

/**
 * Build the Ollama discoverer.
 *
 * @param options - per-request budget override, for tests and for slow machines.
 * @returns the discoverer.
 */
export function createOllamaDiscoverer(options: { readonly requestTimeoutMs?: number } = {}): HostDiscoverer {
  const requestTimeoutMs = Math.max(1, options.requestTimeoutMs ?? REQUEST_TIMEOUT_MS);
  return {
    engine: OLLAMA_ENGINE,
    async discover(host: ModelHost, signal: AbortSignal): Promise<ModelDescriptor[]> {
      const endpoint = host.runtime.endpoint;
      if (endpoint === undefined || endpoint.trim().length === 0) return [];

      const tagsRead = await fetchJson(joinUrl(endpoint, TAGS_PATH), signal, requestTimeoutMs);
      if (!tagsRead.ok) {
        // Unreachable is reported, not swallowed: the registry turns this into a
        // per-host warning and an empty result for this host. An engine that is
        // simply not running must not break startup, which is why the failure
        // stops here instead of propagating to the caller of `generate`.
        throw new Error(`could not list models from ${endpoint}${TAGS_PATH}: ${tagsRead.reason}`);
      }

      const summaries = parseOllamaTags(tagsRead.value);
      const candidates = await mapWithConcurrency(summaries, SHOW_CONCURRENCY, async (summary): Promise<OllamaCandidate> => {
        const details = await readShow(endpoint, summary.name, signal, requestTimeoutMs);
        if (details === undefined) return { summary, detailSource: 'unreadable' };
        return { summary, details, detailSource: 'read' };
      });

      // Two installed models can slugify to the same id (`Foo:1` and `foo-1`).
      // Keep the first and report the rest, so the merge step's id filter has
      // nothing to do and the catalog never logs a duplicate it did not cause.
      const seen = new Set<string>();
      const descriptors: ModelDescriptor[] = [];
      for (const candidate of candidates) {
        const descriptor = mapOllamaModel(candidate, host);
        if (seen.has(descriptor.id)) continue;
        seen.add(descriptor.id);
        descriptors.push(descriptor);
      }
      return descriptors;
    },
  };
}

/**
 * Read `POST /api/show` for one model.
 * @param endpoint - the host's base URL.
 * @param name - the model name to ask about.
 * @param signal - cancellation for the pass.
 * @param timeoutMs - budget for this request.
 * @returns the details, or `undefined` when the endpoint could not be read.
 */
async function readShow(
  endpoint: string,
  name: string,
  signal: AbortSignal,
  timeoutMs: number,
): Promise<OllamaModelDetails | undefined> {
  const read = await fetchJson(joinUrl(endpoint, SHOW_PATH), signal, timeoutMs, {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
  if (!read.ok) return undefined;
  return parseOllamaShow(read.value, name);
}

/**
 * Join a base URL and a path without doubling or dropping the separator.
 * @param endpoint - the base URL, with or without a trailing slash.
 * @param path - the path, with a leading slash.
 * @returns the absolute URL.
 */
export function joinUrl(endpoint: string, path: string): string {
  const base = endpoint.endsWith('/') ? endpoint.slice(0, -1) : endpoint;
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}

/**
 * Map over a list with a bounded number of concurrent workers.
 *
 * A machine with forty installed models should not open forty sockets, and it
 * should not serialize either. Results keep input order, so the emitted
 * descriptor list is deterministic.
 *
 * @param items - the input list.
 * @param limit - maximum concurrent workers.
 * @param worker - the async mapper.
 * @returns results in input order.
 */
export async function mapWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array<R>(items.length);
  let next = 0;
  const workers: Promise<void>[] = [];
  const run = async (): Promise<void> => {
    for (;;) {
      const index = next;
      next += 1;
      if (index >= items.length) return;
      const item = items[index];
      if (item === undefined) continue;
      results[index] = await worker(item, index);
    }
  };
  const width = Math.max(1, Math.min(limit, items.length));
  for (let index = 0; index < width; index += 1) workers.push(run());
  await Promise.all(workers);
  return results;
}
