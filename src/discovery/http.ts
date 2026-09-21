/**
 * Shared plumbing for discovery.
 *
 * Two jobs, both of which every discoverer needs and neither of which belongs to
 * any one engine:
 *
 * 1. **Talking to an engine without trusting it.** The hub's model-facing
 *    boundary is deliberately total: a discovery pass must never throw because a
 *    server is down, slow, or answering with HTML instead of JSON. The readers
 *    here return `{ ok: false, reason }` where the codebase would otherwise
 *    need a `try`/`catch` ladder.
 * 2. **Deriving ids and reading JSON defensively.** Ids have to be stable across
 *    runs (the same installed model must always map to the same id) and must
 *    satisfy the catalog's id grammar, which is a real constraint rather than a
 *    convention: `MODEL_ID_PATTERN` is validated, published in the JSON Schema,
 *    and reaches artifact ids and tool output.
 *
 * Nothing here knows an engine, a model, or a capability.
 *
 * @module dsh-ai-model-hub/discovery/http
 */

import { isValidModelId } from '../catalog/descriptor.ts';
import { isRecord } from '../util/validate.ts';

/** Longest engine response body retained for a diagnostic message, in characters. */
const MAX_REASON_BODY_CHARS = 400;

/** The suffix appended to an engine label to make a valid id prefix. */
const ID_PREFIX_SEPARATOR = '-';

/**
 * The outcome of reading a URL as JSON.
 *
 * A discriminated result rather than an exception, because "Ollama is not
 * running" is an ordinary discovery outcome that produces a warning and an empty
 * result — see the fail-soft contract on `HostDiscoverer`.
 */
export type JsonReadResult =
  | { readonly ok: true; readonly value: unknown }
  | { readonly ok: false; readonly reason: string };

/**
 * `GET` a URL and parse the body as JSON, containing every failure mode.
 *
 * Both the outer `signal` (the discovery pass's own cancellation) and an
 * internal timeout are honoured; whichever fires first wins. A non-2xx response
 * is a failure rather than an attempt to parse an error page as JSON, and the
 * body is quoted (bounded) in the reason so a misconfigured endpoint produces a
 * diagnosable message instead of "unexpected token <".
 *
 * @param url - the absolute URL to read.
 * @param signal - cancellation for the pass.
 * @param timeoutMs - budget for this one request.
 * @param init - optional request overrides, e.g. a `POST` with a JSON body.
 * @returns the parsed value, or a reason it could not be read.
 */
export async function fetchJson(
  url: string,
  signal: AbortSignal,
  timeoutMs: number,
  init: { readonly method?: string; readonly body?: string } = {},
): Promise<JsonReadResult> {
  const controller = new AbortController();
  const onOuterAbort = (): void => controller.abort();
  if (signal.aborted) controller.abort();
  else signal.addEventListener('abort', onOuterAbort, { once: true });
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const headers: Record<string, string> = { accept: 'application/json' };
    if (init.body !== undefined) headers['content-type'] = 'application/json';
    const response = await fetch(url, {
      method: init.method ?? 'GET',
      signal: controller.signal,
      headers,
      ...(init.body === undefined ? {} : { body: init.body }),
    });
    if (!response.ok) {
      // A non-2xx is a failure, not an attempt to parse an error page as JSON.
      // The body is quoted — bounded, because it is an engine's, not ours — so a
      // misconfigured endpoint produces a diagnosable message rather than
      // "unexpected token <".
      const body = (await response.text().catch(() => '')).trim();
      const quoted =
        body.length === 0
          ? ''
          : `: ${body.length <= MAX_REASON_BODY_CHARS ? body : `${body.slice(0, MAX_REASON_BODY_CHARS)}…`}`;
      return { ok: false, reason: `HTTP ${response.status} from ${url}${quoted}` };
    }
    return { ok: true, value: await response.json() };
  } catch (error) {
    // A controller abort is either the pass deadline or the caller cancelling;
    // both are described the same way, because the caller already knows which.
    if (controller.signal.aborted) return { ok: false, reason: `request to ${url} was aborted or timed out` };
    return { ok: false, reason: `could not read ${url}: ${error instanceof Error ? error.message : String(error)}` };
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', onOuterAbort);
  }
}

/**
 * Whether a value can be indexed by string keys.
 *
 * Re-exported from the catalog's validator rather than reimplemented: a second
 * implementation of "is this a JSON object" is exactly the kind of duplicate
 * that drifts.
 *
 * @param value - candidate value.
 * @returns true when the value is a plain object.
 */
export const isRecordLike = isRecord;

/**
 * Read an array-valued property, or an empty array.
 * @param value - the container.
 * @param key - the property name.
 * @returns the array, or `[]` when absent or not an array.
 */
export function readArray(value: unknown, key: string): readonly unknown[] {
  if (!isRecord(value)) return [];
  const found = value[key];
  return Array.isArray(found) ? found : [];
}

/**
 * Read a non-empty string property.
 * @param value - the container.
 * @param key - the property name.
 * @returns the string, or `undefined`.
 */
export function readString(value: unknown, key: string): string | undefined {
  if (!isRecord(value)) return undefined;
  const found = value[key];
  return typeof found === 'string' && found.trim().length > 0 ? found : undefined;
}

/**
 * Read a finite number property.
 * @param value - the container.
 * @param key - the property name.
 * @returns the number, or `undefined`.
 */
export function readNumber(value: unknown, key: string): number | undefined {
  if (!isRecord(value)) return undefined;
  const found = value[key];
  return typeof found === 'number' && Number.isFinite(found) ? found : undefined;
}

/**
 * Find the first key ending in a suffix and read its numeric value.
 *
 * Engine introspection payloads are keyed by a family the server chooses and the
 * hub cannot enumerate ahead of time — Ollama reports a context window as
 * `<family>.context_length`, where `<family>` is whatever the model's
 * architecture is called. Searching for the *suffix* is what keeps this free of
 * a hardcoded family list.
 *
 * @param value - the record to search.
 * @param suffix - the key suffix, e.g. `.context_length`.
 * @returns the first matching finite number, or `undefined`.
 */
export function readNumberBySuffix(value: unknown, suffix: string): number | undefined {
  if (!isRecord(value)) return undefined;
  for (const [key, found] of Object.entries(value)) {
    if (!key.endsWith(suffix)) continue;
    if (typeof found === 'number' && Number.isFinite(found)) return found;
  }
  return undefined;
}

/**
 * Whether any key in a record matches a predicate.
 *
 * Used to detect "this response contains a sub-object whose name mentions X"
 * without needing to know what X is called on any particular engine.
 *
 * @param value - the record to search.
 * @param predicate - tests each key.
 * @returns true when at least one key matches.
 */
export function hasKeyMatching(value: unknown, predicate: (key: string) => boolean): boolean {
  if (!isRecord(value)) return false;
  return Object.keys(value).some((key) => predicate(key));
}

/**
 * Turn an arbitrary engine-reported name into a valid, stable model id.
 *
 * The catalog validates ids against `MODEL_ID_PATTERN` —
 * `^[a-z0-9][a-z0-9._-]*$` — so this is a correctness requirement, not tidiness.
 * Every character that is not in that set becomes a separator, runs of
 * separators collapse, and leading/trailing separators are trimmed. Resolution
 * tags such as Ollama's `:latest` and version dots survive as separators, so
 * `Llama-3.2:3B-Instruct` and `llama 3.2 3b instruct` both land on the same
 * stable id. The mapping is deterministic, which is the property that matters:
 * the same installed model must resolve to the same id on every run, or routing
 * decisions stop being reproducible.
 *
 * @param name - the engine's own name for the model.
 * @param prefix - an engine label prepended to keep ids unique across engines.
 * @param fallback - used when `name` contains nothing usable; defaults to a
 *   stable digest of the original text so two unusable names cannot collide.
 * @returns a valid model id.
 */
export function slugifyModelId(name: string, prefix: string, fallback = 'model'): string {
  const normalize = (value: string): string =>
    value
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, ID_PREFIX_SEPARATOR)
      .replace(/^[._-]+/, '')
      .replace(/[._-]+$/, '')
      .replace(/-{2,}/g, ID_PREFIX_SEPARATOR);

  const prefixSlug = normalize(prefix) || fallback;
  const nameSlug = normalize(name);
  if (nameSlug.length === 0) {
    // Nothing survived — a name written entirely in a non-Latin script, say.
    // A digest keeps the id stable and unique instead of collapsing every such
    // model onto one id.
    return ensureValid(`${prefixSlug}${ID_PREFIX_SEPARATOR}${fallback}${ID_PREFIX_SEPARATOR}${stableDigest(name)}`);
  }
  // `ollama:ollama-3` should not become `ollama-ollama-3`.
  if (nameSlug === prefixSlug || nameSlug.startsWith(`${prefixSlug}${ID_PREFIX_SEPARATOR}`)) {
    return ensureValid(nameSlug);
  }
  return ensureValid(`${prefixSlug}${ID_PREFIX_SEPARATOR}${nameSlug}`);
}

/**
 * Guarantee an id satisfies the catalog's grammar.
 *
 * `slugifyModelId` should already have done so; this is the belt to that
 * braces, because an invalid id is rejected by `parseModelCatalogConfig` and
 * would take the whole catalog down rather than one model.
 *
 * @param candidate - the id to check.
 * @returns a valid id, derived from the candidate when necessary.
 */
function ensureValid(candidate: string): string {
  if (isValidModelId(candidate)) return candidate;
  const repaired = candidate.replace(/[^a-z0-9._-]+/g, ID_PREFIX_SEPARATOR).replace(/^[^a-z0-9]+/, '');
  if (isValidModelId(repaired)) return repaired;
  return `m${ID_PREFIX_SEPARATOR}${stableDigest(candidate)}`;
}

/**
 * A short, stable, hex digest of a string.
 *
 * Deliberately a plain FNV-1a rather than a cryptographic hash: the only
 * requirement is that it is deterministic and cheap, and `node:crypto` in this
 * path would buy nothing.
 *
 * @param value - the text to digest.
 * @returns eight lowercase hex characters.
 */
export function stableDigest(value: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

/**
 * Bytes → gibibytes, rounded up.
 *
 * Rounding up is deliberate and is the whole heuristic: declaring a little more
 * than the weights strictly occupy is conservative for a routing filter, where
 * under-declaring means a model gets chosen on a machine that then fails to load
 * it. Ingestion, KV cache, and framework overhead are all real and none of them
 * are knowable from a file size, so this is an estimate and is documented as one
 * everywhere it is produced.
 *
 * @param bytes - the size reported by the engine.
 * @returns whole gibibytes, at least 1 when anything at all was reported.
 */
export function bytesToGib(bytes: number): number {
  const gib = bytes / 1024 ** 3;
  return Math.max(1, Math.ceil(gib));
}
