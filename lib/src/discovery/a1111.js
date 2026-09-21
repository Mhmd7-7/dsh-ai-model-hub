/**
 * A1111 / Forge discovery.
 *
 * The automatic1111 WebUI (and its Forge fork) exposes a small, stable REST
 * surface that answers everything discovery needs:
 *
 * | Endpoint                  | What it answers                                        |
 * | ------------------------- | ------------------------------------------------------ |
 * | `GET /sdapi/v1/sd-models` | every checkpoint on disk, with its title and hash       |
 * | `GET /sdapi/v1/samplers`  | the sampler names this build actually accepts           |
 * | `GET /sdapi/v1/options`   | which checkpoint is currently loaded                    |
 *
 * The last one matters more than it looks. Only the loaded checkpoint serves
 * without a model-switch cost — a switch can take tens of seconds and may swap
 * tens of gigabytes — so a model that is already loaded is the one routing
 * should prefer. That preference is expressed with the ordinary `priority` field
 * rather than with any new mechanism: the loaded checkpoint gets
 * {@link LOADED_PRIORITY}, below the un-loaded default, and the catalog's
 * existing ordering does the rest.
 *
 * **The engine is never asked to switch checkpoints by this module.** Discovery
 * reports what exists; which checkpoint a request runs against is the adapter's
 * business, driven by the descriptor it was handed.
 *
 * Like every other discoverer here, no checkpoint name appears in this source.
 *
 * @module dsh-ai-model-hub/discovery/a1111
 */
import { fetchJson, isRecordLike, readString, slugifyModelId, stableDigest } from "./http.js";
import { DISCOVERED_PRIORITY, ioForCapabilities } from "./types.js";
/** Engine labels this discoverer answers to. */
export const A1111_ENGINES = ['a1111', 'automatic1111', 'forge', 'sd_webui', 'stable_diffusion_webui'];
/** The primary engine label, used for the descriptor's id prefix and tags. */
export const A1111_ENGINE = 'a1111';
/** The endpoint listing installed checkpoints. */
const MODELS_PATH = '/sdapi/v1/sd-models';
/** The endpoint listing valid sampler names. */
const SAMPLERS_PATH = '/sdapi/v1/samplers';
/** The endpoint reporting the currently loaded checkpoint. */
const OPTIONS_PATH = '/sdapi/v1/options';
/** Budget for one request, in milliseconds. */
const REQUEST_TIMEOUT_MS = 4_000;
/** Tags every discovered model carries. */
const DISCOVERED_TAGS = ['local', 'discovered', A1111_ENGINE];
/**
 * Priority for the checkpoint the engine reports as currently loaded.
 *
 * Below {@link DISCOVERED_PRIORITY}, so between two otherwise-equal discovered
 * checkpoints the loaded one wins. Both are still far above every static model's
 * default, so this is a tie-break among discovered candidates and never
 * outranks an operator's own entry.
 */
export const LOADED_PRIORITY = 200;
/**
 * Best-effort VRAM estimates, in gibibytes, selected by filename pattern.
 *
 * **These are estimates and are treated as such.** The WebUI's checkpoint
 * listing reports a filename and a hash but no file size and no architecture, so
 * there is nothing exact to read. Filename keywords such as `xl` correlate well
 * with "bigger than a 1.5-era model" and cost nothing, but they are a heuristic
 * on a string — deliberately not a table of known checkpoints, which would be
 * both hardcoded and immediately stale. The router uses declared resources as a
 * filter rather than a reservation, so a rough figure is the right kind of
 * answer; an operator who knows better edits the static catalog.
 */
const VRAM_ESTIMATE_PATTERNS = [
    { pattern: /(^|[^a-z])(xl|sdxl|1024)([^a-z]|$)/i, vramGb: 6 },
    { pattern: /(^|[^a-z])(2k|2048|flux|hunyuan)([^a-z]|$)/i, vramGb: 12 },
    { pattern: /(^|[^a-z])(sd15|sd1[._-]?5|v1[._-]5|512)([^a-z]|$)/i, vramGb: 2 },
];
/** The VRAM estimate used when no pattern matches. */
const DEFAULT_VRAM_GB = 4;
/**
 * Parse the body of `GET /sdapi/v1/sd-models`.
 *
 * The body *is* the array, so there is no wrapper key to read. Anything that is
 * not an array — an HTML error page, an object from a mock server — yields no
 * models rather than throwing.
 *
 * @param raw - the parsed JSON body.
 * @returns one summary per usable entry.
 */
export function parseA1111Models(raw) {
    if (!Array.isArray(raw))
        return [];
    const models = [];
    for (const entry of raw)
        addSummary(models, entry);
    return models;
}
/**
 * Append one parsed summary when the entry carries a usable title.
 * @param models - the accumulator.
 * @param entry - one raw array element.
 */
function addSummary(models, entry) {
    const title = readString(entry, 'title') ?? readString(entry, 'model_name');
    if (title === undefined)
        return;
    const modelName = readString(entry, 'model_name');
    const hash = readString(entry, 'hash');
    const filename = readString(entry, 'filename');
    models.push({
        title,
        ...(modelName === undefined ? {} : { modelName }),
        ...(hash === undefined ? {} : { hash }),
        ...(filename === undefined ? {} : { filename }),
    });
}
/**
 * Parse the body of `GET /sdapi/v1/samplers`.
 * @param raw - the parsed JSON body.
 * @returns the sampler names, in the order the engine listed them.
 */
export function parseA1111Samplers(raw) {
    if (!Array.isArray(raw))
        return [];
    const samplers = [];
    for (const entry of raw)
        addSampler(samplers, entry);
    return samplers;
}
/**
 * Append one sampler name when the entry carries one.
 * @param samplers - the accumulator.
 * @param entry - one raw array element.
 */
function addSampler(samplers, entry) {
    const name = readString(entry, 'name') ?? readString(entry, 'aliases');
    if (name !== undefined && !samplers.includes(name))
        samplers.push(name);
}
/**
 * Parse the body of `GET /sdapi/v1/options` for the loaded checkpoint.
 *
 * The WebUI reports it under `sd_model_checkpoint`, whose value is the
 * checkpoint's title (occasionally decorated with a hash suffix). Only the
 * presence and identity of a loaded model is used; nothing else in the options
 * document is read, because none of it is discovery's business.
 *
 * @param raw - the parsed JSON body.
 * @returns the loaded checkpoint's name, or `undefined`.
 */
export function parseA1111Options(raw) {
    if (!isRecordLike(raw))
        return undefined;
    const loaded = readString(raw, 'sd_model_checkpoint');
    return loaded;
}
/**
 * The capabilities every discovered A1111 checkpoint gets.
 *
 * `text_to_image` and `image_to_image` are both served by the same endpoint
 * family (`/sdapi/v1/txt2img` and `/sdapi/v1/img2img`) for every checkpoint the
 * WebUI can load — a property of the engine, not of any checkpoint. Audio,
 * video, and 3D are not reachable through this API at all, so claiming them
 * would route work into a guaranteed failure.
 */
export const A1111_CAPABILITIES = ['text_to_image', 'image_to_image'];
/**
 * Estimate a checkpoint's VRAM need from its filename.
 *
 * A heuristic on a string, documented as such — see
 * {@link VRAM_ESTIMATE_PATTERNS}. It exists so a small card has a chance of
 * routing around a large checkpoint; it is not a substitute for an operator's
 * own entry in the static catalog, which always wins on an id collision.
 *
 * @param names - the filename, title, and model name, any of which may be absent.
 * @returns the estimate in gibibytes.
 */
export function estimateA1111Vram(names) {
    const haystack = names.filter((name) => name !== undefined).join(' ');
    if (haystack.trim().length === 0)
        return DEFAULT_VRAM_GB;
    for (const rule of VRAM_ESTIMATE_PATTERNS) {
        if (rule.pattern.test(haystack))
            return rule.vramGb;
    }
    return DEFAULT_VRAM_GB;
}
/**
 * Map one installed checkpoint into a descriptor.
 *
 * @param summary - the parsed checkpoint.
 * @param host - the host it belongs to.
 * @param facts - engine-wide facts read once per host.
 * @returns the descriptor.
 */
export function mapA1111Model(summary, host, facts) {
    const capabilities = A1111_CAPABILITIES;
    const { inputTypes, outputTypes } = ioForCapabilities(capabilities);
    const id = slugifyModelId(summary.title, host.runtime.engine || A1111_ENGINE, `checkpoint-${stableDigest(summary.title)}`);
    const vramGb = estimateA1111Vram([summary.filename, summary.title, summary.modelName]);
    const loaded = facts.loadedCheckpoint !== undefined && isLoadedCheckpoint(summary, facts.loadedCheckpoint);
    return {
        id,
        name: summary.title,
        type: modelTypeFor(),
        host: host.id,
        capabilities,
        // Explicit, from CAPABILITY_IO: an image-to-image model accepts an `image`
        // as well as a `text`, which is what lets a generated image chain into it.
        inputTypes,
        outputTypes,
        adapterConfig: {
            model: summary.title,
            ...(summary.hash === undefined ? {} : { checkpointHash: summary.hash }),
            discovery: {
                ...(facts.samplers.length === 0 ? {} : { availableSamplers: [...facts.samplers] }),
                loaded,
                ...(summary.filename === undefined ? {} : { filename: summary.filename }),
            },
        },
        resources: { vramGb, ramGb: vramGb, requiresGpu: true },
        priority: loaded ? LOADED_PRIORITY : DISCOVERED_PRIORITY,
        tags: [...DISCOVERED_TAGS, ...(loaded ? ['loaded'] : [])],
        notes: describeA1111Model(summary, facts, loaded, vramGb),
    };
}
/**
 * Whether a parsed checkpoint is the one the engine reports as loaded.
 *
 * The engine's own option value and the listing's title usually match exactly,
 * but the WebUI sometimes decorates the option with a shortened hash, so a
 * prefix match in either direction is accepted. Filenames are compared too,
 * because some builds report the path there instead of the title.
 *
 * @param summary - the parsed checkpoint.
 * @param loaded - what the engine reported.
 * @returns true when this is the loaded checkpoint.
 */
function isLoadedCheckpoint(summary, loaded) {
    const candidates = [summary.title, summary.modelName, summary.filename].filter((value) => value !== undefined);
    return candidates.some((candidate) => candidate === loaded || candidate.startsWith(loaded) || loaded.startsWith(candidate));
}
/**
 * The `type` a discovered A1111 checkpoint declares.
 *
 * Both capabilities are image editing in the sense the vocabulary means — a
 * checkpoint can generate from text and can transform an image — so
 * `image_editing` describes it more precisely than `image_generation`. Routing
 * does not read `type`; this is operator-facing grouping.
 *
 * @returns the model type.
 */
function modelTypeFor() {
    return 'image_editing';
}
/**
 * A provenance note assembled from what the engine reported.
 * @param summary - the parsed checkpoint.
 * @param facts - engine-wide facts.
 * @param loaded - whether this is the loaded checkpoint.
 * @param vramGb - the estimate that was applied.
 * @returns the note.
 */
function describeA1111Model(summary, facts, loaded, vramGb) {
    const factsOut = ['Discovered from the A1111/Forge API; not listed in models.json.'];
    if (loaded)
        factsOut.push('Currently loaded, so it serves without a checkpoint switch.');
    if (summary.filename !== undefined)
        factsOut.push(`File: ${summary.filename}.`);
    if (facts.samplers.length > 0)
        factsOut.push(`${facts.samplers.length} sampler(s) available.`);
    factsOut.push(`VRAM estimate ${vramGb} GiB, inferred from the filename; treat as approximate.`);
    return factsOut.join(' ');
}
/**
 * Build the A1111/Forge discoverer.
 *
 * @param options - per-request budget override, for tests and slow machines.
 * @returns the discoverer.
 */
export function createA1111Discoverer(options = {}) {
    const requestTimeoutMs = Math.max(1, options.requestTimeoutMs ?? REQUEST_TIMEOUT_MS);
    return {
        engine: A1111_ENGINE,
        aliases: A1111_ENGINES.filter((label) => label !== A1111_ENGINE),
        async discover(host, signal) {
            const endpoint = host.runtime.endpoint;
            if (endpoint === undefined || endpoint.trim().length === 0)
                return [];
            const modelsRead = await fetchJson(join(endpoint, MODELS_PATH), signal, requestTimeoutMs);
            if (!modelsRead.ok) {
                // Reported, not swallowed: the registry turns this into a per-host
                // warning and an empty result, so an engine that is not running is a
                // note in the log rather than a failed startup.
                throw new Error(`could not list checkpoints from ${endpoint}${MODELS_PATH}: ${modelsRead.reason}`);
            }
            const summaries = parseA1111Models(modelsRead.value);
            // The engine-wide facts are best-effort: a build that does not expose
            // `/samplers` should still yield models, just with no sampler list.
            const [samplersRead, optionsRead] = await Promise.all([
                fetchJson(join(endpoint, SAMPLERS_PATH), signal, requestTimeoutMs),
                fetchJson(join(endpoint, OPTIONS_PATH), signal, requestTimeoutMs),
            ]);
            const facts = {
                samplers: samplersRead.ok ? parseA1111Samplers(samplersRead.value) : [],
                ...(optionsRead.ok ? optionalLoaded(parseA1111Options(optionsRead.value)) : {}),
            };
            const seen = new Set();
            const descriptors = [];
            for (const summary of summaries) {
                const descriptor = mapA1111Model(summary, host, facts);
                if (seen.has(descriptor.id))
                    continue;
                seen.add(descriptor.id);
                descriptors.push(descriptor);
            }
            return descriptors;
        },
    };
}
/**
 * Wrap an optional loaded-checkpoint name for conditional spreading.
 * @param loaded - the name, or `undefined`.
 * @returns an object with the key, or an empty object.
 */
function optionalLoaded(loaded) {
    return loaded === undefined ? {} : { loadedCheckpoint: loaded };
}
/**
 * Join a base URL and a path without doubling or dropping the separator.
 * @param endpoint - the base URL.
 * @param path - the path.
 * @returns the absolute URL.
 */
function join(endpoint, path) {
    const base = endpoint.endsWith('/') ? endpoint.slice(0, -1) : endpoint;
    return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}
