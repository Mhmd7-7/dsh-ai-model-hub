/**
 * The `http_json` adapter: real image generation over a JSON HTTP endpoint.
 *
 * This is the adapter the catalog examples have always referred to —
 * `config/examples/real-models.example.json` points A1111/Forge and ComfyUI at
 * adapter kind `http_json` — but which shipped unimplemented. Until now the only
 * image adapter was the Phase 1 mock, so every `text_to_image` call returned a
 * deterministic placeholder PNG.
 *
 * What it speaks
 * --------------
 * The de-facto Stable Diffusion HTTP contract popularised by AUTOMATIC1111 and
 * implemented by Forge, reForge, and `stable-diffusion.cpp`'s `sd-server`:
 *
 *     POST /sdapi/v1/txt2img   { prompt, steps, width, height, ... }
 *       -> { images: ["<base64 PNG>", ...] }
 *
 * The response parser is deliberately tolerant — it also accepts a single
 * `image` field, an OpenAI-style `data[].b64_json`, and base64 payloads wrapped
 * in a `data:` URL — because these engines disagree in small ways and a strict
 * parser turns a working engine into a mysterious failure.
 *
 * Why image dimensions are read from the bytes
 * --------------------------------------------
 * Engines silently clamp, round, or ignore requested dimensions. Reporting the
 * dimensions we *asked* for would make the artifact metadata a lie, and
 * downstream models (image_to_3d, image_to_image) read those facts instead of
 * opening the file. So the PNG header is parsed and the real size recorded.
 *
 * @module dsh-ai-model-hub/adapters/http-json
 */
import { ModelHubError } from "../errors.js";
/** Capabilities this adapter can serve. Anything else is refused up front. */
const IMAGE_CAPABILITIES = ['text_to_image', 'image_to_image'];
/** Default generation route, matching the A1111/Forge/sd.cpp convention. */
const DEFAULT_TXT2IMG_PATH = '/sdapi/v1/txt2img';
/** Default editing route, used when the model declares `image_to_image`. */
const DEFAULT_IMG2IMG_PATH = '/sdapi/v1/img2img';
/**
 * Default budget for one generation. Real diffusion on a laptop GPU is slow —
 * a 30-step SDXL render at 1024x1024 can take minutes — so the default is
 * generous and per-model override is expected.
 */
const DEFAULT_TIMEOUT_MS = 600_000;
/**
 * Read a finite number from per-call options, then adapter config.
 * @param options - per-call options.
 * @param config - the model's adapter configuration.
 * @param key - the field name.
 * @returns the number, or `undefined` when neither source supplies one.
 */
function numberSetting(options, config, key) {
    const fromOptions = options[key];
    if (typeof fromOptions === 'number' && Number.isFinite(fromOptions))
        return fromOptions;
    const fromConfig = config[key];
    if (typeof fromConfig === 'number' && Number.isFinite(fromConfig))
        return fromConfig;
    return undefined;
}
/**
 * Read a non-empty string from per-call options, then adapter config.
 * @param options - per-call options.
 * @param config - the model's adapter configuration.
 * @param key - the field name.
 * @returns the string, or `undefined` when neither source supplies one.
 */
function stringSetting(options, config, key) {
    const fromOptions = options[key];
    if (typeof fromOptions === 'string' && fromOptions.length > 0)
        return fromOptions;
    const fromConfig = config[key];
    if (typeof fromConfig === 'string' && fromConfig.length > 0)
        return fromConfig;
    return undefined;
}
/**
 * Read a plain-object setting, e.g. `extraBody`.
 * @param config - the model's adapter configuration.
 * @param key - the field name.
 * @returns the record, or `undefined`.
 */
function recordSetting(config, key) {
    const value = config[key];
    if (value === null || typeof value !== 'object' || Array.isArray(value))
        return undefined;
    return value;
}
/**
 * Resolve generation settings for one invocation.
 *
 * Per-call options win over adapter config, which wins over the model's declared
 * limits. That order lets the agent ask for a specific size without an operator
 * losing control of the defaults.
 *
 * @param invocation - the resolved request.
 * @returns the merged settings.
 * @throws ModelHubError with `INVOCATION_FAILED` when no prompt was supplied.
 */
function settingsFor(invocation) {
    const config = invocation.model.adapterConfig;
    const options = invocation.options;
    const prompt = invocation.prompt;
    if (prompt === undefined || prompt.trim().length === 0) {
        throw new ModelHubError('INVOCATION_FAILED', `capability "${invocation.capability}" requires a \`prompt\` and none was supplied`, { modelId: invocation.model.id, capability: invocation.capability });
    }
    const width = numberSetting(options, config, 'width') ?? invocation.model.limits.maxWidth;
    const height = numberSetting(options, config, 'height') ?? invocation.model.limits.maxHeight;
    return {
        prompt,
        ...(stringSetting(options, config, 'negativePrompt') === undefined
            ? {}
            : { negativePrompt: stringSetting(options, config, 'negativePrompt') }),
        ...(numberSetting(options, config, 'steps') === undefined
            ? {}
            : { steps: numberSetting(options, config, 'steps') }),
        ...(numberSetting(options, config, 'cfgScale') === undefined
            ? {}
            : { cfgScale: numberSetting(options, config, 'cfgScale') }),
        ...(stringSetting(options, config, 'sampler') === undefined
            ? {}
            : { sampler: stringSetting(options, config, 'sampler') }),
        ...(numberSetting(options, config, 'seed') === undefined
            ? {}
            : { seed: numberSetting(options, config, 'seed') }),
        ...(width === undefined ? {} : { width: Math.max(1, Math.round(width)) }),
        ...(height === undefined ? {} : { height: Math.max(1, Math.round(height)) }),
        ...(stringSetting(options, config, 'model') === undefined
            ? {}
            : { checkpoint: stringSetting(options, config, 'model') }),
        timeoutMs: Math.max(1, numberSetting(options, config, 'timeoutMs') ?? DEFAULT_TIMEOUT_MS),
        ...(recordSetting(config, 'extraBody') === undefined
            ? {}
            : { extraBody: recordSetting(config, 'extraBody') }),
    };
}
/**
 * Build the absolute request URL for a model and capability.
 *
 * @param model - the resolved model.
 * @param capability - the capability being served.
 * @returns the URL string, or `undefined` when the endpoint is unusable.
 */
function requestUrl(model, capability) {
    const endpoint = model.runtime.endpoint;
    if (endpoint === undefined)
        return undefined;
    const configured = model.runtime.path;
    const fallback = capability === 'image_to_image' ? DEFAULT_IMG2IMG_PATH : DEFAULT_TXT2IMG_PATH;
    // A model may either name the route explicitly, or name only the txt2img route
    // and let the img2img sibling be derived from it — which is what operators
    // copying the A1111 example expect.
    const path = capability === 'image_to_image' && (configured === undefined || configured === DEFAULT_TXT2IMG_PATH)
        ? configured === undefined
            ? fallback
            : DEFAULT_IMG2IMG_PATH
        : configured ?? fallback;
    try {
        const base = endpoint.endsWith('/') ? endpoint : `${endpoint}/`;
        return new URL(path.replace(/^\//, ''), base).toString();
    }
    catch {
        return undefined;
    }
}
/**
 * Read pixel dimensions out of a PNG's IHDR chunk.
 *
 * Returns `undefined` for anything that is not a PNG or is too short to hold a
 * header, so a non-PNG payload degrades to "dimensions unknown" rather than
 * producing bogus metadata.
 *
 * @param bytes - the encoded image.
 * @returns the dimensions, or `undefined`.
 */
export function readPngSize(bytes) {
    const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    if (bytes.length < 24)
        return undefined;
    for (let index = 0; index < PNG_SIGNATURE.length; index += 1) {
        if (bytes[index] !== PNG_SIGNATURE[index])
            return undefined;
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const width = view.getUint32(16, false);
    const height = view.getUint32(20, false);
    if (width === 0 || height === 0)
        return undefined;
    return { width, height };
}
/**
 * Strip a `data:` URL prefix, leaving raw base64.
 * @param value - the possibly-wrapped payload.
 * @returns bare base64.
 */
function bareBase64(value) {
    const comma = value.indexOf(',');
    if (value.startsWith('data:') && comma !== -1)
        return value.slice(comma + 1);
    return value;
}
/**
 * Pull every base64 image out of an engine response.
 *
 * Tolerant on purpose: `{images:[...]}` is the A1111/Forge/sd.cpp shape,
 * `{image:"…"}` appears in older builds, and `{data:[{b64_json}]}` is what an
 * OpenAI-compatible image route returns.
 *
 * @param payload - the parsed JSON response.
 * @returns the base64 payloads, in the order the engine produced them.
 */
export function extractImages(payload) {
    if (payload === null || typeof payload !== 'object')
        return [];
    const record = payload;
    const single = record['image'];
    if (typeof single === 'string' && single.length > 0)
        return [single];
    const list = record['images'];
    if (Array.isArray(list)) {
        return list.filter((entry) => typeof entry === 'string' && entry.length > 0);
    }
    const data = record['data'];
    if (Array.isArray(data)) {
        return data
            .map((entry) => entry !== null && typeof entry === 'object'
            ? entry['b64_json']
            : undefined)
            .filter((entry) => typeof entry === 'string' && entry.length > 0);
    }
    return [];
}
/**
 * A file-name-safe truncation of a prompt.
 * @param seed - the prompt text.
 * @param maxLength - maximum characters to keep.
 * @returns a slug.
 */
function slug(seed, maxLength = 32) {
    const cleaned = seed
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
    return cleaned.length === 0 ? 'untitled' : cleaned.slice(0, maxLength);
}
/**
 * Create the JSON-over-HTTP image adapter.
 * @returns the adapter instance.
 */
export function createHttpJsonAdapter() {
    return {
        kind: 'http_json',
        displayName: 'JSON HTTP image endpoint (A1111, Forge, stable-diffusion.cpp)',
        supports(model) {
            if (model.runtime.endpoint === undefined) {
                return {
                    ok: false,
                    reason: 'the http_json adapter requires `runtime.endpoint` (e.g. http://127.0.0.1:7860)',
                };
            }
            if (requestUrl(model, 'text_to_image') === undefined) {
                return { ok: false, reason: `runtime.endpoint "${model.runtime.endpoint}" is not a usable URL` };
            }
            const supported = model.capabilities.filter((capability) => IMAGE_CAPABILITIES.includes(capability));
            if (supported.length === 0) {
                return {
                    ok: false,
                    reason: `it declares only ${model.capabilities.join(', ') || 'no capabilities'}, and this adapter serves ` +
                        `${IMAGE_CAPABILITIES.join(', ')}. Point it at an engine adapter that speaks that modality.`,
                };
            }
            return { ok: true };
        },
        async health(model, signal) {
            const started = Date.now();
            const endpoint = model.runtime.endpoint;
            if (endpoint === undefined) {
                return { healthy: false, checkedAt: started, detail: 'no runtime.endpoint configured' };
            }
            const url = requestUrl({ ...model, runtime: { ...model.runtime, path: model.health.path ?? '/sdapi/v1/sd-models' } }, 'text_to_image');
            if (url === undefined) {
                return { healthy: false, checkedAt: started, detail: `endpoint "${endpoint}" is not a valid URL` };
            }
            // Never throws: this runs on a timer and inside the cold-start gate, where
            // an exception would be a bug rather than information.
            const controller = new AbortController();
            const onAbort = () => controller.abort();
            if (signal.aborted)
                controller.abort();
            else
                signal.addEventListener('abort', onAbort, { once: true });
            const timer = setTimeout(() => controller.abort(), model.health.timeoutMs ?? 3_000);
            try {
                const response = await fetch(url, { method: 'GET', signal: controller.signal, headers: { accept: '*/*' } });
                // Any answer below 500 means something is listening and speaking HTTP.
                // Engines disagree about whether a model-listing route is 200 or 404, and
                // both answer the question actually being asked.
                const healthy = response.status < 500;
                return {
                    healthy,
                    checkedAt: started,
                    latencyMs: Date.now() - started,
                    detail: `${url} responded ${response.status}`,
                };
            }
            catch (error) {
                return {
                    healthy: false,
                    checkedAt: started,
                    latencyMs: Date.now() - started,
                    detail: `${url} unreachable: ${error instanceof Error ? error.message : String(error)}`,
                };
            }
            finally {
                clearTimeout(timer);
                signal.removeEventListener('abort', onAbort);
            }
        },
        async invoke(invocation) {
            const model = invocation.model;
            const capability = invocation.capability;
            if (!IMAGE_CAPABILITIES.includes(capability)) {
                throw new ModelHubError('UNSUPPORTED_OPERATION', `the http_json adapter does not implement capability "${capability}"`, { modelId: model.id, capability, supported: [...IMAGE_CAPABILITIES] });
            }
            const url = requestUrl(model, capability);
            if (url === undefined) {
                throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" has no usable runtime.endpoint for the http_json adapter`, { modelId: model.id, endpoint: model.runtime.endpoint });
            }
            const settings = settingsFor(invocation);
            const body = { prompt: settings.prompt };
            if (settings.negativePrompt !== undefined)
                body['negative_prompt'] = settings.negativePrompt;
            if (settings.steps !== undefined)
                body['steps'] = settings.steps;
            if (settings.cfgScale !== undefined)
                body['cfg_scale'] = settings.cfgScale;
            if (settings.sampler !== undefined)
                body['sampler_name'] = settings.sampler;
            if (settings.seed !== undefined)
                body['seed'] = settings.seed;
            if (settings.width !== undefined)
                body['width'] = settings.width;
            if (settings.height !== undefined)
                body['height'] = settings.height;
            if (settings.checkpoint !== undefined)
                body['override_settings_restore_afterwards'] = true;
            // An image_to_image request must carry its source image inline as base64,
            // which is how every engine in this family expects it.
            if (capability === 'image_to_image') {
                const source = invocation.inputs.find((artifact) => artifact.type === 'image');
                if (source === undefined) {
                    throw new ModelHubError('INVOCATION_FAILED', 'image_to_image requires an input artifact of type `image`', { modelId: model.id, providedTypes: invocation.inputs.map((artifact) => artifact.type) });
                }
                const read = await invocation.artifacts.read(source.id);
                body['init_images'] = [Buffer.from(read.bytes).toString('base64')];
            }
            // Operator-supplied tuning is merged last so it can override a derived
            // field, but it can never come from the agent's per-call options.
            if (settings.extraBody !== undefined)
                Object.assign(body, settings.extraBody);
            // One controller, two abort sources, so the reason a request ended is never
            // ambiguous to the classification below.
            const controller = new AbortController();
            let timedOut = false;
            const onCallerAbort = () => controller.abort();
            if (invocation.signal.aborted)
                controller.abort();
            else
                invocation.signal.addEventListener('abort', onCallerAbort, { once: true });
            const timer = setTimeout(() => {
                timedOut = true;
                controller.abort();
            }, settings.timeoutMs);
            let response;
            try {
                invocation.log.debug('http_json: posting generation request', {
                    url,
                    capability,
                    width: settings.width,
                    height: settings.height,
                    steps: settings.steps,
                });
                response = await fetch(url, {
                    method: 'POST',
                    signal: controller.signal,
                    headers: { 'content-type': 'application/json', accept: 'application/json' },
                    body: JSON.stringify(body),
                });
            }
            catch (error) {
                if (timedOut) {
                    throw new ModelHubError('INVOCATION_TIMEOUT', `model "${model.id}" did not answer ${url} within ${settings.timeoutMs} ms`, { modelId: model.id, url, timeoutMs: settings.timeoutMs });
                }
                if (invocation.signal.aborted) {
                    throw new ModelHubError('INVOCATION_ABORTED', 'invocation cancelled while awaiting the engine', {
                        modelId: model.id,
                    });
                }
                throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" could not reach ${url}: ${error instanceof Error ? error.message : String(error)}`, { modelId: model.id, url });
            }
            finally {
                clearTimeout(timer);
                invocation.signal.removeEventListener('abort', onCallerAbort);
            }
            if (!response.ok) {
                // Engines put their real complaint in the body; a bare status code sends
                // the operator hunting through logs for something the engine already said.
                let detail = '';
                try {
                    detail = (await response.text()).slice(0, 500);
                }
                catch {
                    detail = '(response body unreadable)';
                }
                throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" answered ${response.status} ${response.statusText} from ${url}: ${detail}`, { modelId: model.id, url, status: response.status });
            }
            let payload;
            try {
                payload = await response.json();
            }
            catch (error) {
                throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" returned a body that is not JSON: ${error instanceof Error ? error.message : String(error)}`, { modelId: model.id, url });
            }
            const images = extractImages(payload);
            if (images.length === 0) {
                // The single most common operator error is pointing this adapter at a
                // chat endpoint, which answers 200 with a perfectly valid JSON body that
                // contains no image at all. Say that, rather than "no images".
                throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" returned no images from ${url}. Expected { images: [...] }; ` +
                    'if this endpoint serves chat completions, it is not an image engine.', { modelId: model.id, url, keys: Object.keys(payload).slice(0, 10) });
            }
            const outputs = [];
            for (const [index, encoded] of images.entries()) {
                const bytes = new Uint8Array(Buffer.from(bareBase64(encoded), 'base64'));
                if (bytes.length === 0) {
                    throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" returned image ${index} as base64 that decoded to zero bytes`, { modelId: model.id, url, index });
                }
                const size = readPngSize(bytes);
                const artifact = await invocation.artifacts.put({
                    type: 'image',
                    bytes,
                    mimeType: 'image/png',
                    label: `${slug(settings.prompt, 40)}${images.length > 1 ? ` #${index + 1}` : ''}`,
                    producerModelId: model.id,
                    extension: '.png',
                    metadata: {
                        prompt: settings.prompt,
                        ...(settings.negativePrompt === undefined ? {} : { negativePrompt: settings.negativePrompt }),
                        ...(size === undefined ? {} : { width: size.width, height: size.height }),
                        ...(settings.width === undefined ? {} : { requestedWidth: settings.width }),
                        ...(settings.height === undefined ? {} : { requestedHeight: settings.height }),
                        ...(settings.steps === undefined ? {} : { steps: settings.steps }),
                        ...(settings.seed === undefined ? {} : { seed: settings.seed }),
                        ...(settings.checkpoint === undefined ? {} : { checkpoint: settings.checkpoint }),
                        format: 'png',
                        source: 'http_json',
                    },
                });
                outputs.push(artifact);
            }
            invocation.log.info('http_json: generated image(s)', {
                url,
                count: outputs.length,
            });
            const first = outputs[0];
            return {
                outputs,
                value: {
                    count: outputs.length,
                    format: 'png',
                    prompt: settings.prompt,
                    ...(first?.metadata['width'] === undefined ? {} : { width: first.metadata['width'] }),
                    ...(first?.metadata['height'] === undefined ? {} : { height: first.metadata['height'] }),
                },
            };
        },
    };
}
