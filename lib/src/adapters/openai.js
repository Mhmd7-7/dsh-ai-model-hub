/**
 * The OpenAI-compatible chat adapter.
 *
 * This one adapter reaches every engine that speaks the `/v1/chat/completions`
 * shape — Ollama, llama.cpp's `llama-server`, vLLM, LM Studio, KoboldCpp, and
 * anything else that copied the API. That is why it is the highest-value adapter
 * in the set: one file makes a whole family of local engines usable, and no
 * router, catalog, plugin, or DSH code changes to add another one.
 *
 * Three rules it obeys, all of which are contract requirements rather than
 * preferences:
 *
 * - **Never throw for a negative health finding.** An unreachable engine is a
 *   *report*; the periodic prober and the cold-start gate both depend on that.
 * - **Always forward cancellation.** A cancelled invocation must settle promptly
 *   and report `INVOCATION_ABORTED`, never linger holding a socket.
 * - **Persist every output through the artifact store.** Text the agent cannot
 *   reference by id is text it cannot chain into the next capability call.
 *
 * The `model` field sent to the server comes from `adapterConfig.model` (or
 * `runtime.modelPath`), never from the agent: the caller states a *capability*
 * and the catalog states which checkpoint serves it. `options` from the agent
 * may set a small documented set of generation scalars; arbitrary keys are
 * ignored rather than merged into the request, and `extraBody` — the escape
 * hatch for engine-specific tuning such as `num_ctx` or `repeat_penalty` — is
 * read from `adapterConfig` only, so it stays an operator decision.
 *
 * @module dsh-ai-model-hub/adapters/openai
 */
import { ModelHubError } from "../errors.js";
import { isRecord } from "../util/validate.js";
/** The request path used when a descriptor declares none. */
const DEFAULT_REQUEST_PATH = '/v1/chat/completions';
/** Default per-invocation deadline, in milliseconds. */
const DEFAULT_TIMEOUT_MS = 120_000;
/** Longest input artifact text inlined into a request, in characters. */
const MAX_INLINE_TEXT_CHARS = 100_000;
/** Largest input image inlined as a data URL, in bytes. */
const MAX_INLINE_IMAGE_BYTES = 16 * 1024 * 1024;
/** Longest engine response body quoted in a failure message, in characters. */
const MAX_ERROR_BODY_CHARS = 600;
/**
 * The capabilities this adapter serves.
 *
 * Both produce text; `image_understanding` is the same request with image parts
 * added to the user message. Everything else — image, audio, video, 3D
 * generation — belongs to an adapter that speaks that engine's own API, and is
 * refused here with a clear reason rather than a confusing engine error.
 */
const CHAT_CAPABILITIES = ['text_to_text', 'image_understanding'];
/**
 * Read a finite number, or `undefined` when the value is not one.
 * @param value - the candidate.
 * @returns the number, or `undefined`.
 */
function finiteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}
/**
 * Read a non-empty string from a settings object.
 * @param source - the settings object.
 * @param key - the field name.
 * @returns the string, or `undefined`.
 */
function readText(source, key) {
    const value = source[key];
    return typeof value === 'string' && value.trim().length > 0 ? value : undefined;
}
/**
 * Read a non-empty string from per-call options, then adapter config.
 *
 * Options win because the agent asked for this specific call; adapter config is
 * the deployment's default. This precedence is the same one the mock adapter
 * uses, so "options override config" holds across the whole adapter set.
 *
 * @param invocation - the resolved request.
 * @param key - the field name.
 * @returns the string, or `undefined` when neither source supplies one.
 */
function textSetting(invocation, key) {
    return readText(invocation.options, key) ?? readText(invocation.model.adapterConfig, key);
}
/**
 * Read a finite number from per-call options, then adapter config.
 * @param invocation - the resolved request.
 * @param key - the field name.
 * @returns the number, or `undefined` when neither source supplies one.
 */
function numberSetting(invocation, key) {
    return finiteNumber(invocation.options[key]) ?? finiteNumber(invocation.model.adapterConfig[key]);
}
/**
 * Resolve the adapter's settings for one invocation.
 * @param invocation - the resolved request.
 * @returns the merged settings.
 */
function settingsFor(invocation) {
    // The checkpoint name is engine configuration, not agent input: it may come
    // from adapterConfig or from runtime.modelPath, never from `options`.
    const model = readText(invocation.model.adapterConfig, 'model') ?? invocation.model.runtime.modelPath;
    const extraBody = invocation.model.adapterConfig['extraBody'];
    return {
        model,
        temperature: numberSetting(invocation, 'temperature'),
        maxTokens: numberSetting(invocation, 'maxTokens'),
        system: textSetting(invocation, 'system'),
        seed: numberSetting(invocation, 'seed'),
        timeoutMs: Math.max(1, numberSetting(invocation, 'timeoutMs') ?? DEFAULT_TIMEOUT_MS),
        extraBody: isRecord(extraBody) ? extraBody : undefined,
    };
}
/**
 * Compose the absolute request URL for a model.
 *
 * `runtime.path` is resolved against the endpoint's origin, which is the shape
 * every engine in this family documents (`/v1/chat/completions`,
 * `/api/chat`, `/v1/completions`).
 *
 * @param model - the resolved model.
 * @returns the URL, or `undefined` when the endpoint is absent or unusable.
 */
function requestUrl(model) {
    const endpoint = model.runtime.endpoint;
    if (endpoint === undefined)
        return undefined;
    const path = model.runtime.path ?? DEFAULT_REQUEST_PATH;
    try {
        return new URL(path, endpoint.endsWith('/') ? endpoint : `${endpoint}/`).toString();
    }
    catch {
        return undefined;
    }
}
/**
 * Base64-encode bytes for an inline `data:` URL.
 * @param bytes - the raw content.
 * @returns the base64 payload.
 */
function toBase64(bytes) {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
}
/**
 * Build the chat messages for a request.
 *
 * Input artifacts become conversation content in two forms: text artifacts are
 * inlined with a labelled separator so the model can tell the instruction from
 * the material, and images become `image_url` data URLs, which is how every
 * OpenAI-compatible vision endpoint expects them.
 *
 * @param invocation - the resolved request.
 * @param settings - resolved generation settings.
 * @returns the messages, in order.
 * @throws ModelHubError with `INVOCATION_FAILED` when the request carries nothing to act on.
 */
async function buildMessages(invocation, settings) {
    const messages = [];
    if (settings.system !== undefined)
        messages.push({ role: 'system', content: settings.system });
    const textSections = [];
    const images = [];
    for (const artifact of invocation.inputs) {
        if (artifact.type === 'text' || artifact.type === 'json') {
            const { bytes } = await invocation.artifacts.read(artifact.id);
            const decoded = new TextDecoder().decode(bytes);
            const body = decoded.length > MAX_INLINE_TEXT_CHARS
                ? `${decoded.slice(0, MAX_INLINE_TEXT_CHARS)}\n… [truncated: ${decoded.length - MAX_INLINE_TEXT_CHARS} more characters]`
                : decoded;
            textSections.push(`--- input artifact ${artifact.id} ---\n${body}`);
            continue;
        }
        if (artifact.type === 'image') {
            if (invocation.capability !== 'image_understanding') {
                throw new ModelHubError('UNSUPPORTED_OPERATION', `the openai_compatible adapter cannot feed an image to capability "${invocation.capability}"`, { modelId: invocation.model.id, capability: invocation.capability, artifactId: artifact.id });
            }
            const { bytes } = await invocation.artifacts.read(artifact.id);
            if (bytes.byteLength > MAX_INLINE_IMAGE_BYTES) {
                throw new ModelHubError('ARTIFACT_ERROR', `input image "${artifact.id}" is ${bytes.byteLength} bytes, above the ${MAX_INLINE_IMAGE_BYTES}-byte inline limit`, { artifactId: artifact.id, byteLength: bytes.byteLength });
            }
            const mime = artifact.mimeType ?? 'image/png';
            images.push({ type: 'image_url', image_url: { url: `data:${mime};base64,${toBase64(bytes)}` } });
            continue;
        }
        throw new ModelHubError('UNSUPPORTED_OPERATION', `the openai_compatible adapter cannot consume a "${artifact.type}" input`, { modelId: invocation.model.id, artifactId: artifact.id, artifactType: artifact.type });
    }
    const prompt = invocation.prompt?.trim() ?? '';
    const instruction = prompt.length > 0 ? prompt : invocation.capability === 'image_understanding' ? 'Describe this image.' : '';
    const preamble = textSections.length > 0 ? `${textSections.join('\n\n')}\n\n` : '';
    const text = `${preamble}${instruction}`;
    if (text.trim().length === 0) {
        throw new ModelHubError('INVOCATION_FAILED', `capability "${invocation.capability}" requires a \`prompt\` or a text input artifact and neither was supplied`, { modelId: invocation.model.id, capability: invocation.capability });
    }
    // A plain string is used whenever there is no image: some servers reject the
    // content-array form even though the specification allows it.
    if (images.length === 0)
        messages.push({ role: 'user', content: text });
    else
        messages.push({ role: 'user', content: [{ type: 'text', text }, ...images] });
    return messages;
}
/**
 * Extract the assistant text from a chat-completions response.
 *
 * Three shapes are accepted because local servers differ: `message.content` as a
 * string (the specification), `message.content` as an array of text parts (some
 * gateways), and the legacy completions `choice.text`.
 *
 * @param payload - the parsed response body.
 * @returns the text plus the reported finish reason, or `undefined` when the
 *   body carries no usable content.
 */
function extractText(payload) {
    if (!isRecord(payload))
        return undefined;
    const choices = payload['choices'];
    if (!Array.isArray(choices) || choices.length === 0)
        return undefined;
    const first = choices[0];
    if (!isRecord(first))
        return undefined;
    const finishReason = typeof first['finish_reason'] === 'string' ? first['finish_reason'] : undefined;
    const legacy = first['text'];
    if (typeof legacy === 'string')
        return { text: legacy, finishReason };
    const message = first['message'];
    if (!isRecord(message))
        return undefined;
    const content = message['content'];
    if (typeof content === 'string')
        return { text: content, finishReason };
    if (Array.isArray(content)) {
        const joined = content
            .map((part) => (isRecord(part) && typeof part['text'] === 'string' ? part['text'] : ''))
            .join('');
        if (joined.length > 0)
            return { text: joined, finishReason };
    }
    return undefined;
}
/**
 * Read a numeric usage field, or `undefined` when it is absent or not a number.
 * @param usage - the response's `usage` object.
 * @param key - the field name.
 * @returns the count, or `undefined`.
 */
function usageNumber(usage, key) {
    if (!isRecord(usage))
        return undefined;
    return finiteNumber(usage[key]);
}
/**
 * Turn a filename-safe fragment out of arbitrary text.
 * @param value - the source text.
 * @param maxLength - maximum length of the result.
 * @returns a lowercase `[a-z0-9-]` fragment.
 */
function slugify(value, maxLength = 32) {
    const cleaned = value
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
    return cleaned.length === 0 ? 'reply' : cleaned.slice(0, maxLength);
}
/**
 * Classify a transport failure into the right stable code.
 *
 * The distinction matters to the agent: a timeout is worth retrying with a
 * larger budget, an unreachable endpoint means the engine is not running, and a
 * caller cancellation must never be retried on another model.
 *
 * @param error - the caught value.
 * @param invocation - the request that failed.
 * @param url - the URL that was attempted.
 * @param options - which abort fired first.
 * @returns the error to throw.
 */
function transportError(error, invocation, url, options) {
    const modelId = invocation.model.id;
    if (options.callerAborted) {
        return new ModelHubError('INVOCATION_ABORTED', `invocation of "${modelId}" was cancelled`, { modelId, endpoint: url });
    }
    if (options.timedOut) {
        return new ModelHubError('INVOCATION_TIMEOUT', `"${modelId}" exceeded its ${options.timeoutMs} ms budget at ${url}`, { modelId, endpoint: url, timeoutMs: options.timeoutMs });
    }
    const detail = error instanceof Error ? error.message : String(error);
    return new ModelHubError('INVOCATION_FAILED', `could not reach ${url}: ${detail}`, {
        modelId,
        endpoint: url,
        reason: 'transport',
    });
}
/**
 * Create the OpenAI-compatible adapter.
 *
 * @returns the adapter instance.
 */
export function createOpenAiCompatibleAdapter() {
    return {
        kind: 'openai_compatible',
        displayName: 'OpenAI-compatible chat endpoint (Ollama, llama.cpp, vLLM, LM Studio, …)',
        supports(model) {
            if (model.runtime.endpoint === undefined) {
                return {
                    ok: false,
                    reason: 'the openai_compatible adapter requires `runtime.endpoint` (e.g. http://127.0.0.1:11434)',
                };
            }
            if (requestUrl(model) === undefined) {
                return { ok: false, reason: `runtime.endpoint "${model.runtime.endpoint}" is not a usable URL` };
            }
            const unsupported = model.capabilities.filter((capability) => !CHAT_CAPABILITIES.includes(capability));
            if (model.capabilities.length > 0 && unsupported.length === model.capabilities.length) {
                return {
                    ok: false,
                    reason: `it declares only ${unsupported.join(', ')}, and this adapter serves ${CHAT_CAPABILITIES.join(', ')}. ` +
                        'Point it at an engine adapter that speaks that modality.',
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
            let url;
            try {
                url = new URL(model.health.path ?? '/', endpoint.endsWith('/') ? endpoint : `${endpoint}/`).toString();
            }
            catch {
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
            const timer = setTimeout(() => controller.abort(), model.health.timeoutMs ?? 2_000);
            try {
                const response = await fetch(url, { method: 'GET', signal: controller.signal, headers: { accept: '*/*' } });
                // Any answer below 500 means something is listening and speaking HTTP.
                // Engines disagree about whether `/` is 200, 404, or 405, and all three
                // answer the question actually being asked.
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
            if (!CHAT_CAPABILITIES.includes(invocation.capability)) {
                throw new ModelHubError('UNSUPPORTED_OPERATION', `the openai_compatible adapter does not implement capability "${invocation.capability}"`, { modelId: model.id, capability: invocation.capability, supported: [...CHAT_CAPABILITIES] });
            }
            const url = requestUrl(model);
            if (url === undefined) {
                throw new ModelHubError('INVOCATION_FAILED', `model "${model.id}" has no usable runtime.endpoint for the openai_compatible adapter`, { modelId: model.id, endpoint: model.runtime.endpoint });
            }
            const settings = settingsFor(invocation);
            const messages = await buildMessages(invocation, settings);
            const body = { messages, stream: false };
            if (settings.model !== undefined)
                body['model'] = settings.model;
            if (settings.temperature !== undefined)
                body['temperature'] = settings.temperature;
            if (settings.maxTokens !== undefined)
                body['max_tokens'] = settings.maxTokens;
            if (settings.seed !== undefined)
                body['seed'] = settings.seed;
            // Operator-supplied tuning is merged last so it can override a derived
            // field, but it can never come from the agent's per-call options.
            if (settings.extraBody !== undefined)
                Object.assign(body, settings.extraBody);
            // One controller, two abort sources, so the reason a request ended is
            // never ambiguous to the classification below.
            const controller = new AbortController();
            let timedOut = false;
            const callerAborted = () => invocation.signal.aborted;
            const onCallerAbort = () => controller.abort();
            if (invocation.signal.aborted)
                controller.abort();
            else
                invocation.signal.addEventListener('abort', onCallerAbort, { once: true });
            const timer = setTimeout(() => {
                timedOut = true;
                controller.abort();
            }, settings.timeoutMs);
            const started = Date.now();
            try {
                let response;
                let raw;
                try {
                    response = await fetch(url, {
                        method: 'POST',
                        signal: controller.signal,
                        headers: { 'content-type': 'application/json', accept: 'application/json' },
                        body: JSON.stringify(body),
                    });
                    raw = await response.text();
                }
                catch (error) {
                    throw transportError(error, invocation, url, {
                        callerAborted: callerAborted(),
                        timedOut,
                        timeoutMs: settings.timeoutMs,
                    });
                }
                if (!response.ok) {
                    throw new ModelHubError('INVOCATION_FAILED', `${url} returned HTTP ${response.status}: ${raw.trim().slice(0, MAX_ERROR_BODY_CHARS) || '(empty body)'}`, {
                        modelId: model.id,
                        endpoint: url,
                        status: response.status,
                        reason: 'http-status',
                    });
                }
                let payload;
                try {
                    payload = JSON.parse(raw);
                }
                catch (error) {
                    throw new ModelHubError('INVOCATION_FAILED', `${url} returned a body that is not JSON: ${error instanceof Error ? error.message : String(error)}`, { modelId: model.id, endpoint: url, reason: 'malformed-body' });
                }
                const extracted = extractText(payload);
                if (extracted === undefined) {
                    throw new ModelHubError('INVOCATION_FAILED', `${url} returned no message content: ${raw.trim().slice(0, MAX_ERROR_BODY_CHARS) || '(empty body)'}`, { modelId: model.id, endpoint: url, reason: 'empty-completion' });
                }
                const promptTokens = usageNumber(isRecord(payload) ? payload['usage'] : undefined, 'prompt_tokens');
                const completionTokens = usageNumber(isRecord(payload) ? payload['usage'] : undefined, 'completion_tokens');
                const servedModel = isRecord(payload) && typeof payload['model'] === 'string' ? payload['model'] : settings.model;
                const durationMs = Date.now() - started;
                const artifact = await invocation.artifacts.put({
                    type: 'text',
                    text: extracted.text,
                    mimeType: 'text/plain',
                    label: `text from ${model.id} for "${slugify(invocation.prompt ?? '', 24)}"`,
                    producerModelId: model.id,
                    extension: '.txt',
                    metadata: {
                        capability: invocation.capability,
                        model: servedModel ?? model.id,
                        prompt: invocation.prompt ?? '',
                        finishReason: extracted.finishReason ?? 'unknown',
                        durationMs,
                        ...(promptTokens === undefined ? {} : { promptTokens }),
                        ...(completionTokens === undefined ? {} : { completionTokens }),
                        ...(invocation.inputs.length === 0
                            ? {}
                            : { sourceArtifactIds: invocation.inputs.map((artifact) => artifact.id).join(',') }),
                    },
                });
                invocation.log.info('openai-compatible reply', {
                    modelId: model.id,
                    endpoint: url,
                    characters: extracted.text.length,
                    durationMs,
                });
                const value = {
                    text: extracted.text,
                    characters: extracted.text.length,
                    durationMs,
                    finishReason: extracted.finishReason ?? 'unknown',
                };
                if (servedModel !== undefined)
                    value['model'] = servedModel;
                if (promptTokens !== undefined)
                    value['promptTokens'] = promptTokens;
                if (completionTokens !== undefined)
                    value['completionTokens'] = completionTokens;
                return { outputs: [artifact], value };
            }
            finally {
                clearTimeout(timer);
                invocation.signal.removeEventListener('abort', onCallerAbort);
            }
        },
    };
}
