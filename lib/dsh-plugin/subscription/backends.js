/**
 * Subscription backends as AI Hub capability options.
 *
 * A subscription integration is a *backend*, not an engine: it runs on someone
 * else's machine, it has no process here, and the credential, session, and
 * transport belong to the plugin that owns them. So the hub is handed an option
 * — facts, a readiness probe, and a function that does the work — and it
 * supplies everything else: the operator's order, the enable/disable decision,
 * the compatibility filter, the fallback chain, and the artifact contract.
 *
 * Three rules shape this module, and each of them is a requirement rather than a
 * preference:
 *
 * - **The exact selected backend runs, and nothing else.** A capability option's
 *   id *is* its backend (`codex_image`, `grok_video`). The option never asks for
 *   "an image provider"; it asks the subscription tool for that one provider by
 *   name, and refuses when the tool cannot promise it.
 * - **Policy is enforced where the work starts, not where it is planned.** The
 *   hub re-checks permission immediately before dispatch, and this module checks
 *   again before it calls the tool, because the tool is the last boundary the hub
 *   does not own.
 * - **Outputs enter the artifact system.** A path under the harness home is not
 *   something a later workflow step can chain from, so every produced file is
 *   read and re-stored as an artifact the hub hands back like any other.
 *
 * What this module deliberately does *not* do: read a token, open a login flow,
 * call a provider endpoint, or duplicate the subscription plugin's account
 * scheduling. It calls the plugin's own registered tool through the tool
 * registry, which is the only supported way to reach a subscription from another
 * plugin and the only one that keeps the operator's session restrictions, tool
 * policies, and account rotation in force.
 *
 * @module dsh-ai-model-hub/dsh-plugin/subscription/backends
 */
import { readFile } from 'node:fs/promises';
import { CAPABILITY_IO } from "../../src/index.js";
import { ModelHubError, toHubError } from "../../src/index.js";
/**
 * The image and video backends this integration covers.
 *
 * Each entry names its tool by the *canonical* name the subscription plugin
 * registers. That plugin falls back to a plugin-scoped alias
 * (`dsh_subscriptions_image_generate`) when another plugin already owns the
 * plain name, so the resolution below tries both and reports which one it found
 * — a deployment where a different image tool won the name is a real
 * configuration, and silently calling the wrong tool would be worse than saying
 * the integration is not wired.
 */
export const SUBSCRIPTION_BACKENDS = [
    {
        id: 'codex_image',
        name: 'ChatGPT Images',
        provider: 'codex',
        providerModel: 'gpt-image-2',
        capability: 'text_to_image',
        inputTypes: ['text'],
        outputTypes: ['image'],
        tool: 'image_generate',
        providerLabel: 'subscription · ChatGPT (Codex)',
    },
    {
        id: 'codex_image_edit',
        name: 'ChatGPT Images (edit)',
        provider: 'codex',
        providerModel: 'gpt-image-2',
        capability: 'image_to_image',
        inputTypes: ['image', 'text'],
        outputTypes: ['image'],
        tool: 'image_generate',
        providerLabel: 'subscription · ChatGPT (Codex)',
    },
    {
        id: 'grok_image',
        name: 'Grok Images',
        provider: 'grok',
        providerModel: 'grok-imagine-image-2.0',
        capability: 'text_to_image',
        inputTypes: ['text'],
        outputTypes: ['image'],
        tool: 'image_generate',
        providerLabel: 'subscription · Grok (X Premium)',
    },
    {
        id: 'grok_image_edit',
        name: 'Grok Images (edit)',
        provider: 'grok',
        providerModel: 'grok-imagine-image-2.0',
        capability: 'image_to_image',
        inputTypes: ['image', 'text'],
        outputTypes: ['image'],
        tool: 'image_generate',
        providerLabel: 'subscription · Grok (X Premium)',
    },
    {
        id: 'grok_video',
        name: 'Grok Video',
        provider: 'grok',
        providerModel: 'grok-imagine-video-1.5',
        capability: 'video_generation',
        inputTypes: ['text', 'image'],
        outputTypes: ['video'],
        tool: 'video_generate',
        providerLabel: 'subscription · Grok (X Premium)',
    },
];
/**
 * Build the hub options for the subscription image and video backends.
 *
 * @param access - how to reach the subscription plugin's tools.
 * @param log - the plugin's logger.
 * @returns one entry per backend, executable or not.
 */
export function buildSubscriptionBackendOptions(access, log) {
    return SUBSCRIPTION_BACKENDS.map((backend) => buildOne(backend, access, log));
}
/**
 * Build one backend's option.
 * @param backend - the backend.
 * @param access - tool access.
 * @param log - the plugin's logger.
 * @returns the option, marked executable or not.
 */
function buildOne(backend, access, log) {
    const registered = access.resolveTool(backend.tool);
    const facts = {
        id: backend.id,
        name: backend.name,
        type: backend.capability === 'video_generation' ? 'video_generation' : backend.capability === 'image_to_image' ? 'image_editing' : 'image_generation',
        source: 'subscription',
        provider: backend.provider,
        providerLabel: backend.providerLabel,
        capabilities: [backend.capability],
        inputTypes: backend.inputTypes,
        outputTypes: backend.outputTypes,
        // An option whose tool is not registered cannot run, and says so rather than
        // being advertised as available and failing on the first request. A remote
        // backend has no GPU or RAM footprint here, so nothing about this machine
        // participates in the decision.
        executable: registered !== undefined,
        ...(registered === undefined
            ? {
                unsupportedReason: `the subscription plugin's ${backend.tool} tool is not registered in this profile, so nothing can execute ` +
                    `${backend.provider} ${backend.capability}`,
            }
            : {}),
        catalogEnabled: false,
    };
    const option = {
        facts,
        status: () => readiness(backend, registered, access),
        invoke: async (request, context) => {
            if (registered === undefined) {
                throw new ModelHubError('UNSUPPORTED_OPERATION', `${backend.name} cannot run: the subscription plugin's ${backend.tool} tool is not registered`, { optionId: backend.id, capability: request.capability });
            }
            if (access.toolPermitted !== undefined && !access.toolPermitted(registered, backend.provider)) {
                throw new ModelHubError('POLICY_DISABLED', `${backend.name} cannot run: the subscription plugin's own tool policy disables ${registered}`, { optionId: backend.id, capability: request.capability });
            }
            return invokeBackend(backend, registered, access, request, context, log);
        },
    };
    return { facts, executable: registered !== undefined, ...(registered === undefined ? { reason: facts.unsupportedReason } : {}), option };
}
/**
 * Report a backend's live usability.
 *
 * Honest about what can be known from here: whether the tool exists, and whether
 * the subscription plugin's own session policy allows it. Whether a login is
 * currently valid is *not* knowable without asking the provider, and asking on
 * every routing decision would be both slow and a needless network call — so this
 * reports ready and lets the actual call tell the truth, which the plan then
 * treats as a definitive failure and falls past.
 *
 * @param backend - the backend.
 * @param registered - its registered tool name, when it has one.
 * @param access - tool access.
 * @returns the live state.
 */
function readiness(backend, registered, access) {
    if (registered === undefined) {
        return {
            state: 'unsupported',
            reason: `Unsupported: the subscription plugin's ${backend.tool} tool is not registered in this profile.`,
        };
    }
    if (access.toolPermitted !== undefined && !access.toolPermitted(registered, backend.provider)) {
        return {
            state: 'disabled',
            reason: `Disabled: the subscription plugin's own tool policy disables ${registered} for this session.`,
        };
    }
    return {
        state: 'ready',
        reason: `Ready: the ${backend.provider} ${backend.tool} tool is registered and permitted here. Whether its login is currently valid is only knowable by calling it.`,
    };
}
/**
 * Translate the hub's request into the subscription tool's arguments and run it.
 *
 * @param backend - the backend being dispatched.
 * @param tool - its registered tool name.
 * @param access - tool access.
 * @param request - the hub's request.
 * @param context - the hub's invocation context.
 * @param log - the plugin's logger.
 * @returns the artifact ids produced and any structured value.
 */
async function invokeBackend(backend, tool, access, request, context, log) {
    const prompt = (context.prompt ?? '').trim();
    if (backend.capability !== 'image_to_image' && prompt.length === 0) {
        throw new ModelHubError('INVOCATION_FAILED', `${backend.name} needs a prompt and this request carried none`, { optionId: backend.id, capability: context.capability });
    }
    const references = backend.capability === 'image_to_image' ? await imageReferences(context, backend.id) : undefined;
    if (backend.capability === 'image_to_image' && (references === undefined || references.length === 0)) {
        throw new ModelHubError('ARTIFACT_ERROR', `${backend.name} edits an existing image and this request supplied no image input`, { optionId: backend.id, capability: context.capability });
    }
    const args = { prompt, provider: providerArgument(backend.provider) };
    if (references !== undefined)
        args['referenceImages'] = references;
    const size = sizeArgument(context.options);
    if (size !== undefined)
        args['size'] = size;
    const quality = qualityArgument(context.options);
    if (quality !== undefined)
        args['quality'] = quality;
    if (backend.capability === 'video_generation') {
        // `video_generate` takes its own option names; the image tool's `size`
        // vocabulary does not apply, so only what the caller actually set is passed.
        delete args['size'];
        delete args['quality'];
        args['provider'] = undefined;
        delete args['provider'];
        for (const key of ['duration', 'aspect_ratio', 'resolution']) {
            const value = context.options[key];
            if (value !== undefined)
                args[key] = value;
        }
        // Video gives no way to name a provider — the tool is Grok-only — so the
        // option id is the only thing that could be wrong here, and it is fixed.
        void backend.provider;
    }
    log.debug(`dispatching ${context.capability} to ${tool} for ${backend.id} (provider ${backend.provider})`);
    // Past this line the request is the subscription plugin's, and the hub can no
    // longer call it off: the tool may already have an image on the wire. Announcing
    // submission here is what makes a later timeout or disconnect produce an
    // *unresolved* outcome instead of a retry — the one rule that keeps a policy
    // change or a dropped socket from being paid for twice.
    context.onSubmit?.({ optionId: backend.id, provider: backend.provider, tool });
    let raw;
    try {
        // The call is made exactly once, for exactly this provider. A refusal is
        // classified from what the tool said rather than by trying a different
        // provider behind the operator's back — the subscription plugin's own
        // internal provider fallback is closed off at the execution boundary by
        // `enforcement.ts`, which refuses a call that names no provider at all.
        raw = await access.callTool(tool, args, context.signal);
    }
    catch (error) {
        throw classifyToolFailure(error, backend);
    }
    const produced = await storeOutputs(raw, backend, context);
    if (produced.artifactIds.length === 0) {
        throw new ModelHubError('INVOCATION_FAILED', `${backend.name} reported success but produced no file this hub could archive`, { optionId: backend.id, capability: context.capability });
    }
    return produced;
}
/**
 * Classify a failure the subscription tool reported.
 *
 * The distinction this draws decides whether the plan may move on. A refusal the
 * backend *stated* — nobody logged in, quota exhausted, rate limited, rejected
 * credentials — is a definitive "no" and the next permitted option may be tried.
 * Everything else keeps its own code, so the hub's post-submission ambiguity rule
 * can recognise a timeout or a dropped socket and refuse to duplicate the work.
 *
 * @param error - the tool's failure.
 * @param backend - the backend that failed.
 * @returns the error to propagate.
 */
function classifyToolFailure(error, backend) {
    const described = toHubError(error, 'INVOCATION_FAILED', { optionId: backend.id });
    const message = described.message.toLowerCase();
    if (message.includes('no image provider is logged in') ||
        message.includes('no image provider is configured') ||
        message.includes('disabled for this session') ||
        message.includes('quota') ||
        message.includes('unauthorized') ||
        message.includes('forbidden') ||
        message.includes('401') ||
        message.includes('403') ||
        message.includes('429') ||
        message.includes('rate limit')) {
        return new ModelHubError('MODEL_UNAVAILABLE', `${backend.name}: ${described.message}`, {
            optionId: backend.id,
            code: described.code,
        });
    }
    return described;
}
/**
 * The `provider` argument value the subscription tool expects.
 *
 * Exported because the execution-boundary guard has to agree with it exactly: a
 * guard that thought `codex` was spelled one way while the adapter sent another
 * would be a policy hole.
 *
 * @param provider - the subscription provider.
 * @returns the tool's own spelling.
 */
export function providerArgument(provider) {
    return provider === 'codex' ? 'gpt' : 'grok';
}
/**
 * Read the request's image inputs as the subscription tool's own reference shape.
 *
 * The tool takes complete DSH image references — attachment id, media type, byte
 * count, dimensions — not artifact ids, because that is what the attachment store
 * it reads through understands. Converting here rather than asking the agent to
 * do it is what lets one request drive a local ComfyUI workflow and a
 * subscription edit identically.
 *
 * @param context - the hub's invocation context.
 * @param optionId - the option being dispatched, for error details.
 * @returns the references, or `undefined` when the request carried no image.
 */
async function imageReferences(context, optionId) {
    const images = context.inputs.filter((artifact) => artifact.type === 'image');
    if (images.length === 0)
        return undefined;
    const references = [];
    for (const artifact of images) {
        const attachmentId = typeof artifact.metadata['attachmentId'] === 'string' ? artifact.metadata['attachmentId'] : undefined;
        if (attachmentId === undefined) {
            throw new ModelHubError('ARTIFACT_ERROR', `image artifact "${artifact.id}" carries no attachment reference, which the subscription edit API requires. ` +
                'Re-import the image with read_image so it has one.', { optionId, artifactId: artifact.id });
        }
        const mediaType = artifact.mimeType ?? 'image/png';
        if (!['image/png', 'image/jpeg', 'image/webp', 'image/gif'].includes(mediaType)) {
            throw new ModelHubError('ARTIFACT_ERROR', `image artifact "${artifact.id}" is ${mediaType}, which the subscription edit API does not accept`, { optionId, artifactId: artifact.id, mediaType });
        }
        const width = typeof artifact.metadata['width'] === 'number' ? artifact.metadata['width'] : undefined;
        const height = typeof artifact.metadata['height'] === 'number' ? artifact.metadata['height'] : undefined;
        if (width === undefined || height === undefined) {
            throw new ModelHubError('ARTIFACT_ERROR', `image artifact "${artifact.id}" does not record its dimensions, which the subscription edit API requires`, { optionId, artifactId: artifact.id });
        }
        references.push({
            attachmentId,
            mediaType,
            bytes: artifact.byteLength ?? 0,
            width,
            height,
            ...(artifact.label === undefined ? {} : { name: artifact.label }),
        });
    }
    return references;
}
/** The sizes the subscription image API accepts. */
const IMAGE_SIZES = ['1024x1024', '1024x1536', '1536x1024', 'auto'];
/**
 * Translate `{ width, height }` options into the API's `size` value.
 *
 * A pair that does not match one of the provider's own sizes is *dropped*, not
 * rounded: silently rendering at a different resolution than the caller asked for
 * is worse than rendering at the provider default, and the structured result
 * records the drop so it is visible.
 *
 * @param options - the request's option bag.
 * @returns the size string, or `undefined`.
 */
function sizeArgument(options) {
    const explicit = options['size'];
    if (typeof explicit === 'string' && IMAGE_SIZES.includes(explicit))
        return explicit;
    const width = options['width'];
    const height = options['height'];
    if (typeof width !== 'number' || typeof height !== 'number')
        return undefined;
    const candidate = `${String(width)}x${String(height)}`;
    return IMAGE_SIZES.includes(candidate) ? candidate : undefined;
}
/**
 * Normalize a quality option.
 * @param options - the request's option bag.
 * @returns the quality string, or `undefined`.
 */
function qualityArgument(options) {
    const value = options['quality'];
    return typeof value === 'string' && ['low', 'medium', 'high', 'auto'].includes(value) ? value : undefined;
}
/**
 * Archive everything the tool produced into the hub's artifact store.
 *
 * Both the image and the video tool report a filesystem path under the harness
 * home. That path is not an artifact, so a later step cannot consume it by id —
 * which is exactly why the bytes are read back and re-stored. The subscription
 * plugin's own file is left where it is: it may be the only copy the operator's
 * UI shows, and deleting another plugin's output would be rude and lossy.
 *
 * @param raw - the tool's canonical output value.
 * @param backend - the backend that produced it.
 * @param context - the hub's invocation context.
 * @returns the artifact ids and the structured value.
 */
async function storeOutputs(raw, backend, context) {
    const value = typeof raw === 'object' && raw !== null ? raw : {};
    const files = [];
    const producedType = backend.capability === 'video_generation' ? 'video' : 'image';
    if (typeof value.path === 'string' && value.path.length > 0) {
        files.push({
            path: value.path,
            type: producedType,
            ...(producedType === 'video' ? { mimeType: 'video/mp4' } : {}),
        });
    }
    for (const path of value.paths ?? []) {
        if (typeof path === 'string' && path.length > 0)
            files.push({ path, type: 'image' });
    }
    const artifactIds = [];
    const archived = [];
    for (const file of files) {
        let bytes;
        try {
            bytes = await readFile(file.path);
        }
        catch (error) {
            throw new ModelHubError('ARTIFACT_ERROR', `${backend.name} reported "${file.path}" but it could not be read back: ${error instanceof Error ? error.message : String(error)}`, { optionId: backend.id, path: file.path });
        }
        const artifact = await context.artifacts.put({
            type: file.type,
            bytes,
            ...(file.mimeType === undefined ? {} : { mimeType: file.mimeType }),
            label: file.path.split(/[\\/]/).pop() ?? backend.name,
            producerModelId: backend.id,
            metadata: {
                provider: backend.provider,
                providerModel: backend.providerModel,
                sourcePath: file.path,
            },
        });
        artifactIds.push(artifact.id);
        archived.push({ id: artifact.id, source: file.path, bytes: bytes.byteLength });
    }
    return {
        artifactIds,
        value: {
            provider: backend.provider,
            providerModel: backend.providerModel,
            archived,
            ...(value.revisedPrompt === undefined ? {} : { revisedPrompt: value.revisedPrompt }),
            ...(typeof value.url === 'string' ? { providerUrl: value.url } : {}),
            ...(typeof value.duration === 'number' ? { durationSeconds: value.duration } : {}),
        },
    };
}
/**
 * The facts for a backend, before a hub exists to describe it.
 *
 * Used by the execution-boundary guard, which needs to reason about a backend
 * from a tool call's arguments and has no plan in hand.
 *
 * @param id - the option id.
 * @returns its declared facts, or `undefined` when the id is not a backend.
 */
export function backendFacts(id) {
    const backend = SUBSCRIPTION_BACKENDS.find((entry) => entry.id === id);
    if (backend === undefined)
        return undefined;
    return {
        id: backend.id,
        name: backend.name,
        type: 'image_generation',
        source: 'subscription',
        provider: backend.provider,
        providerLabel: backend.providerLabel,
        capabilities: [backend.capability],
        inputTypes: backend.inputTypes,
        outputTypes: backend.outputTypes,
        executable: true,
        catalogEnabled: false,
    };
}
/**
 * The option id one tool call would run on.
 *
 * The mapping is by *(provider, whether the call carries reference images)* and
 * never by tool name alone, because one tool serves four distinct options: the
 * ChatGPT and Grok image options, and each one's editing variant. Getting this
 * wrong would let a call marked "text-to-image" be permitted to run an edit
 * option the operator had disabled.
 *
 * @param tool - the registered tool name.
 * @param provider - the provider the call named, when it named one.
 * @param edits - whether the call carries reference images.
 * @returns the option id, or `undefined` when the tool is not a subscription backend tool.
 */
export function optionIdForToolCall(tool, provider, edits) {
    const canonical = canonicalToolName(tool);
    if (canonical === undefined)
        return undefined;
    if (canonical === 'video_generate')
        return 'grok_video';
    if (canonical !== 'image_generate')
        return undefined;
    if (provider === undefined)
        return undefined;
    if (provider === 'codex')
        return edits ? 'codex_image_edit' : 'codex_image';
    return edits ? 'grok_image_edit' : 'grok_image';
}
/** Every tool name this integration claims, canonical and aliased. */
export function subscriptionToolNames() {
    const names = new Set();
    for (const backend of SUBSCRIPTION_BACKENDS) {
        names.add(backend.tool);
        names.add(`dsh_subscriptions_${backend.tool}`);
    }
    return [...names];
}
/**
 * Map a registered tool name back to its canonical name.
 * @param tool - the registered name.
 * @returns the canonical name, or `undefined`.
 */
export function canonicalToolName(tool) {
    const known = ['image_generate', 'video_generate'];
    if (known.includes(tool))
        return tool;
    const prefix = 'dsh_subscriptions_';
    if (tool.startsWith(prefix)) {
        const stripped = tool.slice(prefix.length);
        return known.includes(stripped) ? stripped : undefined;
    }
    return undefined;
}
/** Re-exported so a caller can reason about a capability's declared inputs. */
export { CAPABILITY_IO };
