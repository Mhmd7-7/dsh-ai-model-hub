/**
 * Discovery tools: what models, what capabilities, what state, what artifacts.
 *
 * These are the tools that let the agent answer "what can this machine do?"
 * before it commits to a plan. They are deliberately separate from the
 * invocation tool so a planning turn can be read-only.
 *
 * None of them names a model, an engine, or a capability in code. Every value
 * they report comes from the catalog at call time.
 *
 * @module dsh-ai-model-hub/dsh-plugin/tools/discovery
 */
import { defineTool } from '@deepseek-ai/dsh-tools';
import { formatArtifact, formatAvailability, formatModel, textBlock, toToolError } from "./support.js";
/**
 * Register the read-only discovery tools.
 *
 * @param ctx - the context whose `tools` registry receives them.
 * @param service - the hub service to read from.
 * @param options - the per-call artifact root resolver, so a listing shows the
 *   calling session's artifacts rather than the boot-time store's.
 */
export function registerDiscoveryTools(ctx, service, options) {
    const hub = service.hub;
    ctx.tools.register(defineTool({
        name: 'list_models',
        description: 'List the local AI models this machine offers, with their capabilities, live status, and resource requirements. ' +
            'Use this to discover what is available before planning work that needs a model. ' +
            'Models are discovered from configuration; none are hard-coded. ' +
            'Optionally filter to models that declare one capability.',
        parameters: {
            capability: {
                type: 'string',
                description: 'Only list models declaring this capability, e.g. text_to_image. Call list_capabilities to see the vocabulary.',
            },
            includeDisabled: {
                type: 'boolean',
                description: 'Include models disabled in configuration. Defaults to true, since a disabled model explains a missing capability.',
            },
        },
        output: {
            schema: {
                type: 'object',
                additionalProperties: false,
                properties: {
                    count: { type: 'integer', required: true },
                    models: {
                        type: 'array',
                        required: true,
                        items: {
                            type: 'object',
                            additionalProperties: false,
                            properties: {
                                id: { type: 'string', required: true },
                                name: { type: 'string', required: true },
                                type: { type: 'string', required: true },
                                capabilities: { type: 'array', required: true, items: { type: 'string' } },
                                availability: { type: 'string', required: true },
                                lifecycle: { type: 'string', required: true },
                                engine: { type: 'string', required: true },
                                endpoint: { type: 'string' },
                                vramGb: { type: 'number', required: true },
                                ramGb: { type: 'number', required: true },
                                startable: { type: 'boolean', required: true },
                                tags: { type: 'array', required: true, items: { type: 'string' } },
                            },
                        },
                    },
                },
            },
            render: (_args, value) => {
                if (value.count === 0) {
                    return textBlock('No models are registered. Check the model catalog configuration (config/models.json).');
                }
                const blocks = value.models.map((model) => [
                    `${model.id} â€” ${model.name}`,
                    `  type: ${model.type}`,
                    `  capabilities: ${model.capabilities.join(', ') || '(none)'}`,
                    `  status: ${model.availability} [${model.lifecycle}]`,
                    `  engine: ${model.engine}${model.endpoint === undefined ? '' : ` at ${model.endpoint}`}`,
                    `  resources: ${model.vramGb} GiB VRAM, ${model.ramGb} GiB RAM`,
                    `  startable by the hub: ${model.startable ? 'yes' : 'no'}`,
                    model.tags.length > 0 ? `  tags: ${model.tags.join(', ')}` : '',
                ]
                    .filter((line) => line.length > 0)
                    .join('\n'));
                return textBlock(`${value.count} model(s) available:\n\n${blocks.join('\n\n')}`);
            },
        },
        execute: async (args) => {
            const views = await Promise.resolve(hub.listModels({ includeDisabled: args.includeDisabled ?? true }));
            const filtered = args.capability === undefined || args.capability.trim().length === 0
                ? views
                : views.filter((view) => view.model.capabilities.includes(args.capability));
            return {
                count: filtered.length,
                models: filtered.map((view) => ({
                    id: view.model.id,
                    name: view.model.name,
                    type: view.model.type,
                    capabilities: [...view.model.capabilities],
                    availability: view.status.availability,
                    lifecycle: view.status.lifecycle,
                    engine: view.model.runtime.engine,
                    ...(view.model.runtime.endpoint === undefined ? {} : { endpoint: view.model.runtime.endpoint }),
                    vramGb: view.model.resources.vramGb,
                    ramGb: view.model.resources.ramGb,
                    startable: view.model.lifecycle.startable,
                    tags: [...view.model.tags],
                })),
            };
        },
    }));
    ctx.tools.register(defineTool({
        name: 'list_capabilities',
        description: 'List the media and text capabilities this machine can currently serve, which models provide each, and which capabilities have no model at all. ' +
            'This is the vocabulary to use with invoke_model. ' +
            'Call this before planning a multi-step workflow so you know which steps are actually possible here.',
        parameters: {},
        output: {
            schema: {
                type: 'object',
                additionalProperties: false,
                properties: {
                    capabilities: {
                        type: 'array',
                        required: true,
                        items: {
                            type: 'object',
                            additionalProperties: false,
                            properties: {
                                capability: { type: 'string', required: true },
                                inputTypes: { type: 'array', required: true, items: { type: 'string' } },
                                outputTypes: { type: 'array', required: true, items: { type: 'string' } },
                                models: {
                                    type: 'array',
                                    required: true,
                                    items: {
                                        type: 'object',
                                        additionalProperties: false,
                                        properties: {
                                            id: { type: 'string', required: true },
                                            availability: { type: 'string', required: true },
                                            priority: { type: 'number', required: true },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    unserved: {
                        type: 'array',
                        required: true,
                        items: {
                            type: 'object',
                            additionalProperties: false,
                            properties: {
                                capability: { type: 'string', required: true },
                                reason: { type: 'string', required: true },
                            },
                        },
                    },
                },
            },
            render: (_args, value) => {
                const lines = [];
                if (value.capabilities.length === 0) {
                    lines.push('No capability is currently served by any enabled model.');
                }
                else {
                    lines.push('Capabilities available on this machine:');
                    for (const entry of value.capabilities) {
                        lines.push(`\n${entry.capability}  (${entry.inputTypes.join('|')} â†’ ${entry.outputTypes.join('|')})`);
                        for (const model of entry.models) {
                            lines.push(`  - ${model.id} [${model.availability}, priority ${model.priority}]`);
                        }
                    }
                }
                if (value.unserved.length > 0) {
                    lines.push('\nCapabilities with no usable model:');
                    for (const entry of value.unserved) {
                        lines.push(`  - ${entry.capability}: ${entry.reason}`);
                    }
                }
                return textBlock(lines.join('\n'));
            },
        },
        execute: async () => {
            const capabilities = hub.listCapabilities().map((view) => ({
                capability: view.capability,
                inputTypes: [...view.inputTypes],
                outputTypes: [...view.outputTypes],
                models: view.modelIds.map((modelId) => {
                    const status = hub.getModelStatus(modelId);
                    return {
                        id: modelId,
                        availability: status.availability,
                        priority: hub.getModel(modelId).model.priority,
                    };
                }),
            }));
            const unserved = hub.listUnservedCapabilities().map((entry) => ({
                capability: entry.capability,
                reason: entry.reason,
            }));
            return { capabilities, unserved };
        },
    }));
    ctx.tools.register(defineTool({
        name: 'get_model_status',
        description: 'Report whether specific models are running, healthy, and available, including the reason when one is not. ' +
            'Use this after a failed invocation to find out what actually went wrong, or before one to check that the machine can serve it. ' +
            'Omit `modelId` to survey every model.',
        parameters: {
            modelId: {
                type: 'string',
                description: 'The model to inspect. Omit to report every registered model.',
            },
        },
        output: {
            schema: {
                type: 'object',
                additionalProperties: false,
                properties: {
                    machine: {
                        type: 'object',
                        required: true,
                        additionalProperties: false,
                        properties: {
                            vramGb: { type: 'number', required: true },
                            ramGb: { type: 'number', required: true },
                            hasGpu: { type: 'boolean', required: true },
                            notes: { type: 'string', required: true },
                        },
                    },
                    statuses: {
                        type: 'array',
                        required: true,
                        items: {
                            type: 'object',
                            additionalProperties: false,
                            properties: {
                                modelId: { type: 'string', required: true },
                                availability: { type: 'string', required: true },
                                lifecycle: { type: 'string', required: true },
                                reason: { type: 'string' },
                                pid: { type: 'integer' },
                                healthy: { type: 'boolean' },
                                healthDetail: { type: 'string' },
                                activeInvocations: { type: 'integer', required: true },
                                startable: { type: 'boolean', required: true },
                                stoppable: { type: 'boolean', required: true },
                                vramGb: { type: 'number', required: true },
                                ramGb: { type: 'number', required: true },
                            },
                        },
                    },
                },
            },
            render: (_args, value) => {
                const lines = [
                    `machine: ${value.machine.ramGb} GiB RAM, ${value.machine.vramGb} GiB VRAM, GPU ${value.machine.hasGpu ? 'present' : 'absent'}`,
                    `  ${value.machine.notes}`,
                    '',
                ];
                for (const status of value.statuses) {
                    lines.push(`${status.modelId}: ${status.availability} [${status.lifecycle}]`);
                    if (status.reason !== undefined)
                        lines.push(`  reason: ${status.reason}`);
                    if (status.pid !== undefined)
                        lines.push(`  pid: ${status.pid}`);
                    if (status.healthy !== undefined) {
                        lines.push(`  health: ${status.healthy ? 'passing' : `failing â€” ${status.healthDetail ?? 'no detail'}`}`);
                    }
                    lines.push(`  active invocations: ${status.activeInvocations}`);
                    lines.push(`  hub can start: ${status.startable ? 'yes' : 'no'}; stop: ${status.stoppable ? 'yes' : 'no'}`);
                    lines.push(`  needs: ${status.vramGb} GiB VRAM, ${status.ramGb} GiB RAM`);
                }
                return textBlock(lines.join('\n'));
            },
        },
        execute: async (args) => {
            try {
                const machine = hub.machineProfile;
                const modelIds = args.modelId === undefined || args.modelId.trim().length === 0
                    ? hub.listModels({ includeDisabled: true }).map((view) => view.model.id)
                    : [args.modelId];
                const statuses = modelIds.map((modelId) => {
                    const status = hub.getModelStatus(modelId);
                    const model = hub.getModel(modelId).model;
                    return {
                        modelId,
                        availability: status.availability,
                        lifecycle: status.lifecycle,
                        ...(status.reason === undefined ? {} : { reason: status.reason }),
                        ...(status.pid === undefined ? {} : { pid: status.pid }),
                        ...(status.health === undefined ? {} : { healthy: status.health.healthy }),
                        ...(status.health?.detail === undefined ? {} : { healthDetail: status.health.detail }),
                        activeInvocations: status.activeInvocations,
                        startable: model.lifecycle.startable,
                        stoppable: model.lifecycle.stoppable,
                        vramGb: model.resources.vramGb,
                        ramGb: model.resources.ramGb,
                    };
                });
                return {
                    machine: { vramGb: machine.vramGb, ramGb: machine.ramGb, hasGpu: machine.hasGpu, notes: machine.notes },
                    statuses,
                };
            }
            catch (error) {
                throw toToolError(error, 'MODEL_NOT_FOUND', { requestedModelId: args.modelId });
            }
        },
    }));
    ctx.tools.register(defineTool({
        name: 'list_artifacts',
        description: 'List artifacts produced earlier â€” images, 3D meshes, audio, text â€” with their ids. ' +
            'Pass an artifact id to invoke_model as an input to feed one model\'s output into another. ' +
            'Use this to find previously generated content in a multi-step workflow.',
        parameters: {
            limit: {
                type: 'integer',
                description: 'Maximum number of artifacts to return, newest first. Defaults to 20.',
            },
        },
        output: {
            schema: {
                type: 'object',
                additionalProperties: false,
                properties: {
                    count: { type: 'integer', required: true },
                    artifacts: {
                        type: 'array',
                        required: true,
                        items: {
                            type: 'object',
                            additionalProperties: false,
                            properties: {
                                id: { type: 'string', required: true },
                                type: { type: 'string', required: true },
                                description: { type: 'string', required: true },
                            },
                        },
                    },
                },
            },
            render: (_args, value) => {
                if (value.count === 0)
                    return textBlock('No artifacts have been produced yet.');
                return textBlock(`${value.count} artifact(s), newest first:\n\n${value.artifacts.map((artifact) => `  ${artifact.description}\n    id: ${artifact.id}`).join('\n')}`);
            },
        },
        execute: async (args, exec) => {
            const limit = Math.max(1, Math.min(200, args.limit ?? 20));
            const artifactRoot = options.artifactRootFor(exec);
            const artifacts = await hub.listArtifacts(limit, artifactRoot);
            return {
                count: artifacts.length,
                artifacts: artifacts.map((artifact) => ({
                    id: artifact.id,
                    type: artifact.type,
                    description: formatArtifact(artifact, '').split('\n')[0] ?? artifact.id,
                })),
            };
        },
    }));
}
/**
 * Register a diagnostic tool that explains routing without invoking anything.
 *
 * Kept separate from `invoke_model` because "which model would you use, and why"
 * is the question an operator asks when a workflow behaved unexpectedly, and
 * answering it must not have side effects.
 *
 * @param ctx - the context whose `tools` registry receives it.
 * @param service - the hub service to read from.
 */
export function registerRoutingTool(ctx, service, options) {
    const hub = service.hub;
    ctx.tools.register(defineTool({
        name: 'explain_routing',
        description: 'Explain which model would be chosen for a capability and why, considering every candidate that was considered and rejected. ' +
            'Read-only: nothing is started or invoked. ' +
            'Use this when a model choice was surprising, or to check that the machine can serve a workflow before running it.',
        parameters: {
            capability: {
                type: 'string',
                required: true,
                description: 'The capability to route, e.g. text_to_image.',
            },
            prompt: {
                type: 'string',
                description: 'The prompt you would send, when it affects the choice.',
            },
            inputArtifactIds: {
                type: 'array',
                description: 'Artifact ids you would pass as inputs. Their kinds are checked against each model.',
                items: { type: 'string' },
            },
        },
        output: {
            schema: {
                type: 'object',
                additionalProperties: false,
                properties: {
                    chosen: { type: 'string', required: true },
                    rationale: { type: 'string', required: true },
                    candidates: {
                        type: 'array',
                        required: true,
                        items: {
                            type: 'object',
                            additionalProperties: false,
                            properties: {
                                modelId: { type: 'string', required: true },
                                eligible: { type: 'boolean', required: true },
                                reason: { type: 'string', required: true },
                            },
                        },
                    },
                },
            },
            render: (_args, value) => {
                const lines = [`Would choose: ${value.chosen}`, `Why: ${value.rationale}`, '', 'Candidates considered:'];
                for (const candidate of value.candidates) {
                    lines.push(`  [${candidate.eligible ? 'eligible' : 'rejected'}] ${candidate.modelId}: ${candidate.reason}`);
                }
                return textBlock(lines.join('\n'));
            },
        },
        execute: async (args, exec) => {
            try {
                const artifactRoot = options.artifactRootFor(exec);
                const decision = await hub.route({
                    capability: args.capability,
                    ...(args.prompt === undefined ? {} : { prompt: args.prompt }),
                    ...(args.inputArtifactIds === undefined
                        ? {}
                        : { inputs: args.inputArtifactIds.map((id) => ({ id })) }),
                }, artifactRoot === undefined ? {} : { artifactRoot });
                return {
                    chosen: decision.modelId,
                    rationale: decision.rationale,
                    candidates: decision.candidates.map((candidate) => ({
                        modelId: candidate.modelId,
                        eligible: candidate.eligible,
                        reason: candidate.reason,
                    })),
                };
            }
            catch (error) {
                throw toToolError(error, 'NO_COMPATIBLE_MODEL', { capability: args.capability });
            }
        },
    }));
}
export { formatAvailability, formatModel };
