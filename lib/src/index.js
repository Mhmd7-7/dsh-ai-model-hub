/**
 * The hub's public surface.
 *
 * Everything a consumer needs is re-exported here, so the DSH plugin, a CLI, a
 * test, or a future HTTP daemon all import from one place and none of them
 * depends on the hub's internal file layout. That indirection is what makes
 * moving a module a non-breaking change.
 *
 * @module dsh-ai-model-hub
 */
// ── Vocabulary ──────────────────────────────────────────────────────────────
export { CAPABILITIES, CAPABILITY_IO, MODEL_TYPES, IO_TYPES, defaultIoFor, isCapability, isIoType, isModelType, } from "./catalog/capabilities.js";
// ── Descriptors and configuration ───────────────────────────────────────────
export { ADAPTER_KINDS, HEALTH_CHECK_KINDS, isValidModelId, parseModelCatalogConfig, resolveDescriptor, } from "./catalog/descriptor.js";
// ── The catalog ─────────────────────────────────────────────────────────────
export { ModelCatalog, catalogFromConfig, checkDescriptorCoherence, summarizeModel } from "./catalog/registry.js";
// ── Artifacts ───────────────────────────────────────────────────────────────
export { describeArtifact, formatBytes, readArtifactConventions } from "./artifacts/types.js";
export { LocalArtifactStore, fileUriToPath, pathToFileUri } from "./artifacts/local-store.js";
// ── Adapters ────────────────────────────────────────────────────────────────
export { AdapterRegistry, lineLogger, silentLogger } from "./adapters/types.js";
export { createMockAdapter, renderMockWav } from "./adapters/mock.js";
export { createOpenAiCompatibleAdapter } from "./adapters/openai.js";
export { createHttpJsonAdapter, extractImages, readPngSize } from "./adapters/http-json.js";
export { createComfyUiAdapter, readPngDimensions } from "./adapters/comfyui.js";
export { MOCK_STL_VERTEX_COUNT, colorFromSeed, renderMockPng, renderMockStl } from "./adapters/png.js";
// ── Router ──────────────────────────────────────────────────────────────────
export { DEFAULT_ROUTING_POLICY, describeResolvedRequest, explainDecision, resolveRequestInputs, routeRequest, } from "./router/router.js";
// ── Runtime ─────────────────────────────────────────────────────────────────
export { RuntimeManager, parseHostPort, probeTcp } from "./runtime/manager.js";
// ── Execution safety ────────────────────────────────────────────────────────
export { DEFAULT_COMMAND_ALLOWLIST, DEFAULT_EXECUTION_POLICY, UnsafeCommandError, assertAllowedCommand, assertSafeArguments, buildChildEnvironment, commandBasename, delay, runCommand, spawnProcess, withPolicy, withTimeout, } from "./util/process.js";
// ── Machine detection ───────────────────────────────────────────────────────
export { probeMachine } from "./machine.js";
// ── Shared types ────────────────────────────────────────────────────────────
export { AVAILABILITY_STATES, LIFECYCLE_STATES, } from "./types.js";
// ── Errors ──────────────────────────────────────────────────────────────────
export { ERROR_CODES, ModelHubError, describeError, toHubError } from "./errors.js";
// ── Validation helpers ──────────────────────────────────────────────────────
export { IssueCollector, formatIssues, isLosslessJson, isRecord, } from "./util/validate.js";
// ── Runtime discovery ───────────────────────────────────────────────────────
export { DEFAULT_DISCOVERY_TIMEOUT_MS, DEFAULT_DISCOVERY_TTL_MS, DISCOVERED_PRIORITY, DiscoveryRegistry, ioForCapabilities, mergeCatalogConfig, staticModelIds, } from "./discovery/types.js";
export { bytesToGib, fetchJson, hasKeyMatching, isRecordLike, readArray, readNumber, readNumberBySuffix, readString, slugifyModelId, stableDigest, } from "./discovery/http.js";
export { OLLAMA_ENGINE, capabilitiesForOllamaModel, createOllamaDiscoverer, joinUrl, mapOllamaModel, mapWithConcurrency, parseOllamaShow, parseOllamaTags, } from "./discovery/ollama.js";
export { A1111_CAPABILITIES, A1111_ENGINE, A1111_ENGINES, LOADED_PRIORITY, createA1111Discoverer, estimateA1111Vram, mapA1111Model, parseA1111Models, parseA1111Options, parseA1111Samplers, } from "./discovery/a1111.js";
export { CAPABILITY_SIGNALS, COMFYUI_ENGINE, COMFYUI_ENGINES, WEIGHT_FIELDS, buildDefaultGraph, capabilitiesForComfyModel, createComfyUiDiscoverer, describeIntrospection, estimateComfyVram, mapComfyWeightFile, parseComfyObjectInfo, } from "./discovery/comfyui.js";
// ── Configuration loading ───────────────────────────────────────────────────
export { findConfigDirectory, loadCatalogConfig, loadCatalogFromAnchors, loadHubFromDisk, } from "./config/load.js";
// ── The hub ─────────────────────────────────────────────────────────────────
export { ModelHub, defaultArtifactRoot } from "./hub.js";
