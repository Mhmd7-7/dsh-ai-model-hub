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
export {
  CAPABILITIES,
  CAPABILITY_IO,
  MODEL_TYPES,
  IO_TYPES,
  defaultIoFor,
  isCapability,
  isIoType,
  isModelType,
} from './catalog/capabilities.ts';
export type { Capability, IoType, ModelType } from './catalog/capabilities.ts';

// ── Descriptors and configuration ───────────────────────────────────────────
export {
  ADAPTER_KINDS,
  HEALTH_CHECK_KINDS,
  isValidModelId,
  parseModelCatalogConfig,
  resolveDescriptor,
} from './catalog/descriptor.ts';
export type {
  AdapterConfig,
  AdapterKind,
  CommandSpec,
  HealthCheckKind,
  HealthCheckSpec,
  LifecycleSpec,
  LimitsSpec,
  ModelCatalogConfig,
  ModelDescriptor,
  ModelHost,
  ResolvedModel,
  ResourceSpec,
  RuntimeSpec,
} from './catalog/descriptor.ts';

// ── The catalog ─────────────────────────────────────────────────────────────
export { ModelCatalog, catalogFromConfig, checkDescriptorCoherence, summarizeModel } from './catalog/registry.ts';
export type { CapabilityView } from './catalog/registry.ts';

// ── Artifacts ───────────────────────────────────────────────────────────────
export { describeArtifact, formatBytes, readArtifactConventions } from './artifacts/types.ts';
export type {
  Artifact,
  ArtifactConventions,
  ArtifactStore,
  ArtifactWriteRequest,
} from './artifacts/types.ts';
export { LocalArtifactStore, fileUriToPath, pathToFileUri } from './artifacts/local-store.ts';

// ── Adapters ────────────────────────────────────────────────────────────────
export { AdapterRegistry, lineLogger, silentLogger } from './adapters/types.ts';
export type {
  AdapterInvocation,
  AdapterLogger,
  AdapterOutput,
  ModelAdapter,
} from './adapters/types.ts';
export { createMockAdapter, renderMockWav } from './adapters/mock.ts';
export { createOpenAiCompatibleAdapter } from './adapters/openai.ts';
export { MOCK_STL_VERTEX_COUNT, colorFromSeed, renderMockPng, renderMockStl } from './adapters/png.ts';

// ── Router ──────────────────────────────────────────────────────────────────
export {
  DEFAULT_ROUTING_POLICY,
  describeResolvedRequest,
  explainDecision,
  resolveRequestInputs,
  routeRequest,
} from './router/router.ts';
export type { ResolvedRequest, RoutingContext, RoutingPolicy } from './router/router.ts';

// ── Runtime ─────────────────────────────────────────────────────────────────
export { RuntimeManager, parseHostPort, probeTcp } from './runtime/manager.ts';
export type { GateDecision, RuntimeManagerOptions } from './runtime/manager.ts';

// ── Execution safety ────────────────────────────────────────────────────────
export {
  DEFAULT_COMMAND_ALLOWLIST,
  DEFAULT_EXECUTION_POLICY,
  UnsafeCommandError,
  assertAllowedCommand,
  assertSafeArguments,
  buildChildEnvironment,
  commandBasename,
  delay,
  runCommand,
  spawnProcess,
  withPolicy,
  withTimeout,
} from './util/process.ts';
export type {
  CommandResult,
  ExecutionPolicy,
  ManagedProcess,
  RunCommandOptions,
} from './util/process.ts';

// ── Machine detection ───────────────────────────────────────────────────────
export { probeMachine } from './machine.ts';
export type { MachineProbeResult } from './machine.ts';

// ── Shared types ────────────────────────────────────────────────────────────
export {
  AVAILABILITY_STATES,
  LIFECYCLE_STATES,
} from './types.ts';
export type {
  ArtifactInput,
  AvailabilityState,
  CatalogOptions,
  HealthReport,
  InvocationRequest,
  InvocationResult,
  LifecycleState,
  MachineProfile,
  ModelRuntimeStatus,
  ModelView,
  RoutingCandidate,
  RoutingDecision,
} from './types.ts';

// ── Errors ──────────────────────────────────────────────────────────────────
export { ERROR_CODES, ModelHubError, describeError, toHubError } from './errors.ts';
export type { ErrorCode, HubErrorDetails } from './errors.ts';

// ── Validation helpers ──────────────────────────────────────────────────────
export {
  IssueCollector,
  formatIssues,
  isLosslessJson,
  isRecord,
} from './util/validate.ts';
export type { JsonValue, ValidationIssue } from './util/validate.ts';

// ── Configuration loading ───────────────────────────────────────────────────
export {
  findConfigDirectory,
  loadCatalogConfig,
  loadCatalogFromAnchors,
  loadHubFromDisk,
} from './config/load.ts';
export type { AnchorLoadOptions, AnchorLoadedCatalog, LoadedCatalog, LoadOptions } from './config/load.ts';

// ── The hub ─────────────────────────────────────────────────────────────────
export { ModelHub, defaultArtifactRoot } from './hub.ts';
export type { HubEvent, HubEventListener, InvokeOptions, ModelHubOptions } from './hub.ts';
