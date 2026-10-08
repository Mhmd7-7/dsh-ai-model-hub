/**
 * The "Local models" scan: what this machine can actually be asked to do.
 *
 * The settings page's Scan button runs this, and it answers one question with two
 * independent halves:
 *
 * 1. **Ollama** — every model the configured instance has installed, read from
 *    its own `/api/tags`.
 * 2. **ComfyUI** — every *complete workflow* it can find, each one a single
 *    selectable unit. Never a checkpoint, a LoRA, a VAE, a ControlNet, a text
 *    encoder, or an individual node: those are implementation details of a
 *    workflow and stay inside it.
 *
 * ## Why workflows, and where they come from
 *
 * A ComfyUI checkpoint is not a capability; a workflow is. Two sources are read,
 * and neither is required:
 *
 * - **The server's own saved workflows**, through `GET /userdata?dir=workflows`
 *   and `GET /userdata/{name}`. This is the integration ComfyUI itself provides,
 *   so it is used when it answers.
 * - **A configured workflow directory**, scanned for `*.json` files. This is the
 *   fallback for a deployment whose workflows live beside the catalog rather than
 *   inside ComfyUI — and it is the only source that works while ComfyUI is down.
 *
 * ## Independence, which is the point
 *
 * The two halves are attempted separately and contain their own failures. An
 * Ollama instance that is not running produces an Ollama warning and no Ollama
 * rows; it never hides the workflows, and a ComfyUI that is down never hides the
 * models. Each result carries its own status, so a partial answer is reported as
 * partial rather than as an empty machine.
 *
 * ## Idempotence
 *
 * Every resource gets an id derived from its *source location*, not a counter, so
 * scanning twice resolves to the same ids. The runnable workflows are then
 * published to the hub as workflow-backed providers, replacing the previous
 * scan's set wholesale — which is what makes a repeat scan a refresh rather than
 * a growing list of duplicates.
 *
 * @module dsh-ai-model-hub/dsh-plugin/scan
 */

import { readFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';

import type { Capability, ModelDescriptor, ModelHost, ModelHub } from '../src/index.ts';
import {
  DISCOVERED_PRIORITY,
  ioForCapabilities,
  readComfyNodeIo,
  scannedOllamaModelId,
  scannedWorkflowId,
  scanWorkflowDocument,
  workflowModelTypeFor,
} from '../src/index.ts';
import type { ComfyNodeIndex } from '../src/index.ts';

/** One directory entry, as the probes report it. */
export interface ScannedDirEntry {
  readonly name: string;
  readonly isDirectory: boolean;
  readonly sizeBytes?: number;
}

/**
 * The effects the scan needs, isolated so a test can describe a machine it does
 * not own. `readFile` is optional: omitting it means the real filesystem, which
 * is only consulted when a workflow directory is configured.
 */
export interface ScanProbes {
  readonly fetchJson: (url: string, timeoutMs: number) => Promise<unknown>;
  readonly isDirectory: (path: string) => Promise<boolean>;
  readonly listDir: (path: string) => Promise<readonly ScannedDirEntry[] | undefined>;
  readonly readFile?: (path: string) => Promise<string>;
}

/** The two kinds of thing the list shows. */
export type LocalResourceKind = 'ollama_model' | 'comfyui_workflow';

/** How ready a resource is to be used. */
export type LocalResourceStatus = 'ready' | 'needs_conversion' | 'invalid' | 'unavailable';

/** One selectable thing in the Local models list. */
export interface LocalResource {
  /** Stable id, derived from the source location so a rescan cannot duplicate it. */
  readonly id: string;
  /** Which kind of resource this is. */
  readonly kind: LocalResourceKind;
  /** The user-facing kind label: `Ollama Model` or `ComfyUI Workflow`. */
  readonly typeLabel: string;
  /** The display name: the workflow's own metadata name, else its filename. */
  readonly name: string;
  /** Where it was found — an endpoint plus route, or a file path. */
  readonly source: string;
  /** How ready it is. */
  readonly status: LocalResourceStatus;
  /** A sentence explaining the status. */
  readonly detail: string;
  /** Whether the hub can run it right now. */
  readonly runnable: boolean;
  /** The catalog provider id, when the hub has one registered for it. */
  readonly modelId?: string;
  /** Capabilities it serves, when it declares any. */
  readonly capabilities?: readonly string[];
  /** Public parameters a caller may set. */
  readonly inputs?: readonly string[];
  /** Public outputs it produces. */
  readonly outputs?: readonly string[];
  /** Size in bytes, when the engine or the filesystem reported one. */
  readonly sizeBytes?: number;
  /** The serialization a workflow was stored in. */
  readonly format?: 'api' | 'ui';
}

/** What one source contributed, so a partial answer can explain itself. */
export interface ScanSourceReport {
  /** The host id, or `workflow-dir` for the configured directory. */
  readonly id: string;
  /** Which kind of resource this source produces. */
  readonly kind: LocalResourceKind;
  /** A human label for the source. */
  readonly label: string;
  /** Whether the source was read successfully. */
  readonly ok: boolean;
  /** What happened, in a sentence. */
  readonly detail: string;
  /** How many resources it contributed. */
  readonly found: number;
}

/** The complete result of one scan. */
export interface LocalScanResult {
  /** Every discovered resource, workflows after models, each id unique. */
  readonly resources: readonly LocalResource[];
  /** One entry per source that was attempted. */
  readonly sources: readonly ScanSourceReport[];
  /** Non-fatal problems, phrased for a user. */
  readonly warnings: readonly string[];
  /** How many providers were published to the hub. */
  readonly registered: number;
  /** When the scan ran. */
  readonly generatedAt: string;
}

/** Options for {@link scanLocalResources}. */
export interface ScanOptions {
  /** The catalog's declared hosts. Engines not named here are not contacted. */
  readonly hosts: readonly ModelHost[];
  /** A directory of API/editor workflow JSON files to scan, when configured. */
  readonly workflowDir?: string;
  /** Budget for one HTTP request, in milliseconds. Defaults to 4000. */
  readonly timeoutMs?: number;
  /** Injectable effects; omitted members fall back to the real machine. */
  readonly probes?: Partial<ScanProbes>;
  /** Whether runnable workflows are published to the hub. Defaults to true. */
  readonly register?: boolean;
  /** Diagnostic sink. */
  readonly log?: (message: string) => void;
}

/** The default request budget; a scan should not stall the settings page. */
const DEFAULT_TIMEOUT_MS = 4_000;

/** How deep the workflow directory is walked. */
const MAX_DIRECTORY_DEPTH = 6;

/** How many workflow files one directory scan will read. */
const MAX_WORKFLOW_FILES = 200;

/** The route that lists a user's saved workflows. */
const WORKFLOW_LIST_QUERY = 'dir=workflows&recurse=true';

/** The real filesystem probes, used for any member the caller did not supply. */
const REAL_PROBES: ScanProbes = {
  fetchJson: async (url, timeoutMs) => {
    const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  },
  isDirectory: async (path) => {
    try {
      return (await stat(path)).isDirectory();
    } catch {
      return false;
    }
  },
  listDir: async (path) => {
    try {
      const entries = await readdir(path, { withFileTypes: true });
      return await Promise.all(
        entries.map(async (entry) => {
          if (entry.isDirectory()) return { name: entry.name, isDirectory: true };
          try {
            const info = await stat(join(path, entry.name));
            return { name: entry.name, isDirectory: false, sizeBytes: info.size };
          } catch {
            return { name: entry.name, isDirectory: false };
          }
        }),
      );
    } catch {
      return undefined;
    }
  },
  readFile: async (path) => await readFile(path, 'utf8'),
};

/**
 * Whether a value is a plain JSON object.
 * @param value - candidate value.
 * @returns true for a non-null, non-array object.
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Whether a host is an Ollama instance.
 *
 * Matched on the engine label rather than the host id, following the rule the
 * discoverers use: an operator may call the row anything, but `runtime.engine`
 * is the label the hub routes to.
 *
 * @param host - the declared host.
 * @returns true when this host speaks Ollama's own API.
 */
function isOllamaHost(host: ModelHost): boolean {
  return /ollama/i.test(host.runtime.engine);
}

/**
 * Strip a trailing slash from an endpoint.
 * @param endpoint - the base URL.
 * @returns the base URL without a trailing slash.
 */
function baseOf(endpoint: string): string {
  return endpoint.endsWith('/') ? endpoint.slice(0, -1) : endpoint;
}

/**
 * The display name of a workflow document.
 *
 * A workflow's own metadata is preferred when it has any — some exports carry a
 * `name`, and pipeline tools often add one — and the filename is the fallback,
 * because a workflow saved by the ComfyUI editor is named by its file.
 *
 * @param raw - the parsed document.
 * @param fallback - the name it was found under.
 * @returns the name to show.
 */
export function workflowDisplayName(raw: unknown, fallback: string): string {
  if (isObject(raw)) {
    for (const key of ['name', 'title', 'workflowName'] as const) {
      const value = raw[key];
      if (typeof value === 'string' && value.trim().length > 0) return value.trim();
    }
    for (const container of ['extra', 'metadata'] as const) {
      const nested = raw[container];
      if (isObject(nested) && typeof nested['name'] === 'string' && (nested['name'] as string).trim().length > 0) {
        return (nested['name'] as string).trim();
      }
    }
  }
  const last = fallback.split(/[/\\]/).pop() ?? fallback;
  return last.replace(/\.json$/i, '');
}

/**
 * Read every workflow document from the server's saved-workflow store.
 *
 * A single unreadable file costs that file, never the pass: each is fetched and
 * parsed independently so one truncated download cannot hide the other twenty.
 *
 * @param probes - the effects to use.
 * @param endpoint - the ComfyUI base URL.
 * @param timeoutMs - budget per request.
 * @returns the documents, or the reason the store could not be listed.
 */
async function readServerWorkflows(
  probes: ScanProbes,
  endpoint: string,
  timeoutMs: number,
): Promise<{ readonly documents: { name: string; raw: unknown; source: string }[]; readonly error?: string }> {
  const base = baseOf(endpoint);
  let listing: unknown;
  try {
    listing = await probes.fetchJson(`${base}/userdata?${WORKFLOW_LIST_QUERY}`, timeoutMs);
  } catch (error) {
    return { documents: [], error: `${base}/userdata did not answer: ${error instanceof Error ? error.message : String(error)}` };
  }

  const entries = Array.isArray(listing) ? listing : isObject(listing) && Array.isArray(listing['files']) ? (listing['files'] as unknown[]) : [];
  const documents: { name: string; raw: unknown; source: string }[] = [];
  for (const entry of entries) {
    const name =
      typeof entry === 'string'
        ? entry
        : isObject(entry) && typeof entry['path'] === 'string'
          ? (entry['path'] as string)
          : isObject(entry) && typeof entry['name'] === 'string'
            ? (entry['name'] as string)
            : undefined;
    if (name === undefined) continue;
    const cleaned = name.replace(/^\/+/, '').trim();
    if (cleaned.length === 0 || !cleaned.toLowerCase().endsWith('.json')) continue;
    const encoded = cleaned.split('/').map((segment) => encodeURIComponent(segment)).join('/');
    const url = `${base}/userdata/${encoded}`;
    try {
      documents.push({ name: cleaned, raw: await probes.fetchJson(url, timeoutMs), source: url });
    } catch (error) {
      documents.push({
        name: cleaned,
        raw: undefined,
        source: `${url} (${error instanceof Error ? error.message : String(error)})`,
      });
    }
  }
  return { documents };
}

/**
 * Every JSON file under a directory, walked breadth-first with a depth and count
 * cap so a huge tree cannot stall the settings page.
 *
 * @param probes - the effects to use.
 * @param root - the directory to walk.
 * @returns the files found, as absolute paths.
 */
async function listWorkflowFiles(
  probes: ScanProbes,
  root: string,
): Promise<{ readonly path: string; readonly sizeBytes?: number }[]> {
  const found: { path: string; sizeBytes?: number }[] = [];
  const queue: { path: string; depth: number }[] = [{ path: root, depth: 0 }];
  while (queue.length > 0 && found.length < MAX_WORKFLOW_FILES) {
    const current = queue.shift() as { path: string; depth: number };
    const entries = (await probes.listDir(current.path)) ?? [];
    for (const entry of entries) {
      if (found.length >= MAX_WORKFLOW_FILES) break;
      const child = join(current.path, entry.name);
      if (entry.isDirectory) {
        if (current.depth < MAX_DIRECTORY_DEPTH) queue.push({ path: child, depth: current.depth + 1 });
        continue;
      }
      if (!/\.json$/i.test(entry.name)) continue;
      found.push({ path: child, ...(entry.sizeBytes === undefined ? {} : { sizeBytes: entry.sizeBytes }) });
    }
  }
  return found;
}

/**
 * Read every workflow document from a configured directory.
 *
 * @param probes - the effects to use.
 * @param root - the directory.
 * @returns the documents, or the reason the directory could not be read.
 */
async function readDirectoryWorkflows(
  probes: ScanProbes,
  root: string,
): Promise<{ readonly documents: { name: string; raw: unknown; source: string; sizeBytes?: number }[]; readonly error?: string }> {
  if (!(await probes.isDirectory(root))) {
    return { documents: [], error: `the configured workflow directory ${root} is not a directory` };
  }
  const readText = probes.readFile ?? REAL_PROBES.readFile;
  const files = await listWorkflowFiles(probes, root);
  const documents: { name: string; raw: unknown; source: string; sizeBytes?: number }[] = [];
  for (const file of files) {
    try {
      const text = await readText!(file.path);
      documents.push({
        name: file.path,
        raw: JSON.parse(text),
        source: file.path,
        ...(file.sizeBytes === undefined ? {} : { sizeBytes: file.sizeBytes }),
      });
    } catch (error) {
      // A malformed file is carried through as an unparseable document so the scan
      // can report it as `invalid` with the parser's own reason, rather than
      // omitting it — a workflow the operator can see but the hub cannot read is
      // exactly the thing this list exists to surface.
      documents.push({
        name: file.path,
        raw: undefined,
        source: `${file.path} (${error instanceof Error ? error.message : String(error)})`,
        ...(file.sizeBytes === undefined ? {} : { sizeBytes: file.sizeBytes }),
      });
    }
  }
  return { documents };
}

/**
 * Describe one Ollama model for the list.
 *
 * @param host - the host it was read from.
 * @param endpoint - the endpoint it was read from.
 * @param entry - the `/api/tags` entry.
 * @param registeredId - the catalog id already serving it, when any.
 * @returns the resource.
 */
function ollamaResource(
  host: ModelHost,
  endpoint: string,
  entry: unknown,
  registeredId: string | undefined,
): LocalResource | undefined {
  if (!isObject(entry)) return undefined;
  const name = typeof entry['name'] === 'string' ? entry['name'] : undefined;
  if (name === undefined || name.trim().length === 0) return undefined;

  const details = isObject(entry['details']) ? entry['details'] : {};
  const parts = [details['parameter_size'], details['quantization_level'], details['family']]
    .filter((part): part is string => typeof part === 'string' && part.length > 0);
  const sizeBytes = typeof entry['size'] === 'number' ? entry['size'] : undefined;
  const id = scannedOllamaModelId(host.id, name);

  return {
    id,
    kind: 'ollama_model',
    typeLabel: 'Ollama Model',
    name,
    source: `${baseOf(endpoint)}/api/tags`,
    status: 'ready',
    detail: parts.length === 0 ? 'installed' : `installed · ${parts.join(' · ')}`,
    runnable: true,
    modelId: registeredId ?? id,
    ...(sizeBytes === undefined ? {} : { sizeBytes }),
  };
}

/**
 * Scan the Ollama instances the catalog declares.
 *
 * @param hub - the live hub, read for already-registered model ids.
 * @param hosts - the declared hosts.
 * @param probes - the effects to use.
 * @param timeoutMs - budget per request.
 * @returns the resources and the per-host report.
 */
async function scanOllama(
  hub: ModelHub,
  hosts: readonly ModelHost[],
  probes: ScanProbes,
  timeoutMs: number,
): Promise<{ resources: LocalResource[]; sources: ScanSourceReport[] }> {
  const resources: LocalResource[] = [];
  const sources: ScanSourceReport[] = [];

  for (const host of hosts.filter(isOllamaHost)) {
    const endpoint = host.runtime.endpoint;
    if (endpoint === undefined || endpoint.trim().length === 0) {
      sources.push({
        id: host.id,
        kind: 'ollama_model',
        label: host.name,
        ok: false,
        detail: 'no runtime.endpoint configured, so its models cannot be listed',
        found: 0,
      });
      continue;
    }
    try {
      const body = await probes.fetchJson(`${baseOf(endpoint)}/api/tags`, timeoutMs);
      const list = isObject(body) && Array.isArray(body['models']) ? (body['models'] as unknown[]) : [];
      const registered = new Map<string, string>();
      for (const model of hub.catalog.listModels()) {
        if (model.hostId !== host.id) continue;
        const tag = model.adapterConfig['model'];
        if (typeof tag === 'string') registered.set(tag, model.id);
      }
      let found = 0;
      for (const entry of list) {
        const tag = isObject(entry) && typeof entry['name'] === 'string' ? entry['name'] : undefined;
        const resource = ollamaResource(host, endpoint, entry, tag === undefined ? undefined : registered.get(tag));
        if (resource === undefined) continue;
        resources.push(resource);
        found += 1;
      }
      sources.push({
        id: host.id,
        kind: 'ollama_model',
        label: host.name,
        ok: true,
        detail: `read ${found} installed model(s) from ${baseOf(endpoint)}/api/tags`,
        found,
      });
    } catch (error) {
      // Contained: this host contributes nothing, and the ComfyUI half still runs.
      sources.push({
        id: host.id,
        kind: 'ollama_model',
        label: host.name,
        ok: false,
        detail: `${host.name} did not answer at ${baseOf(endpoint)}/api/tags: ${error instanceof Error ? error.message : String(error)}`,
        found: 0,
      });
    }
  }
  return { resources, sources };
}

/**
 * Scan the ComfyUI workflows the server has saved and the configured directory
 * holds, and publish the runnable ones as providers.
 *
 * @param hosts - the declared hosts.
 * @param options - the workflow directory, probes, budget, and registration flag.
 * @returns the resources, the per-source report, the warnings, and the descriptors.
 */
async function scanComfyUi(
  hosts: readonly ModelHost[],
  options: {
    readonly workflowDir?: string;
    readonly probes: ScanProbes;
    readonly timeoutMs: number;
    readonly log: (message: string) => void;
  },
): Promise<{
  resources: LocalResource[];
  sources: ScanSourceReport[];
  warnings: string[];
  descriptors: ModelDescriptor[];
  hostsById: Map<string, ModelHost>;
}> {
  const resources: LocalResource[] = [];
  const sources: ScanSourceReport[] = [];
  const warnings: string[] = [];
  const descriptors: ModelDescriptor[] = [];
  const hostsById = new Map<string, ModelHost>();
  const seen = new Set<string>();

  const comfyHosts = hosts.filter((host) => host.adapter === 'comfyui');

  // The node index of the first host that answered, reused for the configured
  // directory: a workflow file beside the catalog belongs to the same engine, so
  // it is interpreted with the same node classes the engine reported.
  let sharedIo: ComfyNodeIndex | undefined;
  let sharedClasses: ReadonlySet<string> | undefined;

  for (const host of comfyHosts) {
    hostsById.set(host.id, host);
    const endpoint = host.runtime.endpoint;
    let io: ComfyNodeIndex | undefined;
    let classes: ReadonlySet<string> | undefined;
    let serverReachable = false;

    if (endpoint !== undefined && endpoint.trim().length > 0) {
      const base = baseOf(endpoint);
      try {
        const info = await options.probes.fetchJson(`${base}/object_info`, options.timeoutMs);
        io = readComfyNodeIo(info);
        classes = new Set(isObject(info) ? Object.keys(info) : []);
        serverReachable = true;
        if (sharedIo === undefined) {
          sharedIo = io;
          sharedClasses = classes;
        }
      } catch (error) {
        warnings.push(
          `${host.name} did not answer /object_info: ${error instanceof Error ? error.message : String(error)}. ` +
            'Workflows from a configured directory are still listed; editor-format ones cannot be converted without it.',
        );
      }

      const listed = await readServerWorkflows(options.probes, endpoint, options.timeoutMs);
      if (listed.error !== undefined) {
        sources.push({
          id: host.id,
          kind: 'comfyui_workflow',
          label: host.name,
          ok: false,
          detail: listed.error,
          found: 0,
        });
      }
      let found = 0;
      for (const document of listed.documents) {
        const resource = workflowResource(host, document, io, classes, options);
        if (resource === undefined) continue;
        if (seen.has(resource.resource.id)) continue;
        seen.add(resource.resource.id);
        resources.push(resource.resource);
        if (resource.descriptor !== undefined) descriptors.push(resource.descriptor);
        found += 1;
      }
      if (listed.error === undefined) {
        sources.push({
          id: host.id,
          kind: 'comfyui_workflow',
          label: host.name,
          ok: true,
          detail: `read ${found} saved workflow(s) from ${base}/userdata`,
          found,
        });
      }
      options.log(
        `${host.name}: ${serverReachable ? 'reachable' : 'unreachable'}, ${found} saved workflow(s)`,
      );
    } else {
      sources.push({
        id: host.id,
        kind: 'comfyui_workflow',
        label: host.name,
        ok: false,
        detail: 'no runtime.endpoint configured, so its saved workflows cannot be listed',
        found: 0,
      });
    }
  }

  // The configured directory is scanned once, and its workflows are attributed to
  // the first ComfyUI host — the engine that will run them.
  const workflowDir = options.workflowDir;
  const owner = comfyHosts[0];
  if (workflowDir !== undefined && workflowDir.trim().length > 0) {
    if (owner === undefined) {
      warnings.push(
        `a workflow directory is configured (${workflowDir}) but no ComfyUI host is declared, so nothing can run them.`,
      );
    } else {
      const read = await readDirectoryWorkflows(options.probes, workflowDir);
      if (read.error !== undefined) {
        sources.push({
          id: 'workflow-dir',
          kind: 'comfyui_workflow',
          label: workflowDir,
          ok: false,
          detail: read.error,
          found: 0,
        });
      }
      let found = 0;
      for (const document of read.documents) {
        const resource = workflowResource(owner, document, sharedIo, sharedClasses, options);
        if (resource === undefined) continue;
        if (seen.has(resource.resource.id)) continue;
        seen.add(resource.resource.id);
        resources.push(resource.resource);
        if (resource.descriptor !== undefined) descriptors.push(resource.descriptor);
        found += 1;
      }
      if (read.error === undefined) {
        sources.push({
          id: 'workflow-dir',
          kind: 'comfyui_workflow',
          label: workflowDir,
          ok: true,
          detail: `read ${found} workflow file(s) from ${workflowDir}`,
          found,
        });
      }
    }
  }

  return { resources, sources, warnings, descriptors, hostsById };
}

/**
 * Turn one workflow document into a list row, and into a provider when it runs.
 *
 * @param host - the ComfyUI host that will run it.
 * @param document - the document, its name, and where it came from.
 * @param io - the node index, when the engine was reachable.
 * @param classes - the installed node classes, when the engine was reachable.
 * @param options - budget and diagnostics.
 * @returns the row and provider, or `undefined` when the document has no name.
 */
function workflowResource(
  host: ModelHost,
  document: { readonly name: string; readonly raw: unknown; readonly source: string; readonly sizeBytes?: number },
  io: ComfyNodeIndex | undefined,
  classes: ReadonlySet<string> | undefined,
  options: { readonly log: (message: string) => void },
): { resource: LocalResource; descriptor?: ModelDescriptor } | undefined {
  if (document.name.trim().length === 0) return undefined;

  const displayName = workflowDisplayName(document.raw, document.name);
  const id = scannedWorkflowId(displayName, document.source);
  const scan = scanWorkflowDocument({
    raw: document.raw,
    name: document.name,
    ...(io === undefined ? {} : { io }),
    ...(classes === undefined ? {} : { classes }),
  });

  const resource: LocalResource = {
    id,
    kind: 'comfyui_workflow',
    typeLabel: 'ComfyUI Workflow',
    name: displayName,
    source: document.source,
    status: scan.readiness,
    detail: scan.detail,
    runnable: scan.runnable,
    format: scan.format,
    capabilities: [...scan.capabilities],
    inputs: [...scan.inputs],
    outputs: [...scan.outputs],
    ...(document.sizeBytes === undefined ? {} : { sizeBytes: document.sizeBytes }),
    ...(scan.runnable ? { modelId: id } : {}),
  };

  if (!scan.runnable || scan.contract === undefined || scan.graph === undefined) {
    options.log(`workflow ${displayName} is ${scan.readiness}: ${scan.detail}`);
    return { resource };
  }

  const { inputTypes, outputTypes } = ioForCapabilities(scan.capabilities);
  const descriptor: ModelDescriptor = {
    id,
    name: displayName,
    type: workflowModelTypeFor(scan.capabilities),
    host: host.id,
    providerKind: 'workflow',
    workflowId: id,
    capabilities: scan.capabilities,
    inputTypes,
    outputTypes,
    adapterConfig: {
      workflow: scan.graph,
      bindings: scan.contract.bindings,
      outputs: scan.contract.outputs,
    },
    resources: { vramGb: 0, ramGb: 0 },
    // Below every static entry, so a hand-declared provider always wins routing
    // against a machine that happens to have a similar workflow saved.
    priority: DISCOVERED_PRIORITY,
    tags: ['local', 'discovered', 'comfyui', 'workflow', 'scanned'],
    enabled: true,
    notes: `Scanned from ${document.source}. Its checkpoint, LoRA and VAE choices are part of the workflow and are not exposed.`,
  };
  return { resource, descriptor };
}

/**
 * Scan this machine for the local AI resources the settings page lists.
 *
 * @param hub - the live hub the runnable workflows are published to.
 * @param options - the declared hosts, the workflow directory, and the effects.
 * @returns the resources, the per-source report, and the warnings.
 */
export async function scanLocalResources(hub: ModelHub, options: ScanOptions): Promise<LocalScanResult> {
  const probes: ScanProbes = { ...REAL_PROBES, ...(options.probes ?? {}) };
  const timeoutMs = Math.max(1, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const log = options.log ?? ((): void => {});
  const warnings: string[] = [];

  // The two halves never share a failure path: each is awaited on its own, and
  // each contains its own errors, so one engine being down can only cost its own
  // rows.
  const ollama = await scanOllama(hub, options.hosts, probes, timeoutMs);
  const comfy = await scanComfyUi(options.hosts, {
    ...(options.workflowDir === undefined ? {} : { workflowDir: options.workflowDir }),
    probes,
    timeoutMs,
    log,
  });

  warnings.push(...comfy.warnings);

  let registered = 0;
  if (options.register !== false) {
    registered = hub.publishScannedModels(comfy.descriptors);
    const runnable = comfy.resources.filter((resource) => resource.runnable).length;
    log(`scan: published ${comfy.descriptors.length} runnable workflow(s) of ${runnable} listed`);
    if (comfy.descriptors.length !== runnable) {
      warnings.push(
        `${runnable - comfy.descriptors.length} workflow(s) could not be published because their id collides with a configured model.`,
      );
    }
  }

  return {
    resources: [...ollama.resources, ...comfy.resources],
    sources: [...ollama.sources, ...comfy.sources],
    warnings,
    registered,
    generatedAt: new Date().toISOString(),
  };
}
