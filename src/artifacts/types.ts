/**
 * The artifact system.
 *
 * An artifact is the *only* thing that crosses a model boundary. Models never
 * hand each other raw bytes, file handles, or engine-specific objects — they
 * hand each other artifact references. That is what makes `text → image → 3D`
 * composable without the router knowing anything about PNG, GLB, or latent space.
 *
 * An artifact is deliberately small, JSON-serializable, and durably addressable:
 * it records *where* content lives plus the metadata a downstream model needs to
 * reason about it (dimensions, duration, format, provenance) without loading it.
 *
 * @module dsh-ai-model-hub/artifacts/types
 */

import type { IoType } from '../catalog/capabilities.ts';

/**
 * A durable reference to one piece of content produced or consumed by a model.
 *
 * `uri` is a plain string so the hub is not tied to one storage backend: the
 * shipped backend uses `file://` URIs on the local filesystem, and an
 * HTTP-backed or object-store backend would use `https://` / `s3://` without
 * changing this type or any consumer of it.
 */
export interface Artifact {
  /** Unique, stable id for this artifact within the hub's store. */
  readonly id: string;
  /** Which kind of content this is; must match a {@link IoType}. */
  readonly type: IoType;
  /**
   * Where the content lives. `file://…` for the local store. A `data:` URI is
   * used only for small inline values that never touch disk.
   */
  readonly uri: string;
  /** MIME type when known, e.g. `image/png`, `model/gltf-binary`. */
  readonly mimeType?: string;
  /** Size in bytes when known. */
  readonly byteLength?: number;
  /** Human-facing label, useful in UI and in artifact listings. */
  readonly label?: string;
  /** Unix epoch milliseconds when the content was produced. */
  readonly createdAt: number;
  /** Id of the model that produced this artifact, when it came from a model. */
  readonly producerModelId?: string;
  /**
   * Kind-specific, losslessly JSON-serializable facts about the content:
   * `{ width, height }` for images, `{ durationSeconds, sampleRate }` for audio,
   * `{ vertexCount, format }` for meshes. Downstream models and the agent read
   * these instead of opening the file.
   */
  readonly metadata: Readonly<Record<string, unknown>>;
}

/** A request to persist new content and obtain its artifact reference. */
export interface ArtifactWriteRequest {
  /** The content kind. */
  readonly type: IoType;
  /** Raw bytes, for binary kinds. */
  readonly bytes?: Uint8Array;
  /** Inline text, for the `text` and `json` kinds. */
  readonly text?: string;
  /** MIME type when the producer knows it. */
  readonly mimeType?: string;
  /** Human-facing label. */
  readonly label?: string;
  /** Id of the producing model. */
  readonly producerModelId?: string;
  /** Caller-supplied extra facts merged into the artifact's metadata. */
  readonly metadata?: Readonly<Record<string, unknown>>;
  /**
   * Preferred file extension including the dot, e.g. `.png`. The store falls
   * back to the MIME type, then to the content kind, when this is omitted.
   */
  readonly extension?: string;
}

/**
 * Where artifacts live.
 *
 * Consumers depend on this interface, never on a concrete store, so a
 * remote/object-store backend can replace the local one without touching the
 * router, adapters, or the DSH plugin.
 */
export interface ArtifactStore {
  /**
   * Persist one piece of content and return its durable reference.
   * @param request - content plus descriptive metadata.
   * @returns the artifact reference for the stored content.
   */
  put(request: ArtifactWriteRequest): Promise<Artifact>;

  /**
   * Resolve a reference previously returned by {@link put}.
   * @param id - the artifact id.
   * @returns the artifact, or `undefined` when no such artifact is stored.
   */
  get(id: string): Promise<Artifact | undefined>;

  /**
   * Read the content behind a reference.
   * @param id - the artifact id.
   * @returns the bytes and the resolved artifact.
   */
  read(id: string): Promise<{ artifact: Artifact; bytes: Uint8Array }>;

  /**
   * Resolve an artifact to a path another local process can open.
   *
   * This is the seam a model runtime uses: an image generator needs a *path* to
   * its input image, not a Uint8Array. Stores that cannot expose a path (a
   * remote object store) materialize a temporary file instead.
   *
   * @param id - the artifact id.
   * @returns an absolute filesystem path plus the artifact it represents.
   */
  resolvePath(id: string): Promise<{ artifact: Artifact; path: string }>;

  /**
   * List artifacts, most recent first.
   * @param limit - maximum number to return.
   * @returns artifact references, newest first.
   */
  list(limit?: number): Promise<Artifact[]>;
}

/**
 * Convenience projection: the metadata keys the hub understands by convention.
 *
 * These are read out of {@link Artifact.metadata} for routing decisions and
 * prompt rendering. They are advisory — a manifest's `inputTypes` is what
 * actually gates compatibility.
 */
export interface ArtifactConventions {
  /** Image pixel dimensions. */
  readonly width?: number;
  /** Image pixel dimensions. */
  readonly height?: number;
  /** Audio/video duration. */
  readonly durationSeconds?: number;
  /** Mesh vertex count. */
  readonly vertexCount?: number;
  /** Container/codec format, e.g. `glb`, `wav`, `png`. */
  readonly format?: string;
}

/**
 * Read the conventional numeric metadata fields off an artifact.
 * @param artifact - the artifact to inspect.
 * @returns whichever conventional fields are present and well-typed.
 */
export function readArtifactConventions(artifact: Artifact): ArtifactConventions {
  const meta = artifact.metadata;
  const result: {
    width?: number;
    height?: number;
    durationSeconds?: number;
    vertexCount?: number;
    format?: string;
  } = {};
  const numberAt = (key: string): number | undefined => {
    const value = meta[key];
    return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
  };
  const stringAt = (key: string): string | undefined => {
    const value = meta[key];
    return typeof value === 'string' ? value : undefined;
  };
  const width = numberAt('width');
  const height = numberAt('height');
  const durationSeconds = numberAt('durationSeconds');
  const vertexCount = numberAt('vertexCount');
  const format = stringAt('format');
  if (width !== undefined) result.width = width;
  if (height !== undefined) result.height = height;
  if (durationSeconds !== undefined) result.durationSeconds = durationSeconds;
  if (vertexCount !== undefined) result.vertexCount = vertexCount;
  if (format !== undefined) result.format = format;
  return result;
}

/**
 * Render one artifact as a single-line human/model-readable handle.
 *
 * This is the canonical way an artifact is described in tool output, so the
 * model sees a stable, compact summary it can reference in a later capability
 * call instead of having file contents pasted into its context.
 *
 * @param artifact - the artifact to describe.
 * @returns a compact handle such as `image artifact_ab12 (1024x1024, image/png, 812 KiB)`.
 */
export function describeArtifact(artifact: Artifact): string {
  const parts: string[] = [];
  const convention = readArtifactConventions(artifact);
  if (convention.width !== undefined && convention.height !== undefined) {
    parts.push(`${convention.width}x${convention.height}`);
  }
  if (convention.durationSeconds !== undefined) {
    parts.push(`${convention.durationSeconds}s`);
  }
  if (convention.vertexCount !== undefined) {
    parts.push(`${convention.vertexCount} verts`);
  }
  if (artifact.mimeType !== undefined) parts.push(artifact.mimeType);
  if (artifact.byteLength !== undefined) parts.push(formatBytes(artifact.byteLength));
  const suffix = parts.length > 0 ? ` (${parts.join(', ')})` : '';
  const label = artifact.label !== undefined ? ` "${artifact.label}"` : '';
  return `${artifact.type} ${artifact.id}${label}${suffix}`;
}

/**
 * Format a byte count for human display.
 * @param bytes - the count.
 * @returns a compact string such as `812 KiB`.
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KiB', 'MiB', 'GiB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit] ?? 'KiB'}`;
}
