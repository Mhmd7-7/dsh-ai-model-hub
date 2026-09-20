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
/**
 * Read the conventional numeric metadata fields off an artifact.
 * @param artifact - the artifact to inspect.
 * @returns whichever conventional fields are present and well-typed.
 */
export function readArtifactConventions(artifact) {
    const meta = artifact.metadata;
    const result = {};
    const numberAt = (key) => {
        const value = meta[key];
        return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
    };
    const stringAt = (key) => {
        const value = meta[key];
        return typeof value === 'string' ? value : undefined;
    };
    const width = numberAt('width');
    const height = numberAt('height');
    const durationSeconds = numberAt('durationSeconds');
    const vertexCount = numberAt('vertexCount');
    const format = stringAt('format');
    if (width !== undefined)
        result.width = width;
    if (height !== undefined)
        result.height = height;
    if (durationSeconds !== undefined)
        result.durationSeconds = durationSeconds;
    if (vertexCount !== undefined)
        result.vertexCount = vertexCount;
    if (format !== undefined)
        result.format = format;
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
export function describeArtifact(artifact) {
    const parts = [];
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
    if (artifact.mimeType !== undefined)
        parts.push(artifact.mimeType);
    if (artifact.byteLength !== undefined)
        parts.push(formatBytes(artifact.byteLength));
    const suffix = parts.length > 0 ? ` (${parts.join(', ')})` : '';
    const label = artifact.label !== undefined ? ` "${artifact.label}"` : '';
    return `${artifact.type} ${artifact.id}${label}${suffix}`;
}
/**
 * Format a byte count for human display.
 * @param bytes - the count.
 * @returns a compact string such as `812 KiB`.
 */
export function formatBytes(bytes) {
    if (bytes < 1024)
        return `${bytes} B`;
    const units = ['KiB', 'MiB', 'GiB'];
    let value = bytes / 1024;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
    }
    return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit] ?? 'KiB'}`;
}
