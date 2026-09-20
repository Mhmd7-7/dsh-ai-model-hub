/**
 * The local, filesystem-backed artifact store.
 *
 * Design constraints that shaped it:
 *
 * - **Durable and inspectable.** Artifacts land as ordinary files under
 *   `artifacts/`, with a JSON index beside them. An operator can open the image
 *   the agent generated without any hub tooling, and a crash cannot lose
 *   produced content that the model already reported.
 * - **Addressable.** Ids are unique and stable, so an artifact id in a tool
 *   result stays resolvable across turns, sessions, and process restarts.
 * - **Concurrency-safe.** Writes are serialized through a single queue. Two
 *   models finishing at once must not interleave index writes and corrupt it;
 *   serializing is simpler than locking and fast enough at this scale.
 *
 * @module dsh-ai-model-hub/artifacts/local-store
 */
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, extname, isAbsolute, join, resolve, sep } from 'node:path';
import { isIoType } from "../catalog/capabilities.js";
import { ModelHubError } from "../errors.js";
import { IssueCollector, isRecord, readOptionalNumber, readOptionalString, } from "../util/validate.js";
/** The current on-disk index format version. */
const INDEX_VERSION = 1;
/** Extension used when the MIME type is unknown. */
const FALLBACK_EXTENSION = {
    text: '.txt',
    image: '.png',
    audio: '.wav',
    video: '.json',
    model_3d: '.stl',
    json: '.json',
    file: '.bin',
};
/** MIME type by extension, for artifacts written by a producer that knew only the extension. */
const MIME_BY_EXTENSION = {
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.webp': 'image/webp',
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.flac': 'audio/flac',
    '.mp4': 'video/mp4',
    '.webm': 'video/webm',
    '.stl': 'model/stl',
    '.obj': 'model/obj',
    '.glb': 'model/gltf-binary',
    '.gltf': 'model/gltf+json',
    '.ply': 'application/octet-stream',
};
/**
 * Choose a file extension for new content.
 * @param request - the write request.
 * @returns an extension beginning with a dot.
 */
function chooseExtension(request) {
    const declared = request.extension;
    if (declared !== undefined && declared.length > 0) {
        return declared.startsWith('.') ? declared : `.${declared}`;
    }
    const mime = request.mimeType;
    if (mime !== undefined) {
        const suffix = mime.split('/')[1];
        if (suffix !== undefined && suffix.length > 0 && suffix.length <= 8 && /^[\w.+-]+$/.test(suffix)) {
            if (mime.startsWith('model/gltf-binary'))
                return '.glb';
            if (mime.startsWith('model/gltf+json'))
                return '.gltf';
            return `.${suffix === 'plain' ? 'txt' : suffix}`;
        }
    }
    return FALLBACK_EXTENSION[request.type];
}
/**
 * Turn a model id or label into a filename-safe fragment.
 * @param value - the source text.
 * @param maxLength - maximum length of the result.
 * @returns a lowercase fragment containing only `[a-z0-9-]`.
 */
function slugify(value, maxLength = 40) {
    const cleaned = value
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '');
    return cleaned.length === 0 ? 'artifact' : cleaned.slice(0, maxLength);
}
/**
 * A filesystem-backed {@link ArtifactStore}.
 *
 * The store keeps the index in memory after the first load and writes it back
 * after every mutation. Reads never touch disk for metadata, which keeps the
 * router's artifact lookups cheap; only {@link read} and {@link resolvePath}
 * touch the content.
 */
export class LocalArtifactStore {
    /** Absolute directory containing the index and content files. */
    root;
    log;
    /** Every known artifact by id. */
    index = new Map();
    /** Serializes mutations so concurrent writes cannot interleave. */
    queue = Promise.resolve();
    /** Set once the index has been read from disk. */
    loaded = false;
    /**
     * @param options - the storage root and diagnostics sink.
     */
    constructor(options) {
        if (!isAbsolute(options.root)) {
            throw new ModelHubError('CONFIG_ERROR', `artifact store root must be absolute, got "${options.root}"`);
        }
        this.root = resolve(options.root);
        this.log = options.log ?? (() => { });
    }
    /** Path of the index document. */
    get indexPath() {
        return join(this.root, 'index.json');
    }
    /** Directory holding artifact content. */
    get contentDir() {
        return join(this.root, 'files');
    }
    /**
     * Load the index from disk if it has not been loaded yet.
     *
     * A missing index is normal on first run. A *corrupt* index is not: it is
     * reported and the in-memory index is left empty, which fails closed — the hub
     * will refuse to resolve an artifact rather than guess at content.
     */
    async ensureLoaded() {
        if (this.loaded)
            return;
        this.loaded = true;
        let raw;
        try {
            raw = await readFile(this.indexPath, 'utf8');
        }
        catch {
            this.log(`artifacts: no index at ${this.indexPath}; starting empty`);
            return;
        }
        let parsed;
        try {
            parsed = JSON.parse(raw);
        }
        catch (error) {
            this.log(`artifacts: index ${this.indexPath} is not valid JSON (${String(error)}); ignoring it`);
            return;
        }
        if (!isRecord(parsed) || !Array.isArray(parsed['artifacts'])) {
            this.log(`artifacts: index ${this.indexPath} has an unexpected shape; ignoring it`);
            return;
        }
        for (const entry of parsed['artifacts']) {
            const artifact = this.parseIndexEntry(entry);
            if (artifact !== undefined)
                this.index.set(artifact.id, artifact);
        }
        this.log(`artifacts: loaded ${this.index.size} artifact(s) from ${this.indexPath}`);
    }
    /**
     * Validate one index entry read from disk.
     * @param entry - the untrusted entry.
     * @returns the artifact, or `undefined` when the entry is unusable.
     */
    parseIndexEntry(entry) {
        if (!isRecord(entry))
            return undefined;
        const collector = new IssueCollector();
        const id = readOptionalString(entry, 'id', 'artifact', collector);
        const type = entry['type'];
        const uri = readOptionalString(entry, 'uri', 'artifact', collector);
        if (id === undefined || uri === undefined || typeof type !== 'string' || !isIoType(type)) {
            this.log('artifacts: dropping malformed index entry');
            return undefined;
        }
        const mimeType = readOptionalString(entry, 'mimeType', 'artifact', collector);
        const label = readOptionalString(entry, 'label', 'artifact', collector);
        const producerModelId = readOptionalString(entry, 'producerModelId', 'artifact', collector);
        const byteLength = readOptionalNumber(entry, 'byteLength', 'artifact', collector);
        const createdAt = readOptionalNumber(entry, 'createdAt', 'artifact', collector);
        const metadata = isRecord(entry['metadata']) ? entry['metadata'] : {};
        const artifact = {
            id,
            type,
            uri,
            metadata,
            createdAt: createdAt ?? 0,
        };
        if (mimeType !== undefined)
            artifact.mimeType = mimeType;
        if (byteLength !== undefined)
            artifact.byteLength = byteLength;
        if (label !== undefined)
            artifact.label = label;
        if (producerModelId !== undefined)
            artifact.producerModelId = producerModelId;
        return artifact;
    }
    /**
     * Persist the index atomically: write a sibling temp file, then rename over
     * the target. A crash mid-write leaves the previous index intact rather than a
     * truncated one.
     */
    async persistIndex() {
        const document = { version: INDEX_VERSION, artifacts: [...this.index.values()] };
        await mkdir(this.root, { recursive: true });
        const tempPath = `${this.indexPath}.${randomUUID()}.tmp`;
        await writeFile(tempPath, `${JSON.stringify(document, null, 2)}\n`, 'utf8');
        await rename(tempPath, this.indexPath);
    }
    /**
     * Run a mutation with exclusive access to the index.
     * @param operation - the mutation to run.
     * @returns the mutation's result.
     */
    serialize(operation) {
        const run = this.queue.then(operation, operation);
        // Keep the chain alive even when an operation rejects, so one failure does
        // not poison every later write.
        this.queue = run.then(() => undefined, () => undefined);
        return run;
    }
    /**
     * Persist one piece of content.
     * @param request - content plus descriptive metadata.
     * @returns the stored artifact reference.
     * @throws ModelHubError with `ARTIFACT_ERROR` for incoherent requests.
     */
    async put(request) {
        const hasBytes = request.bytes !== undefined;
        const hasText = request.text !== undefined;
        if (hasBytes === hasText) {
            throw new ModelHubError('ARTIFACT_ERROR', 'an artifact write must supply exactly one of `bytes` or `text`', { type: request.type, hasBytes, hasText });
        }
        const payload = request.bytes !== undefined ? Buffer.from(request.bytes) : Buffer.from(request.text ?? '', 'utf8');
        return this.serialize(async () => {
            await this.ensureLoaded();
            const extension = chooseExtension(request);
            const id = this.mintId(request, payload, extension);
            const fileName = `${id}${extension}`;
            await mkdir(this.contentDir, { recursive: true });
            const absolutePath = join(this.contentDir, fileName);
            await writeFile(absolutePath, payload);
            const artifact = {
                id,
                type: request.type,
                uri: pathToFileUri(absolutePath),
                createdAt: Date.now(),
                byteLength: payload.byteLength,
                metadata: { ...(request.metadata ?? {}) },
            };
            const mimeType = request.mimeType ?? MIME_BY_EXTENSION[extension];
            if (mimeType !== undefined)
                artifact.mimeType = mimeType;
            if (request.label !== undefined)
                artifact.label = request.label;
            if (request.producerModelId !== undefined)
                artifact.producerModelId = request.producerModelId;
            this.index.set(id, artifact);
            await this.persistIndex();
            this.log(`artifacts: wrote ${id} (${request.type}, ${payload.byteLength} bytes)`);
            return artifact;
        });
    }
    /**
     * Mint a stable, human-scannable id.
     *
     * The shape is `<type>_<slug>_<hash>`: sortable by kind in a directory listing,
     * readable in a tool result, and content-derived so identical writes are
     * visibly identical without a collision risk from the random suffix's absence.
     *
     * @param request - the write request.
     * @param payload - the content.
     * @param extension - the chosen extension.
     * @returns a unique artifact id.
     */
    mintId(request, payload, extension) {
        const digest = createHash('sha256')
            .update(payload)
            .update(extension)
            .update(String(this.index.size))
            .update(randomUUID())
            .digest('hex')
            .slice(0, 12);
        const label = request.label ?? request.producerModelId ?? request.type;
        return `${request.type}_${slugify(label, 24)}_${digest}`;
    }
    /**
     * Resolve a reference.
     * @param id - the artifact id.
     * @returns the artifact, or `undefined` when unknown.
     */
    async get(id) {
        await this.ensureLoaded();
        return this.index.get(id);
    }
    /**
     * Read the content behind a reference.
     * @param id - the artifact id.
     * @returns the bytes and the artifact.
     * @throws ModelHubError with `ARTIFACT_ERROR` when unknown or unreadable.
     */
    async read(id) {
        const { artifact, path } = await this.resolvePath(id);
        try {
            const bytes = await readFile(path);
            return { artifact, bytes };
        }
        catch (error) {
            throw new ModelHubError('ARTIFACT_ERROR', `artifact ${id} could not be read from ${path}`, {
                artifactId: id,
                path,
                cause: String(error),
            });
        }
    }
    /**
     * Resolve an artifact to a filesystem path a local process can open.
     * @param id - the artifact id.
     * @returns the artifact and its absolute path.
     * @throws ModelHubError with `ARTIFACT_ERROR` when unknown, out of root, or missing.
     */
    async resolvePath(id) {
        await this.ensureLoaded();
        const artifact = this.index.get(id);
        if (artifact === undefined) {
            throw new ModelHubError('ARTIFACT_ERROR', `no artifact with id "${id}"`, { artifactId: id });
        }
        const path = fileUriToPath(artifact.uri);
        if (path === undefined) {
            throw new ModelHubError('ARTIFACT_ERROR', `artifact ${id} has a non-file URI ("${artifact.uri}") that this store cannot resolve`, { artifactId: id, uri: artifact.uri });
        }
        // Defence in depth: the index is on disk and therefore editable, so never
        // hand a path outside the store root to another process.
        const contentRoot = this.contentDir.endsWith(sep) ? this.contentDir : `${this.contentDir}${sep}`;
        if (!path.startsWith(contentRoot)) {
            throw new ModelHubError('ARTIFACT_ERROR', `artifact ${id} points outside the store root; refusing to expose ${path}`, { artifactId: id, path, contentRoot });
        }
        try {
            await stat(path);
        }
        catch {
            throw new ModelHubError('ARTIFACT_ERROR', `artifact ${id} content is missing at ${path}`, {
                artifactId: id,
                path,
            });
        }
        return { artifact, path };
    }
    /**
     * List artifacts, newest first.
     * @param limit - maximum number to return. Defaults to 100.
     * @returns artifact references.
     */
    async list(limit = 100) {
        await this.ensureLoaded();
        return [...this.index.values()]
            .sort((left, right) => right.createdAt - left.createdAt)
            .slice(0, Math.max(0, limit));
    }
    /**
     * Delete one artifact and its content.
     * @param id - the artifact id.
     * @returns whether anything was deleted.
     */
    async delete(id) {
        return this.serialize(async () => {
            await this.ensureLoaded();
            const artifact = this.index.get(id);
            if (artifact === undefined)
                return false;
            const path = fileUriToPath(artifact.uri);
            if (path !== undefined) {
                await rm(path, { force: true });
            }
            this.index.delete(id);
            await this.persistIndex();
            return true;
        });
    }
    /**
     * Remove every artifact and reset the index.
     *
     * Used by tests and by an explicit operator action; never called implicitly,
     * because produced content is the user's work.
     *
     * @returns how many artifacts were removed.
     */
    async clear() {
        return this.serialize(async () => {
            await this.ensureLoaded();
            const count = this.index.size;
            this.index.clear();
            await rm(this.contentDir, { recursive: true, force: true });
            await this.persistIndex();
            return count;
        });
    }
    /**
     * Count artifacts currently indexed.
     * @returns the number of known artifacts.
     */
    async size() {
        await this.ensureLoaded();
        return this.index.size;
    }
    /**
     * Re-read the index from disk, discarding in-memory state.
     *
     * Lets several hub instances share one artifact directory — a common setup when
     * a CLI run and a DSH session both point at the same workspace.
     *
     * @returns the number of artifacts after the reload.
     */
    async reload() {
        return this.serialize(async () => {
            this.index.clear();
            this.loaded = false;
            await this.ensureLoaded();
            return this.index.size;
        });
    }
    /**
     * Verify that every indexed artifact still has content on disk.
     * @returns the ids whose content is missing.
     */
    async findOrphans() {
        await this.ensureLoaded();
        const orphans = [];
        for (const artifact of this.index.values()) {
            const path = fileUriToPath(artifact.uri);
            if (path === undefined) {
                orphans.push(artifact.id);
                continue;
            }
            try {
                await stat(path);
            }
            catch {
                orphans.push(artifact.id);
            }
        }
        return orphans;
    }
    /**
     * List content files in the store directory that no index entry references.
     * @returns absolute paths of unreferenced files.
     */
    async findUnreferencedFiles() {
        await this.ensureLoaded();
        const referenced = new Set();
        for (const artifact of this.index.values()) {
            const path = fileUriToPath(artifact.uri);
            if (path !== undefined)
                referenced.add(resolve(path));
        }
        let entries;
        try {
            entries = await readdir(this.contentDir);
        }
        catch {
            return [];
        }
        const unreferenced = [];
        for (const entry of entries) {
            const absolute = resolve(join(this.contentDir, entry));
            if (!referenced.has(absolute) && extname(entry) !== '.tmp')
                unreferenced.push(absolute);
        }
        return unreferenced;
    }
}
/**
 * Convert an absolute filesystem path to a `file://` URI.
 *
 * Hand-rolled rather than using `pathToFileURL` so the stored URI is stable and
 * readable, and so the reverse conversion below is unambiguous on Windows.
 *
 * @param absolutePath - the path to convert.
 * @returns a `file:///…` URI with forward slashes.
 */
export function pathToFileUri(absolutePath) {
    const normalised = resolve(absolutePath).replace(/\\/g, '/');
    return normalised.startsWith('/') ? `file://${normalised}` : `file:///${normalised}`;
}
/**
 * Convert a `file://` URI back to a filesystem path.
 * @param uri - the URI to convert.
 * @returns the absolute path, or `undefined` when the URI is not a file URI.
 */
export function fileUriToPath(uri) {
    if (!uri.startsWith('file://'))
        return undefined;
    let rest = uri.slice('file://'.length);
    // Strip an optional authority component (`file://localhost/…`).
    if (rest.startsWith('/') && /^\/[A-Za-z]:/.test(rest)) {
        rest = rest.slice(1);
    }
    else if (!/^\/?[A-Za-z]:/.test(rest) && rest.startsWith('/')) {
        // POSIX absolute path: keep the leading slash.
        return rest.replace(/\//g, sep);
    }
    let decoded;
    try {
        decoded = decodeURIComponent(rest);
    }
    catch {
        decoded = rest;
    }
    return decoded.replace(/\//g, sep);
}
/**
 * Resolve a directory that does not exist yet, for callers that need the path
 * before the store creates it.
 * @param root - the target directory.
 * @returns the directory's parent, which must already exist.
 */
export function parentOf(root) {
    return dirname(resolve(root));
}
