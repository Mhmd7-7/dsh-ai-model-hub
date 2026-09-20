/**
 * The adapter contract.
 *
 * An adapter is the *only* place engine-specific knowledge lives. It knows how
 * to launch Stable Diffusion, or how to POST to a ComfyUI graph, or which flag
 * makes `llama-server` bind a port. Nothing above it does.
 *
 * The trade is explicit: to add a model whose engine is already supported, you
 * write a config entry. To add a model on a *new* kind of engine, you write one
 * adapter that implements this interface — and you still never touch the router,
 * the catalog, the runtime manager, or DSH.
 *
 * @module dsh-ai-model-hub/adapters/types
 */
/**
 * A registry of adapters keyed by kind.
 *
 * The runtime manager resolves `model.adapter` through this registry. Because
 * lookup is by the model's own declared kind, an adapter is never chosen by
 * model id or name anywhere in the codebase.
 */
export class AdapterRegistry {
    adapters = new Map();
    /**
     * @param adapters - adapters to register immediately.
     */
    constructor(adapters = []) {
        for (const adapter of adapters)
            this.register(adapter);
    }
    /**
     * Register an adapter, replacing any previous one of the same kind.
     *
     * Replacement is allowed deliberately: it is how a test substitutes a fake, and
     * how a plugin overrides a built-in with a real engine. The returned disposer
     * restores the previous binding, which keeps that override scoped.
     *
     * @param adapter - the adapter to register.
     * @returns a disposer that restores whatever was registered before.
     */
    register(adapter) {
        const previous = this.adapters.get(adapter.kind);
        this.adapters.set(adapter.kind, adapter);
        return () => {
            if (previous === undefined)
                this.adapters.delete(adapter.kind);
            else
                this.adapters.set(adapter.kind, previous);
        };
    }
    /**
     * Find the adapter for a kind.
     * @param kind - the adapter kind.
     * @returns the adapter, or `undefined` when none is registered.
     */
    get(kind) {
        return this.adapters.get(kind);
    }
    /**
     * Find the adapter for a kind, failing loudly when absent.
     * @param kind - the adapter kind.
     * @returns the adapter.
     * @throws ModelHubError with `UNSUPPORTED_OPERATION`.
     */
    require(kind) {
        const adapter = this.adapters.get(kind);
        if (adapter === undefined) {
            throw new Error(`no adapter registered for kind "${kind}"`);
        }
        return adapter;
    }
    /** Every registered adapter kind, in registration order. */
    listKinds() {
        return [...this.adapters.keys()];
    }
}
/**
 * A logger that discards everything, used as the default.
 * @returns a silent logger.
 */
export function silentLogger() {
    return {
        debug: () => { },
        info: () => { },
        warn: () => { },
    };
}
/**
 * Build a logger that forwards to a single line-oriented sink.
 * @param sink - receives fully-formatted lines.
 * @param prefix - prepended to every line.
 * @returns the logger.
 */
export function lineLogger(sink, prefix) {
    const emit = (level) => (message, fields) => {
        const suffix = fields === undefined || Object.keys(fields).length === 0
            ? ''
            : ` ${JSON.stringify(fields)}`;
        sink(`${prefix} ${level} ${message}${suffix}`);
    };
    return { debug: emit('debug'), info: emit('info'), warn: emit('warn') };
}
