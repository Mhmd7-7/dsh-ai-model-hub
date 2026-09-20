/**
 * Typed errors and a stable error-code vocabulary.
 *
 * Every layer throws one of these. The DSH plugin layer converts them into
 * tool failures with the code preserved, so the agent can distinguish "no model
 * offers this capability" (a planning problem) from "the model is unhealthy"
 * (a retry-or-fallback problem) from "you violated the sandbox" (a bug).
 *
 * @module dsh-ai-model-hub/errors
 */
/** Stable machine-readable failure codes. */
export const ERROR_CODES = [
    /** A descriptor or manifest failed schema validation. */
    'INVALID_DESCRIPTOR',
    /** Two models or hosts declared the same id. */
    'DUPLICATE_ID',
    /** The requested id is not registered. */
    'MODEL_NOT_FOUND',
    /** No registered model satisfies the request after filtering. */
    'NO_COMPATIBLE_MODEL',
    /** The capability string is not in the vocabulary. */
    'UNKNOWN_CAPABILITY',
    /** A model exists but is not in a state that can serve work. */
    'MODEL_UNAVAILABLE',
    /** The model's process could not be started. */
    'START_FAILED',
    /** The model's health check did not pass within the budget. */
    'HEALTH_CHECK_FAILED',
    /** The adapter reported a domain failure. */
    'INVOCATION_FAILED',
    /** The invocation exceeded its timeout budget. */
    'INVOCATION_TIMEOUT',
    /** The caller cancelled the invocation. */
    'INVOCATION_ABORTED',
    /** The model's declared resource requirements exceed the machine. */
    'INSUFFICIENT_RESOURCES',
    /** A launch or request target was refused by the security guardrails. */
    'UNSAFE_OPERATION',
    /** A configuration file could not be read or parsed. */
    'CONFIG_ERROR',
    /** An artifact was missing, unreadable, or of the wrong kind. */
    'ARTIFACT_ERROR',
    /** An operation is not supported by this model or adapter. */
    'UNSUPPORTED_OPERATION',
    /** The runtime manager was asked to do something for a model with no lifecycle. */
    'LIFECYCLE_UNSUPPORTED',
];
/**
 * The single error type raised by hub layers.
 *
 * `code` is stable and safe to branch on; `message` is for humans and may be
 * shown to the model; `details` carries structured context that must be
 * losslessly JSON-serializable because tool results carry it into the session log.
 */
export class ModelHubError extends Error {
    /** Stable machine-readable code. */
    code;
    /** Structured, JSON-serializable context. */
    details;
    /**
     * @param code - stable failure code.
     * @param message - human-readable explanation, safe to show to the model.
     * @param details - optional structured context; must be losslessly JSON.
     */
    constructor(code, message, details = {}) {
        super(message);
        this.name = 'ModelHubError';
        this.code = code;
        this.details = details;
    }
    /**
     * Project this error into a plain JSON value for tool results and logs.
     * @returns a JSON-safe object with the code, message, and details.
     */
    toJSON() {
        return { code: this.code, message: this.message, details: this.details };
    }
}
/**
 * Narrow an unknown throwable into a message plus optional code and details.
 *
 * Adapters receive third-party errors (fetch failures, spawn failures, HTTP
 * error bodies) whose shape is unknown. This is the one boundary that converts
 * them, so no other module needs an `instanceof` ladder.
 *
 * @param error - the caught value.
 * @returns a normalized description.
 */
export function describeError(error) {
    if (error instanceof ModelHubError) {
        return { code: error.code, message: error.message, details: error.details };
    }
    if (error instanceof Error) {
        const details = { name: error.name };
        const systemCode = error.code;
        if (typeof systemCode === 'string')
            details['systemCode'] = systemCode;
        return { code: 'INVOCATION_FAILED', message: error.message, details };
    }
    return { code: 'INVOCATION_FAILED', message: String(error), details: {} };
}
/**
 * Wrap any throwable as a {@link ModelHubError}, preserving an existing one.
 * @param error - the caught value.
 * @param fallbackCode - code to use when the value carries none.
 * @param context - extra structured context merged into `details`.
 * @returns a hub error safe to propagate.
 */
export function toHubError(error, fallbackCode = 'INVOCATION_FAILED', context = {}) {
    if (error instanceof ModelHubError) {
        if (Object.keys(context).length === 0)
            return error;
        return new ModelHubError(error.code, error.message, { ...error.details, ...context });
    }
    const described = describeError(error);
    const code = described.code === 'INVOCATION_FAILED' ? fallbackCode : described.code;
    return new ModelHubError(code, described.message, { ...described.details, ...context });
}
