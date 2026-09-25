/**
 * Primitive, dependency-free validators.
 *
 * The hub validates untrusted data in three places: model descriptors read from
 * configuration files, artifact indexes read from disk, and invocation requests
 * assembled from model-generated tool arguments. Rather than pull a schema
 * library into the shipped plugin (whose dependencies must resolve inside a DSH
 * profile's `node_modules`), the hub hand-rolls a small, total validator layer.
 *
 * Every reader in this module is *total*: it never throws and never returns a
 * partially-valid value. Callers collect human-readable problems and report them
 * all at once, so a misconfigured model file produces one actionable message
 * instead of failing on the first typo.
 *
 * @module dsh-ai-model-hub/util/validate
 */
/**
 * Whether a value is a plain object (not null, not an array).
 * @param value - candidate value.
 * @returns true when the value can be indexed by string keys.
 */
export function isRecord(value) {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
/**
 * Whether a value is a non-empty, non-whitespace string.
 * @param value - candidate value.
 * @returns true for a usable string.
 */
export function isNonEmptyString(value) {
    return typeof value === 'string' && value.trim().length > 0;
}
/**
 * Whether a value is a finite number.
 * @param value - candidate value.
 * @returns true for a usable number.
 */
export function isFiniteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
}
/**
 * Whether a value is a positive finite number.
 * @param value - candidate value.
 * @returns true for a number strictly greater than zero.
 */
export function isPositiveNumber(value) {
    return isFiniteNumber(value) && value > 0;
}
/**
 * Whether a value is `undefined` or satisfies a predicate.
 * @param value - candidate value.
 * @param predicate - test applied when the value is present.
 * @returns true when the value is absent or passes.
 */
export function isOptional(value, predicate) {
    return value === undefined || predicate(value);
}
/**
 * Whether a value is an array whose every element satisfies a predicate.
 * @param value - candidate value.
 * @param predicate - test applied to each element.
 * @returns true for a fully-conforming array.
 */
export function isArrayOf(value, predicate) {
    return Array.isArray(value) && value.every((item) => predicate(item));
}
/**
 * Whether a value is a string drawn from a fixed vocabulary.
 * @param value - candidate value.
 * @param vocabulary - the allowed strings.
 * @returns true when the value is one of the allowed strings.
 */
export function isOneOf(value, vocabulary) {
    return typeof value === 'string' && vocabulary.includes(value);
}
/**
 * Whether a value survives a structured clone through JSON, i.e. whether putting
 * it in a tool result, session log, or artifact index cannot lose or corrupt it.
 *
 * Rejects cyclic structures, functions, symbols, `undefined` inside containers,
 * and non-finite numbers — exactly the ways a value can look fine in memory and
 * break a durable log later.
 *
 * @param value - candidate value.
 * @param seen - cycle-detection set; callers omit it.
 * @returns true when the value is losslessly JSON-serializable.
 */
export function isLosslessJson(value, seen = new Set()) {
    if (value === null)
        return true;
    switch (typeof value) {
        case 'string':
        case 'boolean':
            return true;
        case 'number':
            return Number.isFinite(value);
        case 'object':
            break;
        default:
            return false;
    }
    if (seen.has(value))
        return false;
    seen.add(value);
    try {
        if (Array.isArray(value)) {
            return value.every((item) => isLosslessJson(item, seen));
        }
        if (!isRecord(value))
            return false;
        // A plain object only. `JSON.stringify(new Date())` yields a string and
        // `JSON.stringify(new Map())` yields `{}`, so a class instance or a built-in
        // container is silently *corrupted* by the round trip rather than rejected by
        // it — which is exactly the loss this function exists to detect, and which a
        // recursive walk over enumerable properties cannot see.
        const prototype = Object.getPrototypeOf(value);
        if (prototype !== Object.prototype && prototype !== null)
            return false;
        return Object.values(value).every((item) => isLosslessJson(item, seen));
    }
    finally {
        seen.delete(value);
    }
}
/** Accumulates issues while reading nested untrusted structure. */
export class IssueCollector {
    issues = [];
    /**
     * Record one problem at a path.
     * @param path - location of the problem.
     * @param message - human-readable explanation.
     */
    add(path, message) {
        this.issues.push({ path, message });
    }
    /**
     * Record a problem only when a condition fails.
     * @param condition - the requirement that must hold.
     * @param path - location of the problem.
     * @param message - human-readable explanation.
     * @returns the condition, so callers can branch on it.
     */
    require(condition, path, message) {
        if (!condition)
            this.add(path, message);
        return condition;
    }
    /** Whether any problem has been recorded. */
    get failed() {
        return this.issues.length > 0;
    }
    /** The problems recorded so far, in insertion order. */
    get collected() {
        return this.issues;
    }
}
/**
 * Render issues as a single operator-facing message.
 * @param label - what was being validated, e.g. a file path.
 * @param issues - the collected problems.
 * @returns a multi-line message listing every problem.
 */
export function formatIssues(label, issues) {
    const lines = issues.map((issue) => `  - ${issue.path}: ${issue.message}`);
    return `${label} failed validation with ${issues.length} problem${issues.length === 1 ? '' : 's'}:\n${lines.join('\n')}`;
}
/**
 * Read an optional string field, returning `undefined` for absence.
 *
 * Records an issue and returns `undefined` when the field is present but not a
 * string, so a caller can keep reading the rest of the object.
 *
 * @param source - the object to read from.
 * @param key - the field name.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the string value, or `undefined`.
 */
export function readOptionalString(source, key, path, collector) {
    const value = source[key];
    if (value === undefined)
        return undefined;
    if (typeof value !== 'string') {
        collector.add(`${path}.${key}`, 'must be a string');
        return undefined;
    }
    return value;
}
/**
 * Read a required, non-empty string field.
 * @param source - the object to read from.
 * @param key - the field name.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the string value, or `undefined` when missing or invalid.
 */
export function readRequiredString(source, key, path, collector) {
    const value = source[key];
    if (value === undefined) {
        collector.add(`${path}.${key}`, 'is required');
        return undefined;
    }
    if (!isNonEmptyString(value)) {
        collector.add(`${path}.${key}`, 'must be a non-empty string');
        return undefined;
    }
    return value;
}
/**
 * Read an optional finite number field.
 * @param source - the object to read from.
 * @param key - the field name.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the number value, or `undefined`.
 */
export function readOptionalNumber(source, key, path, collector) {
    const value = source[key];
    if (value === undefined)
        return undefined;
    if (!isFiniteNumber(value)) {
        collector.add(`${path}.${key}`, 'must be a finite number');
        return undefined;
    }
    return value;
}
/**
 * Read an optional boolean field.
 * @param source - the object to read from.
 * @param key - the field name.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the boolean value, or `undefined`.
 */
export function readOptionalBoolean(source, key, path, collector) {
    const value = source[key];
    if (value === undefined)
        return undefined;
    if (typeof value !== 'boolean') {
        collector.add(`${path}.${key}`, 'must be a boolean');
        return undefined;
    }
    return value;
}
/**
 * Read an optional array of strings.
 * @param source - the object to read from.
 * @param key - the field name.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the array, or `undefined` when absent.
 */
export function readOptionalStringArray(source, key, path, collector) {
    const value = source[key];
    if (value === undefined)
        return undefined;
    if (!isArrayOf(value, (item) => typeof item === 'string')) {
        collector.add(`${path}.${key}`, 'must be an array of strings');
        return undefined;
    }
    return value;
}
/**
 * Read an optional object field.
 * @param source - the object to read from.
 * @param key - the field name.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the object, or `undefined` when absent.
 */
export function readOptionalRecord(source, key, path, collector) {
    const value = source[key];
    if (value === undefined)
        return undefined;
    if (!isRecord(value)) {
        collector.add(`${path}.${key}`, 'must be an object');
        return undefined;
    }
    return value;
}
/**
 * Read an optional enum-valued string field.
 * @param source - the object to read from.
 * @param key - the field name.
 * @param vocabulary - the allowed values.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the matched value, or `undefined`.
 */
export function readOptionalEnum(source, key, vocabulary, path, collector) {
    const value = source[key];
    if (value === undefined)
        return undefined;
    if (!isOneOf(value, vocabulary)) {
        collector.add(`${path}.${key}`, `must be one of ${vocabulary.join(', ')}`);
        return undefined;
    }
    return value;
}
/**
 * Read an optional array of enum-valued strings, dropping (and reporting)
 * unknown members.
 * @param source - the object to read from.
 * @param key - the field name.
 * @param vocabulary - the allowed values.
 * @param path - path used in issue messages.
 * @param collector - issue sink.
 * @returns the validated members, or `undefined` when the field is absent.
 */
export function readOptionalEnumArray(source, key, vocabulary, path, collector) {
    const value = source[key];
    if (value === undefined)
        return undefined;
    if (!Array.isArray(value)) {
        collector.add(`${path}.${key}`, 'must be an array');
        return undefined;
    }
    const accepted = [];
    value.forEach((item, index) => {
        if (isOneOf(item, vocabulary))
            accepted.push(item);
        else
            collector.add(`${path}.${key}[${index}]`, `must be one of ${vocabulary.join(', ')}`);
    });
    return accepted;
}
