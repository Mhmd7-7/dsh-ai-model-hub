/**
 * Drift tests for the published JSON Schema.
 *
 * Two schema representations exist on purpose:
 *
 * - `parseModelCatalogConfig` (TypeScript) is the *enforced* runtime validator,
 *   hand-written so the shipped plugin has no schema-library dependency and can
 *   report every problem at once with actionable paths.
 * - `model-catalog.schema.json` is the *published* schema, for editors,
 *   CI, and anyone generating a catalog outside this codebase.
 *
 * Two representations can drift. These tests make drift a failure: the schema
 * must accept every shipped catalog, and for a table of paired cases it must
 * reach the *same accept/reject verdict* as the runtime validator. That is the
 * strongest guarantee available without generating one from the other, and it is
 * what lets the JSON Schema be trusted as documentation.
 *
 * @module dsh-ai-model-hub/tests/schema.test
 */

import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import { parseModelCatalogConfig } from '../src/index.ts';

/** A JSON Schema node, as far as this test's validator needs to understand one. */
type SchemaNode = boolean | Record<string, unknown>;

/** One validation failure with its location. */
interface SchemaIssue {
  readonly path: string;
  readonly message: string;
}

/**
 * Load the published schema.
 * @returns the parsed schema document.
 */
function loadSchema(): Record<string, unknown> {
  const path = join(process.cwd(), 'src', 'catalog', 'model-catalog.schema.json');
  return JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>;
}

/**
 * Validate a value against the subset of JSON Schema the catalog schema uses.
 *
 * The subset is deliberately small — `type`, `required`, `properties`,
 * `additionalProperties`, `enum`, `pattern`, `minLength`, `maxLength`,
 * `minItems`, `maxItems`, `minimum`, `items`, `$ref`, and `$defs` — because a
 * test that pulled in a full validator would be testing that library rather than
 * this schema. Anything outside the subset is ignored, which is safe here because
 * the schema does not use it.
 *
 * @param root - the whole schema, for `$ref` resolution.
 * @param node - the schema node to apply.
 * @param value - the value being validated.
 * @param path - the current location, for messages.
 * @returns every violation found.
 */
function validate(root: Record<string, unknown>, node: SchemaNode, value: unknown, path: string): SchemaIssue[] {
  if (node === true) return [];
  if (node === false) return [{ path, message: 'no value is allowed here' }];

  const issues: SchemaIssue[] = [];
  const schema = resolveRef(root, node);

  const declaredType = schema['type'];
  if (typeof declaredType === 'string' && !matchesType(declaredType, value)) {
    return [{ path, message: `must be of type ${declaredType}` }];
  }

  const enumValues = schema['enum'];
  if (Array.isArray(enumValues) && !enumValues.includes(value)) {
    return [{ path, message: `must be one of ${enumValues.join(', ')}` }];
  }

  if (typeof value === 'string') {
    const pattern = schema['pattern'];
    if (typeof pattern === 'string' && !new RegExp(pattern).test(value)) {
      issues.push({ path, message: `must match ${pattern}` });
    }
    const minLength = schema['minLength'];
    if (typeof minLength === 'number' && value.length < minLength) {
      issues.push({ path, message: `must be at least ${minLength} characters` });
    }
  }

  if (typeof value === 'number') {
    const minimum = schema['minimum'];
    if (typeof minimum === 'number' && value < minimum) {
      issues.push({ path, message: `must be >= ${minimum}` });
    }
  }

  if (Array.isArray(value)) {
    const minItems = schema['minItems'];
    if (typeof minItems === 'number' && value.length < minItems) {
      issues.push({ path, message: `must have at least ${minItems} item(s)` });
    }
    const maxItems = schema['maxItems'];
    if (typeof maxItems === 'number' && value.length > maxItems) {
      issues.push({ path, message: `must have at most ${maxItems} item(s)` });
    }
    const items = schema['items'];
    if (items !== undefined) {
      value.forEach((item, index) => {
        issues.push(...validate(root, items as SchemaNode, item, `${path}[${index}]`));
      });
    }
  }

  if (isRecord(value) && (declaredType === undefined || declaredType === 'object')) {
    const required = schema['required'];
    if (Array.isArray(required)) {
      for (const key of required) {
        if (typeof key === 'string' && !Object.hasOwn(value, key)) {
          issues.push({ path: `${path}.${key}`, message: 'is required' });
        }
      }
    }
    const properties = schema['properties'];
    if (isRecord(properties)) {
      for (const [key, child] of Object.entries(properties)) {
        if (!Object.hasOwn(value, key)) continue;
        issues.push(...validate(root, child as SchemaNode, value[key], `${path}.${key}`));
      }
    }
    const additional = schema['additionalProperties'];
    if (additional === false && isRecord(properties)) {
      for (const key of Object.keys(value)) {
        if (!Object.hasOwn(properties, key)) {
          issues.push({ path: `${path}.${key}`, message: 'is not an allowed property' });
        }
      }
    }
  }

  return issues;
}

/**
 * Resolve a `$ref` against the document root.
 * @param root - the schema document.
 * @param node - the node that may carry `$ref`.
 * @returns the resolved node.
 */
function resolveRef(root: Record<string, unknown>, node: Record<string, unknown>): Record<string, unknown> {
  const ref = node['$ref'];
  if (typeof ref !== 'string' || !ref.startsWith('#/')) return node;

  let cursor: unknown = root;
  for (const segment of ref.slice(2).split('/')) {
    if (!isRecord(cursor)) return node;
    cursor = cursor[segment];
  }
  return isRecord(cursor) ? cursor : node;
}

/**
 * Whether a value satisfies a JSON Schema primitive type name.
 * @param type - the declared type.
 * @param value - the value to test.
 * @returns whether the type matches.
 */
function matchesType(type: string, value: unknown): boolean {
  switch (type) {
    case 'object':
      return isRecord(value);
    case 'array':
      return Array.isArray(value);
    case 'string':
      return typeof value === 'string';
    case 'number':
      return typeof value === 'number' && Number.isFinite(value);
    case 'integer':
      return typeof value === 'number' && Number.isInteger(value);
    case 'boolean':
      return typeof value === 'boolean';
    case 'null':
      return value === null;
    default:
      return true;
  }
}

/**
 * Whether a value is a plain object.
 * @param value - the candidate.
 * @returns true for a non-array, non-null object.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** A base valid catalog used to derive the paired cases. */
function baseCatalog(): Record<string, unknown> {
  return {
    version: '1',
    hosts: [],
    models: [
      {
        id: 'test_model',
        name: 'Test Model',
        type: 'text_generation',
        capabilities: ['text_to_text'],
        adapter: 'mock',
        runtime: { engine: 'in_process_mock', adapter: 'mock' },
      },
    ],
  };
}

/**
 * A paired case: a document plus the verdict both representations must reach.
 */
interface PairedCase {
  readonly name: string;
  readonly document: unknown;
  readonly valid: boolean;
}

/** Cases where the JSON Schema and the runtime validator must agree. */
const PAIRED_CASES: readonly PairedCase[] = [
  { name: 'a minimal valid catalog', document: baseCatalog(), valid: true },
  {
    name: 'a catalog with every optional field populated',
    document: {
      version: '1',
      hosts: [
        {
          id: 'engine',
          name: 'Engine',
          adapter: 'http_json',
          runtime: { engine: 'e', adapter: 'http_json', endpoint: 'http://127.0.0.1:9000', path: '/api' },
          resources: { vramGb: 8, ramGb: 16, diskGb: 12, requiresGpu: true, allowConcurrentInstances: false },
          lifecycle: {
            startable: true,
            stoppable: true,
            start: { command: 'python', args: ['serve.py'] },
            stop: { command: 'python', args: ['stop.py'] },
            startupTimeoutMs: 1000,
            shutdownTimeoutMs: 500,
            idleTimeoutMs: 1000,
            awaitHealthOnStart: false,
          },
          health: { kind: 'http', path: '/health', timeoutMs: 500 },
          enabled: true,
          notes: 'n',
        },
      ],
      models: [
        {
          id: 'full_model',
          name: 'Full',
          type: 'image_generation',
          capabilities: ['text_to_image', 'image_to_image'],
          inputTypes: ['text', 'image'],
          outputTypes: ['image'],
          version: '1.0.0',
          host: 'engine',
          adapterConfig: { steps: 30 },
          resources: { vramGb: 4 },
          limits: { maxWidth: 1024, maxHeight: 1024, resolutions: ['512x512'], contextTokens: 4096, maxDurationSeconds: 10 },
          lifecycle: { startable: false, stoppable: false },
          health: { kind: 'tcp', timeoutMs: 100 },
          enabled: true,
          priority: 5,
          tags: ['gpu'],
          notes: 'n',
        },
      ],
    },
    valid: true,
  },
  { name: 'a missing models array', document: { version: '1' }, valid: false },
  { name: 'a model with no id', document: { models: [{ name: 'x', type: 'custom', capabilities: ['text_to_text'] }] }, valid: false },
  { name: 'a model with no name', document: { models: [{ id: 'x', type: 'custom', capabilities: ['text_to_text'] }] }, valid: false },
  { name: 'a model with no type', document: { models: [{ id: 'x', name: 'X', capabilities: ['text_to_text'] }] }, valid: false },
  { name: 'a model with an empty capability list', document: { models: [{ id: 'x', name: 'X', type: 'custom', capabilities: [] }] }, valid: false },
  {
    name: 'a model with an unknown capability',
    document: { models: [{ id: 'x', name: 'X', type: 'custom', capabilities: ['teleport'], adapter: 'mock', runtime: { engine: 'e', adapter: 'mock' } }] },
    valid: false,
  },
  {
    name: 'a model with an unknown type',
    document: { models: [{ id: 'x', name: 'X', type: 'wizardry', capabilities: ['text_to_text'], adapter: 'mock', runtime: { engine: 'e', adapter: 'mock' } }] },
    valid: false,
  },
  {
    name: 'an uppercase model id',
    document: { models: [{ id: 'BadID', name: 'X', type: 'custom', capabilities: ['text_to_text'], adapter: 'mock', runtime: { engine: 'e', adapter: 'mock' } }] },
    valid: false,
  },
  {
    name: 'an empty capability-free catalog entry list',
    document: { models: [] },
    valid: true,
  },
  {
    name: 'a runtime with no engine',
    document: { models: [{ id: 'x', name: 'X', type: 'custom', capabilities: ['text_to_text'], adapter: 'mock', runtime: { adapter: 'mock' } }] },
    valid: false,
  },
  {
    name: 'a malformed resolution string',
    document: {
      models: [
        {
          id: 'x',
          name: 'X',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter: 'mock',
          runtime: { engine: 'e', adapter: 'mock' },
          limits: { resolutions: ['not-a-resolution'] },
        },
      ],
    },
    valid: false,
  },
  {
    name: 'an unknown key, which the schema tolerates',
    document: { models: [...(baseCatalog()['models'] as unknown[])], extraTopLevel: true, $comment: 'note' },
    valid: true,
  },
];

describe('published JSON Schema', () => {
  it('is a valid draft 2020-12 document with the expected definitions', () => {
    const schema = loadSchema();
    assert.equal(schema['$schema'], 'https://json-schema.org/draft/2020-12/schema');
    assert.equal(schema['type'], 'object');
    assert.deepEqual(schema['required'], ['models']);
    const defs = schema['$defs'];
    assert.ok(isRecord(defs));
    for (const key of ['capability', 'modelType', 'ioType', 'adapterKind', 'healthCheckKind', 'command', 'runtime', 'resources', 'limits', 'lifecycle', 'health', 'host', 'model']) {
      assert.ok(isRecord(defs[key]), `$defs.${key} is missing`);
    }
  });

  it('enumerates exactly the capability vocabulary the code exposes', async () => {
    const schema = loadSchema();
    const caps = (schema['$defs'] as Record<string, Record<string, unknown>>)['capability'];
    const { CAPABILITIES } = await import('../src/index.ts');
    assert.deepEqual(caps?.['enum'], [...CAPABILITIES]);
  });

  it('enumerates exactly the adapter kinds the code accepts', async () => {
    const schema = loadSchema();
    const kinds = (schema['$defs'] as Record<string, Record<string, unknown>>)['adapterKind'];
    const { ADAPTER_KINDS } = await import('../src/index.ts');
    assert.deepEqual(kinds?.['enum'], [...ADAPTER_KINDS]);
  });

  it('accepts every shipped catalog', () => {
    const schema = loadSchema();
    for (const relative of ['config/models.json', 'config/models.mock.json', 'config/examples/real-models.example.json']) {
      const document = JSON.parse(readFileSync(join(process.cwd(), relative), 'utf8')) as unknown;
      const issues = validate(schema, schema, document, '$');
      assert.deepEqual(issues, [], `${relative} failed the published schema:\n${issues.map((issue) => `  ${issue.path}: ${issue.message}`).join('\n')}`);
    }
  });
});

describe('schema and runtime validator agree', () => {
  for (const testCase of PAIRED_CASES) {
    it(`agrees that "${testCase.name}" is ${testCase.valid ? 'valid' : 'invalid'}`, () => {
      const schema = loadSchema();
      const schemaIssues = validate(schema, schema, testCase.document, '$');
      const runtimeResult = parseModelCatalogConfig(testCase.document, testCase.name);

      assert.equal(
        runtimeResult.ok,
        testCase.valid,
        `runtime validator disagreed: expected ${testCase.valid ? 'valid' : 'invalid'}, got ${runtimeResult.ok ? 'valid' : runtimeResult.message}`,
      );
      assert.equal(
        schemaIssues.length === 0,
        testCase.valid,
        `published schema disagreed: expected ${testCase.valid ? 'valid' : 'invalid'}, got:\n${schemaIssues.map((issue) => `  ${issue.path}: ${issue.message}`).join('\n')}`,
      );
    });
  }
});
