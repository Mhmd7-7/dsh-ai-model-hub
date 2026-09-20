/**
 * Tests for the capability vocabulary and descriptor validation.
 *
 * These cover the layer where a configuration mistake becomes either a clear
 * error or a silent misroute, so they lean hard on the negative cases: what the
 * validator must *refuse*.
 *
 * @module dsh-ai-model-hub/tests/catalog.test
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import {
  CAPABILITIES,
  CAPABILITY_IO,
  ModelCatalog,
  checkDescriptorCoherence,
  defaultIoFor,
  isCapability,
  isIoType,
  isModelType,
  parseModelCatalogConfig,
  resolveDescriptor,
} from '../src/index.ts';

/** A minimal valid catalog document used as the base for mutation tests. */
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

describe('capability vocabulary', () => {
  it('exposes the documented capabilities', () => {
    const expected = [
      'text_to_text',
      'text_to_image',
      'image_to_image',
      'text_to_3d',
      'image_to_3d',
      'audio_generation',
      'speech_to_text',
      'image_understanding',
      'video_generation',
    ];
    assert.deepEqual([...CAPABILITIES], expected);
  });

  it('describes input and output kinds for every capability', () => {
    for (const capability of CAPABILITIES) {
      const io = CAPABILITY_IO[capability];
      assert.ok(io.input.length > 0, `${capability} must declare at least one input kind`);
      assert.ok(io.output.length > 0, `${capability} must declare at least one output kind`);
      for (const kind of [...io.input, ...io.output]) {
        assert.ok(isIoType(kind), `${capability} declares unknown kind ${kind}`);
      }
    }
  });

  it('narrows strings correctly', () => {
    assert.ok(isCapability('text_to_image'));
    assert.ok(!isCapability('text_to_pancakes'));
    assert.ok(isModelType('three_d_generation'));
    assert.ok(!isModelType('image_generation_x'));
    assert.ok(isIoType('model_3d'));
    assert.ok(!isIoType('mesh'));
  });

  it('returns a mutable copy from defaultIoFor so callers cannot corrupt the table', () => {
    const first = defaultIoFor('text_to_image');
    first.input.push('text');
    const second = defaultIoFor('text_to_image');
    assert.deepEqual(second.input, ['text']);
  });
});

describe('descriptor validation', () => {
  it('accepts a minimal valid document', () => {
    const result = parseModelCatalogConfig(baseCatalog(), 'test');
    assert.equal(result.ok, true);
    if (!result.ok) return;
    assert.equal(result.config.models.length, 1);
    assert.equal(result.config.models[0]?.id, 'test_model');
  });

  it('reports every problem at once rather than the first', () => {
    const document = {
      models: [
        { name: 'no id', type: 'nonsense', capabilities: [] },
        { id: 'second', name: 'also broken', type: 'text_generation', capabilities: ['not_a_capability'] },
      ],
    };
    const result = parseModelCatalogConfig(document, 'test');
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.ok(result.issues.length >= 4, `expected several issues, got ${result.issues.length}`);
    const paths = result.issues.map((issue) => issue.path);
    assert.ok(paths.includes('models[0].id'));
    assert.ok(paths.includes('models[0].type'));
    assert.ok(paths.includes('models[0].capabilities'));
    assert.ok(paths.includes('models[1].capabilities[0]'));
  });

  it('rejects a model that declares neither a host nor a runtime', () => {
    const document = baseCatalog();
    const models = document['models'] as Record<string, unknown>[];
    delete models[0]?.['runtime'];
    const result = parseModelCatalogConfig(document, 'test');
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.ok(result.issues.some((issue) => issue.message.includes('host')));
  });

  it('rejects a model that declares both a host and an inline runtime', () => {
    const document = baseCatalog();
    document['hosts'] = [
      { id: 'h', name: 'H', adapter: 'mock', runtime: { engine: 'e', adapter: 'mock' } },
    ];
    const models = document['models'] as Record<string, unknown>[];
    if (models[0] !== undefined) models[0]['host'] = 'h';
    const result = parseModelCatalogConfig(document, 'test');
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.ok(result.issues.some((issue) => issue.message.includes('both')));
  });

  it('rejects a reference to an unknown host', () => {
    const document = baseCatalog();
    const models = document['models'] as Record<string, unknown>[];
    delete models[0]?.['runtime'];
    if (models[0] !== undefined) models[0]['host'] = 'ghost';
    const result = parseModelCatalogConfig(document, 'test');
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.ok(result.issues.some((issue) => issue.message.includes('unknown host')));
  });

  it('rejects duplicate model ids', () => {
    const document = baseCatalog();
    const models = document['models'] as Record<string, unknown>[];
    models.push({ ...(models[0] ?? {}) });
    const result = parseModelCatalogConfig(document, 'test');
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.ok(result.issues.some((issue) => issue.message.includes('duplicate model id')));
  });

  it('rejects input types that contradict the declared capabilities', () => {
    const document = baseCatalog();
    const models = document['models'] as Record<string, unknown>[];
    if (models[0] !== undefined) models[0]['inputTypes'] = ['audio'];
    const result = parseModelCatalogConfig(document, 'test');
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.ok(result.issues.some((issue) => issue.path === 'models[0].inputTypes'));
  });

  it('requires a start command when a model claims to be startable', () => {
    const document = baseCatalog();
    const models = document['models'] as Record<string, unknown>[];
    if (models[0] !== undefined) models[0]['lifecycle'] = { startable: true, stoppable: false };
    const result = parseModelCatalogConfig(document, 'test');
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.ok(result.issues.some((issue) => issue.path === 'models[0].lifecycle.start'));
  });

  it('requires an endpoint for an http adapter', () => {
    const document = baseCatalog();
    const models = document['models'] as Record<string, unknown>[];
    if (models[0] !== undefined) {
      models[0]['adapter'] = 'http_json';
      models[0]['runtime'] = { engine: 'x', adapter: 'http_json' };
    }
    const result = parseModelCatalogConfig(document, 'test');
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.ok(result.issues.some((issue) => issue.path === 'models[0].runtime.endpoint'));
  });

  it('rejects a malformed resolution string', () => {
    const document = baseCatalog();
    const models = document['models'] as Record<string, unknown>[];
    if (models[0] !== undefined) models[0]['limits'] = { resolutions: ['1024x1024', 'big'] };
    const result = parseModelCatalogConfig(document, 'test');
    assert.equal(result.ok, false);
    if (result.ok) return;
    assert.ok(result.issues.some((issue) => issue.path === 'models[0].limits.resolutions[1]'));
  });

  it('ignores unknown keys such as $comment', () => {
    const document = baseCatalog();
    document['$comment'] = 'a note';
    const models = document['models'] as Record<string, unknown>[];
    if (models[0] !== undefined) models[0]['$comment'] = 'another note';
    assert.equal(parseModelCatalogConfig(document, 'test').ok, true);
  });
});

describe('descriptor resolution', () => {
  it('inherits runtime, lifecycle, resources, and health from a host', () => {
    const host = {
      id: 'h',
      name: 'Engine',
      adapter: 'http_json' as const,
      runtime: { engine: 'engine', adapter: 'http_json' as const, endpoint: 'http://127.0.0.1:9000' },
      resources: { vramGb: 8, ramGb: 16 },
      lifecycle: { startable: true, stoppable: true, start: { command: 'python', args: ['serve.py'] }, idleTimeoutMs: 60000 },
      health: { kind: 'http' as const, path: '/health' },
    };
    const descriptor = {
      id: 'child',
      name: 'Child',
      type: 'image_generation' as const,
      capabilities: ['text_to_image'] as const,
      host: 'h',
    };
    const resolved = resolveDescriptor(descriptor, host);
    assert.equal(resolved.adapter, 'http_json');
    assert.equal(resolved.runtime.endpoint, 'http://127.0.0.1:9000');
    assert.equal(resolved.resources.vramGb, 8);
    assert.equal(resolved.resources.ramGb, 16);
    assert.equal(resolved.lifecycle.startable, true);
    assert.equal(resolved.lifecycle.idleTimeoutMs, 60_000);
    assert.equal(resolved.health.path, '/health');
    assert.deepEqual(resolved.inputTypes, ['text']);
    assert.deepEqual(resolved.outputTypes, ['image']);
  });

  it('lets a descriptor override its host field by field', () => {
    const host = {
      id: 'h',
      name: 'Engine',
      adapter: 'mock' as const,
      runtime: { engine: 'engine', adapter: 'mock' as const },
      resources: { vramGb: 8, ramGb: 16 },
    };
    const descriptor = {
      id: 'child',
      name: 'Child',
      type: 'image_generation' as const,
      capabilities: ['text_to_image'] as const,
      host: 'h',
      resources: { vramGb: 2 },
    };
    const resolved = resolveDescriptor(descriptor, host);
    assert.equal(resolved.resources.vramGb, 2, 'descriptor wins');
    assert.equal(resolved.resources.ramGb, 16, 'host fills the gap');
  });

  it('derives a default health check from the adapter kind', () => {
    const cases = [
      ['http_json', 'http'],
      ['openai_compatible', 'http'],
      ['cli', 'none'],
      ['mock', 'none'],
    ] as const;
    for (const [adapter, expected] of cases) {
      const resolved = resolveDescriptor(
        {
          id: 'm',
          name: 'M',
          type: 'custom',
          capabilities: ['text_to_text'],
          adapter,
          runtime: { engine: 'e', adapter, ...(adapter === 'cli' ? {} : { endpoint: 'http://127.0.0.1:1' }) },
        },
        undefined,
      );
      assert.equal(resolved.health.kind, expected, `${adapter} should default to ${expected}`);
    }
  });

  it('refuses to call a model startable when no launch command exists', () => {
    const resolved = resolveDescriptor(
      {
        id: 'm',
        name: 'M',
        type: 'custom',
        capabilities: ['text_to_text'],
        adapter: 'mock',
        runtime: { engine: 'e', adapter: 'mock' },
        lifecycle: { startable: true, stoppable: true },
      },
      undefined,
    );
    assert.equal(resolved.lifecycle.startable, false, 'a startable flag without a command must not survive resolution');
  });
});

describe('ModelCatalog', () => {
  it('orders candidates deterministically by priority then id', () => {
    const catalog = ModelCatalog.fromConfig(
      {
        models: [
          {
            id: 'zebra',
            name: 'Zebra',
            type: 'text_generation',
            capabilities: ['text_to_text'],
            adapter: 'mock',
            runtime: { engine: 'e', adapter: 'mock' },
            priority: 10,
          },
          {
            id: 'alpha',
            name: 'Alpha',
            type: 'text_generation',
            capabilities: ['text_to_text'],
            adapter: 'mock',
            runtime: { engine: 'e', adapter: 'mock' },
            priority: 10,
          },
          {
            id: 'first',
            name: 'First',
            type: 'text_generation',
            capabilities: ['text_to_text'],
            adapter: 'mock',
            runtime: { engine: 'e', adapter: 'mock' },
            priority: 1,
          },
        ],
      },
      'test',
    );
    assert.deepEqual(
      catalog.findModelsByCapability('text_to_text').map((model) => model.id),
      ['first', 'alpha', 'zebra'],
    );
  });

  it('excludes disabled models from routing but keeps them discoverable', () => {
    const catalog = ModelCatalog.fromConfig(
      {
        models: [
          {
            id: 'off',
            name: 'Off',
            type: 'text_generation',
            capabilities: ['text_to_text'],
            adapter: 'mock',
            runtime: { engine: 'e', adapter: 'mock' },
            enabled: false,
          },
        ],
      },
      'test',
    );
    assert.deepEqual(catalog.findModelsByCapability('text_to_text'), []);
    assert.equal(catalog.findModelsByCapabilityIncludingDisabled('text_to_text').length, 1);
    assert.deepEqual(catalog.listUnservedCapabilities().map((entry) => entry.capability), [...CAPABILITIES]);
    const reason = catalog.listUnservedCapabilities().find((entry) => entry.capability === 'text_to_text')?.reason;
    assert.match(reason ?? '', /disabled/);
  });

  it('reports a capability-unaware model as unserved with a distinct reason', () => {
    const catalog = ModelCatalog.fromConfig(
      {
        models: [
          {
            id: 'only_text',
            name: 'Only Text',
            type: 'text_generation',
            capabilities: ['text_to_text'],
            adapter: 'mock',
            runtime: { engine: 'e', adapter: 'mock' },
          },
        ],
      },
      'test',
    );
    const video = catalog.listUnservedCapabilities().find((entry) => entry.capability === 'video_generation');
    assert.match(video?.reason ?? '', /no configured model/);
  });

  it('rejects an unknown model id with MODEL_NOT_FOUND', () => {
    const catalog = ModelCatalog.fromConfig(baseCatalog(), 'test');
    assert.throws(
      () => catalog.requireModel('nope'),
      (error: unknown) =>
        typeof error === 'object' && error !== null && (error as { code?: string }).code === 'MODEL_NOT_FOUND',
    );
  });

  it('reports unsupported resources against the machine profile', () => {
    const catalog = ModelCatalog.fromConfig(
      {
        models: [
          {
            id: 'huge',
            name: 'Huge',
            type: 'image_generation',
            capabilities: ['text_to_image'],
            adapter: 'mock',
            runtime: { engine: 'e', adapter: 'mock' },
            resources: { vramGb: 24, ramGb: 64, requiresGpu: true },
          },
        ],
      },
      'test',
      { machine: { vramGb: 8, ramGb: 32, hasGpu: true, notes: 'test' } },
    );
    const check = catalog.checkResources(catalog.requireModel('huge'));
    assert.equal(check.supported, false);
    assert.match(check.reason ?? '', /24 GiB VRAM/);
  });

  it('refuses a GPU-requiring model on a GPU-less machine', () => {
    const catalog = ModelCatalog.fromConfig(
      {
        models: [
          {
            id: 'gpu_only',
            name: 'GPU Only',
            type: 'image_generation',
            capabilities: ['text_to_image'],
            adapter: 'mock',
            runtime: { engine: 'e', adapter: 'mock' },
            resources: { requiresGpu: true },
          },
        ],
      },
      'test',
      { machine: { vramGb: 0, ramGb: 16, hasGpu: false, notes: 'test' } },
    );
    const check = catalog.checkResources(catalog.requireModel('gpu_only'));
    assert.equal(check.supported, false);
    assert.match(check.reason ?? '', /GPU/);
  });

  it('assumes support when detection returned nothing', () => {
    const catalog = ModelCatalog.fromConfig(
      {
        models: [
          {
            id: 'big',
            name: 'Big',
            type: 'image_generation',
            capabilities: ['text_to_image'],
            adapter: 'mock',
            runtime: { engine: 'e', adapter: 'mock' },
            resources: { vramGb: 80, ramGb: 512, requiresGpu: true },
          },
        ],
      },
      'test',
      { machine: { vramGb: 0, ramGb: 0, hasGpu: false, notes: 'not probed' } },
    );
    assert.equal(catalog.checkResources(catalog.requireModel('big')).supported, true);
  });
});

describe('checkDescriptorCoherence', () => {
  it('flags duplicates and unknown capabilities', () => {
    const problems = checkDescriptorCoherence({
      id: 'm',
      name: 'M',
      type: 'custom',
      capabilities: ['text_to_text', 'text_to_text', 'nope' as never],
    });
    assert.equal(problems.length, 2);
  });
});
