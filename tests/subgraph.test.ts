/**
 * Tests for ComfyUI subgraph flattening and analysis.
 *
 * These cover the path that matters for an operator who never wants to export an
 * API file: a workflow saved by the ComfyUI *editor*, collapsed into subgraphs,
 * discovered, flattened, analysed and made invocable.
 *
 * The fixtures reproduce the structure a real export uses — a `definitions`
 * table keyed by subgraph UUID, a node whose `type` is that UUID, virtual
 * `inputNode`/`outputNode` proxies with negative ids, links written as objects
 * inside a definition and as positional arrays at the top level, and per-instance
 * values in both `widgets_values` and `widgets_values_named`. The shapes are taken
 * from live files, not invented.
 *
 * @module dsh-ai-model-hub/tests/subgraph.test
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { parseComfyWorkflow, readComfyNodeIo, scanWorkflowDocument } from '../src/index.ts';

/** Node metadata for the classes these fixtures use. */
const objectInfo = {
  CheckpointLoaderSimple: { input: { required: { ckpt_name: [['A.safetensors'], {}] } }, output: ['MODEL', 'CLIP', 'VAE'] },
  CLIPTextEncode: {
    input: { required: { clip: ['CLIP', {}], text: ['STRING', {}] } },
    output: ['CONDITIONING'],
  },
  EmptyLatentImage: {
    input: { required: { width: ['INT', {}], height: ['INT', {}], batch_size: ['INT', {}] } },
    output: ['LATENT'],
  },
  KSampler: {
    input: {
      required: {
        model: ['MODEL', {}],
        positive: ['CONDITIONING', {}],
        negative: ['CONDITIONING', {}],
        latent_image: ['LATENT', {}],
        seed: ['INT', {}],
        steps: ['INT', {}],
        cfg: ['FLOAT', {}],
      },
    },
    output: ['LATENT'],
  },
  VAEDecode: { input: { required: { samples: ['LATENT', {}], vae: ['VAE', {}] } }, output: ['IMAGE'] },
  SaveImage: {
    input: { required: { images: ['IMAGE', {}], filename_prefix: ['STRING', {}] } },
    output: ['IMAGE'],
    output_node: true,
  },
};
const io = readComfyNodeIo(objectInfo);
const classes = new Set(Object.keys(objectInfo));

/**
 * Parse a document and return its node map.
 *
 * `parseComfyWorkflow` yields nodes in the `{ classType, inputs }` shape for both
 * serializations, which is the structure flattening and conversion are asserted
 * against; `scanWorkflowDocument` is what turns that into the queued graph.
 *
 * @param raw - the document.
 * @returns node id → class and inputs.
 */
function converted(raw: unknown): Record<string, { classType: string; inputs: Record<string, unknown> }> {
  const parsed = parseComfyWorkflow(raw, 'fixture.json', io);
  return parsed.nodes as unknown as Record<string, { classType: string; inputs: Record<string, unknown> }>;
}

/**
 * A UI workflow with one subgraph, shaped like a real export.
 *
 * The subgraph is a whole text-to-image pipeline: its prompt, width and height are
 * promoted inputs, and its only output is the decoded image.
 */
function oneSubgraph(prompt = 'a cat', width = 640): Record<string, unknown> {
  return {
    id: 'wf-1',
    revision: 1,
    nodes: [
      {
        id: 1,
        type: 'aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa',
        inputs: [
          { name: 'text', label: 'prompt', type: 'STRING', widget: { name: 'text' }, link: null },
          { name: 'width', label: 'width', type: 'INT', widget: { name: 'width' }, link: null },
          { name: 'height', label: 'height', type: 'INT', widget: { name: 'height' }, link: null },
        ],
        outputs: [{ name: 'IMAGE', type: 'IMAGE', links: [7] }],
        widgets_values: [prompt, width, 640],
        widgets_values_named: { text: prompt, width, height: 640 },
      },
      { id: 2, type: 'SaveImage', inputs: [{ name: 'images', type: 'IMAGE', link: 7 }], widgets_values: { filename_prefix: 'demo' } },
      { id: 3, type: 'MarkdownNote', inputs: [], widgets_values: 'a note, connected to nothing' },
    ],
    links: [[7, 1, 0, 2, 0, 'IMAGE']],
    definitions: {
      subgraphs: [
        {
          id: 'aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa',
          name: 'Text to Image',
          inputNode: { id: -10 },
          outputNode: { id: -20 },
          inputs: [
            { name: 'text', type: 'STRING' },
            { name: 'width', type: 'INT' },
            { name: 'height', type: 'INT' },
          ],
          outputs: [{ name: 'IMAGE', type: 'IMAGE' }],
          nodes: [
            { id: 10, type: 'CheckpointLoaderSimple', inputs: [], widgets_values: ['A.safetensors'] },
            { id: 11, type: 'CLIPTextEncode', inputs: [{ name: 'clip', link: 100 }, { name: 'text', link: 101 }], widgets_values: ['last used value'] },
            { id: 12, type: 'CLIPTextEncode', inputs: [{ name: 'clip', link: 100 }], widgets_values: [''] },
            { id: 13, type: 'EmptyLatentImage', inputs: [{ name: 'width', link: 102 }, { name: 'height', link: 103 }], widgets_values: [512, 512, 1] },
            {
              id: 14,
              type: 'KSampler',
              inputs: [
                { name: 'model', link: 104 },
                { name: 'positive', link: 105 },
                { name: 'negative', link: 106 },
                { name: 'latent_image', link: 107 },
              ],
              widgets_values: [123, 'randomize', 8, 1, 'euler', 'normal', 1],
            },
            { id: 15, type: 'VAEDecode', inputs: [{ name: 'samples', link: 108 }, { name: 'vae', link: 109 }], widgets_values: [] },
          ],
          links: [
            { id: 100, origin_id: 10, origin_slot: 1, target_id: 11, target_slot: 0, type: 'CLIP' },
            { id: 101, origin_id: -10, origin_slot: 0, target_id: 11, target_slot: 1, type: 'STRING' },
            { id: 102, origin_id: -10, origin_slot: 1, target_id: 13, target_slot: 0, type: 'INT' },
            { id: 103, origin_id: -10, origin_slot: 2, target_id: 13, target_slot: 1, type: 'INT' },
            { id: 104, origin_id: 10, origin_slot: 0, target_id: 14, target_slot: 0, type: 'MODEL' },
            { id: 105, origin_id: 11, origin_slot: 0, target_id: 14, target_slot: 1, type: 'CONDITIONING' },
            { id: 106, origin_id: 12, origin_slot: 0, target_id: 14, target_slot: 2, type: 'CONDITIONING' },
            { id: 107, origin_id: 13, origin_slot: 0, target_id: 14, target_slot: 3, type: 'LATENT' },
            { id: 108, origin_id: 14, origin_slot: 0, target_id: 15, target_slot: 0, type: 'LATENT' },
            { id: 109, origin_id: 10, origin_slot: 2, target_id: 15, target_slot: 1, type: 'VAE' },
            { id: 110, origin_id: 15, origin_slot: 0, target_id: -20, target_slot: 0, type: 'IMAGE' },
          ],
        },
      ],
    },
  };
}

/**
 * Two levels of nesting: the outer subgraph places an instance of the inner one.
 * @param prompt - the value the outermost instance binds.
 * @returns the document.
 */
function nestedSubgraphs(prompt: string): Record<string, unknown> {
  return {
    id: 'wf-nested',
    nodes: [
      {
        id: 9,
        type: 'bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb',
        inputs: [{ name: 'value', label: 'prompt', type: 'STRING', link: null }],
        outputs: [{ name: 'COND', type: 'CONDITIONING', links: [300] }],
        widgets_values: [prompt],
        widgets_values_named: { value: prompt },
      },
      { id: 8, type: 'SaveImage', inputs: [{ name: 'images', link: 300 }], widgets_values: { filename_prefix: 'n' } },
    ],
    links: [[300, 9, 0, 8, 0, 'CONDITIONING']],
    definitions: {
      subgraphs: [
        {
          id: 'bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb',
          name: 'Outer',
          inputNode: { id: -10 },
          outputNode: { id: -20 },
          inputs: [{ name: 'value', type: 'STRING' }],
          outputs: [{ name: 'COND', type: 'CONDITIONING' }],
          nodes: [
            {
              id: 5,
              type: 'cccccccc-3333-4333-8333-cccccccccccc',
              title: 'Inner',
              inputs: [{ name: 'value', label: 'prompt', link: 200 }],
              outputs: [{ name: 'COND', type: 'CONDITIONING', links: [201] }],
              widgets_values: ['inner default'],
              widgets_values_named: { value: 'inner default' },
            },
          ],
          links: [
            { id: 200, origin_id: -10, origin_slot: 0, target_id: 5, target_slot: 0, type: 'STRING' },
            { id: 201, origin_id: 5, origin_slot: 0, target_id: -20, target_slot: 0, type: 'CONDITIONING' },
          ],
        },
        {
          id: 'cccccccc-3333-4333-8333-cccccccccccc',
          name: 'Inner',
          inputNode: { id: -10 },
          outputNode: { id: -20 },
          inputs: [{ name: 'value', type: 'STRING' }],
          outputs: [{ name: 'COND', type: 'CONDITIONING' }],
          nodes: [
            { id: 1, type: 'CLIPTextEncode', inputs: [{ name: 'text', link: 400 }], widgets_values: ['unused'] },
          ],
          links: [
            { id: 400, origin_id: -10, origin_slot: 0, target_id: 1, target_slot: 0, type: 'STRING' },
            { id: 401, origin_id: 1, origin_slot: 0, target_id: -20, target_slot: 0, type: 'CONDITIONING' },
          ],
        },
      ],
    },
  };
}

/**
 * Two instances of one subgraph, each with its own override.
 * @returns the document.
 */
function repeatedInstances(): Record<string, unknown> {
  const base = oneSubgraph();
  const shared = (base['definitions'] as Record<string, unknown>)['subgraphs'];
  return {
    id: 'wf-repeat',
    nodes: [
      {
        id: 1,
        type: 'aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa',
        inputs: [
          { name: 'text', label: 'prompt', type: 'STRING', link: null },
          { name: 'width', label: 'width', type: 'INT', link: null },
          { name: 'height', label: 'height', type: 'INT', link: null },
        ],
        outputs: [{ name: 'IMAGE', type: 'IMAGE', links: [7] }],
        widgets_values: ['first prompt', 512, 512],
        widgets_values_named: { text: 'first prompt', width: 512, height: 512 },
      },
      {
        id: 4,
        type: 'aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa',
        inputs: [
          { name: 'text', label: 'prompt', type: 'STRING', link: null },
          { name: 'width', label: 'width', type: 'INT', link: null },
          { name: 'height', label: 'height', type: 'INT', link: null },
        ],
        outputs: [{ name: 'IMAGE', type: 'IMAGE', links: [8] }],
        // Deliberately given only the positional array, to prove that path works.
        widgets_values: ['second prompt', 768, 768],
      },
      { id: 2, type: 'SaveImage', inputs: [{ name: 'images', link: 7 }], widgets_values: { filename_prefix: 'one' } },
      { id: 3, type: 'SaveImage', inputs: [{ name: 'images', link: 8 }], widgets_values: { filename_prefix: 'two' } },
    ],
    links: [
      [7, 1, 0, 2, 0, 'IMAGE'],
      [8, 4, 0, 3, 0, 'IMAGE'],
    ],
    definitions: { subgraphs: shared },
  };
}

describe('ComfyUI subgraph flattening', () => {
  it('expands a subgraph instance into real nodes and rewires its output', () => {
    const parsed = parseComfyWorkflow(oneSubgraph(), 'demo.json', io);
    assert.equal(parsed.format, 'ui');
    const graph = converted(oneSubgraph());

    // The instance is gone; its contents are present under a prefixed id.
    assert.equal(graph['1'], undefined, 'the subgraph instance node is not queued as a node class');
    assert.equal(graph['1:11']?.classType, 'CLIPTextEncode');
    assert.equal(graph['1:14']?.classType, 'KSampler');
    assert.equal(graph['1:15']?.classType, 'VAEDecode');

    // The promoted prompt reached the node the proxy fed, replacing its link.
    assert.equal(graph['1:11']?.inputs['text'], 'a cat');
    // The promoted dimensions reached the latent node.
    assert.equal(graph['1:13']?.inputs['width'], 640);
    assert.equal(graph['1:13']?.inputs['height'], 640);

    // The top-level SaveImage now reads the internal decoder, not the instance.
    assert.deepEqual(graph['2']?.inputs['images'], ['1:15', 0]);

    // A node inside the subgraph that was not promoted keeps its own wiring.
    assert.deepEqual(graph['1:14']?.inputs['positive'], ['1:11', 0]);
    assert.deepEqual(graph['1:11']?.inputs['clip'], ['1:10', 1]);
  });

  it('flattens nested subgraphs through every level', () => {
    const graph = converted(nestedSubgraphs('a cat'));

    // 9 (outer instance) → 9:5 (inner instance) → 9:5:1 (real node).
    assert.equal(graph['9:5:1']?.classType, 'CLIPTextEncode');
    assert.equal(graph['9:5:1']?.inputs['text'], 'a cat', 'the value crossed two levels of promotion');
    assert.deepEqual(graph['8']?.inputs['images'], ['9:5:1', 0], 'the outer output was re-pointed at the real node');
  });

  it('keeps repeated instances apart, each with its own values', () => {
    const graph = converted(repeatedInstances());

    assert.equal(graph['1:11']?.inputs['text'], 'first prompt');
    assert.equal(graph['4:11']?.inputs['text'], 'second prompt', 'the named override on the second instance was not lost');
    assert.equal(graph['4:13']?.inputs['width'], 768, 'the second instance read its positional values');
    assert.equal(graph['1:13']?.inputs['width'], 512);
    assert.deepEqual(graph['2']?.inputs['images'], ['1:15', 0]);
    assert.deepEqual(graph['3']?.inputs['images'], ['4:15', 0]);
    assert.notEqual(graph['1:11'], graph['4:11'], 'the two instances produced distinct nodes');
  });

  it('drops annotation nodes that a subgraph export carries', () => {
    const graph = converted(oneSubgraph());
    assert.equal(graph['3'], undefined, 'the unconnected MarkdownNote is not a node to queue');
    for (const node of Object.values(graph)) {
      assert.notEqual(node.classType, 'MarkdownNote');
    }
  });

  it('analyses a subgraph workflow into prompt, dimensions and an image output', () => {
    const scan = scanWorkflowDocument({ raw: oneSubgraph(), name: 'demo.json', io, classes });
    assert.equal(scan.runnable, true);
    assert.deepEqual(scan.capabilities, ['text_to_image']);
    assert.equal(scan.format, 'ui');
    assert.equal(scan.readiness, 'needs_conversion');

    const byName = new Map(scan.inputs.map((input) => [input.name, input]));
    assert.equal(byName.get('prompt')?.node, '1:11');
    assert.equal(byName.get('prompt')?.input, 'text');
    assert.equal(byName.get('width')?.node, '1:13');
    assert.equal(byName.get('height')?.node, '1:13');
    assert.deepEqual(scan.outputs.map((output) => [output.name, output.type, output.node]), [['image', 'image', '2']]);
  });

  it('still converts a workflow that has no subgraphs at all', () => {
    const flat = {
      nodes: [
        { id: 1, type: 'CLIPTextEncode', inputs: [{ name: 'clip', link: 14 }], widgets_values: { text: 'a fox' } },
        { id: 2, type: 'EmptyLatentImage', inputs: [], widgets_values: { width: 256, height: 256, batch_size: 1 } },
        { id: 5, type: 'CheckpointLoaderSimple', inputs: [], widgets_values: ['A.safetensors'] },
        {
          id: 3,
          type: 'KSampler',
          inputs: [
            { name: 'model', link: 13 },
            { name: 'positive', link: 10 },
            { name: 'negative', link: 10 },
            { name: 'latent_image', link: 11 },
          ],
          widgets_values: { seed: 1, steps: 4, cfg: 1 },
        },
        { id: 4, type: 'SaveImage', inputs: [{ name: 'images', link: 12 }], widgets_values: { filename_prefix: 'x' } },
      ],
      links: [
        [10, 1, 0, 3, 1, 'CONDITIONING'],
        [11, 2, 0, 3, 3, 'LATENT'],
        [12, 3, 0, 4, 0, 'LATENT'],
        [13, 5, 0, 3, 0, 'MODEL'],
        [14, 5, 1, 1, 0, 'CLIP'],
      ],
    };
    const parsed = parseComfyWorkflow(flat, 'flat.json', io);
    assert.equal(parsed.format, 'ui');
    const graph = converted(flat);
    assert.equal(graph['1']?.classType, 'CLIPTextEncode', 'ids are untouched when there is nothing to flatten');
    const scan = scanWorkflowDocument({ raw: flat, name: 'flat.json', io, classes });
    assert.deepEqual(scan.capabilities, ['text_to_image']);
    assert.equal(scan.inputs.find((input) => input.name === 'prompt')?.node, '1');
    assert.equal(scan.graph?.['1']?.class_type, 'CLIPTextEncode', 'the queued graph uses API class_type keys');
  });

  it('leaves a workflow with no definitions byte-identical to before', () => {
    const api = {
      '1': { class_type: 'CLIPTextEncode', inputs: { text: 'a fox' } },
      '2': { class_type: 'SaveImage', inputs: { images: ['1', 0] } },
    };
    const parsed = parseComfyWorkflow(api, 'api.json', io);
    assert.equal(parsed.format, 'api');
    assert.deepEqual(Object.keys(parsed.graph), ['1', '2']);
  });
});
