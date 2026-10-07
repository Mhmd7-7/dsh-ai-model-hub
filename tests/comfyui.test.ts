/** Adapter utilities and configured-workflow contract checks.
 * The end-to-end configured HTTP adapter is exercised in comfy-workflow-boundary.test.ts.
 */
import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import { readPngDimensions } from '../src/adapters/comfyui.ts';
import { readContract, validateComfyGraph } from '../src/comfy/workflow.ts';
import { renderMockPng } from '../src/index.ts';

const graph = {
  '42': { class_type: 'CLIPTextEncode', inputs: { text: 'template' } },
  '81': { class_type: 'SaveImage', inputs: { images: ['42', 0] } },
};
const config = { bindings: { prompt: { node: '42', input: 'text' } }, outputs: { image: { node: '81', type: 'image' } } };

describe('comfyui configured workflow contract', () => {
  it('validates explicit prompt and output bindings without guessing node identifiers', () => {
    const contract = readContract(config, ['text_to_image']);
    assert.deepEqual(contract.bindings.prompt, { node: '42', input: 'text' });
    assert.deepEqual(contract.outputs.image, { node: '81', type: 'image' });
    assert.doesNotThrow(() => validateComfyGraph(graph, contract, new Set(['CLIPTextEncode', 'SaveImage'])));
  });
  it('rejects missing nodes and unavailable classes before invocation', () => {
    const contract = readContract(config, ['text_to_image']);
    assert.throws(() => validateComfyGraph(graph, contract, new Set(['SaveImage'])), /required ComfyUI node class is unavailable/);
    assert.throws(() => validateComfyGraph({ '42': graph['42'] }, contract), /output "image" references missing node "81"/);
  });
  it('reads actual image dimensions from a PNG artifact', () => {
    assert.deepEqual(readPngDimensions(renderMockPng(320, 192, 'fixture')), { width: 320, height: 192 });
  });
});
