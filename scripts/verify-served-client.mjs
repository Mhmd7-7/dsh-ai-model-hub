/** Verify the existing DSH server serves the exact edited client artifact. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const base = process.env.DSH_VERIFY_URL || 'http://127.0.0.1:3080';
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 15_000);
try {
  const response = await fetch(base + '/plugins/events', { signal: controller.signal });
  assert.equal(response.status, 200);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let graph;
  while (!graph) {
    const chunk = await reader.read();
    assert.ok(!chunk.done, 'Plugin event stream ended before publishing a graph');
    buffer += decoder.decode(chunk.value, { stream: true });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop();
    for (const frame of frames) {
      const match = frame.match(/^data: ([^\r\n]+)/m);
      if (!match) continue;
      const message = JSON.parse(match[1]);
      if (message.type === 'graph') graph = message.graph;
    }
  }
  await reader.cancel();
  function locate(value) {
    if (!value || typeof value !== 'object') return null;
    if (typeof value.url === 'string' && value.url.includes('dsh-ai-model-hub/client.js')) return value.url;
    for (const child of Object.values(value)) {
      const found = locate(child);
      if (found) return found;
    }
    return null;
  }
  const url = locate(graph);
  assert.ok(url, 'The live module graph must include the AI model hub client');
  const served = await fetch(new URL(url, base), { signal: controller.signal });
  assert.equal(served.status, 200);
  const bytes = Buffer.from(await served.arrayBuffer());
  const source = await readFile(new URL('../dsh-plugin/client.js', import.meta.url));
  const hash = (data) => createHash('sha256').update(data).digest('hex');
  assert.ok(bytes.subarray(0, source.length).equals(source), 'The existing GUI must serve the exact edited client before its generated source-map trailer');
  const trailer = bytes.subarray(source.length).toString('utf8');
  assert.match(trailer, /^\s*;?\s*\/\/# sourceMappingURL=/, 'Only the generated source-map trailer may follow the client');
  console.log(JSON.stringify({ server: base, artifactURL: url, status: served.status, sourceSha256: hash(source), servedSha256: hash(bytes), matchesSource: true }, null, 2));
} finally {
  clearTimeout(timeout);
  controller.abort();
}
