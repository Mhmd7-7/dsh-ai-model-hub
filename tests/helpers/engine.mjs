/**
 * A tiny HTTP engine used by the startup and readiness tests.
 *
 * It exists to make the parts of the lifecycle that actually break observable.
 * A fake `spawnProcess` would let the tests assert that *a* process was asked for;
 * this asserts what actually happened: the port is bound immediately while the
 * readiness route stays 503 for a while, generation submitted before readiness is
 * refused and counted, concurrent requests that race a single startup can be
 * counted, and the process leaves evidence behind when it dies.
 *
 * Flags:
 *
 * - `--port=N` (or `AIMH_TEST_PORT`) — the port to bind. `0` asks the OS for one.
 * - `--ready-after=MS` — how long `/ready` answers 503 before 200.
 * - `--ready-as-markup` — answer `/ready` with an HTML page and status 200, which
 *   is what a half-loaded web UI does and what a readiness check must not accept.
 * - `--never-ready` — never answer `/ready` with 200.
 * - `--exit-immediately` — exit at once, simulating a launch that dies on startup.
 *
 * Routes:
 *
 * - `GET /live` — always 200, so "listening" and "ready" are distinguishable.
 * - `GET /ready` — 503 until ready, then `{"ready":true}`.
 * - `GET /stats` — `{ starts, generations, rejected }`.
 * - `POST /generate` — 503 (and `rejected++`) before readiness, otherwise a small
 *   PNG and `generations++`.
 */

import { createServer } from 'node:http';
import { deflateSync } from 'node:zlib';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const withValue = args.find((entry) => entry.startsWith(`--${name}=`));
  if (withValue !== undefined) return withValue.slice(name.length + 3);
  return args.includes(`--${name}`) ? '1' : fallback;
};

const port = Number(flag('port', process.env['AIMH_TEST_PORT'] ?? '0'));
const readyAfterMs = Number(flag('ready-after', process.env['AIMH_TEST_READY_AFTER_MS'] ?? '0'));
const readyAsMarkup = flag('ready-as-markup', process.env['AIMH_TEST_READY_AS_MARKUP'] ?? '0') === '1';
const neverReady = flag('never-ready', process.env['AIMH_TEST_NEVER_READY'] ?? '0') === '1';

if (args.includes('--exit-immediately')) {
  process.stderr.write('fixture engine: exiting immediately as instructed\n');
  process.exit(3);
}

const startedAt = Date.now();
const stats = { starts: 1, generations: 0, rejected: 0 };

/** A minimal 1x1 PNG, so a "generated image" is a real image. */
function onePixelPng() {
  const crcTable = [];
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crcTable[n] = c >>> 0;
  }
  const crc = (buffer) => {
    let c = 0xffffffff;
    for (const byte of buffer) c = crcTable[(c ^ byte) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type, data) => {
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length, 0);
    const body = Buffer.concat([Buffer.from(type, 'latin1'), data]);
    const checksum = Buffer.alloc(4);
    checksum.writeUInt32BE(crc(body), 0);
    return Buffer.concat([length, body, checksum]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(1, 0);
  header.writeUInt32BE(1, 4);
  header[8] = 8;
  header[9] = 6;
  const raw = Buffer.from([0, 0x12, 0x34, 0x56, 0xff]);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const png = onePixelPng();

/** Whether the engine is ready to serve generation. */
function isReady() {
  if (neverReady) return false;
  return Date.now() - startedAt >= readyAfterMs;
}

const server = createServer((req, res) => {
  const url = req.url ?? '/';
  if (url === '/live') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ listening: true }));
    return;
  }
  if (url === '/ready') {
    if (isReady()) {
      if (readyAsMarkup) {
        res.writeHead(200, { 'content-type': 'text/html' });
        res.end('<html><body>engine ready</body></html>');
        return;
      }
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ready: true }));
      return;
    }
    res.writeHead(503, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ready: false }));
    return;
  }
  if (url === '/stats') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify(stats));
    return;
  }
  if (url === '/generate') {
    if (!isReady()) {
      stats.rejected += 1;
      res.writeHead(503, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'not ready' }));
      return;
    }
    stats.generations += 1;
    res.writeHead(200, { 'content-type': 'application/json' });
    // The A1111/`http_json` image shape: a plain list of base64 strings. The
    // adapter is the shipping one, so the fixture has to speak its protocol.
    res.end(JSON.stringify({ images: [png.toString('base64')], parameters: {}, info: '{}' }));
    return;
  }
  res.writeHead(404, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
});

server.listen(port, '127.0.0.1', () => {
  const address = server.address();
  const bound = typeof address === 'object' && address !== null ? address.port : port;
  process.stdout.write(`fixture engine listening on ${String(bound)}\n`);
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    server.close(() => process.exit(0));
    // Do not let a held connection stall the shutdown the tests are measuring.
    setTimeout(() => process.exit(0), 500).unref();
  });
}
