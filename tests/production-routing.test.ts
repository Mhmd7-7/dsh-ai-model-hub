/** Real installed WebServer/Cordis dispatch against the production plugin entry. */
import { strict as assert } from 'node:assert';
import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { mkdtemp, rm, writeFile, readFile } from 'node:fs/promises';
import ts from 'typescript';
import { homedir, tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { it } from 'node:test';

const require = createRequire(import.meta.url);
const candidates: string[] = [];
if (process.env['DSH_TEST_INSTALL_ROOT']) candidates.push(process.env['DSH_TEST_INSTALL_ROOT']);
try { candidates.push(dirname(require.resolve('@deepseek-ai/dsh/package.json'))); } catch { /* optional host integration */ }
candidates.push(join(homedir(), 'AppData/Roaming/npm/node_modules/@deepseek-ai/dsh'));
const installed = candidates.map((root) => join(root, 'node_modules/@deepseek-ai')).find((root) =>
  existsSync(join(root, 'dsh-host-webserver/lib/index.js')) && existsSync(join(root, 'cordis/lib/index.js')));
const available = installed !== undefined;

it('production scan artifacts match their TypeScript sources', async () => {
  const modules = ['index', 'config', 'compat', 'inventory', 'scan', 'probe-budget', 'priorities-route'];
  for (const name of modules) {
    const sourceUrl = new URL(`../dsh-plugin/${name}.ts`, import.meta.url);
    const source = await readFile(sourceUrl, 'utf8');
    const emitted = ts.transpileModule(source, { fileName: sourceUrl.pathname,
      compilerOptions: { target: ts.ScriptTarget.ES2023, module: ts.ModuleKind.ESNext,
        rewriteRelativeImportExtensions: true, removeComments: false } }).outputText;
    const built = await readFile(new URL(`../lib/dsh-plugin/${name}.js`, import.meta.url), 'utf8');
    assert.equal(built.replace(/\r\n/g, '\n'), emitted.replace(/\r\n/g, '\n'), `run npm run build: stale production ${name}`);
  }
});

for (const delayed of [false, true]) {
  it(`production plugin routes register with real installed Cordis/WebServer (${delayed ? 'delayed' : 'already mounted'} service) and teardown`,
    { skip: !available, timeout: 15000 }, async () => {
      const { Context } = await import(pathToFileURL(join(installed!, 'cordis/lib/index.js')).href);
      const { WebServer } = await import(pathToFileURL(join(installed!, 'dsh-host-webserver/lib/index.js')).href);
      // Package main uses lib, not TS source. Build before this integration check.
      const plugin = await import(new URL('../lib/dsh-plugin/index.js', import.meta.url).href);
      const root = await mkdtemp(join(tmpdir(), 'aimh-production-routing-'));
      const catalog = join(root, 'models.json');
      await writeFile(catalog, JSON.stringify({ version: '1', models: [] }));
      const lines: string[] = [];
      const logger = Object.assign(() => logger, {
        debug: (message: unknown) => lines.push(String(message)),
        info: (message: unknown) => lines.push(String(message)),
        warn: (message: unknown) => lines.push(String(message)),
        error: (message: unknown) => lines.push(String(message)),
      });
      const ctx = new Context().extend({ logger });
      ctx.provide('tools', { register: () => () => {} });
      ctx.provide('systemPrompt', { context: () => () => {}, getContextOrder: () => 100 });
      let fiber: any;
      let webFiber: any;
      try {
        if (!delayed) { webFiber = ctx.plugin(WebServer, { host: '127.0.0.1', port: 0 }); await webFiber.await(); }
        fiber = ctx.plugin(plugin, { configPath: catalog, policyPath: join(root, 'policy.json'),
          healthIntervalMs: 0, idleSweepIntervalMs: 0, probeResources: false, integrateSubscriptions: false });
        await fiber.await();
        if (delayed) { webFiber = ctx.plugin(WebServer, { host: '127.0.0.1', port: 0 }); await webFiber.await(); }
        for (const runtime of ctx.registry.values()) for (const child of runtime.fibers) await child.await();
        const server = ctx.get('webServer');
        assert.ok(server?.port, lines.join('\n'));
        const base = `http://127.0.0.1:${server.port}`;
        const priorities = await fetch(base + '/dsh-ai-model-hub/priorities');
        assert.equal(priorities.status, 200, lines.join('\n'));
        assert.ok(Array.isArray(((await priorities.json()) as { capabilities: unknown }).capabilities));
        // Inventory GET can probe local machine: no mocks and no manufactured success.
        const inventory = await fetch(base + '/dsh-ai-model-hub/inventory');
        assert.equal(inventory.status, 200, lines.join('\n'));
        assert.ok(Array.isArray(((await inventory.json()) as { engines: unknown }).engines));
        const scan = await fetch(base + '/dsh-ai-model-hub/inventory?scan=1');
        assert.equal(scan.status, 200, 'initial mount and Scan use existing scan=1 query');
        assert.ok(Array.isArray(((await scan.json()) as { resources: unknown }).resources));
        assert.equal((await fetch(base + '/dsh-ai-model-hub/inventory')).status, 200, 'plain Refresh also dispatches');
        const method = await fetch(base + '/dsh-ai-model-hub/inventory', { method: 'POST' });
        assert.equal(method.status, 405);
        assert.equal((await fetch(base + '/api/dsh-ai-model-hub/inventory')).status, 404,
          'WebServer has no automatic API prefix');
        await fiber.dispose();
        for (const runtime of ctx.registry.values()) for (const child of runtime.fibers) await child.await();
        assert.equal((await fetch(base + '/dsh-ai-model-hub/inventory')).status, 404);
        assert.equal((await fetch(base + '/dsh-ai-model-hub/priorities')).status, 404);
      } finally {
        if (fiber) await fiber.dispose();
        if (webFiber) await webFiber.dispose();
        await ctx.fiber.dispose();
        const target = join(tmpdir(), root.slice(tmpdir().length).replace(/^[/\\]/, ''));
        assert.equal(target, root, 'verify temporary cleanup target');
        await rm(root, { recursive: true, force: true });
      }
    });
}
