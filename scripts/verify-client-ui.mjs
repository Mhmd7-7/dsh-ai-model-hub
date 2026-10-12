/**
 * Render the Local models component with live, read-only API snapshots.
 * This is a component preview, NOT verification of the authenticated DSH shell.
 * No replacement server, dependency installation, credential access, model
 * invocation, or policy write is performed. All browser API requests are mocked
 * from the GET snapshots; every non-GET/HEAD request is blocked.
 *
 * Run: node scripts/verify-client-ui.mjs
 * Optional: DSH_VERIFY_URL, DSH_VERIFY_PLAYWRIGHT, DSH_VERIFY_PROFILE,
 * DSH_VERIFY_THEME_CLIENT, DSH_VERIFY_CLIENT, DSH_VERIFY_OUTPUT.
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const base = process.env.DSH_VERIFY_URL || 'http://127.0.0.1:3080';
const profile = process.env.DSH_VERIFY_PROFILE || path.join(process.env.DSH_HOME || path.join(homedir(), '.dsh'), 'profiles', 'web');
const clientPath = process.env.DSH_VERIFY_CLIENT || path.join(root, 'dsh-plugin', 'client.js');
const output = process.env.DSH_VERIFY_OUTPUT || path.join(root, 'artifacts', 'ui');

/** Resolve a path to a module or file, reporting what to set when it is absent. */
function resolveFirst(variable, candidates, what) {
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      return require.resolve(candidate);
    } catch {
      // Try the next candidate; the message below names the variable to set.
    }
  }
  throw new Error(
    `${what} was not found. Set ${variable} to its path. Tried:\n  ` +
      candidates.map((candidate) => String(candidate)).join('\n  '),
  );
}

// Playwright: the harness ships no browser automation, and a machine may have it
// under the Node driver or under a Python install. Either path must be given
// explicitly rather than guessed, because a wrong guess silently verifies nothing.
const playwrightPath = resolveFirst(
  'DSH_VERIFY_PLAYWRIGHT',
  [
    process.env.DSH_VERIFY_PLAYWRIGHT,
    'playwright',
    'playwright-core',
  ],
  'Playwright',
);
// The theme plugin owns the --dsw-* tokens the component must inherit; reading its
// stylesheet is what makes the preview use the host's real palette. The package
// exposes the browser half as the `./client` subpath.
const themePath = resolveFirst(
  'DSH_VERIFY_THEME_CLIENT',
  [
    process.env.DSH_VERIFY_THEME_CLIENT,
    (() => {
      try {
        return require.resolve('@deepseek-ai/dsh-client-ui-theme/client', { paths: [profile] });
      } catch {
        return undefined;
      }
    })(),
  ],
  'The DSH theme client',
);
const { chromium } = require(playwrightPath);

async function snapshot(endpoint) {
  const response = await fetch(new URL(endpoint, base), { signal: AbortSignal.timeout(60_000) });
  assert.equal(response.status, 200, `Read-only ${endpoint} must return 200`);
  return response.json();
}

const [inventory, priorities, themeSource, clientBytes] = await Promise.all([
  snapshot('/dsh-ai-model-hub/inventory'),
  snapshot('/dsh-ai-model-hub/priorities'),
  readFile(themePath, 'utf8'),
  readFile(clientPath),
]);
const clientSha256 = createHash('sha256').update(clientBytes).digest('hex');
const cssLiteral = themeSource.match(/var design_platform_css_default = ("(?:[^"\\]|\\.)*");/);
assert.ok(cssLiteral, 'Installed DSH theme must expose its design-platform stylesheet');
const themeCss = JSON.parse(cssLiteral[1]);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const results = [];
const browserErrors = [];
const blockedWrites = [];
const snapshots = new Map([
  ['/dsh-ai-model-hub/inventory', inventory],
  ['/dsh-ai-model-hub/priorities', priorities],
]);

async function checkWidth(page, label) {
  const geometry = await page.evaluate(() => ({
    viewport: window.innerWidth,
    document: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
    page: document.querySelector('.mh-page')?.getBoundingClientRect().width,
    offenders: [...document.querySelectorAll('.mh-page *')].filter((element) => {
      const box = element.getBoundingClientRect();
      return box.width > 0 && (box.right > window.innerWidth + 1 || box.left < -1);
    }).slice(0, 12).map((element) => ({
      tag: element.tagName, class: element.className,
      text: element.textContent?.slice(0, 100),
      right: element.getBoundingClientRect().right,
    })),
  }));
  assert.ok(geometry.document <= geometry.viewport + 1 && geometry.body <= geometry.viewport + 1,
    `${label}: horizontal document overflow: ${JSON.stringify(geometry)}`);
  return geometry;
}

try {
  for (const size of [{ name: 'wide', width: 1120, height: 1000 }, { name: 'mobile', width: 390, height: 844 }]) {
    for (const theme of ['light', 'dark']) {
      const context = await browser.newContext({ viewport: size, colorScheme: theme, reducedMotion: 'reduce' });
      await context.route('**/*', async (route) => {
        const request = route.request();
        const url = new URL(request.url());
        if (!['GET', 'HEAD'].includes(request.method())) {
          blockedWrites.push({ method: request.method(), pathname: url.pathname });
          await route.abort('blockedbyclient');
          return;
        }
        const data = snapshots.get(url.pathname);
        if (data !== undefined && url.origin === new URL(base).origin) {
          await route.fulfill({ status: 200, contentType: 'application/json; charset=utf-8', body: JSON.stringify(data) });
          return;
        }
        // Browser rendering is intentionally isolated from all other endpoints.
        await route.abort('blockedbyclient');
      });
      const page = await context.newPage();
      page.on('pageerror', (error) => browserErrors.push(`${size.name}/${theme}: ${error.message}`));
      // Public JSON gives this preview the real server origin without authenticating.
      await page.goto(new URL('/dsh-ai-model-hub/priorities', base).href);
      await page.setContent(`<!doctype html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
        ${themeCss}
        html { min-width: 0; } body { margin: 0; padding: 24px; background: var(--dsw-alias-bg-base, #fff); color: var(--dsw-alias-label-primary, #202123); font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
        #root { max-width: 1040px; margin: 0 auto; min-width: 0; }
        @media(max-width:540px) { body { padding: 14px; } }
      </style></head><body ${theme === 'dark' ? 'data-ds-dark-theme' : ''}><div id="root"></div></body></html>`);
      await page.addScriptTag({ path: path.join(profile, 'node_modules', 'react', 'umd', 'react.production.min.js') });
      await page.addScriptTag({ path: path.join(profile, 'node_modules', 'react-dom', 'umd', 'react-dom.production.min.js') });
      await page.evaluate(() => {
        const jsx = (type, props, key) => React.createElement(type, key === undefined ? props : { ...props, key });
        window.__ModuleLoader__ = {
          load(registration) {
            if (registration.id !== 'dsh-ai-model-hub') throw new Error('Unexpected plugin registration');
            window.__previewPlugin = registration.factory((specifier) => {
              if (specifier === 'react') return React;
              if (specifier === 'react/jsx-runtime') return { jsx, jsxs: jsx, Fragment: React.Fragment };
              throw new Error(`Unexpected module: ${specifier}`);
            });
          },
        };
      });
      await page.addScriptTag({ path: clientPath });
      await page.evaluate(() => {
        window.__previewPlugin.apply({
          effect(callback) { return callback(); },
          slots: {
            inject(name, callback) {
              if (name !== 'settings.section') throw new Error('Unexpected slot');
              const result = callback();
              if (result && typeof result.next === 'function') { for (const item of result) void item; }
            },
            register(options, Component) {
              if (options.id !== 'dsh-ai-model-hub') throw new Error('Unexpected settings section');
              window.__previewComponent = Component;
              return () => {};
            },
          },
        });
        if (!window.__previewComponent) throw new Error('Local models settings component was not registered');
        ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(window.__previewComponent));
      });
      await page.locator('.mh-page').waitFor();
      await page.waitForFunction(() => document.querySelector('.mh-editor')?.getAttribute('aria-busy') === 'false');
      await page.waitForFunction(() => !document.querySelector('#mh-panel-inventory')?.textContent.includes('scanning…'));
      assert.equal(await page.locator('#mh-panel-inventory').count(), 1);
      assert.equal(await page.locator('#mh-panel-priorities').count(), 1);
      assert.equal(await page.locator('#mh-panel-priorities').getAttribute('hidden'), '');
      await page.evaluate(() => {
        window.__inventoryNode = document.querySelector('#mh-panel-inventory');
        window.__prioritiesNode = document.querySelector('#mh-panel-priorities');
      });
      const inventoryGeometry = await checkWidth(page, `${size.name}/${theme}/inventory`);
      const inventoryScreenshot = path.join(output, `local-models-inventory-${size.name}-${theme}.png`);
      await page.screenshot({ path: inventoryScreenshot, fullPage: true });
      await page.getByRole('button', { name: 'Edit priorities', exact: true }).click();
      assert.equal(await page.getByRole('tab', { name: 'Priorities', exact: true }).getAttribute('aria-selected'), 'true');
      assert.equal(await page.locator('#mh-panel-inventory').getAttribute('hidden'), '');
      assert.equal(await page.locator('#mh-panel-priorities').getAttribute('hidden'), null);
      assert.equal(await page.evaluate(() => window.__inventoryNode === document.querySelector('#mh-panel-inventory') &&
        window.__prioritiesNode === document.querySelector('#mh-panel-priorities')), true, 'Tab switching must preserve panel DOM');

      const cardCount = await page.locator('.mh-card').count();
      assert.equal(cardCount, priorities.capabilities.length, 'All capabilities are represented');
      const search = page.getByRole('searchbox', { name: 'Search capabilities and providers' });
      await search.fill('no-match-for-local-models-verification');
      assert.equal(await page.locator('.mh-card').count(), 0, 'Search must filter unmatched capabilities');
      await search.fill('image');
      const matchedCards = await page.locator('.mh-card').count();
      assert.ok(matchedCards > 0 && matchedCards < cardCount, 'Search must narrow to matching capabilities');
      await search.fill('');
      assert.equal(await page.locator('.mh-card').count(), cardCount, 'Clearing search restores capabilities');
      if (!priorities.globalEnabled && priorities.capabilities.some((entry) =>
        entry.options.some((option) => typeof option.configuredEnabled !== 'boolean'))) {
        assert.ok((await page.locator('.mh-option').count()) > 0, 'Guard scenario must contain options');
        assert.equal(await page.locator('.mh-option button:not([disabled])').count(), 0,
          'Older globally-disabled hosts must not allow edits using unknown saved permissions');
        const explanation = page.getByRole('status').filter({ hasText: 'does not expose saved per-option choices' });
        assert.equal(await explanation.count(), 1, 'Unknown-permissions explanation is shown');
        await page.getByRole('button', { name: 'Enable capabilities', exact: true }).click();
        assert.equal(await page.locator('.mh-option button:not([disabled])').count(), 0,
          'Draft global enable must not unlock edits before the actual policy is read');
        await page.getByRole('button', { name: 'Cancel', exact: true }).click();
        assert.equal(await page.getByRole('button', { name: 'Enable capabilities', exact: true }).count(), 1);
      }
      const priorityGeometry = await checkWidth(page, `${size.name}/${theme}/priorities`);
      const priorityScreenshot = path.join(output, `local-models-priorities-${size.name}-${theme}.png`);
      await page.screenshot({ path: priorityScreenshot, fullPage: true });
      // Keyboard tabs and preserved search state are read-only interactions.
      await search.fill('image');
      await page.getByRole('tab', { name: 'Priorities', exact: true }).press('Home');
      assert.equal(await page.getByRole('tab', { name: 'Inventory', exact: true }).getAttribute('aria-selected'), 'true');
      await page.getByRole('tab', { name: 'Inventory', exact: true }).press('End');
      assert.equal(await search.inputValue(), 'image', 'Search state survives tab switches');
      results.push({ size: size.name, theme, cardCount, matchedCards,
        inventoryGeometry, priorityGeometry, screenshots: [inventoryScreenshot, priorityScreenshot] });
      await context.close();
    }
  }
  assert.deepEqual(blockedWrites, [], 'No policy write must even be attempted');
  assert.deepEqual(browserErrors, [], 'No browser page errors');
  const report = {
    verification: 'Standalone Local models component preview using read-only snapshots from the existing DSH server. Not an authenticated full-shell GUI screenshot.',
    server: base,
    client: clientPath,
    clientSha256,
    themeTokens: Object.fromEntries([
      '--dsw-alias-bg-base', '--dsw-alias-bg-layer-1', '--dsw-alias-bg-layer-2',
      '--dsw-alias-state-success-primary',
    ].map((name) => [name, [...themeCss.matchAll(new RegExp(name + ':([^;}]*)', 'g'))].map((match) => match[1])])),
    policy: { globalEnabled: priorities.globalEnabled, revision: priorities.revision },
    checks: ['Edit priorities navigation', 'Both panels remain mounted', 'Search filtering',
      'Keyboard tabs and state preservation', 'Global-disable edit protection',
      'No horizontal document overflow', 'No browser page errors', 'No attempted policy writes'],
    blockedWrites, browserErrors, results,
  };
  await writeFile(path.join(output, 'verification-report.json'), JSON.stringify(report, null, 2) + '\n', 'utf8');
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
}
