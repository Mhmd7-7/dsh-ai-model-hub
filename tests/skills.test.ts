/**
 * Tests for the skill the plugin ships.
 *
 * The provider is driven directly rather than through a skill registry, because
 * the contract under test is the provider's: the candidate it advertises, the body
 * it loads, and the contained failure it reports when the file beside it is
 * unreadable or malformed. `@deepseek-ai/dsh-skill` is deliberately not a
 * dependency of this project, so a registry stand-in would test the stand-in.
 *
 * The grammar and the rank are the two places where this provider can disagree
 * with the real registry in a way that only shows up inside a running agent, so
 * both are asserted here against the values `@deepseek-ai/dsh-skill` publishes.
 *
 * @module dsh-ai-model-hub/tests/skills.test
 */

import { strict as assert } from 'node:assert';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, it } from 'node:test';

import {
  BUNDLED_SKILL_RANK,
  PROVIDER_NAME,
  SKILL_FILE,
  createModelHubSkillProvider,
} from '../dsh-plugin/skills.ts';
import type { PluginLogger, SkillCandidate } from '../dsh-plugin/types.ts';

/** The kebab-case grammar `@deepseek-ai/dsh-skill` enforces on every candidate. */
const SKILL_NAME_GRAMMAR = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

/** A logger that records what the provider said without printing it. */
function recordingLogger(): { lines: string[]; logger: PluginLogger } {
  const lines: string[] = [];
  const logger: PluginLogger = {
    debug: (message) => lines.push(`debug ${message}`),
    info: (message) => lines.push(`info ${message}`),
    warn: (message) => lines.push(`warn ${message}`),
    error: (message) => lines.push(`error ${message}`),
  };
  return { lines, logger };
}

/** Write a skill file into a throwaway directory and return its path. */
async function fixture(content: string): Promise<{ file: string; cleanup: () => Promise<void> }> {
  const root = await mkdtemp(join(tmpdir(), 'aimh-skill-'));
  const file = join(root, 'SKILL.md');
  await writeFile(file, content, 'utf8');
  return { file, cleanup: () => rm(root, { recursive: true, force: true }) };
}

/** The shipped skill file's own frontmatter, read without the provider's parser. */
function frontmatterOf(source: string): Map<string, string> {
  const block = /^---\r?\n([\s\S]*?)\r?\n---/.exec(source)?.[1];
  assert.notEqual(block, undefined, 'the shipped skill file must carry frontmatter');
  const fields = new Map<string, string>();
  for (const line of (block ?? '').split(/\r?\n/)) {
    const separator = line.indexOf(':');
    if (separator > 0) fields.set(line.slice(0, separator).trim(), line.slice(separator + 1).trim());
  }
  return fields;
}

describe('bundled dsh-ai-model-hub skill', () => {
  it('advertises exactly one candidate that satisfies the registry contract', async () => {
    const { lines, logger } = recordingLogger();
    const provider = createModelHubSkillProvider(logger);

    const candidates = await provider.list({});
    assert.equal(candidates.length, 1);
    const candidate = candidates[0] as SkillCandidate;

    assert.match(candidate.name, SKILL_NAME_GRAMMAR);
    assert.ok(candidate.description.length > 0);
    assert.equal(candidate.provider, PROVIDER_NAME);
    assert.equal(candidate.source, 'bundled');
    assert.equal(candidate.rank, BUNDLED_SKILL_RANK);
    assert.equal(candidate.invocation.modelInvocable, true);
    assert.equal(candidate.invocation.userInvocable, true);
    assert.equal(candidate.path, SKILL_FILE);
    assert.deepEqual(candidate.resourceBase, { kind: 'directory', path: join(SKILL_FILE, '..', '..', '..') });
    assert.deepEqual(lines, [], 'a healthy provider warns about nothing');
  });

  it('takes the name and description from the skill file, not from the code', async () => {
    const { logger } = recordingLogger();
    const provider = createModelHubSkillProvider(logger);

    const source = await readFile(SKILL_FILE, 'utf8');
    const frontmatter = frontmatterOf(source);
    const candidate = (await provider.list({}))[0] as SkillCandidate;

    assert.equal(candidate.name, frontmatter.get('name'));
    assert.equal(candidate.description, frontmatter.get('description'));
    assert.equal(candidate.whenToUse, frontmatter.get('whenToUse'));
  });

  it('loads a body without the frontmatter and without the resource base leaking into it', async () => {
    const { logger } = recordingLogger();
    const provider = createModelHubSkillProvider(logger);

    const candidate = (await provider.list({}))[0] as SkillCandidate;
    const loaded = await provider.get(candidate, {});
    assert.notEqual(loaded, undefined);

    const content = loaded?.content ?? '';
    assert.equal(content.startsWith('---'), false, 'the frontmatter must not reach the model');
    assert.equal(content, content.trim(), 'the body is trimmed');
    assert.ok(content.length > 800, `the body should be substantive, got ${content.length} characters`);
    assert.match(content, /^# dsh-ai-model-hub/m);
    for (const tool of ['invoke_model', 'list_capabilities', 'list_models', 'explain_routing']) {
      assert.ok(content.includes(tool), `the body should name the ${tool} tool`);
    }
    assert.equal(loaded?.name, candidate.name);
    assert.equal(loaded?.description, candidate.description);
  });

  it('refuses a locator it did not issue', async () => {
    const { logger } = recordingLogger();
    const provider = createModelHubSkillProvider(logger);
    const candidate = (await provider.list({}))[0] as SkillCandidate;

    const foreign: SkillCandidate = { ...candidate, locator: 'somewhere-else.md' };
    assert.equal(await provider.get(foreign, {}), undefined);
  });

  it('offers nothing and warns once when the skill file is missing', async () => {
    const root = await mkdtemp(join(tmpdir(), 'aimh-skill-missing-'));
    const { lines, logger } = recordingLogger();
    const provider = createModelHubSkillProvider(logger, join(root, 'absent.md'));

    assert.deepEqual(await provider.list({}), []);
    assert.equal(await provider.get({} as SkillCandidate, {}), undefined);
    assert.equal(lines.length, 1, 'one failure is reported once, not on every collection');
    assert.match(lines[0] ?? '', /^warn .*not offered/);

    await rm(root, { recursive: true, force: true });
  });

  it('contains a malformed skill file instead of degrading the catalog', async () => {
    const cases: readonly (readonly [string, string, RegExp])[] = [
      ['no frontmatter', '# Just a body\n', /frontmatter/],
      ['empty body', '---\nname: bundled-skill\ndescription: something\n---\n', /body .* is empty/],
      [
        'non-kebab-case name',
        '---\nname: Bundled_Skill\ndescription: something\n---\n\n# Body\n',
        /kebab-case/,
      ],
      ['missing description', '---\nname: bundled-skill\n---\n\n# Body\n', /description/],
      [
        'folded description',
        '---\nname: bundled-skill\ndescription: >\n  two lines\n---\n\n# Body\n',
        /multi-line scalar/,
      ],
      [
        'indented continuation',
        '---\nname: bundled-skill\ndescription: first\n  second\n---\n\n# Body\n',
        /flat "key: value"/,
      ],
    ];

    for (const [label, content, message] of cases) {
      const { file, cleanup } = await fixture(content);
      const { lines, logger } = recordingLogger();
      const provider = createModelHubSkillProvider(logger, file);

      assert.deepEqual(await provider.list({}), [], `${label}: offers nothing`);
      assert.equal(lines.length, 1, `${label}: warns exactly once`);
      assert.match(lines[0] ?? '', message, `${label}: names the problem`);
      await cleanup();
    }
  });

  it('honours an aborted signal on both entry points', async () => {
    const { logger } = recordingLogger();
    const provider = createModelHubSkillProvider(logger);
    const controller = new AbortController();
    controller.abort();

    const options = { signal: controller.signal };
    assert.deepEqual(await provider.list(options), []);
    const candidate = (await provider.list({}))[0] as SkillCandidate;
    assert.equal(await provider.get(candidate, options), undefined);
  });
});
