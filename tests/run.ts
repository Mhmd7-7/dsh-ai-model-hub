/**
 * Test entry point.
 *
 * `node --test` normally spawns one child process per test file. That is the
 * right default, but it cannot run under a restricted sandbox (a child spawned
 * with piped stdio is refused with `EPERM`), and it also prevents the plugin's
 * self-referential `dsh-ai-model-hub` imports from resolving, because a bare
 * specifier resolved from `process.cwd()` never reaches the package's own
 * `exports` map.
 *
 * Importing every suite into one process fixes both: Node's test runner collects
 * the `describe`/`it` registrations from these imports and runs them in-process,
 * so resolution happens from a real file inside the package.
 *
 * Run it with:
 *
 *   node --test --experimental-test-isolation=none tests/run.ts
 *
 * @module dsh-ai-model-hub/tests/run
 */

import './catalog.test.ts';
import './artifacts.test.ts';
import './router.test.ts';
import './runtime.test.ts';
import './integration.test.ts';
import './plugin.test.ts';
import './schema.test.ts';
import './openai.test.ts';
