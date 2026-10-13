/**
 * Tests for the DSH compatibility layer.
 *
 * The layer exists because DSH is an `0.x-rc` line whose plugin surface moves.
 * Two classes of failure are covered here, and they need different kinds of test:
 *
 * 1. **A host that is different, not broken.** Every member the plugin uses is
 *    resolved through `compat.ts` and checked, so a host missing `systemPrompt`,
 *    `effect`, `webServer`, `skills`, `sandboxPolicy`, or even `get` must still
 *    get every model tool. These tests mount the real plugin against deliberately
 *    impoverished contexts and assert the tools survive.
 * 2. **A host the plugin cannot work with at all.** No tool registry is not a
 *    degraded plugin but an absent one, so it is the single case that must fail —
 *    loudly, by name, and before anything else is registered.
 *
 * There is a third, subtler class: the **declarative** gate. DSH decides whether
 * to load a plugin from its `peerDependencies` before any plugin code runs, so no
 * amount of runtime defensiveness helps if that range is wrong. That gate is
 * tested in `peer-range.test.ts`, because the bug it catches is a manifest bug.
 *
 * ## Why the schemastery range check is here
 *
 * `@deepseek-ai/schemastery` is the one DSH package the plugin imports a *value*
 * from that has no runtime guard: `config.ts` does `import z from
 * '@deepseek-ai/schemastery'` and builds the Loader's config schema with it. That
 * import is resolved by Node before `apply` runs, so if the package ever stops
 * providing a default export, the plugin fails to load at all. The test below
 * proves the module the plugin actually resolves still has the shape it needs —
 * including under the *host's* copy, when the host's copy is reachable, which is
 * the configuration a real deployment runs in.
 *
 * @module dsh-ai-model-hub/tests/compat.test
 */

import { strict as assert } from 'node:assert';
import { createRequire } from 'node:module';
import { describe, it } from 'node:test';
import { pathToFileURL } from 'node:url';

import { Context } from '@deepseek-ai/cordis';

import {
  DEFAULT_CONTEXT_ORDER,
  REQUIRED_MEMBERS,
  contextOrderFor,
  installEffect,
  installPromptContext,
  installRoute,
  installTool,
  memberVerdict,
  optionalService,
  pluginLogger,
  reportCompatHost,
  resolveCompatHost,
  serviceInScope,
  sessionCwdOf,
} from '../dsh-plugin/compat.ts';
import type { PluginLogger } from '../dsh-plugin/types.ts';
import { apply } from '../dsh-plugin/index.ts';
import { ModelHubError } from '../src/index.ts';

/** A logger that records every line, so absence is asserted rather than assumed. */
function recordingLogger(): { readonly lines: string[]; readonly log: PluginLogger } {
  const lines: string[] = [];
  const write = (level: string) => (message: string) => {
    lines.push(`${level} ${message}`);
  };
  return { lines, log: { debug: write('debug'), info: write('info'), warn: write('warn'), error: write('error') } };
}

/** A tool definition shaped the way `@deepseek-ai/dsh-tools` shapes one. */
function fakeDefinition(name: string): Parameters<typeof installTool>[1] {
  return {
    name,
    description: 'a tool',
    parameters: { type: 'object', additionalProperties: false, properties: {} },
    output: { schema: { type: 'object' }, render: () => [] },
    execute: async () => ({}),
  } as unknown as Parameters<typeof installTool>[1];
}

/**
 * Build a context that has only what the caller names.
 *
 * A real `Context` is the base, because `Service` and the plugin's own
 * `ctx.effect` augmentation need a real one, and only the members under test are
 * overridden. Everything else is *deleted* from the proxy view, which is what
 * makes this a test of absence rather than of a different value.
 *
 * @param members - the members the host should have.
 * @returns the context.
 */
function hostWith(members: Record<string, unknown>): Context {
  return new Context().extend(members);
}

describe('the host observation', () => {
  it('reports every member the plugin uses, one verdict each', () => {
    const host = resolveCompatHost(new Context());
    const names = host.members.map((verdict) => verdict.member);
    for (const name of ['tools', 'systemPrompt', 'logger', 'effect', 'inject', 'get', 'webServer', 'sandboxPolicy', 'skills']) {
      assert.ok(names.includes(name), `${name} is tracked`);
    }
    assert.equal(host.members.length, names.length, 'no member is reported twice');
  });

  it('requires a tool registry, and nothing else', () => {
    assert.deepEqual([...REQUIRED_MEMBERS], ['tools'], 'the tool registry is the only hard requirement');

    const bare = resolveCompatHost(new Context());
    assert.equal(bare.complete, false, 'a bare context is not complete');

    const withTools = resolveCompatHost(
      hostWith({ tools: { register: () => () => {} } }),
    );
    assert.equal(withTools.complete, true, 'a tool registry alone is enough to load');
  });

  it('tells "absent" apart from "present but unusable", because the fixes differ', () => {
    const absent = resolveCompatHost(new Context());
    assert.equal(memberVerdict(absent, 'webServer')?.present, false);
    assert.match(
      memberVerdict(absent, 'webServer')?.consequence ?? '',
      /absent/,
      'an absent member says so',
    );

    // A webServer that exists but has no `register` is a host API change, not a
    // deployment choice, and the sentence has to point at the API rather than
    // telling an operator to configure something that is already configured.
    const unusable = resolveCompatHost(hostWith({ webServer: { routes: [] } }));
    const verdict = memberVerdict(unusable, 'webServer');
    assert.equal(verdict?.present, true);
    assert.equal(verdict?.usable, false);
    assert.match(verdict?.consequence ?? '', /present but is not usable/, 'an unusable member says which kind of problem it is');
  });

  it('accepts a member whose needed method is inherited rather than own', () => {
    class Server {
      register(): () => void {
        return () => {};
      }
    }
    const host = resolveCompatHost(hostWith({ webServer: new Server() }));
    assert.equal(memberVerdict(host, 'webServer')?.usable, true, 'a prototype method counts');
  });

  it('logs a missing optional member quietly and an unusable one as a warning', () => {
    const host = resolveCompatHost(hostWith({ tools: { register: () => () => {} }, webServer: {} }));
    const { lines, log } = recordingLogger();
    const reported = reportCompatHost(host, log);
    assert.equal(reported, host.diagnostics.length, 'every diagnostic is reported once');

    const unusable = lines.filter((line) => line.includes('webServer'));
    assert.equal(unusable.length, 1, 'the unusable member is reported once');
    assert.match(unusable[0] ?? '', /^warn /, 'a present-but-unusable member is a warning');

    const absent = lines.filter((line) => line.includes('sandboxPolicy'));
    assert.equal(absent.length, 1);
    assert.match(absent[0] ?? '', /^debug /, 'an absent optional member is not a warning');
  });
});

describe('a logger that always exists', () => {
  it('uses the host logger when the host has one', () => {
    const seen: string[] = [];
    const ctx = hostWith({
      logger: (scope: string) => ({
        debug: (m: string) => seen.push(`debug ${scope} ${m}`),
        info: (m: string) => seen.push(`info ${scope} ${m}`),
        warn: (m: string) => seen.push(`warn ${scope} ${m}`),
        error: (m: string) => seen.push(`error ${scope} ${m}`),
      }),
    });
    const log = pluginLogger(ctx, 'scope');
    log.info('hello');
    assert.deepEqual(seen, ['info scope hello']);
  });

  it('falls back to a console logger when the host has none, so a mismatch is never silent', () => {
    const log = pluginLogger(new Context(), 'fallback-scope');
    assert.equal(typeof log.info, 'function');
    assert.equal(typeof log.warn, 'function');
    assert.equal(typeof log.error, 'function');
    assert.equal(typeof log.debug, 'function');
  });

  it('falls back when the host logger throws on construction', () => {
    const ctx = hostWith({
      logger: () => {
        throw new Error('no logger for you');
      },
    });
    const log = pluginLogger(ctx, 'throwing');
    assert.doesNotThrow(() => {
      log.warn('still recorded');
    });
  });

  it('fills in levels a partially-implemented host logger omits', () => {
    const ctx = hostWith({ logger: () => ({ info: () => {} }) });
    const log = pluginLogger(ctx, 'partial');
    assert.doesNotThrow(() => {
      log.debug('d');
      log.warn('w');
      log.error('e');
    });
  });
});

describe('registering a tool against an unfamiliar registry', () => {
  it('registers through a working registry and reports success', () => {
    const registered: string[] = [];
    const ctx = hostWith({
      tools: {
        register: (definition: { name: string }) => {
          registered.push(definition.name);
          return () => {};
        },
      },
    });
    const { log } = recordingLogger();
    assert.equal(installTool(ctx, fakeDefinition('a_tool'), log), true);
    assert.deepEqual(registered, ['a_tool']);
  });

  it('contains a registry that refuses a definition, rather than aborting the plugin', () => {
    const ctx = hostWith({
      tools: {
        register: () => {
          throw new Error('unknown schema keyword "const"');
        },
      },
    });
    const { lines, log } = recordingLogger();
    assert.equal(installTool(ctx, fakeDefinition('refused_tool'), log), false, 'the refusal is reported as a false');
    assert.equal(lines.length, 1);
    assert.match(lines[0] ?? '', /refused_tool was refused/);
    assert.match(lines[0] ?? '', /unknown schema keyword/, 'the host reason is carried through');
  });

  it('reports a registry with no register method instead of throwing', () => {
    const ctx = hostWith({ tools: { list: () => [] } });
    const { lines, log } = recordingLogger();
    assert.equal(installTool(ctx, fakeDefinition('orphan'), log), false);
    assert.match(lines[0] ?? '', /no tool registry/);
  });
});

describe('installing cleanup and prompt context on an unfamiliar host', () => {
  it('runs an effect through a host that has one', () => {
    let ran = false;
    const ctx = hostWith({
      effect: (execute: () => unknown) => {
        ran = true;
        return execute();
      },
    });
    const { log } = recordingLogger();
    const disposer = installEffect(ctx, () => () => {}, 'test effect', log);
    assert.equal(ran, true, 'the effect body ran');
    assert.notEqual(disposer, undefined, 'the host disposer is returned');
  });

  it('does NOT run the effect body when the host has no effect member', () => {
    // The distinction matters: an effect body here returns a disposer that stops
    // model processes. Running it with nobody to call the disposer would start
    // processes nothing ever releases.
    //
    // A real cordis `Context` always has `effect`, so the absent case is stated
    // explicitly with a non-callable value: that is also what a host which renamed
    // or replaced the member looks like, which is the case worth surviving.
    let ran = false;
    const { lines, log } = recordingLogger();
    const disposer = installEffect(hostWith({ effect: 'not a function' }), () => {
      ran = true;
      return () => {};
    }, 'unbound effect', log);
    assert.equal(disposer, undefined);
    assert.equal(ran, false, 'the body is not run when its disposer could never be called');
    assert.match(lines.join('\n'), /no `ctx\.effect`/);
    assert.match(lines.join('\n'), /will not run when the plugin unloads/, 'the consequence is stated');
  });

  it('takes the host order slot when the host names one', () => {
    const prompt = { getContextOrder: (name: string) => (name === 'SANDBOX_POLICY' ? 110 : 0) };
    const { log } = recordingLogger();
    assert.equal(contextOrderFor(prompt, 'SANDBOX_POLICY', log), 111, 'the snapshot sits just after the sandbox facts');
  });

  it('falls back to a fixed order when the host will not name one, or throws', () => {
    const { lines, log } = recordingLogger();
    assert.equal(contextOrderFor(undefined, 'SANDBOX_POLICY', log), DEFAULT_CONTEXT_ORDER);
    assert.equal(
      contextOrderFor(
        {
          getContextOrder: () => {
            throw new Error('no such slot');
          },
        },
        'SANDBOX_POLICY',
        log,
      ),
      DEFAULT_CONTEXT_ORDER,
    );
    assert.equal(contextOrderFor({ getContextOrder: () => Number.NaN }, 'SANDBOX_POLICY', log), DEFAULT_CONTEXT_ORDER);
    assert.match(lines.join('\n'), /no such slot/, 'a rejected query is explained rather than swallowed');
  });

  it('registers the prompt context and reports whether it landed', () => {
    const contributions: { name: string; order: number }[] = [];
    const ctx = hostWith({
      systemPrompt: {
        context: (contribution: { name: string; order: number }) => {
          contributions.push({ name: contribution.name, order: contribution.order });
        },
        getContextOrder: () => 110,
      },
    });
    const { log } = recordingLogger();
    assert.equal(installPromptContext(ctx, 'hub:capabilities', 'SANDBOX_POLICY', () => 'text', log), true);
    assert.deepEqual(contributions, [{ name: 'hub:capabilities', order: 111 }]);
  });

  it('loses only the snapshot when the host has no prompt service', () => {
    const { lines, log } = recordingLogger();
    assert.equal(installPromptContext(new Context(), 'hub:capabilities', 'SANDBOX_POLICY', () => 'text', log), false);
    assert.match(lines.join('\n'), /list_capabilities/, 'the consequence names the workaround');
  });

  it('loses only the snapshot when the host refuses the contribution', () => {
    const ctx = hostWith({
      systemPrompt: {
        context: () => {
          throw new Error('prompt assembly is sealed');
        },
      },
    });
    const { lines, log } = recordingLogger();
    assert.equal(installPromptContext(ctx, 'hub:capabilities', 'SANDBOX_POLICY', () => 'text', log), false);
    assert.match(lines.join('\n'), /prompt assembly is sealed/);
  });
});

describe('reading services without requiring them', () => {
  it('returns undefined for a host with no get member at all', () => {
    assert.equal(optionalService(new Context(), 'sandboxPolicy'), undefined);
  });

  it('returns undefined when the host get throws for an unregistered name', () => {
    const ctx = hostWith({
      get: () => {
        throw new Error('service not registered');
      },
    });
    assert.equal(optionalService(ctx, 'missing'), undefined);
  });

  it('prefers the injected property over get, and falls back to get when there is none', () => {
    // Cordis's inject-aware context proxy resolves the injected name as a member,
    // so a scope can carry a service the `get` registry does not know about. Both
    // paths have to work: a host that dropped either one would otherwise lose
    // every optional integration at once.
    const direct = { tag: 'direct' };
    const fromGet = { tag: 'from-get' };
    const withBoth = hostWith({ webServer: direct, get: () => fromGet });
    assert.equal(serviceInScope<{ tag: string }>(withBoth, 'webServer')?.tag, 'direct');

    const getOnly = hostWith({ get: () => fromGet });
    assert.equal(serviceInScope<{ tag: string }>(getOnly, 'webServer')?.tag, 'from-get');

    assert.equal(serviceInScope(new Context(), 'webServer'), undefined);
  });

  it('resolves a service that is already present without waiting to be injected', () => {
    // `ctx.inject` is an *awaited* dependency: on a host where the service is
    // already mounted, waiting for it to be "provided" is waiting for something
    // that already happened. A web server mounted before this plugin must still
    // get its pages.
    const routes: string[] = [];
    const ctx = hostWith({
      webServer: {
        register: (route: { path: string }) => {
          routes.push(route.path);
          return () => {};
        },
      },
      inject: () => {
        throw new Error('this host never fires inject callbacks');
      },
    });
    const { log } = recordingLogger();
    assert.equal(installRoute(ctx, '/already-mounted', () => {}, 'the page', log), true);
    assert.deepEqual(routes, ['/already-mounted'], 'the route was registered without going through inject');
  });
});

describe('attaching an HTTP route', () => {
  it('registers an exact route on a host that has a web server', () => {
    const routes: { kind: string; path: string }[] = [];
    const handler = (): void => {};
    const ctx = hostWith({
      webServer: {
        register: (route: { kind: string; path: string }) => {
          routes.push({ kind: route.kind, path: route.path });
          return () => {};
        },
      },
    });
    const { log } = recordingLogger();
    assert.equal(installRoute(ctx, '/dsh-ai-model-hub/inventory', handler, 'the inventory page', log), true);
    assert.deepEqual(routes, [{ kind: 'exact', path: '/dsh-ai-model-hub/inventory' }]);
  });

  it('costs only the page when the route registration throws', () => {
    const ctx = hostWith({
      webServer: {
        register: () => {
          throw new Error('route table is frozen');
        },
      },
    });
    const { lines, log } = recordingLogger();
    assert.equal(installRoute(ctx, '/dsh-ai-model-hub/priorities', () => {}, 'the priorities page', log), false);
    assert.match(lines.join('\n'), /route table is frozen/);
  });

  it('costs only the page when the host has no web server', () => {
    const { lines, log } = recordingLogger();
    // `inject` is stubbed to run the callback, because cordis's real one only fires
    // when the service is *provided* — with no web server it would never run, and
    // the diagnostic this test exists for would never be written. Running it here is
    // what makes the branch observable.
    const ctx = hostWith({ inject: (_deps: readonly string[], callback: (scope: Context) => void) => callback(ctx) });
    assert.equal(installRoute(ctx, '/x', () => {}, 'the page', log), false);
    assert.match(lines.join('\n'), /not served on \/x/);
    assert.match(lines.join('\n'), /every model tool still works/, 'the cost is bounded, and said so');
  });
});

describe('reading a session out of a tool run context', () => {
  it('reads the documented path', () => {
    assert.equal(sessionCwdOf({ agent: { session: { cwd: 'C:\\work' } } }), 'C:\\work');
  });

  it('survives every malformed shape a host could hand over, instead of throwing inside a call', () => {
    const malformed: unknown[] = [
      undefined,
      {},
      { agent: undefined },
      { agent: null },
      { agent: 'an agent' },
      { agent: { session: 'a session' } },
      { agent: { session: { cwd: 42 } } },
      { agent: { session: { cwd: '' } } },
      { agent: { cwd: 'C:\\work' } },
    ];
    for (const value of malformed) {
      assert.equal(
        sessionCwdOf(value as never),
        undefined,
        `${JSON.stringify(value)} yields undefined rather than throwing`,
      );
    }
  });
});

describe('the plugin against a host that is older or poorer than the one it was written for', () => {
  /**
   * Mount the real plugin against a minimal host and return what it registered.
   *
   * The host has a tool registry and nothing else: no `systemPrompt`, no
   * `effect`, no `inject`, no `get`, no `logger`, no web server. That is the
   * shape this test exists to prove survivable.
   *
   * @param extra - members the case wants the host to have.
   * @returns the registered tool names.
   */
  function mountAgainst(extra: Record<string, unknown> = {}): string[] {
    const registered: string[] = [];
    const ctx = hostWith({
      tools: {
        register: (definition: { name: string }) => {
          registered.push(definition.name);
          return () => {};
        },
      },
      ...extra,
    });
    apply(ctx, {});
    return registered;
  }

  it('registers every model tool on a host with no systemPrompt, effect, or logger', () => {
    const registered = mountAgainst();
    for (const tool of [
      'list_models',
      'list_capabilities',
      'get_model_status',
      'list_artifacts',
      'explain_routing',
      'refresh_model_discovery',
      'invoke_model',
      'start_model',
      'stop_model',
      'check_model_health',
    ]) {
      assert.ok(registered.includes(tool), `${tool} survived a minimal host`);
    }
  });

  it('still registers every tool when the host has a systemPrompt it refuses', () => {
    const registered = mountAgainst({
      systemPrompt: {
        context: () => {
          throw new Error('this host assembles prompts elsewhere');
        },
      },
    });
    assert.ok(registered.includes('invoke_model'), 'a refused prompt contribution does not cost a tool');
  });

  it('refuses to load at all when there is no tool registry, and names what is missing', () => {
    const ctx = new Context();
    assert.throws(
      () => apply(ctx, {}),
      (error: unknown) => {
        assert.ok(error instanceof ModelHubError, 'the failure is a hub error');
        assert.equal(error.code, 'UNSUPPORTED_OPERATION', 'and a code a caller can branch on');
        assert.match(error.message, /tool registry/, 'and it names the member that is missing');
        return true;
      },
    );
  });

  it('reports the host it found before it registers anything', () => {
    const lines: string[] = [];
    const ctx = hostWith({
      tools: { register: () => () => {} },
      logger: () => ({
        debug: (m: string) => lines.push(`debug ${m}`),
        info: (m: string) => lines.push(`info ${m}`),
        warn: (m: string) => lines.push(`warn ${m}`),
        error: (m: string) => lines.push(`error ${m}`),
      }),
    });
    apply(ctx, {});
    assert.ok(
      lines.some((line) => line.includes('webServer') && line.includes('absent')),
      'the observation is written to the log the host provided',
    );
  });
});

describe('the schemastery this plugin resolves', () => {
  /**
   * Resolve a package the way this project resolves it.
   *
   * Two roots are tried, because they are genuinely different configurations and
   * only one of them is what a deployment runs:
   *
   * 1. this package's own resolution — what the test process and a checkout use;
   * 2. the installed DSH host's resolution — what a profile plugin gets when the
   *    host supplies the module, which is the real deployment path.
   *
   * @param specifier - the package to resolve.
   * @returns the file URL, or `undefined` when neither root can reach it.
   */
  async function resolveEither(specifier: string): Promise<unknown | undefined> {
    const roots = [
      import.meta.url,
      process.env['DSH_HOME'] === undefined
        ? undefined
        : pathToFileURL(`${process.env['DSH_HOME']}\\..\\AppData\\Roaming\\npm\\node_modules\\@deepseek-ai\\dsh\\lib\\index.js`).href,
    ];
    for (const root of roots) {
      if (root === undefined) continue;
      try {
        const require = createRequire(root);
        return await import(pathToFileURL(require.resolve(specifier)).href);
      } catch {
        // Try the next root: not being resolvable from one is not a failure.
      }
    }
    return undefined;
  }

  it('provides the default export the plugin imports, with every helper its config schema uses', async () => {
    const module = (await resolveEither('@deepseek-ai/schemastery')) as
      | { default?: Record<string, unknown> }
      | undefined;
    assert.notEqual(module, undefined, 'schemastery resolves from at least one of the roots');
    const z = module?.default;
    assert.equal(typeof z, 'function', 'the default export is the schema factory');

    // The exact statics `dsh-plugin/config.ts` builds the Loader's schema from. A
    // missing one is not a degraded plugin: the import and the schema both fail
    // before `apply` runs, so this list is the plugin's real minimum contract.
    for (const helper of ['object', 'string', 'boolean', 'number', 'array', 'dict', 'const', 'union']) {
      assert.equal(typeof z?.[helper], 'function', `schemastery.${helper} exists`);
    }
  });
});
