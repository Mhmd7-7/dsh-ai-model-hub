/**
 * Tests for subscription policy at the execution boundary.
 *
 * The requirement this defends is blunt: *an agent must not bypass a rejection by
 * calling a subscription tool directly*. A routed `invoke_model` call is easy to
 * govern — the hub owns it. But the subscription plugin registers
 * `image_generate` and `video_generate` as tools in their own right, so an agent
 * that finds its routed request refused can simply call one. Prompt instructions
 * do not close that hole; a guard inside the dispatch pipeline does.
 *
 * Three properties are asserted here, and each of them is a way the hole opens:
 *
 * 1. A disabled backend is refused, whatever the call looks like.
 * 2. A call that does not name a provider is refused, because the tool's own
 *    internal fallback runs *past* the guard and could reach a disabled provider.
 * 3. A call that edits an image is judged against the *editing* option's policy,
 *    not the generation one — two switches that must stay independent.
 *
 * @module dsh-ai-model-hub/tests/subscription-policy.test
 */

import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';

import { CapabilityPolicy } from '../src/index.ts';
import {
  buildSubscriptionBackendOptions,
  canonicalToolName,
  optionIdForToolCall,
  providerArgument,
} from '../dsh-plugin/subscription/backends.ts';
import { checkSubscriptionToolCall, readProvider } from '../dsh-plugin/subscription/enforcement.ts';

/** A policy with the given per-capability decisions. */
function policyWith(order: Record<string, { id: string; enabled: boolean }[]>): CapabilityPolicy {
  const capabilities: Record<string, { order: { id: string; enabled: boolean }[] }> = {};
  for (const [capability, entries] of Object.entries(order)) capabilities[capability] = { order: entries };
  return new CapabilityPolicy({ version: 1, globalEnabled: true, capabilities });
}

/** A silent logger. */
const log = { debug: (): void => {}, info: (): void => {}, warn: (): void => {}, error: (): void => {} };

describe('subscription tool-name resolution', () => {
  it('recognizes the canonical names and the plugin-scoped aliases', () => {
    assert.equal(canonicalToolName('image_generate'), 'image_generate');
    assert.equal(canonicalToolName('dsh_subscriptions_image_generate'), 'image_generate');
    assert.equal(canonicalToolName('video_generate'), 'video_generate');
    assert.equal(canonicalToolName('dsh_subscriptions_video_generate'), 'video_generate');
  });

  it('leaves every other tool alone', () => {
    assert.equal(canonicalToolName('x_search'), undefined);
    assert.equal(canonicalToolName('invoke_model'), undefined);
    assert.equal(canonicalToolName('dsh_subscriptions_x_search'), undefined);
    // A different plugin's image tool must not be caught by the prefix rule.
    assert.equal(canonicalToolName('other_plugin_image_generate'), undefined);
  });

  it('maps a call to the option that would actually run', () => {
    assert.equal(optionIdForToolCall('image_generate', 'codex', false), 'codex_image');
    assert.equal(optionIdForToolCall('image_generate', 'grok', false), 'grok_image');
    assert.equal(optionIdForToolCall('image_generate', 'codex', true), 'codex_image_edit');
    assert.equal(optionIdForToolCall('image_generate', 'grok', true), 'grok_image_edit');
    assert.equal(optionIdForToolCall('video_generate', undefined, false), 'grok_video');
    assert.equal(optionIdForToolCall('dsh_subscriptions_image_generate', 'codex', false), 'codex_image');
    assert.equal(optionIdForToolCall('image_generate', undefined, false), undefined);
  });

  it('reads both spellings of the ChatGPT provider, and rejects a made-up one', () => {
    assert.equal(readProvider({ provider: 'gpt' }), 'codex');
    assert.equal(readProvider({ provider: 'codex' }), 'codex');
    assert.equal(readProvider({ provider: 'CHATGPT' }), 'codex');
    assert.equal(readProvider({ provider: 'grok' }), 'grok');
    assert.equal(readProvider({ provider: 'gemini' }), undefined);
    assert.equal(readProvider({}), undefined);
    assert.equal(readProvider({ provider: 7 }), undefined);
  });

  it('uses the same provider spelling on the wire as the guard expects', () => {
    // If these ever diverge, the guard would be checking one thing while the
    // adapter sent another — a policy hole with no failing test to catch it.
    assert.equal(providerArgument('codex'), 'gpt');
    assert.equal(providerArgument('grok'), 'grok');
    assert.equal(readProvider({ provider: providerArgument('codex') }), 'codex');
    assert.equal(readProvider({ provider: providerArgument('grok') }), 'grok');
  });
});

describe('direct subscription tool calls are governed by the capability policy', () => {
  it('allows a call on a backend the operator enabled', () => {
    const policy = policyWith({ text_to_image: [{ id: 'codex_image', enabled: true }] });
    const verdict = checkSubscriptionToolCall(policy, {
      name: 'image_generate',
      args: { provider: 'gpt', prompt: 'x' },
    });
    assert.equal(verdict.allowed, true);
  });

  it('refuses a call on a backend the operator disabled', () => {
    const policy = policyWith({ text_to_image: [{ id: 'codex_image', enabled: false }] });
    const verdict = checkSubscriptionToolCall(policy, {
      name: 'image_generate',
      args: { provider: 'gpt', prompt: 'x' },
    });
    assert.equal(verdict.allowed, false);
    if (verdict.allowed) return;
    assert.match(verdict.reason, /codex_image/);
    assert.match(verdict.reason, /text_to_image/);
    assert.match(verdict.reason, /not as a fallback/);
  });

  it('refuses a call that names no provider, because the tool would choose one', () => {
    // This is the specific hole: `image_generate` defaults to ChatGPT and falls
    // back to Grok internally. With Grok disabled, letting the tool decide means
    // letting it reach a backend the operator switched off.
    const policy = policyWith({
      text_to_image: [
        { id: 'codex_image', enabled: true },
        { id: 'grok_image', enabled: false },
      ],
    });
    const verdict = checkSubscriptionToolCall(policy, { name: 'image_generate', args: { prompt: 'x' } });
    assert.equal(verdict.allowed, false);
    if (verdict.allowed) return;
    assert.match(verdict.reason, /names no provider/);
    assert.match(verdict.reason, /switched off/);
  });

  it('judges an edit against the editing option, not the generation option', () => {
    // Text-to-image on, image-to-image off: an edit must be refused even though
    // the generation option is permitted, and vice versa.
    const policy = policyWith({
      text_to_image: [{ id: 'codex_image', enabled: true }],
      image_to_image: [{ id: 'codex_image_edit', enabled: false }],
    });
    const generate = checkSubscriptionToolCall(policy, {
      name: 'image_generate',
      args: { provider: 'gpt', prompt: 'x' },
    });
    assert.equal(generate.allowed, true);
    const edit = checkSubscriptionToolCall(policy, {
      name: 'image_generate',
      args: { provider: 'gpt', prompt: 'x', referenceImages: [{ attachmentId: 'sha256:' + 'a'.repeat(64) }] },
    });
    assert.equal(edit.allowed, false);
    if (edit.allowed) return;
    assert.match(edit.reason, /codex_image_edit/);
  });

  it('refuses everything while the global switch is off', () => {
    const policy = new CapabilityPolicy({
      version: 1,
      globalEnabled: false,
      capabilities: {
        text_to_image: { order: [{ id: 'codex_image', enabled: true }] },
        video_generation: { order: [{ id: 'grok_video', enabled: true }] },
      },
    });
    for (const call of [
      { name: 'image_generate', args: { provider: 'gpt', prompt: 'x' } },
      { name: 'video_generate', args: { prompt: 'x' } },
    ]) {
      const verdict = checkSubscriptionToolCall(policy, call);
      assert.equal(verdict.allowed, false);
      if (verdict.allowed) continue;
      assert.match(verdict.reason, /globally/);
    }
  });

  it('refuses a grok video call when the video option is disabled', () => {
    const policy = policyWith({ video_generation: [{ id: 'grok_video', enabled: false }] });
    const verdict = checkSubscriptionToolCall(policy, { name: 'video_generate', args: { prompt: 'x' } });
    assert.equal(verdict.allowed, false);
    if (verdict.allowed) return;
    assert.match(verdict.reason, /grok_video/);
  });

  it('leaves tools the hub holds no policy over entirely alone', () => {
    const policy = policyWith({});
    for (const name of ['x_search', 'dsh_subscriptions_x_search', 'invoke_model', 'read_file']) {
      const verdict = checkSubscriptionToolCall(policy, { name, args: {} });
      assert.equal(verdict.allowed, true, `${name} must not be governed by the capability policy`);
    }
  });

  it('refuses a disabled backend even when the arguments are shaped oddly', () => {
    const policy = policyWith({ text_to_image: [{ id: 'codex_image', enabled: false }] });
    for (const args of [null, 'nonsense', 42, [], { provider: 'gpt', referenceImages: 'not-an-array' }]) {
      const verdict = checkSubscriptionToolCall(policy, { name: 'image_generate', args });
      assert.equal(verdict.allowed, false, `args ${JSON.stringify(args)} must not slip through`);
    }
  });
});

describe('subscription backend options', () => {
  /** Tool access that reports one canonical tool as registered. */
  function accessFor(registered: readonly string[], calls: string[] = []): Parameters<typeof buildSubscriptionBackendOptions>[0] {
    return {
      resolveTool: (canonical) =>
        registered.includes(canonical)
          ? canonical
          : registered.includes(`dsh_subscriptions_${canonical}`)
            ? `dsh_subscriptions_${canonical}`
            : undefined,
      callTool: async (name, args) => {
        calls.push(name);
        void args;
        return { paths: [] };
      },
    };
  }

  it('offers every backend when the tools are registered, all disabled by default', () => {
    const options = buildSubscriptionBackendOptions(accessFor(['image_generate', 'video_generate']), log);
    assert.deepEqual(
      options.map((entry) => entry.facts.id),
      ['codex_image', 'codex_image_edit', 'grok_image', 'grok_image_edit', 'grok_video'],
    );
    for (const entry of options) {
      assert.equal(entry.executable, true);
      assert.equal(entry.facts.source, 'subscription');
      assert.equal(entry.facts.catalogEnabled, false, 'a remote backend never claims to be enabled by the catalog');
      assert.ok(
        entry.facts.capabilities.length === 1,
        'each backend declares exactly the one capability it genuinely serves',
      );
      assert.equal(entry.facts.executable, true);
    }
  });

  it('marks a backend unsupported when its tool is not registered, with a reason', () => {
    const options = buildSubscriptionBackendOptions(accessFor([]), log);
    for (const entry of options) {
      assert.equal(entry.executable, false);
      assert.match(entry.reason ?? '', /not registered/);
      assert.equal(entry.facts.executable, false);
    }
  });

  it('resolves the plugin-scoped alias when the canonical name is taken', () => {
    const options = buildSubscriptionBackendOptions(accessFor(['dsh_subscriptions_image_generate']), log);
    const image = options.find((entry) => entry.facts.id === 'codex_image');
    assert.equal(image?.executable, true);
    const video = options.find((entry) => entry.facts.id === 'grok_video');
    assert.equal(video?.executable, false, 'a tool with no alias registered is still unavailable');
  });

  it('reports readiness per backend rather than inventing a login check', async () => {
    const options = buildSubscriptionBackendOptions(accessFor(['image_generate', 'video_generate']), log);
    const image = options.find((entry) => entry.facts.id === 'codex_image');
    assert.ok(image?.option.status !== undefined);
    const state = await image.option.status();
    assert.equal(state.state, 'ready');
    assert.match(state.reason, /only knowable by calling it/);
  });

  it('reports a missing tool as unsupported rather than as merely stopped', async () => {
    const options = buildSubscriptionBackendOptions(accessFor([]), log);
    const image = options.find((entry) => entry.facts.id === 'codex_image');
    assert.ok(image?.option.status !== undefined);
    const state = await image.option.status();
    assert.equal(state.state, 'unsupported');
  });

  it('refuses to invoke a backend whose tool is not registered', async () => {
    const options = buildSubscriptionBackendOptions(accessFor([]), log);
    const image = options.find((entry) => entry.facts.id === 'codex_image');
    assert.ok(image !== undefined);
    await assert.rejects(
      () =>
        image.option.invoke(
          { capability: 'text_to_image', prompt: 'x' },
          {
            capability: 'text_to_image',
            prompt: 'x',
            inputs: [],
            options: {},
            artifacts: {} as never,
            signal: new AbortController().signal,
            log: () => {},
          },
        ),
      /not registered/,
    );
  });

  it('needs a prompt for generation and an image for an edit', async () => {
    const options = buildSubscriptionBackendOptions(accessFor(['image_generate']), log);
    const context = {
      capability: 'text_to_image' as const,
      inputs: [],
      options: {},
      artifacts: {} as never,
      signal: new AbortController().signal,
      log: () => {},
    };
    const generate = options.find((entry) => entry.facts.id === 'codex_image');
    assert.ok(generate !== undefined);
    await assert.rejects(
      () => generate.option.invoke({ capability: 'text_to_image' }, context),
      /needs a prompt/,
    );

    const edit = options.find((entry) => entry.facts.id === 'codex_image_edit');
    assert.ok(edit !== undefined);
    await assert.rejects(
      () =>
        edit.option.invoke(
          { capability: 'image_to_image', prompt: 'make it blue' },
          { ...context, capability: 'image_to_image', prompt: 'make it blue' },
        ),
      /supplied no image input/,
    );
  });
});
