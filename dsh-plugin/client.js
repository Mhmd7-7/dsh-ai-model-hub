/**
 * The browser half: a "Local models" section in DSH's settings.
 *
 * This file is a **plain side-effect script**, not a module. DSH loads plugin
 * clients through `window.__ModuleLoader__`, and the contract is explicit: no
 * top-level `import` and no top-level `export`, because React and the jsx runtime
 * are handed to the factory's `require` at runtime. That is also why there is no
 * bundler here — hand-written JavaScript satisfies the contract exactly, and a
 * build step would be a second toolchain for one file.
 *
 * The page owns no data. It reads `GET /dsh-ai-model-hub/inventory`, which the host
 * half serves from the live catalog plus filesystem and endpoint probes, so the
 * page can never disagree with what the hub would actually do.
 *
 * Registration target: the `settings.section` slot declared by
 * `@deepseek-ai/dsh-client-ui-settings`, which is how every feature page joins the
 * settings shell.
 */
window.__ModuleLoader__.load({
  id: 'dsh-ai-model-hub-plugin',
  factory: (require) => {
    const module = { exports: {} };

    const React = require('react');
    const { jsx: h } = require('react/jsx-runtime');
    const { useCallback, useEffect, useState } = React;

    /** The host route this page reads. */
    const ENDPOINT = '/dsh-ai-model-hub/inventory';

    /** Monospace stack for paths and commands; the system stack, so nothing is fetched. */
    const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Courier New", monospace';

    const text = {
      primary: 'var(--dsw-alias-label-primary)',
      secondary: 'var(--dsw-alias-label-secondary)',
      tertiary: 'var(--dsw-alias-label-tertiary)',
      border: 'var(--dsw-alias-border-l2)',
      ok: 'var(--dsw-alias-state-ok-primary)',
      error: 'var(--dsw-alias-state-error-primary)',
    };

    /** Format a byte count the same way the host does, for engine-reported sizes. */
    function formatBytes(bytes) {
      if (typeof bytes !== 'number' || !isFinite(bytes)) return '';
      if (bytes < 1024) return bytes + ' B';
      const units = ['KiB', 'MiB', 'GiB', 'TiB'];
      let value = bytes / 1024;
      let unit = 0;
      while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
      }
      return value.toFixed(value >= 10 ? 0 : 1) + ' ' + units[unit];
    }

    /** A small pill, used for engine state and model availability. */
    function Pill(props) {
      const tone = props.tone === 'ok' ? text.ok : props.tone === 'error' ? text.error : text.tertiary;
      return h('span', {
        style: {
          border: '1px solid ' + (props.tone === 'ok' ? tone : text.border),
          color: tone,
          borderRadius: '999px',
          padding: '1px 8px',
          fontSize: '11px',
          whiteSpace: 'nowrap',
        },
        children: props.children,
      });
    }

    /** One labelled value row, with path-like values in monospace and selectable. */
    function Row(props) {
      if (props.value === undefined || props.value === null || props.value === '') return null;
      return h('div', {
        style: { fontSize: '12px', lineHeight: '18px', color: text.secondary, display: 'flex', gap: '6px' },
        children: [
          h('span', { style: { color: text.tertiary, minWidth: '104px', flexShrink: 0 }, children: props.label }),
          h('span', {
            style: {
              color: text.primary,
              wordBreak: 'break-all',
              userSelect: 'text',
              fontFamily: props.mono ? MONO : undefined,
            },
            children: props.value,
          }),
        ],
      });
    }

    /** One engine card: where it is, whether it runs, and what is inside it. */
    function EngineCard(props) {
      const engine = props.engine;
      const tone = engine.configured === false ? 'muted' : engine.running ? 'ok' : 'error';
      const state = engine.configured === false ? 'not configured' : engine.running ? 'running' : 'stopped';

      return h('div', {
        style: {
          border: '1px solid ' + text.border,
          borderRadius: '12px',
          padding: '12px 14px',
          display: 'flex',
          flexDirection: 'column',
          gap: '6px',
        },
        children: [
          h('div', {
            style: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
            children: [
              h('span', { style: { fontSize: '13px', fontWeight: 500, color: text.primary }, children: engine.name }),
              h(Pill, { tone, children: state }),
              engine.startable
                ? h(Pill, { tone: 'ok', children: 'hub can start it' })
                : h(Pill, { tone: 'muted', children: 'external' }),
            ],
          }),
          h(Row, { label: 'engine', value: engine.id + ' · ' + engine.adapter }),
          h(Row, { label: 'listens on', value: engine.endpoint, mono: true }),
          engine.installPath
            ? h(Row, { label: 'installed at', value: engine.installPath, mono: true })
            : h(Row, {
                label: 'installed at',
                value: 'not found — looked in ' + (engine.checkedPaths || []).join(', '),
              }),
          engine.launch
            ? h(Row, {
                label: 'starts with',
                value: [engine.launch.command].concat(engine.launch.args || []).join(' '),
                mono: true,
              })
            : null,
          engine.launch && engine.launch.cwd
            ? h(Row, { label: 'working dir', value: engine.launch.cwd, mono: true })
            : null,
          engine.statusDetail ? h(Row, { label: 'probe', value: engine.statusDetail }) : null,
          engine.models.length > 0
            ? h('div', {
                style: { fontSize: '12px', lineHeight: '20px', display: 'flex', gap: '6px' },
                children: [
                  h('span', {
                    style: { color: text.tertiary, minWidth: '104px', flexShrink: 0 },
                    children: 'models (' + engine.models.length + ')',
                  }),
                  h('span', {
                    style: { display: 'flex', flexWrap: 'wrap', gap: '4px' },
                    children: engine.models.map((model) =>
                      h('span', {
                        key: model.id,
                        title: model.detail || '',
                        style: {
                          border: '1px solid ' + text.border,
                          borderRadius: '6px',
                          padding: '0 6px',
                          fontFamily: MONO,
                          fontSize: '11px',
                          color: text.primary,
                          userSelect: 'text',
                        },
                        children: model.detail ? model.id + '  (' + model.detail + ')' : model.id,
                      }),
                    ),
                  }),
                ],
              })
            : h(Row, { label: 'models', value: engine.modelsSource }),
          ...(engine.storePaths || [])
            .filter((store) => store.exists)
            .map((store) =>
              h('div', {
                key: store.path,
                style: { fontSize: '12px', lineHeight: '18px', display: 'flex', gap: '6px' },
                children: [
                  h('span', {
                    style: { color: text.tertiary, minWidth: '104px', flexShrink: 0 },
                    children: store.label,
                  }),
                  h('span', {
                    style: { color: text.primary, wordBreak: 'break-all', userSelect: 'text', fontFamily: MONO },
                    children:
                      store.path +
                      (store.files && store.files.length > 0
                        ? '  —  ' + store.files.map((file) => file.name + (file.sizeBytes ? ' (' + formatBytes(file.sizeBytes) + ')' : '')).join(', ')
                        : ''),
                  }),
                ],
              }),
            ),
          engine.servedModelIds && engine.servedModelIds.length > 0
            ? h(Row, { label: 'hub uses', value: engine.servedModelIds.join(', '), mono: true })
            : null,
        ],
      });
    }

    /** The settings section itself. */
    function LocalModelsSection() {
      const [state, setState] = useState({ status: 'loading', data: null, error: '' });

      const load = useCallback((probe) => {
        setState((previous) => ({ ...previous, status: previous.data ? 'refreshing' : 'loading' }));
        fetch(ENDPOINT + (probe ? '?probe=1' : ''))
          .then((response) => {
            if (!response.ok) throw new Error('HTTP ' + response.status);
            return response.json();
          })
          .then((data) => setState({ status: 'ready', data, error: '' }))
          .catch((error) =>
            setState({ status: 'error', data: null, error: String((error && error.message) || error) }),
          );
      }, []);

      useEffect(() => {
        load(true);
      }, [load]);

      const button = (label, onClick, disabled) =>
        h('button', {
          type: 'button',
          onClick,
          disabled: disabled === true,
          style: {
            border: '1px solid ' + text.border,
            background: 'transparent',
            color: text.primary,
            borderRadius: '8px',
            padding: '4px 14px',
            fontSize: '12px',
            cursor: disabled === true ? 'default' : 'pointer',
            opacity: disabled === true ? 0.5 : 1,
          },
          children: label,
        });

      const data = state.data;

      return h('section', {
        style: {
          maxWidth: '720px',
          color: text.primary,
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        },
        children: [
          h('h2', {
            style: { margin: 0, fontSize: '16px', fontWeight: 500, lineHeight: '24px' },
            children: 'Local models',
          }),
          h('p', {
            style: { margin: 0, fontSize: '14px', color: text.tertiary, lineHeight: '22px' },
            children:
              'The local engines on this machine — where each one is installed, whether it is running, and what it holds. Read from the active catalog plus a live check, so it always matches what the model hub would actually do.',
          }),
          h('div', {
            style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' },
            children: [
              button(state.status === 'loading' ? 'Checking…' : 'Check now', () => load(true), state.status === 'loading'),
              button('Refresh', () => load(false), state.status === 'loading'),
              state.status === 'refreshing' ? h('span', { style: { fontSize: '12px', color: text.tertiary }, children: 'refreshing…' }) : null,
              state.status === 'error'
                ? h('span', { style: { fontSize: '12px', color: text.error }, children: 'could not read the inventory: ' + state.error })
                : null,
            ],
          }),

          data
            ? h('div', {
                style: {
                  border: '1px solid ' + text.border,
                  borderRadius: '12px',
                  padding: '10px 14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                },
                children: [
                  h(Row, { label: 'catalog', value: data.catalogPath, mono: true }),
                  h(Row, {
                    label: 'artifacts',
                    value: data.artifactRoot || "each session's workspace: <workspace>/artifacts",
                    mono: data.artifactRoot ? true : false,
                  }),
                  h(Row, {
                    label: 'can start',
                    value: data.allowProcessLaunch
                      ? 'yes — the hub starts an engine that is not running (allowProcessLaunch: true)'
                      : 'no — engines must already be running (allowProcessLaunch: false)',
                  }),
                  h(Row, {
                    label: 'machine',
                    value:
                      data.machine.vramGb + ' GiB VRAM · ' + data.machine.ramGb + ' GiB RAM · ' +
                      (data.machine.hasGpu ? 'GPU detected' : 'no GPU detected'),
                  }),
                  h(Row, { label: 'checked at', value: data.generatedAt }),
                ],
              })
            : null,

          ...(data ? data.engines.map((engine) => h(EngineCard, { key: engine.id, engine })) : []),

          data && data.models.length > 0
            ? h('div', {
                style: {
                  border: '1px solid ' + text.border,
                  borderRadius: '12px',
                  padding: '10px 14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                },
                children: [
                  h('div', { style: { fontSize: '13px', fontWeight: 500 }, children: 'Catalog models' }),
                  ...data.models.map((model) =>
                    h('div', {
                      key: model.id,
                      style: {
                        fontSize: '12px',
                        lineHeight: '18px',
                        display: 'flex',
                        gap: '8px',
                        alignItems: 'baseline',
                        flexWrap: 'wrap',
                      },
                      children: [
                        h('span', { style: { fontFamily: MONO, color: text.primary, userSelect: 'text' }, children: model.id }),
                        h(Pill, {
                          tone: model.availability === 'available' ? 'ok' : 'muted',
                          children: model.availability + ' · ' + model.lifecycle,
                        }),
                        h('span', {
                          style: { color: text.tertiary },
                          children:
                            model.capabilities.join(', ') +
                            (model.hostId ? ' · host ' + model.hostId : '') +
                            (model.startable ? ' · startable' : ''),
                        }),
                        model.detail ? h('span', { style: { color: text.tertiary }, children: model.detail }) : null,
                      ],
                    }),
                  ),
                ],
              })
            : null,

          data && (data.capabilities.length > 0 || data.unavailable.length > 0)
            ? h('div', {
                style: {
                  border: '1px solid ' + text.border,
                  borderRadius: '12px',
                  padding: '10px 14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                  fontSize: '12px',
                  lineHeight: '18px',
                },
                children: [
                  h('div', { style: { fontSize: '13px', fontWeight: 500 }, children: 'What this machine can serve' }),
                  ...data.capabilities.map((entry) =>
                    h(Row, { key: entry.capability, label: entry.capability, value: entry.models + ' model(s)' }),
                  ),
                  ...(data.unavailable.length > 0
                    ? [
                        h(Row, {
                          key: 'unavailable',
                          label: 'not available',
                          value: data.unavailable.map((entry) => entry.capability).join(', '),
                        }),
                      ]
                    : []),
                ],
              })
            : null,
        ],
      });
    }

    const name = 'dsh-ai-model-hub';
    const inject = ['slots'];

    /**
     * Register the page into the settings shell.
     * @param ctx - the client context.
     */
    function apply(ctx) {
      ctx.slots.inject('settings.section', function* () {
        yield ctx.slots.register(
          { name: 'settings.section', id: 'dsh-ai-model-hub', order: 35, label: () => 'Local models' },
          LocalModelsSection,
        );
      });
    }

    module.exports = { apply, inject, name };
    return module.exports;
  },
});
