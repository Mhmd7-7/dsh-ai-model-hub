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
// The registration id is the PACKAGE NAME, not a free label: DSH keys composed
// client modules by the loader row's package name, and a bundle registered under
// any other id is stored where nothing ever requires it — the settings section
// then silently never mounts. This file is the client half of the package
// `dsh-ai-model-hub`, so the id is exactly that.
window.__ModuleLoader__.load({
  id: 'dsh-ai-model-hub',
  factory: (require) => {
    const module = { exports: {} };

    const React = require('react');
    const { jsx: h } = require('react/jsx-runtime');
    const { useCallback, useEffect, useRef, useState } = React;

    /** The host route this page reads. */
    const ENDPOINT = '/dsh-ai-model-hub/inventory';

    /** The host route the Capability priorities editor reads and writes. */
    const PRIORITIES_ENDPOINT = '/dsh-ai-model-hub/priorities';

    // Bound a whole request, including body decoding. Latest intent wins even
    // when a transport ignores abort; teardown invalidates all pending callbacks.
    const REQUEST_TIMEOUT_MS = 30000;
    function useRequestTask() {
      const active = useRef(null);
      const sequence = useRef(0);
      const cancel = useCallback(() => {
        sequence.current += 1;
        if (active.current) active.current.abort();
        active.current = null;
      }, []);
      useEffect(() => cancel, [cancel]);
      return useCallback((work, success, failure, finish) => {
        cancel();
        const id = sequence.current;
        const controller = new AbortController();
        active.current = controller;
        let timer;
        const cancelled = new Promise((_, reject) => {
          controller.signal.addEventListener('abort', () => reject(new Error('Request cancelled')), { once: true });
          timer = setTimeout(() => {
            reject(new Error('Request timed out after 30 seconds'));
            controller.abort();
          }, REQUEST_TIMEOUT_MS);
        });
        Promise.race([Promise.resolve().then(() => work(controller.signal)), cancelled])
          .then((value) => { if (id === sequence.current) success(value); })
          .catch((error) => { if (id === sequence.current) failure(error); })
          .finally(() => {
            clearTimeout(timer);
            if (id === sequence.current) {
              active.current = null;
              if (finish) finish();
            }
          });
      }, [cancel]);
    }

    async function readJson(url, signal, init) {
      const response = await fetch(url, { ...init, signal, cache: 'no-store' });
      const body = await response.json().catch((error) => {
        throw new Error('HTTP ' + response.status + ' — invalid JSON response: ' + error.message);
      });
      if (!response.ok) throw new Error('HTTP ' + response.status +
        (body && body.error ? ' — ' + body.error : '') +
        (body && body.issues ? ' — ' + body.issues.join('; ') : ''));
      return body;
    }

    /** Monospace stack for paths and commands; the system stack, so nothing is fetched. */
    const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Courier New", monospace';

    const text = {
      primary: 'var(--dsw-alias-label-primary)',
      secondary: 'var(--dsw-alias-label-secondary)',
      tertiary: 'var(--dsw-alias-label-tertiary)',
      border: 'var(--dsw-alias-border-l2)',
      ok: 'var(--dsw-alias-state-success-primary, #438368)',
      error: 'var(--dsw-alias-state-error-primary)',
    };

    // Scoped to this page. Theme tokens follow DSH's light/dark theme; no fonts,
    // images, or stylesheet requests leave the browser.
    const STYLES = `
      /* The accent is deliberately a fixed green rather than a host token: the
         primary button puts white text on it, and the host's success colour is a
         bright green that does not carry white text at an accessible contrast in
         both themes. Everything structural still comes from the host tokens. */
      .mh-page { --mh-accent: #438368; --mh-surface: var(--dsw-alias-bg-layer-1, rgba(127,127,127,.035)); --mh-soft: rgba(127,127,127,.065); width: 100%; max-width: 1040px; color: var(--dsw-alias-label-primary); font-family: inherit; }
      .mh-page * { box-sizing: border-box; }
      .mh-page button, .mh-page input { font: inherit; }
      .mh-page button:focus-visible, .mh-page input:focus-visible, .mh-page summary:focus-visible { outline: 2px solid var(--mh-accent); outline-offset: 3px; }
      .mh-page [hidden] { display: none !important; }
      .mh-header { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
      .mh-mark { display: grid; place-items: center; width: 48px; height: 48px; border-radius: 15px; background: var(--mh-soft); border: 1px solid var(--dsw-alias-border-l2); }
      .mh-heading { margin: 0; font-size: 24px; font-weight: 650; letter-spacing: -.6px; line-height: 1.3; }
      .mh-subtitle { margin: 5px 0 0; color: var(--dsw-alias-label-secondary); font-size: 13px; line-height: 1.6; }
      .mh-button { display: inline-flex; align-items: center; justify-content: center; gap: 8px; min-height: 36px; padding: 7px 13px; border: 1px solid var(--dsw-alias-border-l2); border-radius: 10px; background: var(--mh-surface); color: var(--dsw-alias-label-primary); font-size: 12px !important; font-weight: 550 !important; cursor: pointer; transition: background .15s, border-color .15s; }
      .mh-button:hover:not(:disabled) { background: var(--mh-soft); border-color: var(--dsw-alias-label-tertiary); }
      .mh-button:disabled { opacity: .45; cursor: not-allowed; }
      .mh-button-primary { background: var(--mh-accent); color: #fff; border-color: var(--mh-accent); }
      .mh-button-primary:hover:not(:disabled) { background: var(--mh-accent); filter: brightness(1.08); }
      .mh-tabs { display: flex; gap: 5px; padding: 5px; width: fit-content; max-width: 100%; border: 1px solid var(--dsw-alias-border-l2); border-radius: 12px; background: var(--mh-soft); margin-bottom: 24px; }
      .mh-tab { display: inline-flex; gap: 8px; align-items: center; padding: 9px 14px; border: 1px solid transparent; border-radius: 8px; background: transparent; color: var(--dsw-alias-label-secondary); cursor: pointer; font-size: 13px !important; }
      .mh-tab[aria-selected=true] { background: var(--mh-surface); color: var(--dsw-alias-label-primary); border-color: var(--dsw-alias-border-l2); box-shadow: 0 2px 5px rgba(0,0,0,.035); }
      .mh-editor { display: flex; flex-direction: column; gap: 18px; }
      .mh-section-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
      .mh-section-heading h2 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: -.3px; }
      .mh-section-heading p { margin: 6px 0 0; font-size: 13px; color: var(--dsw-alias-label-secondary); line-height: 1.6; }
      .mh-panel { border: 1px solid var(--dsw-alias-border-l2); border-radius: 16px; padding: 18px; background: var(--mh-surface); }
      .mh-global { display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
      .mh-global-copy { flex: 1; min-width: 180px; }
      .mh-global-title { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; font-size: 14px; font-weight: 600; }
      .mh-global p { margin: 7px 0 0; font-size: 12px; line-height: 1.65; color: var(--dsw-alias-label-secondary); max-width: 650px; }
      .mh-global-icon { display: grid; place-items: center; width: 42px; height: 42px; flex-shrink: 0; border-radius: 12px; background: var(--mh-soft); color: var(--dsw-alias-label-secondary); }
      .mh-help { font-size: 12px; line-height: 1.7; color: var(--dsw-alias-label-secondary); }
      .mh-help summary { cursor: pointer; width: fit-content; font-weight: 550; color: var(--dsw-alias-label-primary); }
      .mh-help p { margin: 10px 0 0; }
      .mh-toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
      .mh-search { display: flex; align-items: center; gap: 9px; flex: 1; min-width: min(240px, 100%); min-height: 38px; border: 1px solid var(--dsw-alias-border-l2); border-radius: 10px; padding: 0 12px; color: var(--dsw-alias-label-tertiary); background: var(--mh-surface); }
      .mh-search input { width: 100%; min-width: 0; background: transparent; border: 0; padding: 10px 0; font-size: 12px; color: var(--dsw-alias-label-primary); }
      .mh-filter { display: flex; flex-wrap: wrap; gap: 5px; }
      .mh-filter button[aria-pressed=true] { background: var(--mh-soft); border-color: var(--mh-accent); }
      .mh-summary { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 12px; color: var(--dsw-alias-label-secondary); }
      .mh-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr)); gap: 14px; align-items: start; }
      .mh-card { overflow: hidden; border: 1px solid var(--dsw-alias-border-l2); border-radius: 16px; background: var(--mh-surface); }
      .mh-card-head { display: flex; gap: 12px; align-items: flex-start; padding: 17px; }
      .mh-cap-icon { width: 36px; height: 36px; border-radius: 10px; background: var(--mh-soft); display: grid; place-items: center; flex-shrink: 0; color: var(--mh-accent); }
      .mh-card-title { margin: 0; font-size: 14px; line-height: 1.4; font-weight: 600; }
      .mh-card-code { margin-top: 3px; color: var(--dsw-alias-label-tertiary); font-size: 10px; font-family: ${MONO}; overflow-wrap: anywhere; }
      .mh-card-meta { display: flex; flex-wrap: wrap; gap: 7px; align-items: center; padding: 0 17px 13px; font-size: 11px; color: var(--dsw-alias-label-secondary); }
      .mh-options { display: flex; flex-direction: column; gap: 8px; padding: 0 10px 10px; }
      .mh-option { border: 1px solid var(--dsw-alias-border-l2); border-radius: 11px; padding: 12px; background: var(--mh-soft); }
      .mh-option-dragging { border-color: var(--mh-accent); opacity: .65; }
      .mh-option-head { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }
      .mh-rank { display: inline-flex; align-items: center; gap: 5px; color: var(--dsw-alias-label-tertiary); font-size: 11px; min-width: 30px; }
      .mh-option-name { flex: 1; min-width: 100px; font-size: 13px; font-weight: 550; overflow-wrap: anywhere; }
      .mh-option-source { margin: 9px 0 0; font-size: 11px; color: var(--dsw-alias-label-secondary); line-height: 1.6; overflow-wrap: anywhere; }
      .mh-option-controls { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; margin-top: 11px; }
      .mh-move { width: 28px; height: 28px; padding: 0; border: 1px solid var(--dsw-alias-border-l2); border-radius: 7px; background: var(--mh-surface); color: var(--dsw-alias-label-secondary); font-size: 13px !important; cursor: pointer; }
      .mh-move:hover:not(:disabled) { color: var(--dsw-alias-label-primary); border-color: var(--mh-accent); }
      .mh-move:disabled { opacity: .3; cursor: not-allowed; }
      .mh-option-details { margin-top: 9px; font-size: 11px; color: var(--dsw-alias-label-secondary); line-height: 1.65; }
      .mh-option-details summary { cursor: pointer; color: var(--dsw-alias-label-tertiary); }
      .mh-option-details p { margin: 7px 0 0; overflow-wrap: anywhere; }
      .mh-toggle { display: inline-flex; align-items: center; gap: 7px; padding: 0; border: 0; background: transparent; color: var(--dsw-alias-label-secondary); font-size: 11px !important; cursor: pointer; }
      .mh-toggle:disabled { cursor: not-allowed; opacity: .5; }
      .mh-toggle-track { width: 30px; height: 18px; padding: 3px; border-radius: 10px; background: var(--dsw-alias-label-tertiary); display: flex; align-items: center; }
      .mh-toggle[aria-pressed=true] .mh-toggle-track { background: var(--mh-accent); }
      .mh-toggle-knob { width: 12px; height: 12px; border-radius: 50%; background: #fff; transform: translateX(0); transition: transform .15s; }
      .mh-toggle[aria-pressed=true] .mh-toggle-knob { transform: translateX(12px); }
      .mh-empty { padding: 20px 17px; border-top: 1px dashed var(--dsw-alias-border-l2); font-size: 12px; color: var(--dsw-alias-label-secondary); line-height: 1.7; }
      .mh-savebar { position: sticky; bottom: 0; z-index: 2; display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; border: 1px solid var(--dsw-alias-border-l2); border-radius: 13px; padding: 12px 15px; background: var(--dsw-alias-bg-layer-1, Canvas); box-shadow: 0 5px 20px rgba(0,0,0,.06); }
      .mh-savebar-copy { font-size: 12px; line-height: 1.6; color: var(--dsw-alias-label-secondary); flex: 1; min-width: 160px; }
      .mh-savebar-actions { display: flex; gap: 7px; }
      .mh-alert { padding: 12px 15px; border: 1px solid var(--dsw-alias-border-l2); border-radius: 12px; font-size: 12px; line-height: 1.7; color: var(--dsw-alias-label-secondary); background: var(--mh-soft); }
      .mh-alert-error { color: var(--dsw-alias-state-error-primary); }
      .mh-inventory > section { max-width: none !important; gap: 14px !important; }
      .mh-inventory > section > div { border-radius: 14px !important; }
      @media (max-width: 540px) { .mh-heading { font-size: 21px; } .mh-header { gap: 10px; margin-bottom: 18px; } .mh-tab { padding: 8px 10px; } .mh-panel { padding: 14px; } .mh-global-icon { display: none; } .mh-savebar { bottom: 0; } }
      @media (prefers-reduced-motion: reduce) { .mh-page *, .mh-page *::before, .mh-page *::after { transition: none !important; scroll-behavior: auto !important; } }
    `;

    const CAPABILITY_LABELS = {
      text_to_text: ['Text generation', 'text'],
      text_to_image: ['Image generation', 'image'],
      image_to_image: ['Image editing', 'image'],
      text_to_3d: ['Text to 3D', 'box'],
      image_to_3d: ['Image to 3D', 'box'],
      audio_generation: ['Audio generation', 'audio'],
      speech_to_text: ['Speech to text', 'audio'],
      image_understanding: ['Image understanding', 'eye'],
      video_generation: ['Video generation', 'video'],
    };

    function Icon(props) {
      const paths = {
        settings: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M9 3h6l1 3 3 1 2 5-2 5-3 1-1 3H9l-1-3-3-1-2-5 2-5 3-1z',
        inventory: 'M4 4h6v6H4z M14 4h6v6h-6z M4 14h6v6H4z M14 14h6v6h-6z',
        text: 'M4 5h16 M12 5v14 M8 19h8',
        image: 'M4 4h16v16H4z M4 16l5-5 4 4 3-3 4 4 M15 8h.01',
        box: 'M12 3l9 5v8l-9 5-9-5V8z M3 8l9 5 9-5 M12 13v8 M8 5l9 5',
        audio: 'M5 9v6 M9 5v14 M13 3v18 M17 6v12 M21 9v6',
        eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6',
        video: 'M3 5h12v14H3z M15 9l6-4v14l-6-4',
        search: 'M10 3a7 7 0 1 0 0 14 7 7 0 0 0 0-14 M15 15l6 6',
        power: 'M12 2v10 M6 5a9 9 0 1 0 12 0',
        refresh: 'M20 8a9 9 0 1 0 1 7 M20 3v5h-5',
        save: 'M4 3h13l4 4v14H3V3z M7 3v6h10V3 M7 21v-8h10v8',
      };
      return h('svg', {
        width: props.size || 18, height: props.size || 18, viewBox: '0 0 24 24',
        fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round',
        'aria-hidden': true, focusable: false,
        children: h('path', { d: paths[props.name] || paths.settings }),
      });
    }

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
          display: 'inline-flex', alignItems: 'center',
          border: '1px solid ' + text.border,
          background: 'rgba(127,127,127,0.045)',
          color: tone, borderRadius: '999px', padding: '3px 8px',
          fontSize: '10px', lineHeight: '14px', fontWeight: 500, whiteSpace: 'nowrap',
        },
        children: props.children,
      });
    }

    /** A plain outlined button, shared by both sections. */
    function Button(props) {
      const disabled = props.disabled === true;
      return h('button', {
        type: 'button',
        onClick: props.onClick,
        disabled: disabled,
        'aria-label': props.ariaLabel,
        className: 'mh-button' + (props.primary ? ' mh-button-primary' : ''),
        children: props.children,
      });
    }

    /** The same button in positional form, which reads better in a dense list. */
    function button(label, onClick, disabled) {
      return h(Button, { onClick: onClick, disabled: disabled === true, children: label });
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
      const state = engine.state || (engine.running ? 'running' : 'unknown');
      const labels = { running: 'running', installed_unreachable: 'installed · unreachable', not_found: 'not found', unknown: 'unknown' };
      const tone = state === 'running' ? 'ok' : state === 'installed_unreachable' ? 'error' : 'muted';

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
              h(Pill, { tone, children: labels[state] || state }),
              h(Pill, { tone: 'muted', children: engine.configured ? 'configured' : 'not configured' }),
              engine.startable
                ? h(Pill, { tone: 'ok', children: 'hub can start it' })
                : h(Pill, { tone: 'muted', children: 'external' }),
            ],
          }),
          h(Row, { label: 'engine', value: engine.id + ' · ' + engine.adapter }),
          h(Row, { label: 'listens on', value: engine.endpoint, mono: true }),
          engine.installPath
            ? h(Row, { label: 'verified install', value: engine.installPath, mono: true })
            : h(Row, { label: 'installation', value: engine.installation === 'not_found'
                ? 'not found in checked locations (not a machine-wide search)' : 'unknown — no verified installation path' }),
          h(Row, { label: 'configured paths', value: (engine.configuredPaths || []).join(', '), mono: true }),
          h(Row, { label: 'checked paths', value: (engine.checkedPaths || []).join(', '), mono: true }),
          h(Row, { label: 'install evidence', value: engine.installSource }),
          h(Row, { label: 'checked at', value: engine.checkedAt }),
          ...(engine.errors || []).map((error, index) => h('div', {
            key: 'engine-error-' + index, role: 'status', style: { color: text.error, fontSize: '12px' }, children: error,
          })),
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

    /**
     * One scan result: an installed Ollama model, or a complete ComfyUI workflow.
     *
     * The kind label is the point of this row. A workflow is one selectable unit
     * — its checkpoints, LoRAs, VAEs and nodes are deliberately absent, because
     * they are the workflow's business and not something a caller chooses.
     */
    function ResourceItem(props) {
      const resource = props.resource;
      const tone =
        resource.status === 'ready' ? 'ok' : resource.status === 'needs_conversion' ? 'muted' : 'error';
      const kindTone = resource.kind === 'ollama_model' ? 'muted' : 'ok';
      return h('div', {
        style: {
          border: '1px solid ' + text.border,
          borderRadius: '10px',
          padding: '8px 10px',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
        },
        children: [
          h('div', {
            style: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
            children: [
              h('span', {
                style: { fontFamily: MONO, fontSize: '12px', color: text.primary, userSelect: 'text' },
                children: resource.name,
              }),
              h(Pill, { tone: kindTone, children: resource.typeLabel }),
              h(Pill, { tone, children: resource.status }),
            ],
          }),
          h(Row, { label: 'source', value: resource.source, mono: true }),
          h(Row, { label: 'status', value: resource.detail }),
          h(Row, { label: 'availability', value: resource.availability || 'unknown — execution not verified' }),
          typeof resource.engineRunning === 'boolean'
            ? h(Row, { label: 'engine', value: resource.engineRunning ? 'reachable at last check' : 'unreachable at last check' }) : null,
          h(Row, { label: 'freshness', value: resource.freshness || 'scan observation — execution not verified' }),
          h(Row, { label: 'checked at', value: resource.checkedAt }),
          resource.capabilities && resource.capabilities.length > 0
            ? h(Row, { label: 'capabilities', value: resource.capabilities.join(', ') })
            : null,
          resource.inputs && resource.inputs.length > 0
            ? h(Row, {
                label: 'inputs',
                value: resource.inputs
                  .map((input) =>
                    input.name +
                    ' (' + input.kind + ')' +
                    (input.label && input.label !== input.name ? ' — ' + input.label : ''),
                  )
                  .join(', '),
              })
            : null,
          resource.outputs && resource.outputs.length > 0
            ? h(Row, {
                label: 'outputs',
                value: resource.outputs.map((output) => output.name + ' (' + output.type + ')').join(', '),
              })
            : null,
          // Why something was not exposed. Shown rather than hidden: a workflow
          // missing an input it plausibly has is a question the page should answer.
          resource.diagnostics && resource.diagnostics.length > 0
            ? h('div', {
                style: { fontSize: '12px', lineHeight: '18px', color: text.tertiary, userSelect: 'text' },
                children: resource.diagnostics.join(' '),
              })
            : null,
          resource.format ? h(Row, { label: 'format', value: resource.format === 'ui' ? 'editor (UI)' : 'API' }) : null,
          resource.sizeBytes ? h(Row, { label: 'size', value: formatBytes(resource.sizeBytes) }) : null,
          resource.runnable && resource.modelId
            ? h(Row, { label: 'run with', value: 'invoke_model modelId: ' + resource.modelId, mono: true })
            : null,
        ],
      });
    }

    /**
     * The Capability priorities editor.
     *
     * One list per capability, in the operator's saved order, containing every
     * option that genuinely declares it — local models, ComfyUI workflows, and
     * subscription backends in the same list, because from the hub's point of
     * view they are the same kind of thing: a way to serve a capability.
     *
     * The three facts a row shows are deliberately *separate*, because confusing
     * them is the failure this whole feature exists to prevent: whether the
     * operator permits the option (the toggle), where it ranks (the order, movable
     * by drag or by keyboard), and whether it is usable right now (the state pill,
     * with the backend's own reason beside it).
     *
     * Nothing is written until Save. That matters for a list whose whole point is
     * an arrangement: a drag that saved immediately would make a mistake
     * irreversible and a half-finished reorder live.
     */
    function CapabilityPriorities() {
      const [state, setState] = useState({ status: 'loading', error: '', payload: null });
      const [draft, setDraft] = useState(null);
      const [dirty, setDirty] = useState(false);
      const [saving, setSaving] = useState(false);
      const [notice, setNotice] = useState('');
      const [dragging, setDragging] = useState(null);
      const [query, setQuery] = useState('');
      const [filter, setFilter] = useState('all');

      const request = useRequestTask();
      const saveLock = useRef(false);
      const load = useCallback(() => {
        if (saveLock.current) return;
        setState((previous) => ({ ...previous, error: '', status: previous.payload ? 'refreshing' : 'loading' }));
        request((signal) => readJson(PRIORITIES_ENDPOINT, signal), (payload) => {
          setState({ status: 'ready', error: '', payload });
          setDraft(draftFrom(payload));
          setDirty(false);
        }, (error) => setState((previous) => ({ ...previous, status: 'error',
          error: String((error && error.message) || error) })));
      }, [request]);

      useEffect(() => {
        load();
      }, [load]);

      const move = useCallback((capability, from, to) => {
        setDraft((previous) => {
          if (!previous) return previous;
          const list = previous.capabilities[capability];
          if (!list || from < 0 || from >= list.length || to < 0 || to >= list.length || from === to) return previous;
          const next = list.slice();
          const [item] = next.splice(from, 1);
          next.splice(to, 0, item);
          return { ...previous, capabilities: { ...previous.capabilities, [capability]: next } };
        });
        setDirty(true);
        setNotice('');
      }, []);

      const toggle = useCallback((capability, id) => {
        setDraft((previous) => {
          if (!previous) return previous;
          const list = previous.capabilities[capability];
          if (!list) return previous;
          return {
            ...previous,
            capabilities: {
              ...previous.capabilities,
              [capability]: list.map((row) => (row.id === id ? { ...row, enabled: !row.enabled } : row)),
            },
          };
        });
        setDirty(true);
        setNotice('');
      }, []);

      const toggleGlobal = useCallback(() => {
        setDraft((previous) => (previous ? { ...previous, globalEnabled: !previous.globalEnabled } : previous));
        setDirty(true);
        setNotice('');
      }, []);

      const save = useCallback(() => {
        if (!draft || !state.payload || saveLock.current) return;
        saveLock.current = true;
        setSaving(true);
        setNotice('');
        const globalChanged = draft.globalEnabled !== state.payload.globalEnabled;
        const edits = Object.keys(draft.capabilities).filter((capability) => {
          const before = state.payload.capabilities.find((entry) => entry.capability === capability);
          const beforeOrder = before ? before.options.map((option) => option.id + ':' +
            (typeof option.configuredEnabled === 'boolean' ? option.configuredEnabled : option.enabled)) : [];
          const afterOrder = draft.capabilities[capability].map((row) => row.id + ':' + row.enabled);
          return beforeOrder.join('|') !== afterOrder.join('|');
        });
        // The global switch is one document field and is saved on its own; the
        // per-capability edits follow. Sequential rather than parallel so a
        // rejection names the edit that caused it instead of racing another.
        const steps = [];
        if (globalChanged) steps.push({ globalEnabled: draft.globalEnabled });
        for (const capability of edits) {
          steps.push({
            capability,
            order: draft.capabilities[capability].map((row) => ({ id: row.id, enabled: row.enabled })),
          });
        }
        if (steps.length === 0) {
          saveLock.current = false;
          setSaving(false);
          setDirty(false);
          setNotice('Nothing changed.');
          return;
        }
        request(async (signal) => {
          let payload = state.payload;
          for (const step of steps) {
            if (signal.aborted) throw new Error('Request cancelled');
            payload = await readJson(PRIORITIES_ENDPOINT, signal, {
              method: 'PUT', headers: { 'content-type': 'application/json' }, body: JSON.stringify(step),
            });
          }
          return payload;
        }, (payload) => {
          setState({ status: 'ready', error: '', payload });
          setDraft(draftFrom(payload));
          setDirty(false);
          setSaving(false);
          setNotice('Saved. ' + steps.length + ' change(s) written to the capability policy.');
        }, (error) => {
          setNotice('Not saved: ' + String((error && error.message) || error) +
            '. Earlier steps or a timed-out write may have reached the host; retry is idempotent.');
        }, () => {
          saveLock.current = false;
          setSaving(false);
        });
      }, [draft, state.payload, request]);

      const cancel = useCallback(() => {
        setDraft(draftFrom(state.payload));
        setDirty(false);
        setNotice('Reverted to the saved policy.');
      }, [state.payload]);

      const payload = state.payload;
      const capabilities = payload ? payload.capabilities : [];
      const busy = saving || state.status === 'refreshing' || state.status === 'loading';
      const globalEnabled = !!(draft && draft.globalEnabled);
      // Older hosts only expose effective permission while globally disabled.
      // Do not overwrite an unknown saved permission when reordering on those hosts.
      const permissionsKnown = !payload || payload.globalEnabled || capabilities.every((entry) =>
        entry.options.every((option) => typeof option.configuredEnabled === 'boolean'));
      const needle = query.trim().toLowerCase();
      const availableCount = capabilities.filter((entry) => entry.options.length > 0).length;
      const shown = capabilities.filter((entry) => {
        const rows = draft ? draft.capabilities[entry.capability] || [] : [];
        if (filter === 'available' && rows.length === 0) return false;
        if (filter === 'enabled' && !rows.some((row) => row.enabled)) return false;
        const label = CAPABILITY_LABELS[entry.capability] || [entry.capability];
        return !needle || [entry.capability, label[0]].concat(rows.map((row) =>
          [row.name, row.providerLabel, row.id].join(' '))).join(' ').toLowerCase().includes(needle);
      });

      return h('section', {
        className: 'mh-editor',
        'aria-label': 'Capability priorities editor',
        'aria-busy': busy,
        children: [
          h('div', {
            className: 'mh-section-heading',
            children: [
              h('div', { children: [
                h('h2', { children: 'Capability priorities' }),
                h('p', { children: 'Choose what can run. Arrange providers in the order you prefer.' }),
              ] }),
              payload ? h(Pill, { tone: dirty ? 'muted' : 'ok', children: dirty ? 'unsaved changes' : 'Saved policy' }) : null,
            ],
          }),
          h('details', {
            className: 'mh-help',
            children: [
              h('summary', { children: 'How enablement & priority work' }),
              h('p', { children:
                'Enabled permits an option to serve this capability. Disabled forbids it: it is never run, not even as a ' +
                'fallback, and its engine is never started for it. Lower priority means it is tried only after every ' +
                'higher-ranked permitted option has definitively failed. Stopped means the engine is not running yet — ' +
                'an enabled, stopped, startable option keeps its place and is started on demand. Nothing new is enabled ' +
                'until you enable it.',
              }),
            ],
          }),
          state.status === 'error' ? h('div', {
            className: 'mh-alert mh-alert-error', role: 'alert',
            children: [
              'The capability policy could not be read: ' + state.error + '. The policy already in force has not changed. ' +
                (payload ? 'Showing stale policy data from the last successful read. ' : ''),
              h(Button, { onClick: load, children: 'Try again' }),
            ],
          }) : null,
          payload ? h('div', {
            className: 'mh-panel mh-global',
            children: [
              h('div', { className: 'mh-global-icon', children: h(Icon, { name: 'power', size: 22 }) }),
              h('div', {
                className: 'mh-global-copy',
                children: [
                  h('div', { className: 'mh-global-title', children: [
                    'Global switch',
                    h(Pill, { tone: globalEnabled ? 'ok' : 'muted', children: globalEnabled ? 'On' : 'All disabled' }),
                  ] }),
                  h('p', { children: globalEnabled
                    ? 'Only permitted options can run. Switching this off pauses every capability without changing your saved order.'
                    : 'Global disable wins over every per-capability setting: nothing runs, including options you have enabled. Your saved choices remain in place.',
                  }),
                  payload.globalEnabled !== globalEnabled ? h('p', {
                    children: 'Draft only — the global switch changes when you press Save.',
                  }) : null,
                ],
              }),
              h(Button, {
                onClick: toggleGlobal, disabled: busy || !draft,
                ariaLabel: globalEnabled ? 'Disable every capability' : 'Enable capabilities',
                children: globalEnabled ? 'Disable all capabilities' : 'Enable capabilities',
              }),
            ],
          }) : null,
          payload && !permissionsKnown ? h('div', {
            className: 'mh-alert', role: 'status',
            children: 'This host does not expose saved per-option choices while globally disabled. Enable and save the global switch before editing individual options; existing choices will not be overwritten.',
          }) : null,
          payload ? h('div', {
            className: 'mh-toolbar',
            children: [
              h('label', { className: 'mh-search', children: [
                h(Icon, { name: 'search', size: 16 }),
                h('input', {
                  type: 'search', value: query, onChange: (event) => setQuery(event.target.value),
                  'aria-label': 'Search capabilities and providers', placeholder: 'Search capabilities or providers…',
                }),
              ] }),
              h('div', { className: 'mh-filter', 'aria-label': 'Filter capabilities', children:
                [['all', 'All'], ['available', 'With options'], ['enabled', 'Enabled']].map(([value, label]) =>
                  h('button', {
                    key: value, type: 'button', className: 'mh-button', 'aria-pressed': filter === value,
                    onClick: () => setFilter(value), children: label,
                  })),
              }),
              h(Button, {
                onClick: load, disabled: busy || dirty,
                ariaLabel: 'Reload saved capability policy',
                children: [h(Icon, { name: 'refresh', size: 15 }), state.status === 'refreshing' ? 'Reloading…' : 'Reload'],
              }),
            ],
          }) : null,
          payload ? h('div', {
            className: 'mh-summary', children: [
              h('span', { children: String(capabilities.length) + ' capabilities' }),
              h('span', { 'aria-hidden': true, children: '·' }),
              h('span', { children: String(availableCount) + ' with options' }),
              h('span', { 'aria-hidden': true, children: '·' }),
              h('span', { children: globalEnabled ? 'Enabled options keep their saved priority' : 'Execution paused globally' }),
            ],
          }) : null,
          state.status === 'loading' ? h('div', {
            className: 'mh-alert', role: 'status', children: 'Reading the capability policy and checking option status…',
          }) : null,
          h('div', {
            className: 'mh-grid',
            children: shown.map((entry) => h(CapabilityCard, {
              key: entry.capability, entry,
              rows: draft ? draft.capabilities[entry.capability] || [] : [],
              disabled: busy || !draft || !permissionsKnown,
              globalEnabled, dragging,
              onDragStateChange: setDragging, onMove: move, onToggle: toggle,
            })),
          }),
          payload && shown.length === 0 ? h('div', {
            className: 'mh-alert', role: 'status', children: 'No capabilities match this filter.',
          }) : null,
          payload ? h('details', {
            className: 'mh-help', children: [
              h('summary', { children: 'Policy file & revision' }),
              h('div', { style: { marginTop: '10px' }, children: [
                h(Row, { label: 'policy file', value: payload.policyPath, mono: true }),
                h(Row, { label: 'revision', value: String(payload.revision) }),
              ] }),
            ],
          }) : null,
          payload ? h('div', {
            className: 'mh-savebar',
            children: [
              h('div', {
                className: 'mh-savebar-copy', role: 'status', 'aria-live': 'polite',
                children: notice || (dirty ? 'Unsaved changes — Save to apply, or Cancel to discard.' : 'Changes are only applied when you press Save.'),
              }),
              h('div', { className: 'mh-savebar-actions', children: [
                button('Cancel', cancel, busy || !dirty),
                h(Button, {
                  onClick: save, disabled: busy || !dirty, primary: true,
                  children: [h(Icon, { name: 'save', size: 15 }), saving ? 'Saving…' : 'Save'],
                }),
              ] }),
            ],
          }) : null,
        ],
      });
    }

    /** Build the editable draft from a host payload: capability → ordered rows. */
    function draftFrom(payload) {
      if (!payload) return null;
      const capabilities = {};
      for (const entry of payload.capabilities) {
        capabilities[entry.capability] = entry.options.map((option) => ({
          id: option.id,
          name: option.name,
          providerLabel: option.providerLabel,
          source: option.source,
          type: option.type,
          capabilities: option.capabilities,
          inputTypes: option.inputTypes,
          outputTypes: option.outputTypes,
          state: option.state,
          reason: option.reason,
          startable: option.startable,
          executable: option.executable,
          enabled: typeof option.configuredEnabled === 'boolean' ? option.configuredEnabled : option.enabled,
          permissionUnknown: !payload.globalEnabled && typeof option.configuredEnabled !== 'boolean',
          missing: option.missing,
        }));
      }
      return { globalEnabled: payload.globalEnabled, capabilities: capabilities };
    }

    /** One capability's list. */
    function CapabilityCard(props) {
      const entry = props.entry;
      const rows = props.rows;
      const enabledCount = rows.filter((row) => row.enabled).length;
      const unknown = rows.some((row) => row.permissionUnknown);
      const label = CAPABILITY_LABELS[entry.capability] || [entry.capability, 'settings'];
      // Compute the draft display from the edited order, not the stale server plan.
      const permitted = props.globalEnabled ? rows.filter((row) => row.enabled) : [];
      return h('article', {
        className: 'mh-card', 'aria-label': label[0],
        children: [
          h('div', { className: 'mh-card-head', children: [
            h('div', { className: 'mh-cap-icon', children: h(Icon, { name: label[1], size: 20 }) }),
            h('div', { style: { flex: 1, minWidth: 0 }, children: [
              h('h3', { className: 'mh-card-title', children: label[0] }),
              h('div', { className: 'mh-card-code', children: entry.capability }),
            ] }),
            h(Pill, { tone: 'muted', children: rows.length ? String(rows.length) + ' options' : 'Unavailable' }),
          ] }),
          rows.length ? h('div', { className: 'mh-card-meta', children: [
            h('span', { children: unknown ? 'Saved choices paused' : String(enabledCount) + ' enabled' }),
            '·', h('span', { children: !props.globalEnabled ? 'Global switch off' : permitted.length ? 'Top to bottom priority' : 'Nothing permitted' }),
          ] }) : null,
          rows.length === 0 ? h('div', {
            className: 'mh-empty', children: 'No provider is installed for this capability. It stays visible so you can see what is missing.',
          }) : h('div', {
            className: 'mh-options',
            children: rows.map((row, index) => h(PriorityRow, {
              key: row.id, capability: entry.capability, row, index, count: rows.length,
              disabled: props.disabled, globalEnabled: props.globalEnabled,
              dragging: props.dragging, onDragStateChange: props.onDragStateChange,
              onMove: props.onMove, onToggle: props.onToggle,
            })),
          }),
        ],
      });
    }

    /**
     * One option row: order, identity, enablement, live state, and move controls.
     *
     * Dragging is native HTML5 drag-and-drop, and every drag has a keyboard
     * equivalent beside it — Move up, Move down, Move to top, Move to bottom — as
     * real buttons with accessible names. A reorder control that only works with a
     * mouse is not a reorder control for everyone, and the requirement is
     * explicit that this one be reachable from the keyboard.
     */
    function PriorityRow(props) {
      const row = props.row;
      const capability = props.capability;
      const liveState = !props.globalEnabled ? 'Paused globally' : row.state === 'disabled' && row.enabled ? 'Check after saving' : row.state;
      const stateTone = row.enabled
        ? row.state === 'ready'
          ? 'ok'
          : row.state === 'stopped' || row.state === 'starting'
            ? 'muted'
            : 'error'
        : 'muted';
      const isDragging = props.dragging && props.dragging.id === row.id && props.dragging.capability === capability;

      const control = (label, title, onClick, disabled) =>
        h('button', {
          type: 'button',
          'aria-label': title,
          title: title,
          onClick,
          disabled: disabled === true,
          className: 'mh-move',
          children: label,
        });

      return h('div', {
        draggable: props.disabled !== true,
        onDragStart: (event) => {
          if (event.dataTransfer) {
            event.dataTransfer.effectAllowed = 'move';
            // Some browsers refuse to start a drag without payload.
            event.dataTransfer.setData('text/plain', row.id);
          }
          props.onDragStateChange({ capability: capability, id: row.id, index: props.index });
        },
        onDragEnd: () => props.onDragStateChange(null),
        onDragOver: (event) => {
          if (!props.dragging || props.dragging.capability !== capability) return;
          event.preventDefault();
          if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
        },
        onDrop: (event) => {
          event.preventDefault();
          const source = props.dragging;
          props.onDragStateChange(null);
          if (!source || source.capability !== capability) return;
          props.onMove(capability, source.index, props.index);
        },
        'aria-label': row.name + ' — position ' + String(props.index + 1) + ' of ' + String(props.count) + ' for ' + capability,
        className: 'mh-option' + (isDragging ? ' mh-option-dragging' : ''),
        children: [
          h('div', { className: 'mh-option-head', children: [
            h('span', { className: 'mh-rank', children: [
              h('span', { title: 'Drag to reorder', 'aria-hidden': true,
                style: { cursor: props.disabled ? 'default' : 'grab', fontSize: '17px' }, children: '⠿' }),
              String(props.index + 1),
            ] }),
            h('span', { className: 'mh-option-name', children: row.name }),
            row.permissionUnknown ? h(Pill, { tone: 'muted', children: 'Paused' }) : h('button', {
              className: 'mh-toggle', type: 'button',
              onClick: () => props.onToggle(capability, row.id), disabled: props.disabled === true,
              'aria-pressed': row.enabled,
              'aria-label': (row.enabled ? 'Disable ' : 'Enable ') + row.name + ' for ' + capability,
              children: [
                h('span', { className: 'mh-toggle-track', 'aria-hidden': true,
                  children: h('span', { className: 'mh-toggle-knob' }) }),
                h('span', { children: row.enabled ? 'Enabled' : 'Disabled' }),
              ],
            }),
          ] }),
          h('div', { className: 'mh-option-source', children: row.providerLabel + ' · ' + row.source + ' · ' + row.type }),
          h('div', { className: 'mh-option-controls', children: [
            h(Pill, { tone: props.globalEnabled ? stateTone : 'muted', children: liveState }),
            row.startable ? h(Pill, { tone: 'muted', children: 'Starts on demand' }) : null,
            row.missing ? h(Pill, { tone: 'error', children: 'Missing' }) : null,
            h('span', { style: { flex: 1 } }),
            control('↑', 'Move ' + row.name + ' up', () => props.onMove(capability, props.index, props.index - 1), props.disabled === true || props.index === 0),
            control('↓', 'Move ' + row.name + ' down', () => props.onMove(capability, props.index, props.index + 1), props.disabled === true || props.index === props.count - 1),
            control('⤒', 'Move ' + row.name + ' to the top', () => props.onMove(capability, props.index, 0), props.disabled === true || props.index === 0),
            control('⤓', 'Move ' + row.name + ' to the bottom', () => props.onMove(capability, props.index, props.count - 1), props.disabled === true || props.index === props.count - 1),
          ] }),
          h('details', { className: 'mh-option-details', children: [
            h('summary', { children: 'Status & technical details' }),
            h('div', { style: { marginTop: '8px' }, children: [
              h(Row, { label: 'source', value: row.providerLabel + ' · ' + row.source + ' · ' + row.type }),
              h(Row, { label: 'reason', value: row.reason }),
              h(Row, { label: 'option id', value: row.id, mono: true }),
              row.inputTypes && row.inputTypes.length ? h(Row, {
                label: 'in/out', value: row.inputTypes.join(' | ') + ' → ' + (row.outputTypes || []).join(' | '),
              }) : null,
            ] }),
          ] }),
        ],
      });
    }

    /** The settings section itself. */
    function LocalModelsSection() {
      const [state, setState] = useState({ status: 'loading', data: null, error: '', stale: false });

      const request = useRequestTask();
      const lastQuery = useRef('?scan=1');
      const load = useCallback((query) => {
        lastQuery.current = typeof query === 'string' ? query : lastQuery.current;
        setState((previous) => ({ ...previous, error: '', status: previous.data ? 'refreshing' : 'loading' }));
        request(async (signal) => {
          const data = await readJson(ENDPOINT + lastQuery.current, signal);
          const object = (value) => value && typeof value === 'object' && !Array.isArray(value);
          const arrays = ['engines', 'models', 'resources', 'capabilities', 'unavailable'];
          if (!object(data) || !object(data.machine) || arrays.some((key) => !Array.isArray(data[key])) ||
              !object(data.scan) || !Array.isArray(data.scan.sources) || !Array.isArray(data.scan.warnings) ||
              arrays.some((key) => data[key].some((entry) => !object(entry))) ||
              data.engines.some((engine) => !Array.isArray(engine.models)) ||
              data.models.some((model) => !Array.isArray(model.capabilities))) {
            throw new Error('Invalid inventory response from the scan service');
          }
          return data;
        },
          (data) => setState({ status: 'ready', data, error: '', stale: false }),
          (error) => setState((previous) => ({ ...previous, status: 'error', stale: !!previous.data,
            error: String((error && error.message) || error) })));
      }, [request]);

      useEffect(() => {
        // Opening the page *looks*; it never launches. Only the Scan button asks
        // for the engines to be started, which is why the mount read carries no
        // `scan=1` — a settings page appearing on screen must not start a server.
        load('');
      }, [load]);

      const data = state.data;
      const resources = (data && data.resources) || [];
      const ollamaCount = resources.filter((entry) => entry.kind === 'ollama_model').length;
      const workflowCount = resources.filter((entry) => entry.kind === 'comfyui_workflow').length;
      const scanSources = (data && data.scan && data.scan.sources) || [];
      const scanWarnings = (data && data.scan && data.scan.warnings) || [];
      const scanEngines = (data && data.scan && data.scan.engines) || [];
      const scanPhases = (data && data.scan && data.scan.phases) || [];
      const scanComplete = !data || !data.scan ? true : data.scan.complete !== false;
      const launched = scanEngines.filter((engine) => engine.started === true);
      const held = scanEngines.filter((engine) => engine.protectedFromIdleStop === true);
      const PHASE_LABELS = {
        checking: 'Checking engines',
        starting: 'Starting engines',
        waiting: 'Waiting for readiness',
        'engine-ready': 'Engine ready',
        'engine-failed': 'Engine unavailable',
        scanning: 'Scanning',
        complete: 'Complete',
        partial: 'Partially complete',
      };

      return h('section', {
        'aria-busy': state.status === 'loading' || state.status === 'refreshing',
        style: {
          maxWidth: '720px',
          color: text.primary,
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
        },
        children: [          h('h2', {
            style: { margin: 0, fontSize: '16px', fontWeight: 500, lineHeight: '24px' },
            children: 'Local models',
          }),
          h('p', {
            style: { margin: 0, fontSize: '14px', color: text.tertiary, lineHeight: '22px' },
            children:
              'Configured engines and observed local resources. Configuration, verified installation paths and endpoint reachability are separate facts; observations describe the last successful check, not a guarantee that a model will run.',
          }),
          h('div', {
            style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' },
            children: [
              // Never disabled: pressing Scan while a read is in flight is a
              // deliberate act, and the answer is to supersede that read rather
              // than to ignore the click. The server coalesces engine launches, so
              // repeated clicks cost a request, never a second process.
              button(
                state.status === 'refreshing' && lastQuery.current === '?scan=1' ? 'Scanning…' : 'Scan',
                () => load('?scan=1'),
                false,
              ),
              button('Refresh', () => load(''), false),
              data && data.scan && data.scan.managedEngines
                ? h(Pill, { tone: 'muted', children: 'Scan may start stopped engines' })
                : null,
            ],
          }),

          state.status === 'loading'
            ? h('div', {
                style: {
                  border: '1px solid ' + text.border,
                  borderRadius: '12px',
                  padding: '10px 14px',
                  fontSize: '12px',
                  color: text.tertiary,
                },
                children: 'Reading Ollama models and ComfyUI workflows…',
              })
            : null,

          // The stage the last scan reached, and — while one is running — what it is
          // doing now. A scan that starts an engine can take minutes, so silence
          // would read as a hang; the phase names are the answer to "is it stuck?".
          state.status === 'refreshing' && lastQuery.current === '?scan=1'
            ? h('div', {
                style: { fontSize: '12px', color: text.tertiary, lineHeight: '18px' },
                children: 'Scanning… checking engines, starting any that are stopped, waiting for readiness, then reading them.',
              })
            : null,

          data && data.scan && scanPhases.length > 0
            ? h('div', {
                style: { display: 'flex', gap: '6px', flexWrap: 'wrap', alignItems: 'center' },
                children: [
                  h('span', { style: { fontSize: '12px', color: text.tertiary }, children: 'progress' }),
                  ...scanPhases.map((phase, index) =>
                    h(Pill, {
                      key: 'phase-' + index + '-' + phase,
                      tone: phase === 'partial' ? 'error' : phase === 'complete' ? 'ok' : 'muted',
                      children: PHASE_LABELS[phase] || phase,
                    }),
                  ),
                  data.scan.progress
                    ? h('span', { style: { fontSize: '12px', color: text.tertiary }, children: data.scan.progress })
                    : null,
                ],
              })
            : null,

          // A scan that could not read everything says so, and says which engine was
          // the problem. Presenting a partial look as a success is exactly how a
          // user concludes the machine has no models.
          data && data.scan && data.scan.complete === false
            ? h('div', {
                style: {
                  border: '1px solid ' + text.error,
                  borderRadius: '12px',
                  padding: '10px 14px',
                  fontSize: '12px',
                  color: text.error,
                  lineHeight: '18px',
                },
                role: 'status',
                children: [
                  'Partially complete: some engines could not be started or read, so this list may be missing models or workflows. ',
                  ...scanEngines.filter((engine) => engine.ready !== true).map((engine, index) =>
                    h('div', { key: 'engine-problem-' + index, children: engine.detail }),
                  ),
                ],
              })
            : null,

          launched.length > 0
            ? h('div', {
                style: { fontSize: '12px', color: text.tertiary, lineHeight: '18px' },
                children:
                  'This scan started ' + launched.map((engine) => engine.hostName).join(', ') + '. ' +
                  (held.length > 0
                    ? 'They are held against automatic idle unloading until the scan finishes; an engine that was already running is never touched.'
                    : 'Nothing was stopped afterwards.'),
              })
            : null,

          state.status === 'error'
            ? h('div', {
                style: {
                  border: '1px solid ' + text.error,
                  borderRadius: '12px',
                  padding: '10px 14px',
                  fontSize: '12px',
                  color: text.error,
                },
                role: 'alert',
                children: [
                  'The inventory could not be read: ' + state.error + '. ' +
                    (data ? 'Showing stale data from the last successful check. ' : 'No inventory data is available. ') +
                    'This does not mean the engines are absent. ',
                  h(Button, { onClick: () => load(lastQuery.current), children: 'Retry' }),
                ],
              })
            : null,

          data && data.error
            ? h('div', {
                style: {
                  border: '1px solid ' + text.error,
                  borderRadius: '12px',
                  padding: '10px 14px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                },
                children: [
                  h('div', {
                    style: { fontSize: '13px', fontWeight: 500, color: text.error },
                    children: 'The model hub is not loaded',
                  }),
                  h('div', {
                    style: { fontSize: '12px', color: text.primary, lineHeight: '18px', userSelect: 'text' },
                    children: data.error,
                  }),
                  h('div', {
                    style: { fontSize: '12px', color: text.tertiary, lineHeight: '18px' },
                    children:
                      'Fix the catalog file named below and reload the plugin, or restart DeepSeek Harness. ' +
                      'Until then no model tools are available.',
                  }),
                ],
              })
            : null,

          data && !data.error && data.diagnostics && data.diagnostics.length > 0
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
                  h('div', { style: { fontSize: '13px', fontWeight: 500 }, children: 'Catalog notes' }),
                  ...data.diagnostics.map((entry, index) =>
                    h('div', {
                      key: 'diagnostic-' + index,
                      style: {
                        fontSize: '12px',
                        lineHeight: '18px',
                        color: entry.severity === 'error' ? text.error : text.secondary,
                        userSelect: 'text',
                      },
                      children: entry.message,
                    }),
                  ),
                ],
              })
            : null,

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
                  state.stale || state.status === 'refreshing' ? h(Pill, { tone: 'muted',
                    children: state.stale ? (state.status === 'refreshing' ? 'stale — retrying' : 'stale — refresh failed')
                      : 'previous snapshot — refreshing' }) : null,
                ],
              })
            : null,

          data
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
                  h('div', {
                    style: { display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' },
                    children: [
                      h('div', { style: { fontSize: '13px', fontWeight: 500 }, children: 'Local resources' }),
                      h('span', {
                        style: { fontSize: '12px', color: text.tertiary },
                        children:
                          ollamaCount + ' Ollama Model(s) · ' + workflowCount + ' ComfyUI Workflow(s)',
                      }),
                    ],
                  }),
                  h(Row, { label: 'freshness', value: state.stale ? 'stale — last successful inventory' :
                    ((data.scan && data.scan.freshness) || 'last scan snapshot; use Scan to re-read resources') }),
                  h(Row, { label: 'scanned at', value: data.scan && (data.scan.checkedAt || data.scan.scannedAt) }),
                  resources.length === 0
                    ? h('div', {
                        style: { fontSize: '12px', color: text.tertiary, lineHeight: '18px' },
                        children:
                          'Nothing found yet. Press Scan to look for installed Ollama models and complete ComfyUI workflows.',
                      })
                    : h('div', {
                        style: { display: 'flex', flexDirection: 'column', gap: '6px' },
                        children: resources.map((resource) => h(ResourceItem, { key: resource.id, resource })),
                      }),
                  ...scanWarnings.map((warning, index) =>
                    h('div', {
                      key: 'scan-warning-' + index,
                      style: { fontSize: '12px', color: text.error, lineHeight: '18px' },
                      children: warning,
                    }),
                  ),
                  scanSources.length > 0
                    ? h('div', {
                        style: { display: 'flex', flexDirection: 'column', gap: '2px' },
                        children: [
                          h('div', { style: { fontSize: '12px', color: text.tertiary }, children: 'sources' }),
                          ...scanSources.map((source) =>
                            h(Row, {
                              key: source.id + ':' + source.kind,
                              label: source.label,
                              value: (source.ok ? 'read' : 'failed') + ' — ' + source.detail,
                            }),
                          ),
                        ],
                      })
                    : null,
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
                        h(Pill, { tone: 'muted', children: model.freshness === 'live' ? 'live checked' : 'catalog state — not live verified' }),
                        h(Row, { label: 'checked at', value: model.checkedAt }),
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

    /**
     * Keep both panels mounted: switching tabs or rescanning the inventory must
     * never discard a half-finished policy draft. Only visibility changes.
     */
    function LocalModelsPage() {
      const [tab, setTab] = useState('inventory');
      const selectTab = (value) => {
        setTab(value);
        // Keyboard tabs follow the WAI-ARIA roving-tabindex pattern.
        const target = typeof document !== 'undefined' && document.getElementById('mh-tab-' + value);
        if (target) target.focus();
      };
      const onTabKeyDown = (event) => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        selectTab(event.key === 'Home' ? 'inventory' : event.key === 'End' ? 'priorities' : tab === 'inventory' ? 'priorities' : 'inventory');
      };
      return h('div', {
        className: 'mh-page',
        children: [
          h('style', { children: STYLES }),
          h('header', {
            className: 'mh-header', style: { direction: 'ltr' },
            children: [
              h(Button, {
                onClick: () => selectTab('priorities'), ariaLabel: 'Edit priorities',
                children: [h(Icon, { name: 'settings', size: 18 }), 'Edit priorities'],
              }),
              h('div', { className: 'mh-mark', children: h(Icon, { name: 'box', size: 26 }) }),
              h('div', { children: [
                h('h1', { className: 'mh-heading', children: 'Local models' }),
                h('p', { className: 'mh-subtitle', children: 'Your models, providers, and routing preferences — in one place.' }),
              ] }),
            ],
          }),
          h('div', {
            className: 'mh-tabs', role: 'tablist', 'aria-label': 'Local models views',
            children: [['inventory', 'Inventory', 'inventory'], ['priorities', 'Priorities', 'settings']].map(([value, label, icon]) =>
              h('button', {
                key: value, type: 'button', className: 'mh-tab', id: 'mh-tab-' + value,
                role: 'tab', 'aria-selected': tab === value, 'aria-controls': 'mh-panel-' + value,
                tabIndex: tab === value ? 0 : -1,
                onClick: () => setTab(value), onKeyDown: onTabKeyDown,
                children: [h(Icon, { name: icon, size: 16 }), label],
              })),
          }),
          h('div', {
            id: 'mh-panel-inventory', className: 'mh-inventory', role: 'tabpanel',
            'aria-labelledby': 'mh-tab-inventory', hidden: tab !== 'inventory',
            children: h(LocalModelsSection, {}),
          }),
          h('div', {
            id: 'mh-panel-priorities', role: 'tabpanel',
            'aria-labelledby': 'mh-tab-priorities', hidden: tab !== 'priorities',
            children: h(CapabilityPriorities, {}),
          }),
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
          LocalModelsPage,
        );
      });
    }

    module.exports = { apply, inject, name };
    return module.exports;
  },
});
