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
    const { useCallback, useEffect, useState } = React;

    /** The host route this page reads. */
    const ENDPOINT = '/dsh-ai-model-hub/inventory';

    /** The host route the Capability priorities editor reads and writes. */
    const PRIORITIES_ENDPOINT = '/dsh-ai-model-hub/priorities';

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

    /** A plain outlined button, shared by both sections. */
    function Button(props) {
      const disabled = props.disabled === true;
      return h('button', {
        type: 'button',
        onClick: props.onClick,
        disabled: disabled,
        'aria-label': props.ariaLabel,
        style: {
          border: '1px solid ' + text.border,
          background: 'transparent',
          color: text.primary,
          borderRadius: '8px',
          padding: '4px 14px',
          fontSize: '12px',
          cursor: disabled ? 'default' : 'pointer',
          opacity: disabled ? 0.5 : 1,
        },
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

      const load = useCallback(() => {
        setState((previous) => ({ ...previous, status: previous.payload ? 'refreshing' : 'loading' }));
        fetch(PRIORITIES_ENDPOINT)
          .then((response) => {
            if (!response.ok) throw new Error('HTTP ' + response.status);
            return response.json();
          })
          .then((payload) => {
            setState({ status: 'ready', error: '', payload });
            setDraft(draftFrom(payload));
            setDirty(false);
          })
          .catch((error) =>
            setState({ status: 'error', payload: null, error: String((error && error.message) || error) }),
          );
      }, []);

      useEffect(() => {
        load();
      }, [load]);

      const move = useCallback((capability, from, to) => {
        setDraft((previous) => {
          if (!previous) return previous;
          const list = previous[capability];
          if (!list || to < 0 || to >= list.length || from === to) return previous;
          const next = list.slice();
          const [item] = next.splice(from, 1);
          next.splice(to, 0, item);
          return { ...previous, [capability]: next };
        });
        setDirty(true);
        setNotice('');
      }, []);

      const toggle = useCallback((capability, id) => {
        setDraft((previous) => {
          if (!previous) return previous;
          const list = previous[capability];
          if (!list) return previous;
          return {
            ...previous,
            [capability]: list.map((row) => (row.id === id ? { ...row, enabled: !row.enabled } : row)),
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
        if (!draft) return;
        setSaving(true);
        setNotice('');
        const globalChanged = draft.globalEnabled !== state.payload.globalEnabled;
        const edits = Object.keys(draft.capabilities).filter((capability) => {
          const before = state.payload.capabilities.find((entry) => entry.capability === capability);
          const beforeOrder = before ? before.options.map((option) => option.id + ':' + option.enabled) : [];
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
          setSaving(false);
          setDirty(false);
          setNotice('Nothing changed.');
          return;
        }
        (async () => {
          let payload = state.payload;
          for (const step of steps) {
            const response = await fetch(PRIORITIES_ENDPOINT, {
              method: 'PUT',
              headers: { 'content-type': 'application/json' },
              body: JSON.stringify(step),
            });
            const body = await response.json().catch(() => ({}));
            if (!response.ok) {
              throw new Error(
                (body && body.error ? body.error : 'HTTP ' + response.status) +
                  (body && body.issues ? ' — ' + body.issues.join('; ') : ''),
              );
            }
            payload = body;
          }
          setState({ status: 'ready', error: '', payload });
          setDraft(draftFrom(payload));
          setDirty(false);
          setSaving(false);
          setNotice('Saved. ' + steps.length + ' change(s) written to the capability policy.');
        })().catch((error) => {
          setSaving(false);
          setNotice('Not saved: ' + String((error && error.message) || error));
        });
      }, [draft, state.payload]);

      const cancel = useCallback(() => {
        setDraft(draftFrom(state.payload));
        setDirty(false);
        setNotice('Reverted to the saved policy.');
      }, [state.payload]);

      const payload = state.payload;
      const capabilities = payload ? payload.capabilities : [];

      return h('section', {
        style: { maxWidth: '720px', color: text.primary, display: 'flex', flexDirection: 'column', gap: '8px' },
        children: [
          h('h2', {
            style: { margin: '20px 0 0', fontSize: '16px', fontWeight: 500, lineHeight: '24px' },
            children: 'Capability priorities',
          }),
          h('p', {
            style: { margin: 0, fontSize: '14px', color: text.tertiary, lineHeight: '22px' },
            children:
              'Which option serves each capability, in what order, and whether it is allowed to at all. ' +
              'Enabled permits an option to serve this capability. Disabled forbids it: it is never run, not even as a ' +
              'fallback, and its engine is never started for it. Lower priority means it is tried only after every ' +
              'higher-ranked permitted option has definitively failed. Stopped means the engine is not running yet — ' +
              'an enabled, stopped, startable option keeps its place and is started on demand. Nothing new is enabled ' +
              'until you enable it.',
          }),

          state.status === 'error'
            ? h('div', {
                style: {
                  border: '1px solid ' + text.error,
                  borderRadius: '12px',
                  padding: '10px 14px',
                  fontSize: '12px',
                  color: text.error,
                  lineHeight: '18px',
                },
                children:
                  'The capability policy could not be read: ' +
                  state.error +
                  '. Everything else on this page still works; the model hub keeps routing with the policy already in force.',
              })
            : null,

          payload
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
                    style: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
                    children: [
                      h('div', { style: { fontSize: '13px', fontWeight: 500 }, children: 'Global switch' }),
                      h(Pill, {
                        tone: draft && draft.globalEnabled ? 'ok' : 'error',
                        children: draft && draft.globalEnabled ? 'capabilities may run' : 'all disabled',
                      }),
                      h(Button, {
                        onClick: toggleGlobal,
                        disabled: saving || !draft,
                        ariaLabel: draft && draft.globalEnabled ? 'Disable every capability' : 'Enable capabilities',
                        children: draft && draft.globalEnabled ? 'Disable all capabilities' : 'Enable capabilities',
                      }),
                      dirty ? h(Pill, { tone: 'muted', children: 'unsaved changes' }) : null,
                    ],
                  }),
                  h('div', {
                    style: { fontSize: '12px', color: text.tertiary, lineHeight: '18px' },
                    children:
                      'Global disable wins over every per-capability setting: nothing runs, including options you have ' +
                      'enabled, and every request reports that the global switch is off.',
                  }),
                  h(Row, { label: 'policy file', value: payload.policyPath, mono: true }),
                  h(Row, { label: 'revision', value: String(payload.revision) }),
                ],
              })
            : null,

          payload
            ? h('div', {
                style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' },
                children: [
                  button(saving ? 'Saving…' : 'Save', save, saving || !dirty),
                  button('Cancel', cancel, saving || !dirty),
                  button(
                    state.status === 'refreshing' ? 'Reloading…' : 'Reload',
                    load,
                    saving || state.status === 'refreshing',
                  ),
                  notice
                    ? h('span', { style: { fontSize: '12px', color: text.tertiary }, children: notice })
                    : dirty
                      ? h('span', {
                          style: { fontSize: '12px', color: text.tertiary },
                          children: 'Unsaved: press Save to write this order.',
                        })
                      : null,
                ],
              })
            : null,

          state.status === 'loading'
            ? h('div', {
                style: {
                  border: '1px solid ' + text.border,
                  borderRadius: '12px',
                  padding: '10px 14px',
                  fontSize: '12px',
                  color: text.tertiary,
                },
                children: 'Reading the capability policy and probing every option…',
              })
            : null,

          ...capabilities.map((entry) =>
            h(CapabilityCard, {
              key: entry.capability,
              entry,
              rows: draft ? draft.capabilities[entry.capability] || [] : [],
              disabled: saving || !draft,
              dragging: dragging,
              onDragStateChange: setDragging,
              onMove: move,
              onToggle: toggle,
            }),
          ),
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
          enabled: option.enabled,
        }));
      }
      return { globalEnabled: payload.globalEnabled, capabilities: capabilities };
    }

    /** One capability's list. */
    function CapabilityCard(props) {
      const entry = props.entry;
      const rows = props.rows;
      const enabledCount = rows.filter((row) => row.enabled).length;
      return h('div', {
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
              h('span', {
                style: { fontFamily: MONO, fontSize: '13px', fontWeight: 500, color: text.primary, userSelect: 'text' },
                children: entry.capability,
              }),
              h('span', {
                style: { fontSize: '12px', color: text.tertiary },
                children:
                  rows.length +
                  ' option(s) · ' +
                  enabledCount +
                  ' enabled · order: ' +
                  (entry.permitted.length > 0 ? entry.permitted.join(' → ') : 'nothing permitted'),
              }),
            ],
          }),
          entry.inputTypes && entry.inputTypes.length > 0
            ? h(Row, {
                label: 'accepts',
                value: entry.inputTypes.join(', ') + '  →  ' + (entry.outputTypes || []).join(', '),
              })
            : null,
          rows.length === 0
            ? h('div', {
                style: { fontSize: '12px', color: text.tertiary, lineHeight: '18px' },
                children: 'Nothing in this deployment declares this capability. It stays listed so the gap is visible.',
              })
            : h('div', {
                style: { display: 'flex', flexDirection: 'column', gap: '4px' },
                children: rows.map((row, index) =>
                  h(PriorityRow, {
                    key: row.id,
                    capability: entry.capability,
                    row: row,
                    index: index,
                    count: rows.length,
                    disabled: props.disabled,
                    dragging: props.dragging,
                    onDragStateChange: props.onDragStateChange,
                    onMove: props.onMove,
                    onToggle: props.onToggle,
                  }),
                ),
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
      const dim = row.enabled ? 1 : 0.55;
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
          style: {
            border: '1px solid ' + text.border,
            background: 'transparent',
            color: text.primary,
            borderRadius: '6px',
            padding: '1px 6px',
            fontSize: '11px',
            lineHeight: '16px',
            cursor: disabled === true ? 'default' : 'pointer',
            opacity: disabled === true ? 0.4 : 1,
          },
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
        style: {
          border: '1px solid ' + (isDragging ? text.ok : text.border),
          borderRadius: '8px',
          padding: '6px 8px',
          display: 'flex',
          flexDirection: 'column',
          gap: '3px',
          background: isDragging ? 'rgba(127,127,127,0.08)' : 'transparent',
        },
        children: [
          h('div', {
            style: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
            children: [
              h('span', {
                title: 'Drag to reorder',
                'aria-hidden': 'true',
                style: { cursor: props.disabled === true ? 'default' : 'grab', color: text.tertiary, fontSize: '13px' },
                children: '⠿',
              }),
              h('span', {
                style: { fontFamily: MONO, fontSize: '12px', color: text.tertiary, minWidth: '18px' },
                children: String(props.index + 1) + '.',
              }),
              h('span', { style: { fontSize: '13px', color: text.primary, opacity: dim }, children: row.name }),
              h(Pill, { tone: row.enabled ? 'ok' : 'error', children: row.enabled ? 'Enabled' : 'Disabled' }),
              h(Pill, { tone: stateTone, children: row.state }),
              row.missing ? h(Pill, { tone: 'muted', children: 'missing' }) : null,
            ],
          }),
          h(Row, {
            label: 'source',
            value: row.providerLabel + ' · ' + row.source + ' · ' + row.type,
          }),
          h(Row, {
            label: 'reason',
            value: row.reason,
          }),
          row.inputTypes && row.inputTypes.length > 0
            ? h(Row, {
                label: 'in/out',
                value: row.inputTypes.join('|') + ' → ' + (row.outputTypes || []).join('|'),
              })
            : null,
          h('div', {
            style: { display: 'flex', gap: '4px', alignItems: 'center', flexWrap: 'wrap' },
            children: [
              h('button', {
                type: 'button',
                onClick: () => props.onToggle(capability, row.id),
                disabled: props.disabled === true,
                'aria-pressed': row.enabled,
                'aria-label': (row.enabled ? 'Disable ' : 'Enable ') + row.name + ' for ' + capability,
                style: {
                  border: '1px solid ' + text.border,
                  background: 'transparent',
                  color: row.enabled ? text.error : text.ok,
                  borderRadius: '6px',
                  padding: '1px 8px',
                  fontSize: '11px',
                  cursor: props.disabled === true ? 'default' : 'pointer',
                  opacity: props.disabled === true ? 0.5 : 1,
                },
                children: row.enabled ? 'Disable' : 'Enable',
              }),
              control('↑', 'Move ' + row.name + ' up', () => props.onMove(capability, props.index, props.index - 1), props.disabled === true || props.index === 0),
              control('↓', 'Move ' + row.name + ' down', () => props.onMove(capability, props.index, props.index + 1), props.disabled === true || props.index === props.count - 1),
              control('⤒', 'Move ' + row.name + ' to the top', () => props.onMove(capability, props.index, 0), props.disabled === true || props.index === 0),
              control('⤓', 'Move ' + row.name + ' to the bottom', () => props.onMove(capability, props.index, props.count - 1), props.disabled === true || props.index === props.count - 1),
            ],
          }),
        ],
      });
    }

    /** The settings section itself. */
    function LocalModelsSection() {
      const [state, setState] = useState({ status: 'loading', data: null, error: '' });

      const load = useCallback((query) => {
        setState((previous) => ({ ...previous, status: previous.data ? 'refreshing' : 'loading' }));
        fetch(ENDPOINT + (query || ''))
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
        load('?scan=1');
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
      const resources = (data && data.resources) || [];
      const ollamaCount = resources.filter((entry) => entry.kind === 'ollama_model').length;
      const workflowCount = resources.filter((entry) => entry.kind === 'comfyui_workflow').length;
      const scanSources = (data && data.scan && data.scan.sources) || [];
      const scanWarnings = (data && data.scan && data.scan.warnings) || [];

      return h('section', {
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
              'The local engines on this machine — where each one is installed, whether it is running, and what it holds. Read from the active catalog plus a live check, so it always matches what the model hub would actually do.',
          }),
          h('div', {
            style: { display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' },
            children: [
              button(
                state.status === 'loading' ? 'Scanning…' : 'Scan',
                () => load('?scan=1'),
                state.status === 'loading' || state.status === 'refreshing',
              ),
              button('Refresh', () => load(''), state.status === 'loading' || state.status === 'refreshing'),
              state.status === 'refreshing'
                ? h('span', { style: { fontSize: '12px', color: text.tertiary }, children: 'scanning…' })
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
                children: 'Scanning this machine for Ollama models and ComfyUI workflows…',
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
                children:
                  'The scan could not be read: ' +
                  state.error +
                  '. The engines themselves may still be reachable — press Scan to try again.',
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
     * The settings section: the machine inventory, then the capability policy.
     *
     * Two independent components, not one wrapping the other. The inventory is a
     * report — it re-scans, it is read-only, and nothing in it is remembered. The
     * priorities editor is a document with unsaved state, and it must keep that
     * state while the inventory beside it is rescanned. Nesting one inside the
     * other would make a Scan button throw away a half-finished reorder, because a
     * parent re-render is a child remount.
     */
    function LocalModelsPage() {
      return h('div', {
        style: { display: 'flex', flexDirection: 'column', gap: '8px', maxWidth: '720px' },
        children: [h(LocalModelsSection, {}), h(CapabilityPriorities, {})],
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
