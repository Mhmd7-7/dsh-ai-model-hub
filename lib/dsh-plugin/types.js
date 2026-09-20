/**
 * Type bridges for the DeepSeek Harness plugin API.
 *
 * The DSH plugin layer pins its dependency on DSH to this one file plus the
 * handful of `@deepseek-ai/*` imports in the modules beside it. That is a
 * deliberate survivability decision: DSH is a `0.1.x-rc` line, so breaking
 * changes are expected. When one lands, the blast radius is here — not in the
 * catalog, router, runtime, or adapters, which import nothing from DSH at all and
 * are covered by tests that run without DSH installed.
 *
 * Everything added to the `Context` interface below comes from cordis's runtime
 * mixins. They exist at runtime but are absent from the published `Context`
 * declaration, so the augmentation records the real, verified behaviour rather
 * than guessing at an API. Each is documented with the source that establishes it.
 *
 * @module dsh-ai-model-hub/dsh-plugin/types
 */
export {};
