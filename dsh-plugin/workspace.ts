/**
 * Where a tool call's artifacts go.
 *
 * The hub is constructed once, when the profile boots — before any session
 * exists — and one host serves many sessions with different workspaces. So the
 * artifact root cannot be decided at construction, and the plugin's own default
 * (`<process cwd>/artifacts`) is worse than useless for a long-lived `dsh web`,
 * whose working directory has nothing to do with the conversation's.
 *
 * This module answers the question per call instead, from DSH's own view of the
 * calling session: `ctx.sandboxPolicy.resolve({ session }).workspaceRoot` is the
 * session's immutable `cwd` — the same value the file tools treat as the
 * workspace boundary — and artifacts land in `artifacts/` under it. That is the
 * shape the hub already documents ("workspace-local and containable"), just
 * anchored to the right workspace.
 *
 * Precedence, highest first:
 *
 * 1. an explicit `artifactRoot` in the plugin row — an operator who names a
 *    directory gets that directory, in every session;
 * 2. the calling session's workspace, via the sandbox policy;
 * 3. the session's recorded `cwd` directly, when no policy service is mounted;
 * 4. nothing, which leaves the hub's own store in place — the pre-existing
 *    process-working-directory default.
 *
 * @module dsh-ai-model-hub/dsh-plugin/workspace
 */

import { isAbsolute, join } from 'node:path';
import type { Context } from '@deepseek-ai/cordis';
import type { PluginLogger, SandboxPolicyService, ToolCallScope } from './types.ts';

/** The subdirectory artifacts land in, under a workspace. */
export const ARTIFACTS_DIRECTORY = 'artifacts';

/**
 * Answer the artifact root for one tool call.
 *
 * @param runContext - the tool call's run context, or `undefined` outside a call.
 * @returns the absolute root to use, or `undefined` to leave the hub's own store.
 */
export interface ArtifactRootResolver {
  (runContext: ToolCallScope | undefined): string | undefined;
}

/**
 * Read the calling session's workspace out of a run context.
 *
 * @param runContext - the tool call's run context.
 * @returns the absolute session cwd, when the call carries one.
 */
export function sessionCwdOf(runContext: ToolCallScope | undefined): string | undefined {
  const cwd = runContext?.agent?.session?.cwd;
  return typeof cwd === 'string' && cwd.length > 0 ? cwd : undefined;
}

/**
 * Build the resolver the tools call once per invocation.
 *
 * @param ctx - the plugin context, read for the sandbox policy.
 * @param configuredRoot - the deployment's `artifactRoot`, empty when unset.
 * @param log - the plugin's logger, used once per unexpected policy failure.
 * @returns the resolver.
 */
export function createArtifactRootResolver(
  ctx: Context,
  configuredRoot: string,
  log: PluginLogger,
): ArtifactRootResolver {
  if (configuredRoot.length > 0) {
    // An explicit deployment choice outranks the session, and returning nothing
    // is how that is expressed: the hub's own store is already rooted there.
    return () => undefined;
  }

  let warned = false;
  return (runContext) => {
    const session = runContext?.agent?.session;
    const policy = ctx.get('sandboxPolicy') as SandboxPolicyService | undefined;
    if (policy !== undefined && session !== undefined) {
      try {
        const root = policy.resolve({ session }).workspaceRoot;
        if (typeof root === 'string' && isAbsolute(root)) return join(root, ARTIFACTS_DIRECTORY);
      } catch (error) {
        // A policy that cannot resolve is DSH's problem, not a reason to fail the
        // call: fall through to the session's recorded cwd, and warn once.
        if (!warned) {
          warned = true;
          log.warn(
            `could not resolve the sandbox workspace for this call (${error instanceof Error ? error.message : String(error)}); falling back to the session cwd`,
          );
        }
      }
    }
    const cwd = sessionCwdOf(runContext);
    return cwd === undefined || !isAbsolute(cwd) ? undefined : join(cwd, ARTIFACTS_DIRECTORY);
  };
}
