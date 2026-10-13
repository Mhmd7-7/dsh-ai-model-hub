/**
 * The rule for paths written in this plugin's configuration.
 *
 * Three spellings reach the plugin's own settings — `comfyuiWorkflowDir`, and the
 * directory parts of a configured engine's install — and all three mean the same
 * thing, so all three resolve here, once:
 *
 * - **Absolute, either separator.** `C:/workflows/comfy` and `C:\\workflows\\comfy`
 *   are both absolute on Windows, and `path.isAbsolute` already treats a forward
 *   slash in a drive-qualified path correctly.
 * - **Relative.** Relative to **the catalog that configured it**, following
 *   `src/adapters/paths.ts`: a configured directory is part of the catalog's own
 *   deployment, and a long-lived host's working directory says nothing about where
 *   that is.
 * - **Shell-flavoured.** `%USERPROFILE%/ComfyUI` and `~/ComfyUI` are how people
 *   write paths, so both are expanded before resolution, from the host's own
 *   environment — never from a hardcoded machine path, which is the whole point.
 *
 * A path this returns is what the filesystem probes are given, so a configured
 * directory behaves identically on Windows and on a POSIX host.
 *
 * @module dsh-ai-model-hub/dsh-plugin/paths
 */

import { homedir } from 'node:os';
import { isAbsolute, join, resolve } from 'node:path';

/**
 * Expand `%VAR%` and `~` in a configured path.
 *
 * `%VAR%` is Windows' spelling and `~` is everyone's; both are answered from
 * `environment` first, so a host that declares its own `runtime.env` gets its own
 * values rather than the host process's. An unknown variable is left literal
 * rather than replaced with an empty string, because a path with `%MISSING%` in it
 * fails loudly and names itself, while an empty segment silently addresses the
 * wrong directory.
 *
 * @param path - the configured path.
 * @param environment - values to prefer over the process environment.
 * @returns the path with variables and `~` expanded.
 */
export function expandConfiguredPath(
  path: string,
  environment: Readonly<Record<string, string>> | undefined = undefined,
): string {
  let result = path.replace(/%([A-Za-z_][A-Za-z0-9_]*)%/g, (token, name: string) => {
    return environment?.[name] ?? process.env[name] ?? process.env[name.toUpperCase()] ?? token;
  });
  result = result.replace(/\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)/g, (token, braced: string | undefined, plain: string | undefined) => {
    const name = braced ?? plain!;
    return environment?.[name] ?? process.env[name] ?? token;
  });
  if (result === '~') result = homedir();
  else if (result.startsWith('~/') || result.startsWith('~\\')) result = join(homedir(), result.slice(2));
  return result;
}

/**
 * Resolve a path this plugin's configuration wrote to something readable.
 *
 * @param path - the configured path, in any of the three spellings above.
 * @param catalogDir - the directory of the catalog that configured it, when one exists.
 * @param environment - `runtime.env` values to prefer during expansion.
 * @returns an absolute path.
 */
export function resolveConfiguredPath(
  path: string,
  catalogDir: string | undefined,
  environment: Readonly<Record<string, string>> | undefined = undefined,
): string {
  const expanded = expandConfiguredPath(path.trim(), environment);
  if (expanded.length === 0) return expanded;
  // A drive-qualified path with forward slashes is absolute even though a POSIX
  // reading of it is not, which is the one case `isAbsolute` alone gets wrong.
  if (isAbsolute(expanded) || /^[A-Za-z]:[/\\]/.test(expanded) || /^\\\\/.test(expanded)) return expanded;
  return resolve(catalogDir ?? process.cwd(), expanded);
}
