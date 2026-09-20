/**
 * Safe local process execution.
 *
 * This module is the *only* place in the hub that spawns a process. Everything
 * else asks it, which means the security posture is reviewable in one file.
 *
 * The threat model: the agent is untrusted input. It can influence which model
 * runs, but it must never be able to influence *what command line* runs. So the
 * hub never accepts a shell string, never interpolates model output into argv,
 * and refuses binaries outside an allowlist unless the operator opted in.
 *
 * Concretely:
 *
 * - Commands are spawned with `shell: false` and an argv array. There is no
 *   string to inject into, so `; rm -rf /` in a prompt is inert data.
 * - Only commands named in configuration can ever be spawned. The agent cannot
 *   name a command at all — the tool schema exposes no such parameter.
 * - The set of permitted executables is an explicit allowlist, configurable but
 *   defaulting to the inference engines this hub actually integrates with.
 * - Arguments are validated per-element: no NUL, no newlines, bounded length.
 * - Children get a scrubbed environment and are killed as a tree on timeout.
 *
 * @module dsh-ai-model-hub/util/process
 */
import { spawn } from 'node:child_process';
import { ModelHubError } from "../errors.js";
/**
 * Executables the hub will launch without the operator explicitly opting in.
 *
 * This is a deliberate allowlist of well-known local inference runtimes. It is
 * not a security boundary on its own — an operator can widen it — but it makes
 * the default safe: a catalog file edited by a careless copy-paste cannot invent
 * a new executable, and a compromised model config cannot reach a general-purpose
 * interpreter like `sh` or `cmd` unless a human allowed it.
 */
export const DEFAULT_COMMAND_ALLOWLIST = [
    'python',
    'python3',
    'python.exe',
    'py',
    'ollama',
    'llama-server',
    'llama.cpp',
    'llamafile',
    'vllm',
    'koboldcpp',
    'text-generation-launcher',
    'stable-diffusion.cpp',
    'sd',
    'comfy',
    'comfyui',
    'blender',
    'blender.exe',
    'piper',
    'whisper',
    'whisper-cli',
    'main',
    'node',
];
/** Thrown when a spawn request violates the execution policy. */
export class UnsafeCommandError extends ModelHubError {
    /**
     * @param message - why the command was refused.
     * @param details - structured context, including the offending element.
     */
    constructor(message, details = {}) {
        super('UNSAFE_OPERATION', message, details);
    }
}
/** The default policy: allowlisted engines, scrubbed secrets, bounded argv. */
export const DEFAULT_EXECUTION_POLICY = {
    allowlist: DEFAULT_COMMAND_ALLOWLIST,
    allowAnyCommand: false,
    maxArgumentLength: 4096,
    maxArgumentCount: 256,
    scrubEnvPatterns: [
        'API_KEY',
        'TOKEN',
        'SECRET',
        'PASSWORD',
        'CREDENTIAL',
        'DEEPSEEK',
        'OPENAI',
        'ANTHROPIC',
        'AWS_',
        'AZURE_',
        'GOOGLE_',
    ],
};
/** Narrows a policy, used by tests and by stricter deployments. */
export function withPolicy(base, override) {
    return { ...base, ...override };
}
/**
 * The basename of a command, lowercased and without a Windows extension.
 * @param command - an executable name or path.
 * @returns the comparable bare name.
 */
export function commandBasename(command) {
    const normalised = command.replace(/\\/g, '/');
    const lastSegment = normalised.slice(normalised.lastIndexOf('/') + 1);
    return lastSegment.toLowerCase().replace(/\.(exe|cmd|bat|com)$/, '');
}
/**
 * Refuse arguments that cannot safely be passed to a process.
 *
 * A NUL truncates a C string and can smuggle a different argument past a check
 * that inspected the pre-truncation value; a newline is harmless to `spawn` but
 * is a strong signal that someone is building a shell line and got this far by
 * mistake. Both are treated as hostile rather than sanitized.
 *
 * @param args - the candidate argument list.
 * @param policy - the limits to apply.
 * @param context - what is being launched, for the error message.
 * @throws UnsafeCommandError when any element is unacceptable.
 */
export function assertSafeArguments(args, policy, context) {
    if (args.length > policy.maxArgumentCount) {
        throw new UnsafeCommandError(`${context}: refusing to launch with ${args.length} arguments (limit ${policy.maxArgumentCount})`, { argumentCount: args.length, limit: policy.maxArgumentCount });
    }
    args.forEach((arg, index) => {
        if (arg.includes('\0')) {
            throw new UnsafeCommandError(`${context}: argument ${index} contains a NUL byte`, { index });
        }
        if (/[\r\n]/.test(arg)) {
            throw new UnsafeCommandError(`${context}: argument ${index} contains a line break, which suggests a shell command was built as a string`, { index });
        }
        if (arg.length > policy.maxArgumentLength) {
            throw new UnsafeCommandError(`${context}: argument ${index} is ${arg.length} characters (limit ${policy.maxArgumentLength})`, { index, length: arg.length, limit: policy.maxArgumentLength });
        }
    });
}
/**
 * Refuse executables the deployment has not permitted.
 * @param command - the executable name or path from configuration.
 * @param policy - the policy to enforce.
 * @param context - what is being launched, for the error message.
 * @throws UnsafeCommandError when the executable is not permitted.
 */
export function assertAllowedCommand(command, policy, context) {
    if (command.trim().length === 0) {
        throw new UnsafeCommandError(`${context}: command is empty`);
    }
    if (command.includes('\0')) {
        throw new UnsafeCommandError(`${context}: command contains a NUL byte`);
    }
    if (policy.allowAnyCommand)
        return;
    const base = commandBasename(command);
    const allowed = policy.allowlist.some((entry) => commandBasename(entry) === base);
    if (!allowed) {
        throw new UnsafeCommandError(`${context}: "${command}" is not in the command allowlist. ` +
            `Either use an allowlisted engine (${policy.allowlist.slice(0, 8).join(', ')}, …) ` +
            'or set `allowAnyCommand: true` in the hub configuration after reviewing what it runs.', { command, basename: base, allowlist: [...policy.allowlist] });
    }
}
/**
 * Build the child environment: the operator's, minus credential-shaped names,
 * plus the configuration's own variables.
 *
 * @param extra - variables the descriptor explicitly added.
 * @param policy - the scrubbing policy.
 * @returns an environment object safe to hand a model process.
 */
export function buildChildEnvironment(extra, policy) {
    const result = {};
    const scrubbed = policy.scrubEnvPatterns.map((pattern) => pattern.toUpperCase());
    for (const [key, value] of Object.entries(process.env)) {
        if (value === undefined)
            continue;
        const upper = key.toUpperCase();
        if (scrubbed.some((pattern) => upper.includes(pattern)))
            continue;
        result[key] = value;
    }
    if (extra !== undefined) {
        for (const [key, value] of Object.entries(extra))
            result[key] = value;
    }
    return result;
}
/**
 * Run a command to completion and capture its output.
 *
 * Used for health probes and for `nvidia-smi`, never for a model's own request
 * path (adapters that shell out use {@link spawnProcess} so they can stream).
 * Never rejects on a non-zero exit: a failed probe is a result, not an exception,
 * because probes run in the health-check path where throwing would be noise.
 *
 * @param spec - the command and its arguments.
 * @param options - timeout, policy, and environment.
 * @returns the outcome, with `ok` false for every failure mode.
 */
export async function runCommand(spec, options = {}) {
    const policy = options.policy ?? DEFAULT_EXECUTION_POLICY;
    const label = options.label ?? spec.command;
    const timeoutMs = options.timeoutMs ?? 10_000;
    const maxOutputChars = options.maxOutputChars ?? 65_536;
    try {
        assertAllowedCommand(spec.command, policy, label);
        assertSafeArguments(spec.args ?? [], policy, label);
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { ok: false, code: undefined, stdout: '', stderr: message, reason: 'spawn_failed' };
    }
    return new Promise((resolve) => {
        let settled = false;
        let stdout = '';
        let stderr = '';
        let timedOut = false;
        let aborted = false;
        const finish = (result) => {
            if (settled)
                return;
            settled = true;
            clearTimeout(timer);
            options.signal?.removeEventListener('abort', onAbort);
            resolve(result);
        };
        const child = spawn(spec.command, [...(spec.args ?? [])], {
            cwd: spec.cwd,
            env: buildChildEnvironment(options.env, policy),
            shell: false,
            windowsHide: true,
            stdio: ['ignore', 'pipe', 'pipe'],
        });
        const timer = setTimeout(() => {
            timedOut = true;
            child.kill('SIGKILL');
        }, timeoutMs);
        const onAbort = () => {
            aborted = true;
            child.kill('SIGKILL');
        };
        if (options.signal !== undefined) {
            if (options.signal.aborted)
                onAbort();
            else
                options.signal.addEventListener('abort', onAbort, { once: true });
        }
        child.stdout?.on('data', (chunk) => {
            if (stdout.length < maxOutputChars)
                stdout += chunk.toString('utf8');
        });
        child.stderr?.on('data', (chunk) => {
            if (stderr.length < maxOutputChars)
                stderr += chunk.toString('utf8');
        });
        child.on('error', (error) => {
            finish({
                ok: false,
                code: undefined,
                stdout: stdout.slice(0, maxOutputChars),
                stderr: `${error.message}\n${stderr}`.slice(0, maxOutputChars),
                reason: 'spawn_failed',
            });
        });
        child.on('close', (code) => {
            const truncatedOut = stdout.slice(0, maxOutputChars);
            const truncatedErr = stderr.slice(0, maxOutputChars);
            if (timedOut) {
                finish({ ok: false, code: code ?? undefined, stdout: truncatedOut, stderr: truncatedErr, reason: 'timeout' });
                return;
            }
            if (aborted) {
                finish({ ok: false, code: code ?? undefined, stdout: truncatedOut, stderr: truncatedErr, reason: 'aborted' });
                return;
            }
            finish({
                ok: code === 0,
                code: code ?? undefined,
                stdout: truncatedOut,
                stderr: truncatedErr,
                ...(code === 0 ? {} : { reason: 'non_zero_exit' }),
            });
        });
    });
}
/**
 * Start a long-lived process under the execution policy.
 *
 * The child is detached into its own process group on POSIX so that stopping it
 * also stops any worker it spawned — inference servers routinely fork helpers,
 * and killing only the parent leaves the port held.
 *
 * @param spec - the command and its arguments.
 * @param options - policy, environment, and output caps.
 * @returns a handle for observing and stopping the process.
 * @throws UnsafeCommandError when the policy refuses the launch.
 */
export function spawnProcess(spec, options = {}) {
    const policy = options.policy ?? DEFAULT_EXECUTION_POLICY;
    const label = options.label ?? spec.command;
    const maxOutputChars = options.maxOutputChars ?? 16_384;
    assertAllowedCommand(spec.command, policy, label);
    assertSafeArguments(spec.args ?? [], policy, label);
    const child = spawn(spec.command, [...(spec.args ?? [])], {
        cwd: spec.cwd,
        env: buildChildEnvironment(options.env, policy),
        shell: false,
        windowsHide: true,
        detached: process.platform !== 'win32',
        stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout?.on('data', (chunk) => {
        stdout = (stdout + chunk.toString('utf8')).slice(-maxOutputChars);
    });
    child.stderr?.on('data', (chunk) => {
        stderr = (stderr + chunk.toString('utf8')).slice(-maxOutputChars);
    });
    const exited = new Promise((resolve) => {
        child.once('close', (code) => resolve(code ?? undefined));
        child.once('error', () => resolve(undefined));
    });
    let stopped = false;
    return {
        pid: child.pid,
        get running() {
            return child.exitCode === null && child.signalCode === null && !stopped;
        },
        stderrSnapshot: () => stderr,
        stdoutSnapshot: () => stdout,
        async stop(graceMs) {
            if (child.exitCode !== null || child.signalCode !== null)
                return true;
            stopped = true;
            await terminateTree(child, graceMs);
            return child.exitCode !== null || child.signalCode !== null;
        },
        exited,
    };
}
/**
 * Terminate a child and, where the platform allows it, its descendants.
 *
 * POSIX: signal the process group, which catches forked workers. Windows: use
 * `taskkill /T`, because there is no process-group signal and a bare `kill`
 * orphans children that keep the listening port.
 *
 * @param child - the spawned child.
 * @param graceMs - how long to wait after the graceful signal.
 */
async function terminateTree(child, graceMs) {
    const pid = child.pid;
    if (pid === undefined)
        return;
    if (process.platform === 'win32') {
        const ending = new Promise((resolve) => {
            child.once('close', () => resolve());
            setTimeout(resolve, graceMs);
        });
        // Ask politely first so the engine can flush its state.
        try {
            child.kill();
        }
        catch {
            /* already gone */
        }
        await Promise.race([ending, delay(Math.min(graceMs, 2_000))]);
        if (child.exitCode === null && child.signalCode === null) {
            await runCommand({ command: 'taskkill', args: ['/PID', String(pid), '/T', '/F'] }, {
                timeoutMs: 5_000,
                policy: withPolicy(DEFAULT_EXECUTION_POLICY, { allowAnyCommand: true }),
            });
        }
        await Promise.race([ending, delay(graceMs)]);
        return;
    }
    const ending = new Promise((resolve) => {
        child.once('close', () => resolve());
        setTimeout(resolve, graceMs * 2);
    });
    try {
        process.kill(-pid, 'SIGTERM');
    }
    catch {
        try {
            child.kill('SIGTERM');
        }
        catch {
            /* already gone */
        }
    }
    await Promise.race([ending, delay(graceMs)]);
    if (child.exitCode === null && child.signalCode === null) {
        try {
            process.kill(-pid, 'SIGKILL');
        }
        catch {
            try {
                child.kill('SIGKILL');
            }
            catch {
                /* already gone */
            }
        }
    }
    await Promise.race([ending, delay(1_000)]);
}
/**
 * Sleep for a duration.
 * @param ms - milliseconds to wait.
 * @returns a promise that resolves after the delay.
 */
export function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
/**
 * A promise that rejects with `INVOCATION_TIMEOUT` when a deadline passes.
 *
 * Races an operation against a timer without leaking the timer: the winning
 * branch always clears it, so a long-lived process does not accumulate timers.
 *
 * Crucially, the losing branch is *contained*. A same-process operation cannot be
 * hard-killed, so when the deadline or an external abort wins, the operation is
 * still running and will settle later. If it rejects then, and nothing is
 * attached to it, Node treats it as an unhandled rejection and terminates the
 * process — which would turn "this model was slow" into "the whole harness
 * died". Attaching a no-op rejection handler marks the late failure as observed
 * while leaving the reason resolvable for diagnostics.
 *
 * @param operation - the work to bound.
 * @param timeoutMs - the budget in milliseconds.
 * @param label - what is being bounded, for the error message.
 * @param signal - optional external cancellation.
 * @returns the operation's result.
 * @throws ModelHubError with `INVOCATION_TIMEOUT` or `INVOCATION_ABORTED`.
 */
export function withTimeout(operation, timeoutMs, label, signal) {
    return new Promise((resolve, reject) => {
        let settled = false;
        // Bound the abandoned promise's later rejection; see the note above.
        operation.catch(() => undefined);
        const settle = (fn) => {
            if (settled)
                return;
            settled = true;
            clearTimeout(timer);
            signal?.removeEventListener('abort', onAbort);
            fn();
        };
        const timer = setTimeout(() => {
            settle(() => reject(new ModelHubError('INVOCATION_TIMEOUT', `${label} exceeded its ${timeoutMs} ms budget`, {
                timeoutMs,
            })));
        }, timeoutMs);
        const onAbort = () => {
            settle(() => reject(new ModelHubError('INVOCATION_ABORTED', `${label} was cancelled`)));
        };
        if (signal !== undefined) {
            if (signal.aborted) {
                onAbort();
                return;
            }
            signal.addEventListener('abort', onAbort, { once: true });
        }
        operation.then((value) => settle(() => resolve(value)), (error) => settle(() => reject(error)));
    });
}
