/**
 * Machine resource detection.
 *
 * The router can only respect a model's declared VRAM/RAM requirements if it
 * knows what the machine has. Detection here is deliberately conservative:
 * claiming less than the machine offers is safe (it makes routing pick smaller
 * models), while claiming more makes routing choose a model that will fail at
 * load time with an opaque engine error.
 *
 * Two different questions are answered, and the shape of {@link MachineProfile}
 * keeps them apart:
 *
 * - **Capacity** — "could this model ever run here?" — from total RAM and total
 *   VRAM. This is what disqualifies a 24 GB model on an 8 GB card.
 * - **Headroom** — "will it fit right now?" — from free RAM and free VRAM. This
 *   is what stops a second model being started onto a GPU that the first one has
 *   already filled, which on a single-GPU laptop is the difference between a
 *   clean routing rejection and a CUDA out-of-memory crash inside the engine.
 *
 * A figure that cannot be measured is left *absent* rather than guessed at, and
 * every consumer treats absence as "fall back to capacity". Inventing a number
 * here would silently change which models route.
 *
 * @module dsh-ai-model-hub/machine
 */
import { readFile } from 'node:fs/promises';
import { cpus, freemem, totalmem } from 'node:os';
import { statfs } from 'node:fs/promises';
import { DEFAULT_EXECUTION_POLICY, runCommand, withPolicy } from "./util/process.js";
/** Bytes in one gibibyte. */
const GIB = 1024 ** 3;
/** Mebibytes in one gibibyte, for `nvidia-smi`'s reporting unit. */
const MIB_PER_GIB = 1024;
/**
 * The one executable the resource probe runs, and why it is exempt from the
 * command allowlist.
 *
 * The allowlist exists to bound *what the hub launches on a model's behalf*: an
 * agent-influenced catalog must not be able to reach a general-purpose
 * interpreter. `nvidia-smi` is neither agent-influenced nor general-purpose — the
 * arguments are a constant in this file, the output is parsed as numbers, and if
 * the binary is missing or a fake, the probe simply reports no GPU. Without this,
 * detection fails on every machine whose allowlist has been narrowed, and a
 * machine that reports no GPU disqualifies every GPU model.
 */
const GPU_PROBE_COMMAND = 'nvidia-smi';
/**
 * Detect the machine's usable resources.
 *
 * RAM comes from the OS: total from `totalmem()`, free from `freemem()`. Free
 * system RAM is the least trustworthy figure on the machine — every filesystem
 * cache looks "used" to some APIs and "free" to others — so the platform-correct
 * reading is used where one exists (Linux's `MemAvailable`) and no figure at all
 * is reported where none does.
 *
 * GPU memory is probed with `nvidia-smi` when present, which covers the
 * overwhelmingly common local-inference case. The query asks for total *and* free
 * memory per device, so a GPU already holding another model's weights is visible
 * rather than invisible. On macOS the unified-memory architecture means system
 * RAM is the relevant budget and the profile reports no discrete VRAM. A machine
 * with no detectable GPU reports `hasGpu: false` and zero VRAM, which correctly
 * disqualifies models that declare `requiresGpu: true`.
 *
 * @param options - probe controls.
 * @returns the profile plus evidence lines.
 */
export async function probeMachine(options = {}) {
    const evidence = [];
    const totalRamGb = round1(totalmem() / GIB);
    const cpuCount = cpus().length;
    evidence.push(`system RAM ${totalRamGb} GiB across ${cpuCount} logical cores`);
    const availableRamGb = await probeAvailableRam(options.timeoutMs ?? 5_000);
    if (availableRamGb !== undefined) {
        evidence.push(`system RAM available now ${availableRamGb} GiB (${describeRamSource()})`);
    }
    else {
        evidence.push('available system RAM could not be read on this platform; resource checks use total RAM');
    }
    let gpus = [];
    if (options.skipGpuProbe !== true) {
        const probe = await probeNvidia(options.timeoutMs ?? 5_000, options.policy);
        if (probe !== undefined) {
            gpus = probe.gpus;
            evidence.push(probe.detail);
        }
        else {
            evidence.push('no GPU detected via nvidia-smi; GPU-required models will be marked unsupported');
        }
    }
    else {
        evidence.push('GPU probe skipped by configuration');
    }
    const vramGb = round1(gpus.reduce((sum, gpu) => sum + gpu.vramGb, 0));
    const freeKnown = gpus.length > 0 && gpus.every((gpu) => gpu.freeVramGb !== undefined);
    const availableVramGb = freeKnown
        ? round1(gpus.reduce((sum, gpu) => sum + (gpu.freeVramGb ?? 0), 0))
        : undefined;
    let availableDiskGb;
    if (options.skipDiskProbe !== true) {
        const disk = await probeDisk(options.diskPath ?? process.cwd());
        if (disk !== undefined) {
            availableDiskGb = disk.availableGb;
            evidence.push(`${disk.availableGb} GiB free on ${disk.path}`);
        }
        else {
            evidence.push('free disk space could not be read');
        }
    }
    const profile = {
        vramGb,
        ramGb: totalRamGb,
        hasGpu: gpus.length > 0 && vramGb > 0,
        notes: evidence.join('; '),
        gpus,
        platform: process.platform,
        arch: process.arch,
        probedAt: Date.now(),
        ...(availableVramGb === undefined ? {} : { availableVramGb }),
        ...(availableRamGb === undefined ? {} : { availableRamGb }),
        ...(availableDiskGb === undefined ? {} : { availableDiskGb }),
    };
    return { profile, evidence };
}
/**
 * Query `nvidia-smi` for every device's name, total and free memory.
 *
 * Uses `--query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits`,
 * which prints one comma-separated mebibyte pair per GPU. A machine without the
 * tool, or with a driver too old for the `memory.free` field, yields `undefined`
 * rather than an error — absence of a probe result must never fail startup. When
 * only the free-memory field is missing the devices are still reported, with
 * their headroom left unknown; a driver that can name a device but not its free
 * memory is a real case and is worth more than nothing.
 *
 * @param timeoutMs - per-probe timeout.
 * @param policy - the deployment's execution policy, widened for this one command.
 * @returns the per-device facts and a description, or `undefined` when unavailable.
 */
async function probeNvidia(timeoutMs, policy) {
    const result = await runCommand({
        command: GPU_PROBE_COMMAND,
        args: ['--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'],
    }, {
        timeoutMs,
        // Widened from whatever the deployment runs under, and only here: see
        // GPU_PROBE_COMMAND for why this is not the hole it looks like.
        policy: withPolicy(policy ?? DEFAULT_EXECUTION_POLICY, { allowAnyCommand: true }),
        label: 'machine resource probe',
    });
    if (!result.ok)
        return undefined;
    const lines = result.stdout
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);
    if (lines.length === 0)
        return undefined;
    const gpus = [];
    for (const line of lines) {
        const [name, total, free] = line.split(',').map((part) => part.trim());
        const totalMib = Number(total);
        if (!Number.isFinite(totalMib) || totalMib <= 0)
            continue;
        const freeMib = Number(free);
        const deviceName = name !== undefined && name.length > 0 ? name : 'GPU';
        gpus.push({
            name: deviceName,
            vramGb: round1(totalMib / MIB_PER_GIB),
            ...(Number.isFinite(freeMib) && freeMib >= 0 ? { freeVramGb: round1(freeMib / MIB_PER_GIB) } : {}),
        });
    }
    if (gpus.length === 0)
        return undefined;
    const totalGb = round1(gpus.reduce((sum, gpu) => sum + gpu.vramGb, 0));
    const names = gpus.map((gpu) => gpu.name).join(', ');
    const freeText = gpus.every((gpu) => gpu.freeVramGb !== undefined)
        ? `, ${round1(gpus.reduce((sum, gpu) => sum + (gpu.freeVramGb ?? 0), 0))} GiB free`
        : '';
    return {
        gpus,
        detail: `${gpus.length} GPU(s) totalling ${totalGb} GiB VRAM${freeText} (${names})`,
    };
}
/**
 * Read available system RAM the way the platform means it.
 *
 * Linux reports `MemAvailable`, which is the kernel's own estimate of what a new
 * workload could claim without swapping — `freemem()` there returns `MemFree`,
 * which excludes reclaimable page cache and reads alarmingly low on a machine
 * that has simply been reading files. On Windows and macOS `freemem()` is the
 * best available figure and is used directly.
 *
 * @param timeoutMs - budget for the Linux probe.
 * @returns free gibibytes, or `undefined` when it cannot be measured.
 */
async function probeAvailableRam(timeoutMs) {
    if (process.platform === 'linux') {
        const available = await readMemAvailableKb(timeoutMs);
        if (available !== undefined)
            return round1((available * 1024) / GIB);
    }
    try {
        const free = freemem() / GIB;
        if (!Number.isFinite(free) || free < 0)
            return undefined;
        return round1(free);
    }
    catch {
        return undefined;
    }
}
/**
 * Read `MemAvailable` from `/proc/meminfo`, in kibibytes.
 * @param _timeoutMs - unused; `readFile` on procfs does not block meaningfully.
 * @returns the figure, or `undefined` when the file is absent or unparseable.
 */
async function readMemAvailableKb(_timeoutMs) {
    try {
        const text = await readFile('/proc/meminfo', 'utf8');
        const match = /^MemAvailable:\s+(\d+)\s*kB$/m.exec(text);
        if (match === null)
            return undefined;
        const kb = Number(match[1]);
        return Number.isFinite(kb) && kb > 0 ? kb : undefined;
    }
    catch {
        return undefined;
    }
}
/**
 * A human label for where the free-RAM figure came from.
 * @returns `MemAvailable` on Linux, `freemem()` elsewhere.
 */
function describeRamSource() {
    return process.platform === 'linux' ? 'MemAvailable' : 'os.freemem()';
}
/**
 * Measure free space on the filesystem a path lives on.
 *
 * Only free space is reported: a declared `diskGb` requirement is about weights
 * that may not be downloaded yet, so total capacity says little. `statfs` is
 * used rather than a spawned `df`, which keeps this free of a subprocess and of
 * the argument-parsing that a shelled-out probe would need.
 *
 * @param path - any path on the filesystem of interest.
 * @returns free gibibytes plus the path measured, or `undefined` on failure.
 */
async function probeDisk(path) {
    try {
        const stats = await statfs(path);
        const available = (stats.bavail * stats.bsize) / GIB;
        if (!Number.isFinite(available) || available < 0)
            return undefined;
        return { availableGb: round1(available), path };
    }
    catch {
        return undefined;
    }
}
/**
 * Round to one decimal place, the precision resource figures are reported at.
 * @param value - the number to round.
 * @returns the rounded value.
 */
function round1(value) {
    return Math.round(value * 10) / 10;
}
