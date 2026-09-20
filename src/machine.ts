/**
 * Machine resource detection.
 *
 * The router can only respect a model's declared VRAM/RAM requirements if it
 * knows what the machine has. Detection here is deliberately conservative:
 * claiming less than the machine offers is safe (it makes routing pick smaller
 * models), while claiming more makes routing choose a model that will fail at
 * load time with an opaque engine error.
 *
 * @module dsh-ai-model-hub/machine
 */

import { cpus, totalmem } from 'node:os';
import { runCommand } from './util/process.ts';
import type { MachineProfile } from './types.ts';

/** Bytes in one gibibyte. */
const GIB = 1024 ** 3;

/** The detection result plus the evidence behind it. */
export interface MachineProbeResult {
  /** The detected profile. */
  readonly profile: MachineProfile;
  /** How each fact was established, for logs and `getModelStatus` output. */
  readonly evidence: readonly string[];
}

/**
 * Detect the machine's usable resources.
 *
 * RAM comes from the OS. GPU VRAM is probed with `nvidia-smi` when present,
 * which covers the overwhelmingly common local-inference case; on macOS the
 * unified-memory architecture means system RAM is the relevant budget and the
 * profile reports no discrete VRAM. A machine with no detectable GPU reports
 * `hasGpu: false` and zero VRAM, which correctly disqualifies models that
 * declare `requiresGpu: true`.
 *
 * @param options - probe controls.
 * @returns the profile plus evidence lines.
 */
export async function probeMachine(options: {
  /** Skip the (slow) GPU probe and report only OS memory. */
  readonly skipGpuProbe?: boolean;
  /** Per-probe timeout in milliseconds. Defaults to 5000. */
  readonly timeoutMs?: number;
} = {}): Promise<MachineProbeResult> {
  const evidence: string[] = [];
  const totalRamGb = Math.round((totalmem() / GIB) * 10) / 10;
  const cpuCount = cpus().length;
  evidence.push(`system RAM ${totalRamGb} GiB across ${cpuCount} logical cores`);

  let vramGb = 0;
  let hasGpu = false;

  if (options.skipGpuProbe !== true) {
    const probe = await probeNvidia(options.timeoutMs ?? 5_000);
    if (probe !== undefined) {
      vramGb = probe.vramGb;
      hasGpu = probe.vramGb > 0;
      evidence.push(probe.detail);
    } else {
      evidence.push('no GPU detected via nvidia-smi; GPU-required models will be marked unsupported');
    }
  } else {
    evidence.push('GPU probe skipped by configuration');
  }

  return {
    profile: {
      vramGb,
      ramGb: totalRamGb,
      hasGpu,
      notes: evidence.join('; '),
    },
    evidence,
  };
}

/**
 * Query `nvidia-smi` for total VRAM across all devices.
 *
 * Uses `--query-gpu=memory.total --format=csv,noheader,nounits`, which prints
 * one mebibyte count per GPU. A machine without the tool, or with a driver too
 * old for the query, yields `undefined` rather than an error — absence of a
 * probe result must never fail startup.
 *
 * @param timeoutMs - per-probe timeout.
 * @returns the summed VRAM and a description, or `undefined` when unavailable.
 */
async function probeNvidia(
  timeoutMs: number,
): Promise<{ vramGb: number; detail: string } | undefined> {
  const result = await runCommand(
    {
      command: 'nvidia-smi',
      args: ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
    },
    { timeoutMs },
  );
  if (!result.ok) return undefined;

  const lines = result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  if (lines.length === 0) return undefined;

  const names: string[] = [];
  let totalMib = 0;
  for (const line of lines) {
    const [name, memory] = line.split(',').map((part) => part.trim());
    const mib = Number(memory);
    if (!Number.isFinite(mib) || mib <= 0) continue;
    totalMib += mib;
    if (name !== undefined && name.length > 0) names.push(name);
  }
  if (totalMib <= 0) return undefined;

  const vramGb = Math.round((totalMib / 1024) * 10) / 10;
  const detail = `${lines.length} GPU(s) totalling ${vramGb} GiB VRAM${names.length > 0 ? ` (${names.join(', ')})` : ''}`;
  return { vramGb, detail };
}
