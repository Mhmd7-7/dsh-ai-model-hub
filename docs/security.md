# Security

The threat model that shapes every decision here: **the agent is untrusted
input.** A user's prompt, a file the agent read, a web page it fetched, or a
model's own output can all contain text that tries to make the agent do something
it should not. The design assumption is that this will eventually succeed at the
level of *intent*, and that the system's job is to make the resulting *capability*
harmless.

The concrete goal: **the agent can influence which model runs, but it must never
be able to influence what command line runs.**

---

## The command-execution posture

`src/util/process.ts` is the only file in the hub that spawns a process. That is
deliberate — the entire attack surface is reviewable in one place.

### 1. No shell, ever

Children are spawned with `shell: false` and an argv array:

```ts
spawn(spec.command, [...spec.args], { shell: false, stdio: ['ignore', 'pipe', 'pipe'] })
```

There is no command string to inject into. `; rm -rf /`, `` `whoami` ``, `$(...)`,
and `|` are inert data. This one decision removes the entire class of shell
injection rather than trying to escape it correctly.

### 2. An explicit allowlist

Launching is permitted only for well-known local inference engines:

```
python python3 py ollama llama-server llamafile vllm koboldcpp
text-generation-launcher stable-diffusion.cpp sd comfy comfyui
blender piper whisper main node
```

Anything else is refused with an actionable message. This matters because a
catalog file is a document a user might copy from a forum post, and the
allowlist means a careless paste cannot invent a new executable.

**Notably absent, on purpose:** `sh`, `bash`, `zsh`, `cmd`, `powershell`, `rm`,
`curl`, `wget`, `nc`. Those are the pivot points that turn "run my model" into
"run anything".

A deployment that genuinely needs another engine sets `allowAnyCommand: true`,
which is an explicit, auditable decision rather than a silent widening.

### The one exception, and why it is not a hole

The resource probe runs `nvidia-smi`, which is not on that allowlist, and it widens
whatever policy it is given by exactly that one binary (see `GPU_PROBE_COMMAND` in
`src/machine.ts`).

The allowlist exists to bound what the hub launches **on a model's behalf**, where
the catalog — a document that can be copied from anywhere — supplies the command.
The probe is not that: the executable is a constant in the source, the arguments
are a constant in the source, the output is parsed as numbers, and the only failure
mode of a missing or impersonated `nvidia-smi` is a reported GPU figure that is
wrong. Blocking it would mean a machine whose allowlist has been narrowed reports
no GPU, and a machine that reports no GPU disqualifies every GPU model — a silent,
systematic wrong answer in exchange for no security property.

### 3. The agent cannot name a command

This is the part the allowlist cannot provide on its own. Look at
`invoke_model`'s parameters: `capability`, `prompt`, `inputs`, `options`,
`modelId`, `requiredTags`, `timeoutMs`.

There is no field for a command, a path, an argument list, an endpoint, or a
module name. Commands come exclusively from a descriptor file, which is written
by the operator. The only model-identifying input, `modelId`, must match an id the
catalog already knows — a pin to an existing entry, not a way to introduce one.

The agent cannot express a launch it was not already configured to perform.

### 4. Per-argument validation

Even arguments that originate outside the agent are checked, because a
configuration file can be edited by anything that can write to the workspace:

- **NUL bytes** are refused. A NUL truncates a C string, so a value that passed a
  check can be a different value by the time the OS sees it.
- **Line breaks** are refused. They are harmless to `spawn` itself, which makes
  them a reliable signal that someone built a shell line and got this far by
  mistake — worth failing loudly rather than sanitizing.
- **Length and count** are bounded (4096 chars, 256 args), so a runaway config
  cannot produce an unbounded argv.

### 5. A scrubbed environment

Every child inherits the operator's environment *minus* credential-shaped
variables:

```
API_KEY  TOKEN  SECRET  PASSWORD  CREDENTIAL
DEEPSEEK  OPENAI  ANTHROPIC  AWS_  AZURE_  GOOGLE_
```

A model process has no business holding the harness's API keys. Without this,
any engine — including a downloaded model script — could read `DEEPSEEK_API_KEY`
straight out of its own environment and exfiltrate it.

### 6. Bounded, killable processes

- Captured output is capped, so a chatty engine cannot exhaust memory.
- Every one-shot command has a hard timeout.
- Stopping a model signals the **process group** on POSIX and uses
  `taskkill /PID <pid> /T /F` on Windows, because inference servers routinely
  fork helpers and killing only the parent orphans them holding the port.

---

## Two switches, both off by default

| Setting | Default | Meaning |
|---|---|---|
| `allowProcessLaunch` | `false` | The hub will **not** start any engine, regardless of what a descriptor says |
| `allowAnyCommand` | `false` | The command allowlist is enforced |

Both false means the safest useful configuration: the hub talks to engines you
started yourself and never launches anything. This is the recommended default and
is what ships.

Turning `allowProcessLaunch` on is a **two-key** operation by construction: the
descriptor must declare `startable: true` with a `start` command, *and* the
deployment must permit launching. A catalog edit alone can never cause a launch.

### What each switch does and does not buy you

- `allowProcessLaunch: false` means the agent cannot cause a process to start. It
  does **not** stop a *configured and already-running* engine from being invoked.
- The allowlist bounds *which* executables can run. It does not bound what an
  allowlisted engine does — `python` can run any Python file you configured it to
  run, which is why `cwd` and `args` belong to the operator's descriptor.

---

## Artifact store containment

`LocalArtifactStore.resolvePath()` re-checks that a resolved path is inside the
store root *even though the index said so*, because the index is a JSON file on
disk and therefore editable by anything that can write to the workspace. A
tampered or corrupted index that pointed at `~/.ssh/id_rsa` would otherwise hand
an engine an arbitrary file path. `tests/artifacts.test.ts` edits the index
exactly that way and asserts the read is refused.

The store root is workspace-local by default (`<cwd>/artifacts`), so one
conversation's generated content lands beside its code rather than in a shared
global directory another session could collide with.

---

## Failure containment

Security work that turns a failure into an outage is a different bug. Everything
degrades:

| Failure | Behaviour |
|---|---|
| Catalog missing or invalid | Logs the problem, registers **no** tools, lets DSH boot — the agent stays usable for everything that does not need a local model |
| A tool's arguments are invalid | The real DSH schema rejects them before the body runs |
| The model is unavailable | A typed code plus a recovery hint; no retry storm |
| An invocation is abandoned by a timeout | The late rejection is *contained* — without that handler it would become an unhandled rejection and **kill the process** |
| An event listener throws | Contained; a broken log sink cannot fail a user's image generation |

That fourth row was a real bug found by the test suite and is now pinned by a
regression test: `withTimeout` attaches a no-op handler to the losing branch of
its race.

---

## Review checklist

Before enabling process launching in a shared or multi-user environment:

- [ ] `allowProcessLaunch` is `true` only if you actually need the hub to start engines
- [ ] `allowAnyCommand` is `false` unless a specific engine genuinely requires it
- [ ] Every `lifecycle.start` in your catalog was written by you, not copied from an untrusted source
- [ ] Every `cwd` points inside a directory you trust
- [ ] Every `lifecycle.start.command` is on the allowlist, or you have read what it runs
- [ ] The artifact root is inside the workspace
- [ ] The catalog file is not writable by anyone who should not be able to change what runs

## Deliberate non-goals

- **Sandboxing the engine itself.** An allowlisted engine runs with your
  privileges; containing *it* is the operator's job (a container, a VM, a
  separate user). The hub's job is to make sure the agent cannot *choose* what
  runs.
- **Network egress control.** Adapters fetch from endpoints you configured.
  Restricting an engine's outbound network is a container/firewall concern.
- **Protecting a catalog you gave an attacker write access to.** A configuration
  file that says "run this program" is a program that runs. File permissions on
  the configuration are the boundary.
