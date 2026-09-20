<#
.SYNOPSIS
    Install (or update) the dsh-ai-model-hub plugin straight from GitHub.

.DESCRIPTION
    The one-line install for a machine that does not have this repository yet:

        irm https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.ps1 | iex

    ...and the same thing as a file, when you want to pass options:

        .\install.ps1 -Profile web

    Four steps, in this order:

      1. Preflight: Node >= 22.6, plus git, npm and dsh on PATH. Every failure is
         reported here, by name, instead of surfacing later as an opaque module
         error at DSH boot.
      2. Clone — or fast-forward an existing clone of — this repository into
         `$DSH_HOME/plugins/dsh-ai-model-hub` (`~/.dsh/plugins/...` by default),
         which is the directory convention DSH's own plugin store already uses.
      3. Run `scripts/install-plugin.mjs` from that checkout: it installs the
         plugin's dependency closure, adds the plugin to the profile, and
         registers it as a bundle layer.
      4. Verify with `scripts/doctor.mjs`, which imports the plugin exactly the
         way DSH's loader will.

    Why this clones instead of running
    `dsh plugin --profile web add github:Mhmd7-7/dsh-ai-model-hub`: the plugin is
    written in TypeScript and DSH loads it directly through Node's type stripping,
    but Node REFUSES to strip types for any file under `node_modules`
    (ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING). A package installed by pnpm —
    from the registry or from git — always lands inside the profile's
    `node_modules`, so the loader could not import it. The plugin therefore has to
    live in a real checkout OUTSIDE `node_modules` and be linked in from there,
    which is the one thing `dsh plugin add <local path>` does correctly.

    Re-run this script to update: it fast-forwards the clone and reinstalls.

.PARAMETER Profile
    The DSH profile to install into. Defaults to `web`, the profile behind
    `dsh web`. Override with $env:DSH_PROFILE when piping into `iex`.

.PARAMETER InstallDir
    Where to clone the repository. Defaults to
    `$DSH_HOME/plugins/dsh-ai-model-hub`. Override with $env:DSH_MODEL_HUB_DIR.

.PARAMETER Ref
    The branch, tag or commit to install. Defaults to `main`. Override with
    $env:DSH_MODEL_HUB_REF — useful for pinning a release tag.

.PARAMETER Repository
    The git remote to clone. Defaults to the canonical GitHub URL. Override with
    $env:DSH_MODEL_HUB_REPO to install from a fork.

.PARAMETER SkipDoctor
    Install only; skip the post-install verification.

.EXAMPLE
    irm https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.ps1 | iex

.EXAMPLE
    # A different profile, from a pinned tag.
    .\install.ps1 -Profile hubtest -Ref v0.1.0

.EXAMPLE
    # Piped form: parameters are unavailable, so use the environment.
    $env:DSH_PROFILE = 'hubtest'; irm https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.ps1 | iex
#>
[CmdletBinding()]
param(
    [string] $Profile = $(if ($env:DSH_PROFILE) { $env:DSH_PROFILE } else { 'web' }),
    [string] $InstallDir = $env:DSH_MODEL_HUB_DIR,
    [string] $Ref = $(if ($env:DSH_MODEL_HUB_REF) { $env:DSH_MODEL_HUB_REF } else { 'main' }),
    [string] $Repository = $(if ($env:DSH_MODEL_HUB_REPO) { $env:DSH_MODEL_HUB_REPO } else { 'https://github.com/Mhmd7-7/dsh-ai-model-hub.git' }),
    [switch] $SkipDoctor
)

$ErrorActionPreference = 'Stop'

<#
Run a native command and judge it only by its exit code.

Windows PowerShell turns anything a native program writes to stderr into a
terminating error under `$ErrorActionPreference = 'Stop'`, and `git`, `npm` and
`dsh plugin` all write ordinary progress information to stderr. Without this
wrapper a successful install would abort the script.

`2>&1` merges the streams so nothing is treated as an error and all output stays
visible, and the output is discarded so only the exit code is returned.
#>
function Invoke-Native {
    param([string] $Program, [string[]] $Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Program @Arguments 2>&1 | Out-Host
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

<#
Capture a native command's output as text, for the version probes.

Returns an empty string when the command is missing or writes nothing.
#>
function Get-NativeText {
    param([string] $Program, [string[]] $Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = @(& $Program @Arguments 2>&1)
        return ($lines | Where-Object { $_ -is [string] }) -join "`n"
    } catch {
        return ''
    } finally {
        $ErrorActionPreference = $previous
    }
}

<#
Report a fatal problem and stop, with the same "what to do about it" shape the
rest of this project's diagnostics use.
#>
function Stop-Install {
    param([string] $Message)
    Write-Host ''
    Write-Host "install.ps1: $Message" -ForegroundColor Red
    exit 1
}

# ── 1. Preflight ────────────────────────────────────────────────────────────
Write-Host 'dsh-ai-model-hub — install from GitHub' -ForegroundColor Cyan
Write-Host ''

foreach ($tool in @('node', 'npm', 'git', 'dsh')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        $hint = switch ($tool) {
            'node' { 'Install Node 22.6 or newer from https://nodejs.org' }
            'npm' { 'npm ships with Node; reinstall Node if it is missing' }
            'git' { 'Install git from https://git-scm.com — this installer clones the repository' }
            'dsh' { 'Install DeepSeek Harness first: npm install -g @deepseek-ai/dsh' }
        }
        Stop-Install "'$tool' was not found on PATH. $hint"
    }
}

$nodeText = Get-NativeText 'node' @('--version')
$major = 0
$minor = 0
if ($nodeText -match 'v(\d+)\.(\d+)') {
    $major = [int] $Matches[1]
    $minor = [int] $Matches[2]
}
if ($major -lt 22 -or ($major -eq 22 -and $minor -lt 6)) {
    Stop-Install "Node 22.6 or newer is required (found $($nodeText.Trim())). The plugin runs TypeScript directly through Node's type stripping, which older releases do not have."
}

$dshHome = if ($env:DSH_HOME) { $env:DSH_HOME } else { Join-Path $HOME '.dsh' }
if (-not $InstallDir) {
    $InstallDir = Join-Path (Join-Path $dshHome 'plugins') 'dsh-ai-model-hub'
}

# ── 2. A checkout to install from ───────────────────────────────────────────
# Two sources, in priority order: the directory this script itself lives in (so
# running it from a clone installs THAT clone, and never rewrites your working
# tree), then the canonical install directory, cloned on demand.
#
# `$PSScriptRoot` is empty when the script is piped into `iex`, which is exactly
# the case this branch is here to tell apart.
$checkout = $null
$mode = ''

if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot 'dsh-plugin'))) {
    $checkout = (Resolve-Path $PSScriptRoot).Path
    $mode = 'checkout'
    Write-Host "[1/3] Using the checkout this script lives in:" -ForegroundColor Cyan
    Write-Host "      $checkout"
}
elseif ((Test-Path (Join-Path $InstallDir 'dsh-plugin')) -and (Test-Path (Join-Path $InstallDir '.git'))) {
    $checkout = $InstallDir
    $mode = 'update'
    Write-Host "[1/3] Updating the existing clone..." -ForegroundColor Cyan
    Write-Host "      $checkout"
    Write-Host "      ref $Ref of $Repository"

    # Fetch first, then move to the requested ref. The pull is best-effort: a tag
    # or a commit hash checks out but has nothing to fast-forward, and that is not
    # a failure.
    $code = Invoke-Native 'git' @('-C', $checkout, 'fetch', '--prune', 'origin')
    if ($code -ne 0) { Stop-Install "git fetch failed with exit code $code. Check your network and that $Repository is reachable." }

    $code = Invoke-Native 'git' @('-C', $checkout, 'checkout', $Ref)
    if ($code -ne 0) {
        # The ref exists remotely but not locally yet (a new branch, or a commit
        # that is not on any fetched branch tip).
        Invoke-Native 'git' @('-C', $checkout, 'fetch', 'origin', $Ref) | Out-Null
        $code = Invoke-Native 'git' @('-C', $checkout, 'checkout', 'FETCH_HEAD')
        if ($code -ne 0) { Stop-Install "could not check out '$Ref' in $checkout." }
    } else {
        # A fast-forward can legitimately fail, and the common cause is this
        # project's own installer: `npm install` inside the checkout rewrites
        # `dsh-plugin/package-lock.json`, and git refuses to pull over a locally
        # modified file. Say so, rather than reporting an update that did not
        # happen — silently installing the old revision is the worst outcome.
        $code = Invoke-Native 'git' @('-C', $checkout, 'pull', '--ff-only', 'origin', $Ref)
        if ($code -ne 0) {
            Write-Host '      could not fast-forward: the checkout has local changes git will not pull over.'
            Write-Host '      An earlier install rewrites dsh-plugin/package-lock.json, so this is expected.'
            Write-Host '      Installing the revision already present. To discard that generated change and update:'
            Write-Host "        git -C `"$checkout`" checkout -- dsh-plugin/package-lock.json"
        }
    }
}
elseif (Test-Path $InstallDir) {
    Stop-Install "$InstallDir already exists but is not a clone of this repository. Move it aside, or choose another location with -InstallDir."
}
else {
    $mode = 'clone'
    Write-Host "[1/3] Cloning the repository..." -ForegroundColor Cyan
    Write-Host "      $Repository"
    Write-Host "      ref $Ref"
    Write-Host "      into $InstallDir"

    # A shallow clone of a branch or tag is the fast path. It fails for a commit
    # hash — git cannot ask for a single commit by name — so that case falls back
    # to a full clone followed by an explicit checkout.
    #
    # The fallback deliberately does not claim the ref is the problem: it runs for
    # any shallow-clone failure, including an unreachable remote. The full clone
    # below is what produces the real error, and it is the one reported.
    $code = Invoke-Native 'git' @('clone', '--depth', '1', '--branch', $Ref, $Repository, $InstallDir)
    if ($code -ne 0) {
        Write-Host "      shallow clone failed; retrying in full (this also covers a commit hash, which --depth cannot name)..."
        if (Test-Path $InstallDir) { Remove-Item -Recurse -Force $InstallDir }
        $code = Invoke-Native 'git' @('clone', $Repository, $InstallDir)
        if ($code -ne 0) { Stop-Install "git clone failed with exit code $code." }
        $code = Invoke-Native 'git' @('-C', $InstallDir, 'checkout', $Ref)
        if ($code -ne 0) { Stop-Install "'$Ref' is not a branch, tag or commit in $Repository." }
    }
    $checkout = $InstallDir
}

if (-not (Test-Path (Join-Path $checkout 'scripts\install-plugin.mjs'))) {
    Stop-Install "$checkout does not look like dsh-ai-model-hub (no scripts/install-plugin.mjs)."
}

# ── 3. Install into the profile ─────────────────────────────────────────────
# Everything below is the repository's own installer. This script exists to get a
# checkout onto the machine and hand over; it deliberately does not re-implement
# installation, so there is one place where that logic can be right or wrong.
Write-Host ''
Write-Host "[2/3] Installing into profile '$Profile'..." -ForegroundColor Cyan

$installerArgs = @('--no-deprecation', (Join-Path $checkout 'scripts\install-plugin.mjs'), '--profile', $Profile)
if ($SkipDoctor) { $installerArgs += '--skip-doctor' }

Push-Location $checkout
try {
    $code = Invoke-Native 'node' $installerArgs
} finally {
    Pop-Location
}
if ($code -ne 0) {
    Stop-Install "the installer failed with exit code $code. Nothing was half-installed: re-run this script once the cause is fixed."
}

Write-Host ''
Write-Host 'Done.' -ForegroundColor Green
Write-Host ''
Write-Host "Restart DeepSeek Harness so the '$Profile' profile reloads, then ask the agent:"
Write-Host '    List the available AI model capabilities.' -ForegroundColor White
Write-Host ''
Write-Host "  source:  $checkout ($mode)"
Write-Host "  update:  re-run this script; it fast-forwards the clone and reinstalls"
Write-Host "  catalog: the clone ships config/models.json, which the plugin finds"
Write-Host "           without configuration. Edit that file to add real models."
Write-Host "  remove:  dsh plugin --profile $Profile remove dsh-ai-model-hub-plugin"
