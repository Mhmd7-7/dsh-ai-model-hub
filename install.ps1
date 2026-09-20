<#
.SYNOPSIS
    Install (or update) the dsh-ai-model-hub plugin straight from GitHub.

.DESCRIPTION
    The one-line install for a machine that does not have this repository yet:

        irm https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.ps1 | iex

    ...and the same thing as a file, when you want to pass options:

        .\install.ps1 -Profile web

    A preflight, then two steps, because the repository IS the plugin package:

      Preflight: Node >= 22.6, pnpm and dsh on PATH. Every failure is reported
      here, by name, instead of surfacing later as an opaque error.

      [1/2] `dsh plugin --profile <p> add github:Mhmd7-7/dsh-ai-model-hub` -- the
            official CLI installs the package into the profile and, because the
            package declares `dsh.bundle.patch`, registers it as a profile layer.

      [2/2] Verify, unless -SkipDoctor is given: run the installed package's own
            `scripts/doctor.mjs`, which imports the plugin exactly the way DSH's
            loader will.

    Why this no longer clones. It used to clone the repository and run its
    installer, because the plugin was TypeScript-only and Node refuses to strip
    types for a file under `node_modules`
    (ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING) -- so the plugin had to live in a
    real checkout OUTSIDE node_modules and be linked in from there. The package
    now ships its compiled JavaScript (`lib/`), so a plain `dsh plugin add` from
    GitHub works, and the checkout, the clone, `npm install` and the build are all
    gone from this path.

    Re-run this script to update: it re-adds the package and re-resolves the ref.
    To install a local working tree instead, run `npm run install:plugin` in it.

.PARAMETER Profile
    The DSH profile to install into. Defaults to `web`, the profile behind
    `dsh web`. Override with $env:DSH_PROFILE when piping into `iex`.

.PARAMETER Ref
    The branch, tag or commit to install, appended to the package spec as
    `#<ref>`. Defaults to `main`. Override with $env:DSH_MODEL_HUB_REF -- useful
    for pinning a release tag.

.PARAMETER Repository
    Where to install from. Defaults to the canonical GitHub URL. The GitHub
    forms -- `https://github.com/owner/repo(.git)`, `git@github.com:owner/repo.git`
    and a bare `owner/repo` -- all become `github:owner/repo#<ref>`. Override with
    $env:DSH_MODEL_HUB_REPO to install from a fork. Any other value is handed to
    pnpm verbatim, with `#<ref>` appended when it has no fragment of its own.

.PARAMETER InstallDir
    DEPRECATED and ignored. There is no checkout to place any more: the plugin is
    installed into the profile directly from GitHub. Kept so existing command
    lines that pass it keep working.

.PARAMETER SkipDoctor
    Install only; skip the post-install verification.

.EXAMPLE
    irm https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.ps1 | iex

.EXAMPLE
    # A different profile, from a pinned tag.
    .\install.ps1 -Profile hubtest -Ref v0.2.0

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

# The installed package name, and the name of the profile bundle layer it
# registers itself as. They are the same string: the package IS the plugin.
$PluginPackage = 'dsh-ai-model-hub'

<#
Run a native command and judge it only by its exit code.

Windows PowerShell turns anything a native program writes to stderr into a
terminating error under `$ErrorActionPreference = 'Stop'`, and both `pnpm` and
`dsh plugin` write ordinary progress information to stderr. Without this wrapper a
successful install would abort the script.

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

<#
Turn the repository option into a pnpm package spec.

GitHub is the case that matters and the only one with a shorthand: its https URL,
its ssh URL and a bare `owner/repo` all become `github:owner/repo`, which pnpm
fetches as a tarball -- no git binary, no clone. `-Ref` is appended as a `#<ref>`
fragment. Anything else (a fork on another host, a tarball URL) is passed through
untouched, because guessing a shorthand for it would break it.

@param $Repository - the remote or spec the user asked for.
@param $Ref - the branch, tag or commit, or an empty string to leave it unpinned.
@returns the spec to hand to `dsh plugin add`.
#>
function Get-PluginSpec {
    param([string] $Repository, [string] $Ref)

    $repo = $Repository.Trim()
    $slug = $null

    foreach ($prefix in @(
            'https://github.com/',
            'http://github.com/',
            'git+https://github.com/',
            'ssh://git@github.com/')) {
        if ($repo.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            $slug = $repo.Substring($prefix.Length)
            break
        }
    }
    if (-not $slug -and $repo.StartsWith('git@github.com:', [System.StringComparison]::OrdinalIgnoreCase)) {
        $slug = $repo.Substring('git@github.com:'.Length)
    }

    if ($slug) {
        $slug = ($slug -replace '\.git$', '').TrimEnd('/')
        if ($slug -notmatch '^[^/]+/[^/]+$') { $slug = $null }
    } elseif ($repo -notmatch ':' -and $repo -match '^[^/]+/[^/]+$') {
        # A bare `owner/repo`, which is what `github:` means.
        $slug = $repo
    }

    $spec = if ($slug) { "github:$slug" } else { $repo }
    if ($Ref -and $spec -notmatch '#') { $spec = "$spec#$Ref" }
    return $spec
}

# ── 1. Preflight ────────────────────────────────────────────────────────────
Write-Host 'dsh-ai-model-hub -- install from GitHub' -ForegroundColor Cyan
Write-Host ''

if ($InstallDir) {
    Write-Host 'note: -InstallDir / $env:DSH_MODEL_HUB_DIR is deprecated and ignored.' -ForegroundColor Yellow
    Write-Host '      The plugin is installed into the profile straight from GitHub, so there'
    Write-Host '      is no checkout to place. Drop the option; this run carries on without it.'
    Write-Host ''
}

foreach ($tool in @('node', 'pnpm', 'dsh')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        $hint = switch ($tool) {
            'node' { 'Install Node 22.6 or newer from https://nodejs.org' }
            'pnpm' { "dsh plugin forwards to pnpm. Install it with: npm install -g pnpm" }
            'dsh' { 'Install DeepSeek Harness first: npm install -g @deepseek-ai/dsh' }
        }
        Stop-Install "'$tool' was not found on PATH. $hint"
    }
}

# git is no longer part of the install -- `github:` specs are fetched as tarballs --
# so its absence is a note, not a failure. pnpm does fall back to git for some
# refs, which is why this is worth saying out loud rather than not at all.
if (-not (Get-Command 'git' -ErrorAction SilentlyContinue)) {
    Write-Host "note: 'git' was not found on PATH. GitHub packages are fetched as tarballs," -ForegroundColor Yellow
    Write-Host '      so this is normally fine -- but pnpm uses git to resolve an unusual ref.' -ForegroundColor Yellow
    Write-Host '      If the add step below fails with a git error, install https://git-scm.com' -ForegroundColor Yellow
    Write-Host ''
}

$nodeText = Get-NativeText 'node' @('--version')
$major = 0
$minor = 0
if ($nodeText -match 'v(\d+)\.(\d+)') {
    $major = [int] $Matches[1]
    $minor = [int] $Matches[2]
}
if ($major -lt 22 -or ($major -eq 22 -and $minor -lt 6)) {
    Stop-Install "Node 22.6 or newer is required (found $($nodeText.Trim())). $PluginPackage declares engines.node >= 22.6, and DeepSeek Harness needs a modern Node as well."
}

$dshHome = if ($env:DSH_HOME) { $env:DSH_HOME } else { Join-Path $HOME '.dsh' }
$profileDir = Join-Path $dshHome "profiles\$Profile"
$packageDir = Join-Path (Join-Path $profileDir 'node_modules') $PluginPackage

$spec = Get-PluginSpec -Repository $Repository -Ref $Ref

# ── 2. Install into the profile ─────────────────────────────────────────────
# One command, because the CLI does the rest: pnpm installs the package into the
# profile directory, then `dsh plugin` reconciles `dsh.profile.bundles` against the
# installed state and appends this package, whose `dsh.bundle.patch` makes it a
# profile layer. No profile file is edited by hand.
Write-Host "[1/2] Installing into profile '$Profile'..." -ForegroundColor Cyan
Write-Host "      $spec"
$code = Invoke-Native 'dsh' @('plugin', '--profile', $Profile, 'add', $spec)
if ($code -ne 0) {
    Stop-Install "dsh plugin add failed with exit code $code. See pnpm's output above; re-run this script once the cause is fixed."
}

# ── 3. Verify ───────────────────────────────────────────────────────────────
if (-not $SkipDoctor) {
    Write-Host ''
    Write-Host '[2/2] Verifying...' -ForegroundColor Cyan

    if (-not (Test-Path (Join-Path $packageDir 'package.json'))) {
        Stop-Install "the package is not in the profile at $packageDir. Check pnpm's output above; 'dsh plugin --profile $Profile list' shows what the profile actually has."
    }
    Write-Host "      installed: $packageDir"

    # The package's own doctor, when it ships one, is the deep check: it imports
    # the plugin exactly the way DSH's loader will.
    $installedDoctor = Join-Path $packageDir 'scripts\doctor.mjs'
    if (Test-Path $installedDoctor) {
        $code = Invoke-Native 'node' @($installedDoctor, '--profile', $Profile)
        if ($code -ne 0) {
            Stop-Install 'verification failed: the plugin will not load until this passes. Re-run with -SkipDoctor to install without verifying.'
        }
    } else {
        Write-Host '      the installed package ships no scripts/doctor.mjs; skipping the deeper check.'
    }
}

Write-Host ''
Write-Host 'Done.' -ForegroundColor Green
Write-Host ''
Write-Host "Restart DeepSeek Harness so the '$Profile' profile reloads, then ask the agent:"
Write-Host '    List the available AI model capabilities.' -ForegroundColor White
Write-Host ''
Write-Host "  source:  $spec"
Write-Host "  profile: $profileDir  (bundle layer: $PluginPackage)"
Write-Host "  update:  re-run this script; it re-adds the package and re-resolves the ref"
Write-Host "  catalog: the installed package ships config/models.json, which the plugin"
Write-Host "           finds without configuration -- it searches up from the agent's"
Write-Host "           working directory and then from its own installation directory."
Write-Host "           Edit that file, or set searchRoots/configPath in the plugin row."
Write-Host "  remove:  dsh plugin --profile $Profile remove $PluginPackage"
Write-Host ''
Write-Host "  checkout: to install a local working tree instead, run 'npm run install:plugin'"
Write-Host '            in that checkout -- it also runs the peer install and the build.'
