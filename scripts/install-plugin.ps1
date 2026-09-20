<#
.SYNOPSIS
    Install (or update) the dsh-ai-model-hub plugin into a DeepSeek Harness profile.

.DESCRIPTION
    Three steps, in this order, because each one depends on the previous:

    1. `npm install` inside `dsh-plugin/` to materialise the plugin's DSH peer
       closure. The plugin imports `@deepseek-ai/dsh-tools` (which itself needs
       eight more `@deepseek-ai/*` packages), `@deepseek-ai/dsh-llm`,
       `@deepseek-ai/cordis` and `@deepseek-ai/schemastery`. npm installs those
       because they are declared as peer dependencies; pnpm does not.
    2. `dsh plugin add` to install the plugin into the profile. Because the
       package declares `dsh.bundle.patch`, the command also appends it to
       `dsh.profile.bundles`, so no profile file is edited by hand.
    3. Verify with `scripts/doctor.mjs`, which imports the plugin the same way
       DSH's loader will.

    Why step 1 cannot be skipped: DSH installs plugins as *junction links*, so
    Node loads the plugin from its real path in this repository rather than from
    a copy inside the profile. Module resolution therefore walks up from here, and
    the node_modules that matters is `dsh-plugin/node_modules`, not the profile's.

    The script is idempotent — re-run it after editing the plugin, after a DSH
    upgrade, or after moving this repository.

.PARAMETER Profile
    The DSH profile to install into. Defaults to `web`.

.PARAMETER SkipDoctor
    Install only; skip the post-install verification.

.EXAMPLE
    .\scripts\install-plugin.ps1

.EXAMPLE
    # Try it in a throwaway profile first.
    .\scripts\install-plugin.ps1 -Profile hubtest
#>
[CmdletBinding()]
param(
    [string] $Profile = 'web',
    [switch] $SkipDoctor
)

$ErrorActionPreference = 'Stop'

$pluginDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'dsh-plugin'
if (-not (Test-Path (Join-Path $pluginDir 'package.json'))) {
    throw "Could not find the plugin package at $pluginDir"
}

$dshHome = if ($env:DSH_HOME) { $env:DSH_HOME } else { Join-Path $HOME '.dsh' }
$profileDir = Join-Path $dshHome "profiles\$Profile"

<#
Run a native command and judge it only by its exit code.

Windows PowerShell turns anything a native program writes to stderr into a
terminating error under `$ErrorActionPreference = 'Stop'`, and both `npm` and
`dsh plugin` write ordinary progress information to stderr — including the
"initialized profile" notice on first use. Without this wrapper a successful
install would abort the script.

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

Write-Host 'dsh-ai-model-hub plugin installer' -ForegroundColor Cyan
Write-Host "  plugin:  $pluginDir"
Write-Host "  profile: $profileDir"
Write-Host ''

# ── 1. Materialise the plugin's dependencies ───────────────────────────────
Write-Host "[1/3] Installing the plugin's DSH dependencies..." -ForegroundColor Cyan
Push-Location $pluginDir
try {
    # The cache is kept inside the repository so the install does not depend on a
    # writable global npm cache.
    $code = Invoke-Native 'npm' @('install', '--no-audit', '--no-fund', '--cache', (Join-Path $pluginDir '..\.npm-cache'))
    if ($code -ne 0) { throw "npm install failed with exit code $code" }
} finally {
    Pop-Location
}

# Fail early and clearly if the closure is still incomplete: a missing peer here
# becomes an opaque module error much later, at DSH boot.
$missing = @()
foreach ($peer in @(
        '@deepseek-ai/dsh-tools',
        '@deepseek-ai/dsh-llm',
        '@deepseek-ai/cordis',
        '@deepseek-ai/schemastery',
        '@deepseek-ai/dsh-agent',
        '@deepseek-ai/dsh-scope')) {
    if (-not (Test-Path (Join-Path $pluginDir "node_modules\$peer"))) { $missing += $peer }
}
if ($missing.Count -gt 0) {
    throw "The plugin's dependencies are incomplete. Missing: $($missing -join ', '). Run 'npm install' in $pluginDir and read its output."
}
Write-Host "  dependencies present."

# ── 2. Install into the profile ────────────────────────────────────────────
Write-Host ''
Write-Host "[2/3] Installing into profile '$Profile'..." -ForegroundColor Cyan
$code = Invoke-Native 'dsh' @('plugin', '--profile', $Profile, 'add', $pluginDir)
if ($code -ne 0) { throw "dsh plugin add failed with exit code $code" }

# ── 3. Verify ───────────────────────────────────────────────────────────────
if (-not $SkipDoctor) {
    Write-Host ''
    Write-Host '[3/3] Verifying...' -ForegroundColor Cyan
    $code = Invoke-Native 'node' @((Join-Path $PSScriptRoot 'doctor.mjs'), '--profile', $Profile)
    if ($code -ne 0) {
        Write-Host ''
        Write-Host 'Verification failed; the plugin will not load until this passes.' -ForegroundColor Red
        exit $code
    }
}

Write-Host ''
Write-Host 'Done.' -ForegroundColor Green
Write-Host ''
Write-Host "Restart DeepSeek Harness so the '$Profile' profile reloads, then ask the agent:"
Write-Host '    List the available AI model capabilities.' -ForegroundColor White
Write-Host ''
Write-Host 'The plugin discovers its model catalog by walking up from the SESSION'
Write-Host 'working directory, where the agent actually runs. To use the models in'
Write-Host 'this repository, start DSH from this directory, or copy config/models.json'
Write-Host 'into the workspace you normally use.'
