#!/bin/sh
#
# Install (or update) the dsh-ai-model-hub plugin straight from GitHub.
#
# The one-line install for a machine that does not have this repository yet:
#
#     curl -fsSL https://raw.githubusercontent.com/Mhmd7-7/dsh-ai-model-hub/main/install.sh | sh
#
# ...and the same thing as a file, when you want to pass options:
#
#     ./install.sh --profile web
#
# A preflight, then two steps, because the repository IS the plugin package:
#
#   Preflight: Node >= 22.6, pnpm and dsh on PATH. Every failure is reported here,
#   by name, instead of surfacing later as an opaque error.
#
#   [1/2] `dsh plugin --profile <p> add github:Mhmd7-7/dsh-ai-model-hub` -- the
#         official CLI installs the package into the profile and, because the
#         package declares `dsh.bundle.patch`, registers it as a profile layer.
#
#   [2/2] Verify, unless --skip-doctor is given: run the installed package's own
#         `scripts/doctor.mjs`, which imports the plugin exactly the way DSH's
#         loader will.
#
# Why this no longer clones. It used to clone the repository and run its
# installer, because the plugin was TypeScript-only and Node refuses to strip
# types for a file under `node_modules`
# (ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING) -- so the plugin had to live in a
# real checkout OUTSIDE node_modules and be linked in from there. The package now
# ships its compiled JavaScript (`lib/`), so a plain `dsh plugin add` from GitHub
# works, and the checkout, the clone, `npm install` and the build are all gone
# from this path.
#
# Re-run this script to update: it re-adds the package and re-resolves the ref.
# To install a local working tree instead, run `npm run install:plugin` in it.
#
# POSIX sh on purpose: this runs under `curl | sh`, where `sh` may be dash or
# busybox ash, not bash.

set -eu

REPOSITORY="${DSH_MODEL_HUB_REPO:-https://github.com/Mhmd7-7/dsh-ai-model-hub.git}"
REF="${DSH_MODEL_HUB_REF:-main}"
PROFILE="${DSH_PROFILE:-web}"
INSTALL_DIR="${DSH_MODEL_HUB_DIR:-}"
SKIP_DOCTOR=0

# The installed package name, and the name of the profile bundle layer it
# registers itself as. They are the same string: the package IS the plugin.
PLUGIN_PACKAGE='dsh-ai-model-hub'

usage() {
    cat <<'EOF'
Install (or update) the dsh-ai-model-hub plugin from GitHub.

Usage: install.sh [options]

Options:
  -p, --profile <name>     DSH profile to install into (default: web)
  -r, --ref <ref>          branch, tag or commit to install (default: main),
                           appended to the package spec as #<ref>
      --repository <spec>  where to install from (default: the canonical GitHub
                           URL). The GitHub forms -- https://github.com/owner/repo,
                           git@github.com:owner/repo.git, or a bare owner/repo --
                           all become github:owner/repo. Any other value is handed
                           to pnpm verbatim, with #<ref> appended when it has no
                           fragment of its own.
      --skip-doctor        install only; skip the post-install verification
  -d, --dir <path>         DEPRECATED and ignored: there is no checkout to place
                           any more. Kept so existing commands keep working.
  -h, --help               show this help

Environment variables supply the same values, which is how you configure the
`curl | sh` form where arguments are awkward:

  DSH_PROFILE, DSH_MODEL_HUB_REF, DSH_MODEL_HUB_REPO, DSH_MODEL_HUB_DIR
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--profile) PROFILE="${2:?--profile needs a value}"; shift 2 ;;
        -d|--dir) INSTALL_DIR="${2:?--dir needs a value}"; shift 2 ;;
        -r|--ref) REF="${2:?--ref needs a value}"; shift 2 ;;
        --repository) REPOSITORY="${2:?--repository needs a value}"; shift 2 ;;
        --skip-doctor) SKIP_DOCTOR=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "install.sh: unknown argument '$1'" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
done

# Colour only when a human is watching; `curl | sh` into a log file should stay
# plain text.
if [ -t 1 ]; then
    BOLD=$(printf '\033[1m'); DIM=$(printf '\033[2m')
    RED=$(printf '\033[31m'); YELLOW=$(printf '\033[33m')
    GREEN=$(printf '\033[32m'); CYAN=$(printf '\033[36m')
    RESET=$(printf '\033[0m')
else
    BOLD=''; DIM=''; RED=''; YELLOW=''; GREEN=''; CYAN=''; RESET=''
fi

# Report a fatal problem and stop, with the same "what to do about it" shape the
# rest of this project's diagnostics use.
fail() {
    printf '\n%sinstall.sh: %s%s\n' "$RED" "$1" "$RESET" >&2
    exit 1
}

step() { printf '%s%s%s\n' "$CYAN" "$1" "$RESET"; }
note() { printf '%s      %s%s\n' "$DIM" "$1" "$RESET"; }
warn() { printf '%s%s%s\n' "$YELLOW" "$1" "$RESET"; }

# Turn the repository option into a pnpm package spec.
#
# GitHub is the case that matters and the only one with a shorthand: its https
# URL, its ssh URL and a bare `owner/repo` all become `github:owner/repo`, which
# pnpm fetches as a tarball -- no git binary, no clone. `$2` is appended as a
# `#<ref>` fragment. Anything else (a fork on another host, a tarball URL) is
# passed through untouched, because guessing a shorthand for it would break it.
#
# Echoes the spec. Never fails: an unrecognised value is still a spec pnpm can
# be asked about, and pnpm's own error is the one worth reporting.
plugin_spec() {
    repo=$1
    ref=$2
    slug=''

    case "$repo" in
        https://github.com/*)     slug=${repo#https://github.com/} ;;
        http://github.com/*)      slug=${repo#http://github.com/} ;;
        git+https://github.com/*) slug=${repo#git+https://github.com/} ;;
        ssh://git@github.com/*)   slug=${repo#ssh://git@github.com/} ;;
        git@github.com:*)         slug=${repo#git@github.com:} ;;
    esac

    if [ -n "$slug" ]; then
        slug=${slug%.git}
        # Trim trailing slashes without a subshell loop.
        while [ "${slug%/}" != "$slug" ]; do slug=${slug%/}; done
        case "$slug" in
            */*) spec="github:$slug" ;;
            *) spec="$repo" ;;
        esac
    else
        case "$repo" in
            # A bare `owner/repo` is what `github:` means; anything with a scheme
            # or an scp-style host is somebody else's remote and goes verbatim.
            *:*|*//*) spec="$repo" ;;
            */*) spec="github:$repo" ;;
            *) spec="$repo" ;;
        esac
    fi

    if [ -n "$ref" ]; then
        case "$spec" in
            *"#"*) : ;;
            *) spec="$spec#$ref" ;;
        esac
    fi

    printf '%s' "$spec"
}

# ── 1. Preflight ────────────────────────────────────────────────────────────
printf '%sdsh-ai-model-hub -- install from GitHub%s\n\n' "$BOLD" "$RESET"

if [ -n "$INSTALL_DIR" ]; then
    warn 'note: --dir / $DSH_MODEL_HUB_DIR is deprecated and ignored.'
    warn '      The plugin is installed into the profile straight from GitHub, so there'
    warn '      is no checkout to place. Drop the option; this run carries on without it.'
    printf '\n'
fi

for tool in node pnpm dsh; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        case "$tool" in
            node) fail "'node' was not found on PATH. Install Node 22.6 or newer from https://nodejs.org" ;;
            pnpm) fail "'pnpm' was not found on PATH. dsh plugin forwards to pnpm. Install it with: npm install -g pnpm" ;;
            dsh)  fail "'dsh' was not found on PATH. Install DeepSeek Harness first: npm install -g @deepseek-ai/dsh" ;;
        esac
    fi
done

# git is no longer part of the install -- `github:` specs are fetched as tarballs
# -- so its absence is a note, not a failure. pnpm does fall back to git for some
# refs, which is why this is worth saying out loud rather than not at all.
if ! command -v git >/dev/null 2>&1; then
    warn "note: 'git' was not found on PATH. GitHub packages are fetched as tarballs,"
    warn '      so this is normally fine -- but pnpm uses git to resolve an unusual ref.'
    warn '      If the add step below fails with a git error, install https://git-scm.com'
    printf '\n'
fi

NODE_VERSION=$(node --version 2>/dev/null || true)
NODE_MAJOR=$(printf '%s' "$NODE_VERSION" | sed -n 's/^v\([0-9]*\)\..*$/\1/p')
NODE_MINOR=$(printf '%s' "$NODE_VERSION" | sed -n 's/^v[0-9]*\.\([0-9]*\).*$/\1/p')
NODE_MAJOR=${NODE_MAJOR:-0}
NODE_MINOR=${NODE_MINOR:-0}
if [ "$NODE_MAJOR" -lt 22 ] || { [ "$NODE_MAJOR" -eq 22 ] && [ "$NODE_MINOR" -lt 6 ]; }; then
    fail "Node 22.6 or newer is required (found $NODE_VERSION). $PLUGIN_PACKAGE declares engines.node >= 22.6, and DeepSeek Harness needs a modern Node as well."
fi

DSH_HOME_DIR="${DSH_HOME:-$HOME/.dsh}"
PROFILE_DIR="$DSH_HOME_DIR/profiles/$PROFILE"
PACKAGE_DIR="$PROFILE_DIR/node_modules/$PLUGIN_PACKAGE"

SPEC=$(plugin_spec "$REPOSITORY" "$REF")

# ── 2. Install into the profile ─────────────────────────────────────────────
# One command, because the CLI does the rest: pnpm installs the package into the
# profile directory, then `dsh plugin` reconciles `dsh.profile.bundles` against the
# installed state and appends this package, whose `dsh.bundle.patch` makes it a
# profile layer. No profile file is edited by hand.
printf '\n'
step "[1/2] Installing into profile '$PROFILE'..."
note "$SPEC"
dsh plugin --profile "$PROFILE" add "$SPEC" \
    || fail "dsh plugin add failed. See pnpm's output above; re-run this script once the cause is fixed."

# ── 3. Verify ───────────────────────────────────────────────────────────────
if [ "$SKIP_DOCTOR" -eq 0 ]; then
    printf '\n'
    step '[2/2] Verifying...'

    [ -f "$PACKAGE_DIR/package.json" ] \
        || fail "the package is not in the profile at $PACKAGE_DIR. Check pnpm's output above; 'dsh plugin --profile $PROFILE list' shows what the profile actually has."
    note "installed: $PACKAGE_DIR"

    # The package's own doctor, when it ships one, is the deep check: it imports
    # the plugin exactly the way DSH's loader will.
    if [ -f "$PACKAGE_DIR/scripts/doctor.mjs" ]; then
        node "$PACKAGE_DIR/scripts/doctor.mjs" --profile "$PROFILE" \
            || fail 'verification failed: the plugin will not load until this passes. Re-run with --skip-doctor to install without verifying.'
    else
        note 'the installed package ships no scripts/doctor.mjs; skipping the deeper check.'
    fi
fi

printf '\n%sDone.%s\n\n' "$GREEN" "$RESET"
printf "Restart DeepSeek Harness so the '%s' profile reloads, then ask the agent:\n" "$PROFILE"
printf '    List the available AI model capabilities.\n\n'
printf '  %ssource:%s  %s\n' "$BOLD" "$RESET" "$SPEC"
printf '  %sprofile:%s %s  (bundle layer: %s)\n' "$BOLD" "$RESET" "$PROFILE_DIR" "$PLUGIN_PACKAGE"
printf '  %supdate:%s  re-run this script; it re-adds the package and re-resolves the ref\n' "$BOLD" "$RESET"
printf '  %scatalog:%s the installed package ships config/models.json, which the plugin\n' "$BOLD" "$RESET"
printf '           finds without configuration -- it searches up from the agent'"'"'s\n'
printf '           working directory and then from its own installation directory.\n'
printf '           Edit that file, or set searchRoots/configPath in the plugin row.\n'
printf '  %sremove:%s  dsh plugin --profile %s remove %s\n' "$BOLD" "$RESET" "$PROFILE" "$PLUGIN_PACKAGE"
printf '\n'
printf '  %scheckout:%s to install a local working tree instead, run\n' "$BOLD" "$RESET"
printf "             'npm run install:plugin' in that checkout -- it also runs the\n"
printf '             peer install and the build.\n'
