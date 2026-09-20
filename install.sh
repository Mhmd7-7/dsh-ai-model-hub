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
# Four steps, in this order:
#
#   1. Preflight: Node >= 22.6, plus git, npm and dsh on PATH. Every failure is
#      reported here, by name, instead of surfacing later as an opaque module
#      error at DSH boot.
#   2. Clone — or fast-forward an existing clone of — this repository into
#      `$DSH_HOME/plugins/dsh-ai-model-hub` (`~/.dsh/plugins/...` by default),
#      which is the directory convention DSH's own plugin store already uses.
#   3. Run `scripts/install-plugin.mjs` from that checkout: it installs the
#      plugin's dependency closure, adds the plugin to the profile, and registers
#      it as a bundle layer.
#   4. Verify with `scripts/doctor.mjs`, which imports the plugin exactly the way
#      DSH's loader will.
#
# Why this clones instead of running
# `dsh plugin --profile web add github:Mhmd7-7/dsh-ai-model-hub`: the plugin is
# written in TypeScript and DSH loads it directly through Node's type stripping,
# but Node REFUSES to strip types for any file under `node_modules`
# (ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING). A package installed by pnpm —
# from the registry or from git — always lands inside the profile's
# `node_modules`, so the loader could not import it. The plugin therefore has to
# live in a real checkout OUTSIDE `node_modules` and be linked in from there,
# which is the one thing `dsh plugin add <local path>` does correctly.
#
# Re-run this script to update: it fast-forwards the clone and reinstalls.
#
# POSIX sh on purpose: this runs under `curl | sh`, where `sh` may be dash or
# busybox ash, not bash.

set -eu

REPOSITORY="${DSH_MODEL_HUB_REPO:-https://github.com/Mhmd7-7/dsh-ai-model-hub.git}"
REF="${DSH_MODEL_HUB_REF:-main}"
PROFILE="${DSH_PROFILE:-web}"
INSTALL_DIR="${DSH_MODEL_HUB_DIR:-}"
SKIP_DOCTOR=0

usage() {
    cat <<'EOF'
Install (or update) the dsh-ai-model-hub plugin from GitHub.

Usage: install.sh [options]

Options:
  -p, --profile <name>     DSH profile to install into (default: web)
  -d, --dir <path>         where to clone the repository
                           (default: $DSH_HOME/plugins/dsh-ai-model-hub)
  -r, --ref <ref>          branch, tag or commit to install (default: main)
      --repository <url>   git remote to clone (default: the canonical GitHub URL)
      --skip-doctor        install only; skip the post-install verification
  -h, --help               show this help

Environment variables supply the same values, which is how you configure the
`curl | sh` form where arguments are awkward:

  DSH_PROFILE, DSH_MODEL_HUB_DIR, DSH_MODEL_HUB_REF, DSH_MODEL_HUB_REPO
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
    RED=$(printf '\033[31m'); GREEN=$(printf '\033[32m'); CYAN=$(printf '\033[36m')
    RESET=$(printf '\033[0m')
else
    BOLD=''; DIM=''; RED=''; GREEN=''; CYAN=''; RESET=''
fi

# Report a fatal problem and stop, with the same "what to do about it" shape the
# rest of this project's diagnostics use.
fail() {
    printf '\n%sinstall.sh: %s%s\n' "$RED" "$1" "$RESET" >&2
    exit 1
}

step() { printf '%s%s%s\n' "$CYAN" "$1" "$RESET"; }
note() { printf '%s      %s%s\n' "$DIM" "$1" "$RESET"; }

# ── 1. Preflight ────────────────────────────────────────────────────────────
printf '%sdsh-ai-model-hub — install from GitHub%s\n\n' "$BOLD" "$RESET"

for tool in node npm git dsh; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        case "$tool" in
            node) fail "'node' was not found on PATH. Install Node 22.6 or newer from https://nodejs.org" ;;
            npm)  fail "'npm' was not found on PATH. npm ships with Node; reinstall Node if it is missing." ;;
            git)  fail "'git' was not found on PATH. Install git — this installer clones the repository." ;;
            dsh)  fail "'dsh' was not found on PATH. Install DeepSeek Harness first: npm install -g @deepseek-ai/dsh" ;;
        esac
    fi
done

NODE_VERSION=$(node --version 2>/dev/null || true)
NODE_MAJOR=$(printf '%s' "$NODE_VERSION" | sed -n 's/^v\([0-9]*\)\..*$/\1/p')
NODE_MINOR=$(printf '%s' "$NODE_VERSION" | sed -n 's/^v[0-9]*\.\([0-9]*\).*$/\1/p')
NODE_MAJOR=${NODE_MAJOR:-0}
NODE_MINOR=${NODE_MINOR:-0}
if [ "$NODE_MAJOR" -lt 22 ] || { [ "$NODE_MAJOR" -eq 22 ] && [ "$NODE_MINOR" -lt 6 ]; }; then
    fail "Node 22.6 or newer is required (found $NODE_VERSION). The plugin runs TypeScript directly through Node's type stripping, which older releases do not have."
fi

DSH_HOME_DIR="${DSH_HOME:-$HOME/.dsh}"
if [ -z "$INSTALL_DIR" ]; then
    INSTALL_DIR="$DSH_HOME_DIR/plugins/dsh-ai-model-hub"
fi

# ── 2. A checkout to install from ───────────────────────────────────────────
# Two sources, in priority order: the directory this script itself lives in (so
# running it from a clone installs THAT clone, and never rewrites your working
# tree), then the canonical install directory, cloned on demand.
#
# Under `curl | sh` there is no script file at all — `$0` is `sh` or `bash` — and
# that is exactly the case this branch is here to tell apart.
SCRIPT_DIR=''
case "${0:-}" in
    ''|sh|bash|dash|ash|*/sh|*/bash|*/dash|*/ash) SCRIPT_DIR='' ;;
    *) SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" 2>/dev/null && pwd) || SCRIPT_DIR='' ;;
esac

CHECKOUT=''
MODE=''
if [ -n "$SCRIPT_DIR" ] && [ -d "$SCRIPT_DIR/dsh-plugin" ]; then
    CHECKOUT="$SCRIPT_DIR"
    MODE='checkout'
    step '[1/3] Using the checkout this script lives in:'
    note "$CHECKOUT"
elif [ -d "$INSTALL_DIR/dsh-plugin" ] && [ -d "$INSTALL_DIR/.git" ]; then
    CHECKOUT="$INSTALL_DIR"
    MODE='update'
    step '[1/3] Updating the existing clone...'
    note "$CHECKOUT"
    note "ref $REF of $REPOSITORY"

    # Fetch first, then move to the requested ref. The pull is best-effort: a tag
    # or a commit hash checks out but has nothing to fast-forward, and that is not
    # a failure.
    git -C "$CHECKOUT" fetch --prune origin \
        || fail "git fetch failed. Check your network and that $REPOSITORY is reachable."

    if git -C "$CHECKOUT" checkout "$REF"; then
        # A fast-forward can legitimately fail, and the common cause is this
        # project's own installer: `npm install` inside the checkout rewrites
        # `dsh-plugin/package-lock.json`, and git refuses to pull over a locally
        # modified file. Say so, rather than reporting an update that did not
        # happen — silently installing the old revision is the worst outcome.
        if ! git -C "$CHECKOUT" pull --ff-only origin "$REF"; then
            note 'could not fast-forward: the checkout has local changes git will not pull over.'
            note 'An earlier install rewrites dsh-plugin/package-lock.json, so this is expected.'
            note 'Installing the revision already present. To discard that generated change and update:'
            note "  git -C \"$CHECKOUT\" checkout -- dsh-plugin/package-lock.json"
        fi
    else
        # The ref exists remotely but not locally yet (a new branch, or a commit
        # that is not on any fetched branch tip).
        git -C "$CHECKOUT" fetch origin "$REF" || fail "could not fetch '$REF' from $REPOSITORY."
        git -C "$CHECKOUT" checkout FETCH_HEAD || fail "could not check out '$REF' in $CHECKOUT."
    fi
elif [ -e "$INSTALL_DIR" ]; then
    fail "$INSTALL_DIR already exists but is not a clone of this repository. Move it aside, or choose another location with --dir."
else
    MODE='clone'
    step '[1/3] Cloning the repository...'
    note "$REPOSITORY"
    note "ref $REF"
    note "into $INSTALL_DIR"

    # A shallow clone of a branch or tag is the fast path. It fails for a commit
    # hash — git cannot ask for a single commit by name — so that case falls back
    # to a full clone followed by an explicit checkout.
    #
    # The fallback deliberately does not claim the ref is the problem: it runs for
    # any shallow-clone failure, including an unreachable remote. The full clone
    # below is what produces the real error, and it is the one reported.
    if ! git clone --depth 1 --branch "$REF" "$REPOSITORY" "$INSTALL_DIR"; then
        note "shallow clone failed; retrying in full (this also covers a commit hash, which --depth cannot name)..."
        rm -rf "$INSTALL_DIR"
        git clone "$REPOSITORY" "$INSTALL_DIR" || fail "git clone failed."
        git -C "$INSTALL_DIR" checkout "$REF" || fail "'$REF' is not a branch, tag or commit in $REPOSITORY."
    fi
    CHECKOUT="$INSTALL_DIR"
fi

[ -f "$CHECKOUT/scripts/install-plugin.mjs" ] \
    || fail "$CHECKOUT does not look like dsh-ai-model-hub (no scripts/install-plugin.mjs)."

# ── 3. Install into the profile ─────────────────────────────────────────────
# Everything below is the repository's own installer. This script exists to get a
# checkout onto the machine and hand over; it deliberately does not re-implement
# installation, so there is one place where that logic can be right or wrong.
printf '\n'
step "[2/3] Installing into profile '$PROFILE'..."

if [ "$SKIP_DOCTOR" -eq 1 ]; then
    ( cd "$CHECKOUT" && node --no-deprecation scripts/install-plugin.mjs --profile "$PROFILE" --skip-doctor ) \
        || fail "the installer failed. Nothing was half-installed: re-run this script once the cause is fixed."
else
    ( cd "$CHECKOUT" && node --no-deprecation scripts/install-plugin.mjs --profile "$PROFILE" ) \
        || fail "the installer failed. Nothing was half-installed: re-run this script once the cause is fixed."
fi

printf '\n%sDone.%s\n\n' "$GREEN" "$RESET"
printf "Restart DeepSeek Harness so the '%s' profile reloads, then ask the agent:\n" "$PROFILE"
printf '    List the available AI model capabilities.\n\n'
printf '  %ssource:%s  %s (%s)\n' "$BOLD" "$RESET" "$CHECKOUT" "$MODE"
printf '  %supdate:%s  re-run this script; it fast-forwards the clone and reinstalls\n' "$BOLD" "$RESET"
printf '  %scatalog:%s the clone ships config/models.json, which the plugin finds\n' "$BOLD" "$RESET"
printf '           without configuration. Edit that file to add real models.\n'
printf '  %sremove:%s  dsh plugin --profile %s remove dsh-ai-model-hub-plugin\n' "$BOLD" "$RESET" "$PROFILE"
