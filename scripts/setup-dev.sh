#!/usr/bin/env bash
# =============================================================================
# ADK-Rust Development Environment Setup
# =============================================================================
# Installs optional build acceleration tools for your platform.
# Safe to re-run — skips already-installed tools.
#
# Usage:
#   ./scripts/setup-dev.sh          # Install everything recommended
#   ./scripts/setup-dev.sh --check  # Just check what's installed
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok() { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
miss() { echo -e "  ${RED}✗${NC} $1"; }

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=true
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "==================================="
echo " ADK-Rust Dev Environment Setup"
echo " OS: $OS  Arch: $ARCH"
echo "==================================="
echo ""

# ---------------------------------------------------------------------------
# Detect package manager
# ---------------------------------------------------------------------------
install_pkg() {
  local name="$1"
  local brew_name="${2:-$name}"
  local apt_name="${3:-$name}"
  local dnf_name="${4:-$apt_name}"

  if $CHECK_ONLY; then return; fi

  if [[ "$OS" == "Darwin" ]]; then
    if command -v brew &>/dev/null; then
      echo "  Installing $name via brew..."
      brew install "$brew_name" 2>/dev/null || true
    else
      warn "brew not found — install Homebrew first: https://brew.sh"
    fi
  elif [[ "$OS" == "Linux" ]]; then
    if command -v apt-get &>/dev/null; then
      echo "  Installing $name via apt..."
      sudo apt-get install -y "$apt_name" 2>/dev/null || true
    elif command -v dnf &>/dev/null; then
      echo "  Installing $name via dnf..."
      sudo dnf install -y "$dnf_name" 2>/dev/null || true
    else
      warn "No supported package manager found for $name"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Check / install tools
# ---------------------------------------------------------------------------

echo "Core toolchain:"

if command -v rustc &>/dev/null; then
  ok "rustc $(rustc --version | awk '{print $2}')"
else
  miss "rustc — install from https://rustup.rs"
fi

if command -v cargo &>/dev/null; then
  ok "cargo $(cargo --version | awk '{print $2}')"
else
  miss "cargo — install from https://rustup.rs"
fi

# Gate-resolved toolchain — quality gates run through scripts/cargo-pinned.sh,
# which routes to the rust-toolchain.toml pin via rustup when available and
# otherwise requires ambient cargo to match the pin exactly. Report what the
# gates will actually use; ambient cargo alone can mislead when shims are
# shadowed.
PINNED_TOOLCHAIN="$(sed -n 's/^channel *= *"//p' "$(dirname "${BASH_SOURCE[0]}")/../rust-toolchain.toml" 2>/dev/null | cut -d'"' -f1 | head -1)"
GATE_CARGO="$(bash "$(dirname "${BASH_SOURCE[0]}")/cargo-pinned.sh" --version 2>/dev/null | awk '{print $2}')" || GATE_CARGO=""
if [[ -z "$GATE_CARGO" ]]; then
  miss "no cargo resolves the $PINNED_TOOLCHAIN pin for gates — run 'rustup toolchain install $PINNED_TOOLCHAIN' or use 'devenv shell'"
elif [[ "$GATE_CARGO" == "$PINNED_TOOLCHAIN" ]]; then
  ok "gates resolve cargo $GATE_CARGO (matches rust-toolchain.toml pin)"
else
  warn "gates resolve cargo $GATE_CARGO, pin is $PINNED_TOOLCHAIN"
fi

echo ""
echo "Build acceleration (optional):"

# sccache
if command -v sccache &>/dev/null; then
  ok "sccache $(sccache --version | awk '{print $2}')"
else
  miss "sccache — shared compilation cache (speeds up rebuilds significantly)"
  install_pkg sccache
fi

# mold (Linux only)
if [[ "$OS" == "Linux" ]]; then
  if command -v mold &>/dev/null; then
    ok "mold $(mold --version | head -1)"
  else
    miss "mold — fast linker for Linux"
    install_pkg mold
  fi
else
  ok "mold — not needed on macOS (default linker is fast)"
fi

# cmake (needed for openai-webrtc feature / audiopus)
if command -v cmake &>/dev/null; then
  ok "cmake $(cmake --version | head -1 | awk '{print $3}')"
else
  warn "cmake — needed only for openai-webrtc feature (audiopus)"
  install_pkg cmake
fi

echo ""
echo "Test runner (quality gate):"

# cargo-nextest backs one of the three quality gates, and no other installer
# provides it — `cargo test` does not substitute (it skips nextest's isolation).
if command -v cargo-nextest &>/dev/null; then
  ok "cargo-nextest $(cargo nextest --version 2>/dev/null | head -1 | awk '{print $2}')"
else
  miss "cargo-nextest — required by the test gate"
  if ! $CHECK_ONLY; then
    echo "  Installing cargo-nextest via cargo..."
    cargo install cargo-nextest --locked 2>/dev/null || \
      warn "could not install cargo-nextest — see https://nexte.st/docs/installation/"
  fi
fi

echo ""
echo "Feature-gated build tools:"

# protoc — lance's build script needs it, so `adk-rag --features lancedb` cannot
# compile without it. CI installs it on every feature-coverage runner, so a
# missing local copy fails while CI stays green.
if command -v protoc &>/dev/null; then
  ok "protoc $(protoc --version | awk '{print $2}')"
else
  miss "protoc — needed for adk-rag --features lancedb"
  install_pkg protoc protobuf protobuf-compiler
fi

# ALSA headers — examples/desktop_audio (cpal) links alsa-sys on Linux, so the
# pre-push examples gate fails without them. CI installs libasound2-dev on the
# examples runners; macOS uses CoreAudio and needs nothing.
if [[ "$OS" == "Linux" ]]; then
  if pkg-config --exists alsa 2>/dev/null; then
    ok "alsa $(pkg-config --modversion alsa 2>/dev/null)"
  else
    miss "alsa — needed for examples/desktop_audio (pre-push gate)"
    install_pkg alsa-lib alsa-lib libasound2-dev alsa-lib-devel
  fi
else
  ok "alsa — not needed on $OS (CoreAudio)"
fi

# NASM is only consumed by aws-lc-sys on MSVC targets, so Unix hosts never need
# it. Reported rather than silently skipped so the check output matches the
# tool matrix in AGENTS.md on every platform.
ok "nasm — not needed on $OS (aws-lc-sys requires it only on Windows MSVC)"

echo ""
echo "Frontend tooling (ADK Studio UI):"

if command -v node &>/dev/null; then
  ok "node $(node --version)"
else
  miss "node — install Node.js 20+ for ADK Studio UI"
fi

if command -v pnpm &>/dev/null; then
  ok "pnpm $(pnpm --version)"
elif command -v npm &>/dev/null; then
  ok "npm $(npm --version) (pnpm recommended: npm i -g pnpm)"
else
  miss "npm/pnpm — needed for ADK Studio UI"
fi

echo ""
echo "Git hooks (quality gates):"

# Shell-script linter — backs the pre-commit shellcheck gate (see lefthook.yml).
# devenv users get this for free via git-hooks.hooks; install it here so the
# non-Nix lefthook path reaches the same parity.
if command -v shellcheck &>/dev/null; then
  ok "shellcheck $(shellcheck --version | awk '/^version:/ {print $2}')"
else
  miss "shellcheck — shell-script linter for the pre-commit gate"
  install_pkg shellcheck
fi

# lefthook — runs fmt + clippy + shellcheck on pre-commit, nextest on pre-push (see lefthook.yml)
if command -v lefthook &>/dev/null; then
  ok "lefthook $(lefthook version 2>/dev/null | head -1)"
else
  miss "lefthook — git hook runner for the quality gates"
  install_pkg lefthook
  # install_pkg is best-effort: it silently no-ops when lefthook isn't in the
  # platform's default repos (e.g. apt/dnf). Fall back to npm, then warn.
  if ! $CHECK_ONLY && ! command -v lefthook &>/dev/null && command -v npm &>/dev/null; then
    echo "  Installing lefthook via npm..."
    npm install -g lefthook 2>/dev/null || true
  fi
  if ! $CHECK_ONLY && ! command -v lefthook &>/dev/null; then
    warn "could not install lefthook automatically — install it manually: https://lefthook.dev"
  fi
fi

# Register the hooks in this clone
if ! $CHECK_ONLY && command -v lefthook &>/dev/null; then
  if git rev-parse --git-dir &>/dev/null; then
    echo "  Registering git hooks..."
    if lefthook install &>/dev/null; then
      ok "git hooks installed (pre-commit, pre-push)"
    else
      warn "run 'lefthook install' from the repo root to register hooks"
    fi
  else
    warn "not a git repo — run 'lefthook install' from the repo root to register hooks"
  fi
fi

echo ""
echo "Environment variables:"

# sccache wrapper
if [[ -n "${RUSTC_WRAPPER:-}" ]]; then
  ok "RUSTC_WRAPPER=$RUSTC_WRAPPER"
else
  if command -v sccache &>/dev/null; then
    warn "RUSTC_WRAPPER not set — add to your shell profile:"
    echo "       export RUSTC_WRAPPER=sccache"
  fi
fi

# cmake policy (for cmake 4.x + audiopus)
if [[ -n "${CMAKE_POLICY_VERSION_MINIMUM:-}" ]]; then
  ok "CMAKE_POLICY_VERSION_MINIMUM=$CMAKE_POLICY_VERSION_MINIMUM"
else
  if command -v cmake &>/dev/null; then
    CMAKE_MAJOR=$(cmake --version | head -1 | awk '{print $3}' | cut -d. -f1)
    if [[ "$CMAKE_MAJOR" -ge 4 ]]; then
      warn "CMAKE_POLICY_VERSION_MINIMUM not set — needed for cmake 4.x:"
      echo "       export CMAKE_POLICY_VERSION_MINIMUM=3.5"
    fi
  fi
fi

echo ""
if $CHECK_ONLY; then
  echo "Run without --check to install missing tools."
else
  echo "Done. Run 'make help' for build commands."
fi
