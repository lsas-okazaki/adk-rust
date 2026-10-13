#!/usr/bin/env bash
# Run cargo on the rust-toolchain.toml-pinned toolchain, and refuse otherwise.
#
# `cargo` on PATH is not necessarily rustup-managed: a shadowing cargo (e.g.
# from nix) silently ignores rust-toolchain.toml, so quality gates would run
# on the wrong compiler and fail on lints the pinned toolchain never emits.
# This wrapper is fail-closed: it resolves the pin through rustup when
# available, then verifies the cargo it is about to exec reports exactly the
# pinned version. Anything else exits non-zero without running the gate.
#
# Usage: scripts/cargo-pinned.sh <cargo args...>
#
# This prepends the toolchain's bin dir rather than `rustup run`: cargo
# resolves subcommands such as cargo-clippy from PATH, so routing only the
# cargo binary would pair a pinned cargo with a shadowing newer clippy driver.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHANNEL="$(sed -n 's/^channel *= *"//p' "$ROOT/rust-toolchain.toml" 2>/dev/null | cut -d'"' -f1 | head -1)"

if [[ -z "$CHANNEL" ]]; then
  echo "error: cargo-pinned.sh cannot read channel from $ROOT/rust-toolchain.toml" >&2
  exit 1
fi

if command -v rustup &>/dev/null && rustup toolchain list 2>/dev/null | grep -q "^${CHANNEL}"; then
  TOOLCHAIN="$(rustup toolchain list 2>/dev/null | grep -o "^${CHANNEL}[^ ]*" | head -1)"
  TOOLCHAIN_BIN="$(dirname "$(rustup which --toolchain "$TOOLCHAIN" cargo 2>/dev/null)")"
  if [[ -n "$TOOLCHAIN_BIN" && -d "$TOOLCHAIN_BIN" ]]; then
    export PATH="$TOOLCHAIN_BIN:$PATH"
  fi
fi

RESOLVED="$(cargo --version 2>/dev/null | awk '{print $2}')"
if [[ "$RESOLVED" != "$CHANNEL" ]]; then
  echo "error: cargo-pinned.sh refuses to run: resolved cargo ${RESOLVED:-missing}, pin is $CHANNEL (rust-toolchain.toml)" >&2
  echo "  Install the pinned toolchain (rustup toolchain install $CHANNEL) or run from 'devenv shell'." >&2
  exit 1
fi
exec cargo "$@"
