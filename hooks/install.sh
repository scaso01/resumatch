#!/usr/bin/env bash
# Install pre-commit hook
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOKS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/.git/hooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: .git/hooks directory not found. Is this a git repo?"
    exit 1
fi

cp "$SCRIPT_DIR/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"
echo "Pre-commit hook installed."
