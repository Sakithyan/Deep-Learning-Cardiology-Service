#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE_ARG="${1:--d}"

chmod +x "$SCRIPT_DIR/preload"
chmod +x "$SCRIPT_DIR/scripts/check-env.sh"

"$SCRIPT_DIR/preload" "$MODE_ARG"
"$SCRIPT_DIR/scripts/check-env.sh"

echo "Session base ready"
