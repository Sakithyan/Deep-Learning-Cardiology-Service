#!/usr/bin/env bash
set -euo pipefail

echo "===== ENV CHECK ====="

if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null || grep -qiE "microsoft|wsl" /proc/sys/kernel/osrelease 2>/dev/null; then
  echo "[INFO] WSL2 detected"
  echo "[INFO] Recommended: Docker Desktop + WSL integration"
else
  echo "[INFO] Native Linux/VM mode"
fi

check_cmd() {
  local c="$1"
  if command -v "$c" >/dev/null 2>&1; then
    echo "[OK] $c"
  else
    echo "[MISSING] $c"
  fi
}

for c in curl git mvn java docker; do
  check_cmd "$c"
done

echo
echo "===== VERSIONS ====="
command -v curl >/dev/null 2>&1 && curl --version | head -n 1 || true
command -v git >/dev/null 2>&1 && git --version || true
command -v mvn >/dev/null 2>&1 && mvn -version | head -n 2 || true
command -v java >/dev/null 2>&1 && java -version || true
command -v docker >/dev/null 2>&1 && docker --version || true
command -v docker >/dev/null 2>&1 && docker compose version || true

echo
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    echo "[OK] docker daemon reachable"
  else
    echo "[WARN] docker command exists but daemon not reachable"
  fi
fi

if command -v systemctl >/dev/null 2>&1; then
  systemctl is-active docker >/dev/null 2>&1 && echo "[OK] docker service active" || true
fi

echo "[DONE] check finished"
