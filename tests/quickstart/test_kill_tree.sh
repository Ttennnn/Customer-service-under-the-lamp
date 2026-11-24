#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../start.sh
source "$HERE/../../start.sh"

PYTHON="${TEST_PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "SKIP: 找不到 python（可用 TEST_PYTHON 指定）"
  exit 0
fi

tmp="$(mktemp -d)"
PARENT_PID=""
PARENT_WINPID=""
CHILD_WINPID=""

cleanup() {
  if [ -n "$CHILD_WINPID" ]; then
    taskkill //F //T //PID "$CHILD_WINPID" >/dev/null 2>&1 || true
  fi
  if [ -n "$PARENT_PID" ]; then
    kill "$PARENT_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "$PARENT_WINPID" ]; then
    taskkill //F //T //PID "$PARENT_WINPID" >/dev/null 2>&1 || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT

"$PYTHON" -c 'import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
print(child.pid, flush=True)
time.sleep(300)' > "$tmp/child_winpid" &
PARENT_PID=$!

for _ in $(seq 1 50); do
  [ -s "$tmp/child_winpid" ] && break
  sleep 0.2
done
CHILD_WINPID="$(head -n1 "$tmp/child_winpid" 2>/dev/null || true)"
PARENT_WINPID="$(cat "/proc/$PARENT_PID/winpid" 2>/dev/null || true)"

case "$CHILD_WINPID" in
  ''|*[!0-9]*) echo "FAIL 无法取得子进程 Windows PID"; exit 1 ;;
esac
case "$PARENT_WINPID" in
  ''|*[!0-9]*) echo "FAIL 无法取得父进程 Windows PID"; exit 1 ;;
esac

alive() {
  tasklist //FI "PID eq $1" //NH 2>/dev/null | grep -qE "[[:space:]]$1[[:space:]]"
}

if ! winpid_of "$PARENT_PID" >/dev/null 2>&1; then
  echo "FAIL winpid_of 未实现：无法把 MSYS PID 转成 Windows PID"
  exit 1
fi

pass=0
fail=0
check() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    pass=$((pass + 1)); printf 'ok   %s\n' "$label"
  else
    fail=$((fail + 1)); printf 'FAIL %s\n  expected: %s\n  actual:   %s\n' "$label" "$expected" "$actual"
  fi
}

check "winpid_of 与 /proc 一致" "$PARENT_WINPID" "$(winpid_of "$PARENT_PID")"

kill_tree "$PARENT_PID"
sleep 1

if alive "$PARENT_WINPID"; then
  check "父进程已被结束" "dead" "alive"
else
  check "父进程已被结束" "dead" "dead"
fi
if alive "$CHILD_WINPID"; then
  check "子进程已被结束" "dead" "alive"
else
  check "子进程已被结束" "dead" "dead"
fi

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
