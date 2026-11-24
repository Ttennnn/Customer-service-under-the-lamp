#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../start.sh
source "$HERE/../../start.sh"

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

check "两位数按数值排序" "feature/ch10-b" \
  "$(printf 'local:feature/ch2-a\nlocal:feature/ch10-b\nlocal:feature/ch9-c\n' | pick_latest_branch)"
check "同章节号本地优先" "feature/ch10-b" \
  "$(printf 'remote:feature/ch10-a\nlocal:feature/ch10-b\n' | pick_latest_branch)"
check "远程独有更高章节" "feature/ch11-z" \
  "$(printf 'local:feature/ch10-b\nremote:feature/ch11-z\n' | pick_latest_branch)"
check "忽略非章节分支" "feature/ch10-a" \
  "$(printf 'local:feature/misc\nlocal:feature/ch10-a\n' | pick_latest_branch)"

if printf 'local:feature/misc\n' | pick_latest_branch >/dev/null 2>&1; then
  check "无匹配返回非零" "non-zero" "zero"
else
  check "无匹配返回非零" "non-zero" "non-zero"
fi

# 目标校验失败时，不得先执行破坏性的端口清理
ordering_tmp="$(mktemp -d)"
touched="$ordering_tmp/release-port-called"
(
  WT="$ordering_tmp"
  ROOT="$ordering_tmp"
  PY="/bin/false"
  PY_SOURCE="worktree"
  PORT=8000
  printf 'x\n' > "$WT/run.py"
  release_port() { printf 'called\n' > "$touched"; }
  log() { :; }
  die() { exit 1; }
  run_server "feature/ch99-x"
) >/dev/null 2>&1 || true
if [ -s "$touched" ]; then
  check "venv 校验失败时不得先清理 8000" "not-called" "called"
else
  check "venv 校验失败时不得先清理 8000" "not-called" "not-called"
fi

rm -f "$touched" "$ordering_tmp/run.py"
(
  WT="$ordering_tmp"
  ROOT="$ordering_tmp"
  PY="/bin/true"
  PY_SOURCE="worktree"
  PORT=8000
  release_port() { printf 'called\n' > "$touched"; }
  log() { :; }
  die() { exit 1; }
  run_server "feature/ch99-x"
) >/dev/null 2>&1 || true
if [ -s "$touched" ]; then
  check "缺少 run.py 时不得先清理 8000" "not-called" "called"
else
  check "缺少 run.py 时不得先清理 8000" "not-called" "not-called"
fi
rm -rf "$ordering_tmp"

# --- ensure_infra：启动前保证 MySQL(3306)/Milvus(19530) 就绪 ---
infra_tmp="$(mktemp -d)"
mkdir -p "$infra_tmp/repo"
printf 'services: {}\n' > "$infra_tmp/repo/docker-compose.yml"

(
  ROOT="$infra_tmp/repo"
  INFRA_PORTS="3306 19530"
  INFRA_TIMEOUT=2
  port_open() { return 0; }
  docker() { printf '%s\n' "$*" >> "$infra_tmp/calls-ready"; return 0; }
  log() { :; }
  die() { exit 1; }
  ensure_infra
) || true
check "依赖已就绪时不调用 docker" "not-called" \
  "$([ -s "$infra_tmp/calls-ready" ] && echo called || echo not-called)"

(
  ROOT="$infra_tmp/repo"
  INFRA_PORTS="3306 19530"
  INFRA_TIMEOUT=5
  calls="$infra_tmp/calls-up"
  port_open() { grep -q "up -d" "$calls" 2>/dev/null; }
  docker() { printf '%s\n' "$*" >> "$calls"; return 0; }
  log() { :; }
  sleep() { :; }
  die() { exit 1; }
  ensure_infra
) || true
check "未就绪时用仓库根的 compose 启动" "yes" \
  "$(grep -q "compose --project-directory $infra_tmp/repo -f $infra_tmp/repo/docker-compose.yml up -d" "$infra_tmp/calls-up" 2>/dev/null && echo yes || echo no)"

(
  ROOT="$infra_tmp/repo"
  INFRA_PORTS="3306 19530"
  INFRA_TIMEOUT=2
  port_open() { return 1; }
  docker() { return 0; }
  log() { :; }
  sleep() { :; }
  die() { printf '%s\n' "$*" > "$infra_tmp/died"; exit 1; }
  ensure_infra
) || true
check "等待超时给出包含端口的明确错误" "yes" \
  "$(grep -q "19530" "$infra_tmp/died" 2>/dev/null && grep -q "3306" "$infra_tmp/died" && echo yes || echo no)"

(
  ROOT="$infra_tmp/repo"
  INFRA_PORTS="3306 19530"
  INFRA_TIMEOUT=5
  calls="$infra_tmp/calls-daemon"
  daemon_up=0
  port_open() { grep -q "up -d" "$calls" 2>/dev/null; }
  docker() {
    printf '%s\n' "$*" >> "$calls"
    case "$*" in
      "info") [ "$daemon_up" -eq 1 ] ;;
      "desktop start") daemon_up=1 ;;
      "compose version") return 0 ;;
      *) return 0 ;;
    esac
  }
  log() { :; }
  sleep() { :; }
  die() { exit 1; }
  ensure_infra
) || true
check "守护进程未运行时尝试启动 Docker Desktop" "yes" \
  "$(grep -q "^desktop start$" "$infra_tmp/calls-daemon" 2>/dev/null && echo yes || echo no)"

(
  SKIP_INFRA_CHECK=1
  INFRA_PORTS="3306 19530"
  port_open() { return 1; }
  docker() { printf '%s\n' "$*" >> "$infra_tmp/calls-skip"; return 0; }
  log() { :; }
  die() { exit 1; }
  ensure_infra
) || true
check "SKIP_INFRA_CHECK=1 跳过依赖检查" "not-called" \
  "$([ -s "$infra_tmp/calls-skip" ] && echo called || echo not-called)"

(
  WT="$infra_tmp"
  ROOT="$infra_tmp"
  PY="/bin/true"
  PY_SOURCE="worktree"
  PORT=8000
  printf 'x\n' > "$WT/run.py"
  release_port() { printf 'called\n' > "$infra_tmp/released"; }
  ensure_infra() { return 1; }
  log() { :; }
  die() { exit 1; }
  run_server "feature/ch99-x"
) >/dev/null 2>&1 || true
check "依赖启动失败时不得先清理 8000" "not-called" \
  "$([ -s "$infra_tmp/released" ] && echo called || echo not-called)"
rm -rf "$infra_tmp"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
