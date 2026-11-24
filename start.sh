#!/usr/bin/env bash
# 启动当前最新章节分支的本地开发服务。
# 用法见 ./start.sh --help

SCRIPT_DIR=""
ROOT=""
WT=""
PY=""
PY_SOURCE=""
PORT=8000
START_TIMEOUT="${START_TIMEOUT:-60}"
INFRA_PORTS="${INFRA_PORTS:-3306 19530}"
INFRA_TIMEOUT="${INFRA_TIMEOUT:-180}"
SERVER_PID=""

log() {
  printf '[start] %s\n' "$*"
}

die() {
  printf '[start] 错误: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
用法: ./start.sh [--print-target]

  --print-target  只解析并打印将启动的分支、worktree 与 Python，不启动服务
  -h, --help      显示本帮助

环境变量:
  NO_BROWSER=1     启动成功后不自动打开浏览器
  START_TIMEOUT=秒 等待服务可用的最长秒数（默认 60）
  INFRA_TIMEOUT=秒 等待 Docker/依赖服务就绪的最长秒数（默认 180）
  INFRA_PORTS="3306 19530"
                   启动前必须监听的本地端口（默认 MySQL 与 Milvus）
  SKIP_INFRA_CHECK=1
                   跳过依赖服务预检（不自动启动 docker compose）
EOF
}

pick_latest_branch() {
  local best="" best_num=-1 best_is_local=0
  local line kind name num
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    kind="${line%%:*}"
    name="${line#*:}"
    case "$name" in
      feature/ch[0-9]*-*) ;;
      *) continue ;;
    esac
    num="${name#feature/ch}"
    num="${num%%-*}"
    case "$num" in
      ''|*[!0-9]*) continue ;;
    esac
    num=$((10#$num))
    if [ "$num" -gt "$best_num" ]; then
      best="$name"
      best_num="$num"
      best_is_local=0
      if [ "$kind" = "local" ]; then
        best_is_local=1
      fi
    elif [ "$num" -eq "$best_num" ] && [ "$kind" = "local" ] && [ "$best_is_local" -eq 0 ]; then
      best="$name"
      best_is_local=1
    fi
  done
  if [ -z "$best" ]; then
    return 1
  fi
  printf '%s\n' "$best"
}

find_repo_root() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if ! git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    die "启动脚本必须位于 git 仓库内：$SCRIPT_DIR"
  fi
  ROOT="$SCRIPT_DIR"
}

collect_branches() {
  git -C "$ROOT" for-each-ref --format='%(refname:short)' refs/heads/feature \
    | sed 's#^#local:#'
  git -C "$ROOT" for-each-ref --format='%(refname:short)' refs/remotes/origin/feature \
    | sed -e 's#^origin/##' -e 's#^#remote:#'
}

resolve_worktree() {
  local branch="$1"
  local wt dir

  wt="$(git -C "$ROOT" worktree list --porcelain | awk -v b="refs/heads/$branch" '
    /^worktree / { wt = substr($0, 10) }
    /^branch / { if ($2 == b) { print wt; exit } }
  ')"
  if [ -n "$wt" ]; then
    WT="$wt"
    return 0
  fi

  dir="$ROOT/.worktrees/${branch//\//-}"
  if [ -e "$dir" ]; then
    die "目录已存在但不是 $branch 的 worktree：$dir（请先手动处理）"
  fi

  if git -C "$ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
    log "创建 worktree：$dir（$branch）"
    git -C "$ROOT" worktree add "$dir" "$branch" >/dev/null \
      || die "创建 worktree 失败：$branch"
  else
    log "创建 worktree：$dir（origin/$branch → 本地 $branch）"
    git -C "$ROOT" worktree add -b "$branch" "$dir" "origin/$branch" >/dev/null \
      || die "创建 worktree 失败：origin/$branch"
  fi

  if [ ! -f "$dir/.env" ] && [ -f "$ROOT/.env" ]; then
    cp "$ROOT/.env" "$dir/.env"
    log "已从仓库根复制 .env 到新 worktree"
  fi
  WT="$dir"
}

select_python() {
  PY="$WT/.venv/Scripts/python.exe"
  PY_SOURCE="worktree"
  if [ ! -x "$PY" ]; then
    PY="$ROOT/.venv/Scripts/python.exe"
    PY_SOURCE="root-fallback"
  fi
  if [ ! -x "$PY" ]; then
    die "未找到可用的虚拟环境。请在仓库根创建：
  python -m venv .venv
  .venv/Scripts/python.exe -m pip install -r requirements.txt"
  fi
}

release_port() {
  if ! command -v powershell >/dev/null 2>&1; then
    log "未找到 PowerShell，跳过端口清理"
    return 0
  fi
  powershell -NoProfile -ExecutionPolicy Bypass -Command "
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(\$false)
    \$ErrorActionPreference = 'SilentlyContinue'
    \$listeners = @(Get-NetTCPConnection -LocalPort $PORT -State Listen | Select-Object -ExpandProperty OwningProcess -Unique)
    \$targets = @()
    foreach (\$p in \$listeners) {
      \$proc = Get-CimInstance Win32_Process -Filter \"ProcessId=\$p\"
      if (\$proc) {
        \$target = \$proc
        \$parent = Get-CimInstance Win32_Process -Filter ('ProcessId=' + \$proc.ParentProcessId)
        if (\$parent -and \$parent.Name -match '^python') { \$target = \$parent }
        \$targets += \$target
      } else {
        foreach (\$kid in @(Get-CimInstance Win32_Process -Filter \"ParentProcessId=\$p\")) {
          \$targets += \$kid
        }
      }
    }
    foreach (\$t in @(\$targets | Sort-Object ProcessId -Unique)) {
      Write-Output ('[start] 结束占用 $PORT 端口的进程树 PID=' + \$t.ProcessId + ' (' + \$t.Name + ')')
      & taskkill /F /T /PID \$t.ProcessId | Out-Null
    }
  " 2>/dev/null
  sleep 1
}

wait_for_http() {
  local i=0
  local code
  while [ "$i" -lt "$START_TIMEOUT" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "服务进程已退出"
      return 1
    fi
    code="$(curl -s -o /dev/null -m 2 -w '%{http_code}' \
      "http://127.0.0.1:$PORT/" || true)"
    case "$code" in
      2*|3*|4*) return 0 ;;
    esac
    sleep 1
    i=$((i + 1))
  done
  log "等待 ${START_TIMEOUT}s 服务仍未就绪，继续保留服务输出"
  return 1
}

open_browser() {
  if [ "${NO_BROWSER:-0}" = "1" ]; then
    log "NO_BROWSER=1，跳过打开浏览器"
    return 0
  fi
  log "打开 http://127.0.0.1:$PORT/"
  cmd //c start "" "http://127.0.0.1:$PORT/" >/dev/null 2>&1 || true
}

winpid_of() {
  local pid="$1"
  local winpid
  winpid="$(cat "/proc/$pid/winpid" 2>/dev/null || true)"
  case "$winpid" in
    ''|*[!0-9]*) return 1 ;;
  esac
  printf '%s\n' "$winpid"
}

kill_tree() {
  local pid="$1"
  local winpid
  [ -n "$pid" ] || return 0
  if winpid="$(winpid_of "$pid")"; then
    taskkill //F //T //PID "$winpid" >/dev/null 2>&1 && return 0
  fi
  kill "$pid" 2>/dev/null || true
}

port_open() {
  local port="$1"
  (exec 3<>"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
}

infra_ready() {
  local port
  for port in $INFRA_PORTS; do
    port_open "$port" || return 1
  done
  return 0
}

run_compose() {
  local file="$ROOT/docker-compose.yml"
  [ -f "$file" ] || die "找不到 docker-compose.yml：$file"
  if docker compose version >/dev/null 2>&1; then
    docker compose --project-directory "$ROOT" -f "$file" "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose --project-directory "$ROOT" -f "$file" "$@"
  else
    die "未找到 docker compose，无法自动启动依赖服务。"
  fi
}

start_docker_desktop() {
  if docker desktop start >/dev/null 2>&1; then
    return 0
  fi
  local exe
  for exe in \
    "${LOCALAPPDATA:-}/Programs/DockerDesktop/Docker Desktop.exe" \
    "/c/Program Files/Docker/Docker/Docker Desktop.exe"; do
    [ -n "$exe" ] && [ -x "$exe" ] || continue
    powershell -NoProfile -Command "Start-Process -FilePath '$exe'" \
      >/dev/null 2>&1 && return 0
  done
  return 1
}

wait_for_docker() {
  local i=0
  while [ "$i" -lt "$INFRA_TIMEOUT" ]; do
    docker info >/dev/null 2>&1 && return 0
    sleep 1
    i=$((i + 1))
  done
  return 1
}

wait_for_infra() {
  local i=0
  while [ "$i" -lt "$INFRA_TIMEOUT" ]; do
    infra_ready && return 0
    sleep 1
    i=$((i + 1))
  done
  return 1
}

ensure_infra() {
  if [ "${SKIP_INFRA_CHECK:-0}" = "1" ]; then
    log "SKIP_INFRA_CHECK=1，跳过依赖服务检查"
    return 0
  fi
  if infra_ready; then
    log "依赖服务已就绪（端口 $INFRA_PORTS）"
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    die "依赖服务未就绪（端口 $INFRA_PORTS），且未找到 docker。请手动启动 MySQL/Milvus 后重试，或设置 SKIP_INFRA_CHECK=1 跳过检查。"
  fi

  log "依赖服务未就绪（端口 $INFRA_PORTS），尝试启动"
  if ! docker info >/dev/null 2>&1; then
    log "Docker 守护进程未运行，尝试启动 Docker Desktop"
    start_docker_desktop || log "Docker Desktop 启动命令失败，继续等待守护进程"
    if ! wait_for_docker; then
      die "Docker 守护进程在 ${INFRA_TIMEOUT}s 内未就绪。请手动打开 Docker Desktop 后重试。"
    fi
  fi

  run_compose up -d \
    || die "docker compose up -d 失败。请检查 Docker 状态与 docker-compose.yml。"

  if ! wait_for_infra; then
    die "等待 ${INFRA_TIMEOUT}s 后依赖服务仍不可用（端口 $INFRA_PORTS）。请运行：
  docker compose ps
  docker compose logs --tail=50 mysql standalone"
  fi
  log "依赖服务已就绪（端口 $INFRA_PORTS）"
}

run_server() {
  local branch="$1"

  cd "$WT" || die "无法进入 worktree：$WT"
  log "启动 $branch"
  log "worktree: $WT"
  log "python: $PY"
  if [ "$PY_SOURCE" = "root-fallback" ]; then
    log "提示：该 worktree 没有独立 .venv，已回退到仓库根的虚拟环境"
  fi
  [ -f run.py ] || die "worktree 中缺少 run.py：$WT"
  "$PY" -c 'import uvicorn' >/dev/null 2>&1 \
    || die "当前 Python 无法导入 uvicorn：$PY"

  ensure_infra || exit 1

  release_port

  "$PY" run.py &
  SERVER_PID=$!
  trap 'kill_tree "$SERVER_PID"; exit 130' INT TERM

  if wait_for_http; then
    open_browser
  fi
  wait "$SERVER_PID"
}

main() {
  set -euo pipefail

  local mode="run"
  local arg
  for arg in "$@"; do
    case "$arg" in
      --print-target) mode="print" ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        printf '[start] 未知参数: %s\n' "$arg" >&2
        usage >&2
        return 2
        ;;
    esac
  done

  find_repo_root

  local branch
  branch="$(collect_branches | pick_latest_branch)" \
    || die "没有找到 feature/chNN-* 章节分支"

  resolve_worktree "$branch"
  select_python

  if [ "$mode" = "print" ]; then
    printf 'branch=%s\nworktree=%s\npython=%s\nvenv_source=%s\n' \
      "$branch" "$WT" "$PY" "$PY_SOURCE"
    return 0
  fi

  run_server "$branch"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
