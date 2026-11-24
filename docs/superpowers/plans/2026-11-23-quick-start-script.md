# 快捷启动脚本 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供 `start.sh` 与 `start.bat`，一条命令启动当前最新章节分支的 dev server（8000 端口），无需手动找 worktree、切分支或清理旧进程。

**Architecture:** 全部逻辑集中在根目录 `start.sh`，分支选择抽成纯函数 `pick_latest_branch`（从 stdin 读取带 `local:`/`remote:` 前缀的分支名），用 shell 测试脚本覆盖；脚本主体用 `BASH_SOURCE` 守卫，只有直接执行时才跑 `main`。`start.bat` 只负责定位 Git Bash 并转调 `start.sh`，失败时 `pause`。

**Tech Stack:** Bash（Git Bash / MSYS2）、PowerShell（端口清理）、Windows batch（双击入口）、`git worktree`、`curl`。

**Spec:** `docs/superpowers/specs/2026-11-23-quick-start-script-design.md`

## Global Constraints

- 平台：Windows 11 + Git Bash（`bash`、`git`、`curl`、`powershell` 可用）。
- 服务固定用 `python run.py` 启动，端口 8000，`run.py` 自带 `--reload`。
- 等待服务就绪默认 60 秒，可用 `START_TIMEOUT` 覆盖；`NO_BROWSER=1` 跳过打开浏览器。
- 不做 `pip install`、不自动 `git fetch`、不切主检出分支、不启动 8110 分类器。
  （2026-11-24 起依赖预检会执行 `docker compose up -d`，见文末变更说明。）
- 端口清理只处理监听 8000 的进程及其 python 父进程，不按进程名批量清理。
- 脚本与测试文件放在仓库根目录，路径固定：`start.sh`、`start.bat`、`tests/quickstart/test_start_sh.sh`。
- 用户未要求提交，本计划不包含 `git commit` 步骤；完成后由用户决定是否提交。

## Review Focus

- 8000 被上一次的 uvicorn 占用（含 reload 父进程）：再次启动必须结束旧监听与 python 父进程，新实例可访问。Task 2 的二次启动验证覆盖。
- 从任意 cwd / 双击启动：仓库根必须按脚本自身位置解析，不能依赖当前目录。Task 1 的 `/tmp` 调用验证与 Task 3 的双击入口覆盖。
- 章节号排序：`ch2 < ch9 < ch10` 按数值比较，同章节号本地分支优先于远程分支。Task 1 的测试覆盖。
- worktree 没有独立 `.venv`：回退根目录 `.venv` 并在 `--print-target` 中暴露 `venv_source=root-fallback`；两边都没有时给出可执行的建 venv 命令。Task 1 的错误路径覆盖。
- 服务未就绪就打开浏览器 / PowerShell 引号写错导致端口清理静默失败：Task 2 要求轮询到 2xx–4xx 才开浏览器，并用二次启动日志证明清理生效。

---

### Task 1: 分支选择与目标解析（`--print-target`）

**Files:**
- Create: `start.sh`
- Create: `tests/quickstart/test_start_sh.sh`

**Interfaces:**
- Consumes: 无。
- Produces:
  - `pick_latest_branch() -> void`：从 stdin 读 `local:<branch>` / `remote:<branch>` 行，向 stdout 输出章节号最大的 `feature/chNN-*`；没有任何匹配时退出码 1、无输出。
  - `find_repo_root() -> void`：设置全局 `ROOT`（POSIX 路径）。
  - `collect_branches() -> void`：向 stdout 输出带 `local:`/`remote:` 前缀的候选分支。
  - `resolve_worktree(branch) -> void`：设置 `WT`；必要时创建 worktree。
  - `select_python() -> void`：设置 `PY`、`PY_SOURCE`（`worktree` 或 `root-fallback`）。
  - CLI：`start.sh [--print-target|-h|--help]`；`--print-target` 输出四行 `branch=`、`worktree=`、`python=`、`venv_source=`。

- [ ] **Step 1: 写失败测试**

创建 `tests/quickstart/test_start_sh.sh`：

```bash
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

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `bash tests/quickstart/test_start_sh.sh`
Expected: FAIL，`source: .../start.sh: No such file or directory`

- [ ] **Step 3: 实现 `pick_latest_branch` 与脚本骨架**

创建 `start.sh`。结构固定为：函数定义在前，末尾用 `if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi` 守卫。`pick_latest_branch` 规则：用 `${name#feature/ch}`、`${num%%-*}` 取章节号，`$((10#$num))` 转十进制；更大章节号替换；同章节号只有当前最优是 remote 且新候选是 local 时才替换。

- [ ] **Step 4: 运行测试确认通过**

Run: `bash tests/quickstart/test_start_sh.sh`
Expected: 5 个 `ok`，最后 `5 passed, 0 failed`

- [ ] **Step 5: 实现目标解析与 `--print-target`**

在 `start.sh` 中补齐：
- `find_repo_root`：`SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`；`git -C "$SCRIPT_DIR" rev-parse --show-toplevel` 失败则报错退出；`ROOT="$SCRIPT_DIR"`。
- `collect_branches`：`git -C "$ROOT" for-each-ref --format='%(refname:short)' refs/heads/feature` 输出前加 `local:`；`refs/remotes/origin/feature` 去掉 `origin/` 前缀后加 `remote:`。
- `resolve_worktree`：`git -C "$ROOT" worktree list --porcelain | awk -v b="refs/heads/$1" '/^worktree /{wt=substr($0,10)} /^branch /{if ($2==b) {print wt; exit}}'`；找不到时在 `$ROOT/.worktrees/<branch 中的 / 换成 ->` 创建：本地分支 `git worktree add <dir> <branch>`，远程独有分支 `git worktree add -b <branch> <dir> origin/<branch>`；新建且缺少 `.env` 时从 `$ROOT/.env` 复制。
- `select_python`：优先 `$WT/.venv/Scripts/python.exe`，不存在则回退 `$ROOT/.venv/Scripts/python.exe` 并令 `PY_SOURCE=root-fallback`；都不存在则报错并打印 `python -m venv .venv` + `.venv/Scripts/python.exe -m pip install -r requirements.txt`。
- `main --print-target`：依次调用上述函数，输出 `branch=...`、`worktree=...`、`python=...`、`venv_source=...`（worktree 路径用 git 输出的 Windows 风格路径原样打印）。

- [ ] **Step 6: 验证语法与目标解析**

Run:
```bash
bash -n start.sh
chmod +x start.sh
./start.sh --print-target
cd /tmp && /d/AItest/start.sh --print-target
```
Expected: 两次输出相同且包含
`branch=feature/ch10-topic-classifier`、
`worktree=D:/AItest/.worktrees/ch05-agent-workflow`、
`venv_source=worktree`，且 `python=` 指向存在的 `.venv/Scripts/python.exe`。

---

### Task 2: 启动服务、端口清理与浏览器

**Files:**
- Modify: `start.sh`（新增运行模式函数，`main` 在非 `--print-target` 时走运行路径）

**Interfaces:**
- Consumes: Task 1 的 `ROOT`、`WT`、`PY`、`PY_SOURCE`。
- Produces:
  - `release_port() -> void`：结束监听 `$PORT` 的进程及其 `python` 父进程。
  - `wait_for_http() -> 0|1`：`$START_TIMEOUT` 秒内等到 HTTP 2xx–4xx 返回 0；服务进程退出或超时返回 1。
  - `open_browser() -> void`：`NO_BROWSER=1` 时跳过，否则 `cmd //c start "" "http://127.0.0.1:$PORT/"`。
  - `run_server() -> void`：前台运行 `run.py`，`INT`/`TERM` 时结束子进程。

- [ ] **Step 1: 实现 `release_port`**

`powershell -NoProfile -Command` 内使用 `Get-NetTCPConnection -LocalPort $PORT -State Listen` 取 `OwningProcess`，对每个 PID：打印 `[start] 结束占用 8000 端口的进程 PID=<pid> (<name>)`，`Stop-Process -Id <pid> -Force`；若父进程名匹配 `^python`，再打印 `[start] 结束其 uvicorn 父进程 PID=<parent>` 并结束父进程。PowerShell 不可用时打印跳过提示并继续。注意 bash 双引号内所有 PowerShell 变量写成 `\$`。

- [ ] **Step 2: 实现 `wait_for_http` 与 `open_browser`**

`wait_for_http`：最多 `START_TIMEOUT` 次、每次间隔 1 秒；先 `kill -0 "$SERVER_PID"` 检查服务是否已退出；用 `curl -s -o /dev/null -m 2 -w '%{http_code}' "http://127.0.0.1:$PORT/"` 取状态码，匹配 `2*|3*|4*` 即成功。超时时打印未就绪提示并返回 1。`open_browser` 用 `case` 判断 `NO_BROWSER=1`。

- [ ] **Step 3: 实现 `run_server` 并接入 `main`**

`main` 运行模式顺序：`release_port` → `cd "$WT"` → 打印 `[start] 启动 <branch>`、worktree、python、`venv_source=root-fallback` 时的提示 → `"$PY" -c 'import uvicorn'` 校验 → `"$PY" run.py &` 保存 `SERVER_PID` → `trap 'kill "$SERVER_PID" 2>/dev/null; exit 130' INT TERM` → `wait_for_http` 成功才 `open_browser` → `wait "$SERVER_PID"` 并以其退出码退出。

- [ ] **Step 4: 语法检查**

Run: `bash -n start.sh`
Expected: 无输出，退出码 0

- [ ] **Step 5: 端到端启动验证**

Run:
```bash
cd /d/AItest
rm -f /tmp/start1.log /tmp/start2.log
NO_BROWSER=1 ./start.sh > /tmp/start1.log 2>&1 &
P1=$!
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -m 2 -w '%{http_code}' http://127.0.0.1:8000/ || true)
  [ "$code" = "200" ] && break
  sleep 1
done
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/
grep '启动 feature/ch10-topic-classifier' /tmp/start1.log
```
Expected: 输出 `200`；日志含启动分支。

- [ ] **Step 6: 二次启动验证端口清理**

接上一步：
```bash
NO_BROWSER=1 ./start.sh > /tmp/start2.log 2>&1 &
P2=$!
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -m 2 -w '%{http_code}' http://127.0.0.1:8000/ || true)
  [ "$code" = "200" ] && break
  sleep 1
done
grep '结束占用 8000 端口的进程' /tmp/start2.log
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/
kill "$P2" "$P1" 2>/dev/null
sleep 2
netstat -ano | grep -E ':8000 .*LISTENING' || echo 'port free'
```
Expected: 日志含结束占用进程行；状态码 `200`；最后输出 `port free`（若 reload 子进程残留则用 Task 2 的 `release_port` 再清一次并修正 trap）。

---

### Task 3: `start.bat` 双击入口

**Files:**
- Create: `start.bat`

**Interfaces:**
- Consumes: `start.sh` 及其 CLI（`--print-target`、`--help`）。
- Produces: 双击入口；退出码透传 `start.sh` 的退出码，非 0 时 `pause`。

- [ ] **Step 1: 创建 `start.bat`**

实现要点：
- 依次检查 `%ProgramFiles%\Git\bin\bash.exe`、`%ProgramFiles(x86)%\Git\bin\bash.exe`、`%LocalAppData%\Programs\Git\bin\bash.exe`；都没有时用 `where bash` 逐个测试 `uname -s` 输出包含 `MINGW` 或 `MSYS`（排除 `C:\Windows\System32\bash.exe` 的 WSL）。
- 找不到 Git Bash：打印中文提示并 `pause`，`exit /b 1`。
- 找到后执行 `"%BASH%" "%~dp0start.sh" %*`，保存 `%ERRORLEVEL%`，非 0 时 `pause`，最后 `exit /b %CODE%`。

- [ ] **Step 2: 验证 bat 转调与参数透传**

Run:
```bash
cd /d/AItest
cmd //c "start.bat --print-target"
cmd //c "start.bat --help"
```
Expected: 第一条输出与 `./start.sh --print-target` 完全一致；第二条打印用法并退出 0。

- [ ] **Step 3: 手动双击验收**

在资源管理器双击 `start.bat`：应弹出控制台 → 清理旧服务 → 启动 ch10 服务 → 默认浏览器自动打开 `http://127.0.0.1:8000/`；关闭窗口或 Ctrl+C 后 8000 不再监听。此项由用户执行确认。

---

## Self-Review

- Spec coverage：分支选择（Task 1）、worktree 自动创建与 `.env` 复制（Task 1）、venv 回退（Task 1）、端口清理（Task 2）、启动/浏览器/超时（Task 2）、bat 双击（Task 3）、非目标（Global Constraints 明确不做）。
- 步骤自检：每个代码步骤给出函数名、文件与关键值；测试步骤给出命令与预期输出。
- 类型/命名一致：`ROOT`、`WT`、`PY`、`PY_SOURCE`、`SERVER_PID`、`pick_latest_branch` 在 Task 1/2 中名称一致。
- Review Focus 的五类风险均落到 Task 1 测试或 Task 2 二次启动验证。
- 规模：计划与设计文档体量相当，未复制脚本正文。

---

## 变更说明（2026-11-24）：依赖预检

原计划约定「不动 docker」。实际使用中发现 Docker Desktop 不自启、compose 服务
无重启策略，导致每次重启机器后 `/chat` 都因 MySQL 3306 连接被拒而返回
「客服服务暂时不可用」，且 `start.sh` 的就绪检查只请求静态页 `/`，无法发现。

经用户确认采用方案 A，补充：

- `docker-compose.yml` 四个服务加 `restart: unless-stopped`。
- Docker Desktop 设置 `AutoStart: true`（登录时启动）。
- `start.sh` 新增 `ensure_infra`：启动前检查 `INFRA_PORTS`（默认 3306/19530），
  必要时启动 Docker Desktop、执行 `docker compose --project-directory <仓库根>
  up -d` 并等待端口，超时给出明确错误；在端口清理之前执行。
- 新开关：`INFRA_TIMEOUT`（默认 180）、`INFRA_PORTS`、`SKIP_INFRA_CHECK=1`。
- 测试：`tests/quickstart/test_start_sh.sh` 新增 6 个依赖预检测试用例（共 13 个）。

