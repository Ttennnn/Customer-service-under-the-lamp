# 快捷启动脚本设计

日期：2026-11-23
状态：已确认（2026-11-23）

## 意图

一条命令（或双击）启动本项目当前最新的章节服务，不再需要记 worktree 路径、
手动切分支或先清理旧进程。默认固定使用 8000 端口与 `python run.py` 的启动方式。

## 已确认的选择

- 目标分支：自动选择最新章节，而不是写死 ch10。
- 交付形式：同时提供 `start.bat`（可双击）与 `start.sh`（Git Bash/终端）。

## 范围

- 新增 `start.sh`：全部逻辑所在，POSIX sh/bash 实现。
- 新增 `start.bat`：定位 Git Bash 并调用 `start.sh`，失败时 `pause` 保留窗口。
- 两个文件放在仓库根目录（`D:\AItest`），不修改现有应用代码。

## 行为

1. **定位仓库根**：以脚本所在目录为准，验证 `git rev-parse --show-toplevel` 可用。
2. **选择分支**：枚举本地 `feature/chNN-*` 与 `origin/feature/chNN-*`，按章节号
   数值排序取最大；同章节号优先本地分支。章节号按位数比较（ch2 < ch10），
   不使用字符串排序。
3. **定位 worktree**：
   - 用 `git worktree list --porcelain` 查找该分支已检出的 worktree；
   - 没有则 `git worktree add -b <branch> .worktrees/<slug> <ref>`（远程独有分支
     从 `origin/<branch>` 创建）；新 worktree 缺少 `.env` 时从仓库根复制。
4. **选择 Python**：优先该 worktree 的 `.venv/Scripts/python.exe`，否则回退仓库
   根的 `.venv/Scripts/python.exe`；两者都不存在则报错并打印创建虚拟环境的命令。
   回退时提示「该 worktree 未独立创建 venv」。校验 `import uvicorn`，失败则报错。
5. **依赖预检**（2026-11-24 新增）：检查 `INFRA_PORTS`（默认 `3306 19530`，
   即 MySQL 与 Milvus）。已就绪则直接继续；未就绪时：
   - 未找到 `docker` 则报错并提示手动启动或 `SKIP_INFRA_CHECK=1`；
   - Docker 守护进程未运行则尝试 `docker desktop start`（失败再尝试常见安装路径
     的 `Docker Desktop.exe`），并等待守护进程就绪；
   - 执行 `docker compose --project-directory <仓库根> -f <仓库根>/docker-compose.yml
     up -d`，显式指定仓库根是为了避免在 worktree 下启动出另一套空数据卷；
   - 轮询端口直到就绪，超时（`INFRA_TIMEOUT`，默认 180 秒）报错并提示
     `docker compose ps` / `logs`。
   该检查在端口清理之前执行：依赖启动失败时不会先杀掉正在运行的服务。
6. **释放端口 8000**：用 PowerShell（`Get-NetTCPConnection`）找到监听进程，
   结束它；若其父进程是 python（uvicorn --reload 的父进程）一并结束，避免父进程
   重新拉起子进程。仅处理 8000 端口，不按进程名批量清理。
7. **启动**：`cd` 到目标 worktree 后执行所选 Python 的 `run.py`，保持在当前窗口
   前台输出；`Ctrl+C` 或窗口关闭即停止（trap 负责收尾）。
8. **打开浏览器**：后台轮询 `http://127.0.0.1:8000/`，最长 60 秒；返回
   200–499 即视为可用，然后用 `cmd /c start` 打开默认浏览器。超时不视为启动
   失败，仍保留服务输出并给出提示。
9. **开关**：环境变量 `NO_BROWSER=1` 跳过打开浏览器；`START_TIMEOUT` 可调等待秒数
   （默认 60）；`INFRA_TIMEOUT`、`INFRA_PORTS`、`SKIP_INFRA_CHECK=1` 控制依赖预检。
   端口沿用 `run.py` 的 8000。

## 非目标

- 不执行 `pip install`、不升级依赖、不自动 `git fetch`（离线也能启动；新远程
  分支需用户自行 fetch 后才会被发现）。
- 不切换仓库主检出的分支。
- 不启动 ch10 分类器服务（8110）。
- 依赖预检只负责把 compose 服务拉起来并等待端口，不做 `docker compose down`、
  不重建数据卷。

## 约束与风险

- 脚本依赖 Windows + Git Bash + PowerShell；`.bat` 只做 Git Bash 定位与转调。
- 结束进程只针对 8000 端口占用者及其 python 父进程，误杀范围小；端口被非项目
  进程占用时同样会被结束，输出中说明被结束的 PID。
- 自动创建的新 worktree 位于 `.worktrees/`（已在 `.gitignore`）。
- 依赖预检需要本机有 Docker Desktop；冷启动（Docker + MySQL + Milvus）可能耗时
  30–60 秒，期间端口未就绪会持续等待，不会启动一个连不上数据库的服务。
- 端口就绪是「可用」的代理判断：Milvus 可能在端口打开后还需数秒才完全健康，
  极短时间内提问知识库问题可能得到保守拒答，稍等即可。
- `start.*` 与设计文档暂时都是未跟踪文件，是否提交由用户决定。

## 需要构建者决定的部分

- 分支 slug 的具体生成规则（`/` 替换为 `-`）。
- 轮询间隔（1 秒）与对 3xx/4xx 的判定细节。
- 输出文案的措辞与颜色（不做彩色输出，保持跨终端可读）。
