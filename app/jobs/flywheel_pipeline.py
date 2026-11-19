"""飞轮批处理任务:标准化 + 查重 + 待审(Task 6)。

扫描 ``low_confidence_questions`` 中尚未归并的行,逐条调用真实 LLM,
把结果写入 ``review_queue``。导入本模块不会创建 LLM,也不会访问网络;
只有 :func:`main` 读取 settings、构建模型并打开数据库会话。

退出码:
- 0: 报告所有条目均处理成功(可能 created=0, merged=0)
- 1: 至少一条处理失败,或任务异常
"""

from __future__ import annotations

import argparse

from app.config import build_llm, get_settings
from app.db.session import get_session_factory
from app.services.flywheel import process_pending

DEFAULT_LIMIT = 50


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="标准化并查重低置信问题,写入待审队列"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"本轮最多扫描多少条低置信问题（默认 {DEFAULT_LIMIT}）",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help="每轮读取的候选缺口上限（默认 settings.flywheel_candidate_limit）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.limit <= 0:
        print("flywheel: --limit must be a positive integer")
        return 1

    settings = get_settings()
    llm = build_llm(
        settings,
        temperature=0.0,
        max_tokens=settings.flywheel_llm_max_tokens,
        request_timeout=settings.flywheel_timeout_s,
    )

    factory = get_session_factory()
    session = factory()
    try:
        report = process_pending(
            session,
            llm,
            limit=args.limit,
            candidate_limit=args.candidate_limit,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits non-zero
        print(f"flywheel: failed to process pending questions: {exc}")
        return 1
    finally:
        session.close()

    print(
        "flywheel: scanned={scanned} created={created} merged={merged} "
        "failed={failed}".format(
            scanned=report.scanned,
            created=report.created,
            merged=report.merged,
            failed=report.failed,
        )
    )
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
