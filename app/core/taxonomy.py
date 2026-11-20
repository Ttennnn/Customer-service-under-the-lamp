"""The single authoritative 17-class topic taxonomy for ch10.

Every consumer -- data pipelines, prelabel prompts, training label order,
inference, evaluation, APIs, and pages -- imports this module. Never copy the
class list anywhere else.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

TOPIC_CLASSES: tuple[str, ...] = (
    "退换货",
    "物流",
    "尺码",
    "发票",
    "质量问题",
    "运费",
    "优惠活动",
    "价保",
    "支付",
    "订单修改",
    "库存补货",
    "商品信息",
    "保修维修",
    "账号",
    "会员积分",
    "评价",
    "其他",
)

FALLBACK_LABEL = "其他"

TOPIC_BOUNDARIES: dict[str, str] = {
    "退换货": "退货、换货、退款怎么办；修归保修维修，退归这里。",
    "物流": "货走到哪了、什么时候送到；运费的钱事归运费。",
    "尺码": "大小、码数合不合适。",
    "发票": "开票、抬头、报销凭证。",
    "质量问题": "商品本身的毛病。",
    "运费": "运费谁出、运费险理赔；管的是钱，货走到哪了归物流。",
    "优惠活动": "券和活动怎么用、能不能叠。",
    "价保": "买完降价了补不补差价。",
    "支付": "付款环节出的问题。",
    "订单修改": "下单之后改信息、取消订单。",
    "库存补货": "有没有货、什么时候补。",
    "商品信息": "材质、功能、用法。",
    "保修维修": "保修期限、维修换新；修归这里，退归退换货。",
    "账号": "登录、绑定、账号安全。",
    "会员积分": "会员权益、积分怎么用。",
    "评价": "评价、晒单的规则。",
    "其他": "上面都对不上的，先兜底。",
}

TOPIC_EXAMPLES: dict[str, tuple[str, ...]] = {
    "退换货": ("退货", "退款", "退钱", "想退了", "七天无理由还能退不"),
    "物流": ("快递", "发货", "到哪了", "怎么还不动", "海外直邮"),
    "尺码": ("猫窝买大了", "猫别墅尺寸", "项圈偏码", "适合几斤的猫"),
    "发票": ("开发票", "发票抬头开错了", "能开增值税发票吗", "合并开票"),
    "质量问题": ("开胶", "破了个洞", "有瑕疵", "猫砂盆电机坏了"),
    "运费": ("包邮吗", "退货运费谁承担", "运费险怎么赔"),
    "优惠活动": ("优惠券", "满减", "活动价", "能叠加用吗", "双十一有活动吗"),
    "价保": ("刚买就降价了", "能补差价吗", "保价期多久"),
    "支付": ("付不了款", "花呗分期", "扣了两次钱", "货到付款", "数字人民币"),
    "订单修改": ("改地址", "改电话号码", "订单还能取消吗"),
    "库存补货": ("有货吗", "断货了", "什么时候补货", "有现货吗"),
    "商品信息": (
        "什么材质",
        "怎么洗",
        "冻干怎么保存",
        "废砂盒多久倒",
        "猫粮怎么选",
    ),
    "保修维修": ("保修多久", "坏了能修吗", "能换新吗"),
    "账号": ("登录不上", "忘了密码", "换绑手机号", "注销账号"),
    "会员积分": ("积分怎么用", "会员几级", "积分能抵钱吗"),
    "评价": ("评价怎么改", "追评在哪写", "晒单有奖励吗"),
    "其他": ("闲聊", "转人工", "客服几点上班"),
}

LABEL2ID: dict[str, int] = {
    label: index for index, label in enumerate(TOPIC_CLASSES)
}
ID2LABEL: dict[int, str] = {index: label for label, index in LABEL2ID.items()}

TAXONOMY_VERSION: str = hashlib.sha1(
    "|".join(TOPIC_CLASSES).encode("utf-8")
).hexdigest()[:8]


def normalize_labels(labels: Iterable[str]) -> list[str]:
    """Deduplicate labels and order them by the authoritative class order.

    Unknown labels are rejected instead of silently dropped: a typo or an
    out-of-taxonomy label in generated/annotated data is a data bug that must
    be visible before it reaches training.
    """
    seen: set[str] = set()
    for label in labels:
        if label not in LABEL2ID:
            raise ValueError(f"unknown topic label: {label!r}")
        seen.add(label)
    selected = [label for label in TOPIC_CLASSES if label in seen]
    if len(selected) > 1 and FALLBACK_LABEL in selected:
        selected.remove(FALLBACK_LABEL)
    return selected


def labels_from_scores(
    scores: Mapping[str, float],
    threshold: float,
) -> list[str]:
    """Apply a global threshold, with ``其他`` as the exclusive fallback.

    - Labels at or above ``threshold`` are selected.
    - If any specific class is selected, ``其他`` is removed even when its
      score also crossed the line.
    - If nothing crosses the line, the highest-scoring class is returned so a
      prediction is never empty.
    """
    selected = [
        label for label in TOPIC_CLASSES if scores.get(label, 0.0) >= threshold
    ]
    if FALLBACK_LABEL in selected and len(selected) > 1:
        selected.remove(FALLBACK_LABEL)
    if selected:
        return selected
    highest = max(TOPIC_CLASSES, key=lambda label: scores.get(label, 0.0))
    return [highest]
