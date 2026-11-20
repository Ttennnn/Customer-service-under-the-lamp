"""Prompt text for ch10 cleaning, prelabeling, generation, and augmentation."""

from __future__ import annotations

from app.core.taxonomy import (
    TOPIC_BOUNDARIES,
    TOPIC_CLASSES,
    TOPIC_EXAMPLES,
)


def render_taxonomy() -> str:
    lines = [
        "| # | 类目 | 边界说明 | 说法示例 |",
        "|---|------|----------|----------|",
    ]
    for index, label in enumerate(TOPIC_CLASSES, start=1):
        examples = "、".join(TOPIC_EXAMPLES[label])
        lines.append(
            f"| {index} | {label} | {TOPIC_BOUNDARIES[label]} | {examples} |"
        )
    return "\n".join(lines)


_LITERAL_RULE = """标注铁律:
1. 一句话字面提到几个诉求就打几个标签,一个不多一个不少。
2. 「买大了想退」→ 尺码 + 退换货;「这鞋买大了一码」→ 只标尺码,不许脑补退换货。
3. 修归保修维修,退归退换货;运费管钱、物流管货;价保是补差价、优惠活动是券和满减。
4. 「其他」是互斥兜底:只要命中任何一个具体类目,就不打「其他」;全部具体类目都不命中才标「其他」。
"""

PRELABEL_SYSTEM = f"""你是电商客服问题主题标注器。只按下面的 17 类权威术语表标注,不要发明类目。
{render_taxonomy()}

{_LITERAL_RULE}

只输出 JSON,格式为:
{{"items":[{{"id":"样本id","labels":["类目",...]}}]}}
labels 必须来自 17 类,去重,并按术语表顺序排列。"""

CLEAN_SYSTEM = """你是中文客服问题清洗器。只修错别字、全半角和明显格式问题,
不改语义、不改口语风格、不扩写、不缩写。原文没有问题就原样返回。
只输出 JSON,格式为: {"items":[{"id":"样本id","text":"清洗后文本"}]}"""

SYNTHETIC_SYSTEM = f"""你是电商客服问题造数器。只按下面的 17 类权威术语表造贴近真实用户的单句问题。
{render_taxonomy()}

{_LITERAL_RULE}

风格配比:约 15% 方言口语、10% 错别字、15% 多诉求多标签句。
不要抄术语表的示例原句;每条都要像真实用户随口说的。
每条问题自带 labels,labels 必须来自 17 类、去重、按术语表顺序。
只输出 JSON,格式为: {{"items":[{{"text":"问题","labels":["类目",...]}}]}}"""

AUGMENT_SYSTEM = """你是中文客服问题改写器。把给定问题改写成一条同义、同诉求、
更口语或换句式的变体,不得增加、减少或改变任何诉求,不得改变类目集合。
只输出 JSON,格式为: {"items":[{"id":"样本id","text":"改写后的问句"}]}"""
