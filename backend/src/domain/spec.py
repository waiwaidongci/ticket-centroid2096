"""栖桥剧场客服来信分流验收配置。"""


QUEUES = {
    "票务退改": "票务组核对订单、适用退改规则及退款状态。",
    "无障碍服务": "观众服务组确认轮椅席、无障碍入口和陪同安排。",
    "失物招领": "前厅组按场次、座区和物品描述核对失物登记。",
}
MANUAL_QUEUE = "人工分诊"
THEATERS = ("大剧场", "小剧场")

# 预计件数运行参数：维度、维度名称、组别（归口顺序与 QUEUES 一致）。
_GROUP_SLUGS = {"大剧场": "grand", "小剧场": "small", "票务退改": "ticket", "无障碍服务": "access", "失物招领": "lost"}
FORECAST_DIMENSIONS = (
    ("theater", "剧场", THEATERS),
    ("queue", "归口", tuple(QUEUES)),
)
_FORECAST_DEFAULTS = {"大剧场": 55, "小剧场": 45, "票务退改": 50, "无障碍服务": 25, "失物招领": 25}


def forecast_key(dimension, group):
    """预计件数参数键，如 forecast_theater_grand。"""
    return f"forecast_{dimension}_{_GROUP_SLUGS[group]}"


def _forecast_params():
    return [
        {"key": forecast_key(dimension, group), "label": f"{group}预计件数", "type": "int",
         "required": True, "min": 0, "max": 5000, "default": _FORECAST_DEFAULTS[group],
         "help": "演出取消后该组别的预计来信件数；剧场与归口两侧合计需一致才能配平。"}
        for dimension, _label, groups in FORECAST_DIMENSIONS for group in groups
    ]


SPEC = {
    "title": "剧场客服来信分流验收",
    "short_name": "ticket-centroid",
    "subtitle": "用留出件比较来信分流与多数类基线，核对各队列负载和人工分诊量。",
    "organization": "栖桥剧场观众服务中心，四名客服为大小两个剧场处理票务、无障碍服务和失物招领来信。",
    "workflow": (
        "客服从共享邮箱导出去标识的来信，主管复核归口并冻结训练批和留出批。"
        "运营专员用训练批建立各队列的文本质心，在留出批核对错投及人工分诊，"
        "再把逐件验收明细交给票务组、观众服务组和前厅组定排班。"
    ),
    "pains": [
        "演出取消后只有各剧场和各归口的预计件数，需要知道现有分流比基线少分对或多分对多少。",
        "演出取消后退改来信占比上升，总体命中率容易掩盖失物和无障碍请求的错投。",
        "历史邮件和验收批混用时，当天才出现的新词会提前影响词表和权重。",
        "短句或多诉求邮件容易错投，一线需要知道哪些来信应转人工分诊。",
    ],
    "steps": [
        "11 月 20 日 09:00，从共享邮箱整理「栖桥剧场来信分流_20261120.csv」，保留各件的归口、剧场和固定批次。",
        "导入后核对 18 封来信，其中 11 封训练件、7 封留出件。",
        "使用完整训练词表、0.12 最低相似度和默认预计件数运行验收。",
        "在分流结果中比较宏 F1 与训练多数队列的基线，并查看自动分流覆盖率。",
        "逐件检查 QQ-TEST-07 的新词来信和人工分诊原因，按剧场×归口核对预计件数的逐组对账。",
        "将最低相似度调到 0.30，沿用同一批次重跑，比较覆盖率和错投。",
        "主管在验收复核页记录放行或复测决定，在交付页导出逐件分流明细。",
    ],
    "methodology": [
        "CSV 明确标记 train 或 test。训练件拟合词表、逆文档频率、各标签质心和多数类。留出件只参与预测与验收计分。",
        "连续汉字取相邻双字词，单字保留。英文字母和数字按连续串取词并转小写。仅保留训练中出现的词，限制词量时按训练文档频率降序、同频按词序取前 K 个。",
        "以词频乘以 1 + ln((1 + 训练件数)/(1 + 训练文档频率)) 并做 L2 归一化。各类取归一化训练向量的均值，与留出件向量计算余弦相似度。",
        "无命中词、最高分并列或低于阈值时转人工。最高余弦分衡量接近程度。训练多数类并列时按队列名称排序取首项。",
        "两种策略共用训练标签与留出标签的并集计算宏 F1，零分母计零。人工分诊计入真实队列的漏分，覆盖率按自动分流量除以留出总量计算。",
        "剧场只用于分对差对账，不进入词表、质心与预测；留出件须补录剧场，沿用原预测。",
        "预计件数只分给有留出样本的剧场×归口组合，整数分配同时满足剧场与归口两侧合计；按各组合两种命中率差求分对差上下界并列出可达到分配，凑不齐时列出卡住的组。",
        "中断后恢复沿用开跑时保存的预测与预计件数快照；导出发布采用暂存后替换，发布中断保留旧版。",
    ],
    "sample_name": "栖桥剧场来信分流_20261120.csv",
    "default_dataset": "栖桥剧场 11 月客服来信验收批",
    "entity": "客服来信",
    "run": "分流验收",
    "decision": "验收决定",
    "action": "提交验收决定",
    "modes": ["放行试运行", "调整阈值后复测", "补齐标注后复测"],
    "pages": ["值班总览", "来信明细", "分流验收", "分流结果", "验收复核", "分流交付"],
    "fields": [
        {"key": "ticket_id", "label": "来信编号", "type": "string", "required": True, "max_length": 80},
        {"key": "text", "label": "去标识来信", "type": "string", "required": True, "max_length": 1200},
        {"key": "label", "label": "人工归口", "type": "select", "required": True, "options": list(QUEUES)},
        {"key": "theater", "label": "剧场", "type": "select", "required": False, "options": list(THEATERS)},
        {"key": "split", "label": "固定批次", "type": "select", "required": True, "options": ["train", "test"]},
    ],
    "params": [
        {"key": "max_terms", "label": "词表上限", "type": "int", "required": True, "min": 0, "max": 10000, "default": 0,
         "help": "填 0 使用完整训练词表。其他值仅保留训练批里文档频率最高的若干词。"},
        {"key": "min_similarity", "label": "自动分流最低相似度", "type": "number", "required": True,
         "min": 0, "max": 1, "default": 0.12, "help": "最高余弦分低于此值时进入人工分诊。"},
        *_forecast_params(),
    ],
    "fixture": {"min_similarity": 0.30},
    "axes": "分流量图横轴为目标队列，纵轴为留出来信条数，包含人工分诊。",
}
