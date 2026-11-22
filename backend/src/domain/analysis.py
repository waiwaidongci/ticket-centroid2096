"""用固定留出批验收剧场来信的质心分流，并按预计件数对账分对差。

forecast_snapshot 在开跑时固定预测与预计件数，中断恢复后由 summarize 沿用快照汇总，
保证恢复不重算预测、不换预计件数。
"""

import re
from collections import Counter
from math import fsum, isclose, log, sqrt
from unicodedata import normalize

from .allocation import bounds
from .spec import FORECAST_DIMENSIONS, MANUAL_QUEUE, QUEUES, SPEC, THEATERS, forecast_key


_TOKEN_RUNS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9]+")
_THEATER_ORDER = {name: index for index, name in enumerate(THEATERS)}
_QUEUE_ORDER = {name: index for index, name in enumerate(QUEUES)}
_PREDICTION_COLUMNS = (
    ("ticket_id", "留出来信编号"), ("theater", "剧场"), ("actual", "人工归口"),
    ("candidate", "最接近的训练队列"), ("predicted", "分流去向"), ("similarity", "最高余弦分"),
    ("matched_terms", "命中词数"), ("outcome", "分流判定"), ("reason", "判定依据"),
    ("baseline", "多数类去向"),
)
_QUEUE_COLUMNS = (
    ("queue", "队列"), ("training", "训练件数"), ("support", "留出归口量"),
    ("routed", "预计接件量"), ("correct", "正确分流"), ("precision", "精确率"),
    ("recall", "召回率"), ("f1", "F1"), ("baseline_f1", "基线 F1"),
    ("top_terms", "质心代表词"), ("handling", "接件说明"),
)
_COMBO_COLUMNS = (
    ("theater", "剧场"), ("queue", "归口"), ("support", "留出件数"),
    ("model_hits", "分流分对"), ("model_rate", "分流命中率"),
    ("baseline_rate", "基线命中率"), ("diff", "命中率差"),
)
_ALLOC_COLUMNS = (
    ("theater", "剧场"), ("queue", "归口"), ("allocated", "分配件数"),
    ("diff", "命中率差"), ("contribution", "分对差贡献"),
)
_RECON_COLUMNS = (
    ("dimension", "维度"), ("group", "组别"), ("expected", "预计件数"),
    ("upper_allocated", "上界分配"), ("upper_gap", "上界差额"),
    ("lower_allocated", "下界分配"), ("lower_gap", "下界差额"),
)
_BLOCK_COLUMNS = (
    ("dimension", "维度"), ("group", "组别"), ("shortage", "缺口件数"), ("detail", "卡点说明"),
)


def validate_rows(rows):
    """校验 CSV 来信格式和编号唯一性，返回规范化副本。留出件必须补录剧场。"""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("来信明细为空，请按样例导入带归口和批次的记录。")
    tickets = []
    seen = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"第 {index} 条来信格式有误。")
        ticket = {}
        for field in SPEC["fields"]:
            key = field["key"]
            value = row.get(key)
            value = value.strip() if isinstance(value, str) else ""
            if not value:
                if key == "theater" and row.get("split") == "test":
                    raise ValueError(f"第 {index} 条留出来信需补录「剧场」（{'、'.join(field['options'])}）后再验收。")
                if field.get("required", True):
                    raise ValueError(f"第 {index} 条来信的「{field['label']}」缺失或长度超限。")
                ticket[key] = ""
                continue
            if len(value) > field.get("max_length", 300):
                raise ValueError(f"第 {index} 条来信的「{field['label']}」缺失或长度超限。")
            if field["type"] == "select" and value not in field["options"]:
                raise ValueError(f"第 {index} 条来信的「{field['label']}」须为：{'、'.join(field['options'])}。")
            ticket[key] = value
        identifier = normalize("NFKC", ticket["ticket_id"]).casefold()
        if identifier in seen:
            raise ValueError(f"来信编号「{ticket['ticket_id']}」重复，同一编号只能出现于一个批次。")
        seen.add(identifier)
        tickets.append(ticket)
    return tickets


def _parameters(params):
    if not isinstance(params, dict):
        raise ValueError("验收参数格式有误。")
    values = {}
    for field in SPEC["params"]:
        key = field["key"]
        value = params.get(key, field["default"])
        types = (int,) if field["type"] == "int" else (int, float)
        if type(value) not in types or not field["min"] <= value <= field["max"]:
            unit = "整数" if field["type"] == "int" else "数值"
            raise ValueError(f"「{field['label']}」须为 {field['min']} 至 {field['max']} 的{unit}。")
        values[key] = value
    return values


def _counts(text):
    tokens = Counter()
    for run in _TOKEN_RUNS.findall(normalize("NFKC", text).casefold()):
        if run.isascii() or len(run) == 1:
            tokens[run] += 1
        else:
            tokens.update(run[index:index + 2] for index in range(len(run) - 1))
    return tokens


def _unit(vector):
    norm = sqrt(fsum(value * value for value in vector.values()))
    return {term: value / norm for term, value in vector.items()} if norm else {}


def _vector(counts, idf):
    return _unit({term: count * idf[term] for term, count in counts.items() if term in idf})


def _fit(training, label_counts, max_terms):
    documents = [_counts(ticket["text"]) for ticket in training]
    frequency = Counter(term for document in documents for term in document)
    if not frequency:
        raise ValueError("训练批提取不到有效词，请补录含汉字、字母或数字的来信。")
    terms = sorted(frequency, key=lambda term: (-frequency[term], term))
    idf = {
        term: 1 + log((1 + len(training)) / (1 + frequency[term]))
        for term in terms[:max_terms or None]
    }
    sums = {label: Counter() for label in label_counts}
    for ticket, document in zip(training, documents):
        sums[ticket["label"]].update(_vector(document, idf))
    centroids = {
        label: {term: weight / label_counts[label] for term, weight in weights.items()}
        for label, weights in sums.items()
    }
    return idf, centroids


def _prediction(ticket, idf, directions, threshold, majority):
    vector = _vector(_counts(ticket["text"]), idf)
    scores = {
        label: min(1.0, fsum(weight * direction.get(term, 0.0) for term, weight in vector.items()))
        for label, direction in directions.items()
    }
    best = max(scores.values())
    leaders = sorted(
        label for label, score in scores.items() if isclose(score, best, rel_tol=0, abs_tol=1e-12)
    )
    actual = ticket["label"]
    predicted = MANUAL_QUEUE
    candidate = " / ".join(leaders) if best else ""
    if not vector:
        reason = "无训练词表命中"
    elif len(leaders) > 1:
        reason = "最高余弦分并列"
    elif best < threshold and not isclose(best, threshold, rel_tol=0, abs_tol=1e-12):
        reason = "最高余弦分低于阈值"
    else:
        predicted = leaders[0]
        reason = "最高余弦分达到阈值"
    if predicted == MANUAL_QUEUE:
        outcome = "转人工"
    else:
        outcome = "命中" if predicted == actual else "错投"
    return {
        "ticket_id": ticket["ticket_id"], "theater": ticket["theater"], "actual": actual,
        "candidate": candidate, "predicted": predicted, "similarity": round(best, 6),
        "matched_terms": len(vector), "outcome": outcome, "reason": reason, "baseline": majority,
    }


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _table(title, columns, rows):
    return {"title": title, "columns": [{"key": key, "label": label} for key, label in columns], "rows": rows}


def _forecast(params):
    """从运行参数读出两侧预计件数，保持剧场、归口的对账顺序。"""
    return {
        dimension: {group: params[forecast_key(dimension, group)] for group in groups}
        for dimension, _label, groups in FORECAST_DIMENSIONS
    }


def _combo_order(theater, queue):
    return (_THEATER_ORDER.get(theater, 99), theater, _QUEUE_ORDER.get(queue, 99), queue)


def _combos(predictions):
    """按剧场×归口汇总留出样本与两种命中，只保留有留出样本的组合。"""
    groups = {}
    for row in predictions:
        key = (row["theater"], row["actual"])
        combo = groups.setdefault(key, {"theater": key[0], "queue": key[1], "support": 0,
                                        "model_hits": 0, "baseline_hits": 0})
        combo["support"] += 1
        combo["model_hits"] += row["outcome"] == "命中"
        combo["baseline_hits"] += row["baseline"] == row["actual"]
    return sorted(groups.values(), key=lambda combo: _combo_order(combo["theater"], combo["queue"]))


def forecast_snapshot(rows, normalized_params):
    """开跑时计算预测与预计件数快照；中断恢复沿用此快照，不重算预测。"""
    tickets = sorted(validate_rows(rows), key=lambda ticket: ticket["ticket_id"])
    params = _parameters(normalized_params)
    training = [ticket for ticket in tickets if ticket["split"] == "train"]
    holdout = [ticket for ticket in tickets if ticket["split"] == "test"]
    if not training or not holdout:
        raise ValueError("必须同时提供 train 训练件和 test 留出件，批次以导入标记为准。")
    counts = Counter(ticket["label"] for ticket in training)
    if len(counts) < 2:
        raise ValueError("训练批至少需要两个已标注队列，才能验收分流。")
    majority = min(counts, key=lambda label: (-counts[label], label))
    idf, centroids = _fit(training, counts, params["max_terms"])
    directions = {label: _unit(centroid) for label, centroid in centroids.items()}
    predictions = [_prediction(ticket, idf, directions, params["min_similarity"], majority) for ticket in holdout]
    labels = sorted(counts.keys() | {ticket["label"] for ticket in holdout})
    top_terms = {}
    for label in labels:
        centroid = centroids.get(label, {})
        top_terms[label] = "、".join(sorted(centroid, key=lambda term: (-centroid[term], term))[:6])
    return {
        "version": 1,
        "params": params,
        "labels": labels,
        "training_counts": dict(counts),
        "training_total": len(training),
        "vocab_size": len(idf),
        "majority": majority,
        "top_terms": top_terms,
        "empty_centroids": [label for label in labels if not centroids.get(label)],
        "predictions": predictions,
        "combos": _combos(predictions),
        "forecast": _forecast(params),
    }


def _queue_summary(snapshot):
    labels = snapshot["labels"]
    counts = snapshot["training_counts"]
    predictions = snapshot["predictions"]
    majority = snapshot["majority"]
    support = Counter(row["actual"] for row in predictions)
    routed = Counter(row["predicted"] for row in predictions)
    correct = Counter(row["actual"] for row in predictions if row["outcome"] == "命中")
    total = len(predictions)
    rows = []
    model_scores = []
    baseline_scores = []
    for label in labels:
        f1 = _ratio(2 * correct[label], support[label] + routed[label])
        baseline_f1 = _ratio(2 * support[label], support[label] + total) if label == majority else 0.0
        model_scores.append(f1)
        baseline_scores.append(baseline_f1)
        handling = QUEUES[label]
        if label in snapshot["empty_centroids"]:
            handling += " 当前训练词表下无可用质心，请补充训练件或扩大词表。"
        rows.append({
            "queue": label, "training": counts.get(label, 0), "support": support[label], "routed": routed[label],
            "correct": correct[label], "precision": round(_ratio(correct[label], routed[label]), 6),
            "recall": round(_ratio(correct[label], support[label]), 6),
            "f1": round(f1, 6), "baseline_f1": round(baseline_f1, 6),
            "top_terms": snapshot["top_terms"].get(label, ""), "handling": handling,
        })
    rows.append({
        "queue": MANUAL_QUEUE, "training": None, "support": None, "routed": routed[MANUAL_QUEUE],
        "correct": None, "precision": None, "recall": None, "f1": None, "baseline_f1": None,
        "top_terms": "", "handling": "值班客服先确认来件诉求，再转交对应业务组。",
    })
    return rows, fsum(model_scores) / len(labels), fsum(baseline_scores) / len(labels)


def _combo_rows(combos):
    rows = []
    for combo in combos:
        support = combo["support"]
        model_rate = combo["model_hits"] / support
        baseline_rate = combo["baseline_hits"] / support
        rows.append({**combo, "model_rate": round(model_rate, 6), "baseline_rate": round(baseline_rate, 6),
                     "diff": round(model_rate - baseline_rate, 6)})
    return rows


def _allocation_rows(cells, combos_by_key, total):
    rows = []
    for theater, queue in sorted(cells, key=lambda key: _combo_order(*key)):
        combo = combos_by_key[(theater, queue)]
        allocated = cells[(theater, queue)]
        diff = (combo["model_hits"] - combo["baseline_hits"]) / combo["support"]
        rows.append({"theater": theater, "queue": queue, "allocated": allocated,
                     "diff": round(diff, 6), "contribution": round(allocated * diff, 2)})
    rows.append({"theater": "合计", "queue": "—", "allocated": sum(cells.values()),
                 "diff": None, "contribution": round(float(total), 2)})
    return rows


def _reconciliation_rows(forecast, upper_cells, lower_cells, expected_total):
    rows = []
    for dimension, label, _groups in FORECAST_DIMENSIONS:
        for group, expected in forecast[dimension].items():
            slot = 0 if dimension == "theater" else 1
            upper = sum(count for key, count in upper_cells.items() if key[slot] == group)
            lower = sum(count for key, count in lower_cells.items() if key[slot] == group)
            rows.append({"dimension": label, "group": group, "expected": expected,
                         "upper_allocated": upper, "upper_gap": upper - expected,
                         "lower_allocated": lower, "lower_gap": lower - expected})
    rows.append({"dimension": "合计", "group": "—", "expected": expected_total,
                 "upper_allocated": expected_total, "upper_gap": 0,
                 "lower_allocated": expected_total, "lower_gap": 0})
    return rows


def summarize(snapshot):
    """由开跑快照汇总验收结果与分对差对账，纯计算、可重复执行。"""
    params = snapshot["params"]
    predictions = snapshot["predictions"]
    majority = snapshot["majority"]
    queues, macro_f1, baseline_f1 = _queue_summary(snapshot)
    combos = snapshot["combos"]
    forecast = snapshot["forecast"]
    outcome = bounds(combos, forecast["theater"], forecast["queue"])
    feasible = outcome["feasible"]

    outcomes = Counter(row["outcome"] for row in predictions)
    manual = outcomes["转人工"]
    wrong = outcomes["错投"]
    total = len(predictions)
    coverage = (total - manual) / total
    delta = macro_f1 - baseline_f1

    combo_rows = _combo_rows(combos)
    combos_by_key = {(combo["theater"], combo["queue"]): combo for combo in combos}
    metrics = [
        {"label": "分流宏 F1", "value": round(macro_f1, 6), "detail": f"{len(snapshot['labels'])} 个队列等权平均，人工分诊计入漏分"},
        {"label": "多数类宏 F1", "value": round(baseline_f1, 6), "detail": f"训练批多数队列为{majority}（{snapshot['training_counts'].get(majority, 0)} 封）"},
        {"label": "宏 F1 差值", "value": round(delta, 6), "detail": "分流宏 F1 减多数类宏 F1"},
        {"label": "自动分流覆盖率", "value": f"{coverage:.1%}", "detail": f"{total - manual}/{total} 封进入业务队列"},
        {"label": "人工分诊量", "value": manual, "detail": "阈值、并列和未命中词表的逐件原因见明细"},
    ]
    tables = [
        _table("逐件分流明细", _PREDICTION_COLUMNS, predictions),
        _table("队列负载与验收指标", _QUEUE_COLUMNS, queues),
        _table("剧场×归口组合命中率", _COMBO_COLUMNS, combo_rows),
    ]
    sections = [
        {"title": "队列复核与排班", "text": (
            "预计接件量汇总模型去向，人工归口量汇总留出标签。错投会占用接件组的时间，也会让应接件组漏收。"
            "结合逐件依据复核错投和分诊原因，沿用固定留出批比较阈值前后的覆盖率与宏 F1。"
        )},
        {"title": "分对差上下界与逐组对账", "text": (
            "预计件数只分给有留出样本的剧场×归口组合，整数分配同时满足剧场与归口两侧合计。"
            "上界是最有利于现有分流的可达到分配，下界是最不利的可达到分配；命中率差为分流命中率减基线命中率。"
            "页面表格与下载 CSV 内容一致，可按剧场和归口逐组对账。"
        )},
    ]
    if feasible:
        upper_total = outcome["upper"]["total"]
        lower_total = outcome["lower"]["total"]
        upper_value = round(float(upper_total), 2)
        lower_value = round(float(lower_total), 2)
        allocation_note = (
            f"按各组合命中率差对账 {outcome['theater_total']} 件："
            f"分对差上界 {upper_value:+.2f} 件、下界 {lower_value:+.2f} 件。"
        )
        tables += [
            _table("上界可达到分配", _ALLOC_COLUMNS,
                   _allocation_rows(outcome["upper"]["cells"], combos_by_key, upper_total)),
            _table("下界可达到分配", _ALLOC_COLUMNS,
                   _allocation_rows(outcome["lower"]["cells"], combos_by_key, lower_total)),
            _table("逐组对账", _RECON_COLUMNS,
                   _reconciliation_rows(forecast, outcome["upper"]["cells"], outcome["lower"]["cells"],
                                        outcome["theater_total"])),
        ]
        metrics += [
            {"label": "预计总件数", "value": outcome["theater_total"], "detail": "剧场与归口预计合计一致"},
            {"label": "分对差上界", "value": upper_value, "detail": "最有利于现有分流的可达到分配"},
            {"label": "分对差下界", "value": lower_value, "detail": "最不利于现有分流的可达到分配"},
        ]
    else:
        allocation_note = f"预计件数凑不齐：{len(outcome['blocking'])} 个剧场或归口存在缺口，详见卡点明细。"
        tables.append(_table("凑不齐的卡点明细", _BLOCK_COLUMNS, outcome["blocking"]))
        metrics.append({"label": "预计总件数", "value": f"{outcome['theater_total']} / {outcome['queue_total']}",
                        "detail": "剧场合计 / 归口合计；两侧需一致且组合有留出样本"})
        sections.append({"title": "凑不齐的组", "text": (
            "卡点明细列出配不平的剧场或归口及缺口件数。两侧合计不一致时先对齐合计；"
            "组合缺留出样本时调整预计件数，或为留出批补录覆盖更多组合的样本后重跑。"
        )})

    return {
        "state": "warning" if manual or wrong or delta <= 0 or not feasible else "ok",
        "headline": (f"留出 {total} 封，自动分流 {total - manual} 封（错投 {wrong} 封），人工分诊 {manual} 封。"
                     + allocation_note),
        "note": (f"训练 {snapshot['training_total']} 封，使用 {snapshot['vocab_size']} 个训练词，"
                 f"最低相似度 {params['min_similarity']:g}。多数类基线全部投向{majority}。"
                 "剧场只参与对账，不影响预测。"),
        "metrics": metrics,
        "tables": tables,
        "charts": [
            {"title": "留出来信的预计队列接件量", "kind": "bar",
             "series": [{"label": row["queue"], "value": row["routed"]} for row in queues]},
            {"title": "各剧场×归口组合命中率差（分流−基线）", "kind": "bar",
             "series": [{"label": f"{row['theater']}·{row['queue']}", "value": row["diff"]} for row in combo_rows]},
        ],
        "sections": sections,
    }


def analyze(rows, normalized_params):
    """一次性计算：开跑快照加汇总。后台执行分段保存快照，见 services.execution。"""
    return summarize(forecast_snapshot(rows, normalized_params))
