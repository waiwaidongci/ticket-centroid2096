"""按剧场与归口的预计件数，求分对差上下界的整数分配。

预计件数只分给有留出样本的剧场×归口组合，整数件数同时满足两侧合计。
在二分网络上做逐最短路增广（费用先按组合留出量通分为整数），
最小费用对应下界、最大费用对应上界；配不齐时按残量说明卡在哪些组。
"""

from fractions import Fraction
from math import gcd


def _lcm(values):
    result = 1
    for value in values:
        result = result * value // gcd(result, value)
    return result


class _Network:
    """小型残量网络，边为 [终点, 剩余容量, 单位费用, 反向边下标]。"""

    def __init__(self, size):
        self.graph = [[] for _ in range(size)]

    def add(self, source, target, capacity, cost):
        self.graph[source].append([target, capacity, cost, len(self.graph[target])])
        self.graph[target].append([source, 0, -cost, len(self.graph[source]) - 1])
        return len(self.graph[source]) - 1

    def min_cost_flow(self, source, sink, goal):
        """沿当前费用最小路径增广至 goal 或断流，返回（流量, 费用）。节点与边按固定顺序松弛，结果确定。"""
        nodes = len(self.graph)
        flow = cost = 0
        while flow < goal:
            dist = [None] * nodes
            dist[source] = 0
            prev = [None] * nodes
            for _ in range(nodes - 1):
                relaxed = False
                for node in range(nodes):
                    if dist[node] is None:
                        continue
                    for index, edge in enumerate(self.graph[node]):
                        target, capacity, weight, _reverse = edge
                        if capacity and (dist[target] is None or dist[target] > dist[node] + weight):
                            dist[target] = dist[node] + weight
                            prev[target] = (node, index)
                            relaxed = True
                if not relaxed:
                    break
            if prev[sink] is None:
                break
            push = goal - flow
            node = sink
            while node != source:
                origin, index = prev[node]
                push = min(push, self.graph[origin][index][1])
                node = origin
            node = sink
            while node != source:
                origin, index = prev[node]
                edge = self.graph[origin][index]
                edge[1] -= push
                self.graph[node][edge[3]][1] += push
                cost += push * edge[2]
                node = origin
            flow += push
        return flow, cost


def bounds(combos, theater_margins, queue_margins):
    """求满足两侧预计件数的整数分配及分对差上下界。

    combos 为 [{"theater", "queue", "support", "model_hits", "baseline_hits"}]，
    两个 margins 为 {组名: 预计件数}（顺序即对账顺序）。
    返回 {"feasible", "theater_total", "queue_total", "lower", "upper", "blocking"}；
    lower/upper 为 {"cells": {(剧场, 归口): 件数}, "total": Fraction}，不可行时为 None，
    blocking 逐组列出配不齐的剧场或归口及缺口。
    """
    theaters = list(theater_margins)
    queues = list(queue_margins)
    allowed = {}
    for combo in combos:
        if combo["support"] > 0 and combo["theater"] in theater_margins and combo["queue"] in queue_margins:
            allowed[(combo["theater"], combo["queue"])] = combo
    scale = _lcm([combo["support"] for combo in allowed.values()] or [1])
    theater_total = sum(theater_margins.values())
    queue_total = sum(queue_margins.values())
    t_index = {name: index for index, name in enumerate(theaters)}
    q_index = {name: index for index, name in enumerate(queues)}
    source = 0
    sink = 1 + len(theaters) + len(queues)

    def solve(direction):
        net = _Network(sink + 1)
        supply = {name: net.add(source, 1 + t_index[name], theater_margins[name], 0) for name in theaters}
        combo_edges = {}
        for (theater, queue), combo in allowed.items():
            cost = direction * (combo["model_hits"] - combo["baseline_hits"]) * (scale // combo["support"])
            node = 1 + t_index[theater]
            combo_edges[(theater, queue)] = (node, net.add(node, 1 + len(theaters) + q_index[queue], theater_total, cost))
        demand = {
            name: (1 + len(theaters) + q_index[name], net.add(1 + len(theaters) + q_index[name], sink, queue_margins[name], 0))
            for name in queues
        }
        flow, cost = net.min_cost_flow(source, sink, theater_total)
        return net, flow, cost, supply, combo_edges, demand

    net, flow, min_cost, supply, combo_edges, demand = solve(1)
    if flow != theater_total or theater_total != queue_total:
        blocking = []
        for name in theaters:
            residual = net.graph[source][supply[name]][1]
            if residual:
                targets = [queue for theater, queue in allowed if theater == name]
                detail = (f"可分流归口「{'、'.join(targets)}」均已达到预计件数" if targets
                          else "没有留出样本覆盖的组合，无法接收预计件数")
                blocking.append({"dimension": "剧场", "group": name, "shortage": residual, "detail": detail})
        for name in queues:
            node, index = demand[name]
            residual = net.graph[node][index][1]
            if residual:
                sources = [theater for theater, queue in allowed if queue == name]
                detail = (f"可分流剧场「{'、'.join(sources)}」的预计件数已分完" if sources
                          else "没有留出样本覆盖的组合，无法补齐预计件数")
                blocking.append({"dimension": "归口", "group": name, "shortage": residual, "detail": detail})
        return {"feasible": False, "theater_total": theater_total, "queue_total": queue_total,
                "lower": None, "upper": None, "blocking": blocking}

    def cells(network, edges):
        return {key: theater_total - network.graph[node][index][1]
                for key, (node, index) in edges.items()
                if theater_total - network.graph[node][index][1]}

    upper_net, _upper_flow, negated_cost, _supply, upper_edges, _demand = solve(-1)
    return {
        "feasible": True,
        "theater_total": theater_total,
        "queue_total": queue_total,
        "lower": {"cells": cells(net, combo_edges), "total": Fraction(min_cost, scale)},
        "upper": {"cells": cells(upper_net, upper_edges), "total": Fraction(-negated_cost, scale)},
        "blocking": [],
    }
