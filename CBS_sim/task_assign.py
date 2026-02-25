"""
task_assign.py — One-Shot 任务分配

使用匈牙利算法（scipy.optimize.linear_sum_assignment）将 N 个机器人
与 N 个 pod搬运任务做最优最小代价匹配。

代价矩阵：cost[i][j] = 机器人 i 取货 pod j 的曼哈顿距离
"""

from __future__ import annotations
from cbs_types import Pos, Task, Agent
import numpy as np


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def assign_tasks(
    agents: list[Agent],
    tasks: list[Task],
) -> None:
    """
    将 tasks 分配给 agents（in-place 修改 agent.task）。

    如果 len(agents) > len(tasks)，多余的 agent 不分配任务。
    如果 len(agents) < len(tasks)，只分配 len(agents) 个任务。
    """
    n_agents = len(agents)
    n_tasks  = len(tasks)

    if n_agents == 0 or n_tasks == 0:
        return

    # 构建代价矩阵
    n = min(n_agents, n_tasks)
    cost = np.zeros((n_agents, n_tasks), dtype=float)
    for i, agent in enumerate(agents):
        for j, task in enumerate(tasks):
            cost[i, j] = manhattan(agent.start, task.pod_pos)

    # 匈牙利算法
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
    except ImportError:
        # Fallback：贪心分配（scipy 未安装时）
        assigned_tasks: set[int] = set()
        assigned_agents: set[int] = set()
        row_ind, col_ind = _greedy_assign(cost, n_agents, n_tasks)

    for i, j in zip(row_ind, col_ind):
        agents[i].task = tasks[j]

    # 打印分配结果
    print("[TaskAssign] Assignment result:")
    for i, j in zip(row_ind, col_ind):
        a = agents[i]
        t = tasks[j]
        print(
            f"  Agent#{a.agent_id}@{a.start} → "
            f"pod@{t.pod_pos} → station#{t.station_id}@{t.station_pos}  "
            f"(cost={cost[i,j]:.0f})"
        )


def _greedy_assign(
    cost: np.ndarray,
    n_agents: int,
    n_tasks: int,
) -> tuple[list[int], list[int]]:
    """贪心分配（代替匈牙利算法的 fallback）。"""
    assigned_agents: set[int] = set()
    assigned_tasks:  set[int] = set()
    row_ind: list[int] = []
    col_ind: list[int] = []

    while len(row_ind) < min(n_agents, n_tasks):
        best_val = float("inf")
        best_i = best_j = -1
        for i in range(n_agents):
            if i in assigned_agents:
                continue
            for j in range(n_tasks):
                if j in assigned_tasks:
                    continue
                if cost[i, j] < best_val:
                    best_val = cost[i, j]
                    best_i, best_j = i, j

        if best_i == -1:
            break
        row_ind.append(best_i)
        col_ind.append(best_j)
        assigned_agents.add(best_i)
        assigned_tasks.add(best_j)

    return row_ind, col_ind
