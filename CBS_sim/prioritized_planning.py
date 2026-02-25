"""
prioritized_planning.py — 基于优先级的多机器人路径规划（Prioritized Planning）

与 CBS 的关系：
  - CBS（Conflict-Based Search）是最优算法，但在 agent 数量多时计算量大
  - 优先规划（Prioritized Planning）是 CBS 的快速近似方案：
    按优先级顺序逐一规划，已规划的 agent 的路径作为动态障碍注入约束

优先级策略：path length（取货+送货距离）短的 agent 优先级高（先规划）

输出：与 CBS 相同格式的 paths dict，可直接用于可视化
"""

from __future__ import annotations
from cbs_types import Pos, Constraint, Agent
from low_level import plan_full_path, MAX_T as _GLOBAL_MAX_T

import math


def _path_duration_constraint(
    path: list[Pos],
    agent_id: int,
) -> list[Constraint]:
    """
    将已完成规划的 agent 路径转换为顶点约束 + 边约束，
    供后续 agent 规划时避开。

    重要：路径结束后 agent 停在最终位置（DONE 状态），
    必须为 t >= len(path) 的所有时刻也加约束（驻留约束），
    否则后续规划的 agent 会直接"穿越"已完成的 agent。
    """
    constraints: list[Constraint] = []
    for t, pos in enumerate(path):
        # 顶点约束
        constraints.append(Constraint(
            agent_id=agent_id,
            pos=pos,
            timestep=t,
        ))
        # 边约束：换位冲突
        if t + 1 < len(path):
            next_pos = path[t + 1]
            if next_pos != pos:
                constraints.append(Constraint(
                    agent_id=agent_id,
                    pos=pos,
                    timestep=t + 1,
                    prev_pos=next_pos,
                ))

    # ── 驻留约束（Stay-at-Goal）──────────────────────────────────
    # 路径结束后 agent 一直停在 path[-1]（pod 原始位置），
    # 必须禁止其他 agent 在 t >= len(path) 时占用该格。
    final_pos = path[-1]
    for t in range(len(path), _GLOBAL_MAX_T):
        constraints.append(Constraint(
            agent_id=agent_id,
            pos=final_pos,
            timestep=t,
        ))

    return constraints


def prioritized_plan(
    agents: list[Agent],
    rows: int,
    cols: int,
    obstacles: set[Pos],
    priority_order: list[int] | None = None,
) -> dict[int, list[Pos]] | None:
    """
    按优先级顺序逐一规划路径。

    Args:
        agents: Agent 列表（已完成任务分配）
        rows, cols: 地图尺寸
        obstacles: 固定障碍集合
        priority_order: agent_id 排列（优先级从高到低）；
                       None = 按任务路径长度（短→高优先）排序

    Returns:
        paths: {agent_id: list[Pos]}，失败返回 None
    """
    from low_level import manhattan

    if priority_order is None:
        # 按 start→pod + pod→station 曼哈顿距离排序（短的先规划）
        def task_dist(a: Agent) -> float:
            if a.task is None:
                return 0.0
            return (manhattan(a.start, a.task.pod_pos)
                    + manhattan(a.task.pod_pos, a.task.station_pos))
        priority_order = sorted(
            [a.agent_id for a in agents],
            key=lambda aid: task_dist(next(a for a in agents if a.agent_id == aid))
        )

    agent_map = {a.agent_id: a for a in agents}

    # 全局约束集合（累积所有已规划 agent 的路径约束）
    all_constraints: list[Constraint] = []

    paths: dict[int, list[Pos]] = {}

    for rank, aid in enumerate(priority_order):
        a = agent_map[aid]
        print(f"  [PP] Planning Agent#{aid} (priority rank {rank+1}/{len(priority_order)})...")

        if a.task is None:
            paths[aid] = [a.start]
            continue

        # 将全局约束中的 agent_id 替换为当前 agent_id（Space-Time A* 按 agent_id 过滤）
        my_constraints = [
            Constraint(
                agent_id=aid,
                pos=c.pos,
                timestep=c.timestep,
                prev_pos=c.prev_pos,
            )
            for c in all_constraints
        ]

        result = plan_full_path(
            start=a.start,
            pod_pos=a.task.pod_pos,
            station_pos=a.task.station_pos,
            rows=rows,
            cols=cols,
            obstacles=obstacles,
            constraints=my_constraints,
        )

        if result is None:
            print(f"  [PP] Agent#{aid}: NO PATH FOUND!")
            return None

        path, fetch_end_t, deliver_end_t, wait_end_t, return_end_t = result
        a.path          = path
        a.fetch_end_t   = fetch_end_t
        a.deliver_end_t = deliver_end_t
        a.wait_end_t    = wait_end_t
        a.return_end_t  = return_end_t
        paths[aid]      = path

        # 将此 agent 的路径加入全局约束
        path_constraints = _path_duration_constraint(path, aid)
        all_constraints.extend(path_constraints)

        print(
            f"  [PP] Agent#{aid}: {len(path)} steps  "
            f"fetch_end={fetch_end_t}  deliver_end={deliver_end_t}"
        )

    return paths
