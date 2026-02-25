"""
low_level.py — 低层规划器：Space-Time A*

给定：
  - 网格地图（行列数、障碍物集合）
  - 单个 agent 的起点和终点
  - 约束列表（顶点约束 + 边约束）

返回：满足所有约束的最短时空路径（list[Pos]），无解则返回 None。

时空 A* 节点：(row, col, time)
  - 邻居：上下左右 + 原地等待
  - 启发函数：曼哈顿距离（可访问）
  - 约束：跳过被约束的节点/边
"""

from __future__ import annotations
import heapq
from typing import Callable
from cbs_types import Pos, Constraint

# 工作站等待 tick 数（与 ../simulator.py 中的 WAIT_TICKS 一致）
WAIT_TICKS: int = 5


# ---------------------------------------------------------------------------
# 网格方向：上下左右 + 原地等待
# ---------------------------------------------------------------------------
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # (dr, dc)

# 最大时间步（防止无限规划）
MAX_T = 512


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def space_time_astar(
    start: Pos,
    goal: Pos,
    rows: int,
    cols: int,
    obstacles: set[Pos],
    constraints: list[Constraint],
    heuristic: Callable[[Pos], int] | None = None,
    start_t: int = 0,
    max_t: int = MAX_T,
) -> list[Pos] | None:
    """
    Space-Time A*：返回从 start（@start_t）到 goal 的时空路径。
    路径格式：list[Pos]，路径长度 = len(path)，path[0]=start。

    目标条件（标准 MAPF CBS 处理）：
      到达 goal 后，agent 必须等待到所有针对 goal 的约束时间步结束后才算完成。
      即：接受到达 goal 的节点当且仅当 t >= max_goal_constraint_t。
    """
    # 预处理约束集合
    vertex_constraints: set[tuple[Pos, int]] = set()
    edge_constraints:   set[tuple[Pos, Pos, int]] = set()

    for c in constraints:
        if c.is_vertex():
            vertex_constraints.add((c.pos, c.timestep))
        else:
            assert c.prev_pos is not None
            edge_constraints.add((c.prev_pos, c.pos, c.timestep))

    # 找出所有针对 goal 的顶点约束中最大时间步（需等待超过该步才能停在 goal）
    goal_constraint_times = [tt for (pos, tt) in vertex_constraints if pos == goal]
    min_goal_t = max(goal_constraint_times) + 1 if goal_constraint_times else 0

    h = heuristic if heuristic else (lambda p: manhattan(p, goal))

    start_node = (start[0], start[1], start_t)
    open_heap: list[tuple[int, int, tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (h(start), 0, start_node))

    g_score: dict[tuple[int, int, int], int] = {start_node: 0}
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_node: None}

    while open_heap:
        f, g, curr = heapq.heappop(open_heap)
        r, c, t = curr

        # 到达目标：必须在 goal 停留到超过所有约束时间步
        if (r, c) == goal and t >= min_goal_t:
            return _reconstruct(came_from, curr)

        if t >= max_t:
            continue

        nt = t + 1
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            npos = (nr, nc)

            if npos in obstacles:
                continue

            if (npos, nt) in vertex_constraints:
                continue

            curr_pos = (r, c)
            if (curr_pos, npos, nt) in edge_constraints:
                continue

            new_g = g + 1
            next_node = (nr, nc, nt)

            if next_node not in g_score or new_g < g_score[next_node]:
                g_score[next_node] = new_g
                f_val = new_g + h(npos)
                heapq.heappush(open_heap, (f_val, new_g, next_node))
                came_from[next_node] = curr

    return None  # 无解


def _reconstruct(
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None],
    end_node: tuple[int, int, int],
) -> list[Pos]:
    path: list[Pos] = []
    node: tuple[int, int, int] | None = end_node
    while node is not None:
        r, c, _ = node
        path.append((r, c))
        node = came_from.get(node)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# 三段路径规划：start → pod → station
# ---------------------------------------------------------------------------
def plan_full_path(
    start: Pos,
    pod_pos: Pos,
    station_pos: Pos,
    rows: int,
    cols: int,
    obstacles: set[Pos],
    constraints: list[Constraint],
    max_t: int = MAX_T,
) -> tuple[list[Pos], int, int, int, int] | None:
    """
    规划完整四段路径（取货 + 送货 + 等待 + 回返）。

    返回：(path, fetch_end_t, deliver_end_t, wait_end_t, return_end_t)

    - path[0..fetch_end_t]          : start → pod_pos
    - path[fetch_end_t..deliver_end_t]: pod_pos → station_pos
    - path[deliver_end_t..wait_end_t] : 在 station_pos 等待 WAIT_TICKS 步
    - path[wait_end_t..return_end_t]  : station_pos → pod_pos（放回）
    """
    # --- 段 1：start → pod_pos ---
    seg1 = space_time_astar(
        start=start, goal=pod_pos,
        rows=rows, cols=cols, obstacles=obstacles,
        constraints=constraints, start_t=0, max_t=max_t,
    )
    if seg1 is None:
        return None
    fetch_end_t = len(seg1) - 1

    # --- 段 2：pod_pos → station_pos ---
    seg2 = space_time_astar(
        start=pod_pos, goal=station_pos,
        rows=rows, cols=cols, obstacles=obstacles,
        constraints=constraints, start_t=fetch_end_t, max_t=max_t,
    )
    if seg2 is None:
        return None
    deliver_end_t = fetch_end_t + len(seg2) - 1

    # --- 段 3：在 station_pos 等待 WAIT_TICKS 步 ---
    wait_end_t = deliver_end_t + WAIT_TICKS
    wait_segment = [station_pos] * WAIT_TICKS  # 不包括 deliver_end_t 本身

    # --- 段 4：station_pos → pod_pos（回返放回）---
    seg4 = space_time_astar(
        start=station_pos, goal=pod_pos,
        rows=rows, cols=cols, obstacles=obstacles,
        constraints=constraints, start_t=wait_end_t, max_t=max_t,
    )
    if seg4 is None:
        return None
    return_end_t = wait_end_t + len(seg4) - 1

    # 拼接：seg1 + seg2[1:] + wait_segment + seg4[1:]
    full_path = seg1 + seg2[1:] + wait_segment + seg4[1:]

    return full_path, fetch_end_t, deliver_end_t, wait_end_t, return_end_t
