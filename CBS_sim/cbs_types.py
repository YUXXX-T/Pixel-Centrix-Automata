"""
cbs_types.py — CBS仿真基础数据类型

Constraint  : 单个约束（机器人在某时刻不能在某位置）
CTNode      : 约束树节点（高层CBS使用）
Agent       : 机器人代理（起点、终点、已规划路径）
Task        : 搬运任务（pod位置 → 工作站位置）
Conflict    : 两个机器人之间的冲突（顶点冲突 or 边冲突）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple


# ---------------------------------------------------------------------------
# 基础位置类型
# ---------------------------------------------------------------------------
Pos = tuple[int, int]          # (row, col)
TimedPos = tuple[int, int, int]  # (row, col, time)


# ---------------------------------------------------------------------------
# 约束
# ---------------------------------------------------------------------------
class Constraint(NamedTuple):
    """
    单步约束：agent_id 在 timestep 时刻不能占据 pos。
    如果是边约束则还记录 prev_pos（从 prev_pos→pos 这条边的移动被禁止）。
    """
    agent_id: int
    pos: Pos
    timestep: int
    prev_pos: Pos | None = None   # None = 顶点约束; not None = 边约束

    def is_vertex(self) -> bool:
        return self.prev_pos is None

    def is_edge(self) -> bool:
        return self.prev_pos is not None


# ---------------------------------------------------------------------------
# 冲突
# ---------------------------------------------------------------------------
class Conflict(NamedTuple):
    """两个agent之间检测到的冲突。"""
    type: str          # "vertex" 或 "edge"
    agent_a: int
    agent_b: int
    pos: Pos           # 冲突位置（顶点冲突）或目标位置（边冲突）
    prev_pos_a: Pos | None  # 边冲突：agent_a 的出发位置
    prev_pos_b: Pos | None  # 边冲突：agent_b 的出发位置
    timestep: int      # 发生冲突的时间步


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------
@dataclass
class Task:
    task_id: int
    pod_pos: Pos        # Pod 所在位置
    station_pos: Pos    # 目标工作站位置
    station_id: int     # 工作站编号


# ---------------------------------------------------------------------------
# 代理（机器人）
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    agent_id: int
    start: Pos
    task: Task | None = None

    # One-shot 任务分配后填充：
    # path[t] = 在时刻 t 的位置（包含取货+送货+等待+回返四段拼接）
    path: list[Pos] = field(default_factory=list)

    # 完整三段路径的分段索引（便于可视化状态显示）
    fetch_end_t: int = 0     # path[fetch_end_t] = pod_pos（到达 pod）
    deliver_end_t: int = 0   # path[deliver_end_t] = station_pos（到达工作站）
    wait_end_t: int = 0      # 相当于 deliver_end_t + WAIT_TICKS——等待结束
    return_end_t: int = 0    # path[return_end_t] = pod_pos（放回）

    @property
    def goal(self) -> Pos | None:
        """final destination = pod 原始位置"""
        return self.task.pod_pos if self.task else None

    def phase_at(self, t: int) -> str:
        """返回 t 时刻所处任务阶段。"""
        if self.task is None:
            return "IDLE"
        if t <= self.fetch_end_t:
            return "FETCH"
        if t <= self.deliver_end_t:
            return "DELIVER"
        if t <= self.wait_end_t:
            return "WAIT"
        if t <= self.return_end_t:
            return "RETURN"
        return "DONE"


# ---------------------------------------------------------------------------
# CBS 约束树节点
# ---------------------------------------------------------------------------
@dataclass
class CTNode:
    constraints: list[Constraint] = field(default_factory=list)
    paths: dict[int, list[Pos]] = field(default_factory=dict)  # agent_id → path
    cost: int = 0             # sum of path lengths (SIC)

    # 用于优先队列比较
    def __lt__(self, other: CTNode) -> bool:
        return self.cost < other.cost
