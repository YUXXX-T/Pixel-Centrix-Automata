"""
robot.py — Robot

移动策略（两阶段：reserve → execute_move）：
  FETCH_POD  → 跟随 dim=0 吸引场，爬升（找最大邻居）
  DELIVER    → 跟随 dim=tar_id 代价场，下降（找最小邻居=离工作站最近）
  IDLE       → 不移动

reserve 判断逻辑：
  1. 候选格 = 邻居中 is_available 的格子（未 Occ 且未 Res）
  2. FETCH_POD：选 grad[0] 最大的候选，且必须比当前格更大（保证在爬升）
  3. DELIVER：选 grad[tar_id] 最小的候选，且必须比当前格更小（保证在下降）
  4. 写入 candidate.res = robot_id 完成预约
"""

from __future__ import annotations
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid import Grid


class TaskType(Enum):
    IDLE = auto()
    FETCH_POD = auto()
    DELIVER = auto()


class Robot:
    def __init__(self, robot_id: int, start_row: int, start_col: int) -> None:
        self.robot_id = robot_id
        self.row = start_row
        self.col = start_col
        self.task_type: TaskType = TaskType.IDLE
        self.tar_id: int = 0
        self.carrying_pod: bool = False
        self._next_pos: tuple[int, int] | None = None

    @property
    def pos(self) -> tuple[int, int]:
        return (self.row, self.col)

    def assign_fetch(self, tar_id: int) -> None:
        self.tar_id = tar_id
        self.task_type = TaskType.FETCH_POD

    # ------------------------------------------------------------------
    # 阶段 1：预约
    # ------------------------------------------------------------------
    def reserve(self, grid: "Grid") -> None:
        if self.task_type == TaskType.IDLE:
            self._next_pos = None
            return

        current_cell = grid[self.row, self.col]
        candidates = [c for c in grid.neighbors(self.row, self.col)
                      if c.is_available]

        if not candidates:
            self._next_pos = None
            return

        if self.task_type == TaskType.FETCH_POD:
            # ── 吸引场：爬升 dim=0 ────────────────────────────────────
            dim = 0
            best = max(candidates, key=lambda c: c.grad[dim])
            # 只有邻居梯度确实高于当前格才移动
            if best.grad[dim] <= current_cell.grad[dim]:
                self._next_pos = None
                return

        else:  # DELIVER
            # ── 代价场：下降 dim=tar_id ───────────────────────────────
            dim = self.tar_id
            best = min(candidates, key=lambda c: c.grad[dim])
            # 只有邻居代价确实低于当前格才移动
            if best.grad[dim] >= current_cell.grad[dim]:
                self._next_pos = None
                return

        best.res = self.robot_id
        self._next_pos = (best.row, best.col)

    # ------------------------------------------------------------------
    # 阶段 2：执行移动
    # ------------------------------------------------------------------
    def execute_move(self, grid: "Grid") -> bool:
        if self._next_pos is None:
            return False

        nr, nc = self._next_pos
        target = grid[nr, nc]

        if target.res != self.robot_id:
            self._next_pos = None
            return False

        grid[self.row, self.col].occ = False
        self.row, self.col = nr, nc
        target.occ = True
        target.res = None
        self._next_pos = None
        return True

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"Robot#{self.robot_id} @({self.row},{self.col}) "
                f"task={self.task_type.name} tar={self.tar_id} "
                f"carry={self.carrying_pod}")
