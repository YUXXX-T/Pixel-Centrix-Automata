"""
cell.py — Cell Agent
每个 Cell 是地图上的一个格子 Agent，持有：
  - cell_id : (row, col) 元组
  - occ     : 是否当前被某个 robot 占用（bool）
  - res     : 预约字段（int | None）
              含义：若 res = robot_id，表示该 robot 将在下一时间步从其当前格
              移动到本格（握手预约）。None 表示无预约。
  - grad    : N 维梯度向量
              dim=0        → 寻找货架（pod）用的引力场
              dim=tar_id   → 导航到对应工作站用的势能场
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

# 梯度向量维度总数（0 + 最多 N_DIM-1 个工作站编号）
N_DIM: int = 8


@dataclass
class Cell:
    row: int
    col: int
    occ: bool = False                          # 当前占用状态
    res: int | None = None                     # 预约机器人ID（None = 空闲）
    grad: np.ndarray = field(
        default_factory=lambda: np.zeros(N_DIM, dtype=float)
    )

    # ------------------------------------------------------------------
    @property
    def cell_id(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def is_available(self) -> bool:
        """若既未被占用也未被预约，则可用。"""
        return (not self.occ) and (self.res is None)

    # ------------------------------------------------------------------
    def reset_dim(self, dim: int) -> None:
        """将指定维度的梯度清零。"""
        self.grad[dim] = 0.0

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Cell({self.row},{self.col}) "
            f"occ={self.occ} res={self.res} "
            f"grad={np.round(self.grad, 2)}"
        )
