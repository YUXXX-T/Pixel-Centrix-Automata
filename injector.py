"""
injector.py — GradientInjector

职责：
  1. setup_station(tar_id, row, col)
       工作站是地图的永久固件，启动时调用一次。
       初始化 dim=tar_id 的代价场（0 在工作站格，COST_INF 在其余格），
       并预跑波前扩散让代价场覆盖全图。

  2. inject_order(pod_row, pod_col, tar_id)
       接到订单后，向 pod 格注入 MAX_GRAD（dim=0，吸引场），
       并预跑吸引场扩散。
       工作站代价场无需再做任何修改（已在 setup_station 中建立）。

  3. tick_diffuse()
       每 tick 维持 pod 吸引场稳定（若 pod 未被取走）。
       工作站代价场只需少量迭代维持稳定（扰动恢复）。

  4. clear_pod_gradient()
       货架被举起后清除 dim=0 吸引场。

  防碰撞（多机器人）：
       simulator 在每 tick reserve 阶段前，调用
       grid.add_cost_penalty(row, col, tar_id, ROBOT_PENALTY)
       为正在配送的机器人所在格注入高代价，reserve 结束后移除。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grid import Grid

# ── 吸引场参数（dim=0，货架） ────────────────────────────────────────
MAX_GRAD: float = 1000.0
ALPHA: float = 0.90
DELTA_DECAY: float = 10.0
INIT_DIFFUSE_ITERS: int = 60

# ── 代价场参数（dim=K，工作站） ──────────────────────────────────────
STATION_DELTA_DECAY: float = 10.0
INIT_COST_ITERS: int = 300    # 波前需要足够多迭代才能覆盖全图


class GradientInjector:
    def __init__(self, grid: "Grid") -> None:
        self.grid = grid
        self._pod_source: tuple[int, int] | None = None          # dim=0 的 source
        self._station_sources: dict[int, tuple[int, int]] = {}   # dim=K → (row, col)

    # ------------------------------------------------------------------
    # 工作站永久代价场初始化（启动时调用）
    # ------------------------------------------------------------------
    def setup_station(self, tar_id: int, row: int, col: int) -> None:
        """
        初始化第 tar_id 号工作站的代价场（dim=tar_id）。
        工作站格 grad[tar_id] = 0，其余 = COST_INF，然后预跑波前扩散。
        """
        g = self.grid
        g.init_cost_dim(tar_id, (row, col), source_value=0.0)
        g.diffuse_cost(tar_id,
                       delta_decay=STATION_DELTA_DECAY,
                       iterations=INIT_COST_ITERS,
                       source=(row, col), source_value=0.0)
        self._station_sources[tar_id] = (row, col)
        print(f"[Injector] Station#{tar_id} cost field ready: "
              f"(row={row}, col={col}), dim={tar_id}")

    # ------------------------------------------------------------------
    # 订单注入（仅处理货架吸引场 dim=0）
    # ------------------------------------------------------------------
    def inject_order(self, pod_row: int, pod_col: int, tar_id: int) -> None:
        """
        接到订单后注入货架吸引场（dim=0）。
        工作站代价场在 setup_station 时已建立，无需重复注入。
        """
        g = self.grid
        g.inject(pod_row, pod_col, 0, MAX_GRAD)
        self._pod_source = (pod_row, pod_col)

        g.diffuse(0, alpha=ALPHA, delta_decay=DELTA_DECAY,
                  iterations=INIT_DIFFUSE_ITERS,
                  source=(pod_row, pod_col), source_value=MAX_GRAD)

        print(f"[Injector] Pod attraction field injected at "
              f"({pod_row},{pod_col}) → station#{tar_id}")

    # ------------------------------------------------------------------
    # 每 tick 维持梯度场稳定
    # ------------------------------------------------------------------
    def tick_diffuse(self, att_iters: int = 1, cost_iters: int = 1) -> None:
        """
        att_iters  : 吸引场每 tick 的扩散迭代次数（pod 存在时）
        cost_iters : 代价场每 tick 的扰动恢复迭代次数
        """
        # 吸引场（dim=0）
        if self._pod_source is not None:
            sr, sc = self._pod_source
            self.grid.diffuse(0, alpha=ALPHA, delta_decay=DELTA_DECAY,
                              iterations=att_iters,
                              source=(sr, sc), source_value=MAX_GRAD)

        # 代价场（dim=K，工作站）
        for tar_id, (sr, sc) in self._station_sources.items():
            self.grid.diffuse_cost(tar_id,
                                   delta_decay=STATION_DELTA_DECAY,
                                   iterations=cost_iters,
                                   source=(sr, sc), source_value=0.0)

    # ------------------------------------------------------------------
    # 货架被举起：清除吸引场
    # ------------------------------------------------------------------
    def clear_pod_gradient(self) -> None:
        self.grid.clear_dim(0)
        self._pod_source = None
        print("[Injector] Pod attraction field (dim=0) cleared.")
