"""
simulator.py — Simulator (Multi-Robot, 6-Dim)

Grad 维度（与 cell.py 一致）：
  Grad[0]   : Pod 吸引场（爬升，共用）
  Grad[1..4]: 工作站代价场（下降，永久）
  Grad[5]   : 返程代价场（下降，临时）

防碰撞惩罚方向：
  吸引场(dim=0, ascending)  → 对其他机器人所在格注入【低值】（凹坑）+ 2圈扩张
  代价场(dim=1-5, descending)→ 对其他机器人所在格注入【高值】（凸峰）+ 2圈扩张
"""

from __future__ import annotations
from dataclasses import dataclass
from grid import Grid
from robot import Robot, TaskType, POD_DIM, RETURN_DIM
from injector import GradientInjector

WAIT_TICKS: int = 5

# 防碰撞惩罚值
PENALTY_R0: float = 200.0    # 导航机器人自己的 Cell（来自其他机器人）
PENALTY_R1: float = 120.0    # 第 1 圈
PENALTY_R2: float = 60.0     # 第 2 圈

# Wake trail（空间热力尾迹）
WAKE_INIT: float     = 100.0   # 机器人第一次到达 Cell 时的 wake 值
WAKE_N: int          = 2       # 机器人离开 Cell 后 N tick 衰减到 0
WAKE_DELTA: float    = WAKE_INIT / WAKE_N   # 每 tick 衰减量
W2: float            = 0.5     # wake 权重（penalty 权重 w1=1.0 隐含）


@dataclass
class Station:
    tar_id: int
    row: int
    col: int


@dataclass
class Order:
    pod_row: int
    pod_col: int
    tar_id: int
    fulfilled: bool = False


class Simulator:
    def __init__(self, rows: int, cols: int) -> None:
        self.grid     = Grid(rows, cols)
        self.injector = GradientInjector(self.grid)
        self.robots   : list[Robot]          = []
        self._stations: dict[int, Station]   = {}
        self._pending_orders: list[Order]    = []
        self._robot_orders: dict[int, Order] = {}   # robot_id → Order
        self.tick_count: int                 = 0
        self._penalty_snapshot: dict[tuple[int, int, int], float] = {}

    def add_robot(self, robot: Robot) -> None:
        self.robots.append(robot)
        self.grid[robot.row, robot.col].occ = True

    def register_station(self, tar_id: int, row: int, col: int) -> None:
        self._stations[tar_id] = Station(tar_id=tar_id, row=row, col=col)
        self.injector.setup_station(tar_id, row, col)

    def add_order(self, order: Order) -> None:
        self._pending_orders.append(order)

    # ------------------------------------------------------------------
    def _dispatch_orders(self) -> None:
        for robot in self.robots:
            if robot.task_type != TaskType.IDLE:
                continue
            if not self._pending_orders:
                break
            order = self._pending_orders.pop(0)
            self._robot_orders[robot.robot_id] = order
            self.injector.inject_order(order.pod_row, order.pod_col)
            robot.assign_fetch(order.tar_id)
            st = self._stations.get(order.tar_id)
            print(f"[Sim]  Robot#{robot.robot_id} ← pod@({order.pod_row},{order.pod_col}) "
                  f"→ station#{order.tar_id}"
                  + (f"@({st.row},{st.col})" if st else ""))

    # ------------------------------------------------------------------
    def tick(self) -> bool:
        self.tick_count += 1
        self._dispatch_orders()
        self.injector.tick_diffuse(att_iters=1, cost_iters=1)

        # Phase 0: 在所有机器人当前位置打上 wake 标记（首次到达时设为 WAKE_INIT）
        for robot in self.robots:
            cell = self.grid[robot.row, robot.col]
            if cell.wake == 0.0:
                cell.wake = WAKE_INIT

        # Phase 1: 顺序预约（penalty + wake 参与评分）
        for robot in self.robots:
            self._apply_others_penalties(exclude_robot=robot)
            robot.reserve(self.grid)
            self._remove_others_penalties()

        # Phase 2: 执行移动
        for robot in self.robots:
            robot.execute_move(self.grid)

        # Phase 2.5: 全局 wake 衰减（只衰减 wake > 0 且不被 occ 的 Cell）
        for cell in self.grid.all_cells():
            if cell.wake > 0.0 and not cell.occ:
                cell.wake = max(0.0, cell.wake - WAKE_DELTA)

        # Phase 3: 到达检测
        any_active = False
        for robot in self.robots:
            t     = robot.task_type
            order = self._robot_orders.get(robot.robot_id)

            if t == TaskType.WAIT_AT_STATION:
                any_active = True
                robot.wait_ticks -= 1
                if robot.wait_ticks <= 0:
                    assert robot.pod_origin is not None
                    pr, pc = robot.pod_origin
                    self.injector.inject_return_field(pr, pc)
                    robot.task_type = TaskType.RETURN_POD
                    print(f"[Sim]  Robot#{robot.robot_id} → returning to ({pr},{pc})  "
                          f"tick={self.tick_count}")
                continue

            if t == TaskType.IDLE:
                continue

            any_active = True

            if t == TaskType.FETCH_POD and order is not None:
                if robot.row == order.pod_row and robot.col == order.pod_col:
                    robot.carrying_pod = True
                    robot.pod_origin   = (order.pod_row, order.pod_col)
                    robot.task_type    = TaskType.DELIVER
                    self.injector.clear_pod_peak(order.pod_row, order.pod_col)
                    print(f"[Sim]  Robot#{robot.robot_id} lifted pod  tick={self.tick_count}")

            elif t == TaskType.DELIVER:
                st = self._stations.get(robot.tar_id)
                if st and robot.row == st.row and robot.col == st.col:
                    robot.task_type  = TaskType.WAIT_AT_STATION
                    robot.wait_ticks = WAIT_TICKS
                    if order is not None:
                        order.fulfilled = True
                    print(f"[Sim]  Robot#{robot.robot_id} delivered → wait {WAIT_TICKS}  "
                          f"tick={self.tick_count}")

            elif t == TaskType.RETURN_POD:
                assert robot.pod_origin is not None
                pr, pc = robot.pod_origin
                if robot.row == pr and robot.col == pc:
                    robot.carrying_pod = False
                    robot.pod_origin   = None
                    robot.task_type    = TaskType.IDLE
                    self.injector.clear_return_target(pr, pc)
                    self._robot_orders.pop(robot.robot_id, None)
                    print(f"[Sim]  Robot#{robot.robot_id} returned pod → IDLE  "
                          f"tick={self.tick_count}")

        parts = " | ".join(
            f"R{r.robot_id}@({r.row},{r.col}) {r.task_type.name[:4]}"
            for r in self.robots
        )
        print(f"  Tick {self.tick_count:>3} | {parts}")

        return bool(self._pending_orders) or bool(self._robot_orders) or any_active

    # ------------------------------------------------------------------
    # 防碰撞：方向感知惩罚
    def _apply_others_penalties(self, exclude_robot: Robot) -> None:
        """
        对 exclude_robot 的导航维度注入其他机器人的障碍。

        规则：
        1. 跳过 exclude_robot 自身（自己不给自己加惩罚）
        2. 每个其他机器人 B（包括 IDLE）：
           a. 跳过 B 自己所在 Cell（ring 0 不注入，靠 occ 即可）
           b. 在 B 周围 ring 1、ring 2 注入惩罚
           c. 在 exclude_robot 自己所在 Cell 也注入惩罚
              （防止 R 在其他机器人 ring 外形成局部极值而卡住）

        方向：
          ascending(dim=0): 减值（凹坑）
          descending(dim=1-5): 加值（凸峰）
        """
        dim = exclude_robot.nav_dim
        if dim < 0:
            return

        ascending = exclude_robot.ascending
        snap = self._penalty_snapshot
        snap.clear()

        def _inject(cell, penalty_val):
            key = (dim, cell.row, cell.col)
            if key not in snap:
                snap[key] = cell.grad[dim]
            if ascending:
                cell.grad[dim] -= penalty_val
            else:
                cell.grad[dim] += penalty_val

        nav_cell = self.grid[exclude_robot.row, exclude_robot.col]

        for robot in self.robots:
            if robot is exclude_robot:
                continue

            # (c) 在导航机器人自己的 Cell 上注入惩罚（来自该其他机器人）
            _inject(nav_cell, PENALTY_R0)

            # (b) Ring 1 和 Ring 2（跳过 ring 0 = 其他机器人自己的 Cell）
            for dist, penalty in [(1, PENALTY_R1), (2, PENALTY_R2)]:
                for cell in self.grid.cells_at_distance(robot.row, robot.col, dist):
                    _inject(cell, penalty)

    def _remove_others_penalties(self) -> None:
        for (dim, r, c), orig in self._penalty_snapshot.items():
            self.grid[r, c].grad[dim] = orig
        self._penalty_snapshot.clear()

    # ------------------------------------------------------------------
    # 可视化用：在所有活跃维度上注入全部机器人的惩罚
    # ------------------------------------------------------------------
    def apply_viz_penalties(self) -> None:
        """临时注入所有机器人的惩罚效果到对应维度，用于热力图可视化。"""
        snap = self._viz_snap = {}

        # 收集所有活跃维度
        active_dims: set[tuple[int, bool]] = set()
        for robot in self.robots:
            dim = robot.nav_dim
            if dim >= 0:
                active_dims.add((dim, robot.ascending))

        def _viz_inject(cell, d, asc, val):
            key = (d, cell.row, cell.col)
            if key not in snap:
                snap[key] = cell.grad[d]
            if asc:
                cell.grad[d] -= val
            else:
                cell.grad[d] += val

        for src in self.robots:
            for dim, ascending in active_dims:
                # 每个其他机器人的 Cell 上注入惩罚
                for other in self.robots:
                    if other is src:
                        continue
                    _viz_inject(self.grid[other.row, other.col], dim, ascending, PENALTY_R0)

                # Ring 1, Ring 2（跳过 src 自己的 Cell）
                for dist, penalty in [(1, PENALTY_R1), (2, PENALTY_R2)]:
                    for cell in self.grid.cells_at_distance(src.row, src.col, dist):
                        _viz_inject(cell, dim, ascending, penalty)

    def remove_viz_penalties(self) -> None:
        """还原 apply_viz_penalties 的效果。"""
        snap = getattr(self, '_viz_snap', {})
        for (dim, r, c), orig in snap.items():
            self.grid[r, c].grad[dim] = orig
        self._viz_snap = {}

    # ------------------------------------------------------------------
    def run(self, max_ticks: int = 500, callback=None) -> None:
        for _ in range(max_ticks):
            running = self.tick()
            if callback:
                callback(self)
            if not running:
                print(f"\n[Sim] All done in {self.tick_count} ticks.")
                return
        print(f"\n[Sim] Reached max ticks ({max_ticks}).")

