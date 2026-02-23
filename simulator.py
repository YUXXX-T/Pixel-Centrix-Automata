"""
simulator.py — Simulator

Tick 流程：
  0. 派发新订单（inject_order，工作站代价场已由 setup_station 预建）
  1. Injector 维持梯度场（吸引场 + 代价场恢复扰动）
  2. 防碰撞注入：配送中的机器人向当前格注入高代价惩罚
  3. Phase 1: 所有机器人 reserve()
  4. 防碰撞移除：清除刚才注入的惩罚
  5. Phase 2: 所有机器人 execute_move()
  6. 到达检测：
       - FETCH_POD 到达 pod 格 → 举起货架，清除 dim=0 吸引场，切换 DELIVER
       - DELIVER   到达工作站格 → 放下货架，切换 IDLE
"""

from __future__ import annotations
from dataclasses import dataclass, field
from grid import Grid
from robot import Robot, TaskType
from injector import GradientInjector

# 防碰撞梯度惩罚：三圈逐渐衰减
# Ring-0 (机器人所在格)：极高，等同于封锁该格
PENALTY_R0: float = 1e4
# Ring-1 (第一圈邻居)：显著高于自然代价，强烈排斥
PENALTY_R1: float = 800.0
# Ring-2 (第二圈邻居)：中等高于自然代价，较温和的排斥
PENALTY_R2: float = 400.0


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
        self.grid = Grid(rows, cols)
        self.injector = GradientInjector(self.grid)
        self.robots: list[Robot] = []
        self._stations: dict[int, Station] = {}
        self._pending_orders: list[Order] = []
        self._active_order: Order | None = None
        self.tick_count: int = 0
        # 防碰撞快照：{(dim, row, col): 原始値}
        self._penalty_snapshot: dict[tuple[int, int, int], float] = {}

    # ------------------------------------------------------------------
    def add_robot(self, robot: Robot) -> None:
        self.robots.append(robot)
        self.grid[robot.row, robot.col].occ = True

    def register_station(self, tar_id: int, row: int, col: int) -> None:
        """注册工作站并初始化其代价场（启动时调用一次）。"""
        st = Station(tar_id=tar_id, row=row, col=col)
        self._stations[tar_id] = st
        self.injector.setup_station(tar_id, row, col)

    def add_order(self, order: Order) -> None:
        self._pending_orders.append(order)

    # ------------------------------------------------------------------
    def _dispatch_orders(self) -> None:
        if self._active_order is not None or not self._pending_orders:
            return
        order = self._pending_orders.pop(0)
        self._active_order = order

        # 仅注入 pod 吸引场（工作站代价场已预建）
        self.injector.inject_order(order.pod_row, order.pod_col, order.tar_id)

        for robot in self.robots:
            if robot.task_type == TaskType.IDLE:
                robot.assign_fetch(order.tar_id)
                st = self._stations.get(order.tar_id)
                print(f"[Sim]  Robot#{robot.robot_id} assigned: "
                      f"fetch pod@({order.pod_row},{order.pod_col}) "
                      f"→ station#{order.tar_id}"
                      + (f"@({st.row},{st.col})" if st else ""))
                break

    # ------------------------------------------------------------------
    def tick(self) -> bool:
        self.tick_count += 1
        self._dispatch_orders()

        # ── 维持梯度场 ────────────────────────────────────────────────
        self.injector.tick_diffuse(att_iters=1, cost_iters=1)

        # ── Phase 1：顺序预约（每个机器人只看其他机器人的代价惩罚） ────
        # 核心修复：若令机器人看到自己的 ring-1 惩罚，则其候选格全部
        # 变为同一惩罚值，失去方向信息。改为顺序执行：
        #   对每个机器人 R：
        #     1. 注入所有【其他】配送机器人的 ring-0/1/2 惩罚
        #     2. R.reserve()（此时只看到他人惩罚，自身候选格保持自然代价）
        #     3. 还原快照
        for robot in self.robots:
            self._apply_others_penalties(exclude_robot=robot)
            robot.reserve(self.grid)
            self._remove_others_penalties()

        # ── Phase 2：所有机器人执行移动 ───────────────────────────────
        for robot in self.robots:
            robot.execute_move(self.grid)

        # ── 到达检测 ─────────────────────────────────────────────────
        order = self._active_order
        all_idle = True

        for robot in self.robots:
            if robot.task_type == TaskType.IDLE:
                continue
            all_idle = False

            if order is None:
                continue

            if (robot.task_type == TaskType.FETCH_POD and
                    robot.row == order.pod_row and
                    robot.col == order.pod_col):
                robot.carrying_pod = True
                robot.task_type = TaskType.DELIVER
                self.injector.clear_pod_gradient()
                print(f"[Sim]  Robot#{robot.robot_id} lifted pod "
                      f"at tick {self.tick_count}!")

            elif robot.task_type == TaskType.DELIVER:
                st = self._stations.get(robot.tar_id)
                if st and robot.row == st.row and robot.col == st.col:
                    robot.carrying_pod = False
                    robot.task_type = TaskType.IDLE
                    order.fulfilled = True
                    self._active_order = None
                    print(f"[Sim]  Robot#{robot.robot_id} delivered "
                          f"at tick {self.tick_count}! "
                          f"Station#{st.tar_id}@({st.row},{st.col})")

        print(f"  Tick {self.tick_count:>3} | {self.robots[0]}")

        has_work = (self._active_order is not None) or bool(self._pending_orders)
        return has_work or not all_idle

    # ------------------------------------------------------------------
    # 防碰撞：只注入「其他机器人」的惩罚（排除自身）
    # ------------------------------------------------------------------
    def _apply_others_penalties(self, exclude_robot: "Robot") -> None:
        """
        注入所有【其他】配送机器人的三圈代价惩罚，排除 exclude_robot 自身。

        这样 exclude_robot.reserve() 执行时：
          - 看到来自其他机器人的 ring-0/1/2 惩罚（正确的他人排斥信号）
          - 自身 ring-1 候选格保持自然代价（方向信息完整）

        Ring-0 (其他机器人所在格) : PENALTY_R0 — 极高，实质封锁
        Ring-1 (其他机器人第一圈) : PENALTY_R1 — 强烈排斥
        Ring-2 (其他机器人第二圈) : PENALTY_R2 — 中等排斥
        """
        snap = self._penalty_snapshot
        snap.clear()

        ring_config = [(0, PENALTY_R0), (1, PENALTY_R1), (2, PENALTY_R2)]

        for robot in self.robots:
            if robot is exclude_robot:          # 跳过自身
                continue
            # if robot.task_type != TaskType.DELIVER:
            #     continue
            dim = robot.tar_id

            for dist, penalty in ring_config:
                cells = (
                    [self.grid[robot.row, robot.col]] if dist == 0
                    else self.grid.cells_at_distance(robot.row, robot.col, dist)
                )
                for cell in cells:
                    key = (dim, cell.row, cell.col)
                    if key not in snap:
                        snap[key] = cell.grad[dim]
                    # cell.grad[dim] = max(cell.grad[dim], penalty)
                    cell.grad[dim] += penalty


    def _remove_others_penalties(self) -> None:
        """精确还原 _apply_others_penalties 之前的快照值。"""
        for (dim, r, c), orig in self._penalty_snapshot.items():
            self.grid[r, c].grad[dim] = orig
        self._penalty_snapshot.clear()

    # ------------------------------------------------------------------
    def run(self, max_ticks: int = 300, callback=None) -> None:
        for _ in range(max_ticks):
            running = self.tick()
            if callback:
                callback(self)
            if not running:
                print(f"\n[Sim] All orders fulfilled in {self.tick_count} ticks.")
                return
        print(f"\n[Sim] Reached max ticks ({max_ticks}).")
