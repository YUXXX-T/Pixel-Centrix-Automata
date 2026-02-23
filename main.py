"""
main.py — Demo Entry Point
10×10 地图，1 个机器人，1 张订单：
  Robot   : 起点 (0, 0)
  Pod     : (7, 6)
  Station : tar_id=1, 位置 (2, 9)

可视化说明：
  - 左图：dim=0 梯度热力图（寻找货架用）
  - 右图：dim=1 梯度热力图（导航去工作站用）
  - 每个格子中心绘制半透明方块，颜色深浅代表梯度大小
  - 白色网格线标出每个 Cell 边界
  - 绿色圆圈 = Robot（走在格子正中心）
  - 蓝色三角 = Pod 位置
  - 红色星号 = 工作站位置
  - 橙色方块 = Occ 占用中的格子
  - 紫色方块 = Res 已预约的格子
"""

import sys, time
from simulator import Simulator, Order
from robot import Robot


ROWS, COLS     = 10, 10
ROBOT_START    = (0, 0)
POD_POS        = (7, 6)
STATION_ID     = 1
STATION_POS    = (2, 9)
VISUALIZE      = True
MAX_TICKS      = 300
TICK_INTERVAL  = 0.15   # 秒/tick


def build_sim() -> Simulator:
    sim = Simulator(ROWS, COLS)

    # 注册工作站（永久固件，预建代价场）
    sim.register_station(tar_id=STATION_ID,
                         row=STATION_POS[0], col=STATION_POS[1])

    sim.add_robot(Robot(robot_id=0, start_row=ROBOT_START[0], start_col=ROBOT_START[1]))

    # 订单只需指定 pod 位置和目标工作站编号
    sim.add_order(Order(
        pod_row=POD_POS[0], pod_col=POD_POS[1],
        tar_id=STATION_ID,
    ))
    return sim


# ─────────────────────────────────────────────────────────────────────
# Console 模式
# ─────────────────────────────────────────────────────────────────────
def run_console() -> None:
    print("=" * 58)
    print("  Distributed Warehouse System — Console Demo")
    print(f"  Grid: {ROWS}×{COLS}  |  Robot@{ROBOT_START}")
    print(f"  Pod: {POD_POS}  →  Station#{STATION_ID}@{STATION_POS}")
    print("=" * 58)
    sim = build_sim()
    sim.run(max_ticks=MAX_TICKS)


# ─────────────────────────────────────────────────────────────────────
# Matplotlib 动画模式
# ─────────────────────────────────────────────────────────────────────
def run_visual() -> None:
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
    except ImportError:
        print("[main] matplotlib/numpy not found, falling back to console.")
        run_console()
        return

    from injector import MAX_GRAD

    sim = build_sim()
    sim._dispatch_orders()   # 提前注入梯度

    # ── 布局 ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.patch.set_facecolor("#1a1a2e")
    TITLES = [
        "dim=0  Reach Pod: gravitational field (high gradient = pod, robot climbs)",
        f"dim={STATION_ID}  Deliver to Station#{STATION_ID}: cost field (low value = station, robot descends)",
    ]
    CMAPS  = ["YlOrRd", "plasma_r"]   # plasma_r: 低代价=亮色（目标处最亮）
    DIMS   = [0, STATION_ID]

    # 坐标约定：x = col，y = ROWS-1-row（y 轴向上，大值在顶）
    def gy(r: int) -> int:
        """将 grid row 转为显示 y 坐标（y 轴向上）。"""
        return ROWS - 1 - r
    ax_objects = []
    for idx, (ax, title, cmap) in enumerate(zip(axes, TITLES, CMAPS)):
        ax.set_facecolor("#0f0f1a")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlim(-0.5, COLS - 0.5)
        ax.set_ylim(-0.5, ROWS - 0.5)   # y 轴向上：0 在底，ROWS-1 在顶
        ax.set_aspect("equal")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        # 网格线（Cell 边界）
        for x in np.arange(-0.5, COLS, 1):
            ax.axvline(x, color="#334466", linewidth=0.6, zorder=1)
        for y in np.arange(-0.5, ROWS, 1):
            ax.axhline(y, color="#334466", linewidth=0.6, zorder=1)

        # 坐标轴刻度对齐 Cell 中心
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.set_xticklabels(range(COLS), color="#aaaacc", fontsize=7)
        ax.set_yticklabels(range(ROWS), color="#aaaacc", fontsize=7)

        # 热力图：左图 vmax=MAX_GRAD（吸引场），右图 vmax=代价场最大有效值
        mat = np.flipud(sim.grid.grad_matrix(DIMS[idx]))
        from grid import COST_INF
        if DIMS[idx] == 0:
            vmin, vmax = 0, MAX_GRAD
        else:
            # 代价场：过滤掉 COST_INF 方便可视化
            raw = sim.grid.grad_matrix(DIMS[idx])
            finite_max = raw[raw < COST_INF * 0.5].max() if (raw < COST_INF * 0.5).any() else 1
            vmin, vmax = 0, finite_max
            mat = np.flipud(np.clip(raw, 0, finite_max))
        im = ax.imshow(
            mat, vmin=vmin, vmax=vmax, cmap=cmap,
            origin="lower",
            extent=[-0.5, COLS - 0.5, -0.5, ROWS - 0.5],
            aspect="equal", alpha=0.75, zorder=2,
        )
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.yaxis.set_tick_params(color="gray")

        # Occ 覆盖层（橙色方块集合，用 scatter 模拟）
        occ_scat = ax.scatter([], [], s=300, marker="s",
                               color="#ff9900", alpha=0.55, zorder=3, label="Occ")
        # Res 覆盖层（紫色方块）
        res_scat = ax.scatter([], [], s=300, marker="s",
                               color="#cc44ff", alpha=0.45, zorder=3, label="Res")

        # 固定标记：Pod、Station（坐标转换 gy）
        ax.plot(POD_POS[1], gy(POD_POS[0]), "b^", markersize=11, zorder=6,
                label="Pod", markeredgecolor="white", markeredgewidth=0.8)
        ax.plot(STATION_POS[1], gy(STATION_POS[0]), "r*", markersize=15, zorder=6,
                label=f"Station#{STATION_ID}", markeredgecolor="white", markeredgewidth=0.5)

        # Robot 标记（走在格子中心）
        rob_dot, = ax.plot([], [], "o", markersize=14, zorder=7,
                           color="#00ff88", markeredgecolor="white", markeredgewidth=1.2,
                           label="Robot")

        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#1a1a3a", edgecolor="#334466", labelcolor="white")
        ax_objects.append((im, occ_scat, res_scat, rob_dot))

    status_txt = fig.text(
        0.5, 0.01, "Initializing…",
        ha="center", fontsize=10, color="white",
        fontfamily="monospace",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.ion()
    plt.show()

    robot = sim.robots[0]

    def update_frame() -> None:
        # 收集 Occ / Res 格子的坐标（行坐标转换 gy）
        occ_pts = [(c.col, gy(c.row)) for c in sim.grid.all_cells() if c.occ]
        res_pts = [(c.col, gy(c.row)) for c in sim.grid.all_cells() if c.res is not None]

        for dim_idx, (im, occ_scat, res_scat, rob_dot) in enumerate(ax_objects):
            from grid import COST_INF
            raw = sim.grid.grad_matrix(DIMS[dim_idx])
            if DIMS[dim_idx] == 0:
                frame_mat = np.flipud(raw)
            else:
                # 代价场：裁剪 COST_INF 使可视化不被无效值撑满
                finite_mask = raw < COST_INF * 0.5
                cap = raw[finite_mask].max() if finite_mask.any() else 1.0
                frame_mat = np.flipud(np.clip(raw, 0, cap))
            im.set_data(frame_mat)
            occ_scat.set_offsets(occ_pts if occ_pts else [[None, None]])
            res_scat.set_offsets(res_pts if res_pts else [[None, None]])
            # Robot 走在格子中心，行坐标转换 gy
            rob_dot.set_data([robot.col], [gy(robot.row)])

        status_txt.set_text(
            f"Tick {sim.tick_count:>3}  |  "
            f"Robot@({robot.row},{robot.col})  "
            f"task={robot.task_type.name}  "
            f"carry={robot.carrying_pod}"
        )
        fig.canvas.draw()
        fig.canvas.flush_events()

    # 初始帧
    update_frame()
    time.sleep(0.5)

    for _ in range(MAX_TICKS):
        still_running = sim.tick()
        update_frame()
        time.sleep(TICK_INTERVAL)

        if not still_running:
            status_txt.set_text(
                f"✓ Done in {sim.tick_count} ticks! "
                f"Robot delivered pod to Station#{STATION_ID}@{STATION_POS}"
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            break

    plt.ioff()
    plt.show()


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if VISUALIZE:
        run_visual()
    else:
        run_console()
