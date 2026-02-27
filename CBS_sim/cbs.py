"""
cbs.py — 高层CBS规划器（Conflict-Based Search）

实现CBS算法 + Bypass 优化：
1. 初始化：为每个 agent 独立规划忽略其他 agent 的 Space-Time A* 路径
2. 检测第一个冲突
3. 尝试 Bypass：如果加约束后重规划 cost 不增加，直接采用（不分裂）
4. 否则按冲突分裂约束树节点（左子：对 agent_a 约束；右子：对 agent_b 约束）
5. 重新规划受约束 agent 的路径
6. 重复直到无冲突

支持：顶点冲突 + 边冲突（机器人互换位置）
路径结束后 agent 停在最终位置（DONE 状态），仍参与冲突检测。
"""

from __future__ import annotations
import heapq
from collections import defaultdict
from cbs_types import Pos, Constraint, Conflict, CTNode, Agent
from low_level import plan_full_path


# ---------------------------------------------------------------------------
# 冲突检测
# ---------------------------------------------------------------------------

def detect_conflict(paths: dict[int, list[Pos]]) -> Conflict | None:
    """
    检测所有 agent 对之间的第一个冲突（优先时间步小的）。

    顶点冲突：两个 agent 在同一时刻占据同一位置。
    边冲突：  两个 agent 在同一时间步内互换位置（swap）。

    重要：路径结束后 agent 停在最终位置（DONE 状态），仍参与冲突检测。
    """
    agent_ids = list(paths.keys())
    max_t = max((len(p) for p in paths.values()), default=0)

    def _pos(path: list[Pos], t: int) -> Pos:
        return path[t] if t < len(path) else path[-1]

    for t in range(max_t - 1):
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a, b = agent_ids[i], agent_ids[j]
                path_a = paths[a]
                path_b = paths[b]

                pa_t  = _pos(path_a, t)
                pb_t  = _pos(path_b, t)
                pa_t1 = _pos(path_a, t + 1)
                pb_t1 = _pos(path_b, t + 1)

                if pa_t1 == pb_t1:
                    return Conflict(
                        type="vertex",
                        agent_a=a, agent_b=b,
                        pos=pa_t1,
                        prev_pos_a=None, prev_pos_b=None,
                        timestep=t + 1,
                    )

                if pa_t == pb_t1 and pb_t == pa_t1:
                    return Conflict(
                        type="edge",
                        agent_a=a, agent_b=b,
                        pos=pa_t1,
                        prev_pos_a=pa_t,
                        prev_pos_b=pb_t,
                        timestep=t + 1,
                    )

    return None


# ---------------------------------------------------------------------------
# 路径代价（SIC = Sum of Individual Costs）
# ---------------------------------------------------------------------------
def sic(paths: dict[int, list[Pos]]) -> int:
    return sum(len(p) for p in paths.values())


def _make_constraint(conflict: Conflict, agent_id: int) -> Constraint:
    """从冲突构造对指定 agent 的约束。"""
    if conflict.type == "vertex":
        return Constraint(
            agent_id=agent_id,
            pos=conflict.pos,
            timestep=conflict.timestep,
        )
    else:
        prev = (conflict.prev_pos_a if agent_id == conflict.agent_a
                else conflict.prev_pos_b)
        return Constraint(
            agent_id=agent_id,
            pos=conflict.pos,
            timestep=conflict.timestep,
            prev_pos=prev,
        )


# ---------------------------------------------------------------------------
# CBS 节点（扩展 CTNode，增加按 agent 索引的约束和阶段缓存）
# ---------------------------------------------------------------------------
class _CBSNode:
    """内部节点：为性能优化，按 agent_id 索引约束。"""
    __slots__ = ('constraints_by_agent', 'paths', 'phases', 'cost')

    def __init__(self) -> None:
        # agent_id → [Constraint, ...]
        self.constraints_by_agent: dict[int, list[Constraint]] = defaultdict(list)
        self.paths: dict[int, list[Pos]] = {}
        # agent_id → (fetch_end_t, deliver_end_t, wait_end_t, return_end_t)
        self.phases: dict[int, tuple[int, int, int, int]] = {}
        self.cost: int = 0

    def add_constraint(self, c: Constraint) -> None:
        self.constraints_by_agent[c.agent_id].append(c)

    def copy(self) -> _CBSNode:
        n = _CBSNode()
        for aid, cs in self.constraints_by_agent.items():
            n.constraints_by_agent[aid] = list(cs)
        n.paths = {aid: list(p) for aid, p in self.paths.items()}
        n.phases = dict(self.phases)
        n.cost = self.cost
        return n

    def __lt__(self, other: _CBSNode) -> bool:
        return self.cost < other.cost


# ---------------------------------------------------------------------------
# CBS 高层规划
# ---------------------------------------------------------------------------
class CBS:
    def __init__(
        self,
        agents: list[Agent],
        rows: int,
        cols: int,
        obstacles: set[Pos],
        max_nodes: int = 200000,
    ) -> None:
        self.agents   = {a.agent_id: a for a in agents}
        self.rows     = rows
        self.cols     = cols
        self.obstacles = obstacles
        self.max_nodes = max_nodes

    # ------------------------------------------------------------------
    def _replan(
        self,
        agent_id: int,
        agent_constraints: list[Constraint],
    ) -> tuple[list[Pos], int, int, int, int] | None:
        """
        为 agent_id 重新规划完整路径。
        接收已过滤好的 agent 专属约束（无需再过滤）。
        **不修改 self.agents 对象**。
        """
        agent = self.agents[agent_id]
        if agent.task is None:
            return [agent.start], 0, 0, 0, 0

        return plan_full_path(
            start=agent.start,
            pod_pos=agent.task.pod_pos,
            station_pos=agent.task.station_pos,
            rows=self.rows,
            cols=self.cols,
            obstacles=self.obstacles,
            constraints=agent_constraints,
        )

    # ------------------------------------------------------------------
    def solve(self) -> dict[int, list[Pos]] | None:
        """
        执行 CBS 搜索（含 Bypass 优化）。
        返回：agent_id → path（无冲突的完整路径），无解返回 None。
        """
        # --- 根节点：每个 agent 独立规划 ---
        root = _CBSNode()
        for aid in self.agents:
            result = self._replan(aid, [])
            if result is None:
                print(f"[CBS] Agent {aid} has no initial path!")
                return None
            path, fe, de, we, re = result
            root.paths[aid] = path
            root.phases[aid] = (fe, de, we, re)
        root.cost = sic(root.paths)

        open_list: list[tuple[int, int, _CBSNode]] = []
        counter = 0
        heapq.heappush(open_list, (root.cost, counter, root))

        while open_list and counter < self.max_nodes:
            _, _, node = heapq.heappop(open_list)

            conflict = detect_conflict(node.paths)

            if conflict is None:
                # 无冲突！把最终路径和阶段索引写回 agent
                for aid, path in node.paths.items():
                    a = self.agents[aid]
                    a.path = path
                    if aid in node.phases:
                        fe, de, we, re = node.phases[aid]
                        a.fetch_end_t = fe
                        a.deliver_end_t = de
                        a.wait_end_t = we
                        a.return_end_t = re
                print(f"[CBS] Solved! Expanded {counter} nodes, cost={node.cost}")
                return node.paths

            # ── CBS Bypass 优化 ────────────────────────────────
            bypassed = False
            for constrained_agent in (conflict.agent_a, conflict.agent_b):
                new_c = _make_constraint(conflict, constrained_agent)

                trial_constraints = (
                    node.constraints_by_agent.get(constrained_agent, []) + [new_c]
                )
                result = self._replan(constrained_agent, trial_constraints)
                if result is None:
                    continue

                new_path, fe, de, we, re = result
                old_len = len(node.paths[constrained_agent])
                new_len = len(new_path)

                if new_len <= old_len:
                    # Bypass: cost 不增加，直接采用
                    node.add_constraint(new_c)
                    node.paths[constrained_agent] = new_path
                    node.phases[constrained_agent] = (fe, de, we, re)
                    node.cost = sic(node.paths)
                    counter += 1
                    heapq.heappush(open_list, (node.cost, counter, node))
                    bypassed = True
                    break

            if bypassed:
                continue

            # ── 标准 CBS 分裂 ──────────────────────────────────
            for constrained_agent in (conflict.agent_a, conflict.agent_b):
                counter += 1
                new_c = _make_constraint(conflict, constrained_agent)

                child = node.copy()
                child.add_constraint(new_c)

                agent_constraints = child.constraints_by_agent.get(
                    constrained_agent, []
                )
                result = self._replan(constrained_agent, agent_constraints)
                if result is None:
                    continue

                new_path, fe, de, we, re = result
                child.paths[constrained_agent] = new_path
                child.phases[constrained_agent] = (fe, de, we, re)
                child.cost = sic(child.paths)

                heapq.heappush(open_list, (child.cost, counter, child))

        print(f"[CBS] Failed! Expanded {counter} nodes.")
        return None
