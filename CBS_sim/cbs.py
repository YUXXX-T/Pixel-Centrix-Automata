"""
cbs.py — 高层CBS规划器（Conflict-Based Search）

实现标准CBS算法：
1. 初始化：为每个 agent 独立规划忽略其他 agent 的 Space-Time A* 路径
2. 检测第一个冲突
3. 按冲突分裂约束树节点（左子：对 agent_a 约束；右子：对 agent_b 约束）
4. 重新规划受约束 agent 的路径
5. 重复直到无冲突

支持：顶点冲突 + 边冲突（机器人互换位置）

注：Pod 位置不作为硬障碍处理。CBS 通过约束机制自动解决路径冲突。
Pod 只是机器人路径的中间目标（pickup point），不阻挡其他机器人经过。
"""

from __future__ import annotations
import heapq
import copy
from cbs_types import Pos, Constraint, Conflict, CTNode, Agent
from low_level import plan_full_path, space_time_astar


# ---------------------------------------------------------------------------
# 冲突检测
# ---------------------------------------------------------------------------

def detect_conflict(paths: dict[int, list[Pos]]) -> Conflict | None:
    """
    检测所有 agent 对之间的第一个冲突（优先时间步小的）。

    顶点冲突：两个 agent 在同一时刻都还在路径内，且占据同一位置。
    边冲突：  两个 agent 在同一时间步内互换位置（swap）。

    注：超出路径长度的 agent 视为"已完成/离场"，不产生冲突。
    """
    agent_ids = list(paths.keys())
    # 只检查到最长路径的时间步
    max_t = max((len(p) for p in paths.values()), default=0)

    for t in range(max_t - 1):
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a, b = agent_ids[i], agent_ids[j]
                path_a = paths[a]
                path_b = paths[b]

                # 只在两者都还在路径范围内时检测
                in_a = t + 1 < len(path_a)
                in_b = t + 1 < len(path_b)
                if not (in_a and in_b):
                    continue

                pa_t  = path_a[t]
                pb_t  = path_b[t]
                pa_t1 = path_a[t + 1]
                pb_t1 = path_b[t + 1]

                # 顶点冲突（在 t+1 时刻）
                if pa_t1 == pb_t1:
                    return Conflict(
                        type="vertex",
                        agent_a=a, agent_b=b,
                        pos=pa_t1,
                        prev_pos_a=None, prev_pos_b=None,
                        timestep=t + 1,
                    )

                # 边冲突（互换）
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
        max_nodes: int = 5000,
    ) -> None:
        self.agents   = {a.agent_id: a for a in agents}
        self.rows     = rows
        self.cols     = cols
        self.obstacles = obstacles
        self.max_nodes = max_nodes

        # pod 位置（对其他 robot 是"软障碍"，等任务分配后决定如何处理）
        # 在 CBS 中，pod_pos 本身不是硬障碍，但分配给某 agent 的 pod 对其他 agent 是障碍
        self._agent_pod: dict[int, Pos] = {}      # agent_id → pod_pos

    def set_agent_pods(self, pods: dict[int, Pos]) -> None:
        """保留接口兼容性（pod 不再作为硬障碍）。"""
        self._agent_pod = dict(pods)

    # ------------------------------------------------------------------
    def _replan(
        self,
        agent_id: int,
        constraints: list[Constraint],
    ) -> list[Pos] | None:
        """
        为 agent_id 重新规划完整路径（三段：start → pod → station）。
        Pod 位置不作为障碍——CBS 约束已处理所有冲突。
        """
        agent = self.agents[agent_id]
        if agent.task is None:
            return [agent.start]

        my_constraints = [c for c in constraints if c.agent_id == agent_id]

        result = plan_full_path(
            start=agent.start,
            pod_pos=agent.task.pod_pos,
            station_pos=agent.task.station_pos,
            rows=self.rows,
            cols=self.cols,
            obstacles=self.obstacles,   # 只用全局固定障碍
            constraints=my_constraints,
        )
        if result is None:
            return None

        path, fetch_end_t, deliver_end_t, wait_end_t, return_end_t = result
        agent.fetch_end_t   = fetch_end_t
        agent.deliver_end_t = deliver_end_t
        agent.wait_end_t    = wait_end_t
        agent.return_end_t  = return_end_t
        agent.path          = path
        return path

    # ------------------------------------------------------------------
    def solve(self) -> dict[int, list[Pos]] | None:
        """
        执行 CBS 搜索。
        返回：agent_id → path（无冲突的完整路径），无解返回 None。
        """
        # --- 根节点：每个 agent 独立规划 ---
        root = CTNode()
        for aid in self.agents:
            path = self._replan(aid, root.constraints)
            if path is None:
                print(f"[CBS] Agent {aid} has no initial path!")
                return None
            root.paths[aid] = path
        root.cost = sic(root.paths)

        open_list: list[tuple[int, int, CTNode]] = []
        counter = 0
        heapq.heappush(open_list, (root.cost, counter, root))

        while open_list and counter < self.max_nodes:
            _, _, node = heapq.heappop(open_list)

            # 拷贝 agent 阶段信息到当前 node 的路径
            conflict = detect_conflict(node.paths)

            if conflict is None:
                # 无冲突！把最终路径写回 agent
                for aid, path in node.paths.items():
                    self.agents[aid].path = path
                print(f"[CBS] Solved! Expanded {counter} nodes, cost={node.cost}")
                return node.paths

            # 分裂：对 agent_a 和 agent_b 各生成一个子节点
            for constrained_agent in (conflict.agent_a, conflict.agent_b):
                counter += 1
                child = CTNode(
                    constraints=copy.copy(node.constraints),
                    paths=dict(node.paths),
                    cost=node.cost,
                )

                if conflict.type == "vertex":
                    new_c = Constraint(
                        agent_id=constrained_agent,
                        pos=conflict.pos,
                        timestep=conflict.timestep,
                    )
                else:
                    # 边约束：禁止 constrained_agent 在该时刻通过这条边
                    if constrained_agent == conflict.agent_a:
                        new_c = Constraint(
                            agent_id=constrained_agent,
                            pos=conflict.pos,
                            timestep=conflict.timestep,
                            prev_pos=conflict.prev_pos_a,
                        )
                    else:
                        new_c = Constraint(
                            agent_id=constrained_agent,
                            pos=conflict.pos,
                            timestep=conflict.timestep,
                            prev_pos=conflict.prev_pos_b,
                        )

                child.constraints.append(new_c)

                # 重新规划受约束的 agent
                new_path = self._replan(constrained_agent, child.constraints)
                if new_path is None:
                    continue  # 该分支无解，剪枝

                child.paths[constrained_agent] = new_path
                child.cost = sic(child.paths)

                heapq.heappush(open_list, (child.cost, counter, child))

        print(f"[CBS] Failed! Expanded {counter} nodes.")
        return None
