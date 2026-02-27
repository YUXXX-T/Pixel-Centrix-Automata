"""Test CBS with increased node budget (200K) on 10-robot scenario."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

SCENARIO = 10

import world as _world
_world.ACTIVE_CONFIG = str(SCENARIO)
_world._reinit()

from world import ROWS, COLS, OBSTACLES, build_agents_and_tasks
from task_assign import assign_tasks
from cbs import CBS

agents, tasks = build_agents_and_tasks()
assign_tasks(agents, tasks)

print("--- CBS with 200K nodes, SCENARIO=10 ---")
cbs = CBS(agents=agents, rows=ROWS, cols=COLS, obstacles=OBSTACLES, max_nodes=200000)
solution = cbs.solve()

if solution is None:
    print("[RESULT] CBS FAILED with 200K nodes")
    sys.exit(1)

max_t = max((len(p) for p in solution.values()), default=0)
print(f"Makespan = {max_t - 1}")
print(f"SIC = {sum(len(p) for p in solution.values())}")
print("[RESULT] CBS SUCCEEDED")
sys.exit(0)
