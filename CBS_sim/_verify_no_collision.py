"""Quick headless verification: run planning + check for pod collisions without GUI."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

SCENARIO = 10

import world as _world
_world.ACTIVE_CONFIG = str(SCENARIO)
_world._reinit()

from world import ROWS, COLS, STATIONS, OBSTACLES, build_agents_and_tasks
from task_assign import assign_tasks
from prioritized_planning import prioritized_plan

agents, tasks = build_agents_and_tasks()
assign_tasks(agents, tasks)

print("\n--- Planning (PP only) ---")
solution = prioritized_plan(agents=agents, rows=ROWS, cols=COLS, obstacles=OBSTACLES)

if solution is None:
    print("\n[ERROR] No solution found!")
    sys.exit(1)

max_t = max((len(a.path) for a in agents if a.path), default=0)
print(f"\nMakespan = {max_t - 1}")

# -- Check pod collisions --
from collections import defaultdict

collision_count = 0
for t in range(max_t):
    pod_positions = defaultdict(list)
    for a in agents:
        if a.task is None:
            continue
        if t <= a.fetch_end_t:
            pos = a.task.pod_pos
        elif t <= a.return_end_t:
            pos = a.path[t] if t < len(a.path) else a.path[-1]
        else:
            pos = a.task.pod_pos
        pod_positions[pos].append(a.agent_id)

    for pos, aids in pod_positions.items():
        if len(aids) > 1:
            ids_str = ", ".join(f"P{a}" for a in sorted(aids))
            print(f"[COLLISION] Tick {t:>3} | Pods {ids_str} collide at ({pos[0]}, {pos[1]})")
            collision_count += 1

if collision_count == 0:
    print("\n[OK] ZERO pod collisions detected!")
else:
    print(f"\n[FAIL] {collision_count} pod collision(s) detected!")
sys.exit(0 if collision_count == 0 else 1)
