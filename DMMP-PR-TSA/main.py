# =========================================================
# MAIN  –  DMMP-PR-TSA Pipeline
#
# Execution flow:
#
#   1. Generate sensing-demand map  (environment)
#   2. Sample tasks weighted by demand  (environment)
#   3. Generate heterogeneous UAV fleet  (environment)
#
#   4. [D-MODULE]  Capacity-constrained power-diagram
#      region partitioning  (scheduler)
#
#   5. [PR-MODULE]  SOM-based pre-assignment  (pr_module)
#
#   6. [TSA-MODULE]  Q-learning task sequence optimisation
#      per UAV  (rl_agent)
#
#   7. Simulate dynamic events:
#        (a) New urgent task insertion
#        (b) Task location update
#        (c) UAV failure
#      Re-run PR re-assignment + TSA after each event.
#
#   8. Print mission metrics
#   9. Visualise results
# =========================================================
import os
import sys

import time

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import random
import numpy as np

from common.environment  import generate_demand_map, generate_tasks, generate_uavs, generate_new_task
from scheduler    import assign_tasks
from pr_module    import (
    preassign,
    reassign_new_tasks,
    reassign_after_location_update,
    reassign_after_uav_failure,
    cancel_tasks,
)
from rl_agent     import run_tsa_for_fleet, QLearningTrajectoryPlanner
from utils        import print_mission_metrics
from visualization import plot_all, plot_reward_convergence

from common.config import (
    SEED,
    UAV_SEED,
    NUM_TASKS,
    HIGH_PRIORITY_RATIO,
    NUM_UAVS,
    EPOCHS,
    ENABLE_DYNAMIC_EVENTS,
)


# ----------------------------------------------------------
# SEED for reproducibility
# ----------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)


# ==========================================================
# STEP 1-3 : ENVIRONMENT SETUP
# ==========================================================

def setup_environment():

    print("=" * 60)
    print("  DMMP-PR-TSA  |  UAV Remote Sensing Scheduler")
    print("=" * 60)

    print("\n[1] Generating sensing-demand map...")
    demand_map = generate_demand_map(seed=SEED)
    print(f"    Map size : {demand_map.shape[1]} × {demand_map.shape[0]} cells")

    print(f"\n[2] Sampling {NUM_TASKS} tasks from demand map...")
    tasks, _ = generate_tasks(
        num_tasks           = NUM_TASKS,
        high_priority_ratio = HIGH_PRIORITY_RATIO,
        demand_map          = demand_map,
        seed                = SEED,
    )
    p1 = sum(1 for t in tasks if t.priority == 1)
    p2 = sum(1 for t in tasks if t.priority == 2)
    p3 = sum(1 for t in tasks if t.priority == 3)
    print(f"    Tasks : {len(tasks)}  "
          f"(P1={p1}, P2={p2}, P3={p3})")

    print(f"\n[3] Generating {NUM_UAVS} heterogeneous UAVs...")
    uavs = generate_uavs(num_uavs=NUM_UAVS, seed=UAV_SEED)
    for uav in uavs:
        print(f"    {uav}")

    return demand_map, tasks, uavs


# ==========================================================
# STEP 4 : D-MODULE  –  Region Partitioning
# ==========================================================

def run_d_module(tasks, uavs):

    print("\n[4] D-MODULE: Capacity-Constrained Region Partitioning")
    print("    Running power-diagram optimisation with Lagrange multipliers...")

    uavs, unassigned = assign_tasks(tasks, uavs)

    assigned_count = sum(len(u.assigned_tasks) for u in uavs)
    print(f"    Assigned : {assigned_count}/{len(tasks)} tasks")
    if unassigned:
        print(f"    Unassigned (capacity overflow): {len(unassigned)} tasks")

    return uavs, unassigned


# ==========================================================
# STEP 5 : PR-MODULE  –  SOM Pre-Assignment
# ==========================================================

def run_pr_module(tasks, uavs, optimize=True):

    print("\n[5] PR-MODULE: SOM Pre-Assignment")
    print("    Running SOM competitive learning...")

    # Reset task assignment metadata from D-module
    for t in tasks:
        t.assigned_uav = None

    uavs = preassign(tasks, uavs, optimize=optimize)

    return uavs


# ==========================================================
# STEP 6 : TSA-MODULE  –  RL Sequence Optimisation
# ==========================================================

def run_tsa_module(uavs, optimize=True):

    print("\n[6] TSA-MODULE: Q-Learning Task Sequence Adjustment")
    print(f"    Training {EPOCHS} episodes per UAV...")

    reward_logs = {}
    all_routes  = {}

    for uav in uavs:

        if not uav.active or not uav.assigned_tasks:
            all_routes[uav.uav_id]  = []
            reward_logs[uav.uav_id] = []
            continue

        print(f"    UAV {uav.uav_id:02d} "
              f"({len(uav.assigned_tasks)} tasks, "
              f"type {uav.uav_type:+d})...", end=' ')

        planner = QLearningTrajectoryPlanner(uav, uav.assigned_tasks, optimize=optimize)
        logs    = planner.train(epochs=EPOCHS, verbose=False)

        route   = planner.get_best_route()
        if optimize:
            route   = planner.reorder_by_deadline(route)

        uav.assigned_tasks = route
        all_routes[uav.uav_id]  = route
        reward_logs[uav.uav_id] = logs
        print(f"done  (best reward={planner._best_reward:.1f})")

    return all_routes, reward_logs


# ==========================================================
# STEP 7 : DYNAMIC EVENTS
# ==========================================================

def simulate_dynamic_events(tasks, uavs, demand_map, optimize=True):
    """
    Simulate the three representative dynamic events from
    the paper (Section 3.3) and re-run PR + TSA after each.
    Returns the final routes and a log of events.
    """

    if not ENABLE_DYNAMIC_EVENTS:
        print("\n[7] Dynamic events: DISABLED (set ENABLE_DYNAMIC_EVENTS=True)")
        routes, logs = run_tsa_module(uavs, optimize=optimize)
        return routes, logs, []

    event_log = []

    # -------------------------------------------------------
    # EVENT (a): New urgent task insertion
    # -------------------------------------------------------
    print("\n[7a] DYNAMIC EVENT: New urgent task insertion")
    next_id    = max(t.task_id for t in tasks) + 1
    new_tasks  = [
        generate_new_task(next_id + i, demand_map, seed=SEED + 100 + i)
        for i in range(3)
    ]
    for t in new_tasks:
        print(f"     Inserting {t}")

    uavs = reassign_new_tasks(new_tasks, uavs, optimize=optimize)
    tasks.extend(new_tasks)
    event_log.append(('new_task_insertion', len(new_tasks)))

    routes_a, logs_a = run_tsa_module(uavs, optimize=optimize)
    print("    TSA re-optimised after new task insertion.")

    # -------------------------------------------------------
    # EVENT (b): Task location update
    # -------------------------------------------------------
    print("\n[7b] DYNAMIC EVENT: Task location update")
    # Pick a random assigned task and shift its location
    assigned_tasks_flat = [
        t for u in uavs for t in u.assigned_tasks
    ]
    if assigned_tasks_flat:
        from config import MAP_WIDTH, MAP_HEIGHT, GRID_RESOLUTION
        map_max_x = MAP_WIDTH  * GRID_RESOLUTION
        map_max_y = MAP_HEIGHT * GRID_RESOLUTION

        update_target = random.choice(assigned_tasks_flat)
        old_pos = (update_target.x, update_target.y)
        # Shift by a small fraction of the map (≈3–8 % of width)
        shift_scale = map_max_x * 0.05
        update_target.x = min(
            update_target.x + random.uniform(shift_scale * 0.6, shift_scale * 1.6),
            map_max_x - 0.5,
        )
        update_target.y = min(
            update_target.y + random.uniform(shift_scale * 0.6, shift_scale * 1.6),
            map_max_y - 0.5,
        )
        print(
            f"     Task {update_target.task_id} "
            f"moved {old_pos} -> "
            f"({update_target.x:.1f},{update_target.y:.1f})"
        )
        uavs = reassign_after_location_update([update_target], uavs, optimize=optimize)
        event_log.append(('location_update', update_target.task_id))

    routes_b, logs_b = run_tsa_module(uavs, optimize=optimize)
    print("    TSA re-optimised after location update.")

    # -------------------------------------------------------
    # EVENT (c): UAV failure
    # -------------------------------------------------------
    print("\n[7c] DYNAMIC EVENT: UAV failure simulation")
    active_uavs = [u for u in uavs if u.active and u.assigned_tasks]
    if len(active_uavs) > 2:
        failed_uav = random.choice(active_uavs[1:])  # never fail UAV 0
        print(f"     UAV {failed_uav.uav_id} has failed!")
        uavs = reassign_after_uav_failure(failed_uav, uavs, optimize=optimize)
        event_log.append(('uav_failure', failed_uav.uav_id))

    routes_final, logs_final = run_tsa_module(uavs, optimize=optimize)
    print("    TSA re-optimised after UAV failure.")

    return routes_final, logs_final, event_log


# ==========================================================
# STEP 8 : METRICS
# ==========================================================

def report_metrics(uavs, tasks, routes):

    print("\n[8] MISSION METRICS")
    print_mission_metrics(uavs, tasks)

    # Per-UAV task count summary
    print("    Per-UAV task breakdown:")
    for uav in uavs:
        route = routes.get(uav.uav_id, [])
        tag   = "(FAILED)" if not uav.active else ""
        print(
            f"      UAV {uav.uav_id:02d} {tag}: "
            f"{len(uav.assigned_tasks)} assigned, "
            f"{len(route)} in final route"
        )


# ==========================================================
# STEP 9 : VISUALISE
# ==========================================================

def visualise(uavs, routes, tasks, demand_map, reward_logs, save_dir=None, prefix=""):

    print("\n[9] Generating visualisations...")

    plot_all(uavs, routes, tasks, demand_map, save_dir=save_dir, prefix=prefix)
    plot_reward_convergence(reward_logs, save_dir=save_dir, prefix=prefix)


# ==========================================================
# MAIN
# ==========================================================

def main(optimize=True, save_dir=None, prefix=""):
    import math
    from common.config import ENERGY_PER_METER, UAV_SPEED
    from utils import (
        completion_rate,
        high_priority_completion_rate,
        total_travel_distance,
        energy_utilisation,
        compute_utilisation,
    )
    start_time = time.perf_counter()
    # ---- Environment ----
    demand_map, tasks, uavs = setup_environment()

    # ---- D-Module ----
    uavs, _unassigned = run_d_module(tasks, uavs)

    # ---- PR-Module ----
    # uavs = run_pr_module(tasks, uavs, optimize=optimize)

    # ---- TSA + Dynamic Events ----
    routes, reward_logs, event_log = simulate_dynamic_events(
        tasks, uavs, demand_map, optimize=optimize
    )

    # ---- Metrics ----
    report_metrics(uavs, tasks, routes)

    # ---- Visualise ----
    visualise(uavs, routes, tasks, demand_map, reward_logs, save_dir=save_dir, prefix=prefix)

    # Compute metrics
    cr  = completion_rate(uavs, tasks)
    hcr = high_priority_completion_rate(uavs, tasks)
    td  = total_travel_distance(uavs)
    eu  = energy_utilisation(uavs)
    cu  = compute_utilisation(uavs)

    # Compute overloaded UAV count
    overloaded_count = 0
    for u in uavs:
        if not u.active:
            continue
        used_energy = sum(
            t.energy_cost + ENERGY_PER_METER * math.hypot(u.x - t.x, u.y - t.y)
            for t in u.assigned_tasks
        )
        used_hover = sum(
            t.hover_time + math.hypot(u.x - t.x, u.y - t.y) / UAV_SPEED
            for t in u.assigned_tasks
        )
        used_compute = sum(t.compute_load for t in u.assigned_tasks)
        if used_energy > u.max_energy or used_hover > u.max_hover_time or used_compute > u.max_compute:
            overloaded_count += 1

    # Jain's fairness index
    active_uavs = [u for u in uavs if u.active]
    x_i = [len(u.assigned_tasks) for u in active_uavs]
    if x_i and sum(x_i) > 0:
        n_active = len(x_i)
        jains_index = (sum(x_i) ** 2) / (n_active * sum(val ** 2 for val in x_i))
    else:
        jains_index = 0.0

    runtime = time.perf_counter() - start_time

    metrics = {
        "completion_rate": cr,
        "high_priority_completion_rate": hcr,
        "total_travel_distance": td,
        "energy_utilisation": eu,
        "compute_utilisation": cu,
        "overloaded_uav_count": overloaded_count,
        "jains_fairness_index": jains_index,
        "runtime": runtime,
    }

    return metrics


if __name__ == '__main__':
    main()