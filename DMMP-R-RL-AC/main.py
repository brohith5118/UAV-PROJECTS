"""Executable DMMP-R-RL-AC pipeline.

The folder is intentionally split into three stages:
    D  -> total initial allocation in ``scheduler.py``
    PR -> repair/drop-to-backup in ``pr_module.py``
    TSA -> per-UAV route ordering in ``rl_agent.py``
"""

import math
import os
import sys
import time


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.config import (  # noqa: E402
    ENERGY_PER_METER,
    EPOCHS,
    HIGH_PRIORITY_RATIO,
    NUM_TASKS,
    NUM_UAVS,
    SEED,
    UAV_SEED,
    UAV_SPEED,
)
from common.environment import generate_demand_map, generate_tasks, generate_uavs  # noqa: E402
from pr_module import find_overloaded_uavs, repair_assignments  # noqa: E402
from rl_agent import run_tsa_for_fleet  # noqa: E402
from scheduler import assign_tasks  # noqa: E402
from visualization import plot_all  # noqa: E402


TOLERANCE = 1e-9


def distance_between(x1, y1, x2, y2):
    """Euclidean distance used consistently for travel time and energy."""
    return math.hypot(x1 - x2, y1 - y2)


def distance_to(task, uav):
    """Distance from a UAV's current execution position to a task."""
    return distance_between(uav.curr_x, uav.curr_y, task.x, task.y)


def _route_profile(uav, route):
    """Pure no-return route estimate from the UAV depot/current start."""
    current_x = getattr(uav, "curr_x", uav.x)
    current_y = getattr(uav, "curr_y", uav.y)
    clock = 0.0
    profile = {
        "distance": 0.0,
        "energy": 0.0,
        "hover": 0.0,
        "compute": 0.0,
        "deadline_misses": 0,
    }

    for task in route:
        travel_distance = distance_between(current_x, current_y, task.x, task.y)
        travel_time = travel_distance / UAV_SPEED
        clock += travel_time + task.hover_time
        profile["distance"] += travel_distance
        profile["energy"] += travel_distance * ENERGY_PER_METER + task.energy_cost
        profile["hover"] += travel_time + task.hover_time
        profile["compute"] += task.compute_load
        if clock > task.deadline + TOLERANCE:
            profile["deadline_misses"] += 1
        current_x, current_y = task.x, task.y

    return profile


def _transition(task, uav, clock):
    travel_distance = distance_to(task, uav)
    travel_time = travel_distance / UAV_SPEED
    finish = clock + travel_time + task.hover_time
    return {
        "travel_distance": travel_distance,
        "travel_time": travel_time,
        "travel_energy": travel_distance * ENERGY_PER_METER,
        "start": clock + travel_time,
        "finish": finish,
    }


def consume_resources(task, uav, travel_energy=None, travel_time=None):
    """Deduct exactly one leg of travel plus the task workload."""
    if travel_energy is None or travel_time is None:
        transition = _transition(task, uav, 0.0)
        travel_energy = transition["travel_energy"]
        travel_time = transition["travel_time"]

    uav.remaining_energy -= task.energy_cost + travel_energy
    uav.remaining_hover_time -= travel_time + task.hover_time
    uav.remaining_compute -= task.compute_load


def move_to(task, uav, clock):
    """Execute one task if already deemed feasible and return finish time."""
    transition = _transition(task, uav, clock)
    consume_resources(
        task,
        uav,
        transition["travel_energy"],
        transition["travel_time"],
    )
    uav.curr_x = task.x
    uav.curr_y = task.y
    task.start_time = transition["start"]
    task.finish_time = transition["finish"]
    return transition["finish"]


def is_feasible(task, uav, clock):
    """Execution-time physical, type, and deadline feasibility check."""
    if not getattr(uav, "active", True):
        return False
    if not uav.is_compatible(task):
        return False

    transition = _transition(task, uav, clock)
    remaining_energy = (
        uav.remaining_energy - task.energy_cost - transition["travel_energy"]
    )
    remaining_hover = (
        uav.remaining_hover_time - transition["travel_time"] - task.hover_time
    )
    remaining_compute = uav.remaining_compute - task.compute_load

    if (
        remaining_energy < -TOLERANCE
        or remaining_hover < -TOLERANCE
        or remaining_compute < -TOLERANCE
    ):
        return False

    return transition["finish"] <= task.deadline + TOLERANCE


def validate_all_tasks_assigned_once(tasks, uavs):
    assigned_ids = [task.task_id for uav in uavs for task in uav.assigned_tasks]
    expected_ids = [task.task_id for task in tasks]
    if len(assigned_ids) != len(expected_ids):
        raise RuntimeError("D-module did not assign every task exactly once.")
    if set(assigned_ids) != set(expected_ids):
        raise RuntimeError("D-module assignment has missing or unknown tasks.")
    if len(set(assigned_ids)) != len(assigned_ids):
        raise RuntimeError("D-module assignment contains duplicate tasks.")


def validate_repaired_coverage(tasks, uavs, backup_pool):
    assigned_ids = {task.task_id for uav in uavs for task in uav.assigned_tasks}
    backup_ids = {task.task_id for task in backup_pool}
    expected_ids = {task.task_id for task in tasks}
    if assigned_ids & backup_ids:
        raise RuntimeError("PR-module left a task both assigned and in backup.")
    if assigned_ids | backup_ids != expected_ids:
        raise RuntimeError("PR-module lost task ownership during repair.")


def validate_route_contains_same_tasks(uav, route, original_tasks=None):
    source = uav.assigned_tasks if original_tasks is None else original_tasks
    assigned = sorted(task.task_id for task in source)
    planned = sorted(task.task_id for task in route)
    if assigned != planned:
        raise RuntimeError(f"TSA route mismatch for UAV {uav.uav_id}.")


def setup_environment(num_tasks=NUM_TASKS, num_uavs=NUM_UAVS):
    print("=" * 60)
    print("  DMMP-R-RL-AC  |  UAV Remote Sensing Scheduler")
    print("=" * 60)

    print("\n[1] Generating sensing-demand map...")
    demand_map = generate_demand_map(seed=SEED)
    print(f"    Map size : {demand_map.shape[1]} x {demand_map.shape[0]} cells")

    print(f"\n[2] Sampling {num_tasks} tasks from demand map...")
    tasks, _ = generate_tasks(
        num_tasks=num_tasks,
        high_priority_ratio=HIGH_PRIORITY_RATIO,
        demand_map=demand_map,
        seed=SEED,
    )
    p1 = sum(1 for task in tasks if task.priority == 1)
    p2 = sum(1 for task in tasks if task.priority == 2)
    p3 = sum(1 for task in tasks if task.priority == 3)
    print(f"    Tasks : {len(tasks)}  (P1={p1}, P2={p2}, P3={p3})")

    print(f"\n[3] Generating {num_uavs} heterogeneous UAVs...")
    uavs = generate_uavs(num_uavs=num_uavs, seed=UAV_SEED)
    for uav in uavs:
        print(f"    {uav}")

    return demand_map, tasks, uavs


def run_scheduler(tasks, uavs):
    print("=" * 60)
    print("Running D-module Region Partitioning")
    print("=" * 60)

    uavs, _legacy_empty = assign_tasks(tasks, uavs)
    validate_all_tasks_assigned_once(tasks, uavs)

    assigned_count = sum(len(uav.assigned_tasks) for uav in uavs)
    print(f"Total tasks assigned: {assigned_count}/{len(tasks)}\n")

    print("Task Allocation:")
    for uav in uavs:
        print(f"{uav.uav_id}:{[task.task_id for task in uav.assigned_tasks]}")

    print("\nD-module coverage validation: PASS (all tasks assigned exactly once)\n")

    for uav in uavs:
        profile = _route_profile(uav, uav.assigned_tasks)
        print(
            f"UAV {uav.uav_id}: "
            f"E={profile['energy']:.1f}/{uav.max_energy:.1f}, "
            f"H={profile['hover']:.1f}/{uav.max_hover_time:.1f}, "
            f"C={profile['compute']:.1f}/{uav.max_compute:.1f}"
        )
    return uavs, []


def run_pr_module(tasks, uavs):
    print("=" * 60)
    print("Running PR-module Resource-Aware Regret Repair")
    print("=" * 60)

    uavs, backup_pool = repair_assignments(uavs, tasks)
    validate_repaired_coverage(tasks, uavs, backup_pool)

    repaired_count = sum(len(uav.assigned_tasks) for uav in uavs)
    print(f"Tasks retained after repair: {repaired_count}/{len(tasks)}")
    print(f"Backup pool: {[task.task_id for task in backup_pool]}\n")
    return uavs, backup_pool


def find_path(uavs, epochs=EPOCHS, optimize=True, verbose=True):
    original_tasks = {
        uav.uav_id: list(uav.assigned_tasks)
        for uav in uavs
    }
    routes = run_tsa_for_fleet(
        uavs,
        epochs=epochs,
        verbose=False,
        optimize=optimize,
    )
    for uav in uavs:
        route = routes.get(uav.uav_id, [])
        validate_route_contains_same_tasks(
            uav,
            route,
            original_tasks[uav.uav_id],
        )
        if verbose:
            print(f"\nUAV {uav.uav_id} Route:")
            print([task.task_id for task in route])
    return routes


def _reset_execution_state(tasks, uavs):
    for task in tasks:
        task.completed = False
        task.start_time = None
        task.finish_time = None
    for uav in uavs:
        uav.reset_resources()
        uav.reset_position()


def _execute_routes(uavs, routes):
    for uav in uavs:
        clock = 0.0
        for task in routes.get(uav.uav_id, []):
            if not is_feasible(task, uav, clock):
                task.completed = False
                continue
            clock = move_to(task, uav, clock)
            task.completed = True


def _safe_mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _jains_fairness(loads):
    if not loads or sum(loads) <= TOLERANCE:
        return 0.0
    return (sum(loads) ** 2) / (len(loads) * sum(load * load for load in loads))


def _fleet_plan_metrics(uavs, routes):
    profiles = {
        uav.uav_id: _route_profile(uav, routes.get(uav.uav_id, []))
        for uav in uavs
    }
    active_uavs = [uav for uav in uavs if getattr(uav, "active", True)]
    return {
        "total_travel_distance": sum(p["distance"] for p in profiles.values()),
        "planned_deadline_misses": sum(
            p["deadline_misses"] for p in profiles.values()
        ),
        "estimated_energy_utilisation": _safe_mean(
            profiles[uav.uav_id]["energy"] / uav.max_energy
            for uav in active_uavs
            if uav.max_energy > 0
        ),
        "estimated_compute_utilisation": _safe_mean(
            profiles[uav.uav_id]["compute"] / uav.max_compute
            for uav in active_uavs
            if uav.max_compute > 0
        ),
        "jains_fairness_index": _jains_fairness(
            [len(routes.get(uav.uav_id, [])) for uav in active_uavs]
        ),
    }


def _execution_metrics(tasks, uavs):
    total_tasks = len(tasks)
    high_priority = [task for task in tasks if task.priority == 1]
    completed = [task for task in tasks if task.completed]
    high_completed = [
        task for task in high_priority if task.completed
    ]

    return {
        "completion_rate": len(completed) / total_tasks if total_tasks else 0.0,
        "high_priority_completion_rate": (
            len(high_completed) / len(high_priority) if high_priority else 0.0
        ),
        "energy_utilisation": _safe_mean(
            (uav.max_energy - uav.remaining_energy) / uav.max_energy
            for uav in uavs
            if uav.max_energy > 0
        ),
        "compute_utilisation": _safe_mean(
            (uav.max_compute - uav.remaining_compute) / uav.max_compute
            for uav in uavs
            if uav.max_compute > 0
        ),
    }


def main(num_tasks=NUM_TASKS, num_uavs=NUM_UAVS, optimize=True, save_dir=None, prefix=""):
    start = time.perf_counter()

    demand_map, tasks, uavs = setup_environment(
        num_tasks=num_tasks,
        num_uavs=num_uavs,
    )
    uavs, _legacy_empty = run_scheduler(tasks, uavs)
    uavs, backup_pool = run_pr_module(tasks, uavs)
    routes = find_path(uavs, epochs=EPOCHS, optimize=optimize)

    overloaded_count = len(
        find_overloaded_uavs(
            uavs,
            total_task_count=sum(len(uav.assigned_tasks) for uav in uavs),
        )
    )
    plan_metrics = _fleet_plan_metrics(uavs, routes)

    _reset_execution_state(tasks, uavs)
    _execute_routes(uavs, routes)

    metrics = {
        **plan_metrics,
        **_execution_metrics(tasks, uavs),
        "overloaded_uav_count": overloaded_count,
        "backup_pool_size": len(backup_pool),
        "runtime": time.perf_counter() - start,
    }

    plot_all(uavs, routes, tasks, demand_map, save_dir=save_dir, prefix=prefix)

    print(f"Task Completion Rate = {metrics['completion_rate'] * 100:.1f}%")
    print(
        "High Priority Task Completion Rate = "
        f"{metrics['high_priority_completion_rate'] * 100:.1f}%"
    )
    print(f"Total Runtime for the algorithm = {metrics['runtime']:.3f} secs")

    return metrics


if __name__ == "__main__":
    main(num_tasks=30, num_uavs=5)
