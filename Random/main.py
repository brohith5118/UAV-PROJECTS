import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.environment import generate_demand_map,generate_tasks,generate_uavs
from visualization import plot_demand_map, plot_uavs
from scheduler import run_greedy_nearest

from common.config import (
    SEED,
    UAV_SEED,
    NUM_TASKS,
    HIGH_PRIORITY_RATIO,
    NUM_UAVS
)

import time


def setup_environment(num_tasks=None, num_uavs=None, high_priority_ratio=None):
    demand_map = generate_demand_map(SEED)
    print(f"Generated Map with seed {SEED}\n")
    tasks, _ = generate_tasks(
        num_tasks if num_tasks is not None else NUM_TASKS,
        high_priority_ratio if high_priority_ratio is not None else HIGH_PRIORITY_RATIO,
        demand_map,
        SEED
    )
    print(f"Generated {len(tasks)} tasks according to seed {SEED}\n")
    uavs = generate_uavs(
        num_uavs if num_uavs is not None else NUM_UAVS,
        UAV_SEED
    )
    print(f"Generated {NUM_UAVS} UAVs according to seed {UAV_SEED}\n")

    return demand_map, tasks, uavs


# def schedule_tasks(tasks, uavs):
#     unassigned_tasks = assign_tasks(tasks, uavs)
#     return unassigned_tasks


# def run_path_results(uavs):
#     run_path(uavs)
#     print("\nExecuted paths for UAVs. Here are the results:\n")
#     for uav in uavs:
#         print(f"UAV {uav.uav_id} tasks executed successfully: {[task.task_id for task in uav.assigned_tasks if task.completed]}")
#         print(f"UAV {uav.uav_id} tasks failed: {[task.task_id for task in uav.assigned_tasks if not task.completed]}")

def visualize_results(demand_map, tasks, uavs):
    # plot_demand_map(demand_map, tasks)
    print("\nVisualized demand map with task completion status.\n")

    # plot_uavs(uavs)
    print("\nVisualized UAVs and their locations")
    

def print_summary(uavs,tasks):
    completed_tasks = 0
    total_tasks = 0
    for task in tasks:
        if(task.completed == True):
            completed_tasks += 1
        total_tasks += 1
    print(f"Overall completion rate: {completed_tasks * 100/total_tasks}%")


def calculate_return_metrics(uavs, tasks, runtime):
    total_tasks = len(tasks)
    completed_tasks = [task for task in tasks if task.completed]
    high_priority_tasks = [task for task in tasks if task.priority == 1]
    high_priority_completed = [
        task for task in completed_tasks
        if task.priority == 1
    ]
    active_uavs = [uav for uav in uavs if getattr(uav, "active", True)]
    workloads = [len(uav.assigned_tasks) for uav in active_uavs]
    if workloads and sum(workloads) > 0:
        fairness = (sum(workloads) ** 2) / (
            len(workloads) * sum(count ** 2 for count in workloads)
        )
    else:
        fairness = 0.0

    energy_ratios = [
        1.0 - (uav.remaining_energy / uav.max_energy)
        for uav in active_uavs
        if uav.max_energy > 0
    ]
    compute_ratios = [
        1.0 - (uav.remaining_compute / uav.max_compute)
        for uav in active_uavs
        if uav.max_compute > 0
    ]

    return {
        "completion_rate": (
            len(completed_tasks) / total_tasks
            if total_tasks else 0.0
        ),
        "high_priority_completion_rate": (
            len(high_priority_completed) / len(high_priority_tasks)
            if high_priority_tasks else 0.0
        ),
        "energy_utilisation": (
            sum(energy_ratios) / len(energy_ratios)
            if energy_ratios else 0.0
        ),
        "compute_utilisation": (
            sum(compute_ratios) / len(compute_ratios)
            if compute_ratios else 0.0
        ),
        "overloaded_uav_count": sum(
            1
            for uav in active_uavs
            if (
                uav.remaining_energy < -1e-9
                or uav.remaining_hover_time < -1e-9
                or uav.remaining_compute < -1e-9
            )
        ),
        "jains_fairness_index": fairness,
        "runtime": runtime,
        "runtime_seconds": runtime,
    }

def main(num_tasks=NUM_TASKS, num_uavs=NUM_UAVS, high_priority_ratio=HIGH_PRIORITY_RATIO):
    start = time.perf_counter()

    demand_map, tasks, uavs = setup_environment(
        num_tasks=num_tasks,
        num_uavs=num_uavs,
        high_priority_ratio=high_priority_ratio
    )

    unassigned_tasks = run_greedy_nearest(tasks, uavs)

    runtime = time.perf_counter() - start

    print(f"Number of unassigned/failed tasks: {len(unassigned_tasks)}")
    print(f"Unassigned/failed Tasks: {[task.task_id for task in unassigned_tasks]}")

    print("\nGreedy Nearest Paths:\n")
    for uav in uavs:
        completed_ids = [
            task.task_id
            for task in uav.assigned_tasks
            if task.completed
        ]

        print(f"UAV {uav.uav_id} path: {[task.task_id for task in uav.assigned_tasks]}")
        print(f"UAV {uav.uav_id} completed tasks: {completed_ids}")
        print(
            f"UAV {uav.uav_id} remaining energy: {uav.remaining_energy:.2f}, "
            f"remaining hover: {uav.remaining_hover_time:.2f}, "
            f"remaining compute: {uav.remaining_compute:.2f}"
        )
        print()

    visualize_results(demand_map, tasks, uavs)
    print_summary(uavs, tasks)

    return calculate_return_metrics(uavs, tasks, runtime)

if __name__ == "__main__":
    results = main(70, 5)
    print("\nSummary of Results:")
    print(f"Overall Task Completion Rate: {results['completion_rate']:.2%}")
    print(f"High Priority Task Completion Rate: {results['high_priority_completion_rate']:.2%}")
    print(f"Total Runtime: {results['runtime']:.2f} seconds")
