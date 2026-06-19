import os
import sys

import time

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from common.environment  import generate_demand_map, generate_tasks, generate_uavs, generate_new_task
from scheduler    import assign_tasks
from rl_agent     import QLearningPlanner
from common.config import (
    SEED,
    UAV_SEED,
    NUM_TASKS,
    HIGH_PRIORITY_RATIO,
    NUM_UAVS,
    EPOCHS,
    ENABLE_DYNAMIC_EVENTS,
    UAV_SPEED,
    ENERGY_PER_METER
)

def distance_to(task, uav):
    return ((uav.curr_x - task.x)**2 + (uav.curr_y - task.y)**2)**0.5

def consume_resources(task, uav):
    travel_energy = distance_to(task, uav) * ENERGY_PER_METER
    travel_time   = distance_to(task,uav) / UAV_SPEED
    uav.remaining_energy     -= task.energy_cost + travel_energy
    uav.remaining_hover_time -= travel_time + task.hover_time
    uav.remaining_compute    -= task.compute_load


def move_to(task, uav):
    uav.curr_x = task.x
    uav.curr_y = task.y
    consume_resources(task, uav)

def is_feasible(task, uav):
    travel_energy = distance_to(task, uav) * ENERGY_PER_METER
    travel_time   = distance_to(task, uav) / UAV_SPEED
    remaining_energy = uav.remaining_energy
    remaining_hover_time = uav.remaining_hover_time
    remaining_compute = uav.remaining_compute
    remaining_energy     -= task.energy_cost + travel_energy
    remaining_hover_time -= travel_time + task.hover_time
    remaining_compute    -= task.compute_load

    if(remaining_energy < 0 or remaining_hover_time < 0 or remaining_compute < 0):
        return False
    
    current_time = uav.max_hover_time - uav.remaining_hover_time
    if(current_time + travel_time + task.hover_time > task.deadline):
        return False
    
    return True

def setup_environment(num_tasks=NUM_TASKS, num_uavs=NUM_UAVS):

    print("=" * 60)
    print("  DMMP-PR-TSA  |  UAV Remote Sensing Scheduler")
    print("=" * 60)

    print("\n[1] Generating sensing-demand map...")
    demand_map = generate_demand_map(seed=SEED)
    print(f"    Map size : {demand_map.shape[1]} × {demand_map.shape[0]} cells")

    print(f"\n[2] Sampling {num_tasks} tasks from demand map...")
    tasks, _ = generate_tasks(
        num_tasks           = num_tasks,
        high_priority_ratio = HIGH_PRIORITY_RATIO,
        demand_map          = demand_map,
        seed                = SEED,
    )
    p1 = sum(1 for t in tasks if t.priority == 1)
    p2 = sum(1 for t in tasks if t.priority == 2)
    p3 = sum(1 for t in tasks if t.priority == 3)
    print(f"    Tasks : {len(tasks)}  "
          f"(P1={p1}, P2={p2}, P3={p3})")

    print(f"\n[3] Generating {num_uavs} heterogeneous UAVs...")
    uavs = generate_uavs(num_uavs=num_uavs, seed=UAV_SEED)
    for uav in uavs:
        print(f"    {uav}")

    return demand_map, tasks, uavs

def run_scheduler(tasks, uavs):
    print('='*60)
    print("Running D-module Region Partioning")
    print('='*60)

    uavs, unassigned_tasks = assign_tasks(tasks, uavs)

    print(f"Total tasks assigned: {len(tasks) - len(unassigned_tasks)}/{len(tasks)}\n")

    print("Task Allocation:")
    for uav in uavs:
        print(f"{uav.uav_id}:{[task.task_id for task in uav.assigned_tasks]}")

    print("\nUnassigned Tasks:")
    if(len(unassigned_tasks) == 0):
        print("None\n")
    else:
        print(f"{[task.task_id for task in unassigned_tasks]}\n")

    for uav in uavs:
        compute = sum(t.compute_load for t in uav.assigned_tasks)
        hover = sum(t.hover_time for t in uav.assigned_tasks)
        energy = sum(t.energy_cost for t in uav.assigned_tasks)

        print(
            f"UAV {uav.uav_id}: "
            f"E={energy:.1f}/{uav.max_energy:.1f}, "
            f"H={hover:.1f}/{uav.max_hover_time:.1f}, "
            f"C={compute:.1f}/{uav.max_compute:.1f}"
        )
    return uavs, unassigned_tasks

def find_path(uavs):
    all_routes = []
    for uav in uavs:
        planner = QLearningPlanner(uav)
        planner.train(episodes = 500)
        route = planner.extract_route()
        all_routes.append(route)
        print(f"\nUAV {uav.uav_id} Route:")
        print([task.task_id for task in route])
    return all_routes

def main(num_tasks=NUM_TASKS, num_uavs=NUM_UAVS, optimize=True, save_dir=None, prefix=""):
    start = time.perf_counter()

    demand_map, tasks, uavs = setup_environment(num_tasks=num_tasks, num_uavs=num_uavs)
    uavs, unassigned_tasks = run_scheduler(tasks, uavs)
    all_routes = find_path(uavs)

    for uav, route in zip(uavs, all_routes):
        for task in route:
            if not is_feasible(task, uav):
                task.completed = False
                continue
            move_to(task, uav)
            task.completed = True
    
    completed_tasks = 0
    total_tasks = len(tasks)
    high_priority_completed_tasks = 0
    total_high_priority_tasks = 0
    for task in tasks:
        if task.completed:
            completed_tasks += 1
            if task.priority == 1:
                high_priority_completed_tasks += 1
        if task.priority == 1:
            total_high_priority_tasks += 1
    
    completion_rate = completed_tasks / total_tasks
    high_priority_completion_rate = high_priority_completed_tasks / total_high_priority_tasks
    runtime = time.perf_counter() - start

    print(f"Task Completion Rate = {completion_rate * 100}%")
    print(f"High Priority Task Completion Rate = {high_priority_completion_rate * 100}%")
    print(f"Total Runtime for the algorithm = {runtime}(secs)")

    metrics = {
        "completion_rate": completion_rate,
        "high_priority_completion_rate": high_priority_completion_rate,
        "runtime": runtime,
    }

    return metrics

if __name__ == '__main__':
    main(num_tasks=50,num_uavs=9)