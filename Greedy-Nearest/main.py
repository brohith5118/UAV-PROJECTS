import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.environment import generate_demand_map,generate_tasks,generate_uavs
from visualization import plot_demand_map, plot_uavs
from scheduler import assign_tasks, establish_path, run_path

from common.config import (
    SEED,
    UAV_SEED,
    NUM_TASKS,
    HIGH_PRIORITY_RATIO,
    NUM_UAVS
)


def setup_environment():
    demand_map = generate_demand_map(SEED)
    print(f"Generated Map with seed {SEED}\n")
    tasks = generate_tasks(NUM_TASKS,HIGH_PRIORITY_RATIO,demand_map,SEED)
    print(f"Generated {NUM_TASKS} tasks according to seed {SEED}\n")
    uavs = generate_uavs(NUM_UAVS, UAV_SEED)
    print(f"Generated {NUM_UAVS} UAVs according to seed {UAV_SEED}\n")

    return demand_map, tasks, uavs

demand_map, tasks, uavs = setup_environment()
plot_demand_map(demand_map, tasks)

def schedule_tasks(tasks, uavs):
    unassigned_tasks = assign_tasks(tasks, uavs)
    return unassigned_tasks

unassigned_tasks = schedule_tasks(tasks, uavs)
print(f"Number of unassigned tasks: {len(unassigned_tasks)}")
print(f"Unassigned Tasks: {[task.task_id for task in unassigned_tasks]}")

for uav in uavs:
    print(f"UAV {uav.uav_id} assigned tasks: {[task.task_id for task in uav.assigned_tasks]}")

establish_path(tasks, uavs)
print("\nEstablished paths for UAVs based on assigned tasks and priorities.\n")
for uav in uavs:
    print(f"UAV {uav.uav_id} path: {[task.task_id for task in uav.assigned_tasks]}")

def run_path_results(uavs):
    run_path(uavs)
    print("\nExecuted paths for UAVs. Here are the results:\n")
    for uav in uavs:
        print(f"UAV {uav.uav_id} tasks executed successfully: {[task.task_id for task in uav.assigned_tasks if task.completed]}")
        print(f"UAV {uav.uav_id} tasks failed: {[task.task_id for task in uav.assigned_tasks if not task.completed]}")

run_path_results(uavs)

def visualize_results(demand_map, tasks, uavs):
    plot_demand_map(demand_map, tasks)
    print("\nVisualized demand map with task completion status.\n")

    plot_uavs(uavs)
    print("\nVisualized UAVs and their locations")
    

visualize_results(demand_map, tasks, uavs)

def print_summary(uavs,tasks):
    completed_tasks = 0
    total_tasks = 0
    for task in tasks:
        if(task.completed == True):
            completed_tasks += 1
        total_tasks += 1
    print(f"Overall completion rate: {completed_tasks * 100/total_tasks}%")

print_summary(uavs,tasks)