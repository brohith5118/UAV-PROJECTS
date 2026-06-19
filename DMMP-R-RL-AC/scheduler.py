import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.config import NUM_TASKS, UAV_SPEED, ENERGY_PER_METER, UAV_TYPE_MAX_FLIGHT, UAV_TYPE_MAX_COMPUTE   

ALPHA = 0.01
BETA = 300
GAMMA = 15.0
RHO = 0.05
MAX_ITERATIONS = 100

def distance_to(task, uav):
    return ((uav.x - task.x) ** 2 + (uav.y - task.y) ** 2) ** 0.5

def is_feasible(task, uav):
    distance = distance_to(task, uav)

    energy_cost = distance * ENERGY_PER_METER
    flight_time = distance / UAV_SPEED

    if(energy_cost + task.energy_cost > uav.max_energy):
        return False
    if(flight_time + task.hover_time > uav.max_hover_time):
        return False
    if(task.compute_load > uav.max_compute):
        return False
    
    return True

def calculate_cost(task, uav):
    if(is_feasible(task, uav)):
        distance = distance_to(task, uav)

        priority = task.priority

        compute_used = sum(
            t.compute_load
            for t in uav.assigned_tasks
        )

        compute_ratio = (
            compute_used /
            max(1, uav.max_compute)
        )

        energy_used = sum(
            t.energy_cost
            for t in uav.assigned_tasks
        )

        energy_ratio = (
            energy_used /
            max(1, uav.max_energy)
        )

        hover_used = sum(
            t.hover_time
            for t in uav.assigned_tasks
        )

        hover_ratio = (
            hover_used /
            max(1, uav.max_hover_time)
        )

        load_balance_penalty = compute_ratio + hover_ratio + energy_ratio

        lagrange_constraints = uav.mu_energy*task.energy_cost + uav.mu_hover*task.hover_time + uav.mu_compute*task.compute_load

        return ALPHA * distance - GAMMA * priority + lagrange_constraints + BETA * load_balance_penalty
        

    return float('inf')

def assign_tasks(tasks, uavs):
    unassigned_tasks = []

    for uav in uavs:
        uav.assigned_tasks.clear()

    unassigned_tasks = []

    for task in tasks:
        task.assigned_uav = None

    for task in tasks:
        best_cost = float('inf')
        best_uav = None
        for uav in uavs:
            cost = calculate_cost(task, uav)
            if cost < best_cost:
                best_cost = cost
                best_uav = uav
        if best_uav is not None:
            best_uav.assigned_tasks.append(task)
            task.assigned_uav = best_uav
        else:
            unassigned_tasks.append(task)

    return uavs, unassigned_tasks