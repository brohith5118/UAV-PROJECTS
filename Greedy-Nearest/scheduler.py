import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import math

from common.config import ENERGY_PER_METER, UAV_SPEED

def curr_distance(uav, task):
    current_x = getattr(uav, "curr_x", uav.x)
    current_y = getattr(uav, "curr_y", uav.y)
    return math.hypot(current_x - task.x, current_y - task.y)

def is_feasible(uav, task):
    if not uav.active:
        return False
    if uav.curr_energy < task.energy_cost:
        return False
    if uav.curr_hover < task.hover_time:
        return False
    if uav.curr_compute < task.compute_load:
        return False
    return True

def is_feasible_with_travel(uav, task, runtime):
    if not uav.active:
        return False
    travel_dist = curr_distance(uav, task)
    travel_energy_cost = travel_dist * ENERGY_PER_METER
    travel_hover_time = travel_dist / UAV_SPEED
    if(runtime + travel_hover_time + task.hover_time > task.deadline):
        return False
    if uav.curr_energy < task.energy_cost + travel_energy_cost:
        return False
    if uav.curr_hover < task.hover_time + travel_hover_time:
        return False
    if uav.curr_compute < task.compute_load:
        return False
    return True

def assign_tasks(tasks, uavs):
    tasklist = list(tasks)
    for task in tasks:
        best_uav = None
        nearest_dist = float('inf')
        for uav in uavs:
            if(curr_distance(uav,task) < nearest_dist):
                best_uav = uav
                nearest_dist = curr_distance(uav,task)
        if best_uav and is_feasible(best_uav,task):
            best_uav.assign(task)
            best_uav.compute_resource(task)
            tasklist.remove(task)

    return tasklist

def establish_path(tasks,uavs):
    for uav in uavs:
        task_list = []
        for i in range(len(uav.assigned_tasks)):
            nearest_task = None
            nearest_dist = float('inf')
            for task in uav.assigned_tasks:
                if(curr_distance(uav,task) < nearest_dist):
                    nearest_task = task
                    nearest_dist = curr_distance(uav,task)
            task_list.append(nearest_task)
            uav.assigned_tasks.remove(nearest_task)
            uav.move_to(nearest_task)
        
        uav.assigned_tasks = task_list

        uav.assigned_tasks.sort(
            key=lambda task: task.priority,
            reverse=True
        )

def run_path(uavs):
    for uav in uavs:
        uav.reset_position()
        uav.reset_resources()
        runtime = 0
        for task in uav.assigned_tasks:
            travel_dist = curr_distance(uav, task)
            travel_energy = travel_dist * ENERGY_PER_METER
            travel_time = travel_dist / UAV_SPEED
            if is_feasible_with_travel(uav, task,runtime):
                uav.remaining_energy -= task.energy_cost + travel_energy
                uav.remaining_hover_time -= task.hover_time + travel_time
                uav.remaining_compute -= task.compute_load
                runtime += task.hover_time + travel_time
                uav.move_to(task)
                task.completed = True
            else:
                task.completed = False
