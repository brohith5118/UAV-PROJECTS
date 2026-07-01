import os
import sys
import math
import random

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.config import ENERGY_PER_METER, UAV_SPEED


def current_position(uav):
    x = getattr(uav, "curr_x", uav.x)
    y = getattr(uav, "curr_y", uav.y)
    return x, y


def curr_distance(uav, task):
    ux, uy = current_position(uav)
    return math.hypot(ux - task.x, uy - task.y)


def get_remaining_energy(uav):
    return getattr(
        uav,
        "remaining_energy",
        getattr(uav, "curr_energy", uav.max_energy)
    )


def get_remaining_hover(uav):
    return getattr(
        uav,
        "remaining_hover_time",
        getattr(uav, "curr_hover", uav.max_hover_time)
    )


def get_remaining_compute(uav):
    return getattr(
        uav,
        "remaining_compute",
        getattr(uav, "curr_compute", uav.max_compute)
    )


def set_remaining_resources(uav, energy, hover, compute):
    if hasattr(uav, "remaining_energy"):
        uav.remaining_energy = energy

    if hasattr(uav, "remaining_hover_time"):
        uav.remaining_hover_time = hover

    if hasattr(uav, "remaining_compute"):
        uav.remaining_compute = compute

    if hasattr(uav, "curr_energy"):
        uav.curr_energy = energy

    if hasattr(uav, "curr_hover"):
        uav.curr_hover = hover

    if hasattr(uav, "curr_compute"):
        uav.curr_compute = compute


def reset_uav(uav):
    if hasattr(uav, "reset_position"):
        uav.reset_position()
    else:
        uav.curr_x = uav.x
        uav.curr_y = uav.y

    if hasattr(uav, "reset_resources"):
        uav.reset_resources()
    else:
        set_remaining_resources(
            uav,
            uav.max_energy,
            uav.max_hover_time,
            uav.max_compute
        )

    uav.assigned_tasks = []
    uav.runtime = 0.0


def is_feasible_with_travel(uav, task):
    if not getattr(uav, "active", True):
        return False

    runtime = getattr(uav, "runtime", 0.0)

    distance = curr_distance(uav, task)
    travel_energy = distance * ENERGY_PER_METER
    travel_time = distance / UAV_SPEED

    required_energy = task.energy_cost + travel_energy
    required_hover = task.hover_time + travel_time
    required_compute = task.compute_load

    completion_time = runtime + travel_time + task.hover_time

    if completion_time > task.deadline:
        return False

    if get_remaining_energy(uav) < required_energy:
        return False

    if get_remaining_hover(uav) < required_hover:
        return False

    if get_remaining_compute(uav) < required_compute:
        return False

    return True


def execute_task(uav, task):
    distance = curr_distance(uav, task)
    travel_energy = distance * ENERGY_PER_METER
    travel_time = distance / UAV_SPEED

    new_energy = get_remaining_energy(uav) - task.energy_cost - travel_energy
    new_hover = get_remaining_hover(uav) - task.hover_time - travel_time
    new_compute = get_remaining_compute(uav) - task.compute_load

    set_remaining_resources(
        uav,
        new_energy,
        new_hover,
        new_compute
    )

    uav.runtime = getattr(uav, "runtime", 0.0) + travel_time + task.hover_time

    if hasattr(uav, "move_to"):
        uav.move_to(task)
    else:
        uav.curr_x = task.x
        uav.curr_y = task.y

    uav.assigned_tasks.append(task)
    task.completed = True


def run_random_baseline(tasks, uavs, seed=None):
    """
    Random baseline scheduler.

    This algorithm does not optimize task assignment.
    It randomly orders tasks and randomly tries UAVs for each task.
    A task is assigned only if the selected UAV can complete it feasibly.

    Returns:
        remaining_tasks: list of tasks that could not be completed
    """

    if seed is not None:
        random.seed(seed)

    for task in tasks:
        task.completed = False

    for uav in uavs:
        reset_uav(uav)

    shuffled_tasks = list(tasks)
    random.shuffle(shuffled_tasks)

    remaining_tasks = []

    for task in shuffled_tasks:
        shuffled_uavs = list(uavs)
        random.shuffle(shuffled_uavs)

        assigned = False

        for uav in shuffled_uavs:
            if is_feasible_with_travel(uav, task):
                execute_task(uav, task)
                assigned = True
                break

        if not assigned:
            remaining_tasks.append(task)

    return remaining_tasks


# Optional alias if your main.py currently imports assign_tasks
def assign_tasks(tasks, uavs):
    return run_random_baseline(tasks, uavs)


# Optional alias if your main.py currently imports run_greedy_nearest
def run_greedy_nearest(tasks, uavs):
    return run_random_baseline(tasks, uavs)