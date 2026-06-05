# =========================================================
# UTILS  –  shared helper functions
#
# Covers:
#   • Euclidean geometry
#   • TSA reward function  R(s_j, a_j)  (eq 29)
#   • Mission metrics (completion rate, energy usage, etc.)
#   • Deadline compliance check
# =========================================================
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import math

from common.config import UAV_SPEED


# ----------------------------------------------------------
# GEOMETRY
# ----------------------------------------------------------

def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def task_distance(task_a, task_b):
    return math.hypot(task_a.x - task_b.x, task_a.y - task_b.y)


def uav_to_task_distance(uav, task):
    return math.hypot(uav.x - task.x, uav.y - task.y)


# ----------------------------------------------------------
# TSA REWARD FUNCTION  R(s_j, a_j)  (eq 29)
#
# R = c_d · d(TASK_j, TASK_j')
#   + c_p · pri_{j'}
#   + c_t · T^re_u / T^max_u
#   + c_c · (C^re_u − C_{TASK_j'}) / C^max_u
#
# c_d is negative → penalises long travel
# c_p is positive → rewards high-priority next task
# c_t is positive → rewards endurance preservation
# c_c is positive → rewards compute sufficiency
# ----------------------------------------------------------

def calculate_reward(
    current_task,
    next_task,
    uav,
    cd, cp, ct, cc,
):
    """
    TSA reward when transitioning from current_task to
    next_task on *uav*.

    Parameters
    ----------
    current_task : Task  (or a (x,y) tuple for the start position)
    next_task    : Task
    uav          : UAV
    cd, cp, ct, cc : reward coefficients from config
    """

    # Distance term  (cd is negative in config → penalty)
    if hasattr(current_task, 'x'):
        dist = task_distance(current_task, next_task)
    else:
        # current_task is a (x, y) position tuple
        dist = euclidean_distance(
            current_task[0], current_task[1],
            next_task.x, next_task.y
        )

    # Priority reward  (priority 1=critical → highest weight)
    priority_value = {1: 3.0, 2: 2.0, 3: 1.0}
    pri = priority_value.get(next_task.priority, 1.0)

    # Endurance-preservation term
    t_ratio = (
        uav.remaining_hover_time / uav.max_hover_time
        if uav.max_hover_time > 0 else 0.0
    )

    # Compute-sufficiency term
    c_margin = uav.remaining_compute - next_task.compute_load
    c_ratio  = (
        c_margin / uav.max_compute
        if uav.max_compute > 0 else 1.0
    )
    # Clip to avoid negative ratio breaking reward signal
    c_ratio  = max(c_ratio, -1.0)

    reward = (
        cd * dist
        + cp * pri
        + ct * t_ratio
        + cc * c_ratio
    )

    return max(-1000.0, min(1000.0, reward))


# ----------------------------------------------------------
# DEADLINE COMPLIANCE
# ----------------------------------------------------------

def check_deadline(task, finish_time):
    """True iff the task finished before its deadline."""
    return finish_time <= task.deadline


def estimate_finish_time(uav, route, uav_speed, start_time=0.0):
    """
    Estimate sequential finish times for each task in
    *route*, given the UAV starts at its current position
    at *start_time*.

    Returns a list of (task, finish_time) pairs.
    """
    current_x   = uav.x
    current_y   = uav.y
    clock       = start_time
    results     = []

    for task in route:
        travel  = euclidean_distance(
            current_x, current_y, task.x, task.y
        )
        clock  += travel / uav_speed   # travel
        clock  += task.hover_time      # on-site work
        results.append((task, clock))
        current_x = task.x
        current_y = task.y

    return results


def estimate_route_usage(uav, route, uav_speed, energy_per_meter):
    """
    Estimate sequential energy, hover-time, and compute usage for a route.
    This follows the actual execution order after TSA rather than charging
    each task independently from the depot.
    """
    current_x = uav.x
    current_y = uav.y
    total_energy = 0.0
    total_hover = 0.0
    total_compute = 0.0

    for task in route:
        dist = euclidean_distance(current_x, current_y, task.x, task.y)
        total_energy += dist * energy_per_meter + task.energy_cost
        total_hover += dist / uav_speed + task.hover_time
        total_compute += task.compute_load
        current_x = task.x
        current_y = task.y

    return total_energy, total_hover, total_compute


def recompute_route_resources(uav, route=None):
    """Reset residual UAV resources from the current TSA route order."""
    from config import UAV_SPEED, ENERGY_PER_METER

    route = uav.assigned_tasks if route is None else route
    energy, hover, compute = estimate_route_usage(
        uav,
        route,
        UAV_SPEED,
        ENERGY_PER_METER,
    )
    uav.remaining_energy = uav.max_energy - energy
    uav.remaining_hover_time = uav.max_hover_time - hover
    uav.remaining_compute = uav.max_compute - compute
    return energy, hover, compute


# ----------------------------------------------------------
# MISSION METRICS
# ----------------------------------------------------------

def completion_rate(uavs, all_tasks):
    """
    Fraction of tasks completed on time.
    Uses estimate_finish_time for each UAV's route.
    """

    total     = len(all_tasks)
    completed = 0

    for uav in uavs:
        if not uav.assigned_tasks:
            continue
        timeline = estimate_finish_time(
            uav, uav.assigned_tasks, UAV_SPEED
        )
        for task, ft in timeline:
            if check_deadline(task, ft):
                completed += 1

    return completed / total if total > 0 else 0.0


def high_priority_completion_rate(uavs,tasks):
    """Completion rate restricted to priority-1 tasks."""

    hi_total     = 0
    hi_completed = 0

    for task in tasks:
        if task.priority != 1:
            continue
        hi_total += 1

    for uav in uavs:

        timeline = estimate_finish_time(
            uav, uav.assigned_tasks, UAV_SPEED
        )
        for task, ft in timeline:
            if task.priority == 1 and check_deadline(task, ft):
                hi_completed += 1

    return hi_completed / hi_total if hi_total > 0 else 0.0


def total_travel_distance(uavs):
    """Sum of all inter-task distances across the fleet."""
    total = 0.0
    for uav in uavs:
        prev_x, prev_y = uav.x, uav.y
        for task in uav.assigned_tasks:
            total += euclidean_distance(
                prev_x, prev_y, task.x, task.y
            )
            prev_x, prev_y = task.x, task.y
    return total


def energy_utilisation(uavs):
    """Mean fraction of energy budget consumed."""
    if not uavs:
        return 0.0
    ratios = [
        1.0 - uav.remaining_energy / uav.max_energy
        for uav in uavs if uav.max_energy > 0
    ]
    return sum(ratios) / len(ratios) if ratios else 0.0


def compute_utilisation(uavs):
    """Mean fraction of compute budget consumed."""
    active = [u for u in uavs if u.max_compute > 0]
    if not active:
        return 0.0
    ratios = [
        1.0 - u.remaining_compute / u.max_compute
        for u in active
    ]
    return sum(ratios) / len(ratios)


def print_mission_metrics(uavs, all_tasks):
    cr  = completion_rate(uavs, all_tasks)
    hcr = high_priority_completion_rate(uavs, all_tasks)
    td  = total_travel_distance(uavs)
    eu  = energy_utilisation(uavs)
    cu  = compute_utilisation(uavs)

    print("\n=== MISSION METRICS ===")
    print(f"  Overall completion rate    : {cr  * 100:.1f}%")
    print(f"  High-priority completion   : {hcr * 100:.1f}%")
    print(f"  Total travel distance      : {td:.1f} m")
    print(f"  Mean energy utilisation    : {eu  * 100:.1f}%")
    print(f"  Mean compute utilisation   : {cu  * 100:.1f}%")
    print()
