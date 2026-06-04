# =========================================================
# UTILS - shared helper functions
# =========================================================

import math

from config import MAP_WIDTH, MAP_HEIGHT, GRID_RESOLUTION


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
# TSA REWARD FUNCTION
# ----------------------------------------------------------

def calculate_reward(
    current_task,
    next_task,
    uav,
    cd, cp, ct, cc,
):
    """
    TSA transition reward with normalized feature magnitudes.

    The old implementation multiplied raw map distance by CD, which made
    distance dominate the reward by tens of thousands. Here distance is a
    0..1 map-diagonal ratio and the transition reward is clipped to a
    compact range before deadline shaping is applied in the TSA module.
    """

    if hasattr(current_task, "x"):
        dist = task_distance(current_task, next_task)
    else:
        dist = euclidean_distance(
            current_task[0],
            current_task[1],
            next_task.x,
            next_task.y,
        )

    map_diagonal = math.hypot(
        MAP_WIDTH * GRID_RESOLUTION,
        MAP_HEIGHT * GRID_RESOLUTION,
    )
    dist_ratio = min(dist / map_diagonal, 1.0) if map_diagonal > 0 else 0.0

    priority_value = {1: 1.0, 2: 0.65, 3: 0.35}
    pri = priority_value.get(next_task.priority, 0.35)

    t_ratio = (
        uav.remaining_hover_time / uav.max_hover_time
        if uav.max_hover_time > 0 else 0.0
    )

    c_margin = uav.remaining_compute - next_task.compute_load
    c_ratio = (
        c_margin / uav.max_compute
        if uav.max_compute > 0 else 1.0
    )
    c_ratio = max(-1.0, min(1.0, c_ratio))

    reward = (
        cd * dist_ratio
        + cp * pri
        + ct * t_ratio
        + cc * c_ratio
    )

    return max(-250.0, min(250.0, reward))


# ----------------------------------------------------------
# DEADLINE COMPLIANCE
# ----------------------------------------------------------

def check_deadline(task, finish_time):
    """True iff the task finished before its deadline."""
    return finish_time <= task.deadline


def estimate_finish_time(uav, route, uav_speed, start_time=0.0):
    """
    Estimate sequential finish times for a route from the UAV's depot.
    """
    current_x = uav.x
    current_y = uav.y
    clock = start_time
    results = []

    for task in route:
        travel = euclidean_distance(current_x, current_y, task.x, task.y)
        clock += travel / uav_speed
        clock += task.hover_time
        results.append((task, clock))
        current_x = task.x
        current_y = task.y

    return results


# ----------------------------------------------------------
# MISSION METRICS
# ----------------------------------------------------------

def completion_rate(uavs, all_tasks):
    """
    Fraction of tasks completed on time.
    """
    from config import UAV_SPEED

    total = len(all_tasks)
    completed = 0

    for uav in uavs:
        if not uav.assigned_tasks:
            continue
        timeline = estimate_finish_time(uav, uav.assigned_tasks, UAV_SPEED)
        for task, ft in timeline:
            if check_deadline(task, ft):
                completed += 1

    return completed / total if total > 0 else 0.0


def high_priority_completion_rate(uavs, tasks):
    """Completion rate restricted to priority-1 tasks."""
    from config import UAV_SPEED

    hi_total = 0
    hi_completed = 0

    for task in tasks:
        if task.priority == 1:
            hi_total += 1

    for uav in uavs:
        timeline = estimate_finish_time(uav, uav.assigned_tasks, UAV_SPEED)
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
            total += euclidean_distance(prev_x, prev_y, task.x, task.y)
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
    cr = completion_rate(uavs, all_tasks)
    hcr = high_priority_completion_rate(uavs, all_tasks)
    td = total_travel_distance(uavs)
    eu = energy_utilisation(uavs)
    cu = compute_utilisation(uavs)

    print("\n=== MISSION METRICS ===")
    print(f"  Overall completion rate    : {cr * 100:.1f}%")
    print(f"  High-priority completion   : {hcr * 100:.1f}%")
    print(f"  Total travel distance      : {td:.1f} m")
    print(f"  Mean energy utilisation    : {eu * 100:.1f}%")
    print(f"  Mean compute utilisation   : {cu * 100:.1f}%")
    print()
