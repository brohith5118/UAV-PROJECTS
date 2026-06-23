"""Resource-Aware Regret Repair (RARR) for D-module assignments."""

import math
import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.config import (  # noqa: E402
    ENERGY_PER_METER,
    GRID_RESOLUTION,
    MAP_HEIGHT,
    MAP_WIDTH,
    UAV_SPEED,
)


TOLERANCE = 1e-9
MAP_DIAGONAL = max(
    1.0,
    math.hypot(MAP_WIDTH * GRID_RESOLUTION, MAP_HEIGHT * GRID_RESOLUTION),
)

# Eviction score weights.
RESOURCE_EVICTION_WEIGHT = 4.0
DISRUPTION_EVICTION_WEIGHT = 2.0
DEADLINE_EVICTION_WEIGHT = 6.0
BOUNDARY_EVICTION_WEIGHT = 2.0
PRIORITY_PROTECTION_WEIGHT = 3.0
TYPE_MISMATCH_EVICTION_BONUS = 25.0

# Feasible insertion cost weights.
INSERT_DISTANCE_WEIGHT = 4.0
INSERT_RESOURCE_WEIGHT = 3.0
INSERT_DEADLINE_WEIGHT = 2.0
INSERT_LOAD_WEIGHT = 2.0
INSERT_REGION_WEIGHT = 1.5
INSERT_PRIORITY_WEIGHT = 0.5

LAST_BACKUP_POOL = []


def _distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def _position(uav):
    x = getattr(uav, "curr_x", getattr(uav, "x", 0.0))
    y = getattr(uav, "curr_y", getattr(uav, "y", 0.0))
    if math.isfinite(x) and math.isfinite(y):
        return x, y
    return uav.x, uav.y


def _available(uav, resource):
    maximum = max(0.0, float(getattr(uav, f"max_{resource}", 0.0)))
    remaining = float(getattr(uav, f"remaining_{resource}", maximum))
    return min(maximum, max(0.0, remaining))


def _ratio(used, capacity):
    if capacity <= TOLERANCE:
        return 0.0 if used <= TOLERANCE else math.inf
    return used / capacity


def _is_compatible(uav, task):
    check = getattr(uav, "is_compatible", None)
    if callable(check):
        return check(task)
    task_type = getattr(task, "task_type", -1)
    uav_type = getattr(uav, "uav_type", 1)
    if task_type == -1:
        return True
    if task_type == 0:
        return uav_type in (0, 1)
    if task_type == 1:
        return uav_type == 1
    return False


def find_centroid(uav):
    if not uav.assigned_tasks:
        return _position(uav)
    count = len(uav.assigned_tasks)
    return (
        sum(task.x for task in uav.assigned_tasks) / count,
        sum(task.y for task in uav.assigned_tasks) / count,
    )


def centroid_distance(uav, new_task):
    centroid_x, centroid_y = find_centroid(uav)
    return _distance(centroid_x, centroid_y, new_task.x, new_task.y)


def estimate_route(uav, route=None):
    """Return exact no-return route usage and deadline information."""
    route = list(uav.assigned_tasks if route is None else route)
    current_x, current_y = _position(uav)
    clock = max(
        0.0,
        float(getattr(uav, "max_hover_time", 0.0))
        - float(getattr(uav, "remaining_hover_time", 0.0)),
    )
    energy = hover = compute = total_distance = 0.0
    finishes = {}
    deadline_misses = []
    incompatible = []

    for task in route:
        travel = _distance(current_x, current_y, task.x, task.y)
        travel_time = travel / UAV_SPEED
        total_distance += travel
        energy += travel * ENERGY_PER_METER + task.energy_cost
        hover += travel_time + task.hover_time
        compute += task.compute_load
        clock += travel_time + task.hover_time
        finishes[task.task_id] = clock
        if clock > task.deadline + TOLERANCE:
            deadline_misses.append(task)
        if not _is_compatible(uav, task):
            incompatible.append(task)
        current_x, current_y = task.x, task.y

    return {
        "energy": energy,
        "hover": hover,
        "compute": compute,
        "clock": clock,
        "distance": total_distance,
        "finishes": finishes,
        "deadline_misses": deadline_misses,
        "incompatible": incompatible,
    }


def _resource_feasible(uav, profile):
    return (
        profile["energy"] <= _available(uav, "energy") + TOLERANCE
        and profile["hover"] <= _available(uav, "hover_time") + TOLERANCE
        and profile["compute"] <= _available(uav, "compute") + TOLERANCE
    )


def route_is_feasible(uav, route, require_deadlines=False):
    if not getattr(uav, "active", True):
        return False
    profile = estimate_route(uav, route)
    return (
        _resource_feasible(uav, profile)
        and not profile["incompatible"]
        and (not require_deadlines or not profile["deadline_misses"])
    )


def _deadline_overloaded(profile, route):
    misses = profile["deadline_misses"]
    high_priority_miss = any(task.priority == 1 for task in misses)
    tolerated_misses = max(1, math.ceil(0.20 * len(route)))
    return high_priority_miss or len(misses) > tolerated_misses


def _route_is_overloaded(uav, route, count_limit):
    profile = estimate_route(uav, route)
    return (
        not getattr(uav, "active", True)
        or not _resource_feasible(uav, profile)
        or bool(profile["incompatible"])
        or _deadline_overloaded(profile, route)
        or len(route) > count_limit
    )


def find_overloaded_uavs(uavs, total_task_count=None):
    active = [uav for uav in uavs if getattr(uav, "active", True)]
    total = (
        sum(len(uav.assigned_tasks) for uav in uavs)
        if total_task_count is None
        else total_task_count
    )
    expected = total / max(len(active), 1)
    count_limit = max(math.ceil(expected * 1.6), math.ceil(expected) + 2)
    return [
        uav
        for uav in uavs
        if _route_is_overloaded(uav, uav.assigned_tasks, count_limit)
    ]


def _lightweight_route(uav, tasks):
    """Priority/deadline-aware nearest-neighbour route estimate."""
    remaining = list(tasks)
    route = []
    current_x, current_y = _position(uav)
    clock = max(
        0.0,
        uav.max_hover_time - getattr(uav, "remaining_hover_time", uav.max_hover_time),
    )
    while remaining:
        def key(task):
            travel = _distance(current_x, current_y, task.x, task.y)
            finish = clock + travel / UAV_SPEED + task.hover_time
            return (
                int(finish > task.deadline),
                task.priority,
                task.deadline,
                travel,
                task.task_id,
            )

        selected = min(remaining, key=key)
        travel = _distance(current_x, current_y, selected.x, selected.y)
        clock += travel / UAV_SPEED + selected.hover_time
        route.append(selected)
        remaining.remove(selected)
        current_x, current_y = selected.x, selected.y
    return route


def calculate_resource_penalty(uav, new_task):
    """Backward-compatible post-insertion pressure estimate."""
    best_uav, _position_index, best_cost = find_best_feasible_insertion(
        new_task, [uav]
    )
    return best_cost if best_uav is not None else math.inf


def can_accept(uav, task):
    best_uav, _position_index, _cost = find_best_feasible_insertion(task, [uav])
    return best_uav is not None


def _route_disruption(uav, route, index):
    task = route[index]
    if index == 0:
        prev_x, prev_y = _position(uav)
    else:
        prev_x, prev_y = route[index - 1].x, route[index - 1].y
    next_task = route[index + 1] if index + 1 < len(route) else None
    added = _distance(prev_x, prev_y, task.x, task.y)
    if next_task is not None:
        added += _distance(task.x, task.y, next_task.x, next_task.y)
        added -= _distance(prev_x, prev_y, next_task.x, next_task.y)
    return max(0.0, added) / MAP_DIAGONAL


def _boundary_score(task, current_uav, uavs):
    current = centroid_distance(current_uav, task)
    alternatives = [
        centroid_distance(uav, task)
        for uav in uavs
        if uav is not current_uav and getattr(uav, "active", True)
    ]
    if not alternatives:
        return 0.0
    return min(2.0, current / (min(alternatives) + TOLERANCE))


def task_removal_score(task, uav, uavs, route=None):
    """Score high-impact, risky, boundary tasks for possible eviction."""
    route = list(uav.assigned_tasks if route is None else route)
    index = route.index(task)
    profile = estimate_route(uav, route)

    energy_contribution = (
        task.energy_cost
        + _route_disruption(uav, route, index) * MAP_DIAGONAL * ENERGY_PER_METER
    )
    hover_contribution = (
        task.hover_time
        + _route_disruption(uav, route, index) * MAP_DIAGONAL / UAV_SPEED
    )
    resource_contribution = (
        _ratio(energy_contribution, _available(uav, "energy"))
        + _ratio(hover_contribution, _available(uav, "hover_time"))
        + _ratio(task.compute_load, _available(uav, "compute"))
    )
    if not math.isfinite(resource_contribution):
        resource_contribution = 5.0

    finish = profile["finishes"].get(task.task_id, 0.0)
    deadline_risk = max(0.0, finish - task.deadline) / max(task.deadline, 1.0)
    boundary = _boundary_score(task, uav, uavs)
    priority_protection = 4 - getattr(task, "priority", 3)
    mismatch_bonus = 0.0 if _is_compatible(uav, task) else TYPE_MISMATCH_EVICTION_BONUS

    return (
        RESOURCE_EVICTION_WEIGHT * resource_contribution
        + DISRUPTION_EVICTION_WEIGHT * _route_disruption(uav, route, index)
        + DEADLINE_EVICTION_WEIGHT * deadline_risk
        + BOUNDARY_EVICTION_WEIGHT * boundary
        - PRIORITY_PROTECTION_WEIGHT * priority_protection
        + mismatch_bonus
    )


def select_tasks_to_evict(uav, uavs):
    route = list(uav.assigned_tasks)
    return sorted(
        route,
        key=lambda task: (
            task_removal_score(task, uav, uavs, route),
            task.priority,
            task.task_id,
        ),
        reverse=True,
    )


def _insertion_cost(task, uav, old_route, candidate_route, expected_load):
    before = estimate_route(uav, old_route)
    after = estimate_route(uav, candidate_route)
    extra_distance = max(0.0, after["distance"] - before["distance"])
    energy_ratio = _ratio(after["energy"], _available(uav, "energy"))
    hover_ratio = _ratio(after["hover"], _available(uav, "hover_time"))
    compute_ratio = _ratio(after["compute"], _available(uav, "compute"))
    pressure = (
        energy_ratio * energy_ratio
        + hover_ratio * hover_ratio
        + compute_ratio * compute_ratio
    ) / 3.0
    deadline_progress = sum(
        after["finishes"][item.task_id] / max(item.deadline, 1.0)
        for item in candidate_route
    ) / max(len(candidate_route), 1)
    deadline_risk = 0.0
    for item in after["deadline_misses"]:
        tardiness_ratio = (
            after["finishes"][item.task_id] - item.deadline
        ) / max(item.deadline, 1.0)
        deadline_risk += (4 - item.priority) * (2.0 + tardiness_ratio)
    deadline_pressure = deadline_progress + deadline_risk
    load_pressure = len(candidate_route) / max(expected_load, 1.0)
    region_pressure = centroid_distance(uav, task) / MAP_DIAGONAL
    priority_reward = 4 - getattr(task, "priority", 3)

    return (
        INSERT_DISTANCE_WEIGHT * extra_distance / MAP_DIAGONAL
        + INSERT_RESOURCE_WEIGHT * pressure
        + INSERT_DEADLINE_WEIGHT * deadline_pressure
        + INSERT_LOAD_WEIGHT * load_pressure * load_pressure
        + INSERT_REGION_WEIGHT * region_pressure
        - INSERT_PRIORITY_WEIGHT * priority_reward
    )


def _insertion_options(task, uavs):
    active = [uav for uav in uavs if getattr(uav, "active", True)]
    expected_load = (
        (sum(len(uav.assigned_tasks) for uav in active) + 1) / len(active)
        if active
        else 1.0
    )
    options = []
    for uav in active:
        if not _is_compatible(uav, task):
            continue
        route = list(uav.assigned_tasks)
        for position in range(len(route) + 1):
            candidate = route[:position] + [task] + route[position:]
            if not route_is_feasible(uav, candidate, require_deadlines=False):
                continue
            candidate_profile = estimate_route(uav, candidate)
            if _deadline_overloaded(candidate_profile, candidate):
                continue
            cost = _insertion_cost(task, uav, route, candidate, expected_load)
            options.append((cost, uav.uav_id, position, uav))
    return sorted(options, key=lambda item: (item[0], item[1], item[2]))


def find_best_feasible_insertion(task, uavs):
    options = _insertion_options(task, uavs)
    if not options:
        return None, None, math.inf
    cost, _uav_id, position, uav = options[0]
    return uav, position, cost


def _candidate_order(candidate_pool, uavs):
    scored = []
    for task in candidate_pool:
        options = _insertion_options(task, uavs)
        if len(options) >= 2:
            regret = options[1][0] - options[0][0]
        elif options:
            regret = 10.0
        else:
            regret = 20.0
        priority = 4 - getattr(task, "priority", 3)
        scored.append((priority, regret, -task.deadline, task.task_id, task))
    scored.sort(reverse=True)
    return [item[-1] for item in scored]


def update_centroids(uavs):
    for uav in uavs:
        if uav.assigned_tasks:
            uav.centroid_x, uav.centroid_y = find_centroid(uav)
        else:
            uav.centroid_x, uav.centroid_y = _position(uav)


def update_resource_usage(uavs):
    """Store PR estimates without consuming execution-time residual resources."""
    for uav in uavs:
        profile = estimate_route(uav)
        uav.estimated_energy_usage = profile["energy"]
        uav.estimated_hover_usage = profile["hover"]
        uav.estimated_compute_usage = profile["compute"]
        uav.estimated_deadline_misses = len(profile["deadline_misses"])


def repair_assignments(uavs, tasks=None):
    """Repair a D-module allocation and return ``(uavs, backup_pool)``."""
    active = [uav for uav in uavs if getattr(uav, "active", True)]
    all_tasks = list(
        tasks
        if tasks is not None
        else [task for uav in uavs for task in uav.assigned_tasks]
    )
    expected_load = len(all_tasks) / max(len(active), 1)
    count_limit = max(math.ceil(expected_load * 1.6), math.ceil(expected_load) + 2)
    candidate_pool = []
    backup_pool = []

    # Normalize route estimates and remove duplicate/missing ownership safely.
    seen = set()
    for uav in uavs:
        unique = []
        for task in uav.assigned_tasks:
            if task.task_id not in seen:
                unique.append(task)
                seen.add(task.task_id)
        uav.assigned_tasks = (
            _lightweight_route(uav, unique)
            if getattr(uav, "active", True)
            else unique
        )

    if tasks is not None:
        candidate_pool.extend(task for task in all_tasks if task.task_id not in seen)

    # Evict only until each source route is safe; failed UAVs release everything.
    for uav in uavs:
        if not getattr(uav, "active", True):
            candidate_pool.extend(
                task for task in uav.assigned_tasks if not getattr(task, "completed", False)
            )
            uav.assigned_tasks = []
            continue

        removals = 0
        max_removals = len(uav.assigned_tasks)
        while (
            uav.assigned_tasks
            and _route_is_overloaded(uav, uav.assigned_tasks, count_limit)
            and removals < max_removals
        ):
            ranked = select_tasks_to_evict(uav, uavs)
            task = ranked[0]
            uav.assigned_tasks.remove(task)
            task.assigned_uav = None
            candidate_pool.append(task)
            removals += 1
            uav.assigned_tasks = _lightweight_route(uav, uav.assigned_tasks)

    # Regret ordering protects tasks with few alternatives, with priority first.
    unique_candidates = {task.task_id: task for task in candidate_pool}
    for task in _candidate_order(list(unique_candidates.values()), uavs):
        best_uav, position, _cost = find_best_feasible_insertion(task, uavs)
        if best_uav is None:
            task.assigned_uav = None
            backup_pool.append(task)
            continue
        best_uav.assigned_tasks.insert(position, task)
        task.assigned_uav = best_uav

    update_centroids(uavs)
    update_resource_usage(uavs)
    global LAST_BACKUP_POOL
    LAST_BACKUP_POOL = list(backup_pool)
    return uavs, backup_pool


def pr_module(uavs, tasks=None):
    return repair_assignments(uavs, tasks)


def handle_new_task(new_task, uavs, backup_pool=None):
    backup_pool = [] if backup_pool is None else backup_pool
    best_uav, position, _cost = find_best_feasible_insertion(new_task, uavs)
    if best_uav is None:
        new_task.assigned_uav = None
        backup_pool.append(new_task)
    else:
        best_uav.assigned_tasks.insert(position, new_task)
        new_task.assigned_uav = best_uav
    update_centroids(uavs)
    update_resource_usage(uavs)
    return uavs, backup_pool


def handle_uav_failure(failed_uav, uavs, backup_pool=None):
    backup_pool = [] if backup_pool is None else backup_pool
    failed_uav.active = False
    candidates = [
        task
        for task in failed_uav.assigned_tasks
        if not getattr(task, "completed", False)
    ]
    failed_uav.assigned_tasks = []
    for task in _candidate_order(candidates, uavs):
        best_uav, position, _cost = find_best_feasible_insertion(task, uavs)
        if best_uav is None:
            task.assigned_uav = None
            backup_pool.append(task)
        else:
            best_uav.assigned_tasks.insert(position, task)
            task.assigned_uav = best_uav
    update_centroids(uavs)
    update_resource_usage(uavs)
    return uavs, backup_pool


def handle_task_location_change(
    task,
    old_position,
    new_position,
    uavs,
    backup_pool=None,
    movement_threshold=None,
):
    backup_pool = [] if backup_pool is None else backup_pool
    task.x, task.y = new_position
    movement = _distance(old_position[0], old_position[1], *new_position)
    threshold = 0.02 * MAP_DIAGONAL if movement_threshold is None else movement_threshold
    if movement <= threshold:
        update_centroids(uavs)
        update_resource_usage(uavs)
        return uavs, backup_pool

    owner = next((uav for uav in uavs if task in uav.assigned_tasks), None)
    if owner is not None:
        owner.assigned_tasks.remove(task)
    return handle_new_task(task, uavs, backup_pool)


def assign_new_task(tasks, uavs, new_task):
    if new_task not in tasks:
        tasks.append(new_task)
    return handle_new_task(new_task, uavs)


def preassign(tasks, uavs, optimize=True):
    """Legacy wrapper: ensure D allocation exists, then run PR repair."""
    assigned_ids = {task.task_id for uav in uavs for task in uav.assigned_tasks}
    if any(task.task_id not in assigned_ids for task in tasks):
        from scheduler import assign_tasks

        assign_tasks(tasks, uavs)
    repair_assignments(uavs, tasks)
    return uavs


def reassign_new_tasks(new_tasks, uavs, optimize=True):
    backup = []
    for task in new_tasks:
        uavs, backup = handle_new_task(task, uavs, backup)
    return uavs


def reassign_after_location_update(updated_tasks, uavs, optimize=True):
    return repair_assignments(uavs)[0]


def reassign_after_uav_failure(failed_uav, uavs, optimize=True):
    return handle_uav_failure(failed_uav, uavs)[0]


def cancel_tasks(tasks_to_cancel, uavs):
    cancelled_ids = {task.task_id for task in tasks_to_cancel}
    for uav in uavs:
        uav.assigned_tasks = [
            task for task in uav.assigned_tasks if task.task_id not in cancelled_ids
        ]
    update_centroids(uavs)
    update_resource_usage(uavs)
    return uavs
