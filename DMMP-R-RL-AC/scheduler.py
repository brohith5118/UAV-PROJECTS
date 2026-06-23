"""Resource-aware iterative region partitioning for DMMP-R-RL-AC."""

import math
import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.config import (  # noqa: E402
    ENERGY_PER_METER,
    GRID_RESOLUTION,
    ITERATIONS,
    MAP_HEIGHT,
    MAP_WIDTH,
    UAV_SPEED,
)


# All distance and resource terms are normalized before these weights are used.
REGION_WEIGHT = 4.0
BASE_DISTANCE_WEIGHT = 0.75
RESOURCE_WEIGHT = 3.0
LOAD_WEIGHT = 2.0
DRIFT_WEIGHT = 1.5
DEADLINE_WEIGHT = 2.5
PRIORITY_WEIGHT = 1.0
TYPE_WEIGHT = 20.0
SOFT_VIOLATION_WEIGHT = 50.0
INACTIVE_WEIGHT = 100.0

MAX_ITERATIONS = ITERATIONS
CENTROID_EPSILON = 1e-6
TOLERANCE = 1e-9

PRIORITY_VALUE = {1: 1.0, 2: 0.6, 3: 0.25}


def distance(x1, y1, x2, y2):
    """Euclidean distance between two positions."""
    return math.hypot(x1 - x2, y1 - y2)


def _uav_position(uav):
    """Use the current position when available, otherwise the depot position."""
    x = getattr(uav, "curr_x", getattr(uav, "x", 0.0))
    y = getattr(uav, "curr_y", getattr(uav, "y", 0.0))
    if not (math.isfinite(x) and math.isfinite(y)):
        return uav.x, uav.y
    return x, y


def distance_to(task, uav):
    """Distance from the UAV's current/depot position to a task."""
    x, y = _uav_position(uav)
    return distance(x, y, task.x, task.y)


def centroid_distance(task, uav):
    """Distance from a task to the UAV's current region centroid."""
    return distance(uav.centroid_x, uav.centroid_y, task.x, task.y)


def task_travel_cost(task, uav, state=None):
    """Return marginal travel distance, energy, and time for a task."""
    if state is None:
        from_x, from_y = _uav_position(uav)
    else:
        from_x, from_y = state["last_x"], state["last_y"]

    travel_distance = distance(from_x, from_y, task.x, task.y)
    return (
        travel_distance,
        travel_distance * ENERGY_PER_METER,
        travel_distance / UAV_SPEED,
    )


def _available_capacity(uav, resource):
    """Capacity available when partitioning starts, including residual state."""
    maximum = max(0.0, float(getattr(uav, f"max_{resource}", 0.0)))
    remaining = float(getattr(uav, f"remaining_{resource}", maximum))
    return min(maximum, max(0.0, remaining))


def _new_state(uav):
    x, y = _uav_position(uav)
    return {
        "energy": 0.0,
        "hover": 0.0,
        "compute": 0.0,
        "clock": 0.0,
        "count": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "last_x": x,
        "last_y": y,
    }


def _state_from_assignments(uav):
    """Build resource state for compatibility with direct helper calls."""
    state = _new_state(uav)
    for task in uav.assigned_tasks:
        _commit_to_state(task, uav, state)
    return state


def _is_compatible(task, uav):
    compatibility_check = getattr(uav, "is_compatible", None)
    if callable(compatibility_check):
        return compatibility_check(task)

    task_type = getattr(task, "task_type", -1)
    uav_type = getattr(uav, "uav_type", 1)
    if task_type == -1:
        return True
    if task_type == 0:
        return uav_type in (0, 1)
    if task_type == 1:
        return uav_type == 1
    return False


def _candidate_usage(task, uav, state):
    travel_distance, travel_energy, travel_time = task_travel_cost(
        task, uav, state
    )
    return {
        "travel_distance": travel_distance,
        "energy": state["energy"] + travel_energy + task.energy_cost,
        "hover": state["hover"] + travel_time + task.hover_time,
        "compute": state["compute"] + task.compute_load,
        "finish_time": state["clock"] + travel_time + task.hover_time,
    }


def is_feasible(task, uav, state=None):
    """Check activity, type, and cumulative travel/service resources."""
    if not getattr(uav, "active", True) or not _is_compatible(task, uav):
        return False

    state = _state_from_assignments(uav) if state is None else state
    usage = _candidate_usage(task, uav, state)

    if usage["energy"] > _available_capacity(uav, "energy") + TOLERANCE:
        return False
    if usage["hover"] > _available_capacity(uav, "hover_time") + TOLERANCE:
        return False
    if usage["compute"] > _available_capacity(uav, "compute") + TOLERANCE:
        return False

    # The D-module only creates regions; TSA may subsequently reorder each
    # region.  A deadline miss in this provisional insertion order therefore
    # belongs in the cost, rather than making an otherwise feasible task
    # permanently unassignable here.
    return True


def candidate_centroid(task, uav, state=None):
    """Centroid that would result if task were added to this UAV."""
    if state is None:
        count = len(uav.assigned_tasks)
        sum_x = sum(item.x for item in uav.assigned_tasks)
        sum_y = sum(item.y for item in uav.assigned_tasks)
    else:
        count = state["count"]
        sum_x = state["sum_x"]
        sum_y = state["sum_y"]

    return (sum_x + task.x) / (count + 1), (sum_y + task.y) / (count + 1)


def _ratio(used, capacity):
    if capacity <= TOLERANCE:
        return 0.0 if used <= TOLERANCE else math.inf
    return used / capacity


def _soft_ratio(used, capacity, zero_capacity_scale=20.0):
    """Finite pressure ratio used by D-module soft assignment costs."""
    if capacity > TOLERANCE:
        return used / capacity
    if used <= TOLERANCE:
        return 0.0
    return 1.0 + used / max(zero_capacity_scale, 1.0)


def estimate_assignment_cost(
    task,
    uav,
    state,
    expected_tasks_per_uav,
    reference_centroid=None,
):
    """Compute a finite cost, including penalties for soft violations."""

    usage = _candidate_usage(task, uav, state)
    map_diagonal = max(
        1.0,
        math.hypot(
            MAP_WIDTH * GRID_RESOLUTION,
            MAP_HEIGHT * GRID_RESOLUTION,
        ),
    )

    if reference_centroid is None:
        reference_centroid = (uav.centroid_x, uav.centroid_y)

    region_term = distance(
        reference_centroid[0], reference_centroid[1], task.x, task.y
    ) / map_diagonal
    base_term = distance_to(task, uav) / map_diagonal

    new_centroid = candidate_centroid(task, uav, state)
    base_x, base_y = _uav_position(uav)
    drift_term = distance(base_x, base_y, *new_centroid) / map_diagonal

    energy_ratio = _soft_ratio(
        usage["energy"], _available_capacity(uav, "energy")
    )
    hover_ratio = _soft_ratio(
        usage["hover"], _available_capacity(uav, "hover_time")
    )
    compute_capacity = _available_capacity(uav, "compute")
    compute_ratio = _soft_ratio(usage["compute"], compute_capacity)
    resource_term = (
        0.35 * energy_ratio**2
        + 0.35 * hover_ratio**2
        + 0.30 * compute_ratio**2
    )

    count_ratio = (state["count"] + 1) / max(expected_tasks_per_uav, 1.0)
    dominant_resource = max(energy_ratio, hover_ratio, compute_ratio)
    load_term = 0.6 * count_ratio**2 + 0.4 * dominant_resource**2

    deadline = max(float(getattr(task, "deadline", math.inf)), TOLERANCE)
    deadline_term = (
        0.0
        if math.isinf(deadline)
        else (usage["finish_time"] / deadline) ** 2
    )

    priority_value = PRIORITY_VALUE.get(getattr(task, "priority", 3), 0.25)
    mean_headroom = (
        (1.0 - energy_ratio) + (1.0 - hover_ratio) + (1.0 - compute_ratio)
    ) / 3.0
    priority_bonus = priority_value * max(0.0, mean_headroom)
    type_mismatch = 0.0 if _is_compatible(task, uav) else 1.0
    soft_violation = (
        max(0.0, energy_ratio - 1.0) ** 2
        + max(0.0, hover_ratio - 1.0) ** 2
        + max(0.0, compute_ratio - 1.0) ** 2
    )
    inactive_penalty = 0.0 if getattr(uav, "active", True) else 1.0

    return (
        REGION_WEIGHT * region_term
        + BASE_DISTANCE_WEIGHT * base_term
        + RESOURCE_WEIGHT * resource_term
        + LOAD_WEIGHT * load_term
        + DRIFT_WEIGHT * drift_term
        + DEADLINE_WEIGHT * deadline_term
        + TYPE_WEIGHT * type_mismatch
        + SOFT_VIOLATION_WEIGHT * soft_violation
        + INACTIVE_WEIGHT * inactive_penalty
        - PRIORITY_WEIGHT * priority_bonus
    )


def calculate_cost(task, uav):
    """Backward-compatible cost helper for the UAV's current assignments."""
    state = _state_from_assignments(uav)
    expected = max(1.0, state["count"] + 1.0)
    return estimate_assignment_cost(task, uav, state, expected)


def _commit_to_state(task, uav, state):
    usage = _candidate_usage(task, uav, state)
    state["energy"] = usage["energy"]
    state["hover"] = usage["hover"]
    state["compute"] = usage["compute"]
    state["clock"] = usage["finish_time"]
    state["count"] += 1
    state["sum_x"] += task.x
    state["sum_y"] += task.y
    state["last_x"] = task.x
    state["last_y"] = task.y


def update_centroids(uavs_or_task, uav=None):
    """Update region means, while accepting the legacy ``(task, uav)`` call."""
    if uav is not None:
        tasks = list(uav.assigned_tasks)
        task = uavs_or_task
        if task not in tasks:
            tasks.append(task)
        if tasks:
            uav.centroid_x = sum(item.x for item in tasks) / len(tasks)
            uav.centroid_y = sum(item.y for item in tasks) / len(tasks)
        return 0.0

    uavs = uavs_or_task
    total_shift = 0.0
    for uav in uavs:
        old_x, old_y = uav.centroid_x, uav.centroid_y
        if uav.assigned_tasks:
            count = len(uav.assigned_tasks)
            uav.centroid_x = sum(task.x for task in uav.assigned_tasks) / count
            uav.centroid_y = sum(task.y for task in uav.assigned_tasks) / count
        elif not (math.isfinite(old_x) and math.isfinite(old_y)):
            uav.centroid_x, uav.centroid_y = _uav_position(uav)

        total_shift += distance(old_x, old_y, uav.centroid_x, uav.centroid_y)
    return total_shift


def assignment_signature(uavs):
    """Stable representation used to detect unchanged partitions."""
    return tuple(
        (uav.uav_id, tuple(sorted(task.task_id for task in uav.assigned_tasks)))
        for uav in sorted(uavs, key=lambda item: item.uav_id)
    )


def _validate_complete_assignment(uavs, expected_tasks):
    """Remove duplicates and recover any missing task by minimum soft cost."""
    expected_by_id = {task.task_id: task for task in expected_tasks}
    seen = set()
    for uav in uavs:
        unique = []
        for task in uav.assigned_tasks:
            if task.task_id in expected_by_id and task.task_id not in seen:
                unique.append(task)
                seen.add(task.task_id)
        uav.assigned_tasks = unique

    missing = [
        task for task_id, task in expected_by_id.items() if task_id not in seen
    ]
    if missing and not uavs:
        raise ValueError("D-module cannot assign tasks because the UAV list is empty.")

    expected_count = len(expected_tasks) / max(len(uavs), 1)
    states = {uav.uav_id: _state_from_assignments(uav) for uav in uavs}
    for task in missing:
        best_uav = min(
            uavs,
            key=lambda uav: (
                estimate_assignment_cost(
                    task,
                    uav,
                    states[uav.uav_id],
                    expected_count,
                ),
                uav.uav_id,
            ),
        )
        best_uav.assigned_tasks.append(task)
        task.assigned_uav = best_uav
        _commit_to_state(task, best_uav, states[best_uav.uav_id])

    for uav in uavs:
        for task in uav.assigned_tasks:
            task.assigned_uav = uav

    assigned_ids = [
        task.task_id for uav in uavs for task in uav.assigned_tasks
    ]
    if len(assigned_ids) != len(expected_tasks) or len(set(assigned_ids)) != len(expected_tasks):
        raise RuntimeError("D-module assignment validation failed.")


def _task_order(task):
    """Urgent, early-deadline, and demanding tasks receive capacity first."""
    priority = getattr(task, "priority", 3)
    deadline = getattr(task, "deadline", math.inf)
    demand = task.energy_cost + task.hover_time + task.compute_load
    return priority, deadline, -demand, task.task_id


def assign_tasks(tasks, uavs):
    """Assign every task to one UAV using soft, resource-aware region costs."""
    tasks = list(tasks)
    active_uavs = [uav for uav in uavs if getattr(uav, "active", True)]

    for uav in uavs:
        uav.assigned_tasks.clear()
        uav.centroid_x, uav.centroid_y = _uav_position(uav)
    for task in tasks:
        task.assigned_uav = None

    if tasks and not uavs:
        raise ValueError("D-module requires at least one UAV for non-empty tasks.")
    candidate_uavs = active_uavs if active_uavs else list(uavs)
    if not tasks:
        return uavs, []

    ordered_tasks = sorted(tasks, key=_task_order)
    expected_tasks = len(tasks) / len(candidate_uavs)
    previous_signature = None

    for _iteration in range(MAX_ITERATIONS):
        reference_centroids = {
            uav.uav_id: (uav.centroid_x, uav.centroid_y)
            for uav in candidate_uavs
        }
        states = {uav.uav_id: _new_state(uav) for uav in candidate_uavs}

        for uav in uavs:
            uav.assigned_tasks.clear()
        for task in tasks:
            task.assigned_uav = None
        for task in ordered_tasks:
            candidates = []
            for uav in candidate_uavs:
                state = states[uav.uav_id]
                cost = estimate_assignment_cost(
                    task,
                    uav,
                    state,
                    expected_tasks,
                    reference_centroids[uav.uav_id],
                )
                candidates.append((cost, uav.uav_id, uav))

            _, _, best_uav = min(candidates, key=lambda item: (item[0], item[1]))
            best_uav.assigned_tasks.append(task)
            task.assigned_uav = best_uav
            _commit_to_state(task, best_uav, states[best_uav.uav_id])

        centroid_shift = update_centroids(candidate_uavs)
        signature = assignment_signature(uavs)
        if signature == previous_signature or centroid_shift <= CENTROID_EPSILON:
            break
        previous_signature = signature

    _validate_complete_assignment(uavs, tasks)
    update_centroids(uavs)

    # The empty second element preserves the existing pipeline return shape;
    # it is not an unassigned pool. D-module output is always total assignment.
    return uavs, []
