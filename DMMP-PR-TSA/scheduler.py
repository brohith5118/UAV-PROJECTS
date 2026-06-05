# =========================================================
# SCHEDULER  –  D-MODULE
# =========================================================
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import math

from common.config import (
    ALPHA,
    GAMMA,
    RHO,
    LAMBDA_TV,
    ITERATIONS,
    ENERGY_PER_METER,
    UAV_SPEED,
    MAP_WIDTH,
    MAP_HEIGHT,
)


# ----------------------------------------------------------
# DISTANCE HELPERS
# ----------------------------------------------------------

def euclidean(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


# ----------------------------------------------------------
# SEQUENTIAL ROUTE RESOURCE ESTIMATION
# ----------------------------------------------------------

def estimate_route_cost(uav, tasks):

    prev_x = uav.x
    prev_y = uav.y

    total_energy = 0.0
    total_hover = 0.0
    total_compute = 0.0

    for task in tasks:

        dist = euclidean(prev_x, prev_y, task.x, task.y)

        travel_energy = dist * ENERGY_PER_METER
        travel_time = dist / UAV_SPEED

        total_energy += task.energy_cost + travel_energy
        total_hover += task.hover_time + travel_time
        total_compute += task.compute_load

        prev_x = task.x
        prev_y = task.y

    return total_energy, total_hover, total_compute


# ----------------------------------------------------------
# REGION CENTROID UPDATE
# ----------------------------------------------------------

def update_region_centroids(uavs):

    for uav in uavs:

        if not uav.assigned_tasks:
            continue

        n = len(uav.assigned_tasks)

        uav.region_x = (
            sum(t.x for t in uav.assigned_tasks) / n
        )

        uav.region_y = (
            sum(t.y for t in uav.assigned_tasks) / n
        )


# ----------------------------------------------------------
# FEASIBILITY CHECK
# ----------------------------------------------------------

def is_feasible(uav, task):
    """
    Returns True iff assigning *task* to *uav* keeps all
    three resource dimensions within their capacity limits.
    Includes travel overhead (energy + time to reach task).
    Also enforces flight-range constraint (eq 9) and
    type-compatibility constraint (eq 13).
    """

    if not uav.is_compatible(task):
        return False

    tentative_tasks = uav.assigned_tasks + [task]

    total_energy, total_hover, total_compute = (
        estimate_route_cost(uav, tentative_tasks)
    )
    used_hover   = sum(t.hover_time  for t in uav.assigned_tasks)
    used_compute = sum(t.compute_load for t in uav.assigned_tasks)

    if total_energy > uav.max_energy:
        return False

    if total_hover > uav.max_hover_time:
        return False

    if total_compute > uav.max_compute:
        return False

    return True


# ----------------------------------------------------------
# GENERALIZED COST FUNCTION
# ----------------------------------------------------------

def generalized_cost(uav, task, avg_tasks):

    distance = euclidean(
        uav.region_x,
        uav.region_y,
        task.x,
        task.y
    )

    # NORMALIZED DISTANCE
    map_diag = math.hypot(MAP_WIDTH, MAP_HEIGHT)
    norm_distance = distance / map_diag

    # NORMALIZED PRIORITY
    priority_reward = {
        1: 1.0,
        2: 0.6,
        3: 0.2
    }[task.priority]

    load_ratio = (len(uav.assigned_tasks) / max(avg_tasks, 1))

    load_penalty = load_ratio

    total_energy, total_hover, total_compute = (
        estimate_route_cost(
            uav,
            uav.assigned_tasks
        )
    )

    energy_ratio = total_energy / max(uav.max_energy, 1)

    hover_ratio = total_hover / max(uav.max_hover_time, 1)

    compute_ratio = (
        total_compute / max(uav.max_compute, 1)
        if uav.max_compute > 0
        else 0
    )

    resource_penalty = (
        energy_ratio
        + hover_ratio
        + compute_ratio
    )

    # LAGRANGE PENALTY
    lagrange_penalty = (
        uav.mu_energy * energy_ratio
        + uav.mu_hover * hover_ratio
        + uav.mu_compute * compute_ratio
    )


    total_cost = (
        ALPHA * norm_distance
        + 10.0 * load_penalty
        + 2.0 * resource_penalty
        + lagrange_penalty
        - GAMMA * priority_reward
    )

    return total_cost


# ----------------------------------------------------------
# TV REGULARIZATION
# ----------------------------------------------------------

def compactness_penalty(uav, task):

    if not uav.assigned_tasks:
        return 0.0

    n = len(uav.assigned_tasks)

    cx = sum(t.x for t in uav.assigned_tasks) / n
    cy = sum(t.y for t in uav.assigned_tasks) / n

    centroid_dist = euclidean(
        task.x,
        task.y,
        cx,
        cy
    )

    return LAMBDA_TV * centroid_dist


# ----------------------------------------------------------
# LAGRANGE MULTIPLIER UPDATE
# ----------------------------------------------------------

def update_lagrange_multipliers(uav):
    """
    Update all three Lagrange multipliers for UAV u.
    Usage includes travel overhead for energy / hover.
    """

    total_energy, total_hover, total_compute = (
        estimate_route_cost(
            uav,
            uav.assigned_tasks
        )
    )

    uav.mu_energy = max(
        0.0,
        uav.mu_energy +
        RHO * (total_energy - uav.max_energy)
    )

    uav.mu_hover = max(
        0.0,
        uav.mu_hover +
        RHO * (total_hover - uav.max_hover_time)
    )

    uav.mu_compute = max(
        0.0,
        uav.mu_compute +
        RHO * (total_compute - uav.max_compute)
    )


# ----------------------------------------------------------
# GREEDY WARM START ROUTE
# ----------------------------------------------------------

def nearest_neighbor_order(uav):

    if not uav.assigned_tasks:
        return

    remaining = uav.assigned_tasks[:]
    ordered = []

    current_x = uav.x
    current_y = uav.y

    while remaining:

        nearest = min(
            remaining,
            key=lambda t: euclidean(
                current_x,
                current_y,
                t.x,
                t.y
            )
        )

        ordered.append(nearest)

        current_x = nearest.x
        current_y = nearest.y

        remaining.remove(nearest)

    uav.assigned_tasks = ordered


# ----------------------------------------------------------
# MAIN PARTITIONING ALGORITHM
# ----------------------------------------------------------

def assign_tasks(task_list, uavs):
    """
    Capacity-Constrained Power-Diagram Partitioning.
    """
    # INITIAL RESET ONLY ONCE
    for uav in uavs:
        uav.clear_tasks()
        uav.mu_energy  = 0.0
        uav.mu_hover   = 0.0
        uav.mu_compute = 0.0

        uav.region_x = uav.x
        uav.region_y = uav.y

    sorted_tasks = sorted(
        task_list,
        key=lambda t: (
            t.priority,
            -(t.energy_cost +
              t.hover_time +
              t.compute_load)
        )
    )

    unassigned_tasks = []

    for iteration in range(ITERATIONS):

        # STORE OLD ASSIGNMENTS
        old_assignments = {
            u.uav_id: set(
                t.task_id for t in u.assigned_tasks
            )
            for u in uavs
        }

        # CLEAR FOR REFINEMENT
        for uav in uavs:
            uav.clear_tasks()

        unassigned_tasks = []

        avg_tasks = (
            len(task_list) / len(uavs)
            if uavs else 1
        )

        # ASSIGN TASKS
        for task in sorted_tasks:

            feasible_candidates = []

            for uav in uavs:

                if not uav.active:
                    continue

                if not is_feasible(uav, task):
                    continue

                cost = generalized_cost(
                    uav,
                    task,
                    avg_tasks
                )

                cost += compactness_penalty(
                    uav,
                    task
                )

                feasible_candidates.append(
                    (cost, uav)
                )

            if feasible_candidates:

                feasible_candidates.sort(
                    key=lambda x: x[0]
                )

                best_uav = feasible_candidates[0][1]

                best_uav.assigned_tasks.append(task)
                

            else:
                unassigned_tasks.append(task)

        # UPDATE CENTROIDS ONCE PER ITERATION
        update_region_centroids(uavs)

        # UPDATE LAGRANGE MULTIPLIERS
        for uav in uavs:
            update_lagrange_multipliers(uav)

        # CONVERGENCE CHECK
        converged = True

        for uav in uavs:
            update_lagrange_multipliers(uav)

            new_set = set(
                t.task_id
                for t in uav.assigned_tasks
            )

            if new_set != old_assignments[uav.uav_id]:
                converged = False
                break

        if converged:
            print(
                f"[D-Module] Converged "
                f"after {iteration+1} iterations"
            )
            break

    # GREEDY ROUTE INITIALIZATION
    for uav in uavs:
        nearest_neighbor_order(uav)

    # RESET RESOURCES
    for uav in uavs:
        uav.reset_resources()

    # APPLY FINAL RESOURCE CONSUMPTION
    for uav in uavs:

        total_energy, total_hover, total_compute = (
            estimate_route_cost(
                uav,
                uav.assigned_tasks
            )
        )

        uav.remaining_energy = (
            uav.max_energy - total_energy
        )

        uav.remaining_hover_time = (
            uav.max_hover_time - total_hover
        )

        uav.remaining_compute = (
            uav.max_compute - total_compute
        )

    _print_partitioning_summary(
        uavs,
        unassigned_tasks
    )

    return uavs, unassigned_tasks


# ----------------------------------------------------------
# DYNAMIC RE-PARTITIONING  (hysteresis-gated, post Section 5)
# ----------------------------------------------------------

def repartition_with_hysteresis(
    task_list,
    uavs,
    prev_objective,
    hysteresis_threshold=5.0,
    max_reassignments=10,
):
    """
    Re-runs partitioning only if the predicted objective
    improvement exceeds the hysteresis threshold ε.

    Uses a budget B = max_reassignments to limit churn.
    Returns (uavs, new_objective, reassigned_count).
    """

    # Compute current objective
    def objective(uavs_):
        total = 0.0
        for uav in uavs_:
            for t in uav.assigned_tasks:
                dist = math.hypot(uav.x - t.x, uav.y - t.y)
                total += ALPHA * dist - GAMMA * (
                    100 if t.priority == 1 else
                    60  if t.priority == 2 else 20
                )
        return total

    current_obj = objective(uavs)

    # Compute tentative new objective (single pass)
    test_uavs = [_clone_uav(u) for u in uavs]
    assign_tasks(task_list, test_uavs)
    new_obj = objective(test_uavs)

    delta_j = abs(new_obj - current_obj)

    if delta_j < hysteresis_threshold:
        return uavs, current_obj, 0   # no re-partition needed

    # Accept new partition, but cap reassignments
    reassigned = 0
    for u_old, u_new in zip(uavs, test_uavs):
        old_set = set(t.task_id for t in u_old.assigned_tasks)
        new_set = set(t.task_id for t in u_new.assigned_tasks)
        changes = len(old_set.symmetric_difference(new_set)) // 2
        if reassigned + changes > max_reassignments:
            break
        u_old.assigned_tasks = u_new.assigned_tasks
        reassigned += changes

    return uavs, new_obj, reassigned


def _clone_uav(uav):
    """Shallow clone for hysteresis testing."""
    from uav import UAV
    u2 = UAV(
        uav.uav_id,
        uav.x, uav.y,
        uav.uav_type,
        uav.max_energy,
        uav.max_hover_time,
        uav.max_compute,
    )
    u2.mu_energy  = uav.mu_energy
    u2.mu_hover   = uav.mu_hover
    u2.mu_compute = uav.mu_compute
    return u2


# ----------------------------------------------------------
# HELPER: PRINT SUMMARY
# ----------------------------------------------------------

def _print_partitioning_summary(uavs, unassigned):

    print("\n=== REGION PARTITIONING SUMMARY ===")

    for uav in uavs:

        te, th, tf = estimate_route_cost(
            uav,
            uav.assigned_tasks
        )
        th = sum(t.hover_time   for t in uav.assigned_tasks)
        tf = sum(t.compute_load for t in uav.assigned_tasks)

        hp = sum(
            1
            for t in uav.assigned_tasks
            if t.priority == 1
        )

        print(
            f"  UAV {uav.uav_id:02d} (type {uav.uav_type:+d}) | "
            f"tasks={len(uav.assigned_tasks):3d} "
            f"(hi-pri={hp}) | "
            f"E={te:6.1f}/{uav.max_energy:6.1f}J  "
            f"H={th:5.1f}/{uav.max_hover_time:5.1f}s  "
            f"F={tf:5.1f}/{uav.max_compute:5.1f}GHz·s"
        )

    if unassigned:
        print(
            f"\n  WARNING: {len(unassigned)} tasks could not be assigned"
            f" (insufficient fleet capacity)"
        )

        print(
            f"High Priority: {hp}"
        )

        print(
            f"Energy: "
            f"{te:.2f}/{uav.max_energy:.2f}"
        )

        print(
            f"Hover: "
            f"{th:.2f}/{uav.max_hover_time:.2f}"
        )

        print(
            f"Compute: "
            f"{tf:.2f}/{uav.max_compute:.2f}"
        )

    if unassigned:

        print(
            f"\nUnassigned Tasks: "
            f"{len(unassigned)}"
        )

    print()
