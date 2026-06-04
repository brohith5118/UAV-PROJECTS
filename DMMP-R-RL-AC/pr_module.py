# =========================================================
# PR-MODULE  –  Resource-Constrained K-Means++ Pre/Re-Assignment
#
# Replaces the SOM competitive learning with an extremely
# efficient, capacity-aware K-Means++ clustering algorithm.
# =========================================================

import sys
import os
import math
import numpy as np

# Add parent directory to path so we can import root modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import (
    UAV_SPEED,
    ENERGY_PER_METER,
    UAV_TYPE_MAX_FLIGHT,
    UAV_TYPE_MAX_COMPUTE,
    MAP_WIDTH,
    MAP_HEIGHT
)


def rc_kmeans_assign(tasks, uavs, base_x=0.0, base_y=0.0):
    """
    Proposed Resource-Constrained K-Means++ Task Assignment.
    
    Assigns tasks to UAVs by iteratively clustering around centroids,
    while dynamically checking and balancing remaining resources.
    """
    active_uavs = [u for u in uavs if u.active]
    if not active_uavs or not tasks:
        return {t.task_id: None for t in tasks}

    # Initialize centroids of active UAVs to their starting positions
    centroids = {u.uav_id: (u.x, u.y) for u in active_uavs}
    
    # We run 15 iterations of KMeans clustering
    num_iterations = 15
    best_assignment = {t.task_id: None for t in tasks}
    best_assigned_count = -1

    for iter_idx in range(num_iterations):
        # Track simulated remaining capacities starting from their current states
        sim_remaining_energy = {u.uav_id: u.remaining_energy for u in active_uavs}
        sim_remaining_hover = {u.uav_id: u.remaining_hover_time for u in active_uavs}
        sim_remaining_compute = {u.uav_id: u.remaining_compute for u in active_uavs}
        
        current_assignment = {t.task_id: None for t in tasks}
        assigned_tasks_per_uav = {u.uav_id: [] for u in active_uavs}

        # Sort tasks: highest priority first, then largest workload descending
        sorted_tasks = sorted(
            tasks,
            key=lambda t: (
                t.priority,
                -(t.energy_cost + t.hover_time + t.compute_load)
            )
        )

        for task in sorted_tasks:
            best_score = -float('inf')
            selected_uav = None

            for u in active_uavs:
                if not u.is_compatible(task):
                    continue

                # Distance to the cluster centroid
                cx, cy = centroids[u.uav_id]
                dist_to_centroid = math.hypot(task.x - cx, task.y - cy)
                
                # Distance to UAV's actual position
                dist_to_uav = math.hypot(task.x - u.x, task.y - u.y)

                # Flight range constraint
                if dist_to_uav > u.max_flight_range:
                    continue

                # Capacity checks
                e_needed = task.energy_cost + ENERGY_PER_METER * dist_to_uav
                h_needed = task.hover_time + dist_to_uav / UAV_SPEED
                f_needed = task.compute_load

                if sim_remaining_energy[u.uav_id] < e_needed:
                    continue
                if sim_remaining_hover[u.uav_id] < h_needed:
                    continue
                if task.requires_compute and sim_remaining_compute[u.uav_id] < f_needed:
                    continue

                # Score functions balancing spatial proximity, resource margins, and priority
                spatial_score = -dist_to_centroid / MAP_WIDTH  # closer to centroid is better

                energy_margin = (
                    sim_remaining_energy[u.uav_id] - e_needed
                ) / u.max_energy

                hover_margin = (
                    sim_remaining_hover[u.uav_id] - h_needed
                ) / u.max_hover_time

                compute_margin = (
                    sim_remaining_compute[u.uav_id] - f_needed
                ) / max(u.max_compute, 1)

                # Soft penalty as capacities deplete to balance loads
                capacity_score = 15.0 * (energy_margin + hover_margin + compute_margin)
                
                # Priority factor: assign critical tasks to their most preferred UAVs
                priority_score = 10.0 * (4 - task.priority)

                # utilization_bonus = 0.5 * len(
                #     assigned_tasks_per_uav[u.uav_id]
                # )

                score = spatial_score + capacity_score + priority_score

                if score > best_score:
                    best_score = score
                    selected_uav = u

            if selected_uav is not None:
                current_assignment[task.task_id] = selected_uav
                assigned_tasks_per_uav[selected_uav.uav_id].append(task)
                
                # Deduct resources
                dist_to_uav = math.hypot(task.x - selected_uav.x, task.y - selected_uav.y)
                sim_remaining_energy[selected_uav.uav_id] -= (task.energy_cost + ENERGY_PER_METER * dist_to_uav)
                sim_remaining_hover[selected_uav.uav_id] -= (task.hover_time + dist_to_uav / UAV_SPEED)
                sim_remaining_compute[selected_uav.uav_id] -= task.compute_load

        # Update centroids for the next iteration
        for u in active_uavs:
            assigned = assigned_tasks_per_uav[u.uav_id]
            if assigned:
                avg_x = sum(t.x for t in assigned) / len(assigned)
                avg_y = sum(t.y for t in assigned) / len(assigned)
                centroids[u.uav_id] = (avg_x, avg_y)
            else:
                centroids[u.uav_id] = (u.x, u.y)

        # Track the best configuration
        assigned_count = sum(1 for uid in current_assignment.values() if uid is not None)
        if assigned_count > best_assigned_count:
            best_assigned_count = assigned_count
            best_assignment = current_assignment.copy()

    return best_assignment


# ----------------------------------------------------------
# PR-MODULE PUBLIC API
# ----------------------------------------------------------

def preassign(tasks, uavs, base_x=0.0, base_y=0.0, optimize=True):
    """
    Initial task pre-assignment phase.
    Clears old assignments and assigns all tasks.
    """
    for uav in uavs:
        uav.clear_tasks()
        uav.reset_resources()

    assignment = rc_kmeans_assign(tasks, uavs, base_x, base_y)

    for task in tasks:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav

    _print_assignment_summary("PRE-ASSIGNMENT", uavs, tasks)
    return uavs


def reassign_new_tasks(new_tasks, uavs, base_x=0.0, base_y=0.0, optimize=True):
    """
    Incremental assignment of new tasks using remaining capacities.
    """
    assignment = rc_kmeans_assign(new_tasks, uavs, base_x, base_y)

    for task in new_tasks:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav

    print(f"\n[PR] Proposed RC-KMeans inserted {len(new_tasks)} new tasks.")
    return uavs


def reassign_after_location_update(updated_tasks, uavs,
                                   base_x=0.0, base_y=0.0, optimize=True):
    """
    Re-assign updated tasks. Restores old capacity values before reassigning.
    """
    for task in updated_tasks:
        old_uav = task.assigned_uav
        if old_uav is not None and task in old_uav.assigned_tasks:
            old_uav.assigned_tasks.remove(task)
            # Revert resource consumption
            travel_dist = math.hypot(task.x - old_uav.x, task.y - old_uav.y)
            travel_energy = ENERGY_PER_METER * travel_dist
            travel_hover = travel_dist / UAV_SPEED
            old_uav.remaining_energy     += task.energy_cost + travel_energy
            old_uav.remaining_hover_time += task.hover_time + travel_hover
            old_uav.remaining_compute    += task.compute_load

    assignment = rc_kmeans_assign(updated_tasks, uavs, base_x, base_y)

    for task in updated_tasks:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav

    print(f"\n[PR] Proposed RC-KMeans re-assigned {len(updated_tasks)} location-updated tasks.")
    return uavs


def reassign_after_uav_failure(failed_uav, uavs,
                               base_x=0.0, base_y=0.0, optimize=True):
    """
    Redistribute uncompleted tasks from a failed UAV to active UAVs.
    """
    failed_uav.active = False
    orphaned = [
        t for t in failed_uav.assigned_tasks
        if not t.completed
    ]
    failed_uav.assigned_tasks = []

    if not orphaned:
        print(f"\n[PR] UAV {failed_uav.uav_id} failed – no orphaned tasks.")
        return uavs

    active_uavs = [u for u in uavs if u.active]
    assignment  = rc_kmeans_assign(orphaned, active_uavs, base_x, base_y)

    redistributed = 0
    for task in orphaned:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav
            redistributed += 1

    print(
        f"\n[PR] UAV {failed_uav.uav_id} failed – "
        f"proposed RC-KMeans redistributed {redistributed}/{len(orphaned)} tasks."
    )
    return uavs


def cancel_tasks(cancelled_tasks, uavs):
    """
    Remove cancelled tasks and restore capacities.
    """
    cancelled_ids = {t.task_id for t in cancelled_tasks}

    for uav in uavs:
        to_remove = [
            t for t in uav.assigned_tasks
            if t.task_id in cancelled_ids
        ]
        for t in to_remove:
            uav.assigned_tasks.remove(t)
            travel_dist = math.hypot(t.x - uav.x, t.y - uav.y)
            travel_energy = ENERGY_PER_METER * travel_dist
            travel_hover = travel_dist / UAV_SPEED
            uav.remaining_energy     += t.energy_cost + travel_energy
            uav.remaining_hover_time += t.hover_time + travel_hover
            uav.remaining_compute    += t.compute_load

    print(f"\n[PR] Cancelled {len(cancelled_tasks)} tasks.")
    return uavs


def _print_assignment_summary(label, uavs, all_tasks):
    assigned = sum(
        1 for t in all_tasks if t.assigned_uav is not None
    )
    print(f"\n=== {label} (PROPOSED RC-KMEANS) ===")
    print(f"  Tasks assigned: {assigned}/{len(all_tasks)}")
    for uav in uavs:
        if not uav.active:
            continue
        hi = sum(1 for t in uav.assigned_tasks if t.priority == 1)
        print(
            f"  UAV {uav.uav_id:02d} (type {uav.uav_type:+d}) -> "
            f"{len(uav.assigned_tasks)} tasks  (hi-pri={hi})  "
            f"rem-compute={uav.remaining_compute:.1f}  "
            f"rem-hover={uav.remaining_hover_time:.0f}s"
        )
    print()
