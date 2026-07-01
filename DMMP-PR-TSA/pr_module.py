# =========================================================
# PR-MODULE  –  SOM-based Pre-Assignment & Re-Assignment
#
# Implements Algorithm 1 from the paper (Section
# "Network structure and input representation", eq 15–26).
#
# Input feature vectors:
#   INFO_TASK_i = (POS_i, ϕ_i, RES_i)          (eq 15)
#   INFO_UAV_u  = (POS_u, ψ_u, RES_u)          (eq 16)
#
# Matching distance:
#   D(i,u) = D_p + c_ϕ·D_ϕ + c_RES·D_RES       (eq 17)
#
# Neighbourhood update (eq 25–26) ensures topology
# coherence across UAV types.
#
# Dynamic events handled:
#   (1) New task insertion
#   (2) Task location update
#   (3) UAV failure / task redistribution
#   (4) Task cancellation
# =========================================================
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import math
import numpy as np

from common.config import (
    SOM_ITERATIONS,
    SOM_LEARN_RATE,
    C_PHI,
    C_RES,
    C_S,
    C_TIME,
    C_COMP,
    UAV_SPEED,
    ENERGY_PER_METER,
    MAP_WIDTH,
    MAP_HEIGHT,
    GRID_RESOLUTION,
)
from utils import estimate_route_usage, recompute_route_resources


# ----------------------------------------------------------
# MATCHING DISTANCE  D(i,u)  (eq 17–24)
# ----------------------------------------------------------

def spatial_distance(task, uav):
    """D_p = (X_i − X_u)² + (Y_i − Y_u)²  (eq 18)"""
    d2 = (task.x - uav.x) ** 2 + (task.y - uav.y) ** 2
    map_diag2 = (
        (MAP_WIDTH * GRID_RESOLUTION) ** 2
        + (MAP_HEIGHT * GRID_RESOLUTION) ** 2
    )
    return d2 / max(map_diag2, 1.0)


def capability_mismatch(task, uav):
    """
    D_ϕ (eq 19):
      0       if |ψ_u − ϕ_i| ≤ 1  (compatible)
      ∞       otherwise
    """
    if _is_capability_compatible(uav, task):
        return 0.0
    return float('inf')


def _is_capability_compatible(uav, task):
    """Capability feasibility using the project's UAV/task hierarchy."""
    if task.task_type == -1:
        return True
    if task.task_type == 0:
        return uav.uav_type in (0, 1)
    if task.task_type == 1:
        return uav.uav_type == 1
    return False


def time_margin_penalty(task, uav, base_x=0.0, base_y=0.0):
    """
    Δ_time  (eq 20–22):
      diff_time = T^re_u − (d(task,uav)+d(task,base))/v − t^req_i
      penalty   = exp(−c_time · diff_time)  if diff_time ≥ 0
                = ∞                          otherwise
    """
    d_to_task = math.hypot(task.x - uav.x, task.y - uav.y)
    d_to_base = math.hypot(task.x - base_x, task.y - base_y)
    travel    = (d_to_task + d_to_base) / UAV_SPEED
    diff_time = uav.remaining_hover_time - travel - task.hover_time

    if diff_time < 0:
        return float('inf')
    return math.exp(-C_TIME * diff_time)


def compute_margin_penalty(task, uav):
    """
    Δ_comp  (eq 23–24):
      diff_comp = C^re_u − c^req_i
      penalty   = exp(−c_comp · diff_comp)  if diff_comp ≥ 0
                = ∞                          otherwise
    """
    diff_comp = uav.remaining_compute - task.compute_load
    if diff_comp < 0:
        return float('inf')
    return math.exp(-C_COMP * diff_comp)


def matching_distance(task, uav, base_x=0.0, base_y=0.0):
    """
    Full matching distance D(i,u)  (eq 17).
    Returns ∞ if the pair is infeasible.
    """
    d_phi = capability_mismatch(task, uav)
    if d_phi == float('inf'):
        return float('inf')

    d_p   = spatial_distance(task, uav)
    d_res = (
        time_margin_penalty(task, uav, base_x, base_y)
        + compute_margin_penalty(task, uav)
    )

    if d_res == float('inf'):
        return float('inf')

    return d_p + C_PHI * d_phi + C_RES * d_res


def _route_usage(uav, route):
    return estimate_route_usage(uav, route, UAV_SPEED, ENERGY_PER_METER)


def _route_feasible(uav, route):
    if any(not _is_capability_compatible(uav, task) for task in route):
        return False

    energy, hover, compute = _route_usage(uav, route)
    return (
        energy <= uav.max_energy + 1e-9
        and hover <= uav.max_hover_time + 1e-9
        and compute <= uav.max_compute + 1e-9
    )


def _best_insertion_route(uav, route, task, base_x=0.0, base_y=0.0):
    if not _is_capability_compatible(uav, task):
        return None

    best_route = None
    best_score = None
    for pos in range(len(route) + 1):
        candidate = route[:pos] + [task] + route[pos:]
        if not _route_feasible(uav, candidate):
            continue

        energy, hover, compute = _route_usage(uav, candidate)
        deadline_misses = 0
        clock = 0.0
        current_x, current_y = uav.x, uav.y
        for item in candidate:
            clock += math.hypot(current_x - item.x, current_y - item.y) / UAV_SPEED
            clock += item.hover_time
            deadline_misses += int(clock > item.deadline)
            current_x, current_y = item.x, item.y

        dist_to_base = math.hypot(candidate[-1].x - base_x, candidate[-1].y - base_y)
        score = (deadline_misses, hover, energy, compute, dist_to_base)
        if best_score is None or score < best_score:
            best_score = score
            best_route = candidate

    return best_route


# ----------------------------------------------------------
# SOM NEIGHBOURHOOD FUNCTION  n_{u,u*}  (eq 25)
# ----------------------------------------------------------

def neighbourhood(uav, winner_uav, all_uavs):
    """
    n_{u,u*} (eq 25):
      1                        if u == u*
      exp(−S_{u,u*} / c_s)    if |ψ_u − ψ_u*| ≤ 1
      0                        otherwise

    S_{u,u*} is the index distance in the UAV list (proxy
    for SOM grid distance).
    """
    if uav.uav_id == winner_uav.uav_id:
        return 1.0

    if abs(uav.uav_type - winner_uav.uav_type) > 1:
        return 0.0

    idx_u      = next(i for i, v in enumerate(all_uavs)
                      if v.uav_id == uav.uav_id)
    idx_winner = next(i for i, v in enumerate(all_uavs)
                      if v.uav_id == winner_uav.uav_id)
    s = abs(idx_u - idx_winner)
    return math.exp(-s / C_S)


# ----------------------------------------------------------
# SOM FEATURE VECTORS  (residual-based, updated each round)
# ----------------------------------------------------------

def uav_feature(uav):
    """INFO_UAV_u = (X_u, Y_u, ψ_u, T^re_u, C^re_u)"""
    return np.array([
        uav.x,
        uav.y,
        float(uav.uav_type),
        uav.remaining_hover_time,
        uav.remaining_compute,
    ], dtype=np.float64)


def task_feature(task):
    """INFO_TASK_i = (X_i, Y_i, ϕ_i, t^req_i, c^req_i)"""
    return np.array([
        task.x,
        task.y,
        float(task.task_type),
        task.hover_time,
        task.compute_load,
    ], dtype=np.float64)


# ----------------------------------------------------------
# CORE SOM ASSIGNMENT  (Algorithm 1)
# ----------------------------------------------------------

class TempUAVWeightProxy:
    def __init__(
        self,
        uav_feat,
        uav_id,
        rem_energy,
        rem_hover,
        rem_compute,
        uav_map,
        use_weight_position=True,
    ):
        self.uav_id = uav_id
        orig_uav = uav_map[uav_id]
        if use_weight_position:
            self.x = float(uav_feat[0])
            self.y = float(uav_feat[1])
        else:
            self.x = orig_uav.x
            self.y = orig_uav.y
        self.uav_type = orig_uav.uav_type
        
        self.remaining_energy = rem_energy
        self.remaining_hover_time = rem_hover
        self.remaining_compute = rem_compute
        
        self.max_energy = orig_uav.max_energy
        self.max_hover_time = orig_uav.max_hover_time
        self.max_compute = orig_uav.max_compute

    def is_compatible(self, task):
        return _is_capability_compatible(self, task)

    def distance_to(self, task):
        return math.hypot(self.x - task.x, self.y - task.y)

    def time_feasible(self, task, base_x=0.0, base_y=0.0):
        dist_to_task = self.distance_to(task)
        dist_to_base = math.hypot(task.x - base_x, task.y - base_y)
        travel_time = (dist_to_task + dist_to_base) / UAV_SPEED
        return self.remaining_hover_time - travel_time - task.hover_time >= 0


def _task_resource_cost(uav, task):
    dist = math.hypot(uav.x - task.x, uav.y - task.y)
    return (
        task.energy_cost + ENERGY_PER_METER * dist,
        task.hover_time + dist / UAV_SPEED,
        task.compute_load,
    )


def _can_take_task(uav, task, base_x=0.0, base_y=0.0):
    if not _is_capability_compatible(uav, task):
        return False
    e_needed, _h_needed, c_needed = _task_resource_cost(uav, task)
    return (
        uav.remaining_energy >= e_needed
        and uav.time_feasible(task, base_x, base_y)
        and uav.remaining_compute >= c_needed
    )


def _deduct_task_resources(uav_id, uav, task, rem_energy, rem_hover, rem_compute):
    e_needed, h_needed, c_needed = _task_resource_cost(uav, task)
    rem_energy[uav_id] -= e_needed
    rem_hover[uav_id] -= h_needed
    rem_compute[uav_id] -= c_needed


def _recompute_uav_resources(uav):
    recompute_route_resources(uav)


def redistribute_load(uavs, base_x=0.0, base_y=0.0):
    active_uavs = [u for u in uavs if u.active]
    if not active_uavs:
        return uavs

    print("\n[PR] Running Dynamic Reallocation Engine for load balancing...")

    def get_utilizations(u):
        return _route_usage(u, u.assigned_tasks)

    max_passes = 5
    for pass_idx in range(max_passes):
        reallocated_any = False
        overloaded_uavs = []
        
        for u in active_uavs:
            te, th, tf = get_utilizations(u)
            if te > u.max_energy or th > u.max_hover_time or tf > u.max_compute:
                e_ratio = te / u.max_energy
                h_ratio = th / u.max_hover_time
                f_ratio = tf / u.max_compute if u.max_compute > 0 else 0.0
                overflow = max(e_ratio, h_ratio, f_ratio)
                overloaded_uavs.append((overflow, u))
                
        if not overloaded_uavs:
            break
            
        overloaded_uavs.sort(key=lambda x: x[0], reverse=True)
        
        for overflow, u_over in overloaded_uavs:
            tasks_to_migrate = sorted(
                u_over.assigned_tasks,
                key=lambda t: (t.priority, t.energy_cost + t.hover_time + t.compute_load),
                reverse=True
            )
            
            for task in tasks_to_migrate:
                best_score = -float('inf')
                best_target_uav = None
                
                for u_target in active_uavs:
                    if u_target.uav_id == u_over.uav_id:
                        continue
                    if not u_target.is_compatible(task):
                        continue
                        
                    candidate_route = _best_insertion_route(
                        u_target,
                        list(u_target.assigned_tasks),
                        task,
                        base_x,
                        base_y,
                    )
                    if candidate_route is not None:
                        target_te, target_th, _target_tf = _route_usage(u_target, candidate_route)
                        score = -target_th + 10.0 * (1.0 - target_te / u_target.max_energy)
                        if score > best_score:
                            best_score = score
                            best_target_uav = u_target
                            
                if best_target_uav is not None:
                    u_over.assigned_tasks.remove(task)
                    best_target_uav.assigned_tasks = _best_insertion_route(
                        best_target_uav,
                        list(best_target_uav.assigned_tasks),
                        task,
                        base_x,
                        base_y,
                    )
                    task.assigned_uav = best_target_uav
                    reallocated_any = True
                    print(f"      Migrated Task {task.task_id} from UAV {u_over.uav_id} to UAV {best_target_uav.uav_id}")
                    te, th, tf = get_utilizations(u_over)
                    if te <= u_over.max_energy and th <= u_over.max_hover_time and tf <= u_over.max_compute:
                        break
                        
        if not reallocated_any:
            print("      Cannot migrate tasks without violating target capacities. Evicting lowest priority tasks...")
            for overflow, u_over in overloaded_uavs:
                te, th, tf = get_utilizations(u_over)
                if te <= u_over.max_energy and th <= u_over.max_hover_time and tf <= u_over.max_compute:
                    continue
                u_over.assigned_tasks.sort(
                    key=lambda t: (t.priority, t.energy_cost + t.hover_time + t.compute_load),
                    reverse=True,
                )
                while u_over.assigned_tasks and (te > u_over.max_energy or th > u_over.max_hover_time or tf > u_over.max_compute):
                    evicted = u_over.assigned_tasks.pop(0)
                    evicted.assigned_uav = None
                    print(f"      Evicted Task {evicted.task_id} (Priority {evicted.priority}) from UAV {u_over.uav_id} to prevent overload")
                    te, th, tf = get_utilizations(u_over)
            break

    for u in active_uavs:
        _recompute_uav_resources(u)

    return uavs


def som_assign(tasks, uavs, base_x=0.0, base_y=0.0, optimize=True):
    if not tasks or not uavs:
        return {}, {}

    active_uavs = [u for u in uavs if u.active]
    if not active_uavs:
        return {t.task_id: None for t in tasks}, {}

    uav_map = {u.uav_id: u for u in active_uavs}
    planned_routes = {u.uav_id: list(u.assigned_tasks) for u in active_uavs}
    
    rem_energy = {}
    rem_hover = {}
    rem_compute = {}
    for u in active_uavs:
        energy, hover, compute = _route_usage(u, planned_routes[u.uav_id])
        rem_energy[u.uav_id] = u.max_energy - energy
        rem_hover[u.uav_id] = u.max_hover_time - hover
        rem_compute[u.uav_id] = u.max_compute - compute

    uav_features = {u.uav_id: uav_feature(u) for u in active_uavs}

    learn_rate = SOM_LEARN_RATE
    lr_decay   = learn_rate / SOM_ITERATIONS

    for r in range(SOM_ITERATIONS):
        shuffled = list(tasks)
        np.random.default_rng(r).shuffle(shuffled)

        for task in shuffled:
            tf = task_feature(task)
            best_dist  = float('inf')
            best_uav   = None

            for uav in active_uavs:
                if optimize:
                    feat = uav_features[uav.uav_id]
                    uav_proxy = TempUAVWeightProxy(
                        feat, uav.uav_id,
                        rem_energy[uav.uav_id],
                        rem_hover[uav.uav_id],
                        rem_compute[uav.uav_id],
                        uav_map,
                        use_weight_position=True,
                    )
                    d = matching_distance(task, uav_proxy, base_x, base_y)
                else:
                    # Baseline weight-bypass bug (original uav instead of proxy weights)
                    d = matching_distance(task, uav, base_x, base_y)
                if d < best_dist:
                    best_dist = d
                    best_uav  = uav

            if best_uav is None:
                continue

            for uav in active_uavs:
                n = neighbourhood(uav, best_uav, active_uavs)
                if n == 0.0:
                    continue
                feat   = uav_features[uav.uav_id]
                feat  += n * learn_rate * (tf - feat)
                uav_features[uav.uav_id] = feat

        learn_rate = max(0.01, learn_rate - lr_decay)

    assignment = {}
    sorted_tasks = sorted(tasks, key=lambda t: t.priority)

    for task in sorted_tasks:
        best_dist = float('inf')
        best_uav  = None
        best_route = None

        for uav in active_uavs:
            if optimize:
                feat = uav_features[uav.uav_id]
                uav_proxy = TempUAVWeightProxy(
                    feat, uav.uav_id,
                    rem_energy[uav.uav_id],
                    rem_hover[uav.uav_id],
                    rem_compute[uav.uav_id],
                    uav_map,
                    use_weight_position=False,
                )
                candidate_route = _best_insertion_route(
                    uav,
                    planned_routes[uav.uav_id],
                    task,
                    base_x,
                    base_y,
                )
                if candidate_route is None:
                    continue
                dist = matching_distance(task, uav_proxy, base_x, base_y)
            else:
                candidate_route = _best_insertion_route(
                    uav,
                    planned_routes[uav.uav_id],
                    task,
                    base_x,
                    base_y,
                )
                if candidate_route is None:
                    continue
                dist = math.hypot(uav.x - task.x, uav.y - task.y)
            if dist < best_dist:
                best_dist = dist
                best_uav  = uav
                best_route = candidate_route

        if best_uav is not None:
            assignment[task.task_id] = best_uav
            planned_routes[best_uav.uav_id] = best_route
            energy, hover, compute = _route_usage(best_uav, best_route)
            rem_energy[best_uav.uav_id] = best_uav.max_energy - energy
            rem_hover[best_uav.uav_id] = best_uav.max_hover_time - hover
            rem_compute[best_uav.uav_id] = best_uav.max_compute - compute
        else:
            assignment[task.task_id] = None

    return assignment, planned_routes


# ----------------------------------------------------------
# PR-MODULE PUBLIC API
# ----------------------------------------------------------

def preassign(tasks, uavs, base_x=0.0, base_y=0.0, optimize=True):
    has_partition = any(uav.assigned_tasks for uav in uavs)

    if has_partition:
        assigned_ids = set()
        for uav in uavs:
            for task in uav.assigned_tasks:
                task.assigned_uav = uav
                assigned_ids.add(task.task_id)
            _recompute_uav_resources(uav)
        candidate_tasks = [task for task in tasks if task.task_id not in assigned_ids]
    else:
        for uav in uavs:
            uav.clear_tasks()
            uav.reset_resources()
        for task in tasks:
            task.assigned_uav = None
        candidate_tasks = list(tasks)

    assignment, planned_routes = som_assign(candidate_tasks, uavs, base_x, base_y, optimize)

    for uav in uavs:
        uav.assigned_tasks = planned_routes.get(uav.uav_id, uav.assigned_tasks)
        for task in uav.assigned_tasks:
            task.assigned_uav = uav
        _recompute_uav_resources(uav)

    for task in candidate_tasks:
        if assignment.get(task.task_id) is None:
            task.assigned_uav = None

    if optimize:
        uavs = redistribute_load(uavs, base_x, base_y)
    _print_assignment_summary("PRE-ASSIGNMENT", uavs, tasks)
    return uavs


def reassign_new_tasks(new_tasks, uavs, base_x=0.0, base_y=0.0, optimize=True):
    assignment, planned_routes = som_assign(new_tasks, uavs, base_x, base_y, optimize)

    for uav in uavs:
        if uav.active:
            uav.assigned_tasks = planned_routes.get(uav.uav_id, uav.assigned_tasks)
            for task in uav.assigned_tasks:
                task.assigned_uav = uav
            _recompute_uav_resources(uav)

    for task in new_tasks:
        if assignment.get(task.task_id) is None:
            task.assigned_uav = None

    if optimize:
        uavs = redistribute_load(uavs, base_x, base_y)
    print(f"\n[PR] Inserted {len(new_tasks)} new tasks.")
    return uavs


def reassign_after_location_update(updated_tasks, uavs,
                                   base_x=0.0, base_y=0.0, optimize=True):
    for task in updated_tasks:
        old_uav = task.assigned_uav
        if old_uav is not None and task in old_uav.assigned_tasks:
            old_uav.assigned_tasks.remove(task)
            _recompute_uav_resources(old_uav)

    assignment, planned_routes = som_assign(updated_tasks, uavs, base_x, base_y, optimize)

    for uav in uavs:
        if uav.active:
            uav.assigned_tasks = planned_routes.get(uav.uav_id, uav.assigned_tasks)
            for task in uav.assigned_tasks:
                task.assigned_uav = uav
            _recompute_uav_resources(uav)

    for task in updated_tasks:
        if assignment.get(task.task_id) is None:
            task.assigned_uav = None

    if optimize:
        uavs = redistribute_load(uavs, base_x, base_y)
    print(f"\n[PR] Re-assigned {len(updated_tasks)} location-updated tasks.")
    return uavs


def reassign_after_uav_failure(failed_uav, uavs,
                               base_x=0.0, base_y=0.0, optimize=True):
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
    assignment, planned_routes = som_assign(orphaned, active_uavs, base_x, base_y, optimize)

    redistributed = 0
    for uav in active_uavs:
        uav.assigned_tasks = planned_routes.get(uav.uav_id, uav.assigned_tasks)
        for task in uav.assigned_tasks:
            task.assigned_uav = uav
        _recompute_uav_resources(uav)

    for task in orphaned:
        if assignment.get(task.task_id) is None:
            task.assigned_uav = None
        else:
            redistributed += 1

    if optimize:
        uavs = redistribute_load(uavs, base_x, base_y)
    print(
        f"\n[PR] UAV {failed_uav.uav_id} failed – "
        f"redistributed {redistributed}/{len(orphaned)} tasks."
    )
    return uavs


def cancel_tasks(cancelled_tasks, uavs):
    """
    Dynamic event (4): Tasks cancelled – remove from UAV
    lists and return resources.
    """
    cancelled_ids = {t.task_id for t in cancelled_tasks}

    for uav in uavs:
        to_remove = [
            t for t in uav.assigned_tasks
            if t.task_id in cancelled_ids
        ]
        for t in to_remove:
            uav.assigned_tasks.remove(t)
            t.assigned_uav = None
            _recompute_uav_resources(uav)

    print(f"\n[PR] Cancelled {len(cancelled_tasks)} tasks.")
    return uavs


# ----------------------------------------------------------
# HELPER: PRINT SUMMARY
# ----------------------------------------------------------

def _print_assignment_summary(label, uavs, all_tasks):
    assigned = sum(
        1 for t in all_tasks if t.assigned_uav is not None
    )
    print(f"\n=== {label} ===")
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
