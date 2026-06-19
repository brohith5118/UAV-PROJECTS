# =========================================================
# ENVIRONMENT
#
# Generates the grid map G = {g_1,...,g_N} of resolution Δr
# and the heterogeneous UAV fleet U = {1,...,U}.
#
# Task workload vectors ω(g_i) are derived from a synthetic
# "sensing-demand" score that mimics the HTCD pixel-level
# change-label proxy used in the paper.
# =========================================================

import random
import math
import numpy as np

from common.task import Task
from common.uav import UAV

from common.config import (
    MAP_WIDTH,
    MAP_HEIGHT,
    GRID_RESOLUTION,
    NUM_UAVS,
    NUM_TASKS,
    HIGH_PRIORITY_RATIO,
    PRIORITY_DEADLINES,
    TASK_TYPE_RATIO,
    TASK_ENERGY_COST,
    TASK_HOVER_TIME,
    TASK_COMPUTE_LOAD,
    MIN_ENERGY,
    MAX_ENERGY,
    MIN_HOVER_TIME,
    MAX_HOVER_TIME,
    ENERGY_PER_METER,
    UAV_SPEED,
    UAV_TYPE_MAX_FLIGHT,
    UAV_TYPE_MAX_COMPUTE,
)


# ----------------------------------------------------------
# SYNTHETIC SENSING-DEMAND MAP
#
# Simulates the HTCD change-probability raster used in the
# paper to determine region relevance and priority.
# ----------------------------------------------------------

def generate_demand_map(seed=42):
    """
    Returns a 2-D numpy array [MAP_HEIGHT × MAP_WIDTH]
    with values in [0, 1] representing sensing demand /
    change probability at each grid cell.

    We place 3–6 Gaussian "hotspot" regions that represent
    high-interest areas (e.g. disaster zones, urban change).
    """
    rng  = np.random.default_rng(seed)
    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)

    num_hotspots = rng.integers(3, 7)
    ys = np.arange(MAP_HEIGHT)
    xs = np.arange(MAP_WIDTH)
    X, Y = np.meshgrid(xs, ys)

    for _ in range(num_hotspots):
        cx     = rng.uniform(5, MAP_WIDTH  - 5)
        cy     = rng.uniform(5, MAP_HEIGHT - 5)
        sigma  = rng.uniform(4, 12)
        weight = rng.uniform(0.4, 1.0)
        gauss  = weight * np.exp(
            -((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2)
        )
        grid += gauss

    # Normalise to [0, 1]
    grid = grid / (grid.max() + 1e-9)

    # Add mild uniform noise
    grid += rng.uniform(0, 0.1, grid.shape)
    grid = np.clip(grid / grid.max(), 0, 1)

    return grid


def score_to_priority(score, high_ratio=HIGH_PRIORITY_RATIO):
    """
    Convert a continuous demand score to a discrete priority
    level used throughout the paper.
      score >= 1 - high_ratio  →  1 (critical)
      score >= 0.4             →  2 (important)
      else                     →  3 (routine)
    """
    if score >= (1.0 - high_ratio):
        return 1
    elif score >= 0.4:
        return 2
    else:
        return 3


def score_to_task_type(score, rng):
    """
    Higher-demand cells more likely to need onboard compute.
    Draws from the task-type distribution in config.
    """
    thresholds = [
        TASK_TYPE_RATIO[-1],
        TASK_TYPE_RATIO[-1] + TASK_TYPE_RATIO[0],
    ]
    r = rng.random()
    if score > 0.7:
        # Bias towards compute-intensive for hot cells
        return 1 if r < 0.5 else 0
    if r < thresholds[0]:
        return -1
    elif r < thresholds[1]:
        return 0
    else:
        return 1


# ----------------------------------------------------------
# TASK GENERATION
# ----------------------------------------------------------
    """
    Sample *num_tasks* grid cells from G, weighting the
    sampling by demand score (higher-demand cells selected
    more often).

    Each task carries:
      - spatial position (x, y)
      - priority p^pri_i  derived from demand score
      - task type ϕ_i
      - workload vector ω(g_i) = {energy_cost, hover_time,
                                   compute_load}
      - deadline Di  (from paper Table 9 / priority level)
    """

def generate_tasks(num_tasks=NUM_TASKS,high_priority_ratio=HIGH_PRIORITY_RATIO,demand_map=None,seed=0):
    rng = np.random.default_rng(seed)

    if demand_map is None:
        demand_map = generate_demand_map(seed=seed)

    # Flatten and build a probability distribution
    flat_demand = demand_map.flatten()
    probs       = flat_demand / flat_demand.sum()
    cell_count  = MAP_WIDTH * MAP_HEIGHT

    # Sample cells without replacement
    chosen_cells = rng.choice(
        cell_count, size=num_tasks,
        replace=False, p=probs
    )

    task_list = []

    for tid, cell_idx in enumerate(chosen_cells):

        row = cell_idx // MAP_WIDTH
        col = cell_idx %  MAP_WIDTH

        # Cell centre coordinates
        x = col * GRID_RESOLUTION + 0.5
        y = row * GRID_RESOLUTION + 0.5

        score     = float(demand_map[row, col])
        priority  = score_to_priority(score, high_priority_ratio)
        task_type = score_to_task_type(score, rng)

        # ω(g_i) workload vector
        # Energy cost includes travel component (proportional
        # to demand; exact distance added per UAV in scheduler)
        energy_cost  = TASK_ENERGY_COST[task_type]
        hover_time   = TASK_HOVER_TIME[task_type]
        compute_load = TASK_COMPUTE_LOAD[task_type]

        deadline = PRIORITY_DEADLINES[priority]

        task = Task(
            task_id     = tid,
            x           = x,
            y           = y,
            priority    = priority,
            task_type   = task_type,
            energy_cost = energy_cost,
            hover_time  = hover_time,
            compute_load= compute_load,
            deadline    = deadline,
        )
        task_list.append(task)

    return task_list, demand_map


# ----------------------------------------------------------
# UAV GENERATION  (heterogeneous fleet)
# ----------------------------------------------------------

def generate_uavs(num_uavs=NUM_UAVS, seed=99):
    """
    Generate a heterogeneous UAV fleet.

    Fleet composition (roughly balanced, paper section 4.3):
      type -1  acquisition-only   : ~33%
      type  0  balanced           : ~33%
      type  1  compute-capable    : ~33%

    Each UAV starts at a random depot position.
    """
    rng  = np.random.default_rng(seed)
    uavs = []

    type_pool = (
        [-1] * (num_uavs // 3) +
        [ 0] * (num_uavs // 3) +
        [ 1] * (num_uavs - 2 * (num_uavs // 3))
    )
    rng.shuffle(type_pool)

    for uid in range(num_uavs):

        # Depot near map edges (realistic launch pads)
        x = float(rng.uniform(0, MAP_WIDTH * GRID_RESOLUTION))
        y = float(rng.uniform(0, MAP_HEIGHT * GRID_RESOLUTION))
        # x = 0
        # y = 0

        uav_type = type_pool[uid]

        # Max flight time from Table 1
        max_hover = float(UAV_TYPE_MAX_FLIGHT[uav_type])

        # Max energy  ∝  flight time (longer-endurance UAVs
        # carry more battery)
        max_energy = (
            max_hover * ENERGY_PER_METER * UAV_SPEED * 0.1
        )
        max_energy = float(np.clip(max_energy, MIN_ENERGY, MAX_ENERGY))

        # Max compute from Table 1
        max_compute = float(UAV_TYPE_MAX_COMPUTE[uav_type])

        uav = UAV(
            uav_id         = uid,
            x              = x,
            y              = y,
            uav_type       = uav_type,
            max_energy     = max_energy,
            max_hover_time = max_hover,
            max_compute    = max_compute,
        )
        uavs.append(uav)

    return uavs


# ----------------------------------------------------------
# DYNAMIC EVENT HELPERS
# ----------------------------------------------------------

def generate_new_task(task_id, demand_map, seed=None):
    """Spawn a single new urgent task (paper Section 4.2)."""
    rng = np.random.default_rng(seed)

    x = float(rng.uniform(0, MAP_WIDTH * GRID_RESOLUTION))
    y = float(rng.uniform(0, MAP_HEIGHT * GRID_RESOLUTION))

    row = min(int(y/GRID_RESOLUTION), MAP_HEIGHT - 1)
    col = min(int(x/GRID_RESOLUTION), MAP_WIDTH  - 1)
    score = float(demand_map[row, col])

    priority  = 1 if rng.random() < 0.5 else 2  # new tasks tend urgent
    task_type = score_to_task_type(score, rng)

    deadlines = {1: 1000, 2: 1250, 3: 1500}  # Table 11

    return Task(
        task_id     = task_id,
        x           = x,
        y           = y,
        priority    = priority,
        task_type   = task_type,
        energy_cost = float(rng.uniform(5, 20)),
        hover_time  = float(rng.uniform(3, 10)),
        compute_load= float(rng.uniform(5, 15)) if task_type >= 0 else 0.0,
        deadline    = deadlines[priority],
    )