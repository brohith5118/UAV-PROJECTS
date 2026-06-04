import random
import math
import numpy as np

from task import Task
from uav import UAV

from config import (
    MAP_HEIGHT,
    MAP_WIDTH,
    GRID_RESOLUTION,
    TASK_TYPE_RATIO,
    TASK_HOVER_TIME,
    TASK_ENERGY_COST,
    TASK_COMPUTE_COST,
    TASK_DEADLINE,
    UAV_TYPE_MAX_ENERGY,
    UAV_TYPE_MAX_FLIGHT,
    UAV_TYPE_MAX_COMPUTE
)

def generate_demand_map(seed):
    rng = np.random.default_rng(seed)
    grid = np.zeros((MAP_HEIGHT,MAP_WIDTH),dtype = np.float32)

    num_hotspots = rng.integers(3,7)
    ys = np.arange(MAP_HEIGHT)
    xs = np.arange(MAP_WIDTH)
    X,Y = np.meshgrid(xs,ys)

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


def score_to_priority(score, high_ratio):
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


def generate_tasks(num_tasks,high_priority_ratio,demand_map,seed):
    rng = np.random.default_rng(seed)
    tasks = []

    flat_demand = demand_map.flatten()
    probs       = flat_demand / flat_demand.sum()
    cell_count  = MAP_WIDTH * MAP_HEIGHT

    chosen_cells = rng.choice(cell_count,size = num_tasks,replace = False,p = probs)

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
        compute_cost = TASK_COMPUTE_COST[task_type]

        deadline = TASK_DEADLINE[priority]

        task = Task(
            task_id     = tid,
            x           = x,
            y           = y,
            priority    = priority,
            task_type   = task_type,
            energy_cost = energy_cost,
            hover_time  = hover_time,
            compute_cost= compute_cost,
            deadline    = deadline,
        )
        tasks.append(task)

    return tasks
    

def generate_uavs(num_uavs, seed=99):
    rng  = np.random.default_rng(seed)
    uavs = []

    type_pool = (
        [-1] * (num_uavs // 3) +
        [ 0] * (num_uavs // 3) +
        [ 1] * (num_uavs - 2 * (num_uavs // 3))
    )
    rng.shuffle(type_pool)

    for uid in range(num_uavs):

        # Depot near demand_map edges (realistic launch pads)
        x = float(rng.uniform(0, MAP_WIDTH * GRID_RESOLUTION))
        y = float(rng.uniform(0, MAP_HEIGHT * GRID_RESOLUTION))

        uav_type = type_pool[uid]

        # Max flight time from Table 1
        max_hover = float(UAV_TYPE_MAX_FLIGHT[uav_type])

        # Max energy  ∝  flight time (longer-endurance UAVs
        # carry more battery)
        max_energy = float(UAV_TYPE_MAX_ENERGY[uav_type])

        # Max compute from Table 1
        max_compute = float(UAV_TYPE_MAX_COMPUTE[uav_type])

        uav = UAV(
            uav_id         = uid,
            x              = x,
            y              = y,
            uav_type       = uav_type,
            max_energy     = max_energy,
            max_hover      = max_hover,
            max_compute    = max_compute,
        )
        uavs.append(uav)

    return uavs
