import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


from common.config import MAP_WIDTH, MAP_HEIGHT, GRID_RESOLUTION

save_dir = 'generated_figures'
prefix = ''

os.makedirs(save_dir, exist_ok=True)
PRIORITY_COLORS = {
    3: 'green',
    2: 'orange',
    1: 'red'
}

PRIORITY_LABELS = {
    3: 'Low Priority',
    2: 'Medium Priority',
    1: 'High Priority'
}

def plot_demand_map(demand_map, tasks):
    plt.figure(figsize=(10, 10))
    im = plt.imshow(
        demand_map,
        origin='lower',
        extent=[0, MAP_WIDTH * GRID_RESOLUTION,
                0, MAP_HEIGHT * GRID_RESOLUTION],
        cmap='YlOrRd',
        alpha=0.75,
        vmin = 0,vmax = 1
    )

    plt.colorbar(im, label='Demand Intensity')

    for task in tasks:
        plt.scatter(
            task.x,
            task.y,
            c=PRIORITY_COLORS[task.priority],
            edgecolors='black',
            s=60,
            linewidths=0.5
        )

    handles = [
        mpatches.Patch(
            color=PRIORITY_COLORS[p],
            label=PRIORITY_LABELS[p]
        )
        for p in sorted(PRIORITY_COLORS)
    ]

    plt.legend(handles=handles, fontsize=7)

    plt.title('Sensing Demand Map & Task Locations')
    plt.xlabel('X (grid units)')
    plt.ylabel('Y (grid units)')

    plt.xlim(0, MAP_WIDTH * GRID_RESOLUTION)
    plt.ylim(0, MAP_HEIGHT * GRID_RESOLUTION)

    path = os.path.join(save_dir, f'{prefix}demand_map.png')
    plt.savefig(path, dpi=150)
    plt.close()

def plot_uavs(uavs):
    plt.figure(figsize=(10, 10))
    for uav in uavs:
        plt.scatter(
            uav.x,
            uav.y,
            c='blue',
            edgecolors='black',
            s=80,
            linewidths=0.5,
            label=f'UAV {uav.uav_id}'
        )

    plt.title('UAV Positions')
    plt.xlabel('X (grid units)')
    plt.ylabel('Y (grid units)')

    plt.xlim(0, MAP_WIDTH * GRID_RESOLUTION)
    plt.ylim(0, MAP_HEIGHT * GRID_RESOLUTION)

    plt.legend()
    path = os.path.join(save_dir, f'{prefix}uav_positions.png')
    plt.savefig(path, dpi=150)
    plt.close()
