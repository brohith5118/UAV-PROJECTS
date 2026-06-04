# =========================================================
# VISUALIZATION
#
# Produces a multi-panel figure showing:
#   Panel 1  – Sensing-demand heatmap + task locations
#   Panel 2  – Region partition (colour per UAV) + tasks
#   Panel 3  – UAV trajectories after TSA sequencing
#   Panel 4  – Resource utilisation bar chart per UAV
#   Panel 5  – Priority breakdown per UAV
#   Panel 6  – Per-task deadline compliance timeline
# =========================================================

import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import MAP_WIDTH, MAP_HEIGHT, UAV_SPEED, GRID_RESOLUTION, ENERGY_PER_METER
from utils  import estimate_finish_time, check_deadline


# ----------------------------------------------------------
# COLOUR PALETTE  (up to 15 UAVs)
# ----------------------------------------------------------

UAV_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]

PRIORITY_COLORS = {1: '#d62728', 2: '#ff7f0e', 3: '#2ca02c'}
PRIORITY_LABELS = {1: 'Critical (P1)', 2: 'Important (P2)', 3: 'Routine (P3)'}
GENERATED_GRAPH_DIR = 'generated_graphs'


def _uav_color(uav_id):
    return UAV_COLORS[uav_id % len(UAV_COLORS)]


def _resolve_save_dir(save_dir):
    output_dir = save_dir or GENERATED_GRAPH_DIR
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_figure(fig, filename, save_dir=None):
    output_dir = _resolve_save_dir(save_dir)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------
# PANEL 1 – DEMAND MAP  +  RAW TASK SCATTER
# ----------------------------------------------------------

def plot_demand_map(ax, demand_map, tasks):
    """Heatmap of sensing demand with task positions overlaid."""

    im = ax.imshow(
        demand_map,
        origin='lower',
        extent=[0, MAP_WIDTH * GRID_RESOLUTION, 0, MAP_HEIGHT * GRID_RESOLUTION],
        cmap='YlOrRd',
        alpha=0.75,
        vmin=0, vmax=1,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='Sensing demand')

    for task in tasks:
        ax.scatter(
            task.x, task.y,
            c=PRIORITY_COLORS[task.priority],
            s=60, edgecolors='black', linewidths=0.5,
            zorder=5,
        )

    # Legend for priorities
    handles = [
        mpatches.Patch(color=PRIORITY_COLORS[p],
                       label=PRIORITY_LABELS[p])
        for p in sorted(PRIORITY_COLORS)
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper right')
    ax.set_title('Sensing Demand Map & Task Locations', fontsize=10)
    ax.set_xlabel('X (grid units)')
    ax.set_ylabel('Y (grid units)')
    ax.set_xlim(0, MAP_WIDTH * GRID_RESOLUTION)
    ax.set_ylim(0, MAP_HEIGHT * GRID_RESOLUTION)


# ----------------------------------------------------------
# PANEL 2 – REGION PARTITION
# ----------------------------------------------------------

def plot_region_partition(ax, uavs, tasks, demand_map):
    """
    Colour each task dot by the UAV it was assigned to,
    overlay UAV depot positions, and draw convex-hull-like
    region boundaries using scatter alpha blending.
    """

    # Background demand
    ax.imshow(
        demand_map,
        origin='lower',
        extent=[0, MAP_WIDTH * GRID_RESOLUTION, 0, MAP_HEIGHT * GRID_RESOLUTION],
        cmap='Greys',
        alpha=0.25,
        vmin=0, vmax=1,
    )

    # Build task → uav map
    task_uav = {}
    for uav in uavs:
        for t in uav.assigned_tasks:
            task_uav[t.task_id] = uav.uav_id

    # Scatter tasks coloured by assigned UAV
    for task in tasks:
        uid  = task_uav.get(task.task_id, -1)
        col  = _uav_color(uid) if uid >= 0 else '#999999'
        marker = '*' if task.priority == 1 else 'o'
        ax.scatter(
            task.x, task.y,
            c=col, s=80 if task.priority == 1 else 40,
            marker=marker,
            edgecolors='black', linewidths=0.4,
            zorder=5,
        )

    # UAV depot squares
    for uav in uavs:
        ax.scatter(
            uav.x, uav.y,
            marker='s',
            s=180,
            c=_uav_color(uav.uav_id),
            edgecolors='black', linewidths=1.2,
            zorder=8,
        )
        ax.annotate(
            f'U{uav.uav_id}',
            (uav.x, uav.y),
            textcoords='offset points',
            xytext=(5, 5),
            fontsize=6,
        )

    # Centroid circles per UAV region
    for uav in uavs:
        if len(uav.assigned_tasks) < 2:
            continue
        cx = np.mean([t.x for t in uav.assigned_tasks])
        cy = np.mean([t.y for t in uav.assigned_tasks])
        ax.scatter(cx, cy,
                   marker='+', s=120,
                   c=_uav_color(uav.uav_id),
                   linewidths=2, zorder=9)

    ax.set_title('Capacity-Constrained Region Partition', fontsize=10)
    ax.set_xlabel('X (grid units)')
    ax.set_ylabel('Y (grid units)')
    ax.set_xlim(0, MAP_WIDTH * GRID_RESOLUTION)
    ax.set_ylim(0, MAP_HEIGHT * GRID_RESOLUTION)


# ----------------------------------------------------------
# PANEL 3 – UAV TRAJECTORIES  (after TSA)
# ----------------------------------------------------------

def plot_trajectories(ax, uavs, routes, demand_map):
    """
    Draw the RL-optimised execution routes for every UAV.
    Route starts at the UAV depot, visits tasks in order,
    and returns to depot.
    """

    ax.imshow(
        demand_map,
        origin='lower',
        extent=[0, MAP_WIDTH * GRID_RESOLUTION, 0, MAP_HEIGHT * GRID_RESOLUTION],
        cmap='Greys',
        alpha=0.2,
        vmin=0, vmax=1,
    )

    for uav in uavs:
        route = routes.get(uav.uav_id, [])
        col   = _uav_color(uav.uav_id)

        # Depot
        ax.scatter(
            uav.x, uav.y,
            marker='s', s=180,
            c=col,
            edgecolors='black', linewidths=1.2,
            zorder=8,
        )

        if not route:
            continue

        # Build path: depot → task_1 → ... → task_n → depot
        xs = [uav.x] + [t.x for t in route] + [uav.x]
        ys = [uav.y] + [t.y for t in route] + [uav.y]

        ax.plot(
            xs, ys,
            linestyle='--', linewidth=1.4,
            color=col, alpha=0.85,
            zorder=4,
        )

        # Task dots with sequence numbers
        for seq, task in enumerate(route, start=1):
            ax.scatter(
                task.x, task.y,
                c=PRIORITY_COLORS[task.priority],
                s=55, edgecolors='black', linewidths=0.4,
                zorder=6,
            )
            ax.annotate(
                str(seq),
                (task.x, task.y),
                textcoords='offset points',
                xytext=(3, 3),
                fontsize=5.5,
            )

        # Arrow on first leg
        if len(xs) > 2:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
            ax.annotate(
                '',
                xy=(xs[1], ys[1]),
                xytext=(xs[0], ys[0]),
                arrowprops=dict(
                    arrowstyle='->', color=col,
                    lw=1.2,
                ),
            )

    # Legend: UAV colours
    handles = [
        mpatches.Patch(color=_uav_color(u.uav_id),
                       label=f'UAV {u.uav_id} (type {u.uav_type:+d})')
        for u in uavs if uavs.index(u) < 6  # cap legend entries
    ]
    ax.legend(handles=handles, fontsize=6, loc='upper right',
              ncol=2)
    ax.set_title('RL-Optimised Task Execution Trajectories (TSA)', fontsize=10)
    ax.set_xlabel('X (grid units)')
    ax.set_ylabel('Y (grid units)')
    ax.set_xlim(0, MAP_WIDTH * GRID_RESOLUTION)
    ax.set_ylim(0, MAP_HEIGHT * GRID_RESOLUTION)


# ----------------------------------------------------------
# PANEL 4 – RESOURCE UTILISATION  (stacked bars)
# ----------------------------------------------------------

def plot_resource_utilisation(ax, uavs):
    """
    Grouped bar chart showing energy, hover-time, and
    compute utilisation as fractions of capacity for each UAV.
    """

    uav_labels = [f'U{u.uav_id}' for u in uavs]
    x          = np.arange(len(uavs))
    w          = 0.25

    def usage_frac(used, cap):
        return used / cap if cap > 0 else 0.0

    energy_used   = [
        usage_frac(u.max_energy - u.remaining_energy, u.max_energy)
        for u in uavs
    ]
    hover_used    = [
        usage_frac(u.max_hover_time - u.remaining_hover_time,
                   u.max_hover_time)
        for u in uavs
    ]
    compute_used  = [
        usage_frac(u.max_compute - u.remaining_compute, u.max_compute)
        for u in uavs
    ]

    ax.bar(x - w,  energy_used,  w, label='Energy',     color='#1f77b4', alpha=0.8)
    ax.bar(x,      hover_used,   w, label='Hover time', color='#ff7f0e', alpha=0.8)
    ax.bar(x + w,  compute_used, w, label='Compute',    color='#2ca02c', alpha=0.8)

    ax.axhline(1.0, color='red', linestyle=':', linewidth=1,
               label='Capacity limit')
    ax.set_xticks(x)
    ax.set_xticklabels(uav_labels, fontsize=8)
    ax.set_ylabel('Utilisation fraction')
    ax.set_ylim(0, 1.15)
    ax.set_title('Resource Utilisation per UAV', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.5)


# ----------------------------------------------------------
# PANEL 5 – PRIORITY BREAKDOWN  (stacked bar per UAV)
# ----------------------------------------------------------

def plot_priority_breakdown(ax, uavs):
    """
    Stacked bar showing how many P1 / P2 / P3 tasks each
    UAV carries.
    """

    uav_labels = [f'U{u.uav_id}' for u in uavs]
    x          = np.arange(len(uavs))

    p1 = [sum(1 for t in u.assigned_tasks if t.priority == 1) for u in uavs]
    p2 = [sum(1 for t in u.assigned_tasks if t.priority == 2) for u in uavs]
    p3 = [sum(1 for t in u.assigned_tasks if t.priority == 3) for u in uavs]

    ax.bar(x, p1, label='Critical (P1)',   color=PRIORITY_COLORS[1], alpha=0.85)
    ax.bar(x, p2, bottom=p1,               label='Important (P2)', color=PRIORITY_COLORS[2], alpha=0.85)
    p1p2 = [a + b for a, b in zip(p1, p2)]
    ax.bar(x, p3, bottom=p1p2,             label='Routine (P3)',   color=PRIORITY_COLORS[3], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(uav_labels, fontsize=8)
    ax.set_ylabel('Number of tasks')
    ax.set_title('Task Priority Distribution per UAV', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis='y', linestyle='--', alpha=0.5)


# ----------------------------------------------------------
# PANEL 6 – DEADLINE COMPLIANCE GANTT
# ----------------------------------------------------------

def plot_deadline_compliance(ax, uavs, routes):
    """
    Horizontal bar (Gantt-style) per task showing execution
    window.  Bars are green if completed before deadline,
    red otherwise.  Deadline is shown as a vertical line.
    """

    y_pos    = 0
    yticks   = []
    ylabels  = []

    for uav in uavs:
        route = routes.get(uav.uav_id, [])
        if not route:
            continue

        timeline = estimate_finish_time(uav, route, UAV_SPEED)
        clock    = 0.0
        prev_x, prev_y = uav.x, uav.y

        for task, finish in timeline:
            travel = math.hypot(
                task.x - prev_x, task.y - prev_y
            ) / UAV_SPEED
            start  = clock + travel
            dur    = task.hover_time
            on_time = check_deadline(task, finish)

            color = '#2ca02c' if on_time else '#d62728'

            ax.barh(
                y_pos, dur, left=start,
                color=color, alpha=0.75,
                edgecolor='black', linewidth=0.4,
                height=0.6,
            )
            # Deadline marker
            ax.vlines(
                task.deadline, y_pos - 0.4, y_pos + 0.4,
                colors='navy', linewidth=2.0, linestyles='--'
            )

            yticks.append(y_pos)
            ylabels.append(f'T{task.task_id}(U{uav.uav_id})')
            clock    = finish
            prev_x, prev_y = task.x, task.y
            y_pos   += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=5.5)
    ax.set_xlabel('Time (s)')
    ax.set_title('Deadline Compliance (green=on-time, red=late)', fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Legend
    handles = [
        mpatches.Patch(color='#2ca02c', label='On time'),
        mpatches.Patch(color='#d62728', label='Late'),
        plt.Line2D([0], [0], color='navy', linestyle=':',
                   label='Deadline'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='lower right')


# ----------------------------------------------------------
# INDIVIDUAL SPATIAL FIGURES
# ----------------------------------------------------------

def plot_demand_map_figure(demand_map, tasks, save_dir=None, prefix=""):
    """Standalone figure: sensing-demand heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        'Sensing-Demand Heatmap & Task Locations',
        fontsize=13, fontweight='bold',
    )
    plot_demand_map(ax, demand_map, tasks)
    plt.tight_layout()
    _save_figure(fig, f"{prefix}demand_map.png", save_dir)


def plot_region_partition_figure(uavs, tasks, demand_map, save_dir=None, prefix=""):
    """Standalone figure: capacity-constrained region partition."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        'Capacity-Constrained Region Partition\n'
        '(Power-Diagram + Lagrange Multipliers)',
        fontsize=13, fontweight='bold',
    )
    plot_region_partition(ax, uavs, tasks, demand_map)
    plt.tight_layout()
    _save_figure(fig, f"{prefix}region_partition.png", save_dir)


def plot_trajectories_figure(uavs, routes, demand_map, save_dir=None, prefix=""):
    """Standalone figure: RL-optimised UAV trajectories."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        'RL-Optimised UAV Trajectories (TSA Module)\n'
        'Numbers indicate execution order',
        fontsize=13, fontweight='bold',
    )
    plot_trajectories(ax, uavs, routes, demand_map)
    plt.tight_layout()
    _save_figure(fig, f"{prefix}trajectories.png", save_dir)


# ----------------------------------------------------------
# MASTER PLOT FUNCTION
# ----------------------------------------------------------

def plot_all(uavs, routes, tasks, demand_map, save_dir=None, prefix=""):
    """
    Render results across four separate figures:
      Figure 1  – Sensing-demand heatmap
      Figure 2  – Capacity-constrained region partition
      Figure 3  – UAV trajectories after TSA sequencing
      Figure 4  – Resource utilisation + priority breakdown
                  + deadline compliance Gantt
    """

    # ---- Figure 1: Demand map ----
    plot_demand_map_figure(demand_map, tasks, save_dir=save_dir, prefix=prefix)

    # ---- Figure 2: Region partition ---- #
    plot_region_partition_figure(uavs, tasks, demand_map, save_dir=save_dir, prefix=prefix)

    # ---- Figure 3: Trajectories ----
    plot_trajectories_figure(uavs, routes, demand_map, save_dir=save_dir, prefix=prefix)

    # ---- Figure 4: Analytics dashboard ----
    fig = plt.figure(figsize=(18, 10))
    active_uavs = [u for u in uavs if u.active]
    x_i = [len(u.assigned_tasks) for u in active_uavs]
    if x_i and sum(x_i) > 0:
        n_active = len(x_i)
        jains_index = (sum(x_i) ** 2) / (n_active * sum(val ** 2 for val in x_i))
    else:
        jains_index = 0.0

    fig.suptitle(
        f"DMMP-PR-TSA: Mission Analytics Dashboard (Jain's Fairness Index: {jains_index:.3f})",
        fontsize=13, fontweight='bold',
    )

    gs = fig.add_gridspec(2, 1, hspace=0.42, wspace=0.32)

    ax4 = fig.add_subplot(gs[0, 0])   # resource bars (full width top)
    ax5 = fig.add_subplot(gs[1, 0])     # priority breakdown

    plot_resource_utilisation(ax4, uavs)
    plot_priority_breakdown(ax5, uavs)

    plt.tight_layout()
    _save_figure(fig, f"{prefix}analytics_dashboard.png", save_dir)

    # Figure 5: Deadline compliance
    fig_deadline, ax_deadline = plt.subplots(figsize=(8, 5))

    fig_deadline.suptitle(
        "Mission Deadline Compliance Analysis",
        fontsize=13,
        fontweight='bold'
    )

    plot_deadline_compliance(
        ax_deadline,
        uavs,
        routes
    )

    fig_deadline.tight_layout(rect=[0, 0, 1, 0.93])

    _save_figure(fig_deadline, f"{prefix}deadline_compliance.png", save_dir)


# ----------------------------------------------------------
# SUPPLEMENTARY: ROLLOUT RL REWARD CONVERGENCE PLOT
# ----------------------------------------------------------

def plot_reward_convergence(reward_logs: dict, save_dir=None, prefix=""):
    """
    Plot per-UAV episode reward curves from TSA rollout training.

    Parameters
    ----------
    reward_logs : {uav_id: list_of_episode_rewards}
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for uid, rewards in reward_logs.items():
        if not rewards:
            continue
        # Smooth with a 20-episode rolling mean
        arr     = np.array(rewards, dtype=float)
        kernel  = min(20, len(arr))
        smooth  = np.convolve(
            arr, np.ones(kernel) / kernel, mode='valid'
        )
        ax.plot(smooth, label=f'UAV {uid}',
                color=_uav_color(uid), linewidth=1.2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Smoothed episode reward')
    ax.set_title('TSA Rollout RL Convergence (20-episode rolling mean)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    _save_figure(fig, f"{prefix}convergence.png", save_dir)
