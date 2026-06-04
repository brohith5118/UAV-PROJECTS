import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from config import GENERATED_GRAPHS_DIR, GRID_RESOLUTION, MAP_HEIGHT, MAP_WIDTH


UAV_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
]

PRIORITY_COLORS = {1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}
PRIORITY_LABELS = {1: "Routine (P1)", 2: "Important (P2)", 3: "Critical (P3)"}


def _uav_color(uav_id):
    return UAV_COLORS[uav_id % len(UAV_COLORS)]


def _graph_dir(output_dir=None):
    path = Path(output_dir or GENERATED_GRAPHS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig, output_dir, filename):
    path = _graph_dir(output_dir) / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _route_events(result, uav_id):
    route = result.routes.get(uav_id)
    return [] if route is None else route.scheduled_tasks


def plot_demand_map(ax, demand_map, tasks):
    im = ax.imshow(
        demand_map,
        origin="lower",
        extent=[0, MAP_WIDTH * GRID_RESOLUTION, 0, MAP_HEIGHT * GRID_RESOLUTION],
        cmap="YlOrRd",
        alpha=0.75,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Sensing demand")

    for task in tasks:
        ax.scatter(
            task.x,
            task.y,
            c=PRIORITY_COLORS[task.priority],
            s=62,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    handles = [
        mpatches.Patch(color=PRIORITY_COLORS[p], label=PRIORITY_LABELS[p])
        for p in sorted(PRIORITY_COLORS)
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    ax.set_title("Sensing Demand Map and Task Locations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, MAP_WIDTH * GRID_RESOLUTION)
    ax.set_ylim(0, MAP_HEIGHT * GRID_RESOLUTION)


def plot_assignment_map(ax, uavs, tasks, result, demand_map, label="PSO"):
    ax.imshow(
        demand_map,
        origin="lower",
        extent=[0, MAP_WIDTH * GRID_RESOLUTION, 0, MAP_HEIGHT * GRID_RESOLUTION],
        cmap="Greys",
        alpha=0.25,
        vmin=0,
        vmax=1,
    )

    for task in tasks:
        uid = task.assigned_uav
        color = _uav_color(uid) if uid is not None else "#999999"
        marker = "*" if task.priority == 3 else "o"
        ax.scatter(
            task.x,
            task.y,
            c=color,
            s=90 if task.priority == 1 else 45,
            marker=marker,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )

    for uav in uavs:
        ax.scatter(
            uav.x,
            uav.y,
            marker="s",
            s=160,
            c=_uav_color(uav.uav_id),
            edgecolors="black",
            linewidths=1.2,
            zorder=8,
        )
        ax.annotate(f"U{uav.uav_id}", (uav.x, uav.y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    for uav_id, route in result.routes.items():
        events = route.scheduled_tasks
        if len(events) < 2:
            continue
        cx = np.mean([event.task.x for event in events])
        cy = np.mean([event.task.y for event in events])
        ax.scatter(cx, cy, marker="+", s=120, c=_uav_color(uav_id), linewidths=2, zorder=9)

    ax.set_title(f"{label} Task Assignment by UAV")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, MAP_WIDTH * GRID_RESOLUTION)
    ax.set_ylim(0, MAP_HEIGHT * GRID_RESOLUTION)


def plot_trajectories(ax, uavs, result, demand_map, label="PSO"):
    ax.imshow(
        demand_map,
        origin="lower",
        extent=[0, MAP_WIDTH * GRID_RESOLUTION, 0, MAP_HEIGHT * GRID_RESOLUTION],
        cmap="Greys",
        alpha=0.2,
        vmin=0,
        vmax=1,
    )

    for uav in uavs:
        color = _uav_color(uav.uav_id)
        events = _route_events(result, uav.uav_id)

        ax.scatter(
            uav.x,
            uav.y,
            marker="s",
            s=160,
            c=color,
            edgecolors="black",
            linewidths=1.2,
            zorder=8,
        )

        if not events:
            continue

        xs = [uav.x] + [event.task.x for event in events] + [uav.x]
        ys = [uav.y] + [event.task.y for event in events] + [uav.y]
        ax.plot(xs, ys, linestyle="--", linewidth=1.5, color=color, alpha=0.85, zorder=4)

        if len(xs) > 1:
            ax.annotate(
                "",
                xy=(xs[1], ys[1]),
                xytext=(xs[0], ys[0]),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
            )

        for seq, event in enumerate(events, start=1):
            task = event.task
            ax.scatter(
                task.x,
                task.y,
                c=PRIORITY_COLORS[task.priority],
                s=56,
                edgecolors="black",
                linewidths=0.4,
                zorder=6,
            )
            ax.annotate(str(seq), (task.x, task.y), textcoords="offset points", xytext=(3, 3), fontsize=6)

    handles = [
        mpatches.Patch(color=_uav_color(uav.uav_id), label=f"UAV {uav.uav_id} (type {uav.uav_type:+d})")
        for uav in uavs[:8]
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)
    ax.set_title(f"{label} UAV Task Execution Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, MAP_WIDTH * GRID_RESOLUTION)
    ax.set_ylim(0, MAP_HEIGHT * GRID_RESOLUTION)


def plot_resource_utilisation(ax, uavs):
    labels = [f"U{uav.uav_id}" for uav in uavs]
    x = np.arange(len(uavs))
    width = 0.25

    def fraction(used, capacity):
        return used / capacity if capacity > 0 else 0.0

    energy = [fraction(u.max_energy - u.remaining_energy, u.max_energy) for u in uavs]
    hover = [fraction(u.max_hover_time - u.remaining_hover_time, u.max_hover_time) for u in uavs]
    compute = [fraction(u.max_compute - u.remaining_compute, u.max_compute) for u in uavs]

    ax.bar(x - width, energy, width, label="Energy", color="#1f77b4", alpha=0.85)
    ax.bar(x, hover, width, label="Hover time", color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, compute, width, label="Compute", color="#2ca02c", alpha=0.85)
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1, label="Capacity limit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Utilisation fraction")
    ax.set_ylim(0, 1.15)
    ax.set_title("Resource Utilisation per UAV")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.45)


def plot_priority_breakdown(ax, uavs):
    labels = [f"U{uav.uav_id}" for uav in uavs]
    x = np.arange(len(uavs))
    p1 = [sum(1 for task in uav.assigned_tasks if task.priority == 1) for uav in uavs]
    p2 = [sum(1 for task in uav.assigned_tasks if task.priority == 2) for uav in uavs]
    p3 = [sum(1 for task in uav.assigned_tasks if task.priority == 3) for uav in uavs]

    ax.bar(x, p1, label=PRIORITY_LABELS[1], color=PRIORITY_COLORS[1], alpha=0.85)
    ax.bar(x, p2, bottom=p1, label=PRIORITY_LABELS[2], color=PRIORITY_COLORS[2], alpha=0.85)
    p1p2 = [a + b for a, b in zip(p1, p2)]
    ax.bar(x, p3, bottom=p1p2, label=PRIORITY_LABELS[3], color=PRIORITY_COLORS[3], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of tasks")
    ax.set_title("Task Priority Distribution per UAV")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.45)


def plot_deadline_compliance(ax, uavs, result):
    y_pos = 0
    yticks = []
    ylabels = []

    for uav in uavs:
        for event in _route_events(result, uav.uav_id):
            task = event.task
            on_time = event.finish_time <= task.deadline
            color = "#2ca02c" if on_time else "#d62728"
            ax.barh(
                y_pos,
                task.hover_time,
                left=event.start_time,
                color=color,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
                height=0.62,
            )
            ax.vlines(task.deadline, y_pos - 0.4, y_pos + 0.4, colors="navy", linewidth=1.8, linestyles="--")
            yticks.append(y_pos)
            ylabels.append(f"T{task.task_id}(U{uav.uav_id})")
            y_pos += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_title("Deadline Compliance")
    ax.grid(axis="x", linestyle="--", alpha=0.45)
    handles = [
        mpatches.Patch(color="#2ca02c", label="On time"),
        mpatches.Patch(color="#d62728", label="Late"),
        plt.Line2D([0], [0], color="navy", linestyle="--", label="Deadline"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")


def plot_pso_convergence(ax, result, label="PSO"):
    if not result.history:
        ax.text(0.5, 0.5, f"No {label} history available", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(result.history, color="#1f77b4", linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_title(f"{label} Fitness Convergence")
    ax.grid(linestyle="--", alpha=0.45)


def save_all_graphs(uavs, tasks, demand_map, result, output_dir=None, prefix="pso_", label=None):
    output_dir = _graph_dir(output_dir)
    saved_paths = []
    label = label or prefix.strip("_").upper()

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_demand_map(ax, demand_map, tasks)
    saved_paths.append(_save(fig, output_dir, f"{prefix}demand_map.png"))

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_assignment_map(ax, uavs, tasks, result, demand_map, label=label)
    saved_paths.append(_save(fig, output_dir, f"{prefix}assignment_map.png"))

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_trajectories(ax, uavs, result, demand_map, label=label)
    saved_paths.append(_save(fig, output_dir, f"{prefix}trajectories.png"))

    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    task_counts = [len(uav.assigned_tasks) for uav in uavs if getattr(uav, "active", True)]
    fairness = 0.0
    if task_counts and sum(task_counts) > 0:
        fairness = (sum(task_counts) ** 2) / (len(task_counts) * sum(count ** 2 for count in task_counts))
    fig.suptitle(f"{label} Mission Analytics Dashboard | Jain's Fairness Index: {fairness:.3f}", fontweight="bold")
    plot_resource_utilisation(axes[0], uavs)
    plot_priority_breakdown(axes[1], uavs)
    plot_pso_convergence(axes[2], result, label=label)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    saved_paths.append(_save(fig, output_dir, f"{prefix}analytics_dashboard.png"))

    fig, ax = plt.subplots(figsize=(10, max(6, 0.23 * result.completed_count)))
    plot_deadline_compliance(ax, uavs, result)
    fig.tight_layout()
    saved_paths.append(_save(fig, output_dir, f"{prefix}deadline_compliance.png"))

    return saved_paths
