"""Lightweight comparison runner for the DMMP-R-RL-AC pipeline."""

import os
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import main as pipeline_main  # noqa: E402
from common.config import HIGH_PRIORITY_RATIO, SEED, UAV_SEED  # noqa: E402
from common.environment import generate_demand_map, generate_tasks, generate_uavs  # noqa: E402
from pr_module import preassign  # noqa: E402
from rl_agent import QLearningTrajectoryPlanner  # noqa: E402


def _run_case(label, optimize, art_dir, prefix):
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)

    start = time.perf_counter()
    metrics = pipeline_main.main(
        optimize=optimize,
        save_dir=art_dir,
        prefix=prefix,
    )
    elapsed = time.perf_counter() - start
    print(f"{label} finished in {elapsed:.2f} seconds.")
    return metrics, elapsed


def _plot_tsa_reward_curve(art_dir):
    demand_map = generate_demand_map(seed=SEED)
    tasks, _ = generate_tasks(
        num_tasks=40,
        high_priority_ratio=HIGH_PRIORITY_RATIO,
        demand_map=demand_map,
        seed=SEED,
    )
    uavs = generate_uavs(num_uavs=4, seed=UAV_SEED)
    preassign(tasks, uavs, optimize=True)

    active_uavs = [
        uav for uav in uavs
        if getattr(uav, "active", True) and len(uav.assigned_tasks) > 1
    ]
    if not active_uavs:
        return None

    target_uav = active_uavs[0]
    planner = QLearningTrajectoryPlanner(
        target_uav,
        list(target_uav.assigned_tasks),
        optimize=True,
    )
    rewards = planner.train(epochs=500, verbose=False)
    if not rewards:
        return None

    arr = np.asarray(rewards, dtype=float)
    kernel = min(20, len(arr))
    smooth = np.convolve(arr, np.ones(kernel) / kernel, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(smooth, label="Heuristic-guided Q-learning", color="#2ca02c")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed episode reward")
    plt.title(
        "TSA Route Optimization Reward Curve "
        f"(UAV {target_uav.uav_id:02d}, {len(target_uav.assigned_tasks)} tasks)"
    )
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = os.path.join(art_dir, "tsa_reward_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved TSA reward curve to {path}")
    return path


def _format_metric(value, scale=1.0, suffix=""):
    return f"{value * scale:.3f}{suffix}"


def _write_report(art_dir, plain_metrics, proposed_metrics, plain_time, proposed_time):
    speedup = plain_time / proposed_time if proposed_time > 0 else 0.0
    plain_completion = _format_metric(plain_metrics["completion_rate"], 100, "%")
    proposed_completion = _format_metric(
        proposed_metrics["completion_rate"], 100, "%"
    )
    plain_high = _format_metric(
        plain_metrics["high_priority_completion_rate"], 100, "%"
    )
    proposed_high = _format_metric(
        proposed_metrics["high_priority_completion_rate"], 100, "%"
    )
    plain_energy = _format_metric(plain_metrics["energy_utilisation"], 100, "%")
    proposed_energy = _format_metric(
        proposed_metrics["energy_utilisation"], 100, "%"
    )
    plain_compute = _format_metric(plain_metrics["compute_utilisation"], 100, "%")
    proposed_compute = _format_metric(
        proposed_metrics["compute_utilisation"], 100, "%"
    )
    plain_overloaded = plain_metrics["overloaded_uav_count"]
    proposed_overloaded = proposed_metrics["overloaded_uav_count"]
    plain_distance = plain_metrics["total_travel_distance"]
    proposed_distance = proposed_metrics["total_travel_distance"]
    plain_fairness = plain_metrics["jains_fairness_index"]
    proposed_fairness = proposed_metrics["jains_fairness_index"]

    report = f"""# DMMP-R-RL-AC Comparison Report

| Metric | TSA local search off | Proposed DMMP-R-RL-AC |
| :--- | :---: | :---: |
| Overall completion rate | {plain_completion} | {proposed_completion} |
| High-priority completion rate | {plain_high} | {proposed_high} |
| Backup pool size | {plain_metrics['backup_pool_size']} | {proposed_metrics['backup_pool_size']} |
| Overloaded UAV count | {plain_overloaded} | {proposed_overloaded} |
| Total travel distance | {plain_distance:.1f} m | {proposed_distance:.1f} m |
| Mean energy utilisation | {plain_energy} | {proposed_energy} |
| Mean compute utilisation | {plain_compute} | {proposed_compute} |
| Jain fairness index | {plain_fairness:.3f} | {proposed_fairness:.3f} |
| Runtime | {plain_time:.2f} s | {proposed_time:.2f} s |
| Runtime ratio | 1.00x | {speedup:.2f}x |

The comparison keeps the D-module and PR-module identical and toggles only the
TSA route-refinement flag. This avoids mixing architectural changes with route
planning effects.
"""

    report_path = os.path.join(art_dir, "performance_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)
    print(f"Saved performance report to {report_path}")
    return report_path


def run_comparison():
    art_dir = os.path.join(
        os.path.dirname(__file__),
        "generated_graphs",
        "comparison",
    )
    os.makedirs(art_dir, exist_ok=True)

    plain_metrics, plain_time = _run_case(
        "RUNNING TSA WITHOUT LOCAL ROUTE REFINEMENT",
        optimize=False,
        art_dir=art_dir,
        prefix="no_refine_",
    )
    proposed_metrics, proposed_time = _run_case(
        "RUNNING PROPOSED DMMP-R-RL-AC",
        optimize=True,
        art_dir=art_dir,
        prefix="proposed_",
    )

    _plot_tsa_reward_curve(art_dir)
    _write_report(
        art_dir,
        plain_metrics,
        proposed_metrics,
        plain_time,
        proposed_time,
    )


if __name__ == "__main__":
    run_comparison()
