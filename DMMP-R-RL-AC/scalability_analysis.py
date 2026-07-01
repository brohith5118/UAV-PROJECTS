# =========================================================
# SCALABILITY ANALYSIS
# =========================================================

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common import config  # noqa: E402
import main as proposed_main


TASK_COUNTS = [10, 20, 30, 40, 50, 60, 80, 100]
RUNS_PER_SETTING = 3

SAVE_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated_graphs",
    "scalability",
)


def _empty_results():
    return {
        "tasks": [],
        "completion_rate": [],
        "high_priority_completion_rate": [],
        "travel_distance": [],
        "energy_utilisation": [],
        "compute_utilisation": [],
        "fairness": [],
        "overloaded": [],
        "runtime": [],
    }


def _save_plot(results, y, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(results["tasks"], y, marker="o", linewidth=2)
    plt.xlabel("Number of Tasks")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Task Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300)
    plt.close()


def _save_results_csv(results):
    csv_path = os.path.join(SAVE_DIR, "scalability_results.csv")
    headers = [
        "tasks",
        "completion_rate",
        "high_priority_completion_rate",
        "travel_distance",
        "energy_utilisation",
        "compute_utilisation",
        "fairness",
        "overloaded",
        "runtime",
    ]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")

        for i in range(len(results["tasks"])):
            row = [
                results["tasks"][i],
                results["completion_rate"][i],
                results["high_priority_completion_rate"][i],
                results["travel_distance"][i],
                results["energy_utilisation"][i],
                results["compute_utilisation"][i],
                results["fairness"][i],
                results["overloaded"][i],
                results["runtime"][i],
            ]
            f.write(",".join(map(str, row)) + "\n")

    return csv_path


def run_scalability_analysis(task_counts=None, runs_per_setting=RUNS_PER_SETTING):
    os.makedirs(SAVE_DIR, exist_ok=True)

    task_counts = task_counts or TASK_COUNTS
    results = _empty_results()

    for task_count in task_counts:
        print("\n" + "=" * 60)
        print(f"Testing {task_count} Tasks")
        print("=" * 60)

        completion_rates = []
        hp_rates = []
        travel_distances = []
        energy_utils = []
        compute_utils = []
        fairness_scores = []
        overloaded_counts = []
        runtimes = []

        for run in range(runs_per_setting):
            print(f"Run {run + 1}/{runs_per_setting}")

            start_time = time.perf_counter()
            metrics = proposed_main.main(
                num_tasks=task_count,
                num_uavs=config.NUM_UAVS,
                optimize=True,
                save_dir=SAVE_DIR,
                prefix=f"scale_{task_count}_run_{run}_",
            )
            elapsed = time.perf_counter() - start_time

            completion_rates.append(metrics["completion_rate"])
            hp_rates.append(metrics["high_priority_completion_rate"])
            travel_distances.append(metrics["total_travel_distance"])
            energy_utils.append(metrics["energy_utilisation"])
            compute_utils.append(metrics["compute_utilisation"])
            fairness_scores.append(metrics["jains_fairness_index"])
            overloaded_counts.append(metrics["overloaded_uav_count"])
            runtimes.append(elapsed)

        results["tasks"].append(task_count)
        results["completion_rate"].append(np.mean(completion_rates))
        results["high_priority_completion_rate"].append(np.mean(hp_rates))
        results["travel_distance"].append(np.mean(travel_distances))
        results["energy_utilisation"].append(np.mean(energy_utils))
        results["compute_utilisation"].append(np.mean(compute_utils))
        results["fairness"].append(np.mean(fairness_scores))
        results["overloaded"].append(np.mean(overloaded_counts))
        results["runtime"].append(np.mean(runtimes))

    _save_plot(
        results,
        results["completion_rate"],
        "Completion Rate",
        "scalability_completion_rate.png",
    )
    _save_plot(
        results,
        results["high_priority_completion_rate"],
        "High Priority Completion Rate (%)",
        "scalability_high_priority_completion.png",
    )
    _save_plot(
        results,
        results["travel_distance"],
        "Travel Distance",
        "scalability_travel_distance.png",
    )
    _save_plot(
        results,
        results["energy_utilisation"],
        "Energy Utilisation",
        "scalability_energy_utilisation.png",
    )
    _save_plot(
        results,
        results["compute_utilisation"],
        "Compute Utilisation",
        "scalability_compute_utilisation.png",
    )
    _save_plot(results, results["fairness"], "Jain Fairness Index", "scalability_fairness.png")
    _save_plot(
        results,
        results["overloaded"],
        "Overloaded UAV Count",
        "scalability_overloaded_uavs.png",
    )
    _save_plot(results, results["runtime"], "Runtime (seconds)", "scalability_runtime.png")

    csv_path = _save_results_csv(results)

    print("\nScalability analysis completed.")
    print(f"Results saved to:\n{SAVE_DIR}")
    print(f"CSV saved to:\n{csv_path}")
    return results


if __name__ == "__main__":
    run_scalability_analysis()
