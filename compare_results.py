import argparse
import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "comparison_outputs"
PUBLICATION_GRAPH_DIR = OUTPUT_DIR / "publication_graphs"
TABLE_DIR = OUTPUT_DIR / "tables"
GRAPH_DIR = OUTPUT_DIR / "graphs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
PUBLICATION_GRAPH_DIR.mkdir(parents=True, exist_ok=True)


ALGORITHMS = {
    "DMMP-PR-TSA": "DMMP-PR-TSA",
    "DMMP-R-RL-AC": "DMMP-R-RL-AC",
    "Greedy-Nearest": "Greedy-Nearest",
    "PSO": "PSO",
    "DPSO": "DPSO",
}

METRICS = [
    ("completion_rate", "Completion Rate"),
    ("high_priority_completion_rate", "High Priority Completion Rate"),
    ("runtime", "Runtime (seconds)"),
]

ALGORITHM_COLORS = {
    "DMMP-PR-TSA": "#1f77b4",
    "DMMP-R-RL-AC": "#2ca02c",
    "Greedy-Nearest": "#d62728",
    "PSO": "#9467bd",
    "DPSO": "#8c564b",
}

ALGORITHM_MARKERS = {
    "DMMP-PR-TSA": "o",
    "DMMP-R-RL-AC": "s",
    "Greedy-Nearest": "^",
    "PSO": "D",
    "DPSO": "P",
}

LOCAL_MODULES = [
    "main",
    "scheduler",
    "utils",
    "visualization",
    "pr_module",
    "rl_agent",
]


@contextlib.contextmanager
def algorithm_import_context(folder_path):
    """Temporarily put one algorithm folder first so same-named modules do not collide."""
    original_path = list(sys.path)
    removed_modules = {
        name: sys.modules.pop(name)
        for name in LOCAL_MODULES
        if name in sys.modules
    }

    sys.path.insert(0, str(folder_path))
    sys.path.insert(0, str(ROOT_DIR))
    try:
        yield
    finally:
        for name in LOCAL_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(removed_modules)
        sys.path[:] = original_path


def load_algorithm_main(folder_name):
    folder_path = ROOT_DIR / folder_name
    main_path = folder_path / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"Could not find main.py in {folder_path}")

    module_name = f"comparison_{folder_name.replace('-', '_')}_main"
    with algorithm_import_context(folder_path):
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.main


def normalize_metrics(metrics):
    runtime = metrics.get("runtime", metrics.get("runtime_seconds", 0.0))
    return {
        "completion_rate": float(metrics.get("completion_rate", 0.0)),
        "high_priority_completion_rate": float(
            metrics.get("high_priority_completion_rate", 0.0)
        ),
        "runtime": float(runtime),
    }


def call_algorithm(folder_name, num_tasks, num_uavs):
    folder_path = ROOT_DIR / folder_name
    main = load_algorithm_main(folder_name)
    with algorithm_import_context(folder_path):
        metrics = main(num_tasks=num_tasks, num_uavs=num_uavs)
    return normalize_metrics(metrics)


def run_experiment(experiment_name, varying_label, varying_values, fixed_label, fixed_value):
    rows = []
    print(f"\n{'=' * 72}")
    print(f"{experiment_name}: varying {varying_label}, fixed {fixed_label}={fixed_value}")
    print(f"{'=' * 72}")

    for value in varying_values:
        print(f"\nRunning {varying_label}={value}")
        num_tasks = value if varying_label == "tasks" else fixed_value
        num_uavs = value if varying_label == "uavs" else fixed_value

        for algorithm_name, folder_name in ALGORITHMS.items():
            print(f"  {algorithm_name}")
            metrics = call_algorithm(
                folder_name=folder_name,
                num_tasks=num_tasks,
                num_uavs=num_uavs,
            )
            rows.append(
                {
                    "algorithm": algorithm_name,
                    varying_label: value,
                    "num_tasks": num_tasks,
                    "num_uavs": num_uavs,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def save_metric_tables(df, varying_label, experiment_slug):
    csv_path = TABLE_DIR / f"{experiment_slug}.csv"
    html_path = TABLE_DIR / f"{experiment_slug}.html"
    markdown_path = TABLE_DIR / f"{experiment_slug}.md"

    df.to_csv(csv_path, index=False)
    df.to_html(html_path, index=False, float_format=lambda value: f"{value:.4f}")
    markdown_path.write_text(dataframe_to_markdown(df), encoding="utf-8")

    for metric, _label in METRICS:
        pivot = df.pivot(index=varying_label, columns="algorithm", values=metric)
        pivot.to_csv(TABLE_DIR / f"{experiment_slug}_{metric}_pivot.csv")
        (TABLE_DIR / f"{experiment_slug}_{metric}_pivot.md").write_text(
            dataframe_to_markdown(pivot.reset_index()),
            encoding="utf-8",
        )


def dataframe_to_markdown(df):
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _index, row in df.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in df.columns) + " |")
    return "\n".join(lines) + "\n"


def save_table_image(df, varying_label, experiment_slug):
    display_df = df.copy()
    for metric, _label in METRICS:
        display_df[metric] = display_df[metric].map(lambda value: f"{value:.4f}")

    columns = ["algorithm", varying_label, "completion_rate", "high_priority_completion_rate", "runtime"]
    display_df = display_df[columns]

    height = max(4, 0.35 * len(display_df) + 1)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9ead3")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f8fa")
    fig.tight_layout()
    fig.savefig(TABLE_DIR / f"{experiment_slug}_table.png", dpi=200)
    plt.close(fig)


def create_metric_plot(
    df,
    varying_label,
    metric,
    ylabel,
    experiment_slug,
    fixed_label,
    fixed_value,
):
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    for algorithm in ALGORITHMS:
        if algorithm not in set(df["algorithm"]):
            continue

        subset = (
            df[df["algorithm"] == algorithm]
            .sort_values(varying_label)
        )

        ax.plot(
            subset[varying_label],
            subset[metric],
            color=ALGORITHM_COLORS.get(algorithm),
            marker=ALGORITHM_MARKERS.get(algorithm, "o"),
            markersize=7.5,
            linewidth=2.6,
            label=algorithm,
        )

    ax.set_xlabel(
        f"Number of {varying_label.title()}",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylabel(
        ylabel,
        fontsize=13,
        fontweight="bold",
    )

    title = (
        f"{ylabel} vs Number of {varying_label.title()}"
        f"\n({fixed_label}={fixed_value})"
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax.set_xticks(sorted(df[varying_label].unique()))

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    if "rate" in metric:
        ymax = max(df[metric].max() * 1.05, 1.0)
        ax.set_ylim(0, ymax)

    legend = ax.legend(
        fontsize=9.5,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(len(ALGORITHMS), 5),
    )

    legend.get_frame().set_alpha(0.95)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    standard_path = GRAPH_DIR / f"{experiment_slug}_{metric}.png"

    publication_path = (
        PUBLICATION_GRAPH_DIR
        / f"{experiment_slug}_{metric}_publication.png"
    )

    fig.savefig(
        standard_path,
        dpi=200,
        bbox_inches="tight",
    )

    try:
        fig.savefig(
            publication_path,
            dpi=300,
            bbox_inches="tight",
        )
    except OSError as exc:
        print(f"Warning: could not save publication graph {publication_path}: {exc}")

    plt.close(fig)

def save_outputs(df, varying_label, experiment_slug, fixed_label, fixed_value):
    save_metric_tables(df, varying_label, experiment_slug)
    save_table_image(df, varying_label, experiment_slug)
    for metric, label in METRICS:
        create_metric_plot(
            df,
            varying_label,
            metric,
            label,
            experiment_slug,
            fixed_label,
            fixed_value,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare UAV scheduling algorithms across task and UAV scalability experiments."
    )
    parser.add_argument("--task-counts", nargs="+", type=int, default=list(range(10, 101, 10)))
    parser.add_argument("--uav-counts", nargs="+", type=int, default=list(range(3, 21)))
    parser.add_argument("--fixed-uavs", type=int, default=5)
    parser.add_argument("--fixed-tasks", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    task_df = run_experiment(
        experiment_name="Task scalability",
        varying_label="tasks",
        varying_values=args.task_counts,
        fixed_label="uavs",
        fixed_value=args.fixed_uavs,
    )
    save_outputs(
        task_df,
        "tasks",
        f"varying_tasks_static_{args.fixed_uavs}_uavs",
        "uavs",
        args.fixed_uavs,
    )

    uav_df = run_experiment(
        experiment_name="UAV scalability",
        varying_label="uavs",
        varying_values=args.uav_counts,
        fixed_label="tasks",
        fixed_value=args.fixed_tasks,
    )
    save_outputs(
        uav_df,
        "uavs",
        f"varying_uavs_static_{args.fixed_tasks}_tasks",
        "tasks",
        args.fixed_tasks,
    )

    combined_df = pd.concat([task_df, uav_df], ignore_index=True)
    combined_df.to_csv(TABLE_DIR / "combined_comparison.csv", index=False)
    combined_df.to_html(
        TABLE_DIR / "combined_comparison.html",
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )

    print("\nFinished comparison.")
    print(f"Tables saved in: {TABLE_DIR}")
    print(f"Graphs saved in: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
