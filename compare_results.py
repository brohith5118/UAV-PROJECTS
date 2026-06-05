import os
import sys
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------

# Dynamic module loader

# --------------------------------------------------

def load_main(folder_name):
path = os.path.join(ROOT_DIR, folder_name, "main.py")

```
spec = importlib.util.spec_from_file_location(
    folder_name.replace("-", "_"),
    path
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

return module.main
```

# --------------------------------------------------

# Load algorithms

# --------------------------------------------------

algorithms = {
"DMMP-PR-TSA":
load_main("DMMP-PR-TSA"),

```
"DMMP-R-RL-AC":
    load_main("DMMP-R-RL-AC"),

"Greedy":
    load_main("Greedy-Nearest"),

"PSO":
    load_main("PSO"),
```

}

# --------------------------------------------------

# Experiment 1

# Fixed UAVs

# --------------------------------------------------

TASK_COUNTS = [20, 40, 60, 80, 100]
FIXED_UAVS = 4

results_tasks = []

for task_count in TASK_COUNTS:

```
print(f"\nRunning task-count experiment: {task_count}")

for algo_name, algo in algorithms.items():

    print(f"   {algo_name}")

    metrics = algo(
        num_tasks=task_count,
        num_uavs=FIXED_UAVS
    )

    results_tasks.append({
        "algorithm": algo_name,
        "tasks": task_count,
        "completion_rate":
            metrics["completion_rate"],
        "high_priority_completion_rate":
            metrics["high_priority_completion_rate"],
        "runtime":
            metrics["runtime"],
    })
```

df_tasks = pd.DataFrame(results_tasks)

df_tasks.to_csv(
"task_scalability.csv",
index=False
)

# --------------------------------------------------

# Experiment 2

# Fixed Tasks

# --------------------------------------------------

UAV_COUNTS = [2, 4, 6, 8, 10]
FIXED_TASKS = 100

results_uavs = []

for uav_count in UAV_COUNTS:

```
print(f"\nRunning UAV-count experiment: {uav_count}")

for algo_name, algo in algorithms.items():

    print(f"   {algo_name}")

    metrics = algo(
        num_tasks=FIXED_TASKS,
        num_uavs=uav_count
    )

    results_uavs.append({
        "algorithm": algo_name,
        "uavs": uav_count,
        "completion_rate":
            metrics["completion_rate"],
        "high_priority_completion_rate":
            metrics["high_priority_completion_rate"],
        "runtime":
            metrics["runtime"],
    })
```

df_uavs = pd.DataFrame(results_uavs)

df_uavs.to_csv(
"uav_scalability.csv",
index=False
)

# --------------------------------------------------

# Plot helper

# --------------------------------------------------

def create_plot(
dataframe,
x_column,
y_column,
xlabel,
ylabel,
filename
):

```
plt.figure(figsize=(8,5))

for algo in dataframe["algorithm"].unique():

    subset = dataframe[
        dataframe["algorithm"] == algo
    ]

    plt.plot(
        subset[x_column],
        subset[y_column],
        marker="o",
        label=algo
    )

plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(filename)
plt.close()
```

# --------------------------------------------------

# TASK SCALABILITY

# --------------------------------------------------

create_plot(
df_tasks,
"tasks",
"completion_rate",
"Number of Tasks",
"Completion Rate",
"completion_vs_tasks.png"
)

create_plot(
df_tasks,
"tasks",
"high_priority_completion_rate",
"Number of Tasks",
"High Priority Completion Rate",
"high_priority_vs_tasks.png"
)

create_plot(
df_tasks,
"tasks",
"runtime",
"Number of Tasks",
"Runtime (seconds)",
"runtime_vs_tasks.png"
)

# --------------------------------------------------

# UAV SCALABILITY

# --------------------------------------------------

create_plot(
df_uavs,
"uavs",
"completion_rate",
"Number of UAVs",
"Completion Rate",
"completion_vs_uavs.png"
)

create_plot(
df_uavs,
"uavs",
"high_priority_completion_rate",
"Number of UAVs",
"High Priority Completion Rate",
"high_priority_vs_uavs.png"
)

create_plot(
df_uavs,
"uavs",
"runtime",
"Number of UAVs",
"Runtime (seconds)",
"runtime_vs_uavs.png"
)

print("\nFinished.")
print("Generated:")
print("  task_scalability.csv")
print("  uav_scalability.csv")
print("  completion_vs_tasks.png")
print("  high_priority_vs_tasks.png")
print("  runtime_vs_tasks.png")
print("  completion_vs_uavs.png")
print("  high_priority_vs_uavs.png")
print("  runtime_vs_uavs.png")
