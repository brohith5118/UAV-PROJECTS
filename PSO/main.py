import copy

from environment import generate_demand_map, generate_tasks, generate_uavs
from scheduler import DMMPPRTSAScheduler, PSOScheduler
from visualization import save_all_graphs


PRIORITY_NAMES = {
    1: "Low/Routine",
    2: "Medium/Important",
    3: "High/Critical",
}


def print_uavs(uavs):
    print("\nUAV fleet")
    for uav in uavs:
        print(f"  {uav}")


def _scheduled_events(result):
    events = []
    for route in result.routes.values():
        events.extend(route.scheduled_tasks)
    return events


def _task_percentage(count, total):
    return 0.0 if total == 0 else (count / total) * 100.0


def print_summary(result, tasks, uavs, label="Schedule"):
    scheduled_events = _scheduled_events(result)
    assigned_tasks = [event.task for event in scheduled_events]
    on_time_tasks = [event.task for event in scheduled_events if event.finish_time <= event.task.deadline]
    late_tasks = [event.task for event in scheduled_events if event.finish_time > event.task.deadline]
    total_tasks = len(tasks)
    unassigned_count = len(result.unassigned_tasks)

    print(f"\n{label} schedule summary")
    print("=" * 72)
    print(f"  Total tasks generated          : {total_tasks}")
    print(f"  Tasks assigned to UAVs         : {len(assigned_tasks)} ({_task_percentage(len(assigned_tasks), total_tasks):.1f}%)")
    print(f"  Tasks completed on time        : {len(on_time_tasks)} ({_task_percentage(len(on_time_tasks), total_tasks):.1f}%)")
    print(f"  Assigned but missed deadline   : {len(late_tasks)} ({_task_percentage(len(late_tasks), total_tasks):.1f}%)")
    print(f"  Unassigned / infeasible tasks  : {unassigned_count} ({_task_percentage(unassigned_count, total_tasks):.1f}%)")
    print(f"  Overall assignment rate        : {result.assignment_rate:.1%}")
    print(f"  Deadline success rate          : {_task_percentage(len(on_time_tasks), len(assigned_tasks)):.1f}% of assigned tasks")
    print(f"  Total distance travelled       : {result.total_distance:.2f} m")
    print(f"  Mission makespan               : {result.makespan:.2f} s")
    print(f"  Final schedule fitness         : {result.fitness:.2f}")

    print("\nPriority-wise task completion")
    print("-" * 72)
    for priority in sorted(PRIORITY_NAMES):
        generated = [task for task in tasks if task.priority == priority]
        assigned = [task for task in assigned_tasks if task.priority == priority]
        on_time = [task for task in on_time_tasks if task.priority == priority]
        late = [task for task in late_tasks if task.priority == priority]
        unassigned = [task for task in result.unassigned_tasks if task.priority == priority]
        print(
            f"  P{priority} {PRIORITY_NAMES[priority]:17s}: "
            f"generated={len(generated):2d}, "
            f"assigned={len(assigned):2d}, "
            f"on-time={len(on_time):2d}, "
            f"late={len(late):2d}, "
            f"unassigned={len(unassigned):2d}"
        )

    print("\nTask type completion")
    print("-" * 72)
    for task_type, label in [(-1, "Acquisition-only"), (0, "Light compute"), (1, "Compute-intensive")]:
        generated = [task for task in tasks if task.task_type == task_type]
        assigned = [task for task in assigned_tasks if task.task_type == task_type]
        on_time = [task for task in on_time_tasks if task.task_type == task_type]
        late = [task for task in late_tasks if task.task_type == task_type]
        unassigned = [task for task in result.unassigned_tasks if task.task_type == task_type]
        print(
            f"  Type {task_type:+d} {label:18s}: "
            f"generated={len(generated):2d}, "
            f"assigned={len(assigned):2d}, "
            f"on-time={len(on_time):2d}, "
            f"late={len(late):2d}, "
            f"unassigned={len(unassigned):2d}"
        )

    print("\nPer-UAV workload and resource usage")
    print("-" * 72)
    for uav in uavs:
        route = result.routes.get(uav.uav_id)
        events = [] if route is None else route.scheduled_tasks
        priority_counts = {
            priority: sum(1 for event in events if event.task.priority == priority)
            for priority in PRIORITY_NAMES
        }
        energy_used = uav.max_energy - uav.remaining_energy
        hover_used = uav.max_hover_time - uav.remaining_hover_time
        compute_used = uav.max_compute - uav.remaining_compute
        print(
            f"  UAV {uav.uav_id:02d} type={uav.uav_type:+d}: "
            f"tasks={len(events):2d} "
            f"(P1={priority_counts[1]}, P2={priority_counts[2]}, P3={priority_counts[3]}), "
            f"distance={(route.total_distance if route else 0.0):8.1f} m, "
            f"finish={(route.finish_time if route else 0.0):7.1f} s"
        )
        print(
            f"       resource used: "
            f"energy={energy_used:7.1f}/{uav.max_energy:.1f} J, "
            f"hover={hover_used:7.1f}/{uav.max_hover_time:.1f} s, "
            f"compute={compute_used:6.1f}/{uav.max_compute:.1f} GHz*s"
        )

    if late_tasks:
        print("\nDeadline failures")
        print("-" * 72)
        for task in sorted(late_tasks, key=lambda item: item.finish_time - item.deadline, reverse=True):
            delay = task.finish_time - task.deadline
            print(
                f"  T{task.task_id:02d}: "
                f"priority=P{task.priority}, assigned_uav=U{task.assigned_uav}, "
                f"finish={task.finish_time:.1f}s, deadline={task.deadline:.1f}s, "
                f"delay={delay:.1f}s"
            )

    if result.unassigned_tasks:
        print("\nUnassigned tasks")
        print("-" * 72)
        for task in sorted(result.unassigned_tasks, key=lambda item: (item.priority, item.task_id)):
            print(
                f"  T{task.task_id:02d}: "
                f"priority=P{task.priority}, type={task.task_type:+d}, "
                f"energy={task.energy_cost:.1f}J, hover={task.hover_time:.1f}s, "
                f"compute={task.compute_load:.1f}GHz*s, deadline={task.deadline:.1f}s"
            )


def print_routes(result):
    print("\nRoutes")
    for uav_id, route in sorted(result.routes.items()):
        print(
            f"  UAV {uav_id:02d}: "
            f"{len(route.scheduled_tasks)} tasks, "
            f"distance={route.total_distance:.1f} m, "
            f"finish={route.finish_time:.1f} s, "
            f"return={route.return_time:.1f} s"
        )
        for event in route.scheduled_tasks:
            task = event.task
            late = " late" if event.finish_time > task.deadline else ""
            print(
                f"    T{task.task_id:02d} "
                f"pri={task.priority} type={task.task_type:+d} "
                f"start={event.start_time:7.1f}s "
                f"finish={event.finish_time:7.1f}s "
                f"deadline={task.deadline:4.0f}s{late}"
            )

    if result.unassigned_tasks:
        ids = ", ".join(f"T{task.task_id:02d}" for task in result.unassigned_tasks)
        print(f"\nUnassigned: {ids}")


def _clone_scenario(tasks, uavs):
    return copy.deepcopy(tasks), copy.deepcopy(uavs)


def print_comparison(results, total_tasks):
    print("\nAlgorithm comparison")
    print("=" * 72)
    print(
        f"{'Algorithm':18s} {'assigned':>9s} {'on-time':>9s} "
        f"{'late':>7s} {'unassigned':>11s} {'distance(m)':>13s} {'makespan(s)':>12s}"
    )
    for label, result in results:
        assigned = result.completed_count
        late = result.deadline_violations
        on_time = assigned - late
        print(
            f"{label:18s} "
            f"{assigned:4d}/{total_tasks:<4d} "
            f"{on_time:4d}/{total_tasks:<4d} "
            f"{late:7d} "
            f"{len(result.unassigned_tasks):11d} "
            f"{result.total_distance:13.1f} "
            f"{result.makespan:12.1f}"
        )


def main():
    demand_map = generate_demand_map(seed=42)
    tasks, _ = generate_tasks(demand_map=demand_map, seed=42)
    uavs = generate_uavs(seed=99)

    print_uavs(uavs)

    pso_tasks, pso_uavs = _clone_scenario(tasks, uavs)
    dmmp_tasks, dmmp_uavs = _clone_scenario(tasks, uavs)

    pso_result = PSOScheduler().schedule(pso_uavs, pso_tasks)
    dmmp_result = DMMPPRTSAScheduler().schedule(dmmp_uavs, dmmp_tasks)

    print_comparison(
        [("PSO baseline", pso_result), ("DMMP-PR-TSA", dmmp_result)],
        total_tasks=len(tasks),
    )

    print_summary(dmmp_result, dmmp_tasks, dmmp_uavs, label="DMMP-PR-TSA")
    print_routes(dmmp_result)

    saved_graphs = []
    saved_graphs.extend(save_all_graphs(pso_uavs, pso_tasks, demand_map, pso_result, prefix="pso_"))
    saved_graphs.extend(save_all_graphs(dmmp_uavs, dmmp_tasks, demand_map, dmmp_result, prefix="dmmp_"))
    print("\nGenerated graphs")
    for path in saved_graphs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
