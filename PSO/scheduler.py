import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from config import (
    ALPHA,
    BASE_X,
    BASE_Y,
    C_COMP,
    C_PHI,
    C_RES,
    C_S,
    C_TIME,
    CC,
    CD,
    CP,
    CT,
    ENERGY_PER_METER,
    EPOCHS,
    EPSILON,
    GAMMA,
    ITERATIONS,
    LAMBDA_TV,
    PSO_COGNITIVE,
    PSO_DEADLINE_PENALTY,
    PSO_HIGH_PRIORITY_DEADLINE_MULTIPLIER,
    PSO_DISTANCE_WEIGHT,
    PSO_INERTIA,
    PSO_INFEASIBLE_PENALTY,
    PSO_ITERATIONS,
    PSO_MAKESPAN_WEIGHT,
    PSO_PRIORITY_REWARD,
    PSO_SEED,
    PSO_SOCIAL,
    PSO_SWARM_SIZE,
    PSO_TARDINESS_WEIGHT,
    PSO_UNASSIGNED_PENALTY,
    PSO_RETURN_TO_BASE,
    RHO,
    RL_ALPHA,
    RL_GAMMA,
    SOM_ITERATIONS,
    SOM_LEARN_RATE,
    SOM_ROWS,
    UAV_SPEED,
)


@dataclass
class ScheduledTask:
    task: object
    uav: object
    start_time: float
    finish_time: float
    travel_distance: float
    travel_time: float
    travel_energy: float


@dataclass
class UAVRoute:
    uav: object
    scheduled_tasks: List[ScheduledTask] = field(default_factory=list)
    total_distance: float = 0.0
    total_energy: float = 0.0
    total_hover_time: float = 0.0
    total_compute: float = 0.0
    finish_time: float = 0.0
    return_distance: float = 0.0
    return_time: float = 0.0
    return_energy: float = 0.0


@dataclass
class ScheduleResult:
    routes: Dict[int, UAVRoute]
    unassigned_tasks: List[object]
    fitness: float
    completed_count: int
    total_priority_score: float
    total_distance: float
    makespan: float
    deadline_violations: int
    history: List[float] = field(default_factory=list)

    @property
    def assignment_rate(self) -> float:
        total = self.completed_count + len(self.unassigned_tasks)
        return 0.0 if total == 0 else self.completed_count / total


class PSOScheduler:
    """Particle Swarm Optimization baseline for UAV task allocation and sequencing."""

    def __init__(
        self,
        swarm_size: int = PSO_SWARM_SIZE,
        iterations: int = PSO_ITERATIONS,
        inertia: float = PSO_INERTIA,
        cognitive: float = PSO_COGNITIVE,
        social: float = PSO_SOCIAL,
        seed: int = PSO_SEED,
    ):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.rng = np.random.default_rng(seed)

    def schedule(self, uavs: Sequence[object], tasks: Sequence[object]) -> ScheduleResult:
        active_uavs = [uav for uav in uavs if getattr(uav, "active", True)]
        if not active_uavs:
            return ScheduleResult({}, list(tasks), PSO_INFEASIBLE_PENALTY, 0, 0.0, 0.0, 0.0, 0)
        if not tasks:
            return ScheduleResult({uav.uav_id: UAVRoute(uav) for uav in active_uavs}, [], 0.0, 0, 0.0, 0.0, 0.0, 0)

        dimensions = len(tasks) * 2
        lower = np.zeros(dimensions)
        upper = np.concatenate(
            [
                np.full(len(tasks), max(len(active_uavs) - 1, 0), dtype=float),
                np.ones(len(tasks), dtype=float),
            ]
        )

        positions = self.rng.uniform(lower, np.maximum(upper, 1e-9), size=(self.swarm_size, dimensions))
        velocities = self.rng.uniform(-1.0, 1.0, size=(self.swarm_size, dimensions))

        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position: Optional[np.ndarray] = None
        global_best_score = math.inf
        history: List[float] = []

        for _ in range(self.iterations):
            for idx in range(self.swarm_size):
                result = self._decode_and_score(positions[idx], active_uavs, tasks)
                score = result.fitness
                if score < personal_best_scores[idx]:
                    personal_best_scores[idx] = score
                    personal_best_positions[idx] = positions[idx].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[idx].copy()

            history.append(global_best_score)
            if global_best_position is None:
                continue

            r1 = self.rng.random(size=(self.swarm_size, dimensions))
            r2 = self.rng.random(size=(self.swarm_size, dimensions))
            velocities = (
                self.inertia * velocities
                + self.cognitive * r1 * (personal_best_positions - positions)
                + self.social * r2 * (global_best_position - positions)
            )
            positions = np.clip(positions + velocities, lower, upper)

        best = self._decode_and_score(global_best_position, active_uavs, tasks)  # type: ignore[arg-type]
        best.history = history
        self._apply_schedule(best)
        return best

    def _decode_and_score(
        self,
        particle: np.ndarray,
        uavs: Sequence[object],
        tasks: Sequence[object],
    ) -> ScheduleResult:
        task_count = len(tasks)
        assignment_keys = particle[:task_count]
        sequence_keys = particle[task_count:]
        task_order = sorted(range(task_count), key=lambda idx: (sequence_keys[idx], -tasks[idx].priority))

        states = {
            uav.uav_id: {
                "uav": uav,
                "x": float(uav.x),
                "y": float(uav.y),
                "energy": float(uav.max_energy),
                "hover": float(uav.max_hover_time),
                "compute": float(uav.max_compute),
                "time": 0.0,
                "depot_x": float(uav.x),
                "depot_y": float(uav.y),
                "route": UAVRoute(uav),
            }
            for uav in uavs
        }

        unassigned: List[object] = []
        deadline_violations = 0
        priority_score = 0.0

        for task_idx in task_order:
            task = tasks[task_idx]
            preferred_idx = int(round(float(assignment_keys[task_idx])))
            preferred_idx = min(max(preferred_idx, 0), len(uavs) - 1)

            candidates = self._candidate_uav_order(preferred_idx, uavs, states, task)
            scheduled = False
            for uav in candidates:
                event = self._try_schedule_on_state(states[uav.uav_id], task)
                if event is None:
                    continue
                route = states[uav.uav_id]["route"]
                route.scheduled_tasks.append(event)
                route.total_distance += event.travel_distance
                route.total_energy += task.energy_cost + event.travel_energy
                route.total_hover_time += task.hover_time + event.travel_time
                route.total_compute += task.compute_load
                route.finish_time = event.finish_time
                priority_score += task.priority
                if event.finish_time > task.deadline:
                    deadline_violations += 1
                scheduled = True
                break
            if not scheduled:
                unassigned.append(task)

        routes = {}
        for uav in uavs:
            state = states[uav.uav_id]
            route = state["route"]
            if PSO_RETURN_TO_BASE and route.scheduled_tasks:
                route.return_distance = math.hypot(state["x"] - state["depot_x"], state["y"] - state["depot_y"])
                route.return_time = route.return_distance / UAV_SPEED
                route.return_energy = route.return_distance * ENERGY_PER_METER
                route.total_distance += route.return_distance
                route.total_energy += route.return_energy
                route.total_hover_time += route.return_time
                route.finish_time += route.return_time
            routes[uav.uav_id] = route

        completed = task_count - len(unassigned)
        total_distance = sum(route.total_distance for route in routes.values())
        makespan = max((route.finish_time for route in routes.values()), default=0.0)
        weighted_deadline_penalty = 0.0
        for route in routes.values():
            for event in route.scheduled_tasks:
                if event.finish_time > event.task.deadline:
                    multiplier = PSO_HIGH_PRIORITY_DEADLINE_MULTIPLIER if event.task.priority == 3 else 1.0
                    tardiness = event.finish_time - event.task.deadline
                    weighted_deadline_penalty += (
                        PSO_DEADLINE_PENALTY * multiplier
                        + PSO_TARDINESS_WEIGHT * multiplier * tardiness
                    )

        fitness = (
            len(unassigned) * PSO_UNASSIGNED_PENALTY
            + weighted_deadline_penalty
            + total_distance * PSO_DISTANCE_WEIGHT
            + makespan * PSO_MAKESPAN_WEIGHT
            - priority_score * PSO_PRIORITY_REWARD
        )

        return ScheduleResult(
            routes=routes,
            unassigned_tasks=unassigned,
            fitness=fitness,
            completed_count=completed,
            total_priority_score=priority_score,
            total_distance=total_distance,
            makespan=makespan,
            deadline_violations=deadline_violations,
        )

    def _candidate_uav_order(self, preferred_idx: int, uavs: Sequence[object], states: Dict[int, dict], task: object) -> List[object]:
        ranked = []
        for idx, uav in enumerate(uavs):
            state = states[uav.uav_id]
            distance = math.hypot(state["x"] - task.x, state["y"] - task.y)
            preference_gap = abs(idx - preferred_idx)
            ranked.append((preference_gap, distance, uav))
        return [uav for _, _, uav in sorted(ranked, key=lambda item: (item[0], item[1]))]

    def _try_schedule_on_state(self, state: dict, task: object) -> Optional[ScheduledTask]:
        uav = state["uav"]
        if not uav.is_compatible(task):
            return None

        distance = math.hypot(state["x"] - task.x, state["y"] - task.y)
        travel_time = distance / UAV_SPEED
        travel_energy = distance * ENERGY_PER_METER
        required_energy = task.energy_cost + travel_energy
        required_hover = task.hover_time + travel_time
        required_compute = task.compute_load
        reserve_energy = 0.0
        reserve_hover = 0.0
        if PSO_RETURN_TO_BASE:
            return_distance = math.hypot(task.x - state["depot_x"], task.y - state["depot_y"])
            reserve_energy = return_distance * ENERGY_PER_METER
            reserve_hover = return_distance / UAV_SPEED

        if state["energy"] + 1e-9 < required_energy + reserve_energy:
            return None
        if state["hover"] + 1e-9 < required_hover + reserve_hover:
            return None
        if state["compute"] + 1e-9 < required_compute:
            return None

        start_time = state["time"] + travel_time
        finish_time = start_time + task.hover_time

        state["energy"] -= required_energy
        state["hover"] -= required_hover
        state["compute"] -= required_compute
        state["time"] = finish_time
        state["x"] = float(task.x)
        state["y"] = float(task.y)

        return ScheduledTask(
            task=task,
            uav=uav,
            start_time=start_time,
            finish_time=finish_time,
            travel_distance=distance,
            travel_time=travel_time,
            travel_energy=travel_energy,
        )

    def _apply_schedule(self, result: ScheduleResult) -> None:
        for route in result.routes.values():
            uav = route.uav
            uav.clear_tasks()
            uav.reset_resources()
            for event in route.scheduled_tasks:
                task = event.task
                task.assigned_uav = uav.uav_id
                task.start_time = event.start_time
                task.finish_time = event.finish_time
                task.completed = True
                uav.assigned_tasks.append(task)
                uav.remaining_energy -= task.energy_cost + event.travel_energy
                uav.remaining_hover_time -= task.hover_time + event.travel_time
                uav.remaining_compute -= task.compute_load
            uav.remaining_energy -= route.return_energy
            uav.remaining_hover_time -= route.return_time

        for task in result.unassigned_tasks:
            task.assigned_uav = None
            task.start_time = None
            task.finish_time = None
            task.completed = False


    def reschedule_after_event(
        uavs: Sequence[object],
        pending_tasks: Sequence[object],
        new_tasks: Optional[Iterable[object]] = None,
        failed_uav_ids: Optional[Iterable[int]] = None,
        scheduler: Optional[PSOScheduler] = None,
    ) -> ScheduleResult:
        """Re-run PSO after dynamic arrivals or UAV failures."""
        failed = set(failed_uav_ids or [])
        for uav in uavs:
            if uav.uav_id in failed:
                uav.active = False

        combined_tasks = list(pending_tasks)
        if new_tasks:
            combined_tasks.extend(new_tasks)

        pso = scheduler or PSOScheduler()
        return pso.schedule(uavs, combined_tasks)


class DMMPPRTSAScheduler:
    """DMMP-PR-TSA implementation from the attached UAV scheduling paper.

    The implementation follows the paper's three linked modules:
      D   - capacity-constrained power-diagram partitioning, equations (3)-(8)
      PR  - SOM-based feasibility-aware pre/re-assignment, equations (15)-(26)
      TSA - Q-learning task sequence adjustment, equations (27)-(30)

    The local project represents paper grid cells as Task objects, so the D module
    partitions the task set instead of every raster cell.
    """

    def __init__(
        self,
        partition_iterations: int = ITERATIONS,
        som_iterations: int = SOM_ITERATIONS,
        rl_epochs: int = EPOCHS,
        seed: int = PSO_SEED + 101,
    ):
        self.partition_iterations = partition_iterations
        self.som_iterations = som_iterations
        self.rl_epochs = rl_epochs
        self.rng = np.random.default_rng(seed)

    def schedule(self, uavs: Sequence[object], tasks: Sequence[object]) -> ScheduleResult:
        active_uavs = [uav for uav in uavs if getattr(uav, "active", True)]
        if not active_uavs:
            return ScheduleResult({}, list(tasks), PSO_INFEASIBLE_PENALTY, 0, 0.0, 0.0, 0.0, 0)
        if not tasks:
            return ScheduleResult({uav.uav_id: UAVRoute(uav) for uav in active_uavs}, [], 0.0, 0, 0.0, 0.0, 0.0, 0)

        partitions = self._partition_tasks(active_uavs, tasks)
        assignments, pr_unassigned = self._som_preassign(active_uavs, tasks, partitions)
        result = self._tsa_build_schedule(active_uavs, assignments, pr_unassigned)
        result.history = []
        self._apply_schedule(result)
        return result

    # ------------------------------------------------------------------
    # D module: capacity-constrained power-diagram partitioning
    # ------------------------------------------------------------------

    def _partition_tasks(self, uavs: Sequence[object], tasks: Sequence[object]) -> Dict[int, List[object]]:
        mu = {
            uav.uav_id: {"energy": 0.0, "hover": 0.0, "compute": 0.0}
            for uav in uavs
        }
        assignment: Dict[int, int] = {}

        for _ in range(self.partition_iterations):
            loads = {
                uav.uav_id: {"energy": 0.0, "hover": 0.0, "compute": 0.0}
                for uav in uavs
            }

            for task in tasks:
                best_uav = min(
                    uavs,
                    key=lambda uav: self._partition_cost(uav, task, mu[uav.uav_id]),
                )
                assignment[task.task_id] = best_uav.uav_id
                loads[best_uav.uav_id]["energy"] += task.energy_cost
                loads[best_uav.uav_id]["hover"] += task.hover_time
                loads[best_uav.uav_id]["compute"] += task.compute_load

            for uav in uavs:
                uid = uav.uav_id
                mu[uid]["energy"] = max(0.0, mu[uid]["energy"] + RHO * (loads[uid]["energy"] - uav.max_energy))
                mu[uid]["hover"] = max(0.0, mu[uid]["hover"] + RHO * (loads[uid]["hover"] - uav.max_hover_time))
                mu[uid]["compute"] = max(0.0, mu[uid]["compute"] + RHO * (loads[uid]["compute"] - uav.max_compute))

        assignment = self._local_refine_partition(uavs, tasks, assignment, mu)
        partitions = {uav.uav_id: [] for uav in uavs}
        for task in tasks:
            partitions[assignment[task.task_id]].append(task)
        return partitions

    def _partition_cost(self, uav: object, task: object, multipliers: dict) -> float:
        resource_term = (
            multipliers["energy"] * task.energy_cost
            + multipliers["hover"] * task.hover_time
            + multipliers["compute"] * task.compute_load
        )
        return ALPHA * uav.distance_to(task) - GAMMA * task.priority + resource_term

    def _local_refine_partition(
        self,
        uavs: Sequence[object],
        tasks: Sequence[object],
        assignment: Dict[int, int],
        mu: Dict[int, dict],
    ) -> Dict[int, int]:
        current = dict(assignment)
        task_by_id = {task.task_id: task for task in tasks}
        for task in tasks:
            old_uid = current[task.task_id]
            best_uid = old_uid
            best_delta = 0.0
            for uav in uavs:
                if uav.uav_id == old_uid:
                    continue
                old_uav = next(item for item in uavs if item.uav_id == old_uid)
                old_cost = self._partition_cost(old_uav, task, mu[old_uid])
                new_cost = self._partition_cost(uav, task, mu[uav.uav_id])
                tv_delta = self._tv_delta(task, task_by_id.values(), current, old_uid, uav.uav_id)
                delta = (new_cost - old_cost) + LAMBDA_TV * tv_delta
                if delta < best_delta:
                    best_delta = delta
                    best_uid = uav.uav_id
            current[task.task_id] = best_uid
        return current

    def _tv_delta(
        self,
        task: object,
        tasks: Iterable[object],
        assignment: Dict[int, int],
        old_uid: int,
        new_uid: int,
    ) -> float:
        nearest = sorted(
            (other for other in tasks if other.task_id != task.task_id),
            key=lambda other: math.hypot(task.x - other.x, task.y - other.y),
        )[:4]
        delta = 0.0
        for other in nearest:
            other_uid = assignment[other.task_id]
            delta += (0 if new_uid == other_uid else 1) - (0 if old_uid == other_uid else 1)
        return delta

    # ------------------------------------------------------------------
    # PR module: improved SOM pre-assignment and dynamic re-assignment
    # ------------------------------------------------------------------

    def _som_preassign(
        self,
        uavs: Sequence[object],
        tasks: Sequence[object],
        partitions: Dict[int, List[object]],
    ) -> tuple[Dict[int, List[object]], List[object]]:
        states = {
            uav.uav_id: {
                "uav": uav,
                "x": float(uav.x),
                "y": float(uav.y),
                "hover": float(uav.max_hover_time),
                "compute": float(uav.max_compute),
                "energy": float(uav.max_energy),
                "depot_x": float(uav.x),
                "depot_y": float(uav.y),
            }
            for uav in uavs
        }
        nodes = self._initial_som_nodes(uavs)
        ordered_tasks = sorted(tasks, key=lambda item: (-item.priority, item.deadline, item.task_id))

        for iteration in range(max(self.som_iterations, 1)):
            task = ordered_tasks[iteration % len(ordered_tasks)]
            winner_uid = self._best_matching_uav(task, uavs, states, partitions, allow_partition_bias=True)
            if winner_uid is None:
                continue
            eta = SOM_LEARN_RATE * (1.0 - iteration / max(self.som_iterations, 1))
            self._update_som_nodes(nodes, task, winner_uid, eta)

        assignments = {uav.uav_id: [] for uav in uavs}
        unassigned: List[object] = []

        for task in ordered_tasks:
            uid = self._best_matching_uav(task, uavs, states, partitions, allow_partition_bias=True)
            if uid is None:
                unassigned.append(task)
                continue
            state = states[uid]
            distance = math.hypot(state["x"] - task.x, state["y"] - task.y)
            travel_time = distance / UAV_SPEED
            travel_energy = distance * ENERGY_PER_METER
            state["x"] = float(task.x)
            state["y"] = float(task.y)
            state["hover"] -= travel_time + task.hover_time
            state["compute"] -= task.compute_load
            state["energy"] -= travel_energy + task.energy_cost
            assignments[uid].append(task)

        return assignments, unassigned

    def _initial_som_nodes(self, uavs: Sequence[object]) -> Dict[tuple[int, int], np.ndarray]:
        nodes = {}
        for row in range(SOM_ROWS):
            for col, uav in enumerate(uavs):
                jitter = self.rng.normal(0.0, 0.01, size=5)
                nodes[(row, uav.uav_id)] = np.array(
                    [uav.x, uav.y, uav.uav_type, uav.max_hover_time, uav.max_compute],
                    dtype=float,
                ) + jitter
        return nodes

    def _update_som_nodes(
        self,
        nodes: Dict[tuple[int, int], np.ndarray],
        task: object,
        winner_uid: int,
        eta: float,
    ) -> None:
        target = np.array([task.x, task.y, task.task_type, task.hover_time, task.compute_load], dtype=float)
        winner_key = min(
            [key for key in nodes if key[1] == winner_uid],
            key=lambda key: np.linalg.norm(nodes[key][:2] - target[:2]),
        )
        for key, value in nodes.items():
            grid_distance = abs(key[0] - winner_key[0]) + abs(key[1] - winner_key[1])
            type_close = abs(value[2] - nodes[winner_key][2]) <= 1.0
            influence = 1.0 if key == winner_key else (math.exp(-grid_distance / C_S) if type_close else 0.0)
            if influence > 0:
                nodes[key] = value + eta * influence * (target - value)

    def _best_matching_uav(
        self,
        task: object,
        uavs: Sequence[object],
        states: Dict[int, dict],
        partitions: Dict[int, List[object]],
        allow_partition_bias: bool,
    ) -> Optional[int]:
        scored = []
        for uav in uavs:
            state = states[uav.uav_id]
            distance = math.hypot(state["x"] - task.x, state["y"] - task.y)
            if not uav.is_compatible(task):
                continue
            return_distance = math.hypot(task.x - state["depot_x"], task.y - state["depot_y"])
            diff_time = state["hover"] - (distance + return_distance) / UAV_SPEED - task.hover_time
            diff_comp = state["compute"] - task.compute_load
            diff_energy = state["energy"] - (distance + return_distance) * ENERGY_PER_METER - task.energy_cost
            if diff_time < 0 or diff_comp < 0 or diff_energy < 0:
                continue

            d_phi = 0.0 if abs(uav.uav_type - task.task_type) <= 1 else math.inf
            d_res = math.exp(-C_TIME * diff_time) + math.exp(-C_COMP * diff_comp)
            partition_bonus = -GAMMA if allow_partition_bias and task in partitions.get(uav.uav_id, []) else 0.0
            score = distance**2 + C_PHI * d_phi + C_RES * d_res + partition_bonus
            scored.append((score, uav.uav_id))
        if not scored:
            return None
        return min(scored, key=lambda item: item[0])[1]

    # ------------------------------------------------------------------
    # TSA module: Q-learning sequence adjustment
    # ------------------------------------------------------------------

    def _tsa_build_schedule(
        self,
        uavs: Sequence[object],
        assignments: Dict[int, List[object]],
        unassigned: List[object],
    ) -> ScheduleResult:
        routes = {uav.uav_id: UAVRoute(uav) for uav in uavs}
        still_unassigned = list(unassigned)

        for uav in uavs:
            tasks = assignments.get(uav.uav_id, [])
            if not tasks:
                continue
            sequence = self._learn_task_sequence(uav, tasks)
            state = {
                "uav": uav,
                "x": float(uav.x),
                "y": float(uav.y),
                "energy": float(uav.max_energy),
                "hover": float(uav.max_hover_time),
                "compute": float(uav.max_compute),
                "time": 0.0,
                "depot_x": float(uav.x),
                "depot_y": float(uav.y),
                "route": routes[uav.uav_id],
            }
            for task in sequence:
                event = PSOScheduler()._try_schedule_on_state(state, task)
                if event is None:
                    still_unassigned.append(task)
                    continue
                route = routes[uav.uav_id]
                route.scheduled_tasks.append(event)
                route.total_distance += event.travel_distance
                route.total_energy += task.energy_cost + event.travel_energy
                route.total_hover_time += task.hover_time + event.travel_time
                route.total_compute += task.compute_load
                route.finish_time = event.finish_time

            route = routes[uav.uav_id]
            if PSO_RETURN_TO_BASE and route.scheduled_tasks:
                route.return_distance = math.hypot(state["x"] - state["depot_x"], state["y"] - state["depot_y"])
                route.return_time = route.return_distance / UAV_SPEED
                route.return_energy = route.return_distance * ENERGY_PER_METER
                route.total_distance += route.return_distance
                route.total_energy += route.return_energy
                route.total_hover_time += route.return_time
                route.finish_time += route.return_time

        completed = sum(len(route.scheduled_tasks) for route in routes.values())
        total_distance = sum(route.total_distance for route in routes.values())
        makespan = max((route.finish_time for route in routes.values()), default=0.0)
        deadline_violations = sum(
            1
            for route in routes.values()
            for event in route.scheduled_tasks
            if event.finish_time > event.task.deadline
        )
        priority_score = sum(
            event.task.priority
            for route in routes.values()
            for event in route.scheduled_tasks
        )
        fitness = (
            len(still_unassigned) * PSO_UNASSIGNED_PENALTY
            + deadline_violations * PSO_DEADLINE_PENALTY
            + total_distance * PSO_DISTANCE_WEIGHT
            + makespan * PSO_MAKESPAN_WEIGHT
            - priority_score * PSO_PRIORITY_REWARD
        )

        return ScheduleResult(
            routes=routes,
            unassigned_tasks=still_unassigned,
            fitness=fitness,
            completed_count=completed,
            total_priority_score=priority_score,
            total_distance=total_distance,
            makespan=makespan,
            deadline_violations=deadline_violations,
        )

    def _learn_task_sequence(self, uav: object, tasks: Sequence[object]) -> List[object]:
        if len(tasks) <= 1:
            return list(tasks)

        count = len(tasks)
        q_values = np.zeros((count + 1, count), dtype=float)
        base_state = count

        for _ in range(max(self.rl_epochs, 1)):
            current_state = base_state
            current_x = float(uav.x)
            current_y = float(uav.y)
            residual_time = float(uav.max_hover_time)
            residual_compute = float(uav.max_compute)
            remaining = set(range(count))

            while remaining:
                feasible = [
                    idx for idx in remaining
                    if self._transition_feasible(current_x, current_y, residual_time, residual_compute, tasks[idx], uav.x, uav.y)
                ]
                if not feasible:
                    break
                if self.rng.random() < EPSILON:
                    action = int(self.rng.choice(feasible))
                else:
                    action = max(feasible, key=lambda idx: q_values[current_state, idx])

                task = tasks[action]
                distance = math.hypot(current_x - task.x, current_y - task.y)
                travel_time = distance / UAV_SPEED
                reward = self._tsa_reward(distance, task, uav, residual_time, residual_compute)
                next_state = action
                remaining.remove(action)
                next_feasible = list(remaining)
                future = max((q_values[next_state, idx] for idx in next_feasible), default=0.0)
                q_values[current_state, action] = (
                    (1.0 - RL_ALPHA) * q_values[current_state, action]
                    + RL_ALPHA * (reward + RL_GAMMA * future)
                )
                residual_time -= travel_time + task.hover_time
                residual_compute -= task.compute_load
                current_x = float(task.x)
                current_y = float(task.y)
                current_state = next_state

        sequence = []
        current_state = base_state
        current_x = float(uav.x)
        current_y = float(uav.y)
        residual_time = float(uav.max_hover_time)
        residual_compute = float(uav.max_compute)
        remaining = set(range(count))
        while remaining:
            feasible = [
                idx for idx in remaining
                if self._transition_feasible(current_x, current_y, residual_time, residual_compute, tasks[idx], uav.x, uav.y)
            ]
            if not feasible:
                sequence.extend(sorted((tasks[idx] for idx in remaining), key=lambda task: (-task.priority, task.deadline)))
                break
            action = max(feasible, key=lambda idx: (q_values[current_state, idx], tasks[idx].priority, -tasks[idx].deadline))
            task = tasks[action]
            distance = math.hypot(current_x - task.x, current_y - task.y)
            residual_time -= distance / UAV_SPEED + task.hover_time
            residual_compute -= task.compute_load
            current_x = float(task.x)
            current_y = float(task.y)
            current_state = action
            remaining.remove(action)
            sequence.append(task)
        return sequence

    def _transition_feasible(
        self,
        current_x: float,
        current_y: float,
        residual_time: float,
        residual_compute: float,
        task: object,
        depot_x: float,
        depot_y: float,
    ) -> bool:
        distance = math.hypot(current_x - task.x, current_y - task.y)
        return_distance = math.hypot(task.x - depot_x, task.y - depot_y)
        required_time = (distance + return_distance) / UAV_SPEED + task.hover_time
        return residual_time + 1e-9 >= required_time and residual_compute + 1e-9 >= task.compute_load

    def _tsa_reward(
        self,
        distance: float,
        task: object,
        uav: object,
        residual_time: float,
        residual_compute: float,
    ) -> float:
        endurance_ratio = residual_time / uav.max_hover_time if uav.max_hover_time > 0 else 0.0
        if uav.max_compute > 0:
            compute_ratio = (residual_compute - task.compute_load) / uav.max_compute
        else:
            compute_ratio = 1.0 if task.compute_load <= 0 else -1.0
        return CD * (distance / 1000.0) + CP * task.priority + CT * endurance_ratio + CC * compute_ratio

    def _apply_schedule(self, result: ScheduleResult) -> None:
        PSOScheduler()._apply_schedule(result)
