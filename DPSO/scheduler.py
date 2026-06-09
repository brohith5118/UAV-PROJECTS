import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from common.config import (
    BASE_X,
    BASE_Y,
    ENERGY_PER_METER,
    UAV_SPEED,
)


# Paper-inspired defaults for DPSO with pheromone memory. The paper uses
# smaller c1/c2/c3/omega values when pheromone reinforcement is enabled.
DPSO_SWARM_SIZE = 40
DPSO_ITERATIONS = 100
DPSO_COGNITIVE = 0.5
DPSO_SOCIAL = 0.5
DPSO_INERTIA = 0.5
DPSO_MUTATION = 0.5
DPSO_RHO = 0.9
DPSO_TAU_MIN = 0.05
DPSO_TAU_MAX = 1.0
DPSO_SEED = 17

DPSO_UNASSIGNED_PENALTY = 100000.0
DPSO_DEADLINE_PENALTY = 650.0
DPSO_TARDINESS_WEIGHT = 70.0
DPSO_HIGH_PRIORITY_DEADLINE_MULTIPLIER = 3.0
DPSO_PRIORITY_REWARD = 225.0
DPSO_DISTANCE_WEIGHT = 1.0
DPSO_MAKESPAN_WEIGHT = 20.0
DPSO_RETURN_TO_BASE = True


Edge = Tuple[int, int]
VelocityEdge = Tuple[float, int, int]


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


class DPSOScheduler:
    """
    Discrete PSO with pheromone memory adapted from the paper's DTSP model.

    Particle position is a Hamiltonian cycle over task indices, stored as a
    successor array. Velocity is a probabilistic set of directed edges
    (a, x, y). The UAV-specific decoder respects the common resource,
    deadline, range, type-compatibility, energy, hover-time, and compute
    constraints when converting a tour into per-UAV routes.
    """

    def __init__(
        self,
        swarm_size: int = DPSO_SWARM_SIZE,
        iterations: int = DPSO_ITERATIONS,
        cognitive: float = DPSO_COGNITIVE,
        social: float = DPSO_SOCIAL,
        inertia: float = DPSO_INERTIA,
        mutation: float = DPSO_MUTATION,
        rho: float = DPSO_RHO,
        tau_min: float = DPSO_TAU_MIN,
        tau_max: float = DPSO_TAU_MAX,
        seed: int = DPSO_SEED,
        reset_pheromone_on_change: bool = False,
    ):
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.cognitive = cognitive
        self.social = social
        self.inertia = inertia
        self.mutation = mutation
        self.rho = rho
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.reset_pheromone_on_change = reset_pheromone_on_change
        self.rng = np.random.default_rng(seed)
        self.pheromone: Optional[np.ndarray] = None

    def schedule(self, uavs: Sequence[object], tasks: Sequence[object]) -> ScheduleResult:
        active_uavs = [uav for uav in uavs if getattr(uav, "active", True)]
        if not active_uavs:
            return ScheduleResult({}, list(tasks), DPSO_UNASSIGNED_PENALTY, 0, 0.0, 0.0, 0.0, 0)
        if not tasks:
            return ScheduleResult({uav.uav_id: UAVRoute(uav) for uav in active_uavs}, [], 0.0, 0, 0.0, 0.0, 0.0, 0)

        n = len(tasks)
        self._ensure_pheromone(n)

        positions = [self._random_cycle(n) for _ in range(self.swarm_size)]
        velocities: List[List[VelocityEdge]] = [[] for _ in range(self.swarm_size)]

        personal_best_positions = [position.copy() for position in positions]
        personal_best_scores = [math.inf] * self.swarm_size
        global_best_position: Optional[np.ndarray] = None
        global_best_score = math.inf
        history: List[float] = []

        for iteration in range(self.iterations):
            for idx, position in enumerate(positions):
                result = self._decode_and_score(position, active_uavs, tasks)
                score = result.fitness
                if score < personal_best_scores[idx]:
                    personal_best_scores[idx] = score
                    personal_best_positions[idx] = position.copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = position.copy()

            if global_best_position is None:
                continue

            history.append(global_best_score)
            self._evaporate_and_reinforce(global_best_position)

            for idx, position in enumerate(positions):
                velocity = self._next_velocity(
                    current=position,
                    personal_best=personal_best_positions[idx],
                    global_best=global_best_position,
                    previous_velocity=velocities[idx],
                )
                velocities[idx] = velocity
                positions[idx] = self._next_position(position, velocity, iteration)

        best = self._decode_and_score(global_best_position, active_uavs, tasks)  # type: ignore[arg-type]
        best.history = history
        self._apply_schedule(best)
        return best

    def _ensure_pheromone(self, n: int) -> None:
        if self.pheromone is None or self.pheromone.shape != (n, n) or self.reset_pheromone_on_change:
            self.pheromone = np.full((n, n), self.tau_max, dtype=float)
            np.fill_diagonal(self.pheromone, self.tau_min)

    def _random_cycle(self, n: int) -> np.ndarray:
        order = self.rng.permutation(n)
        return self._cycle_from_order(order)

    def _cycle_from_order(self, order: Sequence[int]) -> np.ndarray:
        successor = np.empty(len(order), dtype=int)
        for idx, node in enumerate(order):
            successor[int(node)] = int(order[(idx + 1) % len(order)])
        return successor

    def _edges(self, position: np.ndarray) -> set[Edge]:
        return {(idx, int(position[idx])) for idx in range(len(position))}

    def _next_velocity(
        self,
        current: np.ndarray,
        personal_best: np.ndarray,
        global_best: np.ndarray,
        previous_velocity: Sequence[VelocityEdge],
    ) -> List[VelocityEdge]:
        current_edges = self._edges(current)
        velocity_by_edge: Dict[Edge, float] = {}

        for probability, source, target in previous_velocity:
            velocity_by_edge[(source, target)] = max(
                velocity_by_edge.get((source, target), 0.0),
                probability * self.inertia,
            )

        for source, target in self._edges(personal_best) - current_edges:
            velocity_by_edge[(source, target)] = max(
                velocity_by_edge.get((source, target), 0.0),
                self.cognitive * float(self.rng.random()),
            )

        for source, target in self._edges(global_best) - current_edges:
            velocity_by_edge[(source, target)] = max(
                velocity_by_edge.get((source, target), 0.0),
                self.social * float(self.rng.random()),
            )

        return [
            (min(max(probability, 0.0), 1.0), source, target)
            for (source, target), probability in velocity_by_edge.items()
            if source != target and probability > 1e-12
        ]

    def _next_position(self, current: np.ndarray, velocity: Sequence[VelocityEdge], iteration: int) -> np.ndarray:
        selected: List[Edge] = []

        for probability, source, target in velocity:
            reinforced = self._pheromone_reinforced_probability(probability, source, target, iteration)
            if float(self.rng.random()) <= reinforced:
                selected.append((source, target))

        for source, target in self._edges(current):
            probability = self.mutation * float(self.rng.random())
            if float(self.rng.random()) <= probability:
                selected.append((source, target))

        use_nearest_neighbor = iteration > 0 and iteration % 50 == 0
        return self._complete_cycle(len(current), selected, use_nearest_neighbor)

    def _pheromone_reinforced_probability(self, probability: float, source: int, target: int, iteration: int) -> float:
        assert self.pheromone is not None
        scale = iteration / max(self.iterations, 1)
        reinforced = probability + (float(self.pheromone[source, target]) - 0.5) * scale
        return min(max(reinforced, 0.0), 1.0)

    def _complete_cycle(self, n: int, selected_edges: Sequence[Edge], nearest_neighbor: bool) -> np.ndarray:
        successor = np.full(n, -1, dtype=int)
        used_targets: set[int] = set()

        for source, target in sorted(selected_edges, key=lambda edge: float(self.pheromone[edge[0], edge[1]]), reverse=True):
            if self._can_add_edge(successor, used_targets, source, target, final_edge=False):
                successor[source] = target
                used_targets.add(target)

        while np.any(successor < 0):
            source = self._next_open_source(successor)
            target = self._choose_completion_target(successor, used_targets, source, nearest_neighbor)
            successor[source] = target
            used_targets.add(target)

        return successor

    def _can_add_edge(self, successor: np.ndarray, used_targets: set[int], source: int, target: int, final_edge: bool) -> bool:
        if source == target or successor[source] != -1 or target in used_targets:
            return False
        if final_edge:
            return True
        n = len(successor)
        cursor = target
        seen = {source}
        while successor[cursor] != -1:
            if cursor in seen:
                return False
            seen.add(cursor)
            cursor = int(successor[cursor])
            if cursor == source and len(used_targets) < n - 1:
                return False
        return cursor != source or len(used_targets) == n - 1

    def _next_open_source(self, successor: np.ndarray) -> int:
        open_sources = np.where(successor < 0)[0]
        outbound_pressure = []
        for source in open_sources:
            pheromone_sum = float(np.sum(self.pheromone[source])) if self.pheromone is not None else 0.0
            outbound_pressure.append((-pheromone_sum, int(source)))
        return min(outbound_pressure)[1]

    def _choose_completion_target(
        self,
        successor: np.ndarray,
        used_targets: set[int],
        source: int,
        nearest_neighbor: bool,
    ) -> int:
        n = len(successor)
        final_edge = len(used_targets) == n - 1
        candidates = [
            target
            for target in range(n)
            if self._can_add_edge(successor, used_targets, source, target, final_edge=final_edge)
        ]
        if not candidates:
            candidates = [target for target in range(n) if target != source and target not in used_targets]
        if not candidates:
            candidates = [target for target in range(n) if target != source]

        if nearest_neighbor:
            return min(candidates, key=lambda target: abs(target - source))

        assert self.pheromone is not None
        weights = np.array([max(self.pheromone[source, target], self.tau_min) for target in candidates], dtype=float)
        weights = weights / weights.sum()
        return int(self.rng.choice(candidates, p=weights))

    def _evaporate_and_reinforce(self, best_position: np.ndarray) -> None:
        assert self.pheromone is not None
        self.pheromone *= self.rho
        for source, target in self._edges(best_position):
            self.pheromone[source, target] += 1.0 - self.rho
        np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)
        np.fill_diagonal(self.pheromone, self.tau_min)

    def _order_from_cycle(self, position: np.ndarray, tasks: Sequence[object]) -> List[int]:
        if len(position) == 0:
            return []
        start = min(range(len(tasks)), key=lambda idx: (tasks[idx].priority, tasks[idx].deadline, tasks[idx].task_id))
        order = []
        seen = set()
        cursor = start
        while cursor not in seen and len(order) < len(position):
            seen.add(cursor)
            order.append(cursor)
            cursor = int(position[cursor])
        for idx in range(len(position)):
            if idx not in seen:
                order.append(idx)
        return order

    def _decode_and_score(self, position: np.ndarray, uavs: Sequence[object], tasks: Sequence[object]) -> ScheduleResult:
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

        for task_idx in self._order_from_cycle(position, tasks):
            task = tasks[task_idx]
            candidates = []
            for uav in uavs:
                event = self._preview_schedule_on_state(states[uav.uav_id], task)
                if event is None:
                    continue
                tardiness = max(0.0, event.finish_time - task.deadline)
                load = len(states[uav.uav_id]["route"].scheduled_tasks)
                priority_gain = 4 - task.priority
                score = (
                    tardiness * (3.0 if task.priority == 1 else 1.0),
                    event.finish_time,
                    event.travel_distance,
                    load,
                    -priority_gain,
                )
                candidates.append((score, uav, event))

            if not candidates:
                unassigned.append(task)
                continue

            _score, uav, _event = min(candidates, key=lambda item: item[0])
            event = self._try_schedule_on_state(states[uav.uav_id], task)
            if event is None:
                unassigned.append(task)
                continue

            route = states[uav.uav_id]["route"]
            route.scheduled_tasks.append(event)
            route.total_distance += event.travel_distance
            route.total_energy += task.energy_cost + event.travel_energy
            route.total_hover_time += task.hover_time + event.travel_time
            route.total_compute += task.compute_load
            route.finish_time = event.finish_time
            priority_score += 4 - task.priority
            if event.finish_time > task.deadline:
                deadline_violations += 1

        routes = {}
        for uav in uavs:
            state = states[uav.uav_id]
            route = state["route"]
            if DPSO_RETURN_TO_BASE and route.scheduled_tasks:
                route.return_distance = math.hypot(state["x"] - state["depot_x"], state["y"] - state["depot_y"])
                route.return_time = route.return_distance / UAV_SPEED
                route.return_energy = route.return_distance * ENERGY_PER_METER
                route.total_distance += route.return_distance
                route.total_energy += route.return_energy
                route.total_hover_time += route.return_time
                route.finish_time += route.return_time
            routes[uav.uav_id] = route

        completed = len(tasks) - len(unassigned)
        total_distance = sum(route.total_distance for route in routes.values())
        makespan = max((route.finish_time for route in routes.values()), default=0.0)
        weighted_deadline_penalty = 0.0
        for route in routes.values():
            for event in route.scheduled_tasks:
                if event.finish_time <= event.task.deadline:
                    continue
                multiplier = DPSO_HIGH_PRIORITY_DEADLINE_MULTIPLIER if event.task.priority == 1 else 1.0
                weighted_deadline_penalty += multiplier * (
                    DPSO_DEADLINE_PENALTY
                    + DPSO_TARDINESS_WEIGHT * (event.finish_time - event.task.deadline)
                )

        fitness = (
            len(unassigned) * DPSO_UNASSIGNED_PENALTY
            + weighted_deadline_penalty
            + total_distance * DPSO_DISTANCE_WEIGHT
            + makespan * DPSO_MAKESPAN_WEIGHT
            - priority_score * DPSO_PRIORITY_REWARD
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

    def _preview_schedule_on_state(self, state: dict, task: object) -> Optional[ScheduledTask]:
        snapshot = state.copy()
        return self._try_schedule_on_state(snapshot, task)

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
        if DPSO_RETURN_TO_BASE:
            return_distance = math.hypot(task.x - state["depot_x"], task.y - state["depot_y"])
            reserve_energy = return_distance * ENERGY_PER_METER
            reserve_hover = return_distance / UAV_SPEED

        if distance > state["hover"] * UAV_SPEED + 1e-9:
            return None
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
                task.completed = event.finish_time <= task.deadline
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
        self,
        uavs: Sequence[object],
        pending_tasks: Sequence[object],
        new_tasks: Optional[Iterable[object]] = None,
        failed_uav_ids: Optional[Iterable[int]] = None,
    ) -> ScheduleResult:
        failed = set(failed_uav_ids or [])
        for uav in uavs:
            if uav.uav_id in failed:
                uav.active = False

        combined_tasks = list(pending_tasks)
        if new_tasks:
            combined_tasks.extend(new_tasks)
        if self.reset_pheromone_on_change:
            self.pheromone = None
        return self.schedule(uavs, combined_tasks)
