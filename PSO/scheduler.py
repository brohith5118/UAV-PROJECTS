import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from common.config import (
    BASE_X,
    BASE_Y,
    ENERGY_PER_METER,
    PSO_COGNITIVE,
    PSO_DEADLINE_PENALTY,
    PSO_DISTANCE_WEIGHT,
    PSO_HIGH_PRIORITY_DEADLINE_MULTIPLIER,
    PSO_INERTIA,
    PSO_INFEASIBLE_PENALTY,
    PSO_ITERATIONS,
    PSO_MAKESPAN_WEIGHT,
    PSO_PRIORITY_REWARD,
    PSO_RETURN_TO_BASE,
    PSO_SEED,
    PSO_SOCIAL,
    PSO_SWARM_SIZE,
    PSO_TARDINESS_WEIGHT,
    PSO_UNASSIGNED_PENALTY,
    UAV_SPEED
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
        scheduler: Optional["PSOScheduler"] = None,
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

