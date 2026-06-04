# =========================================================
# TSA-MODULE - Deadline-aware Rollout Reinforcement Learning
# =========================================================

import math
import random

import numpy as np

from config import (
    EPOCHS,
    RL_ALPHA,
    RL_GAMMA,
    EPSILON,
    CD,
    CP,
    CT,
    CC,
    UAV_SPEED,
    ENERGY_PER_METER,
)
from utils import calculate_reward, estimate_finish_time, check_deadline


class TempUAVProxy:
    def __init__(self, max_energy, max_hover_time, max_compute, rem_energy, rem_hover, rem_compute):
        self.max_energy = max_energy
        self.max_hover_time = max_hover_time
        self.max_compute = max_compute
        self.remaining_energy = rem_energy
        self.remaining_hover_time = rem_hover
        self.remaining_compute = rem_compute


class QLearningTrajectoryPlanner:
    """
    Rollout-guided task sequencer with explicit deadline pressure.

    Assignment modules already consume UAV resources to record utilisation, so
    TSA must plan from the UAV's mission budget, not from remaining resources
    after assignment. Otherwise many valid task sets appear infeasible and the
    best reward stays at -inf.
    """

    def __init__(self, uav, tasks, optimize=True):
        self.uav = uav
        self.tasks = list(tasks)
        self.n = len(self.tasks)
        self.optimize = optimize

        if self.n == 0:
            raise ValueError(f"UAV {uav.uav_id}: no assigned tasks for TSA.")

        self.q_table = {}
        self._best_route = None
        self._best_reward = -float("inf")

    def _initial_budget(self):
        return (
            self.uav.max_hover_time,
            self.uav.max_energy,
            self.uav.max_compute,
        )

    def _get_q_values(self, current_state, visited_mask):
        state_key = (current_state, visited_mask)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n, dtype=np.float64)
        return self.q_table[state_key]

    def _travel(self, current_pos, task):
        dist = math.hypot(current_pos[0] - task.x, current_pos[1] - task.y)
        return dist, dist / UAV_SPEED, dist * ENERGY_PER_METER

    def _priority_weight(self, task):
        return {1: 1.0, 2: 0.65, 3: 0.35}.get(task.priority, 0.35)

    def _is_resource_feasible(self, task, travel_t, travel_e, curr_hover, curr_energy, curr_compute):
        return (
            travel_t + task.hover_time <= curr_hover
            and travel_e + task.energy_cost <= curr_energy
            and task.compute_load <= curr_compute
        )

    def _deadline_reward(self, task, finish_time):
        slack = task.deadline - finish_time
        priority = self._priority_weight(task)

        if slack >= 0:
            slack_ratio = min(slack / max(task.deadline, 1.0), 1.0)
            urgency_bonus = 90.0 * priority * (1.0 - slack_ratio)
            on_time_bonus = 120.0 * priority
            return on_time_bonus + urgency_bonus

        lateness_ratio = min((-slack) / max(task.deadline, 1.0), 1.0)
        return -(160.0 + 260.0 * priority * lateness_ratio)

    def _future_deadline_pressure(self, next_pos, remaining_indices, finish_time):
        if not remaining_indices:
            return 0.0

        pressure = 0.0
        for idx in remaining_indices:
            task = self.tasks[idx]
            _, travel_t, _ = self._travel(next_pos, task)
            earliest_finish = finish_time + travel_t + task.hover_time
            slack = task.deadline - earliest_finish
            priority = self._priority_weight(task)

            if slack >= 0:
                pressure += 18.0 * priority
                if slack < 0.15 * task.deadline:
                    pressure += 25.0 * priority
            else:
                pressure -= 45.0 * priority

        return pressure / max(len(remaining_indices), 1)

    def _transition_reward(
        self,
        current_pos,
        task,
        curr_time,
        curr_hover,
        curr_energy,
        curr_compute,
        remaining_after,
    ):
        dist, travel_t, travel_e = self._travel(current_pos, task)
        finish_time = curr_time + travel_t + task.hover_time

        uav_proxy = TempUAVProxy(
            self.uav.max_energy,
            self.uav.max_hover_time,
            self.uav.max_compute,
            curr_energy,
            curr_hover,
            curr_compute,
        )

        reward = calculate_reward(current_pos, task, uav_proxy, CD, CP, CT, CC)
        reward += self._deadline_reward(task, finish_time)
        reward += self._future_deadline_pressure(
            (task.x, task.y),
            remaining_after,
            finish_time,
        )

        if not self._is_resource_feasible(task, travel_t, travel_e, curr_hover, curr_energy, curr_compute):
            reward -= 350.0

        return max(-600.0, min(600.0, reward))

    def _rollout_tail_value(
        self,
        current_pos,
        unvisited,
        curr_time,
        curr_hover,
        curr_energy,
        curr_compute,
    ):
        value = 0.0
        discount = 1.0
        remaining = set(unvisited)
        pos = current_pos
        time_now = curr_time
        hover = curr_hover
        energy = curr_energy
        compute = curr_compute

        while remaining:
            feasible = []
            for idx in remaining:
                task = self.tasks[idx]
                _, travel_t, travel_e = self._travel(pos, task)
                if self._is_resource_feasible(task, travel_t, travel_e, hover, energy, compute):
                    feasible.append(idx)

            if not feasible:
                value -= discount * 200.0 * len(remaining)
                break

            best_idx = max(
                feasible,
                key=lambda i: self._transition_reward(
                    pos,
                    self.tasks[i],
                    time_now,
                    hover,
                    energy,
                    compute,
                    remaining - {i},
                ),
            )
            task = self.tasks[best_idx]
            reward = self._transition_reward(pos, task, time_now, hover, energy, compute, remaining - {best_idx})
            value += discount * reward

            _, travel_t, travel_e = self._travel(pos, task)
            time_now += travel_t + task.hover_time
            hover -= travel_t + task.hover_time
            energy -= travel_e + task.energy_cost
            compute -= task.compute_load
            pos = (task.x, task.y)
            remaining.remove(best_idx)
            discount *= RL_GAMMA

        return value

    def _select_rollout_action(
        self,
        current_pos,
        visited,
        feasible,
        curr_time,
        curr_hover,
        curr_energy,
        curr_compute,
    ):
        best_action = feasible[0]
        best_value = -float("inf")

        for idx in feasible:
            task = self.tasks[idx]
            remaining_after = set(range(self.n)) - visited - {idx}
            immediate = self._transition_reward(
                current_pos,
                task,
                curr_time,
                curr_hover,
                curr_energy,
                curr_compute,
                remaining_after,
            )

            _, travel_t, travel_e = self._travel(current_pos, task)
            next_value = self._rollout_tail_value(
                (task.x, task.y),
                remaining_after,
                curr_time + travel_t + task.hover_time,
                curr_hover - travel_t - task.hover_time,
                curr_energy - travel_e - task.energy_cost,
                curr_compute - task.compute_load,
            )
            value = immediate + RL_GAMMA * next_value

            if value > best_value:
                best_value = value
                best_action = idx

        return best_action

    def _select_action(
        self,
        current_state,
        visited_mask,
        current_pos,
        visited,
        feasible,
        curr_time,
        curr_hover,
        curr_energy,
        curr_compute,
        epsilon,
    ):
        if random.random() < epsilon:
            return random.choice(feasible)

        q_vals = self._get_q_values(current_state, visited_mask)
        q_ready = any(abs(q_vals[idx]) > 1e-9 for idx in feasible)

        rollout_action = self._select_rollout_action(
            current_pos,
            visited,
            feasible,
            curr_time,
            curr_hover,
            curr_energy,
            curr_compute,
        )

        if not q_ready:
            return rollout_action

        q_action = max(feasible, key=lambda idx: q_vals[idx])
        if q_vals[q_action] > q_vals[rollout_action] + 25.0:
            return q_action
        return rollout_action

    def _run_episode(self, epsilon):
        current_pos = (self.uav.x, self.uav.y)
        current_state = 0
        visited = set()
        visited_mask = 0
        route_indices = []
        episode_reward = 0.0
        curr_time = 0.0
        curr_hover, curr_energy, curr_compute = self._initial_budget()

        while len(visited) < self.n:
            feasible = []
            for idx in range(self.n):
                if idx in visited:
                    continue
                task = self.tasks[idx]
                _, travel_t, travel_e = self._travel(current_pos, task)
                if self._is_resource_feasible(task, travel_t, travel_e, curr_hover, curr_energy, curr_compute):
                    feasible.append(idx)

            if not feasible:
                missing = self.n - len(visited)
                episode_reward -= 200.0 * missing
                break

            action = self._select_action(
                current_state,
                visited_mask,
                current_pos,
                visited,
                feasible,
                curr_time,
                curr_hover,
                curr_energy,
                curr_compute,
                epsilon,
            )

            task = self.tasks[action]
            remaining_after = set(range(self.n)) - visited - {action}
            reward = self._transition_reward(
                current_pos,
                task,
                curr_time,
                curr_hover,
                curr_energy,
                curr_compute,
                remaining_after,
            )

            _, travel_t, travel_e = self._travel(current_pos, task)
            next_time = curr_time + travel_t + task.hover_time
            next_hover = curr_hover - travel_t - task.hover_time
            next_energy = curr_energy - travel_e - task.energy_cost
            next_compute = curr_compute - task.compute_load

            next_visited = visited | {action}
            next_visited_mask = visited_mask | (1 << action)
            next_feasible = [i for i in range(self.n) if i not in next_visited]
            next_q_vals = self._get_q_values(action + 1, next_visited_mask)
            max_future_q = max((next_q_vals[i] for i in next_feasible), default=0.0)

            q_vals = self._get_q_values(current_state, visited_mask)
            q_vals[action] = (
                (1.0 - RL_ALPHA) * q_vals[action]
                + RL_ALPHA * (reward + RL_GAMMA * max_future_q)
            )

            episode_reward += reward
            route_indices.append(action)
            visited = next_visited
            visited_mask = next_visited_mask
            current_state = action + 1
            current_pos = (task.x, task.y)
            curr_time = next_time
            curr_hover = next_hover
            curr_energy = next_energy
            curr_compute = next_compute

        if len(route_indices) < self.n:
            missing = [i for i in range(self.n) if i not in set(route_indices)]
            missing.sort(key=lambda i: (self.tasks[i].deadline, self.tasks[i].priority))
            route_indices.extend(missing)

        avg_reward = episode_reward / max(self.n, 1)
        return max(-1000.0, min(1000.0, avg_reward)), route_indices

    def train(self, epochs=EPOCHS, verbose=False):
        """
        Train Q-values while using rollout as the expert policy for exploitation.
        Reward logs are average per-task episode rewards, clipped to [-1000, 1000].
        """
        epsilon = max(EPSILON, 0.35)
        eps_decay = epsilon / max(epochs * 0.75, 1)
        reward_log = []

        for _ in range(epochs):
            ep_reward, route = self._run_episode(epsilon)
            reward_log.append(ep_reward)

            if ep_reward > self._best_reward:
                self._best_reward = ep_reward
                self._best_route = [self.tasks[i] for i in route]

            epsilon = max(0.03, epsilon - eps_decay)

        if verbose:
            print(
                f"  UAV {self.uav.uav_id} TSA: {epochs} eps, "
                f"best avg reward={self._best_reward:.2f}"
            )

        return reward_log

    def get_best_route(self):
        if self._best_route:
            return list(self._best_route)
        return sorted(self.tasks, key=lambda task: (task.deadline, task.priority))

    def reorder_by_deadline(self, route):
        """
        Local search that improves deadline compliance without blindly sorting by EDF.
        """
        route = list(route)

        def score(candidate):
            timeline = estimate_finish_time(self.uav, candidate, UAV_SPEED)
            late_penalty = 0.0
            on_time_weight = 0.0
            for task, finish_time in timeline:
                priority = self._priority_weight(task)
                if check_deadline(task, finish_time):
                    on_time_weight += priority
                else:
                    late_penalty += priority * (finish_time - task.deadline)
            distance = 0.0
            prev = (self.uav.x, self.uav.y)
            for task in candidate:
                distance += math.hypot(prev[0] - task.x, prev[1] - task.y)
                prev = (task.x, task.y)
            return on_time_weight * 10000.0 - late_penalty - 0.02 * distance

        best_score = score(route)
        improved = True
        passes = 0

        while improved and passes < max(3, len(route)):
            passes += 1
            improved = False

            for i in range(len(route)):
                task = route[i]
                for j in range(len(route)):
                    if i == j:
                        continue
                    candidate = route[:]
                    candidate.pop(i)
                    candidate.insert(j, task)
                    candidate_score = score(candidate)
                    if candidate_score > best_score:
                        route = candidate
                        best_score = candidate_score
                        improved = True
                        break
                if improved:
                    break

        return route


# ----------------------------------------------------------
# FLEET-LEVEL TSA
# ----------------------------------------------------------

def run_tsa_for_fleet(uavs, epochs=EPOCHS, verbose=True, optimize=True):
    """
    Run TSA rollout planner for all UAVs in the fleet.
    """
    all_routes = {}

    for uav in uavs:
        if not uav.active:
            all_routes[uav.uav_id] = []
            continue

        if not uav.assigned_tasks:
            if verbose:
                print(f"  UAV {uav.uav_id:02d}: no tasks assigned.")
            all_routes[uav.uav_id] = []
            continue

        if verbose:
            print(
                f"  Running TSA (deadline-aware rollout RL) for UAV {uav.uav_id:02d} "
                f"({len(uav.assigned_tasks)} tasks)..."
            )

        planner = QLearningTrajectoryPlanner(uav, uav.assigned_tasks, optimize=optimize)
        planner.train(epochs=epochs, verbose=verbose)

        route = planner.get_best_route()
        if optimize:
            route = planner.reorder_by_deadline(route)

        uav.assigned_tasks = route
        all_routes[uav.uav_id] = route

    return all_routes