"""Lightweight heuristic-guided Q-learning for UAV task sequencing."""

from collections import defaultdict
import math
import os
import random
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from common.config import (  # noqa: E402
    ENERGY_PER_METER,
    EPOCHS,
    GRID_RESOLUTION,
    MAP_HEIGHT,
    MAP_WIDTH,
    UAV_SPEED,
)


# Tuned for the small per-UAV task sets produced by the D-module. Rewards use
# normalized features, so these values do not depend on the map's metre scale.
LEARNING_RATE = 0.18
DISCOUNT_FACTOR = 0.92
EPSILON_START = 0.35
EPSILON_MIN = 0.03
GUIDANCE_WEIGHT = 8.0
LOCAL_SEARCH_PASSES = 6
STATE_BINS = 5
TOLERANCE = 1e-9

PRIORITY_VALUE = {1: 3.0, 2: 2.0, 3: 1.0}


def _distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


class HeuristicGuidedQLearningPlanner:
    """Plan a resource-feasible task sequence with guided tabular Q-learning."""

    def __init__(self, uav, tasks=None, optimize=True, seed=None):
        self.uav = uav
        self.tasks = list(uav.assigned_tasks if tasks is None else tasks)
        self.num_tasks = len(self.tasks)
        self.n = self.num_tasks
        self.optimize = optimize
        self.q_table = defaultdict(float)

        default_seed = 104729 + 1009 * int(getattr(uav, "uav_id", 0))
        self._rng = random.Random(default_seed if seed is None else seed)
        self._best_route = None
        self._best_key = None
        self._best_reward = -math.inf

        self._map_diagonal = max(
            1.0,
            math.hypot(
                MAP_WIDTH * GRID_RESOLUTION,
                MAP_HEIGHT * GRID_RESOLUTION,
            ),
        )
        self._capacities = (
            self._available("energy"),
            self._available("hover_time"),
            self._available("compute"),
        )

        if self.num_tasks:
            self._seed_with_heuristics()

    def _available(self, resource):
        maximum = max(0.0, float(getattr(self.uav, f"max_{resource}", 0.0)))
        remaining = float(getattr(self.uav, f"remaining_{resource}", maximum))
        return min(maximum, max(0.0, remaining))

    def _start_position(self):
        x = getattr(self.uav, "curr_x", self.uav.x)
        y = getattr(self.uav, "curr_y", self.uav.y)
        if math.isfinite(x) and math.isfinite(y):
            return x, y
        return self.uav.x, self.uav.y

    def distance(self, task_a, task_b):
        """Backward-compatible task-to-task distance helper."""
        return _distance(task_a.x, task_a.y, task_b.x, task_b.y)

    def _initial_rollout(self):
        x, y = self._start_position()
        return {
            "x": x,
            "y": y,
            "clock": 0.0,
            "energy": 0.0,
            "hover": 0.0,
            "compute": 0.0,
        }

    def _transition(self, rollout, action):
        task = self.tasks[action]
        travel = _distance(rollout["x"], rollout["y"], task.x, task.y)
        travel_time = travel / UAV_SPEED
        return {
            "x": task.x,
            "y": task.y,
            "clock": rollout["clock"] + travel_time + task.hover_time,
            "energy": rollout["energy"] + travel * ENERGY_PER_METER + task.energy_cost,
            "hover": rollout["hover"] + travel_time + task.hover_time,
            "compute": rollout["compute"] + task.compute_load,
            "travel": travel,
        }

    @staticmethod
    def _ratio(used, capacity):
        if capacity <= TOLERANCE:
            return 0.0 if used <= TOLERANCE else math.inf
        return used / capacity

    def _resource_ratios(self, rollout):
        return (
            self._ratio(rollout["energy"], self._capacities[0]),
            self._ratio(rollout["hover"], self._capacities[1]),
            self._ratio(rollout["compute"], self._capacities[2]),
        )

    def _is_feasible_transition(self, rollout, action):
        candidate = self._transition(rollout, action)
        return all(ratio <= 1.0 + TOLERANCE for ratio in self._resource_ratios(candidate))

    def get_actions(self, visited_mask, rollout=None):
        """Return unvisited actions, optionally filtered by physical feasibility."""
        actions = [
            idx
            for idx in range(self.num_tasks)
            if not visited_mask & (1 << idx)
        ]
        if rollout is None:
            return actions
        return [idx for idx in actions if self._is_feasible_transition(rollout, idx)]

    @staticmethod
    def _bin(ratio):
        if not math.isfinite(ratio):
            return STATE_BINS
        return min(STATE_BINS - 1, max(0, int(ratio * STATE_BINS)))

    def _state(self, current_index, visited_mask, rollout):
        """Compact state: location/set plus coarse time and resource pressure."""
        max_deadline = max(
            (float(task.deadline) for task in self.tasks),
            default=1.0,
        )
        energy, hover, compute = self._resource_ratios(rollout)
        return (
            current_index,
            visited_mask,
            self._bin(rollout["clock"] / max(max_deadline, 1.0)),
            self._bin(energy),
            self._bin(hover),
            self._bin(compute),
        )

    def _heuristic_score(self, rollout, action, visited_mask):
        """Immediate guidance balancing urgency, proximity, and headroom."""
        task = self.tasks[action]
        candidate = self._transition(rollout, action)
        priority = PRIORITY_VALUE.get(task.priority, 1.0)
        deadline = max(float(task.deadline), 1.0)
        progress = candidate["clock"] / deadline
        tardiness = max(0.0, progress - 1.0)
        on_time = candidate["clock"] <= task.deadline
        ratios = self._resource_ratios(candidate)
        pressure = sum(ratio * ratio for ratio in ratios) / 3.0
        distance_ratio = candidate["travel"] / self._map_diagonal

        remaining = [
            idx
            for idx in range(self.num_tasks)
            if idx != action and not visited_mask & (1 << idx)
        ]
        if remaining:
            next_distance = min(
                _distance(task.x, task.y, self.tasks[idx].x, self.tasks[idx].y)
                for idx in remaining
            ) / self._map_diagonal
        else:
            next_distance = 0.0

        deadline_value = (
            2.5 * priority * (1.0 - min(progress, 1.0))
            if on_time
            else -8.0 * priority * (1.0 + tardiness)
        )
        return (
            3.0 * priority
            + deadline_value
            - 2.0 * distance_ratio
            - 1.5 * pressure
            - 0.5 * next_distance
        )

    def _transition_reward(self, rollout, action):
        """Normalized reward respecting deadline and resource constraints."""
        task = self.tasks[action]
        candidate = self._transition(rollout, action)
        priority = PRIORITY_VALUE.get(task.priority, 1.0)
        deadline = max(float(task.deadline), 1.0)
        finish_ratio = candidate["clock"] / deadline
        ratios = self._resource_ratios(candidate)
        pressure = sum(ratio * ratio for ratio in ratios) / 3.0

        reward = (
            40.0
            + 35.0 * priority
            - 35.0 * (candidate["travel"] / self._map_diagonal)
            - 25.0 * pressure
        )
        if candidate["clock"] <= task.deadline:
            slack_ratio = max(0.0, 1.0 - finish_ratio)
            reward += 80.0 * priority + 20.0 * priority * slack_ratio
        else:
            tardiness_ratio = finish_ratio - 1.0
            reward -= 120.0 * priority + 80.0 * priority * tardiness_ratio
        return max(-500.0, min(500.0, reward))

    def reward(self, current_idx, action):
        """Legacy reward API, evaluated from the partial original-order route."""
        rollout = self._initial_rollout()
        if 0 <= current_idx < self.num_tasks:
            current = self.tasks[current_idx]
            rollout["x"], rollout["y"] = current.x, current.y
        return self._transition_reward(rollout, action)

    def _choose_action(self, state, actions, rollout, visited_mask, epsilon):
        if not actions:
            return None
        if self._rng.random() < epsilon:
            return self._rng.choice(actions)
        return max(
            actions,
            key=lambda action: (
                self.q_table[(state, action)]
                + GUIDANCE_WEIGHT * self._heuristic_score(
                    rollout, action, visited_mask
                ),
                -action,
            ),
        )

    def choose_action(self, state):
        """Legacy epsilon-greedy API for ``(current_index, visited_mask)``."""
        current_index, visited_mask = state
        rollout = self._initial_rollout()
        actions = self.get_actions(visited_mask, rollout)
        expanded_state = self._state(current_index, visited_mask, rollout)
        return self._choose_action(
            expanded_state,
            actions,
            rollout,
            visited_mask,
            EPSILON_START,
        )

    def _run_episode(self, epsilon):
        rollout = self._initial_rollout()
        visited_mask = 0
        current_index = -1
        route = []
        total_reward = 0.0

        while len(route) < self.num_tasks:
            state = self._state(current_index, visited_mask, rollout)
            actions = self.get_actions(visited_mask, rollout)
            if not actions:
                break

            action = self._choose_action(
                state, actions, rollout, visited_mask, epsilon
            )
            reward = self._transition_reward(rollout, action)
            next_rollout = self._transition(rollout, action)
            next_mask = visited_mask | (1 << action)
            next_state = self._state(action, next_mask, next_rollout)
            next_actions = self.get_actions(next_mask, next_rollout)
            max_future = max(
                (self.q_table[(next_state, item)] for item in next_actions),
                default=0.0,
            )

            old_q = self.q_table[(state, action)]
            self.q_table[(state, action)] = old_q + LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * max_future - old_q
            )

            total_reward += reward
            route.append(action)
            visited_mask = next_mask
            current_index = action
            rollout = next_rollout

        return total_reward, route

    def _simulate_route(self, route):
        rollout = self._initial_rollout()
        on_time = high_on_time = 0
        weighted_on_time = 0.0
        weighted_tardiness = 0.0
        total_distance = 0.0
        shaped_reward = 0.0

        for action in route:
            task = self.tasks[action]
            shaped_reward += self._transition_reward(rollout, action)
            rollout = self._transition(rollout, action)
            total_distance += rollout["travel"]
            priority = PRIORITY_VALUE.get(task.priority, 1.0)
            if rollout["clock"] <= task.deadline:
                on_time += 1
                weighted_on_time += priority
                if task.priority == 1:
                    high_on_time += 1
            else:
                weighted_tardiness += priority * (
                    rollout["clock"] - task.deadline
                )

        feasible = (
            len(route) == self.num_tasks
            and all(
                ratio <= 1.0 + TOLERANCE
                for ratio in self._resource_ratios(rollout)
            )
        )
        return {
            "feasible": feasible,
            "on_time": on_time,
            "high_on_time": high_on_time,
            "weighted_on_time": weighted_on_time,
            "weighted_tardiness": weighted_tardiness,
            "distance": total_distance,
            "energy": rollout["energy"],
            "reward": shaped_reward,
        }

    def _route_key(self, route):
        result = self._simulate_route(route)
        return (
            int(result["feasible"]),
            result["weighted_on_time"],
            result["high_on_time"],
            result["on_time"],
            -result["weighted_tardiness"],
            -result["distance"],
            -result["energy"],
        )

    def _greedy_route(self):
        rollout = self._initial_rollout()
        visited_mask = 0
        route = []
        while len(route) < self.num_tasks:
            actions = self.get_actions(visited_mask, rollout)
            if not actions:
                break
            action = max(
                actions,
                key=lambda item: (
                    self._heuristic_score(rollout, item, visited_mask),
                    -item,
                ),
            )
            route.append(action)
            visited_mask |= 1 << action
            rollout = self._transition(rollout, action)
        return route

    def _nearest_route(self):
        rollout = self._initial_rollout()
        visited_mask = 0
        route = []
        while len(route) < self.num_tasks:
            actions = self.get_actions(visited_mask, rollout)
            if not actions:
                break
            action = min(
                actions,
                key=lambda idx: (
                    _distance(
                        rollout["x"],
                        rollout["y"],
                        self.tasks[idx].x,
                        self.tasks[idx].y,
                    ),
                    self.tasks[idx].priority,
                    self.tasks[idx].deadline,
                    idx,
                ),
            )
            route.append(action)
            visited_mask |= 1 << action
            rollout = self._transition(rollout, action)
        return route

    def _improve_route(self, route):
        """Bounded one-task relocation search using the mission objective."""
        route = list(route)
        if len(route) < 2 or len(route) != self.num_tasks:
            return route

        best_key = self._route_key(route)
        for _ in range(LOCAL_SEARCH_PASSES):
            improved = False
            for old_pos in range(len(route)):
                action = route[old_pos]
                shortened = route[:old_pos] + route[old_pos + 1 :]
                for new_pos in range(len(route)):
                    candidate = shortened[:new_pos] + [action] + shortened[new_pos:]
                    key = self._route_key(candidate)
                    if key > best_key:
                        route = candidate
                        best_key = key
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return route

    def _seed_with_heuristics(self):
        original = list(range(self.num_tasks))
        priority_deadline = sorted(
            original,
            key=lambda idx: (
                self.tasks[idx].priority,
                self.tasks[idx].deadline,
                idx,
            ),
        )
        candidates = [
            original,
            priority_deadline,
            self._nearest_route(),
            self._greedy_route(),
        ]
        for candidate in candidates:
            if len(candidate) != self.num_tasks:
                continue
            candidate = self._improve_route(candidate)
            key = self._route_key(candidate)
            if self._best_key is None or key > self._best_key:
                self._best_route = candidate
                self._best_key = key
                self._best_reward = self._simulate_route(candidate)["reward"]

    def train(self, episodes=None, epochs=None, verbose=False):
        """Train with decaying exploration and objective-based early stopping."""
        if episodes is None:
            episodes = EPOCHS if epochs is None else epochs
        episodes = max(0, int(episodes))
        if not self.num_tasks:
            return []

        reward_log = []
        decay = (
            (EPSILON_MIN / EPSILON_START) ** (1.0 / max(episodes - 1, 1))
            if episodes
            else 1.0
        )
        epsilon = EPSILON_START
        minimum_episodes = min(episodes, max(80, 10 * self.num_tasks))
        patience = max(50, 5 * self.num_tasks)
        stale_episodes = 0

        for episode in range(episodes):
            episode_reward, route = self._run_episode(epsilon)
            reward_log.append(episode_reward)
            improved = False
            if len(route) == self.num_tasks:
                key = self._route_key(route)
                if self._best_key is None or key > self._best_key:
                    self._best_route = route
                    self._best_key = key
                    self._best_reward = episode_reward
                    improved = True
            stale_episodes = 0 if improved else stale_episodes + 1
            epsilon = max(EPSILON_MIN, epsilon * decay)
            if (
                episode + 1 >= minimum_episodes
                and stale_episodes >= patience
            ):
                break

        if self._best_route is not None and self.optimize:
            improved = self._improve_route(self._best_route)
            if self._route_key(improved) >= self._best_key:
                self._best_route = improved
                self._best_key = self._route_key(improved)
                self._best_reward = self._simulate_route(improved)["reward"]

        if verbose:
            tail = reward_log[-min(50, len(reward_log)) :]
            mean_reward = sum(tail) / len(tail) if tail else 0.0
            print(
                f"  UAV {self.uav.uav_id:02d} TSA: "
                f"{len(reward_log)}/{episodes} episodes, "
                f"last mean={mean_reward:.1f}, best={self._best_reward:.1f}"
            )
        return reward_log

    def extract_route(self):
        """Return the best complete, resource-feasible route found."""
        if not self.num_tasks:
            return []
        if self._best_route is None:
            self._seed_with_heuristics()
        return [self.tasks[idx] for idx in self._best_route]

    def get_best_route(self):
        return self.extract_route()

    def reorder_by_deadline(self, route):
        """Compatibility hook using the same bounded mission-aware repair."""
        index_by_id = {task.task_id: idx for idx, task in enumerate(self.tasks)}
        indices = [index_by_id[task.task_id] for task in route]
        improved = self._improve_route(indices)
        return [self.tasks[idx] for idx in improved]


class QLearningPlanner(HeuristicGuidedQLearningPlanner):
    """Pipeline-compatible planner used by ``DMMP-R-RL-AC/main.py``."""

    def __init__(self, uav, optimize=True, seed=None):
        super().__init__(uav, uav.assigned_tasks, optimize=optimize, seed=seed)


QLearningTrajectoryPlanner = HeuristicGuidedQLearningPlanner


def run_tsa_for_fleet(uavs, epochs=EPOCHS, verbose=True, optimize=True):
    """Train and apply one planner per active UAV."""
    routes = {}
    for uav in uavs:
        if not getattr(uav, "active", True) or not uav.assigned_tasks:
            routes[uav.uav_id] = []
            continue
        planner = QLearningTrajectoryPlanner(
            uav, uav.assigned_tasks, optimize=optimize
        )
        planner.train(epochs=epochs, verbose=verbose)
        route = planner.get_best_route()
        uav.assigned_tasks = route
        routes[uav.uav_id] = route
    return routes
