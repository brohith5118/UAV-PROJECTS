# =========================================================
# TSA-MODULE  –  RL-based Task Sequence Adjustment
#
# Implements Algorithm 2 from the paper (Section
# "RL-based task sequence adjustment algorithm").
#
# MDP formulation (eq 27–30):
#
#   State space:
#     S = {s_0, s_1, ..., s_m, s_{m+1}}
#     s_j = (X_j, Y_j, pri_j)              (eq 27)
#     s_0 = UAV current position + resources
#     s_{m+1} = terminal (return to base)
#
#   Action space:
#     A = {a_j | a_j: s_j → s_j', j' ≠ j} (eq 28)
#     Restricted to feasibility-filtered assigned tasks.
#
#   Reward (eq 29):
#     R = c_d·d(j,j') + c_p·pri_{j'} +
#         c_t·T^re/T^max + c_c·(C^re−C_{j'})/C^max
#
#   Q-update (eq 30):
#     Q[s,a] ← (1−α)Q[s,a] + α(R + γ·max_a' Q[s',a'])
#
# Key design: compact per-UAV state-action space produced
# by the D and PR stages makes tabular Q-learning efficient
# and convergence fast (few hundred episodes).
# =========================================================
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import math
import random
import numpy as np

from utils import calculate_reward, estimate_finish_time, check_deadline
from common.config import (
    EPOCHS,
    RL_ALPHA,
    RL_GAMMA,
    EPSILON,
    CD, CP, CT, CC,
    UAV_SPEED,
    ENERGY_PER_METER,
)


class TempUAVProxy:
    def __init__(self, max_energy, max_hover_time, max_compute, rem_energy, rem_hover, rem_compute):
        self.max_energy = max_energy
        self.max_hover_time = max_hover_time
        self.max_compute = max_compute
        
        self.remaining_energy = rem_energy
        self.remaining_hover_time = rem_hover
        self.remaining_compute = rem_compute


# ----------------------------------------------------------
# Q-LEARNING TRAJECTORY PLANNER  (per UAV)
# ----------------------------------------------------------

class QLearningTrajectoryPlanner:
    """
    Trains a Q-table over the small task-subset assigned to
    one UAV, then extracts the best execution sequence.

    Attributes
    ----------
    uav      : UAV object (position, residual resources)
    tasks    : list[Task] – the UAV's assigned task subset
    q_table  : dict mapping state key (current_state, visited_mask) to np.ndarray shape (n_tasks,)
    """

    def __init__(self, uav, tasks, optimize=True):

        self.uav      = uav
        self.tasks    = list(tasks)        # local copy
        self.n        = len(self.tasks)
        self.optimize = optimize

        if self.n == 0:
            raise ValueError(
                f"UAV {uav.uav_id}: no assigned tasks for TSA."
            )

        # Index map: task_id → local index
        self.task_index = {
            t.task_id: i for i, t in enumerate(self.tasks)
        }

        # Q-table: dictionary lookup to handle large task sets efficiently
        # key: (current_state, visited_mask) -> np.ndarray of shape (n,)
        self.q_table = {}

        # Track best route found during training
        self._best_route   = None
        self._best_reward  = -float('inf')

        # Bootstrap Q-table using a greedy EDF heuristic route
        if self.optimize:
            self._bootstrap_q_table()

    # --------------------------------------------------
    # HEURISTIC ROUTE GENERATION & BOOTSTRAPPING
    # --------------------------------------------------

    def _find_heuristic_route(self):
        """
        Greedily constructs a route prioritizing deadlines, distances,
        and high priority to guide initial RL exploration.
        """
        current_pos = (self.uav.x, self.uav.y)
        remaining = set(range(self.n))
        route = []
        
        # Track simulated resources
        curr_hover = self.uav.remaining_hover_time
        curr_energy = self.uav.remaining_energy
        curr_compute = self.uav.remaining_compute

        while remaining:
            best_idx = None
            best_score = -float('inf')
            
            for idx in remaining:
                task = self.tasks[idx]
                dist = math.hypot(current_pos[0] - task.x, current_pos[1] - task.y)
                travel_t = dist / UAV_SPEED
                travel_e = dist * ENERGY_PER_METER
                
                # Feasibility check
                if (travel_t + task.hover_time <= curr_hover and
                    travel_e + task.energy_cost <= curr_energy and
                    task.compute_load <= curr_compute):
                    
                    # Balanced score: penalize distance and late deadlines, favor priority
                    score = -0.5 * dist - 0.2 * task.deadline + 10.0 * task.priority
                    if score > best_score:
                        best_score = score
                        best_idx = idx
            
            # If no task is feasible under remaining resources, fallback to nearest task
            if best_idx is None:
                best_idx = min(remaining, key=lambda idx: math.hypot(current_pos[0] - self.tasks[idx].x, current_pos[1] - self.tasks[idx].y))
                
            task = self.tasks[best_idx]
            dist = math.hypot(current_pos[0] - task.x, current_pos[1] - task.y)
            
            curr_hover -= (dist / UAV_SPEED + task.hover_time)
            curr_energy -= (dist * ENERGY_PER_METER + task.energy_cost)
            curr_compute -= task.compute_load
            
            route.append(best_idx)
            remaining.remove(best_idx)
            current_pos = (task.x, task.y)
            
        return route

    def _bootstrap_q_table(self):
        """
        Pre-populates Q-values along the heuristic route with a positive bias.
        """
        heuristic_route = self._find_heuristic_route()
        current_state = 0
        visited_mask = 0
        for action in heuristic_route:
            q_vals = self._get_q_values(current_state, visited_mask)
            q_vals[action] = 100.0  # Strong initialization boost
            visited_mask |= (1 << action)
            current_state = action + 1

    def _get_q_values(self, current_state, visited_mask):
        state_key = current_state if not self.optimize else (current_state, visited_mask)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n, dtype=np.float64)
        return self.q_table[state_key]

    # --------------------------------------------------
    # REWARD  R(s_j, a_j)  (eq 29)
    # --------------------------------------------------

    def _reward(self, from_task_or_pos, to_task_idx, uav):
        """
        Compute reward for transitioning to tasks[to_task_idx].
        from_task_or_pos is either a Task or (x, y) tuple.
        """
        next_task = self.tasks[to_task_idx]
        return calculate_reward(
            from_task_or_pos,
            next_task,
            uav,
            CD, CP, CT, CC,
        )

    # --------------------------------------------------
    # FEASIBILITY FILTER
    # --------------------------------------------------

    def _feasible_actions(self, visited, current_pos, curr_hover, curr_energy, curr_compute):
        """
        Returns indices of unvisited tasks that are still reachable and
        comply with hover, energy, and compute constraints.
        """
        if not self.optimize:
            return [i for i in range(self.n) if i not in visited]

        feasible = []
        for idx in range(self.n):
            if idx in visited:
                continue
            task = self.tasks[idx]
            dist = math.hypot(
                current_pos[0] - task.x,
                current_pos[1] - task.y,
            )
            travel_t = dist / UAV_SPEED
            travel_e = dist * ENERGY_PER_METER
            
            # Strict multi-resource check
            if (travel_t + task.hover_time <= curr_hover and
                travel_e + task.energy_cost <= curr_energy and
                task.compute_load <= curr_compute):
                feasible.append(idx)
                
        # If nothing feasible, open all unvisited as a fallback
        if not feasible:
            feasible = [i for i in range(self.n) if i not in visited]
        return feasible

    # --------------------------------------------------
    # EPSILON-GREEDY ACTION SELECTION
    # --------------------------------------------------

    def _select_action(self, current_state, visited_mask, feasible_actions, epsilon):
        if not feasible_actions:
            return None
        if random.random() < epsilon:
            return random.choice(feasible_actions)
        
        q_vals = self._get_q_values(current_state, visited_mask)
        best_a = feasible_actions[0]
        best_q = q_vals[best_a]
        for a in feasible_actions:
            if q_vals[a] > best_q:
                best_q = q_vals[a]
                best_a = a
        return best_a

    # --------------------------------------------------
    # SINGLE EPISODE
    # --------------------------------------------------

    def _run_episode(self, epsilon):
        """
        One full Q-learning episode with dynamic resource tracking and reward shaping.
        """
        current_pos   = (self.uav.x, self.uav.y)
        current_state = 0          # start node
        visited       = set()
        visited_mask  = 0
        episode_reward = 0.0
        route_indices  = []

        # Dynamic capacity tracking
        curr_hover = self.uav.remaining_hover_time
        curr_energy = self.uav.remaining_energy
        curr_compute = self.uav.remaining_compute
        curr_time = 0.0

        while len(visited) < self.n:
            feasible = self._feasible_actions(visited, current_pos, curr_hover, curr_energy, curr_compute)
            if not feasible:
                break

            action = self._select_action(current_state, visited_mask, feasible, epsilon)
            if action is None:
                break

            next_task = self.tasks[action]
            
            # Physics calculations
            dist = math.hypot(current_pos[0] - next_task.x, current_pos[1] - next_task.y)
            travel_t = dist / UAV_SPEED
            travel_e = dist * ENERGY_PER_METER
            
            arrival_time = curr_time + travel_t
            finish_time = arrival_time + next_task.hover_time
            
            is_on_time = finish_time <= next_task.deadline
            
            # Resource updates
            if self.optimize:
                next_hover = curr_hover - (travel_t + next_task.hover_time)
                next_energy = curr_energy - (travel_e + next_task.energy_cost)
                next_compute = curr_compute - next_task.compute_load
            else:
                next_hover = curr_hover
                next_energy = curr_energy
                next_compute = curr_compute

            # Construct proxy UAV with current resource levels
            if self.optimize:
                uav_proxy = TempUAVProxy(
                    self.uav.max_energy,
                    self.uav.max_hover_time,
                    self.uav.max_compute,
                    curr_energy,
                    curr_hover,
                    curr_compute
                )
            else:
                uav_proxy = self.uav
            
            # Base reward
            r = self._reward(
                current_pos if current_state == 0 else self.tasks[current_state - 1],
                action,
                uav_proxy
            )
            
            # Reward shaping
            if self.optimize:
                if not is_on_time:
                    # Penalize late completions
                    r -= 0.5 * (finish_time - next_task.deadline)
                else:
                    # Reward on-time completion based on priority
                    priority_bonus = {1: 3.0, 2: 2.0, 3: 1.0}.get(next_task.priority, 1.0)
                    r += 50.0 * priority_bonus

                # Penalize resource overruns
                if next_hover < 0:
                    r -= 150.0
                if next_energy < 0:
                    r -= 150.0
                if next_compute < 0:
                    r -= 150.0
                r = max(-1000.0, min(1000.0, r))

            # Max future Q lookup
            next_visited  = visited | {action}
            next_visited_mask = visited_mask | (1 << action)
            next_feasible = [i for i in range(self.n) if i not in next_visited]
            
            if next_feasible:
                next_q_vals = self._get_q_values(action + 1, next_visited_mask)
                max_future_q = max(next_q_vals[j] for j in next_feasible)
            else:
                max_future_q = 0.0

            # Q-table update
            q_vals = self._get_q_values(current_state, visited_mask)
            old_q = q_vals[action]
            q_vals[action] = (
                (1 - RL_ALPHA) * old_q
                + RL_ALPHA * (r + RL_GAMMA * max_future_q)
            )

            episode_reward += r
            visited.add(action)
            visited_mask = next_visited_mask
            route_indices.append(action)
            
            current_pos   = (next_task.x, next_task.y)
            current_state = action + 1
            curr_hover    = next_hover
            curr_energy   = next_energy
            curr_compute  = next_compute
            curr_time     = finish_time

        return max(-1000.0, min(1000.0, episode_reward)), route_indices

    # --------------------------------------------------
    # TRAINING  (Algorithm 2)
    # --------------------------------------------------

    def train(self, epochs=EPOCHS, verbose=False):
        """
        Run Q-learning for *epochs* episodes with decaying
        epsilon (exploration → exploitation).
        """
        epsilon     = EPSILON
        eps_decay   = epsilon / max(epochs * 0.8, 1)

        reward_log  = []

        for ep in range(epochs):

            ep_reward, route = self._run_episode(epsilon)
            reward_log.append(ep_reward)

            # Track best complete route seen
            if (len(route) == self.n
                    and ep_reward > self._best_reward):
                self._best_reward = ep_reward
                self._best_route  = route[:]

            # Decay exploration
            epsilon = max(0.01, epsilon - eps_decay)

        if verbose:
            mean_r = sum(reward_log[-50:]) / min(50, len(reward_log))
            print(
                f"  UAV {self.uav.uav_id} TSA: "
                f"{epochs} eps, "
                f"last-50 mean reward={mean_r:.2f}, "
                f"best route reward={self._best_reward:.2f}"
            )

        return reward_log

    # --------------------------------------------------
    # EXTRACT BEST ROUTE
    # --------------------------------------------------

    def get_best_route(self):
        """
        Return the task execution sequence as an ordered
        list of Task objects.
        """
        if self._best_route and len(self._best_route) == self.n:
            return [self.tasks[i] for i in self._best_route]

        # --- Greedy fallback using Q-table ---
        current_state = 0
        visited_mask = 0
        route = []
        visited = set()

        while len(visited) < self.n:
            remaining = [j for j in range(self.n) if j not in visited]
            if not remaining:
                break
            q_vals = self._get_q_values(current_state, visited_mask)
            next_idx = max(remaining, key=lambda j: q_vals[j])
            route.append(next_idx)
            visited.add(next_idx)
            visited_mask |= (1 << next_idx)
            current_state = next_idx + 1

        return [self.tasks[i] for i in route]

    # --------------------------------------------------
    # DEADLINE-AWARE SEQUENCE REORDER
    # --------------------------------------------------

    def reorder_by_deadline(self, route):
        """
        Given an initial route (list of Tasks), move any
        task whose deadline would be missed to the earliest
        feasible position.

        Returns the adjusted route.

        Guard rails:
          - Only marks adjusted=True when the route actually changes.
          - Caps iterations at n*(n-1)//2 to prevent infinite loops
            when no reordering can fix a structurally infeasible
            deadline (e.g. task moved far away after a location update).
        """
        route       = list(route)
        n           = len(route)
        max_iters   = max(1, n * (n - 1) // 2)
        iteration   = 0

        adjusted = True
        while adjusted and iteration < max_iters:
            adjusted = False
            iteration += 1
            timeline  = estimate_finish_time(
                self.uav, route, UAV_SPEED
            )

            for rank, (task, ft) in enumerate(timeline):
                if check_deadline(task, ft):
                    continue
                # Task is at risk – try moving it earlier
                for pos in range(rank):
                    candidate = route[:pos] + [task] + \
                                route[pos:rank] + route[rank + 1:]
                    tl2 = estimate_finish_time(
                        self.uav, candidate, UAV_SPEED
                    )
                    _, ft2 = tl2[pos]
                    if check_deadline(task, ft2):
                        # A beneficial move was found – apply it
                        route = candidate
                        adjusted = True
                        break   # restart scan from new timeline
                if adjusted:
                    break

        return route


# ----------------------------------------------------------
# FLEET-LEVEL TSA  (run planner for every UAV)
# ----------------------------------------------------------

def run_tsa_for_fleet(uavs, epochs=EPOCHS, verbose=True, optimize=True):
    """
    Trains a Q-learning planner for each UAV in the fleet
    and returns a dict  {uav_id: ordered_task_list}.

    Applies deadline-aware reordering after Q-learning.
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
            print(f"  Training TSA for UAV {uav.uav_id:02d} "
                  f"({len(uav.assigned_tasks)} tasks)...")

        planner = QLearningTrajectoryPlanner(uav, uav.assigned_tasks, optimize=optimize)
        planner.train(epochs=epochs, verbose=verbose)

        route = planner.get_best_route()
        if optimize:
            route = planner.reorder_by_deadline(route)

        uav.assigned_tasks = route
        all_routes[uav.uav_id] = route

    return all_routes