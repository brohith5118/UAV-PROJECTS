import math
import numpy as np
import random
from collections import defaultdict

RL_ALPHA = 0.1
RL_GAMMA = 0.9
EPSILON = 0.2

CP = 75.0
CD = 0.02


class QLearningPlanner:

    def __init__(self, uav):
        self.uav = uav
        self.tasks = uav.assigned_tasks
        self.num_tasks = len(self.tasks)
        self.q_table = defaultdict(float)

    def distance(self, taskA, taskB):
        return ((taskA.x - taskB.x)**2+(taskA.y - taskB.y)**2)**(1/2)

    def reward(self, current_idx, action):
        current_task = self.tasks[current_idx]
        next_task = self.tasks[action]
        distance = self.distance(current_task, next_task)

        return CP * (4 - next_task.priority) - CD * distance

    def get_actions(self, visited_mask):
        actions = []
        for i in range(self.num_tasks):
            if not (visited_mask & (1 << i)):
                actions.append(i)
            
        return actions

    def choose_action(self, state):
        current_idx, visited_mask = state

        actions = self.get_actions(visited_mask)
        if not actions:
            return None
        
        if random.random() < EPSILON:
            return random.choice(actions)

        best_action = None
        best_q = float('-inf')

        for action in actions:
            q = self.q_table[(state, action)]
            if q > best_q:
                best_q = q
                best_action = action

        return best_action

    def train(self, episodes=500):
        if self.num_tasks == 1:
            return

        for _ in range(episodes):
            start_idx = random.randint(0, self.num_tasks-1)
            current_idx = start_idx
            visited_mask = 1 << current_idx

            while True:
                state = (current_idx, visited_mask)
                action = self.choose_action(state)
                if action is None:
                    break

                reward = self.reward(current_idx, action)

                next_mask = visited_mask | (1 << action)
                next_state = (action, next_mask)

                future_actions = self.get_actions(next_mask)
                if future_actions:
                    max_future_q = max(self.q_table[next_state, a] for a in future_actions)
                else:
                    max_future_q = 0

                old_q = self.q_table[(state,action)]
                self.q_table[(state,action)] = old_q + RL_ALPHA *(reward + RL_GAMMA*max_future_q - old_q)
                current_idx = action
                visited_mask = next_mask
    
    def extract_route(self):
        if self.num_tasks == 0:
            return []

        start_idx = max(range(self.num_tasks),key=lambda i: self.tasks[i].priority)
        current_idx = start_idx
        visited_mask = 1 << current_idx
        route = [self.tasks[current_idx]]

        while True:
            state = (current_idx, visited_mask)
            actions = self.get_actions(visited_mask)
            if not actions:
                break
            best_action = max(actions,key=lambda a: self.q_table[(state, a)])

            route.append(self.tasks[best_action])
            visited_mask |= 1 << best_action
            current_idx = best_action
            
        return route