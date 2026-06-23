# =========================================================
# UAV  –  platform model (eq 1, Table 2)
#
# Capability vector C_u(t) = {Cu,E(t), Cu,H(t), Cu,F(t)}
# =========================================================
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import math

from common.config import UAV_SPEED, ENERGY_PER_METER
class UAV:

    def __init__(
        self,
        uav_id,
        x,
        y,
        uav_type,       # -1 | 0 | 1  (Table 1)
        max_energy,     # Cu,E  (J)
        max_hover_time, # Cu,H  (s)
        max_compute,    # Cu,F  (GHz·s)
    ):

        self.uav_id   = uav_id
        self.x        = x
        self.y        = y
        self.uav_type = uav_type   # ψ_{UAV_u}

        self.centroid_x = x
        self.centroid_y = y

        self.curr_x = x
        self.curr_y = y

        # -----------------------------------------------
        # Maximum capacity
        # -----------------------------------------------
        self.max_energy     = max_energy
        self.max_hover_time = max_hover_time
        self.max_compute    = max_compute

        # -----------------------------------------------
        # Residual capacity (updated during execution)
        # -----------------------------------------------
        self.remaining_energy     = max_energy
        self.remaining_hover_time = max_hover_time
        self.remaining_compute    = max_compute
        self.curr_energy     = max_energy
        self.curr_hover = max_hover_time
        self.curr_compute    = max_compute

        # -----------------------------------------------
        # Lagrange multipliers μ_{u,k}  (eq 8)
        # -----------------------------------------------
        self.mu_energy  = 0.0
        self.mu_hover   = 0.0
        self.mu_compute = 0.0

        # -----------------------------------------------
        # Assigned tasks / status
        # -----------------------------------------------
        self.assigned_tasks = []
        self.active         = True   # False after failure

    # --------------------------------------------------
    # Maximum flight range from current residual energy
    # Constraint (9): ||p_u - p_i|| <= D^max_u
    # --------------------------------------------------

    @property
    def max_flight_range(self):
        """Maximum reachable distance (metres) given residual
        hover-time and UAV speed."""
        return self.remaining_hover_time * UAV_SPEED

    # --------------------------------------------------
    # Type-compatibility check (eq 13)
    # ϕ_{u,i} = 1 only when |ψ_u − ϕ_task| <= 1
    # --------------------------------------------------

    def is_compatible(self, task):

        if task.task_type == -1:
            return True

        if task.task_type == 0:
            return self.uav_type in [0,1]

        if task.task_type == 1:
            return self.uav_type == 1

        return False

    # --------------------------------------------------
    # Euclidean distance to a task
    # --------------------------------------------------

    def distance_to(self, task):
        return math.hypot(
            self.x - task.x,
            self.y - task.y
        )

    # --------------------------------------------------
    # Residual-time feasibility: can UAV reach task and
    # return to base without running out of flight time?
    # (eq 22)
    # --------------------------------------------------

    def time_feasible(self, task, base_x=0.0, base_y=0.0):
        dist_to_task = self.distance_to(task)
        dist_to_base = math.hypot(
            task.x - base_x,
            task.y - base_y
        )
        travel_time = (dist_to_task + dist_to_base) / UAV_SPEED
        diff_time = (
            self.remaining_hover_time
            - travel_time
            - task.hover_time
        )
        return diff_time >= 0

    # --------------------------------------------------
    # Compute feasibility
    # --------------------------------------------------

    def compute_feasible(self, task):
        diff_comp = self.remaining_compute - task.compute_load
        return diff_comp >= 0

    # --------------------------------------------------
    # House-keeping
    # --------------------------------------------------

    def clear_tasks(self):
        self.assigned_tasks = []

    def reset_resources(self):
        self.remaining_energy     = self.max_energy
        self.remaining_hover_time = self.max_hover_time
        self.remaining_compute    = self.max_compute

    def reset_position(self):
        self.curr_x = self.x
        self.curr_y = self.y

    def consume_resources(self, task):
        """Deduct task workload from residual capacities."""
        travel_energy = self.distance_to(task) * ENERGY_PER_METER
        travel_time   = self.distance_to(task) / UAV_SPEED
        self.remaining_energy     -= task.energy_cost + travel_energy
        self.remaining_hover_time -= travel_time + task.hover_time
        self.remaining_compute    -= task.compute_load

    def compute_resource(self, task):
        """Deduct task workload from residual capacities."""
        self.remaining_energy     -= task.energy_cost
        self.remaining_hover_time -= task.hover_time
        self.remaining_compute    -= task.compute_load

    def move_to(self, task):
        """Update UAV position to task location."""
        self.curr_x = task.x
        self.curr_y = task.y

    def assign(self, task):
        self.assigned_tasks.append(task)
        task.assigned_uav = self.uav_id

    def __repr__(self):
        return (
            f"UAV {self.uav_id:02d} | "
            f"type={self.uav_type:+d} | "
            f"pos=({self.x:.1f},{self.y:.1f}) | "
            f"E={self.remaining_energy:.0f}/{self.max_energy:.0f}J "
            f"H={self.remaining_hover_time:.0f}/{self.max_hover_time:.0f}s "
            f"F={self.remaining_compute:.1f}/{self.max_compute:.1f}GHz·s"
        )