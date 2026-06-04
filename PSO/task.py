# =========================================================
# TASK  –  grid cell g_i  (eq 2, Table 3)
#
# Workload vector ω(g_i) = {ωE,i , ωH,i , ωF,i}
# =========================================================


class Task:

    def __init__(
        self,
        task_id,
        x,
        y,
        priority,         # 1=routine 2=important 3=critical
        task_type,        # -1 acq-only | 0 mixed | 1 compute-intensive
        energy_cost,      # ωE,i  (J)
        hover_time,       # ωH,i  (s)
        compute_load,     # ωF,i  (GHz·s)
        deadline=1200,    # hard deadline Di (s)
    ):

        self.task_id     = task_id
        self.x           = x
        self.y           = y

        self.priority    = priority    # p^pri_i
        self.task_type   = task_type   # ϕ_{TASK_i}
        self.deadline    = deadline

        # ω(g_i) workload vector
        self.energy_cost  = energy_cost
        self.hover_time   = hover_time
        self.compute_load = compute_load

        # Runtime tracking
        self.assigned_uav  = None
        self.start_time    = None
        self.finish_time   = None
        self.completed     = False

    # --------------------------------------------------
    # Derived helpers
    # --------------------------------------------------

    @property
    def requires_compute(self):
        """True for task types 0 and 1 (need onboard CPU)."""
        return self.task_type >= 0

    @property
    def is_high_priority(self):
        return self.priority == 3

    def __repr__(self):
        return (
            f"Task {self.task_id:02d} | "
            f"({self.x:5.1f},{self.y:5.1f}) | "
            f"pri={self.priority} type={self.task_type:+d} | "
            f"E={self.energy_cost:5.1f}J "
            f"H={self.hover_time:4.1f}s "
            f"F={self.compute_load:5.2f}GHz·s | "
            f"dl={self.deadline}s"
        )
