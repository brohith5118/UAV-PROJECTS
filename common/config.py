# =========================================================
# DMMP-PR-TSA CONFIGURATION
# Based on: "Learning enhanced scheduling and resource
# allocation for heterogeneous UAV swarms in edge
# assisted remote sensing"
# =========================================================

# ---------------------------------------------------------
# MAP / GRID SETTINGS
# ---------------------------------------------------------
SEED = 99
UAV_SEED = 100
MAP_WIDTH  = 100
MAP_HEIGHT = 100

GRID_RESOLUTION = 100

# ---------------------------------------------------------
# TASK SETTINGS
# ---------------------------------------------------------
NUM_TASKS = 40

HIGH_PRIORITY_RATIO = 0.4
# Priority levels (paper Table 9)
# 1 = critical  (deadline 800 s)
# 2 = important (deadline 1000 s)
# 3 = routine   (deadline 1200 s)
PRIORITY_DEADLINES = {
    1: 800,
    2: 1000,
    3: 1200
}

# Task type flags  (paper eq 13 / Table 3)
# -1  acquisition-only (no compute)
#  0  acquisition + light compute
#  1  integrated acquisition+processing (compute-intensive)
TASK_TYPE_RATIO = {-1: 0.4, 0: 0.3, 1: 0.3}

# ---------------------------------------------------------
# UAV FLEET SETTINGS
# ---------------------------------------------------------
NUM_UAVS = 4

# Energy budget  (J)
MIN_ENERGY = 12000
MAX_ENERGY = 20000

# Hovering time budget  (s)
MIN_HOVER_TIME = 1200
MAX_HOVER_TIME = 3000

# Compute budget  (GHz·s)
MIN_COMPUTE = 0.0
MAX_COMPUTE = 120.0

# Flight speed  m/s  (paper Table 1)
UAV_SPEED = 20.0

# Energy consumed per unit distance  (J/m)
ENERGY_PER_METER = 0.3

# UAV type capacities matching paper Table 1
# type -1: long-endurance acquisition-only (no compute)
# type  0: balanced
# type  1: processing-capable (shorter endurance)
UAV_TYPE_MAX_FLIGHT = {-1: 2600, 0: 2200, 1: 1800}   # s
UAV_TYPE_MAX_COMPUTE = {-1: 0.0,  0: 60.0, 1: 120.0}   # GHz·s

# Maximum flight range derived from speed × max flight time
# Used in constraint (9): ||p_u - p_i|| <= D^max_u
# (computed at runtime from uav.max_hover_time * UAV_SPEED)

# ---------------------------------------------------------
# REGION PARTITIONING  (D-module, eq 3–8)
# ---------------------------------------------------------
ALPHA      = 1.0   # distance weight
GAMMA      = 15.0  # priority attraction
RHO        = 0.05  # Lagrange sub-gradient step  ρ
LAMBDA_TV  = 5.0   # TV regularisation  λ_TV
ITERATIONS = 20    # power-diagram iterations

# Periodic re-partitioning hysteresis (section after eq 8)
DELTA_T                = 10    # re-evaluate every Δt seconds
HYSTERESIS_THRESHOLD   = 5.0   # ε: skip if ΔJ < ε
MAX_CELL_REASSIGNMENTS = 10    # budget B

# ---------------------------------------------------------
# SOM PRE/RE-ASSIGNMENT  (PR-module, eq 15–26)
# ---------------------------------------------------------
SOM_ROWS        = 5          # competitive layer is 5×U
SOM_ITERATIONS  = 200        # R iterations (Algorithm 1)
C_PHI           = 3.0        # c_ϕ  task-type matching coeff
C_RES           = 6.0        # c_RES resource matching coeff
C_S             = 20.0       # c_s  node influence (SOM grid)
C_TIME          = 0.1        # c_time for Δtime penalty (eq 21)
C_COMP          = 0.1        # c_comp for Δcomp penalty (eq 23)
SOM_LEARN_RATE  = 0.5        # initial SOM learning rate

# ---------------------------------------------------------
# RL TASK SEQUENCE ADJUSTMENT  (TSA-module, eq 27–30)
# ---------------------------------------------------------
EPOCHS    = 500      # Q-learning episodes
RL_ALPHA  = 0.1      # α  learning rate
RL_GAMMA  = 0.9      # γ  discount factor
EPSILON   = 0.2      # ε-greedy exploration

# Reward coefficients (eq 29)
# Distances are in metres, so c_d is intentionally small to keep
# TSA rewards in a stable range for tabular Q-learning.
CD = -0.02  # c_d  distance penalty  (negative -> minimise)
CP =  75.0  # c_p  priority reward
CT = 120.0  # c_t  endurance-preservation reward
CC =  80.0  # c_c  compute-sufficiency reward

# ---------------------------------------------------------
# DYNAMIC EVENTS
# ---------------------------------------------------------
ENABLE_DYNAMIC_EVENTS   = False  # set to True to enable new task arrivals and UAV failures
NEW_TASK_ARRIVAL_RATE   = 0.1   # probability per time-step
UAV_FAILURE_PROBABILITY = 0.02


#PSO

BASE_X = 0.0
BASE_Y = 0.0

# ---------------------------------------------------------
# PSO BASELINE
# ---------------------------------------------------------
# Particle encoding uses two values per task:
#   - assignment key: preferred UAV index
#   - sequence key: task ordering priority during decoding
PSO_SWARM_SIZE = 60
PSO_ITERATIONS = 150
PSO_INERTIA = 0.72
PSO_COGNITIVE = 1.49
PSO_SOCIAL = 1.49
PSO_SEED = 1

# Fitness weights. Lower fitness is better.
PSO_UNASSIGNED_PENALTY = 100000.0
PSO_INFEASIBLE_PENALTY = 1000000.0
PSO_DEADLINE_PENALTY = 500.0
PSO_TARDINESS_WEIGHT = 60.0
PSO_HIGH_PRIORITY_DEADLINE_MULTIPLIER = 3.0
PSO_PRIORITY_REWARD = 200.0
PSO_DISTANCE_WEIGHT = 1.0
PSO_MAKESPAN_WEIGHT = 20.0
PSO_RETURN_TO_BASE = True


GENERATED_GRAPHS_DIR = "generated_graphs"