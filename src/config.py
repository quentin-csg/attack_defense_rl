"""Global constants, hyperparameters, and reward values.

All magic numbers live here. If a value is used in more than one file,
it MUST be defined here — no inline literals.
"""

MAX_NODES: int = 64
DEFAULT_NETWORK_SIZE: int = 8

N_NODE_FEATURES: int = 13  # 3 (os one-hot) + 10 scalar features
N_GLOBAL_FEATURES: int = 3

# N_ACTION_TYPES is derived dynamically from len(ActionType) in actions.py.
# Decoding: action_type = action // MAX_NODES, target_node = action % MAX_NODES

REWARD_EXFILTRATE: float = 150.0
# Dynamic exfiltrate reward for PCG: ratio × |PER_STEP × max_steps| — always > 1.
REWARD_EXFILTRATE_RATIO: float = 2.0
REWARD_PER_STEP: float = -0.5
REWARD_DETECTED: float = -50.0
REWARD_DETECTED_RATIO: float = 1 / 3
REWARD_NEW_NODE_DISCOVERED: float = 2.0
REWARD_NEW_NODE_COMPROMISED: float = 5.0
REWARD_ROOT_OBTAINED: float = 10.0
# scale = min(1.0, REWARD_EXPLORATION_NODES / n_nodes) keeps total discovery budget constant.
REWARD_EXPLORATION_NODES: int = 20
REWARD_REPEATED_ACTION: float = -2.0

SUSPICION_MAX: float = 100.0
SUSPICION_MIN: float = 0.0

SUSPICION_SCAN: float = 3.0
SUSPICION_ENUMERATE: float = 5.0
SUSPICION_ENUMERATE_AGGRESSIVE: float = 15.0
SUSPICION_EXPLOIT_MIN: float = 10.0
SUSPICION_BRUTE_FORCE: float = 30.0
SUSPICION_PRIVESC: float = 12.0
SUSPICION_CREDENTIAL_DUMP: float = 15.0
SUSPICION_PIVOT: float = 5.0
SUSPICION_LATERAL_MOVE: float = 8.0
SUSPICION_INSTALL_BACKDOOR: float = 10.0
SUSPICION_EXFILTRATE: float = 20.0
SUSPICION_LIST_FILES: float = 5.0
SUSPICION_TUNNEL: float = 5.0
SUSPICION_CLEAN_LOGS_BASE: float = -15.0
SUSPICION_WAIT_DECAY: float = -3.0

TUNNEL_SUSPICION_DIVISOR: float = 2.0
CLEAN_LOGS_SEQUENCE: list[float] = [-15.0, -10.0, -5.0, -2.0]
CLEAN_LOGS_COOLDOWN: int = 1
WAIT_FLOOR_DIVISOR: float = 2.0

EXPLOIT_CRASH_RATE: float = 0.05
BRUTE_FORCE_SUCCESS_RATE: float = 0.90

BLUE_ALERT_THRESHOLD: float = 60.0
BLUE_ROTATE_THRESHOLD: float = 80.0
BLUE_ISOLATE_THRESHOLD: float = 95.0

PATROL_INTERVAL: int = 5
PATROL_DETECTION_SUSPICION: float = 25.0
SURVEILLANCE_MULTIPLIER: float = 2.0

# Thresholds re-randomised at each reset so Red can't learn exact trigger timings.
BLUE_ALERT_NOISE: float = 10.0
BLUE_ROTATE_NOISE: float = 10.0
BLUE_ISOLATE_NOISE: float = 5.0

BLUE_ISOLATE_DURATION: int = 10
BLUE_ROTATE_COOLDOWN: int = 5

# Proportional hot_duration computed in wrappers: max(PATROL_HOT_DURATION, max_steps // 20)
PATROL_HOT_DURATION: int = 20
PATROL_HOT_WEIGHT: float = 5.0
PATROL_NEIGHBOR_WEIGHT: float = 2.5
PATROL_SUSPICION_DELTA_MIN: float = 1.0

PCG_SMALL_NODES: tuple[int, int] = (10, 15)
PCG_SMALL_SUBNETS: tuple[int, int] = (2, 3)
PCG_MEDIUM_NODES: tuple[int, int] = (25, 30)
PCG_MEDIUM_SUBNETS: tuple[int, int] = (4, 5)
PCG_LARGE_NODES: tuple[int, int] = (50, 60)
PCG_LARGE_SUBNETS: tuple[int, int] = (7, 8)

PCG_MAX_STEPS_SMALL: int = 150
PCG_MAX_STEPS_MEDIUM: int = 250
PCG_MAX_STEPS_LARGE: int = 400

PCG_BASE_STEPS: int = 40
PCG_STEPS_PER_HOP: int = 15
PCG_STEPS_PER_NODE: int = 2

CURRICULUM_WORLDS_PER_STAGE: int = 40
CURRICULUM_TIMESTEPS_SMALL: int = 200_000
CURRICULUM_TIMESTEPS_MEDIUM: int = 400_000
CURRICULUM_TIMESTEPS_LARGE: int = 1_000_000

DEFAULT_MAX_STEPS: int = 200
PADDING_VALUE: float = -1.0

RL_LEARNING_RATE: float = 3e-4
RL_N_STEPS: int = 2048
RL_BATCH_SIZE: int = 64
RL_N_EPOCHS: int = 10
RL_ENT_COEF: float = 0.05
RL_GAMMA: float = 0.99
RL_CLIP_RANGE: float = 0.2
RL_NET_ARCH: tuple[int, ...] = (256, 256)
RL_TOTAL_TIMESTEPS: int = 500_000
RL_SAVE_FREQ: int = 500_000
RL_LOG_INTERVAL: int = 1
RL_EVAL_FREQ: int = 25_000
RL_EVAL_EPISODES: int = 20
