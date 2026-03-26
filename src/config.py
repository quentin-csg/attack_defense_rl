"""Global constants, hyperparameters, and reward values.

All magic numbers live here. If a value is used in more than one file,
it MUST be defined here — no inline literals.
"""

# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------
MAX_NODES: int = 64  # increased from 50 for Phase 5 PCG Large networks (50-60 nodes)
DEFAULT_NETWORK_SIZE: int = 8  # fixed topology for Phase 1

# ---------------------------------------------------------------------------
# Observation features per node (13 features)
# ---------------------------------------------------------------------------
#   os_type one-hot (3) + n_services + n_vulns_known + session_level
#   + suspicion_level + is_online + has_backdoor + has_tunnel + has_loot
#   + is_under_surveillance + discovery_level
N_NODE_FEATURES: int = 13  # 3 (os one-hot) + 10 scalar features

# Global features: current_step_norm, n_compromised_norm, n_discovered_norm
N_GLOBAL_FEATURES: int = 3

# ---------------------------------------------------------------------------
# Action space  (CORRECTION 1 — flat Discrete)
# ---------------------------------------------------------------------------
# N_ACTION_TYPES is derived dynamically from len(ActionType) in actions.py
# to stay in sync when new actions are added. DO NOT define it here.
# Total discrete actions = N_ACTION_TYPES * MAX_NODES
# Decoding: action_type = action // MAX_NODES, target_node = action % MAX_NODES

# ---------------------------------------------------------------------------
# Rewards  (CORRECTION 4 — calibrated values)
# ---------------------------------------------------------------------------
REWARD_EXFILTRATE: float = 150.0  # ratio = 150 / (0.5 * 200) = 1.5 > 1
REWARD_PER_STEP: float = -0.5
REWARD_DETECTED: float = -50.0
REWARD_NEW_NODE_DISCOVERED: float = 2.0
REWARD_NEW_NODE_COMPROMISED: float = 5.0
REWARD_ROOT_OBTAINED: float = 10.0
REWARD_REPEATED_ACTION: float = -1.0

# ---------------------------------------------------------------------------
# Suspicion
# ---------------------------------------------------------------------------
SUSPICION_MAX: float = 100.0
SUSPICION_MIN: float = 0.0

# Per-action suspicion costs
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
SUSPICION_TUNNEL: float = 5.0
SUSPICION_CLEAN_LOGS_BASE: float = -15.0  # diminishing: -15, -10, -5, -2
SUSPICION_WAIT_DECAY: float = -3.0

# Tunnel suspicion divisor
TUNNEL_SUSPICION_DIVISOR: float = 2.0

# Clean logs diminishing returns sequence
CLEAN_LOGS_SEQUENCE: list[float] = [-15.0, -10.0, -5.0, -2.0]
CLEAN_LOGS_COOLDOWN: int = 1  # steps between consecutive CLEAN_LOGS

# WAIT floor: suspicion never drops below max_historical / 2
WAIT_FLOOR_DIVISOR: float = 2.0

# ---------------------------------------------------------------------------
# Exploit probabilities
# ---------------------------------------------------------------------------
EXPLOIT_CRASH_RATE: float = 0.05
# Success/fail rates are defined per-vulnerability in VulnType.success_rate

BRUTE_FORCE_SUCCESS_RATE: float = 0.90  # if weak_credentials present

# ---------------------------------------------------------------------------
# Blue Team defaults (Phase 4, but constants defined here)
# ---------------------------------------------------------------------------
BLUE_ALERT_THRESHOLD: float = 60.0
BLUE_ROTATE_THRESHOLD: float = 80.0
BLUE_ISOLATE_THRESHOLD: float = 95.0

PATROL_INTERVAL: int = 5  # mean steps between patrols (Poisson, CORRECTION 3)
PATROL_DETECTION_SUSPICION: float = 25.0
SURVEILLANCE_MULTIPLIER: float = 2.0

# Noisy thresholds — randomised at each reset() so Red can't time-exploit them
BLUE_ALERT_NOISE: float = 10.0    # ALERT threshold = 60 ± 10 (drawn in [50, 70])
BLUE_ROTATE_NOISE: float = 10.0   # ROTATE threshold = 80 ± 10 (drawn in [70, 90])
BLUE_ISOLATE_NOISE: float = 5.0   # ISOLATE threshold = 95 ± 5 (drawn in [90, 100])

# ISOLATE duration: isolated nodes are auto-restored after this many steps
BLUE_ISOLATE_DURATION: int = 10

# ROTATE cooldown: min steps between two ROTATE_CREDENTIALS on the same node
BLUE_ROTATE_COOLDOWN: int = 5

# ---------------------------------------------------------------------------
# PCG — Génération procédurale de réseaux (Phase 5)
# ---------------------------------------------------------------------------
# Tailles de réseau : (min_nodes, max_nodes)
PCG_SMALL_NODES: tuple[int, int] = (10, 15)
PCG_SMALL_SUBNETS: tuple[int, int] = (2, 3)

PCG_MEDIUM_NODES: tuple[int, int] = (25, 30)
PCG_MEDIUM_SUBNETS: tuple[int, int] = (4, 5)

PCG_LARGE_NODES: tuple[int, int] = (50, 60)
PCG_LARGE_SUBNETS: tuple[int, int] = (7, 8)

# max_steps par taille (par défaut — compute_max_steps() peut affiner)
PCG_MAX_STEPS_SMALL: int = 150
PCG_MAX_STEPS_MEDIUM: int = 250
PCG_MAX_STEPS_LARGE: int = 350

# Score de difficulté — coefficients
PCG_BASE_STEPS: int = 40
PCG_STEPS_PER_HOP: int = 15
PCG_STEPS_PER_NODE: int = 2

# Curriculum learning
CURRICULUM_WORLDS_PER_STAGE: int = 5
CURRICULUM_TIMESTEPS_SMALL: int = 100_000
CURRICULUM_TIMESTEPS_MEDIUM: int = 150_000
CURRICULUM_TIMESTEPS_LARGE: int = 200_000

# ---------------------------------------------------------------------------
# Episode limits
# ---------------------------------------------------------------------------
DEFAULT_MAX_STEPS: int = 200

# ---------------------------------------------------------------------------
# Padding sentinel (CORRECTION 2)
# ---------------------------------------------------------------------------
PADDING_VALUE: float = -1.0

# ---------------------------------------------------------------------------
# RL Training — Phase 3 (MaskablePPO)
# ---------------------------------------------------------------------------
RL_LEARNING_RATE: float = 3e-4
RL_N_STEPS: int = 2048          # rollout buffer size — must satisfy N_STEPS % BATCH_SIZE == 0
RL_BATCH_SIZE: int = 64         # minibatch size (2048 / 64 = 32 minibatches)
RL_N_EPOCHS: int = 10           # passes over the rollout buffer per update
RL_ENT_COEF: float = 0.05       # entropy bonus — encourages exploration (higher to prevent Simplex crash)
RL_GAMMA: float = 0.99          # discount factor
RL_CLIP_RANGE: float = 0.2      # PPO clipping parameter
RL_NET_ARCH: tuple[int, ...] = (256, 256)  # hidden layers for policy + value networks
RL_TOTAL_TIMESTEPS: int = 500_000    # default training budget
RL_SAVE_FREQ: int = 10_000          # checkpoint every N env steps
RL_LOG_INTERVAL: int = 1            # log every N rollouts
RL_EVAL_FREQ: int = 5_000           # evaluate every N env steps
RL_EVAL_EPISODES: int = 20          # episodes per evaluation run
