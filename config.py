"""
Configuration parameters for swarm coordination and RL algorithms.

This file contains all hardcoded parameters, starting positions, 
and configuration settings separated from the algorithm logic.
"""
import numpy as np

# =============================================================================
# CONSENSUS DYNAMICS & GRAPH LAPLACIAN PARAMETERS
# =============================================================================

class ConsensusConfig:
    """Parameters for matrix-based coordination from handwritten notes."""
    
    # Communication and sensing
    SENSING_RANGE = 4.0          # r: radius for adjacency matrix aij
    SEPARATION_RANGE = 2.0       # rs: safe distance for separation forces
    MIN_CONNECTIVITY = 0.1       # minimum λ2 required for convergence
    
    # Force strengths (tunable k parameters from notes)
    SEPARATION_GAIN = 1.5        # ks: push strength for separation
    CONSENSUS_GAIN = 0.8         # strength of consensus attraction
    FORMATION_GAIN = 0.5         # strength of shape maintenance
    
    # Formation control
    ROTATION_SPEED = 0.1         # ω: angular velocity for moving formations
    STAY_THRESHOLD = 0.15        # minimum force to trigger movement
    
    # Predefined formations (δi values)
    FORMATIONS = {
        'line': [
            np.array([-2.0, 0.0]),
            np.array([0.0, 0.0]), 
            np.array([2.0, 0.0])
        ],
        'triangle': [
            np.array([0.0, 2.0]),
            np.array([-1.5, -1.0]),
            np.array([1.5, -1.0])
        ],
        'square': [
            np.array([-1.5, -1.5]),
            np.array([1.5, -1.5]),
            np.array([1.5, 1.5]),
            np.array([-1.5, 1.5])
        ],
        'circle': [  # 6 drones in circle
            np.array([2.0 * np.cos(i * np.pi/3), 2.0 * np.sin(i * np.pi/3)]) 
            for i in range(6)
        ]
    }

# =============================================================================
# RL TRAINING CONFIGURATIONS  
# =============================================================================

class PPOConfig:
    """PPO hyperparameters for both 2D and 3D environments."""
    
    # Training schedule
    NUM_UPDATES = 200            # total training updates
    ROLLOUT_STEPS = 1500         # steps per rollout
    EPOCHS = 4                   # PPO epochs per update
    MINIBATCH_SIZE = 256         # minibatch size
    
    # PPO hyperparameters
    GAMMA = 0.99                 # discount factor
    LAMBDA = 0.95                # GAE lambda
    CLIP_EPSILON = 0.2           # PPO clipping parameter
    ENTROPY_COEF = 0.02          # entropy regularization
    VALUE_COEF = 0.5             # value function loss coefficient
    LEARNING_RATE = 3e-4         # Adam learning rate
    
    # Network architecture
    HIDDEN_SIZE = 128            # hidden layer size
    
    # Environment settings
    SEED = 42                    # random seed
    
    # Evaluation
    EVAL_EPISODES = 20           # episodes for evaluation
    EVAL_FREQUENCY = 20          # evaluate every N updates

class Environment2DConfig:
    """Configuration for 2D SwarmCoverageEnv."""
    
    # Grid and agents
    GRID_SIZE = 12               # NxN grid size
    N_DRONES = 4                 # number of drones  
    N_OBSTACLES = 25             # random obstacles (if no bbox)
    OBS_WINDOW = 3               # observation radius
    MAX_STEPS = 80               # episode length
    
    # Battery and energy
    BATTERY_DRAIN_MOVE = 1.0     # battery cost per move
    BATTERY_DRAIN_IDLE = 0.2     # battery cost per idle step
    
    # Rewards
    STEP_PENALTY = 0.02          # penalty per step
    
    # Real-world locations (bbox format: west, south, east, north)
    KNOWN_LOCATIONS = {
        "san_jose": {
            "bbox": (-121.94, 37.32, -121.87, 37.37),
            "grid_size": 25,
        },
        "berkeley": {
            "bbox": (-122.259, 37.870, -122.253, 37.875),
            "grid_size": 20,
        },
        "golden_gate_park": {
            "bbox": (-122.511, 37.765, -122.454, 37.775),
            "grid_size": 25,
        },
    }

class Environment3DConfig:
    """Configuration for 3D SwarmSearch3DEnv."""
    
    # Grid and agents  
    GRID_SIZE = 14               # NxN grid size
    N_DRONES = 3                 # number of drones
    MAX_STEPS = 90               # episode length
    
    # 3D parameters
    H_MAX = 3                    # maximum terrain height
    A_MAX = 4                    # maximum altitude
    OBS_WINDOW = 2               # observation radius
    
    # World generation
    N_BUILDINGS = 12             # number of buildings
    N_EPICENTERS = 2             # damage epicenters
    
    # Detection parameters
    DETECT_CONF_THRESH = 0.75    # confidence threshold for damage detection
    ALARM_AREA = 6               # area threshold for alarms
    
    # Reward weights (from PDF reward section)
    W_COV = 0.02                 # coverage reward weight
    W_DMG = 1.5                  # damage detection weight
    W_AREA = 0.6                 # area measurement weight
    W_ALT = 0.12                 # altitude penalty weight (encourages low flight)
    W_STEP = 0.01                # step penalty
    W_BLOCK = 0.15               # blocked move penalty
    W_ALARM = 5.0                # alarm reward weight

# =============================================================================
# CONSENSUS-ENHANCED RL CONFIGURATION
# =============================================================================

class ConsensusRLConfig:
    """Configuration for RL algorithms enhanced with consensus dynamics."""
    
    # Hybrid approach weights
    RL_WEIGHT = 0.7              # weight of RL policy decision
    CONSENSUS_WEIGHT = 0.3       # weight of consensus dynamics
    
    # Consensus integration modes
    INTEGRATION_MODE = 'additive' # 'additive', 'multiplicative', or 'switching'
    
    # Switching thresholds (for switching mode)
    CONNECTIVITY_THRESHOLD = 0.2  # switch to consensus if λ2 < threshold
    SPREAD_THRESHOLD = 5.0        # switch to consensus if swarm too spread out
    
    # Formation learning
    LEARN_FORMATIONS = True       # whether to learn optimal formations
    FORMATION_REWARD_WEIGHT = 0.5 # reward weight for formation maintenance
    
    # Consensus observation augmentation
    AUGMENT_OBSERVATIONS = True   # add consensus info to RL observations
    CONSENSUS_OBS_SIZE = 8        # size of consensus-related observation features

# =============================================================================
# TRAINING DATA AND MODEL PATHS
# =============================================================================

class PathConfig:
    """File paths for models, configurations, and training data."""
    
    # Model directories
    MODELS_DIR = "models"
    WEIGHTS_DIR = "weights"
    CONFIG_DIR = "configs"
    LOGS_DIR = "logs"
    
    # Model filenames
    PPO_2D_MODEL = "ppo_swarm_2d.pt"
    PPO_3D_MODEL = "ppo_swarm_3d.pt" 
    CONSENSUS_PPO_2D_MODEL = "consensus_ppo_2d.pt"
    CONSENSUS_PPO_3D_MODEL = "consensus_ppo_3d.pt"
    
    # Weight filenames (separated from full model configs)
    PPO_2D_WEIGHTS = "ppo_2d_weights.pt"
    PPO_3D_WEIGHTS = "ppo_3d_weights.pt"
    CONSENSUS_PPO_2D_WEIGHTS = "consensus_ppo_2d_weights.pt"
    CONSENSUS_PPO_3D_WEIGHTS = "consensus_ppo_3d_weights.pt"
    
    # Configuration saves
    TRAINING_CONFIG = "last_training_config.json"
    EVALUATION_RESULTS = "evaluation_results.json"

# =============================================================================
# EXPERIMENTAL CONFIGURATIONS
# =============================================================================

class ExperimentConfig:
    """Configuration for different experimental setups."""
    
    # Ablation studies
    ABLATIONS = {
        'no_consensus': {
            'use_consensus': False,
            'use_formation': False,
            'use_separation': False
        },
        'consensus_only': {
            'use_consensus': True, 
            'use_formation': False,
            'use_separation': False
        },
        'full_coordination': {
            'use_consensus': True,
            'use_formation': True, 
            'use_separation': True
        }
    }
    
    # Hyperparameter sweeps
    SENSING_RANGE_SWEEP = [2.0, 3.0, 4.0, 5.0, 6.0]
    CONSENSUS_WEIGHT_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Multi-agent scales
    SWARM_SIZES = [2, 3, 4, 6, 8, 10]
    
    # Environment variations
    GRID_SIZES = [10, 12, 15, 20]
    OBSTACLE_DENSITIES = [0.1, 0.2, 0.3, 0.4]

# =============================================================================
# STARTING POSITIONS AND INITIAL CONDITIONS
# =============================================================================

class InitialConditions:
    """Starting positions and initial conditions for reproducible experiments."""
    
    # Starting positions for different swarm sizes
    START_POSITIONS = {
        2: [(2, 2), (7, 7)],
        3: [(2, 2), (7, 2), (7, 7)],
        4: [(2, 2), (2, 7), (7, 2), (7, 7)],
        6: [(1, 1), (1, 8), (4, 1), (4, 8), (8, 1), (8, 8)]
    }
    
    # Formation starting configurations
    FORMATION_STARTS = {
        'dispersed': 'random',           # random starting positions
        'clustered': 'center',           # all start near center
        'line': 'horizontal_line',       # start in horizontal line
        'corners': 'corner_positions'    # start at grid corners
    }
    
    # Seeds for reproducible experiments
    EXPERIMENT_SEEDS = [42, 123, 456, 789, 999]
    
# =============================================================================
# HELPER FUNCTIONS FOR CONFIGURATION
# =============================================================================

def get_config(config_name):
    """Get configuration by name."""
    configs = {
        'consensus': ConsensusConfig(),
        'ppo': PPOConfig(),
        'env_2d': Environment2DConfig(),
        'env_3d': Environment3DConfig(),
        'consensus_rl': ConsensusRLConfig(),
        'paths': PathConfig(),
        'experiment': ExperimentConfig(),
        'initial': InitialConditions()
    }
    return configs.get(config_name)

def create_experiment_config(swarm_size, grid_size, sensing_range, consensus_weight):
    """Create a custom configuration for experiments."""
    config = {
        'swarm_size': swarm_size,
        'grid_size': grid_size, 
        'sensing_range': sensing_range,
        'consensus_weight': consensus_weight,
        'start_positions': InitialConditions.START_POSITIONS.get(swarm_size, 'random')
    }
    return config

def validate_config(config):
    """Validate configuration parameters."""
    if hasattr(config, 'SENSING_RANGE') and hasattr(config, 'SEPARATION_RANGE'):
        assert config.SENSING_RANGE > config.SEPARATION_RANGE, \
            "Sensing range must be larger than separation range"
    
    if hasattr(config, 'RL_WEIGHT') and hasattr(config, 'CONSENSUS_WEIGHT'):
        total_weight = config.RL_WEIGHT + config.CONSENSUS_WEIGHT
        assert abs(total_weight - 1.0) < 0.01, \
            f"RL and consensus weights should sum to 1.0, got {total_weight}"
    
    return True