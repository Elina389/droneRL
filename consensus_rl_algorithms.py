"""
Reinforcement Learning algorithms enhanced with consensus dynamics and graph Laplacian coordination.

This module implements the core RL algorithms (PPO) augmented with matrix-based
swarm coordination from the handwritten notes. The key innovation is combining:

1. Learned policies (neural networks) for complex decision making
2. Mathematical guarantees (consensus dynamics) for coordination
3. Graph theory (Laplacian matrices) for emergent swarm behavior

Architecture changes from baseline RL:
- Observations augmented with consensus/connectivity information  
- Action selection combines RL policy outputs with consensus forces
- Reward functions include formation maintenance and connectivity metrics
- Training incorporates multi-agent coordination objectives
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from collections import deque

from consensus_dynamics import ConsensusCoordinator, snap_consensus_to_discrete_actions
from config import PPOConfig, ConsensusRLConfig, ConsensusConfig
from weights_manager import ConsensusWeightsManager


class ConsensusAugmentedActorCritic(nn.Module):
    """
    Actor-Critic network augmented with consensus dynamics information.
    
    Key changes from baseline ActorCritic:
    1. Larger observation space (includes consensus features)
    2. Additional value head for consensus-specific rewards
    3. Separate policy heads for individual vs. coordinated actions
    """
    
    def __init__(self, base_obs_dim, n_actions, consensus_obs_size=8, 
                 hidden=128, use_consensus_value=True):
        super().__init__()
        
        # Total observation includes base environment + consensus features
        total_obs_dim = base_obs_dim + consensus_obs_size
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(total_obs_dim, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        # Actor heads
        self.individual_actor = nn.Linear(hidden, n_actions)  # Standard RL policy
        self.consensus_actor = nn.Linear(hidden, n_actions)   # Consensus-aware policy
        
        # Value heads  
        self.individual_critic = nn.Linear(hidden, 1)         # Standard value function
        if use_consensus_value:
            self.consensus_critic = nn.Linear(hidden, 1)      # Consensus value function
        
        self.use_consensus_value = use_consensus_value
        
    def forward(self, obs, return_both_policies=False):
        """
        Forward pass returning policy logits and value estimates.
        
        Args:
            obs: Augmented observations [batch_size, total_obs_dim]
            return_both_policies: If True, return both individual and consensus policies
            
        Returns:
            If return_both_policies=False: (policy_logits, value)
            If return_both_policies=True: (individual_logits, consensus_logits, individual_value, consensus_value)
        """
        features = self.feature_extractor(obs)
        
        individual_logits = self.individual_actor(features)
        consensus_logits = self.consensus_actor(features)
        
        individual_value = self.individual_critic(features).squeeze(-1)
        
        if self.use_consensus_value:
            consensus_value = self.consensus_critic(features).squeeze(-1)
        else:
            consensus_value = individual_value
        
        if return_both_policies:
            return individual_logits, consensus_logits, individual_value, consensus_value
        else:
            # Default: return consensus-aware policy
            return consensus_logits, consensus_value


class ConsensusEnhancedPPO:
    """
    PPO algorithm enhanced with consensus dynamics coordination.
    
    This combines the learning capabilities of PPO with the mathematical
    guarantees of consensus dynamics from the handwritten notes.
    """
    
    def __init__(self, env, consensus_coordinator=None, config=None):
        self.env = env
        self.config = config or ConsensusRLConfig()
        
        # Initialize consensus coordinator
        if consensus_coordinator is None:
            self.consensus_coordinator = ConsensusCoordinator()
        else:
            self.consensus_coordinator = consensus_coordinator
        
        # Augmented observation space size
        base_obs_dim = env._obs_dim if hasattr(env, '_obs_dim') else env.obs_dim
        consensus_obs_size = ConsensusRLConfig.CONSENSUS_OBS_SIZE
        
        # Create augmented network
        self.network = ConsensusAugmentedActorCritic(
            base_obs_dim=base_obs_dim,
            n_actions=5,  # up, down, left, right, stay
            consensus_obs_size=consensus_obs_size,
            hidden=PPOConfig.HIDDEN_SIZE
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=PPOConfig.LEARNING_RATE)
        
        # Weight manager for saving/loading
        self.weights_manager = ConsensusWeightsManager()
        
        # Training metrics
        self.training_stats = {
            'connectivity_history': deque(maxlen=1000),
            'formation_error_history': deque(maxlen=1000),
            'convergence_time_history': deque(maxlen=1000),
            'episode_rewards': deque(maxlen=100)
        }
        
    def augment_observation(self, base_obs, agent, positions):
        """
        Augment base environment observation with consensus dynamics features.
        
        Added features:
        1. Local connectivity (how many neighbors this drone has)
        2. Distance to swarm center  
        3. Algebraic connectivity (λ2) of current swarm configuration
        4. Formation error (if target formation is set)
        5. Consensus force magnitude
        6. Separation force magnitude
        7. Direction to nearest neighbor
        8. Swarm spread (max distance between any two drones)
        """
        if isinstance(base_obs, torch.Tensor):
            base_obs = base_obs.numpy()
        
        # Get consensus dynamics information
        coordination_forces, diagnostics = self.consensus_coordinator.compute_coordination_forces(positions)
        
        # Extract features
        agent_idx = list(self.env.agents).index(agent) if agent in self.env.agents else 0
        
        # 1. Local connectivity
        adjacency_matrix = diagnostics['adjacency_matrix']
        local_connectivity = adjacency_matrix[agent_idx].sum() if agent_idx < len(adjacency_matrix) else 0
        local_connectivity_norm = local_connectivity / max(1, len(positions) - 1)  # normalize by max possible
        
        # 2. Distance to swarm center
        swarm_center = np.mean(positions, axis=0)
        agent_pos = positions[agent_idx] if agent_idx < len(positions) else np.array([0, 0])
        dist_to_center = np.linalg.norm(agent_pos - swarm_center)
        dist_to_center_norm = dist_to_center / self.env.grid_size  # normalize by grid size
        
        # 3. Global algebraic connectivity
        algebraic_connectivity = diagnostics['algebraic_connectivity']
        
        # 4. Formation error (if formation is set)
        if self.consensus_coordinator.target_formation is not None:
            # Simplified formation error: distance to desired relative position
            if agent_idx < len(self.consensus_coordinator.target_formation):
                desired_pos = (self.consensus_coordinator.formation_center + 
                             self.consensus_coordinator.target_formation[agent_idx])
                formation_error = np.linalg.norm(agent_pos - desired_pos)
                formation_error_norm = formation_error / self.env.grid_size
            else:
                formation_error_norm = 0.0
        else:
            formation_error_norm = 0.0
        
        # 5-6. Force magnitudes
        if agent_idx < len(coordination_forces):
            consensus_force = diagnostics['consensus_forces'][agent_idx]
            separation_force = diagnostics['separation_forces'][agent_idx]
            consensus_force_mag = np.linalg.norm(consensus_force)
            separation_force_mag = np.linalg.norm(separation_force)
        else:
            consensus_force_mag = separation_force_mag = 0.0
        
        # 7. Direction to nearest neighbor
        if len(positions) > 1:
            distances = [np.linalg.norm(agent_pos - pos) for i, pos in enumerate(positions) if i != agent_idx]
            if distances:
                nearest_idx = np.argmin(distances)
                # Adjust index for skipped agent_idx
                if nearest_idx >= agent_idx:
                    nearest_idx += 1
                nearest_pos = positions[nearest_idx]
                direction_to_nearest = (nearest_pos - agent_pos)
                if np.linalg.norm(direction_to_nearest) > 0:
                    direction_to_nearest = direction_to_nearest / np.linalg.norm(direction_to_nearest)
                nearest_dir_x, nearest_dir_y = direction_to_nearest
            else:
                nearest_dir_x = nearest_dir_y = 0.0
        else:
            nearest_dir_x = nearest_dir_y = 0.0
        
        # 8. Swarm spread
        if len(positions) > 1:
            pairwise_distances = [np.linalg.norm(positions[i] - positions[j]) 
                                 for i in range(len(positions)) 
                                 for j in range(i+1, len(positions))]
            swarm_spread = max(pairwise_distances) if pairwise_distances else 0
            swarm_spread_norm = swarm_spread / (self.env.grid_size * np.sqrt(2))  # normalize by max possible
        else:
            swarm_spread_norm = 0.0
        
        # Combine all consensus features
        consensus_features = np.array([
            local_connectivity_norm,    # [0, 1]
            dist_to_center_norm,        # [0, ~1]  
            algebraic_connectivity,     # [0, ~N] where N is number of drones
            formation_error_norm,       # [0, ~1]
            consensus_force_mag,        # [0, ~inf] - force magnitude
            separation_force_mag,       # [0, ~inf] - force magnitude  
            nearest_dir_x,              # [-1, 1] - unit vector component
            nearest_dir_y,              # [-1, 1] - unit vector component
        ], dtype=np.float32)
        
        # Concatenate base observation with consensus features
        augmented_obs = np.concatenate([base_obs, consensus_features])
        
        return augmented_obs
    
    def select_action_hybrid(self, obs, positions, agent):
        """
        Hybrid action selection combining RL policy with consensus dynamics.
        
        Integration modes:
        - 'additive': Combine RL and consensus action probabilities
        - 'multiplicative': Use RL policy weighted by consensus alignment
        - 'switching': Use RL normally, switch to consensus when needed
        """
        with torch.no_grad():
            # Get RL policy distribution
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            individual_logits, consensus_logits, _, _ = self.network(obs_tensor, return_both_policies=True)
            
            rl_probs = torch.softmax(individual_logits[0], dim=0)
            consensus_aware_probs = torch.softmax(consensus_logits[0], dim=0)
            
            # Get consensus dynamics forces
            coordination_forces, diagnostics = self.consensus_coordinator.compute_coordination_forces(positions)
            
            if self.config.INTEGRATION_MODE == 'additive':
                # Weighted combination of RL and consensus policies
                final_probs = (self.config.RL_WEIGHT * rl_probs + 
                              self.config.CONSENSUS_WEIGHT * consensus_aware_probs)
                
            elif self.config.INTEGRATION_MODE == 'multiplicative':
                # Use RL policy but weight by consensus force alignment
                agent_idx = list(self.env.agents).index(agent) if agent in self.env.agents else 0
                
                if agent_idx < len(coordination_forces):
                    consensus_force = coordination_forces[agent_idx]
                    
                    # Compute alignment between each action and consensus force
                    action_deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up, down, left, right, stay
                    alignments = []
                    
                    for dr, dc in action_deltas:
                        action_vector = np.array([dr, dc])
                        if np.linalg.norm(consensus_force) > 0 and np.linalg.norm(action_vector) > 0:
                            alignment = np.dot(consensus_force, action_vector) / (
                                np.linalg.norm(consensus_force) * np.linalg.norm(action_vector)
                            )
                            alignments.append(max(0, alignment))  # Only positive alignments
                        else:
                            alignments.append(0.5)  # neutral alignment for stay action or zero force
                    
                    alignment_weights = torch.tensor(alignments, dtype=torch.float32)
                    final_probs = rl_probs * (1 + self.config.CONSENSUS_WEIGHT * alignment_weights)
                    final_probs = final_probs / final_probs.sum()  # renormalize
                else:
                    final_probs = rl_probs
                    
            elif self.config.INTEGRATION_MODE == 'switching':
                # Switch to consensus when connectivity is low or swarm is too spread
                connectivity = diagnostics['algebraic_connectivity']
                
                # Compute swarm spread
                pairwise_distances = [np.linalg.norm(positions[i] - positions[j]) 
                                    for i in range(len(positions)) 
                                    for j in range(i+1, len(positions))]
                swarm_spread = max(pairwise_distances) if pairwise_distances else 0
                
                use_consensus = (connectivity < self.config.CONNECTIVITY_THRESHOLD or 
                               swarm_spread > self.config.SPREAD_THRESHOLD)
                
                if use_consensus:
                    # Use pure consensus dynamics
                    agent_idx = list(self.env.agents).index(agent) if agent in self.env.agents else 0
                    consensus_actions = snap_consensus_to_discrete_actions(
                        coordination_forces, self.env, positions
                    )
                    if agent in consensus_actions:
                        return consensus_actions[agent]
                    else:
                        final_probs = consensus_aware_probs
                else:
                    final_probs = rl_probs
                    
            else:
                # Default: use consensus-aware network
                final_probs = consensus_aware_probs
            
            # Sample action
            action_dist = Categorical(final_probs)
            action = action_dist.sample().item()
            
            return action
    
    def compute_consensus_reward(self, positions, diagnostics):
        """
        Compute additional reward terms based on consensus dynamics.
        
        Reward components:
        1. Connectivity bonus (higher λ2 = better connected = higher reward)
        2. Formation maintenance (lower formation error = higher reward)  
        3. Convergence progress (approaching consensus = higher reward)
        """
        connectivity_reward = diagnostics['algebraic_connectivity'] * 0.1
        
        # Formation reward (if formation is set)
        formation_reward = 0.0
        if (self.consensus_coordinator.target_formation is not None and 
            hasattr(self.config, 'FORMATION_REWARD_WEIGHT')):
            
            # Compute formation error
            formation_positions = []
            for i, pos in enumerate(positions):
                if i < len(self.consensus_coordinator.target_formation):
                    desired_pos = (self.consensus_coordinator.formation_center + 
                                 self.consensus_coordinator.target_formation[i])
                    error = np.linalg.norm(pos - desired_pos)
                    formation_positions.append(error)
            
            if formation_positions:
                avg_formation_error = np.mean(formation_positions)
                formation_reward = -self.config.FORMATION_REWARD_WEIGHT * avg_formation_error
        
        total_consensus_reward = connectivity_reward + formation_reward
        return total_consensus_reward
    
    def save_model(self, filename_base, training_stats=None):
        """Save the consensus-enhanced model."""
        metadata = {
            'model_type': 'consensus_enhanced_ppo',
            'training_stats': training_stats or dict(self.training_stats),
            'config': {
                'rl_weight': self.config.RL_WEIGHT,
                'consensus_weight': self.config.CONSENSUS_WEIGHT,
                'integration_mode': self.config.INTEGRATION_MODE,
            }
        }
        
        return self.weights_manager.save_consensus_model(
            self.network, self.consensus_coordinator, filename_base
        )
    
    def load_model(self, filename_base):
        """Load a consensus-enhanced model."""
        state_dict, consensus_params, metadata = self.weights_manager.load_consensus_model(filename_base)
        
        # Load network weights
        self.network.load_state_dict(state_dict)
        
        # Restore consensus coordinator parameters
        for param_name, param_value in consensus_params.items():
            if hasattr(self.consensus_coordinator, param_name):
                setattr(self.consensus_coordinator, param_name, param_value)
        
        print(f"Loaded consensus model: {filename_base}")
        return metadata


def train_consensus_enhanced_ppo(env_factory, config=None, save_path="consensus_ppo"):
    """
    Training function for consensus-enhanced PPO.
    
    This function orchestrates the complete training process including:
    1. Environment setup with consensus coordinator
    2. Network initialization with augmented observations
    3. Training loop with both RL and consensus objectives
    4. Evaluation with connectivity and formation metrics
    5. Model saving with separated weights
    """
    if config is None:
        config = ConsensusRLConfig()
    
    # Create environment and consensus coordinator
    env = env_factory()
    consensus_coordinator = ConsensusCoordinator()
    
    # Set up formation if specified
    if hasattr(config, 'TARGET_FORMATION'):
        consensus_coordinator.set_formation(
            ConsensusConfig.FORMATIONS[config.TARGET_FORMATION]
        )
    
    # Create PPO trainer
    ppo_trainer = ConsensusEnhancedPPO(env, consensus_coordinator, config)
    
    print("Starting consensus-enhanced PPO training...")
    print(f"Environment: {env.__class__.__name__}")
    print(f"Integration mode: {config.INTEGRATION_MODE}")
    print(f"RL weight: {config.RL_WEIGHT}, Consensus weight: {config.CONSENSUS_WEIGHT}")
    
    # Training loop would go here...
    # (This is a template - full implementation would include the complete PPO training loop)
    
    # Save trained model
    ppo_trainer.save_model(save_path)
    
    return ppo_trainer


if __name__ == "__main__":
    # Example usage
    from swarm import SwarmCoverageEnv
    
    def create_env():
        return SwarmCoverageEnv(grid_size=12, n_drones=4, max_steps=80)
    
    # Train consensus-enhanced PPO
    trainer = train_consensus_enhanced_ppo(create_env)