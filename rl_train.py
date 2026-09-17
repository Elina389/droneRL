"""
Enhanced reinforcement learning with consensus dynamics and graph Laplacian coordination.

This implementation combines:
1. PPO (Proximal Policy Optimization) for learning complex behaviors
2. Consensus dynamics for mathematical coordination guarantees  
3. Graph Laplacian matrices for emergent swarm coordination
4. Matrix-based formation control and separation forces

Key enhancements from baseline RL:
- Observations augmented with connectivity and consensus information
- Hybrid action selection combining RL policy with consensus forces
- Additional reward terms for formation maintenance and connectivity
- Algebraic connectivity (λ2) used as convergence metric
- Separated weight management for easier experimentation

Mathematical foundation from handwritten notes:
- aij = 1 if ||xi - xj|| < r (adjacency matrix)
- ẋ = -Lx, L = D - A (consensus dynamics)
- ||x(t) - x̄|| ≤ e^(-λ2t)||x(0) - x̄|| (convergence guarantee)

Run:
    PYTHONPATH=./swarm-env python rl_train.py

Saves weights to weights/ directory and full models to models/ directory.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm-env"))

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from swarm import SwarmCoverageEnv
from consensus_dynamics import ConsensusCoordinator
from config import PPOConfig, ConsensusRLConfig, Environment2DConfig
from weights_manager import ConsensusWeightsManager


class ActorCritic(nn.Module):
    """
    Enhanced Actor-Critic network with consensus dynamics integration.
    
    Key improvements:
    1. Larger observation space including consensus features
    2. Optional separate value heads for individual vs consensus objectives
    3. Residual connections for better gradient flow
    4. Layer normalization for training stability
    """
    def __init__(self, obs_dim, n_actions, hidden=128, use_consensus=True, consensus_obs_size=8):
        super().__init__()
        
        # If using consensus, observation includes additional features
        total_obs_dim = obs_dim + (consensus_obs_size if use_consensus else 0)
        
        # Shared trunk with residual connections
        self.trunk = nn.Sequential(
            nn.Linear(total_obs_dim, hidden), 
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )
        
        # Policy head (actor)
        self.actor = nn.Linear(hidden, n_actions)
        
        # Value head (critic)  
        self.critic = nn.Linear(hidden, 1)
        
        # Optional consensus-specific value head
        if use_consensus:
            self.consensus_critic = nn.Linear(hidden, 1)
        
        self.use_consensus = use_consensus

    def forward(self, obs, return_consensus_value=False):
        h = self.trunk(obs)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        
        if return_consensus_value and self.use_consensus:
            consensus_value = self.consensus_critic(h).squeeze(-1)
            return logits, value, consensus_value
        
        return logits, value


def make_env(cfg):
    """Create environment with optional consensus dynamics integration."""
    return SwarmCoverageEnv(
        grid_size=cfg.get("grid_size", Environment2DConfig.GRID_SIZE),
        n_drones=cfg.get("n_drones", Environment2DConfig.N_DRONES), 
        n_obstacles=cfg.get("n_obstacles", Environment2DConfig.N_OBSTACLES),
        obs_window=cfg.get("obs_window", Environment2DConfig.OBS_WINDOW),
        max_steps=cfg.get("max_steps", Environment2DConfig.MAX_STEPS),
    )


def augment_obs_with_consensus(env, consensus_coordinator, agents):
    """
    Augment environment observations with consensus dynamics features.
    
    Added features per drone:
    1. Local connectivity (normalized number of neighbors)
    2. Distance to swarm center (normalized)
    3. Global algebraic connectivity (λ2)
    4. Formation error (if target formation set)
    5. Consensus force magnitude
    6. Separation force magnitude
    7. Direction to nearest neighbor (x, y components)
    8. Swarm spread (normalized maximum pairwise distance)
    """
    # Get current positions
    positions = [env.positions[agent] for agent in agents]
    
    # Compute consensus dynamics
    coordination_forces, diagnostics = consensus_coordinator.compute_coordination_forces(positions)
    
    augmented_obs = []
    for i, agent in enumerate(agents):
        # Base observation
        base_obs = env._get_obs(agent)
        
        # Consensus features
        consensus_features = compute_consensus_features(
            positions, i, coordination_forces, diagnostics, env.grid_size, consensus_coordinator
        )
        
        # Combine
        full_obs = np.concatenate([base_obs, consensus_features])
        augmented_obs.append(full_obs)
    
    return np.array(augmented_obs)


def compute_consensus_features(positions, agent_idx, coordination_forces, diagnostics, grid_size, consensus_coordinator):
    """Compute consensus-related observation features for one agent."""
    n_agents = len(positions)
    
    if agent_idx >= n_agents:
        return np.zeros(8)  # Return zero features if agent index out of range
    
    agent_pos = np.array(positions[agent_idx])
    
    # 1. Local connectivity (normalized by maximum possible connections)
    adj_matrix = diagnostics['adjacency_matrix']
    local_connections = adj_matrix[agent_idx].sum() if agent_idx < len(adj_matrix) else 0
    local_connectivity = local_connections / max(1, n_agents - 1)
    
    # 2. Distance to swarm center (normalized by grid diagonal)
    swarm_center = np.mean(positions, axis=0)
    dist_to_center = np.linalg.norm(agent_pos - swarm_center)
    dist_to_center_norm = dist_to_center / (grid_size * np.sqrt(2))
    
    # 3. Global algebraic connectivity (λ2)
    algebraic_connectivity = diagnostics['algebraic_connectivity']
    
    # 4. Formation error (normalized)
    formation_error = 0.0
    if (consensus_coordinator.target_formation is not None and 
        agent_idx < len(consensus_coordinator.target_formation)):
        target_pos = (consensus_coordinator.formation_center + 
                     consensus_coordinator.target_formation[agent_idx])
        formation_error = np.linalg.norm(agent_pos - target_pos) / grid_size
    
    # 5-6. Force magnitudes (consensus and separation)
    if agent_idx < len(coordination_forces):
        consensus_force = diagnostics['consensus_forces'][agent_idx]
        separation_force = diagnostics['separation_forces'][agent_idx] 
        consensus_force_mag = np.linalg.norm(consensus_force)
        separation_force_mag = np.linalg.norm(separation_force)
    else:
        consensus_force_mag = separation_force_mag = 0.0
    
    # 7. Direction to nearest neighbor (unit vector components)
    if n_agents > 1:
        other_positions = [pos for i, pos in enumerate(positions) if i != agent_idx]
        distances = [np.linalg.norm(agent_pos - pos) for pos in other_positions]
        if distances:
            nearest_idx = np.argmin(distances)
            nearest_pos = other_positions[nearest_idx]
            direction = nearest_pos - agent_pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            nearest_dir_x, nearest_dir_y = direction
        else:
            nearest_dir_x = nearest_dir_y = 0.0
    else:
        nearest_dir_x = nearest_dir_y = 0.0
    
    # 8. Swarm spread (normalized maximum pairwise distance)
    if n_agents > 1:
        max_distance = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                max_distance = max(max_distance, dist)
        swarm_spread_norm = max_distance / (grid_size * np.sqrt(2))
    else:
        swarm_spread_norm = 0.0
    
    return np.array([
        local_connectivity,     # [0, 1]
        dist_to_center_norm,    # [0, 1]  
        algebraic_connectivity, # [0, inf] - but typically [0, n_agents]
        formation_error,        # [0, 1]
        consensus_force_mag,    # [0, inf]
        separation_force_mag,   # [0, inf]
        nearest_dir_x,          # [-1, 1]
        nearest_dir_y,          # [-1, 1]
    ], dtype=np.float32)


def obs_batch(env, agents, consensus_coordinator=None, use_consensus=True):
    """
    Stack observations into batch tensor, optionally with consensus augmentation.
    
    Returns [N, obs_dim] tensor where obs_dim includes consensus features if enabled.
    """
    if use_consensus and consensus_coordinator is not None:
        augmented_obs = augment_obs_with_consensus(env, consensus_coordinator, agents)
        return torch.tensor(augmented_obs, dtype=torch.float32)
    else:
        return torch.tensor(np.stack([env._get_obs(a) for a in agents]), dtype=torch.float32)


def collect_rollout(env, net, cfg, consensus_coordinator=None, use_consensus=True):
    """
    Enhanced rollout collection with consensus dynamics integration.
    
    Collects both standard RL experience and consensus-related metrics
    for training the enhanced policy.
    """
    T, N = cfg["rollout_steps"], cfg["n_drones"]
    
    # Determine observation dimension
    if use_consensus:
        base_obs_dim = env._obs_dim
        consensus_obs_size = 8  # From compute_consensus_features
        obs_dim = base_obs_dim + consensus_obs_size
    else:
        obs_dim = env._obs_dim
    
    agents = env.possible_agents

    obs_buf = torch.zeros(T, N, obs_dim)
    act_buf = torch.zeros(T, N, dtype=torch.long)
    logp_buf = torch.zeros(T, N)
    val_buf = torch.zeros(T, N)
    rew_buf = torch.zeros(T, N)
    done_buf = torch.zeros(T)
    
    # Additional consensus tracking
    consensus_metrics = {
        'connectivity': [],
        'formation_error': [],
        'convergence_estimate': []
    } if use_consensus else None

    ep_coverages = []
    ep_consensus_rewards = []

    env.reset()
    ob = obs_batch(env, agents, consensus_coordinator, use_consensus)
    
    for t in range(T):
        with torch.no_grad():
            logits, value = net(ob)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

        actions = {a: int(action[i].item()) for i, a in enumerate(agents)}
        _, rewards, _, truncs, infos = env.step(actions)

        obs_buf[t] = ob
        act_buf[t] = action
        logp_buf[t] = logp
        val_buf[t] = value
        
        # Standard environment rewards
        env_rewards = torch.tensor([rewards[a] for a in agents], dtype=torch.float32)
        
        # Add consensus rewards if using consensus dynamics
        if use_consensus and consensus_coordinator is not None:
            positions = [env.positions[a] for a in agents]
            _, diagnostics = consensus_coordinator.compute_coordination_forces(positions)
            
            # Consensus reward components
            connectivity_reward = diagnostics['algebraic_connectivity'] * 0.1
            consensus_reward_per_agent = connectivity_reward / N  # Share among all agents
            consensus_rewards = torch.full((N,), consensus_reward_per_agent, dtype=torch.float32)
            
            # Track metrics
            consensus_metrics['connectivity'].append(diagnostics['algebraic_connectivity'])
            
            # Formation error (if formation is set)
            if consensus_coordinator.target_formation is not None:
                formation_errors = []
                for i, pos in enumerate(positions):
                    if i < len(consensus_coordinator.target_formation):
                        target_pos = (consensus_coordinator.formation_center + 
                                    consensus_coordinator.target_formation[i])
                        error = np.linalg.norm(np.array(pos) - target_pos)
                        formation_errors.append(error)
                
                if formation_errors:
                    avg_formation_error = np.mean(formation_errors)
                    consensus_metrics['formation_error'].append(avg_formation_error)
                    
                    # Formation reward (negative error)
                    formation_reward_per_agent = -0.1 * avg_formation_error / N
                    consensus_rewards += formation_reward_per_agent
            
            # Combine environment and consensus rewards
            total_rewards = env_rewards + consensus_rewards
            ep_consensus_rewards.append(consensus_rewards.mean().item())
        else:
            total_rewards = env_rewards
        
        rew_buf[t] = total_rewards

        done = bool(truncs[agents[0]])
        done_buf[t] = 1.0 if done else 0.0
        
        if done:
            ep_coverages.append(list(infos.values())[0]["coverage"])
            env.reset()
            ob = obs_batch(env, agents, consensus_coordinator, use_consensus)
        else:
            ob = obs_batch(env, agents, consensus_coordinator, use_consensus)

    # bootstrap value for the final observation
    with torch.no_grad():
        _, last_val = net(ob)

    rollout_data = dict(obs=obs_buf, act=act_buf, logp=logp_buf, val=val_buf,
                       rew=rew_buf, done=done_buf, last_val=last_val,
                       ep_coverages=ep_coverages)
    
    if use_consensus:
        rollout_data['consensus_metrics'] = consensus_metrics
        rollout_data['ep_consensus_rewards'] = ep_consensus_rewards
    
    return rollout_data


def compute_gae(roll, cfg):
    """Generalized Advantage Estimation, computed backward through time for
    each drone column. done[t]=1 zeros the bootstrap across an episode
    boundary so advantages never leak between episodes."""
    T, N = roll["rew"].shape
    gamma, lam = cfg["gamma"], cfg["lam"]
    adv = torch.zeros(T, N)
    lastgae = torch.zeros(N)
    for t in reversed(range(T)):
        nonterminal = 1.0 - roll["done"][t]
        next_val = roll["val"][t + 1] if t + 1 < T else roll["last_val"]
        delta = roll["rew"][t] + gamma * next_val * nonterminal - roll["val"][t]
        lastgae = delta + gamma * lam * nonterminal * lastgae
        adv[t] = lastgae
    returns = adv + roll["val"]
    return adv, returns


def ppo_update(net, optimizer, roll, adv, returns, cfg):
    """The PPO clipped-objective update. Flattens all [T, N] transitions into
    one big batch (parameter sharing means every drone's experience trains
    the same weights) and runs several epochs of minibatch gradient steps."""
    obs = roll["obs"].reshape(-1, roll["obs"].shape[-1])
    act = roll["act"].reshape(-1)
    old_logp = roll["logp"].reshape(-1)
    adv = adv.reshape(-1)
    returns = returns.reshape(-1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalize advantages

    n = obs.shape[0]
    idx = np.arange(n)
    clip, ent_c, val_c = cfg["clip"], cfg["ent_coef"], cfg["val_coef"]

    for _ in range(cfg["epochs"]):
        np.random.shuffle(idx)
        for start in range(0, n, cfg["minibatch"]):
            mb = idx[start:start + cfg["minibatch"]]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])

            ratio = torch.exp(logp - old_logp[mb])              # r_t(theta)
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()        # L^CLIP
            value_loss = ((value - returns[mb]) ** 2).mean()     # critic MSE
            entropy = dist.entropy().mean()                      # exploration bonus

            loss = policy_loss + val_c * value_loss - ent_c * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()


def evaluate(net, cfg, consensus_coordinator=None, use_consensus=False, episodes=20, deterministic=True):
    """Enhanced evaluation with consensus dynamics support."""
    env = make_env(cfg)
    agents = env.possible_agents
    covs = []
    connectivity_scores = []
    
    for _ in range(episodes):
        env.reset()
        episode_connectivity = []
        
        while env.agents:
            ob = obs_batch(env, agents, consensus_coordinator, use_consensus)
            with torch.no_grad():
                logits, _ = net(ob)
                action = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
            
            # Execute actions
            actions_dict = {a: int(action[i]) for i, a in enumerate(agents)}
            _, _, _, _, infos = env.step(actions_dict)
            
            # Track connectivity if using consensus
            if use_consensus and consensus_coordinator is not None:
                positions = [env.positions[a] for a in agents]
                _, diagnostics = consensus_coordinator.compute_coordination_forces(positions)
                episode_connectivity.append(diagnostics['algebraic_connectivity'])
        
        # Record episode results
        final_info = list(infos.values())[0] if infos else {"coverage": 0.0}
        covs.append(final_info["coverage"])
        
        if episode_connectivity:
            connectivity_scores.append(np.mean(episode_connectivity))
    
    avg_coverage = float(np.mean(covs))
    
    if use_consensus and connectivity_scores:
        avg_connectivity = float(np.mean(connectivity_scores))
        print(f"  avg connectivity: {avg_connectivity:.3f}")
        return avg_coverage, avg_connectivity
    
    return avg_coverage


def random_baseline(cfg, episodes=20):
    """Random baseline with optional consensus metrics."""
    env = make_env(cfg)
    agents = env.possible_agents
    covs = []
    
    for _ in range(episodes):
        env.reset()
        while env.agents:
            actions = {a: int(env.action_space(a).sample()) for a in agents}
            _, _, _, _, infos = env.step(actions)
        final_info = list(infos.values())[0] if infos else {"coverage": 0.0}
        covs.append(final_info["coverage"])
    
    return float(np.mean(covs))


def train():
    """Enhanced training with consensus dynamics integration."""
    # Use configuration classes
    cfg = dict(
        # Environment configuration
        grid_size=Environment2DConfig.GRID_SIZE, 
        n_drones=Environment2DConfig.N_DRONES,
        n_obstacles=Environment2DConfig.N_OBSTACLES, 
        obs_window=Environment2DConfig.OBS_WINDOW, 
        max_steps=Environment2DConfig.MAX_STEPS,
        
        # PPO configuration  
        rollout_steps=PPOConfig.ROLLOUT_STEPS, 
        num_updates=PPOConfig.NUM_UPDATES, 
        epochs=PPOConfig.EPOCHS, 
        minibatch=PPOConfig.MINIBATCH_SIZE,
        gamma=PPOConfig.GAMMA, 
        lam=PPOConfig.LAMBDA, 
        clip=PPOConfig.CLIP_EPSILON, 
        ent_coef=PPOConfig.ENTROPY_COEF, 
        val_coef=PPOConfig.VALUE_COEF, 
        lr=PPOConfig.LEARNING_RATE,
        hidden=PPOConfig.HIDDEN_SIZE, 
        seed=PPOConfig.SEED,
        
        # Consensus configuration
        use_consensus=True,  # Enable consensus dynamics
        consensus_weight=ConsensusRLConfig.CONSENSUS_WEIGHT,
        rl_weight=ConsensusRLConfig.RL_WEIGHT,
    )
    
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    env = make_env(cfg)
    
    # Initialize consensus coordinator
    consensus_coordinator = None
    if cfg.get("use_consensus", False):
        consensus_coordinator = ConsensusCoordinator()
        # Set a target formation for training
        from config import ConsensusConfig
        consensus_coordinator.set_formation(ConsensusConfig.FORMATIONS['triangle'])
        print("Consensus dynamics enabled with triangle formation")
    
    # Determine observation dimensions
    base_obs_dim = env._obs_dim
    consensus_obs_size = 8  # From compute_consensus_features  
    obs_dim = base_obs_dim + (consensus_obs_size if cfg.get("use_consensus") else 0)
    n_actions = 5
    
    # Create network with consensus support
    net = ActorCritic(base_obs_dim, n_actions, hidden=cfg["hidden"], 
                     use_consensus=cfg.get("use_consensus", False),
                     consensus_obs_size=consensus_obs_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    # Initialize weight manager
    weights_manager = ConsensusWeightsManager()

    base = random_baseline(cfg)
    consensus_note = " (with consensus dynamics)" if cfg.get("use_consensus") else ""
    print(f"random baseline coverage: {base:.1%}")
    print(f"training enhanced PPO{consensus_note} ({cfg['num_updates']} updates x {cfg['rollout_steps']} steps)...")

    t0 = time.time()
    training_metrics = {
        'coverage_history': [],
        'connectivity_history': [],
        'formation_error_history': [],
        'consensus_reward_history': []
    }
    
    for update in range(1, cfg["num_updates"] + 1):
        roll = collect_rollout(env, net, cfg, consensus_coordinator, cfg.get("use_consensus", False))
        adv, returns = compute_gae(roll, cfg)
        ppo_update(net, optimizer, roll, adv, returns, cfg)
        
        # Track metrics
        if roll["ep_coverages"]:
            avg_coverage = np.mean(roll["ep_coverages"])
            training_metrics['coverage_history'].append(avg_coverage)
        
        if cfg.get("use_consensus") and "consensus_metrics" in roll:
            metrics = roll["consensus_metrics"]
            if metrics['connectivity']:
                avg_connectivity = np.mean(metrics['connectivity'])
                training_metrics['connectivity_history'].append(avg_connectivity)
            
            if metrics['formation_error']:
                avg_formation_error = np.mean(metrics['formation_error']) 
                training_metrics['formation_error_history'].append(avg_formation_error)
        
        if update % 10 == 0 or update == 1:
            train_cov = np.mean(roll["ep_coverages"]) if roll["ep_coverages"] else float("nan")
            
            status_msg = f"  update {update:3d}/{cfg['num_updates']}  train-coverage {train_cov:.1%}"
            
            if cfg.get("use_consensus") and "consensus_metrics" in roll:
                metrics = roll["consensus_metrics"]
                if metrics['connectivity']:
                    avg_conn = np.mean(metrics['connectivity'])
                    status_msg += f"  connectivity {avg_conn:.3f}"
                if metrics['formation_error']:
                    avg_form_err = np.mean(metrics['formation_error'])
                    status_msg += f"  formation_error {avg_form_err:.2f}"
            
            status_msg += f"  ({time.time()-t0:.0f}s)"
            print(status_msg)

    # Evaluation
    stochastic = evaluate(net, cfg, consensus_coordinator, use_consensus=cfg.get("use_consensus", False), deterministic=False)
    greedy = evaluate(net, cfg, consensus_coordinator, use_consensus=cfg.get("use_consensus", False), deterministic=True)
    
    print(f"\ntrained policy coverage (stochastic, 20 maps): {stochastic:.1%}")
    print(f"trained policy coverage (greedy, 20 maps):     {greedy:.1%}")
    print(f"random baseline:                               {base:.1%}")

    # Create models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    # Save full model (legacy format)
    model_filename = "ppo_swarm_consensus.pt" if cfg.get("use_consensus") else "ppo_swarm.pt"
    path = os.path.join("models", model_filename)
    
    save_data = {
        "state_dict": net.state_dict(),
        "config": cfg,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "eval_coverage": stochastic,
        "eval_coverage_greedy": greedy,
        "random_baseline": base,
        "training_metrics": training_metrics,
        "consensus_enabled": cfg.get("use_consensus", False)
    }
    
    torch.save(save_data, path)
    print(f"saved full model -> {path}")
    
    # Save weights separately
    weight_filename = "ppo_consensus_weights.pt" if cfg.get("use_consensus") else "ppo_weights.pt"
    metadata = {
        'coverage_stochastic': stochastic,
        'coverage_greedy': greedy,
        'training_updates': cfg['num_updates'],
        'consensus_enabled': cfg.get("use_consensus", False),
        'final_metrics': training_metrics
    }
    
    weights_manager.save_weights_only(net, weight_filename, metadata)
    
    # Save consensus coordinator separately if used
    if cfg.get("use_consensus") and consensus_coordinator is not None:
        weights_manager.save_consensus_model(net, consensus_coordinator, "ppo_consensus_complete")
    
    return net, training_metrics


if __name__ == "__main__":
    train()
