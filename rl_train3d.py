"""
Enhanced 3D PPO trainer with consensus dynamics and graph Laplacian coordination.

This extends the 3D disaster-search environment with matrix-based swarm coordination:
1. Consensus dynamics for coordinated movement and formation control
2. Graph Laplacian matrices for emergent swarm behavior  
3. Algebraic connectivity (λ2) for convergence guarantees
4. Formation control for maintaining specific 3D shapes while searching

The learning story: random policy flies chaotically at high altitude with poor
damage detection. Enhanced policy learns to:
- Maintain coordinated formations for systematic search coverage
- Use consensus dynamics to avoid clustering/redundancy
- Balance individual exploration with swarm coordination
- Fly low in formation to maximize damage detection confidence

Mathematical foundation from handwritten notes:
- 3D positions: xi = (row, col, altitude) 
- Consensus: ẋ = -Lx where L = D - A (graph Laplacian)
- Formation: xi*(t) = c(t) + R(ωt)δi (moving formations)
- Separation: ui^sep = ∑(dij<rs) ks(rs - dij)(xi - xj)/dij

Run:
    PYTHONPATH=./swarm-env python rl_train3d.py
    
Saves models to models/ and weights to weights/ directories.
"""
import os
import time

import numpy as np
import torch
from torch.distributions import Categorical

from swarm3d import SwarmSearch3DEnv, N_ACTIONS
from rl_train import ActorCritic, compute_gae, ppo_update
from consensus_dynamics import ConsensusCoordinator
from config import PPOConfig, ConsensusRLConfig, Environment3DConfig
from weights_manager import ConsensusWeightsManager


def make_env(cfg, seed=None):
    """Create 3D environment with configuration parameters."""
    return SwarmSearch3DEnv(
        grid_size=cfg.get("grid_size", Environment3DConfig.GRID_SIZE),
        n_drones=cfg.get("n_drones", Environment3DConfig.N_DRONES),
        max_steps=cfg.get("max_steps", Environment3DConfig.MAX_STEPS), 
        H_MAX=cfg.get("H_MAX", Environment3DConfig.H_MAX), 
        A_MAX=cfg.get("A_MAX", Environment3DConfig.A_MAX),
        obs_window=cfg.get("obs_window", Environment3DConfig.OBS_WINDOW), 
        n_buildings=cfg.get("n_buildings", Environment3DConfig.N_BUILDINGS),
        n_epicenters=cfg.get("n_epicenters", Environment3DConfig.N_EPICENTERS), 
        seed=seed,
    )


def obs_batch_3d(env, consensus_coordinator=None, use_consensus=True):
    """
    Create observation batch for 3D environment with optional consensus augmentation.
    
    3D consensus features:
    1. Local connectivity (2D projection)
    2. Distance to swarm center (3D distance)
    3. Algebraic connectivity
    4. 3D formation error
    5. Consensus force magnitude (3D)
    6. Separation force magnitude (3D)  
    7. Direction to nearest neighbor (x, y components, normalized z difference)
    8. Swarm spread (3D maximum pairwise distance)
    """
    agents = env.possible_agents
    base_obs = torch.tensor(np.stack([env._get_obs(a) for a in agents]), dtype=torch.float32)
    
    if not use_consensus or consensus_coordinator is None:
        return base_obs
    
    # Get 3D positions
    positions_3d = [env.positions[agent] for agent in agents]  # (r, c, a) tuples
    positions_2d = [(r, c) for r, c, a in positions_3d]  # Project to 2D for adjacency
    
    # Compute consensus dynamics (using 2D projection for connectivity)
    coordination_forces, diagnostics = consensus_coordinator.compute_coordination_forces(positions_2d)
    
    # Compute 3D consensus features
    n_agents = len(positions_3d)
    consensus_features_batch = []
    
    for i, agent in enumerate(agents):
        r, c, a = positions_3d[i]
        pos_3d = np.array([r, c, a])
        
        # 1. Local connectivity (same as 2D)
        adj_matrix = diagnostics['adjacency_matrix']
        local_connections = adj_matrix[i].sum() if i < len(adj_matrix) else 0
        local_connectivity = local_connections / max(1, n_agents - 1)
        
        # 2. Distance to 3D swarm center
        swarm_center_3d = np.mean([np.array([r, c, a]) for r, c, a in positions_3d], axis=0)
        dist_to_center_3d = np.linalg.norm(pos_3d - swarm_center_3d)
        # Normalize by 3D grid diagonal  
        grid_diagonal_3d = env.grid_size * np.sqrt(2) + env.A_MAX
        dist_to_center_norm = dist_to_center_3d / grid_diagonal_3d
        
        # 3. Algebraic connectivity (from 2D projection)
        algebraic_connectivity = diagnostics['algebraic_connectivity']
        
        # 4. 3D Formation error (if 3D formation is set)
        formation_error_3d = 0.0
        if (consensus_coordinator.target_formation is not None and 
            i < len(consensus_coordinator.target_formation)):
            # Extend 2D formation to 3D by adding altitude component
            target_2d = consensus_coordinator.target_formation[i]
            target_altitude = env.A_MAX // 2  # Middle altitude as default
            target_3d = np.array([target_2d[0], target_2d[1], target_altitude])
            target_pos_3d = (consensus_coordinator.formation_center[0], 
                            consensus_coordinator.formation_center[1], target_altitude) + target_3d
            formation_error_3d = np.linalg.norm(pos_3d - target_pos_3d) / grid_diagonal_3d
        
        # 5-6. Force magnitudes (extend 2D forces to 3D by adding altitude component)
        if i < len(coordination_forces):
            consensus_force_2d = diagnostics['consensus_forces'][i]
            separation_force_2d = diagnostics['separation_forces'][i] 
            
            # Add altitude component (simple: altitude consensus toward middle)
            altitude_consensus = (env.A_MAX / 2 - a) * 0.1  # Pull toward middle altitude
            consensus_force_3d = np.array([consensus_force_2d[0], consensus_force_2d[1], altitude_consensus])
            
            # Altitude separation (push away if same altitude and close)
            altitude_separation = 0.0
            for j, (rj, cj, aj) in enumerate(positions_3d):
                if i != j and abs(a - aj) < 1 and np.linalg.norm([r-rj, c-cj]) < 2:
                    altitude_separation += 0.1 * np.sign(a - aj)
            separation_force_3d = np.array([separation_force_2d[0], separation_force_2d[1], altitude_separation])
            
            consensus_force_mag = np.linalg.norm(consensus_force_3d)
            separation_force_mag = np.linalg.norm(separation_force_3d)
        else:
            consensus_force_mag = separation_force_mag = 0.0
        
        # 7. Direction to nearest neighbor (3D)
        if n_agents > 1:
            other_positions = [pos for j, pos in enumerate(positions_3d) if j != i]
            distances_3d = [np.linalg.norm(pos_3d - np.array([rj, cj, aj])) for rj, cj, aj in other_positions]
            if distances_3d:
                nearest_idx = np.argmin(distances_3d)
                nearest_pos_3d = np.array(other_positions[nearest_idx])
                direction_3d = nearest_pos_3d - pos_3d
                if np.linalg.norm(direction_3d) > 0:
                    direction_3d = direction_3d / np.linalg.norm(direction_3d)
                nearest_dir_x, nearest_dir_y, nearest_dir_z = direction_3d
            else:
                nearest_dir_x = nearest_dir_y = nearest_dir_z = 0.0
        else:
            nearest_dir_x = nearest_dir_y = nearest_dir_z = 0.0
        
        # 8. 3D Swarm spread
        if n_agents > 1:
            max_distance_3d = 0
            for j in range(n_agents):
                for k in range(j + 1, n_agents):
                    pos_j = np.array(positions_3d[j])
                    pos_k = np.array(positions_3d[k])
                    dist_3d = np.linalg.norm(pos_j - pos_k)
                    max_distance_3d = max(max_distance_3d, dist_3d)
            swarm_spread_norm = max_distance_3d / grid_diagonal_3d
        else:
            swarm_spread_norm = 0.0
        
        # Combine 3D consensus features
        consensus_features = np.array([
            local_connectivity,     # [0, 1]
            dist_to_center_norm,    # [0, 1]  
            algebraic_connectivity, # [0, n_agents]
            formation_error_3d,     # [0, 1]
            consensus_force_mag,    # [0, inf]
            separation_force_mag,   # [0, inf]
            nearest_dir_x,          # [-1, 1]
            nearest_dir_y,          # [-1, 1] 
        ], dtype=np.float32)
        
        consensus_features_batch.append(consensus_features)
    
    # Concatenate base observations with consensus features
    consensus_features_tensor = torch.tensor(np.array(consensus_features_batch), dtype=torch.float32)
    augmented_obs = torch.cat([base_obs, consensus_features_tensor], dim=1)
    
    return augmented_obs


def collect_rollout_3d(env, net, cfg, consensus_coordinator=None, use_consensus=True):
    """Enhanced 3D rollout collection with consensus dynamics integration."""
    T, N = cfg["rollout_steps"], cfg["n_drones"]
    
    # Determine observation dimension
    if use_consensus:
        base_obs_dim = env.obs_dim
        consensus_obs_size = 8  # From obs_batch_3d
        obs_dim = base_obs_dim + consensus_obs_size
    else:
        obs_dim = env.obs_dim
    
    agents = env.possible_agents

    obs_buf = torch.zeros(T, N, obs_dim)
    act_buf = torch.zeros(T, N, dtype=torch.long)
    logp_buf = torch.zeros(T, N)
    val_buf = torch.zeros(T, N)
    rew_buf = torch.zeros(T, N)
    done_buf = torch.zeros(T)
    
    # Episode statistics
    ep_stats = []  # (damage_detected, mean_alt, alarms, explored) per episode
    
    # Additional 3D consensus tracking
    consensus_metrics_3d = {
        'connectivity_3d': [],
        'formation_error_3d': [],
        'altitude_coordination': [],
        'swarm_compactness': []
    } if use_consensus else None

    env.reset()
    ob = obs_batch_3d(env, consensus_coordinator, use_consensus)
    
    for t in range(T):
        with torch.no_grad():
            logits, value = net(ob)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

        acts = {a: int(action[i].item()) for i, a in enumerate(agents)}
        _, rewards, _, truncs, infos = env.step(acts)

        obs_buf[t] = ob
        act_buf[t] = action
        logp_buf[t] = logp
        val_buf[t] = value
        
        # Standard environment rewards
        env_rewards = torch.tensor([rewards[a] for a in agents], dtype=torch.float32)
        
        # Add 3D consensus rewards if using consensus dynamics
        if use_consensus and consensus_coordinator is not None:
            positions_3d = [env.positions[a] for a in agents]
            positions_2d = [(r, c) for r, c, a in positions_3d]
            
            _, diagnostics = consensus_coordinator.compute_coordination_forces(positions_2d)
            
            # 3D-specific consensus reward components
            connectivity_reward = diagnostics['algebraic_connectivity'] * 0.05
            
            # Altitude coordination reward (encourage flying at similar altitudes)
            altitudes = [a for r, c, a in positions_3d]
            altitude_variance = np.var(altitudes) if len(altitudes) > 1 else 0
            altitude_coord_reward = -0.02 * altitude_variance  # Penalize high variance
            
            # Formation maintenance reward for 3D
            formation_reward_3d = 0.0
            if consensus_coordinator.target_formation is not None:
                formation_errors = []
                for i, (r, c, a) in enumerate(positions_3d):
                    if i < len(consensus_coordinator.target_formation):
                        # Use middle altitude as target
                        target_2d = consensus_coordinator.target_formation[i]
                        target_3d = np.array([
                            consensus_coordinator.formation_center[0] + target_2d[0],
                            consensus_coordinator.formation_center[1] + target_2d[1], 
                            env.A_MAX // 2
                        ])
                        actual_3d = np.array([r, c, a])
                        error = np.linalg.norm(actual_3d - target_3d)
                        formation_errors.append(error)
                
                if formation_errors:
                    avg_formation_error = np.mean(formation_errors)
                    formation_reward_3d = -0.03 * avg_formation_error
            
            # Combine consensus rewards
            total_consensus_reward = connectivity_reward + altitude_coord_reward + formation_reward_3d
            consensus_rewards = torch.full((N,), total_consensus_reward / N, dtype=torch.float32)
            
            # Track 3D metrics
            consensus_metrics_3d['connectivity_3d'].append(diagnostics['algebraic_connectivity'])
            consensus_metrics_3d['altitude_coordination'].append(altitude_variance)
            
            if formation_errors:
                consensus_metrics_3d['formation_error_3d'].append(np.mean(formation_errors))
            
            # Swarm compactness (3D volume occupied)
            if len(positions_3d) > 1:
                pos_array = np.array([[r, c, a] for r, c, a in positions_3d])
                ranges = pos_array.max(axis=0) - pos_array.min(axis=0)
                volume = np.prod(ranges + 1)  # +1 to avoid zero
                consensus_metrics_3d['swarm_compactness'].append(volume)
            
            # Combine environment and consensus rewards
            total_rewards = env_rewards + consensus_rewards
        else:
            total_rewards = env_rewards
        
        rew_buf[t] = total_rewards

        done = bool(truncs[agents[0]])
        done_buf[t] = 1.0 if done else 0.0
        
        if done:
            info = list(infos.values())[0]
            ep_stats.append((info["damage_detected"], info["mean_altitude"],
                             info["alarms"], info["explored"]))
            env.reset()
        
        ob = obs_batch_3d(env, consensus_coordinator, use_consensus)

    with torch.no_grad():
        _, last_val = net(ob)
    
    rollout_data = dict(obs=obs_buf, act=act_buf, logp=logp_buf, val=val_buf,
                       rew=rew_buf, done=done_buf, last_val=last_val, ep_stats=ep_stats)
    
    if use_consensus:
        rollout_data['consensus_metrics_3d'] = consensus_metrics_3d
    
    return rollout_data


def evaluate_3d(net, cfg, consensus_coordinator=None, use_consensus=False, episodes=20, deterministic=False):
    """Enhanced 3D evaluation with consensus dynamics support."""
    env = make_env(cfg)
    agents = env.possible_agents
    dmg, alt, alarms, expl = [], [], [], []
    connectivity_scores = []
    formation_scores = []
    
    for _ in range(episodes):
        env.reset()
        episode_connectivity = []
        episode_formation_errors = []
        
        while env.agents:
            ob = obs_batch_3d(env, consensus_coordinator, use_consensus)
            with torch.no_grad():
                logits, _ = net(ob)
                action = (logits.argmax(-1) if deterministic
                          else Categorical(logits=logits).sample())
            
            acts = {a: int(action[i]) for i, a in enumerate(agents)}
            _, _, _, _, infos = env.step(acts)
            
            # Track consensus metrics if enabled
            if use_consensus and consensus_coordinator is not None:
                positions_3d = [env.positions[a] for a in agents]
                positions_2d = [(r, c) for r, c, a in positions_3d]
                _, diagnostics = consensus_coordinator.compute_coordination_forces(positions_2d)
                episode_connectivity.append(diagnostics['algebraic_connectivity'])
                
                # Track 3D formation error
                if consensus_coordinator.target_formation is not None:
                    formation_errors = []
                    for i, (r, c, a) in enumerate(positions_3d):
                        if i < len(consensus_coordinator.target_formation):
                            target_2d = consensus_coordinator.target_formation[i]
                            target_3d = np.array([
                                consensus_coordinator.formation_center[0] + target_2d[0],
                                consensus_coordinator.formation_center[1] + target_2d[1], 
                                env.A_MAX // 2
                            ])
                            actual_3d = np.array([r, c, a])
                            error = np.linalg.norm(actual_3d - target_3d)
                            formation_errors.append(error)
                    if formation_errors:
                        episode_formation_errors.append(np.mean(formation_errors))
        
        # Record episode results
        info = list(infos.values())[0]
        dmg.append(info["damage_detected"])
        alt.append(info["mean_altitude"])
        alarms.append(info["alarms"])
        expl.append(info["explored"])
        
        if episode_connectivity:
            connectivity_scores.append(np.mean(episode_connectivity))
        if episode_formation_errors:
            formation_scores.append(np.mean(episode_formation_errors))
    
    results = dict(damage=np.mean(dmg), altitude=np.mean(alt),
                  alarms=np.mean(alarms), explored=np.mean(expl))
    
    if use_consensus:
        if connectivity_scores:
            results['connectivity'] = np.mean(connectivity_scores)
        if formation_scores:
            results['formation_error'] = np.mean(formation_scores)
    
    return results


def random_baseline_3d(cfg, episodes=20):
    """Enhanced 3D random baseline."""
    env = make_env(cfg)
    agents = env.possible_agents
    dmg, alt, alarms = [], [], []
    
    for _ in range(episodes):
        env.reset()
        while env.agents:
            acts = {a: int(env.action_space(a).sample()) for a in agents}
            _, _, _, _, infos = env.step(acts)
        info = list(infos.values())[0]
        dmg.append(info["damage_detected"])
        alt.append(info["mean_altitude"])
        alarms.append(info["alarms"])
    
    return dict(damage=np.mean(dmg), altitude=np.mean(alt), alarms=np.mean(alarms))


def train():
    """Enhanced 3D training with consensus dynamics integration."""
    # Use configuration classes
    cfg = dict(
        # Environment configuration
        grid_size=Environment3DConfig.GRID_SIZE, 
        n_drones=Environment3DConfig.N_DRONES, 
        max_steps=Environment3DConfig.MAX_STEPS, 
        H_MAX=Environment3DConfig.H_MAX, 
        A_MAX=Environment3DConfig.A_MAX,
        obs_window=Environment3DConfig.OBS_WINDOW, 
        n_buildings=Environment3DConfig.N_BUILDINGS,
        n_epicenters=Environment3DConfig.N_EPICENTERS,
        
        # PPO configuration
        rollout_steps=PPOConfig.ROLLOUT_STEPS, 
        num_updates=PPOConfig.NUM_UPDATES + 50,  # Longer training for 3D
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
    
    env = make_env(cfg, seed=cfg["seed"])
    
    # Initialize consensus coordinator for 3D
    consensus_coordinator = None
    if cfg.get("use_consensus", False):
        consensus_coordinator = ConsensusCoordinator()
        # Set a 3D formation (square formation at middle altitude)
        from config import ConsensusConfig
        consensus_coordinator.set_formation(ConsensusConfig.FORMATIONS['square'])
        print("3D Consensus dynamics enabled with square formation")
    
    # Determine observation dimensions
    base_obs_dim = env.obs_dim
    consensus_obs_size = 8  # From obs_batch_3d
    obs_dim = base_obs_dim + (consensus_obs_size if cfg.get("use_consensus") else 0)
    
    # Create network with consensus support
    net = ActorCritic(base_obs_dim, N_ACTIONS, hidden=cfg["hidden"],
                     use_consensus=cfg.get("use_consensus", False),
                     consensus_obs_size=consensus_obs_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    # Initialize weight manager
    weights_manager = ConsensusWeightsManager()

    base = random_baseline_3d(cfg)
    consensus_note = " (with 3D consensus dynamics)" if cfg.get("use_consensus") else ""
    print(f"random baseline: damage_detected={base['damage']:.1f}  "
          f"mean_alt={base['altitude']:.2f}  alarms={base['alarms']:.2f}")
    print(f"training enhanced 3D PPO{consensus_note} ({cfg['num_updates']} updates x {cfg['rollout_steps']} steps)...")

    t0 = time.time()
    training_metrics_3d = {
        'damage_history': [],
        'altitude_history': [],
        'alarms_history': [],
        'connectivity_history': [],
        'formation_error_history': [],
        'altitude_coordination_history': [],
        'swarm_compactness_history': []
    }
    
    for update in range(1, cfg["num_updates"] + 1):
        roll = collect_rollout_3d(env, net, cfg, consensus_coordinator, cfg.get("use_consensus", False))
        adv, returns = compute_gae(roll, cfg)
        ppo_update(net, optimizer, roll, adv, returns, cfg)
        
        # Track 3D metrics
        if roll["ep_stats"]:
            stats = np.array(roll["ep_stats"], dtype=float)
            if len(stats) > 0:
                training_metrics_3d['damage_history'].append(stats[:,0].mean())
                training_metrics_3d['altitude_history'].append(stats[:,1].mean())
                training_metrics_3d['alarms_history'].append(stats[:,2].mean())
        
        # Track consensus metrics
        if cfg.get("use_consensus") and "consensus_metrics_3d" in roll:
            metrics = roll["consensus_metrics_3d"]
            if metrics['connectivity_3d']:
                avg_connectivity = np.mean(metrics['connectivity_3d'])
                training_metrics_3d['connectivity_history'].append(avg_connectivity)
            
            if metrics['formation_error_3d']:
                avg_formation_error = np.mean(metrics['formation_error_3d'])
                training_metrics_3d['formation_error_history'].append(avg_formation_error)
            
            if metrics['altitude_coordination']:
                avg_alt_coord = np.mean(metrics['altitude_coordination'])
                training_metrics_3d['altitude_coordination_history'].append(avg_alt_coord)
            
            if metrics['swarm_compactness']:
                avg_compactness = np.mean(metrics['swarm_compactness'])
                training_metrics_3d['swarm_compactness_history'].append(avg_compactness)
        
        if update % 20 == 0 or update == 1:
            status_msg = f"  update {update:3d}/{cfg['num_updates']}"
            
            if roll["ep_stats"]:
                s = np.array(roll["ep_stats"], dtype=float)
                status_msg += f"  damage {s[:,0].mean():.1f}  alt {s[:,1].mean():.2f}  alarms {s[:,2].mean():.2f}"
            
            if cfg.get("use_consensus") and "consensus_metrics_3d" in roll:
                metrics = roll["consensus_metrics_3d"]
                if metrics['connectivity_3d']:
                    avg_conn = np.mean(metrics['connectivity_3d'])
                    status_msg += f"  conn {avg_conn:.3f}"
                if metrics['formation_error_3d']:
                    avg_form_err = np.mean(metrics['formation_error_3d'])
                    status_msg += f"  form_err {avg_form_err:.2f}"
            
            status_msg += f"  ({time.time()-t0:.0f}s)"
            print(status_msg)

    # Evaluation
    ev = evaluate_3d(net, cfg, consensus_coordinator, use_consensus=cfg.get("use_consensus", False))
    
    print(f"\ntrained 3D (stochastic, 20 maps): damage_detected={ev['damage']:.1f}  "
          f"mean_alt={ev['altitude']:.2f}  alarms={ev['alarms']:.2f}  explored={ev['explored']:.0%}")
    
    if cfg.get("use_consensus"):
        if 'connectivity' in ev:
            print(f"  avg connectivity: {ev['connectivity']:.3f}")
        if 'formation_error' in ev:
            print(f"  avg formation error: {ev['formation_error']:.2f}")
    
    print(f"random baseline:               damage_detected={base['damage']:.1f}  "
          f"mean_alt={base['altitude']:.2f}  alarms={base['alarms']:.2f}")

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    # Save full model (legacy format)
    model_filename = "ppo_swarm3d_consensus.pt" if cfg.get("use_consensus") else "ppo_swarm3d.pt"
    path = os.path.join("models", model_filename)
    
    save_data = {
        "state_dict": net.state_dict(), 
        "config": cfg,
        "obs_dim": obs_dim, 
        "n_actions": N_ACTIONS,
        "eval": ev, 
        "random_baseline": base,
        "training_metrics_3d": training_metrics_3d,
        "consensus_enabled": cfg.get("use_consensus", False)
    }
    
    torch.save(save_data, path)
    print(f"saved full 3D model -> {path}")
    
    # Save weights separately
    weight_filename = "ppo_3d_consensus_weights.pt" if cfg.get("use_consensus") else "ppo_3d_weights.pt"
    metadata = {
        'damage_detected': ev['damage'],
        'mean_altitude': ev['altitude'], 
        'alarms': ev['alarms'],
        'training_updates': cfg['num_updates'],
        'consensus_enabled': cfg.get("use_consensus", False),
        'final_metrics_3d': training_metrics_3d
    }
    
    weights_manager.save_weights_only(net, weight_filename, metadata)
    
    # Save consensus coordinator separately if used
    if cfg.get("use_consensus") and consensus_coordinator is not None:
        weights_manager.save_consensus_model(net, consensus_coordinator, "ppo_3d_consensus_complete")
    
    return net, training_metrics_3d


if __name__ == "__main__":
    train()
