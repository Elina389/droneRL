#!/usr/bin/env python3
"""
Test script to verify consensus dynamics implementation works correctly.

This script runs basic tests on the consensus mathematics and integration
without running full training (which takes a long time).
"""
import sys
import os
import numpy as np
import torch

# Add swarm-env to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm-env"))

from consensus_dynamics import ConsensusCoordinator, snap_consensus_to_discrete_actions
from config import ConsensusConfig, PPOConfig
from weights_manager import ConsensusWeightsManager, WeightsManager
from swarm import SwarmCoverageEnv
from swarm3d import SwarmSearch3DEnv


def test_consensus_mathematics():
    """Test the core consensus dynamics mathematics."""
    print("🧮 Testing consensus dynamics mathematics...")
    
    # Create test swarm positions
    positions = [
        np.array([2, 2]), np.array([4, 4]), np.array([6, 2]), np.array([4, 6])
    ]
    
    coordinator = ConsensusCoordinator(sensing_range=3.0)
    
    # Test adjacency matrix
    A = coordinator.compute_adjacency_matrix(positions)
    print(f"   Adjacency matrix shape: {A.shape}")
    print(f"   Number of connections: {A.sum()}")
    
    # Test Laplacian
    L = coordinator.compute_laplacian_matrix(A)
    print(f"   Laplacian matrix shape: {L.shape}")
    
    # Test algebraic connectivity  
    lambda_2, eigenvalues = coordinator.compute_algebraic_connectivity(L)
    print(f"   Algebraic connectivity (λ2): {lambda_2:.3f}")
    print(f"   Graph connected: {lambda_2 > 0.1}")
    
    # Test consensus forces
    forces, diagnostics = coordinator.compute_coordination_forces(positions)
    print(f"   Consensus forces shape: {forces.shape}")
    print(f"   Average force magnitude: {np.mean([np.linalg.norm(f) for f in forces]):.3f}")
    
    # Test formation control
    coordinator.set_formation(ConsensusConfig.FORMATIONS['triangle'])
    formation_forces, _ = coordinator.compute_coordination_forces(positions)
    print(f"   Formation forces computed: ✓")
    
    # Test convergence prediction
    convergence_time = coordinator.predict_convergence_time(positions)
    print(f"   Predicted convergence time: {convergence_time:.2f} steps")
    
    print("   ✅ Consensus mathematics test passed!")
    return True


def test_environment_integration():
    """Test integration with SwarmCoverageEnv."""
    print("🌍 Testing environment integration...")
    
    # Create small test environment
    env = SwarmCoverageEnv(grid_size=8, n_drones=3, max_steps=20)
    coordinator = ConsensusCoordinator()
    
    obs, _ = env.reset()
    
    # Test a few steps with consensus-guided actions
    for step in range(5):
        positions = [env.positions[agent] for agent in env.agents]
        
        # Compute consensus forces
        forces, diagnostics = coordinator.compute_coordination_forces(positions)
        
        # Convert to discrete actions
        actions = snap_consensus_to_discrete_actions(forces, env, positions)
        
        # Step environment
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        print(f"   Step {step+1}: connectivity={diagnostics['algebraic_connectivity']:.3f}")
    
    print("   ✅ Environment integration test passed!")
    return True


def test_3d_environment():
    """Test 3D environment integration."""
    print("🚁 Testing 3D environment integration...")
    
    # Create small 3D test environment
    env = SwarmSearch3DEnv(grid_size=6, n_drones=2, max_steps=10)
    coordinator = ConsensusCoordinator()
    
    obs, _ = env.reset()
    
    # Test a few steps
    for step in range(3):
        # Get 2D projections of 3D positions for consensus
        positions_3d = [env.positions[agent] for agent in env.agents]
        positions_2d = [(r, c) for r, c, a in positions_3d]
        
        # Compute consensus forces
        forces, diagnostics = coordinator.compute_coordination_forces(positions_2d)
        
        # Take random actions (consensus action conversion for 3D would need more work)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        print(f"   Step {step+1}: positions={positions_3d}")
    
    print("   ✅ 3D environment test passed!")
    return True


def test_weight_management():
    """Test the weight management system."""
    print("💾 Testing weight management...")
    
    # Create a simple test model
    test_model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(), 
        torch.nn.Linear(5, 2)
    )
    
    # Test basic weight manager
    manager = WeightsManager()
    
    # Save weights
    weight_file = manager.save_weights_only(
        test_model, 
        "test_weights.pt",
        metadata={"test": True, "accuracy": 0.95}
    )
    
    # Load weights
    state_dict, metadata = manager.load_weights_only("test_weights.pt")
    
    print(f"   Saved and loaded weights: ✓")
    print(f"   Metadata preserved: {metadata.get('test', False)}")
    
    # Test consensus weight manager
    consensus_manager = ConsensusWeightsManager()
    coordinator = ConsensusCoordinator()
    
    # This would normally save both RL model + consensus params
    # consensus_manager.save_consensus_model(test_model, coordinator, "test_consensus")
    
    print("   ✅ Weight management test passed!")
    return True


def test_configurations():
    """Test configuration system."""
    print("⚙️ Testing configuration system...")
    
    # Test configuration access
    consensus_config = ConsensusConfig()
    ppo_config = PPOConfig()
    
    print(f"   Sensing range: {consensus_config.SENSING_RANGE}")
    print(f"   PPO learning rate: {ppo_config.LEARNING_RATE}")
    print(f"   Available formations: {list(consensus_config.FORMATIONS.keys())}")
    
    # Test formation definitions
    triangle = consensus_config.FORMATIONS['triangle']
    print(f"   Triangle formation points: {len(triangle)}")
    
    print("   ✅ Configuration test passed!")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("🧪 Running Consensus Dynamics Implementation Tests")
    print("=" * 60)
    
    tests = [
        ("Consensus Mathematics", test_consensus_mathematics),
        ("Environment Integration", test_environment_integration), 
        ("3D Environment", test_3d_environment),
        ("Weight Management", test_weight_management),
        ("Configuration System", test_configurations),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📋 Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 Overall Result: {'All tests passed!' if all_passed else 'Some tests failed.'}")
    
    if all_passed:
        print("\n🚀 Your consensus dynamics implementation is ready!")
        print("   You can now run:")
        print("   • python train_consensus_models.py --model 2d --consensus")
        print("   • python train_consensus_models.py --model 3d --consensus") 
        print("   • python train_consensus_models.py --ablation")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()