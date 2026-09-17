#!/usr/bin/env python3
"""
Complete training script for consensus-enhanced RL models.

This script trains both 2D and 3D models with matrix-based coordination:
1. Standard PPO models (baseline)
2. Consensus-enhanced PPO models (with graph Laplacian coordination)
3. Ablation studies (consensus vs no-consensus)
4. Hyperparameter sweeps

Usage:
    python train_consensus_models.py --model 2d --consensus
    python train_consensus_models.py --model 3d --consensus  
    python train_consensus_models.py --model both --ablation
    python train_consensus_models.py --sweep consensus_weight
"""
import argparse
import sys
import os
import time
import json
from pathlib import Path

# Add swarm-env to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "swarm-env"))

import torch
import numpy as np

# Import our modules
from rl_train import train as train_2d
from rl_train3d import train as train_3d
from config import (PPOConfig, ConsensusRLConfig, Environment2DConfig, 
                   Environment3DConfig, ExperimentConfig, PathConfig)
from weights_manager import ConsensusWeightsManager
from consensus_dynamics import ConsensusCoordinator


def train_model(model_type, use_consensus=True, config_overrides=None):
    """
    Train a single model with specified configuration.
    
    Args:
        model_type: '2d' or '3d'
        use_consensus: Whether to enable consensus dynamics
        config_overrides: Dict of config parameters to override
    
    Returns:
        Tuple of (trained_network, training_metrics, final_performance)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model (consensus={'enabled' if use_consensus else 'disabled'})")
    print(f"{'='*60}")
    
    # Set consensus configuration globally
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(ConsensusRLConfig, key.upper()):
                setattr(ConsensusRLConfig, key.upper(), value)
    
    start_time = time.time()
    
    try:
        if model_type == '2d':
            net, metrics = train_2d()
        elif model_type == '3d':  
            net, metrics = train_3d()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        training_time = time.time() - start_time
        
        print(f"\n✅ {model_type.upper()} training completed in {training_time:.1f}s")
        
        # Extract final performance
        if model_type == '2d':
            final_performance = {
                'coverage': metrics['coverage_history'][-1] if metrics['coverage_history'] else 0,
                'connectivity': metrics['connectivity_history'][-1] if metrics.get('connectivity_history') else 0,
            }
        else:  # 3d
            final_performance = {
                'damage_detected': metrics['damage_history'][-1] if metrics['damage_history'] else 0,
                'mean_altitude': metrics['altitude_history'][-1] if metrics['altitude_history'] else 0,
                'connectivity': metrics['connectivity_history'][-1] if metrics.get('connectivity_history') else 0,
            }
        
        return net, metrics, final_performance
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, None, None


def run_ablation_study(model_type='2d'):
    """
    Run ablation study comparing different consensus configurations.
    
    Tests:
    1. No consensus (baseline PPO)
    2. Consensus only (no formation/separation) 
    3. Full consensus (consensus + formation + separation)
    """
    print(f"\n🧪 Running ablation study for {model_type.upper()} models")
    
    ablation_configs = ExperimentConfig.ABLATIONS
    results = {}
    
    for ablation_name, config in ablation_configs.items():
        print(f"\n--- Ablation: {ablation_name} ---")
        
        # Override configuration
        config_overrides = {
            'use_consensus': config.get('use_consensus', False),
            'consensus_weight': 0.3 if config.get('use_consensus', False) else 0.0,
        }
        
        net, metrics, performance = train_model(
            model_type, 
            use_consensus=config.get('use_consensus', False),
            config_overrides=config_overrides
        )
        
        if performance:
            results[ablation_name] = {
                'config': config,
                'performance': performance,
                'training_time': time.time()
            }
    
    # Save ablation results
    results_path = Path("logs") / f"ablation_study_{model_type}_{int(time.time())}.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Ablation results saved to: {results_path}")
    
    # Print summary
    print("\n--- Ablation Summary ---")
    for name, result in results.items():
        perf = result['performance']
        if model_type == '2d':
            print(f"{name:20s}: coverage={perf.get('coverage', 0):.1%} connectivity={perf.get('connectivity', 0):.3f}")
        else:
            print(f"{name:20s}: damage={perf.get('damage_detected', 0):.1f} alt={perf.get('mean_altitude', 0):.2f} conn={perf.get('connectivity', 0):.3f}")
    
    return results


def run_hyperparameter_sweep(param_name, model_type='2d'):
    """
    Run hyperparameter sweep for specified parameter.
    
    Args:
        param_name: 'sensing_range', 'consensus_weight', etc.
        model_type: '2d' or '3d'
    """
    print(f"\n🔬 Running hyperparameter sweep: {param_name} ({model_type.upper()})")
    
    # Get parameter values to sweep
    if param_name == 'sensing_range':
        values = ExperimentConfig.SENSING_RANGE_SWEEP
    elif param_name == 'consensus_weight':
        values = ExperimentConfig.CONSENSUS_WEIGHT_SWEEP
    else:
        print(f"❌ Unknown parameter for sweep: {param_name}")
        return None
    
    results = {}
    
    for value in values:
        print(f"\n--- {param_name}={value} ---")
        
        config_overrides = {param_name: value}
        
        net, metrics, performance = train_model(
            model_type, 
            use_consensus=True,
            config_overrides=config_overrides
        )
        
        if performance:
            results[str(value)] = {
                'parameter_value': value,
                'performance': performance
            }
    
    # Save sweep results
    results_path = Path("logs") / f"sweep_{param_name}_{model_type}_{int(time.time())}.json"
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Sweep results saved to: {results_path}")
    
    # Print summary
    print(f"\n--- {param_name} Sweep Summary ---")
    for value, result in results.items():
        perf = result['performance']
        if model_type == '2d':
            print(f"{param_name}={value:6s}: coverage={perf.get('coverage', 0):.1%} connectivity={perf.get('connectivity', 0):.3f}")
        else:
            print(f"{param_name}={value:6s}: damage={perf.get('damage_detected', 0):.1f} connectivity={perf.get('connectivity', 0):.3f}")
    
    return results


def compare_models():
    """Compare performance between 2D and 3D models with consensus."""
    print(f"\n⚖️  Comparing 2D vs 3D models with consensus dynamics")
    
    results = {}
    
    # Train 2D model
    print("\nTraining 2D model...")
    net_2d, metrics_2d, perf_2d = train_model('2d', use_consensus=True)
    if perf_2d:
        results['2d'] = {'performance': perf_2d, 'metrics': metrics_2d}
    
    # Train 3D model  
    print("\nTraining 3D model...")
    net_3d, metrics_3d, perf_3d = train_model('3d', use_consensus=True)
    if perf_3d:
        results['3d'] = {'performance': perf_3d, 'metrics': metrics_3d}
    
    # Save comparison
    results_path = Path("logs") / f"model_comparison_{int(time.time())}.json"
    results_path.parent.mkdir(exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return obj
    
    results_serializable = convert_numpy(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n📊 Comparison results saved to: {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train consensus-enhanced RL models')
    parser.add_argument('--model', choices=['2d', '3d', 'both'], default='2d',
                       help='Model type to train')
    parser.add_argument('--consensus', action='store_true', 
                       help='Enable consensus dynamics')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study')
    parser.add_argument('--sweep', choices=['sensing_range', 'consensus_weight'],
                       help='Run hyperparameter sweep')
    parser.add_argument('--compare', action='store_true',
                       help='Compare 2D vs 3D models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create necessary directories
    for directory in [PathConfig.MODELS_DIR, PathConfig.WEIGHTS_DIR, PathConfig.LOGS_DIR]:
        Path(directory).mkdir(exist_ok=True)
    
    print(f"🚀 Starting consensus-enhanced RL training")
    print(f"   Model: {args.model}")
    print(f"   Consensus: {'enabled' if args.consensus else 'disabled'}")
    print(f"   Seed: {args.seed}")
    
    start_time = time.time()
    
    try:
        if args.ablation:
            # Run ablation study
            if args.model == 'both':
                run_ablation_study('2d')
                run_ablation_study('3d')
            else:
                run_ablation_study(args.model)
                
        elif args.sweep:
            # Run hyperparameter sweep
            if args.model == 'both':
                run_hyperparameter_sweep(args.sweep, '2d')
                run_hyperparameter_sweep(args.sweep, '3d')
            else:
                run_hyperparameter_sweep(args.sweep, args.model)
                
        elif args.compare:
            # Compare models
            compare_models()
            
        else:
            # Standard training
            if args.model == 'both':
                train_model('2d', args.consensus)
                train_model('3d', args.consensus)
            else:
                train_model(args.model, args.consensus)
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\n🏁 Total execution time: {total_time:.1f}s")


if __name__ == "__main__":
    main()