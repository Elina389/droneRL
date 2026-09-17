"""
Weight management system for separating model weights from configurations.

This module handles saving, loading, and managing neural network weights
separately from training configurations and hyperparameters.
"""
import os
import torch
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from config import PathConfig


class WeightsManager:
    """
    Manages neural network weights separately from model configurations.
    
    This allows for:
    1. Easier weight sharing between different configurations
    2. Smaller file sizes (weights only, no metadata)
    3. Better organization of trained models
    4. Version control and experiment tracking
    """
    
    def __init__(self, base_dir=".", create_dirs=True):
        self.base_dir = Path(base_dir)
        self.weights_dir = self.base_dir / PathConfig.WEIGHTS_DIR
        self.models_dir = self.base_dir / PathConfig.MODELS_DIR
        self.logs_dir = self.base_dir / PathConfig.LOGS_DIR
        
        if create_dirs:
            self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.weights_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def save_weights_only(self, model, weight_filename, metadata=None):
        """
        Save only the model weights (state_dict) without configuration.
        
        Args:
            model: PyTorch model or state_dict
            weight_filename: Filename for weights (e.g., "ppo_2d_weights.pt")  
            metadata: Optional dict with training info (epoch, loss, etc.)
        """
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
        else:
            state_dict = model
        
        # Prepare weight data
        weight_data = {
            'state_dict': state_dict,
            'save_time': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        if metadata:
            weight_data['metadata'] = metadata
        
        # Save weights
        weight_path = self.weights_dir / weight_filename
        torch.save(weight_data, weight_path)
        
        # Log the save
        self._log_weight_operation('save', weight_filename, metadata)
        
        print(f"Weights saved to: {weight_path}")
        return weight_path
    
    def load_weights_only(self, weight_filename, map_location='cpu'):
        """
        Load only model weights without configuration.
        
        Args:
            weight_filename: Filename of saved weights
            map_location: Device mapping for PyTorch load
            
        Returns:
            state_dict: The model state dictionary
            metadata: Any saved metadata
        """
        weight_path = self.weights_dir / weight_filename
        
        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
        weight_data = torch.load(weight_path, map_location=map_location, weights_only=False)
        
        self._log_weight_operation('load', weight_filename)
        
        return weight_data['state_dict'], weight_data.get('metadata', {})
    
    def save_full_model(self, model, model_filename, config, training_stats=None):
        """
        Save complete model with configuration (for backward compatibility).
        
        Args:
            model: PyTorch model
            model_filename: Filename for full model
            config: Training configuration dict
            training_stats: Training statistics and metrics
        """
        full_model_data = {
            'state_dict': model.state_dict(),
            'config': config,
            'training_stats': training_stats,
            'save_time': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        
        # Add model architecture info if available
        if hasattr(model, 'obs_dim'):
            full_model_data['obs_dim'] = model.obs_dim
        if hasattr(model, 'n_actions'):
            full_model_data['n_actions'] = model.n_actions
        
        model_path = self.models_dir / model_filename
        torch.save(full_model_data, model_path)
        
        print(f"Full model saved to: {model_path}")
        return model_path
    
    def list_available_weights(self):
        """List all available weight files with metadata."""
        weight_files = []
        
        for weight_file in self.weights_dir.glob("*.pt"):
            try:
                data = torch.load(weight_file, map_location='cpu', weights_only=False)
                info = {
                    'filename': weight_file.name,
                    'save_time': data.get('save_time', 'Unknown'),
                    'size_mb': weight_file.stat().st_size / 1024 / 1024,
                    'metadata': data.get('metadata', {})
                }
                weight_files.append(info)
            except Exception as e:
                print(f"Could not read {weight_file}: {e}")
        
        return sorted(weight_files, key=lambda x: x['save_time'], reverse=True)
    
    def compare_weights(self, weight_file1, weight_file2):
        """
        Compare two weight files and report differences.
        
        Returns dict with comparison metrics.
        """
        state_dict1, _ = self.load_weights_only(weight_file1)
        state_dict2, _ = self.load_weights_only(weight_file2)
        
        comparison = {
            'identical': True,
            'different_keys': [],
            'weight_differences': {},
            'total_parameters': {},
        }
        
        # Check if keys match
        keys1 = set(state_dict1.keys())
        keys2 = set(state_dict2.keys())
        
        if keys1 != keys2:
            comparison['identical'] = False
            comparison['different_keys'] = {
                'only_in_file1': keys1 - keys2,
                'only_in_file2': keys2 - keys1
            }
        
        # Compare weights for common keys
        common_keys = keys1 & keys2
        for key in common_keys:
            w1 = state_dict1[key]
            w2 = state_dict2[key]
            
            if not torch.equal(w1, w2):
                comparison['identical'] = False
                diff_stats = {
                    'max_abs_diff': torch.max(torch.abs(w1 - w2)).item(),
                    'mean_abs_diff': torch.mean(torch.abs(w1 - w2)).item(),
                    'relative_diff': (torch.norm(w1 - w2) / torch.norm(w1)).item()
                }
                comparison['weight_differences'][key] = diff_stats
        
        # Parameter counts
        comparison['total_parameters'] = {
            'file1': sum(p.numel() for p in state_dict1.values()),
            'file2': sum(p.numel() for p in state_dict2.values())
        }
        
        return comparison
    
    def backup_weights(self, weight_filename, backup_suffix=None):
        """Create a backup copy of weight file."""
        if backup_suffix is None:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        original_path = self.weights_dir / weight_filename
        if not original_path.exists():
            raise FileNotFoundError(f"Weight file not found: {original_path}")
        
        # Create backup filename
        name_parts = weight_filename.split('.')
        backup_filename = '.'.join(name_parts[:-1]) + f'_backup_{backup_suffix}.' + name_parts[-1]
        backup_path = self.weights_dir / backup_filename
        
        # Copy file
        import shutil
        shutil.copy2(original_path, backup_path)
        
        print(f"Backup created: {backup_path}")
        return backup_path
    
    def _log_weight_operation(self, operation, filename, metadata=None):
        """Log weight operations for tracking."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'filename': filename,
            'metadata': metadata
        }
        
        log_file = self.logs_dir / 'weights_operations.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class ConsensusWeightsManager(WeightsManager):
    """
    Specialized weight manager for consensus-enhanced RL models.
    
    Handles the additional complexity of models that combine RL policies
    with consensus dynamics components.
    """
    
    def save_consensus_model(self, rl_model, consensus_coordinator, filename_base):
        """
        Save both RL model weights and consensus coordinator parameters.
        
        Args:
            rl_model: The neural network policy model
            consensus_coordinator: ConsensusCoordinator instance
            filename_base: Base name for files (e.g., "consensus_ppo_2d")
        """
        # Save RL model weights
        rl_weights_file = f"{filename_base}_rl_weights.pt"
        rl_metadata = {
            'model_type': 'rl_policy',
            'architecture': 'ActorCritic',
            'consensus_enhanced': True
        }
        self.save_weights_only(rl_model, rl_weights_file, rl_metadata)
        
        # Save consensus coordinator parameters
        consensus_params = {
            'sensing_range': consensus_coordinator.sensing_range,
            'separation_range': consensus_coordinator.separation_range,
            'separation_gain': consensus_coordinator.separation_gain,
            'consensus_gain': consensus_coordinator.consensus_gain,
            'formation_gain': consensus_coordinator.formation_gain,
            'min_connectivity': consensus_coordinator.min_connectivity,
            'target_formation': consensus_coordinator.target_formation.tolist() if consensus_coordinator.target_formation is not None else None,
            'formation_center': consensus_coordinator.formation_center.tolist(),
            'rotation_speed': consensus_coordinator.rotation_speed,
        }
        
        consensus_file = f"{filename_base}_consensus_params.json"
        consensus_path = self.weights_dir / consensus_file
        
        with open(consensus_path, 'w') as f:
            json.dump(consensus_params, f, indent=2)
        
        print(f"Consensus model saved:")
        print(f"  RL weights: {rl_weights_file}")
        print(f"  Consensus params: {consensus_file}")
        
        return rl_weights_file, consensus_file
    
    def load_consensus_model(self, filename_base):
        """
        Load both RL model weights and consensus coordinator parameters.
        
        Returns:
            rl_state_dict: RL model state dictionary
            consensus_params: Consensus coordinator parameters
            rl_metadata: RL model metadata
        """
        rl_weights_file = f"{filename_base}_rl_weights.pt"
        consensus_file = f"{filename_base}_consensus_params.json"
        
        # Load RL weights
        rl_state_dict, rl_metadata = self.load_weights_only(rl_weights_file)
        
        # Load consensus parameters
        consensus_path = self.weights_dir / consensus_file
        with open(consensus_path, 'r') as f:
            consensus_params = json.load(f)
        
        # Convert lists back to numpy arrays
        if consensus_params['target_formation'] is not None:
            consensus_params['target_formation'] = np.array(consensus_params['target_formation'])
        consensus_params['formation_center'] = np.array(consensus_params['formation_center'])
        
        return rl_state_dict, consensus_params, rl_metadata


def migrate_old_models():
    """
    Utility function to migrate old model files to new weight management system.
    
    Converts existing .pt files in models/ directory to separate weights and configs.
    """
    weights_manager = WeightsManager()
    models_dir = Path(PathConfig.MODELS_DIR)
    
    if not models_dir.exists():
        print("No models directory found, nothing to migrate.")
        return
    
    for model_file in models_dir.glob("*.pt"):
        try:
            print(f"Migrating {model_file.name}...")
            
            # Load old model file
            model_data = torch.load(model_file, map_location='cpu', weights_only=False)
            
            if 'state_dict' in model_data:
                # Extract components
                state_dict = model_data['state_dict']
                config = model_data.get('config', {})
                
                # Create new weight filename
                base_name = model_file.stem
                weight_filename = f"{base_name}_weights.pt"
                
                # Save weights only
                metadata = {
                    'migrated_from': model_file.name,
                    'original_config': config,
                    'migration_date': datetime.now().isoformat()
                }
                
                weights_manager.save_weights_only(state_dict, weight_filename, metadata)
                print(f"  → Created {weight_filename}")
            
        except Exception as e:
            print(f"  Failed to migrate {model_file.name}: {e}")
    
    print("Migration complete!")


if __name__ == "__main__":
    # Example usage and testing
    weights_manager = WeightsManager()
    
    print("Available weights:")
    for weight_info in weights_manager.list_available_weights():
        print(f"  {weight_info['filename']} ({weight_info['size_mb']:.2f} MB)")