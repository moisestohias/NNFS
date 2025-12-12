"""
Checkpoint management for saving and loading training state.

This module provides utilities for:
- Saving complete training checkpoints (model + optimizer + metadata)
- Loading and resuming training from checkpoints
- Automatic checkpoint rotation to manage disk space
"""

import os
import numpy as np
from datetime import datetime

CHECKPOINT_VERSION = "1.0"


class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.
    
    Features:
        - Automatic checkpoint rotation (keeps last N checkpoints)
        - Compressed storage using np.savez_compressed
        - Full training state preservation
        - Best model tracking
    
    Example:
        >>> checkpoint_mgr = CheckpointManager('./checkpoints', max_to_keep=3)
        >>> # During training
        >>> checkpoint_mgr.save(model, optimizer, epoch=5, step=1000, loss=0.25)
        >>> # To resume
        >>> metadata = checkpoint_mgr.load_latest(model, optimizer)
    """
    
    def __init__(self, save_dir, max_to_keep=5):
        """
        Initialize the checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints.
            max_to_keep: Maximum number of checkpoints to retain.
        """
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        os.makedirs(save_dir, exist_ok=True)
        self.checkpoint_files = self._scan_existing_checkpoints()
    
    def _scan_existing_checkpoints(self):
        """Scans the save directory for existing checkpoints."""
        if not os.path.exists(self.save_dir):
            return []
        
        checkpoints = [
            os.path.join(self.save_dir, f) 
            for f in os.listdir(self.save_dir) 
            if f.startswith('checkpoint_') and f.endswith('.npz')
        ]
        checkpoints.sort(key=os.path.getmtime)
        return checkpoints
    
    def save(self, network, optimizer=None, epoch=0, step=0, 
             loss=None, best_loss=None, **metadata):
        """
        Saves a complete training checkpoint.
        
        Args:
            network: The Network instance to save.
            optimizer: The optimizer instance (optional, for training resumption).
            epoch: Current epoch number.
            step: Current training step.
            loss: Current loss value.
            best_loss: Best loss achieved so far.
            **metadata: Additional metadata to include.
            
        Returns:
            str: Path to the saved checkpoint.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.npz"
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'version': CHECKPOINT_VERSION,
            'epoch': epoch,
            'step': step,
            'timestamp': timestamp,
        }
        
        if loss is not None:
            checkpoint['loss'] = loss
        if best_loss is not None:
            checkpoint['best_loss'] = best_loss
        
        # Add network state
        network_state = network.state_dict()
        checkpoint['network_layer_types'] = np.array(network_state['layer_types'], dtype=object)
        checkpoint['network_num_layers'] = len(network_state['layers'])
        
        for i, layer_state in enumerate(network_state['layers']):
            checkpoint[f'network_layer_{i}_name'] = layer_state.get('layer_name', '')
            params = layer_state.get('params', [])
            checkpoint[f'network_layer_{i}_num_params'] = len(params)
            for j, param in enumerate(params):
                checkpoint[f'network_layer_{i}_param_{j}'] = param
            # Save additional state (e.g., BatchNorm running stats)
            for key, value in layer_state.items():
                if key not in ('layer_name', 'params') and isinstance(value, np.ndarray):
                    checkpoint[f'network_layer_{i}_{key}'] = value
        
        # Add optimizer state if provided
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            checkpoint['optimizer_name'] = opt_state['optimizer_name']
            checkpoint['optimizer_built'] = opt_state.get('optimizer_built', False)
            
            # Save Adam-specific state
            if 'time_step' in opt_state:
                checkpoint['optimizer_time_step'] = opt_state['time_step']
            if 'first_moments' in opt_state:
                checkpoint['optimizer_num_moments'] = len(opt_state['first_moments'])
                for i, m in enumerate(opt_state['first_moments']):
                    checkpoint[f'optimizer_first_moment_{i}'] = m
                for i, m in enumerate(opt_state['second_moments']):
                    checkpoint[f'optimizer_second_moment_{i}'] = m
            
            # Save SGD-specific state
            if 'velocities' in opt_state:
                checkpoint['optimizer_num_velocities'] = len(opt_state['velocities'])
                for i, v in enumerate(opt_state['velocities']):
                    checkpoint[f'optimizer_velocity_{i}'] = v
        
        # Add custom metadata
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                checkpoint[f'meta_{key}'] = np.array(value)
            else:
                checkpoint[f'meta_{key}'] = value
        
        np.savez_compressed(filepath, **checkpoint)
        
        self.checkpoint_files.append(filepath)
        self._cleanup_old_checkpoints()
        
        return filepath
    
    def load(self, path, network, optimizer=None, strict=True):
        """
        Loads a checkpoint from file.
        
        Args:
            path: Path to the checkpoint file.
            network: Network instance to load parameters into.
            optimizer: Optimizer instance to restore state (optional).
            strict: If True, validates architecture matches.
            
        Returns:
            dict: Training metadata (epoch, step, loss, etc.)
        """
        data = np.load(path, allow_pickle=True)
        
        # Reconstruct and load network state
        network_state = self._reconstruct_network_state(data)
        network.load_state_dict(network_state, strict=strict)
        
        # Restore optimizer state if provided
        if optimizer is not None and 'optimizer_name' in data:
            opt_state = self._reconstruct_optimizer_state(data)
            optimizer.load_state_dict(opt_state)
        
        # Extract training metadata
        metadata = {
            'version': str(data.get('version', 'unknown')),
            'epoch': int(data.get('epoch', 0)),
            'step': int(data.get('step', 0)),
        }
        
        if 'loss' in data:
            metadata['loss'] = float(data['loss'])
        if 'best_loss' in data:
            metadata['best_loss'] = float(data['best_loss'])
        if 'timestamp' in data:
            metadata['timestamp'] = str(data['timestamp'])
        
        # Extract custom metadata
        for key in data.files:
            if key.startswith('meta_'):
                attr_name = key[5:]  # Remove 'meta_' prefix
                metadata[attr_name] = data[key]
        
        return metadata
    
    def load_latest(self, network, optimizer=None, strict=True):
        """
        Loads the most recent checkpoint.
        
        Args:
            network: Network instance to load parameters into.
            optimizer: Optimizer instance to restore state (optional).
            strict: If True, validates architecture matches.
        
        Returns:
            dict: Training metadata, or None if no checkpoint exists.
        """
        latest = self.get_latest_checkpoint()
        if latest is None:
            return None
        return self.load(latest, network, optimizer, strict=strict)
    
    def get_latest_checkpoint(self):
        """
        Finds the most recent checkpoint file.
        
        Returns:
            str: Path to latest checkpoint, or None if none exist.
        """
        if not os.path.exists(self.save_dir):
            return None
        
        checkpoints = [
            f for f in os.listdir(self.save_dir) 
            if f.startswith('checkpoint_') and f.endswith('.npz')
        ]
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(
            key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x)),
            reverse=True
        )
        
        return os.path.join(self.save_dir, checkpoints[0])
    
    def _cleanup_old_checkpoints(self):
        """Removes old checkpoints beyond max_to_keep."""
        if len(self.checkpoint_files) > self.max_to_keep:
            to_remove = self.checkpoint_files[:-self.max_to_keep]
            for f in to_remove:
                if os.path.exists(f):
                    os.remove(f)
            self.checkpoint_files = self.checkpoint_files[-self.max_to_keep:]
    
    def _reconstruct_network_state(self, data):
        """Reconstructs network state dict from flattened npz data."""
        num_layers = int(data['network_num_layers'])
        layer_types = list(data['network_layer_types'])
        
        layers = []
        for i in range(num_layers):
            layer_state = {'layer_name': str(data.get(f'network_layer_{i}_name', ''))}
            
            # Reconstruct params list
            num_params = int(data.get(f'network_layer_{i}_num_params', 0))
            layer_state['params'] = [
                data[f'network_layer_{i}_param_{j}'] for j in range(num_params)
            ]
            
            # Reconstruct additional arrays (e.g., running_mean, running_var)
            for key in data.files:
                if key.startswith(f'network_layer_{i}_') and key not in (
                    f'network_layer_{i}_name', f'network_layer_{i}_num_params'
                ) and not key.startswith(f'network_layer_{i}_param_'):
                    attr_name = key[len(f'network_layer_{i}_'):]
                    layer_state[attr_name] = data[key]
            
            layers.append(layer_state)
        
        return {'layers': layers, 'layer_types': layer_types}
    
    def _reconstruct_optimizer_state(self, data):
        """Reconstructs optimizer state dict from flattened npz data."""
        opt_name = str(data['optimizer_name'])
        
        state = {
            'optimizer_name': opt_name,
            'optimizer_built': bool(data.get('optimizer_built', True))
        }
        
        if opt_name == 'Adam':
            state['time_step'] = int(data.get('optimizer_time_step', 1))
            num_moments = int(data.get('optimizer_num_moments', 0))
            state['first_moments'] = [
                data[f'optimizer_first_moment_{i}'] for i in range(num_moments)
            ]
            state['second_moments'] = [
                data[f'optimizer_second_moment_{i}'] for i in range(num_moments)
            ]
        
        elif opt_name == 'SGD':
            num_velocities = int(data.get('optimizer_num_velocities', 0))
            state['velocities'] = [
                data[f'optimizer_velocity_{i}'] for i in range(num_velocities)
            ]
        
        return state


# Convenience functions for quick save/load operations

def save_checkpoint(path, network, optimizer=None, **metadata):
    """
    Convenience function to save a single checkpoint.
    
    Args:
        path: File path for the checkpoint.
        network: Network instance to save.
        optimizer: Optimizer instance (optional).
        **metadata: Additional metadata (epoch, step, loss, etc.)
    """
    save_dict = {'version': CHECKPOINT_VERSION}
    save_dict.update(metadata)
    
    # Add network state
    network_state = network.state_dict()
    save_dict['network_layer_types'] = np.array(network_state['layer_types'], dtype=object)
    save_dict['network_num_layers'] = len(network_state['layers'])
    
    for i, layer_state in enumerate(network_state['layers']):
        save_dict[f'network_layer_{i}_name'] = layer_state.get('layer_name', '')
        params = layer_state.get('params', [])
        save_dict[f'network_layer_{i}_num_params'] = len(params)
        for j, param in enumerate(params):
            save_dict[f'network_layer_{i}_param_{j}'] = param
        for key, value in layer_state.items():
            if key not in ('layer_name', 'params') and isinstance(value, np.ndarray):
                save_dict[f'network_layer_{i}_{key}'] = value
    
    # Add optimizer state
    if optimizer is not None:
        opt_state = optimizer.state_dict()
        save_dict['optimizer_name'] = opt_state['optimizer_name']
        save_dict['optimizer_built'] = opt_state.get('optimizer_built', False)
        
        if 'time_step' in opt_state:
            save_dict['optimizer_time_step'] = opt_state['time_step']
        if 'first_moments' in opt_state:
            save_dict['optimizer_num_moments'] = len(opt_state['first_moments'])
            for i, m in enumerate(opt_state['first_moments']):
                save_dict[f'optimizer_first_moment_{i}'] = m
            for i, m in enumerate(opt_state['second_moments']):
                save_dict[f'optimizer_second_moment_{i}'] = m
        if 'velocities' in opt_state:
            save_dict['optimizer_num_velocities'] = len(opt_state['velocities'])
            for i, v in enumerate(opt_state['velocities']):
                save_dict[f'optimizer_velocity_{i}'] = v
    
    if not path.endswith('.npz'):
        path += '.npz'
    np.savez_compressed(path, **save_dict)


def load_checkpoint(path, network, optimizer=None, strict=True):
    """
    Convenience function to load a single checkpoint.
    
    Args:
        path: Path to the checkpoint file.
        network: Network instance to load into.
        optimizer: Optimizer instance (optional).
        strict: If True, validates architecture matches.
        
    Returns:
        dict: Metadata from the checkpoint.
    """
    # Create a temporary manager to handle the loading
    mgr = CheckpointManager(os.path.dirname(path) or '.', max_to_keep=1)
    return mgr.load(path, network, optimizer, strict=strict)
