import numpy as np
import os


class Network:
    """ The reason why we have the loss layer should be within the network object is for the graidient propgation. """
    def __init__(self): self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, input_, truth):
        """
        Run the entire network, and return the loss.
        :param input_: The input to the network
        :param truth: The ground truth labels to be passed to the last layer
        :return: The calculated loss.
        """
        input_ = self.run(input_)
        return self.layers[-1].forward(input_, truth)

    def run(self, input_, k=-1):
        """
        Run the network for k layers.
        :param k: If positive, run for the first k layers, if negative, ignore the last -k layers. Cannot be 0.
        :param input_: The input to the network
        :return: The output of the second last layer
        """
        k = len(self.layers) if not k else k
        for layer in self.layers[:min(len(self.layers) - 1, k)]:
            input_ = layer.forward(input_)
        return input_

    def backward(self):
        top_grad = 1.0
        for layer in self.layers[::-1]:
            top_grad = layer.backward(top_grad)

    def state_dict(self):
        """
        Returns the complete state of the network.
        
        Returns:
            dict: Contains layer states and layer type information.
        """
        return {
            'layers': [layer.state_dict() for layer in self.layers],
            'layer_types': [layer.layers_name for layer in self.layers]
        }

    def load_state_dict(self, state_dict, strict=True):
        """
        Loads network state from a state dictionary.
        
        Args:
            state_dict: The state dictionary to load.
            strict: If True, raises error on layer type mismatch.
            
        Raises:
            ValueError: If network architecture doesn't match checkpoint.
        """
        saved_types = state_dict.get('layer_types', [])
        
        if strict and len(saved_types) != len(self.layers):
            raise ValueError(
                f"Layer count mismatch: network has {len(self.layers)} layers, "
                f"checkpoint has {len(saved_types)}"
            )
        
        for i, (layer, layer_state) in enumerate(zip(self.layers, state_dict['layers'])):
            if strict and layer.layers_name != saved_types[i]:
                raise ValueError(
                    f"Layer type mismatch at index {i}: "
                    f"expected {layer.layers_name}, got {saved_types[i]}"
                )
            layer.load_state_dict(layer_state)

    def save(self, path):
        """
        Saves the model parameters to a file.
        
        Args:
            path: File path (will add .npz extension if not present)
        """
        state = self.state_dict()
        save_dict = {'layer_types': np.array(state['layer_types'], dtype=object)}
        save_dict['num_layers'] = len(state['layers'])
        
        for i, layer_state in enumerate(state['layers']):
            save_dict[f'layer_{i}_name'] = layer_state.get('layer_name', '')
            params = layer_state.get('params', [])
            save_dict[f'layer_{i}_num_params'] = len(params)
            for j, param in enumerate(params):
                save_dict[f'layer_{i}_param_{j}'] = param
            # Save additional state (e.g., BatchNorm running stats)
            for key, value in layer_state.items():
                if key not in ('layer_name', 'params') and isinstance(value, np.ndarray):
                    save_dict[f'layer_{i}_{key}'] = value
        
        if not path.endswith('.npz'):
            path += '.npz'
        np.savez_compressed(path, **save_dict)

    def load(self, path, strict=True):
        """
        Loads model parameters from a file.
        
        Args:
            path: Path to the checkpoint file.
            strict: If True, validates layer architecture matches.
        """
        if not path.endswith('.npz'):
            path += '.npz'
        
        data = np.load(path, allow_pickle=True)
        state_dict = self._reconstruct_state_dict(data)
        self.load_state_dict(state_dict, strict=strict)

    def _reconstruct_state_dict(self, data):
        """Reconstructs state_dict from flattened npz data."""
        num_layers = int(data['num_layers'])
        layer_types = list(data['layer_types'])
        
        layers = []
        for i in range(num_layers):
            layer_state = {'layer_name': str(data.get(f'layer_{i}_name', ''))}
            
            # Reconstruct params list
            num_params = int(data.get(f'layer_{i}_num_params', 0))
            layer_state['params'] = [data[f'layer_{i}_param_{j}'] for j in range(num_params)]
            
            # Reconstruct additional arrays (e.g., running_mean, running_var)
            for key in data.files:
                if key.startswith(f'layer_{i}_') and key not in (
                    f'layer_{i}_name', f'layer_{i}_num_params'
                ) and not key.startswith(f'layer_{i}_param_'):
                    attr_name = key[len(f'layer_{i}_'):]
                    layer_state[attr_name] = data[key]
            
            layers.append(layer_state)
        
        return {'layers': layers, 'layer_types': layer_types}

