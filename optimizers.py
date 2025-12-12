import numpy as np
import itertools

# All optimzers should define a single method 'update' that updates the network params.

class Optimizer:
    def __init__(self): raise NotImplementedError
    def update(self): raise NotImplementedError
    def state_dict(self): raise NotImplementedError
    def load_state_dict(self, state_dict): raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, l2=0.):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.l2 = l2
        self.optimizer_built = False
        self.params = []
        self.first_moments = []
        self.second_moments = []
        self.time_step = 1

    def update(self, network):
        if not self.optimizer_built:
            self.params.extend(itertools.chain(*[layer.params for layer in network.layers]))
            self.first_moments = [np.zeros_like(param) for param in self.params]
            self.second_moments = [np.zeros_like(param) for param in self.params]
            self.optimizer_built = True
        
        current_grads = list(itertools.chain(*[layer.grads for layer in network.layers]))
        
        for param, grad, first_moment, second_moment in zip(self.params, current_grads,
                                                            self.first_moments, self.second_moments):
            first_moment *= self.beta_1
            first_moment += (1 - self.beta_1) * grad
            second_moment *= self.beta_2
            second_moment += (1 - self.beta_2) * (grad ** 2)
            m_hat = first_moment / (1 - self.beta_1 ** self.time_step)
            v_hat = second_moment / (1 - self.beta_2 ** self.time_step)
            param -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon) + self.l2 * param
        self.time_step += 1

    def state_dict(self):
        """
        Returns the optimizer state for checkpointing.
        
        Returns:
            dict: Contains optimizer name, hyperparameters, and internal state.
        """
        return {
            'optimizer_name': 'Adam',
            'alpha': self.alpha,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'epsilon': self.epsilon,
            'l2': self.l2,
            'time_step': self.time_step,
            'first_moments': [m.copy() for m in self.first_moments],
            'second_moments': [m.copy() for m in self.second_moments],
            'optimizer_built': self.optimizer_built
        }

    def load_state_dict(self, state_dict):
        """
        Restores optimizer state from a state dictionary.
        
        Args:
            state_dict: Dictionary containing optimizer state.
            
        Raises:
            ValueError: If optimizer type doesn't match.
        """
        if state_dict.get('optimizer_name') != 'Adam':
            raise ValueError(
                f"Optimizer mismatch: expected Adam, got {state_dict.get('optimizer_name')}"
            )
        
        self.time_step = state_dict['time_step']
        self.first_moments = [m.copy() for m in state_dict['first_moments']]
        self.second_moments = [m.copy() for m in state_dict['second_moments']]
        self.optimizer_built = state_dict.get('optimizer_built', True)


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0, nesterov=False, l2=0.):
        """
        Stochastic Gradient Descent optimizer.
        
        Args:
            lr: Learning rate
            momentum: Momentum factor (0 to 1). If 0, no momentum is used.
            nesterov: Whether to use Nesterov momentum
            l2: L2 regularization (weight decay) coefficient
        """
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.l2 = l2
        self.optimizer_built = False
        self.params = []
        self.velocities = []
    
    def update(self, network):
        if not self.optimizer_built:
            self.params.extend(itertools.chain(*[layer.params for layer in network.layers]))
            self.velocities = [np.zeros_like(param) for param in self.params]
            self.optimizer_built = True
        
        current_grads = list(itertools.chain(*[layer.grads for layer in network.layers]))
        
        for param, grad, velocity in zip(self.params, current_grads, self.velocities):
            # Add L2 regularization to gradient
            if self.l2 > 0:
                grad = grad + self.l2 * param
            
            # Update velocity with momentum
            if self.momentum > 0:
                velocity *= self.momentum
                velocity += grad
                
                # Apply Nesterov momentum if enabled
                if self.nesterov:
                    param -= self.lr * (grad + self.momentum * velocity)
                else:
                    param -= self.lr * velocity
            else:
                # Standard SGD without momentum
                param -= self.lr * grad

    def state_dict(self):
        """
        Returns the optimizer state for checkpointing.
        
        Returns:
            dict: Contains optimizer name, hyperparameters, and internal state.
        """
        return {
            'optimizer_name': 'SGD',
            'lr': self.lr,
            'momentum': self.momentum,
            'nesterov': self.nesterov,
            'l2': self.l2,
            'velocities': [v.copy() for v in self.velocities],
            'optimizer_built': self.optimizer_built
        }

    def load_state_dict(self, state_dict):
        """
        Restores optimizer state from a state dictionary.
        
        Args:
            state_dict: Dictionary containing optimizer state.
            
        Raises:
            ValueError: If optimizer type doesn't match.
        """
        if state_dict.get('optimizer_name') != 'SGD':
            raise ValueError(
                f"Optimizer mismatch: expected SGD, got {state_dict.get('optimizer_name')}"
            )
        
        self.velocities = [v.copy() for v in state_dict['velocities']]
        self.optimizer_built = state_dict.get('optimizer_built', True)
