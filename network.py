import itertools
import numpy as np

class Network:
    """A sequential neural network"""

    def __init__(self):
        self.layers = []
        self.params = []
        self.grads = []
        self.optimizer_built = False

    def add_layer(self, layer):
        """
        Add a layer to this network. The last layer should be a loss layer.
        :param layer: The Layer object
        :return: self
        """
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
        """
        Run the backward pass and accumulate the gradients.
        """
        top_grad = 1.0
        for layer in self.layers[::-1]:
            top_grad = layer.backward(top_grad)

    def adam_trainstep(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, l2=0.):
        """
        Run the update step after calculating the gradients
        :param alpha: The learning rate
        :param beta_1: The exponential average weight for the first moment
        :param beta_2: The exponential average weight for the second moment
        :param epsilon: The smoothing constant
        :param l2: The l2 decay constant
        """
        if not self.optimizer_built:
            self.params.extend(itertools.chain(*[layer.params for layer in self.layers]))
            self.grads.extend(itertools.chain(*[layer.grads for layer in self.layers]))
            self.first_moments = [np.zeros_like(param) for param in self.params]
            self.second_moments = [np.zeros_like(param) for param in self.params]
            self.time_step = 1
            self.optimizer_built = True
        for param, grad, first_moment, second_moment in zip(self.params, self.grads,
                                                            self.first_moments, self.second_moments):
            first_moment *= beta_1
            first_moment += (1 - beta_1) * grad
            second_moment *= beta_2
            second_moment += (1 - beta_2) * (grad ** 2)
            m_hat = first_moment / (1 - beta_1 ** self.time_step)
            v_hat = second_moment / (1 - beta_2 ** self.time_step)
            param -= alpha * m_hat / (np.sqrt(v_hat) + epsilon) + l2 * param
        self.time_step += 1
