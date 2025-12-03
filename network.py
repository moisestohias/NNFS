class Network:
    """A sequential neural network"""

    def __init__(self):
        self.layers = []

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
