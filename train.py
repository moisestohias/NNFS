from activation import Relu
from network import Network
from utils.utils import MNIST, Batcher
from layers import Linear, Conv2d, MaxPool2d, Pad2DLayer, FlattenLayer, SoftmaxCELayer
import numpy as np


model = Network()
model.add_layer(Conv2d(1, 10, (3, 3)))
model.add_layer(Relu())
model.add_layer(Pad2DLayer((2, 2)))
model.add_layer(Conv2d(10, 10, (3, 3)))
model.add_layer(Relu())
model.add_layer(MaxPool2d((2, 2)))
model.add_layer(Conv2d(10, 10, (3, 3)))
model.add_layer(Relu())
model.add_layer(MaxPool2d((2, 2)))
model.add_layer(FlattenLayer())
model.add_layer(Linear(360, 32))
model.add_layer(Relu())
model.add_layer(Linear(32, 10))
model.add_layer(SoftmaxCELayer())

n_iters=3
MBS=100
mnist = MNIST(Flat=False, OneHot=False)
for x, y in Batcher(mnist, MBS=MBS):
    cost = model.forward(x, y)
    print(f'Curr loss: {cost}')
    model.backward()
    model.adam_trainstep()
    print(cost)
