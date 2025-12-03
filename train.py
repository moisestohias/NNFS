from utils.utils import MNIST, Batcher
from network import Network
from activation import Relu
from layers import Linear, Conv2d, MaxPool2d, Pad2DLayer, FlattenLayer
from optimizers import Adam, SGD
from losses import MSE, SoftmaxCELayer

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
model.add_layer(MSE())
# model.add_layer(SoftmaxCELayer())

# # Basic SGD
# # optimizer = SGD(lr=0.01)

# # # SGD with momentum
# optimizer = SGD(lr=0.01, momentum=0.9)

# # # SGD with Nesterov momentum
# # optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

# # # SGD with L2 regularization
# # optimizer = SGD(lr=0.01, l2=0.0001)

optimizer = Adam()

MBS=32
mnist = MNIST(Flat=False, OneHot=True)
n_batches = 10
step_count = 0
n_steps = 2
costs=[]

for (x, y), _ in zip(Batcher(mnist, MBS=MBS), range(n_batches)):
    cost = model.forward(x, y)
    model.backward()
    optimizer.update(model)
    step_count += 1
    if step_count % n_steps == 0:
        costs.append(cost)
        print(f'Curr loss after {step_count} steps: {cost}')

# import matplotlib.pyplot as plt

# # Create a figure with subplots
# plt.style.use("dark_background")
# fig, ax = plt.subplots(1,1, figsize=(12, 8))
# ax.minorticks_on() # set minor grid lines to True
# ax.grid(True, axis='both', which='minor', linestyle='-', alpha=0.5) # minor ticks
# ax.grid(True, axis='both', which='major', color='k', linestyle='-', alpha=0.6) # major ticks
# ax.plot(costs)
# fig.tight_layout()
# plt.savefig("ADAM-Plain.png", pad_inches=0.0)


# # Define SGD parameters
# sgd_params = {
#     'Basic SGD': {'lr': 0.01},
#     'SGD with Momentum': {'lr': 0.01, 'momentum': 0.9},
#     'SGD with Nesterov Momentum': {'lr': 0.01, 'momentum': 0.9, 'nesterov': True},
#     'SGD with L2 Regularization': {'lr': 0.01, 'l2': 0.0001}
# }

# MBS = 32
# mnist = MNIST(Flat=False, OneHot=False)
# n_batches = 500
# n_steps = 2

# # Create a figure with subplots
# fig, axs = plt.subplots(2,2, figsize=(12, 12))
# axs = axs.flatten()

# # Perform multiple runs
# for i, (name, params) in enumerate(sgd_params.items()):
#     optimizer = SGD(**params)
#     step_count = 0
#     costs = []

#     for (x, y), _ in zip(Batcher(mnist, MBS=MBS), range(n_batches)):
#         cost = model.forward(x, y)
#         model.backward()
#         optimizer.update(model)
#         step_count += 1
#         if step_count % n_steps == 0:
#             costs.append(cost)
#             print(f'Curr loss after {step_count} steps with {name}: {cost}')

#     # Plot the results
#     axs[i].plot(costs)
#     axs[i].set_title(name)
#     axs[i].set_xlabel('Steps')
#     axs[i].set_ylabel('Loss')

# # Layout so plots do not overlap
# fig.tight_layout()

# plt.savefig("SGD.png")
