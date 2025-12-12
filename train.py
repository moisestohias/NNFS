from utils.utils import MNIST, Batcher
from network import Network
from activation import Relu
from layers import Linear, Conv2d, MaxPool2d, Pad2DLayer, FlattenLayer
from optimizers import Adam, SGD
from losses import MSE, SoftmaxCELayer
from checkpoint import CheckpointManager

# Build the model
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

optimizer = Adam()

# Initialize checkpoint manager
checkpoint_mgr = CheckpointManager('./checkpoints', max_to_keep=3)

# Try to resume from checkpoint
metadata = checkpoint_mgr.load_latest(model, optimizer)
if metadata:
    start_step = metadata['step']
    print(f"Resuming from step {start_step}, loss was {metadata.get('loss', 'N/A')}")
else:
    start_step = 0
    print("Starting fresh training")

MBS = 32
mnist = MNIST(Flat=False, OneHot=True)
n_batches = 10
step_count = start_step
n_steps = 2
checkpoint_every = 5  # Save checkpoint every 5 steps
costs = []

for (x, y), _ in zip(Batcher(mnist, MBS=MBS), range(n_batches)):
    cost = model.forward(x, y)
    model.backward()
    optimizer.update(model)
    step_count += 1
    
    if step_count % n_steps == 0:
        costs.append(cost)
        print(f'Curr loss after {step_count} steps: {cost}')
    
    # Save checkpoint periodically
    if step_count % checkpoint_every == 0:
        ckpt_path = checkpoint_mgr.save(
            model, optimizer,
            epoch=0, step=step_count, loss=cost
        )
        print(f'  Saved checkpoint: {ckpt_path}')

# Save final model (model-only, for inference)
model.save('./checkpoints/final_model')
print(f'Saved final model to ./checkpoints/final_model.npz')




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
