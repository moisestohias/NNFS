# 2025-12-02 - Bug fix
## Fix Linear Layer Initialization, Gradient Updates and Stale Gradient References in Network

## Issue Description
The loss was not decreasing during training when running `train.py`. The issue was obvious from inspecting the loss.

## Root Cause Analysis
Upon investigation, two distinct but compounding issues were identified:

1.  **`Linear` Layer Initialization & Gradient Updates**:
    *   **Initialization Order**: In `layers.py`, the `Linear` class called `super().__init__()` at the very end of its `__init__` method. The base `Layer.__init__` resets `self.params` and `self.grads` to empty lists. This effectively wiped out the initialized weights and biases of the `Linear` layer immediately after creation.
    *   **Missing Gradient Assignment**: The `Linear.backward` method calculated gradients (`self.wGrad`, `self.bGrad`) but failed to update the `self.grads` list. This meant the optimizer had no gradient information to work with for these layers.

2.  **Stale Gradient References in `Network` Class**:
    *   In `network.py`, the `adam_trainstep` method collected all layer gradients into a single list (`self.grads`) only once, when `self.optimizer_built` was False.
    *   However, the layers (like `Conv2d` and the fixed `Linear`) were updating their gradients by *replacing* the numpy arrays in their `self.grads` list (e.g., `self.grads[0] = dW`).
    *   Because the `Network` class held references to the *original* (zero-initialized) arrays, it never saw the updated gradients produced by the layers. It continued to optimize using zeros, resulting in no parameter updates.

## Fixes Implemented

### 1. `layers.py`: Fix `Linear` Layer
*   **Moved `super().__init__()`**: The call to the base constructor was moved to the beginning of `Linear.__init__` to ensure `self.params` and `self.grads` are correctly set up and not overwritten.
*   **Updated `backward`**: Added logic to explicitly update `self.grads[0]` and `self.grads[1]` with the calculated gradients.

### 2. `network.py`: Fix `adam_trainstep`
*   **Dynamic Gradient Fetching**: Modified `adam_trainstep` to re-fetch the gradients from all layers (`current_grads`) at the beginning of every training step. This ensures the optimizer always operates on the latest gradient arrays computed during the backward pass.

## Changes Made

### `layers.py`
```python
class Linear(Layer):
    def __init__(self, inF, outF, bias=True, dtype=np.float32):
        super().__init__()  # Moved to top
        # ... (initialization code)
        self.params = [self.w, self.b] if bias else [self.w]
        self.grads = [np.zeros_like(p) for p in self.params]
        # ...

    def backward(self, TopGrad):
        self.zGrad, self.wGrad, self.bGrad = affin_trans_backward(TopGrad, self.z, self.w)
        self.grads[0] = self.wGrad
        if self.b is not None:
            self.grads[1] = self.bGrad
        return self.zGrad
```

### `network.py`
```python
    def adam_trainstep(self, ...):
        # ... (initialization check)
        
        # ADDED: Re-fetch grads to get the latest arrays from layers
        current_grads = list(itertools.chain(*[layer.grads for layer in self.layers]))
        
        for param, grad, first_moment, second_moment in zip(self.params, current_grads, ...):
            # ... (update logic)
```

## Verification
After applying these fixes, `train.py` was executed. The loss was observed to decrease significantly (from ~2.3 to ~0.3) over the course of training, confirming that the network is now learning correctly.
