import numpy as np
from functional import *
from functional import _pair, _corr2d

"""
>Note: Bottom grad, is what mich refere to as the error, which what propagate backward through the net
Naming convention: inShape, outShape, D:dilation, P: padding, S:stride, G:groups, Z:input, W:weight, B:bias, inF, outF
Maybe we need to go back to more informative naming convention.
"""

class Layer:
  """ All layers should Only acccept batch of inputs: NCHW"""
  def __init__(self): 
    self.layers_name = self.__class__.__name__
    self.built = False
    self.params = []
    self.grads = []

  def __call__(self, x): return self.forward(x)
  def __repr__(self): return f"{self.layers_name}(Z)"
  def forward(self, input): pass
  def backward(self, TopGrad): pass

  def state_dict(self):
    """
    Returns a dictionary containing the layer's learnable parameters.
    
    Returns:
        dict: State dictionary with layer name and parameter copies.
    """
    return {
        'layer_name': self.layers_name,
        'params': [p.copy() for p in self.params] if self.params else []
    }

  def load_state_dict(self, state_dict):
    """
    Restores the layer's parameters from a state dictionary.
    
    Args:
        state_dict: Dictionary containing 'params' key with parameter arrays.
        
    Raises:
        ValueError: If parameter shapes don't match.
    """
    if not state_dict.get('params') or not self.params:
        return
    for i, param_data in enumerate(state_dict['params']):
        if self.params[i].shape != param_data.shape:
            raise ValueError(
                f"Shape mismatch in layer '{self.layers_name}' param {i}: "
                f"expected {self.params[i].shape}, got {param_data.shape}"
            )
        np.copyto(self.params[i], param_data)

class Linear(Layer):
    def __init__(self, inF, outF, bias=True, dtype=np.float32):
        super().__init__()
        self.layers_name = self.__class__.__name__
        self.trainable = True
        lim = 1 / np.sqrt(inF) # Only inF used to calculate the limit, avoid saturation..
        self.w = np.random.uniform(-lim, lim, (outF, inF)).astype(dtype) # torch style (outF, inF)
        self.b = np.random.randn(outF).astype(dtype) * 0.1 if bias else None
        self.params = [self.w, self.b] if bias else [self.w]
        self.grads = [np.zeros_like(p) for p in self.params]
        self.inShape, self.outShape = (inF,), (outF,)

    def forward(self, z):
        self.z = z
        return affin_trans(self.z, self.w, self.b) # [MBS,inF][outF,inF].T -> [MBS,outF]

    def backward(self, TopGrad):
        self.zGrad, self.wGrad, self.bGrad = affin_trans_backward(TopGrad, self.z, self.w)
        self.grads[0] = self.wGrad
        if self.b is not None:
            self.grads[1] = self.bGrad
        return self.zGrad

class BatchNorm1D(Layer):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, dtype=np.float32):
        self.layers_name = self.__class__.__name__
        self.trainable = True
        
        # Initialize parameters
        self.gamma = np.ones(num_features, dtype=dtype)  # Scale
        self.beta = np.zeros(num_features, dtype=dtype)   # Shift
        
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]

        # Running averages for inference
        self.running_mean = np.zeros(num_features, dtype=dtype) 
        self.running_var = np.ones(num_features, dtype=dtype)
        self.momentum = momentum
        self.eps = eps
        self.training = True  # Flag to switch between training and inference mode

    def forward(self, x):
        if self.training:
            out, self.cache = batch_norm_forward(x, self.gamma, self.beta, self.eps)
            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x.mean(axis=0)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x.var(axis=0)
        else:
            # During inference, use running mean and variance
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        return out

    def backward(self, TopGrad):
        return batch_norm_backward(TopGrad, self.cache)

    def state_dict(self):
        """Returns state including running statistics for inference."""
        state = super().state_dict()
        state['running_mean'] = self.running_mean.copy()
        state['running_var'] = self.running_var.copy()
        return state

    def load_state_dict(self, state_dict):
        """Restores parameters and running statistics."""
        super().load_state_dict(state_dict)
        if 'running_mean' in state_dict:
            np.copyto(self.running_mean, state_dict['running_mean'])
        if 'running_var' in state_dict:
            np.copyto(self.running_var, state_dict['running_var'])


class BatchNorm2d(Layer):
    def __init__(self, num_features, eps=1e-5, dtype=np.float32):
        self.gamma = np.ones(num_features, dtype=dtype)
        self.beta = np.zeros(num_features, dtype=dtype)
        self.eps = eps
        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]

    def forward(self, x):
        return batch_norm_2d_forward(x, self.gamma, self.beta, self.eps)
        
    def backward(self, dout, cache):
        return batch_norm_2d_backward(dout, cache)


class Conv2d(Layer):
  def __init__(self, inCh, outCh, K, S=1, P=0, D=1, G=1, b=True, inShape=None, dtype=np.float32):
    super().__init__()
    self.K, self.S, self.P, self.D = _pair(K), _pair(S), _pair(P), _pair(D) 
    self.layers_name = self.__class__.__name__
    self.trainable = True
    
    self.outShape = calculateConvOutShape(inShape, self.K, self.S, self.P, self.D) if inShape else None
    
    # Init Weights (Kaiming He)
    scale = np.sqrt(2.0 / (inCh * self.K[0] * self.K[1]))
    self.w = (np.random.randn(outCh, inCh, self.K[0], self.K[1]) * scale).astype(dtype)
    self.b = np.zeros(outCh, dtype=dtype) if b else None
    
    self.params = [self.w, self.b] if b else [self.w]
    self.grads = [np.zeros_like(p) for p in self.params]


  def forward(self, x):
    self.z = x
    out = _corr2d(x, self.w, S=self.S, P=self.P, D=self.D)
    if self.b is not None:
      out += self.b.reshape(1, -1, 1, 1)
    return out

  def backward(self, TopGrad):
    # dW, dZ = corr2d_backward(self.z, self.w, TopGrad, mode="valid") # old
    dW, dZ = corr2d_backward(self.z, self.w, TopGrad, S=self.S, P=self.P, D=self.D)
    self.grads[0] = dW
    if self.b is not None:
      self.grads[1] = np.sum(TopGrad, axis=(0, 2, 3))
    return dZ


class MaxPool2d(Layer):
    def __init__(self, K, S=None, P=0):
        super().__init__()
        self.k, self.s, self.p  = _pair(K), _pair(S), _pair(P)
        self.layers_name = self.__class__.__name__
        self.cache = None

    def forward(self, x):
        out, self.cache = max_pool2d(x, self.k, self.s, self.p, return_indices=True)
        return out

    def backward(self, grad_output):
        return max_pool2d_backward(grad_output, self.cache)

class Dropout(Layer):
    def __init__(self, inShape, p=0.1, dtype=np.float32):
        self.p = p # Probability to Drop
        self.inShape = inShape
        self.outShape = inShape
        self.layers_name = self.__class__.__name__

    def forward(self, input):
        self.mask = np.random.rand(*self.inShape).astype(dtype) < self.p
        output = np.copy(input)
        output[self.mask] = 0
        return output

    def backward(self, TopGrad):
        input_gradient = np.ones(self.inShape)
        input_gradient[self.mask] = 0
        return input_gradient


class Reshape(Layer):
    def __init__(self, inShape, outShape):
        self.layers_name = self.__class__.__name__
        self.inShape = inShape if isinstance(inShape,tuple) else (inShape,)
        self.outShape = outShape if isinstance(outShape,tuple) else (outShape,)
    def forward(self, Z): return Z.reshape(Z.shape[0],*self.outShape)
    def backward(self, TopGrad): return np.reshape(TopGrad, self.inShape) # maybe we need TopGrad[1:]

class Flatten(Reshape):
    def __init__(self, inShape):
        super().__init__()
        self.inShape = inShape if isinstance(inShape,tuple) else (inShape,)
        self.outShape = (np.prod(inShape),)
        self.layers_name = self.__class__.__name__

class LSTM(Layer):
    def __init__(self, input_size, hidden_size, bias=True):
        self.inShape = inShape
        self.outShape = inShape
        self.layers_name = self.__class__.__name__
        self.weight_hh = np.random.randn(hidden_size, 4*hidden_size).astype(dtype) # maps previous hidden to new hidden:  torch: .T 
        self.weight_ih = np.random.randn(input_size, 4*hidden_size).astype(dtype) # maps input to hidden:  torch: .T
        self.bias      = np.random.randn(hidden_size).astype(dtype)
        self.cell_st   = np.random.randn(hidden_size).astype(dtype)
        self.hidden_st = np.random.randn(hidden_size).astype(dtype)
        self.params    = (self.weight_hh, self.weight_ih, self.bias)

    def forward(self, input):
        self.input = input
        H, c, self.cach = lstm(input, self.weight_hh, self.weight_ih, self.bias, self.cell_st, self.hidden_st)
        return c

    def backward(self, TopGrad):
        zGrad, self.wGrad = lstmP(TopGrad, self.input, self.cach)
        return zGrad


# --- 
class Pad2DLayer(Layer):
    """Pads a 2D image with zeros."""

    def __init__(self, pad_shape):
        """
        :param pad_shape: A tuple representing the height and the width padding
        """
        Layer.__init__(self)
        self.pad_shape = pad_shape

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (self.input_shape[0], self.input_shape[1] + 2 * self.pad_shape[0],
                             self.input_shape[1] + 2 * self.pad_shape[1])
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output = pad2D(input_, self.pad_shape)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_pad2D(top_grad, self.pad_shape)
        return self.bottom_grad

# ---

class FlattenLayer(Layer):
    """A layer that flattens all the dimensions except the batch."""

    def __init__(self):
        Layer.__init__(self)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output, self.cache = flatten(input_)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_flatten(top_grad, self.cache)
        return self.bottom_grad
