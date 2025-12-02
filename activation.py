import numpy as np

"""
+ ReLU, LeakyReLU, ELU enable faster and better convergence than sigmoids.
+ [GELU](https://paperswithcode.com/method/gelu): Gaussian Error Linear Unit used in most Transformers(GPT-3, BERT)
+ [Hard-Swish](https://paperswithcode.com/method/hard-swish) 

## TODO: 
  + Make sure you are Clipping correctly
"""


__all__ = [
"Activation",
"Sigmoid",
"Relu",
"Relu6",
"LeakyRelu",
"Elu",
"Swish",
"Tanh",
"Gelu",
"QuickGelu",
"Hardswish",
"Softplus",
"Softmax",
]

from layers import Layer

class Activation(Layer):
  def __init__(self, activation, activationP):
    self.activation = activation
    self.activationP = activationP
    self.params = []
    self.grads = []
    self.layers_name = self.__class__.__name__

  def forward(self, input):
    self.input = input # save input for the backward pass
    return self.activation(self.input)

  def backward(self, topGrad): return np.multiply(topGrad, self.activationP(self.input))


# Precomputed constants
sqrt_2_over_pi = 0.7978845608 # sqrt(2/pi)
alpha = 0.044715
alpha_deriv = 3 * alpha  # 0.134145
CLIP_MIN, CLIP_MAX = -50.0, 50.0
def _clip_exp(x): return np.clip(x, CLIP_MIN, CLIP_MAX)

# ───────────────────────────────────────────────────────────────
def sigmoid(x): return 1.0 / (1.0 + np.exp(-_clip_exp(x)))
def sigmoidP(x): s = sigmoid(x); return s * (1.0 - s)
def relu(x): return np.where(x >= 0, x, 0.0)
def reluP(x): return np.where(x >= 0, 1.0, 0.0)
def leaky_relu(x, alpha=0.01): return np.where(x >= 0, x, alpha * x)
def leaky_reluP(x, alpha=0.01): return np.where(x >= 0, 1.0, alpha)
def relu6(x): return np.minimum(np.maximum(x, 0.0), 6.0)
def relu6P(x): return np.where((x > 0.0) & (x < 6.0), 1.0, 0.0)
def elu(x, alpha=1.0): return np.where(x >= 0, x, alpha * (np.exp(_clip_exp(x)) - 1.0))
def eluP(x, alpha=1.0): return np.where(x >= 0, 1.0, alpha * np.exp(_clip_exp(x)))
def swish(x): s = sigmoid(x); return x * s
def swishP(x): s = sigmoid(x); return s + s * (1.0 - s) * x
silu, siluP = swish, swishP
def tanh(x): return np.tanh(x)
def tanhP(x): t = np.tanh(x); return 1.0 - t * t

def gelu(x): return 0.5*x*(1+np.tanh(sqrt_2_over_pi*(x+alpha*np.power(x,3)))) 
def geluP (x): 
    return 0.5 * (1 + np.tanh(sqrt_2_over_pi * (x + alpha * x**3))) \
    + 0.5 * x * (1 - np.tanh(sqrt_2_over_pi * (x + alpha * x**3))**2) \
    * (sqrt_2_over_pi * (1 + alpha_deriv * x**2))
def quick_gelu(x): return x*sigmoid(x*1.702) # faster version but inacurate
def quick_geluP(x): return 1.702*sigmoidP(x*1.702)
def hardswish(x): return x*relu(x+3.0)/6.0
def hardswishP(x): return 1.0/6.0 *relu(x+3)*(x+1.0)
def softplus(x, limit=20.0, beta=1.0): return (1.0/beta) * np.log(1 + np.exp(x*beta))
def softplusP(limit=20, beta=1.0): _s = np.exp(x*beta) ; return (beta*_s)/(1+_s)


class Sigmoid(Activation):
  def __init__(self): super().__init__(sigmoid, sigmoidP)
class Relu(Activation):
  def __init__(self): super().__init__(relu, reluP)
class Relu6(Activation):
  def __init__(self): super().__init__(relu6, relu6P)
class LeakyRelu(Activation):
  def __init__(self, alpha=0.01): super().__init__(leaky_relu, leaky_reluP)
class Elu(Activation):
  def __init__(self, alpha=0.01): super().__init__(elu, eluP)
class Swish(Activation):
  def __init__(self): super().__init__(swish, swishP)
class Tanh(Activation):
  def __init__(self): super().__init__(tanh, tanhP)
class Gelu(Activation):
  def __init__(self): super().__init__(gelu, geluP)
class QuickGelu(Activation):
  def __init__(self): super().__init__(quick_gelu, quick_geluP)
class Hardswish(Activation):
  def __init__(self): super().__init__(hardswish, hardswishP)
class Softplus(Activation):
  def __init__(self, limit=20.0, beta=1.0): super().__init__(softplus, softplusP)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    def backward(self, topGrad):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, topGrad)
