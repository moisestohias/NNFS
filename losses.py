# looses.py
import numpy as np 
from layers import Layer as _Layer

# Functional Losses
# TODO: protect agains overflow: clip and add epsilon: p = np.clip(p, 1e-15, 1 - 1e-15); log(x + eps)

def mse(y, p): return 0.5*np.mean((y-p)**2)
def mseP(y, p): return (p-y) /np.prod(y.shape)


def bce(y, p): p = np.clip(p, 1e-15, 1 - 1e-15) ; return (-y * np.log(p)-(1-y)*np.log(1-p)).mean()
def bceP(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return ((1 - y) / (1 - p) - y / p) / y.shape[0]



def backward_crossentropy(top_grad, x, y):
    res = np.zeros(x.shape, dtype=x.dtype)
    res[np.arange(x.shape[0]), y] = - np.reciprocal(x[np.arange(x.shape[0]), y]) / x.shape[0]
    return res * top_grad


# we must clip
def softmax(x): s = np.exp(x-x.max(axis=-1, keepdims=True)); return s / s.sum(axis=-1, keepdims=True)
def softmaxP(s): s = s.reshape(-1, 1); return np.diagflat(s) - np.dot(s, s.T)
def logSoftmax(x): s = np.exp(x-x.max()); return s - np.log(s.sum(axis=-1, keepdims=True))
def logSoftmaxP(x): ...
# def softmaxCE(y, pred): return -(np.sum(y * np.log(pred + 1e-12), axis=-1)).mean()
# def softmaxCEP(y, s): return (s-y)/y.shape[0]

def softmaxCE(x, y): 
    x = x-x.max(axis=-1, keepdims=True) # keepdims in the sum, can be set to true or false both will work
    return (np.log(np.exp(x).sum(axis=-1))-x[np.arange(y.shape[0]), y]).mean()
# def softmaxCE(x, y): return crossentropy(softmax(x), y) # not recommended


def backward_softmax_crossentropy(top_grad, inp_softmax, y):
    res = inp_softmax
    res[np.arange(res.shape[0]), y] -= 1
    return top_grad * res / inp_softmax.shape[0]
def crossentropy(x, y): return np.mean(-np.log(x[np.arange(x.shape[0]), y]))

# OOP ========================
class Layer(_Layer):
    def __init__(self): super().__init__()
    def acc(self, y, p): return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

# class MSE(Layer):
#     def forward(self, y, p): return mse(y, p)
#     def backward(self, y, p): return mseP(y,p)


# class MSE(Layer):
#   def forward(self, y, p):
#     self.y, self.p = y, p
#     return mse(y, p)

#   def backward(self, TopGrad):
#     dy = TopGrad * mseP(self.y, self.p)
#     dp = TopGrad * -mseP(self.y, self.p)
#     return dy, dp


class MSE(Layer):
    def __init__(self, inShape=None): super().__init__()
    def forward(self, z, truth):
        self.pred, self.truth = z, truth
        return mse(self.truth, self.pred)
    def backward(self, top_grad=1.0):
        return mseP(self.truth, self.pred)

class CrossEntropy(Layer):
    def forward(self, y, p): return ce(y, p)
    def backward(self, y, p): return ceP(y, p)
class BCrossEntropy(Layer):
    def forward(self, y, p): return bce(y, p)
    def backward(self, y, p): return bceP(y, p)
class SoftmaxCE(Layer):
    def forward(self, y, p): return softmaxCE(y, p)
    def backward(self, y, p): return bceP(y, p)


class SoftmaxCELayer(Layer):
    """Calculates the softmax-crossentropy loss of the given input logits wrt some truth value."""
    def __init__(self, inShape=None):
        super.__init__(self)
        self.layers_name = self.__class__.__name__
        if inShape: self.inShape = inShape if isinstance(inShape,tuple) else (inShape,)

    def forward(self, z, truth):
        """
        :param z: The logits
        :param truth: The indices of the correct classification
        :return: The calculated loss
        """
        self.truth = truth
        self.output, self.cache = softmax_crossentropy(z, self.truth)
        return self.output

    def backward(self, top_grad=1.0):
        self.bottom_grad = backward_softmax_crossentropy(top_grad, self.cache, self.truth)
        return self.bottom_grad



