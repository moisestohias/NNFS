## Functonal 
import numpy as np
from math import floor, ceil
as_strided = np.lib.stride_tricks.as_strided

def _pair(x): return (x, x) if isinstance(x, int) else x


## Convolution
def _pad(Z: np.ndarray, K: np.ndarray, mode: str="valid") -> np.ndarray:
    """ 
    Check arguments and pad for conv/corr 
    This is a function that checks the validity of inputs and applies padding to the input array `Z` based on the specified mode ("valid", "same", or "full") for convolution or correlation operations. It ensures the dimensions and sizes are compatible, then returns the padded input and the kernel.
    """

    modes = "full", "same", "valid" 
    if mode not in modes: raise ValueError(f"mode must be one of {modes}")
    if Z.ndim != K.ndim: raise ValueError("Z and K must have the same number of dimensions")
    if Z.size == 0 or K.size == 0: raise ValueError("Zero-size arrays not supported in convolutions.")
    ZN,ZC,ZH,ZW = Z.shape
    OutCh,KC,KH,KW = K.shape
    if ZC!=KC: raise ValueError(f"Kernel must have the same number channels as Input, got Z.shape:{Z.shape}, W.shape {K.shape}")
    if mode == 'valid' : padding = ((0,0),(0,0), (0,0), (0,0))
    elif mode == 'same':
        # OH = ZH-KH+1 -> ZH=OH+KH-1
        ## Warning: Pytorch pads symmetrically breaking the strict math definition, 
        ## If you want this behavior use these instead, otherwise use the above two lines
        # PadTop, PadBottom = floor((KH-1)/2), ceil((KH-1)/2)
        # PadLeft, PadRigh = floor((KW-1)/2), ceil((KW-1)/2)
        PadTop = PadBottom = KH // 2 
        PadLeft = PadRight = KW // 2
        padding = ((0,0),(0,0), (PadTop, PadBottom),(PadLeft, PadRigh))
    elif mode == 'full':
        PadTop, PadBottom = KH-1, KH-1 # full-convolution aligns kernel edge with the firs pixel of input, thus K-1
        PadLeft, PadRigh = KW-1, KW-1
        padding = ((0,0),(0,0), (PadTop, PadBottom),(PadLeft, PadRigh))
    if np.array(padding).any(): Z = np.pad(Z, padding, mode='constant') 
    return Z, K 

def calculateConvOutShape(inShape, K, S=(1,1), P=(0,0), D=(1,1)):
    """Compute output H, W."""
    return (int(np.floor((inShape[0] + 2*P[0] - D[0]*(K[0]-1) - 1)/S[0] + 1)),
            int(np.floor((inShape[1] + 2*P[1] - D[1]*(K[1]-1) - 1)/S[1] + 1)))

def _corr2d(Z, W, S=(1,1), P=(0,0), D=(1,1)):
    """Perform 2D cross-correlation using as_strided and GEMM."""
    # Apply padding
    Z = np.pad(Z, ((0,0), (0,0), (P[0],P[0]), (P[1],P[1]))) if sum(P) > 0 else Z
    
    N, C, Hin, Win = Z.shape
    Cout, Cin, KH, KW = W.shape
    Hout, Wout = calculateConvOutShape((Hin, Win), (KH, KW), S, (0,0), D)
    
    # NCHW -> NHWC, W: OIHW -> HWIO
    Z_nhwc = Z.transpose(0, 2, 3, 1) 
    W_hwio = W.transpose(2, 3, 1, 0)
    
    # Create strided view: (N, Hout, Wout, KH, KW, C)
    shape = (N, Hout, Wout, KH, KW, C)
    s = Z_nhwc.strides
    strides = (s[0], s[1]*S[0], s[2]*S[1], s[1]*D[0], s[2]*D[1], s[3])
    
    Z_view = as_strided(Z_nhwc, shape=shape, strides=strides)
    
    # GEMM: (N*Hout*Wout, KH*KW*C) @ (KH*KW*C, Cout) -> (N*Hout*Wout, Cout)
    # Note: reshape(-1, Cout) works because W_hwio is (KH, KW, Cin, Cout) -> flattens correctly
    out = Z_view.reshape(-1, KH*KW*C) @ W_hwio.reshape(-1, Cout)
    
    return out.reshape(N, Hout, Wout, Cout).transpose(0, 3, 1, 2)

def conv2d(Z, W, mode="valid"):
    Z_pad, W = _pad(Z, W, mode)
    return _corr2d(Z_pad, W[:, :, ::-1, ::-1], S=(1,1), P=(0,0), D=(1,1))

def corr2d_backward(Z, W, TopGrad, mode="valid"):
    """Compute gradients wrt weights (WGrad) and input (ZGrad)."""
    # WGrad: Convolve Input (as N channels) with TopGrad (as N kernels)
    # Z(N, C, H, W) -> (C, N, H, W)
    # TopGrad(N, OutC, H', W') -> (OutC, N, H', W')
    WGrad = _corr2d(Z.transpose(1, 0, 2, 3), TopGrad.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3)

    # ZGrad: Full convolution of TopGrad with spatially flipped AND channel-transposed W
    # TopGrad has 'OutC' channels. W has 'OutC' filters.
    # To convolve TopGrad, we need filters that accept 'OutC' channels.
    # W is (OutC, InC, H, W) -> Transpose to (InC, OutC, H, W)
    W_T = W.transpose(1, 0, 2, 3) 
    
    # Flip spatially for convolution
    kh, kw = W.shape[2], W.shape[3]
    ZGrad = _corr2d(TopGrad, W_T[:, :, ::-1, ::-1], P=(kh-1, kw-1))
    
    return WGrad, ZGrad

# ---
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
    dW, dZ = corr2d_backward(self.z, self.w, TopGrad, mode="valid")
    self.grads[0] = dW
    if self.b is not None:
      self.grads[1] = np.sum(TopGrad, axis=(0, 2, 3))
    return dZ

