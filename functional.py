"""
x : 
    + 1D: N,C,W 
    + 2D : N,C,H,W
"""

## Functional
import numpy as np
from math import floor, ceil
as_strided = np.lib.stride_tricks.as_strided

def _pair(x): return (x, x) if isinstance(x, int) else x

## Linear (aka Dense)
def affin_trans(Z, W, B=0): return Z.dot(W.T) + B # W: (outF,inF)
def affin_trans_backward(TopGrad, Z, W):
    BGrad = TopGrad.sum(axis=0)
    WGrad = TopGrad.T.dot(Z)
    Zgrad = TopGrad.dot(W)
    return Zgrad, WGrad, BGrad

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

## MaxPool
def get_windows(x, k, s, p, pad_val):
    x_pad = np.pad(x, ((0,0), (0,0), (p[0], p[0]), (p[1], p[1])), mode='constant', constant_values=pad_val)
    n, c, h, w = x.shape
    kh, kw = k
    sh, sw = s
    out_h = (x_pad.shape[2] - kh) // sh + 1
    out_w = (x_pad.shape[3] - kw) // sw + 1
    
    shape = (n, c, out_h, out_w, kh, kw)
    strides = (x_pad.strides[0], x_pad.strides[1], x_pad.strides[2]*sh, x_pad.strides[3]*sw, x_pad.strides[2], x_pad.strides[3])
    return as_strided(x_pad, shape=shape, strides=strides)

def max_pool2d(x, kernel_size, stride=None, padding=0, return_indices=False):
    k = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    s = (stride, stride) if isinstance(stride, int) else (stride if stride else k)
    p = (padding, padding) if isinstance(padding, int) else padding
    
    # 1. Windowing and Max
    # Use -inf for padding so it doesn't affect max
    windows = get_windows(x, k, s, p, pad_val=-np.inf)
    
    if not return_indices:
        return np.max(windows, axis=(-2, -1))
    
    # 2. Retrieve Indices (Argmax)
    # Flatten the window dimensions (KH, KW) -> (K*K)
    win_flat = windows.reshape(*windows.shape[:-2], -1)
    argmax = np.argmax(win_flat, axis=-1)
    out = np.take_along_axis(win_flat, argmax[..., None], axis=-1).squeeze(-1)
    
    # 3. Map to Global Indices
    # We map the local window argmax to the global input array index
    indices = np.arange(x.size).reshape(x.shape)
    # Pad indices with -1 (invalid) matching the data padding
    ind_pad = np.pad(indices, ((0,0), (0,0), (p[0], p[0]), (p[1], p[1])), mode='constant', constant_values=-1)
    # Window indices (no extra padding needed here as we manually padded)
    win_ind = get_windows(ind_pad, k, s, (0,0), pad_val=-1)
    
    win_ind_flat = win_ind.reshape(*win_ind.shape[:-2], -1)
    max_inds = np.take_along_axis(win_ind_flat, argmax[..., None], axis=-1).squeeze(-1)
    
    return out, (x.shape, max_inds)

def max_pool2d_backward(grad_output, cache):
    x_shape, max_inds = cache
    grad_in = np.zeros(np.prod(x_shape))
    
    # Flatten for scattering
    grad_out_flat = grad_output.flatten()
    inds_flat = max_inds.flatten()
    
    # Filter out padding indices (-1)
    mask = inds_flat != -1
    
    # Accumulate gradients (add.at handles overlapping windows/max values)
    np.add.at(grad_in, inds_flat[mask], grad_out_flat[mask])
    
    return grad_in.reshape(x_shape)


# --- pad

def pad2D(image, pad_shape):
    return np.pad(image, ((0, 0), (0, 0), (pad_shape[0], pad_shape[0]), (pad_shape[1], pad_shape[1])), mode='constant')


def backward_pad2D(top_grad, pad_shape):
    return top_grad[:, :, pad_shape[0]:-pad_shape[0], pad_shape[1]:-pad_shape[1]]

def crossentropy(x, y): return np.mean(-np.log(x[np.arange(x.shape[0]), y]))

def softmax(x):
    temp = np.exp(x - x.max(axis=1, keepdims=True))
    res = temp / temp.sum(axis=1, keepdims=True)
    return res

def flatten(x): return x.reshape((x.shape[0], -1)), x.shape
def backward_flatten(top_grad, original_shape): return top_grad.reshape(original_shape)


def backward_crossentropy(top_grad, x, y):
    res = np.zeros(x.shape, dtype=x.dtype)
    res[np.arange(x.shape[0]), y] = - np.reciprocal(x[np.arange(x.shape[0]), y]) / x.shape[0]
    return res * top_grad

def softmax_crossentropy(x, y): 
  s = softmax(x)
  return crossentropy(s, y), s
def backward_softmax_crossentropy(top_grad, inp_softmax, y):
    res = inp_softmax
    res[np.arange(res.shape[0]), y] -= 1
    return top_grad * res / inp_softmax.shape[0]