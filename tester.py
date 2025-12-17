
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Import from convo.py
sys.path.append(os.getcwd())
from convo import _corr2d, corr2d_backward, Conv2d

def test_conv_forward_backward(N, C, H, W, OutC, KH, KW, S, P, D):
    print(f"Testing: N={N}, C={C}, H={H}, W={W}, OutC={OutC}, K=({KH},{KW}), S={S}, P={P}, D={D}")
    
    # Setup PyTorch
    x_pt = torch.randn(N, C, H, W, requires_grad=True, dtype=torch.float64)
    conv_pt = nn.Conv2d(C, OutC, (KH, KW), stride=S, padding=P, dilation=D, bias=False).double()
    
    # Initialize weights
    w_np = conv_pt.weight.data.numpy().astype(np.float64)
    
    # Forward PyTorch
    out_pt = conv_pt(x_pt)
    
    # Top Grad
    top_grad_np = np.random.randn(*out_pt.shape).astype(np.float64)
    top_grad_pt = torch.from_numpy(top_grad_np)
    
    # Backward PyTorch
    out_pt.backward(top_grad_pt)
    
    dx_pt = x_pt.grad.numpy()
    dw_pt = conv_pt.weight.grad.numpy()
    
    # Setup NumPy
    x_np = x_pt.detach().numpy().astype(np.float64)
    
    # Forward NumPy
    # Note: Conv2d class uses _corr2d internally
    conv_np = Conv2d(C, OutC, (KH, KW), S=S, P=P, D=D, b=False, dtype=np.float64)
    conv_np.w = w_np
    
    out_np = conv_np.forward(x_np)
    
    # Backward NumPy
    dx_np = conv_np.backward(top_grad_np)
    dw_np = conv_np.grads[0]
    
    # Compare
    fwd_diff = np.max(np.abs(out_pt.detach().numpy() - out_np))
    dw_diff = np.max(np.abs(dw_pt - dw_np))
    dx_diff = np.max(np.abs(dx_pt - dx_np))
    
    print(f"Forward Diff: {fwd_diff:.2e}")
    print(f"WGrad Diff:   {dw_diff:.2e}")
    print(f"XGrad Diff:   {dx_diff:.2e}")
    
    success = True
    if fwd_diff > 1e-10:
        print("FAILED: Forward pass")
        success = False
    if dw_diff > 1e-10:
        print("FAILED: WGrad")
        success = False
    if dx_diff > 1e-10:
        print("FAILED: XGrad")
        success = False
    
    if success:
        print("SUCCESS")
    print("-" * 30)
    return success

## Add more test cases in here to test our implementation of the convolution answer accordingly briefly (don't explain)
if __name__ == "__main__":
    # Test cases
    # tests = [
    #     # N, C, H, W, OutC, KH, KW, S, P, D
    #     (2, 3, 7, 7, 4, 3, 3, 1, 0, 1), # Simple
    #     (2, 3, 7, 7, 4, 3, 3, 2, 0, 1), # Stride
    #     (2, 3, 7, 7, 4, 3, 3, 1, 1, 1), # Padding
    #     (2, 3, 7, 7, 4, 3, 3, 1, 0, 2), # Dilation
    #     (2, 3, 15, 15, 4, 3, 3, 2, 1, 2), # All
    # ]

    # tests = [
    #     # N, C, H, W, OutC, KH, KW, S, P, D
    #     (2, 3, 7, 7, 4, 3, 3, 1, 0, 1), # Simple
    #     (2, 3, 7, 7, 4, 3, 3, 2, 0, 1), # Stride
    #     (2, 3, 7, 7, 4, 3, 3, 1, 1, 1), # Padding
    #     (2, 3, 7, 7, 4, 3, 3, 1, 0, 2), # Dilation
    #     (2, 3, 15, 15, 4, 3, 3, 2, 1, 2), # All
    #     (1, 1, 5, 5, 1, 2, 2, 1, 0, 1), # Small kernel
    #     (1, 1, 5, 5, 1, 5, 5, 1, 0, 1), # Full kernel size
    #     (3, 3, 10, 10, 8, 4, 4, 3, 2, 1), # Larger stride and padding
    #     (2, 3, 8, 8, 4, 3, 3, 1, 0, 3), # Larger dilation
    #     (1, 2, 6, 6, 3, 2, 2, 2, 1, 2), # Mixed dimensions
    # ]

    
    np.random.seed(42)  # For reproducibility

    tests = []
    for _ in range(10):
        N = np.random.randint(1, 5)   # batch
        C = np.random.randint(1, 6)   # channels
        H = W = np.random.randint(5, 16)  # height, width
        OutC = np.random.randint(1, 10)
        KH = KW = np.random.randint(2, 6)  # kernel size
        S = np.random.randint(1, 4)   # stride
        P = np.random.randint(0, 3)   # padding
        D = np.random.randint(1, 4)   # dilation
        tests.append((N, C, H, W, OutC, KH, KW, S, P, D))

    
    all_success = True
    for t in tests:
        if not test_conv_forward_backward(*t):
            all_success = False
            
    if all_success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)

