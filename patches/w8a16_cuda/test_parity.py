"""Parity unit test: native CUDA W8A16 vs Triton reference."""
import torch
import os, sys

# Add the extension directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def triton_reference(x, w8, scales, biases, N, K, GS=64):
    """Reference W8A16 computation using pure PyTorch (no Triton)."""
    M = x.shape[0]
    KG = K // GS
    # Dequantize weights: w8 (N, K) uint8 -> bf16
    w_bf16 = w8.to(torch.bfloat16)
    # Scales and biases: (N, KG) bf16
    # Compute: out[m, n] = sum_g (sum_k x[m, g*GS+k] * w[n, g*GS+k]) * S[n, g] + sum_g (sum_k x[m, g*GS+k]) * B[n, g]
    acc = torch.zeros(M, N, dtype=torch.float32, device=x.device)
    for g in range(KG):
        x_g = x[:, g*GS:(g+1)*GS]  # (M, GS)
        w_g = w_bf16[:, g*GS:(g+1)*GS]  # (N, GS)
        dot = x_g.to(torch.float32) @ w_g.to(torch.float32).T  # (M, N)
        sums = x_g.to(torch.float32).sum(dim=1)  # (M,)
        s = scales[:, g].to(torch.float32)  # (N,)
        b = biases[:, g].to(torch.float32)  # (N,)
        acc += dot * s.unsqueeze(0) + sums.unsqueeze(1) * b.unsqueeze(0)
    return acc.to(torch.bfloat16)

def test_parity():
    torch.manual_seed(42)
    device = "cuda"
    
    # Test shapes: (M, K, N)
    shapes = [
        (1, 128, 128),      # decode single
        (4, 256, 256),      # decode batch
        (16, 128, 128),     # small prefill
        (32, 256, 256),     # medium
        (64, 128, 128),     # larger
        (1, 512, 512),      # wide
        (8, 1024, 1024),    # large K
    ]
    
    # Import the native CUDA extension
    import w8a16_cuda_qmm
    
    all_pass = True
    for M, K, N in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        w8 = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=device)
        KG = K // 64
        scales = torch.randn(N, KG, dtype=torch.bfloat16, device=device) * 0.01
        biases = torch.randn(N, KG, dtype=torch.bfloat16, device=device) * 0.01
        
        # Native CUDA
        out_cuda = w8a16_cuda_qmm.forward(x, w8, scales, biases, False)
        
        # Reference
        out_ref = triton_reference(x, w8, scales, biases, N, K)
        
        # Compare
        max_diff = (out_cuda.float() - out_ref.float()).abs().max().item()
        mean_diff = (out_cuda.float() - out_ref.float()).abs().mean().item()
        rel_diff = max_diff / max(out_ref.float().abs().max().item(), 1e-8)
        
        # For bf16, atol=1e-2 and rtol=1e-2 should be fine
        passed = torch.allclose(out_cuda.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] M={M:3d} K={K:4d} N={N:4d}  max_diff={max_diff:.4f} mean={mean_diff:.4f} rel={rel_diff:.4f}")
    
    # Also test f32 output
    print("\n  f32 output test:")
    M, K, N = 4, 128, 128
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w8 = torch.randint(0, 256, (N, K), dtype=torch.uint8, device=device)
    KG = K // 64
    scales = torch.randn(N, KG, dtype=torch.bfloat16, device=device) * 0.01
    biases = torch.randn(N, KG, dtype=torch.bfloat16, device=device) * 0.01
    
    out_cuda = w8a16_cuda_qmm.forward(x, w8, scales, biases, True)
    out_ref = triton_reference(x, w8, scales, biases, N, K).float()
    
    max_diff = (out_cuda - out_ref).abs().max().item()
    passed = torch.allclose(out_cuda, out_ref, atol=1e-3, rtol=1e-2)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] M={M} K={K} N={N} f32  max_diff={max_diff:.6f}")
    
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass

if __name__ == "__main__":
    test_parity()
