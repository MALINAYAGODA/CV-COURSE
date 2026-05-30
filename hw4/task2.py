import torch
import triton
import triton.language as tl
from triton.testing import perf_report, Benchmark, do_bench


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128}, num_warps=2),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_N": 2048}, num_warps=16),
    ],
    key=["hidden_size"]
)
@triton.jit
def layernorm_forward_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    saved_mean_ptr,
    saved_rstd_ptr,
    hidden_size: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    col_indices = tl.arange(0, BLOCK_N)
    valid_mask = col_indices < hidden_size
    
    input_data = tl.load(
        input_ptr + row_idx * hidden_size + col_indices,
        mask=valid_mask,
        other=0.0
    ).to(tl.float32)
    
    row_mean = tl.sum(input_data, axis=0) / hidden_size
    centered_data = input_data - row_mean
    variance = tl.sum(centered_data * centered_data, axis=0) / hidden_size
    inv_std = tl.rsqrt(variance + epsilon)
    normalized_data = centered_data * inv_std
    gamma = tl.load(weight_ptr + col_indices, mask=valid_mask, other=0.0).to(tl.float32)
    beta = tl.load(bias_ptr + col_indices, mask=valid_mask, other=0.0).to(tl.float32)
    output_data = normalized_data * gamma + beta
    
    tl.store(output_ptr + row_idx * hidden_size + col_indices, output_data, mask=valid_mask)
    tl.store(saved_mean_ptr + row_idx, row_mean)
    tl.store(saved_rstd_ptr + row_idx, inv_std)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128}, num_warps=2),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
    ],
    key=["hidden_size"]
)
@triton.jit
def layernorm_backward_kernel(
    grad_output_ptr,
    input_ptr,
    weight_ptr,
    saved_mean_ptr,
    saved_rstd_ptr,
    grad_input_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    hidden_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    col_indices = tl.arange(0, BLOCK_N)
    valid_mask = col_indices < hidden_size
    
    grad_out = tl.load(
        grad_output_ptr + row_idx * hidden_size + col_indices,
        mask=valid_mask,
        other=0.0
    ).to(tl.float32)
    
    input_data = tl.load(
        input_ptr + row_idx * hidden_size + col_indices,
        mask=valid_mask,
        other=0.0
    ).to(tl.float32)
    
    gamma = tl.load(weight_ptr + col_indices, mask=valid_mask, other=0.0).to(tl.float32)
    
    row_mean = tl.load(saved_mean_ptr + row_idx)
    inv_std = tl.load(saved_rstd_ptr + row_idx)
    
    normalized_data = (input_data - row_mean) * inv_std
    
    grad_normalized = grad_out * gamma
    
    sum_grad_normalized = tl.sum(grad_normalized, axis=0)
    sum_grad_normalized_times_normalized = tl.sum(grad_normalized * normalized_data, axis=0)
    
    grad_input = (
        grad_normalized 
        - sum_grad_normalized / hidden_size 
        - normalized_data * sum_grad_normalized_times_normalized / hidden_size
    ) * inv_std
    
    tl.store(grad_input_ptr + row_idx * hidden_size + col_indices, grad_input, mask=valid_mask)
    
    grad_gamma = grad_out * normalized_data
    tl.atomic_add(grad_weight_ptr + col_indices, grad_gamma, mask=valid_mask)
    tl.atomic_add(grad_bias_ptr + col_indices, grad_out, mask=valid_mask)


class LayerNormTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, bias, eps=1e-5):
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        
        batch_size, hidden_size = input_tensor.shape
        
        output = torch.empty_like(input_tensor)
        saved_mean = torch.empty((batch_size,), device=input_tensor.device, dtype=torch.float32)
        saved_rstd = torch.empty((batch_size,), device=input_tensor.device, dtype=torch.float32)
        
        grid = (batch_size,)
        layernorm_forward_kernel[grid](
            input_tensor,
            weight,
            bias,
            output,
            saved_mean,
            saved_rstd,
            hidden_size=hidden_size,
            epsilon=eps,
        )
        
        ctx.save_for_backward(input_tensor, weight, bias, saved_mean, saved_rstd)
        ctx.hidden_size = hidden_size
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight, bias, saved_mean, saved_rstd = ctx.saved_tensors
        batch_size = input_tensor.shape[0]
        hidden_size = ctx.hidden_size
        
        grad_input = torch.empty_like(input_tensor)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_bias = torch.zeros_like(bias, dtype=torch.float32)
        
        grid = (batch_size,)
        layernorm_backward_kernel[grid](
            grad_output,
            input_tensor,
            weight,
            saved_mean,
            saved_rstd,
            grad_input,
            grad_weight,
            grad_bias,
            hidden_size=hidden_size,
        )
        
        return grad_input, grad_weight, grad_bias, None


def layernorm_triton(input_tensor, weight, bias, eps=1e-5):
    return LayerNormTritonFunction.apply(input_tensor, weight, bias, eps)


def layernorm_reference(input_tensor, weight, bias, eps=1e-5):
    row_mean = input_tensor.mean(dim=-1, keepdim=True)
    row_var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
    inv_std = torch.rsqrt(row_var + eps)
    normalized = (input_tensor - row_mean) * inv_std
    return normalized * weight + bias


def test_correctness():
    print("=" * 80)
    print("TESTING CORRECTNESS")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    batch_size, hidden_size = 16, 1024
    eps = 1e-5
    
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(hidden_size, device="cuda", dtype=torch.float32, requires_grad=True)
    
    output_triton = layernorm_triton(x, weight, bias, eps)
    
    output_reference = layernorm_reference(x, weight, bias, eps)
    
    print("\nForward pass check:")
    try:
        torch.testing.assert_close(output_triton, output_reference, atol=1e-2, rtol=1e-2)
        print("Forward pass is correct!")
    except AssertionError as e:
        print(f"Forward pass failed: {e}")
        return False
    
    grad_output = torch.randn_like(output_triton)
    
    output_triton.backward(grad_output, retain_graph=True)
    grad_x_triton = x.grad.clone()
    grad_w_triton = weight.grad.clone()
    grad_b_triton = bias.grad.clone()
    
    x.grad = None
    weight.grad = None
    bias.grad = None
    
    output_reference.backward(grad_output)
    grad_x_reference = x.grad.clone()
    grad_w_reference = weight.grad.clone()
    grad_b_reference = bias.grad.clone()
    
    print("\nBackward pass check:")
    try:
        torch.testing.assert_close(grad_x_triton, grad_x_reference, atol=1e-2, rtol=1e-2)
        print("grad_input is correct!")
    except AssertionError as e:
        print(f"grad_input failed: {e}")
        return False
    
    try:
        torch.testing.assert_close(grad_w_triton, grad_w_reference, atol=1e-2, rtol=1e-2)
        print("grad_weight is correct!")
    except AssertionError as e:
        print(f"grad_weight failed: {e}")
        return False
    
    try:
        torch.testing.assert_close(grad_b_triton, grad_b_reference, atol=1e-2, rtol=1e-2)
        print("grad_bias is correct!")
    except AssertionError as e:
        print(f"grad_bias failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80 + "\n")
    return True


def prepare_benchmark_inputs(num_elements, hidden_size=1024, device="cuda"):
    batch_size = max(1, num_elements // hidden_size)
    x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    bias = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    return x, weight, bias


@perf_report([
    Benchmark(
        x_names=["num_elements"],
        x_vals=[2**i for i in range(18, 25)],
        line_arg="implementation",
        line_vals=["triton", "torch", "torch_compile"],
        line_names=["Triton", "PyTorch", "PyTorch Compiled"],
        styles=[("blue", "-"), ("red", "--"), ("green", "-.")],
        ylabel="Latency (ms)",
        plot_name="layernorm_performance_comparison",
        args={},
    )
])
def benchmark_layernorm(num_elements, implementation):
    hidden_size = 1024
    x, weight, bias = prepare_benchmark_inputs(num_elements, hidden_size)
    
    implementations = {
        "triton": lambda: layernorm_triton(x, weight, bias),
        "torch": lambda: layernorm_reference(x, weight, bias),
        "torch_compile": torch.compile(lambda: layernorm_reference(x, weight, bias)),
    }
    
    fn = implementations[implementation]
    median_ms, min_ms, max_ms = do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    
    return median_ms, min_ms, max_ms


def run_simple_benchmark():
    print("=" * 80)
    print("SIMPLE PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    test_sizes = [
        (256, 1024),
        (1024, 1024),
        (4096, 1024),
    ]
    
    for batch_size, hidden_size in test_sizes:
        x = torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.float32)
        weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        bias = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        
        for _ in range(10):
            _ = layernorm_triton(x, weight, bias)
            _ = layernorm_reference(x, weight, bias)
        torch.cuda.synchronize()
        
        triton_time = do_bench(lambda: layernorm_triton(x, weight, bias))[0]
        torch_time = do_bench(lambda: layernorm_reference(x, weight, bias))[0]
        
        speedup = torch_time / triton_time
        
        print(f"\nShape: [{batch_size}, {hidden_size}]")
        print(f"  Triton:  {triton_time:.4f} ms")
        print(f"  PyTorch: {torch_time:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    if test_correctness():
        run_simple_benchmark()
        
        print("Running full benchmark (this may take a while)...")
        benchmark_layernorm.run(print_data=True, show_plots=True)
