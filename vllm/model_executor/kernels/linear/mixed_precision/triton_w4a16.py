# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based W4A16 GEMM kernel for ROCm MI300.

Implements fused int4-weight dequantization + fp16 GEMM in a single kernel,
using GPTQ sequential packing (8 int4 values per int32, shifts [0,4,...,28]).
Plugs into the MPLinearKernel selection system and is preferred over
MarlinLinearKernel/ExllamaLinearKernel on ROCm.

Weight layout expected by this kernel (post-process_weights_after_loading):
  qweight: [K//2, N]  uint8  — rows=K//2 (input, uint8 packed), cols=N 
  scales:  [K//G, N]  fp16/bf16
  qzeros:  [K//G, N//8]  int32  (optional; None for symmetric uint4b8)

Checkpoint layout from compressed_tensors_wNa16 create_weights:
  weight_packed:     [N, K//8]  int32  (output_dim=0, input_dim=1, packed_dim=1)
  weight_scale:      [N, K//G]  fp16   (output_dim=0, input_dim=1)
  weight_zero_point: [N//8, K//G]  int32 (output_dim=0, packed_dim=0)
"""

import torch
from torch.library import register_fake
from typing import Optional

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

TRITON_W4A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
TRITON_W4A16_SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
    scalar_types.uint4,  # asymmetric with explicit zeros
]

configs = [
    triton.Config({'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'SPLIT_K': sk}, num_stages=s, num_warps=w)
        for m in [16, 32, 64, 128]
        for n in [64, 128, 256, 512]
        for k in [32, 64, 128]
        for sk in range(1, 33)
        for w,s in [(8,1),(4,1),(4,2)]
]

def prune_invalid_configs(configs, named_args, **kwargs):
    M = kwargs ['M_BUCKET']
    N = named_args ['N']
    K = named_args ['K']
    gs = kwargs ['group_size']
    pruned_configs = []
    for config in configs:
        BLOCK_M = config.kwargs['BLOCK_M']
        BLOCK_K = config.kwargs['BLOCK_K']
                  
        if M <= 128 and BLOCK_M != max(16, triton.next_power_of_2(M)):
            continue  

        if M > 128 and BLOCK_M < 128:
            continue
            
        if K % BLOCK_K != 0 or gs < BLOCK_K or gs % BLOCK_K != 0 :
            continue 
            
        BLOCK_N = config.kwargs['BLOCK_N']
        split_k = config.kwargs['SPLIT_K']
        n_warps = config.num_warps
        
        active_cu = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) 
        
        if active_cu >= 480:
            candidate = 1
        elif active_cu <= 3:
            candidate = 32  
        else:
            best_util = 100   
            for i in range(1, 9):
                c = i * 120 // active_cu
                if c < 1 or c > 32:
                    continue
                util = triton.cdiv(active_cu * c, 120) / c
                if util < best_util:
                    candidate = c
                    best_util = util
                    
        if split_k != candidate:
            continue
        
            
        if BLOCK_M * BLOCK_N > 128*128 or BLOCK_N * BLOCK_K > 128*128:
            continue
            
        if ( BLOCK_M * BLOCK_N > 64*128 or BLOCK_N * BLOCK_K > 64*128 ) and n_warps < 8:
            continue
            
        if ( BLOCK_M * BLOCK_N < 64*128 and BLOCK_N * BLOCK_K < 64*128 ) and n_warps > 4:
            continue
         
        pruned_configs.append(config)  
    return pruned_configs
    
    
@triton.autotune(
    configs=configs, 
    key=['M_BUCKET', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': prune_invalid_configs 
    },
    reset_to_zero=['c_ptr'],
)
@triton.jit
def triton_w4a16_gemm_kernel(
    # Pointers
    a_ptr,       # [M, K]  fp16/bf16 activations
    b_ptr,       # [K//2, N] uint8 packed 4-bit weights
    scales_ptr,  # [K//G, N]  fp16/bf16 scales
    zeros_ptr,   # [K//G, N//8] int32 packed zeros
    c_ptr,       # [M, N]  fp16/bf16 output
    # Dimensions
    M, 
    N, 
    K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn, 
    stride_cm, stride_cn,
    # Quantization parameters
    group_size: tl.constexpr,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    M_BUCKET: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    num_k_blocks_per_pid = tl.cdiv(num_k_blocks, SPLIT_K)
    k_start_idx = pid_k * num_k_blocks_per_pid
    k_end_idx = min(k_start_idx + num_k_blocks_per_pid, num_k_blocks)

    
    # Setup 1D offsets for scale and zero vector loading
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    scale_mask = offs_sn < N
    
    offs_zn = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    z_mask = offs_zn < N // 2

    # Accumulator in FP32
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K-loop using block pointer advanced scaling
    for k_idx in range(k_start_idx, k_end_idx):
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, k_idx * BLOCK_K),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0)
        )
        
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(K // 2, N),
            strides=(stride_bk, 1), 
            offsets=(k_idx * (BLOCK_K // 2), pid_n * (BLOCK_N)),
            block_shape=(BLOCK_K // 2, BLOCK_N),
            order=(1, 0)
        )
        
        # ---- Load A ----
        a = tl.load(a_block_ptr, boundary_check=(0, 1,))

        # ---- Load B (uint8 mode, half the data footprint in registers) ----
        # b_packed_u8 shape: [BLOCK_K//2, BLOCK_N ]
        b_packed_u8 = tl.load(b_block_ptr, boundary_check=(1,))

        # ---- Fast Unpacking via single int8 Interleave ----
        # Extract low 4-bit and high 4-bit nibbles separately
        # Cast to int8 because later it will be substracted by zero points
        # Merge back to [BLOCK_K, BLOCK_N] using join, trans and reshape
        b_low = (b_packed_u8 & 0x0F).to(tl.int8)
        b_high = ((b_packed_u8 >> 4) & 0x0F).to(tl.int8)  

        b = tl.join(b_low, b_high)
        b = tl.trans(b, (0, 2, 1))
        b = tl.reshape(b, (BLOCK_K, BLOCK_N))


        
        # ---- Compute scale/zero group row index ----
        g_idx = (k_idx * BLOCK_K) // group_size

        # ---- Load Scales ----
        # Others set to 0 to avoid overflowed N
        scale_offset = g_idx * N + offs_sn
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=0.0)

        # ---- Load / Compute ZP (Optimized int8 handling) ----
        if HAS_ZP:
            zeros_ptr_u8 = zeros_ptr.to(tl.pointer_type(tl.uint8))
            
            z_offset = g_idx * (N // 2) + offs_zn
            z_packed_u8 = tl.load(zeros_ptr_u8 + z_offset, mask=z_mask, other=0)
            
            z_low = (z_packed_u8 & 0x0F).to(tl.int8)
            z_high = ((z_packed_u8 >> 4) & 0x0F).to(tl.int8)
            z = tl.interleave(z_low, z_high)
        else:
            z = ZP_BIAS # Zero extra register cost as it handles via scalar broadcast

        # ---- Dequantize ----
        # Keep calculations in int8 up to subtraction, then cast to activation type
        z_val = z[None, :] if HAS_ZP else z
        b_fp = (b - z_val).to(a.dtype) * scales[None, :]

        
        # ---- GEMM Tensor Core Dot ----
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)


    # ---- Store Output C ----
    c = accumulator.to(c_ptr.dtype.element_ty)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)

@torch.library.custom_op("vllm::triton_w4a16_gemm", mutates_args=())           
def triton_w4a16_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [K, N//8] int32
    scales: torch.Tensor,  # [K//G, N] fp16/bf16
    qzeros: torch.Tensor | None,  # [K//G, N//8] int32, or None
    group_size: int,
    zp_bias: int = 8,  # bias for uint4b8 when qzeros is None
) -> torch.Tensor:
    """
    Fused W4A16 GEMM using GPTQ-packed int4 weights.

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [K, N//8], int32 (GPTQ sequential).
        scales:     Per-group scales [K//G, N], same dtype as a.
        qzeros:     Per-group packed zero points [K//G, N//8] int32, or None
                    for symmetric quantization (uses zp_bias instead).
        group_size: Quantization group size (resolved from -1 to K by caller).
        zp_bias:    Constant zero used when qzeros is None (default 8 for uint4b8).

    Returns:
        Output matrix [M, N], same dtype as a.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[1]

    assert b_q.shape == (K // 2, N), (
        f"b_q shape mismatch: {b_q.shape} vs ({K // 2}, {N})"
    )
    assert scales.shape == (K // group_size, N), (
        f"scales shape mismatch: {scales.shape} vs ({K // group_size}, {N})"
    )

    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // 8), (
            f"qzeros shape mismatch: {qzeros.shape}"
        )

    has_zp = qzeros is not None
    # Provide a dummy pointer when HAS_ZP=False (Triton requires a valid ptr)
    zeros_ptr = qzeros if has_zp else b_q
    
    m_bucket = triton.next_power_of_2(M)
    m_bucket = 16 if m_bucket < 16 else m_bucket
    m_bucket = 1024 if m_bucket > 1024 else m_bucket
        
    
    c = torch.zeros((M, N), dtype=a.dtype, device=a.device)
    
    def grid_fn(meta):
        M, BLOCK_M = meta['M'], meta['BLOCK_M']
        N, BLOCK_N = meta['N'], meta['BLOCK_N']
        split_k = meta['SPLIT_K']
        return (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)
    
    triton_w4a16_gemm_kernel[grid_fn](
        a,
        b_q,
        scales,
        zeros_ptr,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        c.stride(0),
        c.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        #BLOCK_M=BLOCK_M,
        #BLOCK_N=BLOCK_N,
        #BLOCK_K=BLOCK_K,
        M_BUCKET=m_bucket,
        #num_warps=num_warps,
        #num_stages=num_stages,
    )
    return c

@register_fake("vllm::triton_w4a16_gemm")
def triton_w4a16_gemm_fake(
    a: torch.Tensor,
    b_q: torch.Tensor,
    scales: torch.Tensor,
    qzeros: Optional[torch.Tensor],
    group_size: int,
    zp_bias: int = 8,
) -> torch.Tensor:
    M, K = a.shape
    # According to assert: assert b_q.shape == (K // 2, N)
    N = b_q.shape[1]  
    return torch.empty((M, N), dtype=a.dtype, device=b_q.device)  
    
class TritonW4A16LinearKernel(MPLinearKernel):
    """
    Triton-based W4A16 GEMM kernel for ROCm (MI300 and newer).

    Supports GPTQ-format int4 weights (uint4b8 symmetric, uint4 asymmetric)
    with grouped quantization. Weight tensors are transposed from the
    compressed-tensors checkpoint layout to the kernel's [K, N//8] layout.
    """

    SUPPORTED_QUANT_TYPES = TRITON_W4A16_SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        # Triton handles capability checks itself
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not (current_platform.is_rocm() or current_platform.is_cuda()):
            return False, "TritonW4A16LinearKernel requires CUDA or ROCm"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        N = c.partition_weight_shape[1]
        if N % 8 != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by 8 "
                "(8 int4 values packed per int32)",
            )

        if c.has_g_idx:
            return (
                False,
                "Activation reordering (g_idx) is not supported by "
                "TritonW4A16LinearKernel",
            )

        gs = c.group_size
        if (
            gs not in TRITON_W4A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {TRITON_W4A16_SUPPORTED_GROUP_SIZES} "
                f"or full K ({c.full_weight_shape[0]})",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return (False, f"Input features {K} not divisible by group size {eff_gs}")
        if K % 32 != 0:
            return (
                False,
                f"{K} cannot be divided by the smallest group size",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Convert compressed-tensors checkpoint layout to kernel layout.

        Checkpoint (from compressed_tensors_wNa16.create_weights):
          weight_packed:     [N, K//8]  int32   input_dim=1, output_dim=0, packed_dim=1
          weight_scale:      [N, K//G]  fp16    input_dim=1, output_dim=0
          weight_zero_point: [N//8, K//G] int32  output_dim=0, packed_dim=0

        Kernel needs:
          qweight: [K//2, N]  uint8   (transpose weight_packed)
          scales:  [K//G, N]  fp16    (transpose weight_scale)
          qzeros:  [K//G, N//8] int32 (transpose weight_zero_point)
        """

        # ---- Transform qweight: [N, K//8] → [K//8, N] → back to [K, N//8] ----
        # permute_param_layout_(x, input_dim=0, output_dim=1) rearranges so that
        # the input(K) dimension is at physical dim 0 and output(N) at dim 1.
        # Checkpoint has input_dim=1, output_dim=0, packed_dim=1 (K is packed).
        # After permute we get [K//8, N] (K packed at dim 0, N at dim 1).
        # The kernel wants [K//2, N] (K at dim 0, N packed at dim 1), so we

        #
        # Simple approach: unpack → repack as [K//2, N].
        # This is done CPU-side at load time (one-time cost).
        
        def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            w = x.data  # [N, K//8] int32
            N_dim, K8 = w.shape
            
            # 1. Unpack int32，we got [N, K8, 8]
            shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
            U = ((w.unsqueeze(-1) >> shifts) & 0xF) 
            
            # 2. Slice the last dim：U_even: (0,2,4,6) and U_odd (1,3,5,7)
            U_even = U[..., 0::2]  # [N, K8, 4]
            U_odd = U[..., 1::2]   # [N, K8, 4]
            
            # 3. Merge by or: [N, K8, 4]
            bytes_packed = (U_odd << 4) | U_even 
            
            # 4. Flatten K dim and then transpose to [K//2, N]
            x.data = bytes_packed.to(torch.uint8).reshape(N_dim, K8 * 4).t().contiguous()
            return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            # x.data is [N, K//G] fp16, bring to [K//G, N]
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                # Checkpoint: [N//8, K//G] int32 (N packed at dim 0, K//G at dim 1)
                # Kernel needs: [K//G, N//8] — just transpose
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        K = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else K

        # For symmetric types (uint4b8), use the scalar bias; no zeros tensor
        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0
        qzeros = None if c.weight_type.has_bias() else w_zp
        output = triton_w4a16_gemm(
            a=x_2d,
            b_q=w_q,
            scales=w_s,
            qzeros=qzeros,
            group_size=group_size,
            zp_bias=zp_bias,
        )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
        
