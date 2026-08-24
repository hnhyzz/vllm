# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class ExllamaLinearKernel(MPLinearKernel):
    # My interleaved MFMA kernel is 4-bit only, so restrict to the 4-bit types.
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4, scalar_types.uint4b8]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "Exllama is only supported on CUDA and ROCm",
            )

        if c.weight_type.size_bits != 4:
            return False, ("My Exllama kernel is 4-bit only (W4A16). "
                           "Switch to a uint4/uint4b8 quant.")

        # The interleaved B[K/32][N/32][64][4] layout requires K and N to be
        # multiples of 32 -- guarantee this so the kernel is never launched with
        # an invalid shape (there is no fallback).
        if c.partition_weight_shape[0] % 32 != 0:
            return False, ("Input features (K) must be a multiple of 32 for "
                           "the interleaved kernel")
        if c.partition_weight_shape[1] % 32 != 0:
            return False, ("Output features (N) must be a multiple of 32 for "
                           "the interleaved kernel")

        if c.has_g_idx and c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return (
                False,
                "Act reordering currently not supported by Exllama, "
                "when the input features are partitioned across "
                "devices",
            )

        if c.act_type != torch.float16:
            return False, "Exllama only supports float16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "Exllama, supported types are: "
                f"{cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                f"Group size ({c.group_size}) must be positive, "
                "Exllama does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.full_weight_shape[0]})",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        device = getattr(layer, self.w_q_name).device

        self.w_zp_name = "qzeros"
        if c.zero_points:
            # Asymmetric: real per-group/column zero points from the checkpoint.
            self._has_zp = True

            def transform_w_zp(x):
                assert isinstance(x, BasevLLMParameter)
                permute_param_layout_(x, input_dim=0, output_dim=1)
                x.data = x.data.contiguous()
                return self._repack_zeros(x.data)

            self._transform_param(layer, self.w_zp_name, transform_w_zp)
        else:
            # Symmetric: no zero-point tensor; signalled to the kernel via
            # use_v2_format (the kernel uses the implicit zero point, 8).
            self._has_zp = False
            setattr(layer, self.w_zp_name,
                    torch.nn.Parameter(
                        torch.empty((0,), dtype=torch.uint8, device=device),
                        requires_grad=False))

        if c.has_g_idx:
            def transform_w_g_idx(x):
                # Exllama wants the permutation array instead of the group
                # indices.
                return torch.argsort(x).to(torch.int)
            self._transform_param(layer, self.w_gidx_name,
                                  transform_w_g_idx)  # type: ignore
        else:
            self.w_gidx_name = "g_idx"
            setattr(layer, self.w_gidx_name,
                    torch.nn.Parameter(
                        torch.empty((0,), dtype=torch.int, device=device),
                        requires_grad=False))

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            assert self.w_gidx_name is not None
            g_idx = getattr(layer, self.w_gidx_name)

            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            ops.gptq_shuffle(x_cont, g_idx, c.weight_type.size_bits)
            # Repack into my interleaved B; vLLM stores it as w_q (a single
            # torch-owned copy, no duplicate, no per-call repack).
            return self._interleave(x_cont, g_idx)

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        # g_idx is only consumed during the one-time repack above (inside
        # _interleave / gptq_shuffle). The interleaved B already encodes the
        # group ordering, and my kernel never reads g_idx (my_gptq_gemm:
        # (void)b_g_idx), so drop the permutation to free HBM. Keep an empty
        # parameter so apply_weights' _get_weight_params still returns non-None.
        setattr(layer, self.w_gidx_name,
                torch.nn.Parameter(
                    torch.empty((0,), dtype=torch.int, device=device),
                    requires_grad=False))

    def _interleave(self, x_cont, g_idx):
        # x_cont: [K/8, N] int32 in the final exllama layout (post permute +
        # shuffle). Build B[K/32][N/32][64][4] int16 DIRECTLY by indexing x_cont,
        # so we never materialize the old [K, N]-shaped int64 intermediates
        # (wq/shifted/phys/nib) that spiked HBM during load. The logical row k
        # maps to physical row perm_inv[k] = argsort(g_idx)[k]; the nibble at a
        # physical row p sat in x_cont[p//8, n] at bit slot G[p%8].
        K8, N = x_cont.shape
        K = K8 * 8
        dev = x_cont.device

        # Value for physical slot s sits at nibble position G[s] (validated).
        G = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dev)

        # perm = argsort(g_idx) maps physical->true, so logical[k] = phys[perm_inv[k]].
        if g_idx.numel() > 0:
            perm_inv = torch.argsort(g_idx)
        else:
            perm_inv = torch.arange(K, device=dev)

        # Interleave: B[k32][n32][lane][j] bit s = nib[k32*32+s*8+(lane>>5)*4+j,
        # n32*32+(lane&31)].
        K32, N32 = K // 32, N // 32
        lane = torch.arange(64, device=dev)
        k32 = torch.arange(K32, device=dev)[:, None, None, None]
        n32 = torch.arange(N32, device=dev)[None, :, None, None]
        l4 = lane[None, None, :, None]
        j4 = torch.arange(4, device=dev)[None, None, None, :]
        ncol = n32 * 32 + (l4 & 31)             # column into x_cont [K32,N32,64,4]
        kbase = k32 * 32 + (l4 >> 5) * 4 + j4   # logical K row; + s*8 per nibble
        B = torch.zeros((K32, N32, 64, 4), dtype=torch.int16, device=dev)
        for s in range(4):
            pi = perm_inv[kbase + s * 8]                    # physical K row
            val = (x_cont[pi // 8, ncol] >> (4 * G[pi % 8])) & 0xF
            B |= (val.to(torch.int16) << (4 * s))
        # int16 holds the bit patterns; the kernel reads them as uint16. Return
        # the SAME [K/8, N] int32 shape as the original exllama weight so that
        # gptq_gemm's N derivation (b_q_weight.size(1)) stays correct; the raw
        # bytes are the interleaved B (kernel reinterprets as uint16).
        return B.view(torch.int32).reshape(K8, N).contiguous()

    def _repack_zeros(self, zp_data):
        # zp_data: [groups, N/8] uint32 (asymmetric zero points, packed 8 per
        # word along N) -> my 2-columns-per-byte uint8 [groups, N/2].
        groups, N8 = zp_data.shape
        N = N8 * 8
        dev = zp_data.device
        n = torch.arange(N, device=dev)
        z = (zp_data.to(torch.int64)[:, n // 8] >> (4 * (n % 8))) & 0xF
        return (z[:, 0::2] | (z[:, 1::2] << 4)).to(torch.uint8).contiguous()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)

        # use_v2_format doubles as the "has zero points" indicator for the
        # kernel: True -> asymmetric (zero-point tensor present); False ->
        # symmetric (no zero point; the kernel uses the implicit zero, 8).
        use_v2_format = self._has_zp

        # The split-K workspace is allocated inside gptq_gemm (C++) via
        # torch::stable at the CUDA-graph-captured address; the kernel discards
        # g_idx, so the Python side allocates no workspace and no static lives
        # in the kernel. split is also chosen by the C++ (single source of truth).
        assert w_g_idx is not None, "Group index is required by Exllama"
        output = ops.gptq_gemm(
            x_2d, w_q, w_zp, w_s, w_g_idx, True, use_v2_format,
            c.weight_type.size_bits
        )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)

    # -- NOTE: split-k is computed in the C++ layer (q_gemm.cu gptq_gemm), not
    #    here; Python no longer sizes a workspace tensor. ------------------------------
