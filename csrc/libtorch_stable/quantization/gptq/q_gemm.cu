// w4a16_gemm.cu -- my-kernel-only W4A16 GEMM for ROCm (gfx908).
//
// Trimmed replacement for the vLLM GPTQ q_gemm.cu. Keeps ONLY:
//   * my interleaved MFMA 32x32x8 kernel (B[K/32][N/32][64][4] int16),
//   * the 4-bit `gptq_shuffle` weight-prep kernels (used by exllama.py before my
//     repack), and
//   * the torch entry points gptq_gemm / gptq_shuffle.
// The original exllama GEMM kernels, the reconstruct+cuBLAS fallback, and the
// 2/3/8-bit shuffle paths are removed. My kernel is 4-bit only; exllama.py
// guarantees 4-bit + K%32==0 + N%32==0, so there is no fallback.
//
//   * asymmetric: b_q_weight's raw bytes ARE my interleaved B (the tensor keeps
//     the [K/8][N] int32 shape); b_gptq_qzeros holds my 2-per-byte uint8 zeros.
//   * symmetric (no zero point): signalled via use_v2_format; kernel uses zp=8.
#include <cstdint>
#include <cstdio>

#include "../../torch_utils.h"
#include <torch/csrc/stable/ops.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "qdq_4.cuh"   // shuffle_4bit_8

namespace vllm {
namespace gptq {

#define THREADS_X 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#if defined(USE_ROCM) && defined(__HIP_PLATFORM_AMD__)

#ifndef W4A_CHECK
#define W4A_CHECK(x) do { hipError_t e_ = (x); (void)e_; assert(e_ == hipSuccess); } while (0)
#endif

// my_gptq_gemm is defined below, used by gptq_gemm.
using u16x4  = unsigned short __attribute__((vector_size(8)));
using f16x4  = _Float16      __attribute__((vector_size(8)));
using f32x16 = float         __attribute__((vector_size(64)));

template <int BM, int BN, int BK, int THREADS>
struct TileGeom {
    static constexpr int WAVES   = THREADS / 64;
    static constexpr int TS      = 32;
    static constexpr int KSIZE   = 8;
    static constexpr int TILES_M = BM / TS;
    static constexpr int TILES_N = BN / TS;
    static constexpr int TILES   = TILES_M * TILES_N;
    static constexpr int WPT     = TILES / WAVES;
    static constexpr int KSTEPS  = BK / KSIZE;
    static_assert(TILES % WAVES == 0, "tiles must divide evenly across waves");
};

__device__ __forceinline__ f16x4 dequant_interleaved(u16x4 V, int s,
                                                     unsigned char zp,
                                                     _Float16 scale) {
    const u16x4 mask = {0x000F, 0x000F, 0x000F, 0x000F};
    u16x4 nib = (V >> (4 * s)) & mask;
    f16x4 f = __builtin_convertvector(nib, f16x4);
    f -= (_Float16)zp;
    return f * scale;
}

// General ASYM + HOIST kernel (zero point read per (group, column) from the
// packed zeros array; scale hoisted when gs > BK). Symmetric models (uint4b8 /
// uint8b128) simply carry a constant zero (8 / 128) in the zeros array.
template <int BM, int BN, int BK, int THREADS, bool ASYM, bool HOIST>
__global__ __launch_bounds__(THREADS)
void w4a16_gemm_kernel(const __half* __restrict__ A,
                       const uint16_t* __restrict__ B,
                       const __half* __restrict__ scales,
                       const uint8_t* __restrict__ zeros,
                       float* __restrict__ D,
                       __half* __restrict__ C,
                       int M, int N, int K, int gs, int split) {
    using G = TileGeom<BM, BN, BK, THREADS>;
    static_assert(G::WAVES == 4 || G::WAVES == 8, "4 or 8 wavefronts only");
    static_assert(BK == 32, "one 32-K macro tile per chunk");

    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int nchunks = K / BK;
    const int nc = (nchunks + split - 1) / split;
    const int c0 = blockIdx.z * nc;
    const int nlocal = std::min(nc, nchunks - c0);
    const int k0 = c0 * BK;
    const bool block_full = (n0 + BN) <= N;

    __shared__ __half A_lds[2][BM * BK];

    u16x4 bw[G::WPT], bn[G::WPT];
    _Float16 sc[G::WPT];
    unsigned char zp[G::WPT];
    f32x16 acc[G::WPT];

#pragma unroll
    for (int t = 0; t < G::WPT; ++t) acc[t] = f32x16{};

    auto lane_col = [&](int t) -> int {
        const int ti = wave * G::WPT + t;
        const int tc = ti % G::TILES_N;
        return n0 + tc * G::TS + (lane & 31);
    };
    auto tile_mrow = [&](int t) -> int {
        const int ti = wave * G::WPT + t;
        return (ti / G::TILES_N) * G::TS;
    };
    auto load_a = [&](int chunk, int stage) {
        const int k = k0 + chunk * BK;
        constexpr int ELEM = BM * BK / THREADS;
        constexpr int PAIRS = ELEM / 2;
#pragma unroll
        for (int j = 0; j < PAIRS; ++j) {
            const int p = tid * PAIRS + j;
            const int r = p / (BK / 2);
            const int pp = p % (BK / 2);
            const int cc = 2 * pp;
            const int mrow = m0 + r;
            uint32_t packed = 0;
            if (mrow < M) __builtin_memcpy(&packed, A + (size_t)mrow * K + k + cc, 4);
            A_lds[stage][r * BK + cc] = *reinterpret_cast<const _Float16*>(&packed);
            A_lds[stage][r * BK + cc + 1] = *reinterpret_cast<const _Float16*>((const char*)&packed + 2);
        }
    };
    [[maybe_unused]] int last_g = -1;
    auto load_sz = [&](int chunk) {
        const int g = (k0 + chunk * BK) / gs;
        if constexpr (HOIST) { if (g == last_g) return; last_g = g; }
#pragma unroll
        for (int t = 0; t < G::WPT; ++t) {
            const int n = lane_col(t);
            const int nc = n < N ? n : N - 1;
            sc[t] = scales[(size_t)g * N + nc];
            // ASYM reads the per-column zero point; SYM (no zero point, signalled
            // via the use_v2_format flag) uses the implicit symmetric zero of a
            // 4-bit range, 8 (dequant = (nib - 8) * scale).
            zp[t] = ASYM ? (zeros[(size_t)g * (N / 2) + nc / 2] >>
                            ((nc & 1) << 2)) & 0xF
                         : 8;
        }
    };
    auto load_b_chunk = [&](u16x4* out, int chunk) {
        const int k32 = (k0 + chunk * BK) / 32;
#pragma unroll
        for (int t = 0; t < G::WPT; ++t) {
            const int n = lane_col(t);
            const int nc = n < N ? n : N - 1;
            const int n32 = nc / 32;
            out[t] = *reinterpret_cast<const u16x4*>(B + ((size_t)k32 * (N / 32) + n32) * 256 + lane * 4);
        }
    };
    auto load_af = [&](int chunk, int stage, int s, int t) -> f16x4 {
        const int row = tile_mrow(t) + (lane & 31);
        const int kk = s * G::KSIZE + (lane >> 5) * 4;
        return *reinterpret_cast<const f16x4*>(&A_lds[stage][row * BK + kk]);
    };
    auto compute_chunk = [&](int chunk, int stage) {
        load_sz(chunk);
#pragma unroll
        for (int s = 0; s < G::KSTEPS; ++s) {
#pragma unroll
            for (int t = 0; t < G::WPT; ++t) {
                f16x4 af = load_af(chunk, stage, s, t);
                f16x4 bf = dequant_interleaved(bw[t], s, zp[t], sc[t]);
                acc[t] = __builtin_amdgcn_mfma_f32_32x32x8f16(af, bf, acc[t], 0, 0, 0);
            }
        }
    };

    if (nlocal > 0) {
        load_a(0, 0);
        load_b_chunk(bw, 0);
        __syncthreads();
        for (int c = 0; c < nlocal; ++c) {
            if (c + 1 < nlocal) { load_a(c + 1, (c + 1) & 1); load_b_chunk(bn, c + 1); }
            compute_chunk(c, c & 1);
#pragma unroll
            for (int t = 0; t < G::WPT; ++t) bw[t] = bn[t];
            __syncthreads();
        }
    } else {
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < G::WPT; ++t) {
        const int n = lane_col(t);
        if (!(block_full || n < N)) continue;
        const int mbase = m0 + tile_mrow(t) + 4 * (lane >> 5);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int m = mbase + 8 * (i >> 2) + (i & 3);
            if (m >= M) continue;
            const float v = acc[t][i];
            if (split > 1) D[((size_t)blockIdx.z * M + m) * N + n] = v;
            else           C[(size_t)m * N + n] = __float2half(v);
        }
    }
}

__global__ void reduce_split(const float* __restrict__ D, __half* __restrict__ C,
                             int total, int split) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float s = 0.0f;
#pragma unroll 4
    for (int k = 0; k < split; ++k) s += D[(size_t)k * total + i];
    C[i] = __float2half(s);
}

struct MyCfg { int bm, bn, bk, threads; };
MyCfg pick_my_cfg(int M) {
    if (M <= 16)  return {32, 1024, 32, 512};
    if (M <= 32)  return {32, 512, 32, 512};
    if (M <= 224) return {32, 1024, 32, 512};
    return {64, 1024, 32, 512};
}
int choose_split(long blocks, int nchunks) {
    const int max_s = std::min(32, nchunks / 2);
    double best = 1e300; int best_s = 1;
    for (int s = 1; s <= max_s; ++s) {
        const int nc = (nchunks + s - 1) / s;
        if (nc < 2) continue;
        const double per_cu = (double)(blocks * s) / 120.0;
        const double slack = std::ceil(per_cu) - per_cu;
        const double cost = 100.0 * slack + (nc < 4 ? 50.0 : 0.0) + 0.1 * s;
        if (cost < best) { best = cost; best_s = s; }
    }
    return best_s;
}

// ---------------------------------------------------------------------------
void launch_my(const MyCfg& c, const __half* A, const uint16_t* B, const __half* scales,
               const uint8_t* zeros, float* D, __half* C,
               int M, int N, int K, int gs, int split,
               const dim3& grid, hipStream_t stream, bool has_zero_points) {
    const bool hoist = (gs > c.bk);
#define LAUNCH(BM, BN, TH, ASYM, HO) w4a16_gemm_kernel<BM, BN, 32, TH, ASYM, HO> \
            <<<grid, TH, 0, stream>>>(A, B, scales, zeros, D, C, M, N, K, gs, split)
#define DISPATCH(BM, BN, TH, HO) \
    if (c.bm == BM && c.bn == BN && c.threads == TH) { \
        if (has_zero_points) { if (hoist) { LAUNCH(BM, BN, TH, true, true);  return; } \
                                      else       { LAUNCH(BM, BN, TH, true, false); return; } } \
        else                  { if (hoist) { LAUNCH(BM, BN, TH, false, true);  return; } \
                                      else       { LAUNCH(BM, BN, TH, false, false); return; } } \
    }
    if (hoist) {
        DISPATCH(32, 512, 512, true)
        DISPATCH(32, 1024, 512, true)
        DISPATCH(64, 1024, 512, true)
    } else {
        DISPATCH(32, 512, 512, false)
        DISPATCH(32, 1024, 512, false)
        DISPATCH(64, 1024, 512, false)
    }
#undef DISPATCH
#undef LAUNCH
    fprintf(stderr, "unsupported my config\n"); assert(false);
}

// ---------------------------------------------------------------------------
// M==1 decode GEMV. Reads the SAME interleaved B[K/32][N/32][64][4] layout as
// the MFMA (no second weight copy). B-memory-bound, so WAVES split the block's
// K-slice in-block (LDS + barrier) so the grid split -- and thus the split-K
// reduce -- stays small. Each block covers WN 32-column N tiles; lanes pair up
// (L and L+32) as the two K-halves of a column, combined with one shfl.
template <int WAVES, int WN>
__global__ __launch_bounds__(64 * WAVES)
void w4a16_gemv_kernel(const __half* __restrict__ A, const uint16_t* __restrict__ B,
                       const __half* __restrict__ scales, const uint8_t* __restrict__ zeros,
                       float* __restrict__ D, __half* __restrict__ C,
                       int M, int N, int K, int gs, int split, int sym) {
    const int tid  = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int kh   = lane >> 5;
    const int nb   = blockIdx.x;
    const int N32  = N / 32;
    const int n32b = nb * WN;
    const int nchunks = K / 32;
    const int nc = (nchunks + split - 1) / split;
    const int c0 = blockIdx.z * nc;
    const int nlocal = std::min(nc, nchunks - c0);
    // -- In-block K split. `nlocal` (not `nc`) so the last grid block never
    //    runs past nchunks (guard against OOB when K isn't split evenly).
    const int wchunks = (nlocal + WAVES - 1) / WAVES;
    const int wc0 = wave * wchunks;
    const int wlocal = std::min(wchunks, nlocal - wc0);

    float acc[WN];
    #pragma unroll
    for (int t = 0; t < WN; ++t) acc[t] = 0.0f;

    if (wlocal > 0) {
        for (int c = wc0; c < wc0 + wlocal; ++c) {
            const int k32 = c0 + c;
            const int g = (k32 * 32) / gs;
            const int baseA = k32 * 32;
            f16x4 af[4];   // shared across all WN columns (A-reuse)
            #pragma unroll
            for (int s = 0; s < 4; ++s)
                af[s] = *reinterpret_cast<const f16x4*>(&A[baseA + 8 * s + 4 * kh]);
            #pragma unroll
            for (int t = 0; t < WN; ++t) {
                const int n32 = n32b + t;
                const int col = n32 * 32 + (lane & 31);
                const float sc = __half2float(scales[(size_t)g * N + col]);
                const unsigned char zp = sym ? 8 :
                    (zeros[(size_t)g * (N / 2) + col / 2] >> ((col & 1) << 2)) & 0xF;
                const float scz = sc * ((float)zp);
                const u16x4 wb = *reinterpret_cast<const u16x4*>(B + ((size_t)(k32 * N32 + n32) * 64 + lane) * 4);
                f16x4 accv = {0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const u16x4 nibs = (wb >> (4 * s)) & (u16x4){0x000F, 0x000F, 0x000F, 0x000F};
                    f16x4 nf = __builtin_convertvector(nibs, f16x4);
                    nf *= (_Float16)sc;
                    accv += (nf - (_Float16)scz) * af[s];   // packed half accumulate
                }
                acc[t] += (float)((accv[0] + accv[1]) + (accv[2] + accv[3]));
            }
        }
    }

    // Combine the two K-halves of each column (lane L and L+32).
    #pragma unroll
    for (int t = 0; t < WN; ++t) acc[t] += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, acc[t], 32);

    // In-block wave reduction over WAVES via LDS; wave 0 writes the result.
    extern __shared__ float ldacc[];   // [WAVES][WN][32]
    if (lane < 32) {
        #pragma unroll
        for (int t = 0; t < WN; ++t) ldacc[(wave * WN + t) * 32 + lane] = acc[t];
    }
    __syncthreads();
    if (wave == 0 && lane < 32) {
        #pragma unroll
        for (int t = 0; t < WN; ++t) {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < WAVES; ++w) s += ldacc[(w * WN + t) * 32 + lane];
            const int col = (n32b + t) * 32 + lane;
            if (split > 1) D[((size_t)blockIdx.z) * N + col] = s;
            else           C[col] = __float2half(s);
        }
    }
}

// Split-K chosen by CU occupancy: target ~4 blocks/CU (tuned; 8 over-splits).
// Adapts to N via NB. For NB=40 -> split 12, NB=136 -> split ~4.
int pick_gemv_split(int nb, int nchunks) {
    int s = (int)((120.0 * 4) / std::max(1, nb) + 0.5);
    s = std::max(1, std::min(s, std::min(32, std::max(1, nchunks / 2))));
    return s;
}

// Launch the M==1 GEMV (WAVES=4, WN=4).
void launch_gemv(const __half* A, const uint16_t* B, const __half* scales,
                 const uint8_t* zeros, float* D, __half* C,
                 int M, int N, int K, int gs, int split,
                 const dim3& grid, hipStream_t stream, bool sym) {
    constexpr int WAVES = 4, WN = 4;
    const size_t dsmem = (size_t)WAVES * WN * 32 * sizeof(float);
    w4a16_gemv_kernel<WAVES, WN><<<grid, 64 * WAVES, dsmem, stream>>>(
        A, B, scales, zeros, D, C, M, N, K, gs, split, sym ? 1 : 0);
    W4A_CHECK(hipGetLastError());
}

// Run the GEMM with my kernel. exllama.py already repacked the tensors into my
// layout: b_q_weight now holds the interleaved B (reinterpreted as uint16). For
// asymmetric quant b_gptq_qzeros holds my 2-per-byte zeros; for symmetric quant
// there is no zero-point tensor and use_v2_format is used as the indicator, so
// the kernel uses the implicit symmetric zero (8). No repack or cache here, and
// no separate weight copy is held.
//
// `split` and `ws` are computed/allocated by the caller (gptq_gemm), which gives
// a torch-pooled, stream-safe, non-retaining workspace instead of a process-wide
// static buffer.
void my_gptq_gemm(const half* a, const uint32_t* b_q_weight,
                  const uint32_t* b_gptq_qzeros, const half* b_gptq_scales,
                  const int* b_g_idx, half* c, int size_m, int size_n,
                  int size_k, int groups, bool use_v2_format, int bit,
                  int split, float* ws) {
    (void)b_g_idx; (void)bit;
    const bool has_zero_points = use_v2_format;   // use_v2_format signals "has zp"
    const uint16_t* B = reinterpret_cast<const uint16_t*>(b_q_weight);
    const __half* scales = b_gptq_scales;
    const uint8_t* zeros = has_zero_points
        ? reinterpret_cast<const uint8_t*>(b_gptq_qzeros)
        : nullptr;
    const int M = size_m, N = size_n, K = size_k, gs = K / groups;
    const cudaStream_t stream = get_current_cuda_stream();

    // Decode (single token) is B-memory-bound: use the interleaved-B GEMV.
    if (M == 1) {
        constexpr int WN = 4;
        const int NB = (N / 32) / WN;
        const dim3 grid(NB, 1, split);
        // sym = no zero points (use_v2_format false -> implicit symmetric zero 8)
        launch_gemv(a, B, scales, zeros, ws, c, M, N, K, gs, split, grid, stream,
                    !has_zero_points);
        if (split > 1) {
            const int total = M * N;
            reduce_split<<<(total + 255) / 256, 256, 0, stream>>>(ws, c, total, split);
        }
        W4A_CHECK(hipGetLastError());
        return;
    }

    MyCfg cfg = pick_my_cfg(M);
    const dim3 grid((M + cfg.bm - 1) / cfg.bm, (N + cfg.bn - 1) / cfg.bn, split);
    launch_my(cfg, a, B, scales, zeros, ws, c, M, N, K, gs, split, grid, stream,
              has_zero_points);
    if (split > 1) {
        const int total = M * N;
        reduce_split<<<(total + 255) / 256, 256, 0, stream>>>(ws, c, total, split);
    }
    W4A_CHECK(hipGetLastError());
}

#endif  // USE_ROCM && __HIP_PLATFORM_AMD__
__global__ void shuffle_4bit_kernel(uint32_t* __restrict__ b_q_weight,
                                    const int size_k, const int size_n) {
  auto n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < size_k) {
    shuffle_4bit_8(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 8;
  }
}

__global__ void make_sequential_4bit_kernel(const uint32_t* __restrict__ w,
                                            uint32_t* __restrict__ w_new,
                                            const int* __restrict__ q_perm,
                                            const int w_width) {
  const uint64_t* w2 = (uint64_t*)w;
  uint64_t* w_new2 = (uint64_t*)w_new;
  int w2_stride = w_width >> 1;
  auto w2_column = THREADS_X * blockIdx.x + threadIdx.x;
  if (w2_column >= w2_stride) return;
  auto w_new2_row = blockIdx.y;
  int q_perm_idx = w_new2_row << 3;
  uint64_t dst = 0;

#pragma unroll
  for (int i = 0; i < 8; i++) {
    int source_row = q_perm[q_perm_idx++];

    int w2_row = source_row >> 3;
    int w2_subrow = source_row & 0x07;
    int w2_row_shift = w2_subrow << 2;
    int wnew2_row_shift = i << 2;

    uint64_t src = w2[w2_row * w2_stride + w2_column];
    src >>= w2_row_shift;
    src &= 0x0000000f0000000f;
    src <<= wnew2_row_shift;
    dst |= src;
  }
  w_new2[w_new2_row * w2_stride + w2_column] = dst;
}


// 4-bit-only shuffle_exllama_weight (weight prep used before my repack).
void shuffle_exllama_weight(uint32_t* q_weight, int* q_perm, int height,
                            int width, int bit) {
  if (bit != 4) {
    fprintf(stderr,
            "gptq_shuffle: only 4-bit is supported by the my-kernel build\n");
    return;
  }
  const cudaStream_t stream = get_current_cuda_stream();
  if (q_perm) {
    uint32_t* new_qweight = NULL;
    cudaMalloc(&new_qweight, height / 32 * 4 * width * sizeof(uint32_t));
    dim3 blockDim, gridDim;
    blockDim.x = THREADS_X;
    blockDim.y = 1;
    gridDim.x = DIVIDE(width, THREADS_X);
    gridDim.y = height / 32 * 4;
    make_sequential_4bit_kernel<<<gridDim, blockDim, 0, stream>>>(
        q_weight, new_qweight, q_perm, width);
    cudaMemcpyAsync(q_weight, new_qweight,
                    height / 32 * 4 * width * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaFree(new_qweight);
  }
  dim3 blockDim, gridDim;
  blockDim.x = THREADS_X;
  blockDim.y = 1;
  gridDim.x = DIVIDE(width, THREADS_X);
  gridDim.y = 1;
  shuffle_4bit_kernel<<<gridDim, blockDim, 0, stream>>>(q_weight, height, width);
}

}  // namespace gptq
}  // namespace vllm

// NOTE: gptq_gemm / gptq_shuffle must live at GLOBAL scope to match the
// torch_bindings.cpp registrations (TORCH_BOX(&gptq_gemm) etc.).

torch::stable::Tensor gptq_gemm(torch::stable::Tensor a,
                                torch::stable::Tensor b_q_weight,
                                torch::stable::Tensor b_gptq_qzeros,
                                torch::stable::Tensor b_gptq_scales,
                                torch::stable::Tensor b_g_idx, bool use_exllama,
                                bool use_v2_format, int64_t bit) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  // b_q_weight keeps the [K/8][N] int32 shape (raw bytes = my interleaved B),
  // so N = b_q_weight.size(1) and K = a.size(1) as before.
  auto c = torch::stable::new_zeros(a, {a.size(0), b_q_weight.size(1)});
#if defined(USE_ROCM) && defined(__HIP_PLATFORM_AMD__)
  {
    const int M = (int)a.size(0), N = (int)c.size(1), K = (int)a.size(1);
    const int groups = (int)b_gptq_scales.size(0);
    // The split-K workspace is allocated here (per call) via torch::stable, so
    // it comes out of the torch caching allocator / CUDA-graph pool at a fixed,
    // replay-stable address -- no process-wide static, and no Python/C++ split
    // mirroring needed (this is the single source of truth for the size).
    // Workspace cap: keep the split-K workspace bounded. For large M the M
    // dimension already fills the CUs, so a small split is enough (and faster).
    // split = min(choose_split(...), WS_CAP_elems / (M*N)); WS_CAP ~ 128 MB.
    static constexpr long long WS_CAP_ELEMS = 32LL * 1024 * 1024;  // 128 MB of float
    auto cap_split = [&](int s) -> int {
        long long cap = WS_CAP_ELEMS / ((long long)M * N);
        if (cap < 1) cap = 1;
        if (s > cap) s = (int)cap;
        return s < 1 ? 1 : s;
    };
    int split;
    if (M == 1) {
      split = vllm::gptq::pick_gemv_split((N / 32) / 4, K / 32);
      // Tuned: the occupancy heuristic over-splits for small N (e.g. N=5120 ->
      // split 24). A smaller split (~8-12) streams B better; cap at 12.
      if (split > 12) split = 12;
    } else {
      vllm::gptq::MyCfg cfg = vllm::gptq::pick_my_cfg(M);
      const long blocks =
          (long)((M + cfg.bm - 1) / cfg.bm) * ((N + cfg.bn - 1) / cfg.bn);
      split = vllm::gptq::choose_split(blocks, K / cfg.bk);
      split = cap_split(split);
    }
    // float workspace holds the split-K partial sums (kernel writes floats).
    torch::stable::Tensor ws_t;
    float* ws = nullptr;
    if (split > 1) {
      ws_t = torch::stable::empty(
          {(int64_t)split * M * N}, c10::ScalarType::Float,
          std::nullopt, a.device());
      ws = (float*)ws_t.data_ptr();
    }
    vllm::gptq::my_gptq_gemm(
        (const half*)a.data_ptr(), (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_gptq_qzeros.data_ptr(),
        (const half*)b_gptq_scales.data_ptr(),
        b_g_idx.device().type() == torch::stable::DeviceType::Meta
            ? NULL
            : (const int*)b_g_idx.data_ptr(),
        (half*)c.data_ptr(), M, N, K, groups, use_v2_format, (int)bit,
        split, ws);
  }
#else
  (void)use_exllama;
  (void)use_v2_format;
  (void)bit;
  fprintf(stderr, "gptq_gemm: my-kernel path is ROCm-only\n");
#endif
  return c;
}

void gptq_shuffle(torch::stable::Tensor q_weight, torch::stable::Tensor q_perm,
                  int64_t bit) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      q_weight.get_device_index());
  vllm::gptq::shuffle_exllama_weight(
      (uint32_t*)q_weight.data_ptr(),
      q_perm.device().type() == torch::stable::DeviceType::Meta ||
              q_perm.numel() == 0
          ? NULL
          : (int*)q_perm.data_ptr(),
      (int)(q_weight.size(0) * 32 / bit), (int)q_weight.size(1), (int)bit);
}
