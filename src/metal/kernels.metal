// The GPU half of tinyllm's Metal backend, for Apple's GPUs: one kernel for each step of the
// forward pass, the same steps as the CPU's (see model.rs). It's compiled from this source when
// tinyllm starts, so building tinyllm needs no Apple tools beyond what every Mac has.
//
// Weights stay as they're stored: bfloat16 (kept as their bits, and widened with a shift, which
// is exact) or 32-bit floats. Everything else is 32-bit. Apple GPUs run 32 threads in lockstep as
// a "simdgroup", which most kernels here use as their unit of work, adding up with simd_sum. As
// everywhere in tinyllm, each result is added up in an order that doesn't depend on what else is
// in the batch, so a request gets the same answer however it's batched.

#include <metal_stdlib>
using namespace metal;

#define SIMD 32
#define MAX_HEAD_DIM 512

inline float bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }

inline float weight_at(device const ushort *w, ulong i) { return bf16(w[i]); }
inline float weight_at(device const float *w, ulong i) { return w[i]; }

// Eight weights starting at w[at], which is a multiple of 8.
inline void load8(device const ushort *w, ulong at, thread float4 &lo, thread float4 &hi) {
    // Two bfloat16s per uint, the first in the low half.
    const uint4 b = *(device const uint4 *)(w + at);
    lo = float4(as_type<float>(b.x << 16), as_type<float>(b.x & 0xFFFF0000u),
                as_type<float>(b.y << 16), as_type<float>(b.y & 0xFFFF0000u));
    hi = float4(as_type<float>(b.z << 16), as_type<float>(b.z & 0xFFFF0000u),
                as_type<float>(b.w << 16), as_type<float>(b.w & 0xFFFF0000u));
}
inline void load8(device const float *w, ulong at, thread float4 &lo, thread float4 &hi) {
    lo = *(device const float4 *)(w + at);
    hi = *(device const float4 *)(w + at + 4);
}

// One threadgroup per row: x[row] = table[tokens[row]] * scale.
struct EmbedArgs {
    uint dim;
    float scale;
};
template <typename T>
inline void embed_rows(device const T *table, device const uint *tokens, device float *x,
                       constant EmbedArgs &a, uint row, uint tid, uint threads) {
    const ulong from = (ulong)tokens[row] * a.dim;
    device float *to = x + (ulong)row * a.dim;
    for (uint i = tid; i < a.dim; i += threads)
        to[i] = weight_at(table, from + i) * a.scale;
}
kernel void embed_bf16(device const ushort *table [[buffer(0)]], device const uint *tokens [[buffer(1)]],
                       device float *x [[buffer(2)]], constant EmbedArgs &a [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]]) {
    embed_rows(table, tokens, x, a, row, tid, threads);
}
kernel void embed_f32(device const float *table [[buffer(0)]], device const uint *tokens [[buffer(1)]],
                      device float *x [[buffer(2)]], constant EmbedArgs &a [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint threads [[threads_per_threadgroup]]) {
    embed_rows(table, tokens, x, a, row, tid, threads);
}

// The sum of squares of a row of `dim` floats (a multiple of 4), added up by a threadgroup of
// NORM_THREADS threads in a fixed order: each thread every NORM_THREADS-th group of 4, then each
// simdgroup's lanes, then the simdgroups in order.
#define NORM_THREADS 256
inline float sum_of_squares(device const float4 *from, uint dim, threadgroup float *partial,
                            uint tid, uint simd, uint lane) {
    float sum = 0.0f;
    for (uint i = tid; i < dim / 4; i += NORM_THREADS)
        sum += dot(from[i], from[i]);
    sum = simd_sum(sum);
    if (lane == 0)
        partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < NORM_THREADS / SIMD; s++)
        total += partial[s];
    return total;
}

// One threadgroup per row: out[row] = x[r] / sqrt(mean(x[r]^2) + eps) * weight, where r is `row`,
// or `rows[row]` when only some rows are wanted, and x's rows are `stride` floats apart.
struct NormArgs {
    uint dim;
    float eps;
    uint select;
    uint stride;
};
kernel void rmsnorm(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                    device const float *weight [[buffer(2)]], device const uint *rows [[buffer(3)]],
                    constant NormArgs &a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[NORM_THREADS / SIMD];
    device const float4 *from = (device const float4 *)(x + (ulong)(a.select ? rows[row] : row) * a.stride);
    const float scale = 1.0f / sqrt(sum_of_squares(from, a.dim, partial, tid, simd, lane) / a.dim + a.eps);
    device float4 *to = (device float4 *)(out + (ulong)row * a.dim);
    device const float4 *w = (device const float4 *)weight;
    for (uint i = tid; i < a.dim / 4; i += NORM_THREADS)
        to[i] = from[i] * scale * w[i];
}

// One threadgroup per row: x[row] += y[row] normalized by its RMS and weighted, for models that
// normalize what each block adds to the residual stream.
kernel void rmsnorm_add(device const float *y [[buffer(0)]], device float *x [[buffer(1)]],
                        device const float *weight [[buffer(2)]], constant NormArgs &a [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[NORM_THREADS / SIMD];
    device const float4 *from = (device const float4 *)(y + (ulong)row * a.stride);
    const float scale = 1.0f / sqrt(sum_of_squares(from, a.dim, partial, tid, simd, lane) / a.dim + a.eps);
    device float4 *to = (device float4 *)(x + (ulong)row * a.dim);
    device const float4 *w = (device const float4 *)weight;
    for (uint i = tid; i < a.dim / 4; i += NORM_THREADS)
        to[i] += from[i] * scale * w[i];
}

// One simdgroup per (head, row): normalize a head of the queries (the first n_q) or keys.
struct HeadNormArgs {
    uint stride;
    uint n_q;
    uint head_dim;
    uint k_offset;
    float eps;
};
kernel void head_norm(device float *qkv [[buffer(0)]], device const float *q_weight [[buffer(1)]],
                      device const float *k_weight [[buffer(2)]],
                      constant HeadNormArgs &a [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint head = group.x, row = group.y;
    const bool is_q = head < a.n_q;
    device float4 *x = (device float4 *)(qkv + (ulong)row * a.stride +
                                         (is_q ? head * a.head_dim : a.k_offset + (head - a.n_q) * a.head_dim));
    device const float4 *weight = (device const float4 *)(is_q ? q_weight : k_weight);
    float sum = 0.0f;
    for (uint i = lane; i < a.head_dim / 4; i += SIMD)
        sum += dot(x[i], x[i]);
    const float scale = 1.0f / sqrt(simd_sum(sum) / a.head_dim + a.eps);
    for (uint i = lane; i < a.head_dim / 4; i += SIMD)
        x[i] *= scale * weight[i];
}

// Matrix multiplication: out[b][m] = scale * dot(w[m], x[b]) (+ bias[m]), added to what's
// there if `accumulate`. Each dot product is added up in one fixed order, the same whatever else
// is in the batch and whichever of the two kernels below does it, so a request's results never
// depend on how it's batched:
//
// - the columns are split into RANGES runs of whole groups of 8 (the last runs can be short or
//   empty), and each run is added up from zero with one fused multiply-add after another, from
//   its first column to its last;
// - then the runs are added together in order.
//
// That's exactly what Apple's 8x8 simdgroup matrix instructions do with a run (each is a chain
// of fused multiply-adds over its 8 columns, carrying on from the last), so matmul_mma can use
// them for big batches while matmul_vec does the same sums one lane at a time for small ones.
// n has to be a multiple of 8.
#define RANGES 8
struct MatmulArgs {
    uint m;
    uint n;
    uint batch;
    float scale;
    uint accumulate;
    uint has_bias;
    // How far apart the output's rows are (usually m).
    uint out_stride;
};

// How many groups of 8 columns each run has.
inline uint run_length(uint groups) { return (groups + RANGES - 1) / RANGES; }

// The end of the sum: scale, bias, and store or add.
inline void finish_output(device float *out, device const float *bias, constant MatmulArgs &a,
                          uint b, uint m, float total) {
    float value = total * a.scale;
    if (a.has_bias)
        value += bias[m];
    device float *at = out + (ulong)b * a.out_stride + m;
    *at = a.accumulate ? *at + value : value;
}

// For small batches: each threadgroup of RANGES simdgroups takes SIMD weight rows, one per lane,
// and up to B rows of x (1, 2, 4, 6 or 8), and simdgroup s works out run s of each of them. So every lane of
// a simdgroup reads the same columns of x at the same time, which the hardware fetches once for
// all of them, and each weight is read once. A lane goes through its run a column at a time,
// doing every row of x's multiply-add for that column together: separate sums, so they don't
// wait for each other, each still added up in column order. The runs' sums meet in threadgroup
// memory and the first simdgroup adds them up in order.
#define VEC_BATCH 8
#define VEC_STEP 2
#define VEC_THREADS (RANGES * SIMD)
#define VEC_PARTIALS (RANGES * VEC_BATCH * SIMD)
template <uint B, typename W>
inline void matmul_vec_rows(device const W *w, device const float *x, device float *out,
                            device const float *bias, constant MatmulArgs &a,
                            threadgroup float *partial, uint2 group, uint simd, uint lane) {
    const uint run = simd;
    const uint m = group.x * SIMD + lane;
    const uint b0 = group.y * B;
    const uint rows = min(B, a.batch - b0);
    const uint groups = a.n / 8, length = run_length(groups);
    const uint first = min(run * length, groups), last = min(first + length, groups);
    // Lanes past the last weight row read it again, and don't store anything; rows of x past
    // the batch's last read it again, and aren't kept.
    const ulong row = (ulong)min(m, a.m - 1) * a.n;
    device const float4 *xs[B];
    for (uint b = 0; b < B; b++)
        xs[b] = (device const float4 *)(x + (ulong)(b0 + min(b, rows - 1)) * a.n);
    float acc[B];
    for (uint b = 0; b < B; b++)
        acc[b] = 0.0f;
    // VEC_STEP groups of weights are loaded at a time, so that more are on their way at once.
    for (uint g = first; g < last; g += VEC_STEP) {
        float4 lo[VEC_STEP], hi[VEC_STEP];
        for (uint u = 0; u < VEC_STEP; u++)
            load8(w, row + min(g + u, last - 1) * 8, lo[u], hi[u]);
        for (uint u = 0; u < VEC_STEP; u++) {
            if (g + u >= last)
                break;
            float4 x0[B], x1[B];
            for (uint b = 0; b < B; b++) {
                x0[b] = xs[b][2 * (g + u)];
                x1[b] = xs[b][2 * (g + u) + 1];
            }
            for (uint b = 0; b < B; b++)
                acc[b] = fma(lo[u].x, x0[b].x, acc[b]);
            for (uint b = 0; b < B; b++)
                acc[b] = fma(lo[u].y, x0[b].y, acc[b]);
            for (uint b = 0; b < B; b++)
                acc[b] = fma(lo[u].z, x0[b].z, acc[b]);
            for (uint b = 0; b < B; b++)
                acc[b] = fma(lo[u].w, x0[b].w, acc[b]);
            for (uint b = 0; b < B; b++)
                acc[b] = fma(hi[u].x, x1[b].x, acc[b]);
            for (uint b = 0; b < B; b++)
                acc[b] = fma(hi[u].y, x1[b].y, acc[b]);
            for (uint b = 0; b < B; b++)
                acc[b] = fma(hi[u].z, x1[b].z, acc[b]);
            for (uint b = 0; b < B; b++)
                acc[b] = fma(hi[u].w, x1[b].w, acc[b]);
        }
    }
    for (uint b = 0; b < rows; b++)
        partial[(run * B + b) * SIMD + lane] = acc[b];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (run != 0 || m >= a.m)
        return;
    for (uint b = 0; b < rows; b++) {
        float total = partial[b * SIMD + lane];
        for (uint r = 1; r < RANGES; r++) {
            if (r * length < groups)
                total += partial[(r * B + b) * SIMD + lane];
        }
        finish_output(out, bias, a, b0 + b, m, total);
    }
}

// For big batches: each threadgroup of 4 simdgroups works out a tile of MM_BN rows of x by MM_BM
// weight rows, MM_BK columns at a time, staged in threadgroup memory as 32-bit floats (weights
// stored column by column, so both operands load straight). Each simdgroup has 16 x 32 of the
// tile, as 2 x 4 simdgroup matrices, plus their totals over the runs so far. Adding a run to a
// total is done as total + run * identity, which rounds once, like total += run.
#define MM_BN 32
#define MM_BM 64
#define MM_BK 32
#define MM_THREADS 128
#define XS_STRIDE (MM_BK + 4)
#define WS_STRIDE (MM_BM + 4)
#define MM_SHARED (MM_BN * XS_STRIDE + MM_BK * WS_STRIDE)
template <typename W>
inline void matmul_mma_rows(device const W *w, device const float *x, device float *out,
                            device const float *bias, constant MatmulArgs &a,
                            threadgroup float *shared, uint2 group, uint simd, uint tid) {
    threadgroup float *xs = shared, *ws = shared + MM_BN * XS_STRIDE;
    const uint m0 = group.x * MM_BM, b0 = group.y * MM_BN;
    const uint sb = (simd / 2) * 16, sm = (simd % 2) * 32;
    const simdgroup_float8x8 identity = simdgroup_float8x8(1.0f);
    simdgroup_float8x8 sums[2][4], totals[2][4];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 4; j++) {
            sums[i][j] = simdgroup_float8x8(0.0f);
            totals[i][j] = simdgroup_float8x8(0.0f);
        }
    const uint groups = a.n / 8, length = run_length(groups);
    bool first_run = true;
    for (uint k0 = 0; k0 < a.n; k0 += MM_BK) {
        for (uint t = tid; t < MM_BM * (MM_BK / 8); t += MM_THREADS) {
            const uint row = t % MM_BM, col = (t / MM_BM) * 8;
            float4 lo = 0.0f, hi = 0.0f;
            if (m0 + row < a.m && k0 + col < a.n)
                load8(w, (ulong)(m0 + row) * a.n + k0 + col, lo, hi);
            threadgroup float *to = ws + col * WS_STRIDE + row;
            to[0 * WS_STRIDE] = lo.x;
            to[1 * WS_STRIDE] = lo.y;
            to[2 * WS_STRIDE] = lo.z;
            to[3 * WS_STRIDE] = lo.w;
            to[4 * WS_STRIDE] = hi.x;
            to[5 * WS_STRIDE] = hi.y;
            to[6 * WS_STRIDE] = hi.z;
            to[7 * WS_STRIDE] = hi.w;
        }
        for (uint t = tid; t < MM_BN * (MM_BK / 4); t += MM_THREADS) {
            const uint row = t / (MM_BK / 4), col = (t % (MM_BK / 4)) * 4;
            float4 value = 0.0f;
            if (b0 + row < a.batch && k0 + col < a.n)
                value = *(device const float4 *)(x + (ulong)(b0 + row) * a.n + k0 + col);
            *(threadgroup float4 *)(xs + row * XS_STRIDE + col) = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint chunk = min((uint)MM_BK, a.n - k0) / 8;
        for (uint kg = 0; kg < chunk; kg++) {
            const uint g = k0 / 8 + kg;
            if (g > 0 && g % length == 0) {
                // A new run starts: add the last one to the totals.
                for (uint i = 0; i < 2; i++)
                    for (uint j = 0; j < 4; j++) {
                        if (first_run)
                            totals[i][j] = sums[i][j];
                        else
                            simdgroup_multiply_accumulate(totals[i][j], sums[i][j], identity, totals[i][j]);
                        sums[i][j] = simdgroup_float8x8(0.0f);
                    }
                first_run = false;
            }
            simdgroup_float8x8 xm[2], wm[4];
            for (uint i = 0; i < 2; i++)
                simdgroup_load(xm[i], xs + (sb + i * 8) * XS_STRIDE + kg * 8, XS_STRIDE);
            for (uint j = 0; j < 4; j++)
                simdgroup_load(wm[j], ws + kg * 8 * WS_STRIDE + sm + j * 8, WS_STRIDE);
            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(sums[i][j], xm[i], wm[j], sums[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 4; j++) {
            if (first_run)
                totals[i][j] = sums[i][j];
            else
                simdgroup_multiply_accumulate(totals[i][j], sums[i][j], identity, totals[i][j]);
        }
    // Through threadgroup memory to the output, so the edges of the batch and matrix are easy.
    threadgroup float *tile = shared;
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(totals[i][j], tile + (sb + i * 8) * MM_BM + sm + j * 8, MM_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = tid; t < MM_BN * MM_BM; t += MM_THREADS) {
        const uint b = t / MM_BM, m = t % MM_BM;
        if (b0 + b < a.batch && m0 + m < a.m)
            finish_output(out, bias, a, b0 + b, m0 + m, tile[t]);
    }
}

#define MATMUL_ARGS(W)                                                                             \
    device const W *w [[buffer(0)]], device const float *x [[buffer(1)]],                          \
        device float *out [[buffer(2)]], device const float *bias [[buffer(3)]],                   \
        constant MatmulArgs &a [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],        \
        uint simd [[simdgroup_index_in_threadgroup]]
// matmul_vecB_bf16 and matmul_vecB_f32 for B rows of x per threadgroup.
#define MATMUL_VEC(B)                                                                              \
    kernel void matmul_vec##B##_bf16(MATMUL_ARGS(ushort), uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup float partial[VEC_PARTIALS];                                                   \
        matmul_vec_rows<B>(w, x, out, bias, a, partial, group, simd, lane);                        \
    }                                                                                              \
    kernel void matmul_vec##B##_f32(MATMUL_ARGS(float), uint lane [[thread_index_in_simdgroup]]) { \
        threadgroup float partial[VEC_PARTIALS];                                                   \
        matmul_vec_rows<B>(w, x, out, bias, a, partial, group, simd, lane);                        \
    }
MATMUL_VEC(1)
MATMUL_VEC(2)
MATMUL_VEC(4)
MATMUL_VEC(6)
MATMUL_VEC(8)
kernel void matmul_mma_bf16(MATMUL_ARGS(ushort), uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[MM_SHARED];
    matmul_mma_rows(w, x, out, bias, a, shared, group, simd, tid);
}
kernel void matmul_mma_f32(MATMUL_ARGS(float), uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[MM_SHARED];
    matmul_mma_rows(w, x, out, bias, a, shared, group, simd, tid);
}

// One threadgroup per row: rotate its queries and keys by their position (pairs i and
// i + head_dim/2 of each head), then store its keys and values in its KV slot.
struct RopeArgs {
    uint stride;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint q_dim;
    uint kv_dim;
};
kernel void rope_store(device float *qkv [[buffer(0)]], device const float *cos_table [[buffer(1)]],
                       device const float *sin_table [[buffer(2)]],
                       device const uint *write_slots [[buffer(3)]], device float *k_pool [[buffer(4)]],
                       device float *v_pool [[buffer(5)]], constant RopeArgs &a [[buffer(6)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]]) {
    const uint half_dim = a.head_dim / 2;
    device float *q = qkv + (ulong)row * a.stride;
    device float *k = q + a.q_dim;
    device float *v = k + a.kv_dim;
    device const float *c = cos_table + (ulong)row * half_dim;
    device const float *s = sin_table + (ulong)row * half_dim;
    for (uint p = tid; p < (a.n_heads + a.n_kv_heads) * half_dim; p += threads) {
        const uint head = p / half_dim, i = p % half_dim;
        device float *h = head < a.n_heads ? q + head * a.head_dim
                                           : k + (head - a.n_heads) * a.head_dim;
        const float x0 = h[i], x1 = h[i + half_dim];
        h[i] = x0 * c[i] - x1 * s[i];
        h[i + half_dim] = x1 * c[i] + x0 * s[i];
    }
    threadgroup_barrier(mem_flags::mem_device);
    device float *key = k_pool + (ulong)write_slots[row] * a.kv_dim;
    device float *value = v_pool + (ulong)write_slots[row] * a.kv_dim;
    for (uint i = tid; i < a.kv_dim; i += threads) {
        key[i] = k[i];
        value[i] = v[i];
    }
}

// One threadgroup of ATT_SIMDS simdgroups per (head, row): attention over the row's KV slots
// (the last `window` of them). The positions are split into ATT_SIMDS runs, one per simdgroup,
// which goes through its run ATT_STEP positions at a time, keeping a running maximum so the
// softmax can be built up as it goes. Each lane holds every 32nd group of 4 elements of the head.
// Then the simdgroups' results are combined in order. How the work is split depends only on the
// row's own length, so its result doesn't depend on what else is in the batch. Needs head_dim to
// be a multiple of 4, and threadgroup memory for ATT_SIMDS * (head_dim + 2) floats.
#define ATT_SIMDS 8
#define ATT_STEP 4
#define HEAD_VECS (MAX_HEAD_DIM / 4 / SIMD)
struct AttnArgs {
    uint stride;
    uint n_heads;
    uint n_kv_heads;
    uint head_dim;
    uint kv_dim;
    uint q_dim;
    float scale;
    float softcap;
    uint window;
};
kernel void attention(device const float *qkv [[buffer(0)]], device const float *k_pool [[buffer(1)]],
                      device const float *v_pool [[buffer(2)]], device const uint *slots [[buffer(3)]],
                      device const uint *row_offset [[buffer(4)]],
                      device const uint *row_len [[buffer(5)]], device float *out [[buffer(6)]],
                      constant AttnArgs &a [[buffer(7)]], threadgroup float *shared [[threadgroup(0)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint simd [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint tid [[thread_index_in_threadgroup]]) {
    const uint head = group.x, row = group.y;
    const uint kv_head = head / (a.n_heads / a.n_kv_heads);
    const uint vecs = a.head_dim / 4;
    device const float4 *query =
        (device const float4 *)(qkv + (ulong)row * a.stride + head * a.head_dim);
    float4 q[HEAD_VECS], o[HEAD_VECS];
    for (uint j = 0; j < HEAD_VECS; j++) {
        const uint f = lane + SIMD * j;
        q[j] = f < vecs ? query[f] : 0.0f;
        o[j] = 0.0f;
    }
    const uint len = row_len[row];
    const uint first = len > a.window ? len - a.window : 0;
    const uint run = (len - first + ATT_SIMDS - 1) / ATT_SIMDS;
    const uint start = first + simd * run, end = min(start + run, len);
    device const uint *row_slots = slots + row_offset[row];
    const ulong head_offset = (ulong)kv_head * a.head_dim;
    float running_max = -INFINITY, total = 0.0f;
    for (uint p = start; p < end; p += ATT_STEP) {
        // Positions past the end of the run score -infinity, which gives them no weight.
        ulong at[ATT_STEP];
        float scores[ATT_STEP];
        for (uint u = 0; u < ATT_STEP; u++)
            at[u] = (ulong)row_slots[min(p + u, end - 1)] * a.kv_dim + head_offset;
        for (uint u = 0; u < ATT_STEP; u++) {
            device const float4 *key = (device const float4 *)(k_pool + at[u]);
            float partial = 0.0f;
            for (uint j = 0; j < HEAD_VECS; j++) {
                const uint f = lane + SIMD * j;
                if (f < vecs)
                    partial += dot(q[j], key[f]);
            }
            float score = simd_sum(partial) * a.scale;
            if (a.softcap > 0.0f)
                score = a.softcap * precise::tanh(score / a.softcap);
            scores[u] = p + u < end ? score : -INFINITY;
        }
        float new_max = running_max;
        for (uint u = 0; u < ATT_STEP; u++)
            new_max = max(new_max, scores[u]);
        const float correction = precise::exp(running_max - new_max);
        float weights[ATT_STEP];
        total *= correction;
        for (uint u = 0; u < ATT_STEP; u++) {
            weights[u] = precise::exp(scores[u] - new_max);
            total += weights[u];
        }
        for (uint j = 0; j < HEAD_VECS; j++) {
            const uint f = lane + SIMD * j;
            if (f < vecs) {
                float4 sum = o[j] * correction;
                for (uint u = 0; u < ATT_STEP; u++)
                    sum += weights[u] * ((device const float4 *)(v_pool + at[u]))[f];
                o[j] = sum;
            }
        }
        running_max = new_max;
    }
    // Each simdgroup leaves its maximum, total and weighted values; then they're combined in
    // order of simdgroup. One that had no positions has a maximum of -infinity and adds nothing.
    threadgroup float *mine = shared + simd * (a.head_dim + 2);
    for (uint j = 0; j < HEAD_VECS; j++) {
        const uint f = lane + SIMD * j;
        if (f < vecs)
            ((threadgroup float4 *)(mine + 2))[f] = o[j];
    }
    if (lane == 0) {
        mine[0] = running_max;
        mine[1] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = -INFINITY;
    for (uint s = 0; s < ATT_SIMDS; s++)
        best = max(best, shared[s * (a.head_dim + 2)]);
    float sum = 0.0f;
    for (uint s = 0; s < ATT_SIMDS; s++) {
        threadgroup float *theirs = shared + s * (a.head_dim + 2);
        sum += theirs[1] * precise::exp(theirs[0] - best);
    }
    device float *to = out + (ulong)row * a.q_dim + head * a.head_dim;
    for (uint d = tid; d < a.head_dim; d += ATT_SIMDS * SIMD) {
        float value = 0.0f;
        for (uint s = 0; s < ATT_SIMDS; s++) {
            threadgroup float *theirs = shared + s * (a.head_dim + 2);
            value += theirs[2 + d] * precise::exp(theirs[0] - best);
        }
        to[d] = value / sum;
    }
}

// out[row][i] = act(gate[i]) * up[i], with gate and up side by side in each row of gate_up.
struct GatedArgs {
    uint hidden;
    uint gelu;
};
kernel void gated(device const float *gate_up [[buffer(0)]], device float *out [[buffer(1)]],
                  constant GatedArgs &a [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
                  uint2 tid [[thread_position_in_threadgroup]],
                  uint2 threads [[threads_per_threadgroup]]) {
    const uint row = group.y, i = group.x * threads.x + tid.x;
    if (i >= a.hidden)
        return;
    const float g = gate_up[(ulong)row * 2 * a.hidden + i];
    const float u = gate_up[(ulong)row * 2 * a.hidden + a.hidden + i];
    float activated;
    if (a.gelu) {
        const float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
        activated = 0.5f * g * (1.0f + precise::tanh(inner));
    } else {
        activated = g / (1.0f + precise::exp(-g));
    }
    out[(ulong)row * a.hidden + i] = activated * u;
}

// Scores divided by `divisor`, then soft-capped if `softcap` isn't 0.
struct LogitArgs {
    uint count;
    float divisor;
    float softcap;
};
kernel void finish_logits(device float *logits [[buffer(0)]], constant LogitArgs &a [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= a.count)
        return;
    float value = logits[i] / a.divisor;
    if (a.softcap > 0.0f)
        value = a.softcap * precise::tanh(value / a.softcap);
    logits[i] = value;
}

// LoRA adapters: a row using adapter a (1 or more; 0 is none) gets B_a (A_a x) added to a
// matrix's output, with every adapter's A stored as [adapters, rank, in] and B as
// [adapters, out, rank], padded with zeros to the largest rank. First h = A x, one simdgroup per
// (j, row); then each output adds its row of B times h. Both add up in a fixed order.
struct LoraArgs {
    uint in_dim;
    uint out_dim;
    uint rank;
};
kernel void lora_shrink(device const float *x [[buffer(0)]], device const float *a [[buffer(1)]],
                        device const uint *adapters [[buffer(2)]], device float *h [[buffer(3)]],
                        constant LoraArgs &args [[buffer(4)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint simds [[simdgroups_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.y, j = group.x * simds + simd;
    if (j >= args.rank)
        return;
    const uint adapter = adapters[row];
    float sum = 0.0f;
    if (adapter != 0) {
        device const float *aj = a + ((ulong)(adapter - 1) * args.rank + j) * args.in_dim;
        device const float *xr = x + (ulong)row * args.in_dim;
        for (uint k = lane; k < args.in_dim; k += SIMD)
            sum += aj[k] * xr[k];
        sum = simd_sum(sum);
    }
    if (lane == 0)
        h[(ulong)row * args.rank + j] = sum;
}
kernel void lora_expand(device const float *h [[buffer(0)]], device const float *b [[buffer(1)]],
                        device const uint *adapters [[buffer(2)]], device float *out [[buffer(3)]],
                        constant LoraArgs &args [[buffer(4)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint2 tid [[thread_position_in_threadgroup]],
                        uint2 threads [[threads_per_threadgroup]]) {
    const uint row = group.y, o = group.x * threads.x + tid.x;
    const uint adapter = adapters[row];
    if (o >= args.out_dim || adapter == 0)
        return;
    device const float *bo = b + ((ulong)(adapter - 1) * args.out_dim + o) * args.rank;
    device const float *hr = h + (ulong)row * args.rank;
    float sum = 0.0f;
    for (uint j = 0; j < args.rank; j++)
        sum += bo[j] * hr[j];
    out[(ulong)row * args.out_dim + o] += sum;
}

// ---------------------------------------------------------------------------------------------
// Mixtures of experts
// ---------------------------------------------------------------------------------------------
// A token goes through the few experts its router scores best, as on CUDA (see kernels.cu):
// moe_route picks each row's experts, moe_sort lays out the (row, expert) pairs grouped by
// expert, each expert's run padded to whole groups of `group_rows` rows, moe_gather copies the
// rows into that order, the grouped matmuls multiply each group by its expert's weights, and
// moe_combine adds up each row's results in order of expert. The grouped matmuls are dispatched
// for as many groups as there could be, and those past the real number stop at once, so the CPU
// never has to wait to see which experts were chosen.
#define MAX_TOP_K 16
#define MAX_EXPERTS 512
#define MAX_GROUPS 32

struct RouteArgs {
    uint experts;
    uint top_k;
    uint renormalize;
    // DeepSeek's routing (see `Routing` in config.rs), rather than the softmax's top_k.
    uint deepseek;
    uint sigmoid;
    uint groups;
    uint top_groups;
    float scale;
    uint has_correction;
};

// One simdgroup per row. The usual routing: the softmax of the scores, the top_k best, weights
// renormalized to add up to 1 if asked. DeepSeek's: weights from the softmax or from sigmoids,
// experts chosen by those plus `correction` from the best `top_groups` of `groups` equal groups
// (scored by their best member, or best two with sigmoids), renormalized if asked, then scaled.
// Ties go to the lower number. Written in order of expert.
kernel void moe_route(device const float *scores [[buffer(0)]],
                      device const float *correction [[buffer(1)]],
                      device uint *topk_ids [[buffer(2)]], device float *topk_weights [[buffer(3)]],
                      constant RouteArgs &a [[buffer(4)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float weight[MAX_EXPERTS], choice[MAX_EXPERTS], group_score[MAX_GROUPS];
    threadgroup uint allowed[MAX_GROUPS];
    device const float *s = scores + (ulong)row * a.experts;
    float largest = -INFINITY, sum = 0.0f;
    if (!a.sigmoid) {
        for (uint i = lane; i < a.experts; i += SIMD)
            largest = max(largest, s[i]);
        largest = simd_max(largest);
        for (uint i = lane; i < a.experts; i += SIMD)
            sum += precise::exp(s[i] - largest);
        sum = simd_sum(sum);
    }
    for (uint i = lane; i < a.experts; i += SIMD) {
        const float w = a.sigmoid ? 1.0f / (1.0f + precise::exp(-s[i])) : precise::exp(s[i] - largest) / sum;
        weight[i] = w;
        // The usual routing chooses by the scores themselves.
        choice[i] = !a.deepseek ? s[i] : a.has_correction ? w + correction[i] : w;
    }
    const uint groups = a.deepseek ? a.groups : 1, per_group = a.experts / groups;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = lane; g < groups; g += SIMD) {
        float first = -INFINITY, second = -INFINITY;
        for (uint i = g * per_group; i < (g + 1) * per_group; i++) {
            const float c = choice[i];
            if (c > first) {
                second = first;
                first = c;
            } else if (c > second) {
                second = c;
            }
        }
        group_score[g] = a.sigmoid ? first + (per_group > 1 ? second : 0.0f) : first;
        allowed[g] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        for (uint k = 0; k < min(a.deepseek ? a.top_groups : 1, groups); k++) {
            int best = -1;
            for (uint g = 0; g < groups; g++)
                if (!allowed[g] && (best < 0 || group_score[g] > group_score[best]))
                    best = g;
            allowed[best] = 1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint chosen[MAX_TOP_K];
    float chosen_weight[MAX_TOP_K];
    float total = 0.0f;
    for (uint k = 0; k < a.top_k; k++) {
        float best = -INFINITY;
        uint best_i = 0xffffffffu;
        for (uint i = lane; i < a.experts; i += SIMD) {
            bool taken = !allowed[i / per_group];
            for (uint j = 0; j < k; j++)
                taken |= chosen[j] == i;
            if (!taken && (choice[i] > best || (choice[i] == best && i < best_i))) {
                best = choice[i];
                best_i = i;
            }
        }
        for (uint step = SIMD / 2; step > 0; step >>= 1) {
            const float other = simd_shuffle_xor(best, step);
            const uint other_i = simd_shuffle_xor(best_i, step);
            if (other > best || (other == best && other_i < best_i)) {
                best = other;
                best_i = other_i;
            }
        }
        chosen[k] = best_i;
        chosen_weight[k] = weight[best_i];
        total += chosen_weight[k];
    }
    if (lane != 0)
        return;
    for (uint k = 1; k < a.top_k; k++) {
        for (uint j = k; j > 0 && chosen[j - 1] > chosen[j]; j--) {
            const uint id = chosen[j];
            chosen[j] = chosen[j - 1];
            chosen[j - 1] = id;
            const float w = chosen_weight[j];
            chosen_weight[j] = chosen_weight[j - 1];
            chosen_weight[j - 1] = w;
        }
    }
    // DeepSeek's renormalization keeps clear of dividing by 0.
    const float divisor = a.renormalize ? total + (a.deepseek ? 1e-20f : 0.0f) : 1.0f;
    for (uint k = 0; k < a.top_k; k++) {
        topk_ids[row * a.top_k + k] = chosen[k];
        topk_weights[row * a.top_k + k] = (a.renormalize ? chosen_weight[k] / divisor : chosen_weight[k]) * (a.deepseek ? a.scale : 1.0f);
    }
}

// One threadgroup: lay out the (row, expert) pairs grouped by expert, each expert's run padded to
// a whole number of groups. Each thread takes some experts and finds their pairs in order, so
// the layout is the same every time. `slot_of[pair]` is where a pair went; `sorted_rows[slot]`
// is its row (or -1 for padding); `group_expert[group]` is each group's expert.
struct SortArgs {
    uint rows;
    uint top_k;
    uint experts;
    uint group_rows;
};
kernel void moe_sort(device const uint *topk_ids [[buffer(0)]], device int *sorted_rows [[buffer(1)]],
                     device uint *slot_of [[buffer(2)]], device uint *group_expert [[buffer(3)]],
                     device uint *num_groups [[buffer(4)]], constant SortArgs &a [[buffer(5)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]]) {
    threadgroup uint counts[MAX_EXPERTS], starts[MAX_EXPERTS + 1];
    const uint pairs = a.rows * a.top_k, size = a.group_rows;
    for (uint e = tid; e < a.experts; e += threads) {
        uint count = 0;
        for (uint i = 0; i < pairs; i++)
            count += topk_ids[i] == e;
        counts[e] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint at = 0;
        for (uint e = 0; e < a.experts; e++) {
            starts[e] = at;
            at += (counts[e] + size - 1) / size * size;
        }
        starts[a.experts] = at;
        *num_groups = at / size;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < a.experts; e += threads) {
        uint at = starts[e];
        for (uint i = 0; i < pairs; i++) {
            if (topk_ids[i] == e) {
                sorted_rows[at] = i / a.top_k;
                slot_of[i] = at;
                at++;
            }
        }
        for (; at < starts[e + 1]; at++)
            sorted_rows[at] = -1;
        for (uint g = starts[e] / size; g < starts[e + 1] / size; g++)
            group_expert[g] = e;
    }
}

// One threadgroup per slot: copy its row (zeros for padding), `dim` floats.
struct GatherArgs {
    uint dim;
    uint group_rows;
};
kernel void moe_gather(device const float4 *x [[buffer(0)]], device const int *sorted_rows [[buffer(1)]],
                       device const uint *num_groups [[buffer(2)]], device float4 *out [[buffer(3)]],
                       constant GatherArgs &a [[buffer(4)]], uint slot [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]]) {
    if (slot >= *num_groups * a.group_rows)
        return;
    const int row = sorted_rows[slot];
    const uint vectors = a.dim / 4;
    for (uint i = tid; i < vectors; i += threads)
        out[(ulong)slot * vectors + i] = row < 0 ? float4(0.0f) : x[(ulong)row * vectors + i];
}

// moe_matmul_*: matmul_* for each group of rows (8 for vec, MM_BN for mma) and its expert, whose
// `[m, n]` weights (and biases) follow the expert before's.
#define MOE_MATMUL_ARGS(W)                                                                         \
    MATMUL_ARGS(W), device const uint *group_expert [[buffer(5)]],                                 \
        device const uint *num_groups [[buffer(6)]]
#define EXPERT_WEIGHTS                                                                             \
    if (group.y >= *num_groups)                                                                    \
        return;                                                                                    \
    const ulong expert = group_expert[group.y];                                                    \
    device const float *expert_bias = bias + (a.has_bias ? expert * a.m : 0);
kernel void moe_matmul_vec_bf16(MOE_MATMUL_ARGS(ushort), uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[VEC_PARTIALS];
    EXPERT_WEIGHTS
    matmul_vec_rows<VEC_BATCH>(w + expert * a.m * a.n, x, out, expert_bias, a, partial, group, simd, lane);
}
kernel void moe_matmul_vec_f32(MOE_MATMUL_ARGS(float), uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[VEC_PARTIALS];
    EXPERT_WEIGHTS
    matmul_vec_rows<VEC_BATCH>(w + expert * a.m * a.n, x, out, expert_bias, a, partial, group, simd, lane);
}
kernel void moe_matmul_mma_bf16(MOE_MATMUL_ARGS(ushort), uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[MM_SHARED];
    EXPERT_WEIGHTS
    matmul_mma_rows(w + expert * a.m * a.n, x, out, expert_bias, a, shared, group, simd, tid);
}
kernel void moe_matmul_mma_f32(MOE_MATMUL_ARGS(float), uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[MM_SHARED];
    EXPERT_WEIGHTS
    matmul_mma_rows(w + expert * a.m * a.n, x, out, expert_bias, a, shared, group, simd, tid);
}

// Each row adds up what its experts made of it, weighted, in order of expert, and adds that to
// the residual stream.
struct CombineArgs {
    uint dim;
    uint top_k;
};
kernel void moe_combine(device const float *down [[buffer(0)]], device const uint *slot_of [[buffer(1)]],
                        device const float *topk_weights [[buffer(2)]], device float *x [[buffer(3)]],
                        constant CombineArgs &a [[buffer(4)]],
                        uint2 group [[threadgroup_position_in_grid]],
                        uint2 tid [[thread_position_in_threadgroup]],
                        uint2 threads [[threads_per_threadgroup]]) {
    const uint row = group.y, i = group.x * threads.x + tid.x;
    if (i >= a.dim)
        return;
    float sum = 0.0f;
    for (uint k = 0; k < a.top_k; k++)
        sum += topk_weights[row * a.top_k + k] * down[(ulong)slot_of[row * a.top_k + k] * a.dim + i];
    x[(ulong)row * a.dim + i] += sum;
}

// For a shared expert with a gate of its own (Qwen2 MoE): x += sigmoid(gate) * y.
kernel void add_gated(device const float *y [[buffer(0)]], device const float *gate [[buffer(1)]],
                      device float *x [[buffer(2)]], constant uint &dim [[buffer(3)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint2 tid [[thread_position_in_threadgroup]],
                      uint2 threads [[threads_per_threadgroup]]) {
    const uint row = group.y, i = group.x * threads.x + tid.x;
    if (i >= dim)
        return;
    const float scale = 1.0f / (1.0f + precise::exp(-gate[row]));
    x[(ulong)row * dim + i] += scale * y[(ulong)row * dim + i];
}

// ---------------------------------------------------------------------------------------------
// DeepSeek's latent attention
// ---------------------------------------------------------------------------------------------
// As on CUDA (see kernels.cu): the cache holds each token's normalized latent (`rank` wide) and
// the rotated part of its keys that all heads share, and each head scores its query against the
// latent through that head's key matrix (folded into the query by matmul_heads), then expands its
// blend of latents with its value matrix. Attention is then one key/value head, whose values are
// the first `rank` elements of its keys.

// matmul_heads_*: matmul_* for each head in turn (the grid's third dimension), with head h's
// weights, inputs and outputs `w_stride`, `x_stride` and `out_offset` elements after head h - 1's.
struct HeadArgs {
    uint w_stride;
    uint x_stride;
    uint out_offset;
};
#define HEADS_ARGS(W)                                                                              \
    device const W *w [[buffer(0)]], device const float *x [[buffer(1)]],                          \
        device float *out [[buffer(2)]], device const float *bias [[buffer(3)]],                   \
        constant MatmulArgs &a [[buffer(4)]], constant HeadArgs &h [[buffer(5)]],                  \
        uint3 group3 [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]]
#define HEAD_PARTS(W)                                                                              \
    const ulong head = group3.z;                                                                   \
    const uint2 group = group3.xy;                                                                 \
    device const W *head_w = w + head * h.w_stride;                                                \
    device const float *head_x = x + head * h.x_stride;                                            \
    device float *head_out = out + head * h.out_offset;
kernel void matmul_heads_vec_bf16(HEADS_ARGS(ushort), uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[VEC_PARTIALS];
    HEAD_PARTS(ushort)
    matmul_vec_rows<VEC_BATCH>(head_w, head_x, head_out, bias, a, partial, group, simd, lane);
}
kernel void matmul_heads_vec_f32(HEADS_ARGS(float), uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[VEC_PARTIALS];
    HEAD_PARTS(float)
    matmul_vec_rows<VEC_BATCH>(head_w, head_x, head_out, bias, a, partial, group, simd, lane);
}
kernel void matmul_heads_mma_bf16(HEADS_ARGS(ushort), uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[MM_SHARED];
    HEAD_PARTS(ushort)
    matmul_mma_rows(head_w, head_x, head_out, bias, a, shared, group, simd, tid);
}
kernel void matmul_heads_mma_f32(HEADS_ARGS(float), uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[MM_SHARED];
    HEAD_PARTS(float)
    matmul_mma_rows(head_w, head_x, head_out, bias, a, shared, group, simd, tid);
}

struct LatentArgs {
    uint n_heads;
    uint nope;
    uint rope;
    uint rank;
    // How far apart rows of the queries (or of the stacked kv_a outputs) are.
    uint stride;
    // Rows of working memory: how far apart one head's part of it is from the next.
    uint rows;
    float eps;
};

// One threadgroup per row: rotate the rotated part of each head's query (`nope` into the head,
// in adjacent pairs, as DeepSeek lays them out), into its place after the latent part of the
// attention query (`[rows][heads][rank + rope]`), and copy each head's unrotated part to
// `q_nope`, `[heads][rows][nope]`, for matmul_heads.
kernel void latent_queries(device const float *q [[buffer(0)]], device const float *cos_table [[buffer(1)]],
                           device const float *sin_table [[buffer(2)]], device float *q_nope [[buffer(3)]],
                           device float *query [[buffer(4)]], constant LatentArgs &a [[buffer(5)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]]) {
    const uint half_rope = a.rope / 2, width = a.nope + a.rope, key = a.rank + a.rope;
    device const float *queries = q + (ulong)row * a.stride;
    device const float *c = cos_table + (ulong)row * half_rope;
    device const float *s = sin_table + (ulong)row * half_rope;
    for (uint t = tid; t < a.n_heads * half_rope; t += threads) {
        const uint head = t / half_rope, i = t % half_rope;
        device const float *pair = queries + head * width + a.nope + 2 * i;
        const float x0 = pair[0], x1 = pair[1];
        device float *to = query + ((ulong)row * a.n_heads + head) * key + a.rank + 2 * i;
        to[0] = x0 * c[i] - x1 * s[i];
        to[1] = x1 * c[i] + x0 * s[i];
    }
    for (uint t = tid; t < a.n_heads * a.nope; t += threads) {
        const uint head = t / a.nope, i = t % a.nope;
        q_nope[((ulong)head * a.rows + row) * a.nope + i] = queries[head * width + i];
    }
}

// One threadgroup of NORM_THREADS per row: normalize the latent (the first `rank` of the row's
// kv_a outputs), rotate the keys' shared part that follows it (adjacent pairs), and store both in
// the row's slot of the cache, `[slots, rank + rope]`.
kernel void latent_store(device const float *kv [[buffer(0)]], device const float *norm [[buffer(1)]],
                         device const float *cos_table [[buffer(2)]], device const float *sin_table [[buffer(3)]],
                         device const uint *write_slots [[buffer(4)]], device float *cache [[buffer(5)]],
                         constant LatentArgs &a [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[NORM_THREADS / SIMD];
    device const float *from = kv + (ulong)row * a.stride;
    const float scale = 1.0f / sqrt(sum_of_squares((device const float4 *)from, a.rank, partial, tid, simd, lane) / a.rank + a.eps);
    device float *to = cache + (ulong)write_slots[row] * (a.rank + a.rope);
    for (uint i = tid; i < a.rank; i += NORM_THREADS)
        to[i] = from[i] * scale * norm[i];
    const uint half_rope = a.rope / 2;
    device const float *c = cos_table + (ulong)row * half_rope;
    device const float *s = sin_table + (ulong)row * half_rope;
    for (uint i = tid; i < half_rope; i += NORM_THREADS) {
        const float x0 = from[a.rank + 2 * i], x1 = from[a.rank + 2 * i + 1];
        to[a.rank + 2 * i] = x0 * c[i] - x1 * s[i];
        to[a.rank + 2 * i + 1] = x1 * c[i] + x0 * s[i];
    }
}

// attention, for the latent cache: one threadgroup of ATT_SIMDS simdgroups per (head, row), the
// query `[rows][heads][rank + rope]`, every head scoring against the same cached rows and blending
// their first `rank` elements, into `out`, `[heads][rows][rank]`. As in attention, the positions
// are split into runs, one per simdgroup, combined in order at the end. Needs rank + rope to be at
// most 640 and rank at most 512, both multiples of 4, and threadgroup memory for
// ATT_SIMDS * (rank + 2) floats.
#define LATENT_KEY_VECS 5
struct LatentAttnArgs {
    uint n_heads;
    uint rank;
    uint rope;
    uint rows;
    float scale;
};
kernel void attention_latent(device const float *query [[buffer(0)]], device const float *cache [[buffer(1)]],
                             device const uint *slots [[buffer(2)]], device const uint *row_offset [[buffer(3)]],
                             device const uint *row_len [[buffer(4)]], device float *out [[buffer(5)]],
                             constant LatentAttnArgs &a [[buffer(6)]],
                             threadgroup float *shared [[threadgroup(0)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint tid [[thread_index_in_threadgroup]]) {
    const uint head = group.x, row = group.y;
    const uint width = a.rank + a.rope, key_vecs = width / 4, value_vecs = a.rank / 4;
    device const float4 *q_in = (device const float4 *)(query + ((ulong)row * a.n_heads + head) * width);
    float4 q[LATENT_KEY_VECS], o[HEAD_VECS];
    for (uint j = 0; j < LATENT_KEY_VECS; j++) {
        const uint f = lane + SIMD * j;
        q[j] = f < key_vecs ? q_in[f] : 0.0f;
    }
    for (uint j = 0; j < HEAD_VECS; j++)
        o[j] = 0.0f;
    const uint len = row_len[row];
    const uint run = (len + ATT_SIMDS - 1) / ATT_SIMDS;
    const uint start = simd * run, end = min(start + run, len);
    device const uint *row_slots = slots + row_offset[row];
    float running_max = -INFINITY, total = 0.0f;
    for (uint p = start; p < end; p += ATT_STEP) {
        // Positions past the end of the run score -infinity, which gives them no weight.
        ulong at[ATT_STEP];
        float scores[ATT_STEP];
        for (uint u = 0; u < ATT_STEP; u++)
            at[u] = (ulong)row_slots[min(p + u, end - 1)] * width;
        for (uint u = 0; u < ATT_STEP; u++) {
            device const float4 *key = (device const float4 *)(cache + at[u]);
            float partial = 0.0f;
            for (uint j = 0; j < LATENT_KEY_VECS; j++) {
                const uint f = lane + SIMD * j;
                if (f < key_vecs)
                    partial += dot(q[j], key[f]);
            }
            const float score = simd_sum(partial) * a.scale;
            scores[u] = p + u < end ? score : -INFINITY;
        }
        float new_max = running_max;
        for (uint u = 0; u < ATT_STEP; u++)
            new_max = max(new_max, scores[u]);
        const float correction = precise::exp(running_max - new_max);
        float weights[ATT_STEP];
        total *= correction;
        for (uint u = 0; u < ATT_STEP; u++) {
            weights[u] = precise::exp(scores[u] - new_max);
            total += weights[u];
        }
        for (uint j = 0; j < HEAD_VECS; j++) {
            const uint f = lane + SIMD * j;
            if (f < value_vecs) {
                float4 sum = o[j] * correction;
                for (uint u = 0; u < ATT_STEP; u++)
                    sum += weights[u] * ((device const float4 *)(cache + at[u]))[f];
                o[j] = sum;
            }
        }
        running_max = new_max;
    }
    threadgroup float *mine = shared + simd * (a.rank + 2);
    for (uint j = 0; j < HEAD_VECS; j++) {
        const uint f = lane + SIMD * j;
        if (f < value_vecs)
            ((threadgroup float4 *)(mine + 2))[f] = o[j];
    }
    if (lane == 0) {
        mine[0] = running_max;
        mine[1] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = -INFINITY;
    for (uint s = 0; s < ATT_SIMDS; s++)
        best = max(best, shared[s * (a.rank + 2)]);
    float sum = 0.0f;
    for (uint s = 0; s < ATT_SIMDS; s++) {
        threadgroup float *theirs = shared + s * (a.rank + 2);
        sum += theirs[1] * precise::exp(theirs[0] - best);
    }
    device float *to = out + ((ulong)head * a.rows + row) * a.rank;
    for (uint d = tid; d < a.rank; d += ATT_SIMDS * SIMD) {
        float value = 0.0f;
        for (uint s = 0; s < ATT_SIMDS; s++) {
            threadgroup float *theirs = shared + s * (a.rank + 2);
            value += theirs[2 + d] * precise::exp(theirs[0] - best);
        }
        to[d] = value / sum;
    }
}

// ---------------------------------------------------------------------------------------------
// EAGLE draft heads
// ---------------------------------------------------------------------------------------------

// One threadgroup per row: copy the row's `dim` floats into `out`, whose rows are `stride` apart,
// at `offset` into the row (keeping a layer's residual stream for an EAGLE head, say).
struct CopyArgs {
    uint dim;
    uint stride;
    uint offset;
};
kernel void copy_rows(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                      constant CopyArgs &a [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint threads [[threads_per_threadgroup]]) {
    for (uint i = tid; i < a.dim; i += threads)
        out[(ulong)row * a.stride + a.offset + i] = x[(ulong)row * a.dim + i];
}

// One threadgroup of NORM_THREADS per row, for an EAGLE head's layer input: the row's embedding
// normalized with `embed_norm`, then its hidden state normalized with `hidden_norm`, side by side
// in `wide` (`[rows, 2 * dim]`).
kernel void eagle_input(device const float *embeds [[buffer(0)]], device const float *hidden [[buffer(1)]],
                        device const float *embed_norm [[buffer(2)]], device const float *hidden_norm [[buffer(3)]],
                        device float *wide [[buffer(4)]], constant NormArgs &a [[buffer(5)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[NORM_THREADS / SIMD];
    device float4 *to = (device float4 *)(wide + (ulong)row * 2 * a.dim);
    for (uint half_index = 0; half_index < 2; half_index++) {
        device const float4 *from =
            (device const float4 *)((half_index == 0 ? embeds : hidden) + (ulong)row * a.dim);
        device const float4 *w = (device const float4 *)(half_index == 0 ? embed_norm : hidden_norm);
        const float scale = 1.0f / sqrt(sum_of_squares(from, a.dim, partial, tid, simd, lane) / a.dim + a.eps);
        for (uint i = tid; i < a.dim / 4; i += NORM_THREADS)
            to[half_index * a.dim / 4 + i] = from[i] * scale * w[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One threadgroup per row: an EAGLE head's scores for its own vocabulary, put in their tokens'
// places in the model's (`targets[i]` for score i), and -infinity for the tokens it doesn't have.
struct ScatterArgs {
    uint draft_vocab;
    uint vocab;
};
kernel void scatter_logits(device const float *draft [[buffer(0)]], device const uint *targets [[buffer(1)]],
                           device float *logits [[buffer(2)]], constant ScatterArgs &a [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]]) {
    device float *to = logits + (ulong)row * a.vocab;
    for (uint i = tid; i < a.vocab; i += threads)
        to[i] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_device);
    device const float *from = draft + (ulong)row * a.draft_vocab;
    for (uint i = tid; i < a.draft_vocab; i += threads)
        to[targets[i]] = from[i];
}
