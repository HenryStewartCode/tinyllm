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
// or `rows[row]` when only some rows are wanted.
struct NormArgs {
    uint dim;
    float eps;
    uint select;
};
kernel void rmsnorm(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                    device const float *weight [[buffer(2)]], device const uint *rows [[buffer(3)]],
                    constant NormArgs &a [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[NORM_THREADS / SIMD];
    device const float4 *from = (device const float4 *)(x + (ulong)(a.select ? rows[row] : row) * a.dim);
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
    device const float4 *from = (device const float4 *)(y + (ulong)row * a.dim);
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
};

// How many groups of 8 columns each run has.
inline uint run_length(uint groups) { return (groups + RANGES - 1) / RANGES; }

// The end of the sum: scale, bias, and store or add.
inline void finish_output(device float *out, device const float *bias, constant MatmulArgs &a,
                          uint b, uint m, float total) {
    float value = total * a.scale;
    if (a.has_bias)
        value += bias[m];
    device float *at = out + (ulong)b * a.m + m;
    *at = a.accumulate ? *at + value : value;
}

// For small batches: each simdgroup takes VEC_ROWS weight rows, with one lane per (row, run),
// and up to VEC_BATCH rows of x.
#define VEC_ROWS (SIMD / RANGES)
#define VEC_BATCH 8
#define VEC_STEP 4
template <typename W>
inline void matmul_vec_rows(device const W *w, device const float *x, device float *out,
                            device const float *bias, constant MatmulArgs &a, uint2 group,
                            uint simd, uint simds, uint lane) {
    const uint run = lane % RANGES;
    const uint m = (group.x * simds + simd) * VEC_ROWS + lane / RANGES;
    const uint b0 = group.y * VEC_BATCH;
    const uint rows = min((uint)VEC_BATCH, a.batch - b0);
    const uint groups = a.n / 8, length = run_length(groups);
    const uint first = min(run * length, groups), last = min(first + length, groups);
    float acc[VEC_BATCH];
    for (uint b = 0; b < VEC_BATCH; b++)
        acc[b] = 0.0f;
    // Lanes past the last weight row read it again, and don't store anything. VEC_STEP groups
    // are loaded at a time, so that more of the weights are on their way at once.
    const ulong row = (ulong)min(m, a.m - 1) * a.n;
    for (uint g = first; g < last; g += VEC_STEP) {
        const uint count = min((uint)VEC_STEP, last - g);
        float4 lo[VEC_STEP], hi[VEC_STEP];
        for (uint u = 0; u < VEC_STEP; u++)
            load8(w, row + min(g + u, last - 1) * 8, lo[u], hi[u]);
        for (uint b = 0; b < VEC_BATCH; b++) {
            if (b < rows) {
                device const float4 *xb = (device const float4 *)(x + (ulong)(b0 + b) * a.n + g * 8);
                float sum = acc[b];
                for (uint u = 0; u < VEC_STEP; u++) {
                    if (u < count) {
                        const float4 x0 = xb[2 * u], x1 = xb[2 * u + 1];
                        sum = fma(lo[u].x, x0.x, sum);
                        sum = fma(lo[u].y, x0.y, sum);
                        sum = fma(lo[u].z, x0.z, sum);
                        sum = fma(lo[u].w, x0.w, sum);
                        sum = fma(hi[u].x, x1.x, sum);
                        sum = fma(hi[u].y, x1.y, sum);
                        sum = fma(hi[u].z, x1.z, sum);
                        sum = fma(hi[u].w, x1.w, sum);
                    }
                }
                acc[b] = sum;
            }
        }
    }
    // The lane with a row's first run adds the others to it, in order.
    const uint base = lane - run;
    for (uint b = 0; b < VEC_BATCH; b++) {
        if (b < rows) {
            float total = acc[b];
            for (uint r = 1; r < RANGES; r++) {
                const float part = simd_shuffle(acc[b], base + r);
                if (r * length < groups)
                    total += part;
            }
            if (run == 0 && m < a.m)
                finish_output(out, bias, a, b0 + b, m, total);
        }
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
kernel void matmul_vec_bf16(MATMUL_ARGS(ushort), uint simds [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    matmul_vec_rows(w, x, out, bias, a, group, simd, simds, lane);
}
kernel void matmul_vec_f32(MATMUL_ARGS(float), uint simds [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    matmul_vec_rows(w, x, out, bias, a, group, simd, simds, lane);
}
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
