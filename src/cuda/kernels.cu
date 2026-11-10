// The GPU half of tinyllm's CUDA backend: one kernel for each step of the forward pass.
//
// build_kernels.sh compiles this file to PTX, NVIDIA's portable GPU assembly, which is embedded
// in the tinyllm binary. The driver turns the PTX into machine code for whichever GPU is present,
// so neither building nor running tinyllm needs the CUDA toolkit.
//
// Every kernel uses blocks of 32 threads, and a GPU runs those 32 as one unit (a "warp") in
// lockstep, so a warp is much like one 32-lane SIMD register. Having the lanes read consecutive
// elements lets the hardware fetch them as a single request, which is what most of the loops
// below are arranged to do: `for (i = lane; i < n; i += WARP)` visits every element once across
// the warp. As on the CPU, values are added up in an order that doesn't depend on how the batch
// is arranged, so a sequence gets bit-identical results whichever others it runs with.
//
// Matrices are stored row-major: row `b` of a `[batch, dim]` matrix starts at `b * dim`.

#include <cuda_bf16.h>
#include <float.h>
#include <mma.h>
#include <stddef.h>

using namespace nvcuda;

#define WARP 32
#define TILE 16
#define TILE_ELEMENTS (TILE * TILE)

// Where element (row, column) of a 16-bit tile-layout matrix is; see "16-bit weights" below.
__device__ __forceinline__ size_t tile_index(int row, int column, int chunks) {
    return ((size_t)(row / TILE) * chunks + column / TILE) * TILE_ELEMENTS + (row % TILE) * TILE +
           column % TILE;
}

// Add a value up across all the lanes of the warp. Every lane ends up with the total.
__device__ __forceinline__ float warp_sum(float v) {
    for (int step = WARP / 2; step > 0; step >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, step);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int step = WARP / 2; step > 0; step >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, step));
    return v;
}

// One block per row: x[row] = embeddings[tokens[row]] * scale.
extern "C" __global__ void embed(const float *embeddings, const int *tokens, float *x, int dim,
                                 float scale) {
    const float *from = embeddings + (size_t)tokens[blockIdx.x] * dim;
    float *to = x + (size_t)blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += WARP)
        to[i] = from[i] * scale;
}

// One block per output row: out = x / sqrt(mean(x^2) + eps) * weight. If `rows` isn't null, it
// says which row of x each output row comes from.
extern "C" __global__ void rmsnorm(const float *x, float *out, const float *weight, int dim,
                                   float eps, const int *rows) {
    const int row = blockIdx.x;
    const float *from = x + (size_t)(rows ? rows[row] : row) * dim;
    float *to = out + (size_t)row * dim;
    float sum = 0;
    for (int i = threadIdx.x; i < dim; i += WARP)
        sum += from[i] * from[i];
    const float scale = 1.0f / sqrtf(warp_sum(sum) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += WARP)
        to[i] = from[i] * scale * weight[i];
}

// Weight rows handled by each warp.
#define BLOCK_ROWS 8

// The dot products of BLOCK_ROWS weight rows with C input rows, one warp per combination.
//
// The lanes hold 32 consecutive columns, so one load fetches 32 elements. Multiplying every
// weight row by every input row then gets BLOCK_ROWS * C multiplies per lane out of
// BLOCK_ROWS + C loads, which is what keeps memory from being the limit.
//
// Blocks are numbered (input group, weight group), so that blocks launched close together use the
// same weights, which then come from the GPU's cache rather than from memory: each weight is read
// from memory about once per step, however large the batch. The last groups can be incomplete:
// the weight matrix is padded with zero rows to a whole group, the input buffers have room for
// padding rows (whose contents don't matter), and nothing is written for rows that aren't real.
template <int C>
__device__ __forceinline__ void matmul_blocks(const float *w, int m, int n, const float *x,
                                              int batch, float *out, const float *bias,
                                              int accumulate) {
    const int lane = threadIdx.x;
    const int first_m = blockIdx.y * BLOCK_ROWS;
    const int first_b = blockIdx.x * C;
    const float *w_rows = w + (size_t)first_m * n;
    const float *x_rows = x + (size_t)first_b * n;

    float acc[BLOCK_ROWS][C];
    for (int a = 0; a < BLOCK_ROWS; a++)
        for (int b = 0; b < C; b++)
            acc[a][b] = 0;

    for (int col = lane; col < n; col += WARP) {
        float wv[BLOCK_ROWS], xv[C];
        for (int a = 0; a < BLOCK_ROWS; a++)
            wv[a] = w_rows[(size_t)a * n + col];
        for (int b = 0; b < C; b++)
            xv[b] = x_rows[(size_t)b * n + col];
        for (int a = 0; a < BLOCK_ROWS; a++)
            for (int b = 0; b < C; b++)
                acc[a][b] = fmaf(wv[a], xv[b], acc[a][b]);
    }

    for (int b = 0; b < C; b++) {
        float value = 0;
        // Every lane gets every total, and lane `a` writes the total for weight row `a`,
        // so that the 8 results land next to each other in one memory request.
        for (int a = 0; a < BLOCK_ROWS; a++) {
            const float total = warp_sum(acc[a][b]);
            if (a == lane)
                value = total;
        }
        const int output_row = first_b + b, column = first_m + lane;
        if (lane < BLOCK_ROWS && output_row < batch && column < m) {
            float *target = out + (size_t)output_row * m + column;
            if (bias)
                value += bias[column];
            *target = accumulate ? *target + value : value;
        }
    }
}

// matmul_C: out[b][m] = dot(w[m], x[b]) (+ bias[m]), added to what's there if `accumulate`. The
// kernel to use is the largest C that's no more than the batch.
#define MATMUL(C)                                                                              \
    extern "C" __global__ void matmul_##C(const float *w, int m, int n, const float *x,        \
                                          int batch, float *out, const float *bias,            \
                                          int accumulate) {                                    \
        matmul_blocks<C>(w, m, n, x, batch, out, bias, accumulate);                            \
    }
MATMUL(1)
MATMUL(2)
MATMUL(4)
MATMUL(8)

// One block per row. Applies rotary position embeddings to the queries (in place) and the keys,
// then stores the keys and values in this token's KV slot. qkv holds, per row, the queries
// (n_heads * head_dim), then the keys and the values (kv_dim each).
//
// Pairs of elements are rotated by an angle that depends on the position, so the dot product of
// a query and a key depends on how far apart they are. Hugging Face checkpoints pair element `i`
// with element `i + half`.
extern "C" __global__ void rope_store(float *qkv, int stride, const float *cos_table,
                                      const float *sin_table, int n_heads, int n_kv_heads,
                                      int head_dim, const int *write_slots, float *k_pool,
                                      float *v_pool) {
    const int row = blockIdx.x, lane = threadIdx.x, half = head_dim / 2;
    const int kv_dim = n_kv_heads * head_dim;
    float *queries = qkv + (size_t)row * stride;
    const float *keys = queries + n_heads * head_dim;
    const float *values = keys + kv_dim;
    const float *c = cos_table + (size_t)row * half;
    const float *s = sin_table + (size_t)row * half;
    const size_t slot = write_slots[row];
    float *k_out = k_pool + slot * kv_dim;
    float *v_out = v_pool + slot * kv_dim;

    for (int t = lane; t < n_heads * half; t += WARP) {
        const int i = t % half;
        float *head = queries + (t / half) * head_dim;
        const float a = head[i], b = head[i + half];
        head[i] = a * c[i] - b * s[i];
        head[i + half] = b * c[i] + a * s[i];
    }
    for (int t = lane; t < n_kv_heads * half; t += WARP) {
        const int i = t % half, offset = (t / half) * head_dim;
        const float a = keys[offset + i], b = keys[offset + i + half];
        k_out[offset + i] = a * c[i] - b * s[i];
        k_out[offset + i + half] = b * c[i] + a * s[i];
    }
    for (int i = lane; i < kv_dim; i += WARP)
        v_out[i] = values[i];
}

// head_dim can be at most this many warps wide.
#define MAX_HEAD_CHUNKS 16

// `cap * tanh(score / cap)`, which some models apply to their attention scores.
__device__ __forceinline__ float soft_cap(float score, float cap) {
    return cap > 0.0f ? cap * tanhf(score / cap) : score;
}

// Causal self-attention. One block per (row, query head); a row sees the KV slots listed for it,
// or as many of the last ones as fit in `window`.
//
// With grouped-query attention, `group` query heads share each key/value head. `scratch` is
// working space with room for one float per (row, head, slot).
extern "C" __global__ void attention(const float *qkv, int stride, const float *k_pool,
                                     const float *v_pool, const int *slot_lists,
                                     const int *row_offset, const int *row_len, float *scratch,
                                     const unsigned *scratch_offset, float *out, int n_heads,
                                     int n_kv_heads, int head_dim, float scale, int window,
                                     float softcap) {
    const int team = blockIdx.x, lane = threadIdx.x;
    const int row = team / n_heads, head = team % n_heads;
    const int all = row_len[row], n = min(all, window), kv_dim = n_kv_heads * head_dim;
    const int group = n_heads / n_kv_heads;
    const int kv_head = (head / group) * head_dim;
    const int *slots = slot_lists + row_offset[row] + (all - n);
    const float *query = qkv + (size_t)row * stride + head * head_dim;
    float *scores = scratch + scratch_offset[row] + (size_t)head * all;

    // How relevant is each earlier token to this one? One lane per token.
    float best = -FLT_MAX;
    for (int i = lane; i < n; i += WARP) {
        const float *key = k_pool + (size_t)slots[i] * kv_dim + kv_head;
        float score = 0;
        for (int j = 0; j < head_dim; j++)
            score = fmaf(query[j], key[j], score);
        score = soft_cap(score * scale, softcap);
        scores[i] = score;
        best = fmaxf(best, score);
    }
    best = warp_max(best);

    // Softmax, shifted by the max for numerical stability.
    float sum = 0;
    for (int i = lane; i < n; i += WARP) {
        const float weight = expf(scores[i] - best);
        scores[i] = weight;
        sum += weight;
    }
    const float inverse = 1.0f / warp_sum(sum);
    __syncwarp(); // the next loop reads what other lanes just wrote

    // Blend their values by that relevance. Here the lanes hold consecutive elements of the
    // value vector instead, going through the tokens one by one.
    const int value_chunks = (head_dim + WARP - 1) / WARP;
    float acc[MAX_HEAD_CHUNKS];
    for (int c = 0; c < value_chunks; c++)
        acc[c] = 0;
    for (int i = 0; i < n; i++) {
        const float weight = scores[i];
        const float *value = v_pool + (size_t)slots[i] * kv_dim + kv_head;
        for (int c = 0; c < value_chunks; c++) {
            const int j = c * WARP + lane;
            if (j < head_dim)
                acc[c] = fmaf(weight, value[j], acc[c]);
        }
    }
    float *to = out + (size_t)row * n_heads * head_dim + head * head_dim;
    for (int c = 0; c < value_chunks; c++) {
        const int j = c * WARP + lane;
        if (j < head_dim)
            to[j] = acc[c] * inverse;
    }
}

// silu(x) = x * sigmoid(x)
struct Silu {
    __device__ __forceinline__ float operator()(float x) const { return x / (1.0f + expf(-x)); }
};

// gelu(x) = x * (1 + tanh(c * (x + 0.044715 * x^3))) / 2, with c = sqrt(2 / pi).
// Written with the exponential: 1 + tanh(y) = 2 / (1 + exp(-2y)).
struct Gelu {
    __device__ __forceinline__ float operator()(float x) const {
        return x / (1.0f + expf(-1.5957691f * (x + 0.044715f * x * x * x)));
    }
};

// One block per row: out = activation(gate) * up. gate_up holds, per row, the gate values and
// then the up values (`hidden` each).
template <typename Activation>
__device__ __forceinline__ void gated(const float *gate_up, float *out, int hidden) {
    const Activation activation;
    const float *gate = gate_up + (size_t)blockIdx.x * 2 * hidden;
    const float *up = gate + hidden;
    float *to = out + (size_t)blockIdx.x * hidden;
    for (int i = threadIdx.x; i < hidden; i += WARP)
        to[i] = activation(gate[i]) * up[i];
}

extern "C" __global__ void swiglu(const float *gate_up, float *out, int hidden) {
    gated<Silu>(gate_up, out, hidden);
}

extern "C" __global__ void geglu(const float *gate_up, float *out, int hidden) {
    gated<Gelu>(gate_up, out, hidden);
}

// One block per (row, head) for Gemma 3, which normalizes each head of the queries and keys.
// qkv holds, per row, the queries and then the keys, `n_heads` and `n_kv_heads` heads of them.
extern "C" __global__ void head_norm(float *qkv, int stride, int n_heads, int n_kv_heads,
                                     int head_dim, const float *q_weight, const float *k_weight,
                                     float eps) {
    const int heads = n_heads + n_kv_heads;
    const int row = blockIdx.x / heads, head = blockIdx.x % heads;
    float *x = qkv + (size_t)row * stride + head * head_dim;
    const float *weight = head < n_heads ? q_weight : k_weight;
    float sum = 0;
    for (int i = threadIdx.x; i < head_dim; i += WARP)
        sum += x[i] * x[i];
    const float scale = 1.0f / sqrtf(warp_sum(sum) / head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += WARP)
        x[i] *= scale * weight[i];
}

// ---------------------------------------------------------------------------------------------
// 16-bit weights on tensor cores
//
// Tensor cores multiply 16x16 tiles of 16-bit numbers in one instruction, keeping a 32-bit
// running total, far faster than the code above can do with 32-bit numbers. Most checkpoints
// store their weights as bfloat16 anyway, so nothing is lost there. The inputs (activations) are
// rounded to bfloat16 just for the multiplication, and everything else stays 32-bit.
//
// Tiles have to be stored contiguously for this to reach full speed: scattered over rows, a tile
// would use only a sliver of each cache line it touches. A matrix is divided into groups of 16
// rows and chunks of 16 columns, and element (row, column) lives at
// (group * chunks + chunk) * 256 + (row % 16) * 16 + (column % 16). The last row group and column
// chunk are padded with zeros.
// ---------------------------------------------------------------------------------------------

extern "C" __global__ void embed_bf16(const __nv_bfloat16 *embeddings, const int *tokens, float *x,
                                      int dim, int chunks, float scale) {
    const int token = tokens[blockIdx.x];
    float *to = x + (size_t)blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += WARP)
        to[i] = __bfloat162float(embeddings[tile_index(token, i, chunks)]) * scale;
}

// A warp waits for each load before going on, and when there are few rows there aren't many
// warps, so memory latency sets the speed unless each warp asks for several things at once.
#define LOAD_AHEAD 4

// As rmsnorm, with the output in tile layout for the next matmul_bf16.
extern "C" __global__ void rmsnorm_bf16(const float *x, __nv_bfloat16 *out, const float *weight,
                                        int dim, float eps, const int *rows, int chunks) {
    const int row = blockIdx.x;
    const float *from = x + (size_t)(rows ? rows[row] : row) * dim;
    float sum = 0;
    for (int i = threadIdx.x; i < dim; i += WARP * LOAD_AHEAD) {
        float value[LOAD_AHEAD];
        for (int j = 0; j < LOAD_AHEAD; j++)
            value[j] = i + j * WARP < dim ? from[i + j * WARP] : 0.0f;
        for (int j = 0; j < LOAD_AHEAD; j++)
            sum += value[j] * value[j];
    }
    const float scale = 1.0f / sqrtf(warp_sum(sum) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += WARP * LOAD_AHEAD) {
        float value[LOAD_AHEAD], factor[LOAD_AHEAD];
        for (int j = 0; j < LOAD_AHEAD; j++) {
            if (i + j * WARP < dim) {
                value[j] = from[i + j * WARP];
                factor[j] = weight[i + j * WARP];
            }
        }
        for (int j = 0; j < LOAD_AHEAD; j++) {
            if (i + j * WARP < dim)
                out[tile_index(row, i + j * WARP, chunks)] =
                    __float2bfloat16(value[j] * scale * factor[j]);
        }
    }
}

// One block per row: x += y / sqrt(mean(y^2) + eps) * weight, for models that normalize what
// each block adds to the residual stream.
extern "C" __global__ void rmsnorm_add(const float *y, float *x, const float *weight, int dim,
                                       float eps) {
    const float *from = y + (size_t)blockIdx.x * dim;
    float *to = x + (size_t)blockIdx.x * dim;
    float sum = 0;
    for (int i = threadIdx.x; i < dim; i += WARP * LOAD_AHEAD) {
        float value[LOAD_AHEAD];
        for (int j = 0; j < LOAD_AHEAD; j++)
            value[j] = i + j * WARP < dim ? from[i + j * WARP] : 0.0f;
        for (int j = 0; j < LOAD_AHEAD; j++)
            sum += value[j] * value[j];
    }
    const float scale = 1.0f / sqrtf(warp_sum(sum) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += WARP * LOAD_AHEAD) {
        float value[LOAD_AHEAD], factor[LOAD_AHEAD], total[LOAD_AHEAD];
        for (int j = 0; j < LOAD_AHEAD; j++) {
            if (i + j * WARP < dim) {
                value[j] = from[i + j * WARP];
                factor[j] = weight[i + j * WARP];
                total[j] = to[i + j * WARP];
            }
        }
        for (int j = 0; j < LOAD_AHEAD; j++) {
            if (i + j * WARP < dim)
                to[i + j * WARP] = total[j] + value[j] * scale * factor[j];
        }
    }
}

// Blocks of WARPS_ACROSS weight groups by TX input groups, one warp for each pairing. Warps in a
// block use the same tiles at about the same time, so they come from the cache after the first.
#define WARPS_ACROSS 4

template <int TX>
__device__ __forceinline__ void matmul_tiles(const __nv_bfloat16 *w, int m, int chunks,
                                             const __nv_bfloat16 *x, int batch, float *out,
                                             const float *bias, int accumulate, float *scratch) {
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int x_group = blockIdx.x * TX + warp / WARPS_ACROSS;
    const int w_group = blockIdx.y * WARPS_ACROSS + warp % WARPS_ACROSS;
    const int first_row = x_group * TILE, first_column = w_group * TILE;
    if (first_row >= batch || first_column >= m)
        return; // this warp only has padding to work on

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    float *target = out + (size_t)first_row * m + first_column;

    // Tensor-core loads and stores need rows at multiples of 16 elements. If the output's rows
    // aren't (a vocabulary of odd size, say), tiles go through a scratch buffer instead.
    const bool aligned = m % TILE == 0;
    float *tile = aligned ? nullptr
                          : scratch + ((size_t)x_group * ((m + TILE - 1) / TILE) + w_group) *
                                          TILE_ELEMENTS;
    if (accumulate && aligned) {
        wmma::load_matrix_sync(acc, target, m, wmma::mem_row_major);
    } else if (accumulate) {
        for (int i = lane; i < TILE_ELEMENTS; i += WARP) {
            const int column = i % TILE;
            tile[i] = first_column + column < m ? target[(size_t)(i / TILE) * m + column] : 0.0f;
        }
        __syncwarp();
        wmma::load_matrix_sync(acc, tile, TILE, wmma::mem_row_major);
    } else {
        wmma::fill_fragment(acc, 0.0f);
    }

    const __nv_bfloat16 *x_tiles = x + (size_t)x_group * chunks * TILE_ELEMENTS;
    const __nv_bfloat16 *w_tiles = w + (size_t)w_group * chunks * TILE_ELEMENTS;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[LOAD_AHEAD];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[LOAD_AHEAD];
    for (int c = 0; c < chunks; c += LOAD_AHEAD) {
        const int n = min(LOAD_AHEAD, chunks - c);
        for (int i = 0; i < n; i++) {
            wmma::load_matrix_sync(a[i], x_tiles + (size_t)(c + i) * TILE_ELEMENTS, TILE);
            wmma::load_matrix_sync(b[i], w_tiles + (size_t)(c + i) * TILE_ELEMENTS, TILE);
        }
        for (int i = 0; i < n; i++)
            wmma::mma_sync(acc, a[i], b[i], acc);
    }

    if (bias) {
        // The same bias values go on every row, which is what a stride of zero reads.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> offsets;
        wmma::load_matrix_sync(offsets, bias + first_column, 0, wmma::mem_row_major);
        for (int i = 0; i < 8; i++)
            acc.x[i] += offsets.x[i];
    }

    if (aligned) {
        wmma::store_matrix_sync(target, acc, m, wmma::mem_row_major);
        return;
    }
    wmma::store_matrix_sync(tile, acc, TILE, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < TILE_ELEMENTS; i += WARP) {
        const int column = i % TILE;
        if (first_column + column < m)
            target[(size_t)(i / TILE) * m + column] = tile[i];
    }
}

// matmul_bf16_TX: out[b][m] = dot(w[m], x[b]) (+ bias[m]), added to what's there if `accumulate`.
// `scratch` is only used when m isn't a multiple of 16, and then needs room for a 16x16 tile for
// every tile of the output.
#define MATMUL_BF16(TX)                                                                        \
    extern "C" __global__ void matmul_bf16_##TX(const __nv_bfloat16 *w, int m, int chunks,     \
                                                const __nv_bfloat16 *x, int batch, float *out, \
                                                const float *bias, int accumulate,             \
                                                float *scratch) {                              \
        matmul_tiles<TX>(w, m, chunks, x, batch, out, bias, accumulate, scratch);              \
    }
MATMUL_BF16(1)
MATMUL_BF16(2)
MATMUL_BF16(4)

// ---------------------------------------------------------------------------------------------
// Mixtures of experts
// ---------------------------------------------------------------------------------------------
// A token goes through the few experts its router scores best. On the GPU that's done in steps
// whose launches don't depend on which experts were chosen, so a recorded graph can replay them:
// moe_route picks each row's experts, moe_sort lays out the (row, expert) pairs grouped by expert
// in runs padded to whole 16-row groups, moe_gather copies the rows into that order, the grouped
// matmul multiplies each group by its expert's weights, and moe_combine adds each row's results
// up, in order of expert. Kernels launched for more groups than there turn out to be stop early.
#define MAX_TOP_K 16
#define MAX_EXPERTS 512

// One warp per row: the softmax of the experts' scores, the top_k best (the lower-numbered of any
// tie), and their weights, renormalized to add up to 1 if asked. Written in order of expert.
extern "C" __global__ void moe_route(const float *scores, int experts, int top_k, int renormalize,
                                     int *topk_ids, float *topk_weights) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const float *s = scores + (size_t)row * experts;
    float max = -FLT_MAX;
    for (int i = lane; i < experts; i += WARP)
        max = fmaxf(max, s[i]);
    max = warp_max(max);
    float sum = 0.0f;
    for (int i = lane; i < experts; i += WARP)
        sum += expf(s[i] - max);
    sum = warp_sum(sum);
    int chosen[MAX_TOP_K];
    float weight[MAX_TOP_K];
    float total = 0.0f;
    for (int k = 0; k < top_k; k++) {
        float best = -FLT_MAX;
        int best_i = 0x7fffffff;
        for (int i = lane; i < experts; i += WARP) {
            bool taken = false;
            for (int j = 0; j < k; j++)
                taken |= chosen[j] == i;
            if (!taken && (s[i] > best || (s[i] == best && i < best_i))) {
                best = s[i];
                best_i = i;
            }
        }
        for (int step = WARP / 2; step > 0; step >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, step);
            const int other_i = __shfl_xor_sync(0xffffffffu, best_i, step);
            if (other > best || (other == best && other_i < best_i)) {
                best = other;
                best_i = other_i;
            }
        }
        chosen[k] = best_i;
        weight[k] = expf(best - max) / sum;
        total += weight[k];
    }
    if (lane != 0)
        return;
    for (int k = 1; k < top_k; k++) {
        for (int j = k; j > 0 && chosen[j - 1] > chosen[j]; j--) {
            const int id = chosen[j];
            chosen[j] = chosen[j - 1];
            chosen[j - 1] = id;
            const float w = weight[j];
            weight[j] = weight[j - 1];
            weight[j - 1] = w;
        }
    }
    for (int k = 0; k < top_k; k++) {
        topk_ids[row * top_k + k] = chosen[k];
        topk_weights[row * top_k + k] = renormalize ? weight[k] / total : weight[k];
    }
}

// One block: lay out the (row, expert) pairs grouped by expert, each expert's run padded to a
// whole number of 16-row groups. Each thread takes some experts and finds their pairs in order,
// so the layout is the same every time. `slot_of[pair]` is where a pair went; `sorted_rows[slot]`
// is its row (or -1 for padding); `group_expert[group]` is each group's expert.
extern "C" __global__ void moe_sort(const int *topk_ids, int rows, int top_k, int experts,
                                    int *sorted_rows, int *slot_of, int *group_expert,
                                    int *num_groups) {
    __shared__ int counts[MAX_EXPERTS];
    __shared__ int starts[MAX_EXPERTS + 1];
    const int pairs = rows * top_k;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        int count = 0;
        for (int i = 0; i < pairs; i++)
            count += topk_ids[i] == e;
        counts[e] = count;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int at = 0;
        for (int e = 0; e < experts; e++) {
            starts[e] = at;
            at += (counts[e] + TILE - 1) / TILE * TILE;
        }
        starts[experts] = at;
        *num_groups = at / TILE;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        int at = starts[e];
        for (int i = 0; i < pairs; i++) {
            if (topk_ids[i] == e) {
                sorted_rows[at] = i / top_k;
                slot_of[i] = at;
                at++;
            }
        }
        for (; at < starts[e + 1]; at++)
            sorted_rows[at] = -1;
        for (int g = starts[e] / TILE; g < starts[e + 1] / TILE; g++)
            group_expert[g] = e;
    }
}

// One block per slot: copy its row of the 16-bit tile-layout input (zeros for padding).
extern "C" __global__ void moe_gather(const __nv_bfloat16 *x, int chunks, const int *sorted_rows,
                                      const int *num_groups, __nv_bfloat16 *out) {
    const int slot = blockIdx.x;
    if (slot >= *num_groups * TILE)
        return;
    const int row = sorted_rows[slot];
    for (int c = threadIdx.x; c < chunks * TILE; c += WARP)
        out[tile_index(slot, c, chunks)] =
            row < 0 ? __float2bfloat16(0.0f) : x[tile_index(row, c, chunks)];
}

// moe_matmul_TX: out = W_e x for each 16-row group of `x`, with `e` that group's expert and the
// experts' weight matrices `expert_stride` elements apart, all in the 16-bit tile layout.
#define MOE_MATMUL(TX)                                                                         \
    extern "C" __global__ void moe_matmul_##TX(                                                \
        const __nv_bfloat16 *w, unsigned long long expert_stride, const int *group_expert,    \
        const int *num_groups, int m, int chunks, const __nv_bfloat16 *x, float *out) {       \
        const int warp = threadIdx.x / WARP;                                                   \
        const int x_group = blockIdx.x * TX + warp / WARPS_ACROSS;                             \
        const int groups = *num_groups;                                                        \
        if (x_group >= groups)                                                                 \
            return;                                                                            \
        matmul_tiles<TX>(w + group_expert[x_group] * expert_stride, m, chunks, x,              \
                         groups * TILE, out, nullptr, 0, nullptr);                             \
    }
MOE_MATMUL(1)
MOE_MATMUL(2)
MOE_MATMUL(4)

// One block per row: add up what its experts made of it, weighted, in order of expert, and add
// that to the residual stream.
extern "C" __global__ void moe_combine(const float *down, int dim, const int *slot_of,
                                       const float *topk_weights, int top_k, float *x) {
    const int row = blockIdx.x;
    for (int c = threadIdx.x; c < dim; c += WARP) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; k++)
            sum += topk_weights[row * top_k + k] *
                   down[(size_t)slot_of[row * top_k + k] * dim + c];
        x[(size_t)row * dim + c] += sum;
    }
}

// ---------------------------------------------------------------------------------------------
// 8-bit weights
// ---------------------------------------------------------------------------------------------
// With --quantization int8, each layer's matrices hold one signed byte per weight and a scale for
// each group of 128 weights along a row: weight = byte * scale. Bytes are exact in bfloat16, so
// they're widened in registers and multiplied on the tensor cores, and each group's sums are
// multiplied by its scale as they're added up: half the bytes of 16-bit weights to read per
// step, for a little rounding.
//
// This uses the tensor cores' own instruction (mma.m16n8k16) rather than wmma, because its
// register layout is documented: the bytes are stored in exactly the order each lane needs them,
// so a lane reads its 8 bytes of a 16x16 tile in one load, with no trip through shared memory.
// For the tile of weight rows n = 0..15 and columns k = 0..15, lane l (g = l / 4, t = l % 4)
// holds, for each half h of the rows, row n = 8h + g at columns 2t, 2t+1, 2t+8 and 2t+9.
#define I8_LOADS 8 // chunks of 16 columns: one group of 128

__device__ __forceinline__ unsigned bf16_pair(float low, float high) {
    __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
    return *reinterpret_cast<unsigned *>(&pair);
}

// acc += A (16x16, row-major) * B (16x8, column-major), with bfloat16 inputs.
__device__ __forceinline__ void mma_bf16(float *acc, const unsigned *a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A weight byte's value as bfloat16 bits: a signed integer, or (FP8) a float with 4 exponent and
// 3 mantissa bits, as in FP8 checkpoints. Either way it's exact in bfloat16, so these are just
// the top halves of the f32s, built with integer operations.
template <bool FP8> __device__ __forceinline__ unsigned bf16_bits(unsigned b);
template <> __device__ __forceinline__ unsigned bf16_bits<false>(unsigned b) {
    return __float_as_uint((float)(signed char)b) >> 16;
}
template <> __device__ __forceinline__ unsigned bf16_bits<true>(unsigned b) {
    const unsigned rest = b & 0x7f;
    // Normal numbers just move their exponent from a bias of 7 to 127; ones below normal are
    // mantissa * 2^-9.
    const unsigned magnitude = rest >= 8 ? (rest << 4) + (120u << 7)
                                         : __float_as_uint((float)rest * (1.0f / 512.0f)) >> 16;
    return magnitude | ((b & 0x80) << 8);
}

template <bool FP8> __device__ __forceinline__ unsigned bf16_pair_of(unsigned low, unsigned high) {
    return bf16_bits<FP8>(low) | (bf16_bits<FP8>(high) << 16);
}

template <int TX, bool FP8>
__device__ __forceinline__ void matmul_tiles_q8(const unsigned char *w, const float *scales, int m,
                                                int chunks, const __nv_bfloat16 *x, int batch,
                                                float *out, const float *bias, int accumulate) {
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int x_group = blockIdx.x * TX + warp / WARPS_ACROSS;
    const int w_group = blockIdx.y * WARPS_ACROSS + warp % WARPS_ACROSS;
    const int first_row = x_group * TILE, first_column = w_group * TILE;
    if (first_row >= batch || first_column >= m)
        return; // this warp only has padding to work on
    const int g = lane / 4, t = lane % 4;

    // Inputs are 16-bit tiles; as pairs of values, row r of a tile starts at word 8r.
    const unsigned *x_words =
        reinterpret_cast<const unsigned *>(x + (size_t)x_group * chunks * TILE_ELEMENTS);
    const int2 *w_lanes =
        reinterpret_cast<const int2 *>(w + (size_t)w_group * chunks * TILE_ELEMENTS) + lane;
    const int groups = (chunks + I8_LOADS - 1) / I8_LOADS;
    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    for (int c = 0; c < chunks; c += I8_LOADS) {
        const int n = min(I8_LOADS, chunks - c);
        float part[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        int2 bytes[I8_LOADS];
        unsigned a[I8_LOADS][4];
#pragma unroll
        for (int i = 0; i < I8_LOADS; i++) {
            if (i < n) {
                bytes[i] = w_lanes[(size_t)(c + i) * WARP];
                const unsigned *tile = x_words + (size_t)(c + i) * (TILE_ELEMENTS / 2);
                a[i][0] = tile[g * 8 + t];
                a[i][1] = tile[(g + 8) * 8 + t];
                a[i][2] = tile[g * 8 + t + 4];
                a[i][3] = tile[(g + 8) * 8 + t + 4];
            }
        }
#pragma unroll
        for (int i = 0; i < I8_LOADS; i++) {
            if (i < n) {
                const unsigned low = bytes[i].x, high = bytes[i].y;
                mma_bf16(part[0], a[i], bf16_pair_of<FP8>(low & 255, (low >> 8) & 255),
                         bf16_pair_of<FP8>((low >> 16) & 255, low >> 24));
                mma_bf16(part[1], a[i], bf16_pair_of<FP8>(high & 255, (high >> 8) & 255),
                         bf16_pair_of<FP8>((high >> 16) & 255, high >> 24));
            }
        }
        // This lane's weight rows are 8h + 2t and 8h + 2t + 1 of the warp's 16.
        const int group = c / I8_LOADS;
        for (int h = 0; h < 2; h++) {
            for (int j = 0; j < 2; j++) {
                const int row = min(first_column + h * 8 + 2 * t + j, m - 1);
                const float scale = scales[(size_t)row * groups + group];
                acc[h][j] += part[h][j] * scale;
                acc[h][j + 2] += part[h][j + 2] * scale;
            }
        }
    }

    // Lane (g, t) has rows g and g + 8, and columns 2t and 2t + 1 of each half.
    for (int h = 0; h < 2; h++) {
        for (int j = 0; j < 4; j++) {
            const int row = first_row + g + (j / 2) * 8;
            const int column = first_column + h * 8 + 2 * t + j % 2;
            if (column >= m)
                continue;
            float value = acc[h][j];
            if (bias)
                value += bias[column];
            float *at = out + (size_t)row * m + column;
            *at = accumulate ? *at + value : value;
        }
    }
}

// matmul_i8_TX: out[b][m] = dot(w[m], x[b]) (+ bias[m]), added to what's there if `accumulate`,
// with 8-bit weights in the lane order above and `scales: [m, groups of 128 columns]`.
#define MATMUL_Q8(NAME, TX, FP8)                                                               \
    extern "C" __global__ void NAME##_##TX(const unsigned char *w, const float *scales, int m,   \
                                           int chunks, const __nv_bfloat16 *x, int batch,      \
                                           float *out, const float *bias, int accumulate) {    \
        matmul_tiles_q8<TX, FP8>(w, scales, m, chunks, x, batch, out, bias, accumulate);       \
    }
MATMUL_Q8(matmul_i8, 1, false)
MATMUL_Q8(matmul_i8, 2, false)
MATMUL_Q8(matmul_i8, 4, false)
MATMUL_Q8(matmul_f8, 1, true)
MATMUL_Q8(matmul_f8, 2, true)
MATMUL_Q8(matmul_f8, 4, true)

// One block per row, as rope_store, with the keys and values stored 16-bit in the tile layout:
// the KV pools are `[slots, kv_dim]` matrices, so a page of 16 slots is one group of tiles.
extern "C" __global__ void rope_store_bf16(float *qkv, int stride, const float *cos_table,
                                           const float *sin_table, int n_heads, int n_kv_heads,
                                           int head_dim, const int *write_slots,
                                           __nv_bfloat16 *k_pool, __nv_bfloat16 *v_pool,
                                           int kv_chunks) {
    const int row = blockIdx.x, lane = threadIdx.x, half = head_dim / 2;
    const int kv_dim = n_kv_heads * head_dim;
    float *queries = qkv + (size_t)row * stride;
    const float *keys = queries + n_heads * head_dim;
    const float *values = keys + kv_dim;
    const float *c = cos_table + (size_t)row * half;
    const float *s = sin_table + (size_t)row * half;
    const int slot = write_slots[row];

    for (int t = lane; t < n_heads * half; t += WARP) {
        const int i = t % half;
        float *head = queries + (t / half) * head_dim;
        const float a = head[i], b = head[i + half];
        head[i] = a * c[i] - b * s[i];
        head[i + half] = b * c[i] + a * s[i];
    }
    for (int t = lane; t < n_kv_heads * half; t += WARP) {
        const int i = t % half, offset = (t / half) * head_dim;
        const float a = keys[offset + i], b = keys[offset + i + half];
        k_pool[tile_index(slot, offset + i, kv_chunks)] = __float2bfloat16(a * c[i] - b * s[i]);
        k_pool[tile_index(slot, offset + i + half, kv_chunks)] =
            __float2bfloat16(b * c[i] + a * s[i]);
    }
    for (int i = lane; i < kv_dim; i += WARP)
        v_pool[tile_index(slot, i, kv_chunks)] = __float2bfloat16(values[i]);
}

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> TileA;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> TileBColumns;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> TileBRows;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> TileTotal;

// Causal self-attention on tensor cores. One block (one warp) per (row, key/value head).
//
// Consecutive rows from one sequence read nearly the same pages, and a tile has room for more
// query heads than one row has per key/value head, so rows can share a block. `team_sizes[row]`
// is how many rows (this one and those following it) this row answers for, and is 0 if another
// row's block is handling this one. Tile rows are then the query heads of the team's rows in
// turn, and a tile multiplication gives the scores of a whole page of 16 tokens at once for all
// of them.
//
// Fetching keys and values from memory is what limits long prompts, so they're each fetched only
// once, in one go. That means working out the softmax in a single pass over the pages, which
// needs each token's weight before the best score is known. Weights are relative to the best
// score in the first page instead, and that works because only ratios between weights matter:
// dividing by the sum of the weights at the end undoes any common factor. Weights can then be
// large if a later page scores much higher, so they're capped well short of overflowing.
//
// `row_len[row]` is how many tokens the row can see, whose pages start at
// page_lists[row_offset[row]]. Rows in a team are consecutive tokens, so they each see one more
// token than the row before. With a `window` a row only sees that many of the latest tokens, and
// pages before the first that any row can see are skipped. `smem` needs room for the query tiles (2 bytes * 256 * chunks) and
// two working tiles, a score tile (4 bytes * 256) and a weight tile (2 bytes * 256).
#define LARGEST_ATTENTION_EXPONENT 60.0f

template <int CHUNKS>
__device__ __forceinline__ void attention_tiles(const float *qkv, int stride,
                                                const __nv_bfloat16 *k_pool,
                                                const __nv_bfloat16 *v_pool, int kv_chunks,
                                                const unsigned *page_lists, const int *row_offset,
                                                const int *row_len, const int *team_sizes,
                                                __nv_bfloat16 *out, int n_heads, int n_kv_heads,
                                                int head_dim, float scale, int out_chunks,
                                                int window, float softcap) {
    extern __shared__ __align__(1024) unsigned char smem[];
    const int block = blockIdx.x, lane = threadIdx.x;
    const int first_row = block / n_kv_heads, kv_head = block % n_kv_heads;
    const int team_size = team_sizes[first_row];
    if (team_size == 0)
        return;
    const int group = n_heads / n_kv_heads;
    // Tile rows are the query heads of this team's rows, one row's heads after another.
    const int tile_rows = team_size * group;
    const int first_len = row_len[first_row];
    const int first_page = max(0, first_len - window) / TILE;
    const int last_page = (first_len + team_size - 2) / TILE;
    const unsigned *pages = page_lists + row_offset[first_row];
    const size_t tile_bytes = TILE_ELEMENTS * sizeof(__nv_bfloat16);

    __nv_bfloat16 *q_tiles = (__nv_bfloat16 *)smem;
    float *scores = (float *)(smem + CHUNKS * tile_bytes);
    __nv_bfloat16 *weights = (__nv_bfloat16 *)(smem + CHUNKS * tile_bytes + 2 * tile_bytes);

    // Lay the queries out as tiles (rows: query heads; columns: 16 elements of the head).
    for (int c = 0; c < CHUNKS; c++) {
        for (int i = lane; i < TILE_ELEMENTS; i += WARP) {
            const int t = i / TILE, j = i % TILE;
            float value = 0.0f;
            if (t < tile_rows) {
                const float *query = qkv + (size_t)(first_row + t / group) * stride +
                                     (kv_head * group + t % group) * head_dim;
                value = query[c * TILE + j];
            }
            q_tiles[c * TILE_ELEMENTS + i] = __float2bfloat16(value);
        }
    }
    __syncwarp();
    TileA a[CHUNKS];
    for (int c = 0; c < CHUNKS; c++)
        wmma::load_matrix_sync(a[c], q_tiles + c * TILE_ELEMENTS, TILE);

    // Whether a tile row's query can see the token in this lane's column of a page.
    const auto visible = [&](int t, int page) {
        const int position = page * TILE + lane, seen = first_len + t / group;
        return position < seen && position >= seen - window;
    };
    // The first page a tile row's query can see anything in.
    const auto query_first_page = [&](int t) {
        return max(0, first_len + t / group - window) / TILE;
    };

    // The reference point for each query's weights (the best score in its first page, as raw
    // scores and as soft-capped ones), and the running total of its weights.
    float reference[TILE], capped_reference[TILE], lane_sum[TILE];
    for (int t = 0; t < tile_rows; t++)
        lane_sum[t] = 0;
    for (int i = lane; i < TILE_ELEMENTS; i += WARP)
        weights[i] = __float2bfloat16(0.0f);
    __syncwarp();
    TileTotal totals[CHUNKS];
    for (int c = 0; c < CHUNKS; c++)
        wmma::fill_fragment(totals[c], 0.0f);

    for (int page = first_page; page <= last_page; page++) {
        const size_t first = ((size_t)pages[page] * kv_chunks + kv_head * CHUNKS) * TILE_ELEMENTS;
        TileBColumns keys[CHUNKS];
        TileBRows values[CHUNKS];
        for (int c = 0; c < CHUNKS; c++) {
            wmma::load_matrix_sync(keys[c], k_pool + first + (size_t)c * TILE_ELEMENTS, TILE);
            wmma::load_matrix_sync(values[c], v_pool + first + (size_t)c * TILE_ELEMENTS, TILE);
        }
        TileTotal total;
        wmma::fill_fragment(total, 0.0f);
        for (int c = 0; c < CHUNKS; c++)
            wmma::mma_sync(total, a[c], keys[c], total);
        wmma::store_matrix_sync(scores, total, TILE, wmma::mem_row_major);
        __syncwarp();

        for (int t = 0; t < tile_rows; t++) {
            if (page != query_first_page(t))
                continue;
            float best = -FLT_MAX;
            if (lane < TILE && visible(t, page))
                best = scores[t * TILE + lane];
            reference[t] = warp_max(best);
            capped_reference[t] = soft_cap(reference[t] * scale, softcap);
        }
        if (lane < TILE) {
            for (int t = 0; t < tile_rows; t++) {
                float weight = 0.0f;
                if (visible(t, page)) {
                    const float score = scores[t * TILE + lane];
                    const float exponent =
                        softcap > 0.0f ? soft_cap(score * scale, softcap) - capped_reference[t]
                                       : (score - reference[t]) * scale;
                    weight = expf(fminf(exponent, LARGEST_ATTENTION_EXPONENT));
                }
                // Summing the rounded weight the multiplication really uses keeps the result a
                // true average, so a dominant token isn't scaled by its weight's rounding error.
                const __nv_bfloat16 rounded = __float2bfloat16(weight);
                weights[t * TILE + lane] = rounded;
                lane_sum[t] += __bfloat162float(rounded);
            }
        }
        __syncwarp();
        TileA weight_tile;
        wmma::load_matrix_sync(weight_tile, weights, TILE);
        for (int c = 0; c < CHUNKS; c++)
            wmma::mma_sync(totals[c], weight_tile, values[c], totals[c]);
    }

    // Divide by the total weight and write out, rounded for the next multiplication.
    float inverse[TILE];
    for (int t = 0; t < tile_rows; t++)
        inverse[t] = 1.0f / warp_sum(lane_sum[t]);
    for (int c = 0; c < CHUNKS; c++) {
        wmma::store_matrix_sync(scores, totals[c], TILE, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < TILE_ELEMENTS; i += WARP) {
            const int t = i / TILE, j = i % TILE;
            if (t < tile_rows) {
                const int row = first_row + t / group, head = kv_head * group + t % group;
                out[tile_index(row, head * head_dim + c * TILE + j, out_chunks)] =
                    __float2bfloat16(scores[i] * inverse[t]);
            }
        }
    }
}

// One kernel per head size, so the compiler can see how many tiles there are and keep them in
// registers. The head is CHUNKS * 16 elements long.
#define ATTENTION_BF16(CHUNKS)                                                                 \
    extern "C" __global__ void attention_bf16_##CHUNKS(                                        \
        const float *qkv, int stride, const __nv_bfloat16 *k_pool, const __nv_bfloat16 *v_pool, \
        int kv_chunks, const unsigned *page_lists, const int *row_offset, const int *row_len,  \
        const int *team_sizes, __nv_bfloat16 *out, int n_heads, int n_kv_heads, int head_dim,  \
        float scale, int out_chunks, int window, float softcap) {                              \
        attention_tiles<CHUNKS>(qkv, stride, k_pool, v_pool, kv_chunks, page_lists, row_offset, \
                                row_len, team_sizes, out, n_heads, n_kv_heads, head_dim, scale, \
                                out_chunks, window, softcap);                                  \
    }
ATTENTION_BF16(1)
ATTENTION_BF16(2)
ATTENTION_BF16(3)
ATTENTION_BF16(4)
ATTENTION_BF16(5)
ATTENTION_BF16(6)
ATTENTION_BF16(7)
ATTENTION_BF16(8)
ATTENTION_BF16(16)

// As gated, with the output in tile layout. One block per SWIGLU_COLUMNS columns of a row.
#define SWIGLU_COLUMNS 256

template <typename Activation>
__device__ __forceinline__ void gated_tiles(const float *gate_up, __nv_bfloat16 *out, int hidden,
                                            int chunks) {
    const Activation activation;
    const int row = blockIdx.x, first = blockIdx.y * SWIGLU_COLUMNS;
    const int end = min(hidden, first + SWIGLU_COLUMNS);
    const float *gate = gate_up + (size_t)row * 2 * hidden;
    const float *up = gate + hidden;
    for (int i = first + threadIdx.x; i < end; i += WARP * LOAD_AHEAD) {
        float g[LOAD_AHEAD], u[LOAD_AHEAD];
        for (int j = 0; j < LOAD_AHEAD; j++) {
            if (i + j * WARP < end) {
                g[j] = gate[i + j * WARP];
                u[j] = up[i + j * WARP];
            }
        }
        for (int j = 0; j < LOAD_AHEAD; j++) {
            if (i + j * WARP < end)
                out[tile_index(row, i + j * WARP, chunks)] =
                    __float2bfloat16(activation(g[j]) * u[j]);
        }
    }
}

extern "C" __global__ void swiglu_bf16(const float *gate_up, __nv_bfloat16 *out, int hidden,
                                       int chunks) {
    gated_tiles<Silu>(gate_up, out, hidden, chunks);
}

extern "C" __global__ void geglu_bf16(const float *gate_up, __nv_bfloat16 *out, int hidden,
                                      int chunks) {
    gated_tiles<Gelu>(gate_up, out, hidden, chunks);
}

// Like swiglu_bf16 and geglu_bf16, for the slots of a mixture of experts.
template <typename Activation>
__device__ __forceinline__ void moe_gated(const float *gate_up, __nv_bfloat16 *out, int hidden,
                                          int chunks, const int *num_groups) {
    if (blockIdx.x >= *num_groups * TILE)
        return;
    gated_tiles<Activation>(gate_up, out, hidden, chunks);
}

extern "C" __global__ void moe_swiglu(const float *gate_up, __nv_bfloat16 *out, int hidden,
                                      int chunks, const int *num_groups) {
    moe_gated<Silu>(gate_up, out, hidden, chunks, num_groups);
}

extern "C" __global__ void moe_geglu(const float *gate_up, __nv_bfloat16 *out, int hidden,
                                     int chunks, const int *num_groups) {
    moe_gated<Gelu>(gate_up, out, hidden, chunks, num_groups);
}

// ---------------------------------------------------------------------------------------------
// Choosing candidates on the GPU
//
// Choosing the next token needs a few tokens' scores, not all of them: anything far behind the
// top score has essentially no chance of being picked. Reducing a row of (say) 150,000 logits to
// the tokens in contention here saves moving them all to the CPU and scanning them there.
// ---------------------------------------------------------------------------------------------

// A vocabulary is a lot for one warp to get through (150,000 tokens, 32 at a time), and scanning
// memory one load at a time would be slow, so these kernels use one big block per row of logits,
// each of whose threads handles every SCAN_THREADS-th token and asks for several at once.
#define SCAN_WARPS WARP
#define SCAN_THREADS (SCAN_WARPS * WARP)
#define SCAN_AHEAD 8

// One big block per row of logits: the last adjustments some models make to their scores.
extern "C" __global__ void finish_logits(float *logits, int vocab, float divisor, float softcap) {
    float *row = logits + (size_t)blockIdx.x * vocab;
    for (int i = threadIdx.x; i < vocab; i += SCAN_THREADS)
        row[i] = soft_cap(row[i] / divisor, softcap);
}

// Calls `visit(score, token)` for this thread's share of a row of scores.
template <typename Visit>
__device__ __forceinline__ void for_each_score(const float *scores, int vocab, Visit visit) {
    for (int i = threadIdx.x; i < vocab; i += SCAN_THREADS * SCAN_AHEAD) {
        float loaded[SCAN_AHEAD];
        for (int j = 0; j < SCAN_AHEAD; j++) {
            const int token = i + j * SCAN_THREADS;
            loaded[j] = token < vocab ? scores[token] : 0.0f;
        }
        for (int j = 0; j < SCAN_AHEAD; j++) {
            const int token = i + j * SCAN_THREADS;
            if (token < vocab)
                visit(loaded[j], token);
        }
    }
}

// Writes every token within `gaps[row]` of the row's best into values/ids (`capacity` entries per
// row), and how many there were into `results[2 * row]`, which can exceed `capacity` if there
// wasn't room. A negative gap skips the row.
//
// If `limits[row]` is positive, only about that many are needed (the sampler's top-k). Counting
// how many tokens fall within a few distances of the best finds a tighter cutoff that still keeps
// at least that many, cheaply, without sorting.
#define GAP_LEVELS 7

__device__ __forceinline__ int warp_sum_int(int v) {
    for (int step = WARP / 2; step > 0; step >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, step);
    return v;
}

extern "C" __global__ void top_candidates(const float *logits, int vocab, const float *gaps,
                                          const int *limits, unsigned *results, float *values,
                                          unsigned *ids, int capacity) {
    __shared__ float best_of_warp[SCAN_WARPS];
    __shared__ int within_of_warp[GAP_LEVELS][SCAN_WARPS];
    const int row = blockIdx.x, warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const float gap = gaps[row];
    if (gap < 0.0f)
        return;
    const float *scores = logits + (size_t)row * vocab;

    float best = -FLT_MAX;
    for_each_score(scores, vocab, [&](float score, int) { best = fmaxf(best, score); });
    best = warp_max(best);
    if (lane == 0)
        best_of_warp[warp] = best;
    __syncthreads();
    best = warp_max(best_of_warp[lane]);

    float cutoff = best - gap;
    if (limits[row] > 0) {
        int within[GAP_LEVELS];
        for (int j = 0; j < GAP_LEVELS; j++)
            within[j] = 0;
        for_each_score(scores, vocab, [&](float score, int) {
            float level = 0.5f;
            for (int j = 0; j < GAP_LEVELS; j++) {
                within[j] += score >= best - level ? 1 : 0;
                level *= 2.0f;
            }
        });
        for (int j = 0; j < GAP_LEVELS; j++) {
            const int total = warp_sum_int(within[j]);
            if (lane == 0)
                within_of_warp[j][warp] = total;
        }
        __syncthreads();
        float level = 0.5f;
        for (int j = 0; j < GAP_LEVELS; j++) {
            if (warp_sum_int(within_of_warp[j][lane]) >= limits[row]) {
                cutoff = fmaxf(cutoff, best - level);
                break;
            }
            level *= 2.0f;
        }
    }

    unsigned *count = results + (size_t)row * 2;
    if (threadIdx.x == 0)
        *count = 0;
    __threadfence();
    __syncthreads();
    float *value_slots = values + (size_t)row * capacity;
    unsigned *id_slots = ids + (size_t)row * capacity;
    for_each_score(scores, vocab, [&](float score, int token) {
        if (score >= cutoff) {
            const unsigned at = atomicAdd(count, 1u);
            if (at < (unsigned)capacity) {
                value_slots[at] = score;
                id_slots[at] = token;
            }
        }
    });
}

// The MurmurHash3 finalizer: a bijection on 32 bits in which every input bit affects every output
// bit, which is all the randomness one input number needs here.
__device__ __forceinline__ unsigned mix32(unsigned x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

// Keep the larger of two (value, token) pairs, or the lower token if the values are the same.
__device__ __forceinline__ void keep_best(float &value, int &token, float other_value,
                                          int other_token) {
    if (other_value > value || (other_value == value && other_token < token)) {
        value = other_value;
        token = other_token;
    }
}

__device__ __forceinline__ void warp_best(float &value, int &token) {
    for (int step = WARP / 2; step > 0; step >>= 1) {
        const float other_value = __shfl_xor_sync(0xffffffffu, value, step);
        const int other_token = __shfl_xor_sync(0xffffffffu, token, step);
        keep_best(value, token, other_value, other_token);
    }
}

// Draw a token from each row, with probability proportional to exp(logit / temperature), writing
// it to results[2 * row + 1]. A temperature that isn't positive skips the row.
//
// Adding independent Gumbel noise to every token's logit / temperature and taking the largest is
// the same as drawing from that distribution (the "Gumbel-max" trick). Each token's noise depends
// only on the row's key and the token's id, so the same key and logits always draw the same token,
// whatever else is running, and a row costs one pass over its scores rather than sorting or
// moving them. Noise from 23 random bits can't lift a token by more than about 16.6, so anything
// that far behind the best (one part in 16 million) never gets picked.
extern "C" __global__ void draw_tokens(const float *logits, int vocab, const float *temperatures,
                                       const unsigned *keys, unsigned *results) {
    __shared__ float value_of_warp[SCAN_WARPS];
    __shared__ int token_of_warp[SCAN_WARPS];
    const int row = blockIdx.x, warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const float temperature = temperatures[row];
    if (temperature <= 0.0f)
        return;
    const float *scores = logits + (size_t)row * vocab;
    const unsigned key_low = keys[row * 2], key_high = keys[row * 2 + 1];
    const float inverse = 1.0f / temperature;

    float best = -FLT_MAX;
    int best_token = 0;
    for_each_score(scores, vocab, [&](float score, int token) {
        const unsigned bits = mix32(mix32((unsigned)token ^ key_low) ^ key_high);
        // Halfway between multiples of 2^-23, so neither end of (0, 1) can come up.
        const float uniform = ((float)(bits >> 9) + 0.5f) / 8388608.0f;
        keep_best(best, best_token, score * inverse - logf(-logf(uniform)), token);
    });
    warp_best(best, best_token);
    if (lane == 0) {
        value_of_warp[warp] = best;
        token_of_warp[warp] = best_token;
    }
    __syncthreads();
    best = value_of_warp[lane];
    best_token = token_of_warp[lane];
    warp_best(best, best_token);
    if (threadIdx.x == 0)
        results[(size_t)row * 2 + 1] = best_token;
}
