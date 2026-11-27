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
#include <cuda_fp16.h>
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
    // A negative token is a row given its own embedding (an image's), placed by place_rows.
    const int token = tokens[blockIdx.x];
    if (token < 0)
        return;
    const float *from = embeddings + (size_t)token * dim;
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
                                     float softcap, const float *sinks) {
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
    // A head's sink (GPT-OSS) joins the softmax as one more score, with no value.
    if (sinks)
        best = fmaxf(best, sinks[head]);

    // Softmax, shifted by the max for numerical stability.
    float sum = 0;
    for (int i = lane; i < n; i += WARP) {
        const float weight = expf(scores[i] - best);
        scores[i] = weight;
        sum += weight;
    }
    const float inverse = 1.0f / (warp_sum(sum) + (sinks ? expf(sinks[head] - best) : 0.0f));
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
    if (token < 0)
        return;
    float *to = x + (size_t)blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += WARP)
        to[i] = __bfloat162float(embeddings[tile_index(token, i, chunks)]) * scale;
}

// A warp waits for each load before going on, and when there are few rows there aren't many
// warps, so memory latency sets the speed unless each warp asks for several things at once.
#define LOAD_AHEAD 4

// As rmsnorm, with the output in tile layout for the next matmul_bf16, and the input's rows
// `stride` elements apart.
extern "C" __global__ void rmsnorm_bf16(const float *x, __nv_bfloat16 *out, const float *weight,
                                        int dim, float eps, const int *rows, int chunks,
                                        int stride) {
    // Any whole number of warps per block, up to 32; the warps' sums are added in order.
    __shared__ float partial[WARP];
    const int row = blockIdx.x, warp = threadIdx.x / WARP, warps = blockDim.x / WARP;
    const float *from = x + (size_t)(rows ? rows[row] : row) * stride;
    float sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += from[i] * from[i];
    sum = warp_sum(sum);
    if (threadIdx.x % WARP == 0)
        partial[warp] = sum;
    __syncthreads();
    float total = 0;
    for (int w = 0; w < warps; w++)
        total += partial[w];
    const float scale = 1.0f / sqrtf(total / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out[tile_index(row, i, chunks)] = __float2bfloat16(from[i] * scale * weight[i]);
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

// The 16-bit matmul splits each tile's columns of the inner dimension into SPLIT_K contiguous
// ranges, one warp each, and adds their sums in range order. With few rows (one request), there
// are otherwise too few warps to keep a large GPU's memory busy. The split doesn't depend on the
// batch, so neither do the results. A block is one output tile's SPLIT_K warps, each keeping
// SPLIT_AHEAD tiles of its range in flight.
#define SPLIT_K 8
#define SPLIT_AHEAD 8

__device__ __forceinline__ void matmul_split(const __nv_bfloat16 *w, int m, int chunks,
                                             const __nv_bfloat16 *x, int batch, float *out,
                                             const float *bias, int accumulate) {
    constexpr int ACROSS = 1, TX = 1;
    __shared__ float partial[ACROSS * TX][SPLIT_K][TILE_ELEMENTS];
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int range = warp % SPLIT_K, pair = warp / SPLIT_K;
    const int x_group = blockIdx.x * TX + pair / ACROSS;
    const int w_group = blockIdx.y * ACROSS + pair % ACROSS;
    const int first_row = x_group * TILE, first_column = w_group * TILE;
    const bool active = first_row < batch && first_column < m;

    if (active) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        // Whole K tiles of cute-rs's tinyllm_linear (32 wide), which adds up the same ranges.
        const int per = ((chunks + SPLIT_K - 1) / SPLIT_K + 1) & ~1;
        const int begin = min(chunks, range * per), end = min(chunks, begin + per);
        const __nv_bfloat16 *x_tiles = x + (size_t)x_group * chunks * TILE_ELEMENTS;
        const __nv_bfloat16 *w_tiles = w + (size_t)w_group * chunks * TILE_ELEMENTS;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[SPLIT_AHEAD];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[SPLIT_AHEAD];
        for (int c = begin; c < end; c += SPLIT_AHEAD) {
            const int n = min(SPLIT_AHEAD, end - c);
#pragma unroll
            for (int i = 0; i < SPLIT_AHEAD; i++)
                if (i < n) {
                    wmma::load_matrix_sync(a[i], x_tiles + (size_t)(c + i) * TILE_ELEMENTS, TILE);
                    wmma::load_matrix_sync(b[i], w_tiles + (size_t)(c + i) * TILE_ELEMENTS, TILE);
                }
#pragma unroll
            for (int i = 0; i < SPLIT_AHEAD; i++)
                if (i < n)
                    wmma::mma_sync(acc, a[i], b[i], acc);
        }
        wmma::store_matrix_sync(partial[pair][range], acc, TILE, wmma::mem_row_major);
    }
    __syncthreads();
    if (!active)
        return;
    // The range's warp writes its share of the tile's elements: the ranges' sums in order, then
    // what was there (if accumulating) and the bias.
    for (int i = range * WARP + lane; i < TILE_ELEMENTS; i += SPLIT_K * WARP) {
        const int row = first_row + i / TILE, column = first_column + i % TILE;
        if (row >= batch || column >= m)
            continue;
        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < SPLIT_K; r++)
            sum += partial[pair][r][i];
        float *target = out + (size_t)row * m + column;
        if (accumulate)
            sum = *target + sum;
        if (bias)
            sum += bias[column];
        *target = sum;
    }
}

// As matmul_tiles (blocks of WARPS_ACROSS weight tiles by TX input groups, a warp for each
// pairing, so a block's warps share tiles through the cache), but adding up the inner dimension
// in matmul_split's ranges: each range's products from zero, then the ranges' sums in order, then
// what was there (if accumulating), then the bias. So it gives the same bits as matmul_split and
// cute-rs's tinyllm_linear, for steps with more rows than suit the one and fewer than the other.
template <int TX>
__device__ __forceinline__ void matmul_ranged(const __nv_bfloat16 *w, int m, int chunks,
                                              const __nv_bfloat16 *x, int batch, float *out,
                                              const float *bias, int accumulate, float *scratch) {
    __shared__ float tiles[WARPS_ACROSS * TX][TILE_ELEMENTS];
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int x_group = blockIdx.x * TX + warp / WARPS_ACROSS;
    const int w_group = blockIdx.y * WARPS_ACROSS + warp % WARPS_ACROSS;
    const int first_row = x_group * TILE, first_column = w_group * TILE;
    if (first_row >= batch || first_column >= m)
        return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc, total;
    wmma::fill_fragment(total, 0.0f);
    const __nv_bfloat16 *x_tiles = x + (size_t)x_group * chunks * TILE_ELEMENTS;
    const __nv_bfloat16 *w_tiles = w + (size_t)w_group * chunks * TILE_ELEMENTS;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[LOAD_AHEAD];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[LOAD_AHEAD];
    const int per = ((chunks + SPLIT_K - 1) / SPLIT_K + 1) & ~1;
    for (int begin = 0; begin < chunks; begin += per) {
        const int end = min(chunks, begin + per);
        wmma::fill_fragment(acc, 0.0f);
        for (int c = begin; c < end; c += LOAD_AHEAD) {
            const int n = min(LOAD_AHEAD, end - c);
#pragma unroll
            for (int i = 0; i < LOAD_AHEAD; i++)
                if (i < n) {
                    wmma::load_matrix_sync(a[i], x_tiles + (size_t)(c + i) * TILE_ELEMENTS, TILE);
                    wmma::load_matrix_sync(b[i], w_tiles + (size_t)(c + i) * TILE_ELEMENTS, TILE);
                }
#pragma unroll
            for (int i = 0; i < LOAD_AHEAD; i++)
                if (i < n)
                    wmma::mma_sync(acc, a[i], b[i], acc);
        }
        for (int i = 0; i < acc.num_elements; i++)
            total.x[i] += acc.x[i];
    }

    // Through the warp's tile in shared memory, adding what was there and the bias element by
    // element.
    float *target = out + (size_t)first_row * m + first_column;
    float *tile = tiles[warp];
    wmma::store_matrix_sync(tile, total, TILE, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < TILE_ELEMENTS; i += WARP) {
        const int row = first_row + i / TILE, column = first_column + i % TILE;
        if (row >= batch || column >= m)
            continue;
        float sum = tile[i];
        float *at = target + (size_t)(i / TILE) * m + i % TILE;
        if (accumulate)
            sum = *at + sum;
        if (bias)
            sum += bias[column];
        *at = sum;
    }
}

// matmul_bf16_TX: out[b][m] = dot(w[m], x[b]) (+ bias[m]), added to what's there if `accumulate`.
// matmul_bf16_1 is matmul_split, the others matmul_ranged, held to 64 registers so two blocks of
// 512 threads fit on a multiprocessor. `scratch` isn't used.
extern "C" __global__ void matmul_bf16_1(const __nv_bfloat16 *w, int m, int chunks,
                                         const __nv_bfloat16 *x, int batch, float *out,
                                         const float *bias, int accumulate, float *scratch) {
    matmul_split(w, m, chunks, x, batch, out, bias, accumulate);
}
#define MATMUL_BF16(TX)                                                                        \
    extern "C" __global__ void __launch_bounds__(32 * WARPS_ACROSS * TX, 8 / TX)              \
        matmul_bf16_##TX(const __nv_bfloat16 *w, int m, int chunks, const __nv_bfloat16 *x,    \
                         int batch, float *out, const float *bias, int accumulate,             \
                         float *scratch) {                                                     \
        matmul_ranged<TX>(w, m, chunks, x, batch, out, bias, accumulate, scratch);             \
    }
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

// One block per row (of 256 threads, so a lone row isn't left to one warp's memory latency): add
// up what its experts made of it (plus each expert's down bias, if they
// have them), weighted, in order of expert, and add that to the residual stream.
extern "C" __global__ void moe_combine(const float *down, int dim, const int *slot_of,
                                       const float *topk_weights, int top_k, float *x,
                                       const int *topk_ids, const float *down_bias) {
    const int row = blockIdx.x;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            float value = down[(size_t)slot_of[row * top_k + k] * dim + c];
            if (down_bias)
                value += down_bias[(size_t)topk_ids[row * top_k + k] * dim + c];
            sum += topk_weights[row * top_k + k] * value;
        }
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
                acc[h][j] = fmaf(part[h][j], scale, acc[h][j]);
                acc[h][j + 2] = fmaf(part[h][j + 2], scale, acc[h][j + 2]);
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

// ---------------------------------------------------------------------------------------------
// 4-bit weights
// ---------------------------------------------------------------------------------------------
// AWQ and GPTQ checkpoints: each weight is a 4-bit value v with a zero point z and scale s for
// its row and group of 128 columns, and weight = (v - z) * s. The values are stored in the same
// lane order as 8-bit weights, two to a byte, so a lane reads its 8 values of a 16x16 tile as one
// 32-bit word: nibbles 0-3 for its row in the first half, 4-7 for the second. v - z is a small
// integer, so it's exact in bfloat16, and it's made with integer operations: the bits of 128 + v
// are 0x4300 | v, and subtracting 128 + z leaves v - z. As with 8-bit weights, each group's sums
// are multiplied by its scale as they're added up.

// Two 4-bit values from the low byte of `byte`, as bfloat16 bits of 128 + v.
__device__ __forceinline__ unsigned offset_pair(unsigned byte) {
    return (byte & 0xF) | ((byte & 0xF0) << 12) | 0x43004300u;
}

__device__ __forceinline__ unsigned minus_zero(unsigned pair, unsigned zero_pair) {
    __nv_bfloat162 difference = __hsub2(*reinterpret_cast<__nv_bfloat162 *>(&pair),
                                        *reinterpret_cast<__nv_bfloat162 *>(&zero_pair));
    return *reinterpret_cast<unsigned *>(&difference);
}

template <int TX>
__device__ __forceinline__ void matmul_tiles_q4(const unsigned *w, const float *scales,
                                                const unsigned char *zeros, int m, int chunks,
                                                const __nv_bfloat16 *x, int batch, float *out,
                                                const float *bias, int accumulate) {
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int x_group = blockIdx.x * TX + warp / WARPS_ACROSS;
    const int w_group = blockIdx.y * WARPS_ACROSS + warp % WARPS_ACROSS;
    const int first_row = x_group * TILE, first_column = w_group * TILE;
    if (first_row >= batch || first_column >= m)
        return; // this warp only has padding to work on
    const int g = lane / 4, t = lane % 4;

    const unsigned *x_words =
        reinterpret_cast<const unsigned *>(x + (size_t)x_group * chunks * TILE_ELEMENTS);
    const unsigned *w_lanes = w + (size_t)w_group * chunks * WARP + lane;
    const int groups = (chunks + I8_LOADS - 1) / I8_LOADS;
    // The weight rows this lane unpacks: g of each half.
    const int unpack_rows[2] = {min(first_column + g, m - 1), min(first_column + 8 + g, m - 1)};
    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    for (int c = 0; c < chunks; c += I8_LOADS) {
        const int n = min(I8_LOADS, chunks - c);
        const int group = c / I8_LOADS;
        unsigned zero_pairs[2];
        for (int h = 0; h < 2; h++) {
            const unsigned z = 0x4300u | zeros[(size_t)unpack_rows[h] * groups + group];
            zero_pairs[h] = z | (z << 16);
        }
        float part[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        unsigned words[I8_LOADS];
        unsigned a[I8_LOADS][4];
#pragma unroll
        for (int i = 0; i < I8_LOADS; i++) {
            if (i < n) {
                words[i] = w_lanes[(size_t)(c + i) * WARP];
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
                const unsigned word = words[i];
                mma_bf16(part[0], a[i], minus_zero(offset_pair(word), zero_pairs[0]),
                         minus_zero(offset_pair(word >> 8), zero_pairs[0]));
                mma_bf16(part[1], a[i], minus_zero(offset_pair(word >> 16), zero_pairs[1]),
                         minus_zero(offset_pair(word >> 24), zero_pairs[1]));
            }
        }
        // This lane's output rows are 8h + 2t and 8h + 2t + 1 of the warp's 16.
        for (int h = 0; h < 2; h++) {
            for (int j = 0; j < 2; j++) {
                const int row = min(first_column + h * 8 + 2 * t + j, m - 1);
                const float scale = scales[(size_t)row * groups + group];
                acc[h][j] = fmaf(part[h][j], scale, acc[h][j]);
                acc[h][j + 2] = fmaf(part[h][j + 2], scale, acc[h][j + 2]);
            }
        }
    }

    for (int h = 0; h < 2; h++) {
        for (int j = 0; j < 4; j++) {
            const int row = first_row + g + (j / 2) * 8;
            const int column = first_column + h * 8 + 2 * t + j % 2;
            if (column >= m)
                continue;
            // In the order of cute-rs's tinyllm_linear_q4: the old value, then the bias.
            float *at = out + (size_t)row * m + column;
            float value = accumulate ? *at + acc[h][j] : acc[h][j];
            if (bias)
                value += bias[column];
            *at = value;
        }
    }
}

// matmul_q4_TX: out[b][m] = dot(w[m], x[b]) (+ bias[m]), added to what's there if `accumulate`,
// with 4-bit weights in the lane order above, and `scales` and `zeros: [m, groups of 128]`.
#define MATMUL_Q4(TX)                                                                          \
    extern "C" __global__ void matmul_q4_##TX(const unsigned *w, const float *scales,          \
                                              const unsigned char *zeros, int m, int chunks,  \
                                              const __nv_bfloat16 *x, int batch, float *out,  \
                                              const float *bias, int accumulate) {            \
        matmul_tiles_q4<TX>(w, scales, zeros, m, chunks, x, batch, out, bias, accumulate);     \
    }
MATMUL_Q4(1)
MATMUL_Q4(2)
MATMUL_Q4(4)

// ---------------------------------------------------------------------------------------------
// LoRA adapters
// ---------------------------------------------------------------------------------------------
// A row using adapter a (1 or more; 0 is none) gets B_a (A_a x) added to a matrix's output, with
// every adapter's A stored as `[adapters, rank, in]` and B transposed, `[adapters, rank, out]`,
// padded with zeros to the largest rank. First h = A x, then out += B h. Each block takes
// LORA_ROWS rows, so that rows sharing an adapter read its matrices once between them. Both add
// up in a fixed order for each row, so its result doesn't depend on the batch.
#define LORA_ROWS 4

// lora_shrink: h[row][j] = dot(A_a[j], x[row]), with x in 16-bit tiles (`tiled`) or 32-bit rows.
// One block per LORA_ROWS rows and group of j, a warp per j, which reads A_a[j] once for the
// block's rows.
extern "C" __global__ void lora_shrink(const void *x, int tiled, int in_dim, const float *a, int rank,
                                       const unsigned *adapters, int batch, float *h) {
    const int first = blockIdx.x * LORA_ROWS;
    const int rows = min(LORA_ROWS, batch - first);
    const int chunks = (in_dim + TILE - 1) / TILE;
    const int lane = threadIdx.x % WARP;
    const int j = blockIdx.y * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (j < rank) {
        for (int r = 0; r < rows; r++) {
            const int row = first + r;
            const unsigned adapter = adapters[row];
            float sum = 0.0f;
            if (adapter != 0) {
                const float *aj = a + ((size_t)(adapter - 1) * rank + j) * in_dim;
                for (int k = lane; k < in_dim; k += WARP) {
                    const float v =
                        tiled ? __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(
                                    x)[tile_index(row, k, chunks)])
                              : reinterpret_cast<const float *>(x)[(size_t)row * in_dim + k];
                    sum += aj[k] * v;
                }
                sum = warp_sum(sum);
            }
            if (lane == 0)
                h[(size_t)row * rank + j] = sum;
        }
    }
}

// lora_expand: out[row][o] += dot(B_a[:, o], h[row]), one thread per o for LORA_ROWS rows.
extern "C" __global__ void lora_expand(const float *h, int rank, const float *b, int out_dim,
                                       const unsigned *adapters, int batch, float *out) {
    const int first = blockIdx.x * LORA_ROWS;
    const int rows = min(LORA_ROWS, batch - first);
    const int o = blockIdx.y * blockDim.x + threadIdx.x;
    if (o >= out_dim)
        return;
    for (int r = 0; r < rows; r++) {
        const unsigned adapter = adapters[first + r];
        if (adapter == 0)
            continue;
        const float *bo = b + (size_t)(adapter - 1) * rank * out_dim + o;
        const float *hr = h + (size_t)(first + r) * rank;
        float sum = 0.0f;
        for (int j = 0; j < rank; j++)
            sum += bo[(size_t)j * out_dim] * hr[j];
        out[(size_t)(first + r) * out_dim + o] += sum;
    }
}

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

// For cute-flashinfer's attention (src/cuda/attention.ptx): each row's rotated queries from
// `qkv`, as bfloat16 rows of `q_dim`.
extern "C" __global__ void queries_bf16(const float *qkv, int stride, __nv_bfloat16 *q, int q_dim) {
    const float *from = qkv + (size_t)blockIdx.x * stride;
    __nv_bfloat16 *to = q + (size_t)blockIdx.x * q_dim;
    for (int i = threadIdx.x; i < q_dim; i += blockDim.x)
        to[i] = __float2bfloat16(from[i]);
}

// Merge the attention kernel's partial states for row `first_row + blockIdx.x`, head
// `blockIdx.y` (one thread per element of the head): scratch rows `start[r] .. start[r] +
// count[r]`, in order, each a normalised output and its base-2 log-sum-exp. Chunks past the
// row's causal reach have lse -inf (and no meaningful output) and are skipped, so a row's
// result doesn't depend on how many such chunks its step gave it. The result goes to `out`
// in tile layout, `q_chunks` chunks wide.
extern "C" __global__ void merge_chunks(const float *tmp_o, const float *tmp_lse, const int *start,
                                        const int *count, int first_row, int n_heads,
                                        int head_dim, __nv_bfloat16 *out, int q_chunks) {
    const int r = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int first = start[r], n = count[r];
    float m = -INFINITY;
    for (int k = 0; k < n; k++)
        m = fmaxf(m, tmp_lse[(size_t)(first + k) * n_heads + head]);
    float sum = 0.0f, total = 0.0f;
    for (int k = 0; k < n; k++) {
        const float lse = tmp_lse[(size_t)(first + k) * n_heads + head];
        if (!(lse > -INFINITY)) // -inf, or NaN to be safe
            continue;
        const float w = exp2f(lse - m);
        sum += w;
        total += w * tmp_o[((size_t)(first + k) * n_heads + head) * head_dim + d];
    }
    out[tile_index(first_row + r, head * head_dim + d, q_chunks)] = __float2bfloat16(total / sum);
}

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> TileA;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> TileBColumns;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> TileBRows;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> TileTotal;

// Causal self-attention on tensor cores, in two kernels that give exactly the same results:
// `attention_bf16_*` for rows generating tokens (few rows per sequence, long contexts), and
// `attention_prefill_*` for prompts (many consecutive rows of one sequence).
//
// Tile rows are query heads: a 16-row tile holds the query heads sharing a key/value head for
// 16 / group consecutive rows, and a tile multiplication gives the scores of a whole page of 16
// tokens at once for all of them.
//
// The softmax: pages fall into ranges of RANGE_PAGES (by their position in the sequence). Within
// a range, one warp goes through its pages in order, weighting each token relative to the best
// score in the first page of the range that the query can see (only ratios between weights
// matter, and weights are capped well short of overflowing). Then the ranges' totals are folded
// into one, in range order, rescaled to a running reference. Pages a query can't see change
// nothing, so a row's result depends only on its own context, whichever kernel and whichever
// tile it's in: results don't depend on the batch.
//
// `row_len[row]` is how many tokens the row can see, whose pages start at
// page_lists[row_offset[row]]. Consecutive rows of a sequence each see one more token than the
// row before. With a `window` a row only sees that many of the latest tokens.
#define LARGEST_ATTENTION_EXPONENT 60.0f
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
#define RANGE_PAGES 32
#define PREFILL_TILES 4
#define PREFILL_PAGES 2

// One warp's softmax state for a range, in the registers of the tensor cores' fragments
// (mma.m16n8k16): lane L (g = L / 4, t = L % 4) looks after tile rows g and g + 8 (h = 0, 1), and
// in each 8 columns, columns 2t and 2t + 1. `totals[c][n][2h + e]` is row g + 8h, column
// c * 16 + n * 8 + 2t + e of the weighted values.
template <int CHUNKS> struct RangeState {
    float totals[CHUNKS][2][4];
    float reference[2], capped[2], lane_sum[2];
    bool referenced[2];
};

template <int CHUNKS> __device__ __forceinline__ void range_reset(RangeState<CHUNKS> &st) {
    for (int c = 0; c < CHUNKS; c++)
        for (int n = 0; n < 2; n++)
            for (int i = 0; i < 4; i++)
                st.totals[c][n][i] = 0.0f;
    for (int h = 0; h < 2; h++) {
        st.lane_sum[h] = 0.0f;
        st.referenced[h] = false;
    }
}

// What a warp's tile needs to know about its rows.
struct TileRows {
    int count;     // tile rows in use
    int first_len; // how many tokens the tile's first row sees
    int group;     // query heads per key/value head
    int window;
};

// Four 8x8 matrices of 16-bit values from shared memory, each lane giving one row's address
// (lanes 8i..8i+7 the rows of matrix i), as fragments: lane L gets row L / 4, columns 2(L % 4)
// and the next, of each (TRANS: of each one transposed).
template <bool TRANS>
__device__ __forceinline__ void load_matrices(unsigned *r, const __nv_bfloat16 *row) {
    const unsigned address = (unsigned)__cvta_generic_to_shared(row);
    if (TRANS)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(address));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                     : "r"(address));
}

// Add P consecutive pages from `first_page` to a range, those whose bit is set in `active`: page p's
// key and value tiles (CHUNKS each) at `k + p * page_stride` and `v + p * page_stride`, in global
// memory or (SHARED) shared memory. The scores of all P pages are found together, so the tensor
// cores have more independent work; the softmax then takes the pages in order, exactly as one at a
// time. The scores stay in registers, laid out as the weights' A fragment needs them.
template <int CHUNKS, bool SHARED = false, int P = 1>
__device__ __forceinline__ void range_pages(RangeState<CHUNKS> &st, const unsigned (*q)[4],
                                            const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                                            int page_stride, int first_page, unsigned active,
                                            TileRows rows, float scale, float softcap, int lane) {
    const int g = lane / 4, t = lane % 4;
    // Keys are stored token by token, so a B fragment (k = dimension, n = token) is 2 words.
    float scores[P][2][4] = {};
#pragma unroll
    for (int c = 0; c < CHUNKS; c++)
#pragma unroll
        for (int p = 0; p < P; p++) {
            if (!(active >> p & 1))
                continue;
            const __nv_bfloat16 *keys = k + p * page_stride + c * TILE_ELEMENTS;
            unsigned b[4];
            if (SHARED) {
                // Matrix i: tokens i / 2 * 8.., dimensions i % 2 * 8..
                const int i = lane / 8;
                load_matrices<false>(b, keys + (i / 2 * 8 + lane % 8) * TILE + i % 2 * 8);
            } else {
                for (int n = 0; n < 2; n++) {
                    const unsigned *key = (const unsigned *)(keys + (n * 8 + g) * TILE + 2 * t);
                    b[2 * n] = key[0];
                    b[2 * n + 1] = key[4];
                }
            }
            for (int n = 0; n < 2; n++)
                mma_bf16(scores[p][n], q[c], b[2 * n], b[2 * n + 1]);
        }
#pragma unroll
    for (int p = 0; p < P; p++) {
        if (!(active >> p & 1))
            continue;
        const int page = first_page + p;
        unsigned weights[4];
        // Once every row has its reference, the best scores aren't needed.
        const bool find_best = __any_sync(0xffffffff, !st.referenced[0] || !st.referenced[1]);
        // Most pages are seen whole by every row, which needs no masks.
        const bool whole = page * TILE + TILE <= rows.first_len &&
                           page * TILE >= rows.first_len + (rows.count - 1) / rows.group -
                                              rows.window;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = g + 8 * h;
            const int seen = rows.first_len + r / rows.group;
            const bool used = r < rows.count;
            bool visible[4];
            for (int i = 0; i < 4; i++) {
                const int position = page * TILE + i / 2 * 8 + 2 * t + i % 2;
                visible[i] =
                    used && (whole || (position < seen && position >= seen - rows.window));
            }
            if (find_best) {
                float best = -FLT_MAX;
                for (int i = 0; i < 4; i++)
                    if (visible[i])
                        best = fmaxf(best, scores[p][i / 2][2 * h + i % 2]);
                best = fmaxf(best, __shfl_xor_sync(0xffffffff, best, 1));
                best = fmaxf(best, __shfl_xor_sync(0xffffffff, best, 2));
                if (!st.referenced[h] && best != -FLT_MAX) {
                    st.reference[h] = best;
                    st.capped[h] = soft_cap(best * scale, softcap);
                    st.referenced[h] = true;
                }
            }
            // Weights are powers of 2 of the exponent in base 2, which one fused multiply-add
            // gives.
            const float scale2 = scale * LOG2E, reference2 = st.reference[h] * scale2;
            float weight[4];
            for (int i = 0; i < 4; i++) {
                const float score = scores[p][i / 2][2 * h + i % 2];
                const float exponent2 =
                    softcap > 0.0f ? (soft_cap(score * scale, softcap) - st.capped[h]) * LOG2E
                                   : fmaf(score, scale2, -reference2);
                weight[i] = visible[i]
                                ? exp2_approx(fminf(exponent2, LARGEST_ATTENTION_EXPONENT * LOG2E))
                                : 0.0f;
            }
            // Summing the rounded weights the multiplication really uses keeps the result a true
            // average, so a dominant token isn't scaled by its weight's rounding error.
            for (int j = 0; j < 2; j++) {
                const __nv_bfloat162 pair = __floats2bfloat162_rn(weight[2 * j], weight[2 * j + 1]);
                st.lane_sum[h] += __low2float(pair) + __high2float(pair);
                weights[2 * j + h] = *reinterpret_cast<const unsigned *>(&pair);
            }
        }
        // Values are stored token by token too, so a B fragment (k = token, n = dimension)
        // takes two halves from consecutive tokens.
        const __nv_bfloat16 *page_values = v + p * page_stride;
        const unsigned short *values = (const unsigned short *)page_values;
#pragma unroll
        for (int c = 0; c < CHUNKS; c++) {
            unsigned b[4];
            if (SHARED) {
                // Matrix i: tokens i % 2 * 8.., dimensions i / 2 * 8.., transposed.
                const int i = lane / 8;
                load_matrices<true>(b, page_values + c * TILE_ELEMENTS +
                                           (i % 2 * 8 + lane % 8) * TILE + i / 2 * 8);
            } else {
                for (int n = 0; n < 2; n++) {
                    const unsigned short *column =
                        values + c * TILE_ELEMENTS + n * 8 + g + 2 * t * TILE;
                    b[2 * n] = column[0] | (unsigned)column[TILE] << 16;
                    b[2 * n + 1] = column[8 * TILE] | (unsigned)column[9 * TILE] << 16;
                }
            }
            for (int n = 0; n < 2; n++)
                mma_bf16(st.totals[c][n], weights, b[2 * n], b[2 * n + 1]);
        }
    }
}

// How to fold row g + 8h of a warp's range into an accumulator whose reference exponent is `old`:
// the accumulator is scaled by `keep` and the range by `add`, and `next` is the new reference.
template <int CHUNKS>
__device__ __forceinline__ void fold_factors(const RangeState<CHUNKS> &st, int h, bool used,
                                             float scale, float softcap, float old, float &next,
                                             float &keep, float &add) {
    used = used && st.referenced[h];
    const float ref = !used ? 0.0f : softcap > 0.0f ? st.capped[h] : st.reference[h] * scale;
    next = used ? fmaxf(old, ref) : old;
    keep = !used ? 1.0f : old == -FLT_MAX ? 0.0f : expf(old - next);
    add = used ? expf(ref - next) : 0.0f;
}

// The total weight of row g + 8h of a range (the same in all 4 lanes of a row).
template <int CHUNKS>
__device__ __forceinline__ float range_sum(const RangeState<CHUNKS> &st, int h) {
    float sum = st.lane_sum[h] + __shfl_xor_sync(0xffffffff, st.lane_sum[h], 1);
    return sum + __shfl_xor_sync(0xffffffff, sum, 2);
}

// Fold a warp's range into an accumulator in shared memory: `acc` (CHUNKS tiles of floats), and
// per tile row a reference exponent `m` (-FLT_MAX before any) and total weight `s`. One warp.
template <int CHUNKS>
__device__ __forceinline__ void range_fold(RangeState<CHUNKS> &st, int count, float scale,
                                           float softcap, float *acc, float *m, float *s,
                                           int lane) {
    const int g = lane / 4, t = lane % 4;
    float keep[2], add[2], sum[2], next[2];
    for (int h = 0; h < 2; h++) {
        const int r = g + 8 * h;
        sum[h] = range_sum(st, h);
        fold_factors(st, h, r < count, scale, softcap, m[r], next[h], keep[h], add[h]);
    }
    __syncwarp();
    for (int c = 0; c < CHUNKS; c++)
        for (int n = 0; n < 2; n++)
            for (int i = 0; i < 4; i++) {
                const int h = i / 2;
                float &a = acc[c * TILE_ELEMENTS + (g + 8 * h) * TILE + n * 8 + 2 * t + i % 2];
                a = a * keep[h] + st.totals[c][n][i] * add[h];
            }
    if (t == 0)
        for (int h = 0; h < 2; h++) {
            const int r = g + 8 * h;
            s[r] = s[r] * keep[h] + sum[h] * add[h];
            m[r] = next[h];
        }
    __syncwarp();
}

// The same accumulator, kept in one warp's registers, laid out as `RangeState::totals`.
template <int CHUNKS> struct RowTotals {
    float acc[CHUNKS][2][4];
    float m[2], s[2];
};

template <int CHUNKS>
__device__ __forceinline__ void range_fold(RangeState<CHUNKS> &st, int count, float scale,
                                           float softcap, RowTotals<CHUNKS> &a, int lane) {
    const int g = lane / 4;
    float keep[2], add[2];
    for (int h = 0; h < 2; h++) {
        const float sum = range_sum(st, h);
        float next;
        fold_factors(st, h, g + 8 * h < count, scale, softcap, a.m[h], next, keep[h], add[h]);
        a.s[h] = a.s[h] * keep[h] + sum * add[h];
        a.m[h] = next;
    }
    for (int c = 0; c < CHUNKS; c++)
        for (int n = 0; n < 2; n++)
            for (int i = 0; i < 4; i++)
                a.acc[c][n][i] = a.acc[c][n][i] * keep[i / 2] + st.totals[c][n][i] * add[i / 2];
}

// The query tile of `count` tile rows starting at `first_row`, as A fragments.
template <int CHUNKS>
__device__ __forceinline__ void load_queries(unsigned (*q)[4], const float *qkv, int stride,
                                             int first_row, int count, int group, int kv_head,
                                             int head_dim, int lane) {
    const int g = lane / 4, t = lane % 4;
    for (int h = 0; h < 2; h++) {
        const int r = g + 8 * h;
        const float *row =
            r < count ? qkv + (size_t)(first_row + r / group) * stride +
                            (kv_head * group + r % group) * head_dim
                      : nullptr;
        for (int c = 0; c < CHUNKS; c++)
            for (int half = 0; half < 2; half++) {
                const int column = c * TILE + half * 8 + 2 * t;
                q[c][h + 2 * half] =
                    row ? bf16_pair(row[column], row[column + 1]) : bf16_pair(0.0f, 0.0f);
            }
    }
}

// Divide the accumulator by the total weight and write the tile rows out, rounded for the next
// multiplication. A head's sink (GPT-OSS) adds to the total weight as one more score, with no
// value. Threads `start`, `start + step`, ... share the work.
template <int CHUNKS>
__device__ __forceinline__ void write_rows(const float *acc, const float *m, const float *s,
                                           int count, int first_row, int group, int kv_head,
                                           int head_dim, const float *sinks, __nv_bfloat16 *out,
                                           int out_chunks, int start, int step) {
    for (int i = start; i < CHUNKS * TILE_ELEMENTS; i += step) {
        const int c = i / TILE_ELEMENTS, t = i % TILE_ELEMENTS / TILE, j = i % TILE;
        if (t >= count)
            continue;
        float total = s[t];
        if (sinks)
            total += expf(
                fminf(sinks[kv_head * group + t % group] - m[t], LARGEST_ATTENTION_EXPONENT));
        const int row = first_row + t / group, head = kv_head * group + t % group;
        out[tile_index(row, head * head_dim + c * TILE + j, out_chunks)] =
            __float2bfloat16(acc[i] / total);
    }
}

// For rows generating tokens: one block per team of rows (`team_sizes[row]` rows from this one,
// 0 if another row's block has this one) and key/value head. Its warps take turns over the
// ranges (warp w: ranges w, w + warps, ...), so a long context is spread over many warps, and
// fold them in order. `smem`: CHUNKS tiles of floats and 2 * 16 floats.
template <int CHUNKS>
__device__ __forceinline__ void attention_tiles(const float *qkv, int stride,
                                                const __nv_bfloat16 *k_pool,
                                                const __nv_bfloat16 *v_pool, int kv_chunks,
                                                const unsigned *page_lists, const int *row_offset,
                                                const int *row_len, const int *team_sizes,
                                                __nv_bfloat16 *out, int n_heads, int n_kv_heads,
                                                int head_dim, float scale, int out_chunks,
                                                int window, float softcap, const float *sinks) {
    extern __shared__ __align__(1024) unsigned char smem[];
    const int lane = threadIdx.x % WARP, warp = threadIdx.x / WARP, warps = blockDim.x / WARP;
    const int first_row = blockIdx.x / n_kv_heads, kv_head = blockIdx.x % n_kv_heads;
    const int team_size = team_sizes[first_row];
    if (team_size == 0)
        return;
    const int group = n_heads / n_kv_heads;
    const TileRows rows = {team_size * group, row_len[first_row], group, window};
    const int first_page = max(0, rows.first_len - window) / TILE;
    const int last_page = (rows.first_len + team_size - 2) / TILE;
    const unsigned *pages = page_lists + row_offset[first_row];

    float *acc = (float *)smem;
    float *m = acc + CHUNKS * TILE_ELEMENTS, *s = m + TILE;
    for (int i = threadIdx.x; i < CHUNKS * TILE_ELEMENTS; i += blockDim.x)
        acc[i] = 0.0f;
    if (threadIdx.x < TILE) {
        m[threadIdx.x] = -FLT_MAX;
        s[threadIdx.x] = 0.0f;
    }
    unsigned q[CHUNKS][4];
    load_queries<CHUNKS>(q, qkv, stride, first_row, rows.count, group, kv_head, head_dim, lane);
    __syncthreads();

    RangeState<CHUNKS> st;
    const int first_range = first_page / RANGE_PAGES, last_range = last_page / RANGE_PAGES;
    for (int base = first_range; base <= last_range; base += warps) {
        const int range = base + warp;
        range_reset(st);
        if (range <= last_range) {
            const int end = min(last_page, range * RANGE_PAGES + RANGE_PAGES - 1);
            for (int page = max(first_page, range * RANGE_PAGES); page <= end; page++) {
                const size_t first =
                    ((size_t)pages[page] * kv_chunks + kv_head * CHUNKS) * TILE_ELEMENTS;
                range_pages<CHUNKS>(st, q, k_pool + first, v_pool + first, 0, page, 1, rows,
                                    scale, softcap, lane);
            }
        }
        for (int w = 0; w < warps; w++) {
            if (warp == w && range <= last_range)
                range_fold<CHUNKS>(st, rows.count, scale, softcap, acc, m, s, lane);
            __syncthreads();
        }
    }
    write_rows<CHUNKS>(acc, m, s, rows.count, first_row, group, kv_head, head_dim, sinks, out,
                       out_chunks, threadIdx.x, blockDim.x);
}

// For prompts: one block per PREFILL_TILES teams of consecutive rows of one sequence
// (`block_rows[row]` rows from this one, 0 if another row's block has this one) and key/value
// head, one warp per team. Each page of keys and values is read once into shared memory for all
// of them, the next page loading while this one is used, which is what makes long prompts fast.
// `smem`: two stages of PREFILL_PAGES pages of keys and values (CHUNKS tiles each).
template <int CHUNKS>
__device__ __forceinline__ void attention_prefill_tiles(
    const float *qkv, int stride, const __nv_bfloat16 *k_pool, const __nv_bfloat16 *v_pool,
    int kv_chunks, const unsigned *page_lists, const int *row_offset, const int *row_len,
    const int *block_rows, __nv_bfloat16 *out, int n_heads, int n_kv_heads, int head_dim,
    float scale, int out_chunks, int window, float softcap, const float *sinks) {
    extern __shared__ __align__(1024) unsigned char smem[];
    const int lane = threadIdx.x % WARP, warp = threadIdx.x / WARP;
    const int block_row = blockIdx.x / n_kv_heads, kv_head = blockIdx.x % n_kv_heads;
    const int total_rows = block_rows[block_row];
    if (total_rows == 0)
        return;
    const int group = n_heads / n_kv_heads, team = TILE / group;
    const int first_row = block_row + warp * team;
    const int my_rows = max(0, min(team, total_rows - warp * team));
    const int block_len = row_len[block_row];
    const TileRows rows = {my_rows * group, block_len + warp * team, group, window};
    const int block_first = max(0, block_len - window) / TILE;
    const int block_last = (block_len + total_rows - 2) / TILE;
    const int my_first = max(0, rows.first_len - window) / TILE;
    const int my_last = (rows.first_len + my_rows - 2) / TILE;
    const unsigned *pages = page_lists + row_offset[block_row];
    // [2 stages][PREFILL_PAGES pages][keys, values]
    __nv_bfloat16 *buffers = (__nv_bfloat16 *)smem;
    const int page_stride = 2 * CHUNKS * TILE_ELEMENTS;

    const int vectors = CHUNKS * TILE_ELEMENTS * 2 / 16;
    auto fetch = [&](int stage) {
        for (int p = 0; p < PREFILL_PAGES; p++) {
            const int page = stage * PREFILL_PAGES + p;
            if (page < block_first || page > block_last)
                continue;
            const size_t first =
                ((size_t)pages[page] * kv_chunks + kv_head * CHUNKS) * TILE_ELEMENTS;
            __nv_bfloat16 *keys = buffers + ((stage & 1) * PREFILL_PAGES + p) * page_stride;
            for (int i = threadIdx.x; i < 2 * vectors; i += blockDim.x) {
                const bool value = i >= vectors;
                const __nv_bfloat16 *from = (value ? v_pool : k_pool) + first;
                const unsigned to = (unsigned)__cvta_generic_to_shared(keys) + i * 16;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(to),
                             "l"((const uint4 *)from + (i - (value ? vectors : 0))));
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    unsigned q[CHUNKS][4];
    if (my_rows > 0)
        load_queries<CHUNKS>(q, qkv, stride, first_row, rows.count, group, kv_head, head_dim,
                             lane);
    RowTotals<CHUNKS> totals;
    for (int c = 0; c < CHUNKS; c++)
        for (int n = 0; n < 2; n++)
            for (int i = 0; i < 4; i++)
                totals.acc[c][n][i] = 0.0f;
    for (int h = 0; h < 2; h++) {
        totals.m[h] = -FLT_MAX;
        totals.s[h] = 0.0f;
    }
    RangeState<CHUNKS> st;
    range_reset(st);

    // Stages hold PREFILL_PAGES pages, which divides RANGE_PAGES, so a range ends with a stage.
    const int first_stage = block_first / PREFILL_PAGES, last_stage = block_last / PREFILL_PAGES;
    fetch(first_stage);
    for (int stage = first_stage; stage <= last_stage; stage++) {
        if (stage < last_stage) {
            fetch(stage + 1);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        if (my_rows > 0) {
            const int first_page = stage * PREFILL_PAGES;
            unsigned active = 0;
            for (int p = 0; p < PREFILL_PAGES; p++)
                if (first_page + p >= my_first && first_page + p <= my_last)
                    active |= 1u << p;
            const __nv_bfloat16 *keys = buffers + (stage & 1) * PREFILL_PAGES * page_stride;
            range_pages<CHUNKS, true, PREFILL_PAGES>(st, q, keys, keys + CHUNKS * TILE_ELEMENTS,
                                                     page_stride, first_page, active, rows, scale,
                                                     softcap, lane);
            const int last_page = first_page + PREFILL_PAGES - 1;
            if (last_page % RANGE_PAGES == RANGE_PAGES - 1 || stage == last_stage) {
                range_fold<CHUNKS>(st, rows.count, scale, softcap, totals, lane);
                range_reset(st);
            }
        }
        __syncthreads();
    }
    if (my_rows == 0)
        return;
    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const int r = g + 8 * h;
        if (r >= rows.count)
            break; // rows g + 8 come after rows g
        const int head = kv_head * group + r % group, row = first_row + r / group;
        float total = totals.s[h];
        if (sinks)
            total += expf(fminf(sinks[head] - totals.m[h], LARGEST_ATTENTION_EXPONENT));
#pragma unroll
        for (int c = 0; c < CHUNKS; c++)
#pragma unroll
            for (int n = 0; n < 2; n++)
#pragma unroll
                for (int e = 0; e < 2; e++)
                    out[tile_index(row, head * head_dim + c * TILE + n * 8 + 2 * t + e,
                                   out_chunks)] =
                        __float2bfloat16(totals.acc[c][n][2 * h + e] / total);
    }
}

// One kernel per head size, so the compiler can see how many tiles there are and keep them in
// registers. The head is CHUNKS * 16 elements long.
#define ATTENTION_BF16(CHUNKS)                                                                 \
    extern "C" __global__ void attention_bf16_##CHUNKS(                                        \
        const float *qkv, int stride, const __nv_bfloat16 *k_pool, const __nv_bfloat16 *v_pool, \
        int kv_chunks, const unsigned *page_lists, const int *row_offset, const int *row_len,  \
        const int *team_sizes, __nv_bfloat16 *out, int n_heads, int n_kv_heads, int head_dim,  \
        float scale, int out_chunks, int window, float softcap, const float *sinks) {          \
        attention_tiles<CHUNKS>(qkv, stride, k_pool, v_pool, kv_chunks, page_lists, row_offset, \
                                row_len, team_sizes, out, n_heads, n_kv_heads, head_dim, scale, \
                                out_chunks, window, softcap, sinks);                           \
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

#define ATTENTION_PREFILL(CHUNKS)                                                              \
    extern "C" __global__ void attention_prefill_##CHUNKS(                                     \
        const float *qkv, int stride, const __nv_bfloat16 *k_pool, const __nv_bfloat16 *v_pool, \
        int kv_chunks, const unsigned *page_lists, const int *row_offset, const int *row_len,  \
        const int *block_rows, __nv_bfloat16 *out, int n_heads, int n_kv_heads, int head_dim,  \
        float scale, int out_chunks, int window, float softcap, const float *sinks) {          \
        attention_prefill_tiles<CHUNKS>(qkv, stride, k_pool, v_pool, kv_chunks, page_lists,    \
                                        row_offset, row_len, block_rows, out, n_heads,         \
                                        n_kv_heads, head_dim, scale, out_chunks, window,       \
                                        softcap, sinks);                                       \
    }
ATTENTION_PREFILL(1)
ATTENTION_PREFILL(2)
ATTENTION_PREFILL(3)
ATTENTION_PREFILL(4)
ATTENTION_PREFILL(5)
ATTENTION_PREFILL(6)
ATTENTION_PREFILL(7)
ATTENTION_PREFILL(8)

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

// ---------------------------------------------------------------------------------------------
// GPT-OSS's experts
// ---------------------------------------------------------------------------------------------
// The activation, for the slots of a mixture of experts, with each expert's gate and up biases
// added first: with g = min(gate, limit) and u = clamp(up, -limit, limit), (u + 1) * g *
// sigmoid(alpha * g). One block per slot and SWIGLU_COLUMNS columns, as moe_swiglu.
extern "C" __global__ void moe_clamped_glu(const float *gate_up, __nv_bfloat16 *out, int hidden,
                                           int chunks, const int *num_groups, const float *bias,
                                           const int *group_expert, float alpha, float limit) {
    const int slot = blockIdx.x;
    if (slot >= *num_groups * TILE)
        return;
    const int first = blockIdx.y * SWIGLU_COLUMNS, end = min(hidden, first + SWIGLU_COLUMNS);
    const float *gate = gate_up + (size_t)slot * 2 * hidden, *up = gate + hidden;
    const float *expert_bias = bias + (size_t)group_expert[slot / TILE] * 2 * hidden;
    for (int i = first + threadIdx.x; i < end; i += WARP) {
        const float g = fminf(gate[i] + expert_bias[i], limit);
        const float u = fminf(fmaxf(up[i] + expert_bias[hidden + i], -limit), limit);
        out[tile_index(slot, i, chunks)] = __float2bfloat16((u + 1.0f) * (g / (1.0f + expf(-alpha * g))));
    }
}

// MXFP4 weights: 4-bit floats (2 exponent bits, 1 mantissa bit: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and
// their negatives) with a power-of-two scale for each 32 columns, stored as an exponent byte
// biased by 127, `[m, cols / 32]`. The 4-bit codes are in the same lane order as 4-bit integers
// (see matmul_q4), so a lane reads its 8 values of a tile as one word. Each value is exact in
// bfloat16, made with integer operations, and each pair of chunks' sums is multiplied by its
// group's scale as they're added up.

// The bfloat16 bits of a 4-bit float.
__device__ __forceinline__ unsigned fp4_bf16(unsigned code) {
    const unsigned m = code & 7;
    // 2^((m >> 1) - 1) * (1 + (m & 1) / 2) for m >= 2; 0 and 0.5 below that.
    const unsigned magnitude = m >= 2 ? (m << 6) + 0x3F00u : m * 0x3F00u;
    return magnitude | ((code & 8) << 12);
}

// The two 4-bit floats in the low byte of `byte` as a pair of bfloat16s, the low nibble first.
__device__ __forceinline__ unsigned fp4_pair(unsigned byte) {
    return fp4_bf16(byte & 15) | (fp4_bf16((byte >> 4) & 15) << 16);
}

#define FP4_GROUP_CHUNKS 2 // 32 columns share a scale

template <int TX>
__device__ __forceinline__ void matmul_tiles_fp4(const unsigned *w, const unsigned char *scales,
                                                 int m, int chunks, const __nv_bfloat16 *x,
                                                 int batch, float *out) {
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int x_group = blockIdx.x * TX + warp / WARPS_ACROSS;
    const int w_group = blockIdx.y * WARPS_ACROSS + warp % WARPS_ACROSS;
    const int first_row = x_group * TILE, first_column = w_group * TILE;
    if (first_row >= batch || first_column >= m)
        return; // this warp only has padding to work on
    const int g = lane / 4, t = lane % 4;
    const unsigned *x_words =
        reinterpret_cast<const unsigned *>(x + (size_t)x_group * chunks * TILE_ELEMENTS);
    const unsigned *w_lanes = w + (size_t)w_group * chunks * WARP + lane;
    const int groups = chunks / FP4_GROUP_CHUNKS;
    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    for (int c = 0; c < chunks; c += I8_LOADS) {
        const int n = min(I8_LOADS, chunks - c);
        unsigned words[I8_LOADS];
        unsigned a[I8_LOADS][4];
#pragma unroll
        for (int i = 0; i < I8_LOADS; i++) {
            if (i < n) {
                words[i] = w_lanes[(size_t)(c + i) * WARP];
                const unsigned *tile = x_words + (size_t)(c + i) * (TILE_ELEMENTS / 2);
                a[i][0] = tile[g * 8 + t];
                a[i][1] = tile[(g + 8) * 8 + t];
                a[i][2] = tile[g * 8 + t + 4];
                a[i][3] = tile[(g + 8) * 8 + t + 4];
            }
        }
#pragma unroll
        for (int p = 0; p < I8_LOADS / FP4_GROUP_CHUNKS; p++) {
            if (p * FP4_GROUP_CHUNKS >= n)
                break;
            float part[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
            for (int q = 0; q < FP4_GROUP_CHUNKS; q++) {
                const int i = p * FP4_GROUP_CHUNKS + q;
                const unsigned word = words[i];
                mma_bf16(part[0], a[i], fp4_pair(word), fp4_pair(word >> 8));
                mma_bf16(part[1], a[i], fp4_pair(word >> 16), fp4_pair(word >> 24));
            }
            // This lane's output rows are 8h + 2t and 8h + 2t + 1 of the warp's 16.
            const int group = (c + p * FP4_GROUP_CHUNKS) / FP4_GROUP_CHUNKS;
            for (int h = 0; h < 2; h++) {
                for (int j = 0; j < 2; j++) {
                    const int row = min(first_column + h * 8 + 2 * t + j, m - 1);
                    const unsigned e = scales[(size_t)row * groups + group];
                    const float scale = e == 0 ? __uint_as_float(1u << 22) : __uint_as_float(e << 23);
                    acc[h][j] += part[h][j] * scale;
                    acc[h][j + 2] += part[h][j + 2] * scale;
                }
            }
        }
    }
    for (int h = 0; h < 2; h++) {
        for (int j = 0; j < 4; j++) {
            const int row = first_row + g + (j / 2) * 8;
            const int column = first_column + h * 8 + 2 * t + j % 2;
            if (column < m)
                out[(size_t)row * m + column] = acc[h][j];
        }
    }
}

// moe_matmul_fp4_TX: as moe_matmul_TX, with MXFP4 weights, each expert's `expert_words` words
// and `expert_scales` scale bytes apart.
#define MOE_MATMUL_FP4(TX)                                                                     \
    extern "C" __global__ void moe_matmul_fp4_##TX(                                            \
        const unsigned *w, unsigned long long expert_words, const unsigned char *scales,       \
        unsigned long long expert_scales, const int *group_expert, const int *num_groups,      \
        int m, int chunks, const __nv_bfloat16 *x, float *out) {                               \
        const int warp = threadIdx.x / WARP;                                                   \
        const int x_group = blockIdx.x * TX + warp / WARPS_ACROSS;                             \
        const int groups = *num_groups;                                                        \
        if (x_group >= groups)                                                                 \
            return;                                                                            \
        const int expert = group_expert[x_group];                                              \
        matmul_tiles_fp4<TX>(w + expert * expert_words, scales + expert * expert_scales, m,    \
                             chunks, x, groups * TILE, out);                                   \
    }
MOE_MATMUL_FP4(1)
MOE_MATMUL_FP4(2)
MOE_MATMUL_FP4(4)

// ---------------------------------------------------------------------------------------------
// DeepSeek's latent attention
// ---------------------------------------------------------------------------------------------
// Each token's keys and values come from one latent vector, `rank` wide (512 in DeepSeek V2 and
// V3), which each head expands with matrices of its own: head h's key is `K_h c` (plus a rotated
// part that all heads share) and its value `V_h c`. Rather than store every head's key and value
// (5,120 numbers a token for V2-Lite's 16 heads, 40,960 for V3's 128), the cache holds just the
// latent and the shared rotated part: 576 numbers. A query `q` scores against the latent through
// `K_h^T q`, worked out once per query (matmul_heads), and attention's result, a blend of
// latents, goes through `V_h` afterwards. That's the same arithmetic in a different order, and
// it makes attention a single key/value head, whose values are the first `rank` elements of
// its keys.

// matmul_heads_TX: matmul_bf16 for each head in turn (blockIdx.z), with head h's weights, inputs
// and outputs `w_stride`, `x_stride` and `out_stride` elements after head h - 1's. `m` has to be
// a multiple of 16.
#define MATMUL_HEADS(TX)                                                                       \
    extern "C" __global__ void matmul_heads_##TX(                                              \
        const __nv_bfloat16 *w, unsigned long long w_stride, int m, int chunks,               \
        const __nv_bfloat16 *x, unsigned long long x_stride, int batch, float *out,           \
        unsigned long long out_stride) {                                                       \
        const size_t head = blockIdx.z;                                                        \
        matmul_tiles<TX>(w + head * w_stride, m, chunks, x + head * x_stride, batch,          \
                         out + head * out_stride, nullptr, 0, nullptr);                        \
    }
MATMUL_HEADS(1)
MATMUL_HEADS(2)
MATMUL_HEADS(4)

// One block per row: rotate the rotated part of each head's query (`nope` elements into the
// head, in adjacent pairs, as DeepSeek lays them out) in place, and copy each head's unrotated
// part into a tile-layout matrix of its own for matmul_heads: head h's is `[rows, nope]`, at
// `head_stride * h`.
extern "C" __global__ void latent_queries(float *q, int stride, int n_heads, int nope, int rope,
                                          const float *cos_table, const float *sin_table,
                                          __nv_bfloat16 *q_nope, unsigned long long head_stride) {
    const int row = blockIdx.x, lane = threadIdx.x, half = rope / 2, width = nope + rope;
    float *queries = q + (size_t)row * stride;
    const float *c = cos_table + (size_t)row * half;
    const float *s = sin_table + (size_t)row * half;
    for (int t = lane; t < n_heads * half; t += blockDim.x) {
        const int i = t % half;
        float *pair = queries + (t / half) * width + nope + 2 * i;
        const float a = pair[0], b = pair[1];
        pair[0] = a * c[i] - b * s[i];
        pair[1] = b * c[i] + a * s[i];
    }
    const int chunks = nope / TILE;
    for (int t = lane; t < n_heads * nope; t += blockDim.x) {
        const int head = t / nope, i = t % nope;
        q_nope[head * head_stride + tile_index(row, i, chunks)] =
            __float2bfloat16(queries[head * width + i]);
    }
}

// One block per row: normalize the latent (the first `rank` of each row's `kv`, rows `stride`
// apart), rotate the keys' shared part that follows it (adjacent pairs), and store both in the
// row's slot of the cache, a `[slots, rank + rope]` tile-layout matrix.
extern "C" __global__ void latent_store(const float *kv, int stride, int rank, int rope,
                                        const float *norm, float eps, const float *cos_table,
                                        const float *sin_table, const int *write_slots,
                                        __nv_bfloat16 *pool) {
    const int row = blockIdx.x, lane = threadIdx.x, half = rope / 2;
    const int chunks = (rank + rope) / TILE, slot = write_slots[row];
    const float *from = kv + (size_t)row * stride;
    float sum = 0;
    for (int i = lane; i < rank; i += WARP)
        sum += from[i] * from[i];
    const float scale = 1.0f / sqrtf(warp_sum(sum) / rank + eps);
    for (int i = lane; i < rank; i += WARP)
        pool[tile_index(slot, i, chunks)] = __float2bfloat16(from[i] * scale * norm[i]);
    const float *c = cos_table + (size_t)row * half;
    const float *s = sin_table + (size_t)row * half;
    for (int i = lane; i < half; i += WARP) {
        const float a = from[rank + 2 * i], b = from[rank + 2 * i + 1];
        pool[tile_index(slot, rank + 2 * i, chunks)] = __float2bfloat16(a * c[i] - b * s[i]);
        pool[tile_index(slot, rank + 2 * i + 1, chunks)] = __float2bfloat16(b * c[i] + a * s[i]);
    }
}

// latent_store for cute-flashinfer's MLA kernel: the same values, in 16-bit floats, the latents
// in `ckv` (`[slots, rank]`) and the rotated parts in `kpe` (`[slots, rope]`).
extern "C" __global__ void latent_store_f16(const float *kv, int stride, int rank, int rope,
                                            const float *norm, float eps, const float *cos_table,
                                            const float *sin_table, const int *write_slots,
                                            __half *ckv, __half *kpe) {
    const int row = blockIdx.x, lane = threadIdx.x, half = rope / 2;
    const size_t slot = write_slots[row];
    const float *from = kv + (size_t)row * stride;
    float sum = 0;
    for (int i = lane; i < rank; i += WARP)
        sum += from[i] * from[i];
    const float scale = 1.0f / sqrtf(warp_sum(sum) / rank + eps);
    for (int i = lane; i < rank; i += WARP)
        ckv[slot * rank + i] = __float2half(__bfloat162float(__float2bfloat16(from[i] * scale * norm[i])));
    const float *c = cos_table + (size_t)row * half;
    const float *s = sin_table + (size_t)row * half;
    for (int i = lane; i < half; i += WARP) {
        const float a = from[rank + 2 * i], b = from[rank + 2 * i + 1];
        kpe[slot * rope + 2 * i] = __float2half(a * c[i] - b * s[i]);
        kpe[slot * rope + 2 * i + 1] = __float2half(b * c[i] + a * s[i]);
    }
}

// One block per row: the queries cute-flashinfer's MLA kernel takes, in 16-bit floats, from each
// head's query through its key matrix (`q_latent`, `[n_heads][rows, rank]`, heads `q_head_stride`
// apart) and the rotated part of each head's query (`q_rope`, rows `q_stride` apart and heads
// `head_width` apart): `q_nope [rows, n_heads, rank]` and `q_pe [rows, n_heads, rope]`.
extern "C" __global__ void mla_queries(const float *q_latent, unsigned long long q_head_stride,
                                       const float *q_rope, int q_stride, int head_width,
                                       int n_heads, int rank, int rope, __half *q_nope,
                                       __half *q_pe) {
    const size_t row = blockIdx.x;
    for (int t = threadIdx.x; t < n_heads * rank; t += blockDim.x) {
        const int head = t / rank, i = t % rank;
        q_nope[row * n_heads * rank + t] =
            __float2half(q_latent[head * q_head_stride + row * rank + i]);
    }
    for (int t = threadIdx.x; t < n_heads * rope; t += blockDim.x) {
        const int head = t / rope, i = t % rope;
        q_pe[row * n_heads * rope + t] =
            __float2half(q_rope[row * q_stride + (size_t)head * head_width + i]);
    }
}

// One block of rank / 8 threads per row and head (grid `[rows, n_heads]`): merge the partial states
// cute-flashinfer's MLA kernel wrote for the row's chunks, in order, as FlashInfer merges them.
// Row `r`'s head `h` (index `r * n_heads + h`) has `merge_count` states, `merge_stride` rows of
// `partial_o` apart from `merge_start`, each with its base-2 log-sum-exp in `partial_lse`. The
// result goes to `out`, each head's `[rows, rank]` in tile layout, `out_head_stride` apart, for
// rows `first_row + r`.
extern "C" __global__ void mla_merge(const __half *partial_o, const float *partial_lse,
                                     const int *merge_start, const int *merge_count,
                                     const int *merge_stride, int first_row, int n_heads, int rank,
                                     __nv_bfloat16 *out, unsigned long long out_head_stride) {
    const int r = blockIdx.x, head = blockIdx.y, col = threadIdx.x * 8;
    const int index = r * n_heads + head;
    const int start = merge_start[index], count = merge_count[index], stride = merge_stride[index];
    float m = -INFINITY, d = 1.0f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < count; k++) {
        const size_t p = (size_t)start + (size_t)k * stride;
        const float l = partial_lse[p];
        const float m_new = fmaxf(m, l);
        const float f1 = exp2f(fmaxf(m - m_new, -INFINITY));
        const float f2 = exp2f(fmaxf(l - m_new, -INFINITY));
        d = d * f1 + f2;
        const __half *o = partial_o + p * rank + col;
#pragma unroll
        for (int e = 0; e < 8; e++)
            acc[e] = acc[e] * f1 + __half2float(o[e]) * f2;
        m = m_new;
    }
    const float inverse = d > 0.0f ? 1.0f / d : 0.0f;
    const int chunks = rank / TILE;
#pragma unroll
    for (int e = 0; e < 8; e++)
        out[head * out_head_stride + tile_index(first_row + r, col + e, chunks)] =
            __float2bfloat16(acc[e] * inverse);
}

// The warps of attention_latent, and the most chunks of a cached row (the latent and the rotated
// part) and of the latent each one works on: rows up to 640 wide, latents up to 512.
#define LATENT_WARPS 4
#define LATENT_KEY_CHUNKS 10
#define LATENT_VALUE_CHUNKS 8

// Attention over the latent cache, on tensor cores: one key/value head that all `n_heads` query
// heads share, its keys `chunks` 16-element chunks wide (the latent, then the rotated part) and
// its values the first `value_chunks` of those (the latent). One block of LATENT_WARPS warps
// per row and 16 query heads (block = row * head_tiles + head tile). With fewer than 16 heads,
// consecutive rows of a sequence share a block, as in attention_tiles: `team_sizes[row]` is how
// many. Tile rows are (row, head) pairs.
//
// The warps share out the work. Each multiplies its share of the key chunks for a page's scores,
// which are then added up in a fixed order, and keeps the running totals of its share of the
// value chunks. The softmax is the single pass of attention_tiles, without windows, caps or sinks,
// which DeepSeek doesn't use.
//
// `q_latent` holds each head's query through that head's key matrix, `[n_heads, rows, rank]`
// (`q_head_stride` elements per head), and the rotated part of each head's query is in `q_rope`,
// rows `q_stride` apart and heads `head_width` apart. The result goes to `out`, each head's
// `[rows, rank]` in tile layout, `out_head_stride` elements apart.
extern "C" __global__ void __launch_bounds__(LATENT_WARPS *WARP)
    attention_latent(const float *q_latent, unsigned long long q_head_stride, const float *q_rope,
                     int q_stride, int head_width, const __nv_bfloat16 *pool, int chunks,
                     int value_chunks, const unsigned *page_lists, const int *row_offset,
                     const int *row_len, const int *team_sizes, __nv_bfloat16 *out,
                     unsigned long long out_head_stride, int n_heads, float scale) {
    extern __shared__ __align__(1024) unsigned char latent_smem[];
    const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    const int head_tiles = (n_heads + TILE - 1) / TILE;
    const int first_row = blockIdx.x / head_tiles, head_tile = blockIdx.x % head_tiles;
    const int team_size = team_sizes[first_row];
    if (team_size == 0)
        return;
    const int per_tile = min(n_heads, TILE), rank = value_chunks * TILE;
    const int first_len = row_len[first_row];
    const int last_page = (first_len + team_size - 2) / TILE;
    const unsigned *pages = page_lists + row_offset[first_row];
    // Tile row t is head `head_tile * 16 + t % per_tile` of the team's row `t / per_tile`.
    const auto valid = [&](int t) {
        return t / per_tile < team_size && head_tile * TILE + t % per_tile < n_heads;
    };

    __nv_bfloat16 *q_tiles = (__nv_bfloat16 *)latent_smem;
    float *partial = (float *)(latent_smem + (size_t)chunks * TILE_ELEMENTS * sizeof(__nv_bfloat16));
    __nv_bfloat16 *weights = (__nv_bfloat16 *)(partial + LATENT_WARPS * TILE_ELEMENTS);
    float *inverse = (float *)(weights + TILE_ELEMENTS);

    for (int i = threadIdx.x; i < chunks * TILE_ELEMENTS; i += LATENT_WARPS * WARP) {
        const int c = i / TILE_ELEMENTS, t = i % TILE_ELEMENTS / TILE, j = i % TILE;
        float value = 0.0f;
        if (valid(t)) {
            const size_t row = first_row + t / per_tile;
            const int head = head_tile * TILE + t % per_tile;
            value = c < value_chunks
                        ? q_latent[head * q_head_stride + row * rank + c * TILE + j]
                        : q_rope[row * q_stride + (size_t)head * head_width + (c - value_chunks) * TILE + j];
        }
        q_tiles[i] = __float2bfloat16(value);
    }
    __syncthreads();

    // This warp's chunks of the keys, and of the values.
    const int key_share = (chunks + LATENT_WARPS - 1) / LATENT_WARPS;
    const int key_begin = warp * key_share, key_end = min(chunks, key_begin + key_share);
    const int value_share = (value_chunks + LATENT_WARPS - 1) / LATENT_WARPS;
    const int value_begin = warp * value_share;
    const int value_end = min(value_chunks, value_begin + value_share);
    TileA a[LATENT_KEY_CHUNKS];
#pragma unroll
    for (int i = 0; i < LATENT_KEY_CHUNKS; i++)
        if (key_begin + i < key_end)
            wmma::load_matrix_sync(a[i], q_tiles + (size_t)(key_begin + i) * TILE_ELEMENTS, TILE);
    TileTotal totals[LATENT_VALUE_CHUNKS];
#pragma unroll
    for (int i = 0; i < LATENT_VALUE_CHUNKS; i++)
        wmma::fill_fragment(totals[i], 0.0f);

    // This warp looks after the softmax of tile rows warp, warp + LATENT_WARPS, ...
    constexpr int OWN_ROWS = TILE / LATENT_WARPS;
    float reference[OWN_ROWS], lane_sum[OWN_ROWS];
    for (int j = 0; j < OWN_ROWS; j++)
        lane_sum[j] = 0.0f;

    for (int page = 0; page <= last_page; page++) {
        const __nv_bfloat16 *tiles = pool + (size_t)pages[page] * chunks * TILE_ELEMENTS;
        TileTotal total;
        wmma::fill_fragment(total, 0.0f);
#pragma unroll
        for (int i = 0; i < LATENT_KEY_CHUNKS; i++) {
            if (key_begin + i < key_end) {
                TileBColumns keys;
                wmma::load_matrix_sync(keys, tiles + (size_t)(key_begin + i) * TILE_ELEMENTS, TILE);
                wmma::mma_sync(total, a[i], keys, total);
            }
        }
        wmma::store_matrix_sync(partial + warp * TILE_ELEMENTS, total, TILE, wmma::mem_row_major);
        __syncthreads();

        for (int j = 0; j < OWN_ROWS; j++) {
            const int t = warp + j * LATENT_WARPS;
            float score = 0.0f;
            if (lane < TILE) {
                for (int w = 0; w < LATENT_WARPS; w++)
                    score += partial[w * TILE_ELEMENTS + t * TILE + lane];
            }
            const bool seen =
                lane < TILE && valid(t) && page * TILE + lane < first_len + t / per_tile;
            // Weights are relative to the best score in the first page (see attention_tiles).
            if (page == 0)
                reference[j] = warp_max(seen ? score : -FLT_MAX);
            float weight = 0.0f;
            if (seen)
                weight = expf(fminf((score - reference[j]) * scale, LARGEST_ATTENTION_EXPONENT));
            const __nv_bfloat16 rounded = __float2bfloat16(weight);
            if (lane < TILE)
                weights[t * TILE + lane] = rounded;
            lane_sum[j] += __bfloat162float(rounded);
        }
        __syncthreads();

        TileA weight_tile;
        wmma::load_matrix_sync(weight_tile, weights, TILE);
#pragma unroll
        for (int i = 0; i < LATENT_VALUE_CHUNKS; i++) {
            if (value_begin + i < value_end) {
                TileBRows values;
                wmma::load_matrix_sync(values, tiles + (size_t)(value_begin + i) * TILE_ELEMENTS,
                                       TILE);
                wmma::mma_sync(totals[i], weight_tile, values, totals[i]);
            }
        }
    }

    for (int j = 0; j < OWN_ROWS; j++) {
        const float sum = warp_sum(lane_sum[j]);
        if (lane == 0)
            inverse[warp + j * LATENT_WARPS] = sum > 0.0f ? 1.0f / sum : 0.0f;
    }
    __syncthreads();
    float *staging = partial + warp * TILE_ELEMENTS;
#pragma unroll
    for (int i = 0; i < LATENT_VALUE_CHUNKS; i++) {
        if (value_begin + i >= value_end)
            continue;
        wmma::store_matrix_sync(staging, totals[i], TILE, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < TILE_ELEMENTS; e += WARP) {
            const int t = e / TILE;
            if (valid(t)) {
                const int row = first_row + t / per_tile, head = head_tile * TILE + t % per_tile;
                out[head * out_head_stride +
                    tile_index(row, (value_begin + i) * TILE + e % TILE, value_chunks)] =
                    __float2bfloat16(staging[e] * inverse[t]);
            }
        }
        __syncwarp();
    }
}

// One block per row: gather each head's `width` outputs (`[n_heads, rows, width]`, `head_stride`
// elements per head) into the row of a `[rows, n_heads * width]` tile-layout matrix.
extern "C" __global__ void heads_to_tiles(const float *x, unsigned long long head_stride,
                                          int n_heads, int width, __nv_bfloat16 *out) {
    const int row = blockIdx.x, chunks = n_heads * width / TILE;
    for (int i = threadIdx.x; i < n_heads * width; i += blockDim.x)
        out[tile_index(row, i, chunks)] =
            __float2bfloat16(x[(i / width) * head_stride + (size_t)row * width + i % width]);
}

// One warp per row, for DeepSeek's routing (see `Routing` in config.rs): weights from the
// softmax of the scores, or from their sigmoids; experts chosen by those plus `correction` (if
// there is one), from the best `top_groups` of `groups` equal groups, each scored by its best
// member (softmax) or its best two (sigmoids); the top_k best of those, renormalized if asked,
// then all multiplied by `scale`. Ties go to the lower number. Written in order of expert.
#define MAX_GROUPS 32
extern "C" __global__ void moe_route_grouped(const float *scores, int experts, int top_k,
                                             int renormalize, int sigmoid, int groups,
                                             int top_groups, float scale,
                                             const float *correction, int *topk_ids,
                                             float *topk_weights) {
    __shared__ float weight[MAX_EXPERTS], choice[MAX_EXPERTS], group_score[MAX_GROUPS];
    __shared__ int allowed[MAX_GROUPS];
    const int row = blockIdx.x, lane = threadIdx.x;
    const float *s = scores + (size_t)row * experts;
    float max = -FLT_MAX, sum = 0.0f;
    if (!sigmoid) {
        for (int i = lane; i < experts; i += WARP)
            max = fmaxf(max, s[i]);
        max = warp_max(max);
        for (int i = lane; i < experts; i += WARP)
            sum += expf(s[i] - max);
        sum = warp_sum(sum);
    }
    for (int i = lane; i < experts; i += WARP) {
        const float w = sigmoid ? 1.0f / (1.0f + expf(-s[i])) : expf(s[i] - max) / sum;
        weight[i] = w;
        choice[i] = correction ? w + correction[i] : w;
    }
    __syncwarp();
    const int per_group = experts / groups;
    for (int g = lane; g < groups; g += WARP) {
        float first = -FLT_MAX, second = -FLT_MAX;
        for (int i = g * per_group; i < (g + 1) * per_group; i++) {
            const float c = choice[i];
            if (c > first) {
                second = first;
                first = c;
            } else if (c > second) {
                second = c;
            }
        }
        group_score[g] = sigmoid ? first + (per_group > 1 ? second : 0.0f) : first;
        allowed[g] = 0;
    }
    __syncwarp();
    if (lane == 0) {
        for (int k = 0; k < min(top_groups, groups); k++) {
            int best = -1;
            for (int g = 0; g < groups; g++)
                if (!allowed[g] && (best < 0 || group_score[g] > group_score[best]))
                    best = g;
            allowed[best] = 1;
        }
    }
    __syncwarp();
    int chosen[MAX_TOP_K];
    float chosen_weight[MAX_TOP_K];
    float total = 0.0f;
    for (int k = 0; k < top_k; k++) {
        float best = -FLT_MAX;
        int best_i = 0x7fffffff;
        for (int i = lane; i < experts; i += WARP) {
            bool taken = !allowed[i / per_group];
            for (int j = 0; j < k; j++)
                taken |= chosen[j] == i;
            if (!taken && (choice[i] > best || (choice[i] == best && i < best_i))) {
                best = choice[i];
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
        chosen_weight[k] = weight[best_i];
        total += chosen_weight[k];
    }
    if (lane != 0)
        return;
    for (int k = 1; k < top_k; k++) {
        for (int j = k; j > 0 && chosen[j - 1] > chosen[j]; j--) {
            const int id = chosen[j];
            chosen[j] = chosen[j - 1];
            chosen[j - 1] = id;
            const float w = chosen_weight[j];
            chosen_weight[j] = chosen_weight[j - 1];
            chosen_weight[j - 1] = w;
        }
    }
    for (int k = 0; k < top_k; k++) {
        const float w = renormalize ? chosen_weight[k] / (total + 1e-20f) : chosen_weight[k];
        topk_ids[row * top_k + k] = chosen[k];
        topk_weights[row * top_k + k] = w * scale;
    }
}

// One block per row, for a shared expert with a gate of its own (Qwen2 MoE): x += sigmoid(gate) * y.
extern "C" __global__ void add_gated(const float *y, const float *gate, int dim, float *x) {
    const int row = blockIdx.x;
    const float scale = 1.0f / (1.0f + expf(-gate[row]));
    for (int i = threadIdx.x; i < dim; i += WARP)
        x[(size_t)row * dim + i] += scale * y[(size_t)row * dim + i];
}

// ---------------------------------------------------------------------------------------------
// EAGLE draft heads
// ---------------------------------------------------------------------------------------------

// One block per row: copy the row's `dim` floats into `out`, whose rows are `stride` apart, at
// `offset` into the row (keeping a layer's residual stream for an EAGLE head, say).
extern "C" __global__ void copy_rows(const float *x, int dim, float *out, int stride, int offset) {
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < dim; i += WARP)
        out[(size_t)row * stride + offset + i] = x[(size_t)row * dim + i];
}

// One block per row: x's row, `cols` floats, as bfloat16 in its place in a tile-layout matrix.
// Tensor parallelism: copy this GPU's part of a block's output (`count` floats, a multiple of 4)
// into every other GPU's copy of it. Writes across the bus don't wait, unlike reads.
// Copy whole pages of a KV pool, `words` 32-bit words each: page pairs[2b] to pairs[2b + 1] in
// block b. For draft branches, which need their own copy of the page where they split off.
extern "C" __global__ void copy_pages(unsigned *pool, const unsigned *pairs, size_t words) {
    const unsigned *from = pool + pairs[2 * blockIdx.x] * words;
    unsigned *to = pool + pairs[2 * blockIdx.x + 1] * words;
    for (size_t i = threadIdx.x; i < words; i += blockDim.x)
        to[i] = from[i];
}

extern "C" __global__ void tp_push(const float4 *part, const unsigned long long *peers, int rank,
                                   int size, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count / 4)
        return;
    const float4 v = part[i];
    for (int r = 0; r < size; r++)
        if (r != rank)
            ((float4 *)peers[r])[i] = v;
}

// Then, one thread: count this point's step (`seq`), and tell every other GPU, by writing the
// count into its flags.
extern "C" __global__ void tp_signal(unsigned *seq, const unsigned long long *peer_flags, int rank,
                                     int size) {
    if (threadIdx.x != 0)
        return;
    const unsigned next = seq[0] + 1;
    seq[0] = next;
    __threadfence_system();
    for (int r = 0; r < size; r++)
        if (r != rank)
            ((volatile unsigned *)peer_flags[r])[rank] = next;
}

// One block per row: once every other GPU's flag has reached this step's count, x += the sum of
// every GPU's part (GPU r's at parts + r * stride, all here), added up in the same order on every
// GPU so they all keep the same x.
extern "C" __global__ void reduce_add(float *x, const float *parts, size_t stride,
                                      const unsigned *seq, const unsigned *flags, int rank,
                                      int count, int dim) {
    if (threadIdx.x == 0) {
        const unsigned want = seq[0];
        for (int r = 0; r < count; r++)
            if (r != rank)
                while (((volatile const unsigned *)flags)[r] < want) {
                }
    }
    __syncwarp();
    __threadfence_system();
    const size_t row = (size_t)blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += WARP) {
        float sum = 0.0f;
        for (int r = 0; r < count; r++)
            sum += __ldcg(parts + r * stride + row + i);
        x[row + i] += sum;
    }
}

// One block per row given its embedding (an image's): x[rows[i]] = values[i].
extern "C" __global__ void place_rows(const float *values, const int *rows, float *x, int dim) {
    const float *from = values + (size_t)blockIdx.x * dim;
    float *to = x + (size_t)rows[blockIdx.x] * dim;
    for (int i = threadIdx.x; i < dim; i += WARP)
        to[i] = from[i];
}

// One warp per (patch, head) of a vision encoder: attention with no mask over the patches of the
// patch's segment, starts[row] to starts[row] + lens[row] (its window, or the whole image). Its
// query is in `queries` (rows `q_stride` apart), and the segment's keys and values are rows of
// `keys` and `values`, `dim` apart. The softmax is built up as it goes, with a running maximum,
// so nothing is stored per key. head_dim at most VISION_HEAD.
#define VISION_HEAD 128
extern "C" __global__ void vision_attention(const float *queries, int q_stride, const float *keys,
                                            const float *values, int dim, int head_dim,
                                            const int *starts, const int *lens, float scale,
                                            float *out) {
    constexpr int PER_LANE = VISION_HEAD / WARP;
    const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const float *q = queries + (size_t)row * q_stride + head * head_dim;
    float qv[PER_LANE], acc[PER_LANE];
    for (int j = 0; j < PER_LANE; j++) {
        const int f = lane + j * WARP;
        qv[j] = f < head_dim ? q[f] : 0.0f;
        acc[j] = 0.0f;
    }
    float best = -INFINITY, total = 0.0f;
    const int first = starts[row], end = first + lens[row];
    for (int p = first; p < end; p++) {
        const float *k = keys + (size_t)p * dim + head * head_dim;
        const float *v = values + (size_t)p * dim + head * head_dim;
        float partial = 0.0f;
        for (int j = 0; j < PER_LANE; j++) {
            const int f = lane + j * WARP;
            if (f < head_dim)
                partial += qv[j] * k[f];
        }
        const float score = warp_sum(partial) * scale;
        const float new_best = fmaxf(best, score);
        const float correction = expf(best - new_best); // 0 for the first key
        const float weight = expf(score - new_best);
        total = total * correction + weight;
        for (int j = 0; j < PER_LANE; j++) {
            const int f = lane + j * WARP;
            if (f < head_dim)
                acc[j] = acc[j] * correction + weight * v[f];
        }
        best = new_best;
    }
    float *o = out + (size_t)row * dim + head * head_dim;
    for (int j = 0; j < PER_LANE; j++) {
        const int f = lane + j * WARP;
        if (f < head_dim)
            o[f] = acc[j] / total;
    }
}

// x = GELU(x), the exact curve, for `count` values.
extern "C" __global__ void gelu_erf(float *x, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752f));
}

extern "C" __global__ void tiles_bf16(const float *x, int cols, __nv_bfloat16 *out, int chunks) {
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < cols; i += WARP)
        out[tile_index(row, i, chunks)] = __float2bfloat16(x[(size_t)row * cols + i]);
}

// One block per row: an EAGLE head's scores for its own vocabulary, put in their tokens' places
// in the model's (`targets[i]` for score i), and -infinity for the tokens it doesn't have.
extern "C" __global__ void scatter_logits(const float *draft, int draft_vocab,
                                          const unsigned *targets, float *logits, int vocab) {
    const int row = blockIdx.x;
    float *to = logits + (size_t)row * vocab;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x)
        to[i] = -INFINITY;
    __syncthreads();
    const float *from = draft + (size_t)row * draft_vocab;
    for (int i = threadIdx.x; i < draft_vocab; i += blockDim.x)
        to[targets[i]] = from[i];
}
