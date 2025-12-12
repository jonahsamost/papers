/*
nvcc -arch=sm_89 -o out test.cu
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <math.h>

using namespace nvcuda;

#define BR 64
#define BC 64
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)
#define WMMA 16

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


__global__ void flash_attn_fwd_kernel(
    const half* __restrict__ Q, 
    const half* __restrict__ K, 
    const half* __restrict__ V, 
    half* __restrict__ O, 
    int N, 
    int d
) {
   extern __shared__ half smem[];
   
   half* sQ = smem;
   half* sK = sQ + (BR * d);
   half* sV = sK + (BC * d);
   half* sS = sV + (BC * d);
   half* sO_scratch = sS + (BR * BC);

   int tx = threadIdx.x;
   int warp_id = tx / WARP_SIZE;
   int lane_id = tx % WARP_SIZE;

   int q_row_start = warp_id * WMMA; 

   int batch_head_offset = blockIdx.z * N * d;
   int q_base_offset = batch_head_offset + (blockIdx.x * BR * d);

   wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, half> acc_O[8];
   for(int i=0; i < (d / WMMA); i++) {
       wmma::fill_fragment(acc_O[i], 0.0);
   }

   float m_prev = -INFINITY;
   float l_prev = 0.0f;

   for (int i = tx; i < BR * d; i += blockDim.x) {
       sQ[i] = Q[q_base_offset + i]; 
   }
   __syncthreads();

   int num_blocks_kv = CEIL_DIV(N, BC);
   for (int j = 0; j < num_blocks_kv; j++) {
       int kv_base_offset = batch_head_offset + (j * BC * d);
       
       __syncthreads();
       for (int i = tx; i < BC * d; i += blockDim.x) {
           sK[i] = K[kv_base_offset + i];
           sV[i] = V[kv_base_offset + i];
       }
       __syncthreads();

       for (int s_col_tile = 0; s_col_tile < (BC / WMMA); s_col_tile++) {
           wmma::fragment<wmma::accumulator, WMMA, WMMA, WMMA, half> acc_S;
           wmma::fill_fragment(acc_S, 0.0);

           for (int d_iter = 0; d_iter < (d / WMMA); d_iter++) {
               
               wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> frag_Q;
               wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::col_major> frag_K;

               int q_offset = (q_row_start * d) + (d_iter * WMMA);
               wmma::load_matrix_sync(frag_Q, &sQ[q_offset], d);

               int k_offset = (s_col_tile * WMMA * d) + (d_iter * WMMA); 
               wmma::load_matrix_sync(frag_K, &sK[k_offset], d);

               wmma::mma_sync(acc_S, frag_Q, frag_K, acc_S);
           }

           int s_store_offset = (q_row_start * BC) + (s_col_tile * WMMA);
           wmma::store_matrix_sync(&sS[s_store_offset], acc_S, BC, wmma::mem_row_major);
       }
        
       __syncthreads();

       if (j > 0) {
           for (int i = 0; i < (d / WMMA); i++) {
               int o_store_offset = (q_row_start * d) + (i * WMMA);
               wmma::store_matrix_sync(&sO_scratch[o_store_offset], acc_O[i], d, wmma::mem_row_major);
           }
       }
       __syncthreads();

       if (lane_id < 16) { 
           // find max and rescale 
           int my_row = q_row_start + lane_id;

           float m_curr = -INFINITY;
           for (int c = 0; c < BC; c++) {
               float val = __half2float(sS[my_row * BC + c]);
               if (val > m_curr) m_curr = val;
           }

           float m_new = fmaxf(m_prev, m_curr);
           float o_scale = (j == 0) ? 1.0f : expf(m_prev - m_new);
           
           if (j > 0) {
               for (int x = 0; x < d; x++) {
                   half val = sO_scratch[my_row * d + x];
                   sO_scratch[my_row * d + x] = __float2half(__half2float(val) * o_scale);
               }
           }

           float l_curr = 0.0f;
           for (int c = 0; c < BC; c++) {
               // rescale current chunk
               float val = __half2float(sS[my_row * BC + c]);
               float p = expf(val - m_new);
               sS[my_row * BC + c] = __float2half(p);
               l_curr += p;
           }

           l_prev = (l_prev * o_scale) + l_curr; // rescale row sum
           m_prev = m_new;
       }
       __syncthreads();

       if (j > 0) {
           for (int i = 0; i < (d / WMMA); i++) {
               int o_load_offset = (q_row_start * d) + (i * WMMA);
               wmma::load_matrix_sync(acc_O[i], &sO_scratch[o_load_offset], d, wmma::mem_row_major);
           }
       }

       for (int o_col_tile = 0; o_col_tile < (d / WMMA); o_col_tile++) {
           for (int v_iter = 0; v_iter < (BC / WMMA); v_iter++) {
               wmma::fragment<wmma::matrix_a, WMMA, WMMA, WMMA, half, wmma::row_major> frag_P;
               wmma::fragment<wmma::matrix_b, WMMA, WMMA, WMMA, half, wmma::row_major> frag_V;

               int p_offset = (q_row_start * BC) + (v_iter * WMMA);
               wmma::load_matrix_sync(frag_P, &sS[p_offset], BC);

               int v_offset = (v_iter * WMMA * d) + (o_col_tile * WMMA);
               wmma::load_matrix_sync(frag_V, &sV[v_offset], d);

               wmma::mma_sync(acc_O[o_col_tile], frag_P, frag_V, acc_O[o_col_tile]);
           }
       }

   }

   for (int i = 0; i < (d / WMMA); i++) {
       int o_store_offset = (q_row_start * d) + (i * WMMA);
       wmma::store_matrix_sync(&sO_scratch[o_store_offset], acc_O[i], d, wmma::mem_row_major);
   }
   __syncthreads();

   if (lane_id < 16) {
       // renormalize by running sum
       int my_row = q_row_start + lane_id;
       float inv_l = 1.0f / l_prev;
       
       for (int x = 0; x < d; x++) {
           float val = __half2float(sO_scratch[my_row * d + x]);
           sO_scratch[my_row * d + x] = __float2half(val * inv_l);
       }
   }
   __syncthreads();

   int output_base = batch_head_offset + (blockIdx.x * BR * d);
   for (int i = tx; i < BR * d; i += blockDim.x) {
       O[output_base + i] = sO_scratch[i];
   }
}

void flash_attn_fwd(const half* Q, const half *K, const half *V, half *output, 
                    int batch, int heads, int N, int d) {

   dim3 grid(CEIL_DIV(N, BR), 1, batch * heads);

   size_t total_smem = sizeof(half) * (BR * d * 2 + BC * d * 2 + BC * BR);
   // more shmem
   cudaFuncSetAttribute(flash_attn_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);

   flash_attn_fwd_kernel<<<grid, THREADS_PER_BLOCK, total_smem>>>(Q, K, V, output, N, d);
   
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));
   }
}



// --- Helper: CPU Reference Implementation (Naive Attention) ---
// This calculates the "Ground Truth" using high-precision floats
void cpu_flash_attn_ref(const float* Q, const float* K, const float* V, float* O, 
                        int batch, int heads, int N, int d) {
    
    // Iterate over Batch (b) and Heads (h)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            
            // Pointer offsets for this specific batch/head
            int offset = (b * heads * N * d) + (h * N * d);
            const float* q_ptr = Q + offset;
            const float* k_ptr = K + offset;
            const float* v_ptr = V + offset;
            float* o_ptr = O + offset;

            // Iterate over each Query Row (i)
            for (int i = 0; i < N; i++) {
                
                // 1. Compute Scores: S = Q[i] * K^T
                // We need to compare Q[i] against every K[j]
                std::vector<float> scores(N);
                float max_score = -1e9;

                for (int j = 0; j < N; j++) {
                    float dot = 0.0f;
                    for (int x = 0; x < d; x++) {
                        dot += q_ptr[i * d + x] * k_ptr[j * d + x];
                    }
                    scores[j] = dot;
                    if (dot > max_score) max_score = dot;
                }

                // 2. Compute Softmax: P = exp(S - max) / sum
                float sum_exp = 0.0f;
                for (int j = 0; j < N; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }

                // 3. Compute Output: O = P * V
                // O[i] = Sum_over_j( P[i,j] * V[j] )
                for (int x = 0; x < d; x++) {
                    float val = 0.0f;
                    for (int j = 0; j < N; j++) {
                        val += (scores[j] / sum_exp) * v_ptr[j * d + x];
                    }
                    o_ptr[i * d + x] = val;
                }
            }
        }
    }
}

// --- Helper: Random Initialization ---
void init_data(half* d_ptr, float* h_ptr, int size) {
    for (int i = 0; i < size; i++) {
        float val = static_cast<float>(rand()) / RAND_MAX; // 0.0 to 1.0
        h_ptr[i] = val;
        d_ptr[i] = __float2half(val);
    }
}

// --- Main Test Function ---
int main() {
    srand(time(NULL));

    // 1. Setup Problem Dimensions
    // Small enough to debug, large enough to trigger tiling
    int batch = 1;
    int heads = 2;
    int N = 256;      // Sequence Length (Must be > 64 to test loop)
    int d = 128;       // Head Dimension (Matches our Kernel Defines)

    long total_elements = batch * heads * N * d;
    size_t bytes = total_elements * sizeof(half);

    printf("Testing FlashAttention: B=%d, H=%d, N=%d, d=%d\n", batch, heads, N, d);

    // 2. Allocate Host Memory
    // We keep float versions for the CPU reference to check accuracy
    std::vector<float> h_Q_float(total_elements);
    std::vector<float> h_K_float(total_elements);
    std::vector<float> h_V_float(total_elements);
    std::vector<float> h_O_ref(total_elements); // CPU Result

    std::vector<half> h_Q_half(total_elements);
    std::vector<half> h_K_half(total_elements);
    std::vector<half> h_V_half(total_elements);
    std::vector<half> h_O_gpu(total_elements);   // GPU Result copied back

    // 3. Initialize Data
    init_data(h_Q_half.data(), h_Q_float.data(), total_elements);
    init_data(h_K_half.data(), h_K_float.data(), total_elements);
    init_data(h_V_half.data(), h_V_float.data(), total_elements);

    // 4. Allocate Device Memory
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);

    // 5. Copy Host -> Device
    cudaMemcpy(d_Q, h_Q_half.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K_half.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V_half.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, bytes);

    // 6. Run GPU Kernel
    printf("Launching Kernel...\n");
    flash_attn_fwd(d_Q, d_K, d_V, d_O, batch, heads, N, d);
    cudaDeviceSynchronize();

    // 7. Copy Result Back
    cudaMemcpy(h_O_gpu.data(), d_O, bytes, cudaMemcpyDeviceToHost);

    // 8. Run CPU Reference
    printf("Running CPU Reference...\n");
    cpu_flash_attn_ref(h_Q_float.data(), h_K_float.data(), h_V_float.data(), h_O_ref.data(), batch, heads, N, d);

    // 9. Validation
    printf("Validating...\n");
    float max_diff = 0.0f;
    float tolerance = 1e-2; // FP16 precision is low, loose tolerance needed
    bool passed = true;

    for (int i = 0; i < total_elements; i++) {
        float gpu_val = __half2float(h_O_gpu[i]);
        float cpu_val = h_O_ref[i];
        float diff = fabs(cpu_val - gpu_val);

        if (diff > max_diff) max_diff = diff;

        if (diff > tolerance) {
            printf("Mismatch at index %d: CPU=%f, GPU=%f, Diff=%f\n", i, cpu_val, gpu_val, diff);
            passed = false;
            break; 
        }
    }

    if (passed) {
        printf("PASSED! Max diff: %f\n", max_diff);
    } else {
        printf("FAILED.\n");
    }

    // 10. Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}