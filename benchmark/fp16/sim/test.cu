/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <ctime>
#include <assert.h>


// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define SQUARE 384

#define MATRIX_M SQUARE
#define MATRIX_N SQUARE
#define MATRIX_K SQUARE



// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, half *c, int M, int N, int K) {
	// Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;
	
   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
 
         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
//      out[idx] = in[idx];
	half a= 1.0f;
      out[idx] = a;
   }
}

int main(int argc, char* argv[]) {
   float *a_fp32;
   float *b_fp32;
   float *c_fp32;
   half *a_fp16;
   half *b_fp16;
   half *c_fp16;
   
   half *c_host_wmma;
   
   
   curandGenerator_t gen;
      
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_fp32, MATRIX_M * MATRIX_N * sizeof(float)));   
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&c_fp16, MATRIX_M * MATRIX_N * sizeof(half)));

   
   c_host_wmma = (half*)malloc(MATRIX_M * MATRIX_N * sizeof(half));

   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));
   curandErrCheck(curandGenerateUniform(gen, c_fp32, MATRIX_M * MATRIX_N));
   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (c_fp16, c_fp32, MATRIX_M * MATRIX_N);
   
   
   curandErrCheck(curandDestroyGenerator(gen));
   
   printf("\nM = %d, N = %d, K = %d.\n\n", MATRIX_M, MATRIX_N, MATRIX_K);

////////////////////////////////////////////////////////////////////////////
   half *a;
   half *b;
   half *c;
   half *c_host_cal;

   a = (half*)malloc(MATRIX_M * MATRIX_K *sizeof(half));
   b = (half*)malloc(MATRIX_K * MATRIX_N *sizeof(half));
   c = (half*)malloc(MATRIX_M * MATRIX_N *sizeof(half));
   c_host_cal = (half*)malloc(MATRIX_M * MATRIX_N *sizeof(half));
   
   cudaErrCheck(cudaMemcpy(a, a_fp16, MATRIX_M * MATRIX_N * sizeof(half), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(b, b_fp16, MATRIX_M * MATRIX_N * sizeof(half), cudaMemcpyDeviceToHost));
   cudaErrCheck(cudaMemcpy(c, c_fp16, MATRIX_M * MATRIX_N * sizeof(half), cudaMemcpyDeviceToHost));

   float t1,t2,t3;
   for(int i = 0 ; i < MATRIX_M; i++){
        for(int j = 0 ; j < MATRIX_N ; j++){
		t3=static_cast<float>(c[i*MATRIX_K+j]);
  		for(int k = 0 ; k < MATRIX_K ; k++){
                	t1 = static_cast<float>(a[i*MATRIX_K+k]);
			t2 = static_cast<float>(b[k*MATRIX_N+j]);
                        t3 +=t1*t2;
		}
		
//		printf("%f ",t3);
        }
  //      printf("\n");
   }


///////////////////////////////////////////////////////////////////////////
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   
   wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_fp16, MATRIX_M, MATRIX_N, MATRIX_K);
   cudaErrCheck(cudaMemcpy(c_host_wmma, c_fp16, MATRIX_M * MATRIX_N * sizeof(half), cudaMemcpyDeviceToHost));
      
   float t;
   for(int i = 0 ; i < MATRIX_M; i++){
	for(int j = 0 ; j < MATRIX_N ; j++){
   		t = static_cast<float>(c_host_wmma[i*MATRIX_N+j]);
		printf("%f ",t);
	}
//	printf("\n");
   }

   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(c_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));
   cudaErrCheck(cudaFree(c_fp16));
   free(c_host_wmma);
   free(a);
   free(b);
   free(c);
   free(c_host_cal);
    
   cudaErrCheck(cudaDeviceReset());
   return 0;
}


