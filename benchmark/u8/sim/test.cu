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
__global__ void wmma_example(unsigned char *a, unsigned char *b, int *c, int M, int N, int K) {
	
// Leading dimensions. Packed with no transpositions.
   int lda = K;
   int ldb = N;
   int ldc = N;
	
   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, unsigned char, wmma::row_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, unsigned char, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

   wmma::fill_fragment(acc_frag, 0);
   
   int t;
   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow*lda + aCol , lda);
         wmma::load_matrix_sync(b_frag, b + bRow*ldb + bCol , ldb);

         
	 // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
         	
      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow*ldc + cCol, ldc, wmma::mem_row_major);


      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow*ldc + cCol, c_frag, ldc, wmma::mem_row_major);
 /*     for(int i = 0 ; i < c_frag.num_elements;i++){
      	t = static_cast<int>(c_frag.x[i]);
        printf("thread C : %d\n", t);
      }*/
   }
}

__global__ void set_value (unsigned char *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      unsigned char a=1;
      in[idx]=a;
   }
}

__global__ void set_value (int *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      int a=1;
      in[idx]=a;
   }
}

int main(int argc, char* argv[]) {
   unsigned char *a_u8;
   unsigned char *b_u8;
   int *c_int;
   
   int *c_host_wmma;
      
   cudaErrCheck(cudaMalloc((void**)&a_u8, MATRIX_M * MATRIX_K * sizeof(unsigned char)));
   cudaErrCheck(cudaMalloc((void**)&b_u8, MATRIX_K * MATRIX_N * sizeof(unsigned char)));
   cudaErrCheck(cudaMalloc((void**)&c_int, MATRIX_M * MATRIX_N * sizeof(int)));   
   
   c_host_wmma = (int*)malloc(MATRIX_M * MATRIX_N * sizeof(int));

   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   set_value <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_u8, MATRIX_M*MATRIX_K);
   set_value <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_u8, MATRIX_K*MATRIX_N);
   set_value <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (c_int, MATRIX_M*MATRIX_N);
    
   printf("\nM = %d, N = %d, K = %d.\n\n", MATRIX_M, MATRIX_N, MATRIX_K);

   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   
   wmma_example <<< gridDim, blockDim >>> (a_u8, b_u8, c_int, MATRIX_M, MATRIX_N, MATRIX_K);
   cudaErrCheck(cudaMemcpy(c_host_wmma, c_int, MATRIX_M * MATRIX_N * sizeof(int), cudaMemcpyDeviceToHost));
      
   int t;
   for(int i = 0 ; i < MATRIX_M; i++){
	for(int j = 0 ; j < MATRIX_N ; j++){
   		t = (c_host_wmma[i*MATRIX_N+j]);
		printf("%d ",t);
	}
	printf("\n");
   }

   cudaErrCheck(cudaFree(a_u8));
   cudaErrCheck(cudaFree(b_u8));
   cudaErrCheck(cudaFree(c_int));
   
   free(c_host_wmma);
   cudaErrCheck(cudaDeviceReset());
   return 0;
}


