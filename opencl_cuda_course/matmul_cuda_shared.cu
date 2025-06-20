#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void matmul_shared(float *a, float *b, float *c, int M, int  N,  int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int tile_height = blockDim.y;
	int tile_width = blockDim.x;

	__shared__ float a_shared[16][16];
	__shared__ float b_shared[16][16];


	int phases = (K + tile_height - 1) / tile_height;
	float sum = 0.0f;
	for (int phase = 0; phase < phases; phase++) {
		int a_row = row;
		int a_col = phase * tile_width + threadIdx.x;

		if (a_row < M && a_col < K) {
			a_shared[threadIdx.y][threadIdx.x] = a[a_row * K + a_col];
		} else {
			a_shared[threadIdx.y][threadIdx.x] = 0.0f;
		}

		int b_row = phase * tile_height + threadIdx.y;
		int b_col = col;
		if (b_row < K && b_col < N) {
			b_shared[threadIdx.y][threadIdx.x] = b[b_row * N + b_col];
		} else {
			b_shared[threadIdx.y][threadIdx.x] = 0.0f;
		}
		__syncthreads();
		
		for (int k = 0; k < tile_height; k++) {
			sum += a_shared[threadIdx.y][k] * b_shared[k][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < M && col < N) {
		c[row * N + col] = sum;
	}
}


extern "C" void runCudaMatMulShared() {
	size_t N = 1024;
	size_t M = 512;
	size_t K = 1024; 


	size_t size_A = M * K * sizeof(float);
	size_t size_B = K * N * sizeof(float);
	size_t size_C = M * N * sizeof(float);

	float* h_A = (float*)malloc(size_A);
	float* h_B = (float*)malloc(size_B);
	float* h_C = (float*)malloc(size_C);

	// Initialize matrices
	for (int i = 0; i < M*K; ++i) {
		h_A[i] = 1.0f;
	}
	for (int i = 0; i < K*N; ++i) {
		h_B[i] = 1.0f;
	}

	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, size_A);
	cudaMalloc(&d_B, size_B);
	cudaMalloc(&d_C, size_C);

	cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);


	dim3 block(16, 16);
	dim3 grid((N + block.x - 1) / block.x,(M + block.y - 1) / block.y);

	// Launch kernel
	matmul_shared << <grid, block >> > (d_A, d_B, d_C, M,N,K);
	cudaDeviceSynchronize();

	// Benchmarking
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Kernel execution time: %.4f ms\n", milliseconds);

	cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

	//print result
	printf("CUDA: C[0]=%f, C[M*N-1]=%f\n", h_C[0], h_C[M * N - 1]);
}