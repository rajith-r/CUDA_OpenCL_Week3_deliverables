#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define M 512
#define N 1024
#define K 1024

__global__ void matMulKernel(float* A, float* B, float* C, int m, int n , int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

extern "C" void runCudaMatrixMul() {
	// Allocate memory for matrices
	int sizeA = M * K * sizeof(float);
	int sizeB = K * N * sizeof(float);
    int size = M * N * sizeof(float);
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;
	}
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 1.0f;
    }

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matMulKernel << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("CUDA kernel execution time: %.4f ms\n", ms);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print a few results
    printf("CUDA: C[0]=%f, C[M*N-1]=%f\n", h_C[0], h_C[M * N - 1]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}