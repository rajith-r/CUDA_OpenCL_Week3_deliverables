#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define M 1024
#define N 1024
#define K 1024

static const char* kernelSource =
"__kernel void matmul(__global float* A, __global float* B, __global float* C, int m, int n, int k) {"
"    int row = get_global_id(1);"
"    int col = get_global_id(0);"
"    float sum = 0.0f;"
"    for (int i = 0; i <k; ++i) {"
"        sum += A[row * k + i] * B[i * n + col];"
"    }"
"    C[row * n + col] = sum;"
"}";

extern "C" void runOpenCLMatrixMul() {
	int size_a = M * K * sizeof(float);
	int size_b = K * N * sizeof(float);
	int size = M * N * sizeof(float);
    float* h_A = (float*)malloc(size_a);
    float* h_B = (float*)malloc(size_b);
    float* h_C = (float*)malloc(size);

    
	// Initialize matrices A and B
    for (int i = 0; i < M * K; i++) {
        h_A[i] = 1.0;
	}
    for (int i = 0; i < K * N; i++) {
		h_B[i] = 1.0;
	}

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_A, d_B, d_C;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_a, h_A, &err);
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_b, h_B, &err);
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matmul", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    int n = N;
	int m = M;
	int k = K;
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);

    size_t global[2] = { N, M };
    cl_event kernel_event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &kernel_event);
    clWaitForEvents(1, &kernel_event);

    cl_ulong start, end;
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double elapsed_time_ms = (end - start) * 1e-6;
    printf("OpenCL kernel execution time: %.4f ms\n", elapsed_time_ms);

    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size, h_C, 0, NULL, NULL);

    printf("OpenCL: C[0]=%f, C[M*N-1]=%f\n", h_C[0], h_C[M * N - 1]);

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_A);
    free(h_B);
    free(h_C);
}