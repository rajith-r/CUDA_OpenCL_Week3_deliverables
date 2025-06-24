#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <iostream>
#define M 1024
#define N 1024
#define K 1024
#define TILE_SIZE 32

static const char* kernelSource =
"__kernel void matmul_shared(__global float* a, __global float* b, __global float* c,"
"                             __local float* a_shared_flat, __local float* b_shared_flat,"
"                             int M, int N, int K) {"
"    int row = get_global_id(1);"
"    int col = get_global_id(0);"
"    int tile_height = get_local_size(1);"
"    int tile_width  = get_local_size(0);"
"    int local_row = get_local_id(1);"
"    int local_col = get_local_id(0);"

"    int phases = (K + tile_width - 1) / tile_width;"
"    float sum = 0.0f;"
"    for (int phase = 0; phase < phases; ++phase) {"
"        int a_row = row;"
"        int a_col = phase * tile_width + local_col;"
"        int b_row = phase * tile_height + local_row;"
"        int b_col = col;"

"        if (a_row < M && a_col < K)"
"            a_shared_flat[local_row * tile_width + local_col] = a[a_row * K + a_col];"
"        else"
"            a_shared_flat[local_row * tile_width + local_col] = 0.0f;"

"        if (b_row < K && b_col < N)"
"            b_shared_flat[local_row * tile_width + local_col] = b[b_row * N + b_col];"
"        else"
"            b_shared_flat[local_row * tile_width + local_col] = 0.0f;"

"        barrier(CLK_LOCAL_MEM_FENCE);"

"        for (int k = 0; k < tile_width; ++k) {"
"            sum += a_shared_flat[local_row * tile_width + k] *"
"                   b_shared_flat[k * tile_width + local_col];"
"        }"
"        barrier(CLK_LOCAL_MEM_FENCE);"
"    }"
"    if (row < M && col < N)"
"        c[row * N + col] = sum;"
"}";

extern "C" void runOpenCLMatMulShared() {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_A, d_B, d_C;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
	//std::cout << "OpenCL Platform: " << platform << std::endl;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	//std::cout << "OpenCL Device: " << device << std::endl;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);


    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, h_A, &err);
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_B, h_B, &err);
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_C, NULL, &err);

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matmul_shared", &err);

    int m_val = M, n_val = N, k_val = K;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, TILE_SIZE * TILE_SIZE * sizeof(float), NULL); // a_shared_flat
    clSetKernelArg(kernel, 4, TILE_SIZE * TILE_SIZE * sizeof(float), NULL); // b_shared_flat
    clSetKernelArg(kernel, 5, sizeof(int), &m_val);
    clSetKernelArg(kernel, 6, sizeof(int), &n_val);
    clSetKernelArg(kernel, 7, sizeof(int), &k_val);


    size_t global[2] = { N, M };
    size_t local[2] = { TILE_SIZE, TILE_SIZE };

    cl_event kernel_event;
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &kernel_event);
    clWaitForEvents(1, &kernel_event);
    // Get profiling info
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double elapsed_time_ms = (time_end - time_start) * 1e-6;  // Convert from ns to ms
    printf("OpenCL kernel execution time: %.3f ms\n", elapsed_time_ms);

    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size_C, h_C, 0, NULL, NULL);

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
