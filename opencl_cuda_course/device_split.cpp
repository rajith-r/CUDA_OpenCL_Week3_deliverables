#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 512

void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s failed (%d)\n", msg, err);
        exit(1);
    }
}

char* load_kernel_source(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) { perror("fopen"); exit(1); }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char* src = malloc(size + 1);
    fread(src, 1, size, fp);
    src[size] = '\0';
    fclose(fp);
    return src;
}

int main() {
    cl_platform_id platform;
    cl_device_id devices[2];
    cl_uint num_devices;
    cl_context context;
    cl_command_queue queues[2];
    cl_program programs[2];
    cl_kernel kernels[2];
    cl_int err;

    float* A = malloc(N * sizeof(float));
    float* B = malloc(N * sizeof(float));
    float* C = malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = N - i;
    }

    // Get platform and devices
    check(clGetPlatformIDs(1, &platform, NULL), "clGetPlatformIDs");
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 2, devices, &num_devices), "clGetDeviceIDs");
    if (num_devices < 2) {
        fprintf(stderr, "Need at least 2 devices (e.g., CPU + GPU)\n");
        return 1;
    }

    context = clCreateContext(NULL, 2, devices, NULL, NULL, &err);
    check(err, "clCreateContext");

    // Create queues
    for (int i = 0; i < 2; i++) {
        queues[i] = clCreateCommandQueue(context, devices[i], 0, &err);
        check(err, "clCreateCommandQueue");
    }

    // Load kernel
    const char* src = load_kernel_source("kernel.cl");

    // Create and build per-device programs
    const char* build_flags[] = { "-DCPU", "-DGPU" };
    const char* kernel_names[] = { "add_cpu", "add_gpu" };

    for (int i = 0; i < 2; i++) {
        programs[i] = clCreateProgramWithSource(context, 1, &src, NULL, &err);
        check(err, "clCreateProgramWithSource");
        err = clBuildProgram(programs[i], 1, &devices[i], build_flags[i], NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(programs[i], devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char* log = malloc(log_size);
            clGetProgramBuildInfo(programs[i], devices[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Build log for device %d:\n%s\n", i, log);
            free(log);
            exit(1);
        }

        kernels[i] = clCreateKernel(programs[i], kernel_names[i], &err);
        check(err, "clCreateKernel");
    }

    // Split data
    int half = N / 2;
    cl_mem buffers_A[2], buffers_B[2], buffers_C[2];
    for (int i = 0; i < 2; i++) {
        int offset = i * half;
        buffers_A[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, half * sizeof(float), A + offset, &err);
        buffers_B[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, half * sizeof(float), B + offset, &err);
        buffers_C[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, half * sizeof(float), NULL, &err);
        check(err, "clCreateBuffer");
    }

    // Set kernel args and launch
    size_t global_size = half;
    for (int i = 0; i < 2; i++) {
        clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &buffers_A[i]);
        clSetKernelArg(kernels[i], 1, sizeof(cl_mem), &buffers_B[i]);
        clSetKernelArg(kernels[i], 2, sizeof(cl_mem), &buffers_C[i]);
        clEnqueueNDRangeKernel(queues[i], kernels[i], 1, NULL, &global_size, NULL, 0, NULL, NULL);
    }

    // Read back results
    for (int i = 0; i < 2; i++) {
        clEnqueueReadBuffer(queues[i], buffers_C[i], CL_TRUE, 0, half * sizeof(float), C + i * half, 0, NULL, NULL);
    }

    printf("C[0] = %f, C[N-1] = %f\n", C[0], C[N - 1]);

    // Cleanup
    for (int i = 0; i < 2; i++) {
        clReleaseMemObject(buffers_A[i]);
        clReleaseMemObject(buffers_B[i]);
        clReleaseMemObject(buffers_C[i]);
        clReleaseKernel(kernels[i]);
        clReleaseProgram(programs[i]);
        clReleaseCommandQueue(queues[i]);
    }

    clReleaseContext(context);
    free(src); free(A); free(B); free(C);
    return 0;
}
