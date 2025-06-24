//#include <stdio.h>
//#include <stdlib.h>
//#include <CL/cl.h>
//
//
//__kernel void add_cpu(__global const float* A, __global const float* B, __global float* C) {
//#ifdef CPU
//    int i = get_global_id(0);
//    C[i] = A[i] + B[i]; // pretend it's optimized for CPU
//#endif
//}
//
//__kernel void add_gpu(__global const float* A, __global const float* B, __global float* C) {
//#ifdef GPU
//    int i = get_global_id(0);
//    C[i] = A[i] + B[i]; // pretend it's optimized for GPU
//#endif
//}
