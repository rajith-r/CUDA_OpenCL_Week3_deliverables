#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <CL/cl.hpp>

#ifdef __cplusplus
extern "C" {
#endif
    void runCudaMatrixMul();
    void runOpenCLMatrixMul();
    void runCudaMatMulShared();
    void runOpenCLMatMulShared();
#ifdef __cplusplus
}
#endif

int main() {
    printf("=== CUDA Matrix Multiplication ===\n");
    runCudaMatrixMul();

    printf("\n=== OpenCL Matrix Multiplication ===\n");
    runOpenCLMatrixMul();

	printf("\n=== CUDA Matrix Multiplication with Shared Memory ===\n");
	runCudaMatMulShared();

    printf("\n=== OpenCL Matrix Multiplication with Shared Memory ===\n");
    runOpenCLMatMulShared();

	printf("\n=== All computations completed successfully ===\n");
   
    return 0;
}