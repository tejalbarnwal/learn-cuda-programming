#include <iostream>
#include <cmath>

// include cuda related which gives us API for tranferiing data
#include <cuda_runtime.h>
// including predefined variabled
#include <device_launch_parameters.h>


// vectorAdd function for device
__global__
void vectorAddKernel(float* A, float* B, float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        C[i] = A[i] + B[i];
        // printf("i= %d, ", i);
        // printf("A[i]= %f, ", A[i]);
        // printf("B[i]= %f, ", B[i]);
        // printf("C[i]= %f, ", C[i]);
    }
}

__global__
void printVector(float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        // std::cout << C[i] << " , "; cout isnt implemented in CUDA
        printf("C[%d] = %f, ", i, C[i]);
    }
}


int main()
{
    float h_A[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_B[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    int n = sizeof(h_A)/ sizeof(h_A[0]);

    float h_C[sizeof(h_A)];

    float *d_A, *d_B, *d_C; 

    // allocate device memory for A, B, C in the global memory
    // where did we specify to use global memory?
    // here d_A will point to device memory region allocated for the A vector
    cudaError_t errAllocateA = cudaMalloc((void**)&d_A, sizeof(h_A));
    if(errAllocateA != cudaSuccess)
    {
        std::cout << cudaGetErrorString(errAllocateA) << " in " << __FILE__ << " at line " << __LINE__ << "\n";
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**)&d_B, sizeof(h_B));
    cudaMalloc((void**)&d_C, sizeof(h_C));

    // copy A and B to device memory
    cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice);

    // kernel launch code to have the device to perform the actual vector addition
    vectorAddKernel <<< std::ceil(n/256.0), 256 >>> (d_A, d_B, d_C, n);
    printVector <<< std::ceil(n/256.0), 256 >>> (d_C, n);
    // copy C from the device memory
    cudaMemcpy(h_C, d_C, sizeof(d_C), cudaMemcpyDeviceToHost);
    // free device vectors
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

