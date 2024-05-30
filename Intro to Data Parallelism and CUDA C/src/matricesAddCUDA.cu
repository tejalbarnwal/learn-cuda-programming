#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <memory>

// include cuda API functions and predefined variables
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// matricesAdd kernel function for device
__global__
void matricesAddKernel(Eigen::MatrixXf *A,
                        Eigen::MatrixXf *B,
                        Eigen::MatrixXf *C,
                        int n)
{
    // printf("kernel\n");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.y;
    if ((i < n) && (j < n))
    {
        printf("kernel\n");
        printf("i: %d, j: %d\n", i, j);
        printf("A[%d, %d] = %f, ", i, j, (*A)(0, 0));
        // (*C)(i, j) = (*A)(i, j) + (*B)(i, j);
        // printf("A[%d, %d] = %f, ", i, j, (*A)(i, j));
        // printf("dA[%d, %d] = %f, ", i, j, A->coeff(i, j));
    }
}

// __global__
// void matricesAddKernel(float *A,
//                         float *B,
//                         float *C,
//                         int n)
// {
//     // printf("kernel\n");
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = threadIdx.y;
//     if ((i < n) && (j < n))
//     {
//         printf("kernel\n");
//         // (*C)(i, j) = (*A)(i, j) + (*B)(i, j);
//         // printf("A[%d, %d] = %f, ", i, j, (*A)(i, j));
//     }
// }

// host stub function
// void matricesAdd(Eigen::MatrixXf &output_matrix,
//                 Eigen::MatrixXf &input_matrix_1,
//                 Eigen::MatrixXf &output_matrix_2,
//                 int n)
// {

// }
void printCudaError(cudaError_t &err)
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n";
        exit(EXIT_FAILURE);
    }
    else
    {
        std::cout << "sucess!\n";
    }
}

// void matricesAdd(std::unique_ptr<Eigen::MatrixXf> &A,
//                 std::unique_ptr<Eigen::MatrixXf> &B,
//                 std::unique_ptr<Eigen::MatrixXf> &C,
//                 int n)

void matricesAdd(Eigen::MatrixXf *A,
                Eigen::MatrixXf *B,
                Eigen::MatrixXf *C,
                int n)
{
    int size = n * n * sizeof(float);
    std::cout << "size: " << size << std::endl;
    std::cout << "sizeof(A): " << sizeof(A) << std::endl;

    // float *A_data = A->data();
    // float *B_data = B->data();
    // float *C_data = C->data();

    Eigen::MatrixXf *d_A, *d_B, *d_C;
    // float *d_A, *d_B, *d_C;

    cudaError_t errorAllocateA = cudaMalloc((void**)&d_A, sizeof(A));
    // cudaError_t errorAllocateA = cudaMalloc((void**)&d_A, sizeof(A_data));
    printCudaError(errorAllocateA);
    cudaError_t errorAllocateB = cudaMalloc((void**)&d_B, sizeof(B));
    // cudaError_t errorAllocateB = cudaMalloc((void**)&d_B, sizeof(B_data));
    printCudaError(errorAllocateB);
    cudaError_t errorAllocateC = cudaMalloc((void**)&d_C, sizeof(C));
    // cudaError_t errorAllocateC = cudaMalloc((void**)&d_C, sizeof(C_data));
    printCudaError(errorAllocateC);

    cudaError_t errorCpyA = cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice);
    // cudaError_t errorCpyA = cudaMemcpy(d_A, A_data, sizeof(A_data), cudaMemcpyHostToDevice);
    printCudaError(errorCpyA);
    printf("A[%d, %d] = %f \n", 0, 0, (*A)(0, 0));
    // printf("dA[%d, %d] = %f, ", 0, 0, d_A->coeff(0, 0));
    cudaError_t errorCpyB = cudaMemcpy(d_B, B, sizeof(B), cudaMemcpyHostToDevice);
    // cudaError_t errorCpyB = cudaMemcpy(d_B, B_data, sizeof(B_data), cudaMemcpyHostToDevice);
    printCudaError(errorCpyB);

    
    matricesAddKernel <<< 1, dim3(10, 10) >>> (d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C, d_C, sizeof(d_C), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main()
{
    int n = 6;
    std::cout << "n: " << n << std::endl;
    std::cout <<"sizeof float: " << sizeof(float) << std::endl;

    // std::unique_ptr<Eigen::MatrixXf> input_matrix_1 = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Random(n, n));
    // std::unique_ptr<Eigen::MatrixXf> input_matrix_2 = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Random(n, n));
    // std::unique_ptr<Eigen::MatrixXf> output_matrix = std::make_unique<Eigen::MatrixXf>(Eigen::MatrixXf::Zero(n, n));

    Eigen::MatrixXf *input_matrix_1 = new Eigen::MatrixXf(Eigen::MatrixXf::Random(n, n));
    Eigen::MatrixXf *input_matrix_2 = new Eigen::MatrixXf(Eigen::MatrixXf::Random(n, n));
    Eigen::MatrixXf *output_matrix = new Eigen::MatrixXf(Eigen::MatrixXf::Zero(n, n));


    std::cout << "size of input matrix 1: " << sizeof(input_matrix_1) << std::endl;
    std::cout << "* size of input matrix 1: " << sizeof(*input_matrix_1) << std::endl;
    std::cout << "size of input matrix 1 data: " << sizeof(input_matrix_1->data()) << std::endl;


    std::cout << "input_matrix_1: " << input_matrix_1->coeff(0, 0) << " | " << (*input_matrix_1)(0, 0) << std::endl;
    std::cout << "input_matrix_2: " << input_matrix_2->coeff(0, 0) << " | " << (*input_matrix_2)(0, 0) << std::endl;

    matricesAdd(input_matrix_1, input_matrix_2, output_matrix, n);

    return 0;
}