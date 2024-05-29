# Intro to Data Parallelism and CUDA C

### What is `CUDA C`?

### Terminologies
* Device refers to the GPU
* Host refers to CPU

#### Parallelism:
* Task Parallelism
* Data Parallelism

### CUDA C Functions & Concepts:
* `cudaMalloc()`: the function is called from host in order to allocate a piece of device global memory for an object
    ```cpp
    cudaError_t cudaMalloc(void** add, int size)

    /*
    cudaError_t: returns error flag if occurred
    add: address of the pointer variable that will be set to point to the allocated object
    size: size of the allocated object in terms of bytes
    */
    ```

* `cudaFree()`: frees object from device global memory
    ```cpp
    cudaError_t cudaFree(T* pointer)
    /*
    pointer: pointer variable to the freed object
    */
    ```
* `cudaMemcpy`: memory data transfers
    ```cpp
    cudaError_t cudaMemcpy(T* destPointer, T* srcPointer, int NoBytesCopy, symbolicConstant)
    /*
    destPointer: pointer to the destination location for the data object to be copied
    srcPointer: pointer to the source location
    NoBytesCopy: no of bytes to be copied
    symbolicConstant: indicate the direction of transfer/ types of memory involved(h to h, h to d, d to h, d to d)
    ** cant be used to copy between different GPUs in multiGPU systems
    */
    ```
* `CUDA Kernel Function`: specifies the code to be executed by all threads during a parallel phase
    * When a host code launches a kernel, the CUDA runtime system generates a grid of threads that ore organized in a two level hierarchy
    * Each grid is organized into array of thread blocks. All blocks of a grid are of the same size and each block can contain upto 1024(based on CUDA 3.0) threads.
    * The number of blocks in the grid and threads in each block is specified the host code when the kernel is launched.
    * In general, the dimensions of thread blocks should be multiple of 32 due hardware efficiency reasons
    * Each thread in a block has a unique `threadIdx` value starting from 0, 1, 2, 3 ...
    * Each block in the grid has a unqiue `blockIdx` value starting from 0, 1, 2, 3...
    * Let say that each block contains 256 threads, then each thread can be globally identified by a global index i, where `i = blockIdx.x * blockDim.x + threadIdx.x`. Here, `blockDim.x = 256`.
    * `threadIdx.x`, `blockDim.x` and `blockIdx.x` are keywords. There is also `.y` and `.z` which will be discussed later.
    * The global index `i` can then be used in order to access device allocated variables.
    * Therefore, by launching a kernel with n or more threads, one can process vector of length `n`.
    * When the host code launches a kernel, it sets the grid and thread block dimensions via execution configuration parameters. The configuration parameters are given between the `<<<` and `>>>` before the traditional C function arguments. The first configuration parameter gives the number of thread blocks in the grid. The second specifies the number of threads in each thread block.
* `CUDA C Qualifier Keywords`
    * `__device__`:
        * Indicated that the function being declared is a CUDA Device Function
        * Device function executes on a CUDA device and can only be called from a kernel function or another device function
        * Generally, indirect function calls and recursions in device and kernel functions to allow maximal portability
    * `__global__`:
        * Indicates that the function being declared is a CUDA Kernel Function
    * `__host__`:
        * Indicates that the function being declared is a CUDA Host Function
        * Host function is a traditional C function that executes on the host and can only be called from another host function
        * By default, all functions in CUDA program are host functions
    *   | Keyword       | Executed On       | Only Callable from the |
        | ------------- | ----------------- | ---------------------- |
        | `__device__`  | device            | device                 |
        | `__global__`  | device            | host                   |
        | `__host__`    | host              | host                   |
