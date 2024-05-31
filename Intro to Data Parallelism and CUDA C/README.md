# Intro to Data Parallelism and CUDA C

### What is `CUDA C`?
    Compute Unified Device Architecture(CUDA)

### Terminologies
* Device refers to the GPU
* Host refers to CPU

#### Parallelism:
* Task Parallelism
* Data Parallelism

### CUDA C API Functions & Concepts:
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
    * `threadIdx.x`, `blockDim.x` and `blockIdx.x` are predefined variables. There is also `.y` and `.z` which will be discussed later.
    * The global index `i` can then be used in order to access device allocated variables.
    * Therefore, by launching a kernel with n or more threads, one can process vector of length `n`.
    * When the host code launches a kernel, it sets the grid and thread block dimensions via execution configuration parameters. The configuration parameters are given between the `<<<` and `>>>` before the traditional C function arguments. The first configuration parameter gives the number of thread blocks in the grid. The second specifies the number of threads in each thread block.
* `CUDA C Qualifier Keywords` to support heterogenous parallel computing:
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
    * If both `__host__` and `__device__` are used in a function declaration, the compiler generates two versions of the function, one for the device and one for the host. 
    * If a function declaration does not have any CUDA extension keyword, the function defaults into a host function.

### Some notes related to compilation
* Do `nvcc --list-gpu-arch` to get supported architectures by the driver to include in `CMakeLists.txt`
* To compile your code in the traditional way, do `nvcc vectorAddCUDA.cu -o <object_file_name>`
* To get default version of cpp in use: `g++ -dM -E -x c++  /dev/null | grep -F __cplusplus`

### Notes related to implemenating matrix addition on CUDA
To Do:
- [x] Difference in `Eigen::MatrixXd` and `Eigen::Matrix4d`:
    - dynamic size unknow at compilation, would know at runtime
    - The data allocated with `Eigen::MatrixXd` is allocated on heap(therefore not contagious necessarily).
    - The data allocated with `Eigen::Matrix4d` is allocated on stack.
    - Eigen predefined types are defined with: 
    ```cpp
    typedef Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options> MyMatrixType
    ```
    - Examples:
    ```cpp
    Matrix<double, 6, Dynamic>                  // Dynamic number of columns (heap allocation)
    Matrix<double, Dynamic, 2>                  // Dynamic number of rows (heap allocation)
    Matrix<double, Dynamic, Dynamic, RowMajor>  // Fully dynamic, row major (heap allocation)
    Matrix<double, 13, 3>                       // Fully fixed (usually allocated on stack)
    ```
    - Therefore, since the memory allocated are not contagious locations. It creates a problem while copying the data with `cudaMemcpy`
    - [CONFIRM] We need the memory to be contiguous in order to use the CUDA API for `cudaMemcpy` and `cudaMemcpy2D`

- [x] Eigen Docs: https://web.archive.org/web/20231010014108/http://eigen.tuxfamily.org/index.php?title=Main_Page#Documentation
- [ ] Can a dynamic eigen matrix be converted into fixed size matrix and then can cuda work?
    - For now, no
    - In case you want to, look at: http://www.trevorsimonton.com/blog/2016/11/16/transfer-2d-array-memory-to-cuda.html
- [ ] Plus there are warnings from Eigen when you compile with CMake !!!
- [ ] Debugging the examples where `Eigen::Matrix4f` working but not `Eigen::MatrixXf`:
    - [ ] We figured through following sources that maybe it is possible that `cudaMemcpy` and `cudaMemCcpy2D` might need contagious memory locations, but not sure, will confirm
    - Further when I use `Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>`, which ensure storage via storage options of Eigen that the data is stored by row(which I think would be contgious [link](https://web.archive.org/web/20230522013531/https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html)), it sill doesnt work?
    - Does it also require stack memory? This cant be true
    - Referring to `darknet` repo, all the tensors are passed via `float*` pointer types
    - Some other links:
        * https://stackoverflow.com/questions/55966046/is-an-eigen-matrix-created-automatically-on-the-heap/55966077#55966077
        * https://stackoverflow.com/questions/22932260/eigen-library-memory-usage-for-dynamic-vectors
        * https://stackoverflow.com/questions/33347751/dynamic-arrays-vs-variable-length-arrays#:~:text=It%20depends%20on%20the%20language,have%20a%20runtime%2Ddetermined%20length.
        * https://unstop.com/blog/difference-between-static-and-dynamic-memory-allocation
        * https://forums.developer.nvidia.com/t/cudamallocpitch-and-cudamemcpy2d/38935/8
        * https://stackoverflow.com/questions/33504943/2d-arrays-with-contiguous-rows-on-the-heap-memory-for-cudamemcpy2d
        * https://stackoverflow.com/questions/6137218/how-can-i-add-up-two-2d-pitched-arrays-using-nested-for-loops
        * https://stackoverflow.com/questions/24280220/in-cuda-why-cudamemcpy2d-and-cudamallocpitch-consume-a-lot-of-time
        * https://forums.developer.nvidia.com/t/cudamallocpitch-and-cudamemcpy2d/38935
        * https://stevengong.co/notes/CUDA-Memory-Allocation
        * https://github.com/NVIDIA/cuda-samples/tree/master

### Nvidia Tools:
Nvidia Nsight and Compute[link](https://github.com/CisMine/Guide-NVIDIA-Tools)
