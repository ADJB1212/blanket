#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __forceinline__ inline __attribute__((always_inline))
#pragma clang force_cuda_host_device begin
#include <math.h>
#pragma clang force_cuda_host_device end
struct Dim {
    unsigned x;
};
extern __device__ const Dim blockIdx, blockDim, threadIdx;
