/*
 * Copyright (c) 2022 Ruilong Li, UC Berkeley.
 */

#pragma once

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Exceptions.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CUDA_GET_THREAD_ID(tid, Q)                         \
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= Q)                                          \
    return
#define CUDA_N_BLOCKS_NEEDED(Q, CUDA_N_THREADS) ((Q - 1) / CUDA_N_THREADS + 1)
#define DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

template <typename scalar_t>
inline __device__ __host__ scalar_t ceil_div(scalar_t a, scalar_t b)
{
  return (a + b - 1) / b;
}