#pragma once
#ifndef CUDA_H
#define CUDA_H

// These code are from YOLOV2

extern int gpu_index;

#ifdef GPU

#define BLOCK 512

// runtime can't be in extern

#include "cuda_runtime.h"

#ifdef __cplusplus

extern "C"{

#endif

// may not use
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#include "cudnn.h"
#endif

void check_error(cudaError_t status);
dim3 cuda_gridsize(size_t n);
dim3 cuda_gridsize_v2(size_t n, size_t n_block);

#ifdef __cplusplus

};

#endif

#endif // GPU
#endif



