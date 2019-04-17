//int gpu_index = 0;

#ifdef GPU

// These code are from YOLOV2

#include "cuda_tools.h"
#include "assert.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define error(info) fprintf(stderr, "%s\n", info);

void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
		fprintf(stderr, "%s\n", buffer);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

dim3 cuda_gridsize_v2(size_t n, size_t n_block) {
	size_t k = (n - 1) / n_block + 1;
	size_t x = k;
	size_t y = 1;
	if (x > 65535) {
		x = ceil(sqrt(k));
		y = (n - 1) / (x*n_block) + 1;
	}
	dim3 d = { x, y, 1 };
	//printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
	return d;
}


#endif
