#pragma once

#include "cuda_runtime.h"
#include <type_traits>

namespace CHJ_MATH_CUDA {

__device__ const double PI = 3.14159265358979323846;


// c = a x b
template<typename TYPE>
__host__ __device__
inline void cross(TYPE*c, TYPE *a, TYPE *b) {
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

template<typename TYPE>
__host__ __device__
inline TYPE normalize_v3(TYPE *arr) {
	TYPE lens = arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2];
	if (lens <= 0) {
		return 0;
	}
	lens = sqrt(lens);
	arr[0] /= lens;
	arr[1] /= lens;
	arr[2] /= lens;
	return lens;
}

template<typename TYPE>
__host__ __device__
inline void add(TYPE *c, TYPE *a, TYPE *b) {
	c[0] = a[0] + b[0];
	c[1] = a[1] + b[1];
	c[2] = a[2] + b[2];
}

template<typename TYPE>
__host__ __device__
inline void add_self(TYPE *c, TYPE *a) {
	c[0] += a[0];
	c[1] += a[1];
	c[2] += a[2];
}

template<typename TYPE>
__host__ __device__
inline void add_(TYPE *c, TYPE *a) {
	c[0] += a[0];
	c[1] += a[1];
	c[2] += a[2];
}

template<typename TYPE>
__host__ __device__
inline TYPE muladd(TYPE *a, TYPE*b) {
	return a[0] * b[0] + a[1] * b[1] * a[2] * b[2];
}

template<typename TYPE>
__host__ __device__
inline void mulsaclar_(TYPE *c, TYPE b) {
	c[0] *= b;
	c[1] *= b;
	c[2] *= b;
}

template<typename TYPE>
__host__ __device__
inline void mulsaclar(TYPE *c, TYPE *a, TYPE b) {
	c[0] = a[0] * b;
	c[1] = a[1] * b;
	c[2] = a[2] * b;
}

template<typename TYPE>
__host__ __device__
inline void sub(TYPE *c, TYPE *a, TYPE *b) {
	c[0] = a[0] - b[0];
	c[1] = a[1] - b[1];
	c[2] = a[2] - b[2];
}

template<typename TYPE>
__host__ __device__
inline void sub_v2(TYPE *c, TYPE *a, TYPE *b) {
	c[0] = a[0] - b[0];
	c[1] = a[1] - b[1];
}

template<typename TYPE>
__host__ __device__
inline TYPE dot_v2(TYPE *a, TYPE *b) {
	return a[0] * b[0] + a[1] * b[1];
}

template<typename TYPE>
__host__ __device__
inline TYPE norm_v3(TYPE *arr) {
	return  sqrt(arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2]);
}

template<typename TYPE>
__host__ __device__
inline TYPE norm_v3_square(TYPE *arr) {
	return  arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2];

}


}
