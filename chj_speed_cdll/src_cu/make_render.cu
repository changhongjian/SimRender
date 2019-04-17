#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>
#include "cuda_tools.h"

extern "C" {

#include <sm_60_atomic_functions.h>
}

#include "common_math_gpu.h"

#include "Interface.h"

namespace MESH_RENDER_GPU {
	typedef unsigned char c_uchar;
	typedef unsigned int c_uint32;
	typedef float F_c;
	typedef F_c* pF_c;
	using namespace CHJ_MATH_CUDA;

	//__global__ void D3F_batch_render_info_kernel(int bsize, int nV, int nF, int imgw, int imgh, pF_c pVs, int* pF, pF_c pdms, int * ptri_ids) {
	__global__ void D3F_batch_render_info_kernel(int bsize, int* dims, pF_c pVs, pF_c pTs, int* pF, pF_c pdms, c_uchar * pimgs) {
		int rid = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

		if (rid >= bsize) return;

		int nV = dims[0];
		int nF = dims[1];
		int imgh = dims[2];
		int imgw = dims[3];

		pF_c pV =  pVs + 3*nV*rid;
		pF_c pT = pTs + 3 * nV*rid;

		pF_c pdm = pdms + imgw*imgh*rid;
		auto pimg = pimgs + imgw*imgh*rid * 3;

		

		for (int i = 0; i < nF; i++) {
			int b3 = i * 3;

			int * vids = pF + b3;

			F_c pa[3], pb[3], pc[3];
			int bid0 = 3 * vids[0];
			int bid1 = 3 * vids[1];
			int bid2 = 3 * vids[2];
			pa[0] = pV[bid1 + 0] - pV[bid0 + 0];
			pa[1] = pV[bid1 + 1] - pV[bid0 + 1];
			pa[2] = pV[bid1 + 2] - pV[bid0 + 2];
			pb[0] = pV[bid2 + 0] - pV[bid0 + 0];
			pb[1] = pV[bid2 + 1] - pV[bid0 + 1];
			pb[2] = pV[bid2 + 2] - pV[bid0 + 2];
			cross(pc, pa, pb);

			// !!! 这里是z<0 不可见
			if (pc[2] < 0) continue;

			int b_v[3] = { pF[b3 + 0] * 3, pF[b3 + 1] * 3, pF[b3 + 2] * 3 };

			F_c rectxy[4] = {
				min(min(pV[b_v[0]], pV[b_v[1]]),pV[b_v[2]]),
				min(min(pV[b_v[0] + 1], pV[b_v[1] + 1]),pV[b_v[2] + 1]),
				max(max(pV[b_v[0]], pV[b_v[1]]),pV[b_v[2]]),
				max(max(pV[b_v[0] + 1], pV[b_v[1] + 1]),pV[b_v[2] + 1]),
			};

			//int bbox[4] = { ceill(rectxy[0]),ceill(rectxy[1]), floorl(rectxy[2]),floorl(rectxy[3]) };
			int bbox[4] = { floor(rectxy[0]),floor(rectxy[1]), ceil(rectxy[2]),ceil(rectxy[3]) }; // larger
																						
			if (bbox[0] < 0)bbox[0] = 0;
			if (bbox[1] < 0)bbox[1] = 0;
			if (bbox[2] > imgw)bbox[2] = imgw - 1;
			if (bbox[3] > imgh)bbox[3] = imgh - 1;

			if (bbox[2] < 0) continue;
			if (bbox[3] < 0) continue;

			F_c tex[3];
			F_c pmat[9];
			F_c pb_c[3];
			F_c coord[2];
			F_c v0[2], v1[2], v2[2];
			pF_c _a, _b, _c;
			_a = pV + b_v[0];
			_b = pV + b_v[1];
			_c = pV + b_v[2];


			// 之所以反过来，是考虑图片的索引 【不过感觉也没几个像素】
			for (int by = bbox[1]; by <= bbox[3]; by++) {
				for (int bx = bbox[0]; bx <= bbox[2]; bx++) {

					coord[0] = bx;
					coord[1] = by;

					sub_v2(v0, _b, _a);
					sub_v2(v1, _c, _a);
					sub_v2(v2, coord, _a);

					F_c d00 = dot_v2(v0, v0);
					F_c d01 = dot_v2(v0, v1);
					F_c d11 = dot_v2(v1, v1);
					F_c d20 = dot_v2(v2, v0);
					F_c d21 = dot_v2(v2, v1);

					F_c denom = d00 * d11 - d01 * d01 + 1e-8;
					F_c inverDeno = denom;
					if (inverDeno != 0) inverDeno = 1 / inverDeno;

					pb_c[1] = (d11 * d20 - d01 * d21) * inverDeno;  // v
					pb_c[2] = (d00 * d21 - d01 * d20) * inverDeno;  // w
					pb_c[0] = 1.0 - pb_c[1] - pb_c[2];  // u

					int k = 0;
					for (; k < 3; k++) {
						if (pb_c[k] > 1 + 1e-4 || pb_c[k] < 0 - 1e-4) break; //@7-5
					}

					if (k == 3) {
						F_c deep = pb_c[0] * pV[b_v[0] + 2] + pb_c[1] * pV[b_v[1] + 2] + pb_c[2] * pV[b_v[2] + 2];

						int xy_id = (imgh - 1 - by)*imgw + bx; 
								   
						if (deep > pdm[xy_id]) { 
							pdm[xy_id] = deep;

							// calculate texture
							pmat[0] = pT[b_v[0]];
							pmat[1] = pT[b_v[0] + 1];
							pmat[2] = pT[b_v[0] + 2];
							pmat[3] = pT[b_v[1]];
							pmat[4] = pT[b_v[1] + 1];
							pmat[5] = pT[b_v[1] + 2];
							pmat[6] = pT[b_v[2]];
							pmat[7] = pT[b_v[2] + 1];
							pmat[8] = pT[b_v[2] + 2];

							// Linear interpolation
							tex[0] = pb_c[0] * pmat[0] + pb_c[1] * pmat[3] + pb_c[2] * pmat[6];
							tex[1] = pb_c[0] * pmat[1] + pb_c[1] * pmat[4] + pb_c[2] * pmat[7];
							tex[2] = pb_c[0] * pmat[2] + pb_c[1] * pmat[5] + pb_c[2] * pmat[8];

							for (int _i = 0; _i < 3; _i++) {
								if (tex[_i] < 0) tex[_i] = 0;
								if (tex[_i] > 1) tex[_i] = 1;
								tex[_i] *= 255;
							}

							pimg[xy_id * 3] = tex[2];
							pimg[xy_id * 3 + 1] = tex[1];
							pimg[xy_id * 3 + 2] = tex[0];

						}

					}

				} // bx
			} // by

		} // for nF


	} // kernel


	__global__ void D3F_batch_nF_render_info_kernel(int bsize, int* dims, pF_c pVs, pF_c pTs, int* pF, pF_c pdms, c_uchar * pimgs) {
		int _id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

		int nF = dims[1];
		int rid = _id / nF;
		if (rid >= bsize) return;

		int Fid = _id % nF;

		int nV = dims[0];
		int imgh = dims[2];
		int imgw = dims[3];

		pF_c pV = pVs + 3 * nV*rid;
		pF_c pT = pTs + 3 * nV*rid;

		pF_c pdm = pdms + imgw*imgh*rid;
		auto pimg = pimgs + imgw*imgh*rid*3;

		int i = Fid;

		int b3 = i * 3;

		int * vids = pF + b3;

		F_c pa[3], pb[3], pc[3];
		int bid0 = 3 * vids[0];
		int bid1 = 3 * vids[1];
		int bid2 = 3 * vids[2];
		pa[0] = pV[bid1 + 0] - pV[bid0 + 0];
		pa[1] = pV[bid1 + 1] - pV[bid0 + 1];
		pa[2] = pV[bid1 + 2] - pV[bid0 + 2];
		pb[0] = pV[bid2 + 0] - pV[bid0 + 0];
		pb[1] = pV[bid2 + 1] - pV[bid0 + 1];
		pb[2] = pV[bid2 + 2] - pV[bid0 + 2];
		cross(pc, pa, pb);

		// !!! nz<0 are invisiable
		if (pc[2] <= 0) return;

		int b_v[3] = { pF[b3 + 0] * 3, pF[b3 + 1] * 3, pF[b3 + 2] * 3 };

		F_c rectxy[4] = {
			min(min(pV[b_v[0]], pV[b_v[1]]),pV[b_v[2]]),
			min(min(pV[b_v[0] + 1], pV[b_v[1] + 1]),pV[b_v[2] + 1]),
			max(max(pV[b_v[0]], pV[b_v[1]]),pV[b_v[2]]),
			max(max(pV[b_v[0] + 1], pV[b_v[1] + 1]),pV[b_v[2] + 1]),
		};

		//int bbox[4] = { ceill(rectxy[0]),ceill(rectxy[1]), floorl(rectxy[2]),floorl(rectxy[3]) };
		int bbox[4] = { floor(rectxy[0]),floor(rectxy[1]), ceil(rectxy[2]),ceil(rectxy[3]) }; // larger
		//int bbox[4] = { floor(rectxy[0])-1,floor(rectxy[1])-1, ceil(rectxy[2])+1,ceil(rectxy[3])+1 }; // larger
																								
		if (bbox[0] < 0) bbox[0] = 0;
		if (bbox[1] < 0) bbox[1] = 0;
		if (bbox[2] >= imgw) bbox[2] = imgw - 1;
		if (bbox[3] >= imgh) bbox[3] = imgh - 1;

		if (bbox[0] >= imgw) return;
		if (bbox[1] >= imgh) return;
		if (bbox[2] < 0) return;
		if (bbox[3] < 0) return;


		F_c pb_c[3];
		F_c coord[2];
		F_c v0[2], v1[2], v2[2];
		pF_c _a, _b, _c;
		_a = pV + b_v[0];
		_b = pV + b_v[1];
		_c = pV + b_v[2];

		F_c tex[3];
		F_c pmat[9];

		for (int by = bbox[1]; by <= bbox[3]; by++) {
			for (int bx = bbox[0]; bx <= bbox[2]; bx++) {

				coord[0] = bx;
				coord[1] = by;

				sub_v2(v0, _b, _a);
				sub_v2(v1, _c, _a);
				sub_v2(v2, coord, _a);

				F_c d00 = dot_v2(v0, v0);
				F_c d01 = dot_v2(v0, v1);
				F_c d11 = dot_v2(v1, v1);
				F_c d20 = dot_v2(v2, v0);
				F_c d21 = dot_v2(v2, v1);

				//F_c denom = d00 * d11 - d01 * d01 + 1e-8;
				F_c denom = d00 * d11 - d01 * d01;
				F_c inverDeno= denom;
				if (inverDeno != 0) inverDeno = 1 / inverDeno;

				pb_c[1] = (d11 * d20 - d01 * d21) * inverDeno;  // v
				pb_c[2] = (d00 * d21 - d01 * d20) * inverDeno;  // w
				pb_c[0] = 1.0 - pb_c[1] - pb_c[2];  // u

				int k = 0;
				for (; k < 3; k++) {
					if (pb_c[k] > 1 + 1e-4 || pb_c[k] < 0 - 1e-4) break; //@18-7-5
				}

				if (k == 3) {
					// compute deep first 
					F_c deep = pb_c[0] * pV[b_v[0] + 2] + pb_c[1] * pV[b_v[1] + 2] + pb_c[2] * pV[b_v[2] + 2];

					// calculate texture
					pmat[0] = pT[b_v[0]];
					pmat[1] = pT[b_v[0] + 1];
					pmat[2] = pT[b_v[0] + 2];
					pmat[3] = pT[b_v[1]];
					pmat[4] = pT[b_v[1] + 1];
					pmat[5] = pT[b_v[1] + 2];
					pmat[6] = pT[b_v[2]];
					pmat[7] = pT[b_v[2] + 1];
					pmat[8] = pT[b_v[2] + 2];

					// Linear interpolation
					tex[0] = pb_c[0] * pmat[0] + pb_c[1] * pmat[3] + pb_c[2] * pmat[6];
					tex[1] = pb_c[0] * pmat[1] + pb_c[1] * pmat[4] + pb_c[2] * pmat[7];
					tex[2] = pb_c[0] * pmat[2] + pb_c[1] * pmat[5] + pb_c[2] * pmat[8];

					for (int _i = 0; _i < 3; _i++) {
						if (tex[_i] < 0) tex[_i] = 0;
						if (tex[_i] > 1) tex[_i] = 1;
						tex[_i] *= 255;
					}

					// remember image are in opencv 3D obj are in opengl, so y should change
					int xy_id = (imgh - 1 - by)*imgw + bx;

					// !!! CHJ_WARN: I want to solve a problem A to some extent.
					// A: If two faces are overlaps but all point to your, parallel rendering may cause problems.

					int __cnt = 0;
					while (true) {
						if (deep > pdm[xy_id]) {
							pdm[xy_id] = deep;
							__cnt++;

							// RGB -> BGR
							pimg[xy_id * 3] = tex[2];
							pimg[xy_id * 3 + 1] = tex[1];
							pimg[xy_id * 3 + 2] = tex[0];

						}
						else break;
						if (__cnt > 3) break;
					}
	
				}

			} // bx
		} // by


	} // kernel


	EXT_C_DLL_VOID D3F_batch_render_info_gpu() {

		int * dims = (int*)c_param.mp["dims"];
		int bsize = dims[0];
		//int nF = dims[1];

		int * gpu_dims = (int*)c_param.mp["gpu_dims"];
		pF_c pV = (F_c *)c_param.mp["V"]; // ogl 
		pF_c pT = (F_c *)c_param.mp["T"]; // RGB
		int* pF = (int *)c_param.mp["F"]; // anticlockwise

		pF_c pdm = (F_c *)c_param.mp["deepmask"];
		//pF_c pimg = (F_c *)c_param.mp["img"];
		c_uchar * img = (c_uchar *)c_param.mp["img_BGR_uint8"];  // ocv


		D3F_batch_render_info_kernel << <1, bsize >> >(bsize, gpu_dims, pV, pT, pF, pdm, img);

		check_error(cudaPeekAtLastError());

	}

	EXT_C_DLL_VOID D3F_batch_nF_render_info_gpu() {

		int * dims = (int*)c_param.mp["dims"];
		int bsize = dims[0];
		int nF = dims[1];
		int * gpu_dims = (int*)c_param.mp["gpu_dims"];

		pF_c pV = (F_c *)c_param.mp["V"]; // ogl 
		pF_c pT = (F_c *)c_param.mp["T"]; // RGB
		int* pF = (int *)c_param.mp["F"]; // anticlockwise

		pF_c pdm = (F_c *)c_param.mp["deepmask"];
		//pF_c pimg = (F_c *)c_param.mp["img"];
		c_uchar * img = (c_uchar *)c_param.mp["img_BGR_uint8"];  // ocv

		D3F_batch_nF_render_info_kernel << <cuda_gridsize(bsize*nF), BLOCK >> >(bsize, gpu_dims, pV, pT, pF, pdm, img);

		//Check the error messages:
		check_error(cudaPeekAtLastError());
		
	}


}
