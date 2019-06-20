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

#define DBG_GPU_1

namespace MESH_RENDER_GPU {
	typedef unsigned int c_uint32;
	typedef float F_c;
	typedef F_c* pF_c;
	using namespace CHJ_MATH_CUDA;


	__global__ void D3F_batch_nF_render_info_kernel(int * locks, int bsize, int* dims, pF_c pVs, int* pF, pF_c pdms, int * ptri_ids) {
		int _id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

		int nF = dims[1];
		int rid = _id / nF;
		if (rid >= bsize) return;

		int Fid = _id % nF;

		int nV = dims[0];
		int imgh = dims[2];
		int imgw = dims[3];

		pF_c pV = pVs + 3 * nV*rid;

		pF_c pdm = pdms + imgw*imgh*rid;
		int * ptri_id = ptri_ids + imgw*imgh*rid;

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
        

		// !!! if z<0: Invisible
		
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
					F_c deep = pb_c[0] * pV[b_v[0] + 2] + pb_c[1] * pV[b_v[1] + 2] + pb_c[2] * pV[b_v[2] + 2];

					// !!! to image index
					int xy_id = (imgh - 1 - by)*imgw + bx; // ц╩спЁкрт3

					// !!! in zbuffer, rendering two triangles into the same place concurrent will get wrong results.
					int __cnt = 0;
					while (true) {
						if (deep > pdm[xy_id]) {
							pdm[xy_id] = deep;
							ptri_id[xy_id] = i; // in the face
							__cnt++;
						}
						else break;
						if (__cnt > 3) break;
					}
					
				}

			} // bx
		} // by


	} // kernel

	EXT_C_DLL_VOID D3F_batch_nF_render_info_gpu_v2() {

		int * dims = (int*)c_param.mp["dims"];
		int bsize = dims[0];
		int nF = dims[1];
		int * gpu_dims = (int*)c_param.mp["gpu_dims"];

		pF_c pV = (F_c *)c_param.mp["V"];
		int* pF = (int *)c_param.mp["F"];

		pF_c pdm = (F_c *)c_param.mp["deepmask"];
		int * ptri_id = (int*)c_param.mp["xy_tri_id"];


		int* locks=nullptr;

		D3F_batch_nF_render_info_kernel << <cuda_gridsize(bsize*nF), BLOCK >> >(locks, bsize, gpu_dims, pV, pF, pdm, ptri_id);

		//Check the error messages:

		check_error(cudaPeekAtLastError());

	}



}
