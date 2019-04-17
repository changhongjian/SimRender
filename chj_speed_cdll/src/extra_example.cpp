/*
I don't provide my python code to invoke the following code. This is my very old C codes. You can ignore it.
But, from the following code, you may know how to use this framework to do more jobs.

*/

/*
#include "Interface.h"
#include "common_math.h"
#include <omp.h>

namespace MESH_RENDER {

	using namespace std;
	using namespace CHJ_MATH;

	// 这里决定使用哪种类型
	typedef float F_c;
	typedef F_c* pF_c;
	typedef cv::Mat1f cv_F; // use opencv
	typedef unsigned char c_uchar;

	EXT_C_DLL_VOID D3F_syn_render_run() {
		int * dims = (int*)c_param.mp["dims"];

		int nV = dims[0];
		int nF = dims[1];
		int imgh = dims[2];
		int imgw = dims[2];

		pF_c pV = (F_c *)c_param.mp["V"];
		pF_c pT = (F_c *)c_param.mp["T"];
		int* pF = (int *)c_param.mp["F"];
		//pF_c img = (F_c *)c_param.mp["img"];
		// @6-11 修改
		c_uchar * img = (c_uchar *)c_param.mp["img_BGR_uint8"];
		pF_c pdm = (F_c *)c_param.mp["deepmask"];


		// 一个循环搞定所有的事情
		// 遍历每个三角形开始制作
		cv_F _mat_temp = cv_F::ones(3, 3);
		cv_F coordinate = cv_F::ones(3, 1);

		// 先算三角形的norm

		omp_lock_t writelock;
		omp_init_lock(&writelock);

		//int64 start = 0, end = 0;
		//start = cv::getTickCount();

		// opemp 并行
		//#pragma omp parallel for
		for (int i = 0; i < nF; i++) {
			cv_F mat = _mat_temp.clone();

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


			// 获得bbox
			int b_v[3] = { pF[b3 + 0] * 3, pF[b3 + 1] * 3, pF[b3 + 2] * 3 };
			pF_c pmat = mat.ptr<F_c>(0);
			pmat[0] = pV[b_v[0]];
			pmat[1] = pV[b_v[0] + 1];
			pmat[3] = pV[b_v[1]];
			pmat[4] = pV[b_v[1] + 1];
			pmat[6] = pV[b_v[2]];
			pmat[7] = pV[b_v[2] + 1];

			F_c rectxy[4] = {
				min(min(pV[b_v[0]], pV[b_v[1]]),pV[b_v[2]]),
				min(min(pV[b_v[0] + 1], pV[b_v[1] + 1]),pV[b_v[2] + 1]),
				max(max(pV[b_v[0]], pV[b_v[1]]),pV[b_v[2]]),
				max(max(pV[b_v[0] + 1], pV[b_v[1] + 1]),pV[b_v[2] + 1]),
			};

			//int bbox[4] = { ceill(rectxy[0]),ceill(rectxy[1]), floorl(rectxy[2]),floorl(rectxy[3]) };
			int bbox[4] = { floor(rectxy[0]),floor(rectxy[1]), ceil(rectxy[2]),ceil(rectxy[3]) }; // larger

																								  // [out]
			cv_F mat_for_barycentric = mat.t().inv();

			// WARNING: 修正bbox
			if (bbox[0] < 0)bbox[0] = 0;
			if (bbox[1] < 0)bbox[1] = 0;
			if (bbox[2] > imgw)bbox[2] = imgw - 1;
			if (bbox[3] > imgh)bbox[3] = imgh - 1;

			if (bbox[2] < 0) continue;
			if (bbox[3] < 0) continue;


			F_c tex[3];
			// 之所以反过来，是考虑图片的索引 【不过也没几个像素】
			for (int by = bbox[1]; by <= bbox[3]; by++) {
				for (int bx = bbox[0]; bx <= bbox[2]; bx++) {

					cv_F coord = coordinate.clone();
					coord(0, 0) = bx;
					coord(1, 0) = by;
					// 接着获得xys, 和 b_c

					cv_F b_c = mat_for_barycentric * coord;

					pF_c pb_c = b_c.ptr<F_c>(0);
					int k = 0;
					for (; k < 3; k++) {
						if (pb_c[k] > 1 + 1e-4 || pb_c[k] < 0 - 1e-4) break; //@7-5
					}

					if (k == 3) {
						// 先计算深度，看是否要做
						F_c deep = pb_c[0] * pV[b_v[0] + 2] + pb_c[1] * pV[b_v[1] + 2] + pb_c[2] * pV[b_v[2] + 2];

						// 获得纹理的内容
						pmat[0] = pT[b_v[0]];
						pmat[1] = pT[b_v[0] + 1];
						pmat[2] = pT[b_v[0] + 2];
						pmat[3] = pT[b_v[1]];
						pmat[4] = pT[b_v[1] + 1];
						pmat[5] = pT[b_v[1] + 2];
						pmat[6] = pT[b_v[2]];
						pmat[7] = pT[b_v[2] + 1];
						pmat[8] = pT[b_v[2] + 2];

						// 采用线性插值
						tex[0] = pb_c[0] * pmat[0] + pb_c[1] * pmat[3] + pb_c[2] * pmat[6];
						tex[1] = pb_c[0] * pmat[1] + pb_c[1] * pmat[4] + pb_c[2] * pmat[7];
						tex[2] = pb_c[0] * pmat[2] + pb_c[1] * pmat[5] + pb_c[2] * pmat[8];

						int tex_tmp[3];
						for (int _i = 0; _i < 3; _i++) {
							if (tex[_i] < 0)tex_tmp[_i] = 0;
							else if (tex[_i] > 1)tex_tmp[_i] = 255;
							else tex_tmp[_i] = (tex[_i] * 255 + 0.5);

						}

					
						int xy_id = (imgh - 1 - by)*imgw + bx; 
						omp_set_lock(&writelock);
						if (deep > pdm[xy_id]) { 
							// 注意这里把RGB换成了BGR			

							img[xy_id * 3] = tex_tmp[0 + 2];
							img[xy_id * 3 + 1] = tex_tmp[0 + 1];
							img[xy_id * 3 + 2] = tex_tmp[0 + 0];

							pdm[xy_id] = deep; // @18-7-5

						}
						omp_unset_lock(&writelock);
					}

				}
			}

			//

		}

		//end = cv::getTickCount();
		//cout << "The differences: " << (end - start) / cv::getTickFrequency() << " s" << endl;

		omp_destroy_lock(&writelock);
	}

} // namespace 

*/