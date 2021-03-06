# -*- coding:utf-8 -* 
import os ; import sys 
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

from lib.comm.geometry import *
from lib.render.render_simple import _cls_batch_render, auto_S

import torch
import platform

'''
3D obj must in opengl coordinate system, its texture should be RGB(0~1) and its face vertices need be counterclockwise arrangement.
You can use meshlab to check.

batch rendering example

'''

def run():
    f1()

def f1():
    # anywhere. In linux, it is so.
    fcdll="chj_speed_cdll/lib/chj_speed_deep3dface.dll"
    if platform.system() == "Linux":
        fcdll="libchj_speed_cdll/lib/chj_speed_deep3dface.so"

    fdir="resource/"

    fobjs=["face1", "face2"]

    fobjs=[fdir+x+".obj" for x in fobjs]

    objs=[get_obj_v_t_f(x) for x in fobjs]

    #rows, cols
    imgh, imgw = 224, 224 
    # image size must be consistent with your 3D objs

    F = objs[0][2]
    nV = objs[0][0].shape[0]
    nF = F.shape[0]
    bsize = len(objs)
    
    Vs = np.stack((objs[0][0], objs[1][0]), 0)
    Ts = np.stack((objs[0][1], objs[1][1]), 0)
    
    #@19-6-20
    # if your `Vs` is not in the image area, use auto_S first (see the code)
    
    Vs = torch.from_numpy(Vs).cuda()
    Ts = torch.from_numpy(Ts).cuda()
    
    cls_batch_render = _cls_batch_render(fcdll)
    cls_batch_render.init(bsize, nV, nF, imgh, imgw, F )
    
    
    if 1==1:
        gpu_imgs = cls_batch_render.render(Vs, Ts)
    else:
        cls_batch_render.pre_set()
        gpu_imgs = cls_batch_render.renderv2(Vs, Ts)
        
        Vs_new = Vs.clone()
        Vs[0] = Vs_new[1]
        Vs[1] = Vs_new[0]
        
        gpu_imgs = cls_batch_render.renderv2(Vs, Ts)
    
    imgs=gpu_imgs.cpu().numpy()
    
    for img in imgs:
        key=showimg(img)
        if key==27: break
        

'''
# test time
for i in range(10):
    t0 = time.clock()
    gpu_imgs = cls_batch_render.render(Vs, Ts)
    imgs=gpu_imgs.cpu().numpy()
    t1 = time.clock()
    print("Total running time1: %.3f ms" % (1000*(t1 - t0)))
    
    t0 = time.time()
    gpu_imgs = cls_batch_render.render(Vs, Ts)
    imgs=gpu_imgs.cpu().numpy()
    t1 = time.time()
    print("Total running time2: %.3f ms" % (1000*(t1 - t0)))

'''


if __name__=="__main__":
    run()