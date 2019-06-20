# -*- coding:utf-8 -* 
import os ; import sys 
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 


from lib.comm.geometry import *
from lib.render.render_auto_diff import cls_render_for_auto_diff

import torch
import platform

'''
3D obj must in opengl coordinate system, its texture should be RGB(0~1) and its face vertices need be counterclockwise arrangement.
You can use meshlab to check.

render images which can be used for autodiff

'''

def run():
    f1()

def f1():
    # anywhere. In linux, it is so.
    fcdll="chj_speed_cdll/lib/chj_speed_deep3dface.dll"
    if platform.system() == "Linux":
        fcdll="chj_speed_cdll/lib/chj_speed_deep3dface.so"

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
    
    Vs = torch.from_numpy(Vs).cuda()
    Ts = torch.from_numpy(Ts).cuda()
    
    _cls_render = cls_render_for_auto_diff(fcdll)
    _cls_render.init_render(F, nV, nF, imgh, imgw, batch_size=bsize)
    
    # this is used for traning networks
    # if Vs or Ts are require_grad, then the imgs are require_grad
    # imgs = _cls_render.run(Vs, Ts)
    
    # just for examples
    imgs = _cls_render.run_get_ocv_imgs(Vs, Ts)
    
    for img in imgs:
        key=showimg(img)
        if key==27: break
        
        
if __name__=="__main__":
    run()
    