# -*- coding:utf-8 -* 
import os ; import sys 
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] ) 

from chj.speed.cdll import CHJ_speed
from chj.comm.geometry import *

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

class _cls_batch_render:
    def __init__(self, fcdll):
        speed = CHJ_speed()
        speed.load_cdll(fcdll)
        self.speed = speed
        
    def init(self, bsize, nV, nF, imgh, imgw, npF):
        self.dims = np.array([bsize, nF], np.int32)
        dims = np.array([
            nV, nF, imgh, imgw
        ], np.int32)
        self.gpu_dims = torch.from_numpy(dims).cuda()
        self.gpu_F = torch.from_numpy(npF).cuda()
        
        nptype = np.float32
        deepmasks = np.ones((bsize, imgh, imgw), nptype)
        self.gpu_deepmasks = torch.from_numpy(deepmasks).cuda()
        
        imgs = np.ones((bsize, imgh, imgw, 3), np.uint8)
        self.imgs = torch.from_numpy(imgs).cuda()
    
    # must be in gpu
    def render(self, Vs, Ts):
        #assert Vs.is_cuda
        
        '''
        Tips:
        Remember we only convert c or gpu pointers, make sure the memories are reserved (not release by python's GC). So I use self.xxx to save them.
        
        Some of the follow variables may only need to call set_mp once, like 
        (self.dims, gpu_dims, F, deepmask, imgs). 
        '''
        speed = self.speed
        speed.set_mp("dims", self.dims)
        speed.set_mp_torch("gpu_dims", self.gpu_dims)
        speed.set_mp_torch("F", self.gpu_F)
        
        self.gpu_deepmasks.fill_(-1e30)
        self.imgs.fill_(0)
        speed.set_mp_torch("deepmask", self.gpu_deepmasks)
        speed.set_mp_torch("img_BGR_uint8", self.imgs)

        speed.set_mp_torch("V", Vs)
        speed.set_mp_torch("T", Ts)
        
        
        # slow, may slower than CPU 
        #speed.cdll.D3F_batch_render_info_gpu()
        # very fast but some pixels may not be rendered
        speed.cdll.D3F_batch_nF_render_info_gpu() 
        
        return self.imgs.clone()

    # ^_^ Finally, I realized what I said above.
    def pre_set(self):
        speed = self.speed
        speed.set_mp("dims", self.dims)
        speed.set_mp_torch("gpu_dims", self.gpu_dims)
        speed.set_mp_torch("F", self.gpu_F)
        speed.set_mp_torch("deepmask", self.gpu_deepmasks)
        speed.set_mp_torch("img_BGR_uint8", self.imgs)
    
    def renderv2(self, Vs, Ts):
        speed = self.speed
        self.gpu_deepmasks.fill_(-1e30)
        self.imgs.fill_(0)
        speed.set_mp_torch("V", Vs)
        speed.set_mp_torch("T", Ts)

        #speed.cdll.D3F_batch_render_info_gpu()
        speed.cdll.D3F_batch_nF_render_info_gpu()
        return self.imgs.clone()
        
run()