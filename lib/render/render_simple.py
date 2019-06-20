
from lib.comm.geometry import *
from lib.speed.cdll import CHJ_speed
import torch

# just for experiments, get the recommend 3D shape to render
def auto_S(S, wh, rate=0.8):
    '''
    transform=[s, 0, tx; 0, s, ty]
    '''

    a = S[:, :2].min(axis=0)
    b = S[:, :2].max(axis=0)
    c = (a+b)/2
    d = (b-a).max()

    # get s
    s = wh * rate / d
    # get tx, ty according to the center point
    c_t = np.array([wh, wh]) / 2
    txy = c_t - s*c
    TS = S * s
    TS[:, :2] += txy #[np.newaxis]
    return TS



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
        
        
