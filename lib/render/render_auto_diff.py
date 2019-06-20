
from lib.comm.geometry import *
from lib.speed.cdll import CHJ_speed
import torch

def create_index_table(h, w):
    x = np.arange(0, w).astype(np.int32)
    x_mat = np.tile(x, (h, 1))
    y = np.linspace(h - 1, 0, h).astype(np.int32)
    y_mat = np.tile(y[:, np.newaxis], (1, w))
    xy_mat = np.array([x_mat, y_mat])

    return xy_mat

# [calc barycentray]
def calc_barycentric(p, a, b, c):
    # pytorch batch
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    denom = d00 * d11 - d01 * d01 + 1e-8
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

#print(create_index_table(3, 2))

class cls_render_for_auto_diff:
    def __init__(self, fcdll):
        speed = CHJ_speed()
        speed.load_cdll(fcdll)
        speed.check_type = 0
        self.speed = speed

    def init_render(self, F, nV, nF, h, w, batch_size=1, nptype=np.float32):
        self.model_F = F.reshape(-1, 3)
        dims = np.array([
            nV, nF, h, w
        ], np.int32)
        self.gpu_dims = torch.from_numpy(dims).cuda()
        
        deepmasks = np.ones((batch_size, h, w), nptype) 
        self.gpu_deepmasks = torch.from_numpy(deepmasks).cuda()
        self.gpu_F = torch.from_numpy(self.model_F).cuda()
        
        xy_tri_ids = np.zeros((batch_size, h, w), np.int32)
        self.gpu_xy_tri_ids = torch.from_numpy(xy_tri_ids).cuda()

        imgs_mask = np.zeros((batch_size, h, w), np.uint8)
        self.gpu_imgs_mask = torch.from_numpy(imgs_mask).cuda()
        
        xy_index_table = create_index_table(h, w) # (2, h, w)
        xy_index_table = xy_index_table.transpose(1, 2, 0).copy()  # (h, w, 2)
        bxy_index_table = np.broadcast_to(xy_index_table, (batch_size, h, w, 2))
        self.gpu_bxy_index_table = torch.from_numpy(bxy_index_table).cuda().float()
        
        each_n = h * w * 3
        bs_4rep = np.arange(batch_size)
        self.rep_img3_ids = bs_4rep.repeat(each_n).reshape(-1)
        
        self.imgw=w
        self.imgh=h
        

    def _get_render_info_th_gpu(self, gpu_Vc3d):
        speed = self.speed
        n_item = gpu_Vc3d.size(0)

        dims = np.array([n_item, self.model_F.shape[0]], np.int32)
        speed.set_mp("dims", dims)
        speed.set_mp_torch("gpu_dims", self.gpu_dims)
        speed.set_mp_torch("F", self.gpu_F)
        speed.set_mp_torch("V", gpu_Vc3d)

        self.gpu_deepmasks.fill_(-1e30)
        self.gpu_xy_tri_ids.fill_(-1)

        speed.set_mp_torch("deepmask", self.gpu_deepmasks)
        speed.set_mp_torch("xy_tri_id", self.gpu_xy_tri_ids)

        speed.cdll.D3F_batch_nF_render_info_gpu_v2()

    def run(self, Vs, Ts):
        if type(Vs)==numpy.ndarray:
            Vs=torch.from_numpy(Vs)
            Ts=torch.from_numpy(Ts)
      
        n = len(Vs.size())
        if n==2:
            Vs=Vs.unsqueeze(0)
            Ts=Ts.unsqueeze(0)
        
        assert type(Ts)==torch.Tensor, "must be torch"
        assert len(Ts.size())==3, "must has bacth"
        
        Vs=Vs.cuda()
        Ts=Ts.cuda()
        
        return self._get_gen_texture(Vs, Ts)
    
    def run_get_np(self, Vs, Ts):
        imgs=self.run(Vs, Ts)
        return imgs.cpu().numpy()
    
    def run_get_ocv_imgs(self, Vs, Ts):
        imgs=self.run(Vs, Ts)
        imgs *= 255
        imgs=imgs.cpu().numpy()
        imgs = imgs.astype(np.uint8)
        imgs = imgs[:, :, :, ::-1]
        return imgs
        
    
    def _get_gen_texture(self, Vs, Rs):
        batch_size = Vs.size(0)
        self._get_render_info_th_gpu(Vs)
        illegal_ids = self.gpu_xy_tri_ids < 0 # !!! 
        self.gpu_imgs_mask.fill_(1)
        self.gpu_imgs_mask[illegal_ids] = 0
        self.gpu_xy_tri_ids[illegal_ids] = 0
        
        # each pixel in a trangle
        Vids = self.model_F[self.gpu_xy_tri_ids.view(-1).cpu().numpy(), :]
       
        Vs = Vs.view(batch_size, -1, 3)
        Rs = Rs.view(batch_size, -1, 3)

        ids = self.rep_img3_ids 
        Vids = Vids.reshape(-1)
        # p(Vs.size(), ids.shape, Vids.shape)

        # each pixel in a trangle which has three vertices and each vertice has x, y, z
        sel_Vs = Vs[ids, Vids].view(-1, 3, 3)
        sel_Rs = Rs[ids, Vids].view(-1, 3, 3)

        ids = self.gpu_imgs_mask.view(-1) == 0

        # Remove associations (no need to be auto differentiation)
        sel_Vs[ids] = 0
        sel_Rs[ids] = 0

        sel_Vs2d = sel_Vs[:, :, :2]
        u, v, w = calc_barycentric(self.gpu_bxy_index_table.view(-1, 2), sel_Vs2d[:, 0], sel_Vs2d[:, 1], sel_Vs2d[:, 2])

        # n_mix, 3 (n_mix is all the pixels of batch)
        th_bary = torch.stack((u, v, w), 1)

        #  n*1*3 x n*3*1 (n=b*imgw*imgh)  
        gen_color_BGR = torch.bmm(th_bary.unsqueeze(1), sel_Rs).view(batch_size, self.imgh, self.imgw , 3)
        # ocv
        return gen_color_BGR

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

