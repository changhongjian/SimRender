# -*- coding:utf-8 -* 

import numpy as np
import torch

import ctypes
from ctypes import *

'''
@18-4-20 first summary
@18-5-17 add some functions for torch gpu
'''


def torch_c_ptr(thmat):
    state_ptr = thmat.data_ptr()
    return ctypes.c_void_p(state_ptr)


class _Render(object):
    def __init__(self):
        self.cdll = None
        self.check_type = 0

    def load_cdll(self, lib_path):
        # print("use dll", lib_path)
        self.cdll = ctypes.cdll.LoadLibrary(lib_path)

    def set_mp(self, nm, npmat):
        if self.check_type == 1:
            if npmat.dtype == np.float32: print(nm, "type is float32")
            if npmat.dtype == np.int64: print(nm, "type is int64")

        pm = npmat.ctypes.data_as(c_void_p)
        # bt=(c_char * 100)().value
        ss = bytes(nm, encoding="ascii")
        self.cdll.set_mp(ss, pm)

    # CHJ_WARN: Avoid to use this function.
    # ctype type is ok,# numpy  may has problem
    def get_mp(self, nm, shape, ctype):
        # if(ctype==np.int32): ctype= types.c_int32
        ss = bytes(nm, encoding="ascii")
        # self.cdll.get_mp.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int32, shape=(3,4))
        self.cdll.get_mp.restype = np.ctypeslib.ndpointer(dtype=ctype, shape=shape)
        npmat = self.cdll.get_mp(ss)
        return npmat

    def set_mp_ext(self, nm, ctype_ptr):
        ss = bytes(nm, encoding="ascii")
        self.cdll.set_mp(ss, ctype_ptr)

    # cpu and gpu both ok.
    def set_mp_torch(self, nm, thmat):
        self.set_mp_ext(nm, torch_c_ptr(thmat))


class Render:
    def __init__(self, lib_path):
        self._render = _Render()
        self._render.load_cdll(lib_path)

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
    def render(self, Vs, Ts, magic=True):
        '''
        Tips:
        Remember we only convert c or gpu pointers, make sure the memories are reserved (not release by python's GC). So I use self.xxx to save them.

        Some of the follow variables may only need to call set_mp once, like
        (self.dims, gpu_dims, F, deepmask, imgs).
        '''

        _render = self._render
        _render.set_mp("dims", self.dims)
        _render.set_mp_torch("gpu_dims", self.gpu_dims)
        _render.set_mp_torch("F", self.gpu_F)

        self.gpu_deepmasks.fill_(-1e30)
        self.imgs.fill_(0)
        _render.set_mp_torch("deepmask", self.gpu_deepmasks)
        _render.set_mp_torch("img_BGR_uint8", self.imgs)

        _render.set_mp_torch("V", Vs)
        _render.set_mp_torch("T", Ts)

        if not magic:
            _render.cdll.D3F_batch_render_info_gpu()
        else:
            _render.cdll.D3F_batch_nF_render_info_gpu()

        return self.imgs.clone()
        # return self.imgs.cpu().numpy()

    # ^_^ Finally, I realized what I say in the Tips.
    def pre_set(self):
        _render = self._render
        _render.set_mp("dims", self.dims)
        _render.set_mp_torch("gpu_dims", self.gpu_dims)
        _render.set_mp_torch("F", self.gpu_F)
        _render.set_mp_torch("deepmask", self.gpu_deepmasks)
        _render.set_mp_torch("img_BGR_uint8", self.imgs)

    def renderv2(self, Vs, Ts):
        _render = self._render
        self.gpu_deepmasks.fill_(-1e30)
        self.imgs.fill_(0)
        _render.set_mp_torch("V", Vs)
        _render.set_mp_torch("T", Ts)

        _render.cdll.D3F_batch_render_info_gpu()
        # speed.cdll.D3F_batch_nF_render_info_gpu()
        return self.imgs.clone()
