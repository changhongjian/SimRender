# -*- coding:utf-8 -* 

import numpy as np

import ctypes
from ctypes import *

'''
@18-4-20 first summary
@18-5-17 add some functions for torch gpu
'''


def torch_c_ptr(thmat):
    state_ptr = thmat.data_ptr()
    return ctypes.c_void_p(state_ptr)

class CHJ_speed:
    def __init__(self):
        self.cdll = None
        self.check_type=0
        pass
    def load_cdll(self, fcdll):
        print("use dll", fcdll)
        self.cdll = ctypes.cdll.LoadLibrary(fcdll)
        
    def set_mp(self, nm, npmat):
        if self.check_type==1:
            if npmat.dtype==np.float32: print(nm,"type is float32")
            if npmat.dtype==np.int64: print(nm,"type is int64")

        pm = npmat.ctypes.data_as(c_void_p)
        # bt=(c_char * 100)().value
        ss = bytes(nm, encoding="ascii")
        self.cdll.set_mp(ss, pm)
    
    # CHJ_WARN: Avoid to use this function.
    # ctype type is ok,# numpy  may has problem
    def get_mp(self, nm, shape, ctype):
        #if(ctype==np.int32): ctype= types.c_int32
        ss = bytes(nm, encoding="ascii")
        #self.cdll.get_mp.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int32, shape=(3,4))
        self.cdll.get_mp.restype = np.ctypeslib.ndpointer(dtype=ctype, shape=shape)
        npmat=self.cdll.get_mp(ss)
        return npmat

    def set_mp_ext(self, nm, ctype_ptr):
        ss = bytes(nm, encoding="ascii")
        self.cdll.set_mp(ss, ctype_ptr)

    # cpu and gpu both ok.
    def set_mp_torch(self, nm, thmat):
        self.set_mp_ext(nm, torch_c_ptr(thmat))

