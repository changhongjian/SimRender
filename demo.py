#!/usr/bin/env python3
# coding: utf-8

import time
import cv2
import torch
from render.lib import Render
from render.geometry import get_obj_v_t_f


class Timer:
    def __init__(self):
        self.end = time.time()

    def elapse(self):
        elpase_ms = (time.time() - self.end) * 1000
        print('Elapse: {:.1f} ms'.format(elpase_ms))
        self.end = time.time()


_numpy_to_tensor = lambda x: torch.from_numpy(x)
_numpy_to_cuda_tensor = lambda x: torch.from_numpy(x).cuda()


def demo():
    torch.cuda.synchronize()
    V, T, F = get_obj_v_t_f('resource/face1.obj')
    # T.fill(0.8)
    nV, nT, nF = V.shape[0], T.shape[0], F.shape[0]
    bsize, h, w = 1, 224, 224
    V = _numpy_to_cuda_tensor(V).unsqueeze(0)
    T = _numpy_to_cuda_tensor(T).unsqueeze(0)

    lib_path = 'lib/lib/librender.so'
    render = Render(lib_path=lib_path)
    render.init(bsize, nV, nF, h, w, F)

    for i in range(10):
        end = time.time()
        imgs = render.render(V, T, magic=True)
        imgs = imgs.cpu().numpy()
        elapse = time.time() - end
        print('Elapse: {:.1f}ms'.format(elapse * 1000))

    cv2.imwrite('res/face1.png', imgs[0])


def main():
    demo()


if __name__ == '__main__':
    main()
