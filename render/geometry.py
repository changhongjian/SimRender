# -*- coding:utf-8 -* 

import numpy as np
import cv2


def showimg(img, nm="pic"):
    cv2.imshow(nm, img)
    return cv2.waitKey()


def readlines(fname):
    with open(fname) as fp:
        list = fp.readlines()
    for id, item in enumerate(list):
        list[id] = item.strip()
    return list


# @2018-12-8
def get_obj_v(fnm):
    lines = readlines(fnm)
    v = []
    for line in lines:
        sz = line.split()

        if len(sz) >= 4 and sz[0] == "v":
            v.append([float(x) for x in sz[1:4]])
        if len(sz) == 4 and sz[0] == "f":
            break
    v = np.array(v).astype(np.float32)

    # print(v.shape)
    return v


def get_obj_v_f(fnm):
    lines = readlines(fnm)
    v = []
    f = []
    for line in lines:
        sz = line.split()

        if len(sz) >= 4 and sz[0] == "v":
            v.append([float(x) for x in sz[1:4]])

        if len(sz) == 4 and sz[0] == "f":
            f.append([int(x) for x in sz[1:4]])

    v = np.array(v).astype(np.float32)
    f = np.array(f).astype(np.int32) - 1  # !!! 1-base to 0-base
    # p(v.shape)
    return v, f


def get_obj_v_t_f(fnm):
    lines = readlines(fnm)
    v = []
    t = []
    f = []
    for line in lines:
        sz = line.split()

        if len(sz) >= 4 and sz[0] == "v":
            v.append([float(x) for x in sz[1:4]])
            t.append([float(x) for x in sz[4:7]])

        if len(sz) == 4 and sz[0] == "f":
            f.append([int(x) for x in sz[1:4]])

    v = np.array(v).astype(np.float32)
    t = np.array(t).astype(np.float32)
    f = np.array(f).astype(np.int32) - 1  # !!! 1-base to 0-base

    return v, t, f
