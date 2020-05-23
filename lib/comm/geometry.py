# -*- coding:utf-8 -* 

import numpy 
import numpy as np

import cv2 as cv
import cv2
def showimg(img,nm="pic"):
    cv2.imshow(nm,img)
    return cv2.waitKey()
    
def readlines(fname):
    with open(fname) as fp:
        list=fp.readlines()
    for id, item in enumerate(list):
        list[id]=item.strip()
    return list
    

#@2018-12-8            
def get_obj_v(fnm):
    lines=readlines(fnm)
    v=[]
    for line in lines:
        sz=line.split()
        
        if len(sz)>=4 and sz[0]=="v":
            v.append( [ float(x) for x in sz[1:4] ] )
        if len(sz)==4 and sz[0]=="f":
            break
    v = np.array( v ).astype(np.float32)
    
    #print(v.shape)
    return v    

def get_obj_v_f(fnm):
    lines=readlines(fnm)
    v=[]
    f=[]
    for line in lines:
        sz=line.split()
        
        if len(sz)>=4 and sz[0]=="v":
            v.append( [ float(x) for x in sz[1:4] ] )

        if len(sz)==4 and sz[0]=="f":
            f.append( [ int(x) for x in sz[1:4] ] )
            
    v = np.array( v ).astype(np.float32)
    f = np.array( f ).astype(np.int32) - 1 #!!! 1-base to 0-base
    #p(v.shape)
    return v, f 

def get_obj_v_t_f(fnm):
    lines=readlines(fnm)
    v=[]
    t=[]
    f=[]
    for line in lines:
        sz=line.split()
        
        if len(sz)>=4 and sz[0]=="v":
            v.append( [ float(x) for x in sz[1:4] ] )
            t.append( [ float(x) for x in sz[4:7] ] )

        if len(sz)==4 and sz[0]=="f":
            f.append( [ int(x) for x in sz[1:4] ] )
            
    v = np.array( v ).astype(np.float32)
    t = np.array( t ).astype(np.float32)
    f = np.array( f ).astype(np.int32) - 1 #!!! 1-base to 0-base
    
    return v, t, f     

def save_obj(fname, vtx, face=None, tex=None, decimals=5, v_fmt=None):
    if v_fmt is not None: print("don't use v_fmt")
    vtx = vtx.reshape(-1, 3)
    if tex is not None:
        tex[tex < 0] = 0
        tex[tex > 1] = 1
        tex = tex.reshape(-1, 3)

    #num = vtx.shape[0]
    if tex is not None:
        vtx = np.hstack( (vtx, tex) )
    vtx=np.around(vtx, decimals=decimals).astype(np.str).tolist()
    with open(fname, "w") as fp:
        for vec in vtx:
            s = " ".join(vec)
            s = f"v {s}\n"
            fp.write(s)

        if face is None: return
        face=face+1
        face = face.astype(np.str).tolist()
        for f in face:
            s = " ".join(f)
            s = f"f {s}\n"
            fp.write(s)
