# Introduction

This is my simple framework to accelerate python with C / CUDA. A useful example about 3D object rendering is provided to show how to use it.

**TIPS about framework**
1. It just a framework about invoking a dynamic link library by python .
2. Python‘s cuda variables is pytorch's cuda tensors.
3. It shows convenient mutual accessing between numpy/pytorch and C/CUDA.
4. Avoid calling C functions on python multi-threads. Multi-thread are suggested to be coded in C (c++11, openmp ..) and invoked by python.
5. C code can also includes another c++ libraries like opencv.


**TIPS about rendering**
1. Supporting batch rendering.
2. I don't provide the example about neural network back propagation.
Actually, if you want to autodiff by pytorch, the rendering code can't be write as this. You can use my framework to compute rendering informations quickly, and then render images only by pytorch operations.


# Getting Started
## Requirements
CUDA

PyTorch

## Running demo
```
# 1. Clone
git clone https://github.com/cleardusk/SimRender.git
cd SimRender

# 2. build the library driven by ctypes
cd lib
mkdir build
cd build
cmake ..
make

# 3. run demo
python3 demo.py
```

The terminal output is
```
Loading from resource/face1.obj
Elapse: 5.8ms
Elapse: 0.4ms
Elapse: 0.4ms
Elapse: 0.4ms
Elapse: 0.4ms
Rendered to res/face1.png
```


The rendered results of MeshLab and ours are:

![res](resource/res.jpg)

Meshlab has added its own lighting and use perspective projection as default. For this rendering, I assume your 3D obj has done perspective projection (z use its original) or orthogonal projection.

# Speed
Rendering a single image only takes 0.4ms on Titan X GPU. With a small batch of 128 samples, it takes about 10ms.  

(On GTX1060 3G) Rendering two 224*224 images takes about 1ms. 

Rough measurements.

| batch size | times (ms) |
|--------|--------|
| 2 | 1 |
|  16      |   4.2     |
|  64      |   16     |
|  128      |   35     |

# Future Work

1. Lighting.

