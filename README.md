# Introduction

This is my simple framework to accelerate python with C / CUDA. A useful example about 3D object rendering is provided to show how to use it.

**TIPS about framework**
1. It just a framework about invoking a dynamic link library by python .
2. Python‘s cuda variables is pytorch's cuda tensors.
3. It shows convenient mutual accessing between numpy/pytorch and C/CUDA.
4. Avoid calling C functions on python multi-threads. Multi-thread are suggested to be coded in C (c++11, openmp ..) and invoked by python.
5. C code can also includes another c++ libraries like opencv.


**TIPS about rendering**
1. If you study deep learning about 3D reconstruction, you may know why I write this example and what the following tips to say.
2. Supporting batch rendering.
3. But, I'm not provide the example about neural network back propagation.
Actually, if you want to autodiff by pytorch, the rendering code can't be write as this. You can use my framework to compute rendering informations quickly, and then render images only by pytorch operations.
4. The rendering code is not perfect and really simple. You may need to correct or modify it according to your tasks.

# Requirements
CUDA

torch (python)

# Demo
To run my code, you must got to dir `chj_speed_cdll` to compile a dynamic link library first. It is only related with cuda, not related with python, numpy or pytorch. So if your are familiar with C++, it will be very easy.

## 1.mesh_render.py
Just see the function `f1()`, you will know how to use it.

Here is the results:

![res](resource/res.jpg)

Meshlab has added its own lighting and use perspective projection as default. For this rendering, I assume your 3D obj has done perspective projection (z use its original) or orthogonal projection.


