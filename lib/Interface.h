#pragma once

#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#define  DLL_API _declspec(dllexport)
#else
#define  DLL_API
#endif // _WIN32

#define EXT_C_DLL extern "C" DLL_API
#define EXT_C_DLL_VOID extern "C" DLL_API void

//#define MAX_RECEIVE_ARRAY 20
//#define MAX_ONCE_RECEIVE 30


#include <cmath>
#include <ctime>
#include <cstdio>
#include <cassert>
#include <cstdarg>
#include <algorithm>
#include<iostream>

using namespace std;

#include<map>
#include<string>

class cls_c_param {
public:
	//void * data[MAX_RECEIVE_ARRAY][MAX_ONCE_RECEIVE];
	map<string, void *> mp;
};

extern cls_c_param c_param;


