cmake_minimum_required(VERSION 2.8)
# choose your name
set(targetname chj_speed_deep3dface)
project(${targetname})
#set(targetname python_use)


#
add_compile_options(-fPIC)
#
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -03")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -03 -std=c++11" )


# openmp support
#find_package(OpenMP REQUIRED)
#
#if(OPENMP_FOUND)
#    message("OPENMP FOUND")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

#set(EXCUTABLE_OUPT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(SRC_DIR ${PROJECT_SOURCE_DIR}/)

include_directories(${SRC_DIR}/src)

#include_directories()
#LINK_DIRECTORIES()

#file(GLOB_RECURSE CURRENT_HEADERS ${SRC_DIR}/src/*.h )
file(GLOB CURRENT_HEADERS ${SRC_DIR}/src/*.h )
file(GLOB CURRENT_SOURCES ${SRC_DIR}/src/*.cpp ${SRC_DIR}/src/*.c )


find_package(CUDA)
ADD_DEFINITIONS(-DGPU -DNO_CUDNN)
include_directories(${SRC_DIR}/src_cu)
IF (WIN32) 
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler /wd4819")
ENDIF()
file(GLOB CU_CURRENT_HEADERS ${SRC_DIR}/src_cu/*.h )
file(GLOB CU_CURRENT_SOURCES ${SRC_DIR}/src_cu/*.cpp ${SRC_DIR}/src_cu/*.c )
file(GLOB CURRENT_CUDA_SOURCES ${SRC_DIR}/src_cu/*.cu )

source_group("Include" FILES ${CURRENT_HEADERS} ${CU_CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES} ${CU_CURRENT_SOURCES})
source_group("CuSource" FILES ${CURRENT_CUDA_SOURCES})


CUDA_COMPILE(CU_OBJ ${CURRENT_CUDA_SOURCES})
cuda_add_library(${targetname} SHARED ${CU_CURRENT_SOURCES} ${CU_CURRENT_HEADERS} ${CU_OBJ} ${CURRENT_SOURCES} ${CURRENT_HEADERS})

target_link_libraries(${targetname} ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})

SET_TARGET_PROPERTIES(${targetname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${LIBRARY_OUTPUT_PATH}) 
SET_TARGET_PROPERTIES(${targetname} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
