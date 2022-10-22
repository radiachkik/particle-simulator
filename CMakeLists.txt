cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(particle_simulator LANGUAGES CXX)

# Set variables
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(OpenGL_GL_PREFERENCE GLVND)
# set(OpenGL_GL_PREFERENCE LEGACY)
message(STATUS "Cuda Architecture: ${CMAKE_CUDA_ARCHITECTURES}")

# Check Dependencies
find_package(CUDA REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

# Directory definition
set (PROJECT_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
set (PROJECT_SRC_DIR      "${PROJECT_SOURCE_DIR}/src")
# set (PROJECT_DEPS_DIR     "${PROJECT_SOURCE_DIR}/deps")

# Source files
file(GLOB_RECURSE SOURCES "${PROJECT_SRC_DIR}/*.cpp" "${PROJECT_SRC_DIR}/*.c")
file(GLOB_RECURSE SOURCES_CUDA "${PROJECT_SRC_DIR}/*.cu")

# Header files
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
cuda_include_directories("${OPENGL_INCLUDE_DIRS}")
include_directories(${GLUT_INCLUDE_DIRS} )
cuda_include_directories("${GLUT_INCLUDE_DIRS}")
include_directories("${PROJECT_INCLUDE_DIRS}")
cuda_include_directories("${PROJECT_INCLUDE_DIRS}")

# Compilation
cuda_add_executable(particle_simulator
        ${SOURCES}
        ${SOURCES_CUDA}
        )
set_target_properties(particle_simulator PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_VERBOSE_BUILD ON)
target_link_libraries(particle_simulator
        ${OPENGL_LIBRARIES}
        ${GLUT_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_curand_LIBRARY}
        )
