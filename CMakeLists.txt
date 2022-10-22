cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(particle_simulator LANGUAGES CXX CUDA)

# Set variables
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(OpenGL_GL_PREFERENCE GLVND)

# Directory definition
set (PROJECT_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
set (PROJECT_SRC_DIR      "${PROJECT_SOURCE_DIR}/src")

# Check Dependencies
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

# Include project files
file(GLOB_RECURSE SOURCES "${PROJECT_SRC_DIR}/*.cpp" "${PROJECT_SRC_DIR}/*.c" "${PROJECT_SRC_DIR}/*.cu")
include_directories(${PROJECT_INCLUDE_DIRS})

# Compilation
add_executable(particle_simulator
        ${SOURCES}
        )
target_link_libraries(particle_simulator
        OpenGL::OpenGL
        GLUT::GLUT
        )
set_target_properties(particle_simulator PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_VERBOSE_BUILD ON
        )
