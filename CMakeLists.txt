# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name for the header-only library
project(IRGeneralLibrary)

# This library is header-only, so we only need to specify the include directory
add_library(IRGeneralLibrary INTERFACE)

# Specify the include directory
target_include_directories(IRGeneralLibrary INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
