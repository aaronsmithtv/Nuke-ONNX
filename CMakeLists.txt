cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(ONNXNukeNode)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Set paths for ONNX Runtime
set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include")
set(ONNXRUNTIME_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib")

# Find Nuke
find_package(Nuke REQUIRED)

# Find Python (using Nuke's embedded Python 3.9)
set(PYTHON_INCLUDE_DIRS ${NUKE_INCLUDE_DIR}/../include/python3.9)
if(APPLE)
    set(PYTHON_LIBRARIES ${NUKE_LIBRARY_DIR}/libpython3.9.dylib)
else()
    set(PYTHON_LIBRARIES ${NUKE_LIBRARY_DIR}/libpython3.9.so)
endif()

# Include directories
include_directories(
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link directories
link_directories(${ONNXRUNTIME_LIB_DIR})

# Source files
set(SOURCE_FILES
        src/ONNXRuntimeOp.cpp
)

# Header files
set(HEADER_FILES
        src/ONNXRuntimeOp.h
        src/ONNXModelManager.h
        src/Utils.h
        src/TensorProcessor.h
        src/ONNXInferenceProcessor.h
)

# Add Nuke plugin
add_nuke_plugin(ONNXRuntimeOp ${SOURCE_FILES})
target_link_libraries(ONNXRuntimeOp ${PYTHON_LIBRARIES})