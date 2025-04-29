# ONNX Runtime Node for Nuke

A plugin for The Foundry's Nuke that enables neural network inference on images using the ONNX Runtime.

**NOTE:** This plugin has been developed and tested on Linux.

## Features

- Load and run ONNX format models directly in Nuke.
- Supports models with multiple inputs (up to 10).
- Integrated normalization option for output values (useful for depth maps, etc.).
- Compatible with various model architectures (image-to-image, segmentation, etc.).
- Displays model information (inputs, outputs, dimensions) in the Nuke console.

## Requirements

- The Foundry Nuke 14.0+ (Nuke's embedded Python 3.9 is also used at compile)
- A C++ compiler supporting C++14 (e.g., GCC 7+)
- CMake 3.10+
- ONNX Runtime library v1.12.0 (you should include this in `third_party/onnxruntime/`). Newer ONNX Runtime versions might require tweaks.

## Building The Plugin (CMake)

1.  **Set Environment Variable for Nuke:**
    Ensure the `NUKE_ROOT` environment variable points to your Nuke installation directory (the one containing `include/` and `libNuke*.so`).
    ```bash
    export NUKE_ROOT=/path/to/your/Nuke14.xVx # Example: /usr/local/Nuke14.1v2
    ```

2.  **Configure and Build:**
    Create a build directory and run CMake, then build using Make. It is very important to configure CMake to use the older C++ ABI (`_GLIBCXX_USE_CXX11_ABI=0`) for compatibility with Nuke, as described in the NDK documentation.

    ```bash
    mkdir build
    cd build
    cmake .. -DNUKE_ROOT=$NUKE_ROOT -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
    make
    ```
    *    `-DNUKE_ROOT=$NUKE_ROOT` tells CMake where to find the Nuke SDK.
    *    `-DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"` sets the required ABI flag.

3.  **Result:**
    The compiled plugin `ONNXRuntimeOp.so` will be located inside the `build/` directory.

## Installation (Linux)

1.  **Create Plugin Directory:** Choose a location for your Nuke plugins. A common place is `~/.nuke/ONNXRuntimeOp`. Create this directory if it doesn't exist.
    ```bash
    mkdir -p ~/.nuke/ONNXRuntimeOp
    ```

2.  **Copy Plugin:** Copy the compiled `ONNXRuntimeOp.so` (from the `build/` directory) into your chosen plugin directory.
    ```bash
    cp build/ONNXRuntimeOp.so ~/.nuke/ONNXRuntimeOp/
    ```

3.  **Copy ONNX Runtime Library:** Create a `lib` subdirectory inside your plugin directory and copy the ONNX Runtime shared library into it. This is necessary because the plugin's RPATH is set to look here (`$ORIGIN/lib`).
    ```bash
    mkdir -p ~/.nuke/ONNXRuntimeOp/lib
    cp third_party/onnxruntime/lib/libonnxruntime.so* ~/.nuke/ONNXRuntimeOp/lib/
    ```
    *   Make sure to copy all related `libonnxruntime.so` files (e.g., `libonnxruntime.so`, `libonnxruntime.so.1.12.0`).

4.  **Add to Nuke Plugin Path:** Ensure Nuke knows where to find your plugin. You can do this by adding the parent directory (`~/.nuke` in this example) to your Nuke plugin path. If this has been successfully set, the _ONNXRuntimeOp_ node should now be available in Nuke.

## Usage

1.  Create an `ONNXRuntimeOp` node.
2.  Connect the primary input image(s) required by your model. Input labels will update with model input names when a model is loaded.
3.  In the node's properties panel, use the `model_path` file browser to select your `.onnx` model file.
4.  The node will attempt to load the model. Check the Nuke console/terminal for full success or error messages.
5.  Configure options:
    *   **Normalize Output:** Check this if your model outputs values outside the typical 0-1 image range (e.g., depth maps). The output will be normalized based on the min/max values found in the tensor.
    *   **Reload Model:** Click to force reloading the model from the specified path.
    *   **Print Model Info:** Click to print detailed information about the loaded model's inputs, outputs, and dimensions to the console.
6.  The node will process the input image(s) through the model and output the results. The output format and resolution may change based on the model's output tensor shape.

## Known Issues / Limitations

*   Currently only tested and supported on Linux
*   Assumes ONNX models use `float` tensors
*   Expects NCHW tensor layout
*   GPU execution (`use_gpu` knob) is currently disabled

## License

MIT License - see the `LICENSE` file for details.