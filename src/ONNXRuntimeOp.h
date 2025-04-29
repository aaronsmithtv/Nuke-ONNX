#pragma once

#include "DDImage/Iop.h"
#include "DDImage/Knobs.h"
#include "DDImage/Row.h"
#include "DDImage/Thread.h"
#include "ONNXModelManager.h"
#include "TensorProcessor.h"
#include "ONNXInferenceProcessor.h"

#include <string>
#include <memory>
#include <vector>
#include <mutex>
#include <map>

/**
 * Nuke operator for running inference with ONNX Runtime models.
 * Allows loading and execution of ML models on images with GPU acceleration support.
 * Supports multiple inputs (1-10) for complex ONNX models.
 */
class ONNXRuntimeOp : public DD::Image::Iop
{
public:
    // Constructor and destructor
    ONNXRuntimeOp(Node* node);
    ~ONNXRuntimeOp() override;

    // Required Nuke plugin overrides
    void _validate(bool for_real) override;
    void _request(int x, int y, int r, int t, DD::Image::ChannelMask channels, int count) override;
    void engine(int y, int x, int r, DD::Image::ChannelMask channels, DD::Image::Row& row) override;
    void knobs(DD::Image::Knob_Callback f) override;
    int knob_changed(DD::Image::Knob* k) override;
    const char* Class() const override;
    const char* node_help() const override;
    static const DD::Image::Iop::Description description;
    std::string input_longlabel(int input) const override;
    void _open() override;

    // Multi-input support
    int minimum_inputs() const override { return 1; }
    int maximum_inputs() const override { return 10; }

private:
    // ONNX model configuration
    const char* _modelPath;     // Path to the ONNX model file
    bool _useGPU;               // Whether to use GPU acceleration
    bool _normalize;            // Whether to normalize output values to [0,1]
    
    // Output configuration
    bool _isSingleChannel;      // Whether output is single-channel (like depth)
    int _outputChannelCount;    // Number of channels in the model output
    float _minValue;            // Minimum value for normalization
    float _maxValue;            // Maximum value for normalization
    
    // Dimensions and format tracking
    DD::Image::FormatPair _formats;  // Nuke format information
    bool _dimensionsSet;             // Whether dimensions have been determined
    int _imgWidth, _imgHeight, _imgChannels;      // Input image dimensions
    int _outputWidth, _outputHeight;              // Output dimensions
    
    // Processing state
    std::unique_ptr<ONNXModelManager> _modelManager;   // Manages ONNX model and inference
    std::unique_ptr<ONNXInferenceProcessor> _inferenceProcessor;   // Handles inference workflow
    DD::Image::Lock _cacheLock;       // Thread safety for caching
    bool _cacheValid;                 // Whether cached data is valid
    bool _processingDone;             // Whether processing is complete
    std::vector<float> _processedData; // Processed tensor data
    
    // Multi-input support
    int _activeInputs;                // Number of active inputs
    
    // Core functionality
    void loadModel();                // Load the ONNX model
    void updateDimensions();         // Update output dimensions based on model info
    void cacheAndProcessImage();     // Process input image through the model
    void preprocessImage(const DD::Image::Iop* input, std::vector<float>& inputTensor);
    
    // Output handling
    void findMinMaxValues();         // Find min/max values for normalization
    void setupOutputChannels(DD::Image::ChannelSet& channels);  // Configure output channels
    
    // UI
    void displayModelInfo();         // Display model info in Nuke UI
    void updateActiveInputs();       // Update count of active inputs based on model
};