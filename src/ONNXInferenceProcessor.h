#pragma once

#include "ONNXModelManager.h"
#include "TensorProcessor.h"
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include "ErrorHandling.h"

/**
 * ONNXInferenceProcessor - Handles the core inference workflow for ONNX models
 * 
 * This class encapsulates the non-Nuke specific processing logic for preparing inputs,
 * running inference, and organizing outputs for ONNX models.
 */
class ONNXInferenceProcessor {
public:
    ONNXInferenceProcessor() : 
        _modelManager(nullptr),
        _inputTensors(),
        _width(0),
        _height(0),
        _channels(0),
        _outputWidth(0),
        _outputHeight(0),
        _outputChannels(0),
        _isSingleChannel(true)
    {}
    
    /**
     * Set the model manager to use for inference
     * @param modelManager Pointer to the model manager
     */
    void setModelManager(ONNXModelManager* modelManager) {
        _modelManager = modelManager;
    }
    
    /**
     * Set the input dimensions
     * @param width Input width
     * @param height Input height  
     * @param channels Input channels
     */
    void setInputDimensions(int width, int height, int channels) {
        if (width <= 0 || height <= 0 || channels <= 0) {
            std::string msg = "Invalid input dimensions: " + 
                                std::to_string(width) + "x" + 
                                std::to_string(height) + " with " + 
                                std::to_string(channels) + " channels";
            throw ConfigurationException(msg);
        }
        
        _width = width;
        _height = height;
        _channels = channels;
    }
    
    /**
     * Get the output dimensions
     * @param width Output parameter for width
     * @param height Output parameter for height
     * @param channels Output parameter for channels
     * @return True if dimensions were retrieved successfully
     */
    bool getOutputDimensions(int& width, int& height, int& channels) const {
        width = _outputWidth;
        height = _outputHeight;
        channels = _outputChannels;
        return (_outputWidth > 0 && _outputHeight > 0 && _outputChannels > 0);
    }
    
    /**
     * Prepare input tensors for inference
     * @param inputCount Number of inputs to process
     * @return True if successful
     */
    void prepareInputs(int inputCount) {
        if (!_modelManager) {
            throw ConfigurationException("Model manager is not set");
        }
            
        if (!_modelManager->isLoaded()) {
            throw ConfigurationException("No model has been loaded in the manager");
        }
        
        if (inputCount <= 0) {
            throw InvalidArgumentException("Input count must be positive: " + std::to_string(inputCount));
        }
            
        try {
            // Clear existing input tensors
            _inputTensors.clear();
            _inputTensors.resize(inputCount);
            
            // Get model input info
            const auto& modelInputNames = _modelManager->getInputNames();
            const auto& modelInputDims = _modelManager->getInputDims();
            
            // Set up each input tensor with correct metadata
            for (int i = 0; i < inputCount; i++) {
                // Set input name from model if available
                if (i < static_cast<int>(modelInputNames.size())) {
                    _inputTensors[i].name = modelInputNames[i];
                } else {
                    _inputTensors[i].name = "";
                }
                
                // Prepare shape based on model expectations
                if (i < static_cast<int>(modelInputDims.size()) && !modelInputDims[i].empty()) {
                    // Use model's expected shape as a template
                    _inputTensors[i].shape = modelInputDims[i];
                    
                    // Override height/width with actual dimensions
                    if (_inputTensors[i].shape.size() >= 4) {
                        // NCHW format: adjust height and width
                        _inputTensors[i].shape[2] = static_cast<int64_t>(_height); 
                        _inputTensors[i].shape[3] = static_cast<int64_t>(_width);
                    }
                } else {
                    // Use default NCHW format if no specific shape info
                    _inputTensors[i].shape = {
                        1, _channels, 
                        static_cast<int64_t>(_height), 
                        static_cast<int64_t>(_width)
                    };
                }
                
                // Mark as not valid initially - will be set to valid when data is added
                _inputTensors[i].valid = false;
            }
        }
        catch (const std::exception& e) {
            throw ConfigurationException(std::string("Error preparing inputs: ") + e.what());
        }
    }
    
    /**
     * Set data for a specific input tensor
     * @param inputIndex The index of the input tensor
     * @param data The tensor data
     */
    void setInputTensorData(int inputIndex, const std::vector<float>& data) {
        if (inputIndex < 0 || inputIndex >= static_cast<int>(_inputTensors.size())) {
            std::string msg = "Input index " + std::to_string(inputIndex) + 
                                " out of range (size: " + std::to_string(_inputTensors.size()) + ")";
            throw InvalidArgumentException(msg);
        }
        
        if (data.empty()) {
            throw InvalidArgumentException("Input tensor data for index " + std::to_string(inputIndex) + " is empty");
        }
            
        _inputTensors[inputIndex].data = data;
        _inputTensors[inputIndex].valid = true;
    }
    
    /**
     * Run inference using prepared input tensors
     * @param outputTensor Output tensor to store results
     */
    void runInference(std::vector<float>& outputTensor) {
        if (!_modelManager) {
            throw ConfigurationException("Model manager is not set");
        }
            
        if (!_modelManager->isLoaded()) {
            throw ConfigurationException("No model has been loaded in the manager");
        }
            
        try {
            // Prepare data for inference
            std::vector<std::vector<float>> inputTensors;
            std::vector<std::vector<int64_t>> inputShapes;
            std::vector<std::string> inputNames;
            
            // Collect all valid input tensors
            for (size_t i = 0; i < _inputTensors.size(); i++) {
                if (_inputTensors[i].valid) {
                    if (_inputTensors[i].data.empty()) {
                        throw ConfigurationException("Input tensor " + std::to_string(i) + " has empty data despite being marked valid");
                    }
                    
                    if (_inputTensors[i].shape.empty()) {
                        throw ConfigurationException("Input tensor " + std::to_string(i) + " has empty shape despite being marked valid");
                    }
                    
                    inputTensors.push_back(_inputTensors[i].data);
                    inputShapes.push_back(_inputTensors[i].shape);
                    inputNames.push_back(_inputTensors[i].name);
                }
            }
            
            // Verify we have at least one valid input
            if (inputTensors.empty()) {
                throw ConfigurationException("No valid input tensors available for inference");
            }
            
            // Clear output
            outputTensor.clear();
            
            // Use multi-input method if multiple valid inputs
            if (inputTensors.size() > 1) {
                try {
                    _modelManager->runInferenceMultiInput(
                        inputTensors, inputShapes, inputNames, outputTensor);
                } catch (const std::exception& e) {
                    // Rethrow underlying exceptions as InferenceException
                    throw InferenceException(std::string("Multi-input inference failed: ") + e.what());
                }
            } 
            // Fall back to single input method if only one valid input
            else if (inputTensors.size() == 1) {
                try {
                    _modelManager->runInference(
                        inputTensors[0], inputShapes[0], outputTensor);
                } catch (const std::exception& e) {
                    // Rethrow underlying exceptions as InferenceException
                    throw InferenceException(std::string("Single-input inference failed: ") + e.what());
                }
            }
            
            // Calculate output dimensions based on result
            // Most models output in NCHW format (batch, channels, height, width)
            _outputWidth = _width; // Default: same as input
            _outputHeight = _height; // Default: same as input
            _outputChannels = 1; // Default: 1 channel
            
            // Try to get model's output information
            const auto& outputDims = _modelManager->getOutputDims();
            if (!outputDims.empty() && !outputDims[0].empty()) {
                // NCHW format: [batch, channels, height, width]
                if (outputDims[0].size() >= 4) {
                    _outputChannels = static_cast<int>(outputDims[0][1]);
                    _outputHeight = static_cast<int>(outputDims[0][2]);
                    _outputWidth = static_cast<int>(outputDims[0][3]);
                }
                // CHW format: [channels, height, width]
                else if (outputDims[0].size() == 3) {
                    _outputChannels = static_cast<int>(outputDims[0][0]);
                    _outputHeight = static_cast<int>(outputDims[0][1]);
                    _outputWidth = static_cast<int>(outputDims[0][2]);
                }
                // HW format: [height, width] - single channel
                else if (outputDims[0].size() == 2) {
                    _outputChannels = 1;
                    _outputHeight = static_cast<int>(outputDims[0][0]);
                    _outputWidth = static_cast<int>(outputDims[0][1]);
                }
            }
            
            // Detect if output should be treated as single-channel
            _isSingleChannel = (_outputChannels == 1);
        }
        catch (const ConfigurationException& e) { // Catch specific config errors first
            throw; // Rethrow config errors directly
        }
        catch (const InferenceException& e) { // Catch specific inference errors
             throw; // Rethrow inference errors directly
        }
        catch (const std::exception& e) {
            // Catch any other standard exceptions during setup or post-processing
            throw InferenceException(std::string("Unexpected error during inference processing: ") + e.what());
        }
    }
    
    /**
     * Check if the output is single-channel
     * @return True if output is single-channel
     */
    bool isSingleChannelOutput() const {
        return _isSingleChannel;
    }
    
    /**
     * Get the number of output channels
     * @return Output channel count
     */
    int getOutputChannelCount() const {
        return _outputChannels;
    }
    
    /**
     * Access a specific input tensor
     * @param index The input tensor index
     * @return Reference to the input tensor
     */
    TensorProcessor::InputTensorInfo& getInputTensor(size_t index) {
        if (index >= _inputTensors.size()) {
            throw std::out_of_range("Input tensor index out of range");
        }
        return _inputTensors[index];
    }
    
    /**
     * Get all input tensors
     * @return Vector of input tensors
     */
    const std::vector<TensorProcessor::InputTensorInfo>& getInputTensors() const {
        return _inputTensors;
    }
    
private:
    ONNXModelManager* _modelManager; // Model manager to use for inference
    std::vector<TensorProcessor::InputTensorInfo> _inputTensors; // Input tensors
    
    // Input dimensions
    int _width;
    int _height;
    int _channels;
    
    // Output dimensions
    int _outputWidth;
    int _outputHeight;
    int _outputChannels;
    bool _isSingleChannel;
}; 