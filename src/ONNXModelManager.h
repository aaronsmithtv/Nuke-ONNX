#pragma once

#include "onnxruntime_cxx_api.h"
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include "ErrorHandling.h"

/**
 * Class to handle ONNX model loading, session management and inference
 */
class ONNXModelManager {
public:
    ONNXModelManager() : 
        _env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelManager"),
        _session(nullptr),
        _allocator(std::make_unique<Ort::AllocatorWithDefaultOptions>()),
        _modelLoaded(false) {}
    
    ~ONNXModelManager() {
        unload();
    }
    
    /**
     * Load ONNX model from file
     */
    void load(const char* modelPath, bool useGPU) {
        if (_modelLoaded) {
            unload();
        }
        
        _modelLoaded = false;

        try {
            // Configure session options
            Ort::SessionOptions sessionOptions;

            // Enable CUDA if requested and available
            if (useGPU) {
                OrtCUDAProviderOptions cudaOptions;
                sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            }

            // Create session
            _session = std::make_unique<Ort::Session>(_env, modelPath, sessionOptions);
            
            // Extract model information
            extractModelInfo();
            
            _modelLoaded = true;
        }
        catch (const Ort::Exception& e) {
            throw ModelLoadException(std::string("ONNX Runtime error: ") + e.what());
        }
        catch (const std::exception& e) {
            throw ModelLoadException(std::string("Standard exception: ") + e.what());
        }
    }
    
    /**
     * Unload the model and free resources
     */
    void unload() {
        _session.reset();
        
        // Clear input and output information
        _inputNames.clear();
        _outputNames.clear();
        _inputDims.clear();
        _outputDims.clear();
        
        _modelLoaded = false;
    }
    
    /**
     * Run inference on input tensor data
     */
    void runInference(const std::vector<float>& inputTensor, 
                     const std::vector<int64_t>& inputShape,
                     std::vector<float>& outputTensor) {
        if (!_modelLoaded || !_session) {
            throw InferenceException("Model not loaded");
        }
        
        // Prepare memory info
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        // Create input tensor
        Ort::Value inputOrtValue = Ort::Value::CreateTensor<float>(
            memoryInfo,
            const_cast<float*>(inputTensor.data()),
            inputTensor.size(),
            inputShape.data(),
            inputShape.size()
        );
        
        // Get input and output names
        const char* inputName = _inputNames[0].c_str();
        const char* outputName = _outputNames[0].c_str();
        
        // Run inference
        auto outputTensors = _session->Run(
            Ort::RunOptions{nullptr},
            &inputName,
            &inputOrtValue,
            1,
            &outputName,
            1
        );
        
        // Process output
        if (outputTensors.size() == 0 || !outputTensors[0].IsTensor()) {
            throw InferenceException("Invalid output tensor from ONNX Runtime");
        }
        
        // Get output tensor info
        auto typeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        auto outputShape = typeInfo.GetShape();
        size_t outputSize = typeInfo.GetElementCount();
        
        // Store updated output shape
        _outputDims[0] = outputShape;
        
        // Get output data
        const float* outputData = outputTensors[0].GetTensorData<float>();
        
        // Copy to output tensor
        outputTensor.assign(outputData, outputData + outputSize);
    }
    
    /**
     * Run inference with multiple input tensors
     * @param inputTensors Vector of input tensors
     * @param inputShapes Vector of input shapes
     * @param inputNames Vector of input names (must match model's input names)
     * @param outputTensor Output tensor data
     * @return True if inference was successful
     */
    bool runInferenceMultiInput(
        const std::vector<std::vector<float>>& inputTensors,
        const std::vector<std::vector<int64_t>>& inputShapes,
        const std::vector<std::string>& inputNames,
        std::vector<float>& outputTensor) {
        
        if (!_modelLoaded || !_session) {
            throw InferenceException("Model not loaded");
        }
        
        if (inputTensors.empty() || inputTensors.size() != inputShapes.size()) {
            throw InvalidArgumentException("Mismatch between input tensors and shapes");
        }
        
        // Validate input names match model inputs
        size_t numInputs = inputNames.size();
        if (numInputs > _inputNames.size()) {
            throw InvalidArgumentException("Too many inputs provided for the model");
        }
        
        // Prepare memory info
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        // Create vector of input tensor values
        std::vector<Ort::Value> inputValues;
        std::vector<const char*> inputNamesCStr;
        
        for (size_t i = 0; i < numInputs; i++) {
            // Create tensor for this input
            Ort::Value inputOrtValue = Ort::Value::CreateTensor<float>(
                memoryInfo,
                const_cast<float*>(inputTensors[i].data()),
                inputTensors[i].size(),
                inputShapes[i].data(),
                inputShapes[i].size()
            );
            
            inputValues.push_back(std::move(inputOrtValue));
            
            // Get the correct input name from model
            const char* inputName = nullptr;
            
            // Try to match with provided name first
            if (!inputNames[i].empty()) {
                // Find matching input name in model
                for (const auto& name : _inputNames) {
                    if (name == inputNames[i]) {
                        inputName = name.c_str();
                        break;
                    }
                }
            }
            
            // If no match, use the default input name
            if (inputName == nullptr) {
                inputName = _inputNames[i].c_str();
            }
            
            inputNamesCStr.push_back(inputName);
        }
        
        // Get first output name
        const char* outputName = _outputNames[0].c_str();
        
        // Run inference
        auto outputTensors = _session->Run(
            Ort::RunOptions{nullptr},
            inputNamesCStr.data(),
            inputValues.data(),
            numInputs,
            &outputName,
            1
        );
        
        // Process output
        if (outputTensors.size() == 0 || !outputTensors[0].IsTensor()) {
            throw InferenceException("Invalid output tensor from ONNX Runtime");
        }
        
        // Get output tensor info
        auto typeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        auto outputShape = typeInfo.GetShape();
        size_t outputSize = typeInfo.GetElementCount();
        
        // Store updated output shape
        _outputDims[0] = outputShape;
        
        // Get output data
        const float* outputData = outputTensors[0].GetTensorData<float>();
        
        // Copy to output tensor
        outputTensor.assign(outputData, outputData + outputSize);
        
        return true;
    }
    
    /**
     * Get model information as a formatted string
     */
    std::string getInfoString() const {
        if (!_modelLoaded || !_session) {
            return "No model loaded";
        }

        std::stringstream info;
        info << "\nONNX Model Information:\n";
        info << "---------------------\n";
        
        // Input information
        info << "Inputs: " << _inputNames.size() << "\n";
        for (size_t i = 0; i < _inputNames.size(); i++) {
            info << "  [" << i << "] " << _inputNames[i] << ": ";
            
            // Format dimensions
            if (!_inputDims[i].empty()) {
                info << "[";
                for (size_t d = 0; d < _inputDims[i].size(); d++) {
                    info << _inputDims[i][d];
                    if (d < _inputDims[i].size() - 1) info << ", ";
                }
                info << "]";
            }
            info << "\n";
        }
        info << "\n";
        
        // Output information
        info << "Outputs: " << _outputNames.size() << "\n";
        for (size_t i = 0; i < _outputNames.size(); i++) {
            info << "  [" << i << "] " << _outputNames[i] << ": ";
            
            // Format dimensions
            if (!_outputDims[i].empty()) {
                info << "[";
                for (size_t d = 0; d < _outputDims[i].size(); d++) {
                    info << _outputDims[i][d];
                    if (d < _outputDims[i].size() - 1) info << ", ";
                }
                info << "]";
            }
            info << "\n";
        }
        info << "\n";
        
        // Try to add model metadata if available
        try {
            Ort::ModelMetadata metadata = _session->GetModelMetadata();
            Ort::AllocatedStringPtr producerNamePtr = metadata.GetProducerNameAllocated(*_allocator);
            Ort::AllocatedStringPtr graphNamePtr = metadata.GetGraphNameAllocated(*_allocator);
            Ort::AllocatedStringPtr descriptionPtr = metadata.GetDescriptionAllocated(*_allocator);
            
            info << "Model Metadata:\n";
            if (strlen(producerNamePtr.get()) > 0) {
                info << "  Producer: " << producerNamePtr.get() << "\n";
            }
            if (strlen(graphNamePtr.get()) > 0) {
                info << "  Graph name: " << graphNamePtr.get() << "\n";
            }
            if (strlen(descriptionPtr.get()) > 0) {
                info << "  Description: " << descriptionPtr.get() << "\n";
            }
        } catch (...) {
            // Metadata might not be available for all models
        }

        return info.str();
    }
    
    // Getters for model information
    bool isLoaded() const { return _modelLoaded; }
    const std::vector<std::vector<int64_t>>& getInputDims() const { return _inputDims; }
    const std::vector<std::vector<int64_t>>& getOutputDims() const { return _outputDims; }
    int getInputCount() const { return _inputNames.size(); }
    int getOutputCount() const { return _outputNames.size(); }
    
    // Get input names for mapping to Nuke inputs
    const std::vector<std::string>& getInputNames() const { return _inputNames; }
    
    // Extract channel and dimension information
    bool getOutputDimensions(int& width, int& height, int& channels) const {
        if (!_modelLoaded || _outputDims.empty() || _outputDims[0].empty()) {
            return false;
        }
        
        // Extract based on shape format
        if (_outputDims[0].size() == 4) {
            // NCHW format: [batch, channels, height, width]
            channels = static_cast<int>(_outputDims[0][1]);
            height = static_cast<int>(_outputDims[0][2]);
            width = static_cast<int>(_outputDims[0][3]);
            return true;
        }
        else if (_outputDims[0].size() == 3) {
            // CHW format: [channels, height, width]
            channels = static_cast<int>(_outputDims[0][0]);
            height = static_cast<int>(_outputDims[0][1]);
            width = static_cast<int>(_outputDims[0][2]);
            return true;
        }
        // Additional formats could be handled here
        
        return false;
    }
    
private:
    void extractModelInfo() {
        // Clear existing info
        _inputNames.clear();
        _outputNames.clear();
        _inputDims.clear();
        _outputDims.clear();

        // Get input and output counts
        size_t inputCount = _session->GetInputCount();
        size_t outputCount = _session->GetOutputCount();

        // Get input info
        _inputNames.resize(inputCount);
        _inputDims.resize(inputCount);

        for (size_t i = 0; i < inputCount; i++) {
            // Get input name
            Ort::AllocatedStringPtr inputNamePtr = _session->GetInputNameAllocated(i, *_allocator);
            _inputNames[i] = inputNamePtr.get();  // Store as std::string

            // Get input type info
            auto typeInfo = _session->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

            // Get dimensions
            _inputDims[i] = tensorInfo.GetShape();
        }

        // Get output info
        _outputNames.resize(outputCount);
        _outputDims.resize(outputCount);

        for (size_t i = 0; i < outputCount; i++) {
            // Get output name
            Ort::AllocatedStringPtr outputNamePtr = _session->GetOutputNameAllocated(i, *_allocator);
            _outputNames[i] = outputNamePtr.get();  // Store as std::string

            // Get output type info
            auto typeInfo = _session->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

            // Get dimensions
            _outputDims[i] = tensorInfo.GetShape();
        }
    }

private:
    Ort::Env _env;
    std::unique_ptr<Ort::Session> _session;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> _allocator;
    bool _modelLoaded;
    
    // Model information
    std::vector<std::string> _inputNames;
    std::vector<std::string> _outputNames;
    std::vector<std::vector<int64_t>> _inputDims;
    std::vector<std::vector<int64_t>> _outputDims;
}; 