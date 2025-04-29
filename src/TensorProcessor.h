#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <stdexcept>

/**
 * TensorProcessor - Handles tensor operations for ONNX inference
 * Contains functions and structures for tensor manipulation independent of Nuke/NDK
 */
class TensorProcessor {
public:
    // Structure to hold input tensor information
    struct InputTensorInfo {
        std::vector<float> data;         // Input tensor data
        std::vector<int64_t> shape;      // Input tensor shape
        std::string name;                // Input tensor name
        bool valid;                      // Whether this input is valid

        InputTensorInfo() : data(), shape(), name(""), valid(false) {}
    };

    /**
     * Find minimum and maximum values in a tensor
     * @param tensorData The tensor data
     * @param minValue Output minimum value found
     * @param maxValue Output maximum value found
     */
    static void findMinMax(const std::vector<float>& tensorData, float& minValue, float& maxValue) {
        if (tensorData.empty()) {
            minValue = 0.0f;
            maxValue = 1.0f;
            return;
        }

        minValue = std::numeric_limits<float>::max();
        maxValue = std::numeric_limits<float>::lowest();

        for (const float& value : tensorData) {
            // Check for NaN or Inf values
            if (std::isnan(value) || std::isinf(value)) {
                continue;  // Skip invalid values
            }
            
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }

        // Prevent division by zero in normalization
        if (minValue == maxValue || std::isnan(minValue) || std::isinf(minValue) || 
            std::isnan(maxValue) || std::isinf(maxValue)) {
            minValue = 0.0f;
            maxValue = 1.0f;
        }
    }

    /**
     * Find min and max values for multi-channel tensors
     * @param tensorData The tensor data
     * @param minValue Output minimum value found
     * @param maxValue Output maximum value found
     * @param channelCount Number of channels
     * @param width Width of the tensor
     * @param height Height of the tensor
     */
    static void findMinMaxMultiChannel(
        const std::vector<float>& tensorData,
        float& minValue, float& maxValue,
        int channelCount, int width, int height
    ) {
        if (tensorData.empty() || width <= 0 || height <= 0 || channelCount <= 0) {
            minValue = 0.0f;
            maxValue = 1.0f;
            return;
        }

        minValue = std::numeric_limits<float>::max();
        maxValue = std::numeric_limits<float>::lowest();
        size_t pointsPerChannel = static_cast<size_t>(width * height);
        
        // Find min and max across all channels
        for (int c = 0; c < channelCount; c++) {
            size_t startIdx = c * pointsPerChannel;
            size_t endIdx = std::min(startIdx + pointsPerChannel, tensorData.size());
            
            // Skip if out of bounds
            if (startIdx >= tensorData.size()) continue;
            
            // Find min/max for this channel
            float channelMin = std::numeric_limits<float>::max();
            float channelMax = std::numeric_limits<float>::lowest();
            
            for (size_t i = startIdx; i < endIdx; i++) {
                // Check for NaN or Inf values
                if (std::isnan(tensorData[i]) || std::isinf(tensorData[i])) {
                    continue;  // Skip invalid values
                }
                
                channelMin = std::min(channelMin, tensorData[i]);
                channelMax = std::max(channelMax, tensorData[i]);
            }
            
            if (channelMin != std::numeric_limits<float>::max() && 
                channelMax != std::numeric_limits<float>::lowest()) {
                minValue = std::min(minValue, channelMin);
                maxValue = std::max(maxValue, channelMax);
            }
        }

        // Check if we found any valid values
        if (minValue == std::numeric_limits<float>::max() || 
            maxValue == std::numeric_limits<float>::lowest() ||
            std::isnan(minValue) || std::isinf(minValue) || 
            std::isnan(maxValue) || std::isinf(maxValue)) {
            minValue = 0.0f;
            maxValue = 1.0f;
        }
        // Prevent division by zero in normalization
        else if (minValue == maxValue) {
            maxValue = minValue + 1.0f;
        }
    }

    /**
     * Normalize a value to the range [0, 1]
     * @param value The value to normalize
     * @param min Minimum value in the range
     * @param max Maximum value in the range
     * @return The normalized value
     */
    static float normalize(float value, float min, float max) {
        // Handle invalid input values
        if (std::isnan(value) || std::isinf(value)) {
            return 0.5f;
        }
        
        // Handle edge case
        if (min >= max || std::isnan(min) || std::isinf(min) || 
            std::isnan(max) || std::isinf(max)) {
            return 0.5f;
        }
        
        // Clamp value to the range [min, max]
        float clampedValue = std::max(min, std::min(max, value));
        return (clampedValue - min) / (max - min);
    }

    /**
     * Get tensor value with bounds checking
     * @param tensorData The tensor data
     * @param x X coordinate
     * @param y Y coordinate
     * @param channelIdx Channel index
     * @param width Width of the tensor
     * @param height Height of the tensor
     * @param isSingleChannel Whether tensor is single channel
     * @param normalize Whether to normalize the output
     * @param minValue Minimum value for normalization
     * @param maxValue Maximum value for normalization
     * @return The tensor value
     */
    static float getTensorValue(
        const std::vector<float>& tensorData,
        int x, int y, int channelIdx,
        int width, int height,
        bool isSingleChannel,
        bool doNormalize,
        float minValue, float maxValue
    ) {
        // Validate parameters
        if (tensorData.empty() || width <= 0 || height <= 0) {
            return 0.0f;
        }
        
        // Bounds check on coordinates
        if (x < 0 || x >= width || y < 0 || y >= height || channelIdx < 0) {
            return 0.0f;
        }
        
        int dataIndex;
        if (isSingleChannel) {
            // Single channel mode (flat index)
            dataIndex = y * width + x;
        } else {
            // Multi-channel mode (NCHW format)
            size_t offset = channelIdx * height * width;
            if (offset >= tensorData.size()) {
                return 0.0f;
            }
            dataIndex = offset + y * width + x;
        }
        
        if (dataIndex >= 0 && dataIndex < static_cast<int>(tensorData.size())) {
            float value = tensorData[dataIndex];
            
            // Check for NaN or Inf
            if (std::isnan(value) || std::isinf(value)) {
                return 0.0f;
            }
            
            return doNormalize ? normalize(value, minValue, maxValue) : value;
        }
        
        return 0.0f;
    }
}; 