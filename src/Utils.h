#pragma once

#include "DDImage/Format.h"
#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Tile.h"
#include "ErrorHandling.h"
#include "TensorProcessor.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace Utils {

/**
 * Extract a complete tile from an Iop (input operator)
 * @param input Reference to the input operator
 * @param channels The channel mask to extract
 * @return A tile containing the requested data
 */
inline DD::Image::Tile extractTile(const DD::Image::Iop &input,
                                   DD::Image::ChannelSet channels) {
  // Get the format and bounds
  DD::Image::Format f = input.format();
  DD::Image::Box box(f.x(), f.y(), f.r(), f.t());

  // Create a tile for the entire image
  DD::Image::Iop *nonConstInput = const_cast<DD::Image::Iop *>(&input);
  return DD::Image::Tile(*nonConstInput, box, channels);
}

/**
 * Overload for pointer input for backward compatibility
 */
inline DD::Image::Tile extractTile(const DD::Image::Iop *input,
                                   DD::Image::ChannelSet channels) {
  if (!input) {
    throw InvalidArgumentException(
        "Null input pointer provided to extractTile");
  }
  return extractTile(*input, channels);
}

/**
 * Convert a tile to NCHW tensor format (batch=1) maintaining original image
 * dimensions Format: [1, channels, height, width]
 */
inline void tileToNCHWTensor(const DD::Image::Tile &tile,
                             std::vector<float> &tensor, int width, int height,
                             int channels) {
  if (width <= 0 || height <= 0 || channels <= 0) {
    throw PreprocessException(
        "Invalid dimensions for tensor conversion: " + std::to_string(width) +
        "x" + std::to_string(height) + " C:" + std::to_string(channels));
  }

  // Resize tensor to hold the data - exact dimensions from the image
  // This preserves original behavior of passing actual dimensions to ONNX
  tensor.resize(1 * channels * height * width);

  // Get the bounds from the tile's box
  const DD::Image::Box &bounds = tile.box();
  int xOffset = bounds.x();
  int yOffset = bounds.y();

  // Extract each channel exactly as is from the image
  for (int c = 0; c < channels; c++) {
    DD::Image::Channel chan = DD::Image::Chan_Red;

    // Map channel index to Nuke channels
    switch (c) {
    case 0:
      chan = DD::Image::Chan_Red;
      break;
    case 1:
      chan = DD::Image::Chan_Green;
      break;
    case 2:
      chan = DD::Image::Chan_Blue;
      break;
    case 3:
      chan = DD::Image::Chan_Alpha;
      break;
    default:
      continue; // Skip unsupported channels
    }

    // Check if this channel exists in the tile
    // Use Nuke's ChannelSet intersection to check for the channel
    if (!(tile.channels() & DD::Image::ChannelSet(chan))) {
      // Fill with zeros if channel is missing
      size_t offset = c * height * width;
      std::fill(tensor.begin() + offset,
                tensor.begin() + offset + height * width, 0.0f);
      continue;
    }

    // Copy data for this channel - preserving exact dimensions
    for (int h = 0; h < height; h++) {
      const float *srcRow = tile[chan][h + yOffset];
      if (!srcRow)
        continue;

      for (int w = 0; w < width; w++) {
        // NCHW format index: n * CHW + c * HW + h * W + w
        // With n = 0 (batch size 1)
        size_t dstIdx = c * height * width + h * width + w;
        tensor[dstIdx] = srcRow[w + xOffset];
      }
    }
  }
}

/**
 * Helper to get the channel component index from a channel name
 * Returns: 0 for red/x, 1 for green/y, 2 for blue/z, 3 for alpha/w, -1 for
 * unknown
 */
inline int getChannelComponentIndex(DD::Image::Channel ch) {
  // Get the name of the channel
  const char *name = DD::Image::getName(ch);
  if (!name)
    return -1;

  // Extract the component part (after the dot)
  std::string fullName(name);
  size_t dotPos = fullName.find_last_of('.');
  if (dotPos == std::string::npos)
    return -1;

  std::string component = fullName.substr(dotPos + 1);

  // Check for standard component names
  if (component == "red" || component == "r" || component == "x")
    return 0;
  if (component == "green" || component == "g" || component == "y")
    return 1;
  if (component == "blue" || component == "b" || component == "z")
    return 2;
  if (component == "alpha" || component == "a" || component == "w")
    return 3;

  return -1;
}

/**
 * Process a Nuke row from tensor data based on the format
 * (single/multi-channel)
 */
inline void
processTensorDataToRow(const std::vector<float> &tensorData, int y, int x,
                       int r, DD::Image::ChannelMask channels,
                       DD::Image::Row &row, const DD::Image::Row &inputRow,
                       int outputWidth, int outputHeight, int channelCount,
                       bool isSingleChannel, bool normalize, float minValue,
                       float maxValue) {
  // Limit the end point to output width
  int endX = std::min(r, outputWidth);

  // Ensure y is within the valid range for the output
  if (y < 0 || y >= outputHeight) {
    // If y is outside the valid range, use the input row data
    row.copy(inputRow, channels, x, r);
    return;
  }

  // Copy from input when needed
  auto copyFromInput = [&](DD::Image::Channel z, int start, int end) {
    float *outPtr = row.writable(z);
    if (!outPtr)
      return;

    const float *inPtr = inputRow[z];
    if (inPtr) {
      for (int i = start; i < end; i++) {
        outPtr[i] = inPtr[i];
      }
    } else {
      for (int i = start; i < end; i++) {
        outPtr[i] = 0.0f;
      }
    }
  };

  // Clear channel to zero
  auto clearChannel = [&](DD::Image::Channel z, int start, int end) {
    float *outPtr = row.writable(z);
    if (!outPtr)
      return;

    for (int i = start; i < end; i++) {
      outPtr[i] = 0.0f;
    }
  };

  // Handle each requested channel
  foreach (z, channels) {
    float *outPtr = row.writable(z);
    if (!outPtr)
      continue;

    // Check if this is a custom named channel (not a standard RGBA channel)
    bool isCustomChannel =
        (z != DD::Image::Chan_Red && z != DD::Image::Chan_Green &&
         z != DD::Image::Chan_Blue && z != DD::Image::Chan_Alpha);

    if (isCustomChannel) {
      // Get the component index based on the channel name
      int componentIndex = getChannelComponentIndex(z);

      if (componentIndex >= 0 && componentIndex < channelCount) {
        // Use the component index (or 0 for single channel mode)
        int channelToUse = isSingleChannel ? 0 : componentIndex;

        for (int i = x; i < endX; i++) {
          outPtr[i] = TensorProcessor::getTensorValue(
              tensorData, i, y, channelToUse, outputWidth, outputHeight,
              isSingleChannel, normalize, minValue, maxValue);
        }
      } else {
        clearChannel(z, x, endX);
      }
    } else if (isSingleChannel) {
      // Single-channel mode (e.g., depth map)
      if (z == DD::Image::Chan_Red) {
        // Put the single output channel in red
        for (int i = x; i < endX; i++) {
          outPtr[i] = TensorProcessor::getTensorValue(
              tensorData, i, y, 0, outputWidth, outputHeight, isSingleChannel,
              normalize, minValue, maxValue);
        }
      } else if (z == DD::Image::Chan_Green || z == DD::Image::Chan_Blue) {
        clearChannel(z, x, endX);
      } else {
        // Preserve alpha from input
        copyFromInput(z, x, endX);
      }
    } else {
      // Multi-channel output mode - map model channels to RGBA
      int outChannel = -1;
      if (z == DD::Image::Chan_Red && channelCount > 0)
        outChannel = 0;
      else if (z == DD::Image::Chan_Green && channelCount > 1)
        outChannel = 1;
      else if (z == DD::Image::Chan_Blue && channelCount > 2)
        outChannel = 2;
      else if (z == DD::Image::Chan_Alpha && channelCount > 3)
        outChannel = 3;

      if (outChannel >= 0 && outChannel < channelCount) {
        for (int i = x; i < endX; i++) {
          outPtr[i] = TensorProcessor::getTensorValue(
              tensorData, i, y, outChannel, outputWidth, outputHeight,
              isSingleChannel, normalize, minValue, maxValue);
        }
      } else if (z == DD::Image::Chan_Alpha && channelCount <= 3) {
        // If model doesn't output alpha, preserve input alpha
        copyFromInput(z, x, endX);
      } else {
        // Preserve other channels from input
        copyFromInput(z, x, endX);
      }
    }
  }
}

/**
 * Display a message by printing it to stderr.
 *
 * @param message The message to display
 * @return Always returns true.
 */
inline bool displayNukeMessage(const std::string &message) {
  std::cerr << message << std::endl;
  return true; // Indicate message was "handled" (printed)
}

/**
 * Information about an output layer for model info display
 */
struct LayerInfo {
  std::string name;               // Layer name
  int numChannels;                // Number of channels in this layer
  DD::Image::ChannelSet channels; // Channels in this layer

  LayerInfo(const std::string &n, int count, const DD::Image::ChannelSet &chans)
      : name(n), numChannels(count), channels(chans) {}
};

/**
 * Generate a formatted model information string
 *
 * @param modelInfoString Base model information from the model manager
 * @param useGPU Whether GPU execution is enabled
 * @param isSingleChannel Whether the model operates in single channel mode
 * @param outputChannelCount Number of output channels
 * @param imgWidth Input image width
 * @param imgHeight Input image height
 * @param outputWidth Output image width
 * @param outputHeight Output image height
 * @param activeInputs Number of active inputs
 * @param modelInputCount Total number of inputs required by the model
 * @param modelInputNames Vector of model input names
 * @param inputConnectionStatus Function to check if an input is connected
 * @param normalize Whether normalization is enabled
 * @param minValue Normalization min value
 * @param maxValue Normalization max value
 * @param getChannelName Function to get channel name from Channel
 * @return Formatted information string
 */
inline std::string buildModelInfoString(
    const std::string &modelInfoString, bool useGPU, bool isSingleChannel,
    int outputChannelCount, int imgWidth, int imgHeight, int outputWidth,
    int outputHeight, int activeInputs, int modelInputCount,
    const std::vector<std::string> &modelInputNames,
    const std::function<bool(int)> &inputConnectionStatus, bool normalize,
    float minValue, float maxValue,
    const std::function<const char *(DD::Image::Channel)> &getChannelName) {
  std::string infoStr = modelInfoString;

  // Add additional information about the node
  std::stringstream additionalInfo;
  additionalInfo << "\nNode Configuration:\n";
  additionalInfo << "-------------------\n";
  additionalInfo << "Execution: " << (useGPU ? "GPU (CUDA)" : "CPU") << "\n";
  additionalInfo << "Processing mode: "
                 << (isSingleChannel ? "Single channel" : "Multi-channel")
                 << "\n";
  additionalInfo << "Output channels: " << outputChannelCount << "\n";
  additionalInfo << "Input dimensions: " << imgWidth << "x" << imgHeight
                 << "\n";
  additionalInfo << "Output dimensions: " << outputWidth << "x" << outputHeight
                 << "\n";

  // Add information about active inputs
  additionalInfo << "\nActive Inputs: " << activeInputs << " of "
                 << modelInputCount << " required\n";
  for (int i = 0; i < activeInputs; i++) {
    additionalInfo << "  Input " << i << ": ";
    if (i < static_cast<int>(modelInputNames.size())) {
      additionalInfo << modelInputNames[i];
    } else {
      additionalInfo << "(unnamed)";
    }

    if (inputConnectionStatus(i)) {
      additionalInfo << " - Connected";
    } else {
      additionalInfo << " - Not connected";
    }
    additionalInfo << "\n";
  }

  if (normalize) {
    additionalInfo << "Normalization: Enabled (min=" << minValue
                   << ", max=" << maxValue << ")\n";
  } else {
    additionalInfo << "Normalization: Disabled\n";
  }

  // Add the additional info to model info
  infoStr += additionalInfo.str();
  return infoStr;
}

} // namespace Utils