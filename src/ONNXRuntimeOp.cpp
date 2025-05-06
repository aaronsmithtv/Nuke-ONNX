#include "ONNXRuntimeOp.h"
#include "DDImage/Format.h"
#include "DDImage/NukeWrapper.h" // For Python integration
#include "DDImage/Tile.h"
#include "DDImage/gl.h"
#include "ErrorHandling.h" // Include error handling
#include "TensorProcessor.h"
#include "Utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>

using namespace DD::Image;

static const char *const CLASS = "ONNXRuntimeOp";
static const char *const HELP = "Runs inference on images using ONNX Runtime";

ONNXRuntimeOp::ONNXRuntimeOp(Node *node)
    : Iop(node), _modelPath(""), _useGPU(false), _normalize(false),
      _isSingleChannel(true), _outputChannelCount(1), _minValue(0.0f),
      _maxValue(1.0f), _formats(), _dimensionsSet(false), _imgWidth(0),
      _imgHeight(0), _imgChannels(0), _outputWidth(0), _outputHeight(0),
      _modelManager(std::make_unique<ONNXModelManager>()),
      _inferenceProcessor(std::make_unique<ONNXInferenceProcessor>()),
      _cacheLock(), _cacheValid(false), _processingDone(false),
      _processedData(), _activeInputs(1) {
  // Initialize the format to use Format::None
  _formats.format(&DD::Image::Format::None);
  _formats.fullSizeFormat(&DD::Image::Format::None);

  // Connect the inference processor to the model manager
  _inferenceProcessor->setModelManager(_modelManager.get());
}

ONNXRuntimeOp::~ONNXRuntimeOp() {
  // Smart pointers will clean up automatically
}

std::string ONNXRuntimeOp::input_longlabel(int input) const {
  char buffer[64];
  const char *label = input_label(input, buffer);

  // Enhanced labeling when model is loaded
  if (_modelManager->isLoaded()) {
    // Get model input names
    const auto &modelInputNames = _modelManager->getInputNames();

    // If we have a name for this input from the model, use it
    if (input < static_cast<int>(modelInputNames.size())) {
      return std::string(label ? label : std::string("Input")) + " (" +
             modelInputNames[input] + ")";
    }
  }

  return label ? std::string(label) : std::string();
}

void ONNXRuntimeOp::updateActiveInputs() {
  if (!_modelManager->isLoaded()) {
    _activeInputs = 1; // Default to 1 input when no model is loaded
    return;
  }

  // Get input count from the model
  int modelInputCount = _modelManager->getInputCount();

  // Set active inputs to the minimum of model inputs and maximum allowed
  _activeInputs = std::min(modelInputCount, maximum_inputs());

  // Ensure at least one input is active
  if (_activeInputs < 1) {
    _activeInputs = 1;
  }
}

void ONNXRuntimeOp::_validate(bool for_real) {
  // First copy input format and bbox
  copy_info();

  if (for_real) {
    // Load model if needed
    if (!_modelManager->isLoaded() && strcmp(_modelPath, "") != 0) {
      try {
        loadModel(); // loadModel now handles its own errors internally
        _dimensionsSet = false;
        _cacheValid = false;
        _processingDone = false;
      } catch (const ONNXPluginError &e) {
        // loadModel might still throw if path is invalid before manager->load
        // is called
        error("%s", e.what());
      }
      // No need to check return value of loadModel anymore
    }

    // Update active inputs based on model
    if (_modelManager->isLoaded()) {
      updateActiveInputs();

      // Check required inputs are connected
      if (_activeInputs > 0 && !input(0)) {
        error("Primary input (input 0) must be connected");
      }

      // Update output dimensions if model is loaded
      if (!_dimensionsSet) {
        updateDimensions(); // updateDimensions handles its own errors
      }
    }
  }

  // Setup output channels
  ChannelSet outputChannels = Mask_RGBA;
  setupOutputChannels(outputChannels);

  set_out_channels(outputChannels);
  info_.turn_on(outputChannels);
}

void ONNXRuntimeOp::_request(int x, int y, int r, int t, ChannelMask channels,
                             int count) {
  // Request the entire image from all active inputs
  // This ensures we have access to complete images for ONNX processing
  Format f = input0().format();

  // We'll always need RGBA channels for processing
  ChannelMask requestChannels = Mask_RGBA;

  // Create the requested output but request RGBA from all inputs
  for (int i = 0; i < _activeInputs; i++) {
    if (input(i)) {
      // Use the format of the first input for consistency
      input(i)->request(f.x(), f.y(), f.r(), f.t(), requestChannels, count);
    }
  }
}

void ONNXRuntimeOp::_open() {
  // Reset cache when the node is opened
  _cacheValid = false;
  _processingDone = false;
  _dimensionsSet = false; // Reset dimensions flag on open
}

void ONNXRuntimeOp::setupOutputChannels(ChannelSet &channels) {
  // Simply ensure the output channels (typically RGBA) are turned on
  info_.turn_on(channels);
}

void ONNXRuntimeOp::engine(int y, int x, int r, ChannelMask channels,
                           Row &row) {
  // When model isn't loaded or operation is aborted, pass through input
  if (!_modelManager->isLoaded() || aborted()) {
    if (input(0)) {
      input0().get(y, x, r, channels, row);
    } else {
      row.erase(channels);
    }
    return;
  }

  // Process the image if needed
  bool processing_succeeded = false;
  { // Scope for lock guard
    Guard guard(_cacheLock);
    if (!_cacheValid) {
      try {
        cacheAndProcessImage();
        processing_succeeded = true;

        // Find min/max values for normalization if successfully processed
        if (_normalize) { // No need to check _processingDone, exception handles
                          // failure
          findMinMaxValues();
        }

      } catch (const ONNXPluginError &e) {
        error("Processing failed: %s", e.what());
        processing_succeeded = false;
      } catch (const std::exception &e) {
        error("Unexpected error during processing: %s", e.what());
        processing_succeeded = false;
      }
      _processingDone =
          processing_succeeded; // Update status based on try-catch
      _cacheValid = true; // Mark cache as checked, even if processing failed
    }
  }

  // If processing failed or y is out of bounds, use input data if available
  if (!_processingDone || y < 0 ||
      y >= _outputHeight) { // Check _processingDone status set by try-catch
    if (input(0)) {
      input0().get(y, x, r, channels, row);
    } else {
      row.erase(channels);
    }
    return;
  }

  // We still need input data for channels we don't process
  Row inputRow(x, r);
  if (input(0)) {
    input0().get(y, x, r, Mask_RGBA, inputRow);
  } else {
    // Clear the row if input is missing
    inputRow.erase(Mask_RGBA);
  }

  Utils::processTensorDataToRow(_processedData, y, x, r, channels, row,
                                inputRow, _outputWidth, _outputHeight,
                                _outputChannelCount, _isSingleChannel,
                                _normalize, _minValue, _maxValue);
}

void ONNXRuntimeOp::cacheAndProcessImage() {
  if (!_modelManager->isLoaded()) {
    throw ConfigurationException(
        "Attempted to process image but no model is loaded");
  }

  // No outer try-catch needed here, exceptions will propagate to engine()

  // Get input image dimensions from first input
  if (!input(0)) {
    throw ConfigurationException("Primary input (input 0) is not connected");
  }
  Format f = input0().format();
  _imgWidth = f.width();
  _imgHeight = f.height();
  _imgChannels = 3; // Assumed RGB for preprocessing

  // Validate dimensions
  if (_imgWidth <= 0 || _imgHeight <= 0) {
    throw ConfigurationException("Invalid input dimensions from Nuke format: " +
                                 std::to_string(_imgWidth) + "x" +
                                 std::to_string(_imgHeight));
  }

  _inferenceProcessor->setInputDimensions(_imgWidth, _imgHeight, _imgChannels);
  _inferenceProcessor->prepareInputs(_activeInputs);

  for (int i = 0; i < _activeInputs; i++) {
    const Iop *currentInput = input(i);
    // Skip disconnected inputs (but input 0 check already happened)
    if (currentInput == nullptr) {
      if (i == 0) { // Should not happen due to earlier check, but defensive
        throw ConfigurationException(
            "Primary input (input 0) became disconnected unexpectedly");
      }
      // Mark corresponding tensor as invalid if an optional input is
      // disconnected
      if (i < static_cast<int>(_inferenceProcessor->getInputTensors().size())) {
        _inferenceProcessor->getInputTensor(i).valid = false;
      }
      continue; // Skip preprocessing for disconnected optional inputs
    }

    // Process this input image (throws on error)
    std::vector<float> inputTensor;
    preprocessImage(currentInput, inputTensor);

    _inferenceProcessor->setInputTensorData(i, inputTensor);
  }

  _processedData.clear();
  _inferenceProcessor->runInference(_processedData);

  if (_processedData.empty()) {
    // Although runInference should throw if the ONNX result is invalid,
    // we add a check here for safety.
    throw InferenceException(
        "Inference completed but resulted in empty output data");
  }

  _inferenceProcessor->getOutputDimensions(_outputWidth, _outputHeight,
                                           _outputChannelCount);
  _isSingleChannel = _inferenceProcessor->isSingleChannelOutput();
}

void ONNXRuntimeOp::preprocessImage(const Iop *input,
                                    std::vector<float> &inputTensor) {
  if (!input) {
    throw InvalidArgumentException(
        "Null input pointer passed to preprocessImage");
  }

  try {
    // Extract image and convert to NCHW tensor format (throws on error)
    Tile tile = Utils::extractTile(*input, Mask_RGB);
    Utils::tileToNCHWTensor(tile, inputTensor, _imgWidth, _imgHeight, 3);
  } catch (const ONNXPluginError &e) {
    // Rethrow specific plugin errors
    throw;
  } catch (const std::exception &e) {
    // Wrap other exceptions as PreprocessException
    throw PreprocessException(
        std::string("Error during image preprocessing: ") + e.what());
  }
}

void ONNXRuntimeOp::findMinMaxValues() {
  if (_processedData.empty()) {
    _minValue = 0.0f;
    _maxValue = 1.0f;
    return;
  }

  // Use TensorProcessor to find min/max values
  if (_isSingleChannel) {
    // For single channel, just find min/max of the whole data
    TensorProcessor::findMinMax(_processedData, _minValue, _maxValue);
  } else {
    // For multi-channel, use multichannel min/max function
    TensorProcessor::findMinMaxMultiChannel(_processedData, _minValue,
                                            _maxValue, _outputChannelCount,
                                            _outputWidth, _outputHeight);
  }
}

void ONNXRuntimeOp::loadModel() {
  _processedData.clear();

  // Model path validation
  if (_modelPath == nullptr || strlen(_modelPath) == 0) {
    throw ConfigurationException("Model path is empty");
  }

  // Load the model with the current settings
  try {
    _modelManager->load(_modelPath,
                        _useGPU); // Throws ModelLoadException on failure

    // Check if model has at least one input
    if (_modelManager->getInputCount() <= 0) {
      _modelManager->unload(); // Unload the invalid model
      throw ModelLoadException("Invalid model: No inputs found");
    }

    // Extract information about channels
    int channels;
    if (_modelManager->getOutputDimensions(_outputWidth, _outputHeight,
                                           channels)) {
      _outputChannelCount = channels;
      _isSingleChannel = (channels == 1);
    } else {
      warning("Could not retrieve fixed output dimensions from model. Output "
              "size might adapt to input.");
      // Assume output dimensions match input initially if not specified
      _outputWidth = _imgWidth > 0 ? _imgWidth : 0;
      _outputHeight = _imgHeight > 0 ? _imgHeight : 0;
      _outputChannelCount = 1; // Default guess
      _isSingleChannel = true;
    }

    updateActiveInputs();

    // printf("ONNX Model Inputs: %d (active: %d)\n",
    //     _modelManager->getInputCount(), _activeInputs);

    _cacheValid = false;
    _processingDone = false;
  } catch (const ONNXPluginError &e) {
    error("Failed to load model: %s",
          e.what()); // Use error() for user feedback
    throw; // Re-throw to signal failure to caller (e.g., knob_changed)
  } catch (const std::exception &e) {
    error("Unexpected error loading model: %s", e.what());
    throw ModelLoadException(std::string("Unexpected: ") +
                             e.what()); // Wrap and re-throw
  }
}

void ONNXRuntimeOp::updateDimensions() {
  // Only update if we have valid output dimensions
  if (_outputWidth <= 0 || _outputHeight <= 0) {
    warning("Invalid output dimensions: %dx%d", _outputWidth, _outputHeight);
    return;
  }

  // Only resize if dimensions differ from input
  if (_outputWidth == _imgWidth && _outputHeight == _imgHeight) {
    _dimensionsSet = true;
    return;
  }

  // Ensure input dimensions are valid
  if (_imgWidth <= 0 || _imgHeight <= 0) {
    if (!input(0)) {
      warning("Cannot determine input dimensions: no input connected");
      return;
    }

    Format inputFormat = input0().format();
    _imgWidth = std::max(1, inputFormat.width());
    _imgHeight = std::max(1, inputFormat.height());

    if (_imgWidth <= 1 || _imgHeight <= 1) {
      warning("Input dimensions too small: %dx%d", _imgWidth, _imgHeight);
      return;
    }
  }

  try {
    // Create new format with output dimensions
    Format newFormat(_outputWidth, _outputHeight, 0,
                     0,             // x, y = 0 for Nuke formats
                     _outputWidth,  // r = width
                     _outputHeight, // t = height
                     input0().format().pixel_aspect());

    // Set format and bounding box
    _formats.formatStorage() = newFormat;
    _formats.format(&_formats.formatStorage());
    _formats.fullSizeFormat(&_formats.formatStorage());
    info_.setFormats(_formats);
    info_.set(0, 0, _outputWidth, _outputHeight);

    // Invalidate cache since tensor needs reshaping
    _cacheValid = false;
    _processingDone = false;

    // Debug output
    // printf("ONNX Resizing: %dx%d -> %dx%d\n", _imgWidth, _imgHeight,
    // _outputWidth, _outputHeight);

    _dimensionsSet = true;
  } catch (const std::exception &e) {
    error("Error updating dimensions: %s", e.what());
    _dimensionsSet = false;
  }
}

void ONNXRuntimeOp::displayModelInfo() {
  // Build the info string using the utility function
  std::string infoStr = Utils::buildModelInfoString(
      _modelManager->getInfoString(), _useGPU, _isSingleChannel,
      _outputChannelCount, _imgWidth, _imgHeight, _outputWidth, _outputHeight,
      _activeInputs, _modelManager->getInputCount(),
      _modelManager->getInputNames(),
      [this](int idx) { return input(idx) != nullptr; }, _normalize, _minValue,
      _maxValue, &DD::Image::getName);

  // Display the message using the simplified utility function (prints to
  // stderr)
  Utils::displayNukeMessage(infoStr);
}

void ONNXRuntimeOp::knobs(Knob_Callback f) {
  File_knob(f, &_modelPath, "model_path", "Model Path");
  Tooltip(f, "Path to ONNX model file");

  // Bool_knob(f, &_useGPU, "use_gpu", "Use GPU");
  // Tooltip(f, "Use GPU for inference if available");

  Bool_knob(f, &_normalize, "normalize", "Normalize Output");
  Tooltip(f, "Normalize output values to range 0-1");

  Divider(f);

  Button(f, "reload_model", "Reload Model");
  Tooltip(f, "Reload the model from disk");

  Button(f, "show_model_info", "Print Model Info");
  Tooltip(f, "Display detailed information about the loaded model");
}

int ONNXRuntimeOp::knob_changed(Knob *k) {
  if (k->name() == "model_path") {
    loadModel();
    _dimensionsSet = false;  // Reset dimensions flag on model path change
    _cacheValid = false;     // Invalidate cache
    _processingDone = false; // Reset processing state
    asapUpdate();            // Request immediate UI refresh
    return 1;
  } else if (k->name() == "reload_model") {
    loadModel();
    _dimensionsSet = false;  // Reset dimensions flag on model reload
    _cacheValid = false;     // Invalidate cache
    _processingDone = false; // Reset processing state
    asapUpdate();            // Request immediate UI refresh
    return 1;
  } else if (k->name() == "use_gpu") {
    // Reload model with new setting
    loadModel();
    _dimensionsSet = false;  // Reset dimensions flag on GPU setting change
    _cacheValid = false;     // Invalidate cache
    _processingDone = false; // Reset processing state
    return 1;
  } else if (k->name() == "normalize") {
    // Invalidate cache to reprocess with normalization
    _cacheValid = false;
    return 1;
  } else if (k->name() == "show_model_info") {
    // Display model information
    displayModelInfo();
    return 1;
  }

  return 0;
}

const char *ONNXRuntimeOp::Class() const { return CLASS; }

const char *ONNXRuntimeOp::node_help() const { return HELP; }

// Build function for creating an instance
static Iop *build(Node *node) {
  // Use smart pointer during construction then release ownership when returning
  // This ensures memory is managed correctly even if construction fails
  auto op = std::make_unique<ONNXRuntimeOp>(node);
  return op.release();
}

// Registration of the plugin
const Iop::Description ONNXRuntimeOp::description(CLASS, "Image/ONNX Runtime",
                                                  build);