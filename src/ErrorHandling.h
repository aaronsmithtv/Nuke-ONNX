#pragma once

#include <stdexcept>
#include <string>

// Base class for ONNX Nuke Plugin exceptions
class ONNXPluginError : public std::runtime_error {
public:
  explicit ONNXPluginError(const std::string &message)
      : std::runtime_error(message) {}
  explicit ONNXPluginError(const char *message) : std::runtime_error(message) {}
};

// Specific exception types
class ModelLoadException : public ONNXPluginError {
public:
  explicit ModelLoadException(const std::string &message)
      : ONNXPluginError("Model Load Error: " + message) {}
  explicit ModelLoadException(const char *message)
      : ONNXPluginError(std::string("Model Load Error: ") + message) {}
};

class InferenceException : public ONNXPluginError {
public:
  explicit InferenceException(const std::string &message)
      : ONNXPluginError("Inference Error: " + message) {}
  explicit InferenceException(const char *message)
      : ONNXPluginError(std::string("Inference Error: ") + message) {}
};

class ConfigurationException : public ONNXPluginError {
public:
  explicit ConfigurationException(const std::string &message)
      : ONNXPluginError("Configuration Error: " + message) {}
  explicit ConfigurationException(const char *message)
      : ONNXPluginError(std::string("Configuration Error: ") + message) {}
};

class PreprocessException : public ONNXPluginError {
public:
  explicit PreprocessException(const std::string &message)
      : ONNXPluginError("Preprocessing Error: " + message) {}
  explicit PreprocessException(const char *message)
      : ONNXPluginError(std::string("Preprocessing Error: ") + message) {}
};

class InvalidArgumentException : public ONNXPluginError {
public:
  explicit InvalidArgumentException(const std::string &message)
      : ONNXPluginError("Invalid Argument: " + message) {}
  explicit InvalidArgumentException(const char *message)
      : ONNXPluginError(std::string("Invalid Argument: ") + message) {}
};