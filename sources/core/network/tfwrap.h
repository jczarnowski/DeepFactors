/*
 * This file is part of DeepFactors.
 *
 * Copyright (C) 2020 Imperial College London
 * 
 * The use of the code within this file and all code within files that make up
 * the software that is DeepFactors is permitted for non-commercial purposes
 * only.  The full terms and conditions that apply to the code within this file
 * are detailed within the LICENSE file and at
 * <https://www.imperial.ac.uk/dyson-robotics-lab/projects/deepfactors/deepfactors-license>
 * unless explicitly stated. By downloading this file you agree to comply with
 * these terms.
 *
 * If you wish to use any of this code for commercial purposes then please
 * email researchcontracts.engineering@imperial.ac.uk.
 *
 */
#ifndef DF_TFWRAP_H_
#define DF_TFWRAP_H_

#include <memory>
#include <vector>
#include <cstring>
#include <cassert>
#include <fstream>

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/c_api_experimental.h>

namespace tf
{

template <typename T>
struct tf_type;

#define DEFINE_TF_TYPE(T1, T2) \
  template <> \
  struct tf_type<T1> { static constexpr TF_DataType value = T2; };

DEFINE_TF_TYPE(float, TF_FLOAT)
DEFINE_TF_TYPE(double, TF_DOUBLE)

class Tensor
{
public:
  typedef std::vector<std::int64_t> TensorDims;

  Tensor() : bytes_(0) {}

  /**
   * @brief Allocates data
   */
  Tensor(TF_DataType type, const TensorDims& dims, std::size_t bytes)
    : type_(type), dims_(dims), bytes_(bytes)
  {
    tensor_ = std::shared_ptr<TF_Tensor>(TF_AllocateTensor(type, dims_.data(),
                                                           static_cast<int>(dims_.size()),
                                                           bytes_), TF_DeleteTensor);
  }

  /**
   * @brief Takes ownership of existing tensor
   */
  Tensor(TF_Tensor* ptr)
    : type_(TF_TensorType(ptr)), bytes_(TF_TensorByteSize(ptr))
  {
    tensor_ = std::shared_ptr<TF_Tensor>(ptr, TF_DeleteTensor);
    dims_ = TensorDims(TF_NumDims(ptr));
    for (std::size_t i = 0; i < dims_.size(); ++i)
    {
      dims_[i] = TF_Dim(ptr, i);
    }
  }

  void copyFrom(const void* data)
  {
    std::memcpy(data_ptr(), data, bytes_);
  }

  template <typename T>
  static Tensor fromData(void* data, const TensorDims& dims)
  {
    assert(dims.size());
    std::size_t size = 1;
    for (auto& d : dims)
      size *= d;
    Tensor tensor(tf_type<T>::value, dims, size * sizeof(T));
    tensor.copyFrom(data);
    return tensor;
  }

  template <typename T>
  static Tensor fromDims(const TensorDims& dims)
  {
    assert(dims.size());
    std::size_t size = 1;
    for (auto& d : dims)
      size *= d;
    return Tensor(tf_type<T>::value, dims, size * sizeof(T));
  }

  TF_Tensor* tensor() { return tensor_.get(); }
  std::size_t bytes() { return bytes_; }
  void* data_ptr() { return TF_TensorData(tensor_.get()); }

private:
  TF_DataType type_;
  TensorDims dims_;
  std::size_t bytes_;
  std::shared_ptr<TF_Tensor> tensor_;
};

class Status
{
public:
  Status() : ptr_(TF_NewStatus(), TF_DeleteStatus) {}
  bool ok() { return TF_GetCode(ptr()) == TF_OK; }
  TF_Status* ptr() { return ptr_.get(); }

  void failOnError(std::string msg)
  {
    if (!ok())
      throw std::runtime_error(msg + ": " + std::string(TF_Message(ptr_.get())));
  }

private:
  std::shared_ptr<TF_Status> ptr_;
};


static TF_Buffer* ReadBufferFromFile(const char* file) {
  std::ifstream f(file, std::ios::binary);
  if (f.fail() || !f.is_open())
    return nullptr;
  if (f.seekg(0, std::ios::end).fail())
    return nullptr;
  auto fsize = f.tellg();
  if (f.seekg(0, std::ios::beg).fail())
    return nullptr;
  if (fsize <= 0)
    return nullptr;

  auto data = static_cast<char*>(std::malloc(fsize));
  if (f.read(data, fsize).fail())
    return nullptr;
  f.close();

  auto buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  return buf;
}

class Graph
{
public:
  Graph(std::string graphdef_path) : path_(graphdef_path)
  {
    graph_ = std::shared_ptr<TF_Graph>(TF_NewGraph(), TF_DeleteGraph);
    importGraphDef(graphdef_path);
  }

  void importGraphDef(std::string path)
  {
    auto buffer = ReadBufferFromFile(path.c_str());
    if (buffer == nullptr)
      throw std::runtime_error("Can't read buffer from file: " + path);

    Status status;
    auto opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph_.get(), buffer, opts, status.ptr());
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);

    if (!status.ok())
      throw std::runtime_error("Failed importing graph from: " + path);
  }

  TF_Output getOpByName(std::string op_name)
  {
    TF_Output op = TF_Output{TF_GraphOperationByName(graph_.get(), op_name.c_str()), 0};
    if (op.oper == nullptr)
      throw std::runtime_error("Can't find op in graph: " + op_name);
    return op;
  }

  TF_Graph* ptr() { return graph_.get(); }
  std::string path() { return path_; }

private:
  std::shared_ptr<TF_Graph> graph_;
  std::string path_;
};

class SessionOptions
{
public:
  SessionOptions() : opts(TF_NewSessionOptions(), TF_DeleteSessionOptions) {}
  TF_SessionOptions* ptr() { return opts.get(); }

  void enable_xla_compilation(bool val)
  {
    enable_xla_compilation_ = val;
    setOptions();
  }

  void gpu_memory_allow_growth(bool val)
  {
    gpu_memory_allow_growth_ = val;
    setOptions();
  }

  void num_cpu_devices(int n)
  {
    num_cpu_devices_ = n;
    setOptions();
  }

  void setOptions()
  {
    TF_Buffer* buf = TF_CreateConfig(enable_xla_compilation_, gpu_memory_allow_growth_, num_cpu_devices_);
    TF_SetConfig(opts.get(), buf->data, buf->length, status.ptr());
    status.failOnError("Failed to set option");
    TF_DeleteBuffer(buf);
  }

private:
  Status status;
  std::shared_ptr<TF_SessionOptions> opts;
  bool enable_xla_compilation_ = false;
  bool gpu_memory_allow_growth_ = false;
  int num_cpu_devices_ = 1;
};

class Session
{
public:
  Session(Graph& graph, SessionOptions& opts)
  {
    session = std::shared_ptr<TF_Session>(TF_NewSession(graph.ptr(), opts.ptr(), status.ptr()), Session::closeAndDelete);
    status.failOnError("Failed to create Session");
  }

  static void closeAndDelete(TF_Session* sess)
  {
    Status status;
    TF_CloseSession(sess, status.ptr());
    status.failOnError("Failed to close Session");
    TF_DeleteSession(sess, status.ptr());
    status.failOnError("Failed to delete Session");
  }

  TF_Session* ptr() { return session.get(); }

private:
  Status status;
  std::shared_ptr<TF_Session> session;
};

class GraphEvaluator
{
public:
  typedef std::vector<std::pair<std::string, tf::Tensor>> TensorDict;

  GraphEvaluator(std::string graph_path, tf::SessionOptions& opts)
    : graph_(graph_path), session_(graph_, opts) {}

  std::vector<tf::Tensor> Run(std::vector<std::pair<std::string, tf::Tensor>>& feed_dict,
                              std::vector<std::string>& fetches)
  {
    std::vector<TF_Output> input_ops;
    std::vector<TF_Tensor*> input_tensors;
    for (auto& kv : feed_dict)
    {
      input_ops.push_back(graph_.getOpByName(kv.first));
      input_tensors.push_back(kv.second.tensor());
    }

    std::vector<TF_Output> output_ops;
    std::vector<TF_Tensor*> output_tensors(fetches.size());
    for (auto& name : fetches)
      output_ops.push_back(graph_.getOpByName(name));

    TF_SessionRun(session_.ptr(),
                  nullptr, // Run options.
                  &input_ops[0], &input_tensors[0], feed_dict.size(), // Input tensors, input tensor values, number of inputs.
                  &output_ops[0], &output_tensors[0], fetches.size(), // Output tensors, output tensor values, number of outputs.
                  nullptr, 0, // Target operations, number of targets.
                  nullptr, // Run metadata.
                  status_.ptr());

    status_.failOnError("Running session failed");

    std::vector<tf::Tensor> output;
    for (auto& tensor : output_tensors)
      output.push_back(tf::Tensor(tensor));
    return output;
  }

private:
  tf::Status status_;
  tf::Graph graph_;
  tf::Session session_;
};

} // namespace tf

#endif // DF_TFWRAP_H_
