////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED

#include "lbann/data_ingestion/readers/metadata.hpp"
#include "lbann/data_ingestion/readers/utils/input_data_type.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class input_distconv_adapter : public data_type_distconv_adapter<TensorDataType>
{
public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  using TensorHost = dc::TensorHost<TensorDataType>;
  using TensorHostShuffler = dc::TensorHostShuffler<TensorDataType>;

  input_distconv_adapter(Layer& layer,
                         data_field_type data_field,
                         const bool shuffle_required);
  virtual ~input_distconv_adapter() = default;

  void setup_layer(size_t workspace_capacity) override;

  TensorHostShuffler& get_shuffler(const TensorHost& src,
                                   const TensorHost& dst);
  void setup_fp_tensors() override;
  std::unique_ptr<TensorDevType> setup_activations_i(int index) const override;
  dc::Shape get_activations_local_shape(int index) const override;
  dc::Shape get_activations_shape(int index) const override;
  void setup_shuffler_buffers(const TensorHost& src, const TensorHost& dst);

  // No bp tensors needed for this layer.
  void setup_prev_error_signals() override {}
  void setup_original_prev_error_signals() override {}
  void setup_error_signals() override {}
  void setup_original_error_signals() override {}
  void setup_bp_tensors() override {}

  bool child_copy_required(size_t output_index) const override;
  bool child_shuffle_required(size_t output_index) const override;

  // Nothing to do here as everything is done in fp_compute_distconv.
  void fp_setup() override {}
  void fp_compute();

private:
  /// @brief Data field accessed by corresponding input layer
  data_field_type m_data_field;

  bool m_is_input_processed;
  std::unique_ptr<TensorHost> m_original_host_tensor;
  std::unique_ptr<TensorHost> m_host_tensor;

  const bool m_shuffle_required;
  std::array<std::unique_ptr<TensorHostShuffler>, 4> m_shufflers;
  std::unique_ptr<TensorDataType> m_shuffler_src_buf;
  size_t m_shuffler_src_buf_size = 0;
  std::unique_ptr<TensorDataType> m_shuffler_dst_buf;
  size_t m_shuffler_dst_buf_size = 0;

  // TODO: Use pinned memory pool
  TensorDataType* m_copy_pinned_buffer = nullptr;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Interface with data reader */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class input_layer : public data_type_layer<TensorDataType>
{
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "input layer only supports DATA_PARALLEL data layout");

public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}
public:
  /// @todo make the map and vector references
  input_layer(lbann_comm* comm, std::string const data_field = "")
    : data_type_layer<TensorDataType>(comm), m_data_field(data_field)
  {

    // Input layers have no parents
    this->m_expected_num_parent_layers = 0;
    this->m_expected_num_child_layers = 1;
  }

  input_layer(const input_layer&) = default;
  input_layer& operator=(const input_layer&) = default;
  input_layer* copy() const override { return new input_layer(*this); }

  std::string get_type() const override { return "input"; }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

  // description get_description() const override {
  //   auto desc = io_layer<TensorDataType>::get_description();
  //   return desc;
  // }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return PROPAGATE_NOTHING; }

  void setup_dims() override;

  void setup_data(size_t max_mini_batch_size) override;

  /** Setup output tensors.
   *  Sets up the effective (global) mini-batch size.
   */
  void fp_setup_outputs() override;

  void fp_compute() override;

  /**
   * Get the dimensions of the underlying data.
   */
  std::vector<El::Int> get_data_dims(const DataReaderMetaData& dr_metadata,
                                     int child_index = 0) const;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

private:
  friend cereal::access;
  input_layer() : input_layer(nullptr) {}

  data_field_type m_data_field;

#ifdef LBANN_HAS_DISTCONV
public:
  /** @brief Extensions for distributed convolutions */
  ///{@
  using distconv_adapter_type =
    input_distconv_adapter<TensorDataType, T_layout, Dev>;
  friend distconv_adapter_type;

protected:
  bool is_distconv_supported() const override
  {
    return Dev == El::Device::CPU && T_layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter() override;
  distconv_adapter_type& get_distconv_adapter() override;
  const distconv_adapter_type& get_distconv_adapter() const override;
  bool keep_original_outputs(int index) const override;
  bool keep_original_gradient_wrt_outputs(int index) const override;
///@}
#endif // LBANN_HAS_DISTCONV
};

LBANN_DEFINE_LAYER_BUILDER(input);

#ifndef LBANN_INPUT_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class input_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_INPUT_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_INPUT_LAYER_HPP_INCLUDED
