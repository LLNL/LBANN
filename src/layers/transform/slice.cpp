////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#define LBANN_SLICE_LAYER_INSTANTIATE
#include "lbann/layers/transform/slice.hpp"

namespace lbann {

namespace {

using dim4 = std::array<size_t, 4>;

/** @brief Concatenate 4D tensors. */
template <typename T>
void concat4d(
  size_t concat_dim,
  const std::vector<const T*>& input_buffer_list,
  const std::vector<dim4>& input_dims_list,
  const std::vector<dim4>& input_strides_list,
  T* output_buffer,
  const dim4& output_strides) {

  // Compute offset corresponding to each input tensor
  std::vector<size_t> output_offset_list;
  output_offset_list.push_back(0);
  for (const auto& input_dims : input_dims_list) {
    auto offset = output_offset_list.back();
    offset += input_dims[concat_dim] * output_strides[concat_dim];
    output_offset_list.push_back(offset);
  }

  // Iterate through input tensors
  for (size_t j=0; j<input_buffer_list.size(); ++j) {
    const auto& input_buffer = input_buffer_list[j];
    const auto& input_dims = input_dims_list[j];
    const auto& input_strides = input_strides_list[j];
    const auto& output_offset = output_offset_list[j];

    // Copy input tensor to corresponding position in output tensor
    LBANN_OMP_PARALLEL_FOR_COLLAPSE4
    for (size_t i0=0; i0<input_dims[0]; ++i0) {
      for (size_t i1=0; i1<input_dims[1]; ++i1) {
        for (size_t i2=0; i2<input_dims[2]; ++i2) {
          for (size_t i3=0; i3<input_dims[3]; ++i3) {
            const auto& x = input_buffer[i0 * input_strides[0]
                                         + i1 * input_strides[1]
                                         + i2 * input_strides[2]
                                         + i3 * input_strides[3]];
            auto& y = output_buffer[output_offset
                                    + i0 * output_strides[0]
                                    + i1 * output_strides[1]
                                    + i2 * output_strides[2]
                                    + i3 * output_strides[3]];
            y = x;
          }
        }
      }
    }

  }

}

/** @brief Slice 4D tensors. */
template <typename T>
void slice4d(
  size_t slice_dim,
  const T* input_buffer,
  const dim4& input_strides,
  const std::vector<T*>& output_buffer_list,
  const std::vector<dim4>& output_dims_list,
  const std::vector<dim4>& output_strides_list) {

  // Compute offset corresponding to each output tensor
  std::vector<size_t> input_offset_list;
  input_offset_list.push_back(0);
  for (const auto& output_dims : output_dims_list) {
    auto offset = input_offset_list.back();
    offset += output_dims[slice_dim] * input_strides[slice_dim];
    input_offset_list.push_back(offset);
  }

  // Iterate through output tensors
  for (size_t j=0; j<output_buffer_list.size(); ++j) {
    auto&& output_buffer = output_buffer_list[j];
    const auto& output_dims = output_dims_list[j];
    const auto& output_strides = output_strides_list[j];
    const auto& input_offset = input_offset_list[j];

    // Copy output tensor to corresponding position in input tensor
    LBANN_OMP_PARALLEL_FOR_COLLAPSE4
    for (size_t i0=0; i0<output_dims[0]; ++i0) {
      for (size_t i1=0; i1<output_dims[1]; ++i1) {
        for (size_t i2=0; i2<output_dims[2]; ++i2) {
          for (size_t i3=0; i3<output_dims[3]; ++i3) {
            auto& x = input_buffer[input_offset
                                   + i0 * input_strides[0]
                                   + i1 * input_strides[1]
                                   + i2 * input_strides[2]
                                   + i3 * input_strides[3]];
            auto& y = output_buffer[i0 * output_strides[0]
                                    + i1 * output_strides[1]
                                    + i2 * output_strides[2]
                                    + i3 * output_strides[3]];
            y = x;
          }
        }
      }
    }

  }

}

} // namespace <anon>

template <typename TensorDataType>
void fp_compute_impl(
  slice_layer<TensorDataType,data_layout::MODEL_PARALLEL,El::Device::CPU>& l) {
  // Tensor views have already been setup in fp_setup_outputs
}

#if 0 /// @todo Restore
template <typename TensorDataType>
void bp_compute_impl(
  slice_layer<TensorDataType,data_layout::MODEL_PARALLEL,El::Device::CPU>& l) {

  // Stack Elemental matrices on top of each other
  // Note: Assume each mini-batch sample is flat.
  auto& input_grad = l.get_error_signals();
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> input_grad_v(
    input_grad.Construct(input_grad.Grid(), input_grad.Root()));
  size_t offset = 0;
  for (size_t j=0; j<static_cast<size_t>(l.get_num_children()); ++j) {
    const auto& output_grad = l.get_prev_error_signals(j);
    El::View(*input_grad_v, input_grad,
             El::IR(offset, offset+output_grad.Height()), El::ALL);
    El::Copy(output_grad, *input_grad_v);
    offset += output_grad.Height();
  }

}
#endif // 0

template <typename TensorDataType>
void fp_compute_impl(
  slice_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::CPU>& l) {

  // Just make a view if there is one output
  if (l.get_num_children() == 1) {
    El::LockedView(l.get_activations(0), l.get_prev_activations());
    return;
  }

  // Check that number of dimensions is valid
  /// @todo Support tensors with arbitrary number of dimensions
  const auto& input_dims = l.get_input_dims();
  const size_t num_dims = input_dims.size();
  if (num_dims > 3) {
    LBANN_ERROR(l.get_type()," layer \"",l.get_name(),"\" ",
                "is operating on ",num_dims,"-D tensors, ",
                "but only 3-D tensors are currently supported");
  }

  // Get dimensions and strides for each output gradient tensor
  std::vector<TensorDataType*> output_buffer_list;
  std::vector<dim4> output_dims_list, output_strides_list;
  const size_t num_outputs = l.get_num_children();
  for (size_t j=0; j<num_outputs; ++j) {
    auto& output = l.get_activations(j);
    const auto& output_dims = l.get_output_dims(j);

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(output_dims.rbegin(), output_dims.rend());
    std::vector<size_t> rstrides(output_dims.size(), 1);
    for (size_t d=1; d<output_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(output.LocalWidth());
    rstrides.push_back(output.LDim());

    // Pad tensor dimensions to 4D
    rdims.resize(4, 1);
    rstrides.resize(4, rstrides.back());

    output_buffer_list.push_back(output.Buffer());
    output_dims_list.push_back({rdims[3], rdims[2], rdims[1], rdims[0]});
    output_strides_list.push_back(
      {rstrides[3], rstrides[2], rstrides[1], rstrides[0]});
  }

  // Get strides for input tensor
  const auto& input = l.get_prev_activations();
  dim4 input_strides;
  {

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(input_dims.rbegin(), input_dims.rend());
    std::vector<size_t> rstrides(input_dims.size(), 1);
    for (size_t d=1; d<input_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(input.LocalWidth());
    rstrides.push_back(input.LDim());

    // Pad tensor dimensions to 4D
    rdims.resize(4, 1);
    rstrides.resize(4, rstrides.back());

    input_strides = {rstrides[3], rstrides[2], rstrides[1], rstrides[0]};
  }

  // Slice 4D tensor
  size_t input_offset = std::accumulate(input_dims.begin(),
                                        input_dims.end(),
                                        1, std::multiplies<int>());
  input_offset /= input_dims[l.m_slice_dim];
  input_offset *= l.m_slice_points.front();
  slice4d<TensorDataType>(
    l.m_slice_dim + (4-num_dims),
    input.LockedBuffer() + input_offset,
    input_strides,
    output_buffer_list,
    output_dims_list,
    output_strides_list);

}

#if 0 /// @todo Restore
template <typename TensorDataType>
void bp_compute_impl(
  slice_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::CPU>& l) {

  // Check that number of dimensions is valid
  /// @todo Support tensors with arbitrary number of dimensions
  const size_t num_dims = l.get_output_dims().size();
  if (num_dims > 3) {
    LBANN_ERROR(l.get_type()," layer \"",l.get_name(),"\" ",
                "is operating on ",num_dims,"-D tensors, ",
                "but only 3-D tensors are currently supported");
  }

  // Get dimensions and strides for each input tensor
  std::vector<const TensorDataType*> input_buffer_list;
  std::vector<dim4> input_dims_list, input_strides_list;
  for (size_t j=0; j<static_cast<size_t>(l.get_num_parents()); ++j) {
    const auto& input = l.get_prev_activations(j);
    const auto& input_dims = l.get_input_dims(j);

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(input_dims.rbegin(), input_dims.rend());
    std::vector<size_t> rstrides(input_dims.size(), 1);
    for (size_t d=1; d<input_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(input.LocalWidth());
    rstrides.push_back(input.LDim());

    // Pad tensor dimensions to 4D
    rdims.resize(4, 1);
    rstrides.resize(4, rstrides.back());

    input_buffer_list.push_back(input.LockedBuffer());
    input_dims_list.push_back({rdims[3], rdims[2], rdims[1], rdims[0]});
    input_strides_list.push_back(
      {rstrides[3], rstrides[2], rstrides[1], rstrides[0]});
  }

  // Get strides for output tensor
  dim4 output_strides;
  auto& output = l.get_activations();
  {
    const auto& output_dims = l.get_output_dims();

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(output_dims.rbegin(), output_dims.rend());
    std::vector<size_t> rstrides(output_dims.size(), 1);
    for (size_t d=1; d<output_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(output.LocalWidth());
    rstrides.push_back(output.LDim());

    // Pad tensor dimensions to 4D
    rdims.resize(4, 1);
    rstrides.resize(4, rstrides.back());

    output_strides = {rstrides[3], rstrides[2], rstrides[1], rstrides[0]};
  }

  // Concatenate 4D tensors
  concat4d<TensorDataType>(
    l.m_slice_dim + (4-num_dims),
    input_buffer_list,
    input_dims_list,
    input_strides_list,
    output.Buffer(),
    output_strides);

}
#endif // 0

// Explicit instantiation
#define PROTO(T)                                        \
  template class slice_layer<                           \
    T, data_layout::DATA_PARALLEL, El::Device::CPU>;    \
  template class slice_layer<                           \
    T, data_layout::MODEL_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
