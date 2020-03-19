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

#define LBANN_LEAKY_RELU_LAYER_INSTANTIATE
#include "lbann/layers/activations/leaky_relu.hpp"

namespace lbann {

namespace {

/** CUDA kernel for forward prop computation. */
template <typename TensorDataType>
__global__ void fp_kernel(TensorDataType negative_slope,
                          El::Int height,
                          El::Int width,
                          const TensorDataType* __restrict__ input,
                          El::Int input_ldim,
                          TensorDataType* __restrict__ output,
                          El::Int output_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_ldim];
    auto& y = output[row + col * output_ldim];
    y = (x > TensorDataType(0.)) ? x : negative_slope * x;
  }
}

/** CUDA kernel for backprop computation. */
template <typename TensorDataType>
__global__ void bp_kernel(TensorDataType negative_slope,
                          El::Int height,
                          El::Int width,
                          const TensorDataType* __restrict__ input,
                          El::Int input_ldim,
                          const TensorDataType* __restrict__ gradient_wrt_output,
                          El::Int gradient_wrt_output_ldim,
                          TensorDataType* __restrict__ gradient_wrt_input,
                          El::Int gradient_wrt_input_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_ldim];
    const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
    auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
    dx = (x > TensorDataType(0.)) ? dy : dy * negative_slope;
  }
}

/** Local forward prop computation. */
template <typename TensorDataType>
void local_fp(TensorDataType negative_slope,
              const El::AbstractMatrix<TensorDataType>& input,
              El::AbstractMatrix<TensorDataType>& output) {

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = input.Height();
  const El::Int width = input.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    fp_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
      negative_slope, height, width,
      input.LockedBuffer(), input.LDim(),
      output.Buffer(), output.LDim());
  }

}

/** Local backprop computation. */
template <typename TensorDataType>
void local_bp(TensorDataType negative_slope,
              const El::AbstractMatrix<TensorDataType>& input,
              const El::AbstractMatrix<TensorDataType>& gradient_wrt_output,
              El::AbstractMatrix<TensorDataType>& gradient_wrt_input) {

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = input.Height();
  const El::Int width = input.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    bp_kernel<<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
      negative_slope, height, width,
      input.LockedBuffer(), input.LDim(),
      gradient_wrt_output.LockedBuffer(), gradient_wrt_output.LDim(),
      gradient_wrt_input.Buffer(), gradient_wrt_input.LDim());
  }

}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
void leaky_relu_layer<TensorDataType, Layout, Device>::fp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    assert_always(Layout == data_layout::DATA_PARALLEL);
    fp_compute_distconv();
    if (!this->early_terminate_last_iteration()) {
      return;
    }
    // fall through the normal code path to obtain reference results
  }
#endif
  local_fp(this->m_negative_slope,
           this->get_local_prev_activations(),
           this->get_local_activations());
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled() && this->early_terminate_last_iteration() &&
      this->dc().keep_original()) {
    this->dc().dump_original_activations();
  }
#endif // LBANN_HAS_DISTCONV
}
template <typename TensorDataType, data_layout Layout, El::Device Device>
void leaky_relu_layer<TensorDataType, Layout, Device>::bp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    assert_always(Layout == data_layout::DATA_PARALLEL);
    bp_compute_distconv();
    if (!this->early_terminate_last_iteration()) {
      return;
    }
  }
#endif // LBANN_HAS_DISTCONV
  local_bp(this->m_negative_slope,
           this->get_local_prev_activations(),
           this->get_local_prev_error_signals(),
           this->get_local_error_signals());
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled() && this->early_terminate_last_iteration() &&
      this->dc().keep_original()) {
    this->dc().dump_original_error_signals();
  }
#endif // LBANN_HAS_DISTCONV
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
void leaky_relu_layer<TensorDataType, Layout, Device>::init_distribution(
    std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
    std::map<dc::Dist*, std::set<dc::Dist*>> &equivalents,
    std::set<dc::Dist*> &updated,
    std::set<dc::Dist*> &invariants)  {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  data_type_layer<TensorDataType>::init_distribution(
      dists, equivalents, updated, invariants);
  if (!this->distconv_enabled()) return;
  auto &layer_dists = dists[this];
  // x == y
  equivalents[&layer_dists[0]].insert(&layer_dists[1]);
  equivalents[&layer_dists[1]].insert(&layer_dists[0]);
  // x == dx
  equivalents[&layer_dists[0]].insert(&layer_dists[2]);
  equivalents[&layer_dists[2]].insert(&layer_dists[0]);
  // dx == dy
  equivalents[&layer_dists[2]].insert(&layer_dists[3]);
  equivalents[&layer_dists[3]].insert(&layer_dists[2]);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void leaky_relu_layer<TensorDataType, Layout, Device>::
fp_compute_distconv() {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  assert_always(this->distconv_enabled());
  dc().m_leaky_relu->forward(dc().get_prev_activations(), m_negative_slope,
                             dc().get_activations());
  dc().copy_out_activations();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void leaky_relu_layer<TensorDataType, Layout, Device>::
bp_compute_distconv() {
  assert_always(Layout == data_layout::DATA_PARALLEL);
  assert_always(this->distconv_enabled());
  dc().m_leaky_relu->backward(dc().get_prev_activations(),
                              dc().get_prev_error_signals(),
                              m_negative_slope,
                              dc().get_error_signals());
  dc().copy_out_error_signals();
}
#endif // LBANN_HAS_DISTCONV

#define PROTO(T)                                      \
  template class leaky_relu_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class leaky_relu_layer<T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
