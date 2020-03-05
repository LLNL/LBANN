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

#define LBANN_SPLIT_LAYER_INSTANTIATE
#include "lbann/layers/transform/split.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Dev>
void split_layer<TensorDataType, Layout, Dev>::bp_compute() {
  auto& gradient_wrt_input = this->get_error_signals();
  if (this->get_num_children() > 0) {
    El::Copy(this->get_prev_error_signals(0), gradient_wrt_input);
  } else {
    El::Zero(gradient_wrt_input);
  }
  for (int i = 1; i < this->get_num_children(); ++i) {
    El::Axpy(TensorDataType{1}, this->get_prev_error_signals(i),
             gradient_wrt_input);
  }
}

template class split_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>;
template class split_layer<DataType, data_layout::MODEL_PARALLEL, El::Device::CPU>;

}// namespace lbann
