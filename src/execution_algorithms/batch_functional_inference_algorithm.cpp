////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/execution_algorithms/batch_functional_inference_algorithm.hpp"

namespace lbann {

El::Matrix<El::Int, El::Device::CPU>
batch_functional_inference_algorithm::infer(observer_ptr<model> model)
{
  size_t const mbs = get_trainer().get_max_mini_batch_size();
  El::Matrix<El::Int, El::Device::CPU> labels(mbs, 1);

  auto c = SGDExecutionContext(execution_mode::inference, mbs);
  model->reset_mode(c, execution_mode::inference);
  get_trainer().get_data_coordinator().reset_mode(c);

  get_trainer().get_data_coordinator().fetch_data(execution_mode::inference);
  model->forward_prop(execution_mode::inference);
  get_labels(*model, labels);

  return labels;
}

void batch_functional_inference_algorithm::get_labels(
  model& model,
  El::Matrix<El::Int, El::Device::CPU>& labels)
{
  Layer const* softmax = nullptr;
  auto const layer_list = model.get_layers();
  for (auto const* const l_tmp : layer_list) {
    if (l_tmp->get_type() == "softmax") {
      softmax = l_tmp;
      break;
    }
  }
  if (!softmax)
    LBANN_ERROR("get_labels only supported when model contains a softmax. This "
                "is a known limitation and we're working on it.");
  try {
    auto const& dtl =
      dynamic_cast<lbann::data_type_layer<float> const&>(*softmax);
    const auto& outputs = dtl.get_activations();

    // Find the prediction for each sample
    El::Int const col_count = outputs.Width();
    El::Int const row_count = outputs.Height();
    LBANN_ASSERT(col_count == labels.Height());
    for (El::Int col = 0; col < col_count; ++col) {
      float max = 0.f;
      El::Int pred_label = 0;
      for (El::Int row = 0; row < row_count; ++row) {
        float const col_value = outputs.Get(row, col);
        if (col_value > max) {
          max = col_value;
          pred_label = row;
        }
      }
      labels(col, 0) = pred_label;
    }
  }
  catch (std::bad_cast const&) {
    LBANN_ERROR("Softmax layer does not have data type \"float\"");
  }
}

} // namespace lbann
