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

#ifndef LBANN_BATCH_INFERENCE_ALGORITHM_HPP
#define LBANN_BATCH_INFERENCE_ALGORITHM_HPP

#include "lbann/callbacks/callback.hpp"
#include "lbann/data_ingestion/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/io/input_layer.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

/** @brief Class for LBANN batch inference algorithms.
 *
 *  This execution algorithm is meant for running inference using a trained
 *  model and samples passed by the user from an external application.  The
 *  algorithm currently assumes that the output layer is a softmax layer.
 */
class batch_functional_inference_algorithm
{
public:
  /** Constructor. */
  batch_functional_inference_algorithm(){};
  /** Copy constructor. */
  batch_functional_inference_algorithm(
    const batch_functional_inference_algorithm& other) = default;
  /** Copy assignment operator. */
  batch_functional_inference_algorithm&
  operator=(const batch_functional_inference_algorithm& other) = default;
  /** Move constructor. */
  batch_functional_inference_algorithm(
    batch_functional_inference_algorithm&& other) = default;
  /** Move assignment operator. */
  batch_functional_inference_algorithm&
  operator=(batch_functional_inference_algorithm&& other) = default;
  /** Destructor. */
  virtual ~batch_functional_inference_algorithm() = default;

  std::string get_name() const { return "batch_functional_inference"; }

  std::string get_type() const { return "batch_functional_inference"; }

  // ===========================================
  // Execution
  // ===========================================

  /** @brief Run model inference on samples and return predicted categories.
   * @param[in] model A trained model
   * @return Matrix of predicted labels (by index)
   */
  El::Matrix<El::Int, El::Device::CPU> infer(observer_ptr<model> model);

protected:
  /** @brief Finds the predicted category in a models softmax layer
   * @param[in] model A model that has been used for inference
   * @param[in] labels A matrix to place predicted category labels
   */
  void get_labels(model& model, El::Matrix<El::Int, El::Device::CPU>& labels);
};

} // namespace lbann

#endif // LBANN_BATCH_INFERENCE_ALGORITHM_HPP
