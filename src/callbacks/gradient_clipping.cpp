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
//
// gradient_clipping .hpp .cpp - Callbacks to clip gradient values in training
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/gradient_clipping.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/weights.hpp"

#include "callback_helpers.hpp"
#include "lbann/proto/callbacks.pb.h"
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

#include <vector>

namespace lbann {
namespace callback {

clip_gradient_norm::clip_gradient_norm()
  : clip_gradient_norm(std::vector<std::string>{})
{}

void clip_gradient_norm::setup(model* m)
{

  // Add all weights if list of weights is not initialized
  std::vector<weights*> weights_list =
    select_things_by_name(m->get_weights(), m_weight_names);
  if (weights_list.empty()) {
    weights_list = m->get_weights();
  }

  // Remove weights that are not being optimized
  std::unordered_set<weights*>().swap(m_weights);
  for (weights* w : weights_list) {
    if (w->has_optimizer()) {
      m_weights.insert(w);
    }
  }
}

template <class Archive>
void clip_gradient_norm::serialize(Archive& ar)
{
  ar(cereal::base_class<callback_base>(this),
     CEREAL_NVP(m_weight_names),
     CEREAL_NVP(m_global_norm),
     CEREAL_NVP(m_value));
}

void clip_gradient_norm::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_clip_gradient_norm();
  msg->set_weights(protobuf::to_space_sep_string(this->m_weight_names));
  msg->set_global_norm(m_global_norm);
  msg->set_value(m_value);
}

struct NormComputer
{
  model& m;
  DataType* global_norm_ptr;
  bool compute_global_norm;
  float norm_value;

  NormComputer(model& arg_m,
               DataType* arg_global_norm_ptr,
               bool arg_compute_global_norm,
               float arg_norm_value)
    : m(arg_m),
      global_norm_ptr(arg_global_norm_ptr),
      compute_global_norm(arg_compute_global_norm),
      norm_value(arg_norm_value)
  {}

  template <typename... Ts>
  void DispatchError(Ts&&...)
  {
    LBANN_ERROR("Unable to dispatch functor.");
  }

  template <typename... Ts>
  void DeductionError(Ts&&...)
  {
    LBANN_ERROR("Unable to deduce an argument type.");
  }

  template <typename TensorDataType>
  void operator()(data_type_weights<TensorDataType>& dtw)
  {
    data_type_optimizer<TensorDataType>* opt = dtw.get_optimizer();
    auto& grad = opt->get_gradient_sharded();
    if (!compute_global_norm) {
      TensorDataType norm;

      // The following call may incur communication (e.g., with sharded weights)
      norm = El::Nrm2(grad);
      if (norm > norm_value) {
        El::Scale(norm_value / norm, grad);
      }
    }
    else {
      const auto& gradmat = grad.LockedMatrix();
      TensorDataType local_norm = TensorDataType(0);
      if ((gradmat.GetDevice() == El::Device::CPU)) {
        const auto& gradmatrix =
          static_cast<const El::Matrix<TensorDataType, El::Device::CPU>&>(
            gradmat);
        local_norm = El::Nrm2(gradmatrix);
#ifdef LBANN_HAS_GPU
      }
      else if ((gradmat.GetDevice() == El::Device::GPU)) {
        const auto& gradmatrix =
          static_cast<const El::Matrix<TensorDataType, El::Device::GPU>&>(
            gradmat);
        hydrogen::gpu_blas::Nrm2(
          size_t(gradmatrix.Width() * gradmatrix.Height()),
          gradmatrix.LockedBuffer(),
          size_t(1),
          &local_norm,
          El::SyncInfoFromMatrix(gradmatrix));
#endif // LBANN_HAS_GPU
      }
      *global_norm_ptr += local_norm * local_norm;
    }
  }
};

void clip_gradient_norm::on_backward_prop_end(model* m)
{
  using WeightsTypes =
    h2::meta::tlist::ExpandTL<data_type_weights, supported_layer_data_type>;
  using Dispatcher = h2::multimethods::
    SwitchDispatcher<NormComputer, void, weights, WeightsTypes>;
  DataType global_norm = 0;
  for (weights* w : this->m_weights) {
    if (w->get_optimizer() != nullptr) {
      Dispatcher::Exec(NormComputer(*m, &global_norm, m_global_norm, m_value),
                       *w);
    }
  }

  if (m_global_norm) {
    global_norm = m->get_comm()->trainer_allreduce(global_norm);
    global_norm = std::sqrt(global_norm);
    if (global_norm > this->m_value) {
      DataType scale = this->m_value / global_norm;
      for (weights* w : this->m_weights) {
        optimizer* opt = w->get_optimizer();
        if (opt != nullptr) {
          auto* dt_opt = dynamic_cast<data_type_optimizer<DataType>*>(opt);
          auto& grad = dt_opt->get_gradient_sharded();
          El::Scale(scale, grad);
        }
      }
    }
  }
}

std::unique_ptr<callback_base> build_clip_gradient_norm_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackClipGradientNorm&>(
      proto_msg);
  return std::make_unique<clip_gradient_norm>(
    parse_list<std::string>(params.weights()),
    params.global_norm(),
    params.value());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::clip_gradient_norm
#define LBANN_CLASS_LIBNAME callback_clip_gradient_norm
#include <lbann/macros/register_class_with_cereal.hpp>
