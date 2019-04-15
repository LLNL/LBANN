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

#include "lbann/proto/factories.hpp"
#include "lbann/objective_functions/layer_term.hpp"

namespace lbann {
namespace proto {

namespace {

/** Instantiate a trainer based on prototext. */
std::unique_ptr<trainer> instantiate_trainer(lbann_comm* comm,
                                             const lbann_data::Trainer& proto_trainer) {
  std::stringstream err;

  // Construct trainer
  return make_unique<trainer>(comm);
}

} // namespace

std::unique_ptr<trainer> construct_trainer(lbann_comm* comm,
                                           const lbann_data::Trainer& proto_trainer) {

  // Instantiate trainer
  auto&& t = instantiate_trainer(comm, proto_trainer);
  const auto& name = proto_trainer.name();
  if (!name.empty()) {
    t->set_name(name);
  }
  return std::move(t);
}

} // namespace proto
} // namespace lbann
