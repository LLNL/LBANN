////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
// lbann_data_reader .hpp .cpp - Input data base class for training, testing
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader.hpp"
#include "lbann/data_readers/data_reader_conduit.hpp"

namespace lbann {

void conduit_data_reader::load() {
}

bool conduit_data_reader::fetch_conduit_node(conduit::Node& sample, int data_id)
{
  // get the pathname to the data, and verify it exists in the conduit::Node
  const conduit::Node& node = get_data_store().get_conduit_node(data_id);
  sample = node;
  return true;
}

void conduit_data_reader::set_data_dims(std::vector<int> dims) {
  m_data_dims = dims;
}

void conduit_data_reader::set_label_dims(std::vector<int> dims) {
  m_label_dims = dims;
}

}  // namespace lbann
