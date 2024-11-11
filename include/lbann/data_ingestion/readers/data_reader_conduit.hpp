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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_CONDUIT_HPP
#define LBANN_DATA_READER_CONDUIT_HPP

#include "lbann/data_readers/data_reader.hpp"
#include "lbann/data_store/data_store_conduit.hpp"

namespace lbann {
/**
 * A generalized data reader for passed in conduit nodes.
 */
class conduit_data_reader : public generic_data_reader
{
public:
  conduit_data_reader* copy() const override { return new conduit_data_reader(*this); }
  bool has_conduit_output() override { return true; }
  void load() override;
  bool fetch_conduit_node(conduit::Node& sample, int data_id) override;

  void set_data_dims(std::vector<int> dims);
  void set_label_dims(std::vector<int> dims);

  std::string get_type() const override { return "conduit_data_reader"; }
  int get_linearized_data_size() const override {
    int data_size = 1;
    for(int i : m_data_dims) {
      data_size *= i;
    }
    return data_size;
  }
  int get_linearized_label_size() const override {
    int label_size = 1;
    for(int i : m_label_dims) {
      label_size *= i;
    }
    return label_size;
  }

private:
  std::vector<int> m_data_dims;
  std::vector<int> m_label_dims;

}; // END: class conduit_data_reader

} // namespace lbann

#endif // LBANN_DATA_READER_CONDUIT_HPP
