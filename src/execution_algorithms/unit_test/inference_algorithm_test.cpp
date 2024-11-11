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

#include "Catch2BasicSupport.hpp"

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

#include "lbann/objective_functions/objective_function.hpp"
#include <lbann/base.hpp>
#include <lbann/execution_algorithms/batch_functional_inference_algorithm.hpp>
#include <lbann/models/model.hpp>
#include <lbann/utils/lbann_library.hpp>

#include "lbann/proto/lbann.pb.h"
#include <google/protobuf/text_format.h>
#include <lbann.pb.h>

namespace pb = ::google::protobuf;

namespace {
// This model is just an input layer into a softmax layer, so we can
// verify the output is correct for a simple input (e.g., a matrix
// filled with 1.0)
std::string const model_prototext = R"ptext(
model {
  layer {
    name: "layer1"
    children: "layer2"
    input {
      data_field: "samples"
    }
  }
  layer {
    name: "layer2"
    parents: "layer1"
    softmax {
    }
  }
}
)ptext";

lbann_data::LbannPB get_model_protobuf()
{
  lbann_data::LbannPB my_proto;
  if (!pb::TextFormat::ParseFromString(model_prototext, &my_proto))
    throw "Parsing protobuf failed.";
  return my_proto;
}
auto make_model(lbann::lbann_comm& comm, lbann_data::LbannPB& my_proto, int class_n)
{
  // Construct a trainer so that the model can register the input layer
  auto& trainer =
    lbann::construct_trainer(&comm, my_proto.mutable_trainer(), my_proto);
  unit_test::utilities::mock_data_reader(trainer, {1, 1, class_n}, class_n);
  auto my_model = lbann::proto::construct_model(&comm,
                                                my_proto.optimizer(),
                                                my_proto.trainer(),
                                                my_proto.model());
  my_model->setup(class_n, {&comm.get_trainer_grid()});
  return my_model;
}

} // namespace

namespace lbann {
void setup_inference_env(lbann_comm* lc,
                         int mbs,
                         std::vector<int> input_dims,
                         std::vector<int> output_dims);
}

TEST_CASE("Test batch_function_inference_algorithm", "[inference]")
{
  using DataType = float;
  using DMat =
    El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>;
  DataType constexpr one = 1.;
  DataType constexpr zero = 0.;
  int const mbs_class_n = 4;

  auto& comm = unit_test::utilities::current_world_comm();
  auto const& g = comm.get_trainer_grid();

  auto my_proto = get_model_protobuf(); // the pbuf msg is a global string

  // Construct a trainer so that the model can register the input layer
  lbann::construct_trainer(&comm, my_proto.mutable_trainer(), my_proto);

  lbann::setup_inference_env(&comm, mbs_class_n, {mbs_class_n}, {mbs_class_n});
  auto model = make_model(comm, my_proto, mbs_class_n);
  auto inf_alg = lbann::batch_functional_inference_algorithm();
  SECTION("Model data insert and forward prop")
  {
    DMat data(mbs_class_n, mbs_class_n, g);
    El::Fill(data, one);
    std::map<std::string, DMat> samples;
    samples["data/samples"] = std::move(data);

    lbann::set_inference_samples(samples);
    inf_alg.infer(model.get());

    auto const& l = model->get_layer(1);
    auto const& dtl = dynamic_cast<lbann::data_type_layer<float> const&>(l);
    auto const& output = dtl.get_activations();

    for (El::Int i = 0; i < output.Height(); i++) {
      for (El::Int j = 0; j < output.Width(); j++) {
        REQUIRE(output.Get(i, j) == Approx(1.0 / mbs_class_n));
      }
    }
  }

  SECTION("Verify inference label correctness")
  {
    DMat data(mbs_class_n, mbs_class_n, g);
    El::Fill(data, zero);
    El::FillDiagonal(data, one);

    std::map<std::string, DMat> samples;
    samples["data/samples"] = std::move(data);
    lbann::set_inference_samples(samples);

    auto labels = inf_alg.infer(model.get());

    for (int i = 0; i < labels.Height(); i++) {
      REQUIRE(labels(i) == i);
    }
  }
}
