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
#include <lbann/layers/io/input_layer.hpp>
#include <lbann/layers/layer.hpp>
#include <lbann/models/model.hpp>
#include <lbann/proto/factories.hpp>
#include <lbann/utils/lbann_library.hpp>
#include <lbann/utils/memory.hpp>
#include <lbann/utils/serialize.hpp>

#include "lbann/proto/lbann.pb.h"
#include <google/protobuf/text_format.h>

namespace pb = ::google::protobuf;

namespace {
// model_prototext string is defined here as a "const std::string".
#include "lenet.prototext.inc"

auto mock_datareader_metadata()
{
  lbann::DataReaderMetaData md;
  auto& md_dims = md.data_dims;
  // This is all that should be needed for this test.
  md_dims[lbann::data_reader_target_mode::CLASSIFICATION] = {10};
  md_dims[lbann::data_reader_target_mode::INPUT] = {1,28,28};
  return md;
}

template <typename T>
auto make_model(lbann::lbann_comm& comm,
                const std::string& model_contents = model_prototext)
{
  lbann_data::LbannPB my_proto;
  if (!pb::TextFormat::ParseFromString(model_contents, &my_proto))
    throw "Parsing protobuf failed.";
  // Construct a trainer so that the model can register the input layer
  auto& trainer =
    lbann::construct_trainer(&comm, my_proto.mutable_trainer(), my_proto);
  unit_test::utilities::mock_data_reader(trainer, {1, 28, 28}, 10);
  auto my_model = lbann::proto::construct_model(&comm,
                                                my_proto.optimizer(),
                                                my_proto.trainer(),
                                                my_proto.model());
  my_model->setup(1UL, {&comm.get_trainer_grid()});
  return my_model;
}

} // namespace

using unit_test::utilities::IsValidPtr;
TEST_CASE("Serializing models", "[mpi][model][serialize]")
{
  auto& comm = unit_test::utilities::current_world_comm();

  auto& g = comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  std::stringstream ss;
  std::unique_ptr<lbann::model> model_src_ptr = make_model<DataType>(comm),
                                model_tgt_ptr;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
  SECTION("Binary archive")
  {
    {
      cereal::BinaryOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(model_src_ptr));
    }
    {
      cereal::BinaryInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(model_tgt_ptr));
      REQUIRE(IsValidPtr(model_tgt_ptr));
    }
    if (IsValidPtr(model_tgt_ptr)) {
      REQUIRE_NOTHROW(model_tgt_ptr->setup(1UL, {&g}));
      // if (comm.get_rank_in_world() == 1)
      //   std::cout << model_tgt_ptr->get_description()
      //             << std::endl;
    }
  }

  SECTION("Rooted binary archive")
  {
    {
      lbann::RootedBinaryOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(model_src_ptr));
    }
    {
      lbann::RootedBinaryInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(model_tgt_ptr));
      REQUIRE(IsValidPtr(model_tgt_ptr));
    }

    if (IsValidPtr(model_tgt_ptr)) {
      REQUIRE_NOTHROW(model_tgt_ptr->setup(1UL, {&g}));
      // if (comm.get_rank_in_world() == 1)
      //   std::cout << model_tgt_ptr->get_description()
      //             << std::endl;
    }
  }
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  SECTION("XML archive")
  {
    {
      cereal::XMLOutputArchive oarchive(ss);
      REQUIRE_NOTHROW(oarchive(model_src_ptr));
    }
    {
      cereal::XMLInputArchive iarchive(ss);
      REQUIRE_NOTHROW(iarchive(model_tgt_ptr));
      REQUIRE(IsValidPtr(model_tgt_ptr));
    }
  }

  SECTION("Rooted XML archive")
  {
    {
      lbann::RootedXMLOutputArchive oarchive(ss, g);
      REQUIRE_NOTHROW(oarchive(model_src_ptr));
    }
    // std::cout << ss.str() << std::endl;
    {
      lbann::RootedXMLInputArchive iarchive(ss, g);
      REQUIRE_NOTHROW(iarchive(model_tgt_ptr));
      REQUIRE(IsValidPtr(model_tgt_ptr));
    }
  }
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
}

// Prototext graphs
const std::string backprop_test_model = R"""(
model {
  layer {
    name: "inp"
    children: "layer1"
    weights: "dummy_inputs"
    weights_layer {
      dims: 3
    }
  }
  layer {
    name: "layer1"
    parents: "inp"
    children: "stopgrad"
    relu {
    }
  }
  layer {
    name: "stopgrad"
    parents: "layer1"
    children: "layer2"
    stop_gradient {
    }
  }
  layer {
    name: "layer2"
    parents: "stopgrad"
    relu {
    }
  }
  weights {
    name: "dummy_inputs"
    initializer {
      value_initializer {
        values: -1.2
        values: 3.4
        values: -5.67
      }
    }
  }
}
)""";

class backprop_test_layer final : public lbann::data_type_layer<float>
{
public:
  backprop_test_layer(lbann::lbann_comm* comm)
    : lbann::data_type_layer<float>(comm),
      m_fprop_computed(false),
      m_bprop_computed(false)
  {}
  backprop_test_layer* copy() const final
  {
    return new backprop_test_layer(*this);
  }

  std::string get_type() const final { return "backprop_test_layer"; }
  lbann::data_layout get_data_layout() const final
  {
    return lbann::data_layout::DATA_PARALLEL;
  }
  El::Device get_device_allocation() const final { return El::Device::CPU; }
  void write_specific_proto(lbann_data::Layer& proto) const final {}
  bool fprop_computed() const { return m_fprop_computed; }
  bool bprop_computed() const { return m_bprop_computed; }

private:
  friend class cereal::access;
  backprop_test_layer() : backprop_test_layer(nullptr) {}

  void setup_dims() final
  {
    lbann::data_type_layer<float>::setup_dims();
    this->set_output_dims(this->get_input_dims());
  }
  void fp_setup_outputs() override
  {
    El::LockedView(this->get_activations(), this->get_prev_activations());
  }
  void bp_setup_gradient_wrt_inputs() override
  {
    El::LockedView(this->get_error_signals(), this->get_prev_error_signals());
  }

  void fp_compute() final { m_fprop_computed = true; }

  void bp_compute() final { m_bprop_computed = true; }

  bool m_fprop_computed, m_bprop_computed;
};

TEST_CASE("Gradient backpropagation stop", "[mpi][model][backprop]")
{
  using DataType = float;

  auto& comm = unit_test::utilities::current_world_comm();

  auto& g = comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);
  std::unique_ptr<lbann::model> model =
    make_model<DataType>(comm, backprop_test_model);

  std::shared_ptr<backprop_test_layer> before_stopgrad =
    std::make_shared<backprop_test_layer>(&comm);
  std::shared_ptr<backprop_test_layer> after_stopgrad =
    std::make_shared<backprop_test_layer>(&comm);

  model->insert_layer(before_stopgrad, "layer1");
  model->insert_layer(after_stopgrad, "layer2");

  // Reset the model after adding the new layers
  model->setup(1UL, {&g}, true);

  REQUIRE(!before_stopgrad->fprop_computed());
  REQUIRE(!before_stopgrad->bprop_computed());
  REQUIRE(!after_stopgrad->fprop_computed());
  REQUIRE(!after_stopgrad->bprop_computed());

  // Run forward
  REQUIRE_NOTHROW(model->forward_prop(lbann::execution_mode::training));
  REQUIRE(before_stopgrad->fprop_computed());
  REQUIRE(after_stopgrad->fprop_computed());

  // Run backprop
  REQUIRE_NOTHROW(model->backward_prop(false));
  REQUIRE(!before_stopgrad->bprop_computed());
  REQUIRE(after_stopgrad->bprop_computed());
}
