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

// This header
#include "lbann/trainers/trainer.hpp"

// LBANN dependencies
#include "lbann/base.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/data_ingestion/data_coordinator.hpp"
#include "lbann/data_ingestion/readers/metadata.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/io/persist_impl.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/serialize.hpp"

// LBANN proto
#include "lbann/proto/lbann.pb.h"

// STL
#include <functional>
#include <memory>
#include <string>

namespace lbann {

////////////////////////////////////////////////////////////
// Constructors and destructor
////////////////////////////////////////////////////////////

trainer::trainer(lbann_comm* comm,
                 std::unique_ptr<data_coordinator> dc,
                 size_t mini_batch_size,
                 std::unique_ptr<TrainingAlgorithm> alg)
  : m_data_coordinator{std::move(dc)},
    m_training_alg{std::move(alg)},
    m_comm{comm},
    m_max_mini_batch_size{mini_batch_size},
    m_background_io_allowed{true}
{
  // Default trainer name
  m_name = "trainer" + std::to_string(m_comm->get_trainer_rank());
  m_data_coordinator->set_trainer(*this);
}

trainer::~trainer() {}

template <class Archive>
void trainer::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_persist),
     CEREAL_NVP(m_max_mini_batch_size),
     CEREAL_NVP(m_root_random_seed),
     CEREAL_NVP(m_random_seed),
     CEREAL_NVP(m_data_seq_random_seed),
     CEREAL_NVP(m_background_io_allowed));
}

////////////////////////////////////////////////////////////
// Trainer specification
////////////////////////////////////////////////////////////

void trainer::set_name(std::string const& name)
{
  if (name.empty()) {
    LBANN_ERROR("attempted to rename trainer \"",
                get_name(),
                "\" with empty string");
  }
  m_name = name;
}

description trainer::get_description() const
{

  // Construct description object
  description desc(get_name());
  desc.add("Background I/O", m_background_io_allowed);

  // Result
  return desc;
}

////////////////////////////////////////////////////////////
// Setup
////////////////////////////////////////////////////////////

void trainer::setup(std::unique_ptr<thread_pool> io_thread_pool,
                    std::map<execution_mode, generic_data_reader*> data_readers)
{
  // Setup I/O threads - set up before setting up the layers (input
  // layer depends on having a properly initialized thread pool)
  m_io_thread_pool = std::move(io_thread_pool);

  m_data_coordinator->setup(*m_io_thread_pool.get(),
                            get_max_mini_batch_size(),
                            data_readers);

  for (auto& [mode, reader] : data_readers) {
    if (!reader->supports_background_io()) {
      allow_background_io_activity(false);
    }
  }

  // Set up callbacks first - allow checkpoint / restart to reload state
  for (auto& cb : m_callbacks) {
    cb->setup(this);
  }
}

/// Check if there is already an execution context for the model in this mode,
/// if not create one
trainer::execution_context_key_pair_t
trainer::check_and_build_execution_context(TrainingAlgorithm& alg,
                                           observer_ptr<model> model,
                                           execution_mode mode)
{
  auto key = std::make_pair(model, mode);
  if (m_model_execution_context.count(key) == 0) {
    /// Create a execution context for each model and execution mode
    std::unique_ptr<ExecutionContext> context;
    if (dynamic_cast<observer_ptr<SGDTrainingAlgorithm>>(&alg) != nullptr) {
      /// @todo BVE FIXME Figure out how to get a good mini-batch size
      /// in here
      context = std::make_unique<SGDExecutionContext>(mode);
    }
    else {
      LBANN_ERROR("Unknown execution algorithm type.");
    }
    m_model_execution_context.emplace(key, std::move(context));
  }
  return key;
}

/// Check if there is already an execution context for the model in this mode,
/// if not create one
trainer::execution_context_key_pair_t
trainer::check_and_build_execution_context(ExecutionContext& c,
                                           model& model,
                                           execution_mode mode)
{
  auto key = std::make_pair(&model, mode);
  if (m_model_execution_context.count(key) == 0) {
    std::unique_ptr<ExecutionContext> context;
    //    observer_ptr<training_algorithm> alg = const_cast
    if (dynamic_cast<observer_ptr</*const */ SGDExecutionContext>>(&c) !=
        nullptr) {
      context = std::make_unique<SGDExecutionContext>(mode);
    }
    else {
      LBANN_ERROR("Unknown execution context type");
    }
    m_model_execution_context.emplace(key, std::move(context));
  }
  return key;
}

ExecutionContext& trainer::get_execution_context(observer_ptr<model> model,
                                                 execution_mode mode)
{
  auto key = std::make_pair(model, mode);
  return get_execution_context(key);
}

ExecutionContext&
trainer::get_execution_context(execution_context_key_pair_t key)
{
  if (m_model_execution_context.count(key) == 0) {
    LBANN_ERROR("No execution context for this model / mode pair");
  }
  return static_cast<SGDExecutionContext&>(
    *(m_model_execution_context[key].get()));
}

bool trainer::execution_context_valid(model& m,
                                      execution_mode mode) const noexcept
{
  return execution_context_valid(std::make_pair(&m, mode));
}

bool trainer::execution_context_valid(
  execution_context_key_pair_t key) const noexcept
{
  return (m_model_execution_context.count(key) != 0);
}

void trainer::delete_execution_context(execution_context_key_pair_t key)
{
  if (m_model_execution_context.count(key) == 0) {
    LBANN_WARNING(
      "Attempting to delete an invalid execution context for model=" +
      (key.first)->get_name() + " / " + to_string(key.second));
  }
  m_model_execution_context.erase(key);
}

/// @todo BVE FIXME seems like there is a bug here about mapping
/// execution contexts to the right model
void trainer::for_each_execution_context(
  std::function<void(observer_ptr<ExecutionContext>)> fn)
{
  for (auto&& c : m_model_execution_context) {
    // auto&& model = c.first.first;
    // auto&& mode = c.first.second;
    auto&& context = c.second;
    fn(context.get());
  }
}

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

void trainer::train(observer_ptr<model> model,
                    El::Int num_epochs,
                    El::Int num_batches)
{
  // FIXME (trb 04/22/21): This is a temporary fix to support old PFE
  // model descriptions.
  if (!m_training_alg) {
    std::unique_ptr<SGDTerminationCriteria> stopping;
    if (num_epochs)
      stopping = std::make_unique<EpochTerminationCriteria>(num_epochs);
    else
      stopping = std::make_unique<BatchTerminationCriteria>(num_batches);

    m_training_alg =
      std::make_unique<SGDTrainingAlgorithm>("sgd_train",
                                             std::move(stopping),
                                             /*suppress_timer=*/false);
  }
  m_training_alg->setup_models({model}, get_max_mini_batch_size(), get_grids());

  // FIXME (trb 04/27/2021): This is a hack to support the current
  // checkpoint/restart mechanisms. This needs to be refactored to be
  // agnostic to the training algorithm. At this time, only SGD is
  // properly C/R-able.
  if (m_training_alg->get_type() == "sgd") {
    auto key = check_and_build_execution_context(*m_training_alg,
                                                 model,
                                                 execution_mode::training);
    m_training_alg->apply(*(m_model_execution_context[key]),
                          *model,
                          get_data_coordinator(),
                          execution_mode::training);
  }
  else {
    m_training_alg->apply(*model, get_data_coordinator());
  }
}

// NOTE (trb 04/19/2021): Currently, "evaluate" is just defined as
// "run forward prop and look at objective
// functions/metrics/etc". This is currently implemented in
// `sgd_training_algorithm`, so we just exploit that for now. This
// could should be refactored in the future.
void trainer::evaluate(observer_ptr<model> model,
                       execution_mode mode,
                       El::Int num_batches)
{
  auto sgd = std::make_unique<SGDTrainingAlgorithm>(
    "sgd_evaluate",
    std::make_unique<EpochTerminationCriteria>(/*num_epochs=*/1UL),
    /*suppress_timer=*/true);
  auto ctxt = sgd->get_new_execution_context();
  ctxt->set_execution_mode(mode);
  model->reset_mode(*ctxt, execution_mode::invalid);

  sgd->setup_models({model}, get_max_mini_batch_size(), get_grids());

  if (m_comm->get_grid_type() == GridType::NO_GRID or
      m_comm->get_grid_type() == GridType::PRIMARY_GRID) {
    sgd->evaluate(*ctxt,
                  *model,
                  get_data_coordinator(),
                  mode,
                  EpochTerminationCriteria(/*num_epochs=*/1UL));
  }
}

// =============================================
// Sub-grid management
// =============================================

std::vector<El::Grid*> trainer::get_grids() const
{
  std::vector<El::Grid*> grids;
  grids.reserve(m_grids.size() + 1);
  grids.push_back(&get_comm()->get_trainer_grid());
  for (const auto& g : m_grids) {
    grids.push_back(g.get());
  }
  return grids;
}

void trainer::add_grid(std::unique_ptr<El::Grid> g)
{
  m_grids.emplace_back(std::move(g));
}

// =============================================
// Checkpointing
// =============================================

bool trainer::save_to_checkpoint_shared()
{
  for_each_execution_context([this](observer_ptr<ExecutionContext> ctx) {
    ctx->save_to_checkpoint_shared(this->get_persist_obj());
  });
  save_rng_to_checkpoint_shared(get_persist_obj(), m_comm);

  if (m_comm->am_trainer_master()) {
    write_cereal_archive(*this,
                         get_persist_obj(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                         "trainer.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                         "trainer.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
  }

  return get_data_coordinator().save_to_checkpoint_shared(get_persist_obj());
}

bool trainer::load_from_checkpoint_shared(persist& p)
{
  try {
    load_from_shared_cereal_archive(*this,
                                    p,
                                    *get_comm(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                    "trainer.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                    "trainer.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
  }
  catch (NonexistentArchiveFile const& e) {
    LBANN_MSG(e.what());
    return false;
  }

  return get_data_coordinator().load_from_checkpoint_shared(p);
}

bool trainer::load_from_checkpoint_shared(model& m, ExecutionContext& c)
{
  // Reload the RNG once the trainer and all of the  models are setup
  // to avoid spurious turns of the RNGs
  load_rng_from_checkpoint(get_persist_obj(), m_comm);

  execution_mode current_mode = c.get_execution_mode();
  for (execution_mode mode : execution_mode_iterator()) {
    /// Restart should optionally load any other valid contexts
    if (mode == execution_mode::invalid) {
      continue;
    }
    trainer::execution_context_key_pair_t key;
    try {
      if (current_mode == mode) {
        /// Restart has to be able to load the currently running execution
        /// context
        c.load_from_checkpoint_shared(get_persist_obj());
      }
      else {
        key = check_and_build_execution_context(c, m, mode);
        auto& evaluation_context =
          static_cast<SGDExecutionContext&>(get_execution_context(key));
        evaluation_context.load_from_checkpoint_shared(get_persist_obj());
      }
    }
    catch (NonexistentArchiveFile const& e) {
      // Ignore the exception if the file is not for the current execution mode
      if (current_mode == mode) {
        LBANN_ERROR("Failed to restart model, invalid execution mode: ",
                    to_string(current_mode),
                    "\n\n  e.what(): ",
                    e.what(),
                    "\n");
      }
      else {
        delete_execution_context(key);
      }
    }
  }

  return get_data_coordinator().load_from_checkpoint_shared(get_persist_obj());
}

bool trainer::save_to_checkpoint_distributed()
{
  for_each_execution_context([this](observer_ptr<ExecutionContext> ctx) {
    ctx->save_to_checkpoint_distributed(this->get_persist_obj());
  });
  save_rng_to_checkpoint_distributed(get_persist_obj(), m_comm);
  return get_data_coordinator().save_to_checkpoint_shared(get_persist_obj());
}

bool trainer::load_from_checkpoint_distributed(persist& p)
{
  read_cereal_archive(*this,
                      p,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                      "trainer.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                      "trainer.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return get_data_coordinator().load_from_checkpoint_distributed(p);
}

bool trainer::load_from_checkpoint_distributed(model& m, ExecutionContext& c)
{
  load_rng_from_checkpoint(get_persist_obj(), m_comm);

  execution_mode current_mode = c.get_execution_mode();

  for (execution_mode mode : execution_mode_iterator()) {
    /// Restart should optionally load any other valid contexts
    if (mode == execution_mode::invalid) {
      continue;
    }
    trainer::execution_context_key_pair_t key;
    try {
      if (current_mode == mode) {
        /// Restart has to be able to load the currently running  execution
        /// context
        c.load_from_checkpoint_distributed(get_persist_obj());
      }
      else {
        key = check_and_build_execution_context(c, m, mode);
        auto& evaluation_context =
          static_cast<SGDExecutionContext&>(get_execution_context(key));
        evaluation_context.load_from_checkpoint_distributed(get_persist_obj());
      }
    }
    catch (NonexistentArchiveFile const&) {
      // Ignore the exception if the file is not for the current execution mode
      if (current_mode == mode) {
        LBANN_ERROR("Failed to restart model, invalid execution mode: " +
                    to_string(current_mode));
      }
      else {
        delete_execution_context(key);
      }
    }
  }
  return get_data_coordinator().load_from_checkpoint_distributed(
    get_persist_obj());
}

void trainer::write_proto(lbann_data::Trainer& proto)
{
  proto.Clear();
  if (m_comm->am_world_master()) {
    proto.set_mini_batch_size(m_max_mini_batch_size);
  }
}

} // namespace lbann
