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

#include "lbann/utils/lbann_library.hpp"
#include "lbann/comm.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/data_ingestion/data_reader.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/exception.hpp"

#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/callbacks/dump_weights.hpp"
#include "lbann/callbacks/load_model.hpp"
#include "lbann/callbacks/save_model.hpp"
#include "lbann/models/model.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/proto/factories.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/omp_diagnostics.hpp"
#include "lbann/utils/threads/thread_utils.hpp"

#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/model.pb.h"
#include <conduit/conduit_node.hpp>

#include <cstdlib>
#include <memory>

namespace lbann {

namespace {

std::unique_ptr<trainer> global_trainer_;

void cleanup_trainer_atexit() { global_trainer_ = nullptr; }

} // namespace

trainer& get_trainer()
{
  LBANN_ASSERT(global_trainer_);
  return *global_trainer_;
}
trainer const& get_const_trainer()
{
  LBANN_ASSERT(global_trainer_);
  return *global_trainer_;
}

bool trainer_exists()
{
  if (global_trainer_ == nullptr) {
    return false;
  }
  else {
    return true;
  }
}

void finalize_trainer() { global_trainer_.reset(); }

// Creates a datareader metadata to get around the need for an actual
// datareader in inference only mode
auto mock_dr_metadata(std::vector<int> input_dims,
                      std::vector<int> output_dims) {
  DataReaderMetaData drmd;
  auto& md_dims = drmd.data_dims;
  md_dims[data_reader_target_mode::INPUT] = input_dims;
  md_dims[data_reader_target_mode::CLASSIFICATION] = output_dims;
  return drmd;
}

// Sets data samples for lbann-core inference
void set_inference_samples(std::vector<conduit::Node> &samples) {
  data_coordinator& dc = global_trainer_->get_data_coordinator();
  generic_data_reader* dr = dc.get_data_reader(execution_mode::inference);
  data_store_conduit& ds = dr->get_data_store();

  // Push samples into data_store
  std::vector<int> indices(samples.size());
  std::iota(indices.begin(), indices.end(), 0);
  dr->set_shuffled_indices(indices);
  int data_id = 0;
  for (auto& node : samples) {
    ds.set_conduit_node(data_id, node);
    ++data_id;
  }

  // Call setup on data_coordinator to have proper init of input buffers
  std::map<execution_mode, generic_data_reader*> data_readers = {
    {execution_mode::inference, dr}};
  dc.setup(global_trainer_->get_io_thread_pool(),
           global_trainer_->get_max_mini_batch_size(),
           data_readers);
}

void setup_inference_env(lbann_comm* lc,
                         int mbs,
                         std::vector<int> input_dims,
                         std::vector<int> output_dims)
{
  // Setup data reader
  auto reader = std::make_unique<conduit_data_reader>();
  reader->set_comm(lc);
  reader->set_num_parallel_readers(lc->get_procs_per_trainer());
  reader->set_mini_batch_size(mbs);
  reader->set_data_dims(input_dims);
  reader->set_label_dims(output_dims);
  reader->set_shuffle(false);

  // Add data store
  reader->set_data_store(new lbann::data_store_conduit(reader.get()));

  // Setup data coordinator and trainer
  std::unique_ptr<thread_pool> io_thread_pool =
    construct_io_thread_pool(lc, false);
  auto io_threads_per_process = io_thread_pool->get_num_threads();
  std::unique_ptr<data_coordinator> dc =
    lbann::make_unique<buffered_data_coordinator<float>>(lc);
  global_trainer_ =
    lbann::make_unique<trainer>(lc, std::move(dc), mbs, nullptr);
  int root_random_seed = lbann_default_random_seed;
  int random_seed = root_random_seed;
  int data_seq_random_seed = root_random_seed;
  init_random(random_seed, io_threads_per_process);
  init_data_seq_random(data_seq_random_seed);
  init_ltfb_random(root_random_seed);
  global_trainer_->set_random_seeds(root_random_seed,
                                    random_seed,
                                    data_seq_random_seed);

  // FIXME: The global trainer holds a data coordinator that holds
  // this map. When the aforementioned data coordinator is destroyed,
  // it deletes the data readers that it's holding. Therefore, we
  // release the pointer that we hold and hope no exceptions occur
  // before the data coordinator takes control of the pointer.
  std::map<execution_mode, generic_data_reader*> data_readers = {
    {execution_mode::inference, reader.release()}};
  global_trainer_->setup(std::move(io_thread_pool), data_readers);
}

// Loads a model from checkpoint and sets up model for inference
std::unique_ptr<model>
load_inference_model(lbann_comm* lc,
                     std::string cp_dir,
                     int mbs,
                     std::vector<int> input_dims,
                     std::vector<int> output_dims) {
  // Setup trainer, dr, dc, ds
  setup_inference_env(lc, mbs, input_dims, output_dims);

  // Load checkpoint
  persist p;
  p.open_restart(cp_dir.c_str());
  auto m = make_unique<model>(lc, nullptr, nullptr);
  m->load_from_checkpoint_shared(p);
  p.close_restart();

  m->setup(mbs, get_trainer().get_grids());

  return m;
}

El::Matrix<El::Int, El::Device::CPU>
inference(observer_ptr<model> model) {
  auto inf_alg = batch_functional_inference_algorithm();
  return inf_alg.infer(model);
}

/// Split the MPI communicator into trainers
/// Return the
int allocate_trainer_resources(lbann_comm* comm)
{
  auto& arg_parser = global_argument_parser();
  int procs_per_trainer = arg_parser.get<int>(LBANN_OPTION_PROCS_PER_TRAINER);
  int trainer_grid_height =
    arg_parser.get<int>(LBANN_OPTION_TRAINER_GRID_HEIGHT);
  int trainer_primary_grid_size =
    arg_parser.get<int>(LBANN_OPTION_TRAINER_PRIMARY_GRID_SIZE);
  bool trainer_create_two_models =
    arg_parser.get<bool>(LBANN_OPTION_TRAINER_CREATE_TWO_MODELS);
  bool trainer_async_comm_subgrid =
    arg_parser.get<bool>(LBANN_OPTION_TRAINER_ENABLE_SUBGRID_ASYNC_COMM);
  bool trainer_topo_aware_subgrid =
    arg_parser.get<bool>(LBANN_OPTION_TRAINER_ENABLE_TOPO_AWARE_SUBGRID);

  if (procs_per_trainer == 0) {
    procs_per_trainer = comm->get_procs_in_world();
  }

  // Set up the communicator and get the grid based on the commandline spec.
  // We do not currently support splitting different trainers in different ways,
  // as this implies different grids.
  if (procs_per_trainer != comm->get_procs_per_trainer() ||
      trainer_grid_height != comm->get_trainer_grid().Height()) {
    comm->split_trainers(procs_per_trainer, trainer_grid_height);
  }

  // Split trainer when sub-grid parallelism is enabled
  if (trainer_primary_grid_size > 0) {
    comm->split_trainer_grid(trainer_primary_grid_size,
                             trainer_create_two_models,
                             trainer_async_comm_subgrid,
                             trainer_topo_aware_subgrid);
  }

  return procs_per_trainer;
}

/// Construct a trainer that contains a lbann comm object and threadpool
trainer& construct_trainer(lbann_comm* comm,
                           lbann_data::Trainer* pb_trainer,
                           lbann_data::LbannPB& pb)
{
  int const procs_per_trainer = comm->get_procs_per_trainer();
  auto const& arg_parser = global_argument_parser();

  // Set the number of OMP threads for the trainer (Note that the
  // LBANN option will inherit from std::getenv("OMP_NUM_THREADS")
  auto num_omp_threads = arg_parser.get<int>(LBANN_OPTION_OMP_NUM_THREADS);
  omp_set_num_threads(num_omp_threads);

  // Check to see if the model wants to reduce the I/O parallelism
  bool const serialized_io = pb_trainer->serialize_io();
  if (comm->am_trainer_master()) {
    if (serialized_io) {
      std::cout << "Trainer " << pb_trainer->name()
                << " serialized the I/O threads" << std::endl;
    }
  }

  // Initalize a per-trainer I/O thread pool
  std::unique_ptr<thread_pool> io_thread_pool =
    construct_io_thread_pool(comm, serialized_io);

  // Setup I/O threads
  auto const io_threads_per_process = io_thread_pool->get_num_threads();
  auto const io_threads_offset = io_thread_pool->get_threads_offset();

  // Set algorithmic blocksize in Hydrogen
  if (pb_trainer->hydrogen_block_size() > 0) {
    El::SetBlocksize(pb_trainer->hydrogen_block_size());
  }

  // Display how the OpenMP threads are provisioned
  // if (opts->has_string("print_affinity")) {
  //   display_omp_setup();
  // }

  // User feedback
  //    print_parameters(comm, pb);

  // Initalize trainer
  global_trainer_ = proto::construct_trainer(comm, *pb_trainer);

  // FIXME (trb 04/09/21): This ensures that the trainer is destroyed
  // before the Hydrogen threadpools come destruction time. This is a
  // hack and we should more carefully consider how globals from our
  // various packages interact, both explicitly and implicitly.
  std::atexit(cleanup_trainer_atexit);

  // If the checkpoint directory has been overridden reset it before
  // setting up the trainer
  if (arg_parser.get<std::string>(LBANN_OPTION_CKPT_DIR) != "") {
    for (auto&& c : global_trainer_->get_callbacks()) {
      {
        auto* const cb = dynamic_cast<callback::checkpoint*>(c);
        if (cb != nullptr) {
          cb->set_checkpoint_dir(
            arg_parser.get<std::string>(LBANN_OPTION_CKPT_DIR));
          if (comm->am_trainer_master()) {
            std::cout << "Setting the checkpoint directory to "
                      << cb->get_checkpoint_dir() << std::endl;
          }
        }
      }
    }
  }
  if (arg_parser.get<std::string>(LBANN_OPTION_RESTART_DIR) != "") {
    for (auto&& c : global_trainer_->get_callbacks()) {
      {
        auto* const cb = dynamic_cast<callback::checkpoint*>(c);
        if (cb != nullptr) {
          cb->set_restart_dir(
            arg_parser.get<std::string>(LBANN_OPTION_RESTART_DIR));
          if (comm->am_trainer_master()) {
            std::cout << "Setting the restart directory to "
                      << cb->get_restart_dir() << std::endl;
          }
        }
      }
    }
  }

  // Root of the random seed tree
  int const root_random_seed =
    (pb_trainer->random_seed() ? pb_trainer->random_seed()
                               : lbann_default_random_seed);

  // Random seed used for the general RNGs
  int random_seed = root_random_seed;
  // Random seed used for the RNG used to fetch data
  int data_seq_random_seed = root_random_seed;

  // Initialize models differently if needed.
#ifndef LBANN_DETERMINISTIC
  if (!pb_trainer->random_init_trainers_identically()) {
    random_seed = hash_combine(random_seed, comm->get_trainer_rank());
    data_seq_random_seed = random_seed;
  }
#else
  if (comm->am_world_master()) {
    std::cout << std::string(116, '-') << '\n'
              << "ALERT: executing with LBANN_DETERMINISTIC flag to minimize "
                 "reduce numerical variance -- performance will be degraded\n"
              << std::string(116, '-') << std::endl;
  }
  if (!pb_trainer->random_init_trainers_identically()) {
    if (comm->am_trainer_master()) {
      std::cout << "WARNING: forcing 'random_init_trainers_identically' "
                << "due to sequential consistency" << std::endl;
    }
  }
#endif

  // Initialize the general RNGs and the data sequence RNGs
  int max_io_rng_banks = arg_parser.get<int>(LBANN_OPTION_MAX_IO_RNG_BANKS);
  // Create a set of RNG banks for both training and validation type phases
  init_random(random_seed, max_io_rng_banks * 2);
  init_data_seq_random(data_seq_random_seed);
  init_ltfb_random(root_random_seed);
  global_trainer_->set_random_seeds(root_random_seed,
                                    random_seed,
                                    data_seq_random_seed);

  bool const allow_global_statistics =
    arg_parser.get<bool>(LBANN_OPTION_ALLOW_MULTITRAINER_GLOBAL_STATISTICS);
  bool const multitrainer_verbose =
    arg_parser.get<bool>(LBANN_OPTION_MULTITRAINER_VERBOSE);
  std::vector<int> root_random_seeds;
  std::vector<int> random_seeds;
  std::vector<int> data_seq_random_seeds;
  if (allow_global_statistics && multitrainer_verbose) {
    // Collect everyone's random seeds
    root_random_seeds.resize(comm->get_procs_in_world());
    random_seeds.resize(comm->get_procs_in_world());
    data_seq_random_seeds.resize(comm->get_procs_in_world());
    comm->world_all_gather(root_random_seed, root_random_seeds);
    comm->world_all_gather(random_seed, random_seeds);
    comm->world_all_gather(data_seq_random_seed, data_seq_random_seeds);
  }
  else {
    // Collect random seeds from everyone in the trainer
    root_random_seeds.resize(procs_per_trainer);
    random_seeds.resize(procs_per_trainer);
    data_seq_random_seeds.resize(procs_per_trainer);
    comm->trainer_all_gather(root_random_seed, root_random_seeds);
    comm->trainer_all_gather(random_seed, random_seeds);
    comm->trainer_all_gather(data_seq_random_seed, data_seq_random_seeds);
  }

  // Update the sample lists to accomodate multi-trainer / multi-model
  // specification
  customize_data_readers_sample_list(*comm, pb);

  // Initialize data readers
  //@todo: code not in place for correctly handling image preprocessing
  std::map<execution_mode, generic_data_reader*> data_readers;
  init_data_readers(comm, pb, data_readers);

  global_trainer_->setup(std::move(io_thread_pool), data_readers);

  if (arg_parser.get<bool>(LBANN_OPTION_DISABLE_BACKGROUND_IO_ACTIVITY)) {
    global_trainer_->allow_background_io_activity(false);
  }

  // Create sub-grids in block order
  const int num_subgrids_block =
    arg_parser.get<int>(LBANN_OPTION_NUM_SUBGRIDS_BLOCK_ORDER);
  if (num_subgrids_block > 0) {

    // Check sub-grid size
    const int trainer_size = comm->get_procs_per_trainer();
    const int subgrid_size = trainer_size / num_subgrids_block;
    if (trainer_size != subgrid_size * num_subgrids_block) {
      LBANN_ERROR("attempted to divide a trainer grid with ",
                  trainer_size,
                  " processes ",
                  "into ",
                  num_subgrids_block,
                  " equally-sized sub-grids");
    }

    // Construct sub-grids
    std::vector<int> trainer_ranks(subgrid_size);
    for (int root_rank = 0; root_rank < trainer_size;
         root_rank += subgrid_size) {
      std::iota(trainer_ranks.begin(), trainer_ranks.end(), root_rank);
      El::mpi::Comm trainer_comm;
      El::mpi::Group trainer_group, subgrid_group;
      El::mpi::Dup(comm->get_trainer_comm(), trainer_comm);
      El::mpi::CommGroup(trainer_comm, trainer_group);
      El::mpi::Incl(trainer_group,
                    trainer_ranks.size(),
                    trainer_ranks.data(),
                    subgrid_group);
      global_trainer_->add_grid(make_unique<El::Grid>(std::move(trainer_comm),
                                                      subgrid_group,
                                                      subgrid_size,
                                                      El::COLUMN_MAJOR));
      El::mpi::Free(trainer_group);
    }
  }

  // Report useful information
  if (comm->am_world_master()) {
    print_lbann_configuration(comm, io_threads_per_process, io_threads_offset);
    std::cout << "\n" << global_trainer_->get_description() << std::endl;

    // User feedback
    print_parameters(*comm,
                     pb,
                     root_random_seeds,
                     random_seeds,
                     data_seq_random_seeds);
  }

  return *global_trainer_;
}

// Setup I/O thread pool that is shared across all models
std::unique_ptr<thread_pool> construct_io_thread_pool(lbann_comm* comm,
                                                      bool serialized_io)
{
  int max_io_threads = num_free_cores_per_process(comm);
  // Allow the trainer to override the command-line option or environment
  // variable
  if (serialized_io) {
    max_io_threads = 1;
  }

  auto& arg_parser = global_argument_parser();
  int req_io_threads = arg_parser.get<int>(LBANN_OPTION_NUM_IO_THREADS);
  int max_io_rng_banks = arg_parser.get<int>(LBANN_OPTION_MAX_IO_RNG_BANKS);
  // Limit the number of I/O threads to:
  //   < number of available free cores per process
  //   < number of RNG banks provisioned
  // and at least one
  int num_io_threads = std::max(
    std::min(max_io_rng_banks, std::min(max_io_threads, req_io_threads)),
    1);

  auto io_threads_offset = free_core_offset(comm);

  if (comm->am_world_master()) {
    std::cout << "\tNum. I/O Threads: " << num_io_threads
              << " (Limited to # Unused Compute Cores [" << max_io_threads
              << "] # of RNG banks [" << max_io_rng_banks
              << "] or 1) at offset " << io_threads_offset << std::endl;
  }

  auto io_thread_pool = std::make_unique<thread_pool>();
  io_thread_pool->launch_pinned_threads(num_io_threads, io_threads_offset);

  return io_thread_pool;
}

std::unique_ptr<model> build_model_from_prototext(
  int argc,
  char** argv,
  const lbann_data::Trainer* pb_trainer,
  lbann_data::LbannPB& pb,
  lbann_comm* comm,
  thread_pool& io_thread_pool,
  std::vector<std::shared_ptr<callback_base>>& shared_callbacks)
{

  bool master = comm->am_world_master();
  if (master) {
    std::cerr << "starting build_model_from_prototext" << std::endl;
  }

  std::ostringstream err;

  // Save info to file; this includes the complete prototext (with any
  // over-rides from the cmd line) and various other info
  save_session(*comm, argc, argv, pb);

  // Display how the OpenMP threads are provisioned
  auto& arg_parser = global_argument_parser();
  if (arg_parser.get<bool>(LBANN_OPTION_PRINT_AFFINITY)) {
    display_omp_setup();
  }

  // Initalize model
  std::unique_ptr<model> ret_model =
    proto::construct_model(comm, pb.optimizer(), pb.trainer(), pb.model());

  // Add the trainer's callbacks to the model
  for (auto&& c : shared_callbacks) {
    ret_model->add_callback(c);
  }

  // If the checkpoint directory has been overridden reset it before
  // setting up the model
  if (arg_parser.get<std::string>(LBANN_OPTION_CKPT_DIR) != "") {
    for (auto&& c : ret_model->get_callbacks()) {
      {
        auto* cb = dynamic_cast<callback::dump_weights*>(c);
        if (cb != nullptr) {
          cb->set_target_dir(
            arg_parser.get<std::string>(LBANN_OPTION_CKPT_DIR));
          if (comm->am_trainer_master()) {
            std::cout << "Setting the dump weights directory to "
                      << cb->get_target_dir() << std::endl;
          }
        }
      }
      {
        auto* cb = dynamic_cast<callback::save_model*>(c);
        if (cb != nullptr) {
          cb->set_target_dir(
            arg_parser.get<std::string>(LBANN_OPTION_CKPT_DIR));
          if (comm->am_trainer_master()) {
            std::cout << "Setting the dump weights directory to "
                      << cb->get_target_dir() << std::endl;
          }
        }
      }
    }
  }

  if (arg_parser.get<std::string>(LBANN_OPTION_LOAD_MODEL_WEIGHTS_DIR) != "") {
    callback::load_model* cb = nullptr;
    for (auto&& c : ret_model->get_callbacks()) {
      cb = dynamic_cast<callback::load_model*>(c);
      if (cb != nullptr) {
        break;
      }
    }

    std::string active_load_model_dir;
    std::string load_model_dir =
      arg_parser.get<std::string>(LBANN_OPTION_LOAD_MODEL_WEIGHTS_DIR);
    if (arg_parser.get<bool>(LBANN_OPTION_LOAD_MODEL_WEIGHTS_DIR_IS_COMPLETE)) {
      active_load_model_dir = load_model_dir;
    }
    else {
      size_t epochLast = std::numeric_limits<size_t>::max();
      ;
      size_t stepLast = std::numeric_limits<size_t>::max();
      ;
      execution_mode mode = execution_mode::invalid;
      visitor_hook hook = visitor_hook::invalid;
      active_load_model_dir =
        callback::get_last_shared_checkpoint_filename("sgd", load_model_dir);

      // get last epoch and step saved.
      int success = callback::read_latest(active_load_model_dir,
                                          &hook,
                                          &mode,
                                          &epochLast,
                                          &stepLast);
      if (!success) {
        LBANN_ERROR("Unable to find the latest checkpoint ",
                    active_load_model_dir);
        return nullptr;
      }
      active_load_model_dir =
        callback::get_shared_checkpoint_dirname("sgd",
                                                load_model_dir,
                                                hook,
                                                mode,
                                                epochLast,
                                                stepLast) +
        ret_model->get_name() + '/';
    }

    if (cb == nullptr) {
      std::vector<std::string> dirs = {active_load_model_dir};
      std::unique_ptr<callback::load_model> load_model_cb =
        std::make_unique<callback::load_model>(dirs);
      cb = load_model_cb.get();
      ret_model->add_callback(std::move(load_model_cb));
#ifdef LBANN_DEBUG
      if (comm->am_trainer_master()) {
        LBANN_WARNING(
          "command line flag --load_model_dir was provided but there was no "
          "explicit load_model callback, adding one automagically!");
      }
#endif
    }
    else {
      cb->add_dir(
        arg_parser.get<std::string>(LBANN_OPTION_LOAD_MODEL_WEIGHTS_DIR));
    }
  }

  // restart model from checkpoint if we have one
  //@todo
  // model->restartShared();

  return ret_model;
}

void print_lbann_configuration(lbann_comm* comm,
                               int io_threads_per_process,
                               int io_threads_offset)
{
  // Report hardware settings
  std::cout << "Hardware properties (for master process)" << std::endl
            << "  Processes on node          : " << comm->get_procs_per_node()
            << std::endl
            << "  Total number of processes  : " << comm->get_procs_in_world()
            << std::endl
            << "  OpenMP threads per process : " << omp_get_max_threads()
            << std::endl
            << "  I/O threads per process (+offset) : "
            << io_threads_per_process << " (+" << io_threads_offset << ")"
            << std::endl
            << "  Background I/O enabled     : "
            << get_trainer().background_io_activity_allowed() << std::endl;
#ifdef HYDROGEN_HAVE_GPU
  std::cout << "  GPUs on node               : " << hydrogen::gpu::DeviceCount()
            << std::endl;
#endif // HYDROGEN_HAVE_GPU
  std::cout << std::endl;

  // Report build settings
  std::cout << "Running: LLNL LBANN version: " << LBANN_MAKE_STR(LBANN_VERSION)
#ifdef LBANN_GIT_VERSION
            << " (" << LBANN_MAKE_STR(LBANN_GIT_VERSION) << ")"
#endif
            << std::endl;
#ifdef HYDROGEN_VERSION
  std::cout << "         LLNL Hydrogen version: " << HYDROGEN_VERSION
#ifdef HYDROGEN_GIT_VERSION
            << " (" << HYDROGEN_GIT_VERSION << ")"
#endif
            << std::endl;
#endif
#ifdef DIHYDROGEN_VERSION
  std::cout << "         LLNL DiHydrogen version: " << DIHYDROGEN_VERSION
#ifdef DIHYDROGEN_GIT_VERSION
            << " (" << DIHYDROGEN_GIT_VERSION << ")"
#endif
            << std::endl;
#endif
#ifdef AL_VERSION
  std::cout << "         LLNL Aluminum version: " << AL_VERSION
#ifdef AL_GIT_VERSION
            << " (" << AL_GIT_VERSION << ")"
#endif
            << std::endl;
#endif
  std::cout << std::endl;

  std::cout << "Build settings" << std::endl;
  std::cout << "  Type     : ";
#ifdef LBANN_DEBUG
  std::cout << "Debug" << std::endl;
#else
  std::cout << "Release" << std::endl;
#endif // LBANN_DEBUG
  std::cout << "  Aluminum : ";
#ifdef LBANN_HAS_ALUMINUM
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_ALUMINUM
  std::cout << "  GPU      : ";
#ifdef LBANN_HAS_GPU
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_GPU
  std::cout << "  cuDNN    : ";
#ifdef LBANN_HAS_CUDNN
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // LBANN_HAS_CUDNN
  std::cout << "  CUB      : ";
#ifdef HYDROGEN_HAVE_CUB
  std::cout << "detected" << std::endl;
#else
  std::cout << "NOT detected" << std::endl;
#endif // HYDROGEN_HAVE_CUB
  {
    const auto* env = std::getenv("MV2_USE_CUDA");
    std::cout << "  MV2_USE_CUDA : " << (env != nullptr ? env : "") << std::endl;
    std::cout << std::endl;
  }

#ifdef LBANN_HAS_ROCM
  std::cout << "  MIOpen DB Cache : " << std::endl;
  const auto* env_db = std::getenv("MIOPEN_USER_DB_PATH");
  std::cout << "    MIOPEN_USER_DB_PATH : " << (env_db != nullptr ? env_db : "")
            << std::endl;
  const auto* env_cache = std::getenv("MIOPEN_CUSTOM_CACHE_DIR");
  std::cout << "    MIOPEN_CUSTOM_CACHE_DIR : "
            << (env_cache != nullptr ? env_cache : "") << std::endl;
#endif // LBANN_HAS_ROCM

#ifdef LBANN_HAS_DIHYDROGEN
  std::cout << "DiHydrogen Features:" << std::endl;
  std::cout << "  DaCe : ";
#if H2_HAS_DACE
  std::cout << "enabled" << std::endl;
  {
    const auto* env = std::getenv("DISTCONV_JIT_CACHEPATH");
    std::cout << "  DISTCONV_JIT_CACHEPATH : " << (env != nullptr ? env : "N/A") << std::endl;
    std::cout << std::endl;
  }
#else
  std::cout << "disabled" << std::endl;
#endif // H2_HAS_DACE
  std::cout << std::endl;
#endif // LBANN_HAS_DIHYDROGEN

#ifdef LBANN_HAS_ALUMINUM
  std::cout << "Aluminum Features:" << std::endl;
  std::cout << "  NCCL : ";
#ifdef AL_HAS_NCCL
  std::cout << "enabled" << std::endl;
#else
  std::cout << "disabled" << std::endl;
#endif // AL_HAS_NCCL
  std::cout << std::endl;
#endif // LBANN_HAS_ALUMINUM

  // Report model settings
  const auto& grid = comm->get_trainer_grid();
  std::cout << "Trainer settings" << std::endl
            << "  Trainers              : " << comm->get_num_trainers()
            << std::endl
            << "  Processes per trainer : " << comm->get_procs_per_trainer()
            << std::endl
            << "  Grid dimensions       : " << grid.Height() << " x "
            << grid.Width() << std::endl;
  std::cout << std::endl;
}

} // namespace lbann
