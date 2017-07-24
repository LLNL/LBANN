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

#ifndef LBANN_LAYERS_INPUT_LAYER_PARTITIONED_MINIBATCH_HPP_INCLUDED
#define LBANN_LAYERS_INPUT_LAYER_PARTITIONED_MINIBATCH_HPP_INCLUDED

#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/data_distributions/partitioned_minibatch.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {

template <data_layout T_layout = data_layout::DATA_PARALLEL>
class input_layer_partitioned_minibatch : public input_layer, public partitioned_minibatch {
 public:
  /// @todo make the map and vector references
  input_layer_partitioned_minibatch(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers)
    : input_layer(comm, num_parallel_readers, data_readers),
      partitioned_minibatch(comm, std::min(num_parallel_readers, Layer::m_comm->get_procs_per_model()), data_readers) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "partitioned_minibatch only supports DATA_PARALLEL");
    // Setup the data distribution
    initialize_distributed_matrices();
  }
  input_layer_partitioned_minibatch* copy() const {
    throw lbann_exception("Cannot copy input_layer_partitioned_minibatch");
    return nullptr;
  }

  std::string get_name() const { return "input:partitioned"; }

  virtual inline void initialize_distributed_matrices() {
    input_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_data() {
    input_layer::setup_data();
    int max_mb_size = this->m_neural_network_model->get_max_mini_batch_size();
    if(io_layer::m_data_sets_span_models) {
      int base_offset = Layer::m_comm->get_rank_in_model();
      int batch_stride = Layer::m_comm->get_num_models() * max_mb_size;
      int model_offset = Layer::m_comm->get_model_rank() * max_mb_size;
      //cout << "["<< Layer::m_comm->get_rank_in_world() << "] Setting up input layer, with " << Layer::m_comm->get_num_models() << " models and " << m_num_parallel_readers_training << " parallel readers and " << max_mb_size << " mb size, which gives a stride of " << batch_stride << " and my model offset is " << model_offset << " and my base offset is " << base_offset /*(Layer::m_comm->get_rank_in_model() * max_mb_size)*/ << endl;
      io_layer::setup_data_readers_for_training(base_offset,
                                                          batch_stride,
                                                          partitioned_minibatch::m_num_parallel_readers_training,
                                                          model_offset);
      partitioned_minibatch::calculate_num_iterations_per_epoch_spanning_models(max_mb_size,
                                                                               this->m_training_dataset.data_reader);
      /// Note that the data readers for evaluation should not be partitioned over multiple models (otherwise each model will be scored on a different set of data)
      io_layer::setup_data_readers_for_evaluation(Layer::m_comm->get_rank_in_model(),
                                                  max_mb_size,
                                                  partitioned_minibatch::m_num_parallel_readers_testing);
      partitioned_minibatch::calculate_num_iterations_per_epoch_single_model(max_mb_size,
                                                                             this->m_validation_dataset.data_reader);
      partitioned_minibatch::calculate_num_iterations_per_epoch_single_model(max_mb_size, 
                                                                             this->m_testing_dataset.data_reader);
    } else {
      io_layer::setup_data_readers_for_training(Layer::m_comm->get_rank_in_model(),
                                                          max_mb_size,
                                                          partitioned_minibatch::m_num_parallel_readers_training);
      io_layer::setup_data_readers_for_evaluation(Layer::m_comm->get_rank_in_model(),
                                                            max_mb_size,
                                                            partitioned_minibatch::m_num_parallel_readers_testing);
    }

    partitioned_minibatch::m_local_data_valid = false;
    partitioned_minibatch::m_local_reader_done = false;
    partitioned_minibatch::m_num_data_per_epoch = 0;
  }

  void fp_compute() {
    //  generic_data_reader *data_reader = input_layer::select_data_reader();
    //int num_parallel_readers = get_num_parallel_readers();

    //  DISPLAY_MATRIX(m_activations);
    int num_samples_fetched = partitioned_minibatch::fetch_to_local_matrix(this->m_activations->Matrix());

    // Use the predetermined size of the mini-batch to set the current
    // batch size for the neural network
    int num_samples_in_batch = partitioned_minibatch::get_current_mini_batch_size();

    input_layer::update_num_samples_processed(num_samples_in_batch);

    /// Let each rank know this size of the current mini-batch
    /// Note that this field has to be updated before distributing the data
    this->m_neural_network_model->set_current_mini_batch_size(num_samples_in_batch);
  }

  /**
   * Once a mini-batch is processed, resuffle the data for the next batch if necessary
   */
  bool update_compute() {
    return partitioned_minibatch::is_data_set_processed();
  }


  int fetch_from_data_reader(Mat& M_local) {
    generic_data_reader *data_reader = input_layer::select_data_reader();
    return data_reader->fetch_data(M_local);
  }

  void preprocess_data_samples(Mat& M_local, int num_samples_in_batch) {
    return;
  }

  bool update_data_reader() {
    generic_data_reader *data_reader = input_layer::select_data_reader();
    return data_reader->update();
  }

  execution_mode get_execution_mode() {
    return this->m_execution_mode;
  }
};

}

#endif  // LBANN_LAYERS_INPUT_LAYER_PARTITIONED_MINIBATCH_HPP_INCLUDED
