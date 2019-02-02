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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_store/data_store_jag.hpp"

#ifdef LBANN_HAS_CONDUIT

#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/models/model.hpp"
#include <unordered_set>


namespace lbann {

data_store_jag::data_store_jag(
  generic_data_reader *reader, model *m) :
  generic_data_store(reader, m), 
  m_ds_indices_were_exchanged(false) {
  set_name("data_store_jag");
}

data_store_jag::~data_store_jag() {
}

void data_store_jag::setup() {
  double tm1 = get_time();
  std::stringstream err;

  if (m_master) {
    std::cout << "starting data_store_jag::setup() for role: " << m_reader->get_role() << "\n";
  }

  // I suspect we'll never go out-of-memory ...
  if (! m_in_memory) {
    LBANN_ERROR("out-of-memory mode for data_store_jag has not been implemented");
  }

  generic_data_store::setup();

  //sanity check
  if (m_master) {
    std::cout << "num shuffled_indices: " << m_shuffled_indices->size() << "\n";
  }

  data_reader_jag_conduit *jag_reader = dynamic_cast<data_reader_jag_conduit*>(m_reader);
  if (jag_reader == nullptr) {
    LBANN_ERROR(" dynamic_cast<data_reader_jag_conduit*>(m_reader) failed");
  }

  /// m_all_minibatch_indices[j] will contain all indices that
  //  will be passed to data_reader::fetch_datum in one epoch
  build_all_minibatch_indices();

  // allocate buffers that are used in exchange_data()
  m_send_buffer.resize(m_np);
  m_send_buffer_2.resize(m_np);
  m_send_requests.resize(m_np);
  m_recv_requests.resize(m_np);
  m_status.resize(m_np);
  m_outgoing_msg_sizes.resize(m_np);
  m_incoming_msg_sizes.resize(m_np);
  m_recv_buffer.resize(m_np);

  if (m_master) {
    std::cout << "TIME for data_store_jag setup: " << get_time() - tm1 << "\n";
  }
}

// this gets called at the beginning of each epoch (except for epoch 0)
//
// Note: conduit has a very nice interface for communicating nodes
//       in non-blocking scenarios. Unf, for blocking we need to
//       handle things ourselves. TODO: possible modify conduit to
//       handle non-blocking comms
void data_store_jag::exchange_data() {
  double tm1 = get_time();

  if (m_master) std::cerr << "starting exchange_data; epoch: "<<m_model->get_cur_epoch()<< " data size: "<<m_data.size()<<"\n";

  if (! m_ds_indices_were_exchanged) {
    if (m_master) std::cerr << "calling exchange_ds_indices()\n";
      // fills in m_owner; this maps a sample index to the owning processor
      // Also fill in m_my_datastore_indices, which is the set of indices
      // (and associated samples) that I own
      exchange_ds_indices();
      m_ds_indices_were_exchanged = true;
  }

  //========================================================================
  //build map: proc -> global indices that P_x needs for this epoch, and
  //                   which I own
  std::vector<std::unordered_set<int>> proc_to_indices(m_np);
  for (size_t j=0; j<m_all_minibatch_indices.size(); j++) {
    for (auto idx : m_all_minibatch_indices[j]) {
      int index = (*m_shuffled_indices)[idx];
      // P_j needs the sample that corresponds to 'index' in order
      // to complete the next epoch
      if (m_my_datastore_indices.find(index) != m_my_datastore_indices.end()) {
        proc_to_indices[j].insert(index);
      }
    }
  }

  if (m_master) std::cout << "exchange_data; built map\n";
  MPI_Barrier(MPI_COMM_WORLD);

  //========================================================================
  //part 1: exchange the sizes of the data

  // m_send_buffer[j] is a conduit::Node that contains
  // all samples that this proc will send to P_j
  for (int p=0; p<m_np; p++) {
    m_send_buffer[p].reset();
    for (auto idx : proc_to_indices[p]) {
      m_send_buffer[p][std::to_string(idx)] = m_data[idx];
    }

    // code in the following method is a modification of code from
    // conduit/src/libs/relay/conduit_relay_mpi.cpp
    build_node_for_sending(m_send_buffer[p], m_send_buffer_2[p]);

    m_outgoing_msg_sizes[p] = m_send_buffer_2[p].total_bytes_compact();
    MPI_Isend((void*)&m_outgoing_msg_sizes[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &m_send_requests[p]);
  }

  //start receives for sizes of the data
  for (int p=0; p<m_np; p++) {
    MPI_Irecv((void*)&m_incoming_msg_sizes[p], 1, MPI_INT, p, 0, MPI_COMM_WORLD, &m_recv_requests[p]);
  }

  // wait for all msgs to complete
  MPI_Waitall(m_np, m_send_requests.data(), m_status.data());
  MPI_Waitall(m_np, m_recv_requests.data(), m_status.data());

  //========================================================================
  //part 2: exchange the actual data

  // start sends for outgoing data
  for (int p=0; p<m_np; p++) {
    const void *s = m_send_buffer_2[p].data_ptr();
    MPI_Isend(s, m_outgoing_msg_sizes[p], MPI_BYTE, p, 1, MPI_COMM_WORLD, &m_send_requests[p]);
  }

  // start recvs for incoming data
  for (int p=0; p<m_np; p++) {
    m_recv_buffer[p].set(conduit::DataType::uint8(m_incoming_msg_sizes[p]));
    MPI_Irecv(m_recv_buffer[p].data_ptr(), m_incoming_msg_sizes[p], MPI_BYTE, p, 1, MPI_COMM_WORLD, &m_recv_requests[p]);
  }

  // wait for all msgs to complete
  MPI_Waitall(m_np, m_send_requests.data(), m_status.data());
  MPI_Waitall(m_np, m_recv_requests.data(), m_status.data());

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

  conduit::Node nd;
  m_minibatch_data.clear();
  for (int p=0; p<m_np; p++) {
    conduit::uint8 *n_buff_ptr = (conduit::uint8*)m_recv_buffer[p].data_ptr();
    conduit::Node n_msg;
    n_msg["schema_len"].set_external((conduit::int64*)n_buff_ptr);
    n_buff_ptr +=8;
    n_msg["schema"].set_external_char8_str((char*)(n_buff_ptr));
    conduit::Schema rcv_schema;
    conduit::Generator gen(n_msg["schema"].as_char8_str());
    gen.walk(rcv_schema);
    n_buff_ptr += n_msg["schema"].total_bytes_compact();
    n_msg["data"].set_external(rcv_schema,n_buff_ptr);
    nd.reset();
    nd.update(n_msg["data"]);
    const std::vector<std::string> &names = nd.child_names();
    for (auto t : names) {
      conduit::Node n3 = nd[t];
      m_minibatch_data[atoi(t.c_str())] = n3;
    }
  }

  if (m_master) std::cout << "data_store_jag::exchange_data time: " << get_time() - tm1 << "\n";
}

void data_store_jag::set_conduit_node(int data_id, conduit::Node &node) {
  if (m_data.find(data_id) != m_data.end()) {
    LBANN_ERROR("duplicate data_id: " + std::to_string(data_id) + " in data_store_jag::set_conduit_node");
  }
  m_data[data_id] = node;
}

const conduit::Node & data_store_jag::get_conduit_node(int data_id, bool any_node) const {
  if (any_node) {
    LBANN_ERROR("data_store_jag::get_conduit_node called with any_node = true; this is not yet functional; please contact Dave Hysom");
  }

  std::unordered_map<int, conduit::Node>::const_iterator t = m_minibatch_data.find(data_id);
  if (t == m_minibatch_data.end()) {
    LBANN_ERROR("failed to find data_id: " + std::to_string(data_id) + " in m_minibatch_data; m_minibatch_data.size: " + std::to_string(m_minibatch_data.size()) + "; epoch:"  + std::to_string(m_model->get_cur_epoch()));
  }

  return t->second;
}

void data_store_jag::build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out) {

  conduit::Schema s_data_compact;
  if( node_in.is_compact() && node_in.is_contiguous()) {
    s_data_compact = node_in.schema();
  } else {
    node_in.schema().compact_to(s_data_compact);
  }

  std::string snd_schema_json = s_data_compact.to_json();

  conduit::Schema s_msg;
  s_msg["schema_len"].set(conduit::DataType::int64());
  s_msg["schema"].set(conduit::DataType::char8_str(snd_schema_json.size()+1));
  s_msg["data"].set(s_data_compact);

  conduit::Schema s_msg_compact;
  s_msg.compact_to(s_msg_compact);
  node_out.reset();
  node_out.set(s_msg_compact);
  node_out["schema"].set(snd_schema_json);
  node_out["data"].update(node_in);
}


// fills in m_owner; this maps a sample index to the owning processor
// Also fill in m_my_datastore_indices, which is the set of indices that I own
void data_store_jag::exchange_ds_indices() {
  m_my_datastore_indices.clear();
  int my_num_indices = m_data.size();
  std::vector<int> counts(m_np);
  m_comm->trainer_all_gather<int>(my_num_indices, counts);

  std::vector<int> displ(m_np);
  displ[0] = 0;
  for (size_t j=1; j<counts.size(); j++) {
    displ[j] = displ[j-1] + counts[j-1];
  }

  //recv vector
  int n = std::accumulate(counts.begin(), counts.end(), 0);
  std::vector<int> all_indices(n);

  std::vector<int> mine;
  mine.reserve(m_data.size());
  for (auto t : m_data) {
    mine.push_back(t.first);
  }

  //receive the indices
  m_comm->all_gather<int>(mine, all_indices, counts, displ, m_comm->get_trainer_comm());

  //fill in the final data structure
  m_owner.clear();
  for (int proc=0; proc<m_np; proc++) {
    for (int i=displ[proc]; i<displ[proc]+counts[proc]; i++) {
      if (m_owner.find(all_indices[i]) != m_owner.end()) {
        LBANN_ERROR("duplicate index in m_owner");
      }
      m_owner[all_indices[i]] = proc;
      if (proc == m_rank) {
        m_my_datastore_indices.insert(all_indices[i]);
      }
    }
  }
}

void data_store_jag::build_all_minibatch_indices() {
  m_all_minibatch_indices.clear();
  m_all_minibatch_indices.resize(m_np);
  for (size_t idx=0; idx<m_shuffled_indices->size(); ++idx) {
    int owner = idx % m_np;
    m_all_minibatch_indices[owner].push_back(idx);
  }
}

}  // namespace lbann

#endif //#ifdef LBANN_HAS_CONDUIT
