////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/data_store/data_store_conduit.hpp"

#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/data_readers/data_reader_image.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/options.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/file_utils.hpp"
#include <unordered_set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <unistd.h>
#include <sys/statvfs.h>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>


namespace lbann {

data_store_conduit::data_store_conduit(
  generic_data_reader *reader) :
  m_reader(reader) {

  m_comm = m_reader->get_comm();
  if (m_comm == nullptr) {
    LBANN_ERROR(" m_comm is nullptr");
  }

  m_world_master = m_comm->am_world_master();
  m_trainer_master = m_comm->am_trainer_master();
  m_rank_in_trainer = m_comm->get_rank_in_trainer();
  m_rank_in_world = m_comm->get_rank_in_world();
  m_np_in_trainer = m_comm->get_procs_per_trainer();

  options *opts = options::get();

  if (opts->get_bool("debug")) {
    std::stringstream ss;
    ss << "debug_" << m_reader->get_role() << "." << m_comm->get_rank_in_world();
    m_output = new std::ofstream(ss.str().c_str());
    m_debug_filename = ss.str();
    if (m_world_master) {
      std::cout << "opened " << ss.str() << " for writing\n";
    }
  }

  if (opts->has_string("data_store_spill")) {
    m_spill_dir_base = opts->get_string("data_store_spill");
    make_dir_if_it_doesnt_exist(m_spill_dir_base);
    m_comm->trainer_barrier();
    m_spill_dir_base = m_spill_dir_base + "/conduit_" + std::to_string(m_rank_in_world);
    make_dir_if_it_doesnt_exist(m_spill_dir_base);
    m_cur_spill_dir = -1;
    m_num_files_in_cur_spill_dir = m_max_files_per_directory;
  }

  m_is_local_cache = opts->get_bool("data_store_cache");
  if (m_is_local_cache && !opts->get_bool("preload_data_store")) {
    LBANN_ERROR("data_store_cache is currently only implemented for preload mode; this will change in the future. For now, pleas pass both flags: data_store_cache and --preload_data_store");
  }

  if (m_world_master) {
    if (m_is_local_cache) {
      std::cout << "data_store_conduit is running in local_cache mode\n";
    } else {
      std::cout << "data_store_conduit is running in multi-message mode\n";
    }
  }
}

data_store_conduit::~data_store_conduit() {
  if (m_output) {
    m_output->close();
  }
  if (m_is_local_cache && m_mem_seg) {
    int sanity = shm_unlink(m_seg_name.c_str());
    if (sanity != 0) {
      std::cout << "\nWARNING: shm_unlink failed in data_store_conduit::~data_store_conduit()\n";
    }
    sanity = munmap(reinterpret_cast<void*>(m_mem_seg), m_mem_seg_length);
    if (sanity != 0) {
      std::cout << "\nWARNING: munmap failed in data_store_conduit::~data_store_conduit()\n";
    }
  }
}

data_store_conduit::data_store_conduit(const data_store_conduit& rhs) {
  copy_members(rhs);
}

data_store_conduit::data_store_conduit(const data_store_conduit& rhs, const std::vector<int>& ds_sample_move_list) {
  copy_members(rhs, ds_sample_move_list);
}

data_store_conduit& data_store_conduit::operator=(const data_store_conduit& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  copy_members(rhs);
  return (*this);
}

void data_store_conduit::set_data_reader_ptr(generic_data_reader *reader) { 
  m_reader = reader; 
  if (options::get()->get_bool("debug")) {
    std::stringstream ss;
    ss << "debug_" << m_reader->get_role() << "." << m_comm->get_rank_in_world();
    m_output = new std::ofstream(ss.str().c_str());
    m_debug_filename = ss.str();
    if (m_world_master) {
      std::cout << "data_store_conduit::set_data_reader_ptr; opened " << ss.str() << " for writing\n";
    }
  }
}

void data_store_conduit::copy_members(const data_store_conduit& rhs, const std::vector<int>& ds_sample_move_list) {
  m_is_setup = rhs.m_is_setup;
  m_preload = rhs.m_preload;
  m_explicit_loading = rhs.m_explicit_loading;
  m_owner_map_mb_size = rhs.m_owner_map_mb_size;
  m_compacted_sample_size = rhs.m_compacted_sample_size;
  m_is_local_cache = rhs.m_is_local_cache;
  m_node_sizes_vary = rhs.m_node_sizes_vary;
  m_have_sample_sizes = rhs.m_have_sample_sizes;
  m_reader = rhs.m_reader;
  m_comm = rhs.m_comm;
  m_world_master = rhs.m_world_master;
  m_trainer_master = rhs.m_trainer_master;
  m_rank_in_trainer = rhs.m_rank_in_trainer;
  m_np_in_trainer = rhs.m_np_in_trainer;
  m_owner = rhs.m_owner;
  m_shuffled_indices = rhs.m_shuffled_indices;
  m_sample_sizes = rhs.m_sample_sizes;
  m_mem_seg = rhs.m_mem_seg;
  m_mem_seg_length = rhs.m_mem_seg_length;
  m_seg_name = rhs.m_seg_name;
  m_image_offsets = rhs.m_image_offsets;
  if (m_output) {
    LBANN_ERROR("m_output should be nullptr");
  }

  /// This block needed when carving a validation set from the training set
  //if (options::get()->get_bool("debug") && !m_output) {
  if(ds_sample_move_list.size() == 0) {
    m_data = rhs.m_data;
  } else {
    /// Move indices on the list from the data and owner maps in the RHS data store to the new data store
    for(auto&& i : ds_sample_move_list) {

      if(rhs.m_data.find(i) != rhs.m_data.end()){
        /// Repack the nodes because they don't seem to copy correctly
        //
        //dah - previously this code block only contained the line:
        //  build_node_for_sending(rhs.m_data[i]["data"], m_data[i]);
        //However, this resulted in errors in the schema; not sure why,
        //as it used to work; some change in the conduit library?
        conduit::Node n2;
        const std::vector<std::string> &names = rhs.m_data[i]["data"].child_names();
        const std::vector<std::string> &names2 = rhs.m_data[i]["data"][names[0]].child_names();
        for (auto t : names2) {
          n2[names[0]][t] = rhs.m_data[i]["data"][names[0]][t];
        }
        build_node_for_sending(n2, m_data[i]);
      }
      rhs.m_data.erase(i);

      /// Removed migrated nodes from the original data store's owner list
      if(rhs.m_owner.find(i) != rhs.m_owner.end()) {
        m_owner[i] = rhs.m_owner[i];
        rhs.m_owner.erase(i);
      }
    }
  }

  /// Clear the pointer to the data reader, this cannot be copied
  m_reader = nullptr;
  m_shuffled_indices = nullptr;

  //these will probably zero-length, but I don't want to make assumptions
  //as to state when copy_member is called
  m_minibatch_data = rhs.m_minibatch_data;
  m_send_buffer = rhs.m_send_buffer;
  m_send_buffer_2 = rhs.m_send_buffer_2;
  m_send_requests = rhs.m_send_requests;
  m_recv_requests = rhs.m_recv_requests;
  m_recv_buffer = rhs.m_recv_buffer;
  m_outgoing_msg_sizes = rhs.m_outgoing_msg_sizes;
  m_incoming_msg_sizes = rhs.m_incoming_msg_sizes;
  m_compacted_sample_size = rhs.m_compacted_sample_size;
  m_indices_to_send = rhs.m_indices_to_send;
  m_indices_to_recv = rhs.m_indices_to_recv;
}

void data_store_conduit::setup(int mini_batch_size) {
  if (m_world_master) {
    std::cout << "starting data_store_conduit::setup() for role: " << m_reader->get_role() << "\n";
    if (m_is_local_cache) {
      std::cout << "data store mode: local cache\n";
    } else {
      std::cout << "data store mode: exchange_data via individual samples\n";
    }
  }

  double tm1 = get_time();
  m_owner_map_mb_size = mini_batch_size;

  m_is_setup = true;

  if (m_is_local_cache && m_preload) {
    preload_local_cache();
  }

  if (m_world_master) {
    std::cout << "TIME for data_store_conduit setup: " << get_time() - tm1 << "\n";
  }

}

void data_store_conduit::setup_data_store_buffers() {
  // allocate buffers that are used in exchange_data()
  m_send_buffer.resize(m_np_in_trainer);
  m_send_buffer_2.resize(m_np_in_trainer);
  m_send_requests.resize(m_np_in_trainer);
  m_recv_requests.resize(m_np_in_trainer);
  m_outgoing_msg_sizes.resize(m_np_in_trainer);
  m_incoming_msg_sizes.resize(m_np_in_trainer);
  m_recv_buffer.resize(m_np_in_trainer);
}

void data_store_conduit::set_preloaded_conduit_node(int data_id, const conduit::Node &node) {
  // note: at this point m_data[data_id] = node
  if (m_output) {
    (*m_output) << "set_preloaded_conduit_node: " << data_id << std::endl;
  }
  m_mutex.lock();
  conduit::Node n2 = node;
  build_node_for_sending(n2, m_data[data_id]);
  m_mutex.unlock();
  if (!m_node_sizes_vary) {
    error_check_compacted_node(m_data[data_id], data_id);
  } else {
    m_mutex.lock();
    m_sample_sizes[data_id] = m_data[data_id].total_bytes_compact();
    m_mutex.unlock();
  }
}

void data_store_conduit::error_check_compacted_node(const conduit::Node &nd, int data_id) {
  if (m_compacted_sample_size == 0) {
    m_compacted_sample_size = nd.total_bytes_compact();
    if (m_world_master) {
      std::cout << "num bytes for nodes to be transmitted: " << nd.total_bytes_compact() << " per node" << std::endl;
    }
  } else if (m_compacted_sample_size != nd.total_bytes_compact() && !m_node_sizes_vary) {
    LBANN_ERROR("Conduit node being added data_id: ", data_id,
                " is not the same size as existing nodes in the data_store ",
                m_compacted_sample_size, " != ", nd.total_bytes_compact(),
                " role: ", m_reader->get_role());
  }
  if (!nd.is_contiguous()) {
    LBANN_ERROR("m_data[",  data_id, "] does not have a contiguous layout");
  }
  if (nd.data_ptr() == nullptr) {
    LBANN_ERROR("m_data[", data_id, "] does not have a valid data pointer");
  }
  if (nd.contiguous_data_ptr() == nullptr) {
    LBANN_ERROR("m_data[", data_id, "] does not have a valid contiguous data pointer");
  }
}


void data_store_conduit::set_conduit_node(int data_id, conduit::Node &node, bool already_have) {

  if (m_output) {
    (*m_output) << "set_conduit_node: " << data_id << std::endl;
  }

  if (m_is_local_cache && m_preload) {
    LBANN_ERROR("you called data_store_conduit::set_conduit_node, but you're running in local cache mode with preloading; something is broken; please contact Dave Hysom");
  }

  m_mutex.lock();
  if (already_have == false && m_data.find(data_id) != m_data.end()) {
    m_mutex.unlock();
    LBANN_ERROR("duplicate data_id: ", data_id, " in data_store_conduit::set_conduit_node; role: ", m_reader->get_role());
  }
  m_mutex.unlock();

  if (already_have && is_local_cache()) {
    m_mutex.lock();
    if (m_data.find(data_id) == m_data.end()) {
      m_mutex.unlock();
      LBANN_ERROR("you claim the passed node was obtained from this data_store, but the data_id (", data_id, ") doesn't exist in m_data");
    }
    m_mutex.unlock();
    return;
  }

  if (is_local_cache()) {
    m_mutex.lock();
    m_data[data_id] = node;
    m_mutex.unlock();
  }

  else {
    if (m_spill) {
      conduit::Node n2;
      build_node_for_sending(node, n2);
      error_check_compacted_node(n2, data_id);
      m_mutex.lock();
      m_sample_sizes[data_id] = n2.total_bytes_compact();
      m_mutex.unlock();
      spill_conduit_node(node, data_id);
      m_spilled_nodes[data_id] = m_cur_spill_dir;
    }

    else {
      m_mutex.lock();
      build_node_for_sending(node, m_data[data_id]);
      error_check_compacted_node(m_data[data_id], data_id);
      m_sample_sizes[data_id] = m_data[data_id].total_bytes_compact();
      m_mutex.unlock();
    }  
  }
}

const conduit::Node & data_store_conduit::get_conduit_node(int data_id) const {
  if (m_output) {
    (*m_output) << "get_conduit_node: " << data_id << std::endl;
  }

  if (is_local_cache()) {
    std::unordered_map<int, conduit::Node>::const_iterator t3 = m_data.find(data_id);
    if (t3 == m_data.end()) {
      LBANN_ERROR("(local cache) failed to find data_id: ", data_id, " in m_data; m_data.size: ", m_data.size());
    }
    return t3->second;
  }

  std::unordered_map<int, conduit::Node>::const_iterator t2 = m_minibatch_data.find(data_id);
  // if not preloaded, and get_label() or get_response() is called,
  // we need to check m_data
  if (t2 == m_minibatch_data.end()) {
    std::unordered_map<int, conduit::Node>::const_iterator t3 = m_data.find(data_id);
    if (t3 != m_data.end()) {
      return t3->second["data"];
    }
    LBANN_ERROR("failed to find data_id: ", data_id, " in m_minibatch_data; m_minibatch_data.size: ", m_minibatch_data.size(), " and also failed to find it in m_data; m_data.size: ", m_data.size(), "; role: ", m_reader->get_role());
    if (m_output) {
      (*m_output) << "failed to find data_id: " << data_id << " in m_minibatch_data; my m_minibatch_data indices: ";
      for (auto t : m_minibatch_data) {
        (*m_output) << t.first << " ";
      }
      (*m_output) << std::endl;
    }
  }

  return t2->second;
}

// code in the following method is a modification of code from
// conduit/src/libs/relay/conduit_relay_mpi.cpp
void data_store_conduit::build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out) {
  node_out.reset();
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

  if(!node_out.is_contiguous()) {
    LBANN_ERROR("node_out does not have a contiguous layout");
  }
  if(node_out.data_ptr() == nullptr) {
    LBANN_ERROR("node_out does not have a valid data pointer");
  }
  if(node_out.contiguous_data_ptr() == nullptr) {
    LBANN_ERROR("node_out does not have a valid contiguous data pointer");
  }
}

void data_store_conduit::exchange_data_by_sample(size_t current_pos, size_t mb_size) {
  if (! m_is_setup) {
    LBANN_ERROR("setup(mb_size) has not been called");
  }

  /// exchange sample sizes if they are non-uniform (imagenet);
  /// this will only be called once, during the first call to
  /// exchange_data_by_sample at the beginning of the 2nd epoch,
  /// or during the first call th exchange_data_by_sample() during
  /// the first epoch if preloading
  if (m_node_sizes_vary && !m_have_sample_sizes) {
    exchange_sample_sizes();
  }

  if (m_output) {
    (*m_output) << "starting data_store_conduit::exchange_data_by_sample; mb_size: " << mb_size << std::endl;
  }

  int num_send_req = build_indices_i_will_send(current_pos, mb_size);
  if (m_spill) {
    load_spilled_conduit_nodes();
  }

  int num_recv_req = build_indices_i_will_recv(current_pos, mb_size);

  m_send_requests.resize(num_send_req);
  m_recv_requests.resize(num_recv_req);
  m_recv_buffer.resize(num_recv_req);
  m_recv_data_ids.resize(num_recv_req);

  //========================================================================
  //part 2: exchange the actual data

  // start sends for outgoing data
  size_t ss = 0;
  for (int p=0; p<m_np_in_trainer; p++) {
    const std::unordered_set<int> &indices = m_indices_to_send[p];
    for (auto index : indices) {
      if (m_data.find(index) == m_data.end()) {
        LBANN_ERROR("failed to find data_id: ", index, " to be sent to ", p, " in m_data");
      }
      const conduit::Node& n = m_data[index];
      const El::byte *s = reinterpret_cast<const El::byte*>(n.data_ptr());
      if(!n.is_contiguous()) {
        LBANN_ERROR("data_id: ", index, " does not have a contiguous layout");
      }
      if(n.data_ptr() == nullptr) {
        LBANN_ERROR("data_id: ", index, " does not have a valid data pointer");
      }
      if(n.contiguous_data_ptr() == nullptr) {
        LBANN_ERROR("data_id: ", index, " does not have a valid contiguous data pointer");
      }

      size_t sz = m_compacted_sample_size;

      if (m_node_sizes_vary) {
        if (m_sample_sizes.find(index) == m_sample_sizes.end()) {
          LBANN_ERROR("m_sample_sizes.find(index) == m_sample_sizes.end() for index: ", index, "; m_sample_sizes.size: ", m_sample_sizes.size());
        }
        sz = m_sample_sizes[index];
      }

      if (m_output) {
        (*m_output) << "sending " << index << " size: " << sz << " to " << p << std::endl;
      }

      m_comm->nb_tagged_send<El::byte>(s, sz, p, index, m_send_requests[ss++], m_comm->get_trainer_comm());
    }
  }

  // sanity checks
  if (ss != m_send_requests.size()) {
    LBANN_ERROR("ss != m_send_requests.size; ss: ", ss, " m_send_requests.size: ", m_send_requests.size());
  }

  // start recvs for incoming data
  ss = 0;

  for (int p=0; p<m_np_in_trainer; p++) {
    const std::unordered_set<int> &indices = m_indices_to_recv[p];
    int sanity = 0;
    for (auto index : indices) {
      ++sanity;
      int sz = m_compacted_sample_size;
      if (m_node_sizes_vary) {
        if (m_sample_sizes.find(index) == m_sample_sizes.end()) {
          LBANN_ERROR("m_sample_sizes.find(index) == m_sample_sizes.end() for index: ", index, "; m_sample_sizes.size(): ", m_sample_sizes.size(), " role: ", m_reader->get_role(), " for index: ", sanity, " of ", indices.size());
        }
        sz = m_sample_sizes[index];
      }

      m_recv_buffer[ss].set(conduit::DataType::uint8(sz));
      El::byte *r = reinterpret_cast<El::byte*>(m_recv_buffer[ss].data_ptr());
      m_comm->nb_tagged_recv<El::byte>(r, sz, p, index, m_recv_requests[ss], m_comm->get_trainer_comm());
      m_recv_data_ids[ss] = index;
      ++ss;
    }
  }

  // sanity checks
  if (ss != m_recv_buffer.size()) {
    LBANN_ERROR("ss != m_recv_buffer.size; ss: ", ss, " m_recv_buffer.size: ", m_recv_buffer.size());
  }
  if (m_recv_requests.size() != m_recv_buffer.size()) {
    LBANN_ERROR("m_recv_requests.size != m_recv_buffer.size; m_recv_requests: ", m_recv_requests.size(), " m_recv_buffer.size: ", m_recv_buffer.size());
  }

  // wait for all msgs to complete
  m_comm->wait_all(m_send_requests);
  m_comm->wait_all(m_recv_requests);

  //========================================================================
  //part 3: construct the Nodes needed by me for the current minibatch

  conduit::Node nd;
  m_minibatch_data.clear();
  double tm2 = get_time();
  for (size_t j=0; j < m_recv_buffer.size(); j++) {
    conduit::uint8 *n_buff_ptr = (conduit::uint8*)m_recv_buffer[j].data_ptr();
    conduit::Node n_msg;
    n_msg["schema_len"].set_external((conduit::int64*)n_buff_ptr);
    n_buff_ptr +=8;
    n_msg["schema"].set_external_char8_str((char*)(n_buff_ptr));
    conduit::Schema rcv_schema;
    conduit::Generator gen(n_msg["schema"].as_char8_str());
    gen.walk(rcv_schema);
    n_buff_ptr += n_msg["schema"].total_bytes_compact();
    n_msg["data"].set_external(rcv_schema,n_buff_ptr);

    int data_id = m_recv_data_ids[j];
    m_minibatch_data[data_id].set_external(n_msg["data"]);
  }
  m_rebuild_time += (get_time() - tm2);
}

int data_store_conduit::build_indices_i_will_recv(int current_pos, int mb_size) {
  m_indices_to_recv.clear();
  m_indices_to_recv.resize(m_np_in_trainer);
  int k = 0;
  for (int i=current_pos; i< current_pos + mb_size; ++i) {
    auto index = (*m_shuffled_indices)[i];
    if ((i % m_owner_map_mb_size) % m_np_in_trainer == m_rank_in_trainer) {
      int owner = m_owner[index];
      m_indices_to_recv[owner].insert(index);
      k++;
    }
  }
  return k;
}

int data_store_conduit::build_indices_i_will_send(int current_pos, int mb_size) {
  m_indices_to_send.clear();
  m_indices_to_send.resize(m_np_in_trainer);
  int k = 0;
  if (m_output) {
    (*m_output) << "build_indices_i_will_send; cur pos: " << current_pos << " mb_size: " << mb_size << " m_data.size: " << m_data.size() << "\n";
  }
  for (int i = current_pos; i < current_pos + mb_size; i++) {
    auto index = (*m_shuffled_indices)[i];
    /// If this rank owns the index send it to the (i%m_np)'th rank
    if (m_data.find(index) != m_data.end()) {
      m_indices_to_send[(i % m_owner_map_mb_size) % m_np_in_trainer].insert(index);

      // Sanity check
      if (m_owner[index] != m_rank_in_trainer) {
        std::stringstream s;
        s << "error for i: "<<i<<" index: "<<index<< " m_owner: " << m_owner[index] << " me: " << m_rank_in_trainer;
        LBANN_ERROR(s.str());
      }
      k++;
    }
  }
  return k;
}

void data_store_conduit::build_preloaded_owner_map(const std::vector<int>& per_rank_list_sizes) {
  m_owner.clear();
  int owning_rank = 0;
  size_t per_rank_list_range_start = 0;
  for (size_t i = 0; i < m_shuffled_indices->size(); i++) {
    const auto per_rank_list_size = per_rank_list_sizes[owning_rank];
    if(i == (per_rank_list_range_start + per_rank_list_size)) {
      ++owning_rank;
      per_rank_list_range_start += per_rank_list_size;
    }
    m_owner[(*m_shuffled_indices)[i]] = owning_rank;
  }
}

#if 0
void data_store_conduit::build_owner_map(int mini_batch_size) {
  if (m_world_master) std::cout << "starting data_store_conduit::build_owner_map for role: " << m_reader->get_role() << " with mini_batch_size: " << mini_batch_size << " num indices: " << m_shuffled_indices->size() << "\n";
  if (mini_batch_size == 0) {
    LBANN_ERROR("mini_batch_size == 0; can't build owner_map");
  }
  m_owner.clear();
  m_owner_map_mb_size = mini_batch_size;
  for (size_t i = 0; i < m_shuffled_indices->size(); i++) {
    auto index = (*m_shuffled_indices)[i];
    /// To compute the owner index first find its position inside of
    /// the mini-batch (mod mini-batch size) and then find how it is
    /// striped across the ranks in the trainer
    m_owner[index] = (i % m_owner_map_mb_size) % m_np_in_trainer;
  }
}
#endif

const conduit::Node & data_store_conduit::get_random_node() const {
  size_t sz = m_data.size();

  // Deal with edge case
  if (sz == 0) {
    LBANN_ERROR("can't return random node since we have no data (set_conduit_node has never been called)");
  }

  int offset = random() % sz;
  auto it = std::next(m_data.begin(), offset);
  return it->second;
}

const conduit::Node & data_store_conduit::get_random_node(const std::string &field) const {
  auto node = get_random_node();
  //return node;
  return node[field];
}

conduit::Node & data_store_conduit::get_empty_node(int data_id) {
  if (m_data.find(data_id) != m_data.end()) {
    LBANN_ERROR("we already have a node with data_id= ", data_id);
  }
  return m_data[data_id];
}

void data_store_conduit::purge_unused_samples(const std::vector<int>& indices) {
  if (m_output) {
    (*m_output) << " starting purge_unused_samples; indices.size(): " << indices.size() << " data.size(): " << m_data.size() << std::endl;
  }
  /// Remove unused indices from the data and owner maps
  for(auto&& i : indices) {
    if(m_data.find(i) != m_data.end()){
      m_data.erase(i);
    }
    if(m_owner.find(i) != m_owner.end()) {
      m_owner.erase(i);
    }
  }
  if (m_output) {
    (*m_output) << " leaving  purge_unused_samples; indices.size(): " << indices.size() << " data.size(): " << m_data.size() << std::endl;
  }
}

void data_store_conduit::compact_nodes() {
  for(auto&& j : *m_shuffled_indices) {
    if(m_data.find(j) != m_data.end()){
      if(! (m_data[j].is_contiguous() && m_data[j].is_compact()) ) {
        /// Repack the nodes because they don't seem to copy correctly
        conduit::Node node = m_data[j]["data"];
        m_data.erase(j);
        build_node_for_sending(node, m_data[j]);
      }
    }
  }
}

int data_store_conduit::get_index_owner(int idx) {
  if (m_owner.find(idx) == m_owner.end()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << " idx: " << idx << " was not found in the m_owner map;"
        << " map size: " << m_owner.size();
    throw lbann_exception(err.str());
  }
  return m_owner[idx];
}

void data_store_conduit::check_mem_capacity(lbann_comm *comm, const std::string sample_list_file, size_t stride, size_t offset) {
  if (comm->am_world_master()) {
    // note: we only estimate memory required by the data reader/store

    // get avaliable memory
    std::ifstream in("/proc/meminfo");
    std::string line;
    std::string units;
    double a_mem = 0;
    while (getline(in, line)) {
      if (line.find("MemAvailable:")) {
        std::stringstream s3(line);
        s3 >> line >> a_mem >> units;
        if (units != "kB") {
          LBANN_ERROR("units is ", units, " but we only know how to handle kB; please contact Dave Hysom");
        }
        break;
      }
    }
    in.close();
    if (a_mem == 0) {
      LBANN_ERROR("failed to find MemAvailable field in /proc/meminfo");
    }

    // a lot of the following is cut-n-paste from the sample list class;
    // would like to use the sample list class directly, but this
    // is quicker than figuring out how to modify the sample_list.
    // Actually there are at least three calls, starting from
    // data_reader_jag_conduit, before getting to the code that
    // loads the sample list file names

    // get list of conduit files that I own, and compute my num_samples
    std::ifstream istr(sample_list_file);
    if (!istr.good()) {
      LBANN_ERROR("failed to open ", sample_list_file, " for reading");
    }

    std::string base_dir;
    std::getline(istr, line);  //exclusiveness; discard

    std::getline(istr, line);
    std::stringstream s5(line);
    int included_samples;
    int excluded_samples;
    size_t num_files;
    s5 >> included_samples >> excluded_samples >> num_files;

    std::getline(istr, base_dir); // base dir; discard

    const std::string whitespaces(" \t\f\v\n\r");
    size_t cnt_files = 0u;
    int my_sample_count = 0;

    conduit::Node useme;
    bool got_one = false;

    // loop over conduit filenames
    while (std::getline(istr, line)) {
      const size_t end_of_str = line.find_last_not_of(whitespaces);
      if (end_of_str == std::string::npos) { // empty line
        continue;
      }
      if (cnt_files++ >= num_files) {
        break;
      }
      if ((cnt_files-1)%stride != offset) {
        continue;
      }
      std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
      std::string filename;
      sstr >> filename >> included_samples >> excluded_samples;
      my_sample_count += included_samples;

      // attempt to load a JAG sample
      if (!got_one) {
        hid_t hdf5_file_hnd;
        try {
          hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read(base_dir + '/' + filename);
        } catch (conduit::Error const& e) {
          LBANN_ERROR(" failed to open ", base_dir, '/', filename, " for reading");
        }
        std::vector<std::string> sample_names;
        try {
          conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", sample_names);
        } catch (conduit::Error const& e) {
          LBANN_ERROR("hdf5_group_list_child_names() failed");
        }

        for (auto t : sample_names) {
          std::string key = "/" + t + "/performance/success";
          try {
            conduit::relay::io::hdf5_read(hdf5_file_hnd, key, useme);
          } catch (conduit::Error const& e) {
            LBANN_ERROR("failed to read success flag for ", key);
          }
          if (useme.to_int64() == 1) {
            got_one = true;
            try {
              key = "/" + t;
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, useme);
            } catch (conduit::Error const& e) {
              LBANN_ERROR("failed to load JAG sample: ", key);
            }
            break;
          }
        } // end: for (auto t : sample_names)

        conduit::relay::io::hdf5_close_file(hdf5_file_hnd);
      } // end: attempt to load a JAG sample
    } // end: loop over conduit filenames
    istr.close();
    // end: get list of conduit files that I own, and compute my num_samples

    if (! got_one) {
      LBANN_ERROR("failed to find any successful JAG samples");
    }

    // compute memory for the compacted nodes this processor owns
    double bytes_per_sample = useme.total_bytes_compact() / 1024;
    double  procs_per_node = comm->get_procs_per_node();
    double mem_this_proc = bytes_per_sample * my_sample_count;
    double mem_this_node = mem_this_proc * procs_per_node;

    std::cout
      << "\n"
      << "==============================================================\n"
      << "Estimated memory requirements for JAG samples:\n"
      << "Memory for one sample:             " <<  bytes_per_sample << " kB\n"
      << "Total mem for a single rank:       " << mem_this_proc << " kB\n"
      << "Samples per proc:                  " << my_sample_count << "\n"
      << "Procs per node:                    " << procs_per_node << "\n"
      << "Total mem for all ranks on a node: " << mem_this_node << " kB\n"
      << "Available memory: " << a_mem << " kB (RAM only; not virtual)\n";
    if (mem_this_node > static_cast<double>(a_mem)) {
      std::cout << "\nYOU DO NOT HAVE ENOUGH MEMORY\n"
        << "==============================================================\n\n";
      LBANN_ERROR("insufficient memory to load data\n");
    } else {
      double m = 100 * mem_this_node / a_mem;
      std::cout << "Estimate that data will consume at least " << m << " % of memory\n"
        << "==============================================================\n\n";
    }
  }

  comm->trainer_barrier();
}

bool data_store_conduit::has_conduit_node(int data_id) const {
  std::unordered_map<int, conduit::Node>::const_iterator t = m_data.find(data_id);
  if (m_output) {
    (*m_output) << "has_conduit_node( " << data_id << " ) = " << (t == m_data.end()) << std::endl;
  }
  return t != m_data.end();
}

void data_store_conduit::set_shuffled_indices(const std::vector<int> *indices) {
  m_shuffled_indices = indices;
}

void data_store_conduit::exchange_sample_sizes() {
  if (m_output) {
    (*m_output) << "starting data_store_conduit::exchange_sample_sizes" << std::endl;
  }

  int my_count = m_sample_sizes.size();
  std::vector<int> all_counts(m_np_in_trainer);
  m_comm->all_gather(&my_count, 1, all_counts.data(), 1,  m_comm->get_trainer_comm());

  if (m_output) {
    for (size_t h=0; h<all_counts.size(); h++) {
      (*m_output) << "num samples owned by P_" << h << " is " << all_counts[h] << std::endl;
    }
  }

  std::vector<size_t> my_sizes(m_sample_sizes.size()*2);
  size_t j = 0;
  for (auto t : m_sample_sizes) {
    my_sizes[j++] = t.first;
    my_sizes[j++] = t.second;
  }

  std::vector<size_t> other_sizes;
  for (int k=0; k<m_np_in_trainer; k++) {
    if (m_output) {
      (*m_output) << "sample sizes for P_" << k << std::endl;
      flush_debug_file();
    }
    other_sizes.resize(all_counts[k]*2);
    if (m_rank_in_trainer == k) {
      m_comm->broadcast<size_t>(k, my_sizes.data(), all_counts[k]*2,  m_comm->get_trainer_comm());
    } else {
      m_comm->broadcast<size_t>(k, other_sizes.data(), all_counts[k]*2,  m_comm->get_trainer_comm());

      for (size_t i=0; i<other_sizes.size(); i += 2) {
        if (m_sample_sizes.find(other_sizes[i]) != m_sample_sizes.end()) {
          if (m_output) {
            (*m_output) << "SAMPLE SIZES for P_" << k << std::endl;
            for (size_t h=0; h<other_sizes.size(); h += 2) {
              (*m_output) << other_sizes[h] << " SIZE: " << other_sizes[h+1] << std::endl;
            }
            flush_debug_file();
          }
          LBANN_ERROR("m_sample_sizes.find(other_sizes[i]) != m_sample_sizes.end() for data_id: ", other_sizes[i]);
        }
        m_sample_sizes[other_sizes[i]] = other_sizes[i+1];
      }
    }
  }

  m_have_sample_sizes = true;
}

void data_store_conduit::set_is_preloaded() {
if (m_world_master) std::cout << "starting data_store_conduit::set_is_preloaded(); m_preload: " << m_preload << std::endl;
  //this should be called by generic_data_reader, however, it may also
  //be called by callbacks/ltfb.cpp
  if (m_preload) {
    return;
  }
  m_preload = true;
  if (options::get()->has_string("data_store_test_checkpoint")) {
    std::string dir = options::get()->get_string("data_store_test_checkpoint");
    test_checkpoint(dir);
  }
}

void data_store_conduit::get_image_sizes(std::unordered_map<int,size_t> &file_sizes, std::vector<std::vector<int>> &indices) {
  /// this block fires if image sizes have been precomputed
  if (options::get()->has_string("image_sizes_filename")) {
    LBANN_ERROR("not yet implemented");
    //TODO dah - implement, if this becomes a bottleneck (but I don't think it will)
  }

  else {
    // get list of image file names
    image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
    if (image_reader == nullptr) {
      LBANN_ERROR("data_reader_image *image_reader = dynamic_cast<data_reader_image*>(m_reader) failed");
    }
    const std::vector<image_data_reader::sample_t> &image_list = image_reader->get_image_list();

    // get sizes of files for which I'm responsible
    std::vector<size_t> my_image_sizes;
    for (size_t h=m_rank_in_trainer; h<m_shuffled_indices->size(); h += m_np_in_trainer) {
      const std::string fn = m_reader->get_file_dir() + '/' + image_list[(*m_shuffled_indices)[h]].first;
      std::ifstream in(fn.c_str());
      if (!in) {
        LBANN_ERROR("failed to open ", fn, " for reading; file_dir: ", m_reader->get_file_dir(), "  fn: ", image_list[h].first, "; role: ", m_reader->get_role());
      }
      in.seekg(0, std::ios::end);
      my_image_sizes.push_back((*m_shuffled_indices)[h]);
      my_image_sizes.push_back(in.tellg());
      in.close();
    }
    int my_count = my_image_sizes.size();

    std::vector<int> counts(m_np_in_trainer);
    m_comm->all_gather<int>(&my_count, 1, counts.data(), 1, m_comm->get_trainer_comm());

    //my_image_sizes[h*2] contains the image index
    //my_image_sizes[h*2+1] contains the image sizee

    //fill in displacement vector for gathering the actual image sizes
    std::vector<int> disp(m_np_in_trainer + 1);
    disp[0] = 0;
    for (size_t h=0; h<counts.size(); ++h) {
      disp[h+1] = disp[h] + counts[h];
    }

    std::vector<size_t> work(image_list.size()*2);
    m_comm->trainer_all_gather<size_t>(my_image_sizes, work, counts, disp);
    indices.resize(m_np_in_trainer);
    for (int h=0; h<m_np_in_trainer; h++) {
      indices[h].reserve(counts[h]);
      size_t start = disp[h];
      size_t end = disp[h+1];
      for (size_t k=start; k<end; k+= 2) {
        size_t idx = work[k];
        size_t size = work[k+1];
        indices[h].push_back(idx);
        file_sizes[idx] = size;
      }
    }
  }
}

void data_store_conduit::compute_image_offsets(std::unordered_map<int,size_t> &sizes, std::vector<std::vector<int>> &indices) {
  size_t offset = 0;
  for (size_t p=0; p<indices.size(); p++) {
    for (auto idx : indices[p]) {
      if (sizes.find(idx) == sizes.end()) {
        LBANN_ERROR("sizes.find(idx) == sizes.end() for idx: ", idx);
      }
      size_t sz = sizes[idx];
      m_image_offsets[idx] = offset;
      offset += sz;
    }
  }
}


void data_store_conduit::allocate_shared_segment(std::unordered_map<int,size_t> &sizes, std::vector<std::vector<int>> &indices) {
  off_t size = 0;
  for (auto &&t : sizes) {
    size += t.second;
  }
  m_mem_seg_length = size;

  struct statvfs stat;
  int x = statvfs("/dev/shm", &stat);
  if (x != 0) {
    LBANN_ERROR("statvfs failed\n");
  }
  size_t avail_mem = stat.f_bsize*stat.f_bavail;
  double percent = 100.0 * m_mem_seg_length / avail_mem;
  std::stringstream msg;
  msg << "  size of required shared memory segment: " << m_mem_seg_length  << "\n"
      << "  available mem: " << avail_mem << "\n"
      << "  required size is " << percent << " percent of available\n";
  if (m_world_master) {
    std::cout << "\nShared memory segment statistics:\n"
              << msg.str() << "\n";
  }
  if (m_mem_seg_length >= avail_mem) {
    LBANN_ERROR("insufficient available memory:\n", msg.str());
  }

  //need to ensure name is unique across all data readers
  m_seg_name = "/our_town_" + m_reader->get_role();

  //in case a previous run was aborted, attempt to remove the file, which
  //may or may not exist
  shm_unlink(m_seg_name.c_str());
  int node_id = m_comm->get_rank_in_node();
  if (node_id == 0) {
    std::remove(m_seg_name.c_str());
  }
  m_comm->trainer_barrier();

  int shm_fd = -1;

  if (node_id == 0) {
    shm_fd = shm_open(m_seg_name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
    if (shm_fd == -1) {
      LBANN_ERROR("shm_open failed");
    }
    int v = ftruncate(shm_fd, size);
    if (v != 0) {
      LBANN_ERROR("ftruncate failed for size: ", size);
    }
    void *m = mmap(0, size, PROT_WRITE | PROT_READ, MAP_SHARED, shm_fd, 0);
    if (m == MAP_FAILED) {
      LBANN_ERROR("mmap failed");
    }
    m_mem_seg = reinterpret_cast<char*>(m);
    std::fill_n(m_mem_seg, m_mem_seg_length, 1);
    int sanity = msync(static_cast<void*>(m_mem_seg), m_mem_seg_length, MS_SYNC);
    if (sanity != 0) {
      LBANN_ERROR("msync failed");
    }
  }

  m_comm->barrier(m_comm->get_node_comm());

  if (node_id != 0) {
    shm_fd = shm_open(m_seg_name.c_str(), O_RDONLY, 0666);
    if (shm_fd == -1) {
      LBANN_ERROR("shm_open failed for filename: ", m_seg_name);
    }
    void *m = mmap(0, size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (m == MAP_FAILED) {
      LBANN_ERROR("mmap failed");
    }
    m_mem_seg = reinterpret_cast<char*>(m);

    struct stat b;
    int sanity = fstat(shm_fd, &b);
    if (sanity == -1) {
      LBANN_ERROR("fstat failed");
    }
    if (b.st_size != size) {
      LBANN_ERROR("b.st_size= ", b.st_size, " should be equal to ", size);
    }
  }
  close(shm_fd);
}

void data_store_conduit::preload_local_cache() {
  std::unordered_map<int,size_t> file_sizes;
  std::vector<std::vector<int>> indices;

  double tm1 = get_time();
  if (m_world_master) std::cout << "calling get_image_sizes" << std::endl;
  get_image_sizes(file_sizes, indices);
  if (m_world_master) std::cout << "  get_image_sizes time: " << (get_time()-tm1) << std::endl;
  tm1 = get_time();
  //indices[j] contains the indices (wrt m_reader->get_image_list())
  //that P_j will read from disk, and subsequently bcast to all others
  //
  //file_sizes maps an index to its file size

  if (m_world_master) std::cout << "calling allocate_shared_segment" << std::endl;
  allocate_shared_segment(file_sizes, indices);
  if (m_world_master) std::cout << "  allocate_shared_segment time: " << (get_time()-tm1) << std::endl;
  tm1 = get_time();

  if (m_world_master) std::cout << "calling read_files" << std::endl;
  std::vector<char> work;
  read_files(work, file_sizes, indices[m_rank_in_trainer]);
  if (m_world_master) std::cout << "  read_files time: " << (get_time()- tm1) << std::endl;
  tm1 = get_time();

  if (m_world_master) std::cout << "calling compute_image_offsets" << std::endl;
  compute_image_offsets(file_sizes, indices);
  if (m_world_master) std::cout << "  compute_image_offsets time: " << (get_time()-tm1) << std::endl;
  tm1 = get_time();

  if (m_world_master) std::cout << "calling exchange_images" << std::endl;
  exchange_images(work, file_sizes, indices);
  if (m_world_master) std::cout << "  exchange_images time: " << (get_time()-tm1) << std::endl;
  tm1 = get_time();

  if (m_world_master) std::cout << "calling build_conduit_nodes" << std::endl;
  build_conduit_nodes(file_sizes);
  if (m_world_master) std::cout << "  build_conduit_nodes time: " << (get_time()-tm1) << std::endl;
}

void data_store_conduit::read_files(std::vector<char> &work, std::unordered_map<int,size_t> &sizes, std::vector<int> &indices) {

  //reserve space for reading this proc's files into a contiguous memory space
  size_t n = 0;
  for (size_t j=0; j<indices.size(); ++j) {
    n += sizes[indices[j]];
  }
  work.resize(n);

  if (m_output) {
    (*m_output) << "data_store_conduit::read_files; requested work size: " << n << std::endl;
  }

  //get the list of images from the data reader
  image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<image_data_reader::sample_t> &image_list = image_reader->get_image_list();

  //read the images
  size_t offset = 0;
  if (m_world_master) std::cout << "  my num files: " << indices.size() << std::endl;
  for (size_t j=0; j<indices.size(); ++j) {
    int idx = indices[j];
    size_t s = sizes[idx];
    const std::string fn = m_reader->get_file_dir() + '/' + image_list[idx].first;
    std::ifstream in(fn, std::ios::in | std::ios::binary);
    in.read(work.data()+offset, s);
    in.close();
    offset += s;
  }
  if (m_world_master) std::cout << "  finished reading files\n";
}

void data_store_conduit::build_conduit_nodes(std::unordered_map<int,size_t> &sizes) {
  image_data_reader *image_reader = dynamic_cast<image_data_reader*>(m_reader);
  const std::vector<image_data_reader::sample_t> &image_list = image_reader->get_image_list();
  for (size_t idx=0; idx<image_list.size(); idx++) {
    int label = image_list[idx].second;
    size_t offset = m_image_offsets[idx];
    size_t sz = sizes[idx];
    conduit::Node &node = m_data[idx];
    node[LBANN_DATA_ID_STR(idx) + "/label"].set(label);
    node[LBANN_DATA_ID_STR(idx) + "/buffer_size"] = sz;
    char *c = m_mem_seg + offset;
    node[LBANN_DATA_ID_STR(idx) + "/buffer"].set_external_char_ptr(c, sz);
  }
}

void data_store_conduit::fillin_shared_images(const std::vector<char> &images, size_t offset) {
  memcpy(m_mem_seg+offset, reinterpret_cast<const void*>(images.data()), images.size());
}

void data_store_conduit::exchange_images(std::vector<char> &work, std::unordered_map<int,size_t> &image_sizes, std::vector<std::vector<int>> &indices) {
  std::vector<char> work2;
  int node_rank = m_comm->get_rank_in_node();
  size_t offset = 0;
  for (int p=0; p<m_np_in_trainer; p++) {
    if (m_rank_in_trainer == p) {
      m_comm->trainer_broadcast<char>(p, work.data(), work.size());
      if (node_rank == 0) {
        fillin_shared_images(work, offset);
      }
    } else {
      size_t sz = 0;
      for (auto idx : indices[p]) {
        sz += image_sizes[idx];
      }
      work2.resize(sz);
      m_comm->trainer_broadcast<char>(p, work2.data(), sz);
      if (node_rank == 0) {
        fillin_shared_images(work2, offset);
      }
    }

    for (size_t r=0; r<indices[p].size(); r++) {
      offset += image_sizes[indices[p][r]];
    }
  }

  m_comm->barrier(m_comm->get_node_comm());
}

void data_store_conduit::exchange_owner_maps() {
  if (m_output) {
    (*m_output) << "\nstarting data_store_conduit::exchange_owner_maps\n\n";
  }
  int my_count = m_owner.size();
  std::vector<int> all_counts(m_np_in_trainer);
  m_comm->all_gather(&my_count, 1, all_counts.data(), 1,  m_comm->get_trainer_comm());

  std::vector<size_t> my_sizes(m_owner.size());
  size_t j = 0;
  for (auto t : m_owner) {
    my_sizes[j++] = t.first;
  }

  std::vector<size_t> other_sizes;
  for (int k=0; k<m_np_in_trainer; k++) {
    other_sizes.resize(all_counts[k]);
    if (m_rank_in_trainer == k) {
      m_comm->broadcast<size_t>(k, my_sizes.data(), all_counts[k],  m_comm->get_trainer_comm());
    } else {
      m_comm->broadcast<size_t>(k, other_sizes.data(), all_counts[k],  m_comm->get_trainer_comm());
      for (size_t i=0; i<other_sizes.size(); ++i) {
        if (m_owner.find(other_sizes[i]) != m_owner.end()) {

          if (m_output) {
            (*m_output) << "data_store_conduit::exchange_owner_maps, duplicate data_id: " << other_sizes[i] << "; k= " << k << "\nm_owner:\n";
            for (auto t : m_owner) (*m_output) << "data_id: " << t.first << " owner: " << t.second << std::endl;
            (*m_output) << "\nother_sizes[k]: ";
            for (auto t : other_sizes) (*m_output) << t << " ";
            (*m_output) << std::endl;
            flush_debug_file();
          }

          LBANN_ERROR("duplicate data_id: ", other_sizes[i], " role: ", m_reader->get_role(), "; m_owner[",other_sizes[i],"] = ", m_owner[other_sizes[i]]);
        }
        m_owner[other_sizes[i]] = k;
      }
    }
  }
}

void data_store_conduit::exchange_mini_batch_data(size_t current_pos, size_t mb_size) {
  double tm1 = get_time();
  if (is_local_cache()) {
    return;
  }
  if (m_reader->at_new_epoch()) {
    if (m_world_master && m_cur_epoch > 0) {
      std::cout << "time for exchange_mini_batch_data calls: " 
                << m_exchange_time << std::endl
                << "time for constructing conduit Nodes: " << m_rebuild_time 
                << std::endl;
      std::cout << std::endl;
      m_exchange_time = 0.;
      m_rebuild_time = 0.;
    }
    ++m_cur_epoch;
  }

  if (m_reader->at_new_epoch() && !m_preload && !m_is_local_cache && m_cur_epoch == 1) {
    exchange_owner_maps();
  }

  exchange_data_by_sample(current_pos, mb_size);
  m_exchange_time += (get_time() - tm1);
}

void data_store_conduit::flush_debug_file() {
  if (!m_output) {
    return;
  }
  m_output->close();
  m_output->open(m_debug_filename.c_str(), std::ios::app);
}

size_t data_store_conduit::get_num_indices() const {
  size_t num = m_data.size();
  size_t n = m_comm->trainer_allreduce<size_t>(num);
  return n;
}

void data_store_conduit::test_checkpoint(const std::string &checkpoint_dir) {
  if (m_world_master) {
    std::cout << "starting data_store_conduit::test_checkpoint\n"
              << "here is part of the owner map; m_owner.size(): " << m_owner.size() << std::endl;
    size_t j = 0;
    for (auto t : m_owner) {
      ++j;
      std::cout << "  sample_id: " << t.first << " owner: " << t.second << std::endl;
      if (j >= 10) break;
    }
    print_variables();
    std::cout << "\nCalling spill_to_file(testme_xyz)" << std::endl;
  }
  spill_to_file(checkpoint_dir);

  std::unordered_map<int,int> sanity = m_owner;
  m_owner.clear();
  m_sample_sizes.clear();
  m_data.clear();
  m_cur_epoch = -1;

  m_is_setup = false;
  m_preload = false;
  m_explicit_loading = true;
  m_owner_map_mb_size = 0;
  m_compacted_sample_size = 0;
  m_node_sizes_vary = true;

  if (m_world_master) {
    print_variables();
  }


  if (m_world_master) {
    std::cout << "Cleared the owner map; m_owner.size() = " << m_owner.size() << std::endl
              << "Calling load_from_file" << std::endl;
  }
  load_from_file(checkpoint_dir, nullptr);
  if (m_world_master) {
    std::cout << "Here is part of the re-loaded owner map; map.size(): " << m_owner.size() << std::endl;
    size_t j = 0;
    for (auto t : m_owner) {
      ++j;
      std::cout << "  sample_id: " << t.first << " owner: " << t.second << std::endl;
      if (j >= 10) break;
    }
    print_variables();
  }

  for (auto t : m_owner) {
    if (sanity.find(t.first) == sanity.end()) {
      LBANN_ERROR("sanity.find(t.first) == sanity.end() for t.first= ", t.first);
    } else if (sanity[t.first] != m_owner[t.first]) {
      LBANN_ERROR("sanity[t.first] != m_owner[t.first] for t.first= ", t.first, " and m_owner[t.first]= ", m_owner[t.first]);
    }
  }

  m_comm->global_barrier();
}

void data_store_conduit::make_dir_if_it_doesnt_exist(const std::string &dir_name) {
  int node_rank = m_comm->get_rank_in_node();
  if (node_rank == 0) {
    bool exists = file::directory_exists(dir_name);
    if (!exists) {
      if (m_world_master) {
        std::cout << "data_store_conduit; the directory '" << dir_name << "' doesn't exist; creating it\n";
      }
      file::make_directory(dir_name);
    }
  }
}

void data_store_conduit::spill_to_file(std::string dir_name) {
  make_dir_if_it_doesnt_exist(dir_name);
  m_comm->trainer_barrier();
  const std::string conduit_dir = get_conduit_dir_name(dir_name);
  make_dir_if_it_doesnt_exist(conduit_dir);


  const std::string metadata_fn = get_metadata_fn(dir_name);
  std::ofstream metadata(metadata_fn);
  if (!metadata) {
    LBANN_ERROR("failed to open ", metadata_fn, " for writing");
  }

  //TODO should have two levels of directory to ensure
  //     there's not too many files in any directory
  metadata << conduit_dir << "\n";
  int cur_dir = -1;
  int num_files = m_max_files_per_directory;
  const std::string my_conduit_dir = conduit_dir + "/conduit_" + std::to_string(m_rank_in_world);
  std::string cur_dir_name;
  for (auto t : m_data) {
    if (num_files == m_max_files_per_directory) {
      num_files = 0;
      cur_dir += 1;
      cur_dir_name = conduit_dir + "/" + to_string(cur_dir);
      bool exists = file::directory_exists(cur_dir_name);
      if (!exists) {
        file::make_directory(cur_dir_name);
      }
    }

    const std::string f = cur_dir_name + '/' + std::to_string(t.first);
    t.second.save(cur_dir_name + '/' + std::to_string(t.first));
    metadata << cur_dir << "/" << t.first << " " << t.first << std::endl;
    ++num_files;
  }
  metadata.close();

  // checkpoint remaining state using cereal
  const std::string fn = get_cereal_fn(dir_name);
  std::ofstream os(fn);
  if (!os) {
    LBANN_ERROR("failed to open ", fn, " for writing");
  }

  {
  cereal::XMLOutputArchive archive(os);
    archive(CEREAL_NVP(m_cur_epoch), 
            CEREAL_NVP(m_is_setup),
            CEREAL_NVP(m_preload), 
            CEREAL_NVP(m_explicit_loading),
            CEREAL_NVP(m_owner_map_mb_size), 
            CEREAL_NVP(m_compacted_sample_size), 
            CEREAL_NVP(m_is_local_cache),
            CEREAL_NVP(m_node_sizes_vary), 
            CEREAL_NVP(m_have_sample_sizes),
            CEREAL_NVP(m_owner),
            CEREAL_NVP(m_sample_sizes));
  }
  os.close();
}

void data_store_conduit::load_from_file(std::string dir_name, generic_data_reader *reader) {
  if (m_world_master) std::cout << "starting data_store_conduit::load_from_file" << std::endl;

  bool exists = file::directory_exists(dir_name);
  if (!exists) {
    LBANN_ERROR("cannot load data_store from file, since the specified directory ", dir_name, "doesn't exist");
  }

  const std::string fn = get_cereal_fn(dir_name);
  std::ifstream in(fn);
  if (!in) {
    LBANN_ERROR("failed to open ", m_cereal_fn, " for reading");
  }

  cereal::XMLInputArchive iarchive(in);
  iarchive(m_cur_epoch, m_is_setup,
           m_preload, m_explicit_loading,
           m_owner_map_mb_size,
           m_compacted_sample_size, m_is_local_cache,
           m_node_sizes_vary, m_have_sample_sizes,
           m_owner, m_sample_sizes);

  if (reader != nullptr) {
    m_reader = reader;
    m_comm = m_reader->get_comm();
    m_shuffled_indices = &(m_reader->get_shuffled_indices());
    m_world_master = m_comm->am_world_master();
    m_trainer_master = m_comm->am_trainer_master();
    m_rank_in_trainer = m_comm->get_rank_in_trainer();
    m_rank_in_world = m_comm->get_rank_in_world();
    m_np_in_trainer = m_comm->get_procs_per_trainer();
  }  

  const std::string metadata_fn = get_metadata_fn(dir_name);
  std::ifstream metadata(metadata_fn);
  if (!metadata) {
    LBANN_ERROR("failed to open ", metadata_fn, " for reading");
  }

  std::string base_dir;
  getline(metadata, base_dir);
  std::string tmp;
  int sample_id;
  while (metadata >> tmp >> sample_id) {
    if (tmp.size() > 2) {
      const std::string fn2 = base_dir + "/" + tmp;
      conduit::Node nd;
      nd.load(fn2);

      // if you're wondering why we're doing the following, see the
      // note in copy_members: "Repack the nodes ..."
      conduit::Node n2;
      const std::vector<std::string> &names = nd["data"].child_names();
      const std::vector<std::string> &names2 = nd["data"][names[0]].child_names();
      for (auto t : names2) {
        n2[names[0]][t] = nd["data"][names[0]][t];
      }
      //build_node_for_sending(n2, m_data[sample_id]);
      build_node_for_sending(nd["data"], m_data[sample_id]);
    }
  }
  metadata.close();

  m_was_loaded_from_file = true;
}

void data_store_conduit::print_variables() {
  if (!m_world_master) {
    return;
  }
  std::cout << "m_cur_epoch: " << m_cur_epoch << std::endl
            << "m_is_setup: " << m_is_setup << std::endl
            << "m_preload: " << m_preload << std::endl
            << "m_explicit_loading: " << m_explicit_loading << std::endl
            << "m_owner_map_mb_size: " << m_owner_map_mb_size << std::endl
            << "m_compacted_sample_size: " << m_compacted_sample_size << std::endl
            << "m_node_sizes_vary: " << m_node_sizes_vary << std::endl;
}

std::string data_store_conduit::get_conduit_dir_name(const std::string& dir_name) const {
  return dir_name + "/conduit_" + std::to_string(m_rank_in_world);
}

std::string data_store_conduit::get_metadata_fn(const std::string& dir_name) const {
  return dir_name + "/metadata_" + std::to_string(m_rank_in_world);
}

std::string data_store_conduit::get_cereal_fn(const std::string& dir_name) const {
  return dir_name + '/' + m_cereal_fn + "_" + std::to_string(m_rank_in_world); 
}

void data_store_conduit::spill_conduit_node(const conduit::Node &node, int data_id) {
  if (m_num_files_in_cur_spill_dir == m_max_files_per_directory) {
    m_num_files_in_cur_spill_dir = 0;
    m_cur_dir += 1;
    m_cur_dir = m_spill_dir_base + "/" + to_string(m_cur_dir);
    bool exists = file::directory_exists(m_cur_dir);
    if (!exists) {
      file::make_directory(m_cur_dir);
    }
    node.save(m_cur_dir + '/' + std::to_string(data_id));
  }
}

void data_store_conduit::load_spilled_conduit_nodes() {
  std::unordered_set<int> indices_that_are_already_loaded;
  std::unordered_set<int> indices_to_be_loaded;
  for (const auto &t : m_indices_to_send) {
    for (const auto &t2 : t) {
      if (m_data.find(t2) == m_data.end()) {
        indices_to_be_loaded.insert(t2);
      } else {
        indices_that_are_already_loaded.insert(t2);
      }
    }
  }

  //note: in general, I expect that indices_that_are_already_loaded will be empty;
  //      one exception may be during the first call to exchange_data() at the 
  //      beginning of an epoch
  for (const auto &t : m_data) {
    if (indices_that_are_already_loaded.find(t.first) == indices_that_are_already_loaded.end()) {
      m_data.erase(t.first);
    }
  }

  for (const auto &t : indices_to_be_loaded) {
    std::unordered_map<int, int>::const_iterator it = m_spilled_nodes.find(t);
    if (it == m_spilled_nodes.end()) {
      LBANN_ERROR("t == m_spilled_nodes.end()");
    }
    const std::string fn = m_spill_dir_base + "/" + std::to_string(it->second);
    conduit::Node node;
    node.load(fn);
    build_node_for_sending(node, m_data[t]);
  }
}


}  // namespace lbann
