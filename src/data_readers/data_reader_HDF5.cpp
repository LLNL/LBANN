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
/////////////////////////////////////////////////////////////////////////////////
#include "lbann/data_readers/data_reader_HDF5.hpp"
#include "lbann/utils/timer.hpp"
#include "conduit/conduit_relay_mpi.hpp"

using namespace std; //XX

namespace lbann {


int hdf5_data_reader::fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) {
  return 0;
}

hdf5_data_reader::hdf5_data_reader(bool shuffle) 
  : data_reader_sample_list(shuffle) {
}

hdf5_data_reader::hdf5_data_reader(const hdf5_data_reader& rhs)
  : data_reader_sample_list(rhs) {
  copy_members(rhs);
}

hdf5_data_reader& hdf5_data_reader::operator=(const hdf5_data_reader& rhs) {
  if (this == &rhs) {
    return (*this);
  }
  data_reader_sample_list::operator=(rhs);
  copy_members(rhs);
  return (*this);
}

void hdf5_data_reader::copy_members(const hdf5_data_reader &rhs) {
  m_data_schema = rhs.m_data_schema;
  m_experiment_schema = rhs.m_experiment_schema;
}

void hdf5_data_reader::load() {
  if(is_master()) {
    std::cout << "hdf5_data_reader - starting load" << std::endl;
  }
  double tm1 = get_time();

  data_reader_sample_list::load();

  // load the schemas (yes, these are actually Nodes, but they play
  // the part of schemas, so that's what I'm colling them)
  options *opts = options::get();
  if (!opts->has_string("data_schema_fn")) {
    LBANN_ERROR("you must include --data_schema_fn=<string>");
  }
  load_schema(opts->get_string("data_schema_fn"), m_data_schema);

  if (!opts->has_string("experiment_schema_fn")) {
    LBANN_ERROR("you must include --experiment_schema_fn=<string>");
  }
  load_schema(opts->get_string("experiment_schema_fn"), m_experiment_schema);

  // get ptrs to all nodes
  get_schema_ptrs(&m_experiment_schema, m_experiment_schema_nodes);
  get_schema_ptrs(&m_data_schema, m_data_schema_nodes);

  parse_schemas();

  // may go away; for now, this reader only supports preloading mode
  opts->set_option("use_data_store", true);

  // the usual boilerplate (we should wrap this in a function)
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();
  instantiate_data_store();
  select_subset_of_data();

  if (is_master()) {
    std::cout << "hdf5_data_reader::load() time: " << (get_time() - tm1) 
              << " num samples: " << m_shuffled_indices.size() << std::endl;
  }
}

void hdf5_data_reader::load_schema(std::string filename, conduit::Node &schema) {
cout << "starting load_schema for: " << filename << endl;
  // master loads the schema then bcasts to all others.
  // for now this is an MPI_WORLD_COMM operation
  if (filename == "") {
    LBANN_ERROR("load_schema was passed an empty filename; did you call set_schema_filename?");
  }
  if (is_master()) {
    conduit::relay::io::load(filename, schema);
  }

  conduit::relay::mpi::broadcast_using_schema(schema, m_comm->get_world_master(), m_comm->get_world_comm().GetMPIComm());
}

void hdf5_data_reader::do_preload_data_store() {
  double tm1 = get_time();
  if (is_master()) {
    std::cout << "starting hdf5_data_reader::do_preload_data_store()\n";
  }

  // TODO: construct a more efficient owner mapping, and set it in the data store.

 for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != m_rank_in_model) {
      continue;
    }
    try {
      conduit::Node & node = m_data_store->get_empty_node(index);
      load_sample(node, index);
      m_data_store->set_preloaded_conduit_node(index, node);
    } catch (conduit::Error const& e) {
      LBANN_ERROR(" :: trying to load the node ", index, " and caught conduit exception: ", e.what());
    }
  }
  // Once all of the data has been preloaded, close all of the file handles
  for (size_t idx=0; idx < m_shuffled_indices.size(); idx++) {
    int index = m_shuffled_indices[idx];
    if(m_data_store->get_index_owner(index) != m_rank_in_model) {
      continue;
    }
    close_file(index);
  }

  if (is_master()) {
    std::cout << "loading data for role: " << get_role() << " took " << get_time() - tm1 << "s" << std::endl;
  }
}

// TODO: does this work? maybe not ...
int hdf5_data_reader::get_linearized_size(const std::string &key) const {
LBANN_ERROR("FIXME!");
#if 0
XX
  if (m_data_pathnames.find(key) == m_data_pathnames.end()) {
    LBANN_ERROR("requested key: ", key, " is not in the schema");
  }
  conduit::DataType dt = m_data_schema.dtype();
  return dt.number_of_elements();
#endif
  return 0;
}

// Loads the fields that are specified in the user supplied schema.
// On entry, 'node,' which was obtained from the data_store, contains a 
// single top-level node which is the sample_id.
void hdf5_data_reader::load_sample(conduit::Node &node, size_t index) {
  // get file handle
  hid_t file_handle;
  std::string sample_name;
  data_reader_sample_list::open_file(index, file_handle, sample_name);

  // load data for the  field names that the user specified in their schema;
  // first, we load each field separately; at the end of this method
  // we call munge_data, in which we pack and or normalize, etc, the data
  for (const auto &p : m_all_exp_leaves) {
    const std::string pathname = p->path();

    // check that the requested data exists
    const std::string original_path = "/" + sample_name + "/" + pathname;
    if (!conduit::relay::io::hdf5_has_path(file_handle, original_path)) {
      LBANN_ERROR("hdf5_has_path failed for path: ", original_path);
    }

    // get the new path-name (prepend the index)
    std::stringstream ss2;
    ss2 << LBANN_DATA_ID_STR(index) << '/' << pathname;
    const std::string useme_path(ss2.str());

    // load the field's data into a conduit::Node
    conduit::relay::io::hdf5_read(file_handle, original_path, node[useme_path]);
  }

  munge_data(node);

  // XXX debug block; may go away
  static bool debug = true;
  if (debug) {
    const conduit::Schema &s = node.schema();
    std::cout << "\n=======================================================\n"
                 "Node schema as it will be handed to the data store, at the\n"
                 "end of hdf5_data_reader::load_sample\n";
    s.print();
    debug = false;
  }
}

// on entry, 'node' contains data specified by the user's schema,
// with a single top-level node that contains the sample ID
void hdf5_data_reader::munge_data(conduit::Node &node) {

  // sanity check: assumption: all data has a single top-level node
  size_t n = node.number_of_children();
  if (n != 1) {
    LBANN_ERROR("n= ", n, "; should be 1");
  }


  // Case #1: there is no user-supplied metadata (from the user-supplied schema)
  if (m_packed_to_field_names_map.size() == 0) { 
    return;
  }

  // TODO: Case #X: normalize, etc.
  
  // Case #2: pack some or all of data
  std::unordered_set<std::string> used_field_names;
  std::vector<lbann::DataType> tmp_data; //temporary storage

  for (const auto t : m_packed_to_field_names_map) {

    // allocate storage for packing data
    const std::string &packed_name = t.first;
    if (m_packed_name_to_num_elts.find(packed_name) == m_packed_name_to_num_elts.end()) {
      LBANN_ERROR("m_packed_name_to_num_elts.find(packed_name) == m_packed_name_to_num_elts.end()");
    }
    tmp_data.reserve( m_packed_name_to_num_elts[packed_name] );

    // copy data for the requested fields into tmp_data
    for (const auto &field_name : t.second) {
      if (m_field_name_to_num_elts.find(field_name) == m_field_name_to_num_elts.end()) {
        LBANN_ERROR("m_field_name_to_num_elts.find(", field_name, ") failed");
      }
      used_field_names.insert(field_name);
      try {

        // assumption: node has a single top-level child
        // that is the lbann-assigned sample id
        const conduit::Node &n3 = node.child(0);
        auto vv = n3[field_name].value();
        size_t num_elts = m_field_name_to_num_elts[field_name];
        if (static_cast<conduit::index_t>(num_elts) != n3[field_name].dtype().number_of_elements()) {
          LBANN_ERROR("this should be impossible");
        }
        const std::string &dt = n3[field_name].dtype().name();

        if (dt == "float32") {
          const float* v = n3[field_name].value();
          for (size_t k=0; k<num_elts; k++) {
            tmp_data.emplace_back(v[k]);
          }
        } else if (dt == "float64") {
          const double* v = n3[field_name].value();
          for (size_t k=0; k<num_elts; k++) {
            tmp_data.emplace_back(v[k]);
          }
        } else {
          LBANN_ERROR("Please contact Dave Hysom, or other developer, to add support for your data type: ", dt);
        }

      } catch (conduit::Error const& e) {
        LBANN_ERROR("lbann caught conduit exception: ", e.what());
      }
    } // for field_name

    // remove the data that we've packed
    conduit::Node &n4 = node.child(0);
    for (const auto &field_name : t.second) {
       n4.remove(field_name);
    }
    // TODO: think about what to do: this can leave (pun intended)
    // empty nodes and subtrees; should these be removed? Easiest
    // thing is to leave them be ...

    node[packed_name] = tmp_data;
  } // for pack_to_field_name 
}

void hdf5_data_reader::parse_schemas() {

std::cout << "\nXX starting ::parse_schemas\n\n";

  // Get the pathnames for the fields that the user specified are
  // to be used in the current experiment. 
  //
  // Note that m_experiment_schema
  // may contain pruned trees, wrt m_data_schema; this obviates the need 
  // for the user to specify every field name. Hence, to get the complete
  // list of field names, we get the leaves from the m_experiment_schema, 
  // then trace them to the leaves in the m_data_schema (which is never
  // pruned)
  std::vector<const conduit::Node*> leaves_exp;
  get_leaves(&m_experiment_schema, leaves_exp);

  for (auto node : leaves_exp) {
    // WARNING: possible fragility ahead
    const std::string &pathname = node->path();
    size_t k = pathname.find('/');
    if (k != std::string::npos) {
      std::string pack_name = pathname.substr(0, k);
      std::string leaf_path = pathname.substr(k+1);
      auto iter = m_data_schema_nodes.find(leaf_path);
      if (iter == m_data_schema_nodes.end()) {
        LBANN_ERROR("failed to find ", leaf_path, " in m_data_schema_nodes");
      }
      std::vector<const conduit::Node*> leaves_data;
      get_leaves(iter->second, leaves_data);
      for (auto &field_node : leaves_data) {
        m_all_exp_leaves.insert(field_node);
        m_packed_to_field_names_map[pack_name].push_back(field_node->path());
        m_field_name_to_packed_map[field_node->path()] = pack_name;
      }
    }
  }

  tabulate_packing_memory_requirements();
}

// recursive
void hdf5_data_reader::get_schema_ptrs(conduit::Node* schema, std::unordered_map<std::string, conduit::Node*> &schema_name_map) {

  // process the input node 
  const std::string &path = schema->path();
  if (path == "") {
    if (!schema->is_root()) {
      LBANN_ERROR("path == '' but not root");
    }  
  } else {
    if (schema_name_map.find(path) != schema_name_map.end()) {
      LBANN_ERROR("duplicate pathname: ", path);
    }
    schema_name_map[path] = schema;
  }

  // recurse for each child
  int n_children = schema->number_of_children();
  for (int j=0; j<n_children; j++) {
    get_schema_ptrs(&schema->child(j), schema_name_map);
  }
}


// recursive
void hdf5_data_reader::get_leaves(const conduit::Node* schema_in, std::vector<const conduit::Node*> &leaves, std::string ignore_child_branch) {

  // nursery rhyme
  int n = schema_in->number_of_children();
  int n_children = 0;
  for (int j=0; j<n; j++) {
    if (schema_in->child_ptr(j)->name() != ignore_child_branch) {
      ++n_children;
    }
  }

  if (n_children == 0) {
    leaves.push_back(schema_in);
    return;
  } else {
    for (int j=0; j<n_children; j++) {
      if (schema_in->child_ptr(j)->name() != ignore_child_branch) {
        get_leaves(schema_in->child_ptr(j), leaves);
      }
    }
  }
}

void hdf5_data_reader::tabulate_packing_memory_requirements() {

  // load the schema from the actual data
  conduit::Schema schema;
  load_schema_from_data(schema);

  // fill in field_name -> num elts
  for (const auto &pack_name : m_packed_to_field_names_map) {
    size_t total_elts = 0;
    for (const auto &field_name : pack_name.second) {
      try {
        const conduit::Schema &s = schema.fetch_existing(field_name);
        size_t n_elts = s.dtype().number_of_elements();
        m_field_name_to_num_elts[field_name] = n_elts;
        total_elts += n_elts;
      } catch (conduit::Error const& e) {
        LBANN_ERROR("caught conduit::Error: ", e.what());
      }
    }
    m_packed_name_to_num_elts[pack_name.first] = total_elts;
  }
}

void hdf5_data_reader::load_schema_from_data(conduit::Schema &schema) {
  std::string json;
  if (is_master()) {
    // Load a node, then grab the schema. Note that we're loading the 
    // entire node; TODO fix this (but only if it proves too slow)
    conduit::Node node;
    const sample_t& s = m_sample_list[0];
    sample_file_id_t id = s.first;
    std::stringstream ss;
    ss << m_sample_list.get_samples_dirname() << "/" 
       << m_sample_list.get_samples_filename(id);
    conduit::relay::io::load(ss.str(), "hdf5", node);
    const conduit::Schema &tmp_schema = node.schema();
    json = tmp_schema.to_json(); 
  }
  m_comm->broadcast<std::string>(0, json, m_comm->get_world_comm());
  conduit::Schema data_schema(json);
  schema = data_schema.child(0);
}

} // namespace lbann
