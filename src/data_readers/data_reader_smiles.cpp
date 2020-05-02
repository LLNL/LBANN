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

#include "lbann/data_readers/data_reader_smiles.hpp"
#include "lbann/data_store/data_store_conduit.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/utils/commify.hpp"
#include "lbann/utils/lbann_library.hpp"
#include <mutex>

namespace lbann {

smiles_data_reader::smiles_data_reader(const bool shuffle)
  : generic_data_reader(shuffle) {}

smiles_data_reader::smiles_data_reader(const smiles_data_reader& rhs)  : generic_data_reader(rhs) {
  copy_members(rhs);
}

smiles_data_reader::~smiles_data_reader() {
  if (m_missing_chars.size()) {
    if (is_master()) {
      std::cout << std::endl << "The following tokens were in SMILES strings, but were missing from the vocabulary: ";
      for (const auto t : m_missing_chars) {
        std::cout << t << " ";
      }
    }
    std::cout << std::endl;
  }
}

smiles_data_reader& smiles_data_reader::operator=(const smiles_data_reader& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }
  generic_data_reader::operator=(rhs);
  copy_members(rhs);
  return (*this);
}


void smiles_data_reader::copy_members(const smiles_data_reader &rhs) {
  if(rhs.m_data_store != nullptr) {
      m_data_store = new data_store_conduit(rhs.get_data_store());
  }
  m_num_samples = rhs.m_num_samples;
  m_data_store->set_data_reader_ptr(this);
  m_linearized_data_size = rhs.m_linearized_data_size;
  m_linearized_label_size = rhs.m_linearized_label_size;
  m_linearized_response_size = rhs.m_linearized_response_size;
  m_num_labels = rhs.m_num_labels;
  m_pad = rhs.m_pad;
  m_unk = rhs.m_unk;
  m_bos = rhs.m_bos;
  m_eos = rhs.m_eos;
  m_has_header = rhs.m_has_header;
  m_delimiter = rhs.m_delimiter;
  m_missing_char_in_vocab_count = rhs.m_missing_char_in_vocab_count;
  m_missing_chars = rhs.m_missing_chars;
  m_fast_experimental = rhs.m_fast_experimental;
}

void smiles_data_reader::load() {
  if(is_master()) {
    std::cout << "starting load for role: " << get_role() << std::endl;
  }

  options *opts = options::get();

  m_fast_experimental = true;

  #if 0
  TODO: leaving this in, for possible future development using the data_store

  m_fast_experimental = false;
  if (options::get()->get_bool("fast_experimental")) {
    m_fast_experimental = true;
    if (is_master()) {
      std::cerr << "\nSMILES_DATA_READER is running in --fast_experimental mode\n";
    }
  } else { // run in normal --preload_data_store mode
    if (!opts->has_int("preload_data_store")) {
      if (is_master()) {
        LBANN_WARNING("setting --preload_data_store");
      }
      opts->set_option("preload_data_store", 1);
    }
  }
  #endif

  if (!opts->has_int("sequence_length")) {
    LBANN_ERROR("you must pass --sequence_length=<int> on the cmd line");
  }
  m_linearized_data_size = opts->get_int("sequence_length") +2;

  // load the vocabulary; this is a map: string -> short
  load_vocab();

  // Count total number of samples in the file
  //m_has_header = !opts->get_bool("no_header");
  // TODO: fix this!
  m_has_header = true;

  const std::string infile = get_file_dir() + "/" + get_data_filename();
  m_total_samples = get_num_lines(infile);
  if (m_has_header) {
    --m_total_samples;
  }

  // Get the number of samples to use (this is separate from "percent_of_data_to_use," etc);
  m_num_samples = m_total_samples;
  if (opts->has_int("num_samples")) {
    m_num_samples = opts->get_int("num_samples");
  }
  if (m_num_samples > m_total_samples) {
    LBANN_ERROR("You requested to use ", m_num_samples, " samples, but input file only contains ", m_total_samples);
  }

  // Do the usual things we always do to finish up ...
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  resize_shuffled_indices();

  instantiate_data_store();
  select_subset_of_data();

  // TODO: does this work if we carve of a validation set?
  if (m_fast_experimental) {
    setup_fast_experimental();
  }

  print_statistics();
}

// TODO: relook if/when we start using data_store
void smiles_data_reader::do_preload_data_store() {
  if (is_master()) {
    std::cout << "starting do_preload_data_store; num indices: " 
              << utils::commify(m_shuffled_indices.size()) 
              << " for role: " << get_role() << std::endl;
  }
  m_data_store->set_node_sizes_vary();
  const std::string infile = get_file_dir() + "/" + get_data_filename();
  std::ifstream in(infile.c_str());
  if (!in) {
    LBANN_ERROR("failed to open data file: ", infile, " for reading");
  }
  std::string line;
  if (m_has_header) {
    getline(in, line); 
  }

  //TODO: this is terrible! Maybe: have root scan the file and bcast a map:
  //      offset->line number
  int rank = m_comm->get_rank_in_trainer();
  conduit::Node node;
  for (size_t data_id=0; data_id<m_shuffled_indices.size(); data_id++) {
    getline(in, line);
    int index = m_shuffled_indices[data_id];
    if (m_data_store->get_index_owner(index) != rank) {
      continue;
    }
    construct_conduit_node(index, line, node);
    m_data_store->set_preloaded_conduit_node(index, node);
  }
  in.close();
}

bool smiles_data_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  if (m_fast_experimental) {
    std::vector<short> data;
    get_sample(data_id, data);

    size_t j;
    for (j = 0; j < data.size(); ++j) {
      X(j, mb_idx) = data[j]; 
    }
    for (int jj = j; jj<m_linearized_data_size; jj++) {
      X(jj, mb_idx) = m_pad;
    }
  }

  #if 0
  // Leaving the following in for now for possible future development

  // run with data_store in preload mode
  else {
    bool have_node = true;
    conduit::Node node;
    if (m_data_store != nullptr) {
      if (data_store_active()) {
        //get data from node from data store
        const conduit::Node& ds_node = m_data_store->get_conduit_node(data_id);
        node.set_external(ds_node);
        have_node = true;
      } else if (priming_data_store()) {
        //get data from file, and stuff it in the data store;
        // data is retrieved in vector<short> data
        have_node = false;
        LBANN_ERROR("explicit loading not currently supported; it should be impossible to be here");
      } else {
        have_node = false;
        if (get_role() != "test") {
          LBANN_ERROR("using data_store for smiles_data_reader::fetch_datum is not implemented for role=test (actually, you probably shouldn't be here; please contact Dave Hysom)");
        } else {
         LBANN_ERROR("data_store is active, but data_store_active()=false and priming_data_store() = false. It should be impossible to be here; please contact Dave Hysom");
        }
      }
    }
  
    // this block fires if not using data store
    else {
       LBANN_ERROR("smiles_data_reader requires the data_store; this may change in the future");
      //read_datum(data_id, data_vector);
    }
  
    short *v;
    int n = 0;
    if (have_node) {
      //get v* from conduit node
      v =  node[LBANN_DATA_ID_STR(data_id) + "/data"].value(); 
      n = node[LBANN_DATA_ID_STR(data_id) + "/size"].value();
    } else {
      //TODO
    }
    int j;
    for (j = 0; j < n; ++j) {
      X(j, mb_idx) = v[j]; 
    }
    for (; j<m_linearized_data_size; j++) {
      X(j, mb_idx) = m_pad;
    }
  }
#endif

  return true;
}

bool smiles_data_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("smiles_data_reader::fetch_label is not implemented");
  return true;
}

bool smiles_data_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  LBANN_ERROR("smiles_data_reader::fetch_response is not implemented");
  return true;
}


//user feedback
void smiles_data_reader::print_statistics() const {
  if (!is_master()) {
    return;
  }

  std::cout << "\n======================================================\n";
  std::cout << "role: " << get_role() << std::endl;
  std::cout << "num samples=" << m_shuffled_indices.size() << std::endl;
  std::cout << "max sequence length=" << m_linearized_data_size << std::endl;
  std::cout << "num features=" << m_linearized_data_size << std::endl;
  std::cout << "delimiter= " << m_delimiter << std::endl;
  std::cout << "pad index= " << m_pad << std::endl;
  std::cout << "vocab size= " << m_vocab.size() << std::endl;
  std::cout << "======================================================\n\n";
}

void smiles_data_reader::load_vocab() {
  options *opts = options::get();
  if (!opts->has_string("vocab")) {
    LBANN_ERROR("you must pass --vocab=<string> on the command line");
  }
  const std::string fn = opts->get_string("vocab");
  std::ifstream in(fn.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", fn, " for reading; this is the vocabulary file");
  }
  std::string token;
  short id;
  int sanity = 4;
  while (in >> token >> id) {
    m_vocab_inv[id] = token;
    if (token.size() == 1) {
      m_vocab[token[0]] = id;
    }  
    if (token == "<pad>") {
      m_pad = id;
      --sanity;
    }
    if (token == "<unk>") {
      m_unk = id;
      --sanity;
    }
    if (token == "<bos>") {
      m_bos = id;
      --sanity;
    }
    if (token == "<eos>") {
      m_eos = id;
      --sanity;
    }
  }
  in.close();
  if (sanity) {
    LBANN_ERROR("failed to find <pad> and/or <unk> and/or <bos> and/or <eos> in vocab file: ", fn);
  }
  if (opts->has_int("pad_index")) {
    short tmp = opts->get_int("pad_index");
    if (tmp != m_pad) {
      LBANN_ERROR("you passed --pad_index=", tmp, " but we got --pad_index=", m_pad, " from the vocabulary file");
    }
  }
}

int smiles_data_reader::get_num_lines(std::string fn) {
  double tm1 = get_time();
  if (is_master()) {
    std::cout << "starting: count number of lines in the input file" << std::endl;
  }

  int count = 0;
  if (is_master()) {
    std::ifstream in(fn.c_str());
    if (!in) {
      LBANN_ERROR("failed to open data file: ", fn, " for reading");
    }
    std::string line;
    if (m_has_header) {
      getline(in,line);
    }
    while(getline(in,line)) {
      ++count;
    }
    in.close();

    std::cout << "smiles_data_reader::get_num_lines; num_lines: " 
              << count << " time: " << get_time()-tm1 << std::endl;
  }
  m_comm->broadcast<int>(0, &count, 1, m_comm->get_world_comm());
  if (is_master()) {
    std::cout << "time to count number of lines in the input file: " << get_time() - tm1 << std::endl;
  }
  return count;
}

void smiles_data_reader::construct_conduit_node(int data_id, const std::string &line, conduit::Node &node) {
#if 0
  node.reset();
  size_t j = line.find('\t');
  if (j == std::string::npos) {
    j = line.find(',');
  }
  if (j == std::string::npos) {
    LBANN_ERROR("failed to find delimit character (tab or comma) in line: ", line);
  }
  std::string sm = line.substr(0, j);
  std::vector<short> data;
  encode_smiles(sm, data, data_id);
  node[LBANN_DATA_ID_STR(data_id) + "/data"].set(data);
  int sz = data.size();
  node[LBANN_DATA_ID_STR(data_id) + "/size"].set(sz);
#endif
}


void smiles_data_reader::decode_smiles(const std::vector<short> &data, std::string &out) {
  std::stringstream s;
  for (const auto &t : data) {
    if (m_vocab_inv.find(t) == m_vocab_inv.end()) {
      std::stringstream s2;
      s2 <<"failed to find: " << t <<" in m_vocab_inv for input data: ";
      for (auto tt : data) {
        s2 << tt << " ";
      }
      LBANN_ERROR(s2.str());
    }
    const std::string &x = m_vocab_inv[t];
    if (x == "<unk>") {
      s << "<unk>";
    } else if (!(x == "<bos>" || x == "<eos>" || x == "<pad>")) {
      s << m_vocab_inv[t];
    } 
  }
  out = s.str();
}

// TODO: encode_smiles should work on const char* instead of string; this
//       make it more generalizable, re, test_encode
void smiles_data_reader::encode_smiles(const char *smiles, short size, std::vector<short> &data, int data_id) {
  static int count = 0;

  int stop = size;
  if (stop+2 > m_linearized_data_size) { //+2 is for <bos> and <eos>
    stop = m_linearized_data_size-2;
    if (count < 20) {
      ++count;
      LBANN_WARNING("data_id: ", data_id, " smiles string size is ", size, "; losing ", (size-(m_linearized_data_size-2)), " characters");
    }
  }

  data.clear();
  data.reserve(stop+2);
  data.push_back(m_bos);
  for (int j=0; j<stop; j++) {
    if (m_vocab.find(smiles[j]) == m_vocab.end()) {
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_missing_chars.insert(smiles[j]);
        ++m_missing_char_in_vocab_count;
        if (m_missing_char_in_vocab_count < 20) {
          std::stringstream ss;
          ss << "rank: " << m_comm->get_rank_in_trainer() << "; character not in vocab >>" << smiles[j] << "<<; idx: " << j << "; data_id: " << data_id << "; string length: " << size << "; will use length: " << stop;
          std::cerr << ss.str();
        }
      }
      data.push_back(m_unk);
    } else {
      data.push_back(m_vocab[smiles[j]]);
    }
  }
  data.push_back(m_eos);
}

void smiles_data_reader::get_sample(int sample_id, std::vector<short> &sample_out) {
  std::unordered_map<int, std::pair<size_t, short>>::const_iterator iter = m_sample_lookup.find(sample_id);
  if (iter == m_sample_lookup.end()) {
    std::stringstream s;
    s << "; m_sample_lookup.size: " << m_sample_lookup.size() << " known data_ids: ";
    for (auto t : m_sample_lookup) s << t.first << " ";
    LBANN_ERROR("failed to find data_id ", sample_id, " in m_sample_lookup", s.str());
  }

  size_t offset = iter->second.first;
  short size = iter->second.second;
  if (offset + size > m_data.size()) {
    LBANN_ERROR("offset: ", offset, " + size: ", size, " is > m_data.size(): ", m_data.size());
  }

  const char *smiles = m_data.data()+offset;
  encode_smiles(smiles, size, sample_out, sample_id);
}

int smiles_data_reader::get_smiles_string_length(const std::string &line, int line_number) {
  if (m_delimiter == '\0') {
    return line.size();
  }
  size_t k = line.find(m_delimiter);
  if (k == std::string::npos) {
    LBANN_ERROR("failed to find delimit character >>", m_delimiter, " in line: ", line, " which is line number ", line_number);
  }
  return k;
}


// TODO: break into several function calls ??
// TODO: some/most of the following could/should be in the data_store ??
void smiles_data_reader::setup_fast_experimental() {
  double tm3 = get_time();
  if (is_master()) {
    std::cout << "\nSTARTING smiles_data_reader::setup_fast_experimental() " << std::endl << std::endl;
  }  

  // Get delimiter; default delimiter is none ('\0'), though it's likely
  // to be ',' or '\t', since we're likely reading csv files
  options *opts = options::get();
  if (opts->has_string("delimiter")) {
    const std::string d = opts->get_string("delimiter");
    m_delimiter = d[0];
  }
  if (is_master()) {
    std::cout << "USING delimiter character: " << m_delimiter << std::endl;
  }

  // This will hold: (dataum_id, datum_offset, datum length) for each sample
  std::vector<size_t> sample_offsets(m_shuffled_indices.size()*3);

  // Will hold size of above buffer, for bcasting
  size_t buffer_size;

  if (is_master()) {
    double tm1 = get_time();

    // Open input file and discard header line, if it exists
    const std::string infile = get_file_dir() + "/" + get_data_filename();
    std::ifstream in(infile.c_str());
    if (!in) {
      LBANN_ERROR("failed to open data file: ", infile, " for reading");
    }
    std::string line;
    if (m_has_header) {
      getline(in, line);
    }  

    // Part 1: compute memory requirements for local cache

    // Get max sample id, which will be the number of lines we need to 
    // read from file. This is needed if (1) not using 100% of data,
    // and/or (2) carving off part of train data to use as validation.
    std::unordered_set<int> samples_to_use;
    int max_sample_id = 0; 
    for (size_t j=0; j<m_shuffled_indices.size(); j++) {
      samples_to_use.insert(m_shuffled_indices[j]);
      max_sample_id = m_shuffled_indices[j] > max_sample_id ? m_shuffled_indices[j] : max_sample_id;
    }
    ++max_sample_id;

    // Construct sample_offsets vector
    sample_offsets.clear();
    size_t offset = 0;
    for (int j=0; j<max_sample_id; j++) {
      getline(in, line);
      if (line.size() < 5) {
        LBANN_ERROR("read ", j, " lines from file; could not read another. --num_samples is probably incorrect");
      }
      if (samples_to_use.find(j) != samples_to_use.end()) {
        int k = get_smiles_string_length(line, j);
        sample_offsets.push_back(j);
        sample_offsets.push_back(offset);
        sample_offsets.push_back(k);
        offset += k;
      }
    }
    buffer_size = offset;
    m_data.resize(buffer_size);

    // Part 2: Fill in the data buffer
    in.seekg(0);
    if (m_has_header) {
      getline(in, line); 
    }  
    offset = 0;
    for (int j=0; j<max_sample_id; j++) {
      getline(in, line);
      if (samples_to_use.find(j) != samples_to_use.end()) {
        int k = get_smiles_string_length(line, j);
        for (int n=0; n<k; n++) {
          m_data[n+offset] = line[n];
        }
        offset += k;
      }
    }

    if (sample_offsets.size()/3 != m_shuffled_indices.size()) {
      LBANN_ERROR("sample_offsets.size()/3: ", sample_offsets.size()/3, " should be equal to m_shuffled_indices.size which is ", m_shuffled_indices.size());
    }
    std::cout << "P_0 time for computing sample sizes and filling buffer: " << get_time() - tm1 << std::endl;
  }

  // Construct lookup table for locating samples in the m_data vector (aka, the sample buffer)
  m_comm->broadcast<size_t>(0, sample_offsets.data(), sample_offsets.size(), m_comm->get_world_comm());
  for (size_t j=0; j<sample_offsets.size(); j += 3) {
    m_sample_lookup[sample_offsets[j]] = 
      std::make_pair(sample_offsets[j+1], sample_offsets[j+2]);
  }

  // Bcast the sample buffer
  m_comm->broadcast<size_t>(0, &buffer_size, 1, m_comm->get_world_comm());
  m_data.resize(buffer_size);

  int full_rounds = m_data.size() / INT_MAX;
  int last_round = m_data.size() % INT_MAX;
  size_t the_offset = 0;

  for (int j=0; j<full_rounds; j++) {
    m_comm->broadcast<char>(0, m_data.data()+the_offset, INT_MAX, m_comm->get_world_comm());
    the_offset += INT_MAX;
  }
  if (last_round) {
    m_comm->broadcast<char>(0, m_data.data()+the_offset, last_round, m_comm->get_world_comm());
  }

  if (is_master()) {
    std::cout << "total time for loading data: " << get_time()-tm3 << std::endl
              << "num samples: " << m_sample_lookup.size() << std::endl;
  }

  // Only used for testing/debugging during development
  if (opts->get_bool("test_encode")) {
    test_encode();
  }
}

void smiles_data_reader::test_encode() {
  // What this does: at this point, P_0 has read and bcast the data set,
  // and each rank has built a lookup table. Below, P_1 looks up each
  // data_id; encodes the string (E1); reads the string from file (S2); 
  // decodes E1 to produce string S1; compares S1 and S2 for equality.
  double tm1 = get_time();
  if (is_master()) {
    std::cout << "STARTING TEST_ENCODE" << std::endl;
  }
  if (m_comm->get_rank_in_world() != 1) {
    return;
  }

  // option: testing the test ;)
  bool fail = options::get()->get_bool("make_test_fail");

  // Build ordered set of data_ids so we can more easily iterate
  // through the file -- instead of jumping around
  std::set<int> data_ids;
  for (auto t : m_sample_lookup) {
    data_ids.insert(t.first);
  }

  // Open input file and discard header (if it exists)
  const std::string infile = get_file_dir() + "/" + get_data_filename();
  std::ifstream in(infile.c_str());
  if (!in) {
    LBANN_ERROR("failed to open ", infile, " for reading");
  }
  std::string line;
  if (m_has_header) {
    getline(in, line);
  }

  size_t num_tested = 0;

  std::vector<short> encoded;
  std::string decoded;
  int sample_id = -1;
  while (getline(in, line)) {
    ++sample_id;

    if (data_ids.find(sample_id) != data_ids.end()) {
      ++num_tested;
      // encode then decode the datum that is stored in memory
      get_sample(sample_id, encoded);
      decode_smiles(encoded, decoded); 

      // get datum length from the line we've just read from file
      size_t k = get_smiles_string_length(line, sample_id);
      std::string S2(line.data(), k);

      // test the test! Optionally make the test fail;
      // assumes smiles string contains at least 8 characters,
      // and no string contains "~~~~"
      if (num_tested > 10 && fail) {
        for (size_t h=3; h<7; h++) {
          S2[h] = '~';
        }  
      }

      // conduct tests
      if (S2.size() != decoded.size()) {
        LBANN_ERROR("S2.size (", S2.size(), ") != decoded.size (", decoded.size(), ")");
      }
      if (S2 != decoded) {
          LBANN_ERROR("test_encoded failed; string from memory: ", decoded, "; string from file: ", S2, "; should be equal"); 
      }
    }
  }
  in.close();

  if (num_tested != m_sample_lookup.size()) {
    LBANN_ERROR("num_tested= ", num_tested, "; m_sample_lookup.size()= ", m_sample_lookup.size(), "; should be equal");
  }

  std::cout << "ENDING TEST_ENCODE; time: " << get_time()-tm1 << std::endl;
}

}  // namespace lbann
