#ifndef __SAMPLE_LIST_HPP__
#define __SAMPLE_LIST_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename SN = std::string>
class sample_list {
 public:
  /// The type of the native identifier of a sample rather than an arbitrarily assigned index
  using sample_name_t = SN;
  /// The type for arbitrarily assigned index
  using sample_id_t = size_t;
  /// Type for list of sample names in a sample file
  using sample_t = std::template pair<std::string, sample_name_t>;
  using flat_sample_list_t = std::template vector< sample_t >;
  using samples_t = std::template vector<sample_name_t>;
  /// Type for list where an element is the list of samples in a sample file
  using sample_files_t = std::template vector< std::pair<std::string, samples_t> >;

  sample_list() : m_num_partitions(1u) {}

  /// Set the number of partitions and clear internal states
  bool set_num_partitions(size_t n);

  /// Load a sample list from a file
  bool load(const std::string& samplelist_file);

  /// Extract a sample list from a serialized sample list in a string
  bool load_from_string(const std::string& samplelist);

  /// Write the current sample list into a file
  bool write(const std::string& out_filename) const;

  /// Clear internal states
  void clear();

  /// Serialize sample list for a partition
  bool to_string(size_t p, std::string& sstr) const;

  size_t get_samples_per_hdf5_file(std::istream& ifstr);
 protected:

  /// Populate m_samples_per_file by reading from input stream
  size_t get_samples_per_file(std::istream& istr);

  /// Extract m_samples_per_file by parsing a serialized string
  size_t get_samples_per_file(const std::string& samplelist);

  /// Populate the list of starting sample id for each sample file
  bool get_sample_range_per_file();

  /// Populate the list of starting sample id for each partition
  bool get_sample_range_per_part();

  /// Find the range of sample files that covers the range of samples of a partition
  bool find_sample_files_of_part(size_t p, size_t& sf_begin, size_t& sf_end) const;

  static std::string to_string(const std::string& s);

  template <typename T>
  static std::string to_string(const T v);

 protected:

  /// The number of partitions to divide samples into
  size_t m_num_partitions;

  /** In a sample file list, each line begins with a sample file name
   * that is followed by the names of the samples in the file.
   */
  sample_files_t m_samples_per_file;

  /// Contains list of all sample
  flat_sample_list_t m_sample_list;

  /// Contains starting sample id of each file
  std::vector<sample_id_t> m_sample_range_per_file;

  /// Contains starting sample id of each partition
  std::vector<sample_id_t> m_sample_range_per_part;

  /// indices to m_samples_per_file used for shuffling
  std::vector<unsigned> m_shuffled_indices;

};


} // end of namespace

#include "sample_list_impl.hpp"

#endif // __SAMPLE_LIST_HPP__
