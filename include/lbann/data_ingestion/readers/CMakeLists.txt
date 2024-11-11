################################################################################
## Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################
# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  compound_data_reader.hpp
  metadata.hpp
  # Data readers
  data_reader_cifar10.hpp
  data_reader_conduit.hpp
  data_reader_csv.hpp
  data_reader_image.hpp
  data_reader_HDF5.hpp
  data_reader_merge_features.hpp
  data_reader_merge_samples.hpp
  data_reader_mnist.hpp
  data_reader_nci.hpp
  data_reader_node2vec.hpp
  data_reader_python.hpp
  data_reader_python_dataset.hpp
  data_reader_synthetic.hpp
  data_reader_smiles.hpp
  data_reader_sample_list.hpp
  data_reader_sample_list_impl.hpp
  )

if (LBANN_HAS_CNPY)
  set_full_path(CNPY_HDR_FILES
    data_reader_npz_ras_lipid.hpp
    data_reader_numpy.hpp
    data_reader_numpy_npz.hpp
    data_reader_numpy_npz_conduit.hpp
    data_reader_pilot2_molecular.hpp
    )
  list(APPEND THIS_DIR_HEADERS ${CNPY_HDR_FILES})
endif ()

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/data_reader_hdf5_legacy.hpp")
endif ()

if (LBANN_HAS_OPENCV)
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/data_reader_imagenet.hpp")
endif ()

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
