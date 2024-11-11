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
# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  metadata.cpp
  data_reader_cifar10.cpp
  data_reader_conduit.cpp
  data_reader_csv.cpp
  data_reader_image.cpp
  data_reader_jag_conduit.cpp
  data_reader_merge_features.cpp
  data_reader_merge_samples.cpp
  data_reader_mesh.cpp
  data_reader_mnist.cpp
  data_reader_nci.cpp
  data_reader_node2vec.cpp
  data_reader_synthetic.cpp
  data_reader_python.cpp
  data_reader_python_dataset.cpp
  data_reader_smiles.cpp
  data_reader_HDF5.cpp
  )

if (LBANN_HAS_CNPY)
  set_full_path(CNPY_SRC_FILES
    data_reader_npz_ras_lipid.cpp
    data_reader_numpy.cpp
    data_reader_numpy_npz.cpp
    data_reader_numpy_npz_conduit.cpp
    data_reader_pilot2_molecular.cpp
    )
  list(APPEND THIS_DIR_SOURCES ${CNPY_SRC_FILES})
endif ()

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/data_reader_hdf5_legacy.cpp")
endif ()

if (LBANN_HAS_OPENCV)
  list(APPEND THIS_DIR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/data_reader_imagenet.cpp")
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
