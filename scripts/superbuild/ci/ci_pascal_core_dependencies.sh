################################################################################
## Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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

# Set to ON (or any CMake truthy value) to build all of the
# dependencies of the LBANN stack
BUILD_EXTERNAL_TPLS=ON

# Set to the directory with the top-level CMakeLists.txt file for LBANN
LBANN_SRC_DIR=$(git rev-parse --show-toplevel)

# Set to the directory with the top-level SuperBuild CMakeLists.txt file
SUPERBUILD_SRC_DIR=${LBANN_SRC_DIR}/scripts/superbuild

# Setup the common environment
source ${SUPERBUILD_SRC_DIR}/ci/ci_pascal_env.sh

# Set to the preferred install directory
INSTALL_PREFIX=${INSTALL_PREFIX_EXTERNALS}

# Set to the preferred build directory
BUILD_DIR=${TMPDIR}/lbann-superbuild-core-dependencies

#export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

cmake \
    -G Ninja \
    -S ${SUPERBUILD_SRC_DIR} \
    -B ${BUILD_DIR} \
    \
    -D CMAKE_PREFIX_PATH=${FWD_CMAKE_PREFIX_PATH} \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    \
    -D CMAKE_C_COMPILER=$(which gcc) \
    -D CMAKE_CXX_COMPILER=$(which g++) \
    -D CMAKE_CUDA_COMPILER=$(command -v nvcc) \
    -D CMAKE_CUDA_HOST_COMPILER=$(command -v g++) \
    -D CMAKE_Fortran_COMPILER=$(which gfortran) \
    \
    -D CMAKE_EXE_LINKER_FLAGS=${EXTRA_LINK_FLAGS} \
    -D CMAKE_SHARED_LINKER_FLAGS=${EXTRA_LINK_FLAGS} \
    \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CUDA_STANDARD=17 \
    -D CMAKE_CUDA_ARCHITECTURES=${CUDA_GPU_ARCH} \
    \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
    \
    -D LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY="PKG_LC" \
    -D LBANN_SB_DEFAULT_CUDA_OPTS=ON \
    \
    -D LBANN_SB_BUILD_adiak=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_Caliper=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_adiak_BUILD_SHARED_LIBS=ON \
    -D LBANN_SB_Caliper_BUILD_SHARED_LIBS=ON \
    \
    -D LBANN_SB_BUILD_Catch2=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_cereal=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_Clara=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_CNPY=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_protobuf=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_spdlog=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_zstr=${BUILD_EXTERNAL_TPLS} \
    \
    -D LBANN_SB_BUILD_Conduit=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_HDF5=${BUILD_EXTERNAL_TPLS} \
    \
    -D LBANN_SB_BUILD_JPEG-TURBO=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_BUILD_OpenCV=${BUILD_EXTERNAL_TPLS} \
    -D LBANN_SB_OpenCV_TAG=4.x