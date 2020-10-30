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
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_UTILS_DNN_LIB_CUDNN_CONVOLUTION_HPP_
#define LBANN_UTILS_DNN_LIB_CUDNN_CONVOLUTION_HPP_

#include "lbann/utils/ml_enums.hpp"
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "utils.hpp"

namespace lbann
{

#ifdef LBANN_HAS_CUDNN
namespace dnn_lib
{

using namespace cudnn;

template <typename TensorDataType, typename ScalarParameterType>
void convolution_forward(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  ConvolutionDescriptor const& convDesc,
  fwd_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& yDesc,
  El::AbstractMatrix<TensorDataType>& y,
  El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(hipdnnConvolutionForward(handle_manager.get(),
                                      &alpha,
                                      xDesc,
                                      x.LockedBuffer(),
                                      wDesc,
                                      w.LockedBuffer(),
                                      convDesc,
                                      cudnn::to_cudnn(alg),
                                      workSpace.Buffer(),
                                      workSpace.Height()*sizeof(TensorDataType),
                                      &beta,
                                      yDesc,
                                      y.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_forward(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  ConvolutionDescriptor const& convDesc,
  fwd_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& yDesc,
  El::AbstractMatrix<TensorDataType>& y)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(y),
                                     gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(w),
                                     gpu::get_sync_info(x));
  convolution_forward(alpha_in, xDesc, x, wDesc, w, convDesc, alg,
                      workSpace, beta_in, yDesc, y, multisync);
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_data(
  ScalarParameterType const& alpha_in,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  bwd_data_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dxDesc,
  El::AbstractMatrix<TensorDataType>& dx,
  El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(hipdnnConvolutionBackwardData(handle_manager.get(),
                                      &alpha,
                                      wDesc,
                                      w.LockedBuffer(),
                                      dyDesc,
                                      dy.LockedBuffer(),
                                      convDesc,
                                      cudnn::to_cudnn(alg),
                                      workSpace.Buffer(),
                                      workSpace.Height()*sizeof(TensorDataType),
                                      &beta,
                                      dxDesc,
                                      dx.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_data(
  ScalarParameterType const& alpha_in,
  FilterDescriptor const& wDesc,
  El::AbstractDistMatrix<TensorDataType> const& w,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  bwd_data_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dxDesc,
  El::AbstractMatrix<TensorDataType>& dx)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(dx),
                                     gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(dy),
                                     gpu::get_sync_info(w));
  convolution_backward_data(alpha_in, wDesc, w, dyDesc, dy, convDesc, alg,
                            workSpace, beta_in, dxDesc, dx, multisync);

}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_bias(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dbDesc,
  El::AbstractDistMatrix<TensorDataType>& db,
  El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(hipdnnConvolutionBackwardBias(handle_manager.get(),
                                           &alpha,
                                           dyDesc,
                                           dy.LockedBuffer(),
                                           &beta,
                                           dbDesc,
                                           db.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_bias(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ScalarParameterType const& beta_in,
  TensorDescriptor const& dbDesc,
  El::AbstractDistMatrix<TensorDataType>& db)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(db),
                                     gpu::get_sync_info(dy));
  convolution_backward_bias(alpha_in, dyDesc, dy,
                            beta_in, dbDesc, db,
                            multisync);
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_filter(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  bwd_filter_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  FilterDescriptor const& dwDesc,
  El::AbstractDistMatrix<TensorDataType>& dw,
  El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(hipdnnConvolutionBackwardFilter(handle_manager.get(),
                                      &alpha,
                                      xDesc,
                                      x.LockedBuffer(),
                                      dyDesc,
                                      dy.LockedBuffer(),
                                      convDesc,
                                      cudnn::to_cudnn(alg),
                                      workSpace.Buffer(),
                                      workSpace.Height()*sizeof(TensorDataType),
                                      &beta,
                                      dwDesc,
                                      dw.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void convolution_backward_filter(
  ScalarParameterType const& alpha_in,
  TensorDescriptor const& xDesc,
  El::AbstractMatrix<TensorDataType> const& x,
  TensorDescriptor const& dyDesc,
  El::AbstractMatrix<TensorDataType> const& dy,
  ConvolutionDescriptor const& convDesc,
  bwd_filter_conv_alg alg,
  El::Matrix<TensorDataType, El::Device::GPU>& workSpace,
  ScalarParameterType const& beta_in,
  FilterDescriptor const& dwDesc,
  El::AbstractDistMatrix<TensorDataType>& dw)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(dw),
                                     gpu::get_sync_info(workSpace),
                                     gpu::get_sync_info(dy),
                                     gpu::get_sync_info(x));
  convolution_backward_filter(alpha_in, xDesc, x, dyDesc, dy, convDesc, alg,
                              workSpace, beta_in, dwDesc, dw, multisync);
}

template <typename TensorDataType, typename ScalarParameterType>
void add_tensor(ScalarParameterType const& alpha_in,
                TensorDescriptor const& aDesc,
                El::AbstractDistMatrix<TensorDataType> const& A,
                ScalarParameterType const& beta_in,
                TensorDescriptor const& cDesc,
                El::AbstractMatrix<TensorDataType>& C,
                El::SyncInfo<El::Device::GPU> const& si)
{
  using LibScalingParamT = dnn_lib::ScalingParamType<TensorDataType>;
  auto handle_manager = internal::make_default_handle_manager(si);
  auto alpha = El::To<LibScalingParamT>(alpha_in);
  auto beta = El::To<LibScalingParamT>(beta_in);
  CHECK_CUDNN(hipdnnAddTensor(handle_manager.get(),
                             &alpha,
                             aDesc,
                             A.LockedBuffer(),
                             &beta,
                             cDesc,
                             C.Buffer()));
}

template <typename TensorDataType, typename ScalarParameterType>
void add_tensor(ScalarParameterType const& alpha_in,
                TensorDescriptor const& aDesc,
                El::AbstractDistMatrix<TensorDataType> const& A,
                ScalarParameterType const& beta_in,
                TensorDescriptor const& cDesc,
                El::AbstractMatrix<TensorDataType>& C)
{
  auto multisync = El::MakeMultiSync(gpu::get_sync_info(C),
                                     gpu::get_sync_info(A));
  add_tensor(alpha_in, aDesc, A, beta_in, cDesc, C, multisync);
}

}// namespace cudnn
#endif // LBANN_HAS_CUDNN
}// namespace lbann
#endif // LBANN_UTILS_DNN_LIB_CUDNN_CONVOLUTION_HPP_
