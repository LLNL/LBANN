///////////////////////////////////////////////////////////////////////////////
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

#define LBANN_UNARY_LAYER_INSTANTIATE
#include "lbann/layers/math/unary.hpp"
#include "lbann/utils/entrywise_operator.hpp"

namespace lbann {

namespace {

// =========================================================
// Operator objects for entry-wise unary layers
// =========================================================
// Note: Unary operator corresponds to forward prop step
// (\f$ y = f(x) \f$) and binary operator corresponds to
// back prop step
// (\f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$).

/** Logical not operator. */
template <typename TensorDataType>
struct logical_not_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    const auto& b = x != El::TypeTraits<TensorDataType>::Zero() && !std::isnan(x);
    return !b ? El::TypeTraits<TensorDataType>::One() : El::TypeTraits<TensorDataType>::Zero();
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Absolute value operator. */
template <typename TensorDataType>
struct abs_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return x >= El::TypeTraits<TensorDataType>::Zero() ? x : -x;
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    if      (x > El::TypeTraits<TensorDataType>::Zero()) { return dy;   }
    else if (x < El::TypeTraits<TensorDataType>::Zero()) { return -dy;  }
    else               { return El::TypeTraits<TensorDataType>::Zero(); }
  }
};

/** Negative operator. */
template <typename TensorDataType>
struct negative_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return -x;
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy;
  }
};

/** Sign operator. */
template <typename TensorDataType>
struct sign_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    if      (x > El::TypeTraits<TensorDataType>::Zero()) { return El::TypeTraits<TensorDataType>::One();  }
    else if (x < El::TypeTraits<TensorDataType>::Zero()) { return -El::TypeTraits<TensorDataType>::One(); }
    else               { return El::TypeTraits<TensorDataType>::Zero(); }
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Round operator. */
template <typename TensorDataType>
struct round_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::round(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Ceiling operator. */
template <typename TensorDataType>
struct ceil_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::ceil(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Floor operator. */
template <typename TensorDataType>
struct floor_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::floor(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return El::TypeTraits<TensorDataType>::Zero();
  }
};

/** Reciprocal operator.
 *  If a standard reciprocal produces an infinity or NaN, El::TypeTraits<TensorDataType>::Zero() is
 *  output instead.
 */
template <typename TensorDataType>
struct reciprocal_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return 1 / x;
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    if (dy == El::TypeTraits<TensorDataType>::Zero()) { return El::TypeTraits<TensorDataType>::Zero(); }
    else            { return - dy / (x*x); }
  }
};

/** Square operator. */
template <typename TensorDataType>
struct square_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return x*x;
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return 2*x * dy;
  }
};


/** Square root operator. */
template <typename TensorDataType>
struct sqrt_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return El::Sqrt(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (2 * El::Sqrt(x));
  }
};

/** Reciprocal square root operator. */
template <typename TensorDataType>
struct rsqrt_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return 1 / El::Sqrt(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& s = El::Sqrt(x);
    return - dy / (2 * x * s);
  }
};

/** Safe reciprocal operator. */
template <typename TensorDataType>
struct safe_reciprocal_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    const auto& y = 1 / x;
    if (std::isfinite(y)) { return y; }
    else                  { return El::TypeTraits<TensorDataType>::Zero(); }
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& y = 1 / x;
    if (std::isfinite(y)) { return - dy * y*y; }
    else                  { return El::TypeTraits<TensorDataType>::Zero(); }
  }
};

/** ExpEl::TypeTraits<TensorDataType>::One()ntial operator. */
template <typename TensorDataType>
struct exp_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::exp(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * std::exp(x);
  }
};

/** ExpEl::TypeTraits<TensorDataType>::One()ntial minus El::TypeTraits<TensorDataType>::One() operator. */
template <typename TensorDataType>
struct expm1_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::expm1(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * std::exp(x);
  }
};

/** Natural logarithm operator. */
template <typename TensorDataType>
struct log_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::log(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / x;
  }
};

/** Natural logarithm El::TypeTraits<TensorDataType>::One() plus operator. */
template <typename TensorDataType>
struct log1p_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::log1p(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (x + El::TypeTraits<TensorDataType>::One());
  }
};

/** Cosine operator. */
template <typename TensorDataType>
struct cos_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::cos(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy * std::sin(x);
  }
};

/** Sine operator. */
template <typename TensorDataType>
struct sin_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::sin(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * std::cos(x);
  }
};

/** Tangent operator. */
template <typename TensorDataType>
struct tan_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::tan(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& c = std::cos(x);
    return dy / (c*c);
  }
};

/** Arccosine operator. */
template <typename TensorDataType>
struct acos_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::acos(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy / El::Sqrt(El::TypeTraits<TensorDataType>::One() - x*x);
  }
};

/** Arcsine operator. */
template <typename TensorDataType>
struct asin_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::asin(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / El::Sqrt(El::TypeTraits<TensorDataType>::One() - x*x);
  }
};

/** Arctangent operator. */
template <typename TensorDataType>
struct atan_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::atan(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (El::TypeTraits<TensorDataType>::One() + x*x);
  }
};

/** Hyperbolic cosine operator. */
template <typename TensorDataType>
struct cosh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::cosh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * std::sinh(x);
  }
};

/** Hyperbolic sine operator. */
template <typename TensorDataType>
struct sinh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::sinh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy * std::cosh(x);
  }
};

/** Hyperbolic tangent operator. */
template <typename TensorDataType>
struct tanh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::tanh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    const auto& c = std::cosh(x);
    return dy / (c*c);
  }
};

/** Hyperbolic arccosine operator. */
template <typename TensorDataType>
struct acosh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::acosh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return -dy / (El::Sqrt(x - El::TypeTraits<TensorDataType>::One()) * El::Sqrt(x + El::TypeTraits<TensorDataType>::One()));
  }
};

/** Hyperbolic arcsine operator. */
template <typename TensorDataType>
struct asinh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::asinh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / El::Sqrt(El::TypeTraits<TensorDataType>::One() + x*x);
  }
};

/** Hyperbolic arctangent operator. */
template <typename TensorDataType>
struct atanh_op {
  inline TensorDataType operator()(const TensorDataType& x) const {
    return std::atanh(x);
  }
  inline TensorDataType operator()(const TensorDataType& x, const TensorDataType& dy) const {
    return dy / (El::TypeTraits<TensorDataType>::One() - x*x);
  }
};

} // namespace

// Template instantiation
#define INSTANTIATE(layer, op)                                          \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::fp_compute() {            \
    apply_entrywise_unary_operator<op>(                                 \
      this->get_prev_activations(),                                     \
      this->get_activations());                                         \
  }                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::bp_compute() {            \
    apply_entrywise_binary_operator<op>(                                \
      this->get_prev_activations(),                                     \
      this->get_prev_error_signals(),                                   \
      this->get_error_signals());                                       \
  }                                                                     \
  UNARY_ETI_INST_MACRO_DEV(layer, El::Device::CPU)

INSTANTIATE(logical_not_layer, logical_not_op);
INSTANTIATE(abs_layer, abs_op);
INSTANTIATE(negative_layer, negative_op);
INSTANTIATE(sign_layer, sign_op);
INSTANTIATE(round_layer, round_op);
INSTANTIATE(ceil_layer, ceil_op);
INSTANTIATE(floor_layer, floor_op);
INSTANTIATE(reciprocal_layer, reciprocal_op);
INSTANTIATE(square_layer, square_op);
INSTANTIATE(sqrt_layer, sqrt_op);
INSTANTIATE(rsqrt_layer, rsqrt_op);
INSTANTIATE(safe_reciprocal_layer, safe_reciprocal_op);
INSTANTIATE(exp_layer, exp_op);
INSTANTIATE(expm1_layer, expm1_op);
INSTANTIATE(log_layer, log_op);
INSTANTIATE(log1p_layer, log1p_op);
INSTANTIATE(cos_layer, cos_op);
INSTANTIATE(sin_layer, sin_op);
INSTANTIATE(tan_layer, tan_op);
INSTANTIATE(acos_layer, acos_op);
INSTANTIATE(asin_layer, asin_op);
INSTANTIATE(atan_layer, atan_op);
INSTANTIATE(cosh_layer, cosh_op);
INSTANTIATE(sinh_layer, sinh_op);
INSTANTIATE(tanh_layer, tanh_op);
INSTANTIATE(acosh_layer, acosh_op);
INSTANTIATE(asinh_layer, asinh_op);
INSTANTIATE(atanh_layer, atanh_op);

} // namespace lbann
