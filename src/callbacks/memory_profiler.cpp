////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
// memory_profiler .hpp .cpp - Itemized memory usage profiling.
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/memory_profiler.hpp"
#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/weights/data_type_weights.hpp"
#include "lbann/weights/weights.hpp"

#include "lbann/proto/callbacks.pb.h"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/distconv_adapter.hpp"
#endif

#include "h2/patterns/multimethods/SwitchDispatcher.hpp"

#include <algorithm>
#include <string>

#ifdef LBANN_HAS_GPU
#include <h2/gpu/memory_utils.hpp>
#endif

namespace lbann {
namespace callback {
namespace {

using Datatypes = h2::meta::TL<float,
                               double
#ifdef LBANN_HAS_HALF
                               ,
                               cpu_fp16
#endif
#ifdef LBANN_HAS_GPU_FP16
                               ,
                               fp16
#endif
                               >;
// Need a unary template
template <typename T>
using UnaryConstDTL = const data_type_layer<T, T>;

template <typename T>
using ConstADM = std::add_const_t<El::AbstractDistMatrix<T>>;

using LayerTypes = h2::meta::tlist::ExpandTL<UnaryConstDTL, Datatypes>;
using DistMatrixTypes = h2::meta::tlist::ExpandTL<ConstADM, Datatypes>;

// Now create the functors
/**
 * @brief Prints out the shape and allocated size of a matrix to the stream
 * given in the second argument. Returns the allocated size as well.
 */
template <typename T>
size_t report_dist_matrix(El::AbstractDistMatrix<T> const& m,
                          std::ostream& stream)
{
  size_t const allocated = m.AllocatedMemory() * sizeof(T);
  stream << m.Height() << " x " << m.Width()
         << " (local shape: " << m.LocalHeight() << " x " << m.LocalWidth()
         << "). Size: " << allocated / 1048576.0 << " MiB" << std::endl;
  return allocated;
}

#ifdef LBANN_HAS_DISTCONV
template <typename T>
size_t report_distconv_matrix(::lbann::dc::TensorDev<T> const& m,
                              std::ostream& stream)
{
  size_t const allocated = m.get_local_real_size() * sizeof(T);
  auto const& shp = m.get_shape();
  auto const& lshp = m.get_local_real_shape();
  stream << shp[0];
  for (int i = 1; i < shp.num_dims(); ++i) {
    stream << " x " << shp[i];
  }
  stream << " (local shape (with halo): " << lshp[0];
  for (int i = 1; i < lshp.num_dims(); ++i) {
    stream << " x " << lshp[i];
  }
  stream << "). Size: " << allocated / 1048576.0 << " MiB" << std::endl;
  return allocated;
}
#else
template <typename T>
size_t report_distconv_matrix(T const& m, std::ostream& stream)
{
  stream << "Distconv is disabled" << std::endl;
  return 0;
}
#endif

template <typename T>
size_t get_activation_and_error_signal_size(data_type_layer<T> const& dtl,
                                            std::ostream& reps)
{
  size_t allocated = 0;
  for (int i = 0; i < dtl.get_num_children(); ++i) {
    if (dtl.get_num_children() == 1)
      reps << "    Activations: ";
    else
      reps << "    Activations (" << i << "): ";

    if (dtl.distconv_enabled()) {
#ifdef LBANN_HAS_DISTCONV
      auto const& child = dtl.get_child_layer(i);
      auto const& dcact = dtl.get_distconv_adapter().get_activations(child);
      allocated += report_distconv_matrix(dcact, reps);
      // Add activations if child layer is not distconv-enabled
      if (!child.distconv_enabled()) {
        auto const& act = dtl.get_activations(i);
        reps << "      + non-distconv adapter: ";
        allocated += report_dist_matrix(act, reps);
      }
#endif
    }
    else {
      auto const& act = dtl.get_activations(i);
      allocated += report_dist_matrix(act, reps);
    }
  }
  return allocated;
}

struct ReportDistMatrix
{
  template <typename T>
  size_t operator()(std::ostream& os, El::AbstractDistMatrix<T> const& m)
  {
    return report_dist_matrix(m, os);
  }
  template <typename... Args>
  static size_t DeductionError(Args&&...)
  {
    LBANN_ERROR("Unknown matrix type.");
  }
  static size_t DispatchError(std::ostream&, El::BaseDistMatrix const&)
  {
    LBANN_ERROR("Failed to dispatch \"report_dist_matrix\"");
  }
};

// This makes two things easier. First, it allows us to call the
// int-indexed "get_activations" function, rather than having to go
// through the pointers to the child layers. Second, it allows us to
// amortize our dynamic_cast stack over multiple calls to
// "report_dist_matrix", which should (statically) dispatch directly
// here rather than going through the dynamically dispatched overload.
struct GetActivationAndErrorSignalSize
{
  /**
   *  @brief Print and get maximal activation/error signal size for a layer.
   */
  template <typename T>
  size_t operator()(std::ostream& reps, data_type_layer<T> const& dtl)
  {
    return get_activation_and_error_signal_size(dtl, reps);
  }
  template <typename... Args>
  static size_t DeductionError(Args&&...)
  {
    LBANN_ERROR("Unknown layer type.");
  }
  static size_t DispatchError(std::ostream&, Layer const& l)
  {
    LBANN_ERROR("Failed to dispatch for layer \"", l.get_name(), "\"");
  }
}; // struct GetActivationAndErrorSignalSize

size_t report_dist_matrix(El::BaseDistMatrix const& m, std::ostream& os)
{
  using Dispatcher =
    h2::multimethods::SwitchDispatcher<ReportDistMatrix,
                                       size_t,
                                       El::BaseDistMatrix const,
                                       DistMatrixTypes>;
  return Dispatcher::Exec(ReportDistMatrix{}, m, os);
}

size_t get_activation_and_error_signal_size(Layer const& x, std::ostream& os)
{
  using Dispatcher =
    h2::multimethods::SwitchDispatcher<GetActivationAndErrorSignalSize,
                                       size_t,
                                       Layer const,
                                       LayerTypes>;
  return Dispatcher::Exec(GetActivationAndErrorSignalSize{}, x, os);
}

} // namespace

/**
 * @brief Returns the currently used memory, or 0 if LBANN was not compiled with
 * GPU support.
 * TODO(later): Gather across all ranks?
 */
size_t get_used_gpu_memory()
{
#ifdef LBANN_HAS_GPU
  auto const [available, total] = h2::gpu::mem_info();
  // TODO(later): Might be nicer to return a struct with gathered information
  // (min, max, median across ranks)
  return total - available;
#else
  return 0;
#endif
}

/**
 * @brief Returns the total memory, or 0 if LBANN was not compiled with
 * GPU support.
 */
static inline size_t get_total_gpu_memory()
{
#ifdef LBANN_HAS_GPU
  return h2::gpu::mem_info().total;
#else
  return 0;
#endif
}

memory_profiler::memory_profiler(bool detailed_first_step)
  : callback_base(), m_detailed_first_step(detailed_first_step)
{
#ifndef LBANN_HAS_GPU
  LBANN_WARNING(
    "Without GPU support, the memory profiler callback does not ",
    "provide raw memory usage information, only expected allocation size.");
#endif
}

memory_profiler::~memory_profiler() {}

template <class Archive>
void memory_profiler::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)));
}

void memory_profiler::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_memory_profiler();
  msg->set_detailed_first_step(m_detailed_first_step);
}

void memory_profiler::on_setup_begin(model* m)
{
  size_t total_gpu_mem;
  m_initial_memory_usage = get_used_gpu_memory();

  // Simple memory test for other processes that use GPU memory
  total_gpu_mem = get_total_gpu_memory();
  if (total_gpu_mem > 0) {
    double memratio = static_cast<double>(m_initial_memory_usage) /
                      static_cast<double>(total_gpu_mem);
    if (memratio > 0.5) {
      LBANN_WARNING("GPU memory usage prior to LBANN allocation is ",
                    static_cast<int>(memratio * 100),
                    "%. LBANN may not operate properly.");
    }
  }

  // Initial memory printout
  auto comm = m->get_comm();
  bool should_print = comm->am_trainer_master();
  if (should_print) {
    size_t free = total_gpu_mem - m_initial_memory_usage;
    std::cout << "MEM: Initial available memory: " << free / 1048576.0 << " / "
              << total_gpu_mem / 1048576.0 << " MiB." << std::endl;
  }
}

namespace {
struct MemUsage
{
  std::string report;
  size_t total_mem;

  MemUsage(const std::string& r, size_t m) : report(r), total_mem(m) {}
  bool operator<(const MemUsage& other) const
  {
    return total_mem < other.total_mem;
  }
};
} // namespace

void memory_profiler::report_mem_usage(model* m)
{
  size_t total = 0, weight_mem = 0, opt_state_mem = 0, other_mem = 0;
  std::vector<MemUsage> usage;
  std::map<weights*, std::string> already_reported;

  // Traverse the graph layer by layer
  for (auto& layer : m->get_layers()) {
    std::ostringstream reps;
    size_t layer_total = 0, layer_total_acts = 0;

    reps << "  " << layer->get_name() << " (" << layer->get_type()
         << "):" << std::endl;

    // Get maximal activation/error signal size (suboptimal approximation)
    {
      size_t const allocated = m_act_sizes[layer];
      reps << m_act_report[layer];
      layer_total_acts += allocated;
      layer_total += allocated;
    }

    // Weight accounting
    if (layer->num_weights() > 0) {
      // Nicer printout for multiple weights
      if (layer->num_weights() > 1) {
        reps << "    Weights:" << std::endl;
      }

      for (size_t i = 0; i < layer->num_weights(); ++i) {
        weights* w = &layer->get_weights(i);

        if (layer->num_weights() > 1) {
          reps << "      " << w->get_name() << ": ";
        }
        else {
          reps << "    Weights (" << w->get_name() << "): ";
        }

        // Weights already reported in another layer
        if (already_reported.find(w) != already_reported.end()) {
          reps << "See " << already_reported[w] << std::endl;
          continue;
        }

        // Report weight tensor
        size_t allocated = report_dist_matrix(w->get_values_sharded(), reps);
        weight_mem += allocated;
        layer_total += allocated;
        already_reported[w] = layer->get_name();

        // Optimizer state accounting
        auto* opt = w->get_optimizer();
        if (opt != nullptr) {
          allocated = opt->get_state_size();
          if (allocated > 0) {
            if (layer->num_weights() > 1)
              reps << "  ";
            reps << "      Optimizer state: " << allocated / 1048576.0 << " MB"
                 << std::endl;
            opt_state_mem += allocated;
            layer_total += allocated;
          }
        }
      }
    }

    // TODO(later): Get declared workspace (intrusive)

    // Get the rest of the allocated memory during setup
    size_t unaccounted = m_unaccounted_setup_layer[layer];
    if (layer_total_acts > unaccounted) {
      reps << "    Excess activation memory: " << layer_total_acts << std::endl;
      layer_total_acts = 0;
    }
    else {
      unaccounted -= layer_total_acts;
    }
    // TODO: Subtract workspace from unaccounted so that the field is accurate
    // unaccounted -= workspace;

    // If still larger than zero, report
    if (unaccounted > 0) {
      reps << "    Other: " << unaccounted / 1048576.0 << " MiB" << std::endl;
      layer_total += unaccounted;
      other_mem += unaccounted;
    }

    if (layer_total > 0) {
      total += layer_total;

      // Only account for activations when sorting
      layer_total += layer_total_acts;

      reps << "    Total: " << layer_total / 1048576.0 << " MiB" << std::endl;
      reps << std::endl;
      usage.emplace_back(MemUsage(reps.str(), layer_total));
    }
  }

  // Add extraneous weights
  for (auto& weight : m->get_weights()) {
    if (already_reported.find(weight) == already_reported.end()) {
      size_t weight_total = 0;
      std::stringstream reps;
      reps << "  DETACHED weight " << weight->get_name() << ": ";

      // Report weight tensor
      size_t allocated = report_dist_matrix(weight->get_values_sharded(), reps);
      weight_mem += allocated;
      weight_total += allocated;
      already_reported[weight] = weight->get_name();
      reps << std::endl;

      // Optimizer state accounting
      auto* opt = weight->get_optimizer();
      if (opt != nullptr) {
        allocated = opt->get_state_size();
        if (allocated > 0) {
          reps << "    Optimizer state: " << allocated / 1048576.0 << " MB";
          opt_state_mem += allocated;
          weight_total += allocated;
        }
      }
      if (weight_total > 0) {
        reps << "    Total: " << weight_total / 1048576.0 << " MiB"
             << std::endl;
        reps << std::endl;
        usage.emplace_back(MemUsage(reps.str(), weight_total));
      }
      total += weight_total;
    }
  }

  // Collection is complete, report memory usage in descending order
  std::sort(usage.begin(), usage.end());
  for (auto riter = usage.rbegin(); riter != usage.rend(); ++riter) {
    std::cout << riter->report;
  }

  std::cout << "MEM: Total expected model memory: " << total / 1048576.0
            << " MiB (weights: " << weight_mem / 1048576.0
            << " MiB, optimizer state: " << opt_state_mem / 1048576.0
            << " MiB, other: " << other_mem / 1048576.0 << " MiB)."
            << std::endl;
}

void memory_profiler::on_setup_end(model* m)
{
  auto comm = m->get_comm();
  bool should_print = comm->am_trainer_master();

  // Print total used memory
  m_step0_usage = m_setup_end_usage = get_used_gpu_memory();
  if (m_setup_end_usage > m_initial_memory_usage && should_print) {
    std::cout << "MEM: Total actual memory usage after setup: "
              << (m_setup_end_usage - m_initial_memory_usage) / 1048576.0
              << " MiB." << std::endl;
  }
  m_current_step = 0;
}

void memory_profiler::first_step_accounting(model* m, const std::string& msg)
{
  // Collect raw used memory as necessary
  if (m_current_step == 0) {
    size_t current_usage = get_used_gpu_memory();
    if (current_usage > m_step0_usage) {
      auto comm = m->get_comm();
      bool should_print = comm->am_trainer_master();
      if (should_print) {
        std::cout << "MEM: Allocated memory " << msg << ": " << std::fixed
                  << (current_usage - m_step0_usage) / 1048576.0 << " MiB."
                  << std::endl;
        if (m_detailed_first_step) {
          std::cout << "Breakdown:" << std::endl;
          size_t remainder = current_usage - m_setup_end_usage;
          for (auto const& k : m_unaccounted_fp_layer) {
            if (k.second > 0) {
              std::cout << "  Layer " << k.first->get_name() << ": " << k.second
                        << " bytes (forward)" << std::endl;
              remainder -= k.second;
            }
          }
          for (auto const& k : m_unaccounted_bp_layer) {
            if (k.second > 0) {
              std::cout << "  Layer " << k.first->get_name() << ": " << k.second
                        << " bytes (backprop)" << std::endl;
              remainder -= k.second;
            }
          }
          if (remainder > 0) {
            std::cout << "  Unaccounted remainder: " << remainder << " bytes"
                      << std::endl;
          }
        }
      }
      m_step0_usage = current_usage;
    }
  }
}

void memory_profiler::on_forward_prop_begin(model* m)
{
  first_step_accounting(m, "between setup and first forward prop");
}

void memory_profiler::on_forward_prop_end(model* m)
{
  first_step_accounting(m, "in first forward prop");
}

void memory_profiler::on_backward_prop_begin(model* m)
{
  first_step_accounting(m, "between first forward and backprop");
}

void memory_profiler::on_backward_prop_end(model* m)
{
  first_step_accounting(m, "in first backprop");
}

void memory_profiler::on_optimize_begin(model* m)
{
  first_step_accounting(m, "between backprop and optimizer step");
}

void memory_profiler::on_optimize_end(model* m)
{
  first_step_accounting(m, "in first optimizer step");
}

void memory_profiler::on_batch_end(model* m)
{
  // Collect raw used memory as necessary
  switch (m_current_step) {
  case 0:
    first_step_accounting(m, "between first optimizer step and step end");
    break;
  case 1:
    m_step1_usage = get_used_gpu_memory();
    m_peak_mem_usage = 0;
    break;
  case 2:
    m_step2_usage = get_used_gpu_memory();
    break;
  default:
    break;
  }

  auto comm = m->get_comm();
  bool should_print = comm->am_trainer_master();

  // Print collected activation and weight size
  if (should_print && m_current_step == 0) {
    std::cout << "MEM: Memory usage by layer (in descending order):"
              << std::endl;
    report_mem_usage(m);
  }

  // Check for and print leak report
  if (should_print && m_current_step == 2) {
    double third_step = m_step2_usage > m_step1_usage
                          ? (m_step2_usage - m_step1_usage) / 1048576.0
                          : 0.0;
    if (m_step2_usage > m_step1_usage && should_print) {
      LBANN_WARNING(
        "MEM: Potential memory leak discovered (step 3 consumes more "
        "memory than step 2). Difference: ",
        third_step,
        " MiB.");
      size_t remainder = m_step2_usage - m_step1_usage;
      for (auto const& k : m_unaccounted_fp_layer) {
        if (k.second > 0) {
          LBANN_WARNING("  Layer ",
                        k.first->get_name(),
                        ": ",
                        k.second,
                        " bytes (forward)");
          remainder -= k.second;
        }
      }
      for (auto const& k : m_unaccounted_bp_layer) {
        if (k.second > 0) {
          LBANN_WARNING("  Layer ",
                        k.first->get_name(),
                        ": ",
                        k.second,
                        " bytes (backprop)");
          remainder -= k.second;
        }
      }
      if (remainder > 0) {
        LBANN_WARNING("  Unaccounted remainder: ", remainder, " bytes");
      }
    }

    if (should_print && m_peak_mem_usage > 0) {
      std::cout << "MEM: Peak memory usage: "
                << (m_peak_mem_usage - m_initial_memory_usage) / 1048576.0
                << " MiB." << std::endl;

#ifdef HYDROGEN_HAVE_CUB
      // Print memory pool report, if exists
      auto& pool = El::cub::MemoryPool();
      pool.Report(std::cout);
#endif // HYDROGEN_HAVE_CUB
    }
  }

  // Increment step counter
  if (m_current_step < 4) {
    ++m_current_step;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Per-layer memory accounting

void memory_profiler::on_setup_begin(model* m, Layer* l)
{
  // Layer setup accounting
  m_unaccounted_setup_layer[l] = get_used_gpu_memory();
}

void memory_profiler::on_setup_end(model* m, Layer* l)
{
  // Count total bytes allocated by this layer's setup. Accounting is done
  // on model setup end.
  LBANN_ASSERT(m_unaccounted_setup_layer.find(l) !=
               m_unaccounted_setup_layer.end());

  size_t current_mem = get_used_gpu_memory();
  if (current_mem > m_unaccounted_setup_layer[l]) {
    m_unaccounted_setup_layer[l] = current_mem - m_unaccounted_setup_layer[l];
  }
  else {
    m_unaccounted_setup_layer[l] = 0;
  }
}

void memory_profiler::collect_peak_usage()
{
  if (m_current_step == 2) { // Collect peak memory usage in 3rd step
    size_t current_usage = get_used_gpu_memory();
    if (current_usage > m_peak_mem_usage) {
      m_peak_mem_usage = current_usage;
    }
  }
}

void memory_profiler::on_forward_prop_begin(model* m, Layer* l)
{
  if (m_current_step >= 0 && m_current_step <= 2) {
    m_unaccounted_fp_layer[l] = get_used_gpu_memory();
    collect_peak_usage();
  }
}
void memory_profiler::on_forward_prop_end(model* m, Layer* l)
{
  if (m_current_step >= 0 && m_current_step <= 2) {
    LBANN_ASSERT(m_unaccounted_fp_layer.find(l) !=
                 m_unaccounted_fp_layer.end());

    size_t current_mem = get_used_gpu_memory();
    if (current_mem > m_unaccounted_fp_layer[l]) {
      m_unaccounted_fp_layer[l] = current_mem - m_unaccounted_fp_layer[l];
    }
    else {
      m_unaccounted_fp_layer[l] = 0;
    }
    collect_peak_usage();

    std::ostringstream ss;
    m_act_sizes[l] = get_activation_and_error_signal_size(*l, ss);
    m_act_report[l] = ss.str();
  }
}
void memory_profiler::on_backward_prop_begin(model* m, Layer* l)
{
  if (m_current_step >= 0 && m_current_step <= 2) {
    m_unaccounted_bp_layer[l] = get_used_gpu_memory();
    collect_peak_usage();
  }
}
void memory_profiler::on_backward_prop_end(model* m, Layer* l)
{
  if (m_current_step >= 0 && m_current_step <= 2) {
    LBANN_ASSERT(m_unaccounted_bp_layer.find(l) !=
                 m_unaccounted_bp_layer.end());

    size_t current_mem = get_used_gpu_memory();
    if (current_mem > m_unaccounted_bp_layer[l]) {
      m_unaccounted_bp_layer[l] = current_mem - m_unaccounted_bp_layer[l];
    }
    else {
      m_unaccounted_bp_layer[l] = 0;
    }
    collect_peak_usage();
  }
}

std::unique_ptr<callback_base> build_memory_profiler_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>&)
{
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackMemoryProfiler&>(
      proto_msg);
  return std::make_unique<memory_profiler>(params.detailed_first_step());
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::memory_profiler
#define LBANN_CLASS_LIBNAME callback_memory_profiler
#include <lbann/macros/register_class_with_cereal.hpp>
