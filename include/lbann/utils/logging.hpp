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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LOGGING_HPP_INCLUDED
#define LBANN_LOGGING_HPP_INCLUDED

#include <h2/utils/Logger.hpp>

#include <iostream>
#include <cstdlib>
#include <vector>

namespace lbann {
namespace logging {

// Better than using raw strings
enum LBANN_Logger_ID
{
  LOG_RT,
  LOG_IO,
  LOG_TRAIN,
};

void setup_loggers();

// Raw string may be useful for debugging
char const* logger_id_str(LBANN_Logger_ID id);

// Access the actual logger object
h2::Logger& get(LBANN_Logger_ID id);

}// namespace logging
}// namespace lbann

// #defines can go here. Make sure they can go anywhere:
#define LBANN_LOG(logger_id, level, ...) \
  do { \
    auto& lbann_log_logger = ::lbann::logging::get(logger_id); \
    if (lbann_log_logger.should_log(level)) { \
      lbann_log_logger.get().log(::spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, ::h2::to_spdlog_level(level), __VA_ARGS__); \
    } \
  } while (0)

#define LBANN_TRACE(logger_id, ...) LBANN_LOG(logger_id, ::h2::Logger::LogLevelType::TRACE, __VA_ARGS__)
#define LBANN_DEBUG(logger_id, ...) LBANN_LOG(logger_id, ::h2::Logger::LogLevelType::DEBUG, __VA_ARGS__)
#define LBANN_INFO(logger_id, ...) LBANN_LOG(logger_id, ::h2::Logger::LogLevelType::INFO, __VA_ARGS__)
#define LBANN_WARN(logger_id, ...) LBANN_LOG(logger_id, ::h2::Logger::LogLevelType::WARN, __VA_ARGS__)
#define LBANN_ERR(logger_id, ...) LBANN_LOG(logger_id, ::h2::Logger::LogLevelType::ERR, __VA_ARGS__)
#define LBANN_CRIT(logger_id, ...) LBANN_LOG(logger_id, ::h2::Logger::LogLevelType::CRIT, __VA_ARGS__)

#define LBANN_RT_TRACE(...) LBANN_TRACE(::lbann::logging::LBANN_Logger_ID::LOG_RT, __VA_ARGS__)


/*
#define LBANN_RT_DEBUG(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::DEBUG))                  \
    logger.get().debug(__VA_ARGS__);                                       \

#define LBANN_RT_INFO(...)                                                 \
  if (logger.should_log(h2::Logger::LogLevelType::INFO))                   \
    logger.get().info(__VA_ARGS__);                                        \

#define LBANN_RT_WARN(...)                                                 \
  if (logger.should_log(h2::Logger::LogLevelType::WARN))                   \
    logger.get().warn(__VA_ARGS__);                                        \

#define LBANN_RT_ERROR(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::ERROR))                  \
    logger.get().error(__VA_ARGS__);                                       \

#define LBANN_RT_CRITICAL(...)                                             \
  if (logger.should_log(h2::Logger::LogLevelType::CRITICAL))               \
    logger.get().critical(__VA_ARGS__);                                    \

#define LBANN_IO_TRACE(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::TRACE))                  \
    logger.get().trace(__VA_ARGS__);                                       \

#define LBANN_IO_DEBUG(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::DEBUG))                  \
    logger.get().debug(__VA_ARGS__);                                       \

#define LBANN_IO_INFO(...)                                                 \
  if (logger.should_log(h2::Logger::LogLevelType::INFO))                   \
    logger.get().info(__VA_ARGS__);                                        \

#define LBANN_IO_WARN(...)                                                 \
  if (logger.should_log(h2::Logger::LogLevelType::WARN))                   \
    logger.get().warn(__VA_ARGS__);                                        \

#define LBANN_IO_ERROR(...)                                                \
  if (logger.should_log(h2::Logger::LogLevelType::ERROR))                  \
    logger.get().error(__VA_ARGS__);                                       \

#define LBANN_IO_CRITICAL(...)                                             \
  if (logger.should_log(h2::Logger::LogLevelType::CRITICAL))               \
    logger.get().critical(__VA_ARGS__);                                    \

#define LBANN_TRAINING_TRACE(...)                                          \
  if (logger.should_log(h2::Logger::LogLevelType::TRACE))                  \
    logger.get().trace(__VA_ARGS__);                                       \

#define LBANN_TRAINING_DEBUG(...)                                          \
  if (logger.should_log(h2::Logger::LogLevelType::DEBUG))                  \
    logger.get().debug(__VA_ARGS__);                                       \

#define LBANN_TRAINING_INFO(...)                                           \
  if (logger.should_log(h2::Logger::LogLevelType::INFO))                   \
    logger.get().info(__VA_ARGS__);                                        \

#define LBANN_TRAINING_WARN(...)                                           \
  if (logger.should_log(h2::Logger::LogLevelType::WARN))                   \
    logger.get().warn(__VA_ARGS__);                                        \

#define LBANN_TRAINING_ERROR(...)                                          \
  if (logger.should_log(h2::Logger::LogLevelType::ERROR))                  \
    logger.get().error(__VA_ARGS__);                                       \

#define LBANN_TRAINING_CRITICAL(...)                                       \
  if (logger.should_log(h2::Logger::LogLevelType::CRITICAL))               \
    logger.get().critical(__VA_ARGS__);                                    \
*/
#endif // LBANN_LOGGING_HPP_INCLUDED
