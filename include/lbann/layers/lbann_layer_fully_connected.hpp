////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
// lbann_layer_fully_connected .hpp .cpp - Dense, fully connected, layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
#define LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED

#include "lbann/layers/lbann_layer.hpp"
#include "lbann/layers/lbann_layer_activations.hpp"
#include <string>



namespace lbann
{
    // FullyConnectedLayer : dense layer class
    class FullyConnectedLayer : public Layer
    {
    public:
      FullyConnectedLayer(uint index,
                          int numPrevNeurons,
                          uint numNeurons,
                          uint miniBatchSize,
                          activation_type activationType,
                          weight_initialization init,
                          lbann_comm* comm,
                          Optimizer *optimizer,
                          std::vector<regularizer*> regs={});
      ~FullyConnectedLayer();
      void setup(int numPrevNeurons);
      DistMat& get_weights_biases() { return WB_view; }
      DistMat& get_weights_biases_gradient() { return WB_D_view; }
      DistMat& get_activations() { return Acts_view; }
      bool update();
      DataType checkGradient(Layer& PrevLayer, const DataType Epsilon=1e-4);
      DataType computeCost(DistMat &deltas);
      DataType WBL2norm();

        // bool saveToFile(std::string FileDir);
        // bool loadFromFile(std::string FileDir);

    private:

      const weight_initialization m_weight_initialization;

      /// Views of the weight matrix that allow you to separate activation weights from bias weights
      DistMat m_activation_weights_v;
      DistMat m_bias_weights_v;
      DistMat m_activation_weights_gradient_v;
      DistMat m_bias_weights_gradient_v;
      DistMat m_bias_bp_t;
      DataType m_bias_term;

      /** View of the WB matrix, except for the bottom row. */
      DistMat WB_view;
      /** View of the WB_D matrix, except for the bottom row. */
      DistMat WB_D_view;
      /** View of the Acts matrix, except for the bottom row. */
      DistMat Acts_view;

    public:
      //Probability of dropping neuron/input used in dropout_layer
      //Range 0 to 1; default is -1 => no dropout
      DataType  WBL2NormSum;

    protected:
      void fp_linearity();
      void bp_linearity();
    };

}


#endif // LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
