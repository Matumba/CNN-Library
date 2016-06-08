// Copyright 2016 by Glukhov V. O. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include "base_layer.hpp"
#include "util.hpp"

namespace cnn
{
	namespace nn
	{
		class FullyConnectedLayer : public BaseLayer
		{
		public:
			FullyConnectedLayer(arma::uword in, arma::uword out,
			                    std::unique_ptr<BaseActivationFunction> activFunc);

			void Forward(std::shared_ptr<arma::Cube<float>> input) override;
			std::pair<tensor4d, tensor4d> Backward(
				const std::shared_ptr<arma::Cube<float>>& prevLocalLoss) override;
			std::pair<tensor4d, tensor4d> Backward2nd(
				const std::shared_ptr<arma::Cube<float>>& prevLocalLoss) override;
		};

		inline
		FullyConnectedLayer::FullyConnectedLayer(arma::uword in, arma::uword out,
		                                         std::unique_ptr<BaseActivationFunction> activFunc)
			: BaseLayer(in, out, 1, 1, std::move(activFunc))
		{
			biasWeights_ = tensor4d(out, 1, 1, 1);
		}
	}
}
