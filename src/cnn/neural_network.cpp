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
#include "neural_network.hpp"

namespace cnn
{
	namespace nn
	{

		void NeuralNetwork::Forward()
		{
#ifndef NDEBUG
			assert(!in_->is_empty());
			assert(!layers_.empty());
#endif
			layers_[0]->Forward(in_->Output());
			std::size_t amount = layers_.size();
			for (std::size_t i = 1; i < amount; ++i) {
				layers_[i]->Forward(layers_[i - 1]->Output());
			}
		}

		std::vector<std::pair<tensor4d, tensor4d>> NeuralNetwork::Backpropagation()
		{
			
			std::shared_ptr<arma::Cube<double>> hypothesis = layers_.back()->Output();
			const arma::Col<double> &labels = in_->Labels();
			std::shared_ptr<arma::Cube<double>> loss = std::make_shared<arma::Cube<double>>(
				labels.n_rows, 1, 1);
			for (arma::uword i = 0; i < labels.n_rows; ++i) {
				loss->slice(0)(i, 0) = costFunc_->Derivative(labels(i), hypothesis->slice(0)(i, 0));
			}

			arma::uword size = layers_.size();
			std::vector<std::pair<tensor4d, tensor4d>> result(size);
			result[size - 1] = layers_[size - 1]->Backward(loss);
			for (arma::sword i = size - 1; i > 0; --i) {
				loss = layers_[i]->LocalLoss();
				result[i - 1] = layers_[i - 1]->Backward(loss);
			}
			return result;
		}

		
		std::vector<std::pair<tensor4d, tensor4d>> NeuralNetwork::Backpropagation_2nd()
		{
			//TODO:
			std::shared_ptr<arma::Cube<double>> loss = std::make_shared<arma::Cube<double>>(
				in_->Labels().n_rows, 1, 1);
			std::shared_ptr<arma::Cube<double>> hypothesis = layers_.back()->Output();
			const arma::Col<double> &labels = in_->Labels();
			for (arma::uword i = 0; i < labels.n_rows; ++i) {
				loss->slice(0)(i, 0) = costFunc_->SecondDerivative(labels(i),
																   hypothesis->slice(0)(i, 0));
			}
			arma::uword size = layers_.size();
			std::vector<std::pair<tensor4d, tensor4d>> result(size);
			result[size - 1] = layers_[size - 1]->Backward2nd(loss);
			for (arma::sword i = size - 1; i > 0; --i) {
				loss = layers_[i]->LocalLoss();
				result[i - 1] = layers_[i - 1]->Backward2nd(loss);
			}
			return result;
		}
	}
}