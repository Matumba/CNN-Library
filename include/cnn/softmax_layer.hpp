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
		class SoftMaxLayer : public BaseLayer
		{
		public:
			SoftMaxLayer(arma::uword in, arma::uword out);
			void Forward(std::shared_ptr<arma::Cube<double>> input) override;
			std::pair<tensor4d, tensor4d> Backward(
				const std::shared_ptr<arma::Cube<double>>& prevLocalLoss) override;
			std::pair<tensor4d, tensor4d> Backward2nd(
				const std::shared_ptr<arma::Cube<double>>& prevLocalLoss) override;
		private:
			void ComputeOutput() const;
		};

		inline SoftMaxLayer::SoftMaxLayer(arma::uword in, arma::uword out)
			: BaseLayer(in, out, 1, 1, nullptr)
		{
			biasWeights_ = tensor4d(out, 1, 1, 1);
		}

		inline void SoftMaxLayer::ComputeOutput() const
		{
			double maxVal = receptiveField_->slice(0).col(0).max();
			double denominator = arma::sum<arma::Col<double>>(
				arma::exp(receptiveField_->slice(0).col(0) - maxVal));		
			
			double numerator;
			for (arma::uword r = 0; r < receptiveField_->n_rows; ++r) {
				numerator = std::exp((*receptiveField_)(r, 0, 0) - maxVal);
				(*output_)(r, 0, 0) = numerator / denominator;
			}
		}
	}
}