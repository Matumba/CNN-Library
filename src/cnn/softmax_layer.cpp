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
#include "softmax_layer.hpp"

namespace cnn
{
	namespace nn
	{
		void SoftMaxLayer::Forward(std::shared_ptr<arma::Cube<double>> input)
		{
#ifndef NDEBUG
			assert(initialized_);
			assert(input->n_elem == weights_.n_rows
				   && "the input signal is not equal to the expected size");
#endif
			// check is previous layer was fully-connected
			if (input->n_slices == 1 && input->n_cols == 1) {
				input_ = input;
			} else {
				input_ = std::make_shared<arma::Cube<double>>(vectorise(input->get_ref()));
			}


			if (!receptiveField_) {
				receptiveField_ = std::make_shared<arma::Cube<double>>(weights_.n_cols, 1, 1);
			} else {
				receptiveField_->set_size(weights_.n_cols, 1, 1);
			}

			receptiveField_->slice(0).col(0) = weights_.data[0].slice(0).t()
				* input_->slice(0).col(0);
			receptiveField_->slice(0).col(0) += biasWeights_.data[0].slice(0).col(0);

			if (!output_) {
				output_ = std::make_shared<arma::Cube<double>>(weights_.n_cols, 1, 1);
			}
			ComputeOutput();
		}

		std::pair<tensor4d, tensor4d> SoftMaxLayer::Backward(
			const std::shared_ptr<arma::Cube<double>>& prevLocalLoss)
		{
			using namespace arma;
#ifndef NDEBUG
			assert(prevLocalLoss);
			assert(prevLocalLoss->n_slices == 1 && prevLocalLoss->n_cols == 1);
#endif

			arma::uword input_height = input_->n_rows;
			//propogate current delta to previous layer:
			if (!localLoss_) {
				localLoss_ = std::make_shared<arma::Cube<double>>(input_height, 1, 1);
			} else {
				localLoss_->set_size(input_height, 1, 1);
			}
			localLoss_->slice(0) = weights_.data[0].slice(0) * prevLocalLoss->slice(0);

			arma::uword output_height = receptiveField_->n_rows;
			//compute gradients
			std::pair<tensor4d, tensor4d> result = std::make_pair(
				tensor4d(input_height, output_height, 1, 1),
				tensor4d(output_height, 1, 1, 1));
			result.first.data[0].slice(0) = input_->slice(0).col(0)
				* prevLocalLoss->slice(0).col(0).t();
			result.second.data[0].slice(0).col(0) = prevLocalLoss->slice(0).col(0);

			return result;
		}

		std::pair<tensor4d, tensor4d> SoftMaxLayer::Backward2nd(
			const std::shared_ptr<arma::Cube<double>>& prevLocalLoss)
		{
			using namespace arma;
#ifndef NDEBUG
			assert(prevLocalLoss);
			assert(prevLocalLoss->n_slices == 1 && prevLocalLoss->n_cols == 1);
#endif

			arma::uword input_height = input_->n_rows;
			//propogate current delta to previous layer:
			if (!localLoss_) {
				localLoss_ = std::make_shared<arma::Cube<double>>(input_height, 1, 1);
			} else {
				localLoss_->set_size(input_height, 1, 1);
			}
			localLoss_->slice(0) = arma::square(weights_.data[0].slice(0))
				* prevLocalLoss->slice(0);

			arma::uword output_height = receptiveField_->n_rows;
			//compute gradients
			std::pair<tensor4d, tensor4d> result = std::make_pair(
				tensor4d(input_height, output_height, 1, 1),
				tensor4d(output_height, 1, 1, 1));
			result.first.data[0].slice(0) = arma::square(input_->slice(0).col(0))
				* prevLocalLoss->slice(0).col(0).t();
			result.second.data[0].slice(0).col(0) = prevLocalLoss->slice(0).col(0);

			return result;
		}
	}
}