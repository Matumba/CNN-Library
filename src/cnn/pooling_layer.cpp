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
#include "pooling_layer.hpp"
#include <armadillo>
#include <cmath>

namespace cnn
{
	namespace nn
	{

		void BasePoolingLayer::Forward(std::shared_ptr<arma::Cube<float>> input)
		{
			using namespace arma;

			input_ = input;
			uword output_height = (input_->n_rows - kernel_size_.height);
			uword output_width = (input_->n_cols - kernel_size_.width);

#ifndef NDEBUG
			assert(output_height % stride_ == 0);
			assert(output_width % stride_ == 0);
			//TODO: for release build in case of working in another thread
			//TODO: we should use std::exception_ptr for thread-safety
#endif
			output_height = output_height / stride_ + 1;
			output_width = output_width / stride_ + 1;

			if (!receptiveField_) {
				receptiveField_ = std::make_shared<Cube<float>>(output_height, output_width,
																input_->n_slices, fill::zeros);
			} // convolution and pooling layer don't have fixed data for input signals
			  // so we need always resize our receptive field 
			else {
				receptiveField_->set_size(output_height, output_width, input_->n_slices);
				receptiveField_->fill(0.0);
			}

			SubSample(output_height, output_width);

			// currently common to use the activation function after convolution layer
			// instead of a subsample layer
			if (!activFunc_) {
				output_ = receptiveField_;
			} else {
				if (!output_) {
					output_ = std::make_shared<Cube<float>>(output_height, output_width,
															input_->n_slices);
				} else {
					output_->set_size(output_height, output_width, input_->n_slices);
				}
				activFunc_->Compute(receptiveField_, output_);
			}
		}

		std::pair<tensor4d, tensor4d> MaxPoolingLayer::Backward(
			const std::shared_ptr<arma::Cube<float>>& prevLocalLoss)
		{
			using namespace arma;
#ifndef NDEBUG
			assert(prevLocalLoss);
#endif
			// top layer was 1d. we need reshape error to 3d
			if (prevLocalLoss->n_slices == 1 && prevLocalLoss->n_cols == 1) {
				(*prevLocalLoss) = unvectorise(prevLocalLoss->get_ref(), output_->n_rows,
											   output_->n_cols, output_->n_slices);
			}

			if (!localLoss_) {
				localLoss_ = std::make_shared<Cube<float>>(
					input_->n_rows, input_->n_cols, input_->n_slices, fill::zeros);
			} else {
				localLoss_->set_size(input_->n_rows, input_->n_cols, input_->n_slices);
				localLoss_->fill(0.0);
			}

			for (uword channel = 0; channel < connectIndexes_.n_slices; ++channel) {
				uword lossCol = 0;
				uword lossRow = 0;
				for (uword column = 0; column < connectIndexes_.n_cols; column += stride_) {
					for (uword row = 0; row < connectIndexes_.n_rows; row += stride_) {
						// from top to bottom propagates only connected losses
						if (connectIndexes_(row, column, channel)) {
							(*localLoss_)(row, column, channel) = (*prevLocalLoss)(
								lossRow, lossCol, channel);
							++lossRow;
						}
					}
					++lossCol;
				}
			}

			if (activFunc_) {
				localLoss_->transform([&] (float value)
				{
					if (!value) {
						return 0.0f;
					} else
						return activFunc_->Derivative(value);
				});
			}

			// pool layer doesn't has weights
			return std::make_pair(tensor4d(), tensor4d());
		}

		std::pair<tensor4d, tensor4d> MaxPoolingLayer::Backward2nd(
			const std::shared_ptr<arma::Cube<float>>& prevLocalLoss)
		{
			using namespace arma;
#ifndef NDEBUG
			assert(prevLocalLoss);
#endif
			// top layer was 1d. we need reshape error to 3d
			if (prevLocalLoss->n_slices == 1 && prevLocalLoss->n_cols == 1) {
				(*prevLocalLoss) = unvectorise(prevLocalLoss->get_ref(), output_->n_rows,
											   output_->n_cols, output_->n_slices);
			}

			if (!localLoss_) {
				localLoss_ = std::make_shared<Cube<float>>(
					input_->n_rows, input_->n_cols, input_->n_slices, fill::zeros);
			} else {
				localLoss_->set_size(input_->n_rows, input_->n_cols, input_->n_slices);
				localLoss_->fill(0.0);
			}

			for (uword channel = 0; channel < connectIndexes_.n_slices; ++channel) {
				uword lossCol = 0;
				uword lossRow = 0;
				for (uword column = 0; column < connectIndexes_.n_cols; column += stride_) {
					for (uword row = 0; row < connectIndexes_.n_rows; row += stride_) {
						// from top to bottom propagates only connected losses
						if (connectIndexes_(row, column, channel)) {
							(*localLoss_)(row, column, channel) = (*prevLocalLoss)(
								lossRow, lossCol, channel);
							++lossRow;
						}
					}
					++lossCol;
				}
			}

			if (activFunc_) {
				localLoss_->transform([&](float value)
				{
					if (!value) {
						return 0.0f;
					} else
						return std::pow(activFunc_->Derivative(value), 2);
				});
			}

			// pool layer doesn't has weights
			return std::make_pair(tensor4d(), tensor4d());
		}


		void MaxPoolingLayer::SubSample(arma::uword output_height,
										arma::uword output_width) noexcept
		{
			using namespace arma;
			connectIndexes_.set_size(output_height, output_width, input_->n_slices);
			connectIndexes_.fill(0);

			float maxVal;
			uword rowIdx, colIdx;
			for (uword channel = 0; channel < input_->n_slices; ++channel) {
				// arma store data in column-major order
				for (uword column = 0; column < input_->n_cols; column += stride_) {
					for (uword row = 0; row < input_->n_rows; row += stride_) {
						maxVal = (*input_).slice(channel)(span(row, row + stride_ - 1),
														  span(column, column + stride_ - 1)
														  ).max(rowIdx, colIdx);
						(*receptiveField_).slice(channel)(rowIdx, colIdx) = maxVal;
						connectIndexes_.slice(channel)(rowIdx, colIdx) = 1;
					}
				}
			}
		}


	}
}