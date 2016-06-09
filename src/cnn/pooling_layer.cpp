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
		void MaxPoolingLayer::Forward(std::shared_ptr<arma::Cube<float>> input)
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
			} else {
				receptiveField_->set_size(output_height, output_width, input_->n_slices);
				receptiveField_->fill(0.0);
			}

			connectIndexes_.set_size(input_->n_rows, input_->n_cols, input_->n_slices);
			connectIndexes_.fill(0.0);
			double maxVal;
			uword rowIdx, colIdx;

			for (uword d = 0; d < input_->n_slices; ++d) {
				uword out_col = 0;
				for (uword c = 0; c < input_->n_cols; c += stride_) {
					uword out_row = 0;
					for (uword r = 0; r < input_->n_rows; r += stride_) {
						maxVal = input_->slice(d)(span(r, r + kernel_size_.height - 1),
						                          span(c, c + kernel_size_.height - 1)).max(rowIdx, colIdx);
						(*receptiveField_)(out_row, out_col, d) = maxVal;
						connectIndexes_(r + rowIdx, c + colIdx, d) = 1;
						++out_row;
					}
					++out_col;
				}
			}

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
				localLoss_->fill(0.0f);
			}
			uword rowIdx, colIdx;
			for (uword d = 0; d < input_->n_slices; ++d) {
				uword lossCol = 0;
				for (uword c = 0; c < input_->n_cols; c += stride_) {
					uword lossRow = 0;
					for (uword r = 0; r < input_->n_rows; r += stride_) {
						// from top to bottom propagates only connected losses
						connectIndexes_.slice(d)(span(r, r + kernel_size_.height - 1),
						                         span(c, c + kernel_size_.height - 1)
						               ).max(rowIdx, colIdx);
						(*localLoss_)(r + rowIdx, c + colIdx, d) = (*prevLocalLoss)(
							lossRow, lossCol, d);
						++lossRow;
					}
					++lossCol;
				}
			}

			if (activFunc_) {
				localLoss_->transform([&] (float value) {
					if (!value) {
						return value;
					} else
						return activFunc_->Derivative(value);
				});
			}

			// pool layer doesn't has weights
			return
					std::make_pair(tensor4d(), tensor4d());
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
				localLoss_->fill(0.0f);
			}
			uword rowIdx, colIdx;
			for (uword d = 0; d < input_->n_slices; ++d) {
				uword lossCol = 0;
				for (uword c = 0; c < input_->n_cols; c += stride_) {
					uword lossRow = 0;
					for (uword r = 0; r < input_->n_rows; r += stride_) {
						// from top to bottom propagates only connected losses
						connectIndexes_.slice(d)(span(r, r + kernel_size_.height - 1),
												 span(c, c + kernel_size_.height - 1)
												 ).max(rowIdx, colIdx);
						(*localLoss_)(r + rowIdx, c + colIdx, d) = (*prevLocalLoss)(
							lossRow, lossCol, d);
						++lossRow;
					}
					++lossCol;
				}
			}

			if (activFunc_) {
				localLoss_->transform([&](float value) {
					if (!value) {
						return value;
					} else
						return std::pow(activFunc_->Derivative(value), 2);
				});
			}

			// pool layer doesn't has weights
			return std::make_pair(tensor4d(), tensor4d());
		}
	}
}
