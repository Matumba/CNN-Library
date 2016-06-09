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
#include "convolutional_layer.hpp"
#include <cmath>

namespace cnn
{
	namespace nn
	{
		void ConvolutionalLayer::im2col(const std::shared_ptr<arma::Cube<float>>& src_data,
		                                const tensor4d& src_kernel, arma::Mat<float>& dst_data,
		                                arma::Mat<float>& dst_kernel,
		                                arma::uword height, arma::uword width) const noexcept
		{
			arma::uword kernel_size = src_kernel.n_rows * src_kernel.n_cols;
			dst_data.set_size(src_kernel.n_rows * src_kernel.n_cols * src_data->n_slices,
			                  height * width);
			dst_kernel.set_size(src_kernel.n_size, src_kernel.n_rows
			                    * src_kernel.n_cols * src_data->n_slices);

			for (arma::uword c = 0; c < src_data->n_slices; ++c) {
				for (arma::uword col = 0; col < width; ++col) {
					for (arma::uword row = 0; row < height; ++row) {
						dst_data(arma::span(c * kernel_size, c * kernel_size
						                    + kernel_size - 1),
						         col * height + row
						) = arma::vectorise(src_data->slice(c)(
							arma::span(row * stride_, row * stride_ + src_kernel.n_rows - 1),
							arma::span(col * stride_, col * stride_ + src_kernel.n_cols - 1)));
					}
				}

				for (arma::uword k = 0; k < src_kernel.n_size; ++k) {
					dst_kernel(k, arma::span(c * kernel_size, c * kernel_size
					                         + kernel_size - 1)
					) = arma::vectorise(src_kernel.data[k].slice(c)).t();
				}
			}
		}

		void ConvolutionalLayer::im2col(const std::shared_ptr<arma::Cube<float>>& src_data,
		                                const std::shared_ptr<arma::Cube<float>>& src_kernel,
		                                arma::Mat<float>& dst_data,
		                                arma::Mat<float>& dst_kernel,
		                                arma::uword height, arma::uword width) const noexcept
		{
			/*using namespace arma;
			uword kernel_size = src_kernel->n_rows * src_kernel->n_cols;
			uword kernel_depth = src_kernel->n_slices;
			dst_data.set_size(kernel_size, height * width * src_data->n_slices);
			dst_kernel.set_size(kernel_depth, kernel_size);

			for (uword c = 0; c < src_data->n_slices; ++c) {
				for (uword col = 0; col < width; ++col) {
					for (uword row = 0; row < height; ++row) {
						dst_data(span::all, (c * height * width) + (col * height + row)
						) = arma::vectorise(src_data->slice(c)(
							span(row * stride_, row * stride_ + src_kernel->n_rows - 1),
							span(col * stride_, col * stride_ + src_kernel->n_cols - 1)));
					}
				}
			}
			for (uword k = 0; k < kernel_depth; ++k) {
				dst_kernel(k, span::all) = arma::vectorise(src_kernel->slice(k)).t();
			}*/
		}

		void ConvolutionalLayer::Forward(std::shared_ptr<arma::Cube<float>> input)
		{
			using namespace arma;
#ifndef NDEBUG
			assert(initialized_);
			assert((input->n_rows - kernel_size_.height + 2 * padding_.height) % stride_ == 0);
			assert((input->n_cols - kernel_size_.width + 2 * padding_.width) % stride_ == 0);
			assert(input->n_slices == weights_.n_slices);
#endif

			if (padding_.height == 0 && padding_.width == 0) {
				input_ = input;
			} else {
				AddPadding(input_, input->n_rows, input->n_cols, input->n_slices);
			}
			(*input_)(span(padding_.height, padding_.height + input->n_rows - 1),
			          span(padding_.width, padding_.width + input->n_cols - 1),
			          span::all) = *input;

			Mat<float> input2col;
			Mat<float> kernel2col;

			uword output_height = (input_->n_rows - kernel_size_.height) / stride_ + 1;
			uword output_width = (input_->n_cols - kernel_size_.width) / stride_ + 1;

			im2col(input_, weights_, input2col, kernel2col, output_height, output_width);

			Mat<float> cross_correlation = kernel2col * input2col;

			if (!receptiveField_) {
				receptiveField_ = std::make_shared<Cube<float>>(output_height, output_width,
				                                                 n_filters_);
			} // convolution and pooling layer don't have fixed data for input signals
			// so we need always resize our receptive field 
			else {
				receptiveField_->set_size(output_height, output_width, n_filters_);
			}

			float bias;
			for (uword k = 0; k < n_filters_; ++k) {
				bias = 0;
				for (uword c = 0; c < biasWeights_.n_slices; ++c) {
					bias += biasWeights_.data[k](0, 0, c);
				}
				receptiveField_->slice(k) = arma::reshape(cross_correlation.row(k),
				                                          output_height, output_width) + bias;
			}


			if (activFunc_) {
				if (!output_) {
					output_ = std::make_shared<Cube<float>>(output_height, output_width,
					                                         n_filters_);
				} else {
					output_->set_size(output_height, output_width, n_filters_);
				}
				activFunc_->Compute(receptiveField_, output_);
			} else {
				output_ = receptiveField_;
			}
		}


		std::pair<tensor4d, tensor4d> ConvolutionalLayer::Backward(
			const std::shared_ptr<arma::Cube<float>>& prevLocalLoss)
		{
			using namespace arma;
#ifndef NDEBUG
			assert(prevLocalLoss);
#endif
			// if top layer was 1d tensor we need reshape input error to 3d
			if (prevLocalLoss->n_slices == 1 && prevLocalLoss->n_cols == 1) {
				(*prevLocalLoss) = unvectorise(prevLocalLoss->get_ref(), output_->n_rows,
				                               output_->n_cols, output_->n_slices);
			}

			if (!localLoss_) {
				localLoss_ = std::make_shared<Cube<float>>(
					input_->n_rows, input_->n_cols, input_->n_slices);
			} else {
				localLoss_->set_size(input_->n_rows, input_->n_cols, input_->n_slices);
			}

			uword output_height = output_->n_rows;
			uword output_width = output_->n_cols;
			uword output_depth = output_->n_slices;
			Cube<float> dfdz(output_height, output_width,
			                  output_depth);
			for (uword c = 0; c < output_depth; ++c) {
				for (uword col = 0; col < output_width; ++col) {
					for (uword row = 0; row < output_height; ++row) {
						dfdz(row, col, c) = activFunc_->Derivative(
							(*receptiveField_)(row, col, c));
					}
				}
			}
			(*prevLocalLoss) %= dfdz;
			//compute gradient
			std::pair<tensor4d, tensor4d> result = std::make_pair(
				tensor4d(weights_.n_rows, weights_.n_cols,
				         weights_.n_slices, n_filters_),
				tensor4d(1, 1, biasWeights_.n_slices, n_filters_));
			// if on forward propagate was used padding for input then
			// input_ on backward stage has already been padded
			//Mat<float> input2col, kernel2col;
			//Mat<float> cross_correlation;
			//uword input_depth = input_->n_slices;
			//im2col(input_, prevLocalLoss, input2col, kernel2col,
			//       kernel_size_.height, kernel_size_.width);
			////output size = [prevLocalLoss->n_slices; n_filters * kernel_size_h * kernel_size_w] 
			//cross_correlation = kernel2col * input2col;
			//for (uword k = 0; k < n_filters_; ++k) {
			//	for (uword c = 0; c < weights_.n_slices; ++c)
			//		result.first.data[k].slice(c) = arma::reshape(
			//			cross_correlation(k, span(c * kernel_size_.height 
			//									  * kernel_size_.width,
			//									  c * kernel_size_.height 
			//									  * kernel_size_.width
			//									  + kernel_size_.height
			//									  * kernel_size_.width - 1)),
			//			weights_.n_rows, weights_.n_cols);
			//}
			////compute gradient for bias:
			//for (uword c = 0; c < output_depth; ++c) {
			//	float sum = arma::sum(arma::sum(prevLocalLoss->slice(c)));
			//	for (uword d = 0; d < biasWeights_.n_slices; ++d) {
			//		result.second.data[c](0, 0, d) = sum;
			//	}
			//}

			//// propagate error to bottom layer:

			//uword unpadded_input_height = input_->n_rows - 2 * padding_.height;
			//uword unpadded_input_width = input_->n_cols - 2 * padding_.width;
			//// we must add zeros on borders to input loss to get conv result dimension
			//// equal to input signals
			//uword pad_h = (unpadded_input_height -
			//		((prevLocalLoss->n_rows - kernel_size_.height) / stride_ + 1)) / 2;
			//uword pad_w = (unpadded_input_width -
			//		((prevLocalLoss->n_cols - kernel_size_.width) / stride_ + 1)) / 2;
			//std::shared_ptr<arma::Cube<float>> paddedPrevLoss = std::make_shared<
			//	arma::Cube<float>>(prevLocalLoss->n_rows + 2 * pad_h,
			//	                    prevLocalLoss->n_cols + 2 * pad_w,
			//	                    prevLocalLoss->n_slices, fill::zeros);

			//(*paddedPrevLoss)(span(pad_h, pad_h + output_height - 1),
			//                  span(pad_w, pad_w + output_width - 1), span::all
			//) = std::move((*prevLocalLoss));

			//// for propagate error to the previous layer we should use
			//// convolution instead cross-correlation
			//tensor4d flippedKernel(kernel_size_.height, kernel_size_.width,
			//                       n_filters_, input_depth);
			//for (uword n = 0; n < input_depth; ++n) {
			//	for (uword c = 0; c < n_filters_; ++c) {
			//		flippedKernel.data[n].slice(c) = flipud(fliplr(weights_.data[c].slice(n)));
			//	}
			//}

			//im2col(paddedPrevLoss, flippedKernel, input2col, kernel2col,
			//       unpadded_input_height, unpadded_input_width);
			//Mat<float> convolution = kernel2col * input2col;
			//if (!localLoss_) {
			//	localLoss_ = std::make_shared<Cube<float>>(unpadded_input_height,
			//	                                            unpadded_input_width,
			//	                                            input_depth);
			//} else {
			//	localLoss_->set_size(unpadded_input_height,
			//						 unpadded_input_width,
			//						 input_depth);
			//}
			//for (uword c = 0; c < input_depth; ++c) {
			//	(*localLoss_).slice(c) = arma::reshape(convolution.row(c),
			//	                                       unpadded_input_height,
			//	                                       unpadded_input_width);
			//}

			return result;
		}

		std::pair<tensor4d, tensor4d> ConvolutionalLayer::Backward2nd(
			const std::shared_ptr<arma::Cube<float>>& prevLocalLoss)
		{
			using namespace arma;
#ifndef NDEBUG
			assert(prevLocalLoss);
#endif
			// if top layer was 1d tensor we need reshape input error to 3d
			if (prevLocalLoss->n_slices == 1 && prevLocalLoss->n_cols == 1) {
				(*prevLocalLoss) = unvectorise(prevLocalLoss->get_ref(), output_->n_rows,
				                               output_->n_cols, output_->n_slices);
			}

			if (!localLoss_) {
				localLoss_ = std::make_shared<Cube<float>>(
					input_->n_rows, input_->n_cols, input_->n_slices);
			} else {
				localLoss_->set_size(input_->n_rows, input_->n_cols, input_->n_slices);
			}

			uword output_height = output_->n_rows;
			uword output_width = output_->n_cols;
			uword output_depth = output_->n_slices;
			Cube<float> dfdz(output_height, output_width,
			                  output_depth);
			for (uword c = 0; c < output_depth; ++c) {
				for (uword col = 0; col < output_width; ++col) {
					for (uword row = 0; row < output_height; ++row) {
						dfdz(row, col, c) = std::pow(activFunc_->Derivative(
							                             (*receptiveField_)(row, col, c)), 2);
					}
				}
			}
			(*prevLocalLoss) %= dfdz;

			//compute gradient
			std::pair<tensor4d, tensor4d> result = std::make_pair(
				tensor4d(weights_.n_rows, weights_.n_cols,
				         weights_.n_slices, n_filters_),
				tensor4d(1, 1, biasWeights_.n_slices, n_filters_));
//			// if on forward propagate was used padding for input then
//			// input_ on backward stage has already been padded
//			Mat<float> input2col, kernel2col;
//			Mat<float> cross_correlation;
//			uword input_depth = input_->n_slices;
//
//			// in 2nd order backpropagation we must square input
//			std::shared_ptr<Cube<float>> squaredInput = std::make_shared<Cube<float>>(
//				input_->n_rows, input_->n_cols, input_depth);
//			for (uword c = 0; c < input_depth; ++c) {
//				squaredInput->slice(c) = arma::square(input_->slice(c));
//			}
//
//
//			im2col(squaredInput, prevLocalLoss, input2col, kernel2col,
//			       kernel_size_.height, kernel_size_.width);
//			//output size = [prevLocalLoss->n_slices; n_filters * kernel_size_h * kernel_size_w] 
//			cross_correlation = kernel2col * input2col;
//			for (uword k = 0; k < n_filters_; ++k) {
//				for (uword c = 0; c < weights_.n_slices; ++c)
//					result.first.data[k].slice(c) = arma::reshape(
//						cross_correlation(k, span(c * kernel_size_.height 
//												  * kernel_size_.width,
//												  c * kernel_size_.height 
//												  * kernel_size_.width
//												  + kernel_size_.height
//												  * kernel_size_.width - 1)),
//						weights_.n_rows, weights_.n_cols);
//			}
//			//compute gradient for bias:
//			for (uword c = 0; c < output_depth; ++c) {
//				float sum = arma::sum(arma::sum(prevLocalLoss->slice(c)));
//				for (uword d = 0; d < biasWeights_.n_slices; ++d) {
//					result.second.data[c](0, 0, d) = sum;
//				}
//			}
//
//			// propagate error to bottom layer:
//			uword unpadded_input_height = input_->n_rows - 2 * padding_.height;
//			uword unpadded_input_width = input_->n_cols - 2 * padding_.width;
//			// we must add zeros on borders to input loss to get conv result dimension
//			// equal to input signals
//			uword pad_h = (unpadded_input_height -
//				((prevLocalLoss->n_rows - kernel_size_.height) / stride_ + 1)) / 2;
//			uword pad_w = (unpadded_input_width -
//				((prevLocalLoss->n_cols - kernel_size_.width) / stride_ + 1)) / 2;
//			std::shared_ptr<arma::Cube<float>> paddedPrevLoss = std::make_shared<
//				arma::Cube<float>>(prevLocalLoss->n_rows + 2 * pad_h,
//									prevLocalLoss->n_cols + 2 * pad_w,
//									prevLocalLoss->n_slices, fill::zeros);
//
//			(*paddedPrevLoss)(span(pad_h, pad_h + output_height - 1),
//							  span(pad_w, pad_w + output_width - 1), span::all
//							  ) = std::move((*prevLocalLoss));
//			// for propagate error to the previous layer we should use
//			// convolution instead cross-correlation
//			// in 2nd order backpropagation we must square our filters
//			tensor4d squaredFlippedKernel(kernel_size_.height, kernel_size_.width,
//			                              n_filters_, input_depth);
//			for (uword n = 0; n < input_depth; ++n) {
//				for (uword c = 0; c < n_filters_; ++c) {
//					squaredFlippedKernel.data[n].slice(c) = flipud(fliplr(
//						arma::square(weights_.data[c].slice(n))));
//				}
//			}
//			im2col(paddedPrevLoss, squaredFlippedKernel, input2col, kernel2col,
//			       unpadded_input_height, unpadded_input_width);
//			Mat<float> convolution = kernel2col * input2col;
//			if (!localLoss_) {
//				localLoss_ = std::make_shared<Cube<float>>(unpadded_input_height,
//				                                            unpadded_input_width,
//				                                            input_depth);
//			} else {
//			localLoss_->set_size(unpadded_input_height,
//								 unpadded_input_width,
//								 input_depth);
//			}
//			for (uword c = 0; c < input_depth; ++c) {
//				(*localLoss_).slice(c) = arma::reshape(convolution.row(c),
//				                                       unpadded_input_height,
//				                                       unpadded_input_width);
//			}
//
			return result;
		}
	}
}
