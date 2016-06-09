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
		class BasePoolingLayer : public BaseLayer
		{
		public:
			BasePoolingLayer(kernel_size_t kernel_size,
			                 std::size_t stride);
		protected:
			// each pooling layer use itself subsample method
			// you should implement this method for your class
			//virtual void SubSample(arma::uword output_height,
			//					   arma::uword output_width) noexcept = 0;

		protected:
			kernel_size_t kernel_size_;
			/*pad_size_t padding_;*/
			std::size_t stride_;
		};

		class MaxPoolingLayer final: public BasePoolingLayer
		{
		public:
			MaxPoolingLayer(kernel_size_t kernel_size,
							 std::size_t stride);
			void Forward(std::shared_ptr<arma::Cube<float>> input) override;
			std::pair<tensor4d, tensor4d> Backward(
				const std::shared_ptr<arma::Cube<float>>& prevLocalLoss) override;
			std::pair<tensor4d, tensor4d> Backward2nd(
				const std::shared_ptr<arma::Cube<float>>& prevLocalLoss) override;
		protected:
			//void SubSample(arma::uword output_height, arma::uword output_width) noexcept override;

		private:
			// when we propagate signals from bottom to top
			// we're using sliding window and vanishes all signals in its range except max
			// for correct propagate local errors we should vanishes all errors for disconnected signals
			// to reduce memory usage we may change cube to vector of vector which store bool values
			arma::Cube<arma::uword> connectIndexes_;

		};

		inline BasePoolingLayer::BasePoolingLayer(kernel_size_t kernel_size,
												  std::size_t stride)
			: BaseLayer(0, 0, 0, 0, nullptr),
			kernel_size_(kernel_size), stride_(stride) {}

		inline MaxPoolingLayer::MaxPoolingLayer(kernel_size_t kernel_size,
												std::size_t stride)
			: BasePoolingLayer(kernel_size, stride) {}

	}
}
