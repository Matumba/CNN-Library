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

		class ConvolutionalLayer final : public BaseLayer
		{
		public:
			ConvolutionalLayer(kernel_size_t kernel_size, std::size_t kernel_count,
							   arma::uword depth, std::size_t stride,
							   pad_size_t padding = pad_size_t(0, 0),
							   std::unique_ptr<BaseActivationFunction> activFun = std::make_unique<ReLU>());
			~ConvolutionalLayer() = default;

			void Forward(std::shared_ptr<arma::Cube<float>> input) override;
			std::pair<tensor4d, tensor4d> Backward(
				const std::shared_ptr<arma::Cube<float>>& prevLocalLoss) override;
			std::pair<tensor4d, tensor4d> Backward2nd(
				const std::shared_ptr<arma::Cube<float>>& prevLocalLoss) override;

		private:
			// add zero padding on borders
			void AddPadding(std::shared_ptr<arma::Cube<float>> &src,
							arma::uword n_rows, arma::uword n_cols,
							arma::uword n_slices) noexcept;
			void im2col(const std::shared_ptr<arma::Cube<float>> &src_data,
						const tensor4d& src_kernel, arma::Mat<float> &dst_data,
						arma::Mat<float> &dst_kernel,
						arma::uword height, arma::uword width) const noexcept;
			//im2col version for unsymmetric data. used for computing gradients
			void im2col(const std::shared_ptr<arma::Cube<float>>& src_data,
						const std::shared_ptr<arma::Cube <float>>& src_kernel,
						arma::Mat<float>& dst_data,
						arma::Mat<float>& dst_kernel,
						arma::uword height, arma::uword width) const noexcept;

		private:
			// hyperparameters:
			// spatial extent of sliding window
			kernel_size_t kernel_size_;
			// the amount of zero padding
			kernel_size_t padding_;
			// number of filters = output depth
			std::size_t n_filters_;
			std::size_t stride_;
		};

		inline
		ConvolutionalLayer::ConvolutionalLayer(kernel_size_t kernel_size, std::size_t kernel_count,
											   arma::uword depth, std::size_t stride,
											   pad_size_t padding,
											   std::unique_ptr<BaseActivationFunction> activFun)
			: BaseLayer(kernel_size.height, kernel_size.width, depth, kernel_count,
						std::move(activFun)),
			kernel_size_(kernel_size), padding_(padding),
			n_filters_(kernel_count), stride_(stride)
		{
			biasWeights_ = tensor4d(1, 1, depth, kernel_count);
		}


		inline
		// ReSharper disable once CppMemberFunctionMayBeConst
		void ConvolutionalLayer::AddPadding(std::shared_ptr<arma::Cube<float>>& src,
											arma::uword n_rows, arma::uword n_cols,
											arma::uword n_slices) noexcept
		{
			// add zero padding on border
			src = std::make_shared<arma::Cube<float>>(n_rows + 2 * padding_.height,
													  n_cols + 2 * padding_.width,
													  n_slices);
			src->fill(0.0);
		}

	}
}
