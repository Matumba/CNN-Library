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
#include "util.hpp"
#include <memory>
#include <cmath>


namespace cnn
{
	namespace nn
	{
		class BaseActivationFunction
		{
		public:

			virtual ~BaseActivationFunction() = default;
			virtual void Compute(const std::shared_ptr<arma::Cube<float>>& src,
			                     const std::shared_ptr<arma::Cube<float>>& dst) const noexcept = 0;
			virtual float Derivative(float value) const noexcept = 0;
		};

		class ReLU : public BaseActivationFunction
		{
		public:
			void Compute(const std::shared_ptr<arma::Cube<float>>& src,
			             const std::shared_ptr<arma::Cube<float>>& dst) const noexcept override;

			float Derivative(float value) const noexcept override;
		};

		class Tanh : public BaseActivationFunction
		{
		public:
			void Compute(const std::shared_ptr<arma::Cube<float>>& src,
			             const std::shared_ptr<arma::Cube<float>>& dst) const noexcept override;
			float Derivative(float value) const noexcept override;
		};

		class SoftMax : public BaseActivationFunction
		{
		public:
			void Compute(const std::shared_ptr<arma::Cube<float>>& src,
			             const std::shared_ptr<arma::Cube<float>>& dst) const noexcept override;
			float Derivative(float value) const noexcept override;
		};


		inline void ReLU::Compute(const std::shared_ptr<arma::Cube<float>>& src,
		                          const std::shared_ptr<arma::Cube<float>>& dst) const noexcept
		{
#ifndef NDEBUG
			// we're pass by reference so shared_ptr doesn't guarantee that src and dst is't free
			assert(src);
			assert(dst);
			assert(src->n_slices == dst->n_slices && src->n_rows == dst->n_rows);
#endif

			for (arma::uword s = 0; s < src->n_slices; ++s) {
				// arma store data in column-major order
				for (arma::uword c = 0; c < src->n_cols; ++c) {
					for (arma::uword r = 0; r < src->n_rows; ++r) {
						(*dst)(r, c, s) = std::max(0.0f, (*src)(r, c, s));
					}
				}
			}
		}

		inline float ReLU::Derivative(float value) const noexcept
		{
			return 1 / (1 + std::exp(-value));
		}

		inline void Tanh::Compute(const std::shared_ptr<arma::Cube<float>>& src, const std::shared_ptr<arma::Cube<float>>& dst) const noexcept
		{
			for (arma::uword s = 0; s < src->n_slices; ++s) {
				// arma store data in column-major order
				for (arma::uword c = 0; c < src->n_cols; ++c) {
					for (arma::uword r = 0; r < src->n_rows; ++r) {
						(*dst)(r, c, s) = std::tanh((*src)(r, c, s));
					}
				}
			}
		}

		inline float Tanh::Derivative(float value) const noexcept
		{
			return 1 - std::tanh(value) * std::tanh(value);
		}

		inline void SoftMax::Compute(const std::shared_ptr<arma::Cube<float>>& src,
		                             const std::shared_ptr<arma::Cube<float>>& dst) const noexcept
		{
#ifndef NDEBUG
			assert(src);
			assert(dst);
			// softmax should be using in output row-vector
			assert(src->n_slices == 1 && src->n_cols == 1);
			assert(src->n_slices == dst->n_slices && src->n_rows == dst->n_rows);
#endif
			float maxVal = src->slice(0).col(0).max();
			float denominator = arma::sum<arma::Col<float>>(
				arma::exp(src->slice(0).col(0) - maxVal));		

			float numerator;
			for (arma::uword r = 0; r < src->n_rows; ++r) {
				numerator = std::exp((*src)(r, 0, 0) - maxVal);
				(*dst)(r, 0, 0) = numerator / denominator;
			}
		}

		inline float SoftMax::Derivative(float value) const noexcept
		{
			return value * (1 - value);
		}
	}
}
