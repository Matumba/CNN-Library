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

namespace cnn
{
	namespace nn
	{
		class BaseCostFunction
		{
		public:
			virtual ~BaseCostFunction() = default;

			virtual float Compute(const arma::Col<float>& labels,
			                      const arma::Col<float>& hypothesis) const noexcept = 0;
			virtual float Derivative(float label, float hypothesis) const noexcept = 0;
			// 2nd derivative for euclidian loss and cross-entropy = 1 
			// virtual float Derivative_2nd(float label, float hypothesis) const noexcept = 0;
		};

		class Loglikelihood : public BaseCostFunction
		{
		public:
			float Compute(const arma::Col<float>& labels,
						  const arma::Col<float>& hypothesis) const noexcept override;
			float Derivative(float label, float hypothesis) const noexcept override;
		};

		class EuclidianLoss : public BaseCostFunction
		{
		public:
			float Compute(const arma::Col<float>& labels,
			              const arma::Col<float>& hypothesis) const noexcept override;
			float Derivative(float label, float hypothesis) const noexcept override;
		};

		class CrossEntropy : public BaseCostFunction
		{
		public:
			float Compute(const arma::Col<float>& labels,
			              const arma::Col<float>& hypothesis) const noexcept override;
			float Derivative(float label, float hypothesis) const noexcept override;
		};

		inline float EuclidianLoss::Compute(const arma::Col<float>& labels,
		                                    const arma::Col<float>& hypothesis) const noexcept
		{
			arma::Col<float> diff = labels - hypothesis;
			return arma::as_scalar(diff.t() * diff) / 2.0f;
		}

		inline float EuclidianLoss::Derivative(float label, float hypothesis) const noexcept
		{
			return hypothesis - label;
		}

		inline float CrossEntropy::Compute(const arma::Col<float>& labels,
		                                   const arma::Col<float>& hypothesis) const noexcept
		{
			return -1 * arma::as_scalar(labels.t() * arma::trunc_log(hypothesis));
		}

		inline float CrossEntropy::Derivative(float label, float hypothesis) const noexcept
		{
			return hypothesis - label;
		}
	}
}
