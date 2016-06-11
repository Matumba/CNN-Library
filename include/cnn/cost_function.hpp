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
#include <cmath>

namespace cnn
{
	namespace nn
	{
		class BaseCostFunction
		{
		public:
			virtual ~BaseCostFunction() = default;

			virtual double Compute(const arma::Col<double>& labels,
			                       const arma::Col<double>& hypothesis) const noexcept = 0;
			virtual double Derivative(double label, double hypothesis) const noexcept = 0;
			virtual double SecondDerivative(double label, double hypothesis) const noexcept = 0;
		};


		class EuclidianLoss : public BaseCostFunction
		{
		public:
			double Compute(const arma::Col<double>& labels,
			               const arma::Col<double>& hypothesis) const noexcept override;
			double Derivative(double label, double hypothesis) const noexcept override;
			double SecondDerivative(double label, double hypothesis) const noexcept override;
		};

		class CrossEntropy : public BaseCostFunction
		{
		public:
			// instead of use output from softmax layer we get receptive fields
			// from softmax layer and compute log-softmax using log-sum-exp trick
			// to prevent numerical underflow
			double Compute(const arma::Col<double>& labels,
			               const arma::Col<double>& receptiveFields) const noexcept override;
			// actually this not de/dy. this de/dz for cross entropy with softmax function
			double Derivative(double label, double hypothesis) const noexcept override;
			double SecondDerivative(double label, double hypothesis) const noexcept override;
		private:
			double LogSumExp(const arma::Col<double>& data, double max) const;
		};

		inline double EuclidianLoss::Compute(const arma::Col<double>& labels,
		                                     const arma::Col<double>& hypothesis) const noexcept
		{
			arma::Col<double> diff = labels - hypothesis;
			return arma::as_scalar(diff.t() * diff) / 2.0;
		}

		inline double EuclidianLoss::Derivative(double label, double hypothesis) const noexcept
		{
			return hypothesis - label;
		}

		inline double EuclidianLoss::SecondDerivative(double label, double hypothesis) const noexcept
		{
			return 1;
		}

		inline double CrossEntropy::Compute(const arma::Col<double>& labels,
		                                    const arma::Col<double>& hypothesis) const noexcept
		{
#ifndef NDEBUG
			assert(hypothesis.n_rows == labels.n_rows);
#endif
			double maxVal = hypothesis.max();
			double logSum = LogSumExp(hypothesis, maxVal);
			double sum = 0;
			for (arma::uword i = 0; i < hypothesis.n_rows; ++i) {
				sum += labels(i) * (hypothesis(i) - maxVal - logSum);
			}
			return -sum;
		}

		inline double CrossEntropy::Derivative(double label, double hypothesis) const noexcept
		{
			return hypothesis - label;
		}

		inline double CrossEntropy::SecondDerivative(double label, double hypothesis) const noexcept
		{
			return label - 2 * hypothesis + std::pow(hypothesis, 2);
		}

		inline double CrossEntropy::LogSumExp(const arma::Col<double>& data, double max) const
		{
			double sum = arma::sum(arma::exp(data - max));
			return max + std::log(sum);
		}
	}
}
