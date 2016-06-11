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
#include "activation_function.hpp"
#include "util.hpp"
#include <armadillo>
#include <memory>
#include <utility>
#include <cstdint>

namespace cnn
{
	namespace nn
	{
		class BaseLayer
		{
		public:
			BaseLayer(arma::uword height, arma::uword width, arma::uword depth,
					  std::size_t amount, std::unique_ptr<BaseActivationFunction> activFunc);
			virtual ~BaseLayer() = default;
			// propagate signal from bottom to top
			virtual void Forward(std::shared_ptr<arma::Cube<double>> input) = 0;
			// propagate error from top to bottom and compute gradient
			virtual std::pair<tensor4d, tensor4d> Backward(
				const std::shared_ptr<arma::Cube<double>> &prevLocalLoss) = 0;
			// propagate error from top to bottom and compute hessian
			virtual std::pair<tensor4d, tensor4d> Backward2nd(
				const std::shared_ptr<arma::Cube<double>> &prevLocalLoss) = 0;
			// get propagated local error to for previous layer
			const std::shared_ptr<arma::Cube<double>>& LocalLoss() const noexcept;
			std::shared_ptr<arma::Cube<double>> Output() const noexcept;
			std::shared_ptr<arma::Cube<double>> ReceptiveField() const noexcept;

			tensor4d& Weights() noexcept
			{
				return weights_;
			}
			tensor4d& BiasWeights() noexcept
			{
				return biasWeights_;
			}

			//const tensor4d& GetWeights() const noexcept;
			//const tensor4d& GetBiasWeights() const noexcept;
			//void SetWeights(const tensor4d& weights);
			//void SetWeights(tensor4d&& weights);
			//void SetBiasWeights(const tensor4d& biasWeights);
			//void SetBiasWeights(tensor4d&& biasWeights);

			bool LoadWeights(std::ifstream& in);
			bool SaveWeights(std::ofstream& out) const;
			// initialize all weights in this layer using Gaussian distribution
			void InitWeights() noexcept;
			bool is_initialized() const noexcept;
		protected:
			tensor4d weights_;
			tensor4d biasWeights_;

			// propagated local error to the next layer
			std::shared_ptr<arma::Cube<double>> localLoss_;
			// y = f(v)
			std::shared_ptr<arma::Cube<double>> output_;
			// v = operator(input, weights)
			std::shared_ptr<arma::Cube<double>> receptiveField_;
			// forwarded signal from previous layer
			std::shared_ptr<arma::Cube<double>> input_;
			// nonlinearity
			std::unique_ptr<BaseActivationFunction> activFunc_;

			//weights parameters
//			std::size_t amount_;
//			arma::uword depth_;
//			arma::uword width_;
//			arma::uword height_;
			// weights status
			bool initialized_;
		};


		inline BaseLayer::BaseLayer(arma::uword height, arma::uword width, arma::uword depth,
									std::size_t amount,
									std::unique_ptr<BaseActivationFunction> activFunc)
			: weights_(height, width, depth, amount)
			/*, biasWeights_(1, 1, depth, amount)*/,
			activFunc_(std::move(activFunc)),
//			amount_(amount), depth_(depth), width_(width), height_(height),
			initialized_(false)
		{}


		inline const std::shared_ptr<arma::Cube<double>>& BaseLayer::LocalLoss() const noexcept
		{
			return localLoss_;
		}

		inline std::shared_ptr<arma::Cube<double>> BaseLayer::Output() const noexcept
		{
			return output_;
		}

		inline std::shared_ptr<arma::Cube<double>> BaseLayer::ReceptiveField() const noexcept
		{
			return receptiveField_;
		}

		/*inline const tensor4d& BaseLayer::GetWeights() const noexcept
		{
			return weights_;
		}

		inline const tensor4d& BaseLayer::GetBiasWeights() const noexcept
		{
			return biasWeights_;
		}

		inline void BaseLayer::SetWeights(const tensor4d& weights)
		{
			weights_ = weights;
		}

		inline void BaseLayer::SetWeights(tensor4d&& weights)
		{
			weights_ = std::move(weights_);
		}

		inline void BaseLayer::SetBiasWeights(const tensor4d& biasWeights)
		{
			biasWeights_ = biasWeights;
		}

		inline void BaseLayer::SetBiasWeights(tensor4d&& biasWeights)
		{
			biasWeights_ = std::move(biasWeights);
		}*/

		inline bool BaseLayer::LoadWeights(std::ifstream& in)
		{
			// for all common or polling layers
			if (weights_.n_size == 0)
				return true;
			if (!in.is_open())
				return false;
			for (std::size_t n = 0; n < weights_.n_size; ++n) {
				if (!(weights_.data[n].load(in, arma::arma_binary) 
					&& biasWeights_.data[n].load(in, arma::arma_binary))) {
					in.clear();
					return false;
				}
			}
			initialized_ = true;
			return true;
		}

		inline bool BaseLayer::SaveWeights(std::ofstream& out) const
		{
			if (weights_.n_size == 0)
				return true;
			if (!out.is_open())
				return false;
			for (std::size_t n = 0; n < weights_.n_size; ++n) {
				if (!(weights_.data[n].save(out, arma::arma_binary)
					  && biasWeights_.data[n].save(out, arma::arma_binary))) {
					out.clear();
					return false;
				}
			}
			return true;
		}

		inline void BaseLayer::InitWeights() noexcept
		{
			if (weights_.n_size == 0)
				return;

			for (std::size_t n = 0; n < weights_.n_size; ++n) {
				weights_.data[n] = 0.1 * arma::randn<arma::Cube<double>>(
					weights_.n_rows, weights_.n_cols, weights_.n_slices);
				biasWeights_.data[n] = 0.1 * arma::randn<arma::Cube<double>>(
					biasWeights_.n_rows, biasWeights_.n_cols, biasWeights_.n_slices);
			}
			initialized_ = true;
		}

		inline bool BaseLayer::is_initialized() const noexcept
		{
			return initialized_;
		}
	}
}
