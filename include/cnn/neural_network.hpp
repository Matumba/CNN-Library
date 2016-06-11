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
#include "image_loader.hpp"
#include "input_layer.hpp"
#include "activation_function.hpp"
#include "cost_function.hpp"
#include "softmax_layer.hpp"
#include "fully_connected_layer.hpp"
#include "pooling_layer.hpp"
#include "convolutional_layer.hpp"
#include <armadillo>

#include <memory>
#include <vector>
#include <fstream>
#include <cstddef>

namespace cnn
{
	namespace nn
	{
		class NeuralNetwork
		{
		public:
			NeuralNetwork(std::unique_ptr<BaseImageLoader> loader,
						  std::unique_ptr<BaseCostFunction> costFunction);

			void AppendLayer(std::unique_ptr<BaseLayer> layer);
			std::size_t Size() const noexcept;
			const std::wstring& LabelName(std::size_t id) const;

			void InitWeights() noexcept;
			bool is_initialized() const noexcept;
			bool LoadWeights(std::ifstream& in);
			bool SaveWeights(std::ofstream& out) const;

			bool LoadTestImage();
			bool LoadTrainImage();
			void SetInputImage(std::shared_ptr<arma::Cube<double>> image);

			std::shared_ptr<arma::Cube<double>> Hypothesis() const noexcept;;
			std::shared_ptr<arma::Cube<double>> Output(std::size_t layerIdx) const noexcept;
			std::shared_ptr<arma::Cube<double>> ReceptiveField(std::size_t layerIdx) const noexcept;
			double Error();

			tensor4d& Weights(std::size_t layerIdx) noexcept
			{
				return layers_[layerIdx]->Weights();
			}
			tensor4d& BiasWeights(std::size_t layerIdx) noexcept
			{
				return layers_[layerIdx]->BiasWeights();
			}

			//propagate signals from bottom
			void Forward();
			// compute Gradient
			std::vector<std::pair<tensor4d, tensor4d>> Backpropagation();
			// compute Hessian
			std::vector<std::pair<tensor4d, tensor4d>> Backpropagation_2nd();
		private:
			std::vector<std::unique_ptr<BaseLayer>> layers_;
			std::unique_ptr<BaseCostFunction> costFunc_;
			std::unique_ptr<InputLayer> in_;

			bool initialized_;
		};

		inline 
		NeuralNetwork::NeuralNetwork(std::unique_ptr<BaseImageLoader> loader,
									 std::unique_ptr<BaseCostFunction> costFunction)
			: in_(std::make_unique<InputLayer>(std::move(loader))),
			costFunc_(std::move(costFunction)),
			initialized_(false)
		{}

		inline void NeuralNetwork::AppendLayer(std::unique_ptr<BaseLayer> layer)
		{
			layers_.emplace_back(std::move(layer));
		}

		inline std::size_t NeuralNetwork::Size() const noexcept
		{
			return layers_.size();
		}

		inline const std::wstring& NeuralNetwork::LabelName(std::size_t id) const
		{
			return in_->LabelName(id);
		}

		inline void NeuralNetwork::InitWeights() noexcept
		{
			for(std::unique_ptr<BaseLayer> & item : layers_) 
				item->InitWeights();
			initialized_ = true;
		}

		inline bool NeuralNetwork::is_initialized() const noexcept
		{
			return initialized_;
		}

		inline bool NeuralNetwork::LoadWeights(std::ifstream& in)
		{
			bool flag = true;
			for (std::unique_ptr<BaseLayer> & item : layers_) {
				flag = item->LoadWeights(in);
				if (!flag)
					return flag;
			}
			initialized_ = true;
			return flag;
		}

		inline bool NeuralNetwork::SaveWeights(std::ofstream& out) const
		{
			if (!initialized_ || layers_.empty())
				return false;
			bool flag = true;
			for (const std::unique_ptr<BaseLayer> & item : layers_) {
				flag = item->SaveWeights(out);
				if (!flag)
					return flag;
			}
			return flag;
		}

		inline bool NeuralNetwork::LoadTestImage()
		{
			return in_->LoadTestImage();
		}

		inline bool NeuralNetwork::LoadTrainImage()
		{
			return in_->LoadTrainImage();
		}

		inline void NeuralNetwork::SetInputImage(std::shared_ptr<arma::Cube<double>> image)
		{
			in_->SetCustomImage(std::move(image));
		}

		inline std::shared_ptr<arma::Cube<double>> NeuralNetwork::Hypothesis() const noexcept
		{
			return layers_.back()->Output();
		}

		inline
		std::shared_ptr<arma::Cube<double>> NeuralNetwork::ReceptiveField(
			std::size_t layerIdx) const noexcept
		{
#ifndef NDEBUG
			assert(layerIdx < layers_.size());
			assert(initialized_);
#endif
			return layers_[layerIdx]->ReceptiveField();
		}

		inline 
		std::shared_ptr<arma::Cube<double>> NeuralNetwork::Output(arma::uword layerIdx) const noexcept
		{
#ifndef NDEBUG
			assert(layerIdx < layers_.size());
			assert(initialized_);
#endif
			return layers_[layerIdx]->Output();
		}

		inline double NeuralNetwork::Error()
		{
#ifndef NDEBUG
			assert(!in_->Labels().empty());
#endif
			
			if (SoftMaxLayer *output = dynamic_cast<SoftMaxLayer*>(layers_.back().get())) {
				return costFunc_->Compute(in_->Labels(),
										  layers_.back()->ReceptiveField()->slice(0).col(0));
			} else {
				return costFunc_->Compute(in_->Labels(), layers_.back()->Output()->slice(0).col(0));
			}
		}

	}
}
