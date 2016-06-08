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

#include "image_loader.hpp"
#include <armadillo>
#include <memory>
namespace cnn
{
	namespace nn
	{
		class InputLayer
		{
		public:
			InputLayer(std::unique_ptr<BaseImageLoader> loader);

			bool LoadTestImage();
			bool LoadTrainImage();

			void SetCustomImage(std::shared_ptr<arma::Cube<float>> image);

			bool is_empty() const noexcept;
			std::shared_ptr<arma::Cube<float>> Output() const noexcept;
			const arma::Col<float>& Labels() const noexcept;

			const std::wstring& LabelName(std::size_t id) const;

		private:
			arma::Col<float> labels_;
			std::shared_ptr<arma::Cube<float>> output_;
			std::unique_ptr<BaseImageLoader> loader_;
		};

		inline 
		InputLayer::InputLayer(std::unique_ptr<BaseImageLoader> loader)
			: loader_(std::move(loader)) {}

		inline bool InputLayer::LoadTestImage()
		{
			return loader_->LoadTestImage(output_, labels_);
		}

		inline bool InputLayer::LoadTrainImage()
		{
			return loader_->LoadTrainImage(output_, labels_);
		}

		inline void 
		InputLayer::SetCustomImage(std::shared_ptr<arma::Cube<float>> image)
		{
			output_ = image;
			labels_.reset();
		}

		inline bool InputLayer::is_empty() const noexcept
		{
#ifndef NDEBUG
			assert(output_);
#endif
			return output_->is_empty();
		}

		inline std::shared_ptr<arma::Cube<float>> InputLayer::Output() const noexcept
		{
			return output_;
		}

		inline const arma::Col<float>& InputLayer::Labels() const noexcept
		{
			return labels_;
		}

		inline const std::wstring& InputLayer::LabelName(std::size_t id) const
		{
			return loader_->LabelName(id);
		}
	}
}

