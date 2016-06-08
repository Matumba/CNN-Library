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
#include "image_loader.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <random>
#include <cstddef>

namespace cnn
{
	bool LfwLoader::LoadTestImage(std::shared_ptr<arma::Cube<float>>& dst, 
								  arma::Col<float> &labels)
	{
		if (testDataSet_.empty())
			return false;
		std::size_t amount = testDataSet_.size();
		static std::mt19937 gen(std::random_device().operator()());
		std::uniform_int_distribution<std::size_t> uid(0, amount);

		arma::uword id = uid(gen);
		labels.set_size(trainDataSet_.size());
		labels.fill(0);
		return loadImage(testDataSet_[id].first, dst);
	}

	bool LfwLoader::LoadTrainImage(std::shared_ptr<arma::Cube<float>>& dst, 
								   arma::Col<float> &labels)
	{
		if (trainDataSet_.empty())
			return false;
		std::size_t amount = trainDataSet_.size();
		static std::mt19937 gen(std::random_device().operator()());
		std::uniform_int_distribution<std::size_t> uid(0, amount - 1);

		arma::uword id = uid(gen);
		labels.set_size(trainDataSet_.size());
		labels.fill(0);
		labels(id) = 1;
		return loadImage(trainDataSet_[id].first, dst);
	}

	bool LfwLoader::loadImage(const std::wstring &folder,
							  std::shared_ptr<arma::Cube<float>>& dst) const
	{
		namespace fs = boost::filesystem;
		fs::path dir_path(dataset_dir_ + folder);
		if (!fs::is_directory(dir_path))
			return false;
		fs::directory_iterator begin(dir_path);
		fs::directory_iterator end;
		arma::uword amount = std::distance(begin, end);
		std::vector<fs::directory_entry> imagesPath;
		imagesPath.reserve(amount);
		std::copy_if(fs::directory_iterator(dir_path), fs::directory_iterator(),
					 std::back_inserter(imagesPath), [](fs::directory_entry &x)
		{
			return fs::is_regular_file(x.path())
				&& x.path().filename().string() != std::string("Thumbs.db");
		});


		static std::mt19937 gen(std::random_device().operator()());
		std::uniform_int_distribution<std::size_t> uid(0, imagesPath.size() - 1);

		std::size_t number = uid(gen);
		std::string image_path = imagesPath[number].path().string();
		cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
		// my region of interest
		cv::Rect ROI(70, 60, 140, 140);
		// Note that this doesn't copy the data
		cv::Mat croppedImage = image(ROI);
		cv::Mat scaleImage;
		cv::resize(croppedImage, scaleImage, scaleSize_, 0, 0, CV_INTER_LINEAR);
		dst = std::make_shared<arma::Cube<float>>(cvMat2armaCube(scaleImage));
		//feature scaling
		dst->transform([](float val)
		{
			return val / 255.0f;
		});
		return true;
	}
}
