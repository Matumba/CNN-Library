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
#include <boost/filesystem.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <string>
#include <utility>
#include <memory>
#include <cstddef>

namespace cnn
{

	class BaseImageLoader
	{
	public:
		virtual ~BaseImageLoader() = default;
		virtual bool LoadTestImage(std::shared_ptr<arma::Cube<double>>& dst,
						   arma::Col<double>& labels) = 0;
		virtual bool LoadTrainImage(std::shared_ptr<arma::Cube<double>>& dst,
								   arma::Col<double>& labels) = 0;
		virtual const std::wstring& LabelName(std::size_t id) const = 0;
	};

	class LfwLoader final : public BaseImageLoader
	{
	public:
		//TODO: add defines for use std::string instead std::wstring for *nix
		LfwLoader(const std::wstring& dataSetPath, const std::wstring& trainPath,
		          const std::wstring& testPath, cv::Size scaleSize);


		bool LoadTestImage(std::shared_ptr<arma::Cube<double>>& dst,
		                   arma::Col<double>& labels) override;


		bool LoadTrainImage(std::shared_ptr<arma::Cube<double>>& dst,
		                    arma::Col<double>& labels) override;

		const std::wstring& LabelName(std::size_t id) const override;

	private:
		bool loadImage(const std::wstring& folder, std::shared_ptr<arma::Cube<double>>& dst) const;

	private:
		std::vector<std::pair<std::wstring, arma::uword>> trainDataSet_;
		std::vector<std::pair<std::wstring, arma::uword>> testDataSet_;
		std::vector<std::wstring> labels_;
		std::wstring dataset_dir_;
		cv::Size scaleSize_;
	};

	inline
	LfwLoader::LfwLoader(const std::wstring& dataSetPath, const std::wstring& trainPath,
	                     const std::wstring& testPath, cv::Size scaleSize)
		: dataset_dir_(dataSetPath), scaleSize_(scaleSize)
	{
		std::wifstream in;
		in.open(trainPath);
#ifndef NDEBUG
		assert(in.is_open()); {
			namespace fs = boost::filesystem;
			assert(fs::is_directory(fs::path(dataset_dir_)));
		}
#endif
		if (dataset_dir_.back() != std::wstring(L"\\").back()) {
			dataset_dir_.append(L"\\");
		}

		std::size_t folders_amount;
		in >> folders_amount;
		trainDataSet_.reserve(folders_amount);
		labels_.reserve(folders_amount + 1);
		labels_.emplace_back(L"Unknown");

		std::wstring folder_name;
		arma::uword image_amount;
		while (in >> folder_name >> image_amount) {
			trainDataSet_.emplace_back(folder_name, image_amount);
			labels_.emplace_back(folder_name);
		}
		in.close();
		in.open(testPath);
#ifndef NDEBUG
		assert(in.is_open());
#endif
		in >> folders_amount;
		testDataSet_.reserve(folders_amount);
		while (in >> folder_name >> image_amount)
			testDataSet_.emplace_back(folder_name, image_amount);
	}

	inline const std::wstring& LfwLoader::LabelName(std::size_t id) const
	{
#ifndef NDEBUG
		assert(id < labels_.size());
#endif

		return labels_[id];
	}

}
