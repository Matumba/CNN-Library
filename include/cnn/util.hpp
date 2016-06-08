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
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <armadillo>
#include <vector>
#include <cstdint>
#include "assert.h"

namespace cnn
{
	struct tensor4d
	{
		tensor4d();
		tensor4d(arma::uword height, arma::uword width, arma::uword depth, std::size_t count);
		tensor4d(const tensor4d &item);
		tensor4d(tensor4d &&item);
		tensor4d& operator=(const tensor4d &item);
		tensor4d& operator=(tensor4d &&item);

		std::vector<arma::Cube<float>> data;
		std::size_t n_size;
		arma::uword n_rows;
		arma::uword n_cols;
		arma::uword n_slices;
	};

	struct kernel_size_t
	{
		explicit kernel_size_t(std::size_t h, std::size_t w)
			: height(h), width(w)
		{
		}

		explicit kernel_size_t(std::size_t size)
			: kernel_size_t(size, size)
		{
		}

		std::size_t height;
		std::size_t width;
	};

	typedef kernel_size_t pad_size_t;

	inline tensor4d::tensor4d()
		: n_rows(0), n_cols(0), n_size(0), n_slices(0)
	{
	}

	inline tensor4d::tensor4d(arma::uword height, arma::uword width, arma::uword depth, std::size_t count)
		: n_rows(height), n_cols(width), n_slices(depth), n_size(count)
	{
		data.reserve(n_size);
		for (std::size_t i = 0; i < n_size; ++i)
			data.push_back(arma::Cube<float>(n_rows, n_cols, n_slices));
	}

	inline tensor4d::tensor4d(const tensor4d& item)
		: data(item.data),
		n_rows(item.n_rows), n_cols(item.n_cols), n_slices(item.n_cols), n_size(item.n_cols)
	{
	}

	inline tensor4d::tensor4d(tensor4d&& item)
		: data(std::move(item.data)), 
		n_rows(item.n_rows), n_cols(item.n_cols), n_slices(item.n_cols), n_size(item.n_cols)
	{}

	inline tensor4d& tensor4d::operator=(const tensor4d& item)
	{
		data = item.data;
		n_rows = item.n_rows;
		n_cols = item.n_cols;
		n_slices = item.n_cols;
		n_size = item.n_cols;
		return *this;
	}

	inline tensor4d& tensor4d::operator=(tensor4d&& item)
	{
		data= std::move(item.data);
		n_rows = item.n_rows;
		n_cols = item.n_cols;
		n_slices = item.n_cols;
		n_size = item.n_cols;

		item.n_rows = 0;
		item.n_cols = 0;
		item.n_cols = 0;
		item.n_cols = 0;
		return *this;
	}

	arma::Cube<float> vectorise(const arma::Cube<float>& src);

	arma::Cube<float> unvectorise(const arma::Cube<float>& src, arma::uword height,
	                              arma::uword width, arma::uword depth);

	//convert cv mat with 3 channels to arma cube
	arma::Cube<float> cvMat2armaCube(const cv::Mat& src);
	cv::Mat armaMat2cvMat(const arma::Mat<float> &src);
}
