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

#include "util.hpp"

namespace cnn
{
	arma::Cube<float> vectorise(const arma::Cube<float>& src)
	{
		// already reshaped to column
		//		if (src.n_slices == 1 && src.n_cols == 1)
		//			return src;
		using namespace arma;

		Cube<float> dst(src.n_elem, 1, 1);
		uword size = 0;
		for (uword c = 0; c < src.n_slices; ++c) {
			dst.slice(0)(span(size, size + src.n_elem_slice - 1), 0
						 ) = vectorise(src.slice(c));
			size += src.n_elem_slice;
		}

		return dst;
	}

	arma::Cube<float> unvectorise(const arma::Cube<float>& src, arma::uword height,
								  arma::uword width, arma::uword depth)
	{
		using namespace arma;
#ifndef NDEBUG
		{
			assert(height != 0);
			assert(width != 0);
			assert(depth != 0);
			assert(height * width * depth == src.n_rows * src.n_cols * src.n_slices);
		}
#endif
		//		if (!(arma.n_slices == 1 && arma.n_cols == 1)) {
		//			return;
		//		}

		arma::Cube<float> dst(height, width, depth);
		uword size = 0;
		for (uword c = 0; c < depth; ++c) {
			for (uword column = 0; column < width; ++column) {
				dst.slice(c).col(column) = src.slice(0)(span(size, size + height - 1), 0);
				size += height;
			}
		}
		return dst;
	}

	arma::Cube<float> cvMat2armaCube(const cv::Mat& src)
	{
		cv::Mat f_image;
		src.convertTo(f_image, CV_32FC3);
		arma::uword n_channels = 3;
		std::vector<cv::Mat_<float>> channels;
		channels.reserve(n_channels);

		arma::Cube<float> cube(f_image.cols, f_image.rows, n_channels);
		for (arma::uword channel = 0; channel < n_channels; ++channel)
			channels.emplace_back(f_image.rows, f_image.cols, cube.slice(channel).memptr());
		cv::split(f_image, channels);

		return cube;
	}

	cv::Mat armaMat2cvMat(const arma::Mat<float> &src)
	{
		cv::Mat_<float> temp{ int(src.n_cols), int(src.n_rows), const_cast<float*>(src.memptr()) };
		cv::Mat dst;
		temp.convertTo(dst, CV_8UC1);
		return dst;
	}
}