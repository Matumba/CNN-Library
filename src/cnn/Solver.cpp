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
#include "solver.hpp"
#include "util.hpp"
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

namespace cnn
{
	namespace solver
	{
		void SgdSolver::Solve()
		{
			using namespace arma;
#ifndef NDEBUG
			assert(net_->is_initialized());
#endif
			std::vector<std::pair<cnn::tensor4d, cnn::tensor4d>> gradient;
			std::vector<std::pair<cnn::tensor4d, cnn::tensor4d>> iter_gradient;
			for (uword epoch = 0; epoch < max_epoch_; ++epoch) {
				float error = 0.0;
				net_->LoadTrainImage();
				net_->Forward();
				error += net_->Error();
				gradient = net_->Backpropagation();
				// training network on training dataset
				std::cout << boost::format(
					"compute error on training dataset for %1% samples on %2% training epoches..."
				) % batch_size_ % (epoch + 1) << "\n";
				for (uword i = 1; i < batch_size_; ++i) {
					net_->LoadTrainImage();
					net_->Forward();
					error += net_->Error();
					iter_gradient = net_->Backpropagation();
					for (std::size_t n = 0; n < gradient.size(); ++n) {
						// weights
						for (uword item = 0; item < gradient[n].first.data.size(); ++item) {
							gradient[n].first.data[item] += iter_gradient[n].first.data[item];
						}
						//biases
						for (uword item = 0; item < gradient[n].second.data.size(); ++item) {
							gradient[n].second.data[item] += iter_gradient[n].second.data[item];
						}
					}
				}
				// get average deltas
				for (std::size_t n = 0; n < gradient.size(); ++n) {
					// weights
					for (uword item = 0; item < gradient[n].first.data.size(); ++item) {
						gradient[n].first.data[item] /= batch_size_;
					}
					//biases
					for (uword item = 0; item < gradient[n].second.data.size(); ++item) {
						gradient[n].second.data[item] /= batch_size_;
					}
				}
				error /= batch_size_;
				std::cout << "training error = " << error << "\n";
				std::cout << "update weights...\n";
				for (std::size_t n = 0; n < gradient.size(); ++n) {
					cnn::tensor4d& weigths = net_->Weights(n);
					cnn::tensor4d& bias_weigths = net_->BiasWeights(n);
					for (uword item = 0; item < weigths.data.size(); ++item) {
						for (uword slice = 0; slice < weigths.n_slices; ++slice) {
							for (uword col = 0; col < weigths.n_cols; ++col) {
								for (uword row = 0; row < weigths.n_rows; ++row) {
									weigths.data[item](row, col, slice) += learning_rate_
										* gradient[n].first.data[item](row, col, slice);
								}
							}
						}
					}
					//biases
					for (uword item = 0; item < bias_weigths.data.size(); ++item) {
						for (uword slice = 0; slice < bias_weigths.n_slices; ++slice) {
							for (uword col = 0; col < bias_weigths.n_cols; ++col) {
								for (uword row = 0; row < bias_weigths.n_rows; ++row) {
									bias_weigths.data[item](row, col, slice) += learning_rate_
										* gradient[n].second.data[item](row, col, slice);
								}
							}
						}
					}
				}

				if (snapshot_interval_ != 0 && (epoch + 1) % snapshot_interval_ == 0) {
					boost::filesystem::ofstream out;
					std::wstring path = snapshot_prefix_ + (boost::wformat(L"_%1%") % (epoch + 1)).str();
					out.open(path, std::ios::binary);
					if (!out.is_open()) {
						std::cout << "cannot save weights to file\n";
					} else {
						net_->SaveWeights(out);
						out.close();
					}
				}
			}
		}

		void SdlmSolver::Solve()
		{
			using namespace arma;
#ifndef NDEBUG
			assert(net_->is_initialized());
#endif
			std::vector<std::pair<cnn::tensor4d, cnn::tensor4d>> gradient;
			std::vector<std::pair<cnn::tensor4d, cnn::tensor4d>> iter_gradient;
			std::vector<std::pair<cnn::tensor4d, cnn::tensor4d>> current_hessian;
			std::vector<std::pair<cnn::tensor4d, cnn::tensor4d>> new_hessian;
			std::vector<std::pair<cnn::tensor4d, cnn::tensor4d>> old_hessian;
			std::vector<std::pair<cnn::tensor4d, cnn::tensor4d>> iter_hessian;
			for (uword epoch = 0; epoch < max_epoch_; ++epoch) {
				float error = 0.0;
				if (test_interval_ != 0 && (epoch + 1) % test_interval_ == 0) {
					std::cout << boost::format(
						"compute error on test dataset for %1% samples on %2% training epoches..."
					) % test_size_ % (epoch + 1) << "\n";

					// compute error on test data-set
					for (uword i = 0; i < test_size_; ++i) {
						net_->LoadTestImage();
						net_->Forward();
						error += net_->Error();
					}
					error /= test_size_;
					std::cout << "test error = " << error << "\n";
				}
				error = 0.0;
				net_->LoadTrainImage();
				net_->Forward();
				error += net_->Error();
				gradient = net_->Backpropagation();
				current_hessian = net_->Backpropagation_2nd();
				// training network on training dataset
				std::cout << boost::format(
					"compute error on training dataset for %1% samples on %2% training epoches..."
				) % batch_size_ % (epoch + 1) << "\n";
				for (uword i = 1; i < batch_size_; ++i) {
					net_->LoadTrainImage();
					net_->Forward();
					error += net_->Error();
					iter_gradient = net_->Backpropagation();
					iter_hessian = net_->Backpropagation_2nd();

					for (std::size_t n = 0; n < gradient.size(); ++n) {
						// weights
						for (uword item = 0; item < gradient[n].first.data.size(); ++item) {
							gradient[n].first.data[item] += iter_gradient[n].first.data[item];
							current_hessian[n].first.data[item] += iter_hessian[n].first.data[item];
						}
						//biases
						for (uword item = 0; item < gradient[n].second.data.size(); ++item) {
							gradient[n].second.data[item] += iter_gradient[n].second.data[item];
							current_hessian[n].second.data[item] += iter_hessian[n].second.data[item];
						}
					}
				}
				// get average deltas
				for (std::size_t n = 0; n < gradient.size(); ++n) {
					// weights
					for (uword item = 0; item < gradient[n].first.data.size(); ++item) {
						gradient[n].first.data[item] /= batch_size_;
						current_hessian[n].first.data[item] /= batch_size_;
					}
					//biases
					for (uword item = 0; item < gradient[n].second.data.size(); ++item) {
						gradient[n].second.data[item] /= batch_size_;
						current_hessian[n].second.data[item] /= batch_size_;
					}
				}
				error /= batch_size_;
				std::cout << "training error = " << error << "\n";
				std::cout << "update weights...\n";
				new_hessian = std::move(current_hessian);
				if (!old_hessian.empty()) {
					// new_hessian = (1 - gamma) % old_hessian + gamma % new_hessian
					for (std::size_t n = 0; n < new_hessian.size(); ++n) {
						// weights
						for (uword item = 0; item < new_hessian[n].first.data.size(); ++item) {
							new_hessian[n].first.data[item]
									= (1 - gamma_) * old_hessian[n].first.data[item]
									+ gamma_ * new_hessian[n].first.data[item];
						}
						//biases
						for (uword item = 0; item < new_hessian[n].second.data.size(); ++item) {
							new_hessian[n].second.data[item]
									= (1 - gamma_) * old_hessian[n].second.data[item]
									+ gamma_ * new_hessian[n].second.data[item];
						}
					}
				}
				// found optimal learning rate for all weights and update in-place
				for (std::size_t n = 0; n < new_hessian.size(); ++n) {
					cnn::tensor4d& weigths = net_->Weights(n);
					cnn::tensor4d& bias_weigths = net_->BiasWeights(n);
					float local_learning_rate;
					for (uword item = 0; item < weigths.data.size(); ++item) {
						for (uword slice = 0; slice < weigths.n_slices; ++slice) {
							for (uword col = 0; col < weigths.n_cols; ++col) {
								for (uword row = 0; row < weigths.n_rows; ++row) {
									local_learning_rate = learning_rate_
											/ (new_hessian[n].first.data[item](row, col, slice) + mu_);
									weigths.data[item](row, col, slice) += local_learning_rate 
											* gradient[n].first.data[item](row, col, slice);
								}
							}
						}
					}
					//biases
					for (uword item = 0; item < bias_weigths.data.size(); ++item) {
						for (uword slice = 0; slice < bias_weigths.n_slices; ++slice) {
							for (uword col = 0; col < bias_weigths.n_cols; ++col) {
								for (uword row = 0; row < bias_weigths.n_rows; ++row) {
									local_learning_rate = learning_rate_
											/ (new_hessian[n].second.data[item](row, col, slice) + mu_);
									bias_weigths.data[item](row, col, slice) += local_learning_rate 
											* gradient[n].second.data[item](row, col, slice);
								}
							}
						}
					}

				}
				old_hessian = std::move(new_hessian);

				if (snapshot_interval_ != 0 && (epoch + 1) % snapshot_interval_ == 0) {
					boost::filesystem::ofstream out;
					std::wstring path = snapshot_prefix_ + (boost::wformat(L"_%1%") % (epoch + 1)).str();
					out.open(path, std::ios::binary);
					if (!out.is_open()) {
						std::cout << "cannot save weights to file\n";
					} else {
						net_->SaveWeights(out);
						out.close();
					}
				}
			}
		}
	}
}
