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
#include "neural_network.hpp"
#include <memory>

namespace cnn
{
	namespace solver
	{
		class BaseSolver
		{
		public:
			BaseSolver(std::shared_ptr<nn::NeuralNetwork> network,
					   arma::uword batch_size,
					   float learning_rate,
					   arma::uword max_epoch,
					   arma::uword test_interval,
					   arma::uword test_size,
					   arma::uword snapshot_interval,
					   std::wstring snapshot_prefix = L"");
			virtual ~BaseSolver() = default;
			virtual void Solve() = 0;
		protected:
			std::shared_ptr<nn::NeuralNetwork> net_;
			std::wstring snapshot_prefix_;

			float learning_rate_;
			// training batch_size
			arma::uword batch_size_;
			// maximum number of iterations
			arma::uword max_epoch_;
			// test error on test set every test_interval
			arma::uword test_interval_;
			// every test_interval epoch test test_size data from test data-set
			arma::uword test_size_;
			// write to file every snapshot_interval epoches
			arma::uword snapshot_interval_;

		};

		class SgdSolver final : public BaseSolver
		{
		public:
			SgdSolver(std::shared_ptr<nn::NeuralNetwork> network,
					  arma::uword batch_size, float learning_rate,
					  arma::uword max_epoch, arma::uword test_interval,
					  arma::uword test_size, arma::uword snapshot_interval,
					  std::wstring snapshot_prefix = L"")
				: BaseSolver(network, batch_size, learning_rate, max_epoch,
							 test_interval, test_size, snapshot_interval, snapshot_prefix){}

			void Solve() override;
		};

		class SdlmSolver final : public BaseSolver
		{
		public:
			SdlmSolver(std::shared_ptr<nn::NeuralNetwork> network,
					   arma::uword batch_size,
					   float learning_rate,
					   float mu,
					   float gamma,
					   arma::uword max_epoch,
					   arma::uword test_interval,
					   arma::uword test_size,
					   arma::uword snaprshot_interval,
					   std::wstring snapshot_prefix = L"");


			void Solve() override;
		private:
			// Levenberg–Marquardt hyperparameters
			float mu_;
			// smth like momentum for computing diagonal hessian
			float gamma_;
		};

		inline 
		BaseSolver::BaseSolver(std::shared_ptr<nn::NeuralNetwork> network,
							   arma::uword batch_size, float learning_rate,
							   arma::uword max_epoch, arma::uword test_interval,
							   arma::uword test_size, arma::uword snapshot_interval,
							   std::wstring snapshot_prefix)
			: net_(network), batch_size_(batch_size), learning_rate_(learning_rate),
			max_epoch_(max_epoch), test_interval_(test_interval),
			test_size_(test_size), snapshot_interval_(snapshot_interval),
			snapshot_prefix_(snapshot_prefix)
		{
			
		}

		inline 
		SdlmSolver::SdlmSolver(std::shared_ptr<nn::NeuralNetwork> network,
							   arma::uword batch_size, float learning_rate,
							   float mu, float gamma, arma::uword max_epoch,
							   arma::uword test_interval, arma::uword test_size,
							   arma::uword snaprshot_interval,
							   std::wstring snapshot_prefix)
			: BaseSolver(network, batch_size, learning_rate, max_epoch,
						 test_interval, test_size, snaprshot_interval,
						 snapshot_prefix), mu_(mu), gamma_(gamma)
		{}

	}
}

