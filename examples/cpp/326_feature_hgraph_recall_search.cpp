// Copyright 2024-present the vsag project
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

#include <vsag/vsag.h>

#include <exception>
#include <iostream>
#include <random>
#include <vector>

int
main() {
    try {
        constexpr int64_t dim = 8;
        constexpr int64_t count = 100;

        std::mt19937 random(47);
        std::uniform_real_distribution<float> distribution;
        std::vector<int64_t> ids(count);
        std::vector<float> vectors(count * dim);
        for (int64_t i = 0; i < count; ++i) {
            ids[i] = i;
        }
        for (auto& value : vectors) {
            value = distribution(random);
        }

        auto base = vsag::Dataset::Make()
                        ->NumElements(count)
                        ->Dim(dim)
                        ->Ids(ids.data())
                        ->Float32Vectors(vectors.data())
                        ->Owner(false);
        auto index = vsag::Factory::CreateIndex("hgraph", R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 8,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 100,
            "build_thread_count": 1
        }
    })")
                         .value();
        index->Build(base).value();
        auto immutable_result = index->SetImmutable();
        if (not immutable_result.has_value()) {
            std::cerr << "SetImmutable failed: " << immutable_result.error().message << '\n';
            return 1;
        }

        std::vector<float> calibration_vectors(10 * dim);
        for (auto& value : calibration_vectors) {
            value = distribution(random);
        }
        auto calibration_queries = vsag::Dataset::Make()
                                       ->NumElements(10)
                                       ->Dim(dim)
                                       ->Float32Vectors(calibration_vectors.data())
                                       ->Owner(false);

        const std::vector<vsag::RecallTarget> targets = {{5, 0.9F}, {10, 0.9F}};
        auto calibration = index->CalibrateRecallSearch(targets, calibration_queries);
        if (not calibration.has_value()) {
            std::cerr << "CalibrateRecallSearch failed: " << calibration.error().message << '\n';
            return 1;
        }
        for (const auto& result : calibration.value()) {
            std::cout << "top_k=" << result.top_k << ", recall=" << result.recall
                      << ", search_parameters=" << result.search_parameters << '\n';
        }

        auto query = vsag::Dataset::Make()
                         ->NumElements(1)
                         ->Dim(dim)
                         ->Float32Vectors(calibration_vectors.data())
                         ->Owner(false);
        auto result = index->KnnSearch(query, 5, 0.9F).value();
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Recall search example failed: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "Recall search example failed with an unknown error\n";
        return 1;
    }
}
