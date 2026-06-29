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

#include "autotune.h"

#include <H5Cpp.h>

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "autotune_index_policy.h"

using namespace nlohmann::literals;

namespace {

std::string
MakeTempFile(const std::string& name) {
    auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path);
    out << "placeholder";
    return path.string();
}

float
L2(const std::vector<float>& train,
   const std::vector<float>& test,
   int64_t train_id,
   int64_t query_id,
   int64_t dim) {
    float sum = 0.0F;
    for (int64_t i = 0; i < dim; ++i) {
        const float diff = train[train_id * dim + i] - test[query_id * dim + i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void
WriteDenseEvalDataset(const std::string& path) {
    constexpr int64_t base_count = 64;
    constexpr int64_t query_count = 8;
    constexpr int64_t dim = 8;
    constexpr int64_t gt_k = 10;

    std::vector<float> train(base_count * dim);
    std::vector<float> test(query_count * dim);
    for (int64_t i = 0; i < base_count; ++i) {
        for (int64_t j = 0; j < dim; ++j) {
            train[i * dim + j] = static_cast<float>((i * 17 + j * 13) % 101) / 101.0F + 0.001F * i;
        }
    }
    for (int64_t i = 0; i < query_count; ++i) {
        const int64_t source = (i * 7) % base_count;
        for (int64_t j = 0; j < dim; ++j) {
            test[i * dim + j] = train[source * dim + j] + 0.0001F * static_cast<float>(j);
        }
    }

    std::vector<int64_t> neighbors(query_count * gt_k);
    std::vector<float> distances(query_count * gt_k);
    for (int64_t query_id = 0; query_id < query_count; ++query_id) {
        std::vector<std::pair<float, int64_t>> ranked;
        ranked.reserve(base_count);
        for (int64_t base_id = 0; base_id < base_count; ++base_id) {
            ranked.emplace_back(L2(train, test, base_id, query_id, dim), base_id);
        }
        std::sort(ranked.begin(), ranked.end());
        for (int64_t k = 0; k < gt_k; ++k) {
            neighbors[query_id * gt_k + k] = ranked[k].second;
            distances[query_id * gt_k + k] = ranked[k].first;
        }
    }

    std::remove(path.c_str());
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    {
        auto attr = file.createAttribute("distance", str_type, H5::DataSpace(H5S_SCALAR));
        std::string value = "euclidean";
        attr.write(str_type, value);
    }
    {
        hsize_t dims[2] = {base_count, dim};
        H5::DataSpace space(2, dims);
        auto dataset = file.createDataSet("/train", H5::PredType::NATIVE_FLOAT, space);
        dataset.write(train.data(), H5::PredType::NATIVE_FLOAT);
    }
    {
        hsize_t dims[2] = {query_count, dim};
        H5::DataSpace space(2, dims);
        auto dataset = file.createDataSet("/test", H5::PredType::NATIVE_FLOAT, space);
        dataset.write(test.data(), H5::PredType::NATIVE_FLOAT);
    }
    {
        hsize_t dims[2] = {query_count, gt_k};
        H5::DataSpace space(2, dims);
        auto dataset = file.createDataSet("/neighbors", H5::PredType::NATIVE_INT64, space);
        dataset.write(neighbors.data(), H5::PredType::NATIVE_INT64);
    }
    {
        hsize_t dims[2] = {query_count, gt_k};
        H5::DataSpace space(2, dims);
        auto dataset = file.createDataSet("/distances", H5::PredType::NATIVE_FLOAT, space);
        dataset.write(distances.data(), H5::PredType::NATIVE_FLOAT);
    }
}

bool
HasTunableParam(const vsag::autotune::JsonType& policy,
                const std::string& path,
                const std::string& scope) {
    for (const auto& param : policy["tunable_params"]) {
        if (param["path"] == path && param["scope"] == scope) {
            return true;
        }
    }
    return false;
}

bool
HasFixedDefault(const vsag::autotune::JsonType& policy,
                const std::string& path,
                const vsag::autotune::JsonType& value) {
    for (const auto& param : policy["fixed_defaults"]) {
        if (param["path"] == path && param["value"] == value) {
            return true;
        }
    }
    return false;
}

}  // namespace

TEST_CASE("AutoTune expands arrays, ranges and value escapes") {
    auto expanded = vsag::autotune::ExpandJsonForTest(R"({
        "a": [1, 2],
        "b": {"$range": {"start": 10, "stop": 20, "step": 10}},
        "c": {"$value": [3, 4]}
    })"_json);

    REQUIRE(expanded.size() == 4);
    REQUIRE(expanded[0]["c"].is_array());
    REQUIRE(expanded[0]["c"].size() == 2);
}

TEST_CASE("AutoTune index policies describe tunable parameter spaces") {
    const auto hgraph_policy = vsag::autotune::internal::DescribeIndexTunePolicy("hgraph");
    REQUIRE(hgraph_policy["name"] == "hgraph");
    REQUIRE(hgraph_policy["tunable_params"].size() == 4);
    REQUIRE(hgraph_policy["fixed_defaults"].empty());
    REQUIRE(HasTunableParam(
        hgraph_policy, "/create_params/index_param/base_quantization_type", "build"));
    REQUIRE(HasTunableParam(hgraph_policy, "/create_params/index_param/max_degree", "build"));
    REQUIRE(HasTunableParam(hgraph_policy, "/create_params/index_param/ef_construction", "build"));
    REQUIRE(HasTunableParam(hgraph_policy, "/search_params/hgraph/ef_search", "search"));

    const auto ivf_policy = vsag::autotune::internal::DescribeIndexTunePolicy("ivf");
    REQUIRE(ivf_policy["name"] == "ivf");
    REQUIRE(ivf_policy["tunable_params"].size() == 3);
    REQUIRE(ivf_policy["fixed_defaults"].size() == 2);
    REQUIRE(
        HasTunableParam(ivf_policy, "/create_params/index_param/base_quantization_type", "build"));
    REQUIRE(HasTunableParam(ivf_policy, "/create_params/index_param/buckets_count", "build"));
    REQUIRE(HasTunableParam(ivf_policy, "/search_params/ivf/scan_buckets_count", "search"));
    REQUIRE(
        HasFixedDefault(ivf_policy, "/create_params/index_param/partition_strategy_type", "ivf"));
    REQUIRE(HasFixedDefault(ivf_policy, "/create_params/index_param/ivf_train_type", "kmeans"));
}

TEST_CASE("AutoTune generates covered hgraph and ivf trials") {
    auto data_path = MakeTempFile("vsag_autotune_candidate_test.hdf5");
    auto request = R"({
        "version": 1,
        "data_path": "",
        "indexes": [
            {
                "name": "hgraph",
                "create_params": {
                    "dim": 128,
                    "dtype": "float32",
                    "metric_type": "l2",
                    "index_param": {
                        "base_quantization_type": ["fp32", "sq8_uniform"],
                        "max_degree": [16, 32],
                        "ef_construction": 100
                    }
                },
                "search_params": {
                    "hgraph": {
                        "ef_search": [40, 80, 120]
                    }
                }
            },
            {
                "name": "ivf",
                "create_params": {
                    "dim": 128,
                    "dtype": "float32",
                    "metric_type": "l2",
                    "index_param": {
                        "partition_strategy_type": "ivf",
                        "base_quantization_type": ["fp32", "sq8_uniform"],
                        "buckets_count": [512, 1024],
                        "ivf_train_type": "kmeans"
                    }
                },
                "search_params": {
                    "ivf": {
                        "scan_buckets_count": [16, 32, 64]
                    }
                }
            }
        ],
        "constraints": {
            "recall_at_k": 0.5
        },
        "execution": {
            "workspace_path": "/tmp/vsag_autotune_candidate_test",
            "max_trials": 24
        }
    })"_json;
    request["data_path"] = data_path;

    auto candidates = vsag::autotune::GenerateCandidatesForTest(request);
    REQUIRE(candidates["candidate_count"] == 24);
    REQUIRE(candidates["trial_count"] == 24);
    REQUIRE(candidates["trials"][0]["eval_type"] == "build,search");

    std::remove(data_path.c_str());
}

TEST_CASE("AutoTune applies index policy defaults for hgraph and ivf") {
    auto data_path = MakeTempFile("vsag_autotune_policy_defaults_test.hdf5");
    auto request = R"({
        "version": 1,
        "data_path": "",
        "indexes": [
            {
                "name": "hgraph",
                "create_params": {
                    "dim": 128,
                    "dtype": "float32",
                    "metric_type": "l2"
                }
            },
            {
                "name": "ivf",
                "create_params": {
                    "dim": 128,
                    "dtype": "float32",
                    "metric_type": "l2"
                }
            }
        ],
        "constraints": {
            "recall_at_k": 0.5
        }
    })"_json;
    request["data_path"] = data_path;

    auto candidates = vsag::autotune::GenerateCandidatesForTest(request);
    REQUIRE(candidates["candidate_count"] == 36);
    REQUIRE(candidates["trial_count"] == 36);

    uint64_t hgraph_count = 0;
    uint64_t ivf_count = 0;
    for (const auto& trial : candidates["trials"]) {
        if (trial["index_name"] == "hgraph") {
            ++hgraph_count;
        }
        if (trial["index_name"] == "ivf") {
            ++ivf_count;
        }
    }
    REQUIRE(hgraph_count == 24);
    REQUIRE(ivf_count == 12);

    const auto& first_hgraph = candidates["trials"][0];
    REQUIRE(first_hgraph["index_name"] == "hgraph");
    REQUIRE(first_hgraph["create_params"]["index_param"].contains("base_quantization_type"));
    REQUIRE(first_hgraph["create_params"]["index_param"].contains("max_degree"));
    REQUIRE(first_hgraph["create_params"]["index_param"].contains("ef_construction"));
    REQUIRE(first_hgraph["search_params"]["hgraph"].contains("ef_search"));

    const auto& first_ivf = candidates["trials"][24];
    REQUIRE(first_ivf["index_name"] == "ivf");
    REQUIRE(first_ivf["create_params"]["index_param"]["partition_strategy_type"] == "ivf");
    REQUIRE(first_ivf["create_params"]["index_param"].contains("base_quantization_type"));
    REQUIRE(first_ivf["create_params"]["index_param"].contains("buckets_count"));
    REQUIRE(first_ivf["create_params"]["index_param"]["ivf_train_type"] == "kmeans");
    REQUIRE(first_ivf["search_params"]["ivf"].contains("scan_buckets_count"));

    std::remove(data_path.c_str());
}

TEST_CASE("AutoTune plans search-only trials for an existing index") {
    auto data_path = MakeTempFile("vsag_autotune_search_only_test.hdf5");
    auto index_path = MakeTempFile("vsag_autotune_existing.index");
    auto request = R"({
        "version": 1,
        "data_path": "",
        "index_path": "",
        "indexes": [
            {
                "name": "hgraph",
                "create_params": {
                    "dim": 128,
                    "dtype": "float32",
                    "metric_type": "l2",
                    "index_param": {
                        "base_quantization_type": "fp32",
                        "max_degree": 16,
                        "ef_construction": 100
                    }
                },
                "search_params": {
                    "hgraph": {
                        "ef_search": [40, 80]
                    }
                }
            }
        ],
        "constraints": {
            "recall_at_k": 0.5
        }
    })"_json;
    request["data_path"] = data_path;
    request["index_path"] = index_path;

    auto candidates = vsag::autotune::GenerateCandidatesForTest(request);
    REQUIRE(candidates["candidate_count"] == 2);
    REQUIRE(candidates["trial_count"] == 2);
    REQUIRE(candidates["trials"][0]["eval_type"] == "search");
    REQUIRE(candidates["trials"][0]["index_path"] == index_path);
    REQUIRE(candidates["trials"][1]["eval_type"] == "search");
    REQUIRE(candidates["trials"][1]["index_path"] == index_path);

    std::remove(data_path.c_str());
    std::remove(index_path.c_str());
}

TEST_CASE("AutoTune returns structured failure result") {
    auto result_path =
        (std::filesystem::temp_directory_path() / "vsag_autotune_failed_result.json").string();
    std::remove(result_path.c_str());
    auto request = R"({
        "version": 1,
        "data_path": "/tmp/vsag_autotune_missing_dataset.hdf5",
        "indexes": [
            {
                "name": "hgraph",
                "create_params": {
                    "dim": 8,
                    "dtype": "float32",
                    "metric_type": "l2"
                }
            }
        ],
        "constraints": {
            "recall_at_k": 0.5
        },
        "output": {
            "result_path": ""
        }
    })"_json;
    request["output"]["result_path"] = result_path;

    auto result = vsag::autotune::RunAutoTune(request);
    REQUIRE(result["status"] == "failed");
    REQUIRE(result["trial_count"] == 0);
    REQUIRE(result["failure"]["message"].get<std::string>().find("data_path does not exist") !=
            std::string::npos);
    REQUIRE(std::filesystem::exists(result_path));
    std::remove(result_path.c_str());
}

TEST_CASE("AutoTune rejects unsupported P0 contract fields") {
    auto data_path = MakeTempFile("vsag_autotune_contract_test.hdf5");
    auto request = R"({
        "version": 1,
        "data_path": "",
        "indexes": [
            {
                "name": "hgraph",
                "create_params": {
                    "dim": 8,
                    "dtype": "float32",
                    "metric_type": "l2"
                }
            }
        ],
        "constraints": {
            "unknown_metric": 1.0
        }
    })"_json;
    request["data_path"] = data_path;

    auto result = vsag::autotune::RunAutoTune(request);
    REQUIRE(result["status"] == "failed");
    REQUIRE(result["failure"]["message"].get<std::string>().find("unsupported constraint") !=
            std::string::npos);

    request["constraints"] = {{"recall_at_k", 0.5}};
    request["execution"] = {{"search_mode", "range"}};
    result = vsag::autotune::RunAutoTune(request);
    REQUIRE(result["status"] == "failed");
    REQUIRE(result["failure"]["message"].get<std::string>().find("unsupported in AutoTune P0") !=
            std::string::npos);

    request["execution"] = {{"search_mode", "knn"}};
    request["indexes"][0]["name"] = "unsupported_index";
    result = vsag::autotune::RunAutoTune(request);
    REQUIRE(result["status"] == "failed");
    REQUIRE(result["failure"]["message"].get<std::string>().find("unsupported index") !=
            std::string::npos);

    std::remove(data_path.c_str());
}

TEST_CASE("AutoTune runs real eval build-search integration for hgraph and ivf") {
    const auto dataset_path =
        (std::filesystem::temp_directory_path() / "vsag_autotune_dense_eval_test.hdf5").string();
    const auto workspace_path =
        (std::filesystem::temp_directory_path() / "vsag_autotune_dense_eval_workspace").string();
    const auto result_path = (std::filesystem::path(workspace_path) / "result.json").string();
    std::filesystem::remove_all(workspace_path);
    WriteDenseEvalDataset(dataset_path);

    auto request = R"({
        "version": 1,
        "data_path": "",
        "indexes": [
            {
                "name": "hgraph",
                "create_params": {
                    "dim": 8,
                    "dtype": "float32",
                    "metric_type": "l2",
                    "index_param": {
                        "base_quantization_type": ["fp32", "sq8_uniform"],
                        "max_degree": [8, 16],
                        "ef_construction": 40
                    }
                },
                "search_params": {
                    "hgraph": {
                        "ef_search": [10, 20, 30]
                    }
                }
            },
            {
                "name": "ivf",
                "create_params": {
                    "dim": 8,
                    "dtype": "float32",
                    "metric_type": "l2",
                    "index_param": {
                        "partition_strategy_type": "ivf",
                        "base_quantization_type": ["fp32", "sq8_uniform"],
                        "buckets_count": [4, 8],
                        "ivf_train_type": "kmeans"
                    }
                },
                "search_params": {
                    "ivf": {
                        "scan_buckets_count": [1, 2, 4]
                    }
                }
            }
        ],
        "constraints": {
            "recall_at_k": 0.0,
            "latency_avg_ms": 1000.0,
            "memory_peak_mb": 65536.0
        },
        "execution": {
            "top_k": 3,
            "search_mode": "knn",
            "search_query_count": 0,
            "num_threads_building": 2,
            "num_threads_searching": 2,
            "workspace_path": "",
            "keep_intermediate": false,
            "max_trials": 24
        },
        "output": {
            "result_path": "",
            "include_trials": true
        }
    })"_json;
    request["data_path"] = dataset_path;
    request["execution"]["workspace_path"] = workspace_path;
    request["output"]["result_path"] = result_path;

    auto result = vsag::autotune::RunAutoTune(request);
    REQUIRE(result["status"] == "success");
    REQUIRE(result["trial_count"] == 24);
    REQUIRE(result["trials"].size() == 24);
    REQUIRE(result["recommendation"].is_object());
    REQUIRE(std::filesystem::exists(result_path));

    uint64_t hgraph_count = 0;
    uint64_t ivf_count = 0;
    uint64_t failed_count = 0;
    for (const auto& trial : result["trials"]) {
        if (trial["index_name"] == "hgraph") {
            ++hgraph_count;
        }
        if (trial["index_name"] == "ivf") {
            ++ivf_count;
        }
        if (trial["status"] != "success") {
            ++failed_count;
        }
        REQUIRE(trial["metrics"].contains("recall_at_k"));
        REQUIRE(trial["metrics"].contains("latency_avg_ms"));
        REQUIRE(trial["metrics"].contains("memory_peak_mb"));
    }
    REQUIRE(hgraph_count == 12);
    REQUIRE(ivf_count == 12);
    REQUIRE(failed_count == 0);

    std::filesystem::remove_all(workspace_path);
    std::remove(dataset_path.c_str());
}
