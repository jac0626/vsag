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
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "autotune_candidate.h"
#include "autotune_candidate_generator.h"
#include "autotune_internal.h"
#include "autotune_planner.h"
#include "autotune_request.h"
#include "eval_dataset.h"
#include "vsag/constants.h"
#include "vsag/options.h"

using namespace nlohmann::literals;

namespace {

using vsag::autotune::JsonType;
using vsag::autotune::internal::AutoTuneRequest;
using vsag::autotune::internal::BuildCandidateContext;
using vsag::autotune::internal::CandidateGenerator;
using vsag::autotune::internal::CandidateSpec;
using vsag::autotune::internal::DatasetDescription;
using vsag::autotune::internal::EvaluationResult;
using vsag::autotune::internal::ObjectiveSpec;
using vsag::autotune::internal::SearchCandidateContext;

class StaticCandidateGenerator final : public CandidateGenerator {
public:
    StaticCandidateGenerator(std::vector<JsonType> build_patches,
                             std::vector<JsonType> search_patches)
        : build_patches_(std::move(build_patches)), search_patches_(std::move(search_patches)) {
    }

    std::vector<JsonType>
    GenerateBuildPatches(const BuildCandidateContext&) const override {
        return build_patches_;
    }

    std::vector<JsonType>
    GenerateSearchPatches(const SearchCandidateContext&) const override {
        return search_patches_;
    }

private:
    std::vector<JsonType> build_patches_;
    std::vector<JsonType> search_patches_;
};
class ConditionalIvfSearchCandidateGenerator final : public CandidateGenerator {
public:
    std::vector<JsonType>
    GenerateBuildPatches(const BuildCandidateContext&) const override {
        return {};
    }

    std::vector<JsonType>
    GenerateSearchPatches(const SearchCandidateContext& context) const override {
        const auto& ivf = context.user_search_params.at("ivf");
        if (!ivf.at("enable_reorder").is_boolean()) {
            throw std::runtime_error("conditional generator expected a concrete enable_reorder");
        }
        if (ivf.at("enable_reorder").get<bool>()) {
            return {JsonType{{"ivf", {{"factor", 2.0}}}}};
        }
        return {};
    }
};

std::string
UniqueTempPath(const std::string& stem, const std::string& extension = "") {
    static std::atomic<uint64_t> ordinal{0};
    const auto timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return (std::filesystem::temp_directory_path() /
            (stem + "-" + std::to_string(timestamp) + "-" + std::to_string(ordinal.fetch_add(1)) +
             extension))
        .string();
}

class ScopedPath {
public:
    explicit ScopedPath(std::string path) : path_(std::move(path)) {
    }

    ~ScopedPath() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    ScopedPath(const ScopedPath&) = delete;
    ScopedPath&
    operator=(const ScopedPath&) = delete;

    const std::string&
    Get() const {
        return path_;
    }

private:
    std::string path_;
};

class ScopedBlockSizeLimit {
public:
    explicit ScopedBlockSizeLimit(uint64_t value)
        : original_(vsag::Options::Instance().block_size_limit()) {
        vsag::Options::Instance().set_block_size_limit(value);
    }

    ~ScopedBlockSizeLimit() {
        vsag::Options::Instance().set_block_size_limit(original_);
    }

    ScopedBlockSizeLimit(const ScopedBlockSizeLimit&) = delete;
    ScopedBlockSizeLimit&
    operator=(const ScopedBlockSizeLimit&) = delete;

private:
    uint64_t original_;
};

void
WriteTextFile(const std::string& path, const std::string& contents = "placeholder") {
    std::ofstream output(path, std::ios::binary);
    REQUIRE(output.good());
    output << contents;
    REQUIRE(output.good());
}

JsonType
ReadJsonFile(const std::string& path) {
    std::ifstream input(path);
    REQUIRE(input.good());
    JsonType result;
    input >> result;
    REQUIRE((input.good() || input.eof()));
    return result;
}

void
RequireExactKeys(const JsonType& object, std::initializer_list<const char*> expected_keys) {
    INFO(object.dump(2));
    REQUIRE(object.is_object());
    REQUIRE(object.size() == expected_keys.size());
    for (const auto* key : expected_keys) {
        REQUIRE(object.contains(key));
    }
}

void
RequireTextBefore(const std::string& text, const std::string& left, const std::string& right) {
    INFO(text);
    const auto left_position = text.find(left);
    const auto right_position = text.find(right);
    REQUIRE(left_position != std::string::npos);
    REQUIRE(right_position != std::string::npos);
    REQUIRE(left_position < right_position);
}

float
L2(const std::vector<float>& base,
   const std::vector<float>& query,
   int64_t base_id,
   int64_t query_id,
   int64_t dim) {
    float sum = 0.0F;
    for (int64_t i = 0; i < dim; ++i) {
        const float difference = base[base_id * dim + i] - query[query_id * dim + i];
        sum += difference * difference;
    }
    return std::sqrt(sum);
}

void
WriteDenseEvalDataset(const std::string& path) {
    constexpr int64_t kBaseCount = 96;
    constexpr int64_t kQueryCount = 8;
    constexpr int64_t kDim = 8;
    constexpr int64_t kGroundTruthK = 10;

    std::vector<float> train(kBaseCount * kDim);
    std::vector<float> test(kQueryCount * kDim);
    for (int64_t i = 0; i < kBaseCount; ++i) {
        for (int64_t j = 0; j < kDim; ++j) {
            train[i * kDim + j] = static_cast<float>((i * 17 + j * 13) % 101) / 101.0F + 0.001F * i;
        }
    }
    for (int64_t i = 0; i < kQueryCount; ++i) {
        const int64_t source = (i * 11) % kBaseCount;
        for (int64_t j = 0; j < kDim; ++j) {
            test[i * kDim + j] = train[source * kDim + j] + 0.0001F * static_cast<float>(j);
        }
    }

    std::vector<int64_t> neighbors(kQueryCount * kGroundTruthK);
    std::vector<float> distances(kQueryCount * kGroundTruthK);
    for (int64_t query_id = 0; query_id < kQueryCount; ++query_id) {
        std::vector<std::pair<float, int64_t>> ranked;
        ranked.reserve(kBaseCount);
        for (int64_t base_id = 0; base_id < kBaseCount; ++base_id) {
            ranked.emplace_back(L2(train, test, base_id, query_id, kDim), base_id);
        }
        std::sort(ranked.begin(), ranked.end());
        for (int64_t k = 0; k < kGroundTruthK; ++k) {
            neighbors[query_id * kGroundTruthK + k] = ranked[k].second;
            distances[query_id * kGroundTruthK + k] = ranked[k].first;
        }
    }

    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    {
        auto attribute = file.createAttribute("distance", string_type, H5::DataSpace(H5S_SCALAR));
        std::string value = "euclidean";
        attribute.write(string_type, value);
    }
    {
        hsize_t dimensions[2] = {kBaseCount, kDim};
        H5::DataSpace space(2, dimensions);
        auto dataset = file.createDataSet("/train", H5::PredType::NATIVE_FLOAT, space);
        dataset.write(train.data(), H5::PredType::NATIVE_FLOAT);
    }
    {
        hsize_t dimensions[2] = {kQueryCount, kDim};
        H5::DataSpace space(2, dimensions);
        auto dataset = file.createDataSet("/test", H5::PredType::NATIVE_FLOAT, space);
        dataset.write(test.data(), H5::PredType::NATIVE_FLOAT);
    }
    {
        hsize_t dimensions[2] = {kQueryCount, kGroundTruthK};
        H5::DataSpace space(2, dimensions);
        auto dataset = file.createDataSet("/neighbors", H5::PredType::NATIVE_INT64, space);
        dataset.write(neighbors.data(), H5::PredType::NATIVE_INT64);
    }
    {
        hsize_t dimensions[2] = {kQueryCount, kGroundTruthK};
        H5::DataSpace space(2, dimensions);
        auto dataset = file.createDataSet("/distances", H5::PredType::NATIVE_FLOAT, space);
        dataset.write(distances.data(), H5::PredType::NATIVE_FLOAT);
    }
}

JsonType
MakeWorkload(uint64_t top_k, uint64_t concurrency) {
    return JsonType{{"top_k", top_k}, {"concurrency", concurrency}};
}

JsonType
MakeValidRequest(const std::string& data_path, const std::string& index_name = "hgraph") {
    return JsonType{{"version", 1},
                    {"data_path", data_path},
                    {"indexes",
                     JsonType::array({JsonType{{"name", index_name},
                                               {"create_params", JsonType::object()},
                                               {"search_params", JsonType::object()}}})},
                    {"workload", MakeWorkload(10, 1)},
                    {"constraints", {{"recall_at_k", 0.0}, {"index_size_mb", 1024.0}}},
                    {"objective", {{"metric", "latency_avg_ms"}}},
                    {"tuning_config",
                     {{"workspace_path", UniqueTempPath("vsag-autotune-unit-workspace")},
                      {"keep_intermediate", false},
                      {"max_trials", 1000}}},
                    {"output", JsonType::object()}};
}

DatasetDescription
DenseDatasetDescription(uint64_t ground_truth_k = 100, uint64_t base_count = 100000) {
    return DatasetDescription{
        128, "float32", "l2", base_count, 1000, ground_truth_k, "dense_vectors"};
}

template <typename Callable>
void
RequireThrowsContaining(Callable&& callable, const std::string& expected) {
    bool threw = false;
    try {
        callable();
    } catch (const std::exception& error) {
        threw = true;
        INFO(error.what());
        REQUIRE(std::string(error.what()).find(expected) != std::string::npos);
    }
    REQUIRE(threw);
}

AutoTuneRequest
ParseAndResolve(JsonType request, const DatasetDescription& dataset = DenseDatasetDescription()) {
    auto parsed = vsag::autotune::internal::ParseAutoTuneRequest(request);
    vsag::autotune::internal::ResolveAutoTuneRequest(parsed, dataset);
    return parsed;
}

AutoTuneRequest
SelectionRequest() {
    AutoTuneRequest request;
    request.workload.top_k = 10;
    request.workload.concurrency = 1;
    request.constraints = {{"build_seconds", 10.0},
                           {"index_size_mb", 100.0},
                           {"recall_at_k", 0.9},
                           {"latency_avg_ms", 10.0}};
    request.objective = ObjectiveSpec{"latency_avg_ms"};
    return request;
}

JsonType
MakeBuildResult(const std::string& build_id, double build_seconds, double index_size_mb) {
    return JsonType{
        {"build_id", build_id},
        {"status", "success"},
        {"index_name", "hgraph"},
        {"create_params", {{"build", build_id}}},
        {"metrics", {{"build_seconds", build_seconds}, {"index_size_mb", index_size_mb}}},
        {"artifacts", {{"index_path", "/tmp/" + build_id + ".index"}}}};
}

JsonType
MakeTrialResult(const std::string& trial_id,
                const std::string& build_id,
                double recall,
                double latency,
                double build_seconds = 2.0,
                double index_size_mb = 20.0) {
    return JsonType{{"trial_id", trial_id},
                    {"build_id", build_id},
                    {"status", "success"},
                    {"search_params", {{"hgraph", {{"ef_search", latency * 10.0}}}}},
                    {"metrics",
                     {{"build_seconds", build_seconds},
                      {"index_size_mb", index_size_mb},
                      {"recall_at_k", recall},
                      {"latency_avg_ms", latency}}}};
}

const JsonType&
FindBuild(const JsonType& result, const std::string& index_name) {
    for (const auto& build : result["builds"]) {
        if (build.value("index_name", "") == index_name) {
            return build;
        }
    }
    FAIL("build not found: " << index_name);
    return result;
}

}  // namespace

TEST_CASE("AutoTune V1 request parser enforces the public contract") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-request", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get());
    const auto parsed = vsag::autotune::internal::ParseAutoTuneRequest(request);
    REQUIRE(parsed.version == 1);
    REQUIRE(parsed.indexes.size() == 1);
    REQUIRE(parsed.workload.top_k == 10);
    REQUIRE(parsed.workload.concurrency == 1);
    REQUIRE(parsed.objective.metric == "latency_avg_ms");
    REQUIRE_FALSE(parsed.output.include_raw_eval);
    REQUIRE_FALSE(parsed.effective_request["workload"].contains("build"));
    REQUIRE(parsed.effective_request["output"]["include_raw_eval"] == false);

    SECTION("unknown top-level and workload fields are rejected") {
        auto invalid = request;
        invalid["mode"] = "knn";
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.mode is unsupported");

        invalid = request;
        invalid["workload"]["build"] = {{"threads", 2}};
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.workload.build is unsupported");

        invalid = request;
        invalid["workload"]["mode"] = "range";
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.workload.mode is unsupported");

        invalid = request;
        invalid["workload"]["query_count"] = 10;
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.workload.query_count is unsupported");
    }

    SECTION("index names and search parameter objects are canonicalized") {
        auto uppercase = request;
        uppercase["indexes"][0]["name"] = "HGRAPH";
        uppercase["indexes"][0]["search_params"] = JsonType{{"HGRAPH", {{"ef_search", 40}}}};
        const auto uppercase_parsed = vsag::autotune::internal::ParseAutoTuneRequest(uppercase);
        REQUIRE(uppercase_parsed.indexes[0].name == "hgraph");
        REQUIRE(uppercase_parsed.indexes[0].search_params ==
                JsonType{{"hgraph", {{"ef_search", 40}}}});

        uppercase["indexes"][0]["search_params"]["hgraph"] = {{"ef_search", 80}};
        RequireThrowsContaining(
            [&uppercase]() { vsag::autotune::internal::ParseAutoTuneRequest(uppercase); },
            "contains duplicate index parameter objects");
    }

    SECTION("top_k is resource bounded") {
        auto invalid = request;
        invalid["workload"]["top_k"] = 1000001;
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "top_k exceeds the V1 limit");

        invalid = request;
        invalid["workload"]["top_k"] = 100000;
        invalid["workload"]["concurrency"] = 48;
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "top_k * concurrency exceeds the V1 result-buffer limit");
    }

    SECTION("objective direction is owned by the metric registry") {
        auto invalid = request;
        invalid["objective"]["direction"] = "min";
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.objective.direction is unsupported");
    }

    SECTION("V1 accepts exactly one scenario with top-level constraints and objective") {
        auto invalid = request;
        invalid["workload"] =
            JsonType{{"searches", JsonType::array({MakeWorkload(10, 1), MakeWorkload(100, 1)})}};
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.workload.searches is unsupported");

        invalid = request;
        invalid["constraints"] = JsonType::object();
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.constraints must not be empty");

        invalid = request;
        invalid["objective"]["workload"] = "top10";
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.objective.workload is unsupported");
    }

    SECTION("removed V1 metrics are rejected") {
        auto invalid = request;
        invalid["constraints"] = {{"memory_peak_mb", 1024.0}};
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "unsupported metric: memory_peak_mb");
    }

    SECTION("dataset and existing index paths must be regular files") {
        ScopedPath directory_path(UniqueTempPath("vsag-autotune-directory"));
        REQUIRE(std::filesystem::create_directories(directory_path.Get()));

        auto invalid = request;
        invalid["data_path"] = directory_path.Get();
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "data_path must name a readable regular file");

        invalid = request;
        invalid["index_path"] = directory_path.Get();
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "index_path must name a readable regular file");
    }

    SECTION("raw eval diagnostics are explicit and type checked") {
        auto enabled = request;
        enabled["output"]["include_raw_eval"] = true;
        const auto enabled_request = vsag::autotune::internal::ParseAutoTuneRequest(enabled);
        REQUIRE(enabled_request.output.include_raw_eval);
        REQUIRE(enabled_request.effective_request["output"]["include_raw_eval"] == true);

        auto invalid = request;
        invalid["output"]["include_raw_eval"] = "true";
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.output.include_raw_eval must be a boolean");

        invalid = request;
        invalid["output"]["include_trials"] = true;
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "request.output.include_trials is unsupported");
    }
}

TEST_CASE("AutoTune resolves dataset metadata and validates user-provided metadata") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-metadata", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get());
    auto parsed = vsag::autotune::internal::ParseAutoTuneRequest(request);
    vsag::autotune::internal::ResolveAutoTuneRequest(parsed, DenseDatasetDescription(100));

    REQUIRE(parsed.dataset_resolved);
    REQUIRE(parsed.indexes[0].create_params["dim"] == 128);
    REQUIRE(parsed.indexes[0].create_params["dtype"] == "float32");
    REQUIRE(parsed.indexes[0].create_params["metric_type"] == "l2");
    REQUIRE_FALSE(parsed.indexes[0].create_params.contains("repr"));
    REQUIRE_FALSE(parsed.indexes[0].create_params.contains("index_param"));
    REQUIRE(parsed.effective_request["dataset_description"]["query_count"] == 1000);
    REQUIRE(parsed.effective_request["dataset_description"]["ground_truth_k"] == 100);

    SECTION("matching explicit metadata remains valid") {
        auto explicit_request = request;
        explicit_request["indexes"][0]["create_params"] =
            JsonType{{"dim", 128}, {"dtype", "float32"}, {"metric_type", "l2"}, {"repr", "dense"}};
        auto explicit_parsed = vsag::autotune::internal::ParseAutoTuneRequest(explicit_request);
        REQUIRE_NOTHROW(vsag::autotune::internal::ResolveAutoTuneRequest(
            explicit_parsed, DenseDatasetDescription(100)));
    }

    SECTION("native create parameters pass through candidate generation") {
        auto with_extra_info = request;
        with_extra_info["indexes"][0]["create_params"]["extra_info_size"] = 8;
        auto parsed = ParseAndResolve(with_extra_info, DenseDatasetDescription(100));
        const auto generation = vsag::autotune::internal::GenerateCandidates(parsed);
        REQUIRE_FALSE(generation.empty());
        for (const auto& candidate : generation) {
            REQUIRE(candidate.create_params["extra_info_size"] == 8);
        }
    }

    SECTION("record representation must match the dense dataset") {
        for (const auto* invalid_repr : {"sparse", "multi_vector"}) {
            auto mismatch = request;
            mismatch["indexes"][0]["create_params"]["repr"] = invalid_repr;
            RequireThrowsContaining(
                [&mismatch]() { ParseAndResolve(mismatch, DenseDatasetDescription(100)); },
                "create_params.repr must match the dataset");
        }

        auto candidate_axis = request;
        candidate_axis["indexes"][0]["create_params"]["repr"] =
            JsonType::array({"dense", "sparse"});
        RequireThrowsContaining(
            [&candidate_axis]() { ParseAndResolve(candidate_axis, DenseDatasetDescription(100)); },
            "create_params.repr must match the dataset");
    }

    SECTION("metadata mismatch fails before candidate generation") {
        auto mismatch = request;
        mismatch["indexes"][0]["create_params"]["dim"] = 64;
        auto mismatch_parsed = vsag::autotune::internal::ParseAutoTuneRequest(mismatch);
        RequireThrowsContaining(
            [&mismatch_parsed]() {
                vsag::autotune::internal::ResolveAutoTuneRequest(mismatch_parsed,
                                                                 DenseDatasetDescription(100));
            },
            "create_params.dim must match the dataset");
    }

    SECTION("native build thread parameters remain ordinary candidate axes") {
        auto hgraph = request;
        hgraph["indexes"][0]["create_params"]["index_param"]["build_thread_count"] =
            JsonType::array({1, 2});
        const auto hgraph_candidates =
            vsag::autotune::internal::GenerateCandidates(ParseAndResolve(hgraph));
        std::set<int64_t> hgraph_threads;
        for (const auto& candidate : hgraph_candidates) {
            hgraph_threads.emplace(
                candidate.create_params["index_param"]["build_thread_count"].get<int64_t>());
        }
        REQUIRE(hgraph_threads == std::set<int64_t>{1, 2});

        auto ivf = MakeValidRequest(data_path.Get(), "ivf");
        ivf["indexes"][0]["create_params"]["index_param"]["thread_count"] = JsonType::array({3, 4});
        const auto ivf_candidates =
            vsag::autotune::internal::GenerateCandidates(ParseAndResolve(ivf));
        std::set<int64_t> ivf_threads;
        for (const auto& candidate : ivf_candidates) {
            ivf_threads.emplace(
                candidate.create_params["index_param"]["thread_count"].get<int64_t>());
        }
        REQUIRE(ivf_threads == std::set<int64_t>{3, 4});
    }

    SECTION("ground truth must cover every requested top k") {
        auto narrow = vsag::autotune::internal::ParseAutoTuneRequest(request);
        RequireThrowsContaining(
            [&narrow]() {
                vsag::autotune::internal::ResolveAutoTuneRequest(narrow,
                                                                 DenseDatasetDescription(5));
            },
            "top_k exceeds dataset ground_truth_k");
    }

    SECTION("V1 rejects dense int8 before candidate generation") {
        auto int8_dataset = DenseDatasetDescription();
        int8_dataset.dtype = vsag::DATATYPE_INT8;
        RequireThrowsContaining([&]() { ParseAndResolve(request, int8_dataset); },
                                "supports only float32 dense datasets");
    }
}

TEST_CASE("AutoTune expands HGraph defaults for the request workload") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-hgraph-candidates", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get());
    request["indexes"][0]["create_params"] = JsonType{
        {"index_param",
         {{"base_quantization_type", "fp32"}, {"max_degree", 16}, {"ef_construction", 100}}}};
    request["workload"] = MakeWorkload(100, 4);

    auto parsed = ParseAndResolve(request, DenseDatasetDescription(100));
    const auto candidates = vsag::autotune::internal::GenerateCandidates(parsed);
    REQUIRE(candidates.size() == 3);

    std::set<int64_t> values;
    for (const auto& candidate : candidates) {
        values.emplace(candidate.search_params["hgraph"]["ef_search"].get<int64_t>());
        REQUIRE(candidate.create_params["dim"] == 128);
        REQUIRE_FALSE(candidate.create_params["index_param"].contains("build_thread_count"));
    }
    REQUIRE(values == std::set<int64_t>{100, 200, 400});

    const auto plan = vsag::autotune::internal::PlanTrials(
        parsed, candidates, parsed.tuning_config.workspace_path);
    REQUIRE(plan.builds.size() == 1);
    REQUIRE(plan.trials.size() == 3);
    for (const auto& trial : plan.trials) {
        REQUIRE(trial.build_id == plan.builds[0].build_id);
    }
}

TEST_CASE("AutoTune expands IVF candidates without copying native defaults") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-ivf-candidates", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get(), "ivf");
    auto parsed = ParseAndResolve(request);
    const auto candidates = vsag::autotune::internal::GenerateCandidates(parsed);

    REQUIRE(candidates.size() == 12);
    std::set<std::string> quantizers;
    std::set<int64_t> buckets;
    std::set<int64_t> scanned_buckets;
    for (const auto& candidate : candidates) {
        const auto& index_param = candidate.create_params["index_param"];
        REQUIRE_FALSE(index_param.contains("thread_count"));
        REQUIRE_FALSE(index_param.contains("partition_strategy_type"));
        REQUIRE_FALSE(index_param.contains("ivf_train_type"));
        quantizers.emplace(index_param["base_quantization_type"].get<std::string>());
        buckets.emplace(index_param["buckets_count"].get<int64_t>());
        scanned_buckets.emplace(
            candidate.search_params["ivf"]["scan_buckets_count"].get<int64_t>());
    }
    REQUIRE(quantizers == std::set<std::string>{"fp32", "sq8_uniform"});
    REQUIRE(buckets == std::set<int64_t>{1024, 2048});
    REQUIRE(scanned_buckets == std::set<int64_t>{16, 32, 64});

    auto explicit_partition = request;
    explicit_partition["indexes"][0]["create_params"]["index_param"]["partition_strategy_type"] =
        "gno_imi";
    auto explicit_parsed = ParseAndResolve(explicit_partition);
    const auto explicit_candidates = vsag::autotune::internal::GenerateCandidates(explicit_parsed);
    REQUIRE_FALSE(explicit_candidates.empty());
    for (const auto& candidate : explicit_candidates) {
        REQUIRE(candidate.create_params["index_param"]["partition_strategy_type"] == "gno_imi");
    }
}

TEST_CASE("AutoTune CandidateGenerator cannot overwrite an explicit user parameter") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-candidate-generator-overwrite", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get());
    request["indexes"][0]["create_params"] = JsonType{{"index_param", {{"max_degree", 16}}}};
    const auto parsed = ParseAndResolve(request);
    const StaticCandidateGenerator generator({JsonType{{"index_param", {{"max_degree", 32}}}}},
                                             {JsonType::object()});

    RequireThrowsContaining(
        [&parsed, &generator]() {
            vsag::autotune::internal::GenerateCandidates(parsed, generator);
        },
        "candidate generator must not overwrite user parameter hgraph create_params.index_param."
        "max_degree");
}

TEST_CASE("AutoTune CandidateGenerator preserves correlated build tuples") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-correlated-candidates", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get());
    request["indexes"][0]["create_params"] =
        JsonType{{"index_param", {{"base_quantization_type", "fp32"}}}};
    request["indexes"][0]["search_params"] = JsonType{{"hgraph", {{"ef_search", 40}}}};
    const auto parsed = ParseAndResolve(request);
    const StaticCandidateGenerator generator(
        {JsonType{{"index_param", {{"max_degree", 16}, {"ef_construction", 100}}}},
         JsonType{{"index_param", {{"max_degree", 32}, {"ef_construction", 200}}}}},
        {JsonType::object()});

    const auto generation = vsag::autotune::internal::GenerateCandidates(parsed, generator);
    REQUIRE(generation.size() == 2);

    std::set<std::pair<int64_t, int64_t>> build_tuples;
    for (const auto& candidate : generation) {
        const auto& index_param = candidate.create_params["index_param"];
        build_tuples.emplace(index_param["max_degree"].get<int64_t>(),
                             index_param["ef_construction"].get<int64_t>());
    }
    REQUIRE(build_tuples == std::set<std::pair<int64_t, int64_t>>{{16, 100}, {32, 200}});

    const auto plan = vsag::autotune::internal::PlanTrials(
        parsed, generation, parsed.tuning_config.workspace_path);
    REQUIRE(plan.builds.size() == 2);
    REQUIRE(plan.trials.size() == 2);
}

TEST_CASE("AutoTune CandidateGenerator completes each concrete search tuple independently") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-conditional-search-generator", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get(), "ivf");
    request["indexes"][0]["create_params"] = JsonType{{"index_param",
                                                       {{"base_quantization_type", "fp32"},
                                                        {"buckets_count", 16},
                                                        {"ivf_train_type", "kmeans"},
                                                        {"use_reorder", true}}}};
    request["indexes"][0]["search_params"] = JsonType{
        {"ivf", {{"scan_buckets_count", 8}, {"enable_reorder", JsonType::array({true, false})}}}};
    const auto parsed = ParseAndResolve(request);
    const ConditionalIvfSearchCandidateGenerator generator;

    const auto generation = vsag::autotune::internal::GenerateCandidates(parsed, generator);
    REQUIRE(generation.size() == 2);

    std::set<std::pair<bool, bool>> search_tuples;
    for (const auto& candidate : generation) {
        const auto& ivf = candidate.search_params["ivf"];
        const auto enable_reorder = ivf["enable_reorder"].get<bool>();
        search_tuples.emplace(enable_reorder, ivf.contains("factor"));
        if (enable_reorder) {
            REQUIRE(ivf["factor"] == 2.0);
        }
    }
    REQUIRE(search_tuples == std::set<std::pair<bool, bool>>{{false, false}, {true, true}});
}

TEST_CASE("AutoTune CandidateGenerator preserves native parameters for evaluation") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-generator-native-param", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get());
    request["indexes"][0]["create_params"] =
        JsonType{{"index_param", {{"base_quantization_type", "fp32"}}}};
    request["indexes"][0]["search_params"] = JsonType{{"hgraph", {{"ef_search", 40}}}};
    const auto parsed = ParseAndResolve(request);
    const StaticCandidateGenerator generator(
        {JsonType{{"index_param",
                   {{"max_degree", 16}, {"ef_construction", 100}, {"ef_constuction", 100}}}}},
        {JsonType::object()});

    const auto generation = vsag::autotune::internal::GenerateCandidates(parsed, generator);
    REQUIRE(generation.size() == 1);
    REQUIRE(generation[0].create_params["index_param"]["ef_constuction"] == 100);
}

TEST_CASE("AutoTune candidate expansion honors ranges and max_trials") {
    const auto expanded = vsag::autotune::internal::ExpandJson(R"({
        "degree": [16, 32],
        "construction": {"$range": {"start": 100, "stop": 200, "step": 100}}
    })"_json);
    REQUIRE(expanded.size() == 4);

    const auto floating_range = vsag::autotune::internal::ExpandJson(
        JsonType{{"$range", {{"start", 0.1}, {"stop", 0.3}, {"step", 0.1}}}});
    REQUIRE(floating_range.size() == 3);
    REQUIRE(std::abs(floating_range.back().get<double>() - 0.3) < 1.0e-12);

    const auto descending_range = vsag::autotune::internal::ExpandJson(
        JsonType{{"$range", {{"start", 3}, {"stop", -1}, {"step", -2}}}});
    REQUIRE(descending_range == JsonType::array({3, 1, -1}));

    const double large_start = 1.0e20;
    const double large_stop = std::nextafter(large_start, std::numeric_limits<double>::infinity());
    const JsonType non_advancing_range{
        {"$range", {{"start", large_start}, {"stop", large_stop}, {"step", 1.0}}}};
    RequireThrowsContaining(
        [&non_advancing_range]() { vsag::autotune::internal::ExpandJson(non_advancing_range); },
        "$range step is too small to advance");

    JsonType too_many_fields = JsonType::object();
    for (uint64_t i = 0; i < 129; ++i) {
        too_many_fields["field_" + std::to_string(i)] = 1;
    }
    RequireThrowsContaining(
        [&too_many_fields]() { vsag::autotune::internal::ExpandJson(too_many_fields); },
        "object field count exceeds the V1 safety limit");

    ScopedPath data_path(UniqueTempPath("vsag-autotune-trial-limit", ".hdf5"));
    WriteTextFile(data_path.Get());
    auto request = MakeValidRequest(data_path.Get());
    request["indexes"][0]["create_params"] = JsonType{
        {"index_param",
         {{"base_quantization_type", "fp32"}, {"max_degree", 16}, {"ef_construction", 100}}}};
    request["tuning_config"]["max_trials"] = 2;

    auto parsed = ParseAndResolve(request, DenseDatasetDescription(100));
    RequireThrowsContaining([&parsed]() { vsag::autotune::internal::GenerateCandidates(parsed); },
                            "trial count exceeds tuning_config.max_trials");

    request["indexes"] = JsonType::array({JsonType{
        {"name", "ivf"},
        {"create_params",
         {{"index_param", {{"base_quantization_type", "fp32"}, {"buckets_count", 100001}}}}},
        {"search_params",
         {{"ivf",
           {{"scan_buckets_count",
             {{"$range", {{"start", 1}, {"stop", 100001}, {"step", 1}}}}}}}}}}});
    request["tuning_config"]["max_trials"] = 1;
    parsed = ParseAndResolve(request, DenseDatasetDescription(100, 200000));
    RequireThrowsContaining([&parsed]() { vsag::autotune::internal::GenerateCandidates(parsed); },
                            "trial count exceeds tuning_config.max_trials");

    request["indexes"] = JsonType::array(
        {JsonType{{"name", "ivf"},
                  {"create_params",
                   {{"index_param", {{"base_quantization_type", "fp32"}, {"buckets_count", 32}}}}},
                  {"search_params", JsonType::object()}}});
    request["tuning_config"]["max_trials"] = 2;
    parsed = ParseAndResolve(request, DenseDatasetDescription(100));
    const auto filtered_generation = vsag::autotune::internal::GenerateCandidates(parsed);
    const auto& filtered_candidates = filtered_generation;
    REQUIRE(filtered_candidates.size() == 2);
    REQUIRE(filtered_candidates[0].search_params["ivf"]["scan_buckets_count"] == 16);
    REQUIRE(filtered_candidates[1].search_params["ivf"]["scan_buckets_count"] == 32);

    request["indexes"].push_back(request["indexes"][0]);
    parsed = ParseAndResolve(request, DenseDatasetDescription(100));
    const auto duplicate_generation = vsag::autotune::internal::GenerateCandidates(parsed);
    REQUIRE(duplicate_generation.size() == 2);
}

TEST_CASE("AutoTune preserves explicit index candidates without generation-time filtering") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-hgraph-clamp", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get());
    request["indexes"][0]["create_params"] = JsonType{
        {"index_param",
         {{"base_quantization_type", "fp32"}, {"max_degree", 16}, {"ef_construction", 100}}}};
    request["indexes"][0]["search_params"] =
        JsonType{{"hgraph", {{"ef_search", JsonType::array({50, 150})}}}};
    request["workload"] = MakeWorkload(100, 1);

    auto parsed = ParseAndResolve(request, DenseDatasetDescription(100));
    const auto generation = vsag::autotune::internal::GenerateCandidates(parsed);
    const auto& candidates = generation;
    REQUIRE(candidates.size() == 2);

    std::set<int64_t> values;
    for (const auto& candidate : candidates) {
        values.emplace(candidate.search_params["hgraph"]["ef_search"].get<int64_t>());
    }
    REQUIRE(values == std::set<int64_t>{50, 150});

    auto mixed_index_request = request;
    const auto invalid_ivf =
        JsonType{{"name", "ivf"},
                 {"create_params",
                  {{"index_param", {{"base_quantization_type", "fp32"}, {"buckets_count", 32}}}}},
                 {"search_params", {{"ivf", {{"scan_buckets_count", 64}}}}}};
    mixed_index_request["indexes"] = JsonType::array({invalid_ivf, request["indexes"][0]});
    auto mixed_parsed = ParseAndResolve(mixed_index_request, DenseDatasetDescription(100));
    const auto mixed_generation = vsag::autotune::internal::GenerateCandidates(mixed_parsed);
    REQUIRE(mixed_generation.size() == 3);
    std::map<std::string, uint64_t> candidates_by_index;
    for (const auto& candidate : mixed_generation) {
        ++candidates_by_index[candidate.index_name];
    }
    REQUIRE(candidates_by_index["hgraph"] == 2);
    REQUIRE(candidates_by_index["ivf"] == 1);
}

TEST_CASE("AutoTune existing-index planning is search-only and requires a concrete build") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-existing-data", ".hdf5"));
    ScopedPath index_path(UniqueTempPath("vsag-autotune-existing", ".index"));
    WriteTextFile(data_path.Get());
    WriteTextFile(index_path.Get());

    auto request = MakeValidRequest(data_path.Get());
    request["index_path"] = index_path.Get();
    request["indexes"][0]["create_params"] = JsonType{
        {"index_param",
         {{"base_quantization_type", "fp32"}, {"max_degree", 16}, {"ef_construction", 100}}}};
    request["indexes"][0]["search_params"] =
        JsonType{{"hgraph", {{"ef_search", JsonType::array({20, 40})}}}};

    auto parsed = ParseAndResolve(request);
    const auto candidates = vsag::autotune::internal::GenerateCandidates(parsed);
    const auto plan = vsag::autotune::internal::PlanTrials(
        parsed, candidates, parsed.tuning_config.workspace_path);
    REQUIRE(plan.builds.size() == 1);
    REQUIRE(plan.builds[0].use_existing_index);
    REQUIRE_FALSE(plan.builds[0].cleanup_index_after_build_group);
    REQUIRE(plan.builds[0].index_path == index_path.Get());
    REQUIRE(plan.trials.size() == 2);

    SECTION("build-only metrics cannot be constrained or optimized") {
        auto invalid = request;
        invalid["constraints"] = {{"build_seconds", 10.0}};
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "unavailable for an existing index");

        invalid = request;
        invalid["objective"] = JsonType{{"metric", "build_and_search_seconds"}};
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "unavailable for an existing index");
    }

    SECTION("one existing artifact maps to one index specification") {
        auto invalid = request;
        invalid["indexes"].push_back(invalid["indexes"][0]);
        RequireThrowsContaining(
            [&invalid]() { vsag::autotune::internal::ParseAutoTuneRequest(invalid); },
            "index_path requires exactly one indexes[] specification");
    }

    SECTION("create parameters must not expand for an existing artifact") {
        auto expanding = request;
        expanding["indexes"][0]["create_params"]["index_param"]["max_degree"] =
            JsonType::array({16, 32});
        auto expanding_parsed = ParseAndResolve(expanding);
        RequireThrowsContaining(
            [&expanding_parsed]() {
                vsag::autotune::internal::GenerateCandidates(expanding_parsed);
            },
            "index_path requires exactly one concrete create_params candidate");
    }

    SECTION("result_path must not overwrite input data or an existing index") {
        auto data_alias = request;
        data_alias["output"] = {{"result_path", data_path.Get()}};
        RequireThrowsContaining(
            [&data_alias]() { vsag::autotune::internal::ParseAutoTuneRequest(data_alias); },
            "result_path must not alias data_path");

        auto index_alias = request;
        index_alias["output"] = {{"result_path", index_path.Get()}};
        RequireThrowsContaining(
            [&index_alias]() { vsag::autotune::internal::ParseAutoTuneRequest(index_alias); },
            "result_path must not alias index_path");
    }
}

TEST_CASE("AutoTune selector compares complete create and search candidates") {
    auto request = SelectionRequest();
    EvaluationResult evaluation;
    evaluation.build_results = {MakeBuildResult("build-a", 2.0, 20.0)};
    evaluation.trial_results = {MakeTrialResult("fast", "build-a", 0.95, 2.0),
                                MakeTrialResult("slower", "build-a", 0.99, 3.0)};

    const auto selection = vsag::autotune::internal::SelectResult(request, evaluation);
    INFO(selection.dump(2));
    REQUIRE(selection["status"] == "success");
    REQUIRE(selection["recommendation"]["evidence"]["trial_id"] == "fast");
    REQUIRE(selection["recommendation"]["workload"] == JsonType{{"top_k", 10}, {"concurrency", 1}});
    REQUIRE(selection["recommendation"]["search_params"]["hgraph"]["ef_search"] == 20.0);
    REQUIRE_FALSE(selection["recommendation"].contains("constraint_evaluation"));
    REQUIRE_FALSE(selection["recommendation"].contains("selection_reason"));

    SECTION("objective direction comes from the metric registry") {
        request.objective = ObjectiveSpec{"recall_at_k"};
        const auto recall_selection = vsag::autotune::internal::SelectResult(request, evaluation);
        REQUIRE(recall_selection["recommendation"]["evidence"]["trial_id"] == "slower");
    }

    SECTION("a build metric objective is read from each complete trial") {
        request.objective = ObjectiveSpec{"index_size_mb"};
        evaluation.build_results.push_back(MakeBuildResult("build-b", 3.0, 10.0));
        evaluation.trial_results.push_back(MakeTrialResult("b", "build-b", 0.91, 9.0, 3.0, 10.0));

        const auto build_selection = vsag::autotune::internal::SelectResult(request, evaluation);
        REQUIRE(build_selection["status"] == "success");
        REQUIRE(build_selection["recommendation"]["evidence"]["build_id"] == "build-b");
    }
}

TEST_CASE("AutoTune selector reports best effort and execution failure distinctly") {
    auto request = SelectionRequest();

    SECTION("an infeasible candidate is explanation-only best effort") {
        EvaluationResult evaluation;
        evaluation.build_results = {MakeBuildResult("build-a", 2.0, 20.0)};
        evaluation.trial_results = {MakeTrialResult("infeasible", "build-a", 0.80, 2.0)};

        const auto selection = vsag::autotune::internal::SelectResult(request, evaluation);
        REQUIRE(selection["status"] == "no_candidate_satisfied");
        REQUIRE(selection["recommendation"].is_null());
        REQUIRE(selection["best_effort"]["evidence"]["trial_id"] == "infeasible");
        REQUIRE(selection["best_effort"]["violation_summary"]["violated_constraint_count"] == 1);
        REQUIRE(selection["best_effort"].contains("constraint_evaluation"));
        REQUIRE_FALSE(selection["best_effort"].contains("selection_reason"));
    }

    SECTION("no successful trial is a structured selection failure") {
        EvaluationResult evaluation;
        evaluation.build_results = {MakeBuildResult("build-a", 2.0, 20.0)};
        auto trial = MakeTrialResult("failed", "build-a", 0.95, 2.0);
        trial["status"] = "failed";
        evaluation.trial_results = {std::move(trial)};

        const auto selection = vsag::autotune::internal::SelectResult(request, evaluation);
        REQUIRE(selection["status"] == "failed");
        REQUIRE(selection["best_effort"].is_null());
        REQUIRE(selection["failure"]["code"] == "no_successful_trial");
    }
}

TEST_CASE("AutoTune selector fails when a successful result lacks an objective metric") {
    SECTION("the objective metric is required on every successful trial") {
        auto request = SelectionRequest();
        EvaluationResult evaluation;
        evaluation.build_results = {MakeBuildResult("build-a", 2.0, 20.0)};
        auto trial = MakeTrialResult("missing-objective", "build-a", 0.95, 2.0);
        trial["metrics"].erase("latency_avg_ms");
        evaluation.trial_results = {std::move(trial)};

        const auto selection = vsag::autotune::internal::SelectResult(request, evaluation);
        INFO(selection.dump(2));
        REQUIRE(selection["status"] == "failed");
        REQUIRE(selection["recommendation"].is_null());
        REQUIRE(selection["best_effort"].is_null());
        REQUIRE(selection["failure"]["stage"] == "selection");
        REQUIRE(selection["failure"]["code"] == "objective_metric_missing");
        REQUIRE(selection["failure"]["metric"] == "latency_avg_ms");
        REQUIRE(selection["failure"]["trial_id"] == "missing-objective");
    }

    SECTION("a build metric objective must be present in the complete trial metrics") {
        auto request = SelectionRequest();
        request.objective = ObjectiveSpec{"index_size_mb"};
        EvaluationResult evaluation;
        evaluation.build_results = {MakeBuildResult("build-missing-objective", 2.0, 20.0)};
        auto trial = MakeTrialResult("missing-index-size", "build-missing-objective", 0.95, 2.0);
        trial["metrics"].erase("index_size_mb");
        evaluation.trial_results = {std::move(trial)};

        const auto selection = vsag::autotune::internal::SelectResult(request, evaluation);
        INFO(selection.dump(2));
        REQUIRE(selection["status"] == "failed");
        REQUIRE(selection["failure"]["stage"] == "selection");
        REQUIRE(selection["failure"]["code"] == "objective_metric_missing");
        REQUIRE(selection["failure"]["metric"] == "index_size_mb");
        REQUIRE(selection["failure"]["trial_id"] == "missing-index-size");
    }
}

TEST_CASE("AutoTune result summary exposes the exact CLI contract") {
    const JsonType recommendation{{"index_name", "hgraph"},
                                  {"create_params", {{"index_param", {{"max_degree", 32}}}}},
                                  {"search_params", {{"hgraph", {{"ef_search", 80}}}}},
                                  {"workload", {{"top_k", 10}, {"concurrency", 1}}},
                                  {"metrics", {{"index_size_mb", 12.0}, {"recall_at_k", 0.95}}},
                                  {"future_recommendation_field", true}};
    JsonType report{{"version", 1},
                    {"status", "success"},
                    {"elapsed_seconds", 12.5},
                    {"recommendation", recommendation},
                    {"best_effort", nullptr},
                    {"report_path", "/tmp/vsag-autotune-result.json"},
                    {"failure", nullptr},
                    {"input_request", {{"version", 1}}},
                    {"effective_request", JsonType::object()},
                    {"environment", {{"logical_cores", 8}}},
                    {"builds", JsonType::array({JsonType{{"build_id", "build-1"}}})},
                    {"trials", JsonType::array({JsonType{{"trial_id", "trial-1"}}})}};
    const auto original_report = report;

    const auto summary = vsag::autotune::internal::MakeResultSummary(report);
    REQUIRE(summary == JsonType{{"version", 1},
                                {"status", "success"},
                                {"elapsed_seconds", 12.5},
                                {"recommendation", recommendation},
                                {"report_path", "/tmp/vsag-autotune-result.json"}});

    const auto formatted = vsag::autotune::internal::FormatResultSummaryForCli(report);
    REQUIRE(JsonType::parse(formatted) == summary);
    RequireTextBefore(formatted, "\n  \"recommendation\"", "\n  \"status\"");
    RequireTextBefore(formatted, "\n  \"status\"", "\n  \"elapsed_seconds\"");
    RequireTextBefore(formatted, "\n  \"elapsed_seconds\"", "\n  \"report_path\"");
    RequireTextBefore(formatted, "\n    \"index_name\"", "\n    \"create_params\"");
    RequireTextBefore(formatted, "\n    \"create_params\"", "\n    \"search_params\"");
    RequireTextBefore(formatted, "\n    \"search_params\"", "\n    \"workload\"");
    RequireTextBefore(formatted, "\n    \"workload\"", "\n    \"metrics\"");
    REQUIRE(formatted.find("future_recommendation_field") != std::string::npos);
    REQUIRE(report == original_report);

    SECTION("failed reports keep only the public failure") {
        report["status"] = "failed";
        report["recommendation"] = nullptr;
        report["failure"] =
            JsonType{{"stage", "validation"}, {"code", "invalid_request"}, {"message", "bad"}};
        const auto failed_summary = vsag::autotune::internal::MakeResultSummary(report);
        RequireExactKeys(failed_summary,
                         {"version", "status", "elapsed_seconds", "report_path", "failure"});
        REQUIRE(failed_summary["status"] == "failed");
        REQUIRE_FALSE(failed_summary.contains("recommendation"));
        REQUIRE_FALSE(failed_summary.contains("best_effort"));
        REQUIRE(failed_summary["failure"] == report["failure"]);

        const auto failed_formatted = vsag::autotune::internal::FormatResultSummaryForCli(report);
        REQUIRE(JsonType::parse(failed_formatted) == failed_summary);
        RequireTextBefore(failed_formatted, "\n  \"failure\"", "\n  \"status\"");
        RequireTextBefore(failed_formatted, "\n    \"stage\"", "\n    \"code\"");
        RequireTextBefore(failed_formatted, "\n    \"code\"", "\n    \"message\"");
    }

    SECTION("no-candidate reports contain only the applicable result branch") {
        report["status"] = "no_candidate_satisfied";
        report["recommendation"] = nullptr;
        report["best_effort"] = recommendation;
        const auto no_candidate_summary = vsag::autotune::internal::MakeResultSummary(report);
        const auto no_candidate_formatted =
            vsag::autotune::internal::FormatResultSummaryForCli(report);
        REQUIRE(JsonType::parse(no_candidate_formatted) == no_candidate_summary);
        RequireTextBefore(no_candidate_formatted, "\n  \"best_effort\"", "\n  \"status\"");
        REQUIRE_FALSE(no_candidate_summary.contains("recommendation"));
        REQUIRE_FALSE(no_candidate_summary.contains("failure"));
    }
}

TEST_CASE("AutoTune runs HGraph and IVF full-grid KNN evaluation for one workload") {
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath dataset_path(UniqueTempPath("vsag-autotune-integration", ".hdf5"));
    ScopedPath workspace_path(UniqueTempPath("vsag-autotune-integration-workspace"));
    WriteDenseEvalDataset(dataset_path.Get());
    const auto result_path = (std::filesystem::path(workspace_path.Get()) / "result.json").string();

    auto request = JsonType{
        {"version", 1},
        {"data_path", dataset_path.Get()},
        {"indexes",
         JsonType::array(
             {JsonType{{"name", "hgraph"},
                       {"create_params",
                        {{"index_param",
                          {{"base_quantization_type", "fp32"},
                           {"max_degree", 8},
                           {"ef_construction", 40},
                           {"build_thread_count", 2}}}}},
                       {"search_params", {{"hgraph", {{"ef_search", JsonType::array({8, 16})}}}}}},
              JsonType{{"name", "ivf"},
                       {"create_params",
                        {{"index_param",
                          {{"base_quantization_type", "fp32"},
                           {"buckets_count", 4},
                           {"thread_count", 2}}}}},
                       {"search_params",
                        {{"ivf", {{"scan_buckets_count", JsonType::array({1, 4})}}}}}}})},
        {"workload", MakeWorkload(3, 2)},
        {"constraints",
         {{"recall_at_k", 0.0}, {"index_size_mb", 1024.0}, {"build_seconds", 1000.0}}},
        {"objective", {{"metric", "index_size_mb"}}},
        {"tuning_config",
         {{"workspace_path", workspace_path.Get()},
          {"keep_intermediate", true},
          {"max_trials", 4}}},
        {"output", {{"result_path", result_path}, {"include_raw_eval", true}}}};

    const auto result = vsag::autotune::RunAutoTune(request);
    INFO(result.dump(2));
    REQUIRE(result["status"] == "success");
    REQUIRE(result["report_path"] == result_path);
    REQUIRE(result["input_request"] == request);
    REQUIRE(result["effective_request"]["dataset_description"]["dim"] == 8);
    REQUIRE(result["effective_request"]["dataset_description"]["query_count"] == 8);
    REQUIRE(result["effective_request"]["output"]["include_raw_eval"] == true);
    REQUIRE(result["environment"].is_object());
    REQUIRE(result["evaluation_strategy"]["name"] == "full_grid");
    REQUIRE(result["evaluation_strategy"]["query_coverage"] == "full_dataset");
    REQUIRE(result["build_count"] == 2);
    REQUIRE(result["build_group_count"] == 2);
    REQUIRE(result["trial_count"] == 4);
    REQUIRE(result["builds"].size() == 2);
    REQUIRE(result["trials"].size() == 4);
    REQUIRE(result["recommendation"]["workload"] == MakeWorkload(3, 2));
    REQUIRE(result["recommendation"].contains("search_params"));
    REQUIRE(std::filesystem::exists(result_path));
    REQUIRE(ReadJsonFile(result_path) == result);

    std::map<std::string, uint64_t> trials_by_build;
    for (const auto& build : result["builds"]) {
        RequireExactKeys(build,
                         {"build_id",
                          "status",
                          "index_name",
                          "eval_type",
                          "create_params",
                          "metrics",
                          "constraint_evaluation",
                          "artifacts",
                          "failure",
                          "elapsed_seconds",
                          "raw_eval_result"});
        REQUIRE(build["status"] == "success");
        REQUIRE(build["eval_type"] == "build");
        REQUIRE(build["metrics"].contains("build_seconds"));
        REQUIRE(build["metrics"].contains("index_size_mb"));
        REQUIRE(build["metrics"].contains("index_memory_mb"));
        REQUIRE(build["raw_eval_result"]["action"] == "build");
        const auto& effective_index_params = build["raw_eval_result"]["index_info"]["index_param"];
        if (build["index_name"] == "hgraph") {
            REQUIRE(effective_index_params["build_thread_count"] == 2);
        } else {
            REQUIRE(effective_index_params["thread_count"] == 2);
        }
        REQUIRE(build["artifacts"]["use_existing_index"] == false);
        REQUIRE(build["artifacts"]["cleaned"] == false);
        REQUIRE(std::filesystem::exists(build["artifacts"]["index_path"].get<std::string>()));
    }

    for (const auto& trial : result["trials"]) {
        RequireExactKeys(trial,
                         {"trial_id",
                          "build_id",
                          "status",
                          "top_k",
                          "search_params",
                          "metrics",
                          "constraint_evaluation",
                          "execution",
                          "failure",
                          "elapsed_seconds",
                          "raw_eval_result"});
        REQUIRE(trial["status"] == "success");
        REQUIRE(trial["metrics"].contains("recall_at_k"));
        REQUIRE(trial["metrics"].contains("latency_avg_ms"));
        REQUIRE(trial["metrics"].contains("latency_p99_ms"));
        REQUIRE(trial["metrics"].contains("qps"));
        REQUIRE(trial["metrics"].contains("search_seconds"));
        REQUIRE(trial["metrics"].contains("build_and_search_seconds"));
        REQUIRE(trial["metrics"].contains("build_seconds"));
        REQUIRE(trial["metrics"].contains("index_size_mb"));
        REQUIRE(trial["metrics"].contains("index_memory_mb"));
        REQUIRE_FALSE(trial["constraint_evaluation"].contains("shared"));
        REQUIRE_FALSE(trial["constraint_evaluation"].contains("workload"));
        REQUIRE(trial["constraint_evaluation"].contains("satisfied_constraints"));
        RequireExactKeys(trial["execution"],
                         {"query_count",
                          "requested_concurrency",
                          "index_instance_reuse",
                          "load_policy",
                          "reload_succeeded",
                          "index_deserialize_count"});
        REQUIRE(trial["execution"]["query_count"] == 8);
        REQUIRE(trial["execution"]["index_instance_reuse"] == false);
        REQUIRE(trial["execution"]["load_policy"] == "fresh_deserialize_per_trial");
        REQUIRE(trial["execution"]["reload_succeeded"] == true);
        REQUIRE(trial["execution"]["index_deserialize_count"] == 1);
        REQUIRE(trial["raw_eval_result"]["statistics_query_count"] == 8);
        ++trials_by_build[trial["build_id"].get<std::string>()];
    }
    REQUIRE(trials_by_build.size() == 2);
    for (const auto& [build_id, count] : trials_by_build) {
        INFO(build_id);
        REQUIRE(count == 2);
    }

    const auto& hgraph_build = FindBuild(result, "hgraph");
    const auto hgraph_index_path = hgraph_build["artifacts"]["index_path"].get<std::string>();

    auto existing_request = request;
    existing_request.erase("output");
    existing_request["index_path"] = hgraph_index_path;
    existing_request["indexes"] = JsonType::array(
        {JsonType{{"name", "hgraph"},
                  {"create_params", hgraph_build["create_params"]},
                  {"search_params", {{"hgraph", {{"ef_search", JsonType::array({8, 16})}}}}}}});
    existing_request["constraints"] = {{"index_size_mb", 1024.0}};
    existing_request["tuning_config"]["keep_intermediate"] = false;
    existing_request["tuning_config"]["max_trials"] = 2;

    const auto existing_result = vsag::autotune::RunAutoTune(existing_request);
    INFO(existing_result.dump(2));
    REQUIRE(existing_result["status"] == "success");
    REQUIRE(existing_result["build_count"] == 0);
    REQUIRE(existing_result["build_group_count"] == 1);
    REQUIRE(existing_result["trial_count"] == 2);
    REQUIRE(existing_result["report_path"].is_null());
    REQUIRE(existing_result["effective_request"]["output"]["include_raw_eval"] == false);
    RequireExactKeys(existing_result["builds"][0],
                     {"build_id",
                      "status",
                      "index_name",
                      "eval_type",
                      "create_params",
                      "metrics",
                      "constraint_evaluation",
                      "artifacts",
                      "failure",
                      "elapsed_seconds"});
    REQUIRE(existing_result["builds"][0]["eval_type"] == "existing_index");
    REQUIRE_FALSE(existing_result["builds"][0].contains("raw_eval_result"));
    REQUIRE_FALSE(existing_result["builds"][0]["metrics"].contains("build_seconds"));
    for (const auto& trial : existing_result["trials"]) {
        RequireExactKeys(trial,
                         {"trial_id",
                          "build_id",
                          "status",
                          "top_k",
                          "search_params",
                          "metrics",
                          "constraint_evaluation",
                          "execution",
                          "failure",
                          "elapsed_seconds"});
        REQUIRE(trial["status"] == "success");
        REQUIRE_FALSE(trial["metrics"].contains("build_seconds"));
        REQUIRE_FALSE(trial["metrics"].contains("build_and_search_seconds"));
        REQUIRE(trial["metrics"].contains("index_size_mb"));
        REQUIRE(trial["metrics"].contains("index_memory_mb"));
        REQUIRE_FALSE(trial.contains("raw_eval_result"));
        REQUIRE(trial["execution"]["query_count"] == 8);
        REQUIRE(trial["execution"]["index_deserialize_count"] == 1);
    }
}

TEST_CASE("AutoTune reports request failures with structured evidence") {
    SECTION("request validation failure") {
        auto request = MakeValidRequest(UniqueTempPath("vsag-autotune-missing", ".hdf5"));
        const auto result = vsag::autotune::RunAutoTune(request);
        RequireExactKeys(result,
                         {"version",
                          "run_id",
                          "run_workspace_path",
                          "report_path",
                          "input_request",
                          "effective_request",
                          "environment",
                          "evaluation_strategy",
                          "objective",
                          "status",
                          "elapsed_seconds",
                          "elapsed_breakdown_seconds",
                          "recommendation",
                          "best_effort",
                          "trial_count",
                          "build_count",
                          "build_group_count",
                          "builds",
                          "trials",
                          "failure"});
        REQUIRE(result["status"] == "failed");
        REQUIRE(result["failure"]["stage"] == "validation");
        REQUIRE(result["failure"]["code"] == "invalid_request");
        REQUIRE(result["builds"].empty());
        REQUIRE(result["trials"].empty());
        REQUIRE(result["report_path"].is_null());
        REQUIRE(result["run_id"].is_null());
        REQUIRE(result["run_workspace_path"].is_null());
        REQUIRE(result["effective_request"].is_null());
        REQUIRE(result["objective"].is_null());
        REQUIRE(result["evaluation_strategy"]["name"] == "full_grid");

        const auto& elapsed_breakdown = result["elapsed_breakdown_seconds"];
        REQUIRE(elapsed_breakdown.contains("validation"));
        REQUIRE(elapsed_breakdown["validation"].is_number());
        REQUIRE(elapsed_breakdown["validation"].get<double>() >= 0.0);

        const auto summary = vsag::autotune::internal::MakeResultSummary(result);
        RequireExactKeys(summary, {"version", "status", "elapsed_seconds", "failure"});
        REQUIRE(summary["failure"] == result["failure"]);
    }

    SECTION("malformed HDF5 is a structured validation failure") {
        ScopedPath dataset_path(UniqueTempPath("vsag-autotune-malformed", ".hdf5"));
        ScopedPath result_path(UniqueTempPath("vsag-autotune-malformed-result", ".json"));
        WriteTextFile(dataset_path.Get(), "not an HDF5 file");
        auto request = MakeValidRequest(dataset_path.Get());
        request["output"]["result_path"] = result_path.Get();
        const auto result = vsag::autotune::RunAutoTune(request);
        REQUIRE(result["status"] == "failed");
        REQUIRE(result["failure"]["stage"] == "validation");
        REQUIRE(result["failure"]["code"] == "invalid_request");
        REQUIRE(result["failure"]["message"].get<std::string>().find(
                    "failed to load evaluation dataset") != std::string::npos);
        REQUIRE(result["report_path"] == result_path.Get());
        REQUIRE(std::filesystem::exists(result_path.Get()));
        REQUIRE(ReadJsonFile(result_path.Get()) == result);
    }
}

TEST_CASE("AutoTune failed serialization never retains an incomplete artifact") {
    ScopedBlockSizeLimit block_size_limit(256UL * 1024);
    ScopedPath dataset_path(UniqueTempPath("vsag-autotune-failed-serialize", ".hdf5"));
    ScopedPath workspace_path(UniqueTempPath("vsag-autotune-failed-serialize-workspace"));
    WriteDenseEvalDataset(dataset_path.Get());

    AutoTuneRequest request;
    request.data_path = dataset_path.Get();
    request.dataset = DatasetDescription{8, "float32", "l2", 96, 8, 10, "dense_vectors"};
    request.dataset_resolved = true;
    request.workload.top_k = 3;
    request.workload.concurrency = 1;
    request.tuning_config.workspace_path = workspace_path.Get();
    request.tuning_config.keep_intermediate = true;
    request.tuning_config.max_trials = 1;

    CandidateSpec candidate;
    candidate.index_name = "hgraph";
    candidate.create_params = JsonType{{"dim", 8},
                                       {"dtype", "float32"},
                                       {"metric_type", "l2"},
                                       {"index_param",
                                        {{"base_quantization_type", "fp32"},
                                         {"max_degree", 8},
                                         {"ef_construction", 40},
                                         {"build_thread_count", 1}}}};
    candidate.search_params = JsonType{{"hgraph", {{"ef_search", 8}}}};

    const auto plan = vsag::autotune::internal::PlanTrials(
        request, {candidate}, request.tuning_config.workspace_path);
    REQUIRE(plan.builds.size() == 1);
    REQUIRE(plan.trials.size() == 1);
    REQUIRE_FALSE(plan.builds[0].cleanup_index_after_build_group);

    const auto artifact_path = plan.builds[0].index_path;
    REQUIRE(std::filesystem::create_directories(artifact_path));
    REQUIRE(std::filesystem::is_directory(artifact_path));

    const auto evaluation = vsag::autotune::internal::EvaluatePlan(request, plan);
    REQUIRE(evaluation.build_results.size() == 1);
    REQUIRE(evaluation.trial_results.size() == 1);

    const auto& build = evaluation.build_results[0];
    INFO(build.dump(2));
    REQUIRE(build["status"] == "failed");
    REQUIRE(build["failure"]["stage"] == "build");
    REQUIRE(build["failure"]["code"] == "build_evaluation_failed");
    REQUIRE(build["failure"]["message"].get<std::string>().find("complete index artifact") !=
            std::string::npos);
    REQUIRE(build["artifacts"]["cleanup_planned"] == false);
    REQUIRE(build["artifacts"]["expected_to_exist_after_response"] == false);
    REQUIRE_FALSE(std::filesystem::exists(artifact_path));
    REQUIRE(evaluation.trial_results[0]["failure"]["code"] == "build_failed");
}

TEST_CASE("AutoTune contextual defaults remain legal for concrete index candidates") {
    ScopedPath data_path(UniqueTempPath("vsag-autotune-contextual-defaults", ".hdf5"));
    WriteTextFile(data_path.Get());

    auto request = MakeValidRequest(data_path.Get(), "ivf");
    auto parsed = ParseAndResolve(request, DenseDatasetDescription(100, 8));
    auto generation = vsag::autotune::internal::GenerateCandidates(parsed);
    REQUIRE(generation.size() == 8);

    std::set<int64_t> scans;
    for (const auto& candidate : generation) {
        REQUIRE(candidate.create_params["index_param"]["buckets_count"] == 8);
        scans.emplace(candidate.search_params["ivf"]["scan_buckets_count"].get<int64_t>());
    }
    REQUIRE(scans == std::set<int64_t>{1, 2, 4, 8});

    request["indexes"][0]["create_params"] =
        JsonType{{"index_param", {{"base_quantization_type", "fp32"}, {"buckets_count", 8}}}};
    parsed = ParseAndResolve(request, DenseDatasetDescription(100, 8));
    generation = vsag::autotune::internal::GenerateCandidates(parsed);
    REQUIRE(generation.size() == 4);

    auto hgraph_request = MakeValidRequest(data_path.Get());
    hgraph_request["indexes"][0]["create_params"] =
        JsonType{{"index_param", {{"base_quantization_type", "fp32"}, {"max_degree", 128}}}};
    auto hgraph_parsed = ParseAndResolve(hgraph_request);
    const auto hgraph_generation = vsag::autotune::internal::GenerateCandidates(hgraph_parsed);
    REQUIRE(hgraph_generation.size() == 6);
    std::set<int64_t> constructions;
    for (const auto& candidate : hgraph_generation) {
        constructions.emplace(
            candidate.create_params["index_param"]["ef_construction"].get<int64_t>());
    }
    REQUIRE(constructions == std::set<int64_t>{128, 256});
}
