// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <H5Cpp.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "case/build_eval_case.h"
#include "case/search_eval_case.h"
#include "eval_config.h"
#include "eval_dataset.h"
#include "monitor/latency_monitor.h"
#include "monitor/memory_peak_monitor.h"
#include "monitor/recall_monitor.h"
#include "monitor/search_record.h"
#include "vsag/index.h"

namespace {

using vsag::eval::BuildEvalCase;
using vsag::eval::EvalConfig;
using vsag::eval::EvalDataset;
using vsag::eval::LatencyMonitor;
using vsag::eval::MemoryPeakMonitor;
using vsag::eval::RecallMonitor;
using vsag::eval::SearchEvalCase;
using vsag::eval::SearchRecord;

constexpr int64_t BASE_COUNT = 6;
constexpr int64_t QUERY_COUNT = 4;
constexpr int64_t DIM = 2;
constexpr int64_t GROUND_TRUTH_COUNT = 4;

std::string
MakeTempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

template <typename Function>
void
RequireExceptionMessage(Function&& function, const std::string& expected_message) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception& error) {
        caught = true;
        REQUIRE(error.what() == expected_message);
    }
    REQUIRE(caught);
}

float
L2(const std::vector<float>& train,
   const std::vector<float>& test,
   int64_t train_id,
   int64_t query_id) {
    float sum = 0.0F;
    for (int64_t i = 0; i < DIM; ++i) {
        const float diff = train[train_id * DIM + i] - test[query_id * DIM + i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

void
WriteDenseDataset(const std::string& path,
                  int64_t distance_query_count = QUERY_COUNT,
                  int64_t distance_k = GROUND_TRUTH_COUNT) {
    const std::vector<float> train = {
        0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2.0F, 0.0F, 0.0F, 2.0F};
    const std::vector<float> test = {0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    std::vector<int64_t> neighbors(QUERY_COUNT * GROUND_TRUTH_COUNT);
    std::vector<float> distances(QUERY_COUNT * GROUND_TRUTH_COUNT);
    for (int64_t query_id = 0; query_id < QUERY_COUNT; ++query_id) {
        std::vector<std::pair<float, int64_t>> ranked;
        for (int64_t base_id = 0; base_id < BASE_COUNT; ++base_id) {
            ranked.emplace_back(L2(train, test, base_id, query_id), base_id);
        }
        std::sort(ranked.begin(), ranked.end());
        for (int64_t k = 0; k < GROUND_TRUTH_COUNT; ++k) {
            neighbors[query_id * GROUND_TRUTH_COUNT + k] = ranked[k].second;
            distances[query_id * GROUND_TRUTH_COUNT + k] = ranked[k].first;
        }
    }

    std::remove(path.c_str());
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    auto attribute = file.createAttribute("distance", str_type, H5::DataSpace(H5S_SCALAR));
    const std::string metric = "euclidean";
    attribute.write(str_type, metric);

    hsize_t train_dims[2] = {BASE_COUNT, DIM};
    H5::DataSpace train_space(2, train_dims);
    file.createDataSet("/train", H5::PredType::NATIVE_FLOAT, train_space)
        .write(train.data(), H5::PredType::NATIVE_FLOAT);

    hsize_t test_dims[2] = {QUERY_COUNT, DIM};
    H5::DataSpace test_space(2, test_dims);
    file.createDataSet("/test", H5::PredType::NATIVE_FLOAT, test_space)
        .write(test.data(), H5::PredType::NATIVE_FLOAT);

    hsize_t neighbor_dims[2] = {QUERY_COUNT, GROUND_TRUTH_COUNT};
    H5::DataSpace neighbor_space(2, neighbor_dims);
    file.createDataSet("/neighbors", H5::PredType::NATIVE_INT64, neighbor_space)
        .write(neighbors.data(), H5::PredType::NATIVE_INT64);

    hsize_t distance_dims[2] = {static_cast<hsize_t>(distance_query_count),
                                static_cast<hsize_t>(distance_k)};
    H5::DataSpace distance_space(2, distance_dims);
    file.createDataSet("/distances", H5::PredType::NATIVE_FLOAT, distance_space)
        .write(distances.data(), H5::PredType::NATIVE_FLOAT);
}

void
AddFilterMetadata(const std::string& path,
                  const std::vector<int64_t>& train_labels,
                  const std::vector<int64_t>& test_labels,
                  const std::vector<float>& valid_ratios) {
    if (train_labels.size() != static_cast<uint64_t>(BASE_COUNT) ||
        test_labels.size() != static_cast<uint64_t>(QUERY_COUNT) || valid_ratios.empty()) {
        throw std::invalid_argument("invalid filter metadata test fixture");
    }
    H5::H5File file(path, H5F_ACC_RDWR);
    hsize_t train_label_dims[1] = {train_labels.size()};
    H5::DataSpace train_label_space(1, train_label_dims);
    file.createDataSet("/train_labels", H5::PredType::NATIVE_INT64, train_label_space)
        .write(train_labels.data(), H5::PredType::NATIVE_INT64);

    hsize_t test_label_dims[1] = {test_labels.size()};
    H5::DataSpace test_label_space(1, test_label_dims);
    file.createDataSet("/test_labels", H5::PredType::NATIVE_INT64, test_label_space)
        .write(test_labels.data(), H5::PredType::NATIVE_INT64);

    hsize_t ratio_dims[1] = {valid_ratios.size()};
    H5::DataSpace ratio_space(1, ratio_dims);
    file.createDataSet("/valid_ratios", H5::PredType::NATIVE_FLOAT, ratio_space)
        .write(valid_ratios.data(), H5::PredType::NATIVE_FLOAT);
}

void
WriteInt8DenseDataset(const std::string& path, const std::string& metric) {
    constexpr int64_t kBaseCount = 2;
    constexpr int64_t kQueryCount = 1;
    constexpr int64_t kDim = 2;
    constexpr int64_t kGroundTruthCount = 2;
    const std::vector<int8_t> train = {3, 4, 1, 1};
    const std::vector<int8_t> test = {1, 0};
    const std::vector<int64_t> neighbors = {1, 0};
    const std::vector<float> distances = {1.0F, 2.0F};

    std::remove(path.c_str());
    H5::H5File file(path, H5F_ACC_TRUNC);
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    auto attribute = file.createAttribute("distance", str_type, H5::DataSpace(H5S_SCALAR));
    attribute.write(str_type, metric);

    hsize_t train_dims[2] = {kBaseCount, kDim};
    H5::DataSpace train_space(2, train_dims);
    file.createDataSet("/train", H5::PredType::ALPHA_I8, train_space)
        .write(train.data(), H5::PredType::ALPHA_I8);

    hsize_t test_dims[2] = {kQueryCount, kDim};
    H5::DataSpace test_space(2, test_dims);
    file.createDataSet("/test", H5::PredType::ALPHA_I8, test_space)
        .write(test.data(), H5::PredType::ALPHA_I8);

    hsize_t result_dims[2] = {kQueryCount, kGroundTruthCount};
    H5::DataSpace result_space(2, result_dims);
    file.createDataSet("/neighbors", H5::PredType::NATIVE_INT64, result_space)
        .write(neighbors.data(), H5::PredType::NATIVE_INT64);
    file.createDataSet("/distances", H5::PredType::NATIVE_FLOAT, result_space)
        .write(distances.data(), H5::PredType::NATIVE_FLOAT);
}

class FakeIndex : public vsag::Index {
public:
    tl::expected<std::vector<int64_t>, vsag::Error>
    Build(const vsag::DatasetPtr& base) override {
        ++build_count;
        return std::vector<int64_t>{};
    }

    tl::expected<vsag::DatasetPtr, vsag::Error>
    KnnSearch(const vsag::DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              vsag::BitsetPtr invalid) const override {
        RecordSearchStart();
        const uint64_t active = active_search_count.fetch_add(1) + 1;
        uint64_t observed = max_active_search_count.load();
        while (observed < active &&
               not max_active_search_count.compare_exchange_weak(observed, active)) {
        }
        ++search_count;
        if (search_delay.count() > 0) {
            std::this_thread::sleep_for(search_delay);
        }
        active_search_count.fetch_sub(1);
        if (search_error) {
            RecordSearchEnd();
            return tl::unexpected(
                vsag::Error(vsag::ErrorType::INTERNAL_ERROR, "injected query error"));
        }

        if (fixed_result != nullptr) {
            RecordSearchEnd();
            return fixed_result;
        }
        auto* ids = new int64_t[result_ids.size()];
        auto* distances = new float[result_ids.size()];
        std::copy(result_ids.begin(), result_ids.end(), ids);
        std::fill(distances, distances + result_ids.size(), 0.0F);
        auto result = vsag::Dataset::Make();
        result->NumElements(1)
            ->Dim(static_cast<int64_t>(result_ids.size()))
            ->Ids(ids)
            ->Distances(distances)
            ->Owner(true);
        RecordSearchEnd();
        return result;
    }

    tl::expected<vsag::DatasetPtr, vsag::Error>
    KnnSearch(const vsag::DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        return KnnSearch(query, k, parameters, vsag::BitsetPtr{});
    }

    tl::expected<vsag::DatasetPtr, vsag::Error>
    RangeSearch(const vsag::DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size) const override {
        return UnsupportedSearch();
    }

    tl::expected<vsag::DatasetPtr, vsag::Error>
    RangeSearch(const vsag::DatasetPtr& query,
                float radius,
                const std::string& parameters,
                vsag::BitsetPtr invalid,
                int64_t limited_size) const override {
        return UnsupportedSearch();
    }

    tl::expected<vsag::DatasetPtr, vsag::Error>
    RangeSearch(const vsag::DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size) const override {
        return UnsupportedSearch();
    }

    tl::expected<vsag::BinarySet, vsag::Error>
    Serialize() const override {
        return vsag::BinarySet{};
    }

    tl::expected<void, vsag::Error>
    Serialize(std::ostream& out_stream) override {
        if (serialize_error) {
            return tl::unexpected(
                vsag::Error(vsag::ErrorType::INTERNAL_ERROR, "injected serialize error"));
        }
        out_stream.write("fake", 4);
        if (fail_serialize_stream) {
            out_stream.setstate(std::ios::badbit);
        }
        return {};
    }

    tl::expected<void, vsag::Error>
    Deserialize(const vsag::BinarySet& binary_set) override {
        return {};
    }

    tl::expected<void, vsag::Error>
    Deserialize(const vsag::ReaderSet& reader_set) override {
        return {};
    }

    tl::expected<void, vsag::Error>
    Deserialize(std::istream& in_stream) override {
        if (deserialize_error) {
            return tl::unexpected(
                vsag::Error(vsag::ErrorType::INVALID_BINARY, "injected deserialize error"));
        }
        if (fail_deserialize_stream) {
            in_stream.setstate(std::ios::badbit);
        }
        return {};
    }

    int64_t
    GetNumElements() const override {
        return BASE_COUNT;
    }

    uint64_t
    GetMemoryUsage() const override {
        return 1234;
    }

    double
    GetObservedSearchWindowSeconds() const {
        const int64_t start_ns = first_search_start_ns.load();
        const int64_t end_ns = last_search_end_ns.load();
        if (start_ns == std::numeric_limits<int64_t>::max() || end_ns <= start_ns) {
            return 0.0;
        }
        return static_cast<double>(end_ns - start_ns) / 1'000'000'000.0;
    }

    std::vector<int64_t> result_ids{0, 1};
    vsag::DatasetPtr fixed_result{nullptr};
    std::chrono::milliseconds search_delay{0};
    bool serialize_error{false};
    bool fail_serialize_stream{false};
    bool deserialize_error{false};
    bool fail_deserialize_stream{false};
    bool search_error{false};
    std::atomic<uint64_t> build_count{0};
    mutable std::atomic<uint64_t> search_count{0};
    mutable std::atomic<uint64_t> active_search_count{0};
    mutable std::atomic<uint64_t> max_active_search_count{0};
    mutable std::atomic<int64_t> first_search_start_ns{std::numeric_limits<int64_t>::max()};
    mutable std::atomic<int64_t> last_search_end_ns{0};

private:
    static int64_t
    NowNanoseconds() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    void
    RecordSearchStart() const {
        const int64_t now_ns = NowNanoseconds();
        int64_t observed = first_search_start_ns.load(std::memory_order_relaxed);
        while (now_ns < observed && not first_search_start_ns.compare_exchange_weak(
                                        observed, now_ns, std::memory_order_relaxed)) {
        }
    }

    void
    RecordSearchEnd() const {
        const int64_t now_ns = NowNanoseconds();
        int64_t observed = last_search_end_ns.load(std::memory_order_relaxed);
        while (now_ns > observed && not last_search_end_ns.compare_exchange_weak(
                                        observed, now_ns, std::memory_order_relaxed)) {
        }
    }

    static tl::expected<vsag::DatasetPtr, vsag::Error>
    UnsupportedSearch() {
        return tl::unexpected(
            vsag::Error(vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION, "not implemented"));
    }
};

EvalConfig
MakeSearchConfig() {
    EvalConfig config;
    config.action_type = "search";
    config.index_name = "fake";
    config.build_param = "{}";
    config.search_param = "{}";
    config.search_mode = "knn";
    config.top_k = 2;
    config.search_query_count = QUERY_COUNT;
    config.query_limit_count = 0;
    config.num_threads_searching = 2;
    config.enable_memory = false;
    return config;
}

class LargeGroundTruthDataset : public EvalDataset {
public:
    LargeGroundTruthDataset() {
        number_of_query_ = 48;
        neighbors_shape_ = {48, 100000};
    }
};

void
WriteIndexPlaceholder(const std::string& path) {
    std::ofstream output(path, std::ios::binary);
    output.write("fake", 4);
}

}  // namespace

TEST_CASE("EvalDataset exposes ground-truth metadata and validates aligned shapes") {
    const auto valid_path = MakeTempPath("vsag_eval_correctness_valid.hdf5");
    const auto invalid_path = MakeTempPath("vsag_eval_correctness_invalid.hdf5");
    WriteDenseDataset(valid_path);
    auto dataset = EvalDataset::Load(valid_path);
    REQUIRE(dataset->GetGroundTruthK() == GROUND_TRUTH_COUNT);
    REQUIRE(dataset->GetMetric() == "euclidean");
    REQUIRE(dataset->GetInfo()["dataset_info"]["ground_truth_k"] == GROUND_TRUTH_COUNT);
    REQUIRE(dataset->GetInfo()["dataset_info"]["metric"] == "euclidean");

    WriteDenseDataset(invalid_path, QUERY_COUNT - 1, GROUND_TRUTH_COUNT);
    RequireExceptionMessage([&]() { EvalDataset::Load(invalid_path); },
                            "neighbors and distances shapes do not match");
    std::remove(valid_path.c_str());
    std::remove(invalid_path.c_str());
}

TEST_CASE("EvalDataset uses dtype-aware distances for dense int8 vectors") {
    const int8_t left[] = {3, 4};
    const int8_t zero[] = {0, 0};
    const int8_t diagonal[] = {1, 1};
    const int8_t axis[] = {1, 0};
    const uint64_t dim = 2;

    const auto euclidean_path = MakeTempPath("vsag_eval_correctness_int8_l2.hdf5");
    WriteInt8DenseDataset(euclidean_path, "euclidean");
    auto dataset = EvalDataset::Load(euclidean_path);
    REQUIRE(dataset->GetTrainDataType() == vsag::DATATYPE_INT8);
    REQUIRE(dataset->GetDistanceFunc()(left, zero, &dim) == Catch::Approx(5.0F));

    const auto angular_path = MakeTempPath("vsag_eval_correctness_int8_angular.hdf5");
    WriteInt8DenseDataset(angular_path, "angular");
    dataset = EvalDataset::Load(angular_path);
    const float expected_angular = 1.0F - 1.0F / std::sqrt(2.0F);
    REQUIRE(dataset->GetDistanceFunc()(diagonal, axis, &dim) == Catch::Approx(expected_angular));

    std::remove(euclidean_path.c_str());
    std::remove(angular_path.c_str());
}

TEST_CASE("LatencyMonitor uses per-call latency and finite batch metrics") {
    LatencyMonitor monitor(3);
    monitor.SetMetrics("qps");
    monitor.SetMetrics("avg_latency");
    monitor.SetMetrics("percent_latency");
    monitor.Start();
    for (double latency : {1.0, 2.0, 3.0}) {
        SearchRecord record;
        record.latency_ms = latency;
        monitor.Record(&record);
    }
    monitor.Stop();
    const auto result = monitor.GetResult();
    REQUIRE(result["latency_avg(ms)"].get<double>() == Catch::Approx(2.0));
    REQUIRE(result["latency_detail(ms)"]["p99"].get<double>() == Catch::Approx(2.0));
    REQUIRE(result["duration(s)"].get<double>() >= 0.0);
    REQUIRE(std::isfinite(result["qps"].get<double>()));

    LatencyMonitor empty_monitor;
    empty_monitor.SetMetrics("qps");
    empty_monitor.SetMetrics("avg_latency");
    empty_monitor.SetMetrics("percent_latency");
    empty_monitor.Start();
    empty_monitor.Stop();
    const auto empty_result = empty_monitor.GetResult();
    REQUIRE(empty_result["qps"] == 0.0);
    REQUIRE(empty_result["latency_avg(ms)"] == 0.0);
    REQUIRE(empty_result["latency_detail(ms)"]["p99"] == 0.0);
}

TEST_CASE("Search QPS excludes result statistics and bookkeeping time") {
    const auto dataset_path = MakeTempPath("vsag_eval_correctness_qps.hdf5");
    const auto index_path = MakeTempPath("vsag_eval_correctness_qps.index");
    WriteDenseDataset(dataset_path);
    WriteIndexPlaceholder(index_path);
    auto dataset = EvalDataset::Load(dataset_path);

    auto* ids = new int64_t[2]{0, 1};
    auto* distances = new float[2]{0.0F, 0.0F};
    auto fixed_result = vsag::Dataset::Make();
    fixed_result->NumElements(1)->Dim(2)->Ids(ids)->Distances(distances)->Owner(true)->Statistics(
        "{\"padding\":\"" + std::string(256 * 1024, 'x') + "\"}");

    auto index = std::make_shared<FakeIndex>();
    index->fixed_result = std::move(fixed_result);
    index->search_delay = std::chrono::milliseconds(2);
    auto config = MakeSearchConfig();
    config.search_query_count = 8;
    config.enable_recall = false;
    config.enable_percent_recall = false;

    SearchEvalCase eval_case("unused", index_path, index, config, dataset);
    const auto wall_start = std::chrono::steady_clock::now();
    const auto result = eval_case.Run();
    const double wall_duration_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    const double search_duration_seconds = result["duration(s)"].get<double>();
    const double observed_search_window_seconds = index->GetObservedSearchWindowSeconds();

    REQUIRE(index->search_count == config.search_query_count);
    REQUIRE(result["statistics_query_count"] == config.search_query_count);
    REQUIRE(observed_search_window_seconds > 0.0);
    REQUIRE(search_duration_seconds >= observed_search_window_seconds);
    REQUIRE(search_duration_seconds > 0.0);
    REQUIRE(search_duration_seconds < wall_duration_seconds * 0.5);
    REQUIRE(
        result["qps"].get<double>() ==
        Catch::Approx(static_cast<double>(config.search_query_count) / search_duration_seconds));

    std::remove(dataset_path.c_str());
    std::remove(index_path.c_str());
}

TEST_CASE("MemoryPeakMonitor samples transient resident memory") {
    MemoryPeakMonitor monitor("transient-test");
    monitor.Start();

    constexpr uint64_t ALLOCATION_BYTES = 32ULL * 1024ULL * 1024ULL;
    void* allocation =
        mmap(nullptr, ALLOCATION_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(allocation != MAP_FAILED);
    std::memset(allocation, 1, ALLOCATION_BYTES);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    REQUIRE(munmap(allocation, ALLOCATION_BYTES) == 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    monitor.Stop();

    const auto peak = monitor.GetResult()["memory_peak(transient-test)"].get<std::string>();
    REQUIRE(peak.find("MB") != std::string::npos);
    REQUIRE(std::stof(peak) >= 16.0F);
}

TEST_CASE("RecallMonitor uses requested k and ignores empty invalid and duplicate results") {
    const auto dataset_path = MakeTempPath("vsag_eval_correctness_recall.hdf5");
    WriteDenseDataset(dataset_path);
    auto dataset = EvalDataset::Load(dataset_path);

    const int64_t candidates[] = {
        dataset->GetNeighbors(0)[0], dataset->GetNeighbors(0)[0], -1, BASE_COUNT + 10};
    SearchRecord record;
    record.neighbors = candidates;
    record.ground_truth_neighbors = dataset->GetNeighbors(0);
    record.dataset = dataset.get();
    record.query_data = dataset->GetOneTest(0);
    record.returned_count = 4;
    record.requested_k = 4;
    record.ground_truth_count = dataset->GetGroundTruthK();

    RecallMonitor monitor;
    monitor.SetMetrics("avg_recall");
    monitor.Start();
    monitor.Record(&record);
    monitor.Stop();
    REQUIRE(monitor.GetResult()["recall_avg"].get<double>() == Catch::Approx(0.25));

    record.returned_count = 1;
    monitor.Start();
    monitor.Record(&record);
    monitor.Stop();
    REQUIRE(monitor.GetResult()["recall_avg"].get<double>() == Catch::Approx(0.25));

    record.neighbors = nullptr;
    record.returned_count = 0;
    monitor.Start();
    monitor.Record(&record);
    monitor.Stop();
    REQUIRE(monitor.GetResult()["recall_avg"] == 0.0);
    std::remove(dataset_path.c_str());
}

TEST_CASE("SearchEvalCase runs the complete query set once with configured concurrency") {
    const auto dataset_path = MakeTempPath("vsag_eval_correctness_search.hdf5");
    const auto index_path = MakeTempPath("vsag_eval_correctness_search.index");
    WriteDenseDataset(dataset_path);
    WriteIndexPlaceholder(index_path);
    auto dataset = EvalDataset::Load(dataset_path);
    auto index = std::make_shared<FakeIndex>();
    index->search_delay = std::chrono::milliseconds(10);

    auto config = MakeSearchConfig();
    SearchEvalCase eval_case("unused-dataset-path", index_path, index, config, dataset);
    const auto result = eval_case.Run();
    REQUIRE(index->search_count == QUERY_COUNT);
    REQUIRE(index->max_active_search_count > 1);
    REQUIRE(index->max_active_search_count <= config.num_threads_searching);
    REQUIRE(result["statistics_query_count"] == QUERY_COUNT);
    REQUIRE(result["index_deserialize_count"] == 1);
    REQUIRE(result["index_memory(B)"] == 1234);
    REQUIRE(result["duration(s)"].get<double>() > 0.0);
    REQUIRE(std::isfinite(result["qps"].get<double>()));
    REQUIRE(std::isfinite(result["latency_avg(ms)"].get<double>()));
    REQUIRE(std::isfinite(result["latency_detail(ms)"]["p99"].get<double>()));
    REQUIRE(result["recall_avg"].get<double>() >= 0.0);
    REQUIRE(result["recall_avg"].get<double>() <= 1.0);

    constexpr uint64_t kRepeatedQueryCount = 5000;
    auto repeated_index = std::make_shared<FakeIndex>();
    auto repeated_config = MakeSearchConfig();
    repeated_config.search_query_count = kRepeatedQueryCount;
    SearchEvalCase repeated_case(
        "unused-dataset-path", index_path, repeated_index, repeated_config, dataset);
    const auto repeated_result = repeated_case.Run();
    REQUIRE(repeated_index->search_count == kRepeatedQueryCount);
    REQUIRE(repeated_result["statistics_query_count"] == kRepeatedQueryCount);
    REQUIRE(repeated_result["recall_avg"].get<double>() >= 0.0);
    REQUIRE(repeated_result["recall_avg"].get<double>() <= 1.0);

    std::remove(dataset_path.c_str());
    std::remove(index_path.c_str());
}

TEST_CASE("SearchEvalCase accepts implemented modes and rejects unsupported inputs") {
    const auto dataset_path = MakeTempPath("vsag_eval_correctness_errors.hdf5");
    const auto index_path = MakeTempPath("vsag_eval_correctness_errors.index");
    WriteDenseDataset(dataset_path);
    WriteIndexPlaceholder(index_path);
    auto dataset = EvalDataset::Load(dataset_path);

    auto config = MakeSearchConfig();
    config.search_mode = "knn";
    REQUIRE_NOTHROW(
        SearchEvalCase("unused", index_path, std::make_shared<FakeIndex>(), config, dataset));

    config.search_mode = "knn_filter";
    RequireExceptionMessage(
        [&]() {
            SearchEvalCase("unused", index_path, std::make_shared<FakeIndex>(), config, dataset);
        },
        "knn_filter requires train_labels, test_labels and valid_ratios");

    for (const auto& mode : {"range", "range_filter", "hybrid"}) {
        config.search_mode = mode;
        RequireExceptionMessage(
            [&]() {
                SearchEvalCase(
                    "unused", index_path, std::make_shared<FakeIndex>(), config, dataset);
            },
            "unsupported search_mode '" + std::string(mode) +
                "'; supported modes are knn and knn_filter");
    }

    auto range_yaml = YAML::Load(R"(
datapath: /tmp/test.hdf5
type: search
index_name: hgraph
create_params: '{}'
search_params: '{}'
search_mode: range
)");
    RequireExceptionMessage(
        [&]() { EvalConfig::CheckKeyAndType(range_yaml); },
        "unsupported search_mode: range; supported modes are knn and knn_filter");

    config.search_mode = "knn";
    config.top_k = GROUND_TRUTH_COUNT + 1;
    RequireExceptionMessage(
        [&]() {
            SearchEvalCase("unused", index_path, std::make_shared<FakeIndex>(), config, dataset);
        },
        "top_k exceeds the dataset ground-truth width");

    auto high_k_config = MakeSearchConfig();
    high_k_config.top_k = 100000;
    high_k_config.num_threads_searching = 48;
    high_k_config.search_query_count = 48;
    auto large_ground_truth = std::make_shared<LargeGroundTruthDataset>();
    RequireExceptionMessage(
        [&]() {
            SearchEvalCase("unused",
                           index_path,
                           std::make_shared<FakeIndex>(),
                           high_k_config,
                           large_ground_truth);
        },
        "top_k * num_threads_searching exceeds the result-buffer safety limit");

    auto excessive_concurrency = MakeSearchConfig();
    excessive_concurrency.num_threads_searching = QUERY_COUNT + 1;
    RequireExceptionMessage(
        [&]() {
            SearchEvalCase("unused",
                           index_path,
                           std::make_shared<FakeIndex>(),
                           excessive_concurrency,
                           dataset);
        },
        "num_threads_searching must not exceed the number of executed queries");

    auto latency_only_config = config;
    latency_only_config.enable_recall = false;
    latency_only_config.enable_percent_recall = false;
    auto latency_only_index = std::make_shared<FakeIndex>();
    SearchEvalCase latency_only_case(
        "unused", index_path, latency_only_index, latency_only_config, dataset);
    const auto latency_only_result = latency_only_case.Run();
    REQUIRE(latency_only_index->search_count == QUERY_COUNT);
    REQUIRE_FALSE(latency_only_result.contains("recall_avg"));
    REQUIRE_FALSE(latency_only_result.contains("recall_detail"));

    config.top_k = 2;
    auto error_index = std::make_shared<FakeIndex>();
    error_index->deserialize_error = true;
    SearchEvalCase deserialize_error_case("unused", index_path, error_index, config, dataset);
    RequireExceptionMessage([&]() { deserialize_error_case.Run(); },
                            "failed to deserialize index: injected deserialize error");
    REQUIRE(error_index->search_count == 0);

    auto query_error_index = std::make_shared<FakeIndex>();
    query_error_index->search_error = true;
    SearchEvalCase query_error_case("unused", index_path, query_error_index, config, dataset);
    RequireExceptionMessage([&]() { query_error_case.Run(); }, "query error: injected query error");
    REQUIRE(query_error_index->search_count > 0);
    REQUIRE(query_error_index->search_count <= QUERY_COUNT);

    auto bad_stream_index = std::make_shared<FakeIndex>();
    bad_stream_index->fail_deserialize_stream = true;
    SearchEvalCase bad_stream_case("unused", index_path, bad_stream_index, config, dataset);
    RequireExceptionMessage([&]() { bad_stream_case.Run(); },
                            "failed to deserialize index: index stream read failed");
    REQUIRE(bad_stream_index->search_count == 0);
    std::remove(dataset_path.c_str());
    std::remove(index_path.c_str());
}

TEST_CASE("SearchEvalCase validates knn_filter dataset semantics") {
    const auto valid_path = MakeTempPath("vsag_eval_correctness_filter_valid.hdf5");
    const auto mismatch_path = MakeTempPath("vsag_eval_correctness_filter_mismatch.hdf5");
    const auto label_range_path = MakeTempPath("vsag_eval_correctness_filter_label_range.hdf5");
    const auto index_path = MakeTempPath("vsag_eval_correctness_filter.index");
    WriteIndexPlaceholder(index_path);

    const std::vector<int64_t> zero_train_labels(BASE_COUNT, 0);
    const std::vector<int64_t> zero_test_labels(QUERY_COUNT, 0);
    WriteDenseDataset(valid_path);
    AddFilterMetadata(valid_path, zero_train_labels, zero_test_labels, {1.0F});
    auto valid_dataset = EvalDataset::Load(valid_path);
    REQUIRE(valid_dataset->HasValidRatios());
    REQUIRE(valid_dataset->GetNumberOfLabels() == 1);
    REQUIRE(valid_dataset->GetValidRatio(0) == 1.0F);

    auto config = MakeSearchConfig();
    config.search_mode = "knn_filter";
    REQUIRE_NOTHROW(
        SearchEvalCase("unused", index_path, std::make_shared<FakeIndex>(), config, valid_dataset));

    WriteDenseDataset(mismatch_path);
    AddFilterMetadata(mismatch_path, {0, 1, 0, 1, 0, 1}, zero_test_labels, {0.5F});
    auto mismatch_dataset = EvalDataset::Load(mismatch_path);
    RequireExceptionMessage(
        [&]() {
            SearchEvalCase(
                "unused", index_path, std::make_shared<FakeIndex>(), config, mismatch_dataset);
        },
        "knn_filter recall requires label-filtered ground truth");

    WriteDenseDataset(label_range_path);
    const std::vector<int64_t> one_train_labels(BASE_COUNT, 1);
    const std::vector<int64_t> one_test_labels(QUERY_COUNT, 1);
    AddFilterMetadata(label_range_path, one_train_labels, one_test_labels, {1.0F});
    auto label_range_dataset = EvalDataset::Load(label_range_path);
    RequireExceptionMessage(
        [&]() {
            SearchEvalCase(
                "unused", index_path, std::make_shared<FakeIndex>(), config, label_range_dataset);
        },
        "label is outside the valid_ratios range");

    const auto no_ratio_path = MakeTempPath("vsag_eval_correctness_filter_no_ratio.hdf5");
    WriteDenseDataset(no_ratio_path);
    auto no_ratio_dataset = EvalDataset::Load(no_ratio_path);
    REQUIRE_FALSE(no_ratio_dataset->HasValidRatios());
    RequireExceptionMessage([&]() { static_cast<void>(no_ratio_dataset->GetValidRatio(0)); },
                            "dataset does not contain valid_ratios");

    std::remove(valid_path.c_str());
    std::remove(mismatch_path.c_str());
    std::remove(label_range_path.c_str());
    std::remove(no_ratio_path.c_str());
    std::remove(index_path.c_str());
}

TEST_CASE("BuildEvalCase checks serialization and reports stable raw memory") {
    const auto dataset_path = MakeTempPath("vsag_eval_correctness_build.hdf5");
    const auto index_path = MakeTempPath("vsag_eval_correctness_build.index");
    WriteDenseDataset(dataset_path);
    auto dataset = EvalDataset::Load(dataset_path);
    EvalConfig config;
    config.action_type = "build";
    config.index_name = "fake";
    config.build_param = "{}";
    config.enable_memory = false;

    auto index = std::make_shared<FakeIndex>();
    BuildEvalCase eval_case("unused", index_path, index, config, dataset);
    const auto result = eval_case.Run();
    REQUIRE(index->build_count == 1);
    REQUIRE(result["index_memory(B)"] == 1234);
    REQUIRE(std::isfinite(result["duration(s)"].get<double>()));
    REQUIRE(std::isfinite(result["tps"].get<double>()));

    auto error_index = std::make_shared<FakeIndex>();
    error_index->serialize_error = true;
    BuildEvalCase serialize_error_case("unused", index_path, error_index, config, dataset);
    RequireExceptionMessage([&]() { serialize_error_case.Run(); },
                            "failed to serialize index: injected serialize error");

    auto bad_stream_index = std::make_shared<FakeIndex>();
    bad_stream_index->fail_serialize_stream = true;
    BuildEvalCase bad_stream_case("unused", index_path, bad_stream_index, config, dataset);
    RequireExceptionMessage([&]() { bad_stream_case.Run(); },
                            "failed to serialize index: index stream write failed");
    std::remove(dataset_path.c_str());
    std::remove(index_path.c_str());
}
