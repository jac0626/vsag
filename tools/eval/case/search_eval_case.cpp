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

#include "./search_eval_case.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "../monitor/latency_monitor.h"
#include "../monitor/memory_peak_monitor.h"
#include "../monitor/recall_monitor.h"
#include "../monitor/search_record.h"
#include "typing.h"
#include "vsag/filter.h"
#include "vsag_exception.h"

namespace vsag::eval {

namespace {

struct stored_search_result {
    vsag::DatasetPtr result;
    std::vector<int64_t> neighbors;
    int64_t query_id{0};
    double latency_ms{0.0};
};

constexpr uint64_t MAX_BUFFERED_QUERY_RESULTS = 4096;
constexpr uint64_t MAX_BUFFERED_NEIGHBOR_IDS = 1'000'000;

uint64_t
get_run_query_count(uint64_t query_count, const EvalConfig& config) {
    if (config.query_limit_count > 0) {
        return std::min(query_count, config.query_limit_count);
    }
    return std::max(query_count, config.search_query_count);
}

uint64_t
get_search_batch_size(uint64_t run_query_count,
                      uint64_t top_k,
                      uint64_t concurrency,
                      bool retain_neighbors) {
    uint64_t preferred_batch_size = MAX_BUFFERED_QUERY_RESULTS;
    if (retain_neighbors) {
        const uint64_t query_limit_from_neighbors =
            std::max<uint64_t>(1, MAX_BUFFERED_NEIGHBOR_IDS / top_k);
        preferred_batch_size = std::min(preferred_batch_size, query_limit_from_neighbors);
    }
    preferred_batch_size = std::max(preferred_batch_size, concurrency);
    return std::min(run_query_count, preferred_batch_size);
}

class OmpThreadCountGuard {
public:
    explicit OmpThreadCountGuard(int thread_count)
        : previous_thread_count_(omp_get_max_threads()), previous_dynamic_(omp_get_dynamic()) {
        omp_set_dynamic(0);
        omp_set_num_threads(thread_count);
    }

    ~OmpThreadCountGuard() {
        omp_set_num_threads(previous_thread_count_);
        omp_set_dynamic(previous_dynamic_);
    }

    OmpThreadCountGuard(const OmpThreadCountGuard&) = delete;
    OmpThreadCountGuard&
    operator=(const OmpThreadCountGuard&) = delete;

private:
    int previous_thread_count_;
    int previous_dynamic_;
};

void
validate_knn_filter_dataset(const EvalDataset& dataset, uint64_t requested_k, bool recall_enabled) {
    const auto train_labels = dataset.GetTrainLabels();
    const auto test_labels = dataset.GetTestLabels();
    if (train_labels == nullptr || test_labels == nullptr || !dataset.HasValidRatios()) {
        throw std::invalid_argument(
            "knn_filter requires train_labels, test_labels and valid_ratios");
    }

    const auto query_count = dataset.GetNumberOfQuery();
    const auto base_count = dataset.GetNumberOfBase();
    for (int64_t query_id = 0; query_id < query_count; ++query_id) {
        const auto query_label = test_labels[query_id];
        static_cast<void>(dataset.GetValidRatio(query_label));
        if (!recall_enabled) {
            continue;
        }
        const auto* ground_truth = dataset.GetNeighbors(query_id);
        for (uint64_t k = 0; k < requested_k; ++k) {
            const auto base_id = ground_truth[k];
            if (base_id < 0 || base_id >= base_count) {
                throw std::invalid_argument("knn_filter ground truth contains an invalid base id");
            }
            if (train_labels[base_id] != query_label) {
                throw std::invalid_argument(
                    "knn_filter recall requires label-filtered ground truth");
            }
        }
    }
}

}  // namespace

class FilterObj : public vsag::Filter {
public:
    FilterObj(const std::shared_ptr<int64_t[]>& train_labels, int64_t test_label, float valid_ratio)
        : train_labels_(train_labels), test_label_(test_label), valid_ratio_(valid_ratio) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return train_labels_[id] == test_label_;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return valid_ratio_;
    }

private:
    const std::shared_ptr<int64_t[]>& train_labels_;
    int64_t test_label_;
    float valid_ratio_;
};

SearchEvalCase::SearchEvalCase(const std::string& dataset_path,
                               const std::string& index_path,
                               vsag::IndexPtr index,
                               EvalConfig config,
                               EvalDatasetPtr dataset)
    : EvalCase(dataset_path, index_path, std::move(index), std::move(dataset)),
      config_(std::move(config)) {
    if (config_.enable_memory) {
        memory_monitor_baseline_pages_ = MemoryPeakMonitor::GetCurrentResidentPages();
    }
    if (config_.search_mode == "knn") {
        search_type_ = SearchType::KNN;
    } else if (config_.search_mode == "knn_filter") {
        search_type_ = SearchType::KNN_FILTER;
    } else {
        throw std::invalid_argument("unsupported search_mode '" + config_.search_mode +
                                    "'; supported modes are knn and knn_filter");
    }
    if (config_.top_k <= 0) {
        throw std::invalid_argument("top_k must be positive");
    }
    if (config_.num_threads_searching <= 0) {
        throw std::invalid_argument("num_threads_searching must be positive");
    }
    if (dataset_ptr_->GetNumberOfQuery() <= 0) {
        throw std::invalid_argument("dataset must contain at least one query");
    }
    const auto run_query_count =
        get_run_query_count(static_cast<uint64_t>(dataset_ptr_->GetNumberOfQuery()), config_);
    const auto requested_concurrency = static_cast<uint64_t>(config_.num_threads_searching);
    if (requested_concurrency > run_query_count) {
        throw std::invalid_argument(
            "num_threads_searching must not exceed the number of executed queries");
    }
    const bool recall_enabled = config_.enable_recall || config_.enable_percent_recall;
    if (recall_enabled && static_cast<uint64_t>(config_.top_k) > dataset_ptr_->GetGroundTruthK()) {
        throw std::invalid_argument("top_k exceeds the dataset ground-truth width");
    }
    if (recall_enabled &&
        static_cast<uint64_t>(config_.top_k) > MAX_BUFFERED_NEIGHBOR_IDS / requested_concurrency) {
        throw std::invalid_argument(
            "top_k * num_threads_searching exceeds the result-buffer safety limit");
    }
    if (search_type_ == SearchType::KNN_FILTER) {
        validate_knn_filter_dataset(
            *dataset_ptr_, static_cast<uint64_t>(config_.top_k), recall_enabled);
    }
    this->init_monitor();
}

void
SearchEvalCase::init_monitor() {
    this->init_latency_monitor();
    this->init_recall_monitor();
    this->init_memory_monitor();
}

void
SearchEvalCase::init_latency_monitor() {
    latency_monitor_ = std::make_shared<LatencyMonitor>(this->dataset_ptr_->GetNumberOfQuery());
    if (config_.enable_qps) {
        latency_monitor_->SetMetrics("qps");
    }
    if (config_.enable_latency) {
        latency_monitor_->SetMetrics("avg_latency");
    }
    if (config_.enable_percent_latency) {
        latency_monitor_->SetMetrics("percent_latency");
    }
    this->monitors_.emplace_back(latency_monitor_);
}

void
SearchEvalCase::init_recall_monitor() {
    if (config_.enable_recall or config_.enable_percent_recall) {
        recall_monitor_ = std::make_shared<RecallMonitor>(this->dataset_ptr_->GetNumberOfQuery());
        if (config_.enable_recall) {
            recall_monitor_->SetMetrics("avg_recall");
        }
        if (config_.enable_percent_recall) {
            recall_monitor_->SetMetrics("percent_recall");
        }
        this->monitors_.emplace_back(recall_monitor_);
    }
}

void
SearchEvalCase::init_memory_monitor() {
    if (config_.enable_memory) {
        memory_monitor_ =
            std::make_shared<MemoryPeakMonitor>("search", memory_monitor_baseline_pages_);
        this->monitors_.emplace_back(memory_monitor_);
    }
}

JsonType
SearchEvalCase::Run() {
    return this->run_search_once();
}

JsonType
SearchEvalCase::RunWithSearchParam(const std::string& search_param) {
    config_.search_param = search_param;
    return this->run_search_once();
}

void
SearchEvalCase::LoadIndex() {
    if (index_loaded_) {
        return;
    }
    std::ifstream infile(this->index_path_, std::ios::binary);
    if (!infile.good()) {
        throw std::runtime_error("failed to open index path: " + this->index_path_);
    }
    this->deserialize(infile);
    index_loaded_ = true;
    ++index_deserialize_count_;
}

JsonType
SearchEvalCase::run_search_once() {
    this->reset_run_state();
    this->LoadIndex();
    ++search_run_ordinal_;
    switch (search_type_) {
        case KNN:
            this->do_knn_search();
            break;
        case KNN_FILTER:
            this->do_knn_filter_search();
            break;
    }
    auto result = this->process_result();
    if (config_.delete_index_after_search) {
        std::remove(this->index_path_.c_str());
    }
    return result;
}

void
SearchEvalCase::reset_run_state() {
    this->monitors_.clear();
    latency_monitor_.reset();
    recall_monitor_.reset();
    memory_monitor_.reset();
    this->init_monitor();
    statistics_query_count_.store(0);
    statistics_dist_cmp_.store(0);
    statistics_hops_.store(0);
    statistics_io_cnt_.store(0);
    statistics_io_time_ms_.store(0);
    statistics_reorder_distance_count_.store(0);
    statistics_reorder_lower_bound_probe_count_.store(0);
    statistics_rabitq_filter_count_.store(0);
    statistics_rabitq_full_count_.store(0);
    statistics_rabitq_filter_fallback_full_count_.store(0);
    statistics_rabitq_reorder_hint_full_count_.store(0);
    statistics_rabitq_reorder_fallback_full_count_.store(0);
    actual_concurrency_ = 0;
}

void
SearchEvalCase::deserialize(std::ifstream& infile) {
    auto result = this->index_->Deserialize(infile);
    if (not result.has_value()) {
        throw std::runtime_error("failed to deserialize index: " + result.error().message);
    }
    if (infile.bad()) {
        throw std::runtime_error("failed to deserialize index: index stream read failed");
    }
}

void
SearchEvalCase::do_knn_search() {
    this->do_knn_search(false);
}

void
SearchEvalCase::do_knn_search(bool use_filter) {
    const auto topk = static_cast<uint64_t>(config_.top_k);
    const auto requested_k = static_cast<int64_t>(config_.top_k);
    const auto query_count = static_cast<uint64_t>(this->dataset_ptr_->GetNumberOfQuery());
    const uint64_t run_query_count = get_run_query_count(query_count, config_);
    if (run_query_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("search query count exceeds the supported range");
    }
    this->logger_->Debug("query count is " + std::to_string(run_query_count));

    const auto train_labels = use_filter ? dataset_ptr_->GetTrainLabels() : nullptr;
    const auto test_labels = use_filter ? dataset_ptr_->GetTestLabels() : nullptr;
    if (use_filter && (train_labels == nullptr || test_labels == nullptr)) {
        throw std::invalid_argument("filtered search requires train_labels and test_labels");
    }
    const auto dataset = dataset_ptr_;
    const auto index = index_;
    const auto search_param = config_.search_param;
    const bool dense_vectors = dataset->GetVectorType() == DENSE_VECTORS;
    const bool float32_data = dataset->GetTestDataType() == vsag::DATATYPE_FLOAT32;
    const bool int8_data = dataset->GetTestDataType() == vsag::DATATYPE_INT8;
    const auto record_statistics = [this](const vsag::DatasetPtr& result) {
        this->record_statistics(result);
    };

    OmpThreadCountGuard thread_count_guard(config_.num_threads_searching);
    latency_monitor_->Start();
    if (recall_monitor_ != nullptr) {
        recall_monitor_->Start();
    }
    const bool retain_neighbors = recall_monitor_ != nullptr;
    const auto search_batch_size =
        get_search_batch_size(run_query_count,
                              topk,
                              static_cast<uint64_t>(config_.num_threads_searching),
                              retain_neighbors);
    std::vector<stored_search_result> stored_results(search_batch_size);
    std::vector<vsag::DatasetPtr> batch_queries(search_batch_size);
    std::vector<std::shared_ptr<FilterObj>> batch_filters;
    if (use_filter) {
        batch_filters.resize(search_batch_size);
    }
    double search_duration_seconds = 0.0;
    std::exception_ptr evaluation_exception;

    for (uint64_t batch_begin = 0; batch_begin < run_query_count;
         batch_begin += search_batch_size) {
        const auto batch_query_count = std::min(search_batch_size, run_query_count - batch_begin);
        for (uint64_t offset = 0; offset < batch_query_count; ++offset) {
            const uint64_t run_id = batch_begin + offset;
            const uint64_t query_id = run_id % query_count;
            const auto query_index = static_cast<int64_t>(query_id);
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dataset->GetDim())->Owner(false);
            const void* query_vector = dataset->GetOneTest(query_index);
            if (dense_vectors) {
                if (float32_data) {
                    query->Float32Vectors((const float*)query_vector);
                } else if (int8_data) {
                    query->Int8Vectors((const int8_t*)query_vector);
                }
            } else {
                query->SparseVectors((const SparseVector*)query_vector);
            }
            batch_queries[offset] = std::move(query);
            if (use_filter) {
                const auto test_label = test_labels[query_index];
                batch_filters[offset] = std::make_shared<FilterObj>(
                    train_labels, test_label, dataset->GetValidRatio(test_label));
            }
            stored_results[offset].result.reset();
            stored_results[offset].neighbors.clear();
            stored_results[offset].query_id = query_index;
        }
        std::atomic<bool> search_failed{false};
        std::exception_ptr search_exception;
        std::mutex search_exception_mutex;
        auto capture_search_exception = [&](std::exception_ptr exception) {
            std::lock_guard<std::mutex> lock(search_exception_mutex);
            if (not search_failed.load(std::memory_order_relaxed)) {
                search_exception = std::move(exception);
                search_failed.store(true, std::memory_order_release);
            }
        };

        uint64_t batch_concurrency = 0;
        std::chrono::steady_clock::time_point batch_search_start;
        std::chrono::steady_clock::time_point batch_search_end;
        if (memory_monitor_ != nullptr) {
            memory_monitor_->Start();
        }
#pragma omp parallel default(none) shared(batch_filters,            \
                                          batch_queries,            \
                                          batch_concurrency,        \
                                          batch_query_count,        \
                                          capture_search_exception, \
                                          index,                    \
                                          requested_k,              \
                                          search_failed,            \
                                          search_param,             \
                                          stored_results,           \
                                          batch_search_end,         \
                                          batch_search_start,       \
                                          use_filter)
        {
#pragma omp barrier
#pragma omp single
            {
                batch_concurrency = static_cast<uint64_t>(omp_get_num_threads());
                batch_search_start = std::chrono::steady_clock::now();
            }
#pragma omp for schedule(dynamic)
            for (int64_t offset = 0; offset < static_cast<int64_t>(batch_query_count); ++offset) {
                if (search_failed.load(std::memory_order_acquire)) {
                    continue;
                }
                try {
                    const auto result_offset = static_cast<uint64_t>(offset);
                    const auto query_start = std::chrono::steady_clock::now();
                    tl::expected<vsag::DatasetPtr, vsag::Error> result;
                    if (use_filter) {
                        result = index->KnnSearch(batch_queries[result_offset],
                                                  requested_k,
                                                  search_param,
                                                  batch_filters[result_offset]);
                    } else {
                        result = index->KnnSearch(
                            batch_queries[result_offset], requested_k, search_param);
                    }
                    const auto query_end = std::chrono::steady_clock::now();
                    const double query_duration_seconds =
                        std::chrono::duration<double>(query_end - query_start).count();
                    if (not result.has_value()) {
                        throw std::runtime_error("query error: " + result.error().message);
                    }
                    auto& stored = stored_results[result_offset];
                    stored.result = std::move(result.value());
                    stored.latency_ms = query_duration_seconds * 1000.0;
                } catch (...) {
                    capture_search_exception(std::current_exception());
                }
            }
#pragma omp single
            batch_search_end = std::chrono::steady_clock::now();
        }
        if (memory_monitor_ != nullptr) {
            memory_monitor_->Stop();
        }
        search_duration_seconds +=
            std::chrono::duration<double>(batch_search_end - batch_search_start).count();
        actual_concurrency_ = batch_concurrency;

        if (search_failed.load(std::memory_order_acquire)) {
            evaluation_exception = search_exception;
            break;
        }

        try {
            for (uint64_t offset = 0; offset < batch_query_count; ++offset) {
                auto& stored = stored_results[offset];
                if (stored.result == nullptr) {
                    throw std::runtime_error("query result is missing");
                }
                const int64_t returned_count = stored.result->GetDim();
                const uint64_t result_count =
                    returned_count > 0 ? std::min(static_cast<uint64_t>(returned_count), topk)
                                       : uint64_t{0};
                const int64_t* ids = stored.result->GetIds();
                if (result_count > 0 && ids == nullptr) {
                    throw std::runtime_error("query result contains no ids");
                }
                if (retain_neighbors && result_count > 0) {
                    const auto signed_result_count = static_cast<int64_t>(result_count);
                    stored.neighbors.assign(ids, ids + signed_result_count);
                }
                record_statistics(stored.result);

                SearchRecord latency_record;
                latency_record.latency_ms = stored.latency_ms;
                latency_monitor_->Record(&latency_record);

                if (recall_monitor_ != nullptr) {
                    SearchRecord recall_record;
                    recall_record.neighbors =
                        stored.neighbors.empty() ? nullptr : stored.neighbors.data();
                    recall_record.ground_truth_neighbors =
                        dataset_ptr_->GetNeighbors(stored.query_id);
                    recall_record.dataset = dataset_ptr_.get();
                    recall_record.query_data = dataset_ptr_->GetOneTest(stored.query_id);
                    recall_record.returned_count = stored.neighbors.size();
                    recall_record.requested_k = topk;
                    recall_record.ground_truth_count = dataset_ptr_->GetGroundTruthK();
                    recall_monitor_->Record(&recall_record);
                }
                stored.result.reset();
            }
        } catch (...) {
            evaluation_exception = std::current_exception();
            break;
        }
    }

    latency_monitor_->Stop(search_duration_seconds);
    if (recall_monitor_ != nullptr) {
        recall_monitor_->Stop();
    }
    if (evaluation_exception != nullptr) {
        std::rethrow_exception(evaluation_exception);
    }
}

void
SearchEvalCase::do_knn_filter_search() {
    this->do_knn_search(true);
}

JsonType
SearchEvalCase::process_result() {
    JsonType result;
    for (auto& monitor : this->monitors_) {
        const auto& one_result = monitor->GetResult();
        EvalCase::MergeJsonType(one_result, result);
    }
    result["action"] = "search";
    result["search_mode"] = config_.search_mode;
    result["index_info"] = JsonType::parse(config_.build_param);
    result["search_param"] = config_.search_param;
    result["index"] = config_.index_name;
    result["index_memory(B)"] = this->index_->GetMemoryUsage();
    try {
        auto detail = this->index_->GetMemoryUsageDetail();
        for (const auto& [name, size] : detail) {
            result["memory_detail(B)"][name] = size;
        }
    } catch (const std::exception& e) {
        result["memory_detail_error"] = e.what();
    }
    EvalCase::MergeJsonType(this->basic_info_, result);
    result["index_deserialize_count"] = index_deserialize_count_;
    result["search_run_ordinal"] = search_run_ordinal_;
    result["requested_concurrency"] = config_.num_threads_searching;
    result["actual_concurrency"] = actual_concurrency_;
    result["statistics_query_count"] = this->statistics_query_count_.load();
    result["statistics_total"] = this->statistics_total_json();
    result["statistics_avg_per_query"] = this->statistics_avg_json();
    return result;
}

namespace {

uint64_t
parse_stat_value(const std::string& value) {
    if (value.empty()) {
        return 0;
    }
    return static_cast<uint64_t>(std::strtoull(value.c_str(), nullptr, 10));
}

void
set_stat_value(JsonType& json, const char* key, uint64_t value) {
    json[key] = value;
}

void
set_avg_stat_value(JsonType& json, const char* key, uint64_t value, uint64_t count) {
    const float avg = count == 0 ? 0.0F : static_cast<float>(value) / static_cast<float>(count);
    json[key] = avg;
}

}  // namespace

void
SearchEvalCase::record_statistics(const vsag::DatasetPtr& result) {
    constexpr const char* dist_cmp_key = "dist_cmp";
    constexpr const char* hops_key = "hops";
    constexpr const char* io_cnt_key = "io_cnt";
    constexpr const char* io_time_ms_key = "io_time_ms";
    constexpr const char* reorder_distance_count_key = "reorder_distance_count";
    constexpr const char* reorder_lower_bound_probe_count_key = "reorder_lower_bound_probe_count";
    constexpr const char* rabitq_filter_count_key = "rabitq_filter_count";
    constexpr const char* rabitq_full_count_key = "rabitq_full_count";
    constexpr const char* rabitq_filter_fallback_full_count_key =
        "rabitq_filter_fallback_full_count";
    constexpr const char* rabitq_reorder_hint_full_count_key = "rabitq_reorder_hint_full_count";
    constexpr const char* rabitq_reorder_fallback_full_count_key =
        "rabitq_reorder_fallback_full_count";

    const auto values = result->GetStatistics({dist_cmp_key,
                                               hops_key,
                                               io_cnt_key,
                                               io_time_ms_key,
                                               reorder_distance_count_key,
                                               reorder_lower_bound_probe_count_key,
                                               rabitq_filter_count_key,
                                               rabitq_full_count_key,
                                               rabitq_filter_fallback_full_count_key,
                                               rabitq_reorder_hint_full_count_key,
                                               rabitq_reorder_fallback_full_count_key});
    this->statistics_dist_cmp_.fetch_add(parse_stat_value(values[0]), std::memory_order_relaxed);
    this->statistics_hops_.fetch_add(parse_stat_value(values[1]), std::memory_order_relaxed);
    this->statistics_io_cnt_.fetch_add(parse_stat_value(values[2]), std::memory_order_relaxed);
    this->statistics_io_time_ms_.fetch_add(parse_stat_value(values[3]), std::memory_order_relaxed);
    this->statistics_reorder_distance_count_.fetch_add(parse_stat_value(values[4]),
                                                       std::memory_order_relaxed);
    this->statistics_reorder_lower_bound_probe_count_.fetch_add(parse_stat_value(values[5]),
                                                                std::memory_order_relaxed);
    this->statistics_rabitq_filter_count_.fetch_add(parse_stat_value(values[6]),
                                                    std::memory_order_relaxed);
    this->statistics_rabitq_full_count_.fetch_add(parse_stat_value(values[7]),
                                                  std::memory_order_relaxed);
    this->statistics_rabitq_filter_fallback_full_count_.fetch_add(parse_stat_value(values[8]),
                                                                  std::memory_order_relaxed);
    this->statistics_rabitq_reorder_hint_full_count_.fetch_add(parse_stat_value(values[9]),
                                                               std::memory_order_relaxed);
    this->statistics_rabitq_reorder_fallback_full_count_.fetch_add(parse_stat_value(values[10]),
                                                                   std::memory_order_relaxed);
    this->statistics_query_count_.fetch_add(1, std::memory_order_relaxed);
}

JsonType
SearchEvalCase::statistics_total_json() const {
    JsonType json;
    set_stat_value(json, "dist_cmp", this->statistics_dist_cmp_.load());
    set_stat_value(json, "hops", this->statistics_hops_.load());
    set_stat_value(json, "io_cnt", this->statistics_io_cnt_.load());
    set_stat_value(json, "io_time_ms", this->statistics_io_time_ms_.load());
    set_stat_value(json, "reorder_distance_count", this->statistics_reorder_distance_count_.load());
    set_stat_value(json,
                   "reorder_lower_bound_probe_count",
                   this->statistics_reorder_lower_bound_probe_count_.load());
    set_stat_value(json, "rabitq_filter_count", this->statistics_rabitq_filter_count_.load());
    set_stat_value(json, "rabitq_full_count", this->statistics_rabitq_full_count_.load());
    set_stat_value(json,
                   "rabitq_filter_fallback_full_count",
                   this->statistics_rabitq_filter_fallback_full_count_.load());
    set_stat_value(json,
                   "rabitq_reorder_hint_full_count",
                   this->statistics_rabitq_reorder_hint_full_count_.load());
    set_stat_value(json,
                   "rabitq_reorder_fallback_full_count",
                   this->statistics_rabitq_reorder_fallback_full_count_.load());
    return json;
}

JsonType
SearchEvalCase::statistics_avg_json() const {
    JsonType json;
    const auto count = this->statistics_query_count_.load();
    set_avg_stat_value(json, "dist_cmp", this->statistics_dist_cmp_.load(), count);
    set_avg_stat_value(json, "hops", this->statistics_hops_.load(), count);
    set_avg_stat_value(json, "io_cnt", this->statistics_io_cnt_.load(), count);
    set_avg_stat_value(json, "io_time_ms", this->statistics_io_time_ms_.load(), count);
    set_avg_stat_value(
        json, "reorder_distance_count", this->statistics_reorder_distance_count_.load(), count);
    set_avg_stat_value(json,
                       "reorder_lower_bound_probe_count",
                       this->statistics_reorder_lower_bound_probe_count_.load(),
                       count);
    set_avg_stat_value(
        json, "rabitq_filter_count", this->statistics_rabitq_filter_count_.load(), count);
    set_avg_stat_value(
        json, "rabitq_full_count", this->statistics_rabitq_full_count_.load(), count);
    set_avg_stat_value(json,
                       "rabitq_filter_fallback_full_count",
                       this->statistics_rabitq_filter_fallback_full_count_.load(),
                       count);
    set_avg_stat_value(json,
                       "rabitq_reorder_hint_full_count",
                       this->statistics_rabitq_reorder_hint_full_count_.load(),
                       count);
    set_avg_stat_value(json,
                       "rabitq_reorder_fallback_full_count",
                       this->statistics_rabitq_reorder_fallback_full_count_.load(),
                       count);
    return json;
}

}  // namespace vsag::eval
