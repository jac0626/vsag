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

#include <omp.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <system_error>

#include "algorithm/hgraph/hgraph.h"
#include "autotune_internal.h"
#include "eval_config.h"
#include "evaluator.h"
#include "index/index_impl.h"
#include "vsag/factory.h"

namespace vsag::autotune::internal {

namespace {

constexpr double BYTES_PER_MEBIBYTE = 1024.0 * 1024.0;

class ScopedOpenMpThreads {
public:
    ScopedOpenMpThreads() : original_(omp_get_max_threads()) {
    }

    ~ScopedOpenMpThreads() {
        omp_set_num_threads(original_);
    }

private:
    int original_;
};

double
elapsed(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::optional<double>
number(const JsonType& value, const std::string& key) {
    if (!value.is_object() || !value.contains(key) || !value[key].is_number()) {
        return std::nullopt;
    }
    return value[key].get<double>();
}

void
set_metric(MetricMap& metrics, const std::string& name, const std::optional<double>& value) {
    if (value.has_value()) {
        metrics[name] = *value;
    }
}

MetricMap
build_metrics(const JsonType& raw, const std::string& index_path) {
    MetricMap metrics;
    set_metric(metrics, "build_seconds", number(raw, "duration(s)"));
    const auto memory = number(raw, "index_memory(B)");
    if (memory.has_value() && *memory > 0.0) {
        metrics["index_memory_mb"] = *memory / BYTES_PER_MEBIBYTE;
    }
    std::error_code error;
    const auto bytes = std::filesystem::file_size(index_path, error);
    if (!error) {
        metrics["index_size_mb"] = static_cast<double>(bytes) / BYTES_PER_MEBIBYTE;
    }
    return metrics;
}

MetricMap
projected_build_metrics(const MetricMap& source,
                        const IndexPtr& index,
                        const std::string& index_path,
                        double cumulative_projection_seconds) {
    auto metrics = source;
    const auto build = metrics.find("build_seconds");
    if (build != metrics.end()) {
        build->second += cumulative_projection_seconds;
    }
    metrics["index_memory_mb"] = static_cast<double>(index->GetMemoryUsage()) / BYTES_PER_MEBIBYTE;
    std::error_code error;
    const auto bytes = std::filesystem::file_size(index_path, error);
    if (!error) {
        metrics["index_size_mb"] = static_cast<double>(bytes) / BYTES_PER_MEBIBYTE;
    }
    return metrics;
}

MetricMap
search_metrics(const JsonType& raw, double seconds) {
    MetricMap metrics;
    set_metric(metrics, "recall_at_k", number(raw, "recall_avg"));
    set_metric(metrics, "latency_avg_ms", number(raw, "latency_avg(ms)"));
    set_metric(metrics, "qps", number(raw, "qps"));
    if (raw.contains("latency_detail(ms)") && raw["latency_detail(ms)"].is_object()) {
        set_metric(metrics, "latency_p99_ms", number(raw["latency_detail(ms)"], "p99"));
    }
    const auto memory = number(raw, "index_memory(B)");
    if (memory.has_value() && *memory > 0.0) {
        metrics["index_memory_mb"] = *memory / BYTES_PER_MEBIBYTE;
    }
    metrics["search_seconds"] = seconds;
    return metrics;
}

JsonType
metrics_json(const MetricMap& metrics) {
    JsonType result = JsonType::object();
    for (const auto& [name, value] : metrics) {
        result[name] = value;
    }
    return result;
}

eval::EvalConfig
build_config(const Candidate& candidate) {
    eval::EvalConfig config;
    config.index_name = candidate.index_name;
    config.build_param = candidate.create_params.dump();
    config.enable_memory = false;
    return config;
}

eval::EvalConfig
search_config(const RequestContext& request, const Candidate& candidate) {
    auto config = build_config(candidate);
    config.search_param = candidate.search_params.dump();
    config.search_mode = "knn";
    config.top_k = static_cast<int>(request.top_k);
    config.search_query_count = request.query_count;
    config.num_threads_searching = static_cast<int32_t>(request.concurrency);
    config.enable_memory = false;
    config.enable_recall = request.enable_recall;
    config.enable_percent_recall = false;
    return config;
}

IndexPtr
create_index(const Candidate& candidate) {
    auto created = Factory::CreateIndex(candidate.index_name, candidate.create_params.dump());
    if (!created.has_value()) {
        throw std::runtime_error(created.error().message);
    }
    return created.value();
}

void
serialize_index(const IndexPtr& index, const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    std::filesystem::create_directories(parent);
    std::ofstream output(path, std::ios::binary);
    if (!output.good()) {
        throw std::runtime_error("failed to open index artifact: " + path);
    }
    auto serialized = index->Serialize(output);
    if (!serialized.has_value()) {
        throw std::runtime_error(serialized.error().message);
    }
}

std::optional<uint32_t>
hgraph_max_degree(const Candidate& candidate) {
    if (candidate.index_name != "hgraph" || !candidate.create_params.is_object() ||
        !candidate.create_params.contains("index_param") ||
        !candidate.create_params["index_param"].is_object()) {
        return std::nullopt;
    }
    const auto& params = candidate.create_params["index_param"];
    if (!params.contains("max_degree") || !params["max_degree"].is_number_integer()) {
        return std::nullopt;
    }
    uint64_t degree = 0;
    if (params["max_degree"].is_number_unsigned()) {
        degree = params["max_degree"].get<uint64_t>();
    } else {
        const auto signed_degree = params["max_degree"].get<int64_t>();
        if (signed_degree <= 0) {
            return std::nullopt;
        }
        degree = static_cast<uint64_t>(signed_degree);
    }
    if (degree == 0 || degree > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(degree);
}

std::string
build_group_key(const Candidate& candidate) {
    auto create_params = candidate.create_params;
    const auto degree = hgraph_max_degree(candidate);
    if (degree.has_value()) {
        create_params["index_param"].erase("max_degree");
        return "hgraph_degree\n" + create_params.dump();
    }
    return candidate.index_name + "\n" + create_params.dump();
}

std::shared_ptr<HGraph>
get_hgraph(const IndexPtr& index) {
    auto impl = std::dynamic_pointer_cast<IndexImpl<HGraph>>(index);
    if (impl == nullptr) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<HGraph>(impl->GetInnerIndex());
}

bool
uses_native_build_cost(const RequestContext& request) {
    return request.objective == "build_seconds" ||
           request.objective == "build_and_search_seconds" ||
           request.constraints.find("build_seconds") != request.constraints.end() ||
           request.constraints.find("build_and_search_seconds") != request.constraints.end();
}

}  // namespace

void
EvaluateEfSearchRange(const HGraphEfSearchRange& range,
                      double recall_target,
                      const std::function<std::optional<double>(int64_t)>& evaluate) {
    auto low = range.start;
    const auto low_recall = evaluate(low);
    if (!low_recall.has_value() || *low_recall >= recall_target || low == range.stop) {
        return;
    }

    auto high = low;
    while (low < range.stop) {
        high = low > range.stop / 2 ? range.stop : low * 2;
        const auto high_recall = evaluate(high);
        if (!high_recall.has_value()) {
            return;
        }
        if (*high_recall >= recall_target) {
            break;
        }
        if (high == range.stop) {
            return;
        }
        low = high;
    }

    while (high - low > 1) {
        const auto middle = low + (high - low) / 2;
        const auto middle_recall = evaluate(middle);
        if (!middle_recall.has_value()) {
            return;
        }
        if (*middle_recall >= recall_target) {
            high = middle;
        } else {
            low = middle;
        }
    }
}

Evaluation
EvaluateCandidates(const IndexTuningRequest& tuning_request,
                   const std::vector<Candidate>& candidates,
                   const std::string& run_path) {
    const auto& request = tuning_request.context;
    ScopedOpenMpThreads openmp_threads;
    Evaluation evaluation;
    std::map<std::string, std::vector<uint64_t>> groups;
    const auto reuse_hgraph_degrees = !uses_native_build_cost(request);
    for (uint64_t i = 0; i < candidates.size(); ++i) {
        const auto key = reuse_hgraph_degrees
                             ? build_group_key(candidates[i])
                             : candidates[i].index_name + "\n" + candidates[i].create_params.dump();
        groups[key].emplace_back(i);
    }

    uint64_t build_number = 0;
    uint64_t trial_number = 0;

    struct build_state {
        IndexPtr index;
        JsonType report;
        MetricMap metrics;
        std::string path;
    };

    const auto artifact_path = [&run_path](const std::string& build_id) {
        return (std::filesystem::path(run_path) / "artifacts" / (build_id + ".index")).string();
    };
    const auto cleanup = [&request](const std::string& path) {
        if (request.keep_intermediate) {
            return;
        }
        std::error_code error;
        std::filesystem::remove(path, error);
    };

    const auto build_index = [&](const Candidate& candidate) {
        const auto build_id = "build-" + std::to_string(build_number++);
        const auto index_path = artifact_path(build_id);
        build_state state{nullptr,
                          {{"build_id", build_id},
                           {"index_name", candidate.index_name},
                           {"create_params", candidate.create_params},
                           {"strategy", "full_build"},
                           {"status", "failed"},
                           {"metrics", JsonType::object()},
                           {"failure", nullptr},
                           {"artifacts",
                            {{"index_path", index_path},
                             {"source", "generated"},
                             {"use_existing_index", false},
                             {"retained", request.keep_intermediate}}}},
                          {},
                          index_path};

        const auto start = std::chrono::steady_clock::now();
        try {
            state.index = create_index(candidate);
            auto raw = eval::EvaluateBuild(state.index, request.dataset, build_config(candidate));
            serialize_index(state.index, index_path);
            state.metrics = build_metrics(raw, index_path);
            if (request.include_raw_eval) {
                state.report["raw_eval_result"] = std::move(raw);
            }
            state.report["metrics"] = metrics_json(state.metrics);
            state.report["status"] = "success";
        } catch (const std::exception& error) {
            state.report["failure"] = Failure("build", "build_evaluation_failed", error.what());
            std::error_code cleanup_error;
            std::filesystem::remove(index_path, cleanup_error);
            state.report["artifacts"]["retained"] = false;
            state.index.reset();
        }
        state.report["elapsed_seconds"] = elapsed(start);
        evaluation.builds.emplace_back(state.report);
        return state;
    };

    const auto record_projection = [&](const Candidate& candidate,
                                       const IndexPtr& index,
                                       const build_state& source,
                                       double cumulative_projection_seconds,
                                       double projection_step_seconds) {
        const auto build_id = "build-" + std::to_string(build_number++);
        const auto index_path = artifact_path(build_id);
        build_state state{index,
                          {{"build_id", build_id},
                           {"source_build_id", source.report["build_id"]},
                           {"index_name", candidate.index_name},
                           {"create_params", candidate.create_params},
                           {"strategy", "degree_projection"},
                           {"projection_seconds", cumulative_projection_seconds},
                           {"status", "failed"},
                           {"metrics", JsonType::object()},
                           {"failure", nullptr},
                           {"artifacts",
                            {{"index_path", index_path},
                             {"source", "degree_projection"},
                             {"use_existing_index", false},
                             {"retained", request.keep_intermediate}}}},
                          {},
                          index_path};

        const auto start = std::chrono::steady_clock::now();
        try {
            serialize_index(index, index_path);
            state.metrics = projected_build_metrics(
                source.metrics, index, index_path, cumulative_projection_seconds);
            state.report["metrics"] = metrics_json(state.metrics);
            state.report["status"] = "success";
        } catch (...) {
            std::error_code cleanup_error;
            std::filesystem::remove(index_path, cleanup_error);
            throw;
        }
        state.report["elapsed_seconds"] = projection_step_seconds + elapsed(start);
        evaluation.builds.emplace_back(state.report);
        return state;
    };

    const auto evaluate_searches = [&](const build_state& build,
                                       const std::vector<uint64_t>& indexes) {
        const auto evaluate = [&](const Candidate& candidate) -> std::optional<double> {
            const auto trial_id = "trial-" + std::to_string(trial_number++);
            JsonType trial{{"trial_id", trial_id},
                           {"build_id", build.report["build_id"]},
                           {"index_name", candidate.index_name},
                           {"create_params", candidate.create_params},
                           {"search_params", candidate.search_params},
                           {"status", "failed"},
                           {"metrics", metrics_json(build.metrics)},
                           {"failure", nullptr},
                           {"artifacts", build.report["artifacts"]}};
            std::optional<double> recall;
            const auto search_start = std::chrono::steady_clock::now();
            if (build.report["status"] != "success") {
                trial["failure"] =
                    Failure("search", "build_failed", "search skipped because build failed");
            } else {
                try {
                    const auto measured_start = std::chrono::steady_clock::now();
                    auto raw = eval::EvaluateSearch(
                        build.index, request.dataset, search_config(request, candidate));
                    auto metrics = search_metrics(raw, elapsed(measured_start));
                    for (const auto& [name, value] : build.metrics) {
                        metrics.emplace(name, value);
                    }
                    if (metrics.find("build_seconds") != metrics.end()) {
                        metrics["build_and_search_seconds"] =
                            metrics["build_seconds"] + metrics["search_seconds"];
                    }
                    trial["metrics"] = metrics_json(metrics);
                    trial["status"] = "success";
                    recall = number(trial["metrics"], "recall_at_k");
                    if (request.include_raw_eval) {
                        trial["raw_eval_result"] = std::move(raw);
                    }
                } catch (const std::exception& error) {
                    trial["failure"] = Failure("search", "search_evaluation_failed", error.what());
                }
            }
            trial["elapsed_seconds"] = elapsed(search_start);
            evaluation.trials.emplace_back(std::move(trial));
            return recall;
        };

        for (const auto candidate_index : indexes) {
            const auto& candidate = candidates[candidate_index];
            if (!candidate.ef_search_range.has_value()) {
                evaluate(candidate);
                continue;
            }

            const auto recall_target = request.constraints.at("recall_at_k");
            const auto evaluate_ef_search = [&](int64_t ef_search) {
                auto concrete = candidate;
                concrete.search_params["hgraph"]["ef_search"] = ef_search;
                return evaluate(concrete);
            };
            EvaluateEfSearchRange(*candidate.ef_search_range, recall_target, evaluate_ef_search);
        }
    };

    const auto evaluate_full_group = [&](const std::vector<uint64_t>& indexes) {
        auto state = build_index(candidates[indexes.front()]);
        evaluate_searches(state, indexes);
        cleanup(state.path);
        return state;
    };

    for (const auto& [unused, indexes] : groups) {
        (void)unused;
        std::map<uint32_t, std::vector<uint64_t>, std::greater<>> degree_groups;
        bool all_have_degrees = true;
        for (const auto candidate_index : indexes) {
            const auto degree = hgraph_max_degree(candidates[candidate_index]);
            if (!degree.has_value()) {
                all_have_degrees = false;
                break;
            }
            degree_groups[*degree].emplace_back(candidate_index);
        }
        if (!all_have_degrees || degree_groups.size() <= 1) {
            evaluate_full_group(indexes);
            continue;
        }

        auto degree = degree_groups.begin();
        auto source = build_index(candidates[degree->second.front()]);
        evaluate_searches(source, degree->second);
        cleanup(source.path);

        auto hgraph = get_hgraph(source.index);
        bool can_project = source.report["status"] == "success" && hgraph != nullptr &&
                           hgraph->CanReduceMaxDegree();
        bool prepared = false;
        double projection_seconds = 0.0;
        for (++degree; degree != degree_groups.end(); ++degree) {
            if (!can_project) {
                evaluate_full_group(degree->second);
                continue;
            }

            try {
                const auto start = std::chrono::steady_clock::now();
                if (!prepared) {
                    hgraph->PrepareDegreeReduction();
                    prepared = true;
                }
                hgraph->ReduceMaxDegree(degree->first);
                const auto projection_step_seconds = elapsed(start);
                projection_seconds += projection_step_seconds;
                auto projected = record_projection(candidates[degree->second.front()],
                                                   source.index,
                                                   source,
                                                   projection_seconds,
                                                   projection_step_seconds);
                evaluate_searches(projected, degree->second);
                cleanup(projected.path);
            } catch (const std::exception&) {
                can_project = false;
                evaluate_full_group(degree->second);
            }
        }
    }
    return evaluation;
}

Evaluation
EvaluateCandidates(const SearchTuningRequest& tuning_request,
                   const std::vector<Candidate>& candidates) {
    ScopedOpenMpThreads openmp_threads;
    const auto& request = tuning_request.context;
    Evaluation evaluation;
    uint64_t trial_number = 0;

    const auto evaluate = [&](const Candidate& candidate) -> std::optional<double> {
        JsonType trial{{"trial_id", "trial-" + std::to_string(trial_number++)},
                       {"index_name", candidate.index_name},
                       {"search_params", candidate.search_params},
                       {"status", "failed"},
                       {"metrics", JsonType::object()},
                       {"failure", nullptr}};
        std::optional<double> recall;
        const auto start = std::chrono::steady_clock::now();
        try {
            const auto measured_start = std::chrono::steady_clock::now();
            auto raw = eval::EvaluateSearch(
                tuning_request.index, request.dataset, search_config(request, candidate));
            const auto metrics = search_metrics(raw, elapsed(measured_start));
            trial["metrics"] = metrics_json(metrics);
            trial["status"] = "success";
            recall = number(trial["metrics"], "recall_at_k");
            if (request.include_raw_eval) {
                trial["raw_eval_result"] = std::move(raw);
            }
        } catch (const std::exception& error) {
            trial["failure"] = Failure("search", "search_evaluation_failed", error.what());
        }
        trial["elapsed_seconds"] = elapsed(start);
        evaluation.trials.emplace_back(std::move(trial));
        return recall;
    };

    for (const auto& candidate : candidates) {
        if (!candidate.ef_search_range.has_value()) {
            evaluate(candidate);
            continue;
        }
        const auto recall_target = request.constraints.at("recall_at_k");
        const auto evaluate_ef_search = [&](int64_t ef_search) {
            auto concrete = candidate;
            concrete.search_params["hgraph"]["ef_search"] = ef_search;
            return evaluate(concrete);
        };
        EvaluateEfSearchRange(*candidate.ef_search_range, recall_target, evaluate_ef_search);
    }
    return evaluation;
}

}  // namespace vsag::autotune::internal
