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

#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

#include "autotune_candidate.h"
#include "autotune_environment.h"
#include "autotune_internal.h"
#include "autotune_planner.h"
#include "autotune_request.h"
#include "eval_dataset.h"

namespace vsag::autotune {

namespace {

class Hdf5ErrorSilencer {
public:
    Hdf5ErrorSilencer() {
        if (H5Eget_auto2(H5E_DEFAULT, &handler_, &client_data_) >= 0) {
            active_ = H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) >= 0;
        }
    }

    ~Hdf5ErrorSilencer() {
        if (active_) {
            H5Eset_auto2(H5E_DEFAULT, handler_, client_data_);
        }
    }

    Hdf5ErrorSilencer(const Hdf5ErrorSilencer&) = delete;
    Hdf5ErrorSilencer&
    operator=(const Hdf5ErrorSilencer&) = delete;

private:
    H5E_auto2_t handler_{nullptr};
    void* client_data_{nullptr};
    bool active_{false};
};

class PhaseTimer {
public:
    PhaseTimer(JsonType& elapsed_breakdown, const char* phase)
        : elapsed_breakdown_(elapsed_breakdown), phase_(phase), start_(internal::Clock::now()) {
    }

    ~PhaseTimer() noexcept {
        try {
            elapsed_breakdown_[phase_] = internal::ElapsedSeconds(start_);
        } catch (...) {
        }
    }

    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer&
    operator=(const PhaseTimer&) = delete;

private:
    JsonType& elapsed_breakdown_;
    const char* phase_;
    internal::Clock::time_point start_;
};

class RunWorkspace {
public:
    RunWorkspace(const std::string& base_path, bool keep_intermediate)
        : keep_intermediate_(keep_intermediate) {
        const auto runs_path = std::filesystem::path(base_path) / "runs";
        std::error_code error;
        std::filesystem::create_directories(runs_path, error);
        internal::Require(!error, "failed to create AutoTune runs directory: " + error.message());

        static std::atomic<uint64_t> next_run_ordinal{0};
        for (uint64_t attempt = 0; attempt < 100; ++attempt) {
            const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            const auto ordinal = next_run_ordinal.fetch_add(1, std::memory_order_relaxed);
            run_id_ = "run-" + std::to_string(now) + "-" + std::to_string(ordinal);
            path_ = (runs_path / run_id_).string();

            error.clear();
            if (std::filesystem::create_directory(path_, error)) {
                return;
            }
            internal::Require(!error || error == std::errc::file_exists,
                              "failed to create AutoTune run workspace: " + error.message());
        }
        internal::Require(false, "failed to allocate a unique AutoTune run workspace");
    }

    ~RunWorkspace() {
        if (keep_intermediate_ || path_.empty()) {
            return;
        }
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    RunWorkspace(const RunWorkspace&) = delete;
    RunWorkspace&
    operator=(const RunWorkspace&) = delete;

    [[nodiscard]] const std::string&
    Id() const {
        return run_id_;
    }

    [[nodiscard]] const std::string&
    Path() const {
        return path_;
    }

private:
    bool keep_intermediate_{false};
    std::string run_id_;
    std::string path_;
};

std::string
resolve_metric_type(const std::string& dataset_metric) {
    if (dataset_metric == "euclidean") {
        return "l2";
    }
    if (dataset_metric == "angular") {
        return "cosine";
    }
    if (dataset_metric == "ip") {
        return "ip";
    }
    internal::Require(false, "unsupported dataset metric: " + dataset_metric);
    return "";
}

internal::DatasetDescription
describe_dataset(const eval::EvalDatasetPtr& dataset) {
    internal::Require(dataset != nullptr, "failed to load evaluation dataset");
    internal::Require(dataset->GetDim() > 0, "dataset dim must be positive");
    internal::Require(dataset->GetNumberOfQuery() > 0, "dataset must contain queries");

    internal::DatasetDescription description;
    description.dim = static_cast<uint64_t>(dataset->GetDim());
    description.dtype = dataset->GetTrainDataType();
    description.metric_type = resolve_metric_type(dataset->GetMetric());
    description.base_count = static_cast<uint64_t>(dataset->GetNumberOfBase());
    description.query_count = static_cast<uint64_t>(dataset->GetNumberOfQuery());
    description.ground_truth_k = dataset->GetGroundTruthK();
    description.vector_type = dataset->GetVectorType();
    return description;
}

eval::EvalDatasetPtr
load_evaluation_dataset(const std::string& path) {
    Hdf5ErrorSilencer error_silencer;
    try {
        return eval::EvalDataset::Load(path);
    } catch (const H5::Exception& error) {
        throw std::runtime_error("failed to load evaluation dataset: " + error.getDetailMsg());
    }
}

JsonType
objective_to_json(const internal::ObjectiveSpec& objective) {
    return JsonType{{"metric", objective.metric}};
}

JsonType
evaluation_strategy_evidence() {
    return JsonType{{"name", "full_grid"},
                    {"query_coverage", "full_dataset"},
                    {"artifact_reuse", "build_file_only"},
                    {"index_instance_reuse", false}};
}

std::string
failure_code_for_stage(const std::string& stage) {
    if (stage == "validation") {
        return "invalid_request";
    }
    if (stage == "workspace") {
        return "workspace_initialization_failed";
    }
    if (stage == "candidate_generation") {
        return "candidate_generation_failed";
    }
    if (stage == "planning") {
        return "trial_planning_failed";
    }
    if (stage == "evaluation") {
        return "evaluation_failed";
    }
    if (stage == "selection") {
        return "selection_failed";
    }
    if (stage == "report") {
        return "report_write_failed";
    }
    return "autotune_failed";
}

}  // namespace

JsonType
RunAutoTune(const JsonType& input_request) {
    const auto total_start = internal::Clock::now();
    std::string stage = "validation";
    JsonType elapsed_breakdown = JsonType::object();
    JsonType environment = JsonType::object();
    std::optional<internal::AutoTuneRequest> request;
    std::vector<internal::CandidateSpec> candidates;
    std::unique_ptr<RunWorkspace> workspace;
    internal::AutoTunePlan plan;
    internal::EvaluationResult evaluation;
    JsonType selection;

    try {
        {
            PhaseTimer timer(elapsed_breakdown, "validation");
            request = internal::ParseAutoTuneRequest(input_request);
            const auto dataset = load_evaluation_dataset(request->data_path);
            internal::ResolveAutoTuneRequest(*request, describe_dataset(dataset));
            environment = internal::CollectEnvironmentEvidence();
        }

        stage = "workspace";
        {
            PhaseTimer timer(elapsed_breakdown, "workspace");
            workspace = std::make_unique<RunWorkspace>(request->tuning_config.workspace_path,
                                                       request->tuning_config.keep_intermediate);
        }

        stage = "candidate_generation";
        {
            PhaseTimer timer(elapsed_breakdown, "candidate_generation");
            candidates = internal::GenerateCandidates(*request);
        }

        stage = "planning";
        {
            PhaseTimer timer(elapsed_breakdown, "planning");
            plan = internal::PlanTrials(*request, candidates, workspace->Path());
        }

        stage = "evaluation";
        {
            PhaseTimer timer(elapsed_breakdown, "evaluation");
            evaluation = internal::EvaluatePlan(*request, plan);
        }

        stage = "selection";
        {
            PhaseTimer timer(elapsed_breakdown, "selection");
            selection = internal::SelectResult(*request, evaluation);
        }

        JsonType result{
            {"version", 1},
            {"run_id", workspace->Id()},
            {"run_workspace_path", workspace->Path()},
            {"report_path",
             request->output.result_path.empty() ? JsonType(nullptr)
                                                 : JsonType(request->output.result_path)},
            {"input_request", input_request},
            {"effective_request", request->effective_request},
            {"environment", environment},
            {"evaluation_strategy", evaluation_strategy_evidence()},
            {"objective", objective_to_json(request->objective)},
            {"status", selection["status"]},
            {"elapsed_breakdown_seconds", elapsed_breakdown},
            {"recommendation", selection["recommendation"]},
            {"best_effort", selection["best_effort"]},
            {"trial_count", evaluation.trial_results.size()},
            {"build_count", evaluation.executed_build_count},
            {"build_group_count", evaluation.build_results.size()},
            {"failure", selection["failure"]},
            {"builds",
             internal::MakeBuildReport(evaluation.build_results, request->output.include_raw_eval)},
            {"trials",
             internal::MakeTrialReport(
                 evaluation.trial_results, request->workload, request->output.include_raw_eval)}};
        result["elapsed_seconds"] = internal::ElapsedSeconds(total_start);

        stage = "report";
        try {
            internal::WriteJsonFile(request->output.result_path, result);
        } catch (const std::exception& error) {
            result["status"] = "failed";
            result["recommendation"] = nullptr;
            result["best_effort"] = nullptr;
            result["report_path"] = nullptr;
            result["failure"] = JsonType{
                {"stage", "report"}, {"code", "report_write_failed"}, {"message", error.what()}};
        }
        return result;
    } catch (const std::exception& error) {
        auto result = internal::MakeFailedResult(
            input_request, stage, failure_code_for_stage(stage), error.what(), total_start);
        result["elapsed_breakdown_seconds"] = elapsed_breakdown;
        result["environment"] = environment;
        result["evaluation_strategy"] = evaluation_strategy_evidence();
        if (workspace != nullptr) {
            result["run_id"] = workspace->Id();
            result["run_workspace_path"] = workspace->Path();
        }
        if (request.has_value()) {
            result["effective_request"] = request->effective_request;
            result["objective"] = objective_to_json(request->objective);
        }
        const auto result_path = request.has_value() ? request->output.result_path : std::string();
        result["report_path"] = result_path.empty() ? JsonType(nullptr) : JsonType(result_path);
        if (!result_path.empty()) {
            try {
                internal::WriteJsonFile(result_path, result);
            } catch (const std::exception& write_error) {
                result["report_path"] = nullptr;
                result["failure"]["result_write_error"] = write_error.what();
            }
        }
        return result;
    }
}

std::string
RunAutoTune(const std::string& request_json) {
    const auto total_start = internal::Clock::now();
    JsonType request;
    try {
        request = JsonType::parse(request_json);
    } catch (const JsonType::parse_error& error) {
        return internal::MakeFailedResult(
                   JsonType(nullptr),
                   "validation",
                   "invalid_request",
                   "request_json is not valid JSON: " + std::string(error.what()),
                   total_start)
            .dump();
    }
    return RunAutoTune(request).dump();
}

}  // namespace vsag::autotune
