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

#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tuning/auto_tuning_api.h"

namespace {

struct PocOptions {
    std::string dataset_path;
    std::string json_output_path;
    std::string request_json_path;
    std::string source_type = "existing_index";
    uint64_t base_count = 0;
    uint64_t query_count = 0;
    uint64_t max_trials = 0;
    uint64_t topk = 10;
    double target_recall = 0.80;
    bool source_type_explicit = false;
    bool query_count_explicit = false;
    bool max_trials_explicit = false;
    bool topk_explicit = false;
    bool target_recall_explicit = false;
    bool request_evaluates_all_queries = false;
};

struct DatasetBundle {
    vsag::DatasetPtr base;
    vsag::DatasetPtr queries;
    std::vector<int64_t> ids;
    std::vector<float> base_vectors;
    std::vector<float> query_vectors;
    uint64_t dim = 0;
    std::string source;
};

double
ElapsedMs(std::chrono::steady_clock::time_point started_at) {
    const auto elapsed = std::chrono::steady_clock::now() - started_at;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

void
PrintUsage(const char* binary) {
    std::cout << "Usage:\n"
              << "  " << binary << "\n"
              << "  " << binary
              << " --dataset /root/data/sift-128-euclidean.hdf5"
                 " --base-count 10000 --query-count 100 --target-recall 0.90"
                 " --json-output /tmp/hgraph_auto_tuning_report.json\n"
              << "  " << binary
              << " --source-type raw_dataset"
                 " --json-output /tmp/hgraph_auto_tuning_raw_report.json\n"
              << "  " << binary << " --max-trials 2\n"
              << "  " << binary
              << " --request-json src/tuning/examples/hgraph_auto_tuning_max_trials_request.json"
                 " --json-output /tmp/hgraph_auto_tuning_request_report.json\n";
    std::cout << "  " << binary
              << " --request-json src/tuning/examples/hgraph_auto_tuning_rebuild_request.json"
                 " --json-output /tmp/hgraph_auto_tuning_rebuild_report.json\n";
}

uint64_t
ParseUint64(const std::string& value, const std::string& name) {
    const auto parsed = std::stoull(value);
    if (parsed == 0) {
        throw std::invalid_argument(name + " must be greater than 0");
    }
    return parsed;
}

PocOptions
ParseOptions(int argc, char** argv) {
    PocOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(name + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--dataset") {
            options.dataset_path = require_value(arg);
        } else if (arg == "--json-output") {
            options.json_output_path = require_value(arg);
        } else if (arg == "--request-json") {
            options.request_json_path = require_value(arg);
        } else if (arg == "--source-type") {
            options.source_type = require_value(arg);
            options.source_type_explicit = true;
        } else if (arg == "--base-count") {
            options.base_count = ParseUint64(require_value(arg), arg);
        } else if (arg == "--query-count") {
            options.query_count = ParseUint64(require_value(arg), arg);
            options.query_count_explicit = true;
        } else if (arg == "--max-trials") {
            options.max_trials = ParseUint64(require_value(arg), arg);
            options.max_trials_explicit = true;
        } else if (arg == "--topk") {
            options.topk = ParseUint64(require_value(arg), arg);
            options.topk_explicit = true;
        } else if (arg == "--target-recall") {
            options.target_recall = std::stod(require_value(arg));
            options.target_recall_explicit = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    return options;
}

void
FinalizeOptions(PocOptions& options) {
    if (options.dataset_path.empty()) {
        if (options.base_count == 0) {
            options.base_count = 200;
        }
        if (options.query_count == 0) {
            options.query_count = 8;
        }
    } else {
        if (options.base_count == 0) {
            options.base_count = 10000;
        }
        if (options.query_count == 0 && not options.request_evaluates_all_queries) {
            options.query_count = 100;
        }
    }
    if (options.source_type != "existing_index" && options.source_type != "raw_dataset") {
        if (options.request_json_path.empty()) {
            throw std::invalid_argument("--source-type must be existing_index or raw_dataset");
        }
        throw std::invalid_argument("$.source.type must be existing_index or raw_dataset");
    }
}

void
ValidateRequestJsonOptions(const PocOptions& options) {
    if (options.source_type_explicit || options.query_count_explicit ||
        options.max_trials_explicit || options.topk_explicit || options.target_recall_explicit) {
        throw std::invalid_argument(
            "--request-json cannot be combined with --source-type, "
            "--query-count, --max-trials, --topk, or --target-recall");
    }
}

std::string
ReadTextFile(const std::string& path) {
    std::ifstream input(path);
    if (not input.is_open()) {
        throw std::runtime_error("failed to open request json path: " + path);
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

bool
ReadUint64Value(const nlohmann::json& value, uint64_t& output) {
    if (value.is_number_unsigned()) {
        output = value.get<uint64_t>();
        return true;
    }
    if (value.is_number_integer()) {
        const auto signed_value = value.get<int64_t>();
        if (signed_value < 0) {
            return false;
        }
        output = static_cast<uint64_t>(signed_value);
        return true;
    }
    return false;
}

uint64_t
ReadRequestUint64(const nlohmann::json& parent, const std::string& key, const std::string& path) {
    uint64_t value = 0;
    if (not parent.contains(key) || not ReadUint64Value(parent.at(key), value)) {
        throw std::invalid_argument(path + "." + key + " must be a uint64 in --request-json");
    }
    return value;
}

void
SetStringFromRequest(std::string& target,
                     bool was_explicit,
                     const std::string& value,
                     const std::string& field) {
    if (was_explicit && target != value) {
        throw std::invalid_argument("--" + field + " conflicts with --request-json " + field);
    }
    target = value;
}

void
SetUint64FromRequest(uint64_t& target,
                     bool was_explicit,
                     uint64_t value,
                     const std::string& field) {
    if (was_explicit && target != value) {
        throw std::invalid_argument("--" + field + " conflicts with --request-json " + field);
    }
    target = value;
}

void
SetDoubleFromRequest(double& target, bool was_explicit, double value, const std::string& field) {
    if (was_explicit && target != value) {
        throw std::invalid_argument("--" + field + " conflicts with --request-json " + field);
    }
    target = value;
}

void
ApplyRequestJsonToOptions(const std::string& request_json, PocOptions& options) {
    nlohmann::json request;
    try {
        request = nlohmann::json::parse(request_json);
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("failed to parse --request-json: ") + e.what());
    }
    if (not request.is_object()) {
        throw std::invalid_argument("--request-json must contain a json object");
    }

    if (not request.contains("source") || not request.at("source").is_object() ||
        not request.at("source").contains("type") ||
        not request.at("source").at("type").is_string()) {
        throw std::invalid_argument("--request-json must contain source.type");
    }
    SetStringFromRequest(options.source_type,
                         options.source_type_explicit,
                         request.at("source").at("type").get<std::string>(),
                         "source-type");

    if (not request.contains("workload") || not request.at("workload").is_object()) {
        throw std::invalid_argument("--request-json must contain workload.topk");
    }
    const auto topk = ReadRequestUint64(request.at("workload"), "topk", "$.workload");
    if (topk == 0) {
        throw std::invalid_argument("$.workload.topk must be greater than 0 in --request-json");
    }
    SetUint64FromRequest(options.topk, options.topk_explicit, topk, "topk");

    if (request.contains("evaluation") && request.at("evaluation").is_object() &&
        request.at("evaluation").contains("query_count")) {
        const auto query_count =
            ReadRequestUint64(request.at("evaluation"), "query_count", "$.evaluation");
        if (query_count == 0) {
            options.request_evaluates_all_queries = true;
        } else {
            SetUint64FromRequest(
                options.query_count, options.query_count_explicit, query_count, "query-count");
        }
    } else {
        options.request_evaluates_all_queries = true;
    }

    if (request.contains("objective") && request.at("objective").is_object() &&
        request.at("objective").contains("recall_at_k") &&
        request.at("objective").at("recall_at_k").is_object() &&
        request.at("objective").at("recall_at_k").contains("min")) {
        const auto& target = request.at("objective").at("recall_at_k").at("min");
        if (not target.is_number()) {
            throw std::invalid_argument("$.objective.recall_at_k.min must be a number");
        }
        SetDoubleFromRequest(options.target_recall,
                             options.target_recall_explicit,
                             target.get<double>(),
                             "target-recall");
    }

    if (request.contains("budget") && request.at("budget").is_object() &&
        request.at("budget").contains("max_trials")) {
        const auto max_trials = ReadRequestUint64(request.at("budget"), "max_trials", "$.budget");
        SetUint64FromRequest(
            options.max_trials, options.max_trials_explicit, max_trials, "max-trials");
    }
}

const char*
StageName(vsag::TuningStage stage) {
    switch (stage) {
        case vsag::TuningStage::WORKLOAD_VALIDATION:
            return "workload_validation";
        case vsag::TuningStage::SEARCH_SPACE_CONSTRUCTION:
            return "search_space_construction";
        case vsag::TuningStage::BUILD_PARAMETER_TUNING:
            return "build_parameter_tuning";
        case vsag::TuningStage::QUANTIZER_TUNING:
            return "quantizer_tuning";
        case vsag::TuningStage::CANDIDATE_GENERATION:
            return "candidate_generation";
        case vsag::TuningStage::CANDIDATE_VALIDATION:
            return "candidate_validation";
        case vsag::TuningStage::CANDIDATE_PRUNING:
            return "candidate_pruning";
        case vsag::TuningStage::TRIAL_PLANNING:
            return "trial_planning";
        case vsag::TuningStage::TRIAL_EXECUTION:
            return "trial_execution";
        case vsag::TuningStage::SELECTION:
            return "selection";
    }
    return "unknown";
}

const char*
StageStatusName(vsag::TuningStageStatus status) {
    switch (status) {
        case vsag::TuningStageStatus::COMPLETED:
            return "completed";
        case vsag::TuningStageStatus::SKIPPED:
            return "skipped";
        case vsag::TuningStageStatus::FAILED:
            return "failed";
    }
    return "unknown";
}

const char*
TrialStatusName(vsag::TuningTrialStatus status) {
    switch (status) {
        case vsag::TuningTrialStatus::COMPLETED:
            return "completed";
        case vsag::TuningTrialStatus::SKIPPED:
            return "skipped";
        case vsag::TuningTrialStatus::FAILED:
            return "failed";
    }
    return "unknown";
}

vsag::DatasetPtr
MakeFloat32Dataset(uint64_t num_vectors,
                   uint64_t dim,
                   std::vector<int64_t>& ids,
                   std::vector<float>& vectors) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(static_cast<int64_t>(num_vectors))
        ->Dim(static_cast<int64_t>(dim))
        ->Ids(ids.empty() ? nullptr : ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    return dataset;
}

std::pair<uint64_t, uint64_t>
GetMatrixShape(const H5::H5File& file, const std::string& name) {
    const auto dataset = file.openDataSet(name);
    const auto dataspace = dataset.getSpace();
    if (dataspace.getSimpleExtentNdims() != 2) {
        throw std::runtime_error(name + " must be a 2D matrix");
    }

    hsize_t dims[2] = {0, 0};
    dataspace.getSimpleExtentDims(dims, nullptr);
    return {static_cast<uint64_t>(dims[0]), static_cast<uint64_t>(dims[1])};
}

std::vector<float>
ReadFloat32MatrixPrefix(const H5::H5File& file,
                        const std::string& name,
                        uint64_t rows,
                        uint64_t cols) {
    auto dataset = file.openDataSet(name);
    auto file_space = dataset.getSpace();

    const hsize_t offset[2] = {0, 0};
    const hsize_t count[2] = {static_cast<hsize_t>(rows), static_cast<hsize_t>(cols)};
    file_space.selectHyperslab(H5S_SELECT_SET, count, offset);

    H5::DataSpace memory_space(2, count);
    std::vector<float> values(rows * cols);
    dataset.read(values.data(), H5::PredType::NATIVE_FLOAT, memory_space, file_space);
    return values;
}

DatasetBundle
MakeSyntheticDataset(const PocOptions& options) {
    DatasetBundle bundle;
    bundle.dim = 16;
    bundle.source = "synthetic";
    const auto query_count = options.query_count == 0 ? 8 : options.query_count;

    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;

    bundle.ids.resize(options.base_count);
    bundle.base_vectors.resize(options.base_count * bundle.dim);
    for (uint64_t i = 0; i < options.base_count; ++i) {
        bundle.ids[i] = static_cast<int64_t>(i);
    }
    for (auto& value : bundle.base_vectors) {
        value = distrib_real(rng);
    }

    std::vector<int64_t> query_ids;
    bundle.query_vectors.resize(query_count * bundle.dim);
    for (auto& value : bundle.query_vectors) {
        value = distrib_real(rng);
    }

    bundle.base =
        MakeFloat32Dataset(options.base_count, bundle.dim, bundle.ids, bundle.base_vectors);
    bundle.queries = MakeFloat32Dataset(query_count, bundle.dim, query_ids, bundle.query_vectors);
    return bundle;
}

DatasetBundle
LoadHdf5Dataset(const PocOptions& options) {
    H5::Exception::dontPrint();

    H5::H5File file(options.dataset_path, H5F_ACC_RDONLY);
    const auto train_shape = GetMatrixShape(file, "/train");
    const auto test_shape = GetMatrixShape(file, "/test");
    if (train_shape.second != test_shape.second) {
        throw std::runtime_error("train and test dimensions must match");
    }
    if (options.base_count > train_shape.first) {
        throw std::runtime_error("base-count exceeds train row count");
    }
    const auto query_count = options.query_count == 0 ? test_shape.first : options.query_count;
    if (query_count > test_shape.first) {
        throw std::runtime_error("query-count exceeds test row count");
    }

    DatasetBundle bundle;
    bundle.dim = train_shape.second;
    bundle.source = options.dataset_path;
    bundle.ids.resize(options.base_count);
    for (uint64_t i = 0; i < options.base_count; ++i) {
        bundle.ids[i] = static_cast<int64_t>(i);
    }
    bundle.base_vectors = ReadFloat32MatrixPrefix(file, "/train", options.base_count, bundle.dim);
    bundle.query_vectors = ReadFloat32MatrixPrefix(file, "/test", query_count, bundle.dim);

    std::vector<int64_t> query_ids;
    bundle.base =
        MakeFloat32Dataset(options.base_count, bundle.dim, bundle.ids, bundle.base_vectors);
    bundle.queries = MakeFloat32Dataset(query_count, bundle.dim, query_ids, bundle.query_vectors);
    return bundle;
}

std::string
MakeHGraphBuildParameters(uint64_t dim) {
    return R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": )" +
           std::to_string(dim) +
           R"(,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 100,
            "build_thread_count": 1
        }
    })";
}

std::string
MakeAutoTuningRequestJson(const PocOptions& options, uint64_t dim) {
    nlohmann::json request{
        {"version", 1},
        {"index_type", "hgraph"},
        {"source", {{"type", options.source_type}}},
        {"workload", {{"topk", options.topk}}},
        {"config", {{"search_parameters", {{"hgraph", {{"factor", 2}}}}}}},
        {"objective", {{"recall_at_k", {{"min", options.target_recall}}}}},
        {"search_space",
         {{"search", {{"hgraph.ef_search", {{"values", {0, 10, 20, 40, 80, 160, 320, 1201}}}}}}}},
        {"evaluation",
         {{"query_count", options.query_count}, {"successive_halving", {{"enabled", false}}}}}};
    if (options.max_trials > 0) {
        request["budget"] = {{"max_trials", options.max_trials}};
    }
    if (options.source_type == "raw_dataset") {
        request["config"]["build_parameters"] =
            nlohmann::json::parse(MakeHGraphBuildParameters(dim));
    }
    return request.dump();
}

vsag::IndexPtr
BuildIndex(const std::string& index_type,
           const std::string& build_parameters,
           const vsag::DatasetPtr& base) {
    auto index = vsag::Factory::CreateIndex(index_type, build_parameters);
    if (not index.has_value()) {
        throw std::runtime_error("failed to create " + index_type +
                                 " index: " + index.error().message);
    }

    auto build_result = index.value()->Build(base);
    if (not build_result.has_value()) {
        throw std::runtime_error("failed to build " + index_type +
                                 " index: " + build_result.error().message);
    }
    return index.value();
}

float
L2Distance(const float* lhs, const float* rhs, uint64_t dim) {
    float distance = 0.0F;
    for (uint64_t i = 0; i < dim; ++i) {
        const float diff = lhs[i] - rhs[i];
        distance += diff * diff;
    }
    return distance;
}

vsag::DatasetPtr
BuildGroundTruth(const vsag::DatasetPtr& base,
                 const vsag::DatasetPtr& queries,
                 uint64_t topk,
                 std::vector<int64_t>& ground_truth_ids,
                 std::vector<float>& ground_truth_distances) {
    const auto base_count = static_cast<uint64_t>(base->GetNumElements());
    const auto query_count = static_cast<uint64_t>(queries->GetNumElements());
    const auto dim = static_cast<uint64_t>(base->GetDim());
    const auto* base_vectors = base->GetFloat32Vectors();
    const auto* query_vectors = queries->GetFloat32Vectors();
    const auto* base_ids = base->GetIds();

    ground_truth_ids.assign(query_count * topk, 0);
    ground_truth_distances.assign(query_count * topk, 0.0F);

    for (uint64_t query_id = 0; query_id < query_count; ++query_id) {
        std::vector<std::pair<float, int64_t>> neighbors;
        neighbors.reserve(base_count);
        const auto* query = query_vectors + query_id * dim;
        for (uint64_t base_id = 0; base_id < base_count; ++base_id) {
            const auto* candidate = base_vectors + base_id * dim;
            neighbors.emplace_back(L2Distance(query, candidate, dim), base_ids[base_id]);
        }

        std::partial_sort(
            neighbors.begin(), neighbors.begin() + static_cast<int64_t>(topk), neighbors.end());
        for (uint64_t i = 0; i < topk; ++i) {
            const auto output_offset = query_id * topk + i;
            ground_truth_ids[output_offset] = neighbors[i].second;
            ground_truth_distances[output_offset] = neighbors[i].first;
        }
    }

    auto ground_truth = vsag::Dataset::Make();
    ground_truth->NumElements(static_cast<int64_t>(query_count))
        ->Dim(static_cast<int64_t>(topk))
        ->Ids(ground_truth_ids.data())
        ->Distances(ground_truth_distances.data())
        ->Owner(false);
    return ground_truth;
}

void
PrintInput(const PocOptions& options, const DatasetBundle& bundle) {
    std::cout << "Input:" << std::endl;
    std::cout << "  source=" << bundle.source << std::endl;
    if (not options.request_json_path.empty()) {
        std::cout << "  request_json=" << options.request_json_path << std::endl;
    }
    std::cout << "  source_type=" << options.source_type << std::endl;
    std::cout << "  base_count=" << options.base_count << std::endl;
    std::cout << "  query_count=" << bundle.queries->GetNumElements() << std::endl;
    std::cout << "  dim=" << bundle.dim << std::endl;
    std::cout << "  topk=" << options.topk << std::endl;
    std::cout << "  target_recall=" << options.target_recall << std::endl;
    std::cout << "  max_trials=" << options.max_trials << std::endl;
    std::cout << "  base_search_parameters={\"hgraph\":{\"factor\":2}}" << std::endl;
    std::cout << "  ef_search_candidates={0,10,20,40,80,160,320,1201}" << std::endl;
}

void
PrintStages(const vsag::AutoTuningReport& report) {
    std::cout << "\nStage report:" << std::endl;
    for (const auto& stage : report.stages) {
        std::cout << "  " << StageName(stage.stage) << " status=" << StageStatusName(stage.status)
                  << " input=" << stage.input_count << " output=" << stage.output_count
                  << " message=\"" << stage.message << "\"" << std::endl;
    }
}

std::string
CandidateDescription(const vsag::TuningCandidateReport& candidate) {
    nlohmann::json result = nlohmann::json::object();
    for (const auto& patch : candidate.patches) {
        try {
            result[patch.path] = nlohmann::json::parse(patch.value);
        } catch (const std::exception&) {
            result[patch.path] = patch.value;
        }
    }
    if (not result.contains("hgraph.ef_search") && candidate.ef_search > 0) {
        result["hgraph.ef_search"] = candidate.ef_search;
    }
    return result.dump();
}

void
PrintTrials(const vsag::AutoTuningReport& report) {
    std::cout << "\nTrial report:" << std::endl;
    for (const auto& trial : report.trial_report.trials) {
        std::cout << "  trial=" << trial.trial_id
                  << " candidate=" << CandidateDescription(trial.candidate)
                  << " status=" << TrialStatusName(trial.status);
        if (trial.status == vsag::TuningTrialStatus::COMPLETED) {
            std::cout << " recall=" << trial.evaluation.recall.average
                      << " latency_ms=" << trial.evaluation.latency.average_ms
                      << " qps=" << trial.evaluation.qps;
        }
        if (not trial.message.empty()) {
            std::cout << " message=\"" << trial.message << "\"";
        }
        std::cout << std::endl;
    }
}

void
PrintRecommendation(const vsag::AutoTuningReport& report) {
    if (report.recommendation.has_value()) {
        const auto& trial = report.recommendation.value();
        std::cout << "\nRecommendation: candidate=" << CandidateDescription(trial.candidate)
                  << " recall=" << trial.evaluation.recall.average
                  << " latency_ms=" << trial.evaluation.latency.average_ms << std::endl;
        return;
    }

    std::cout << "\nRecommendation: no candidate reached the target recall" << std::endl;
    if (report.best_effort.has_value()) {
        const auto& trial = report.best_effort.value();
        std::cout << "Best effort: candidate=" << CandidateDescription(trial.candidate)
                  << " recall=" << trial.evaluation.recall.average
                  << " latency_ms=" << trial.evaluation.latency.average_ms << std::endl;
    }
}

void
WriteJsonReport(const vsag::AutoTuningReport& report, const std::string& path) {
    if (path.empty()) {
        return;
    }
    std::ofstream output(path);
    if (not output.is_open()) {
        throw std::runtime_error("failed to open json output path: " + path);
    }
    output << vsag::SerializeAutoTuningReportJson(report) << std::endl;
}

vsag::AutoTuningRequest
MakeAutoTuningRequest(const PocOptions& options,
                      const DatasetBundle& bundle,
                      const vsag::DatasetPtr& ground_truth,
                      const std::string& request_json,
                      double& build_elapsed_ms) {
    const auto build_started_at = std::chrono::steady_clock::now();

    vsag::AutoTuningApiContext context;
    context.base = bundle.base;
    context.queries = bundle.queries;
    context.ground_truth = ground_truth;

    if (options.source_type == "existing_index") {
        context.index = BuildIndex("hgraph", MakeHGraphBuildParameters(bundle.dim), bundle.base);
    }

    const auto effective_request_json =
        request_json.empty() ? MakeAutoTuningRequestJson(options, bundle.dim) : request_json;
    auto parse_result = vsag::PrepareAutoTuningRequestJson(effective_request_json, context);
    build_elapsed_ms = ElapsedMs(build_started_at);
    if (not parse_result.Succeeded()) {
        throw std::runtime_error("failed to prepare auto tuning request: " +
                                 vsag::SerializeAutoTuningErrorJson(parse_result));
    }
    return parse_result.request;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        vsag::init();
        auto options = ParseOptions(argc, argv);
        std::string request_json;
        if (not options.request_json_path.empty()) {
            ValidateRequestJsonOptions(options);
            request_json = ReadTextFile(options.request_json_path);
            ApplyRequestJsonToOptions(request_json, options);
        }
        FinalizeOptions(options);

        const auto load_started_at = std::chrono::steady_clock::now();
        auto bundle =
            options.dataset_path.empty() ? MakeSyntheticDataset(options) : LoadHdf5Dataset(options);
        const auto load_elapsed_ms = ElapsedMs(load_started_at);

        if (options.topk > options.base_count) {
            throw std::invalid_argument("topk must be no greater than base-count");
        }

        const auto ground_truth_started_at = std::chrono::steady_clock::now();
        std::vector<int64_t> ground_truth_ids;
        std::vector<float> ground_truth_distances;
        auto ground_truth = BuildGroundTruth(
            bundle.base, bundle.queries, options.topk, ground_truth_ids, ground_truth_distances);
        const auto ground_truth_elapsed_ms = ElapsedMs(ground_truth_started_at);

        double build_elapsed_ms = 0.0;
        auto request =
            MakeAutoTuningRequest(options, bundle, ground_truth, request_json, build_elapsed_ms);

        vsag::AutoTuningPipeline pipeline;
        const auto report = pipeline.Tune(request);

        PrintInput(options, bundle);
        std::cout << "\nPreparation elapsed_ms:" << std::endl;
        std::cout << "  load=" << load_elapsed_ms << std::endl;
        std::cout << "  build_index=" << build_elapsed_ms << std::endl;
        std::cout << "  ground_truth=" << ground_truth_elapsed_ms << std::endl;
        std::cout << "\nTuning elapsed_ms=" << report.elapsed_ms << std::endl;
        PrintStages(report);
        PrintTrials(report);
        PrintRecommendation(report);
        WriteJsonReport(report, options.json_output_path);

        return report.Succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "hgraph_auto_tuning_poc failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
