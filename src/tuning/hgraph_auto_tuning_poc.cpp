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
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tuning/tuning_pipeline.h"

namespace {

struct PocOptions {
    std::string dataset_path;
    uint64_t base_count = 0;
    uint64_t query_count = 0;
    uint64_t topk = 10;
    double target_recall = 0.80;
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
                 " --base-count 10000 --query-count 100 --target-recall 0.90\n";
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
        } else if (arg == "--base-count") {
            options.base_count = ParseUint64(require_value(arg), arg);
        } else if (arg == "--query-count") {
            options.query_count = ParseUint64(require_value(arg), arg);
        } else if (arg == "--topk") {
            options.topk = ParseUint64(require_value(arg), arg);
        } else if (arg == "--target-recall") {
            options.target_recall = std::stod(require_value(arg));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

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
        if (options.query_count == 0) {
            options.query_count = 100;
        }
    }
    return options;
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
    bundle.query_vectors.resize(options.query_count * bundle.dim);
    for (auto& value : bundle.query_vectors) {
        value = distrib_real(rng);
    }

    bundle.base =
        MakeFloat32Dataset(options.base_count, bundle.dim, bundle.ids, bundle.base_vectors);
    bundle.queries =
        MakeFloat32Dataset(options.query_count, bundle.dim, query_ids, bundle.query_vectors);
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
    if (options.query_count > test_shape.first) {
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
    bundle.query_vectors = ReadFloat32MatrixPrefix(file, "/test", options.query_count, bundle.dim);

    std::vector<int64_t> query_ids;
    bundle.base =
        MakeFloat32Dataset(options.base_count, bundle.dim, bundle.ids, bundle.base_vectors);
    bundle.queries =
        MakeFloat32Dataset(options.query_count, bundle.dim, query_ids, bundle.query_vectors);
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
    std::cout << "  base_count=" << options.base_count << std::endl;
    std::cout << "  query_count=" << options.query_count << std::endl;
    std::cout << "  dim=" << bundle.dim << std::endl;
    std::cout << "  topk=" << options.topk << std::endl;
    std::cout << "  target_recall=" << options.target_recall << std::endl;
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

void
PrintTrials(const vsag::AutoTuningReport& report) {
    std::cout << "\nTrial report:" << std::endl;
    for (const auto& trial : report.ef_search.trials) {
        std::cout << "  trial=" << trial.trial_id << " ef_search=" << trial.candidate.ef_search
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
        std::cout << "\nRecommendation: ef_search=" << trial.candidate.ef_search
                  << " recall=" << trial.evaluation.recall.average
                  << " latency_ms=" << trial.evaluation.latency.average_ms << std::endl;
        return;
    }

    std::cout << "\nRecommendation: no candidate reached the target recall" << std::endl;
    if (report.best_effort.has_value()) {
        const auto& trial = report.best_effort.value();
        std::cout << "Best effort: ef_search=" << trial.candidate.ef_search
                  << " recall=" << trial.evaluation.recall.average
                  << " latency_ms=" << trial.evaluation.latency.average_ms << std::endl;
    }
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        vsag::init();
        const auto options = ParseOptions(argc, argv);

        const auto load_started_at = std::chrono::steady_clock::now();
        auto bundle =
            options.dataset_path.empty() ? MakeSyntheticDataset(options) : LoadHdf5Dataset(options);
        const auto load_elapsed_ms = ElapsedMs(load_started_at);

        if (options.topk > options.base_count) {
            throw std::invalid_argument("topk must be no greater than base-count");
        }

        const auto build_started_at = std::chrono::steady_clock::now();
        auto hgraph_index =
            BuildIndex("hgraph", MakeHGraphBuildParameters(bundle.dim), bundle.base);
        const auto build_elapsed_ms = ElapsedMs(build_started_at);

        const auto ground_truth_started_at = std::chrono::steady_clock::now();
        std::vector<int64_t> ground_truth_ids;
        std::vector<float> ground_truth_distances;
        auto ground_truth = BuildGroundTruth(
            bundle.base, bundle.queries, options.topk, ground_truth_ids, ground_truth_distances);
        const auto ground_truth_elapsed_ms = ElapsedMs(ground_truth_started_at);

        vsag::AutoTuningRequest request;
        request.index = hgraph_index;
        request.queries = bundle.queries;
        request.ground_truth = ground_truth;
        request.topk = options.topk;
        request.query_count = options.query_count;
        request.target_recall = options.target_recall;
        request.base_search_parameters = R"({"hgraph":{"factor":2}})";
        request.ef_search_candidates = {0, 10, 20, 40, 80, 160, 320, 1201};

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

        return report.Succeeded() ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "hgraph_auto_tuning_poc failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
