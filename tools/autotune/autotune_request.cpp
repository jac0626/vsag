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

#include "autotune_request.h"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <set>
#include <utility>

#include "autotune_candidate_generator.h"
#include "autotune_constraints.h"
#include "autotune_internal.h"
#include "vsag/constants.h"

namespace vsag::autotune::internal {

namespace {

void
require_known_keys(const JsonType& object,
                   std::initializer_list<const char*> known_keys,
                   const std::string& path) {
    Require(object.is_object(), path + " must be an object");
    std::set<std::string> known;
    for (const auto* key : known_keys) {
        known.emplace(key);
    }
    for (const auto& item : object.items()) {
        Require(known.find(item.key()) != known.end(), path + "." + item.key() + " is unsupported");
    }
}

std::string
parse_required_string(const JsonType& object, const std::string& key, const std::string& path) {
    Require(object.contains(key), path + "." + key + " is required");
    Require(object[key].is_string(), path + "." + key + " must be a string");
    auto value = object[key].get<std::string>();
    Require(!value.empty(), path + "." + key + " must not be empty");
    return value;
}

uint64_t
parse_positive_uint64(const JsonType& object, const std::string& key, const std::string& path) {
    Require(object.contains(key), path + "." + key + " is required");
    Require(object[key].is_number_integer() || object[key].is_number_unsigned(),
            path + "." + key + " must be an integer");
    if (object[key].is_number_unsigned()) {
        const auto value = object[key].get<uint64_t>();
        Require(value > 0, path + "." + key + " must be positive");
        return value;
    }
    const auto value = object[key].get<int64_t>();
    Require(value > 0, path + "." + key + " must be positive");
    return static_cast<uint64_t>(value);
}

ConstraintMap
parse_constraints(const JsonType& value, const std::string& path, bool use_existing_index) {
    Require(value.is_object(), path + " must be an object");

    ConstraintMap constraints;
    for (const auto& item : value.items()) {
        Require(item.value().is_number(), path + "." + item.key() + " must be a number");
        const auto threshold = item.value().get<double>();
        constraints.emplace(item.key(), threshold);
    }
    ValidateConstraints(constraints, use_existing_index, path);
    return constraints;
}

ObjectiveSpec
parse_objective(const JsonType& value, bool use_existing_index, const std::string& path) {
    require_known_keys(value, {"metric"}, path);
    const auto metric = parse_required_string(value, "metric", path);
    return ResolveObjective(metric, use_existing_index, path);
}

JsonType
constraints_to_json(const ConstraintMap& constraints) {
    JsonType value = JsonType::object();
    for (const auto& [metric, threshold] : constraints) {
        value[metric] = threshold;
    }
    return value;
}

JsonType
objective_to_json(const ObjectiveSpec& objective) {
    return JsonType{{"metric", objective.metric}};
}

JsonType
make_effective_request(const AutoTuneRequest& request) {
    JsonType effective{{"version", request.version}, {"data_path", request.data_path}};
    if (!request.index_path.empty()) {
        effective["index_path"] = request.index_path;
    }

    effective["indexes"] = JsonType::array();
    for (const auto& index : request.indexes) {
        JsonType effective_index{{"name", index.name},
                                 {"create_params", index.create_params},
                                 {"search_params", index.search_params}};
        effective["indexes"].push_back(std::move(effective_index));
    }

    effective["workload"] =
        JsonType{{"top_k", request.workload.top_k}, {"concurrency", request.workload.concurrency}};

    effective["constraints"] = constraints_to_json(request.constraints);
    effective["objective"] = objective_to_json(request.objective);
    effective["tuning_config"] =
        JsonType{{"workspace_path", request.tuning_config.workspace_path},
                 {"keep_intermediate", request.tuning_config.keep_intermediate},
                 {"max_trials", request.tuning_config.max_trials}};
    effective["output"] = JsonType::object();
    if (!request.output.result_path.empty()) {
        effective["output"]["result_path"] = request.output.result_path;
    }
    effective["output"]["include_raw_eval"] = request.output.include_raw_eval;

    if (request.dataset_resolved) {
        effective["dataset_description"] =
            JsonType{{"dim", request.dataset.dim},
                     {"dtype", request.dataset.dtype},
                     {"metric_type", request.dataset.metric_type},
                     {"base_count", request.dataset.base_count},
                     {"query_count", request.dataset.query_count},
                     {"ground_truth_k", request.dataset.ground_truth_k},
                     {"vector_type", request.dataset.vector_type}};
    }
    return effective;
}

void
validate_metadata_field(const JsonType& create_params,
                        const std::string& key,
                        const JsonType& expected,
                        const std::string& index_name) {
    if (!create_params.contains(key)) {
        return;
    }
    Require(create_params[key] == expected,
            index_name + " create_params." + key + " must match the dataset");
}

void
require_readable_regular_file(const std::string& path, const std::string& field_name) {
    std::error_code error;
    const bool is_regular = std::filesystem::is_regular_file(path, error);
    Require(!error && is_regular, field_name + " must name a readable regular file: " + path);
    std::ifstream input(path, std::ios::binary);
    Require(input.good(), field_name + " must name a readable regular file: " + path);
}

}  // namespace

AutoTuneRequest
ParseAutoTuneRequest(const JsonType& input_request) {
    require_known_keys(input_request,
                       {"version",
                        "data_path",
                        "index_path",
                        "indexes",
                        "workload",
                        "constraints",
                        "objective",
                        "tuning_config",
                        "output"},
                       "request");

    AutoTuneRequest request;
    Require(input_request.contains("version"), "request.version is required");
    Require(input_request["version"].is_number_integer(), "request.version must be an integer");
    const auto version = input_request["version"].get<int64_t>();
    Require(version == 1, "only AutoTune request version 1 is supported");
    request.version = static_cast<uint64_t>(version);

    request.data_path = parse_required_string(input_request, "data_path", "request");
    require_readable_regular_file(request.data_path, "data_path");
    if (input_request.contains("index_path")) {
        request.index_path = parse_required_string(input_request, "index_path", "request");
        require_readable_regular_file(request.index_path, "index_path");
    }

    Require(input_request.contains("indexes"), "request.indexes is required");
    Require(input_request["indexes"].is_array() && !input_request["indexes"].empty(),
            "request.indexes must be a non-empty array");
    if (!request.index_path.empty()) {
        Require(input_request["indexes"].size() == 1,
                "index_path requires exactly one indexes[] specification");
    }
    for (uint64_t i = 0; i < input_request["indexes"].size(); ++i) {
        const auto& value = input_request["indexes"][i];
        const auto path = "request.indexes[" + std::to_string(i) + "]";
        require_known_keys(value, {"name", "create_params", "search_params"}, path);
        IndexSpec index;
        index.name = NormalizeIndexName(parse_required_string(value, "name", path));
        Require(SupportsDefaultCandidateGeneration(index.name), "unsupported index: " + index.name);
        if (value.contains("create_params")) {
            Require(value["create_params"].is_object(), path + ".create_params must be an object");
            index.create_params = value["create_params"];
        }
        if (value.contains("search_params")) {
            Require(value["search_params"].is_object(), path + ".search_params must be an object");
            for (const auto& item : value["search_params"].items()) {
                const auto normalized_key = NormalizeIndexName(item.key());
                Require(normalized_key == index.name,
                        path + ".search_params." + item.key() + " is unsupported");
                Require(!index.search_params.contains(index.name),
                        path + ".search_params contains duplicate index parameter objects");
                index.search_params[index.name] = item.value();
            }
        }
        request.indexes.emplace_back(std::move(index));
    }

    Require(input_request.contains("workload"), "request.workload is required");
    const auto& workload = input_request["workload"];
    require_known_keys(workload, {"top_k", "concurrency"}, "request.workload");
    request.workload.top_k = parse_positive_uint64(workload, "top_k", "request.workload");
    request.workload.concurrency =
        parse_positive_uint64(workload, "concurrency", "request.workload");
    Require(request.workload.top_k <= MAX_V1_TOP_K,
            "request.workload.top_k exceeds the V1 limit of " + std::to_string(MAX_V1_TOP_K));
    Require(request.workload.concurrency <= 200,
            "request.workload.concurrency must be in [1, 200]");
    Require(request.workload.top_k <= MAX_V1_CONCURRENT_NEIGHBOR_IDS / request.workload.concurrency,
            "request.workload.top_k * concurrency exceeds the V1 result-buffer limit of " +
                std::to_string(MAX_V1_CONCURRENT_NEIGHBOR_IDS) + " neighbor ids");

    Require(input_request.contains("constraints"), "request.constraints is required");
    request.constraints = parse_constraints(
        input_request["constraints"], "request.constraints", !request.index_path.empty());
    Require(input_request.contains("objective"), "request.objective is required");
    request.objective = parse_objective(
        input_request["objective"], !request.index_path.empty(), "request.objective");

    if (input_request.contains("tuning_config")) {
        const auto& value = input_request["tuning_config"];
        require_known_keys(
            value, {"workspace_path", "keep_intermediate", "max_trials"}, "request.tuning_config");
        if (value.contains("workspace_path")) {
            request.tuning_config.workspace_path =
                parse_required_string(value, "workspace_path", "request.tuning_config");
        }
        if (value.contains("keep_intermediate")) {
            Require(value["keep_intermediate"].is_boolean(),
                    "request.tuning_config.keep_intermediate must be a boolean");
            request.tuning_config.keep_intermediate = value["keep_intermediate"].get<bool>();
        }
        if (value.contains("max_trials")) {
            request.tuning_config.max_trials =
                parse_positive_uint64(value, "max_trials", "request.tuning_config");
            Require(request.tuning_config.max_trials <= MAX_V1_TRIALS,
                    "request.tuning_config.max_trials exceeds the V1 limit of " +
                        std::to_string(MAX_V1_TRIALS));
        }
    }

    if (input_request.contains("output")) {
        const auto& value = input_request["output"];
        require_known_keys(value, {"result_path", "include_raw_eval"}, "request.output");
        if (value.contains("result_path")) {
            request.output.result_path =
                parse_required_string(value, "result_path", "request.output");
        }
        if (value.contains("include_raw_eval")) {
            Require(value["include_raw_eval"].is_boolean(),
                    "request.output.include_raw_eval must be a boolean");
            request.output.include_raw_eval = value["include_raw_eval"].get<bool>();
        }
    }
    if (!request.output.result_path.empty()) {
        Require(!PathsAlias(request.output.result_path, request.data_path),
                "request.output.result_path must not alias data_path");
        Require(!PathsAlias(request.output.result_path, request.index_path),
                "request.output.result_path must not alias index_path");
    }

    request.effective_request = make_effective_request(request);
    return request;
}

void
ResolveAutoTuneRequest(AutoTuneRequest& request, const DatasetDescription& dataset) {
    Require(!request.dataset_resolved, "dataset metadata has already been resolved");
    Require(dataset.dim > 0, "dataset dim must be positive");
    Require(!dataset.dtype.empty(), "dataset dtype must not be empty");
    Require(!dataset.metric_type.empty(), "dataset metric_type must not be empty");
    Require(dataset.base_count > 0, "dataset base_count must be positive");
    Require(dataset.query_count > 0, "dataset query_count must be positive");
    Require(dataset.ground_truth_k > 0, "dataset ground_truth_k must be positive");
    Require(dataset.vector_type == "dense_vectors",
            "hgraph and ivf AutoTune require a dense_vectors dataset");
    Require(dataset.dtype == vsag::DATATYPE_FLOAT32,
            "AutoTune V1 supports only float32 dense datasets");

    Require(request.workload.top_k <= dataset.ground_truth_k,
            "workload top_k exceeds dataset ground_truth_k");
    Require(request.workload.concurrency <= dataset.query_count,
            "workload concurrency exceeds dataset query_count");

    for (auto& index : request.indexes) {
        validate_metadata_field(index.create_params, "dim", dataset.dim, index.name);
        validate_metadata_field(index.create_params, "dtype", dataset.dtype, index.name);
        validate_metadata_field(
            index.create_params, "metric_type", dataset.metric_type, index.name);
        validate_metadata_field(index.create_params, "repr", "dense", index.name);
        index.create_params["dim"] = dataset.dim;
        index.create_params["dtype"] = dataset.dtype;
        index.create_params["metric_type"] = dataset.metric_type;
    }

    request.dataset = dataset;
    request.dataset_resolved = true;
    request.effective_request = make_effective_request(request);
}

}  // namespace vsag::autotune::internal
