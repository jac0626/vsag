
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

#include "hgraph_parameter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <sstream>

#include "datacell/extra_info_datacell_parameter.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/graph_datacell_parameter.h"
#include "datacell/graph_interface_parameter.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/odescent/odescent_graph_parameter.h"
#include "inner_string_params.h"
#include "utils/param_compat_macros.h"
#include "vsag/constants.h"

namespace vsag {

namespace {

constexpr uint64_t ADAPTIVE_EF_MIN_SAMPLE_COUNT = 100;
constexpr uint64_t ADAPTIVE_EF_MIN_CAP = 100;
constexpr uint64_t ADAPTIVE_EF_MAX_TOPK = 100;
constexpr uint64_t ADAPTIVE_EF_MAX_TARGET_COUNT = 64;
constexpr uint64_t ADAPTIVE_EF_MAX_TOPK_COUNT = 64;
constexpr uint64_t ADAPTIVE_EF_MAX_VALUE =
    static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

std::string
trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\n\r");
    return value.substr(first, last - first + 1);
}

std::vector<float>
parse_adaptive_ef_targets(const JsonType& json) {
    CHECK_ARGUMENT(json.IsString(), "adaptive_ef targets must be a comma-separated string");
    const auto values = trim(json.GetString());
    CHECK_ARGUMENT(values.empty() or values.back() != ',',
                   "adaptive_ef targets must not contain empty values");
    std::vector<float> result;
    std::stringstream ss(values);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        CHECK_ARGUMENT(not token.empty(), "adaptive_ef targets must not contain empty values");
        float value = 0.0F;
        std::string::size_type consumed = 0;
        try {
            value = std::stof(token, &consumed);
        } catch (const std::exception&) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("invalid adaptive_ef target: {}", token));
        }
        CHECK_ARGUMENT(consumed == token.size(),
                       fmt::format("invalid adaptive_ef target: {}", token));
        result.push_back(value);
    }
    return result;
}

std::vector<uint64_t>
parse_adaptive_ef_topks(const JsonType& json) {
    CHECK_ARGUMENT(json.IsString(), "adaptive_ef topks must be a comma-separated string");
    const auto values = trim(json.GetString());
    CHECK_ARGUMENT(values.empty() or values.back() != ',',
                   "adaptive_ef topks must not contain empty values");
    std::vector<uint64_t> result;
    std::stringstream ss(values);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        CHECK_ARGUMENT(not token.empty(), "adaptive_ef topks must not contain empty values");
        CHECK_ARGUMENT(
            std::all_of(
                token.begin(), token.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; }),
            fmt::format("invalid adaptive_ef topk: {}", token));
        uint64_t value = 0;
        try {
            value = std::stoull(token);
        } catch (const std::exception&) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("invalid adaptive_ef topk: {}", token));
        }
        result.push_back(value);
    }
    return result;
}

void
validate_adaptive_ef_build_parameters(const HGraphParameter& params) {
    CHECK_ARGUMENT(
        params.adaptive_ef_sample_count >= ADAPTIVE_EF_MIN_SAMPLE_COUNT,
        fmt::format("adaptive_ef sample_count must be at least {}", ADAPTIVE_EF_MIN_SAMPLE_COUNT));
    CHECK_ARGUMENT(params.adaptive_ef_cap >= ADAPTIVE_EF_MIN_CAP,
                   fmt::format("adaptive_ef ef_cap must be at least {}", ADAPTIVE_EF_MIN_CAP));
    CHECK_ARGUMENT(params.adaptive_ef_cap <= ADAPTIVE_EF_MAX_VALUE,
                   "adaptive_ef ef_cap exceeds int64_t range");
    CHECK_ARGUMENT(not params.adaptive_ef_targets.empty(), "adaptive_ef targets must not be empty");
    CHECK_ARGUMENT(params.adaptive_ef_targets.size() <= ADAPTIVE_EF_MAX_TARGET_COUNT,
                   "adaptive_ef has too many targets");
    for (uint64_t i = 0; i < params.adaptive_ef_targets.size(); ++i) {
        const float target = params.adaptive_ef_targets[i];
        CHECK_ARGUMENT(std::isfinite(target) and target > 0.0F and target <= 1.0F,
                       fmt::format("adaptive_ef target {} must be finite and in (0, 1]", target));
        if (i > 0) {
            CHECK_ARGUMENT(target - params.adaptive_ef_targets[i - 1] >= 1e-4F,
                           "adaptive_ef targets must be sorted and at least 0.0001 apart");
        }
    }
    CHECK_ARGUMENT(not params.adaptive_ef_topks.empty(), "adaptive_ef topks must not be empty");
    CHECK_ARGUMENT(params.adaptive_ef_topks.size() <= ADAPTIVE_EF_MAX_TOPK_COUNT,
                   "adaptive_ef has too many topks");
    for (uint64_t i = 0; i < params.adaptive_ef_topks.size(); ++i) {
        const uint64_t topk = params.adaptive_ef_topks[i];
        CHECK_ARGUMENT(
            topk >= 1 and topk <= ADAPTIVE_EF_MAX_TOPK,
            fmt::format("adaptive_ef topk {} must be in [1, {}]", topk, ADAPTIVE_EF_MAX_TOPK));
        if (i > 0) {
            CHECK_ARGUMENT(topk > params.adaptive_ef_topks[i - 1],
                           "adaptive_ef topks must be sorted and unique");
        }
    }
}

uint64_t
parse_adaptive_ef_search_value(const JsonType& json, const std::string& name) {
    CHECK_ARGUMENT(json.IsNumberUnsigned(),
                   fmt::format("adaptive_ef {} must be an unsigned integer", name));
    const uint64_t value = json.GetUint64();
    CHECK_ARGUMENT(value <= ADAPTIVE_EF_MAX_VALUE,
                   fmt::format("adaptive_ef {} exceeds int64_t range", name));
    CHECK_ARGUMENT(
        value == 0 or value >= ADAPTIVE_EF_MIN_CAP,
        fmt::format("adaptive_ef {} must be 0 or at least {}", name, ADAPTIVE_EF_MIN_CAP));
    return value;
}

bool
is_supported_adaptive_ef_alpha(float alpha) {
    return alpha == 0.20F or alpha == 0.10F or alpha == 0.05F;
}

}  // namespace

HGraphParameter::HGraphParameter(const JsonType& json) : HGraphParameter() {
    this->FromJson(json);
}

HGraphParameter::HGraphParameter() : name(INDEX_TYPE_HGRAPH) {
}

void
HGraphParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);

    if (json.Contains(HGRAPH_USE_ELP_OPTIMIZER_KEY)) {
        this->use_elp_optimizer = json[HGRAPH_USE_ELP_OPTIMIZER_KEY].GetBool();
    }

    if (json.Contains(HGRAPH_IGNORE_REORDER_KEY)) {
        this->ignore_reorder = json[HGRAPH_IGNORE_REORDER_KEY].GetBool();
    }

    if (json.Contains(HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY)) {
        this->build_by_base = json[HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY].GetBool();
    }

    CHECK_ARGUMENT(json.Contains(BASE_CODES_KEY),
                   fmt::format("hgraph parameters must contains {}", BASE_CODES_KEY));
    const auto& base_codes_json = json[BASE_CODES_KEY];
    this->base_codes_param = CreateFlattenParam(base_codes_json);

    if (use_reorder && this->reorder_source != HGRAPH_REORDER_SOURCE_BASE) {
        CHECK_ARGUMENT(json.Contains(PRECISE_CODES_KEY),
                       fmt::format("hgraph parameters must contains {}", PRECISE_CODES_KEY));
        const auto& precise_codes_json = json[PRECISE_CODES_KEY];
        this->precise_codes_param = CreateFlattenParam(precise_codes_json);
    }

    CHECK_ARGUMENT(json.Contains(GRAPH_KEY),
                   fmt::format("hgraph parameters must contains {}", GRAPH_KEY));
    const auto& graph_json = json[GRAPH_KEY];

    GraphStorageTypes graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT;
    if (graph_json.Contains(GRAPH_STORAGE_TYPE_KEY)) {
        const auto graph_storage_type_str = graph_json[GRAPH_STORAGE_TYPE_KEY].GetString();
        if (graph_storage_type_str == GRAPH_STORAGE_TYPE_VALUE_COMPRESSED) {
            graph_storage_type = GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_COMPRESSED;
        }

        if (graph_storage_type_str != GRAPH_STORAGE_TYPE_VALUE_COMPRESSED &&
            graph_storage_type_str != GRAPH_STORAGE_TYPE_VALUE_FLAT) {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("invalid graph_storage_type: {}", graph_storage_type_str));
        }
    }
    this->bottom_graph_param =
        GraphInterfaceParameter::GetGraphParameterByJson(graph_storage_type, graph_json);

    hierarchical_graph_param = std::make_shared<SparseGraphDatacellParameter>();
    hierarchical_graph_param->max_degree_ = this->bottom_graph_param->max_degree_ / 2;
    if (graph_storage_type == GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT) {
        auto graph_param =
            std::dynamic_pointer_cast<GraphDataCellParameter>(this->bottom_graph_param);
        if (graph_param != nullptr) {
            hierarchical_graph_param->remove_flag_bit_ = graph_param->remove_flag_bit_;
            hierarchical_graph_param->support_delete_ = graph_param->support_remove_;
            hierarchical_graph_param->use_reverse_edges_ = graph_param->use_reverse_edges_;
        } else {
            hierarchical_graph_param->support_delete_ = false;
        }
    } else {
        hierarchical_graph_param->support_delete_ = false;
    }

    if (json.Contains(EF_CONSTRUCTION_KEY)) {
        this->ef_construction = json[EF_CONSTRUCTION_KEY].GetUint64();
        CHECK_ARGUMENT(this->ef_construction > 0, "ef_construction must be positive");
    }

    if (json.Contains(ALPHA_KEY)) {
        this->alpha = json[ALPHA_KEY].GetFloat();
    }

    if (json.Contains(BUILD_THREAD_COUNT_KEY)) {
        this->build_thread_count = json[BUILD_THREAD_COUNT_KEY].GetUint64();
    }

    if (graph_json.Contains(GRAPH_TYPE_KEY)) {
        graph_type = graph_json[GRAPH_TYPE_KEY].GetString();
        if (graph_type == GRAPH_TYPE_VALUE_ODESCENT) {
            odescent_param = std::make_shared<ODescentParameter>();
            odescent_param->FromJson(graph_json);
        }
    }

    if (json.Contains(SUPPORT_DUPLICATE)) {
        this->support_duplicate = json[SUPPORT_DUPLICATE].GetBool();
        if (this->bottom_graph_param != nullptr) {
            this->bottom_graph_param->support_duplicate_ = this->support_duplicate;
        }
    }
    if (json.Contains(DEDUPLICATE_STORAGE)) {
        this->deduplicate_storage = json[DEDUPLICATE_STORAGE].GetBool();
    }
    if (this->deduplicate_storage && not this->support_duplicate) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "deduplicate_storage requires support_duplicate to be true");
    }
    if (json.Contains(DUPLICATE_DISTANCE_THRESHOLD)) {
        this->duplicate_distance_threshold = json[DUPLICATE_DISTANCE_THRESHOLD].GetFloat();
    }
    if (json.Contains(SUPPORT_FORCE_REMOVE)) {
        this->support_force_remove = json[SUPPORT_FORCE_REMOVE].GetBool();
    }
    if (this->deduplicate_storage && this->support_force_remove) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "deduplicate_storage does not support force remove because duplicate groups share "
            "physical vector slots");
    }
    if (json.Contains(HGRAPH_PERSIST_SOURCE_ID_KEY)) {
        this->persist_source_id = json[HGRAPH_PERSIST_SOURCE_ID_KEY].GetBool();
    }
    if (json.Contains("adaptive_ef")) {
        const auto& ada = json["adaptive_ef"];
        if (ada.Contains("enable")) {
            this->adaptive_ef_enable = ada["enable"].GetBool();
        }
        if (ada.Contains("sample_count")) {
            CHECK_ARGUMENT(ada["sample_count"].IsNumberUnsigned(),
                           "adaptive_ef sample_count must be an unsigned integer");
            this->adaptive_ef_sample_count = ada["sample_count"].GetUint64();
        }
        if (ada.Contains("ef_cap")) {
            CHECK_ARGUMENT(ada["ef_cap"].IsNumberUnsigned(),
                           "adaptive_ef ef_cap must be an unsigned integer");
            this->adaptive_ef_cap = ada["ef_cap"].GetUint64();
        }
        if (ada.Contains("targets")) {
            this->adaptive_ef_targets = parse_adaptive_ef_targets(ada["targets"]);
        }
        if (ada.Contains("topks")) {
            this->adaptive_ef_topks = parse_adaptive_ef_topks(ada["topks"]);
        }
    }
    if (this->adaptive_ef_enable) {
        validate_adaptive_ef_build_parameters(*this);
    }
}

JsonType
HGraphParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json[TYPE_KEY].SetString(INDEX_TYPE_HGRAPH);

    json[HGRAPH_USE_ELP_OPTIMIZER_KEY].SetBool(this->use_elp_optimizer);
    json[HGRAPH_IGNORE_REORDER_KEY].SetBool(this->ignore_reorder);
    json[REORDER_SOURCE_KEY].SetString(this->reorder_source);
    json[BASE_CODES_KEY].SetJson(this->base_codes_param->ToJson());
    json[GRAPH_KEY].SetJson(this->bottom_graph_param->ToJson());
    json[EF_CONSTRUCTION_KEY].SetUint64(this->ef_construction);
    json[ALPHA_KEY].SetFloat(this->alpha);
    json[SUPPORT_DUPLICATE].SetBool(this->support_duplicate);
    json[DEDUPLICATE_STORAGE].SetBool(this->deduplicate_storage);
    json[DUPLICATE_DISTANCE_THRESHOLD].SetFloat(this->duplicate_distance_threshold);
    json[SUPPORT_FORCE_REMOVE].SetBool(this->support_force_remove);
    json[HGRAPH_PERSIST_SOURCE_ID_KEY].SetBool(this->persist_source_id);
    json[TRAIN_SAMPLE_COUNT_KEY].SetInt(this->train_sample_count);
    json["adaptive_ef"]["enable"].SetBool(this->adaptive_ef_enable);
    json["adaptive_ef"]["sample_count"].SetUint64(this->adaptive_ef_sample_count);
    json["adaptive_ef"]["ef_cap"].SetUint64(this->adaptive_ef_cap);
    {
        std::string ts;
        for (uint64_t i = 0; i < this->adaptive_ef_targets.size(); ++i) {
            ts += (i ? "," : "") + std::to_string(this->adaptive_ef_targets[i]);
        }
        json["adaptive_ef"]["targets"].SetString(ts);
    }
    {
        std::string topks;
        for (uint64_t i = 0; i < this->adaptive_ef_topks.size(); ++i) {
            topks += (i ? "," : "") + std::to_string(this->adaptive_ef_topks[i]);
        }
        json["adaptive_ef"]["topks"].SetString(topks);
    }
    return json;
}

bool
HGraphParameter::CheckCompatibility(const ParamPtr& other) const {
    PARAM_CAST_OR_RETURN(HGraphParameter, p, other);
    auto have_reorder = this->use_reorder && not this->ignore_reorder;
    auto have_reorder_other = p->use_reorder && not p->ignore_reorder;
    if (have_reorder != have_reorder_other) {
        logger::error(
            "HGraphParameter::CheckCompatibility: use_reorder and ignore_reorder must be the same");
        return false;
    }
    CHECK_SUB_PARAM(*this, *p, base_codes_param);
    if (have_reorder) {
        CHECK_FIELD_EQ(*this, *p, reorder_source);
        if (this->reorder_source != HGRAPH_REORDER_SOURCE_BASE) {
            if (not this->precise_codes_param ||
                not this->precise_codes_param->CheckCompatibility(p->precise_codes_param)) {
                logger::error(
                    "HGraphParameter::CheckCompatibility: precise_codes_param is not compatible");
                return false;
            }
        }
    }
    CHECK_SUB_PARAM(*this, *p, bottom_graph_param);
    CHECK_FIELD_EQ(*this, *p, use_attribute_filter);
    CHECK_FIELD_EQ(*this, *p, support_duplicate);
    CHECK_FIELD_EQ(*this, *p, deduplicate_storage);
    CHECK_FIELD_EQ(*this, *p, duplicate_distance_threshold);
    CHECK_FIELD_EQ(*this, *p, support_force_remove);
    return true;
}

HGraphSearchParameters
HGraphSearchParameters::FromJson(const std::string& json_string) {
    auto params = JsonType::Parse(json_string);

    HGraphSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.Contains(INDEX_TYPE_HGRAPH),
                   fmt::format("parameters must contains {}", INDEX_TYPE_HGRAPH));

    obj.IndexSearchParameter::FromJson(params[INDEX_TYPE_HGRAPH]);

    CHECK_ARGUMENT(
        params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_EF_RUNTIME),
        fmt::format(
            "parameters[{}] must contains {}", INDEX_TYPE_HGRAPH, HGRAPH_PARAMETER_EF_RUNTIME));
    const auto& ef_search_json = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_EF_RUNTIME];
    CHECK_ARGUMENT(ef_search_json.IsNumberInteger(), "ef_search must be an integer");
    if (ef_search_json.IsNumberUnsigned()) {
        CHECK_ARGUMENT(ef_search_json.GetUint64() <=
                           static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                       "ef_search exceeds int64_t range");
    }
    obj.ef_search = ef_search_json.GetInt();
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_HOPS_LIMIT)) {
        obj.hops_limit = params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_HOPS_LIMIT].GetInt();
    }
    if (params[INDEX_TYPE_HGRAPH].Contains("adaptive_ef")) {
        const auto& ada = params[INDEX_TYPE_HGRAPH]["adaptive_ef"];
        if (ada.Contains("target_recall")) {
            obj.adaptive_ef_target_recall = ada["target_recall"].GetFloat();
            CHECK_ARGUMENT(std::isfinite(obj.adaptive_ef_target_recall) and
                               obj.adaptive_ef_target_recall > 0.0F and
                               obj.adaptive_ef_target_recall <= 1.0F,
                           "adaptive_ef target_recall must be finite and in (0, 1]");
        }
        if (ada.Contains("alpha")) {
            obj.adaptive_ef_alpha = ada["alpha"].GetFloat();
        }
        CHECK_ARGUMENT(is_supported_adaptive_ef_alpha(obj.adaptive_ef_alpha),
                       "adaptive_ef alpha must be one of 0.2, 0.1, or 0.05");
        if (ada.Contains("ef_cap")) {
            obj.adaptive_ef_cap = parse_adaptive_ef_search_value(ada["ef_cap"], "ef_cap");
        }
        if (ada.Contains("force_ef")) {
            obj.adaptive_ef_force = parse_adaptive_ef_search_value(ada["force_ef"], "force_ef");
        }
        CHECK_ARGUMENT(obj.adaptive_ef_target_recall == 0.0F or obj.adaptive_ef_force == 0,
                       "adaptive_ef target_recall and force_ef are mutually exclusive");
        CHECK_ARGUMENT(obj.adaptive_ef_target_recall > 0.0F or obj.adaptive_ef_force > 0,
                       "adaptive_ef requires target_recall or force_ef");
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_USE_EXTRA_INFO_FILTER)) {
        obj.use_extra_info_filter =
            params[INDEX_TYPE_HGRAPH][HGRAPH_USE_EXTRA_INFO_FILTER].GetBool();
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_RABITQ_ONE_BIT_SEARCH)) {
        obj.rabitq_one_bit_search =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_RABITQ_ONE_BIT_SEARCH].GetBool();
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HGRAPH_PARAMETER_BRUTE_FORCE_THRESHOLD)) {
        obj.brute_force_threshold =
            params[INDEX_TYPE_HGRAPH][HGRAPH_PARAMETER_BRUTE_FORCE_THRESHOLD].GetFloat();
        CHECK_ARGUMENT(  // NOLINT
            (0.0F <= obj.brute_force_threshold) and (obj.brute_force_threshold <= 1.0F),
            fmt::format("brute_force_threshold({}) must in range[0.0, 1.0]",
                        obj.brute_force_threshold));
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(RABITQ_QUANTIZATION_ERROR_RATE_KEY)) {
        obj.rabitq_error_rate =
            params[INDEX_TYPE_HGRAPH][RABITQ_QUANTIZATION_ERROR_RATE_KEY].GetFloat();
        CHECK_ARGUMENT(std::isfinite(obj.rabitq_error_rate),
                       fmt::format("rabitq_error_rate must be finite and positive, got {}",
                                   obj.rabitq_error_rate));
        CHECK_ARGUMENT(obj.rabitq_error_rate > 0.0F,
                       fmt::format("rabitq_error_rate must be finite and positive, got {}",
                                   obj.rabitq_error_rate));
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HNSW_PARAMETER_SKIP_RATIO)) {
        obj.skip_ratio = params[INDEX_TYPE_HGRAPH][HNSW_PARAMETER_SKIP_RATIO].GetFloat();
        CHECK_ARGUMENT((0.0F <= obj.skip_ratio) and (obj.skip_ratio <= 1.0F),  // NOLINT
                       fmt::format("skip_ratio({}) must be in range [0.0, 1.0]", obj.skip_ratio));
    }
    if (params[INDEX_TYPE_HGRAPH].Contains(HNSW_PARAMETER_SKIP_STRATEGY)) {
        CHECK_ARGUMENT(
            params[INDEX_TYPE_HGRAPH][HNSW_PARAMETER_SKIP_STRATEGY].IsString(),
            fmt::format("parameters[{}] must be string type", HNSW_PARAMETER_SKIP_STRATEGY));
        obj.skip_strategy_type = parse_filter_search_skip_strategy_type(
            params[INDEX_TYPE_HGRAPH][HNSW_PARAMETER_SKIP_STRATEGY].GetString());
    }

    return obj;
}
}  // namespace vsag
