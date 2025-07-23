
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

#include "gno_imi_parameter.h"

#include <fmt/format.h>

#include "inner_string_params.h"
#include "logger.h"

namespace vsag {

GNOIMIParameter::GNOIMIParameter() = default;

void
GNOIMIParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(
        json.contains(GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY),
        fmt::format("ivf parameters must contains {}", GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY));
    this->first_order_buckets_count = json[GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY];

    if (json.contains(GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY)) {
        this->second_order_buckets_count = json[GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY];
    } else {
        this->second_order_buckets_count = this->first_order_buckets_count;
    }
}

JsonType
GNOIMIParameter::ToJson() const {
    JsonType json;
    json[GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY] = this->first_order_buckets_count;
    json[GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY] = this->second_order_buckets_count;
    return json;
}
bool
GNOIMIParameter::CheckCompatibility(const ParamPtr& other) const {
    auto gno_imi_param = std::dynamic_pointer_cast<GNOIMIParameter>(other);
    if (!gno_imi_param) {
        logger::error(
            "GNOIMIParameter::CheckCompatibility: "
            "other parameter is not GNOIMIParameter");
        return false;
    }
    if (this->first_order_buckets_count != gno_imi_param->first_order_buckets_count) {
        logger::error(
            "GNOIMIParameter::CheckCompatibility: "
            "first_order_buckets_count mismatch: {} != {}",
            this->first_order_buckets_count,
            gno_imi_param->first_order_buckets_count);
        return false;
    }
    if (this->second_order_buckets_count != gno_imi_param->second_order_buckets_count) {
        logger::error(
            "GNOIMIParameter::CheckCompatibility: "
            "second_order_buckets_count mismatch: {} != {}",
            this->second_order_buckets_count,
            gno_imi_param->second_order_buckets_count);
        return false;
    }
    return true;
}
}  // namespace vsag
