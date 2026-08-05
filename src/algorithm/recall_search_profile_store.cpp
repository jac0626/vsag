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

#include "recall_search_profile_store.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vsag/errors.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

constexpr char RECALL_SEARCH_PROFILES[] = "recall_search_profiles";
// Release builds use -Ofast, under which std::isfinite may assume finite inputs.
constexpr uint64_t DOUBLE_EXPONENT_MASK = 0x7FF0000000000000ULL;

struct ProfilePoint {
    double target_recall{0.0};
    double validated_recall{0.0};
    std::shared_ptr<const std::string> search_parameters{};
};

using ProfileCurve = std::vector<ProfilePoint>;

void
ValidateRecall(double recall, const char* name) {
    static_assert(sizeof(double) == sizeof(uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    uint64_t bits{0};
    std::memcpy(&bits, &recall, sizeof(bits));
    const bool finite = (bits & DOUBLE_EXPONENT_MASK) != DOUBLE_EXPONENT_MASK;
    if (not finite or recall < 0.0 or recall > 1.0) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT, name, " must be finite and in the range [0, 1]");
    }
}

void
ValidateEntry(const RecallSearchProfileEntry& entry) {
    if (entry.top_k <= 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "top_k must be greater than 0");
    }
    ValidateRecall(entry.target_recall, "target_recall");
    ValidateRecall(entry.validated_recall, "validated_recall");
    if (entry.validated_recall < entry.target_recall) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "validated_recall must be at least target_recall");
    }
    if (entry.search_parameters.empty()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "search_parameters must not be empty");
    }
    auto parameters = nlohmann::json::parse(entry.search_parameters, nullptr, false);
    if (parameters.is_discarded() or not parameters.is_object()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "search_parameters must be a valid JSON object");
    }
}

ProfilePoint
MakePoint(const RecallSearchProfileEntry& entry) {
    ValidateEntry(entry);
    return ProfilePoint{
        .target_recall = entry.target_recall,
        .validated_recall = entry.validated_recall,
        .search_parameters = std::make_shared<const std::string>(entry.search_parameters),
    };
}

void
Upsert(ProfileCurve& curve, ProfilePoint point) {
    auto previous = std::find_if(curve.begin(), curve.end(), [&point](const auto& item) {
        return item.target_recall == point.target_recall;
    });
    if (previous != curve.end()) {
        curve.erase(previous);
    }
    curve.emplace_back(std::move(point));
    std::sort(curve.begin(), curve.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.validated_recall, lhs.target_recall) <
               std::tie(rhs.validated_recall, rhs.target_recall);
    });
}

}  // namespace

struct RecallSearchProfileStore::Snapshot {
    using PathCurves = std::unordered_map<std::string, ProfileCurve>;
    std::unordered_map<int64_t, PathCurves> curves{};
};

RecallSearchProfileStore::RecallSearchProfileStore()
    : snapshot_(std::make_shared<const Snapshot>()) {
}

void
RecallSearchProfileStore::Update(const RecallSearchProfileEntry& entry) {
    auto point = MakePoint(entry);
    std::lock_guard<std::mutex> lock(this->update_mutex_);
    auto current = std::atomic_load_explicit(&this->snapshot_, std::memory_order_acquire);
    auto next = std::make_shared<Snapshot>(*current);
    auto& curve = next->curves[entry.top_k][entry.path];
    Upsert(curve, std::move(point));
    std::shared_ptr<const Snapshot> published = std::move(next);
    std::atomic_store_explicit(&this->snapshot_, std::move(published), std::memory_order_release);
}

RecallSearchProfileStore::SearchParametersPtr
RecallSearchProfileStore::Resolve(int64_t top_k,
                                  double target_recall,
                                  const std::string& path) const {
    if (top_k <= 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "top_k must be greater than 0");
    }
    ValidateRecall(target_recall, "target_recall");

    auto snapshot = std::atomic_load_explicit(&this->snapshot_, std::memory_order_acquire);
    auto top_k_iter = snapshot->curves.find(top_k);
    if (top_k_iter == snapshot->curves.end()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "no recall search profile for the requested top_k and path");
    }
    auto curve_iter = top_k_iter->second.find(path);
    if (curve_iter == top_k_iter->second.end()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "no recall search profile for the requested top_k and path");
    }

    const auto& curve = curve_iter->second;
    auto point_iter = std::lower_bound(
        curve.begin(), curve.end(), target_recall, [](const auto& point, double recall) {
            return point.validated_recall < recall;
        });
    if (point_iter == curve.end()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "no calibrated search parameters satisfy the requested recall");
    }
    return point_iter->search_parameters;
}

void
RecallSearchProfileStore::AppendTo(JsonType& basic_info) const {
    auto snapshot = std::atomic_load_explicit(&this->snapshot_, std::memory_order_acquire);
    if (snapshot->curves.empty()) {
        return;
    }

    struct SerializableEntry {
        int64_t top_k{0};
        std::string path{};
        ProfilePoint point;
    };
    std::vector<SerializableEntry> entries;
    for (const auto& [top_k, paths] : snapshot->curves) {
        for (const auto& [path, curve] : paths) {
            for (const auto& point : curve) {
                entries.push_back({.top_k = top_k, .path = path, .point = point});
            }
        }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return std::tie(lhs.top_k, lhs.path, lhs.point.validated_recall) <
               std::tie(rhs.top_k, rhs.path, rhs.point.validated_recall);
    });

    nlohmann::json profiles = nlohmann::json::array();
    for (const auto& entry : entries) {
        profiles.push_back({
            {"top_k", entry.top_k},
            {"target_recall", entry.point.target_recall},
            {"validated_recall", entry.point.validated_recall},
            {"path", entry.path},
            {"search_parameters", *entry.point.search_parameters},
        });
    }
    (*basic_info.GetInnerJson())[RECALL_SEARCH_PROFILES] = std::move(profiles);
}

void
RecallSearchProfileStore::RestoreFrom(const JsonType& basic_info, const EntryValidator& validator) {
    auto next = std::make_shared<Snapshot>();
    if (basic_info.Contains(RECALL_SEARCH_PROFILES)) {
        try {
            const auto& profiles = (*basic_info.GetInnerJson()).at(RECALL_SEARCH_PROFILES);
            if (not profiles.is_array()) {
                throw std::invalid_argument("recall_search_profiles must be an array");
            }
            for (const auto& json : profiles) {
                RecallSearchProfileEntry entry{
                    .top_k = json.at("top_k").get<int64_t>(),
                    .target_recall = json.at("target_recall").get<double>(),
                    .validated_recall = json.at("validated_recall").get<double>(),
                    .path = json.at("path").get<std::string>(),
                    .search_parameters = json.at("search_parameters").get<std::string>(),
                };
                if (validator) {
                    validator(entry);
                }
                auto& curve = next->curves[entry.top_k][entry.path];
                Upsert(curve, MakePoint(entry));
            }
        } catch (const VsagException& exception) {
            throw VsagException(
                ErrorType::INVALID_BINARY, "invalid recall search profile: ", exception.what());
        } catch (const std::exception& exception) {
            throw VsagException(
                ErrorType::INVALID_BINARY, "invalid recall search profile: ", exception.what());
        }
    }

    std::lock_guard<std::mutex> lock(this->update_mutex_);
    std::shared_ptr<const Snapshot> published = std::move(next);
    std::atomic_store_explicit(&this->snapshot_, std::move(published), std::memory_order_release);
}

}  // namespace vsag
