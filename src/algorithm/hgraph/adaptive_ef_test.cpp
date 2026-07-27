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

#include "adaptive_ef.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "unittest.h"

namespace vsag {
namespace {

AdaptiveEfHead
MakeHead(uint64_t topk = 100, float target = 0.9F) {
    AdaptiveEfHead head;
    head.topk = topk;
    head.target_recall = target;
    head.feat_mean.assign(kAdaptiveEfFeatureCount, 0.0F);
    head.feat_stdev.assign(kAdaptiveEfFeatureCount, 1.0F);
    head.weights.assign(2 * kAdaptiveEfFeatureCount + 1, 0.0F);
    head.weights.back() = std::log2(100.0F);
    for (bool& enabled : head.alpha_enabled) {
        enabled = true;
    }
    return head;
}

AdaptiveEfState
MakeState() {
    AdaptiveEfState state;
    state.enabled = true;
    state.calibrated = true;
    state.sample_count = 20;
    state.targets = {0.9F};
    state.topks = {100};
    state.ef_cap = 200;
    state.ef_grid = {100, 125, 150, 175, 200};
    state.spearman = 0.75F;
    state.data_mean = {0.0F, 0.0F};
    state.data_var = {1.0F, 1.0F};
    state.heads = {MakeHead()};
    return state;
}

template <typename T>
void
AppendPod(std::string& blob, const T& value) {
    blob.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void
AppendVector(std::string& blob, const std::vector<T>& values) {
    AppendPod(blob, static_cast<uint64_t>(values.size()));
    if (not values.empty()) {
        blob.append(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    }
}

std::string
MakeVersionOneBlob() {
    std::string blob;
    const auto head = MakeHead();
    AppendPod(blob, uint64_t{1});
    AppendPod(blob, true);
    AppendPod(blob, true);
    AppendPod(blob, 0.75F);
    AppendPod(blob, uint64_t{20});
    AppendPod(blob, uint64_t{200});
    AppendVector(blob, std::vector<float>{0.9F});
    AppendVector(blob, std::vector<float>{0.0F, 0.0F});
    AppendVector(blob, std::vector<float>{1.0F, 1.0F});
    AppendPod(blob, uint64_t{1});
    AppendPod(blob, head.target_recall);
    AppendVector(blob, head.feat_mean);
    AppendVector(blob, head.feat_stdev);
    AppendVector(blob, head.weights);
    blob.append(reinterpret_cast<const char*>(head.margins), sizeof(head.margins));
    return blob;
}

}  // namespace

TEST_CASE("AdaptiveEf accepts only calibrated alpha values", "[ut][AdaptiveEf]") {
    REQUIRE(AdaptiveEfState::AlphaIndex(0.2F) == 0);
    REQUIRE(AdaptiveEfState::AlphaIndex(0.1F) == 1);
    REQUIRE(AdaptiveEfState::AlphaIndex(0.05F) == 2);
    REQUIRE(AdaptiveEfState::AlphaIndex(0.15F) == -1);
    REQUIRE(AdaptiveEfState::AlphaIndex(0.01F) == -1);
}

TEST_CASE("AdaptiveEf finds heads by both top-k and target", "[ut][AdaptiveEf]") {
    auto state = MakeState();
    state.topks = {10, 100};
    state.heads.insert(state.heads.begin(), MakeHead(10));

    REQUIRE(state.FindHead(10, 0.9F) != nullptr);
    REQUIRE(state.FindHead(10, 0.9F)->topk == 10);
    REQUIRE(state.FindHead(100, 0.9F) != nullptr);
    REQUIRE(state.FindHead(50, 0.9F) == nullptr);
    REQUIRE(state.FindHead(10, 0.95F) == nullptr);
}

TEST_CASE("AdaptiveEf prediction clamps features and snaps to its ef grid", "[ut][AdaptiveEf]") {
    auto state = MakeState();
    auto head = state.heads.front();
    head.weights[0] = 0.1F;
    float features[kAdaptiveEfFeatureCount] = {};
    features[0] = 1000.0F;

    const double prediction = AdaptiveEfState::PredictLog2FromFeatures(head, features);
    REQUIRE(std::abs(prediction - (std::log2(100.0) + 0.8)) < 1e-5);
    REQUIRE(state.PredictFromFeatures(head, features, 0.2F, 100, 200) == 175);

    head.weights[0] = 0.0F;
    head.weights.back() = std::log2(126.0F);
    REQUIRE(state.PredictFromFeatures(head, features, 0.2F, 100, 200) == 150);
    REQUIRE(state.PredictFromFeatures(head, features, 0.2F, 100, 140) == 140);

    head.alpha_enabled[0] = false;
    REQUIRE_THROWS(state.PredictFromFeatures(head, features, 0.2F, 100, 200));
    REQUIRE_THROWS(state.PredictFromFeatures(head, features, 0.15F, 100, 200));
}

TEST_CASE("AdaptiveEf prediction fails safe for a NaN feature", "[ut][AdaptiveEf]") {
    auto state = MakeState();
    const auto& head = state.heads.front();
    float features[kAdaptiveEfFeatureCount] = {};
    features[0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE(state.PredictFromFeatures(head, features, 0.2F, 100, 200) == 200);
}

TEST_CASE("AdaptiveEf Spearman uses average ranks for ties", "[ut][AdaptiveEf]") {
    const std::vector<float> tied = {1.0F, 1.0F, 2.0F, 2.0F};
    const std::vector<float> increasing = {1.0F, 2.0F, 3.0F, 4.0F};
    const float correlation = AdaptiveEfState::SpearmanCorrelation(tied, increasing);
    REQUIRE(std::abs(correlation - std::sqrt(0.8F)) < 1e-6F);

    REQUIRE(AdaptiveEfState::SpearmanCorrelation({1.0F, 1.0F, 1.0F}, {1.0F, 2.0F, 3.0F}) == 0.0F);
    REQUIRE(AdaptiveEfState::SpearmanCorrelation({1.0F, 2.0F}, {1.0F}) == 0.0F);
}

TEST_CASE("AdaptiveEf state version two round trips and validates", "[ut][AdaptiveEf]") {
    auto state = MakeState();
    state.disabled_reason = "diagnostic";
    const std::string blob = state.SerializeToString();

    AdaptiveEfState restored;
    restored.DeserializeFromString(blob);
    restored.Validate(2);
    REQUIRE(restored.enabled);
    REQUIRE(restored.calibrated);
    REQUIRE(restored.disabled_reason == "diagnostic");
    REQUIRE((restored.topks == std::vector<uint64_t>{100}));
    REQUIRE((restored.ef_grid == std::vector<uint64_t>{100, 125, 150, 175, 200}));
    REQUIRE(restored.heads.size() == 1);
    REQUIRE(restored.heads.front().topk == 100);
    REQUIRE(restored.heads.front().alpha_enabled[2]);

    std::string truncated = blob;
    truncated.pop_back();
    REQUIRE_THROWS(restored.DeserializeFromString(truncated));
    REQUIRE_THROWS(restored.DeserializeFromString(blob + "x"));
}

TEST_CASE("AdaptiveEf deserializes version one fail closed", "[ut][AdaptiveEf]") {
    AdaptiveEfState restored;
    restored.DeserializeFromString(MakeVersionOneBlob());

    REQUIRE((restored.topks == std::vector<uint64_t>{100}));
    REQUIRE((restored.ef_grid == std::vector<uint64_t>{100, 125, 150, 175, 200}));
    REQUIRE(restored.heads.size() == 1);
    REQUIRE(restored.heads.front().topk == 100);
    REQUIRE_FALSE(restored.calibrated);
    REQUIRE(restored.disabled_reason == "legacy adaptive_ef state must be rebuilt");
    REQUIRE_FALSE(restored.heads.front().alpha_enabled[0]);
    REQUIRE_FALSE(restored.heads.front().alpha_enabled[1]);
    REQUIRE_FALSE(restored.heads.front().alpha_enabled[2]);
}

TEST_CASE("AdaptiveEf validation rejects malformed model vectors", "[ut][AdaptiveEf]") {
    auto state = MakeState();
    state.heads.front().feat_stdev[0] = 0.0F;
    REQUIRE_THROWS(state.Validate(2));

    state = MakeState();
    state.data_var[0] = -1.0F;
    REQUIRE_THROWS(state.Validate(2));

    state = MakeState();
    state.ef_grid.pop_back();
    REQUIRE_THROWS(state.Validate(2));
}

TEST_CASE("AdaptiveEf validation rejects a calibrated disabled state", "[ut][AdaptiveEf]") {
    auto state = MakeState();
    state.enabled = false;
    REQUIRE_THROWS(state.Validate(2));
}

TEST_CASE("AdaptiveEf training rejects overlapping fit and calibration rows", "[ut][AdaptiveEf]") {
    std::vector<float> features(2 * kAdaptiveEfFeatureCount, 0.0F);
    std::vector<float> required = {std::log2(100.0F), std::log2(125.0F)};
    REQUIRE_THROWS(AdaptiveEfState::TrainHead(features, required, {0}, {0}, 0.9F, uint64_t{10}));
}

TEST_CASE("AdaptiveEf fixed gate refines a sparse matched-pass baseline", "[ut][AdaptiveEf]") {
    std::map<uint64_t, AdaptiveEfFixedGatePoint> observations;
    const auto evaluate = [&observations](uint64_t ef) {
        const AdaptiveEfFixedGatePoint point{ef < 137 ? uint64_t{8} : uint64_t{9},
                                             static_cast<double>(ef) * 10.0};
        observations.emplace(ef, point);
        return point;
    };

    const auto match = MatchAdaptiveEfFixedGate(9, {100, 200, 400}, evaluate);
    REQUIRE(match.trustworthy);
    REQUIRE(match.matched);
    REQUIRE(match.ef == 137);
    REQUIRE(match.success == 9);
    REQUIRE(match.total_cost == 1370.0);
    REQUIRE(observations.count(136) == 1);
    REQUIRE(observations.count(137) == 1);

    const auto dominance = MatchAdaptiveEfFixedGate(10, {100, 200, 400}, evaluate);
    REQUIRE(dominance.trustworthy);
    REQUIRE_FALSE(dominance.matched);
    REQUIRE(dominance.success == 9);
    REQUIRE(dominance.ef == 200);
    REQUIRE(dominance.total_cost == 2000.0);
}

TEST_CASE("AdaptiveEf fixed gate rejects observed non-monotonic baselines", "[ut][AdaptiveEf]") {
    const auto non_monotonic_success = [](uint64_t ef) {
        const uint64_t success = ef == 100 ? 8 : (ef == 200 ? 7 : 9);
        return AdaptiveEfFixedGatePoint{success, static_cast<double>(ef)};
    };
    auto match = MatchAdaptiveEfFixedGate(9, {100, 200, 300}, non_monotonic_success);
    REQUIRE_FALSE(match.trustworthy);

    const auto post_match_regression = [](uint64_t ef) {
        const uint64_t success = ef == 100 ? 8 : (ef == 200 ? 9 : 7);
        return AdaptiveEfFixedGatePoint{success, static_cast<double>(ef)};
    };
    match = MatchAdaptiveEfFixedGate(9, {100, 200, 300}, post_match_regression);
    REQUIRE_FALSE(match.trustworthy);

    const auto post_match_cost_regression = [](uint64_t ef) {
        const uint64_t success = ef < 200 ? uint64_t{8} : uint64_t{9};
        const double cost = ef == 300 ? 150.0 : static_cast<double>(ef);
        return AdaptiveEfFixedGatePoint{success, cost};
    };
    match = MatchAdaptiveEfFixedGate(9, {100, 200, 300}, post_match_cost_regression);
    REQUIRE_FALSE(match.trustworthy);

    const auto non_monotonic_cost = [](uint64_t ef) {
        return AdaptiveEfFixedGatePoint{ef < 150 ? uint64_t{8} : uint64_t{9},
                                        ef == 125 ? 90.0 : static_cast<double>(ef)};
    };
    match = MatchAdaptiveEfFixedGate(9, {100, 200}, non_monotonic_cost);
    REQUIRE_FALSE(match.trustworthy);
}

TEST_CASE("AdaptiveEf fixed gate accepts only trustworthy same-rate savings", "[ut][AdaptiveEf]") {
    AdaptiveEfFixedGateMatch fixed{
        .ef = 137, .success = 9, .total_cost = 1000.0, .matched = true, .trustworthy = true};
    REQUIRE(AdaptiveEfFixedGatePasses(900.0, fixed, 0.10));
    REQUIRE_FALSE(AdaptiveEfFixedGatePasses(901.0, fixed, 0.10));

    fixed.matched = false;
    REQUIRE_FALSE(AdaptiveEfFixedGatePasses(100.0, fixed, 0.10));
    fixed.matched = true;
    fixed.trustworthy = false;
    REQUIRE_FALSE(AdaptiveEfFixedGatePasses(100.0, fixed, 0.10));

    fixed.trustworthy = true;
    REQUIRE(AdaptiveEfFixedGatePasses(1000.0, fixed, 0.0));
    REQUIRE_FALSE(AdaptiveEfFixedGatePasses(1000.1, fixed, 0.0));
}

}  // namespace vsag
