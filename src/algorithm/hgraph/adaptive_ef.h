
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

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vsag {

// Per-query adaptive ef-search ("declarative recall"): a linear quantile head
// predicts, from the state of an ordinary ef=k search at its would-be
// termination point, the ef needed to reach a declared recall target; the
// search then resumes in place (see InnerSearchParam::adaptive_ef_hook).
//
// The state is index-owned: dataset moment vectors (mean + per-dim variance of
// the stored vectors) and one trained head per calibrated recall target.
// Calibration can run inside HGraph::Build() for backward compatibility or
// explicitly on an existing index through Index::EnableAdaptiveEf().

constexpr int64_t kAdaptiveEfFeatureCount = 13;

struct AdaptiveEfFixedGatePoint {
    uint64_t success{0};
    double total_cost{0};
};

struct AdaptiveEfFixedGateMatch {
    uint64_t ef{0};
    uint64_t success{0};
    double total_cost{0};
    bool matched{false};
    bool trustworthy{false};
};

// Refine a sparse fixed-ef frontier to an integer ef with at least required_success.
// The evaluator must measure the same held-out query subset at every ef. Observed
// non-monotonic success or work fails closed instead of overstating fixed-ef cost.
AdaptiveEfFixedGateMatch
MatchAdaptiveEfFixedGate(uint64_t required_success,
                         const std::vector<uint64_t>& sparse_efs,
                         const std::function<AdaptiveEfFixedGatePoint(uint64_t)>& evaluate);

// Apply the same-rate and relative-cost acceptance rule to one aggregate or stratum.
bool
AdaptiveEfFixedGatePasses(double adaptive_total_cost,
                          const AdaptiveEfFixedGateMatch& fixed_match,
                          double min_relative_saving);

struct AdaptiveEfHead {
    uint64_t topk{100};
    float target_recall{0.0F};
    std::vector<float> feat_mean;           // 13
    std::vector<float> feat_stdev;          // 13
    std::vector<float> weights;             // 27 = 13 linear + 13 squared + bias
    float margins[3] = {0.0F, 0.0F, 0.0F};  // conformal, alpha in {0.2, 0.1, 0.05}
    bool alpha_enabled[3] = {false, false, false};
};

class AdaptiveEfState {
public:
    // ---- configuration (from the build parameter or EnableAdaptiveEf()) ----
    bool enabled{false};
    uint64_t sample_count{1000};
    std::vector<float> targets{0.90F, 0.95F, 0.99F};
    std::vector<uint64_t> topks{100};
    uint64_t ef_cap{5000};
    std::vector<uint64_t> ef_grid;

    // ---- calibrated state ----
    bool calibrated{false};
    float spearman{0.0F};
    std::string disabled_reason;
    std::vector<float> data_mean;  // dim, over stored vectors
    std::vector<float> data_var;   // dim, per-dimension variance
    std::vector<AdaptiveEfHead> heads;

public:
    static int
    AlphaIndex(float alpha);

    const AdaptiveEfHead*
    FindHead(uint64_t topk, float target_recall) const;

    // Compatibility helper for version-1 states, which were calibrated only at top-k=100.
    const AdaptiveEfHead*
    FindHead(float target_recall) const;

    // mu_q = 1 - <q_unit, mean>, sigma_q = sqrt(sum(q_unit^2 * var)); O(dim)
    void
    QueryMoments(const float* query, int64_t dim, float* mu_q, float* sigma_q, float* q_norm) const;

    // dists_asc: current top candidates, ascending. Returns the predicted ef.
    uint64_t
    Predict(const AdaptiveEfHead& head,
            const std::vector<float>& dists_asc,
            float mu_q,
            float sigma_q,
            float q_norm,
            float alpha,
            uint64_t ef_min,
            uint64_t ef_cap) const;

    // Predict from precomputed raw features. Both calibration gates and runtime
    // search must use this method so feature clamping, alpha handling, and
    // ef-grid snapping remain identical.
    uint64_t
    PredictFromFeatures(const AdaptiveEfHead& head,
                        const float* features,
                        float alpha,
                        uint64_t ef_min,
                        uint64_t ef_cap) const;

    // Returns the pre-margin log2(ef) prediction using the same standardized,
    // clamped feature transform as PredictFromFeatures().
    static double
    PredictLog2FromFeatures(const AdaptiveEfHead& head, const float* features);

    static void
    ComputeFeatures(
        const std::vector<float>& dists_asc, float mu_q, float sigma_q, float q_norm, float* out13);

    // Fit one head with pinball loss (Adam, full batch) on log2(required_ef),
    // then compute conformal margins on the calibration rows.
    static AdaptiveEfHead
    TrainHead(const std::vector<float>& features,  // n x 13
              const std::vector<float>& log2_required,
              const std::vector<uint64_t>& fit_rows,
              const std::vector<uint64_t>& cal_rows,
              float target_recall,
              uint64_t topk = 100);

    static float
    SpearmanCorrelation(const std::vector<float>& a, const std::vector<float>& b);

    std::string
    SerializeToString() const;

    void
    DeserializeFromString(const std::string& blob);

    // Validate the configuration and calibrated model against the index dimension.
    // Throws std::runtime_error on incompatible or corrupt state.
    void
    Validate(int64_t dim) const;
};

}  // namespace vsag
