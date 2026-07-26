
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
// Everything is calibrated inside HGraph::Build() when the `adaptive_ef` build
// parameter is enabled.

constexpr int64_t kAdaptiveEfFeatureCount = 13;

struct AdaptiveEfHead {
    float target_recall{0.0F};
    std::vector<float> feat_mean;   // 13
    std::vector<float> feat_stdev;  // 13
    std::vector<float> weights;     // 27 = 13 linear + 13 squared + bias
    float margins[3] = {0.0F, 0.0F, 0.0F};  // conformal, alpha in {0.2, 0.1, 0.05}

    float
    Margin(float alpha) const {
        if (alpha >= 0.15F) {
            return margins[0];
        }
        if (alpha >= 0.075F) {
            return margins[1];
        }
        return margins[2];
    }
};

class AdaptiveEfState {
public:
    // ---- configuration (from the build parameter) ----
    bool enabled{false};
    uint64_t sample_count{1000};
    std::vector<float> targets{0.90F, 0.95F, 0.99F};
    uint64_t ef_cap{5000};

    // ---- calibrated state ----
    bool calibrated{false};
    float spearman{0.0F};
    std::string disabled_reason;
    std::vector<float> data_mean;  // dim, over stored vectors
    std::vector<float> data_var;   // dim, per-dimension variance
    std::vector<AdaptiveEfHead> heads;

public:
    const AdaptiveEfHead*
    FindHead(float target_recall) const;

    // mu_q = 1 - <q_unit, mean>, sigma_q = sqrt(sum(q_unit^2 * var)); O(dim)
    void
    QueryMoments(const float* query, int64_t dim, float* mu_q, float* sigma_q, float* q_norm) const;

    // dists_asc: current top candidates, ascending. Returns the predicted ef.
    static uint64_t
    Predict(const AdaptiveEfHead& head,
            const std::vector<float>& dists_asc,
            float mu_q,
            float sigma_q,
            float q_norm,
            float alpha,
            uint64_t ef_min,
            uint64_t ef_cap);

    static void
    ComputeFeatures(const std::vector<float>& dists_asc,
                    float mu_q,
                    float sigma_q,
                    float q_norm,
                    float* out13);

    // Fit one head with pinball loss (Adam, full batch) on log2(required_ef),
    // then compute conformal margins on the calibration rows.
    static AdaptiveEfHead
    TrainHead(const std::vector<float>& features,  // n x 13
              const std::vector<float>& log2_required,
              const std::vector<uint64_t>& fit_rows,
              const std::vector<uint64_t>& cal_rows,
              float target_recall);

    static float
    SpearmanCorrelation(const std::vector<float>& a, const std::vector<float>& b);

    std::string
    SerializeToString() const;

    void
    DeserializeFromString(const std::string& blob);
};

}  // namespace vsag
