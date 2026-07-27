
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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace vsag {

AdaptiveEfFixedGateMatch
MatchAdaptiveEfFixedGate(uint64_t required_success,
                         const std::vector<uint64_t>& sparse_efs,
                         const std::function<AdaptiveEfFixedGatePoint(uint64_t)>& evaluate) {
    if (sparse_efs.empty()) {
        throw std::invalid_argument("adaptive_ef fixed gate requires a non-empty ef grid");
    }
    for (uint64_t i = 0; i < sparse_efs.size(); ++i) {
        if (sparse_efs[i] == 0 or (i > 0 and sparse_efs[i] <= sparse_efs[i - 1])) {
            throw std::invalid_argument(
                "adaptive_ef fixed gate requires a positive, increasing ef grid");
        }
    }

    std::vector<AdaptiveEfFixedGatePoint> sparse_points;
    sparse_points.reserve(sparse_efs.size());
    uint64_t first_matched = sparse_efs.size();
    uint64_t fallback = 0;
    for (uint64_t i = 0; i < sparse_efs.size(); ++i) {
        auto point = evaluate(sparse_efs[i]);
        if (not std::isfinite(point.total_cost) or point.total_cost < 0) {
            throw std::runtime_error("adaptive_ef fixed gate observed an invalid cost");
        }
        sparse_points.push_back(point);
        if (point.success >= required_success and first_matched == sparse_efs.size()) {
            first_matched = i;
        }
        if (point.success > sparse_points[fallback].success or
            (point.success == sparse_points[fallback].success and
             point.total_cost < sparse_points[fallback].total_cost)) {
            fallback = i;
        }
        if (i > 0 and (point.total_cost < sparse_points[i - 1].total_cost or
                       point.success < sparse_points[i - 1].success)) {
            return {sparse_efs[fallback],
                    sparse_points[fallback].success,
                    sparse_points[fallback].total_cost,
                    false,
                    false};
        }
    }

    if (first_matched == sparse_efs.size()) {
        return {sparse_efs[fallback],
                sparse_points[fallback].success,
                sparse_points[fallback].total_cost,
                false,
                true};
    }
    if (first_matched == 0) {
        return {sparse_efs[0], sparse_points[0].success, sparse_points[0].total_cost, true, true};
    }

    uint64_t lower_ef = sparse_efs[first_matched - 1];
    auto lower = sparse_points[first_matched - 1];
    uint64_t upper_ef = sparse_efs[first_matched];
    auto upper = sparse_points[first_matched];
    while (lower_ef + 1 < upper_ef) {
        const uint64_t middle_ef = lower_ef + (upper_ef - lower_ef) / 2;
        const auto middle = evaluate(middle_ef);
        if (not std::isfinite(middle.total_cost) or middle.total_cost < 0) {
            throw std::runtime_error("adaptive_ef fixed gate observed an invalid cost");
        }
        if (middle.success < lower.success or middle.success > upper.success or
            middle.total_cost < lower.total_cost or middle.total_cost > upper.total_cost) {
            return {upper_ef, upper.success, upper.total_cost, false, false};
        }
        if (middle.success >= required_success) {
            upper_ef = middle_ef;
            upper = middle;
        } else {
            lower_ef = middle_ef;
            lower = middle;
        }
    }
    return {upper_ef, upper.success, upper.total_cost, true, true};
}

bool
AdaptiveEfFixedGatePasses(double adaptive_total_cost,
                          const AdaptiveEfFixedGateMatch& fixed_match,
                          double min_relative_saving) {
    if (not std::isfinite(adaptive_total_cost) or adaptive_total_cost < 0 or
        not std::isfinite(fixed_match.total_cost) or fixed_match.total_cost < 0 or
        not std::isfinite(min_relative_saving) or min_relative_saving < 0 or
        min_relative_saving >= 1) {
        return false;
    }
    return fixed_match.trustworthy and fixed_match.matched and
           adaptive_total_cost <= fixed_match.total_cost * (1.0 - min_relative_saving);
}

int
AdaptiveEfState::AlphaIndex(float alpha) {
    constexpr float kTolerance = 1e-6F;
    constexpr float kAlphas[] = {0.2F, 0.1F, 0.05F};
    for (int index = 0; index < 3; ++index) {
        if (std::abs(alpha - kAlphas[index]) <= kTolerance) {
            return index;
        }
    }
    return -1;
}

const AdaptiveEfHead*
AdaptiveEfState::FindHead(uint64_t topk, float target_recall) const {
    for (const auto& head : heads) {
        if (head.topk == topk and std::abs(head.target_recall - target_recall) < 1e-4F) {
            return &head;
        }
    }
    return nullptr;
}

const AdaptiveEfHead*
AdaptiveEfState::FindHead(float target_recall) const {
    return FindHead(100, target_recall);
}

void
AdaptiveEfState::QueryMoments(
    const float* query, int64_t dim, float* mu_q, float* sigma_q, float* q_norm) const {
    double norm = 0;
    for (int64_t d = 0; d < dim; ++d) {
        norm += static_cast<double>(query[d]) * query[d];
    }
    norm = std::sqrt(std::max(norm, 1e-24));
    double dot = 0;
    double quad = 0;
    for (int64_t d = 0; d < dim; ++d) {
        double qd = query[d] / norm;
        dot += qd * data_mean[d];
        quad += qd * qd * data_var[d];
    }
    *mu_q = static_cast<float>(1.0 - dot);
    *sigma_q = static_cast<float>(std::sqrt(std::max(quad, 1e-24)));
    *q_norm = static_cast<float>(norm);
}

void
AdaptiveEfState::ComputeFeatures(
    const std::vector<float>& dists_asc, float mu_q, float sigma_q, float q_norm, float* out13) {
    const auto n = static_cast<int64_t>(dists_asc.size());
    double mean = 0;
    for (float v : dists_asc) {
        mean += v;
    }
    mean /= static_cast<double>(n);
    double var = 0;
    for (float v : dists_asc) {
        var += (v - mean) * (v - mean);
    }
    var /= static_cast<double>(n);
    const float d1 = dists_asc.front();
    const float dk = dists_asc.back();
    const float safe_sigma = std::max(sigma_q, 1e-9F);
    out13[0] = d1;
    out13[1] = dk;
    out13[2] = dists_asc[n / 2];
    out13[3] = dk - d1;
    out13[4] = dk - dists_asc[n * 9 / 10];
    out13[5] = dists_asc[n / 10] - d1;
    out13[6] = static_cast<float>(mean);
    out13[7] = static_cast<float>(std::sqrt(var));
    out13[8] = mu_q;
    out13[9] = sigma_q;
    out13[10] = (dk - mu_q) / safe_sigma;
    out13[11] = (d1 - mu_q) / safe_sigma;
    out13[12] = q_norm;
}

double
AdaptiveEfState::PredictLog2FromFeatures(const AdaptiveEfHead& head, const float* features) {
    if (features == nullptr or head.feat_mean.size() != kAdaptiveEfFeatureCount or
        head.feat_stdev.size() != kAdaptiveEfFeatureCount or
        head.weights.size() != 2 * kAdaptiveEfFeatureCount + 1) {
        throw std::invalid_argument("adaptive_ef head or features have invalid sizes");
    }
    double prediction = head.weights[2 * kAdaptiveEfFeatureCount];
    if (not std::isfinite(prediction)) {
        throw std::invalid_argument("adaptive_ef head contains an invalid bias");
    }
    for (int64_t j = 0; j < kAdaptiveEfFeatureCount; ++j) {
        if (not std::isfinite(head.feat_mean[j]) or not std::isfinite(head.feat_stdev[j]) or
            head.feat_stdev[j] <= 0.0F or not std::isfinite(head.weights[j]) or
            not std::isfinite(head.weights[kAdaptiveEfFeatureCount + j])) {
            throw std::invalid_argument("adaptive_ef head contains invalid model values");
        }
        double z = (features[j] - head.feat_mean[j]) / head.feat_stdev[j];
        if (std::isnan(z)) {
            return std::numeric_limits<double>::infinity();
        }
        z = std::clamp(z, -8.0, 8.0);
        prediction += head.weights[j] * z + head.weights[kAdaptiveEfFeatureCount + j] * z * z;
    }
    return prediction;
}

uint64_t
AdaptiveEfState::PredictFromFeatures(const AdaptiveEfHead& head,
                                     const float* features,
                                     float alpha,
                                     uint64_t ef_min,
                                     uint64_t prediction_cap) const {
    const int alpha_index = AlphaIndex(alpha);
    if (alpha_index < 0) {
        throw std::invalid_argument("adaptive_ef alpha must be one of 0.2, 0.1, or 0.05");
    }
    if (not head.alpha_enabled[alpha_index]) {
        throw std::invalid_argument("adaptive_ef target/top-k/alpha combination is not enabled");
    }
    if (features == nullptr) {
        throw std::invalid_argument("adaptive_ef features must not be null");
    }
    if (ef_grid.empty()) {
        throw std::runtime_error("adaptive_ef ef grid is empty");
    }

    const uint64_t effective_cap = std::min(ef_cap, prediction_cap);
    if (effective_cap == 0 or ef_min > effective_cap) {
        throw std::invalid_argument("adaptive_ef cap must be positive and at least ef_min");
    }

    const double log2_ef = PredictLog2FromFeatures(head, features) + head.margins[alpha_index];
    uint64_t requested_ef = effective_cap;
    if (std::isfinite(log2_ef)) {
        const double ef = std::pow(2.0, log2_ef);
        if (std::isfinite(ef) and ef < static_cast<double>(effective_cap)) {
            requested_ef = std::max(ef_min, static_cast<uint64_t>(std::max(1.0, std::ceil(ef))));
        }
    }

    const auto rung = std::lower_bound(ef_grid.begin(), ef_grid.end(), requested_ef);
    if (rung == ef_grid.end() or *rung > effective_cap) {
        return effective_cap;
    }
    return *rung;
}

uint64_t
AdaptiveEfState::Predict(const AdaptiveEfHead& head,
                         const std::vector<float>& dists_asc,
                         float mu_q,
                         float sigma_q,
                         float q_norm,
                         float alpha,
                         uint64_t ef_min,
                         uint64_t prediction_cap) const {
    if (dists_asc.empty()) {
        throw std::invalid_argument("adaptive_ef candidate distances must not be empty");
    }
    float feats[kAdaptiveEfFeatureCount];
    ComputeFeatures(dists_asc, mu_q, sigma_q, q_norm, feats);
    return PredictFromFeatures(head, feats, alpha, ef_min, prediction_cap);
}

AdaptiveEfHead
AdaptiveEfState::TrainHead(const std::vector<float>& features,
                           const std::vector<float>& log2_required,
                           const std::vector<uint64_t>& fit_rows,
                           const std::vector<uint64_t>& cal_rows,
                           float target_recall,
                           uint64_t topk) {
    constexpr int64_t kF = kAdaptiveEfFeatureCount;
    constexpr float kTau = 0.8F;
    if (topk == 0 or not std::isfinite(target_recall) or target_recall <= 0.0F or
        target_recall > 1.0F) {
        throw std::invalid_argument("adaptive_ef head has invalid top-k or target recall");
    }
    if (log2_required.empty() or features.size() % kF != 0 or
        features.size() / kF != log2_required.size() or fit_rows.empty() or cal_rows.empty()) {
        throw std::invalid_argument("adaptive_ef training data has invalid sizes");
    }
    if (std::any_of(features.begin(),
                    features.end(),
                    [](float value) { return not std::isfinite(value); }) or
        std::any_of(log2_required.begin(), log2_required.end(), [](float value) {
            return not std::isfinite(value);
        })) {
        throw std::invalid_argument("adaptive_ef training data contains non-finite values");
    }
    std::vector<bool> row_seen(log2_required.size(), false);
    for (uint64_t row : fit_rows) {
        if (row >= log2_required.size() or row_seen[row]) {
            throw std::invalid_argument("adaptive_ef fit rows are invalid");
        }
        row_seen[row] = true;
    }
    for (uint64_t row : cal_rows) {
        if (row >= log2_required.size() or row_seen[row]) {
            throw std::invalid_argument("adaptive_ef calibration rows are invalid");
        }
        row_seen[row] = true;
    }
    AdaptiveEfHead head;
    head.topk = topk;
    head.target_recall = target_recall;
    std::fill(std::begin(head.alpha_enabled), std::end(head.alpha_enabled), true);
    head.feat_mean.assign(kF, 0.0F);
    head.feat_stdev.assign(kF, 1.0F);
    for (int64_t j = 0; j < kF; ++j) {
        double m = 0;
        for (auto i : fit_rows) {
            m += features[i * kF + j];
        }
        m /= static_cast<double>(fit_rows.size());
        double s = 0;
        for (auto i : fit_rows) {
            double d = features[i * kF + j] - m;
            s += d * d;
        }
        head.feat_mean[j] = static_cast<float>(m);
        head.feat_stdev[j] =
            static_cast<float>(std::max(std::sqrt(s / static_cast<double>(fit_rows.size())), 1e-6));
    }
    const int64_t p = 2 * kF + 1;
    head.weights.assign(p, 0.0F);
    std::vector<double> z(fit_rows.size() * kF);
    for (uint64_t r = 0; r < fit_rows.size(); ++r) {
        for (int64_t j = 0; j < kF; ++j) {
            z[r * kF + j] =
                (features[fit_rows[r] * kF + j] - head.feat_mean[j]) / head.feat_stdev[j];
            z[r * kF + j] = std::clamp(z[r * kF + j], -8.0, 8.0);
        }
    }
    {
        std::vector<float> ys;
        ys.reserve(fit_rows.size());
        for (auto i : fit_rows) {
            ys.push_back(log2_required[i]);
        }
        std::sort(ys.begin(), ys.end());
        head.weights[p - 1] = ys[static_cast<uint64_t>(kTau * static_cast<float>(ys.size() - 1))];
    }
    std::vector<double> m1(p, 0.0);
    std::vector<double> m2(p, 0.0);
    const double lr = 0.03;
    for (int iter = 1; iter <= 4000; ++iter) {
        std::vector<double> grad(p, 0.0);
        for (uint64_t r = 0; r < fit_rows.size(); ++r) {
            const double* zr = z.data() + r * kF;
            double pred = head.weights[p - 1];
            for (int64_t j = 0; j < kF; ++j) {
                pred += head.weights[j] * zr[j] + head.weights[kF + j] * zr[j] * zr[j];
            }
            double err = log2_required[fit_rows[r]] - pred;
            double gsign = err > 0 ? -static_cast<double>(kTau) : (1.0 - kTau);
            for (int64_t j = 0; j < kF; ++j) {
                grad[j] += gsign * zr[j];
                grad[kF + j] += gsign * zr[j] * zr[j];
            }
            grad[p - 1] += gsign;
        }
        for (int64_t j = 0; j < p; ++j) {
            grad[j] /= static_cast<double>(fit_rows.size());
            m1[j] = 0.9 * m1[j] + 0.1 * grad[j];
            m2[j] = 0.999 * m2[j] + 0.001 * grad[j] * grad[j];
            double mh = m1[j] / (1 - std::pow(0.9, iter));
            double vh = m2[j] / (1 - std::pow(0.999, iter));
            head.weights[j] -= static_cast<float>(lr * mh / (std::sqrt(vh) + 1e-8));
        }
    }
    // one-sided conformal margins on calibration rows
    std::vector<float> resid;
    resid.reserve(cal_rows.size());
    for (auto i : cal_rows) {
        const double pred = PredictLog2FromFeatures(head, features.data() + i * kF);
        resid.push_back(log2_required[i] - static_cast<float>(pred));
    }
    std::sort(resid.begin(), resid.end());
    const float alphas[3] = {0.2F, 0.1F, 0.05F};
    for (int a = 0; a < 3; ++a) {
        double level =
            std::min(1.0,
                     std::ceil((1.0 - alphas[a]) * static_cast<double>(resid.size() + 1)) /
                         static_cast<double>(resid.size()));
        auto idx = static_cast<uint64_t>(level * static_cast<double>(resid.size() - 1));
        head.margins[a] = resid[std::min(idx, static_cast<uint64_t>(resid.size() - 1))];
    }
    return head;
}

float
AdaptiveEfState::SpearmanCorrelation(const std::vector<float>& a, const std::vector<float>& b) {
    const auto n = static_cast<int64_t>(a.size());
    if (n < 3 or b.size() != a.size()) {
        return 0.0F;
    }
    auto ranks = [n](const std::vector<float>& v) {
        std::vector<int64_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return v[x] < v[y]; });
        std::vector<double> rank(n);
        int64_t begin = 0;
        while (begin < n) {
            int64_t end = begin + 1;
            while (end < n and v[order[begin]] == v[order[end]]) {
                ++end;
            }
            const double average_rank =
                (static_cast<double>(begin) + static_cast<double>(end - 1)) / 2.0;
            for (int64_t i = begin; i < end; ++i) {
                rank[order[i]] = average_rank;
            }
            begin = end;
        }
        return rank;
    };
    if (std::any_of(a.begin(), a.end(), [](float value) { return not std::isfinite(value); }) or
        std::any_of(b.begin(), b.end(), [](float value) { return not std::isfinite(value); })) {
        return 0.0F;
    }
    auto ra = ranks(a);
    auto rb = ranks(b);
    double ma = 0;
    double mb = 0;
    for (int64_t i = 0; i < n; ++i) {
        ma += ra[i];
        mb += rb[i];
    }
    ma /= static_cast<double>(n);
    mb /= static_cast<double>(n);
    double num = 0;
    double da = 0;
    double db = 0;
    for (int64_t i = 0; i < n; ++i) {
        num += (ra[i] - ma) * (rb[i] - mb);
        da += (ra[i] - ma) * (ra[i] - ma);
        db += (rb[i] - mb) * (rb[i] - mb);
    }
    const double denominator = std::sqrt(da * db);
    if (denominator <= std::numeric_limits<double>::epsilon()) {
        return 0.0F;
    }
    return static_cast<float>(num / denominator);
}

namespace {

constexpr uint64_t kAdaptiveEfStateVersion = 2;
constexpr uint64_t kMaxAdaptiveEfTargets = 64;
constexpr uint64_t kMaxAdaptiveEfTopks = 64;
constexpr uint64_t kMaxAdaptiveEfGridSize = 4096;
constexpr uint64_t kMaxAdaptiveEfDimension = 1U << 20;
constexpr uint64_t kMaxAdaptiveEfHeads = kMaxAdaptiveEfTargets * kMaxAdaptiveEfTopks;
constexpr uint64_t kMaxAdaptiveEfReasonLength = 1U << 20;

template <typename T>
void
write_pod(std::string& out, const T& v) {
    out.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
void
read_pod(const std::string& in, uint64_t& off, T& v) {
    if (off > in.size() or sizeof(T) > in.size() - off) {
        throw std::runtime_error("adaptive_ef state blob truncated");
    }
    std::memcpy(&v, in.data() + off, sizeof(T));
    off += sizeof(T);
}

void
write_bool(std::string& out, bool value) {
    write_pod(out, static_cast<uint8_t>(value));
}

void
read_bool(const std::string& in, uint64_t& off, bool& value) {
    uint8_t encoded = 0;
    read_pod(in, off, encoded);
    if (encoded > 1) {
        throw std::runtime_error("adaptive_ef state has invalid boolean");
    }
    value = encoded == 1;
}

template <typename T>
void
write_vec(std::string& out, const std::vector<T>& v) {
    write_pod(out, static_cast<uint64_t>(v.size()));
    if (not v.empty()) {
        out.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
    }
}

template <typename T>
void
read_vec(const std::string& in, uint64_t& off, std::vector<T>& v, uint64_t max_count) {
    uint64_t n = 0;
    read_pod(in, off, n);
    if (n > max_count or off > in.size() or n > (in.size() - off) / sizeof(T)) {
        throw std::runtime_error("adaptive_ef state blob truncated");
    }
    v.resize(n);
    if (n > 0) {
        std::memcpy(v.data(), in.data() + off, n * sizeof(T));
    }
    off += n * sizeof(T);
}

void
write_string(std::string& out, const std::string& value) {
    write_pod(out, static_cast<uint64_t>(value.size()));
    out.append(value);
}

void
read_string(const std::string& in, uint64_t& off, std::string& value, uint64_t max_length) {
    uint64_t length = 0;
    read_pod(in, off, length);
    if (length > max_length or off > in.size() or length > in.size() - off) {
        throw std::runtime_error("adaptive_ef state blob truncated");
    }
    value.assign(in.data() + off, length);
    off += length;
}

std::vector<uint64_t>
legacy_ef_grid(uint64_t ef_cap) {
    std::vector<uint64_t> result;
    constexpr uint64_t kLegacyGrid[] = {100,  125,  150,  175,  200,  250,  300,  350,
                                        400,  500,  600,  700,  800,  1000, 1200, 1400,
                                        1600, 2000, 2400, 2800, 3200, 4000, 5000};
    for (uint64_t ef : kLegacyGrid) {
        if (ef <= ef_cap) {
            result.push_back(ef);
        }
    }
    if (ef_cap > 0 and (result.empty() or result.back() != ef_cap)) {
        result.push_back(ef_cap);
    }
    return result;
}

bool
contains_close(const std::vector<float>& values, float target) {
    return std::any_of(values.begin(), values.end(), [target](float value) {
        return std::abs(value - target) < 1e-4F;
    });
}

}  // namespace

std::string
AdaptiveEfState::SerializeToString() const {
    Validate(static_cast<int64_t>(data_mean.size()));
    std::string out;
    write_pod(out, kAdaptiveEfStateVersion);
    write_bool(out, enabled);
    write_bool(out, calibrated);
    write_pod(out, spearman);
    write_pod(out, sample_count);
    write_pod(out, ef_cap);
    write_vec(out, targets);
    write_vec(out, topks);
    write_vec(out, ef_grid);
    write_string(out, disabled_reason);
    write_vec(out, data_mean);
    write_vec(out, data_var);
    write_pod(out, static_cast<uint64_t>(heads.size()));
    for (const auto& head : heads) {
        write_pod(out, head.topk);
        write_pod(out, head.target_recall);
        write_vec(out, head.feat_mean);
        write_vec(out, head.feat_stdev);
        write_vec(out, head.weights);
        out.append(reinterpret_cast<const char*>(head.margins), sizeof(head.margins));
        for (bool alpha_enabled : head.alpha_enabled) {
            write_bool(out, alpha_enabled);
        }
    }
    return out;
}

void
AdaptiveEfState::DeserializeFromString(const std::string& blob) {
    uint64_t off = 0;
    uint64_t version = 0;
    read_pod(blob, off, version);
    if (version != 1 and version != kAdaptiveEfStateVersion) {
        throw std::runtime_error("unsupported adaptive_ef state version");
    }
    read_bool(blob, off, enabled);
    read_bool(blob, off, calibrated);
    read_pod(blob, off, spearman);
    read_pod(blob, off, sample_count);
    read_pod(blob, off, ef_cap);
    read_vec(blob, off, targets, kMaxAdaptiveEfTargets);
    if (version == kAdaptiveEfStateVersion) {
        read_vec(blob, off, topks, kMaxAdaptiveEfTopks);
        read_vec(blob, off, ef_grid, kMaxAdaptiveEfGridSize);
        read_string(blob, off, disabled_reason, kMaxAdaptiveEfReasonLength);
    } else {
        topks = {100};
        ef_grid = legacy_ef_grid(ef_cap);
        disabled_reason.clear();
    }
    read_vec(blob, off, data_mean, kMaxAdaptiveEfDimension);
    read_vec(blob, off, data_var, kMaxAdaptiveEfDimension);
    uint64_t head_count = 0;
    read_pod(blob, off, head_count);
    if (head_count > kMaxAdaptiveEfHeads or head_count > blob.size() / sizeof(uint64_t)) {
        throw std::runtime_error("adaptive_ef state head count is invalid");
    }
    heads.resize(head_count);
    for (auto& head : heads) {
        if (version == kAdaptiveEfStateVersion) {
            read_pod(blob, off, head.topk);
        } else {
            head.topk = 100;
        }
        read_pod(blob, off, head.target_recall);
        read_vec(blob, off, head.feat_mean, kAdaptiveEfFeatureCount);
        read_vec(blob, off, head.feat_stdev, kAdaptiveEfFeatureCount);
        read_vec(blob, off, head.weights, 2 * kAdaptiveEfFeatureCount + 1);
        if (off > blob.size() or sizeof(head.margins) > blob.size() - off) {
            throw std::runtime_error("adaptive_ef state blob truncated");
        }
        std::memcpy(head.margins, blob.data() + off, sizeof(head.margins));
        off += sizeof(head.margins);
        if (version == kAdaptiveEfStateVersion) {
            for (bool& alpha_enabled : head.alpha_enabled) {
                read_bool(blob, off, alpha_enabled);
            }
        } else {
            std::fill(std::begin(head.alpha_enabled), std::end(head.alpha_enabled), false);
        }
    }
    if (off != blob.size()) {
        throw std::runtime_error("adaptive_ef state blob has trailing bytes");
    }
    if (version == 1) {
        calibrated = false;
        disabled_reason = "legacy adaptive_ef state must be rebuilt";
    }
    Validate(static_cast<int64_t>(data_mean.size()));
}

void
AdaptiveEfState::Validate(int64_t dim) const {
    auto fail = [](const std::string& reason) {
        throw std::runtime_error("invalid adaptive_ef state: " + reason);
    };
    auto finite = [](const std::vector<float>& values) {
        return std::all_of(
            values.begin(), values.end(), [](float value) { return std::isfinite(value); });
    };

    if (dim < 0) {
        fail("negative dimension");
    }
    if (static_cast<uint64_t>(dim) > kMaxAdaptiveEfDimension) {
        fail("dimension exceeds serialization limit");
    }
    if (calibrated and not enabled) {
        fail("calibrated state must be enabled");
    }
    if (not std::isfinite(spearman) or spearman < -1.0F or spearman > 1.0F) {
        fail("spearman must be finite and in [-1, 1]");
    }
    if (enabled and sample_count == 0) {
        fail("sample_count must be positive");
    }
    if (enabled and ef_cap == 0) {
        fail("ef_cap must be positive");
    }
    if (targets.empty() or targets.size() > kMaxAdaptiveEfTargets or not finite(targets)) {
        fail("targets must be non-empty and finite");
    }
    for (uint64_t i = 0; i < targets.size(); ++i) {
        if (targets[i] <= 0.0F or targets[i] > 1.0F) {
            fail("target recall must be in (0, 1]");
        }
        if (i > 0 and targets[i] - targets[i - 1] < 1e-4F) {
            fail("targets must be strictly increasing and unambiguous");
        }
    }
    if (topks.empty() or topks.size() > kMaxAdaptiveEfTopks) {
        fail("topks must be non-empty");
    }
    for (uint64_t i = 0; i < topks.size(); ++i) {
        if (topks[i] == 0 or topks[i] > ef_cap) {
            fail("top-k must be positive and no greater than ef_cap");
        }
        if (i > 0 and topks[i] <= topks[i - 1]) {
            fail("topks must be strictly increasing");
        }
    }
    if (ef_grid.size() > kMaxAdaptiveEfGridSize) {
        fail("ef grid exceeds serialization limit");
    }
    if (calibrated and ef_grid.empty()) {
        fail("calibrated state requires an ef grid");
    }
    for (uint64_t i = 0; i < ef_grid.size(); ++i) {
        if (ef_grid[i] == 0 or ef_grid[i] > ef_cap) {
            fail("ef grid entries must be positive and no greater than ef_cap");
        }
        if (i > 0 and ef_grid[i] <= ef_grid[i - 1]) {
            fail("ef grid must be strictly increasing");
        }
    }
    if (not ef_grid.empty() and ef_grid.back() != ef_cap) {
        fail("ef grid must include ef_cap as its final entry");
    }

    if (data_mean.size() != data_var.size() or not finite(data_mean) or not finite(data_var)) {
        fail("dataset moments have invalid sizes or non-finite values");
    }
    if ((calibrated or not data_mean.empty()) and data_mean.size() != static_cast<uint64_t>(dim)) {
        fail("dataset moment dimension mismatch");
    }
    if (std::any_of(data_var.begin(), data_var.end(), [](float value) { return value < 0.0F; })) {
        fail("dataset variance must be non-negative");
    }

    if (disabled_reason.size() > kMaxAdaptiveEfReasonLength) {
        fail("disabled reason exceeds serialization limit");
    }
    if (heads.size() > kMaxAdaptiveEfHeads) {
        fail("head count exceeds serialization limit");
    }
    bool any_enabled = false;
    std::set<std::pair<uint64_t, float>> head_keys;
    for (const auto& head : heads) {
        if (head.topk == 0 or std::find(topks.begin(), topks.end(), head.topk) == topks.end()) {
            fail("head has an unknown top-k");
        }
        if (not std::isfinite(head.target_recall) or
            not contains_close(targets, head.target_recall)) {
            fail("head has an unknown target recall");
        }
        if (not head_keys.emplace(head.topk, head.target_recall).second) {
            fail("duplicate head for top-k and target recall");
        }
        if (head.feat_mean.size() != kAdaptiveEfFeatureCount or
            head.feat_stdev.size() != kAdaptiveEfFeatureCount or
            head.weights.size() != 2 * kAdaptiveEfFeatureCount + 1) {
            fail("head feature vectors have invalid sizes");
        }
        if (not finite(head.feat_mean) or not finite(head.feat_stdev) or not finite(head.weights)) {
            fail("head contains non-finite model values");
        }
        if (std::any_of(head.feat_stdev.begin(), head.feat_stdev.end(), [](float value) {
                return value <= 0.0F;
            })) {
            fail("head feature standard deviations must be positive");
        }
        for (int alpha_index = 0; alpha_index < 3; ++alpha_index) {
            if (not std::isfinite(head.margins[alpha_index])) {
                fail("head contains a non-finite margin");
            }
            any_enabled = any_enabled or head.alpha_enabled[alpha_index];
        }
    }
    if (calibrated and (heads.empty() or not any_enabled)) {
        fail("calibrated state has no enabled head");
    }
}

}  // namespace vsag
