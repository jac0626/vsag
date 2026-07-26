
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
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace vsag {

const AdaptiveEfHead*
AdaptiveEfState::FindHead(float target_recall) const {
    for (const auto& head : heads) {
        if (std::abs(head.target_recall - target_recall) < 1e-4F) {
            return &head;
        }
    }
    return nullptr;
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
AdaptiveEfState::ComputeFeatures(const std::vector<float>& dists_asc,
                                 float mu_q,
                                 float sigma_q,
                                 float q_norm,
                                 float* out13) {
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

uint64_t
AdaptiveEfState::Predict(const AdaptiveEfHead& head,
                         const std::vector<float>& dists_asc,
                         float mu_q,
                         float sigma_q,
                         float q_norm,
                         float alpha,
                         uint64_t ef_min,
                         uint64_t ef_cap) {
    if (dists_asc.empty()) {
        return ef_min;
    }
    float feats[kAdaptiveEfFeatureCount];
    ComputeFeatures(dists_asc, mu_q, sigma_q, q_norm, feats);
    double acc = head.weights[2 * kAdaptiveEfFeatureCount];
    for (int64_t j = 0; j < kAdaptiveEfFeatureCount; ++j) {
        double z = (feats[j] - head.feat_mean[j]) / head.feat_stdev[j];
        z = std::clamp(z, -8.0, 8.0);
        acc += head.weights[j] * z + head.weights[kAdaptiveEfFeatureCount + j] * z * z;
    }
    double ef = std::pow(2.0, acc + head.Margin(alpha));
    ef = std::min(ef, static_cast<double>(ef_cap));
    return std::max(ef_min, static_cast<uint64_t>(ef));
}

AdaptiveEfHead
AdaptiveEfState::TrainHead(const std::vector<float>& features,
                           const std::vector<float>& log2_required,
                           const std::vector<uint64_t>& fit_rows,
                           const std::vector<uint64_t>& cal_rows,
                           float target_recall) {
    constexpr int64_t kF = kAdaptiveEfFeatureCount;
    constexpr float kTau = 0.8F;
    AdaptiveEfHead head;
    head.target_recall = target_recall;
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
        double pred = head.weights[p - 1];
        for (int64_t j = 0; j < kF; ++j) {
            double zv = (features[i * kF + j] - head.feat_mean[j]) / head.feat_stdev[j];
            pred += head.weights[j] * zv + head.weights[kF + j] * zv * zv;
        }
        resid.push_back(log2_required[i] - static_cast<float>(pred));
    }
    std::sort(resid.begin(), resid.end());
    const float alphas[3] = {0.2F, 0.1F, 0.05F};
    for (int a = 0; a < 3; ++a) {
        double level = std::min(
            1.0,
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
    if (n < 3) {
        return 0.0F;
    }
    auto ranks = [n](const std::vector<float>& v) {
        std::vector<int64_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int64_t x, int64_t y) { return v[x] < v[y]; });
        std::vector<double> rank(n);
        for (int64_t i = 0; i < n; ++i) {
            rank[order[i]] = static_cast<double>(i);
        }
        return rank;
    };
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
    return static_cast<float>(num / std::max(std::sqrt(da * db), 1e-12));
}

namespace {

template <typename T>
void
write_pod(std::string& out, const T& v) {
    out.append(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
void
read_pod(const std::string& in, uint64_t& off, T& v) {
    if (off + sizeof(T) > in.size()) {
        throw std::runtime_error("adaptive_ef state blob truncated");
    }
    std::memcpy(&v, in.data() + off, sizeof(T));
    off += sizeof(T);
}

void
write_vec(std::string& out, const std::vector<float>& v) {
    write_pod(out, static_cast<uint64_t>(v.size()));
    out.append(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
}

void
read_vec(const std::string& in, uint64_t& off, std::vector<float>& v) {
    uint64_t n = 0;
    read_pod(in, off, n);
    if (off + n * sizeof(float) > in.size()) {
        throw std::runtime_error("adaptive_ef state blob truncated");
    }
    v.resize(n);
    std::memcpy(v.data(), in.data() + off, n * sizeof(float));
    off += n * sizeof(float);
}

}  // namespace

std::string
AdaptiveEfState::SerializeToString() const {
    std::string out;
    constexpr uint64_t kVersion = 1;
    write_pod(out, kVersion);
    write_pod(out, enabled);
    write_pod(out, calibrated);
    write_pod(out, spearman);
    write_pod(out, sample_count);
    write_pod(out, ef_cap);
    write_vec(out, targets);
    write_vec(out, data_mean);
    write_vec(out, data_var);
    write_pod(out, static_cast<uint64_t>(heads.size()));
    for (const auto& head : heads) {
        write_pod(out, head.target_recall);
        write_vec(out, head.feat_mean);
        write_vec(out, head.feat_stdev);
        write_vec(out, head.weights);
        out.append(reinterpret_cast<const char*>(head.margins), sizeof(head.margins));
    }
    return out;
}

void
AdaptiveEfState::DeserializeFromString(const std::string& blob) {
    uint64_t off = 0;
    uint64_t version = 0;
    read_pod(blob, off, version);
    if (version != 1) {
        throw std::runtime_error("unsupported adaptive_ef state version");
    }
    read_pod(blob, off, enabled);
    read_pod(blob, off, calibrated);
    read_pod(blob, off, spearman);
    read_pod(blob, off, sample_count);
    read_pod(blob, off, ef_cap);
    read_vec(blob, off, targets);
    read_vec(blob, off, data_mean);
    read_vec(blob, off, data_var);
    uint64_t head_count = 0;
    read_pod(blob, off, head_count);
    heads.resize(head_count);
    for (auto& head : heads) {
        read_pod(blob, off, head.target_recall);
        read_vec(blob, off, head.feat_mean);
        read_vec(blob, off, head.feat_stdev);
        read_vec(blob, off, head.weights);
        if (off + sizeof(head.margins) > blob.size()) {
            throw std::runtime_error("adaptive_ef state blob truncated");
        }
        std::memcpy(head.margins, blob.data() + off, sizeof(head.margins));
        off += sizeof(head.margins);
    }
}

}  // namespace vsag
