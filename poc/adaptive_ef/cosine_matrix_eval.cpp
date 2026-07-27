// Copyright 2026-present the vsag project
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

#include <vsag/vsag.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr uint64_t kInitialEf = 100;
constexpr double kAlpha = 0.05;
constexpr std::array<uint64_t, 3> kTopKs = {10, 50, 100};
constexpr std::array<double, 3> kRecallTargets = {0.90, 0.95, 0.99};
constexpr std::array<uint64_t, 23> kFixedEfFrontier = {
    100, 125,  150,  175,  200,  250,  300,  350,  400,  500,  600, 700,
    800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 4000, 5000};

struct MatrixHeader {
    uint64_t rows{0};
    uint64_t cols{0};
    uint64_t element_count{0};
    uint64_t payload_bytes{0};
};

struct SearchSummary {
    double qps{0.0};
    double avg_recall{0.0};
    double mean_dist_cmp{0.0};
    std::array<double, kRecallTargets.size()> pass_fractions{};
};

template <typename T>
MatrixHeader
InspectMatrix(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (not input) {
        throw std::runtime_error("cannot open matrix file: " + path);
    }

    int32_t rows = 0;
    int32_t cols = 0;
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (not input) {
        throw std::runtime_error("matrix header is truncated: " + path);
    }
    if (rows <= 0 or cols <= 0) {
        throw std::runtime_error("matrix rows and columns must be positive: " + path);
    }

    const uint64_t row_count = static_cast<uint64_t>(rows);
    const uint64_t col_count = static_cast<uint64_t>(cols);
    if (row_count > std::numeric_limits<uint64_t>::max() / col_count) {
        throw std::runtime_error("matrix element count overflows: " + path);
    }
    const uint64_t element_count = row_count * col_count;
    if (element_count > std::numeric_limits<uint64_t>::max() / sizeof(T)) {
        throw std::runtime_error("matrix payload size overflows: " + path);
    }
    const uint64_t payload_bytes = element_count * sizeof(T);
    if (payload_bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("matrix payload is too large for this platform: " + path);
    }

    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    if (file_size < 0 or static_cast<uint64_t>(file_size) != 2 * sizeof(int32_t) + payload_bytes) {
        throw std::runtime_error("matrix file size does not match its header: " + path);
    }
    return {row_count, col_count, element_count, payload_bytes};
}

template <typename T>
std::vector<T>
ReadMatrixPayload(const std::string& path, const MatrixHeader& header) {
    std::ifstream input(path, std::ios::binary);
    if (not input) {
        throw std::runtime_error("cannot open matrix file: " + path);
    }
    input.seekg(2 * sizeof(int32_t), std::ios::beg);
    std::vector<T> values(header.element_count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(header.payload_bytes));
    if (not input) {
        throw std::runtime_error("matrix payload is truncated: " + path);
    }
    return values;
}

uint64_t
ParseUnsigned(const std::string& text, const std::string& context) {
    if (text.empty()) {
        throw std::runtime_error("missing unsigned integer for " + context);
    }
    uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} or parsed.ptr != text.data() + text.size()) {
        throw std::runtime_error("invalid unsigned integer for " + context + ": " + text);
    }
    return value;
}

std::string
NormalizedAbsolutePath(const std::string& path) {
    std::error_code error;
    auto absolute = fs::absolute(path, error);
    if (error) {
        throw std::runtime_error("cannot resolve path " + path + ": " + error.message());
    }
    return absolute.lexically_normal().string();
}

bool
PathExists(const std::string& path) {
    std::error_code error;
    const bool exists = fs::exists(path, error);
    if (error) {
        throw std::runtime_error("cannot inspect path " + path + ": " + error.message());
    }
    return exists;
}

std::string
BuildParameters(uint64_t dim) {
    std::ostringstream params;
    params << R"({"dtype":"float32","metric_type":"cosine","dim":)" << dim
           << R"(,"index_param":{"base_quantization_type":"fp32","max_degree":64,)"
           << R"("ef_construction":500}})";
    return params.str();
}

std::string
AdaptiveTrainingParameters() {
    return R"({"sample_count":1000,"ef_cap":5000,"targets":"0.90,0.95,0.99",)"
           R"("topks":"10,50,100"})";
}

std::string
AdaptiveSearchParameters(double target) {
    std::ostringstream params;
    params << std::fixed << std::setprecision(2)
           << R"({"hgraph":{"ef_search":100,"adaptive_ef":{"target_recall":)" << target
           << R"(,"alpha":)" << kAlpha << "}}}";
    return params.str();
}

std::string
FixedSearchParameters(uint64_t ef_search) {
    return R"({"hgraph":{"ef_search":)" + std::to_string(ef_search) + "}}";
}

vsag::IndexPtr
CreateIndex(const std::string& build_parameters) {
    auto created = vsag::Factory::CreateIndex("hgraph", build_parameters);
    if (not created.has_value()) {
        throw std::runtime_error("failed to create HGraph: " + created.error().message);
    }
    return std::move(created).value();
}

vsag::IndexPtr
LoadIndex(const std::string& index_path, const std::string& build_parameters) {
    auto index = CreateIndex(build_parameters);
    std::ifstream input(index_path, std::ios::binary);
    if (not input) {
        throw std::runtime_error("cannot open index for reading: " + index_path);
    }
    auto loaded = index->Deserialize(input);
    if (not loaded.has_value()) {
        throw std::runtime_error("failed to load index: " + loaded.error().message);
    }
    std::cout << "Loaded index: " << index_path << '\n';
    return index;
}

void
SaveIndex(const vsag::IndexPtr& index, const std::string& index_path) {
    const std::string temporary_path = index_path + ".tmp";
    if (PathExists(temporary_path)) {
        throw std::runtime_error("temporary index path already exists: " + temporary_path);
    }
    {
        std::ofstream output(temporary_path, std::ios::binary | std::ios::trunc);
        if (not output) {
            throw std::runtime_error("cannot open index for writing: " + temporary_path);
        }
        auto serialized = index->Serialize(output);
        if (not serialized.has_value()) {
            throw std::runtime_error("failed to serialize index: " + serialized.error().message);
        }
    }
    std::error_code error;
    fs::rename(temporary_path, index_path, error);
    if (error) {
        throw std::runtime_error("cannot publish index " + index_path + ": " + error.message());
    }
    std::cout << "Saved index: " << index_path << '\n';
}

vsag::IndexPtr
PrepareIndex(const std::string& base_index_path,
             const std::string& trained_index_path,
             const std::string& build_parameters,
             const std::string& train_path,
             const MatrixHeader& train_header) {
    if (PathExists(trained_index_path)) {
        auto index = LoadIndex(trained_index_path, build_parameters);
        if (index->GetNumElements() != static_cast<int64_t>(train_header.rows)) {
            throw std::runtime_error("trained index vector count does not match train.fbin");
        }
        return index;
    }

    vsag::IndexPtr index;
    if (PathExists(base_index_path)) {
        index = LoadIndex(base_index_path, build_parameters);
    } else {
        index = CreateIndex(build_parameters);
        auto train = ReadMatrixPayload<float>(train_path, train_header);
        std::vector<int64_t> ids(train_header.rows);
        std::iota(ids.begin(), ids.end(), 0);
        auto base = vsag::Dataset::Make();
        base->NumElements(static_cast<int64_t>(train_header.rows))
            ->Dim(static_cast<int64_t>(train_header.cols))
            ->Ids(ids.data())
            ->Float32Vectors(train.data())
            ->Owner(false);

        const auto start = std::chrono::steady_clock::now();
        auto built = index->Build(base);
        if (not built.has_value()) {
            throw std::runtime_error("failed to build HGraph: " + built.error().message);
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "Built base index in " << std::fixed << std::setprecision(2) << seconds
                  << " s\n";
        SaveIndex(index, base_index_path);
    }

    if (index->GetNumElements() != static_cast<int64_t>(train_header.rows)) {
        throw std::runtime_error("index vector count does not match train.fbin");
    }

    const auto start = std::chrono::steady_clock::now();
    auto enabled = index->EnableAdaptiveEf(AdaptiveTrainingParameters());
    if (not enabled.has_value()) {
        throw std::runtime_error("failed to train adaptive_ef: " + enabled.error().message);
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Adaptive-ef training finished in " << std::fixed << std::setprecision(2)
              << seconds << " s; calibrated=" << (enabled.value() ? "true" : "false") << '\n';
    SaveIndex(index, trained_index_path);
    return index;
}

uint64_t
ReadDistanceComparisons(const vsag::DatasetPtr& result, uint64_t query_index) {
    const auto values = result->GetStatistics({"dist_cmp"});
    if (values.size() != 1 or values[0].empty()) {
        throw std::runtime_error("search result has no dist_cmp statistic for query " +
                                 std::to_string(query_index));
    }
    return ParseUnsigned(values[0], "dist_cmp");
}

SearchSummary
Evaluate(const vsag::IndexPtr& index,
         const std::vector<float>& queries,
         const MatrixHeader& query_header,
         const std::vector<int32_t>& ground_truth,
         const MatrixHeader& gt_header,
         uint64_t query_count,
         uint64_t topk,
         const std::string& search_parameters) {
    SearchSummary summary;
    double search_seconds = 0.0;
    std::array<uint64_t, kRecallTargets.size()> pass_counts{};
    uint64_t total_hits = 0;
    uint64_t total_dist_cmp = 0;

    for (uint64_t query_index = 0; query_index < query_count; ++query_index) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(static_cast<int64_t>(query_header.cols))
            ->Float32Vectors(queries.data() + query_index * query_header.cols)
            ->Owner(false);

        const auto start = std::chrono::steady_clock::now();
        auto searched = index->KnnSearch(query, static_cast<int64_t>(topk), search_parameters);
        search_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (not searched.has_value()) {
            throw std::runtime_error("search failed for query " + std::to_string(query_index) +
                                     ": " + searched.error().message);
        }
        auto result = std::move(searched).value();
        total_dist_cmp += ReadDistanceComparisons(result, query_index);

        if (result->GetDim() < 0) {
            throw std::runtime_error("search returned a negative result count");
        }
        std::unordered_set<int64_t> returned_ids;
        returned_ids.reserve(static_cast<uint64_t>(result->GetDim()));
        for (int64_t rank = 0; rank < result->GetDim(); ++rank) {
            returned_ids.insert(result->GetIds()[rank]);
        }

        std::unordered_set<int64_t> expected_ids;
        expected_ids.reserve(topk);
        const uint64_t gt_offset = query_index * gt_header.cols;
        for (uint64_t rank = 0; rank < topk; ++rank) {
            expected_ids.insert(ground_truth[gt_offset + rank]);
        }
        uint64_t hits = 0;
        for (const int64_t id : returned_ids) {
            hits += expected_ids.count(id) > 0 ? 1 : 0;
        }
        total_hits += hits;
        const double recall = static_cast<double>(hits) / static_cast<double>(topk);
        for (uint64_t target_index = 0; target_index < kRecallTargets.size(); ++target_index) {
            pass_counts[target_index] +=
                recall + std::numeric_limits<double>::epsilon() >= kRecallTargets[target_index] ? 1
                                                                                                : 0;
        }
    }

    if (search_seconds <= 0.0) {
        throw std::runtime_error("search timer did not advance");
    }
    summary.qps = static_cast<double>(query_count) / search_seconds;
    summary.avg_recall = static_cast<double>(total_hits) / static_cast<double>(query_count * topk);
    summary.mean_dist_cmp = static_cast<double>(total_dist_cmp) / static_cast<double>(query_count);
    for (uint64_t target_index = 0; target_index < kRecallTargets.size(); ++target_index) {
        summary.pass_fractions[target_index] =
            static_cast<double>(pass_counts[target_index]) / static_cast<double>(query_count);
    }
    return summary;
}

void
PrintHeader() {
    std::cout << '\n'
              << std::left << std::setw(14) << "policy" << std::right << std::setw(7) << "topk"
              << std::setw(10) << "target" << std::setw(8) << "alpha" << std::setw(9) << "ef"
              << std::setw(12) << "qps" << std::setw(11) << "pass" << std::setw(13) << "avg_recall"
              << std::setw(15) << "mean_dist_cmp" << '\n';
}

void
WriteRow(std::ofstream& csv,
         const std::string& policy,
         uint64_t topk,
         double target,
         const std::string& alpha,
         uint64_t ef_search,
         uint64_t query_count,
         const SearchSummary& summary,
         uint64_t target_index) {
    csv << policy << ',' << topk << ',' << std::fixed << std::setprecision(2) << target << ','
        << alpha << ',' << ef_search << ',' << query_count << ',' << std::setprecision(6)
        << summary.qps << ',' << summary.pass_fractions[target_index] << ',' << summary.avg_recall
        << ',' << summary.mean_dist_cmp << ",ok,\n";
    csv.flush();

    const std::string display_policy =
        policy == "fixed" ? policy + "_" + std::to_string(ef_search) : policy;
    std::cout << std::left << std::setw(14) << display_policy << std::right << std::setw(7) << topk
              << std::setw(10) << std::fixed << std::setprecision(2) << target << std::setw(8)
              << alpha << std::setw(9) << ef_search << std::setw(12) << std::setprecision(1)
              << summary.qps << std::setw(11) << std::setprecision(4)
              << summary.pass_fractions[target_index] << std::setw(13) << summary.avg_recall
              << std::setw(15) << std::setprecision(1) << summary.mean_dist_cmp << '\n';
}

std::string
CsvField(const std::string& value) {
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += '"';
        }
        escaped += ch;
    }
    return escaped + '"';
}

void
WriteRejectedRow(std::ofstream& csv,
                 uint64_t topk,
                 double target,
                 uint64_t query_count,
                 const std::string& error) {
    csv << "adaptive," << topk << ',' << std::fixed << std::setprecision(2) << target << ','
        << kAlpha << ',' << kInitialEf << ',' << query_count << ",,,,,rejected," << CsvField(error)
        << '\n';
    csv.flush();
    std::cout << std::left << std::setw(14) << "adaptive" << std::right << std::setw(7) << topk
              << std::setw(10) << std::fixed << std::setprecision(2) << target << std::setw(8)
              << kAlpha << std::setw(9) << kInitialEf << "  REJECTED: " << error << '\n';
}

void
ValidateOutputPaths(const std::string& train_path,
                    const std::string& query_path,
                    const std::string& gt_path,
                    const std::string& base_index_path,
                    const std::string& trained_index_path,
                    const std::string& result_path) {
    const std::string normalized_base_index = NormalizedAbsolutePath(base_index_path);
    const std::string normalized_trained_index = NormalizedAbsolutePath(trained_index_path);
    const std::string normalized_result = NormalizedAbsolutePath(result_path);
    if (normalized_base_index == normalized_trained_index or
        normalized_base_index == normalized_result or
        normalized_trained_index == normalized_result) {
        throw std::runtime_error("base index, trained index, and result paths must be different");
    }
    for (const auto& input_path : {train_path, query_path, gt_path}) {
        const std::string normalized_input = NormalizedAbsolutePath(input_path);
        if (normalized_base_index == normalized_input or
            normalized_trained_index == normalized_input or normalized_result == normalized_input) {
            throw std::runtime_error("output paths must not overwrite an input matrix");
        }
    }
}

int
Run(int argc, char** argv) {
    if (argc < 5 or argc > 8) {
        std::cerr << "usage: " << argv[0]
                  << " <data_dir> <base_index_path> <trained_index_path> <result.csv>"
                     " [query_count [only_topk [only_fixed_ef]]]\n";
        return 2;
    }

    const fs::path data_dir(argv[1]);
    const std::string train_path = (data_dir / "train.fbin").string();
    const std::string query_path = (data_dir / "test.fbin").string();
    const std::string gt_path = (data_dir / "gt.ibin").string();
    const std::string base_index_path = argv[2];
    const std::string trained_index_path = argv[3];
    const std::string result_path = argv[4];
    ValidateOutputPaths(
        train_path, query_path, gt_path, base_index_path, trained_index_path, result_path);

    const MatrixHeader train_header = InspectMatrix<float>(train_path);
    const MatrixHeader query_header = InspectMatrix<float>(query_path);
    const MatrixHeader gt_header = InspectMatrix<int32_t>(gt_path);
    if (train_header.cols != query_header.cols) {
        throw std::runtime_error("train.fbin and test.fbin dimensions differ");
    }
    if (query_header.rows != gt_header.rows) {
        throw std::runtime_error("test.fbin and gt.ibin row counts differ");
    }
    if (gt_header.cols < kTopKs.back()) {
        throw std::runtime_error("gt.ibin width must be at least 100");
    }
    if (train_header.rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) or
        train_header.cols > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("matrix dimensions exceed the VSAG API limits");
    }

    uint64_t query_count = query_header.rows;
    if (argc == 6) {
        query_count = ParseUnsigned(argv[5], "query_count");
        if (query_count == 0 or query_count > query_header.rows) {
            throw std::runtime_error("query_count must be in [1, test.fbin rows]");
        }
    }
    uint64_t only_topk = 0;
    uint64_t only_fixed_ef = 0;
    const bool adaptive_only = argc == 7;
    if (argc >= 7) {
        query_count = ParseUnsigned(argv[5], "query_count");
        if (query_count == 0 or query_count > query_header.rows) {
            throw std::runtime_error("query_count must be in [1, test.fbin rows]");
        }
        only_topk = ParseUnsigned(argv[6], "only_topk");
        if (only_topk != 0 and std::find(kTopKs.begin(), kTopKs.end(), only_topk) == kTopKs.end()) {
            throw std::runtime_error("only_topk must be 0, 10, 50, or 100");
        }
    }
    if (argc == 8) {
        only_fixed_ef = ParseUnsigned(argv[7], "only_fixed_ef");
        if (only_fixed_ef == 0) {
            throw std::runtime_error("only_fixed_ef must be positive");
        }
    }

    std::cout << "Dataset: metric=cosine, train=" << train_header.rows << "x" << train_header.cols
              << ", test=" << query_header.rows << "x" << query_header.cols
              << ", gt=" << gt_header.rows << "x" << gt_header.cols
              << ", evaluating=" << query_count << '\n';

    vsag::init();
    auto index = PrepareIndex(base_index_path,
                              trained_index_path,
                              BuildParameters(train_header.cols),
                              train_path,
                              train_header);
    auto queries = ReadMatrixPayload<float>(query_path, query_header);
    auto ground_truth = ReadMatrixPayload<int32_t>(gt_path, gt_header);

    std::ofstream csv(result_path, std::ios::trunc);
    if (not csv) {
        throw std::runtime_error("cannot open CSV for writing: " + result_path);
    }
    csv << "policy,topk,target_recall,alpha,ef_search,queries,qps,pass_fraction,"
           "avg_recall,mean_dist_cmp,status,error\n";
    PrintHeader();

    for (const uint64_t topk : kTopKs) {
        if (only_topk != 0 and topk != only_topk) {
            continue;
        }
        if (only_fixed_ef == 0) {
            for (uint64_t target_index = 0; target_index < kRecallTargets.size(); ++target_index) {
                const double target = kRecallTargets[target_index];
                try {
                    const auto summary = Evaluate(index,
                                                  queries,
                                                  query_header,
                                                  ground_truth,
                                                  gt_header,
                                                  query_count,
                                                  topk,
                                                  AdaptiveSearchParameters(target));
                    WriteRow(csv,
                             "adaptive",
                             topk,
                             target,
                             "0.05",
                             kInitialEf,
                             query_count,
                             summary,
                             target_index);
                } catch (const std::runtime_error& error) {
                    const std::string message = error.what();
                    if (message.find("adaptive_ef") == std::string::npos) {
                        throw;
                    }
                    WriteRejectedRow(csv, topk, target, query_count, message);
                }
            }
        }
        if (adaptive_only) {
            continue;
        }

        std::vector<uint64_t> fixed_efs(kFixedEfFrontier.begin(), kFixedEfFrontier.end());
        if (only_fixed_ef != 0) {
            fixed_efs = {only_fixed_ef};
        }
        for (const uint64_t ef_search : fixed_efs) {
            const auto summary = Evaluate(index,
                                          queries,
                                          query_header,
                                          ground_truth,
                                          gt_header,
                                          query_count,
                                          topk,
                                          FixedSearchParameters(ef_search));
            for (uint64_t target_index = 0; target_index < kRecallTargets.size(); ++target_index) {
                WriteRow(csv,
                         "fixed",
                         topk,
                         kRecallTargets[target_index],
                         "",
                         ef_search,
                         query_count,
                         summary,
                         target_index);
            }
        }
    }

    std::cout << "\nWrote CSV: " << result_path << '\n';
    return 0;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        return Run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
