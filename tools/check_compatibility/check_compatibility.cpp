
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

#include <vsag/vsag.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string>
split_string(const std::string& input, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string item;

    while (getline(ss, item, delimiter)) {
        result.push_back(item);
    }

    return result;
}

std::string
asset_path(const std::string& compatibility_index_dir, const std::string& filename) {
    if (compatibility_index_dir.back() == '/') {
        return compatibility_index_dir + filename;
    }
    return compatibility_index_dir + "/" + filename;
}

bool
read_json(const std::string& json_path, std::string& json) {
    std::ifstream infile(json_path);
    if (not infile.is_open()) {
        std::cerr << "failed to open json file: " << json_path << std::endl;
        return false;
    }
    std::stringstream buffer;
    buffer << infile.rdbuf();
    infile.close();

    json = buffer.str();
    if (json.empty()) {
        std::cerr << "json file is empty: " << json_path << std::endl;
        return false;
    }
    return true;
}

int
main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "input error" << std::endl;
        return -1;
    }

    std::string input_str = argv[1];
    auto strs = split_string(input_str, '_');
    if (strs.size() < 2) {
        std::cerr << "input error" << std::endl;
        return -1;
    }
    auto algo_name = strs[1];

    std::string compatibility_index_dir = "/tmp";
    const auto* compatibility_index_dir_env = std::getenv("COMPATIBILITY_INDEX_DIR");
    if (compatibility_index_dir_env != nullptr and compatibility_index_dir_env[0] != '\0') {
        compatibility_index_dir = compatibility_index_dir_env;
    }

    std::string index_path = asset_path(compatibility_index_dir, input_str + ".index");
    std::string search_json_path =
        asset_path(compatibility_index_dir, input_str + "_search.json");
    std::string build_json_path = asset_path(compatibility_index_dir, input_str + "_build.json");

    std::string build_json;
    std::string search_json;

    auto log_error = [&]() { std::cerr << input_str << " failed " << std::endl; };

    if (not read_json(build_json_path, build_json) or
        not read_json(search_json_path, search_json)) {
        log_error();
        return -1;
    }

    auto index = vsag::Factory::CreateIndex(algo_name, build_json);
    if (not index.has_value()) {
        log_error();
        return -1;
    }
    auto algo = index.value();
    std::ifstream index_file(index_path, std::ios::binary);
    if (not index_file.is_open()) {
        std::cerr << "failed to open index file: " << index_path << std::endl;
        log_error();
        return -1;
    }
    auto load_index = algo->Deserialize(index_file);
    if (not load_index.has_value()) {
        log_error();
        return -1;
    }
    int64_t dim = 512;
    auto count = 500;
    std::string origin_data_path = asset_path(compatibility_index_dir, "random_512d_10K.bin");
    std::ifstream ifs(origin_data_path, std::ios::binary);
    if (not ifs.is_open()) {
        std::cerr << "failed to open dataset file: " << origin_data_path << std::endl;
        log_error();
        return -1;
    }
    std::vector<float> data(count * dim);
    ifs.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    if (ifs.gcount() != static_cast<std::streamsize>(data.size() * sizeof(float))) {
        std::cerr << "dataset file is incomplete: " << origin_data_path << std::endl;
        log_error();
        return -1;
    }

    for (int i = 0; i < count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(data.data() + i * dim)->Owner(false);
        auto knn_result = algo->KnnSearch(query, 1, search_json);
        if (not knn_result.has_value()) {
            log_error();
            return -1;
        }
        if (knn_result.value()->GetIds()[0] != i) {
            log_error();
            return -1;
        }
    }
    std::cout << input_str << " success " << std::endl;
    return 0;
}
