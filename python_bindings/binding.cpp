// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You mayM_PI may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

#include "fmt/format.h"
#include "iostream"
#include "vsag/binaryset.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/factory.h"
#include "vsag/index.h"
#include "vsag/logger.h"
#include "vsag/options.h"
#include "vsag/readerset.h"
#include "vsag/vsag.h"

namespace py = pybind11;

namespace {

// FIX 1: DECLARE exception variables here, but DO NOT initialize them.
// They will be initialized inside the PYBIND11_MODULE function.
py::exception<std::runtime_error> PyVsagError;
py::exception<std::runtime_error> PyUnknownError;
py::exception<std::runtime_error> PyInternalError;
py::exception<std::runtime_error> PyInvalidParameterError;
py::exception<std::runtime_error> PyWrongStatusError;
py::exception<std::runtime_error> PyBuildTwiceError;
py::exception<std::runtime_error> PyUnsupportedIndexError;
py::exception<std::runtime_error> PyUnsupportedOperationError;
py::exception<std::runtime_error> PyDimensionNotEqualError;
py::exception<std::runtime_error> PyIndexEmptyError;
py::exception<std::runtime_error> PyReadError;
py::exception<std::runtime_error> PyInvalidBinaryError;

// This function now correctly throws the declared exception variables.
[[noreturn]] void
HandleError(const vsag::Error& err) {
    std::string msg = fmt::format("vsag error: {}", err.message);
    switch (err.type) {
        case vsag::ErrorType::INVALID_ARGUMENT:
            throw PyInvalidParameterError(msg);
        case vsag::ErrorType::UNSUPPORTED_INDEX:
            throw PyUnsupportedIndexError(msg);
        case vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION:
            throw PyUnsupportedOperationError(msg);
        case vsag::ErrorType::DIMENSION_NOT_EQUAL:
            throw PyDimensionNotEqualError(msg);
        case vsag::ErrorType::INDEX_EMPTY:
            throw PyIndexEmptyError(msg);
        case vsag::ErrorType::BUILD_TWICE:
            throw PyBuildTwiceError(msg);
        case vsag::ErrorType::INTERNAL_ERROR:
            throw PyInternalError(msg);
        case vsag::ErrorType::READ_ERROR:
            throw PyReadError(msg);
        case vsag::ErrorType::INVALID_BINARY:
            throw PyInvalidBinaryError(msg);
        case vsag::ErrorType::WRONG_STATUS:
            throw PyWrongStatusError(msg);
        case vsag::ErrorType::UNKNOWN_ERROR:
        default:
            throw PyUnknownError(msg);
    }
}

}  // namespace

void
SetLoggerOff() {
    // FIX 2: Use .logger() instead of ->logger()
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kOFF);
}

void
SetLoggerInfo() {
    // FIX 2: Use .logger() instead of ->logger()
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kINFO);
}

void
SetLoggerDebug() {
    // FIX 2: Use .logger() instead of ->logger()
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);
}

struct SparseVectors {
    std::vector<vsag::SparseVector> sparse_vectors;
    uint32_t num_elements;
    uint32_t num_non_zeros;

    SparseVectors(uint32_t num_elements)
        : sparse_vectors(num_elements), num_elements(num_elements), num_non_zeros(0) {
    }
};

SparseVectors
BuildSparseVectorsFromCSR(py::array_t<uint32_t> index_pointers,
                          py::array_t<uint32_t> indices,
                          py::array_t<float> values) {
    auto buf_ptr = index_pointers.request();
    auto buf_idx = indices.request();
    auto buf_val = values.request();

    if (buf_ptr.ndim != 1 || buf_idx.ndim != 1 || buf_val.ndim != 1) {
        throw std::invalid_argument("all inputs must be 1-dimensional");
    }

    if (buf_ptr.shape[0] < 2) {
        throw std::invalid_argument("index_pointers length must be at least 2");
    }
    uint32_t num_elements = buf_ptr.shape[0] - 1;

    const uint32_t* ptr_data = index_pointers.data();
    const uint32_t* idx_data = indices.data();
    const float* val_data = values.data();

    uint32_t num_non_zeros = ptr_data[num_elements];

    if (static_cast<size_t>(num_non_zeros) != buf_idx.shape[0]) {
        throw std::invalid_argument(
            fmt::format("Size of 'indices'({}) must equal index_pointers[last]",
                        buf_idx.shape[0],
                        num_non_zeros));
    }
    if (static_cast<size_t>(num_non_zeros) != buf_val.shape[0]) {
        throw std::invalid_argument(
            fmt::format("Size of 'values'({}) must equal index_pointers[last]({})",
                        buf_val.shape[0],
                        num_non_zeros));
    }

    if (ptr_data[0] != 0) {
        throw std::invalid_argument("index_pointers[0] must be 0");
    }
    for (uint32_t i = 1; i <= num_elements; ++i) {
        if (ptr_data[i] < ptr_data[i - 1]) {
            throw std::invalid_argument(
                fmt::format("index_pointers[{}]({}) > index_pointers[{}]({})",
                            i - 1,
                            ptr_data[i - 1],
                            i,
                            ptr_data[i]));
        }
    }

    SparseVectors svs(num_elements);
    svs.num_non_zeros = num_non_zeros;

    for (uint32_t i = 0; i < num_elements; ++i) {
        uint32_t start = ptr_data[i];
        uint32_t end = ptr_data[i + 1];
        uint32_t len = end - start;

        svs.sparse_vectors[i].len_ = len;
        svs.sparse_vectors[i].ids_ = const_cast<uint32_t*>(idx_data + start);
        svs.sparse_vectors[i].vals_ = const_cast<float*>(val_data + start);
    }

    return svs;
}

class Index {
public:
    Index(std::string name, const std::string& parameters) {
        if (auto index = vsag::Factory::CreateIndex(name, parameters)) {
            index_ = index.value();
        } else {
            HandleError(index.error());
        }
    }

public:
    void
    Build(py::array_t<float> vectors, py::array_t<int64_t> ids) {
        auto vec_buf = vectors.request();
        auto id_buf = ids.request();

        if (vec_buf.ndim != 2) {
            throw std::invalid_argument("vectors must be a 2-dimensional array (n, dim)");
        }
        if (id_buf.ndim != 1) {
            throw std::invalid_argument("ids must be a 1-dimensional array");
        }
        if (vec_buf.shape[0] != id_buf.shape[0]) {
            throw std::invalid_argument("Number of vectors and ids must match");
        }

        size_t num_elements = vec_buf.shape[0];
        size_t dim = vec_buf.shape[1];

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(dim)
            ->NumElements(num_elements)
            ->Ids(ids.mutable_data())
            ->Float32Vectors(vectors.mutable_data());

        auto result = index_->Build(dataset);
        if (!result) {
            HandleError(result.error());
        }

        this->dim_ = dim;
    }

    void
    SparseBuild(py::array_t<uint32_t> index_pointers,
                py::array_t<uint32_t> indices,
                py::array_t<float> values,
                py::array_t<int64_t> ids) {
        auto batch = BuildSparseVectorsFromCSR(index_pointers, indices, values);

        auto buf_id = ids.request();
        if (buf_id.ndim != 1) {
            throw std::invalid_argument("ids must be 1-dimensional");
        }
        if (batch.num_elements != buf_id.shape[0]) {
            throw std::invalid_argument(
                fmt::format("Length of 'ids'({}) must match number of vectors({})",
                            buf_id.shape[0],
                            batch.num_elements));
        }

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->NumElements(batch.num_elements)
            ->Ids(ids.data())
            ->SparseVectors(batch.sparse_vectors.data());

        auto result = index_->Build(dataset);
        if (!result) {
            HandleError(result.error());
        }

        this->dim_ = -1;  // -1 indicates sparse index
    }

    void
    Add(py::array_t<float> vectors, py::array_t<int64_t> ids) {
        if (this->dim_ <= 0) {
            throw PyWrongStatusError("Cannot add to a sparse index or an uninitialized index");
        }

        auto vec_buf = vectors.request();
        auto id_buf = ids.request();

        if (vec_buf.ndim != 2) {
            throw std::invalid_argument("vectors must be a 2-dimensional array (n, dim)");
        }
        if (id_buf.ndim != 1) {
            throw std::invalid_argument("ids must be a 1-dimensional array");
        }
        if (vec_buf.shape[0] != id_buf.shape[0]) {
            throw std::invalid_argument("Number of vectors and ids must match");
        }

        size_t num_elements = vec_buf.shape[0];
        size_t dim = vec_buf.shape[1];

        if (dim != (size_t)this->dim_) {
            throw PyDimensionNotEqualError(
                fmt::format("Vector dimension mismatch: index has dim {}, adding vector with dim {}",
                            this->dim_,
                            dim));
        }

        auto dataset = vsag::Dataset::Make();
        dataset->Owner(false)
            ->Dim(dim)
            ->NumElements(num_elements)
            ->Ids(ids.mutable_data())
            ->Float32Vectors(vectors.mutable_data());

        auto result = index_->Add(dataset);
        if (!result) {
            HandleError(result.error());
        }
    }

    void
    Remove(int64_t id) {
        auto result = index_->Remove(id);
        if (!result) {
            HandleError(result.error());
        }
        if (!result.value()) {
            throw PyInvalidParameterError(fmt::format("Failed to remove id: {}. May not exist.", id));
        }
    }

    py::object
    KnnSearch(py::array_t<float> vector, size_t k, std::string& parameters) {
        auto vec_buf = vector.request();
        if (vec_buf.ndim != 1) {
            throw std::invalid_argument("vector must be a 1-dimensional array");
        }

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(vector.size())
            ->Float32Vectors(vector.mutable_data())
            ->Owner(false);

        auto result = index_->KnnSearch(query, k, parameters);
        if (!result) {
            HandleError(result.error());
        }

        auto res_dataset = result.value();
        auto res_k = res_dataset->GetDim();  // Result k is in Dim
        auto res_ids = res_dataset->GetIds();
        auto res_dists = res_dataset->GetDistances();

        py::array_t<int64_t> ids(res_k);
        py::array_t<float> dists(res_k);
        std::memcpy(ids.mutable_data(), res_ids, res_k * sizeof(int64_t));
        std::memcpy(dists.mutable_data(), res_dists, res_k * sizeof(float));

        return py::make_tuple(ids, dists);
    }

    py::tuple
    SparseKnnSearch(py::array_t<uint32_t> index_pointers,
                    py::array_t<uint32_t> indices,
                    py::array_t<float> values,
                    uint32_t k,
                    const std::string& parameters) {
        auto batch = BuildSparseVectorsFromCSR(index_pointers, indices, values);

        std::vector<uint32_t> shape{batch.num_elements, k};
        auto res_ids = py::array_t<int64_t>(shape);
        auto res_dists = py::array_t<float>(shape);

        auto ids_view = res_ids.mutable_unchecked<2>();
        auto dists_view = res_dists.mutable_unchecked<2>();

        for (uint32_t i = 0; i < batch.num_elements; ++i) {
            auto query = vsag::Dataset::Make();
            query->Owner(false)
                ->NumElements(1)
                ->SparseVectors(batch.sparse_vectors.data() + i);

            auto result = index_->KnnSearch(query, k, parameters);
            if (!result) {
                HandleError(result.error());
            }

            auto res_dataset = result.value();
            auto res_k = res_dataset->GetDim();
            auto res_ids_ptr = res_dataset->GetIds();
            auto res_dists_ptr = res_dataset->GetDistances();

            for (uint32_t j = 0; j < k; ++j) {
                if (j < (uint32_t)res_k) {
                    ids_view(i, j) = res_ids_ptr[j];
                    dists_view(i, j) = res_dists_ptr[j];
                } else {
                    ids_view(i, j) = -1;
                    dists_view(i, j) = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }

        return py::make_tuple(res_ids, res_dists);
    }

    py::object
    RangeSearch(py::array_t<float> point, float threshold, std::string& parameters) {
        auto vec_buf = point.request();
        if (vec_buf.ndim != 1) {
            throw std::invalid_argument("vector must be a 1-dimensional array");
        }

        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(point.size())
            ->Float32Vectors(point.mutable_data())
            ->Owner(false);

        auto result = index_->RangeSearch(query, threshold, parameters);
        if (!result) {
            HandleError(result.error());
        }

        auto res_dataset = result.value();
        auto res_count = res_dataset->GetDim();  // Result count is in Dim
        auto res_ids = res_dataset->GetIds();
        auto res_dists = res_dataset->GetDistances();

        py::array_t<int64_t> ids(res_count);
        py::array_t<float> dists(res_count);
        std::memcpy(ids.mutable_data(), res_ids, res_count * sizeof(int64_t));
        std::memcpy(dists.mutable_data(), res_dists, res_count * sizeof(float));

        return py::make_tuple(ids, dists);
    }

    // OVERLOAD 1: Save to single file (for HNSW, SINDI, etc.)
    void
    SaveFile(const std::string& filename) {
        try {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file for writing: " + filename);
            }
            auto result = index_->Serialize(file);
            if (!result) {
                HandleError(result.error());
            }
            file.close();
        } catch (const std::exception& e) {
            throw PyInternalError(fmt::format("Failed to save index to file: {}", e.what()));
        }
    }

    // OVERLOAD 2: Save to directory (for DiskANN)
    py::dict
    SaveDirectory(const std::string& directory) {
        auto bs_result = index_->Serialize();
        if (!bs_result) {
            HandleError(bs_result.error());
        }
        auto binary_set = bs_result.value();

        py::dict file_sizes;
        std::filesystem::create_directories(directory);

        try {
            auto keys = binary_set.GetKeys();
            for (auto key : keys) {
                vsag::Binary b = binary_set.Get(key);
                std::string file_path =
                    (std::filesystem::path(directory) / ("diskann.index." + key)).string();
                std::ofstream file(file_path, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Failed to open file for writing: " + file_path);
                }
                file.write((const char*)b.data.get(), b.size);
                file.close();
                file_sizes[py::str(key)] = b.size;
            }
        } catch (const std::exception& e) {
            throw PyInternalError(fmt::format("Failed to save index to directory: {}", e.what()));
        }
        return file_sizes;
    }

    // OVERLOAD 1: Load from single file (for HNSW, SINDI, etc.)
    void
    LoadFile(const std::string& filename) {
        try {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file for reading: " + filename);
            }
            auto result = index_->Deserialize(file);
            if (!result) {
                HandleError(result.error());
            }
            file.close();
        } catch (const std::exception& e) {
            throw PyInternalError(fmt::format("Failed to load index from file: {}", e.what()));
        }
        this->dim_ = -1;  // Dimension is unknown after loading
    }

    // OVERLOAD 2: Load from directory (for DiskANN)
    void
    LoadDirectory(const std::string& directory, py::dict file_sizes, bool in_memory) {
        // 'in_memory' is unused in the C++ layer but kept for API compatibility
        (void)in_memory;

        vsag::ReaderSet rs;
        try {
            for (auto item : file_sizes) {
                std::string key = item.first.cast<std::string>();
                size_t size = item.second.cast<size_t>();
                std::string file_path =
                    (std::filesystem::path(directory) / ("diskann.index." + key)).string();

                auto reader = vsag::Factory::CreateLocalFileReader(file_path, 0, size);
                rs.Set(key, reader);
            }
        } catch (const std::exception& e) {
            throw PyInternalError(fmt::format("Failed to create readers: {}", e.what()));
        }

        auto result = index_->Deserialize(rs);
        if (!result) {
            HandleError(result.error());
        }
        this->dim_ = -1;  // Dimension is unknown after loading
    }

    int64_t
    GetDim() const {
        if (this->dim_ == -1) {
            py::print(
                "Warning: Index dimension is unknown (e.g., loaded from file or sparse). "
                "Returning -1.");
        }
        return this->dim_;
    }

    int64_t
    GetNumElements() const {
        return index_->GetNumElements();
    }

    int64_t
    GetMemoryUsage() const {
        return index_->GetMemoryUsage();
    }

private:
    std::shared_ptr<vsag::Index> index_;
    int64_t dim_ = 0;  // Cache dimension
};

PYBIND11_MODULE(_pyvsag, m) {
    m.doc() = "Python bindings for the vsag library";

    m.def("set_logger_off", &SetLoggerOff, "Set vsag logger level to OFF");
    m.def("set_logger_info", &SetLoggerInfo, "Set vsag logger level to INFO");
    m.def("set_logger_debug", &SetLoggerDebug, "Set vsag logger level to DEBUG");

    m.def("version", &vsag::version, "Get the vsag library version string");
    m.def("init", &vsag::init, "Initialize the vsag library");

    // FIX 1: Initialize the exception variables in the module scope.
    // Initialize the base exception
    PyVsagError = py::exception<std::runtime_error>(m, "VsagError");

    // Initialize derived exceptions, setting their base class in Python
    PyUnknownError = py::exception<std::runtime_error>(m, "UnknownError", PyVsagError.ptr());
    PyInternalError = py::exception<std::runtime_error>(m, "InternalError", PyVsagError.ptr());
    PyInvalidParameterError =
        py::exception<std::runtime_error>(m, "InvalidParameterError", PyVsagError.ptr());
    PyWrongStatusError =
        py::exception<std::runtime_error>(m, "WrongStatusError", PyVsagError.ptr());
    PyBuildTwiceError = py::exception<std::runtime_error>(m, "BuildTwiceError", PyVsagError.ptr());
    PyUnsupportedIndexError =
        py::exception<std::runtime_error>(m, "UnsupportedIndexError", PyVsagError.ptr());
    PyUnsupportedOperationError =
        py::exception<std::runtime_error>(m, "UnsupportedOperationError", PyVsagError.ptr());
    PyDimensionNotEqualError =
        py::exception<std::runtime_error>(m, "DimensionNotEqualError", PyVsagError.ptr());
    PyIndexEmptyError = py::exception<std::runtime_error>(m, "IndexEmptyError", PyVsagError.ptr());
    PyReadError = py::exception<std::runtime_error>(m, "ReadError", PyVsagError.ptr());
    PyInvalidBinaryError =
        py::exception<std::runtime_error>(m, "InvalidBinaryError", PyVsagError.ptr());

    // Remove the broken exception translator
    // py::register_exception_translator(...); // REMOVED

    py::class_<Index>(m, "Index")
        .def(py::init<std::string, std::string&>(), py::arg("name"), py::arg("parameters"))

        .def("build",
             static_cast<void (Index::*)(py::array_t<float>, py::array_t<int64_t>)>(
                 &Index::Build),
             py::arg("vectors"),
             py::arg("ids"),
             "Build the index with dense vectors.")
        .def("build",
             static_cast<void (Index::*)(py::array_t<uint32_t>,
                                        py::array_t<uint32_t>,
                                        py::array_t<float>,
                                        py::array_t<int64_t>)>(&Index::SparseBuild),
             py::arg("index_pointers"),
             py::arg("indices"),
             py::arg("values"),
             py::arg("ids"),
             "Build the index with sparse vectors in CSR format.")

        .def("add",
             &Index::Add,
             py::arg("vectors"),
             py::arg("ids"),
             "Add new dense vectors to the index.")

        .def("remove", &Index::Remove, py::arg("id"), "Remove a vector from the index by its ID.")

        .def("knn_search",
             static_cast<py::object (Index::*)(
                 py::array_t<float>, size_t, std::string&)>(&Index::KnnSearch),
             py::arg("vector"),
             py::arg("k"),
             py::arg("parameters"),
             "Perform KNN search with a single dense vector.")
        .def("knn_search",
             &Index::SparseKnnSearch,
             py::arg("index_pointers"),
             py::arg("indices"),
             py::arg("values"),
             py::arg("k"),
             py::arg("parameters"),
             "Perform KNN search with sparse vectors in CSR format (supports batch).")

        .def("range_search",
             &Index::RangeSearch,
             py::arg("vector"),
             py::arg("threshold"),
             py::arg("parameters"),
             "Perform range search with a single dense vector.")

        // Overloaded save/load methods
        .def("save",
             static_cast<void (Index::*)(const std::string&)>(&Index::SaveFile),
             py::arg("filename"),
             "Save the index to a single file (for HNSW, SINDI, etc.).")
        // FIX 3: Corrected typo from static_CALCULATOR_static_cast
        .def("save",
             static_cast<py::dict (Index::*)(const std::string&)>(&Index::SaveDirectory),
             py::arg("directory"),
             "Save the index to a directory (for DiskANN, etc.) and return file sizes.")

        .def("load",
             static_cast<void (Index::*)(const std::string&)>(&Index::LoadFile),
             py::arg("filename"),
             "Load the index from a single file (for HNSW, SINDI, etc.).")
        .def("load",
             static_cast<void (Index::*)(const std::string&, py::dict, bool)>(
                 &Index::LoadDirectory),
             py::arg("directory"),
             py::arg("file_sizes"),
             py::arg("in_memory"),
             "Load the index from a directory (for DiskANN, etc.).")

        // Read-only properties
        .def_property_readonly(
            "dim",
            &Index::GetDim,
            "Get the dimension of the vectors in the index (-1 if sparse or loaded).")
        .def_property_readonly("num_elements",
                               &Index::GetNumElements,
                               "Get the total number of vectors in the index.")
        .def_property_readonly("memory_usage",
                               &Index::GetMemoryUsage,
                               "Get the estimated memory usage of the index in bytes.");
}
