# AutoTune (Offline Parameter Tuning)

AutoTune automates the offline process of expanding index parameter candidates, building indexes,
running the VSAG evaluation engine, applying constraints, and selecting a reproducible complete
configuration. V1 is a full-grid, full-query baseline: it reduces manual scripting, but it does not
promise to be faster than a hand-written grid search.

AutoTune V1 supports [HGraph](../indexes/hgraph.md) and [IVF](../indexes/ivf.md), dense `float32`
HDF5 datasets, and unfiltered KNN workloads. It can jointly evaluate build, quantization, and
search parameters, or tune only search parameters for an existing index.

## Build the CLI

Tools are disabled by default. From the VSAG repository root, enable them for a release build:

```bash
VSAG_ENABLE_TOOLS=ON make release
```

The resulting executable is:

```text
build-release/tools/autotune/autotune
```

To configure and build only the AutoTune target with CMake:

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TOOLS=ON \
  -DENABLE_CXX11_ABI=ON
cmake --build build-release --target autotune --parallel
```

`ENABLE_CXX11_ABI` is on by default, but AutoTune is not added to the build when it is disabled.
See [Building VSAG](../development/building.md) for platform prerequisites. Prefer the release
binary when stdout will be consumed as JSON.

V1 does not install the optional C++ library or the CLI executable. Use both from the build tree.

## Prepare the dataset

`data_path` must point to an HDF5 file containing base vectors, queries, and ground truth. AutoTune
infers `dim`, `dtype`, and `metric_type` from the file, so they do not need to be repeated in the
request. See [HDF5 Dataset Format](dataset_format.md) for the required layout.

V1 accepts only dense `float32` data. Every trial evaluates the complete query set in the file.

## Create a request

The following request builds one HGraph index and evaluates three `ef_search` values. Replace the
dataset path before running it.

```json
{
  "version": 1,
  "data_path": "/data/sift-128-euclidean.hdf5",
  "indexes": [
    {
      "name": "hgraph",
      "create_params": {
        "index_param": {
          "base_quantization_type": "fp32",
          "max_degree": 32,
          "ef_construction": 200
        }
      },
      "search_params": {
        "hgraph": {
          "ef_search": [40, 80, 120]
        }
      }
    }
  ],
  "workload": {
    "top_k": 10,
    "concurrency": 1
  },
  "constraints": {
    "recall_at_k": 0.90,
    "index_size_mb": 2048.0
  },
  "objective": {
    "metric": "latency_avg_ms"
  },
  "tuning_config": {
    "workspace_path": "/tmp/vsag_autotune",
    "keep_intermediate": true,
    "max_trials": 16
  },
  "output": {
    "result_path": "/tmp/vsag_autotune/report.json",
    "include_raw_eval": false
  }
}
```

Save it as `request.json`. In this example, the build parameters are fixed and `ef_search` is a
candidate axis, so AutoTune builds once and runs three search trials. `keep_intermediate: true`
keeps generated index files after the request; set it to `false` when only the recommended
parameters are needed.

Each parameter leaf can be expressed as:

| Form | Meaning |
| --- | --- |
| `"max_degree": 32` | One fixed value. |
| `"max_degree": [16, 32]` | Two candidate values. |
| `"ef_search": {"$range":{"start":40,"stop":120,"step":40}}` | Inclusive range. |

Candidate axes form a Cartesian product. When supported fields are omitted, the HGraph or IVF
built-in proposer can add candidates for those missing fields. Explicit user values always take
precedence. AutoTune does not choose an index when `indexes` is omitted; `indexes` is required.

The HGraph proposer covers `base_quantization_type`, `max_degree`, `ef_construction`, and
`ef_search`. The IVF proposer covers `base_quantization_type`, `buckets_count`, and
`scan_buckets_count`. Other omitted fields remain absent and use the concrete index's native
defaults.

## Run the CLI

The CLI accepts exactly one request file:

```bash
set -o pipefail
./build-release/tools/autotune/autotune request.json \
  | tee /tmp/autotune-summary.json
```

The command writes a small summary to stdout. The full report is written to `output.result_path`
when that field is configured.

## Read the summary

With the release CLI and the default log settings, stdout contains only fields that apply to the
current result.

The display order is status-aware and puts the actionable result first. A successful run starts
with `recommendation`; an unsatisfied run starts with `best_effort`; a failed run starts with
`failure`. Only that applicable result branch is emitted; unrelated branches are not printed as
`null`. `report_path` is emitted only when the full report was written successfully. Inside a
recommendation, `index_name`, `create_params`, and `search_params` appear
before the workload, metrics, and evidence. The full report is unchanged. JSON consumers must
still treat object-key order as non-semantic.

| Field | How to use it |
| --- | --- |
| `recommendation` | Emitted on success; selected feasible complete create/search configuration. |
| `best_effort` | Emitted when no candidate satisfies all constraints; explanation only. |
| `failure` | Emitted on failure; structured `stage`, `code`, and `message`. |
| `status` | `success`, `no_candidate_satisfied`, or `failed`. |
| `elapsed_seconds` | End-to-end tuning time before report-file writing and cleanup. |
| `report_path` | Emitted only when the full report was written successfully. |
| `version` | Output contract version; V1 is `1`. |

Interpret `status` as follows:

| Status | Meaning | CLI exit code |
| --- | --- | --- |
| `success` | `recommendation` satisfies all constraints. | `0` |
| `no_candidate_satisfied` | No complete candidate met every constraint. | `0` |
| `failed` | Validation, planning, evaluation, selection, or report writing failed. | non-zero |

Do not use the exit code alone to decide whether a recommendation exists.
`no_candidate_satisfied` is a completed tuning run, but its `best_effort` result is not guaranteed
to satisfy the request and must not be presented as a feasible recommendation.

For a successful result, read:

- `recommendation.index_name` and `recommendation.create_params` for index creation;
- `recommendation.search_params` for search;
- `recommendation.workload` for the benchmark `top_k` and `concurrency`;
- `recommendation.metrics` for all available build and search metrics;
- `recommendation.evidence` to locate the source build and trial in the full report;
- `recommendation.artifacts.expected_to_exist_after_response` before using an artifact path.

With `jq`, a compact view is:

```bash
jq '{status, elapsed_seconds, recommendation, best_effort, failure}' \
  /tmp/autotune-summary.json
```

## Read the full report

Use the full report to audit candidate generation and understand failures:

```bash
jq '{status, build_count, build_group_count, trial_count, environment}' \
  /tmp/vsag_autotune/report.json

jq '.builds[] | {
  build_id, status, index_name, create_params, metrics, constraint_evaluation, failure
}' /tmp/vsag_autotune/report.json

jq '.trials[] | {
  trial_id, build_id, status, search_params, metrics,
  constraint_evaluation, execution, failure
}' /tmp/vsag_autotune/report.json
```

The most useful full-report sections are:

| Section | Contents |
| --- | --- |
| `input_request` | The original parsed request JSON value. |
| `effective_request` | Normalized request plus dataset metadata inferred by AutoTune. |
| `environment` | VSAG, OS, CPU, memory, and SIMD evidence. |
| `elapsed_breakdown_seconds` | Per-phase execution time. |
| `builds[]` | One record per unique build configuration, including metrics and failures. |
| `trials[]` | One record per complete build/search candidate, including metrics and failures. |

When a build or search failure is represented by a record, `builds[].failure` or
`trials[].failure` contains its structured cause. `constraint_evaluation.violated_constraints`
shows the expected and actual values when a measured candidate misses a threshold.
`include_raw_eval: true` adds raw eval diagnostics, but those fields are not a stable AutoTune
metric contract.

`trials[].execution` is diagnostic evidence rather than a selection metric.
`requested_concurrency` is the request value. The underlying eval does not currently report the
actual worker count, so V1 does not invent an `actual_concurrency` value. V1 uses
`load_policy: "fresh_deserialize_per_trial"` and does not reuse an in-memory index between trials.

## Constraints, objectives, and workloads

Constraint direction and objective preference are fixed by the metric; requests do not include a
`direction` field.

| Metric | Constraint | Objective preference |
| --- | --- | --- |
| `recall_at_k` | at least | higher |
| `qps` | at least | higher |
| `latency_avg_ms` | at most | lower |
| `latency_p99_ms` | at most | lower |
| `search_seconds` | at most | lower |
| `build_and_search_seconds` | at most | lower |
| `build_seconds` | at most | lower |
| `index_size_mb` | at most | lower |
| `index_memory_mb` | at most | lower |

Top-level `constraints` is a non-empty object and may combine build and search metrics. The single
top-level `objective` selects one complete create/search candidate after all constraints are
applied. It contains only `metric`; V1 does not accept workload names or direction fields.

`concurrency` is part of the single search workload and configures the eval query workers. The `qps`
value is the throughput metric reported by the underlying evaluation engine at that concurrency;
it is not an index parameter candidate.

`build_seconds` excludes serialization. Values named `*_mb` use 1024 x 1024 bytes;
`index_memory_mb` comes from the index's `GetMemoryUsage()` result and is not process peak RSS.
`search_seconds` comes from eval when available; otherwise it measures the complete search eval
run, which can include deserialize and eval overhead. `build_and_search_seconds` is the sum of the
reported build and search times.

## Tune an existing index

Add `index_path` to run search-only tuning. The request must contain exactly one index spec and one
concrete set of create parameters that describes the existing serialized index. Search parameters
may still contain multiple candidates, but the request still describes exactly one workload.

`data_path` remains required because queries and ground truth are still needed. The create
parameters, including any proposer completion, must expand to exactly one build candidate.

```json
{
  "index_path": "/indexes/sift-hgraph.index",
  "indexes": [
    {
      "name": "hgraph",
      "create_params": {
        "index_param": {
          "base_quantization_type": "fp32",
          "max_degree": 32,
          "ef_construction": 200
        }
      },
      "search_params": {
        "hgraph": {
          "ef_search": [40, 80, 120]
        }
      }
    }
  ]
}
```

This fragment replaces the corresponding fields in the complete request shown earlier. AutoTune
does not modify or delete `index_path`. Build-only metrics such as `build_seconds` and
`build_and_search_seconds` are unavailable in this mode.

## Optional C++ API

The source-tree `vsag::autotune` CMake target accepts the same request and returns the compact full
report as a JSON string:

```cmake
add_subdirectory(path/to/vsag)
target_link_libraries(my_tuner PRIVATE vsag::autotune)
```

```cpp
#include <vsag/autotune.h>

const std::string report_json = vsag::autotune::RunAutoTune(request_json);
```

The target is available only when VSAG is configured with `ENABLE_TOOLS=ON` and
`ENABLE_CXX11_ABI=ON`. V1 does not export it from the installed VSAG package.

V1 does not guarantee that concurrent `RunAutoTune` calls in the same process are safe; serialize
them at the caller.

## V1 operating guidance

- V1 evaluates the full candidate grid and the full query set. Start with small candidate arrays
  and use `tuning_config.max_trials` to catch accidental combinatorial expansion.
- `max_trials` counts complete create/search trials. It defaults to 1,000 and has a V1 maximum
  of 100,000.
- V1 accepts one workload per request. Tune top-10 latency and top-100 throughput as independent
  requests; a caller may batch those requests without changing their recommendation semantics.
- `top_k` must not exceed the dataset's ground-truth width. `concurrency` must not exceed 200 or
  the query count, and `top_k * concurrency` must not exceed 1,000,000 neighbor IDs.
- Latency, QPS, and time-based results depend on the machine and VSAG build. Benchmark on hardware
  and software matching the intended deployment environment.
- V1 supports only unfiltered KNN. Use the lower-level [evaluation tool](eval.md) for range or
  filtered benchmark modes.
- Build thread controls remain native index creation parameters, not workload or tuning-control
  fields.
