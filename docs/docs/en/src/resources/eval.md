# Performance Evaluation Tool (eval_performance)

eval_performance is VSAG's command-line evaluation tool under tools/eval/. The built executable is
build-release/tools/eval/eval_performance. It runs real build, load, and search operations for one
concrete index configuration and reports throughput, latency, recall, and resource metrics.

## Building

Tools are disabled by default and must be enabled explicitly:

~~~bash
VSAG_ENABLE_TOOLS=ON make release

# Or use CMake directly
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release -j
~~~

HDF5 must be installed (Ubuntu: apt install libhdf5-dev; CentOS: yum install hdf5-devel).

## Search Modes

The standalone eval tool accepts `knn` and `knn_filter`; `knn` remains the default. The
unimplemented `range` and `range_filter` modes are rejected during configuration instead of
emitting zero-valued success results. AutoTune V1 invokes only unfiltered KNN.

## Command-Line Mode

The command-line type accepts build or search. First build and serialize the index:

~~~bash
CREATE_PARAMS='{"dim":128,"dtype":"float32","metric_type":"l2",'\
'"index_param":{"base_quantization_type":"fp32","max_degree":32,'\
'"ef_construction":300}}'
./build-release/tools/eval/eval_performance \
    --datapath /tmp/sift-128-euclidean.hdf5 \
    --index_name hgraph \
    --type build \
    --create_params "$CREATE_PARAMS" \
    --index_path /tmp/vsag_eval/hgraph_fp32.index
~~~

Then load the same artifact and run KNN:

~~~bash
CREATE_PARAMS='{"dim":128,"dtype":"float32","metric_type":"l2",'\
'"index_param":{"base_quantization_type":"fp32","max_degree":32,'\
'"ef_construction":300}}'
./build-release/tools/eval/eval_performance \
    --datapath /tmp/sift-128-euclidean.hdf5 \
    --index_name hgraph \
    --type search \
    --create_params "$CREATE_PARAMS" \
    --search_params '{"hgraph":{"ef_search":60}}' \
    --index_path /tmp/vsag_eval/hgraph_fp32.index \
    --search_mode knn \
    --search-query-count 100000 \
    --topk 10
~~~

`--search-query-count` sets the minimum number of query operations for a standalone benchmark. If
it exceeds the dataset query count, eval repeats dataset queries cyclically; otherwise every query
is evaluated once. AutoTune overrides this value with the dataset query count, so each AutoTune
trial still executes every query exactly once. Other useful flags include
`--delete-index-after-search` and the `--disable_*` flags for individual metrics.

## YAML Configuration Mode

Pass the YAML file as the positional argument:

~~~bash
./build-release/tools/eval/eval_performance my_eval.yaml
~~~

A YAML case may use type: build,search to build, serialize, load, and search in one case:

~~~yaml
global:
  num_threads_building: 8
  num_threads_searching: 16
  exporters:
    print-directly:
      to: stdout
      format: table
    save-to-file:
      to: "file:///tmp/eval_results.json"
      format: json

eval_case1:
  datapath: /tmp/sift-128-euclidean.hdf5
  type: build,search
  index_name: hgraph
  create_params: >-
    {"dim":128,"dtype":"float32","metric_type":"l2",
    "index_param":{"base_quantization_type":"fp32","max_degree":32,
    "ef_construction":300}}
  search_params: '{"hgraph":{"ef_search":60}}'
  index_path: /tmp/vsag_eval/hgraph_fp32.index
  search_mode: knn
  search_query_count: 100000
  topk: 10
~~~

One YAML file may define multiple named cases. Each entry under global.exporters is a named map,
not a list item. See tools/eval/eval_template.yaml for the complete shape.

## Metrics

- Build: duration, TPS, and index memory.
- Search quality: average recall and recall quantiles.
- Search efficiency: QPS, average latency, and P50/P80/P90/P95/P99 latency.
- Resources: index memory reported by the concrete index.
- Optional process RSS: `memory_peak(build)` and `memory_peak(search)` report the maximum
  increase observed at operation boundaries and by a 5 ms sampler. They may miss allocations
  whose complete lifetime is shorter than one sampling interval.

For KNN modes with recall metrics enabled, topk must not exceed the HDF5 ground-truth width;
otherwise recall is not well-defined. AutoTune V1 always measures recall, also caps top_k at
1,000,000, and therefore enforces both limits before evaluation. The eval runner also requires
`topk * num_threads_searching <= 1,000,000` when recall is enabled, so a retained result batch can
provide the requested concurrency without unbounded neighbor buffering; AutoTune validates the
same rule as `top_k * concurrency <= 1,000,000`.

Latency, QPS, and elapsed-time results describe the machine that ran the benchmark. When an
AutoTune recommendation depends on those metrics, run the benchmark on a deployment-equivalent
machine, including CPU/SIMD capabilities, core count, memory, VSAG build, and concurrency settings.

## Output

Each exporter specifies a format and a to destination:

- format: table (or text), json, or line_protocol.
- to:
    - stdout.
    - file://<path>, which overwrites the file.
    - influxdb://<host>:<port>/<path>?<query>, used with line_protocol.

Without exporters, results are printed to stdout as a table.

## HTTP Monitoring

Batch configurations may enable the embedded HTTP service:

~~~yaml
global:
  http_server:
    enabled: true
    port: 8080
~~~

It exposes the current case, total case count, progress, and latest metrics.

## Datasets

The eval tool works with HDF5 datasets from
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks), such as
sift-128-euclidean.hdf5 and gist-960-euclidean.hdf5. See
[HDF5 Dataset Format](dataset_format.md) for the complete contract.

## References

- Source: [tools/eval](https://github.com/antgroup/vsag/tree/main/tools/eval)
- Local entry point: [tools/eval/README.md](../../../../../tools/eval/README.md)
- Standard-environment results: [Reference Performance](performance.md)
