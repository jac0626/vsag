# 性能评估工具（eval_performance）

eval_performance 是 VSAG 自带的命令行性能评估工具，位于 tools/eval/。编译后二进制路径为
build-release/tools/eval/eval_performance。它对一份明确的索引配置执行真实 build、load 和
search，并输出吞吐、延迟、召回率和资源指标。

## 构建

tools 默认不会编译，需要显式开启：

~~~bash
VSAG_ENABLE_TOOLS=ON make release

# 或直接使用 CMake
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DENABLE_TOOLS=ON
cmake --build build-release -j
~~~

需要系统安装 HDF5（Ubuntu：apt install libhdf5-dev；CentOS：yum install hdf5-devel）。

## 搜索模式

独立 eval 工具接受 `knn` 和 `knn_filter`，默认值仍是 `knn`。尚未实现的 `range` 和
`range_filter` 会在配置阶段被拒绝，不再输出全零的成功结果。AutoTune V1 的范围更窄，
只会调用无过滤 KNN。

## 命令行模式

命令行的 type 只接受 build 或 search。先构建并序列化索引：

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

再从相同路径加载索引并执行 KNN：

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

`--search-query-count` 设置独立 benchmark 的最少 query 操作数。若它大于数据集 query 数，
eval 会循环复用数据集 query；否则每条 query 执行一次。AutoTune 会把该值覆盖为数据集 query
数，因此每个 AutoTune trial 仍严格对每条 query 执行一次。其他常用参数包括
`--delete-index-after-search` 和用于关闭单项指标的 `--disable_*` 开关。

## YAML 配置模式

YAML 文件作为位置参数直接传入：

~~~bash
./build-release/tools/eval/eval_performance my_eval.yaml
~~~

YAML case 可以使用 type: build,search，在一个 case 中先构建、序列化，再加载并搜索：

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

一份 YAML 可以包含多个具名 case。global.exporters 下的每一项是具名 map，不是数组。
完整字段参考 tools/eval/eval_template.yaml。

## 指标

- 构建：duration、TPS 和索引内存。
- 搜索效果：平均召回率和召回率分位数。
- 搜索效率：QPS、平均延迟和 P50/P80/P90/P95/P99 延迟。
- 资源：具体索引报告的索引内存。
- 可选进程 RSS：`memory_peak(build)` 和 `memory_peak(search)` 是操作边界及每 5 ms
  采样观察到的最大增量；生命周期短于一次采样间隔的完整内存分配可能不会被捕获。

对启用 recall 指标的 KNN 模式，topk 必须不超过 HDF5 ground truth 的宽度，否则 recall 没有
有效定义。AutoTune V1 始终测量 recall，还把 top_k 上限固定为 1,000,000，因此会在评估前
同时检查这两个限制。启用 recall 时，eval runner 还要求
`topk * num_threads_searching <= 1,000,000`，确保保留结果的 batch 能提供请求并发且 neighbor
缓冲不会无界增长；AutoTune 对应校验为 `top_k * concurrency <= 1,000,000`。

延迟、QPS 和耗时指标只描述执行 benchmark 的当前机器。AutoTune recommendation 依赖这些
指标时，应在与部署环境同规格的机器上评估，包括 CPU/SIMD、核数、内存、VSAG 构建和并发
配置。

## 输出

每个 exporter 指定 format 和 to：

- format：table（或 text）、json、line_protocol。
- to：
    - stdout。
    - file://<path>，覆盖写文件。
    - influxdb://<host>:<port>/<path>?<query>，与 line_protocol 配合使用。

未配置 exporter 时默认以 table 格式输出到 stdout。

## HTTP 监控

批量配置可以启用内嵌 HTTP 服务：

~~~yaml
global:
  http_server:
    enabled: true
    port: 8080
~~~

服务展示当前 case、总 case 数、进度和最近指标。

## 数据集

eval 工具可以使用
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) 的 HDF5 数据集，例如
sift-128-euclidean.hdf5 和 gist-960-euclidean.hdf5。详细格式见
[HDF5 数据集格式](dataset_format.md)。

## 参考

- 源码：[tools/eval](https://github.com/antgroup/vsag/tree/main/tools/eval)
- 本地入口：[tools/eval/README_zh.md](../../../../../tools/eval/README_zh.md)
- 标准环境结果：[标准环境性能参考](performance.md)
