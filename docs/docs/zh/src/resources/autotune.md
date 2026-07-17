# AutoTune（离线参数调优）

AutoTune 用于自动完成离线参数调优流程：展开索引参数候选、构建索引、
调用 VSAG 评测引擎、应用约束并选出可复现的参数组合。V1 使用全网格并
评测全量 query，以建立正确性基线；它减少人工编写脚本的工作，但不承诺
比人工网格搜索更快。

AutoTune V1 支持 [HGraph](../indexes/hgraph.md) 和
[IVF](../indexes/ivf.md)、dense `float32` HDF5 数据集以及无过滤 KNN
workload。它既可以联合评测构建、量化与搜索参数，也可以
针对已有索引只调搜索参数。

## 编译 CLI

工具默认不参与编译。在 VSAG 仓库根目录执行：

```bash
VSAG_ENABLE_TOOLS=ON make release
```

编译产物为：

```text
build-release/tools/autotune/autotune
```

也可以用 CMake 只编译 AutoTune target：

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TOOLS=ON \
  -DENABLE_CXX11_ABI=ON
cmake --build build-release --target autotune --parallel
```

`ENABLE_CXX11_ABI` 默认开启，但关闭后不会把 AutoTune 加入构建。平台依赖见
[编译 VSAG](../development/building.md)。如果要把 stdout 当作 JSON 处理，建议使用
Release 版本的 CLI。

V1 不安装可选 C++ library 或 CLI executable；请从 build tree 使用二者。

## 准备数据集

`data_path` 必须指向一个包含 base、query 和 ground truth 的 HDF5 文件。AutoTune 会从
文件中推断 `dim`、`dtype` 和 `metric_type`，因此 request 不需要重复填写
这些元数据。
数据格式见 [HDF5 数据集格式](dataset_format.md)。

V1 只接受 dense `float32` 数据；每个 trial 都会评测文件中的完整 query 集。

## 编写 request

下面的 request 构建一次 HGraph，并评测三个 `ef_search` 值。运行前请替换
数据集路径。

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

将其保存为 `request.json`。本例固定了 build 参数，只把 `ef_search` 作为
候选轴，因此只构建一次索引并执行三个 search trial。
`keep_intermediate: true` 会在请求结束后保留生成的索引文件；如果只需要
推荐参数，可以设为 `false`。

每个参数叶子可以使用以下写法：

| 写法 | 含义 |
| --- | --- |
| `"max_degree": 32` | 一个固定值。 |
| `"max_degree": [16, 32]` | 两个候选值。 |
| `"ef_search": {"$range":{"start":40,"stop":120,"step":40}}` | 包含终点的范围。 |

多个候选轴会形成笛卡尔积。省略受支持的字段时，HGraph 或 IVF 的内置
proposer 可以只为缺失字段补充候选；用户显式值始终优先。AutoTune 不会在
省略 `indexes` 时自动选择索引，`indexes` 是必填字段。

HGraph proposer 会补充 `base_quantization_type`、`max_degree`、
`ef_construction` 和 `ef_search`。IVF proposer 会补充
`base_quantization_type`、`buckets_count` 和 `scan_buckets_count`。
其他省略字段保持缺失，由具体索引使用原生默认值。

## 运行 CLI

CLI 只接受一个 request 文件：

```bash
set -o pipefail
./build-release/tools/autotune/autotune request.json \
  | tee /tmp/autotune-summary.json
```

命令会把精简 summary 写到 stdout。配置 `output.result_path` 后，完整报告会
写入该路径。

## 阅读 summary

使用 Release CLI 和默认日志设置时，stdout JSON 只包含当前结果有意义的字段。

展示顺序会根据状态把可操作结果放在最前面。成功时先输出
`recommendation`，约束不满足时先输出 `best_effort`，失败时先输出
`failure`。只输出实际适用的结果分支，不用 `null` 填充其他分支；只有完整报告成功写入时
才输出 `report_path`。recommendation 内部先输出 `index_name`、`create_params` 和
`search_params`，随后输出 workload、指标与证据，便于定位和复制。
full report 不受影响。
JSON 调用方仍不能把对象字段顺序当作语义契约。

| 字段 | 阅读方式 |
| --- | --- |
| `recommendation` | 成功时输出；满足约束的一套完整 create/search 配置。 |
| `best_effort` | 没有候选满足全部约束时输出；仅用于解释。 |
| `failure` | 失败时输出；包含结构化 `stage`、`code` 和 `message`。 |
| `status` | `success`、`no_candidate_satisfied` 或 `failed`。 |
| `elapsed_seconds` | 不含报告文件写入和清理的端到端调优耗时。 |
| `report_path` | 仅在完整报告成功写入时输出。 |
| `version` | 输出契约版本；V1 固定为 `1`。 |

`status` 的语义如下：

| status | 含义 | CLI 退出码 |
| --- | --- | --- |
| `success` | `recommendation` 满足全部约束。 | `0` |
| `no_candidate_satisfied` | 评测已完成，但没有完整候选满足全部约束。 | `0` |
| `failed` | validation、planning、evaluation、selection 或报告写入失败。 | 非零 |

不能只根据退出码判断是否得到 recommendation。`no_candidate_satisfied`
表示调优任务正常完成，但其 `best_effort` 不保证满足请求，不能把它当作
满足约束的推荐结果。

当 `status=success` 时，重点读取：

- `recommendation.index_name` 和 `recommendation.create_params`：创建索引所需参数；
- `recommendation.search_params`：搜索所需参数；
- `recommendation.workload`：benchmark 的 `top_k` 和 `concurrency`；
- `recommendation.metrics`：全部可用的 build 和 search 指标；
- `recommendation.evidence`：在完整报告中定位来源 build 和 trial；
- 使用 artifact 路径前检查 `recommendation.artifacts.expected_to_exist_after_response`。

可以用 `jq` 查看关键字段：

```bash
jq '{status, elapsed_seconds, recommendation, best_effort, failure}' \
  /tmp/autotune-summary.json
```

## 阅读完整报告

完整报告用于审计候选展开、逐项指标和失败原因：

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

完整报告中最常用的部分是：

| 部分 | 内容 |
| --- | --- |
| `input_request` | 用户原始 request 解析后的 JSON 值。 |
| `effective_request` | 规范化 request，以及 AutoTune 从数据集推断出的元数据。 |
| `environment` | VSAG、操作系统、CPU、内存和 SIMD 证据。 |
| `elapsed_breakdown_seconds` | 各执行阶段的耗时。 |
| `builds[]` | 每个唯一 build 配置一条记录，包括指标和失败。 |
| `trials[]` | 每个完整 build/search 候选一条记录，包括指标和失败。 |

当 build 或 search 失败被表示为报告记录时，`builds[].failure` 或
`trials[].failure` 会包含结构化原因。`constraint_evaluation.violated_constraints` 会记录
未达到阈值时的期望值和实际值。`include_raw_eval: true` 可以附加 eval
原始诊断信息，但这些字段不属于稳定的 AutoTune 指标契约。

`trials[].execution` 是诊断证据，不参与结果选择。`requested_concurrency` 是
request 值。底层 eval 当前不报告实际 worker 数，因此 V1 不虚构
`actual_concurrency`。V1 使用 `load_policy: "fresh_deserialize_per_trial"`，
不同 trial 不复用内存中的索引实例。

## 约束、目标和 workload

约束方向和目标偏好由 metric 固定，request 不填写 `direction`。

| metric | 约束语义 | 目标偏好 |
| --- | --- | --- |
| `recall_at_k` | 大于等于 | 越大越优 |
| `qps` | 大于等于 | 越大越优 |
| `latency_avg_ms` | 小于等于 | 越小越优 |
| `latency_p99_ms` | 小于等于 | 越小越优 |
| `search_seconds` | 小于等于 | 越小越优 |
| `build_and_search_seconds` | 小于等于 | 越小越优 |
| `build_seconds` | 小于等于 | 越小越优 |
| `index_size_mb` | 小于等于 | 越小越优 |
| `index_memory_mb` | 小于等于 | 越小越优 |

顶层 `constraints` 是非空对象，可以同时包含 build 和 search 指标。唯一的顶层
`objective` 在应用全部约束后，从完整 create/search 候选中选出 recommendation。
它只包含 `metric`；V1 不接受 workload 名称或 direction 字段。

`concurrency` 属于这个单一 search workload，用于配置 eval 的 query worker。`qps` 是底层
评测引擎在该并发度下报告的吞吐指标；它不是索引参数候选。

`build_seconds` 不包含序列化。名称为 `*_mb` 的指标按 1024 x 1024 bytes 换算；
`index_memory_mb` 来自索引的 `GetMemoryUsage()`，不是进程峰值 RSS。
`search_seconds` 优先使用 eval 输出；底层没有该值时，它测量完整 search
eval run，
可能包含 deserialize 和 eval 开销。`build_and_search_seconds` 是报告中的 build 与
search 时间之和。

## 调优已有索引

增加 `index_path` 即可进入 search-only 调优。request 必须只包含一个 index
spec，并用一组确定的 create 参数准确描述已序列化索引；search 参数仍可
有多个候选，但 request 仍然只描述一个 workload。

`data_path` 仍然必填，因为评测仍需要 query 和 ground truth。create 参数加上
proposer 补齐后，必须只展开出一个 build candidate。

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

这个片段用于替换前面完整 request 中的对应字段。AutoTune 不会修改或删除
`index_path`；该模式无法提供 `build_seconds` 和 `build_and_search_seconds` 等需要
本次构建的指标。

## 可选 C++ API

源码树内的 `vsag::autotune` CMake target 接受相同的 request，并以 JSON 字符串返回
compact full report：

```cmake
add_subdirectory(path/to/vsag)
target_link_libraries(my_tuner PRIVATE vsag::autotune)
```

```cpp
#include <vsag/autotune.h>

const std::string report_json = vsag::autotune::RunAutoTune(request_json);
```

只有在配置 VSAG 时同时开启 `ENABLE_TOOLS=ON` 和 `ENABLE_CXX11_ABI=ON`，该 target 才可用。
V1 不从安装后的 VSAG package 导出它。

V1 不保证同一进程内并发调用 `RunAutoTune` 的安全性；调用方应串行执行。

## V1 使用建议

- V1 会评测完整候选网格和完整 query 集。先使用较小的候选数组，并通过
  `tuning_config.max_trials` 防止意外的组合爆炸。
- `max_trials` 统计完整 create/search trial，默认值为 1,000，V1 上限为
  100,000。
- V1 每个 request 只接受一个 workload。top-10 延迟和 top-100 吞吐应分别调优；
  调用方可以批量提交这些独立请求，但不能改变每个请求的 recommendation 语义。
- `top_k` 不能超过数据集 ground-truth 宽度；`concurrency` 不能超过 200 或 query
  数量；`top_k * concurrency` 不能超过 1,000,000 个 neighbor ID。
- latency、QPS 和时间指标依赖机器与 VSAG 构建。benchmark 的软硬件环境应与
  预期部署环境保持一致。
- V1 只支持无过滤 KNN。range 或 filtered benchmark 请使用更底层的
  [性能评估工具](eval.md)。
- 构建线程仍是具体索引的原生 create 参数，不属于 workload 或 tuning control。
