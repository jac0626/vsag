# AutoTune API v1 草案

状态：P6 草案，已对齐当前 `tools/autotune` 实现

本文档定义 VSAG AutoTune 的用户输入、输出结果和执行语义。它不是公开稳定 API
文档，当前用途是作为实现与评审的共同契约。后续 API 稳定后，再同步到
`docs/docs/{zh,en}/src/` 的用户文档中。

当前实现入口是 `tools/autotune/autotune`，输入为 JSON request 文件，核心路径
直接复用 `tools/eval` 的进程内 build/search 能力。本文档中标为“当前已实现”的行为
必须和代码保持一致；标为“后续”的行为只代表 API 演进方向。

## 1. 定位

AutoTune 的第一阶段定位是：在现有 `eval_performance` 能力之上增加一层官方的参数
枚举、评估编排和结果选择。

第一阶段要解决的问题：

- 用户不需要手写脚本生成大量 eval 配置。
- 用户可以把可调参数写成数组或范围，由系统展开成候选组合。
- 用户用约束表达目标，例如最低召回、最大延迟、最大内存。
- 系统执行 build/search 评估，返回最优参数、完整试验记录和耗时。

第一阶段不解决的问题：

- 不承诺比人工网格搜索更快。
- 不实现复杂优化器、学习型搜索、自动剪枝策略。
- 不把 `Index::Tune()` 或 Build Cache 作为核心路径。
- 不让系统在用户未指定索引集合时自动选择索引类型。
- 不支持 query sampling、successive halving、build cache、分布式执行。
- 不支持自动调用 `Index::Tune()` 热修改已有索引。

当前已实现能力：

- 支持 JSON request。
- 支持 `hgraph` 和 `ivf` 两类索引候选。
- 支持按 build group 只构建一次，再对同一构建产物执行多个 search trial。
- 支持已有索引上的 search-only 调优。
- 支持数组候选、`$range` 候选和 `$value` 数组转义。
- 支持系统默认补齐 HGraph / IVF 的基础候选空间。
- 支持 `knn` search mode。
- 支持 build 指标与 search 指标合并后的约束过滤。
- 支持推荐结果选择、完整 build/trial 报告和结构化失败输出。

## 2. 设计原则

1. 输入尽量接近现有 eval 输入。
2. 用户侧不暴露 eval 的 `type` 字段。AutoTune 根据输入推导 build group 和
   search trial。
3. `indexes` 必须显式给出。索引类型差异很大，系统不在 v1 中替用户猜索引集合。
4. `create_params` 和 `search_params` 使用 VSAG 现有参数结构，不重新发明参数名。
5. 标量表示固定值，数组表示候选值，缺失字段表示系统默认候选策略。
6. 约束优先。系统先筛选满足约束的候选，再从满足约束的候选中选择最低平均延迟。
7. 所有系统补齐的默认值必须进入输出，避免用户拿不到可复现实验配置。

## 3. Request 总体结构

AutoTune request 使用 JSON 语义定义。实现可以支持 JSON 文件、YAML 文件或 C++ API
对象，但字段含义必须一致。

```json
{
  "version": 1,
  "data_path": "/data/sift-128-euclidean.hdf5",
  "indexes": [
    {
      "name": "hgraph",
      "create_params": {},
      "search_params": {}
    }
  ],
  "constraints": {
    "recall_at_k": 0.95
  },
  "execution": {
    "top_k": 10,
    "search_mode": "knn",
    "num_threads_building": 48,
    "num_threads_searching": 48,
    "workspace_path": "/tmp/vsag_autotune",
    "keep_intermediate": false
  },
  "output": {
    "result_path": "/tmp/vsag_autotune/result.json",
    "include_trials": true
  }
}
```

## 4. 顶层字段

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `version` | 是 | int | API 版本。当前固定为 `1`。 |
| `data_path` | 是 | string | 评估数据集路径。第一阶段要求是 eval 可读取的数据集。 |
| `index_path` | 否 | string | 已存在索引路径。存在时可以用于 search-only 调优。 |
| `indexes` | 是 | array | 候选索引集合。每个元素对应一个索引类型和它的候选参数空间。 |
| `constraints` | 是 | object | 硬约束。至少应包含一个质量或资源约束。 |
| `execution` | 否 | object | 执行控制，例如 topk、搜索模式、线程数、工作目录。 |
| `output` | 否 | object | 输出控制，例如结果文件路径、是否保留完整 trials。 |

### 4.1 `data_path`

`data_path` 是必填字段，即使传入了 `index_path` 也不能省略。

原因是现有 eval search 需要读取 query 和 ground truth 来计算召回、延迟、QPS 等指标。
仅有一个索引文件无法完成质量评估。

第一阶段要求 `data_path` 指向 eval 已支持的数据集格式，例如
`sift-128-euclidean.hdf5`。如果未来需要支持“base vectors、query vectors、ground truth
分成多个文件”的输入，应扩展为新的数据源字段，而不是改变 `data_path` 的含义。

### 4.2 `index_path`

`index_path` 表示一个已存在索引。它是只读输入，不允许 AutoTune 覆盖。

当 request 展开后只有一个 build candidate 时，AutoTune 可以对 `index_path` 执行
search-only 评估。此时 `index_path` 只读，`build_count = 0`，`builds[].eval_type`
等价于 `existing_index`。

当 request 传入 `index_path`，但 `indexes` 展开后存在多个 build candidate 时，当前实现会
返回结构化失败，失败信息包含 `index_path can only be used`。这样可以避免用户以为正在调
已有索引，实际却触发临时 rebuild。

第一阶段若复用 `eval_performance`，即使是 search-only，也仍然需要给出能够创建索引对象的
`create_params`。原因是 eval 当前会先通过 `index_name + create_params` 创建 index 对象，
然后再从 `index_path` 反序列化。

## 5. `indexes`

`indexes` 是候选索引数组。数组中的每个元素独立描述一种索引类型。

```json
{
  "name": "hgraph",
  "create_params": {
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
      "base_quantization_type": "fp32",
      "max_degree": [16, 32, 48],
      "ef_construction": 300
    }
  },
  "search_params": {
    "hgraph": {
      "ef_search": [40, 80, 120]
    }
  }
}
```

字段说明：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| `name` | 是 | string | VSAG 索引名。当前支持 `hgraph`、`ivf`。 |
| `create_params` | 否 | object | 传给 `Factory::CreateIndex` 的参数结构。 |
| `search_params` | 否 | object | 传给 search API 的参数结构。 |

支持的索引集合由 AutoTune 实现显式声明。未声明支持的索引必须在 validation 阶段失败。
当前只声明支持 `hgraph` 和 `ivf`；如果用户传入其他索引名，应返回结构化失败结果，
失败信息包含 `unsupported index`。

后续可以增加 `sindi`、`brute_force` 等索引 policy。`diskann`、`hnsw`、`sparse`
不进入 v1 支持集合。

## 6. 参数候选表达

AutoTune 不改造 VSAG 参数系统。参数对象中的叶子值按以下规则解释：

| 写法 | 含义 |
| --- | --- |
| `"max_degree": 32` | 固定值，只生成一个候选。 |
| `"max_degree": [16, 32, 48]` | 候选值数组，展开为三个候选。 |
| `"ef_search": {"$range": {"start": 40, "stop": 200, "step": 40}}` | 闭区间范围，生成 `40, 80, 120, 160, 200`。 |
| `"centroids": {"$value": [1, 2, 3]}` | 参数真实值就是数组，不作为候选集合展开。 |

### 6.1 数组语义

数组默认表示候选集合。例如：

```json
{
  "index_param": {
    "max_degree": [16, 32],
    "ef_construction": [100, 200]
  }
}
```

展开后生成四个 build 参数组合：

```text
max_degree=16, ef_construction=100
max_degree=16, ef_construction=200
max_degree=32, ef_construction=100
max_degree=32, ef_construction=200
```

如果某个 VSAG 参数的真实值就是数组，必须使用 `$value`，避免和候选集合冲突。

### 6.2 缺失参数

缺失参数由系统默认策略补齐。默认策略必须满足两点：

1. 每个索引类型有明确、可测试、可文档化的默认候选集。
2. 输出中必须包含所有补齐后的最终参数。

例如用户只写：

```json
{
  "name": "hgraph",
  "create_params": {
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2"
  }
}
```

系统可以为 HGraph 自动补齐 `base_quantization_type`、`max_degree`、
`ef_construction` 和 `ef_search` 的候选集。具体默认值不在 API 层写死，
由实现侧的 index policy 维护。

## 7. `constraints`

`constraints` 是硬约束。候选必须全部满足已声明约束，才进入最终选择集合。
约束方向由字段名固定决定，不需要用户再写 `min` 或 `max`。

```json
{
  "constraints": {
    "recall_at_k": 0.95,
    "latency_avg_ms": 2.0,
    "memory_peak_mb": 8192,
    "build_seconds": 3600,
    "index_size_mb": 2048
  }
}
```

约束字段：

| 字段 | 约束语义 | 说明 |
| --- | --- | --- |
| `recall_at_k` | 下限 | KNN 召回率必须大于等于该值。`k` 来自 `execution.top_k`。 |
| `latency_avg_ms` | 上限 | 平均单 query 延迟必须小于等于该值。 |
| `latency_p99_ms` | 上限 | P99 单 query 延迟必须小于等于该值。 |
| `qps` | 下限 | 查询吞吐必须大于等于该值。 |
| `memory_peak_mb` | 上限 | 峰值内存必须小于等于该值。 |
| `build_seconds` | 上限 | 构建耗时必须小于等于该值。 |
| `index_size_mb` | 上限 | 索引产物大小必须小于等于该值。 |

指标来源：

- `recall_at_k`、`latency_avg_ms`、`latency_p99_ms`、`qps` 来自 search eval。
- `build_seconds` 来自 build eval。复用同一 build group 的多个 search trial 共享同一个
  `build_seconds`。
- `index_size_mb` 来自 build group 的索引产物大小。
- `memory_peak_mb` 是 build 侧和 search 侧峰值内存的较大值。
- 使用已有 `index_path` 时，AutoTune 不执行 build，因此不会产生 `build_seconds`。如果用户在
  existing index 场景声明 `build_seconds` 约束，该 trial 会因为缺少指标而不满足约束。

当前只接受上表列出的约束名。未知约束名必须在 validation 阶段失败，失败信息包含
`unsupported constraint`。

如果某个 trial 没有产生约束要求的指标，该 trial 视为不满足约束，并在
`violated_constraints` 中记录 `missing_metric`。因此用户应只声明本次评估能产生的指标。

## 8. `execution`

`execution` 控制评估过程，不表达调优目标。

```json
{
  "execution": {
    "top_k": 10,
    "search_mode": "knn",
    "search_query_count": 10000,
    "num_threads_building": 48,
    "num_threads_searching": 48,
    "workspace_path": "/tmp/vsag_autotune",
    "keep_intermediate": false,
    "max_trials": 1000
  }
}
```

字段说明：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `top_k` | `10` | KNN topk。 |
| `search_mode` | `knn` | 搜索模式。当前只支持 `knn`。 |
| `search_query_count` | `0` | 参与评估的 query 数。`0` 表示使用 eval 默认的全量 query 行为。 |
| `num_threads_building` | `1` | 构建线程数。 |
| `num_threads_searching` | `1` | 搜索线程数。 |
| `workspace_path` | `/tmp/vsag_autotune` | build artifact、eval 配置和中间结果目录。 |
| `keep_intermediate` | `false` | 是否保留 build artifact 和 eval 配置。 |
| `max_trials` | 无限制 | 候选组合数量上限。超过时 validation 失败。 |

### 8.1 `search_mode`

`search_mode` 的语义与 eval 保持一致。当前只支持：

- `knn`

如果用户传入其他值，当前实现返回结构化失败结果，失败信息包含
`execution.search_mode is unsupported`。

后续可以在不改变 request 结构的前提下增加：

- `range`
- `knn_filter`
- `range_filter`

## 9. 执行类型推导

用户不传 `type`。AutoTune 根据候选空间推导 eval 执行方式。

| 输入情况 | 推导执行方式 | 说明 |
| --- | --- | --- |
| 无 `index_path` | build group + search trials | 每个唯一 build candidate 只构建一次，再复用该索引执行多个 search trial。 |
| 有 `index_path`，只有一个 build candidate | existing index + search trials | 不执行 build，加载已有索引，评估不同 search 参数。 |
| 有 `index_path`，存在多个 build candidates | validation 失败 | 当前实现拒绝该输入，避免静默 rebuild。 |
| 只要求构建指标，不要求搜索指标 | build-only | 后续能力，当前不支持。 |

第一阶段主路径是 build group + search trials 和 existing index + search trials。

## 10. 结果选择规则

AutoTune 默认不暴露 `objective` 字段。

固定选择规则：

1. 过滤所有失败 trial。
2. 过滤不满足 `constraints` 的 trial。
3. 如果存在满足约束的 trial，选择 `latency_avg_ms` 最低的 trial。
4. 如果 `latency_avg_ms` 相同，依次比较 `memory_peak_mb`、`build_seconds`、trial id。
5. 如果没有 trial 满足约束，返回 `status = "no_candidate_satisfied"`，同时给出
   `best_effort`。`best_effort` 按 `recall_at_k` 最高、`latency_avg_ms` 最低排序。

这个规则保证结果可解释、可复现。复杂目标函数可以在后续版本增加，但不进入第一阶段输入。

## 11. Response 总体结构

```json
{
  "version": 1,
  "status": "success",
  "elapsed_seconds": 128.41,
  "elapsed_breakdown_seconds": {
    "validation": 0.02,
    "candidate_generation": 0.01,
    "evaluation": 128.35,
    "selection": 0.03
  },
  "recommendation": {
    "trial_id": "hgraph-000002",
    "index_name": "hgraph",
    "create_params": {
      "dim": 128,
      "dtype": "float32",
      "metric_type": "l2",
      "index_param": {
        "base_quantization_type": "fp32",
        "max_degree": 32,
        "ef_construction": 300
      }
    },
    "search_params": {
      "hgraph": {
        "ef_search": 80
      }
    },
    "metrics": {
      "recall_at_k": 0.957,
      "latency_avg_ms": 1.73,
      "latency_p99_ms": 4.91,
      "qps": 27742.0,
      "memory_peak_mb": 6144.0,
      "build_seconds": 96.4,
      "index_size_mb": 1240.5
    },
    "selection_reason": "satisfied constraints and had the lowest latency_avg_ms"
  },
  "best_effort": null,
  "trial_count": 2,
  "build_count": 1,
  "build_group_count": 1,
  "failure": null,
  "builds": [
    {
      "build_id": "hgraph-build-000001",
      "status": "success",
      "index_name": "hgraph",
      "eval_type": "build",
      "create_params": {
        "dim": 128,
        "dtype": "float32",
        "metric_type": "l2",
        "index_param": {
          "base_quantization_type": "fp32",
          "max_degree": 32,
          "ef_construction": 300
        }
      },
      "metrics": {
        "build_seconds": 96.4,
        "memory_peak_mb": 6144.0,
        "index_size_mb": 1240.5
      },
      "elapsed_seconds": 96.8,
      "artifacts": {
        "index_path": "/tmp/vsag_autotune/trials/hgraph-build-000001.index",
        "use_existing_index": false,
        "cleanup_index_after_build_group": true
      },
      "failure": null
    }
  ],
  "trials": [
    {
      "trial_id": "hgraph-000001",
      "status": "success",
      "build_id": "hgraph-build-000001",
      "index_name": "hgraph",
      "eval_type": "search",
      "create_params": {
        "dim": 128,
        "dtype": "float32",
        "metric_type": "l2",
        "index_param": {
          "base_quantization_type": "fp32",
          "max_degree": 32,
          "ef_construction": 300
        }
      },
      "search_params": {
        "hgraph": {
          "ef_search": 40
        }
      },
      "metrics": {
        "recall_at_k": 0.942,
        "latency_avg_ms": 1.12,
        "latency_p99_ms": 3.86,
        "qps": 30321.0,
        "memory_peak_mb": 6144.0,
        "build_seconds": 96.4,
        "index_size_mb": 1240.5
      },
      "satisfied_constraints": false,
      "violated_constraints": [
        {
          "name": "recall_at_k",
          "direction": "min",
          "expected": 0.95,
          "actual": 0.942
        }
      ],
      "elapsed_seconds": 15.4,
      "artifacts": {
        "index_path": "/tmp/vsag_autotune/trials/hgraph-build-000001.index"
      },
      "failure": null
    },
    {
      "trial_id": "hgraph-000002",
      "status": "success",
      "build_id": "hgraph-build-000001",
      "index_name": "hgraph",
      "eval_type": "search",
      "create_params": {
        "dim": 128,
        "dtype": "float32",
        "metric_type": "l2",
        "index_param": {
          "base_quantization_type": "fp32",
          "max_degree": 32,
          "ef_construction": 300
        }
      },
      "search_params": {
        "hgraph": {
          "ef_search": 80
        }
      },
      "metrics": {
        "recall_at_k": 0.957,
        "latency_avg_ms": 1.73,
        "latency_p99_ms": 4.91,
        "qps": 27742.0,
        "memory_peak_mb": 6144.0,
        "build_seconds": 96.4,
        "index_size_mb": 1240.5
      },
      "satisfied_constraints": true,
      "violated_constraints": [],
      "elapsed_seconds": 15.9,
      "artifacts": {
        "index_path": "/tmp/vsag_autotune/trials/hgraph-build-000001.index"
      },
      "failure": null
    }
  ]
}
```

### 11.1 `elapsed_seconds`

顶层 `elapsed_seconds` 是端到端墙钟时间，从 request validation 开始，到 result 写出结束。

它包含：

- 输入校验时间。
- 候选生成时间。
- build group 执行时间。
- search trial 执行时间。
- 指标解析和结果选择时间。

`builds[].elapsed_seconds` 是单个 build group 的耗时。`trials[].elapsed_seconds` 是单个
search trial 的耗时，不包含 build 耗时。trial 的 `metrics.build_seconds` 来自对应
build group 的 build 指标，不来自 trial 自身耗时。

### 11.2 `status`

| 值 | 说明 |
| --- | --- |
| `success` | 至少一个 trial 成功且满足全部约束。 |
| `no_candidate_satisfied` | 有 trial 成功，但没有 trial 满足全部约束。 |
| `failed` | 没有可用 trial，或 request validation 失败。 |

### 11.3 `failure`

`failure` 是顶层失败原因。

- `status = "success"` 时，`failure` 为 `null`。
- `status = "no_candidate_satisfied"` 时，`failure` 为 `null`，原因在 `best_effort` 和
  trial 的 `violated_constraints` 中。
- `status = "failed"` 时，`failure` 是对象，至少包含 `message`。

validation 失败时，`trial_count = 0`，`build_count = 0`，`build_group_count = 0`，
通常没有 `builds` 和 `trials` 字段。

所有 trial 都失败时，顶层 `failure.message` 为 `all trials failed`，每个 trial 的
`failure` 字段记录各自的失败原因。

### 11.4 `build_count` 和 `build_group_count`

`build_group_count` 是本次 plan 中唯一 build candidate 的数量。

`build_count` 是实际执行 build 的次数：

- 无 `index_path` 时，`build_count == build_group_count`。
- 有 `index_path` 且输入合法时，`build_group_count = 1`，`build_count = 0`。

这两个字段用于解释 AutoTune 的执行成本。`trial_count` 仍然表示 search trial 数量，也就是
最终参与约束过滤和结果选择的候选数量。

### 11.5 `builds`

`builds` 是 build group 记录。第一阶段默认在 `output.include_trials = true` 时输出。
如果 `output.include_trials = false`，响应可以省略 `builds` 和 `trials`。

每个 build 记录包含：

- `build_id`：build group 的稳定 ID。
- `status`：`success` 或 `failed`。
- `eval_type`：无 `index_path` 时为 `build`；复用已有索引时为 `existing_index`。
- 展开后的完整 `create_params`。
- build 侧指标。
- build group 耗时。
- `artifacts.index_path`。
- `artifacts.use_existing_index`。
- `artifacts.cleanup_index_after_build_group`。
- `failure`。

如果 build 失败，该 build group 下所有 search trial 都会失败，失败原因包含
`build failed`。

### 11.6 `trials`

`trials` 是完整试验记录。第一阶段默认保留，方便 review 和复现实验。
如果 `output.include_trials = false`，响应可以省略该字段。

每个 trial 必须记录：

- `build_id`。
- 展开后的完整 `create_params`。
- 展开后的完整 `search_params`。
- eval 执行类型。当前实现中 trial 固定为 `search`。
- 成功、失败或跳过状态。
- 合并后的指标结果。
- trial 耗时。
- 失败原因。
- 可选产物路径。

trial 的 `metrics` 是 build metrics 和 search metrics 的合并结果：

- `recall_at_k`、`latency_avg_ms`、`latency_p99_ms`、`qps` 来自 search eval。
- `build_seconds` 来自 build eval。
- `index_size_mb` 来自 build group 的索引产物。
- `memory_peak_mb` 是 build/search 两侧峰值内存的较大值。

当 `execution.keep_intermediate = false` 时，AutoTune 会在 build group 下所有 search trial
完成后删除临时索引。此时 `artifacts.index_path` 表示该 trial 使用过的路径，不保证响应
返回后文件仍存在。

## 12. 示例

### 12.1 最小 HGraph 调优

用户只指定数据、索引类型和约束。系统补齐 HGraph 默认候选空间。

```json
{
  "version": 1,
  "data_path": "/data/sift-128-euclidean.hdf5",
  "indexes": [
    {
      "name": "hgraph",
      "create_params": {
        "dim": 128,
        "dtype": "float32",
        "metric_type": "l2"
      }
    }
  ],
  "constraints": {
    "recall_at_k": 0.95
  },
  "execution": {
    "top_k": 10
  }
}
```

### 12.2 显式 HGraph build/search 调优

```json
{
  "version": 1,
  "data_path": "/data/sift-128-euclidean.hdf5",
  "indexes": [
    {
      "name": "hgraph",
      "create_params": {
        "dim": 128,
        "dtype": "float32",
        "metric_type": "l2",
        "index_param": {
          "base_quantization_type": ["fp32", "sq8_uniform"],
          "max_degree": [16, 32, 48],
          "ef_construction": {
            "$range": {
              "start": 100,
              "stop": 300,
              "step": 100
            }
          }
        }
      },
      "search_params": {
        "hgraph": {
          "ef_search": [40, 80, 120, 160]
        }
      }
    }
  ],
  "constraints": {
    "recall_at_k": 0.95,
    "latency_avg_ms": 2.0
  },
  "execution": {
    "top_k": 10,
    "search_mode": "knn",
    "num_threads_building": 48,
    "num_threads_searching": 48
  }
}
```

### 12.3 已有索引 search-only 调优

```json
{
  "version": 1,
  "data_path": "/data/sift-128-euclidean.hdf5",
  "index_path": "/indexes/sift_hgraph.index",
  "indexes": [
    {
      "name": "hgraph",
      "create_params": {
        "dim": 128,
        "dtype": "float32",
        "metric_type": "l2",
        "index_param": {
          "base_quantization_type": "fp32",
          "max_degree": 32,
          "ef_construction": 300
        }
      },
      "search_params": {
        "hgraph": {
          "ef_search": [40, 80, 120, 160]
        }
      }
    }
  ],
  "constraints": {
    "recall_at_k": 0.95
  },
  "execution": {
    "top_k": 10
  }
}
```

### 12.4 多索引候选

不同索引的参数空间分别写在各自的 `indexes[]` 元素内，不做位置对齐。

```json
{
  "version": 1,
  "data_path": "/data/sift-128-euclidean.hdf5",
  "indexes": [
    {
      "name": "hgraph",
      "create_params": {
        "dim": 128,
        "dtype": "float32",
        "metric_type": "l2",
        "index_param": {
          "base_quantization_type": "fp32",
          "max_degree": [16, 32],
          "ef_construction": [100, 200]
        }
      },
      "search_params": {
        "hgraph": {
          "ef_search": [40, 80, 120]
        }
      }
    },
    {
      "name": "ivf",
      "create_params": {
        "dim": 128,
        "dtype": "float32",
        "metric_type": "l2",
        "index_param": {
          "partition_strategy_type": "ivf",
          "buckets_count": [1024, 2048, 4096],
          "base_quantization_type": "fp32",
          "ivf_train_type": "kmeans"
        }
      },
      "search_params": {
        "ivf": {
          "scan_buckets_count": [16, 32, 64]
        }
      }
    }
  ],
  "constraints": {
    "recall_at_k": 0.95,
    "memory_peak_mb": 8192
  },
  "execution": {
    "top_k": 10
  }
}
```

### 12.5 P5 输出示例

完整 P5 response 示例见：

```text
tools/autotune/examples/sift_hgraph_ivf_autotune_result_p5.json
```

该示例展示：

- `build_count`、`build_group_count`、`trial_count` 的区别。
- `builds[]` 中的 build group 记录。
- `trials[]` 中复用同一 `build_id` 的多个 search trial。
- `metrics` 中 build 指标和 search 指标的合并结果。
- `raw_eval_result.build` 和 `raw_eval_result.search` 的分离。

## 13. 第一阶段实现边界

第一阶段必须完成：

- request validation。
- 参数候选展开。
- 系统默认候选补齐。
- build group 编排。
- search trial 编排。
- build metrics 与 search metrics 合并。
- existing index search-only 编排。
- eval 结果解析。
- 约束过滤和推荐结果选择。
- 完整 JSON result 输出。
- request validation 失败时的结构化 JSON 输出。
- HGraph 和 IVF build/search 主路径验证。

第一阶段可以延后：

- `range`、`knn_filter`、`range_filter` 的完整验收。
- Build Cache 复用。
- `Index::Tune()` 热修改闭环。
- trial 剪枝、successive halving、query sampling。
- 分布式执行。
- 自动索引类型选择。

## 14. 后续优化插入点

本文档的 API 不要求第一阶段实现优化器，但保留以下内部扩展点：

1. 候选生成后、plan 生成前：可以加入非法组合剪枝。
2. build group 执行前：可以查找 build cache，命中后跳过 build。
3. build group 执行后、search trial 前：可以做 build 侧约束剪枝，例如
   `build_seconds` 或 `index_size_mb` 已经超限时跳过该 group 的 search。
4. search trial 内部：可以先用少量 query 评估，再对 finalist 做 full validation。
5. result selection 前：可以增加多目标排序或 Pareto frontier 报告。

这些优化不应改变用户输入结构。它们只改变执行成本和 trial 调度方式。
