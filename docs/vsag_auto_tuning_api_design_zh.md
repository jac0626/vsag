# VSAG 自动调优 API 设计

> 状态：讨论草案。
>
> 本文档专注于自动调优框架的 API 输入输出模型。它不是最终用户手册；功能稳定后，应同步到
> `docs/docs/{en,zh}/src/`。当前实现只覆盖本文档的 P0 子集：HGraph 存量索引的
> `ef_search` 调优。

## 1. 设计目标

自动调优 API 需要同时支持两类入口：

1. **存量索引调优**：用户已经有一个构建好的索引，主要调 search-time 参数，例如
   `ef_search`。
2. **原始数据集调优**：用户提供 base dataset，框架可以构建多个候选索引，用于调 build 参数、
   quantizer 和 search 参数。

因此输入模型不能只围绕 `IndexPtr` 设计，也不能只围绕 raw dataset 设计。推荐使用明确的
discriminated union：

```text
TuningRequest
  +-- source.type = existing_index
  |     +-- index handle or serialized index path
  |
  +-- source.type = raw_dataset
        +-- base dataset or dataset path
```

`source.type` 决定 pipeline 可以做什么。

| Source | 主要用途 | 可调参数 | 注意事项 |
| --- | --- | --- | --- |
| `existing_index` | 存量索引 search 调优、线上回归 | search/runtime | 默认不 rebuild；build/quantizer 仅作元信息。 |
| `raw_dataset` | 完整调优、离线探索、构建新索引 | build、quantizer、search/runtime 参数 | 需要 base dataset；成本更高。 |

同时，VSAG 不只有 HGraph。完整 API 必须把 **索引类型** 作为一等概念，并明确列出支持范围：

- `hgraph`
- `ivf`
- `pyramid`
- `bruteforce`
- `sindi`
- 其他后续显式注册 backend 的索引

不在支持列表内、或没有注册 tuner backend 的索引类型，API 直接返回
`unsupported_index_type`，不实现 tuner。

不同索引类型的参数空间、合法性检查、候选生成策略、可应用方式都不同。API schema 可以统一，但
执行时必须通过 `index_type` 路由到对应的 tuner backend。

## 2. 顶层 Request Schema

推荐长期稳定输入结构：

```json
{
  "version": 1,
  "index_type": "hgraph",

  "source": {},
  "workload": {},
  "config": {},
  "objective": {},
  "search_space": {},
  "evaluation": {},
  "budget": {},
  "output": {}
}
```

字段职责：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `version` | 是 | API schema 版本。 |
| `index_type` | 是 | 索引类型，用于路由到对应 tuner backend，例如 `hgraph`、`ivf`。 |
| `source` | 是 | 存量索引或原始数据集输入。 |
| `workload` | 是 | query、ground truth、topk、metric 等评估 workload。 |
| `config` | 是 | 当前配置或 baseline 配置。 |
| `objective` | 是 | 调优目标和约束。 |
| `search_space` | 是 | 允许调哪些参数、候选值是什么。 |
| `evaluation` | 否 | 评估方式，例如 query sampling、repeat、successive halving。 |
| `budget` | 否 | trial 数、耗时、构建次数等预算限制。 |
| `output` | 否 | 输出报告细节控制。 |

## 3. Index Type 与能力模型

`index_type` 不是普通标签，它决定以下行为：

- 参数命名空间。
- 参数合法性检查。
- 参数应用方式：search patch、`Index::Tune()`、rebuild、reload。
- 默认 search space 生成方式。
- 剪枝规则。
- 推荐策略。

建议核心模块维护一个 registry：

```text
TuningRegistry
  +-- hgraph  -> HGraphTuningBackend
  +-- ivf     -> IvfTuningBackend
  +-- pyramid -> PyramidTuningBackend
  +-- sindi   -> SindiTuningBackend
```

每个 backend 暴露能力声明：

```cpp
struct IndexTuningCapability {
    std::string index_type;
    bool supports_existing_index_search_tuning = false;
    bool supports_raw_dataset_build_tuning = false;
    bool supports_quantizer_tuning = false;
    bool supports_hot_tune = false;
    std::vector<std::string> supported_search_parameters;
    std::vector<std::string> supported_build_parameters;
    std::vector<std::string> supported_quantizer_parameters;
};
```

### 3.1 参数路径规范

推荐所有可调参数都使用带索引类型前缀的路径：

```text
<index_type>.<parameter_name>
```

示例：

```text
hgraph.ef_search
hgraph.factor
hgraph.max_degree
hgraph.ef_construction
hgraph.base_quantization_type

ivf.nprobe
ivf.nlist
ivf.base_quantization_type

pyramid.ef_search
pyramid.subindex_ef_search
pyramid.hierarchies

sindi.n_candidate
sindi.query_prune_ratio
```

这样做有几个好处：

- 同名参数不会混淆，例如多个索引都可能有 `ef_search`。
- search space 可以被统一解析，但由 index backend 决定如何应用。
- 报告里的 candidate 可以跨索引类型比较和存档。

### 3.2 能力矩阵草案

下面是 API 层面的能力草案，不代表当前全部已经实现。

| Index type | search tuning | build tuning | quantizer tuning | existing index | raw dataset | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `hgraph` | 是 | 是 | 是 | 是 | 是 | P0 先实现 existing index + `hgraph.ef_search`。 |
| `ivf` | 是 | 是 | 是 | 是 | 是 | 需要区分聚类/分桶参数和量化参数。 |
| `pyramid` | 是 | 是 | 是 | 是 | 是 | 需要处理层级参数和子索引 search 参数。 |
| `bruteforce` | 有限 | 否 | 是 | 是 | 是 | 主要是量化/重排相关，不调图参数。 |
| `sindi` | 是 | 是 | 不同 | 是 | 是 | 稀疏向量 workload，指标和数据格式不同于 dense top-k。 |

不纳入自动调优 scope 的 index：

| Index type | API 行为 |
| --- | --- |
| `hnsw` | 返回 `unsupported_index_type`，不实现 backend。 |
| `diskann` | 返回 `unsupported_index_type`，不实现 backend。 |
| `sparse` | 返回 `unsupported_index_type`，不实现 backend。 |

### 3.3 Backend 职责

每个 index backend 至少需要实现：

| 职责 | 说明 |
| --- | --- |
| `ValidateSearchSpace` | 判断用户给的参数路径是否属于该索引，候选值是否合法。 |
| `GenerateCandidates` | 从 search space 生成 typed candidates。 |
| `PruneCandidate` | 做静态剪枝，例如范围、参数组合冲突、source 能力不满足。 |
| `ApplyCandidate` | 把 candidate 应用到 search params、`Index::Tune()` 或 rebuild 参数。 |
| `EvaluateCandidate` | 可复用通用 evaluator，也可补充索引特定指标。 |
| `SelectRecommendation` | 使用通用 recommender 或索引特定 tie-breaker。 |

P0 可以先把这些职责折叠在 HGraph `ef_search` tuner 内部；但 API 设计上要给后续 backend 留位置。

## 4. Source 输入

### 4.1 存量索引：`existing_index`

适用于已有索引，默认只调低成本 search/runtime 参数。

```json
{
  "source": {
    "type": "existing_index",
    "index": "<Index handle>",
    "serialized_index_path": null
  }
}
```

C++ 内部 API 形态可以是：

```cpp
enum class TuningSourceType {
    EXISTING_INDEX,
    RAW_DATASET,
};

struct ExistingIndexSource {
    IndexPtr index = nullptr;
    std::string serialized_index_path;
};
```

字段规则：

- `index` 和 `serialized_index_path` 至少提供一个。
- library-level API 优先使用 `IndexPtr`。
- CLI/file API 可以使用 `serialized_index_path`，由工具层负责加载。
- 对 `existing_index`，不要求用户提供 build 参数。

### 4.2 原始数据集：`raw_dataset`

适用于完整离线调优，可以构建多个候选索引。

```json
{
  "source": {
    "type": "raw_dataset",
    "base": "<Dataset>",
    "base_path": null
  }
}
```

C++ 内部 API 形态可以是：

```cpp
struct RawDatasetSource {
    DatasetPtr base = nullptr;
    std::string base_path;
};
```

字段规则：

- `base` 和 `base_path` 至少提供一个。
- library-level API 优先使用 `DatasetPtr`。
- CLI/file API 可以使用 `base_path`，例如 HDF5 或其他数据文件。
- 如果要调 build/quantizer 参数，通常需要 `raw_dataset`。

### 4.3 Source 与 index_type 的一致性

对于 `existing_index`，`index_type` 必须和索引实际类型一致：

```text
request.index_type == source.index->GetIndexType()
```

如果 API 使用字符串，例如 `"hgraph"`，需要有明确映射：

| API `index_type` | `IndexType` |
| --- | --- |
| `hgraph` | `IndexType::HGRAPH` |
| `ivf` | `IndexType::IVF` |
| `pyramid` | `IndexType::PYRAMID` |
| `bruteforce` | `IndexType::BRUTEFORCE` |
| `sindi` | `IndexType::SINDI` |

`IndexType::HNSW`、`IndexType::DISKANN`、`IndexType::SPARSE` 即使仍存在于枚举或兼容代码中，
也不应注册自动调优 backend；调优请求应在 validation 阶段以 `unsupported_index_type`
失败。

如果不一致，应该在 workload validation 阶段失败，不能继续调优。

对于 `raw_dataset`，`index_type` 决定使用哪个 factory/backend 构建候选索引。

### 4.4 为什么不把 build 参数放在 `source` 下

`source` 只描述“数据或索引来自哪里”。build 参数有两种完全不同的语义：

- 存量索引场景：它是 **current config metadata**，描述这个索引当初怎么构建。
- 原始数据集场景：它是 **baseline config**，为未被 search space 覆盖的字段提供默认值。

所以推荐把配置统一放在 `config` 里，而不是放在 `source` 里。

## 5. Config 输入

`config` 描述当前配置或 baseline 配置。

```json
{
  "config": {
    "build_parameters": {},
    "search_parameters": {
      "hgraph": {
        "factor": 2
      }
    }
  }
}
```

字段语义：

| 字段 | existing_index | raw_dataset |
| --- | --- | --- |
| `build_parameters` | 可选。作为 current config metadata。 | 推荐提供。作为 build candidate baseline。 |
| `search_parameters` | 推荐提供。作为 search 参数 baseline。 | 推荐提供。作为 search candidate 的 baseline。 |

注意：

- `search_space` 表示“哪些参数要调”。
- `config` 表示“未调参数从哪里继承”。
- 调优时应只 patch 当前 candidate 的字段，不应覆盖完整参数对象。

例如 `index_type = hgraph` 且调 `hgraph.ef_search` 时，下面的 `factor` 必须保留：

```json
{
  "config": {
    "search_parameters": {
      "hgraph": {
        "factor": 2,
        "use_extra_info_filter": false
      }
    }
  },
  "search_space": {
    "search": {
      "hgraph.ef_search": {
        "values": [40, 80, 120]
      }
    }
  }
}
```

生成 trial 时应得到：

```json
{
  "hgraph": {
    "factor": 2,
    "use_extra_info_filter": false,
    "ef_search": 80
  }
}
```

不同索引类型的 `config.build_parameters` 和 `config.search_parameters` 形状可以不同。API 层不应把
它们拍平成一个通用字段集合，而应保留索引原有 JSON 结构，再通过参数路径 patch：

```json
{
  "config": {
    "build_parameters": {
      "dtype": "float32",
      "metric_type": "l2",
      "dim": 128,
      "index_param": {}
    },
    "search_parameters": {
      "hgraph": {}
    }
  }
}
```

## 6. Workload 输入

`workload` 描述如何评估候选。

```json
{
  "workload": {
    "queries": "<Dataset>",
    "query_path": null,
    "ground_truth": "<Dataset>",
    "ground_truth_path": null,
    "topk": 10,
    "metric_type": "l2",
    "dim": 128
  }
}
```

字段规则：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `queries` / `query_path` | 是 | 查询向量。library-level API 用 `DatasetPtr`，CLI 用 path。 |
| `ground_truth` / `ground_truth_path` | 推荐 | 已计算好的 top-k ground truth。 |
| `topk` | 是 | 评估 recall@k 的 k。 |
| `metric_type` | raw_dataset 推荐 | 如果需要构建索引或计算 ground truth，需要 metric。 |
| `dim` | raw_dataset 推荐 | 如果需要构建索引或校验数据，需要 dim。 |

Ground truth 支持两种模式：

1. 用户直接传入：最快、最明确、推荐。
2. 框架计算：只有在提供 `raw_dataset.base + queries + metric_type + topk` 时可选开启。

Ground truth 自动计算可能很贵，所以建议放在 `evaluation.ground_truth` 下显式控制。

## 7. Objective 输入

`objective` 描述“什么叫好”。

```json
{
  "objective": {
    "primary": "latency",
    "recall_at_k": {
      "min": 0.95
    },
    "constraints": {
      "max_latency_p99_ms": 10.0,
      "max_latency_avg_ms": null,
      "min_qps": null,
      "max_memory_bytes": 17179869184,
      "max_build_seconds": null
    }
  }
}
```

推荐语义：

- 先过滤不满足 hard constraints 的候选。
- 在 feasible candidates 中按 `primary` 选择最优。
- P0 可以先实现 `recall_at_k.min`，并以最小 `ef_search` 作为成本代理。

`primary` 可选值建议：

| 值 | 说明 |
| --- | --- |
| `latency` | 在满足 recall/约束下优先降低 latency。 |
| `qps` | 在满足 recall/约束下优先提升 QPS。 |
| `memory` | 在满足 recall/约束下优先降低 memory。 |
| `build_time` | 在满足 recall/约束下优先降低 build time。 |
| `pareto` | 输出 Pareto frontier，不只给单点推荐。 |

## 8. Search Space 输入

`search_space` 描述允许调哪些参数。

推荐使用带 `index_type` 前缀的参数路径作为 key：

```json
{
  "search_space": {
    "search": {
      "hgraph.ef_search": {
        "values": [20, 40, 80, 120, 200, 400, 800]
      }
    },
    "build": {
      "hgraph.max_degree": {
        "values": [16, 32, 64]
      },
      "hgraph.ef_construction": {
        "values": [100, 200, 400]
      }
    },
    "quantizer": {
      "hgraph.base_quantization_type": {
        "values": ["fp32", "sq8", "sq8_uniform"]
      }
    }
  }
}
```

参数组语义：

| 参数组 | 修改代价 | existing_index 默认支持 | raw_dataset 默认支持 |
| --- | --- | --- | --- |
| `search` | 低 | 是 | 是 |
| `runtime` | 低 | 是 | 是 |
| `quantizer` | 中 | 仅当 `Tune()` 或 raw vector 可用时 | 是 |
| `build` | 高 | 否，除非额外提供 base 并允许 rebuild | 是 |

`values` 是最简单的离散候选。后续可以扩展：

```json
{
  "hgraph.ef_search": {
    "range": {
      "min": 20,
      "max": 1000,
      "scale": "log2"
    }
  }
}
```

P0 建议只支持显式 `values`。

### 8.1 Search space 与 index_type 的关系

API 应要求 `search_space` 中的参数路径必须匹配 `index_type`：

```json
{
  "index_type": "hgraph",
  "search_space": {
    "search": {
      "hgraph.ef_search": {
        "values": [40, 80, 120]
      }
    }
  }
}
```

如果 `index_type = hgraph`，但用户传入：

```json
{
  "search_space": {
    "search": {
      "ivf.nprobe": {
        "values": [8, 16]
      }
    }
  }
}
```

应该在 search-space validation 阶段失败。

如果未来支持联合索引或复合索引，可以扩展为：

```json
{
  "index_type": "composite",
  "components": [
    {
      "name": "dense",
      "index_type": "hgraph"
    },
    {
      "name": "bucket",
      "index_type": "ivf"
    }
  ]
}
```

但第一版不建议支持 composite。

## 9. Evaluation 输入

`evaluation` 描述如何执行 trial。

```json
{
  "evaluation": {
    "query_count": 1000,
    "warmup_query_count": 100,
    "repeat": 1,
    "threads": 1,
    "seed": 42,
    "ground_truth": {
      "mode": "provided"
    },
    "sampling": {
      "enabled": false,
      "method": "prefix",
      "sample_query_count": 100
    },
    "successive_halving": {
      "enabled": false,
      "min_query_count": 100,
      "reduction_factor": 3,
      "final_full_validation": true
    }
  }
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `query_count` | 本次评估使用多少条 query；`0` 表示全部。 |
| `warmup_query_count` | 计时前预热查询数。 |
| `repeat` | 每个 trial 重复次数。 |
| `threads` | 搜索线程数。 |
| `seed` | sampling 或随机策略的 seed。 |
| `ground_truth.mode` | `provided` 或 `compute`。 |
| `sampling` | 单轮 query sampling。 |
| `successive_halving` | 多轮低成本到高成本评估。 |

`sampling` 和 `successive_halving` 不应混为一谈：

- `sampling` 是“这次评估用 query 子集”。
- `successive_halving` 是“多轮评估和淘汰策略”，每轮可以使用不同 query_count。

## 10. Budget 输入

`budget` 描述资源限制。

```json
{
  "budget": {
    "max_trials": 100,
    "max_build_trials": 20,
    "timeout_seconds": 3600,
    "per_trial_timeout_seconds": 300,
    "max_index_memory_bytes": null,
    "working_directory": "/tmp/vsag_tune"
  }
}
```

建议规则：

- 超过 `max_trials` 的 candidate 应标记为 skipped，并说明 budget exceeded。
- 超时 trial 应标记为 failed 或 timeout，不能静默丢弃。
- raw_dataset 模式下应要求 `working_directory` 或由框架创建临时目录。

## 11. Output 控制

`output` 控制报告详细程度。

```json
{
  "output": {
    "include_all_trials": true,
    "include_skipped_trials": true,
    "include_failed_trials": true,
    "include_pareto_frontier": true,
    "top_n": 5
  }
}
```

P0 建议总是输出 all trials，便于调试和复现。

## 12. 完整示例：HGraph 存量索引调 search 参数

```json
{
  "version": 1,
  "index_type": "hgraph",
  "source": {
    "type": "existing_index",
    "index": "<Index handle>"
  },
  "workload": {
    "queries": "<Dataset>",
    "ground_truth": "<Dataset>",
    "topk": 10
  },
  "config": {
    "search_parameters": {
      "hgraph": {
        "factor": 2,
        "use_extra_info_filter": false
      }
    }
  },
  "objective": {
    "primary": "latency",
    "recall_at_k": {
      "min": 0.95
    }
  },
  "search_space": {
    "search": {
      "hgraph.ef_search": {
        "values": [20, 40, 80, 120, 200, 400, 800, 1200]
      }
    }
  },
  "evaluation": {
    "query_count": 1000,
    "successive_halving": {
      "enabled": false
    }
  }
}
```

对 `topk = 10`，HGraph `ef_search` 当前合法上限是：

```text
max(100 * topk, 1000) = 1000
```

因此 `1200` 会被剪枝并记录 reason。其他候选逐个评估，推荐满足 recall 的最小
`ef_search`。

## 13. 完整示例：HGraph 原始数据集调 build + quantizer + search

```json
{
  "version": 1,
  "index_type": "hgraph",
  "source": {
    "type": "raw_dataset",
    "base": "<Dataset>"
  },
  "workload": {
    "queries": "<Dataset>",
    "ground_truth": "<Dataset>",
    "topk": 10,
    "metric_type": "l2",
    "dim": 128
  },
  "config": {
    "build_parameters": {
      "dtype": "float32",
      "metric_type": "l2",
      "dim": 128,
      "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 32,
        "ef_construction": 200,
        "build_thread_count": 16
      }
    },
    "search_parameters": {
      "hgraph": {
        "factor": 2
      }
    }
  },
  "objective": {
    "primary": "latency",
    "recall_at_k": {
      "min": 0.95
    },
    "constraints": {
      "max_memory_bytes": 17179869184
    }
  },
  "search_space": {
    "build": {
      "hgraph.max_degree": {
        "values": [16, 32, 64]
      },
      "hgraph.ef_construction": {
        "values": [100, 200, 400]
      }
    },
    "quantizer": {
      "hgraph.base_quantization_type": {
        "values": ["fp32", "sq8", "sq8_uniform"]
      }
    },
    "search": {
      "hgraph.ef_search": {
        "values": [40, 80, 120, 200, 400, 800]
      }
    }
  },
  "evaluation": {
    "query_count": 1000,
    "successive_halving": {
      "enabled": true,
      "min_query_count": 100,
      "reduction_factor": 3,
      "final_full_validation": true
    }
  },
  "budget": {
    "max_trials": 100,
    "max_build_trials": 20,
    "timeout_seconds": 3600,
    "working_directory": "/tmp/vsag_tune"
  }
}
```

这个模式下，`config.build_parameters` 是 baseline/template。`search_space` 中没有覆盖的构建
字段从 baseline 继承。

## 14. 完整示例：IVF 存量索引调 search 参数

下面示例展示同一 API 如何服务非 HGraph 的长期保留索引。具体参数名称应以对应索引的正式
参数文档为准。

```json
{
  "version": 1,
  "index_type": "ivf",
  "source": {
    "type": "existing_index",
    "index": "<Index handle>"
  },
  "workload": {
    "queries": "<Dataset>",
    "ground_truth": "<Dataset>",
    "topk": 10
  },
  "config": {
    "search_parameters": {
      "ivf": {
        "nprobe": 32
      }
    }
  },
  "objective": {
    "primary": "latency",
    "recall_at_k": {
      "min": 0.95
    }
  },
  "search_space": {
    "search": {
      "ivf.nprobe": {
        "values": [8, 16, 32, 64, 128]
      }
    }
  },
  "evaluation": {
    "query_count": 1000
  }
}
```

这个例子和 HGraph 的差异不是顶层 schema，而是：

- `index_type` 不同。
- `config.search_parameters` 的命名空间不同。
- `search_space` 参数路径不同。
- validator、candidate applier、剪枝规则、推荐 tie-breaker 由 IVF backend 负责。

## 15. Output Report Schema

推荐长期稳定输出结构：

```json
{
  "version": 1,
  "succeeded": true,
  "status": "succeeded",
  "request": {},
  "elapsed_ms": 123.4,
  "stages": [],
  "recommendation": {},
  "best_effort": {},
  "trials": [],
  "pareto_frontier": [],
  "summary": {}
}
```

`request` 是本次 tuning 输入的机器可读摘要，用来支持结果回放、审计和文档化。它不是
`AutoTuningApiContext` 的完整序列化，因此不会包含真实的 `IndexPtr`、`DatasetPtr` 或向量
内容。P0 当前输出的 `request` 形态如下：

```json
{
  "index_type": "hgraph",
  "source": {
    "type": "existing_index"
  },
  "workload": {
    "topk": 10
  },
  "config": {
    "build_parameters": null,
    "search_parameters": {
      "hgraph": {
        "factor": 2
      }
    }
  },
  "objective": {
    "primary": "latency",
    "recall_at_k": {
      "min": 0.95
    }
  },
  "search_space": {
    "search": {
      "hgraph.ef_search": {
        "values": [0, 10, 20, 40, 80]
      }
    }
  },
  "evaluation": {
    "query_count": 100,
    "effective_query_count": 100,
    "successive_halving": {
      "enabled": false
    }
  },
  "budget": {
    "max_trials": 0
  }
}
```

`evaluation.query_count` 是输入中请求的 query 数；`evaluation.effective_query_count` 是 pipeline
实际用于评估的 query 数。`query_count = 0` 表示使用全部 query，因此这两个值可能不同。
`budget.max_trials = 0` 表示不限制 trial 数。

### 15.1 Stage Result

```json
{
  "stage": "candidate_pruning",
  "status": "completed",
  "message": "skipped invalid or budgeted ef_search candidates",
  "input_count": 8,
  "output_count": 7
}
```

Stage status：

| 状态 | 语义 |
| --- | --- |
| `completed` | 阶段正常完成。 |
| `skipped` | 阶段按配置跳过，属于预期行为。 |
| `failed` | 阶段失败，报告应包含原因。 |

### 15.2 Trial Result

```json
{
  "trial_id": 3,
  "candidate": {
    "hgraph.ef_search": 120
  },
  "search_parameters_patch": {
    "hgraph": {
      "ef_search": 120
    }
  },
  "status": "completed",
  "message": "",
  "evaluation": {
    "status": "success",
    "error_message": "",
    "query_count": 100,
    "recall": {
      "average": 0.951,
      "p0": 0.8,
      "p10": 0.9,
      "p30": 0.95,
      "p50": 0.95,
      "p70": 1.0,
      "p90": 1.0
    },
    "latency": {
      "average_ms": 1.2,
      "p50_ms": 1.0,
      "p90_ms": 2.0,
      "p95_ms": 2.4,
      "p99_ms": 3.1
    },
    "qps": 8200,
    "memory_bytes": 8589934592
  }
}
```

Trial status：

| 状态 | 语义 |
| --- | --- |
| `completed` | candidate 已评估完成。 |
| `skipped` | candidate 被剪枝或预算跳过。 |
| `failed` | candidate 执行失败，例如参数非法、构建失败、搜索失败。 |
| `timeout` | 后续可扩展，表示 trial 超时。 |

### 15.3 Recommendation

```json
{
  "recommendation": {
    "candidate": {
      "hgraph.ef_search": 120
    },
    "search_parameters_patch": {
      "hgraph": {
        "ef_search": 120
      }
    },
    "evaluation": {
      "recall": {
        "average": 0.951
      },
      "qps": 8200
    },
    "reason": "smallest ef_search satisfying recall_at_k >= 0.95"
  }
}
```

如果没有候选满足 hard constraints：

- `recommendation` 为空。
- `best_effort` 填充已评估候选中最好的一个。
- P0 当前 `status` 为 `failed`，长期 typed API 可以再细分为 `no_feasible_candidate`。
- `selection` stage 应为 `skipped` 或 `failed`，并说明没有可行候选。

## 16. C++ Typed API 草案

下面是面向核心模块的 C++ typed API 草案。它不一定立即 public expose。

```cpp
enum class TuningSourceType {
    EXISTING_INDEX,
    RAW_DATASET,
};

struct ExistingIndexSource {
    IndexPtr index = nullptr;
    std::string serialized_index_path;
};

struct RawDatasetSource {
    DatasetPtr base = nullptr;
    std::string base_path;
};

struct TuningSource {
    TuningSourceType type = TuningSourceType::EXISTING_INDEX;
    ExistingIndexSource existing_index;
    RawDatasetSource raw_dataset;
};

struct TuningWorkload {
    DatasetPtr queries = nullptr;
    DatasetPtr ground_truth = nullptr;
    std::string query_path;
    std::string ground_truth_path;
    uint64_t topk = 0;
    std::string metric_type;
    uint64_t dim = 0;
};

struct TuningConfig {
    std::string build_parameters;
    std::string search_parameters;
};

struct TuningObjective {
    double min_recall = 0.0;
    std::string primary = "latency";
    std::optional<double> max_latency_p99_ms;
    std::optional<double> min_qps;
    std::optional<uint64_t> max_memory_bytes;
};

struct DiscreteParameterSpace {
    std::string parameter_path;
    std::vector<std::string> values;
};

struct TuningSearchSpace {
    std::vector<DiscreteParameterSpace> build;
    std::vector<DiscreteParameterSpace> quantizer;
    std::vector<DiscreteParameterSpace> search;
};

struct TuningEvaluationOptions {
    uint64_t query_count = 0;
    uint64_t warmup_query_count = 0;
    uint64_t repeat = 1;
    uint64_t threads = 1;
    uint64_t seed = 0;
    bool enable_sampling = false;
    bool enable_successive_halving = false;
};

struct TuningBudget {
    uint64_t max_trials = 0;
    uint64_t max_build_trials = 0;
    uint64_t timeout_seconds = 0;
    uint64_t per_trial_timeout_seconds = 0;
    std::string working_directory;
};

struct TuningRequest {
    uint64_t version = 1;
    std::string index_type = "hgraph";
    TuningSource source;
    TuningWorkload workload;
    TuningConfig config;
    TuningObjective objective;
    TuningSearchSpace search_space;
    TuningEvaluationOptions evaluation;
    TuningBudget budget;
};
```

为了支持多索引类型，建议补充 backend registry：

```cpp
struct IndexTuningBackend {
    virtual IndexTuningCapability
    GetCapability() const = 0;

    virtual TuningValidationResult
    ValidateRequest(const TuningRequest& request) const = 0;

    virtual std::vector<TuningCandidate>
    GenerateCandidates(const TuningRequest& request) const = 0;

    virtual TrialResult
    RunTrial(const TuningRequest& request, const TuningCandidate& candidate) const = 0;
};

class TuningRegistry {
public:
    void
    Register(std::unique_ptr<IndexTuningBackend> backend);

    const IndexTuningBackend&
    Resolve(const std::string& index_type) const;
};
```

## 17. P0 内部契约

当前代码先固定 HGraph P0 内部契约。这个契约不是长期完整 API，也不是 public API；它只覆盖
已经实现并通过 POC 验证的链路：

```text
source.type = existing_index
index_type = hgraph
tunable parameter = hgraph.ef_search
```

P0 明确不在 `Tune()` 内部构建索引。调用方需要先准备好：

```text
IndexPtr hgraph_index
DatasetPtr queries
DatasetPtr ground_truth
topk
target_recall
base_search_parameters
ef_search_candidates
```

`base_search_parameters` 是当前 search 参数模板。P0 tuner 只 patch
`base_search_parameters["hgraph"]["ef_search"]`，其它字段必须保留，例如：

```json
{
  "hgraph": {
    "factor": 2,
    "ef_search": 20
  }
}
```

P0 输出必须包含：

```text
elapsed_ms
stages
trials
recommendation
best_effort
skipped/failed reason
```

`elapsed_ms` 只统计 `AutoTuningPipeline::Tune()` 内部耗时，不包含 POC 外层的数据读取、索引
构建或 ground truth 计算。每个 trial 的 `latency.average_ms` 表示该候选的平均单 query 搜索
延迟，和 `elapsed_ms` 不是同一个指标。

P0 的成功条件：

```text
没有 FAILED stage
存在 recommendation
```

如果所有已评估候选都达不到 `target_recall`，报告保留 `best_effort`，但
`AutoTuningReport::Succeeded()` 返回 `false`。

## 18. 当前实现子集

当前代码中的 `AutoTuningRequest` 是上面长期 API 的 HGraph P0 子集：

```cpp
struct AutoTuningRequest {
    IndexPtr index;
    DatasetPtr queries;
    DatasetPtr ground_truth;
    std::string source_type;
    uint64_t topk;
    uint64_t query_count;
    double target_recall;
    std::string index_name;
    std::string build_parameters;
    std::string base_search_parameters;
    std::vector<uint64_t> ef_search_candidates;
    uint64_t max_trials;
    bool enable_build_parameter_tuning;
    bool enable_quantizer_tuning;
    bool enable_successive_halving;
};
```

等价于长期 schema：

```json
{
  "version": 1,
  "index_type": "hgraph",
  "source": {
    "type": "existing_index",
    "index": "<IndexPtr>"
  },
  "workload": {
    "queries": "<DatasetPtr>",
    "ground_truth": "<DatasetPtr>",
    "topk": 10
  },
  "config": {
    "search_parameters": {}
  },
  "objective": {
    "recall_at_k": {
      "min": 0.95
    }
  },
  "search_space": {
    "search": {
      "hgraph.ef_search": {
        "values": []
      }
    }
  },
  "evaluation": {
    "query_count": 0
  },
  "budget": {
    "max_trials": 0
  }
}
```

当前代码新增了内部 JSON 契约层：

```cpp
AutoTuningApiParseResult
ParseAutoTuningRequestJson(const std::string& request_json,
                           const AutoTuningApiContext& context);

std::string
SerializeAutoTuningReportJson(const AutoTuningReport& report);
```

`AutoTuningApiContext` 承载不能直接从 JSON 反序列化的 C++ 对象：

```cpp
struct AutoTuningApiContext {
    IndexPtr index;
    DatasetPtr base;
    DatasetPtr queries;
    DatasetPtr ground_truth;
};
```

因此 P0 JSON request 只表达语义配置；`source.index`、`source.base`、`workload.queries` 和
`workload.ground_truth` 对应的实际对象仍由调用方通过 `AutoTuningApiContext` 传入。

`source.type = existing_index` 时，parser 直接使用 `context.index`。`source.type = raw_dataset`
时，parser 使用 `context.base` 和 `config.build_parameters` 构建一个 baseline HGraph index，再把
这个 index 交给现有 `AutoTuningPipeline` 调 `hgraph.ef_search`。这个 raw dataset 路径只解决
“从原始数据集生成 baseline index 后调 search 参数”的输入形态，不表示 build/quantizer 参数
搜索已经实现。

P0 JSON parser 当前明确拒绝以下输入：

- `source.type` 不是 `existing_index` 或 `raw_dataset`，返回 `unsupported_source_type`。
- `index_type != hgraph`，返回 `unsupported_index_type`。
- 非 `search_space.search.hgraph.ef_search` 的 search 参数路径，返回 `unsupported_parameter`。
- 非空 `search_space.build` 或 `search_space.quantizer`，返回 `unsupported_search_space`。
- `evaluation.successive_halving.enabled = true`，返回 `unsupported_evaluation_strategy`。
- `evaluation.warmup_query_count`，返回 `unsupported_evaluation_option`。
- `source.type = existing_index` 且 `config.build_parameters` 非空，返回 `unsupported_config`。
- `source.type = raw_dataset` 且缺少 `config.build_parameters`，返回 `missing_field`。
- `objective.primary` 非 `latency`，返回 `unsupported_objective`。
- `objective.constraints`，返回 `unsupported_objective`。
- `budget` 中除 `max_trials` 以外的非空字段，返回 `unsupported_budget`。
- `output`，返回 `unsupported_output`。

P0 已实现：

- `index_type = hgraph` 的特化路径
- `existing_index` + `IndexPtr`
- `raw_dataset` + `DatasetPtr base` + baseline HGraph build
- `queries` + `ground_truth`
- `topk`
- `query_count`
- `target_recall`
- `config.search_parameters`
- `search_space.search.hgraph.ef_search`
- `budget.max_trials`
- stage report
- trial report
- report request summary
- tuning elapsed time
- recommendation
- best effort
- P0 JSON request parser
- P0 JSON report serializer
- 内部 POC target：`hgraph_auto_tuning_poc`

P0 未实现但已有 pipeline 槽位：

- 多索引 backend registry
- IVF、Pyramid、BruteForce、SINDI 等其他索引类型
- raw dataset 场景下的 build 参数调优
- raw dataset 场景下的 quantizer 调优
- successive halving
- timeout、memory、working directory 等预算控制
- Pareto frontier
- public CLI/file request loading

## 19. 设计约束和建议

1. `source` 不应承载 build 参数；它只描述输入来自哪里。
2. `config` 是 baseline/current config；`search_space` 才是调参空间。
3. `index_type` 必须决定 backend，不同索引类型不能共享硬编码参数规则。
4. `search_space` 参数路径必须匹配 `index_type`，除非未来显式支持 composite index。
5. 对存量索引，`config.build_parameters` 只能作为可选 metadata。
6. 对 raw dataset，如果要 build index，必须能得到完整 build parameters。来源可以是
   `config.build_parameters`、默认参数生成器，或二者合并。
7. 所有 skipped/failed candidate 必须进入报告。
8. 最终 recommendation 必须基于真实评估指标，而不是只基于 proxy。
9. API schema 必须版本化，避免后续扩展破坏兼容性。
