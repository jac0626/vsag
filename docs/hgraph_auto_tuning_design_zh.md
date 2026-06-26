# HGraph 自动调优框架设计

> 状态：讨论草案。
>
> 本文档描述 VSAG 中 HGraph 自动调优框架的分阶段工程设计。它不是用户手册，而是后续实现和
> 评审时参考的设计文档。功能稳定后，面向用户的使用说明应同步到
> `docs/docs/{en,zh}/src/`。

## 1. 背景

HGraph 的参数大致可以分为几层：

| 参数层 | 代表参数 | 修改代价 | 主要影响 |
| --- | --- | --- | --- |
| 构建配置 | `max_degree`、`ef_construction`、`alpha`、graph type | 高。需要重建图。 | recall 上限、图内存、构建时间、路径质量。 |
| 表示配置 | `base_quantization_type`、`precise_quantization_type` | 中。可能需要重建编码。 | 内存、距离计算成本、recall。 |
| 搜索配置 | `ef_search`、`factor`、`enable_reorder` | 低。query-time 参数。 | recall、latency、QPS。 |
| 运行时环境配置 | prefetch / runtime 参数 | 低。不改变图结构。 | latency、QPS、cache 行为。 |

人工调参通常是下面这个循环：

```text
选择参数 -> 构建或加载索引 -> 运行查询 -> 计算指标 -> 调整参数
```

这个流程成本高、可复现性差，也很难跨数据集比较。自动调优框架的目标是把它变成一个可复现的
工作流：

```text
用户约束 -> 候选生成 -> 真实评测 -> Pareto frontier -> 推荐配置
```

第一版应该优先保证正确性、可复现性和报告清晰度，而不是一开始就追求复杂的搜索策略。

## 2. 目标

- 为 HGraph 提供可重复执行的调优 workflow。
- 从固定 build config 开始，优先调低成本的 search 参数。
- 输出完整 trial 记录，而不是只返回一个推荐值。
- 基于真实 full-chain metrics 选择推荐配置。
- 支持从 `ef_search` 调优逐步扩展到 representation 调优和少量 build candidates。
- 让核心逻辑能够被 CLI 工具和未来 library API 复用。

## 3. 非目标

初始实现不追求：

- 完全替代人工调参。
- 搜索完整 HGraph 参数空间。
- 提供 learned tuner、Bayesian optimizer 或复杂预测模型。
- 承诺所有 representation 参数都可以通过 `Index::Tune()` 热切换。
- 把现有 `Index::Tune()` 改造成大而全的自动调优 API。
- 自动为所有 workload 生成 ground truth。
- 保证 sampled-query 结果与 full-query 结果完全一致。

## 4. VSAG 现有能力与注意点

### 4.1 可复用能力

- HGraph 支持 query-time `ef_search`，搜索参数形如：

  ```json
  {"hgraph": {"ef_search": 100}}
  ```

- `tools/eval/eval_performance` 已经可以评测 recall、recall 分位、QPS、latency 分位、
  memory 和 build time。
- HGraph 内部已有 ELP optimizer，用小规模 grid search 调整 runtime prefetch 参数。
- HGraph 有 `Tune()` 实现，可以在部分场景下基于 raw vector 重建向量编码。
- HGraph 暴露了 memory estimate 和 memory usage 相关接口。

### 4.2 必须明确的限制

- 当前 `Index::Tune()` 返回 `expected<bool, Error>`，不能直接承载调优曲线、Pareto frontier
  或完整报告。
- 当前 HGraph `Tune()` 需要 raw vector 可用；如果索引没有 raw vector 来源，调优会返回
  `false`。
- 当前 HGraph `Tune()` 主要按 quantizer name 判断是否需要重建。PQ、RaBitQ 等同一
  quantizer 内部子参数变化，未必会触发重编码，除非后续扩展实现。
- `eval_performance` 位于 `tools/eval/`，依赖 HDF5、YAML 等工具层组件。主库不应直接依赖
  这些工具内部实现。
- ELP optimizer 可以作为 grid-search 先例，但它不是 target-recall tuner。它不消费用户
  query set、ground truth 或 target recall。

这些限制不是 blocker，但会影响第一版的设计边界。

## 5. 设计原则

1. 最终推荐必须由 full-chain metrics 决定。
2. proxy metrics 可以用于剪枝，但不能作为最终推荐依据。
3. 第一版应尽量简单、确定、可复现。
4. 每个 skipped 或 failed candidate 都必须记录原因。
5. 调优结果应能通过输入配置和输出报告复现。
6. representation tuning 必须区分可热切换参数和需要 rebuild/reload 的参数。
7. CLI 和 library-level 代码应共享核心数据模型。

## 6. 总体架构

```text
TuningRequest
  |
  v
CandidateGenerator
  |
  v
TrialRunner
  |
  +--> CandidateApplier
  |      - 应用 search 参数
  |      - 在支持时通过 HGraph Tune 应用 representation 参数
  |      - 后续阶段可执行 rebuild
  |
  +--> Evaluator
         - 执行搜索
         - 计算 recall
         - 计算 latency 和 QPS
         - 采集 memory
  |
  v
ResultStore
  |
  v
Recommender
  |
  +--> 约束过滤
  +--> Pareto frontier 生成
  +--> best feasible 选择
  |
  v
TuningReport
```

## 7. 模块边界建议

### 7.1 核心调优模块

建议位置：

```text
src/tuning/
```

核心模块应避免依赖 HDF5、YAML、命令行解析和工具层库。它面向内存对象工作：

- `IndexPtr`
- query `DatasetPtr`
- ground-truth `DatasetPtr`
- candidate 定义
- target constraints

主要组件：

| 组件 | 职责 |
| --- | --- |
| `TuningRequest` | 用户目标、数据引用、候选空间、预算和评测选项。 |
| `CandidateGenerator` | 生成 search、representation，以及后续 build candidates。 |
| `CandidateApplier` | 应用一个 candidate，或返回不能应用的原因。 |
| `Evaluator` | 执行查询并计算指标。 |
| `TrialRunner` | 管理 candidate 执行、错误、超时和缓存。 |
| `ParetoFrontier` | 计算非支配候选集合。 |
| `Recommender` | 根据用户约束选择推荐配置。 |
| `TuningReport` | 稳定的输出数据模型。 |

### 7.2 CLI / 工具模块

建议位置可以二选一：

```text
tools/tune/
```

或放在现有评测工具附近：

```text
tools/eval/tune/
```

CLI 模块负责：

- HDF5 dataset 加载。
- YAML / JSON 配置解析。
- 文件输出。
- 可选 progress 展示。
- 可选复用 `eval_performance` 中已有组件。

CLI 应调用核心调优模块，不应重复实现调优算法。

## 8. 请求模型

> API 输入输出的完整设计见
> [`vsag_auto_tuning_api_design_zh.md`](vsag_auto_tuning_api_design_zh.md)。本节只保留早期
> CLI 配置草图，后续讨论以 API 设计文档为准。

未来 CLI 配置可以类似：

```yaml
index_name: hgraph
dataset:
  path: /tmp/sift-128-euclidean.hdf5
  query_count: 1000
  topk: 10

build:
  create_params: >
    {"dim":128,"dtype":"float32","metric_type":"l2",
     "index_param":{"base_quantization_type":"fp32","max_degree":32,"ef_construction":300}}
  index_path: /tmp/vsag_tune/hgraph.index
  reuse_existing_index: true

targets:
  recall_at_k:
    min: 0.95
  qps:
    min: 5000
  latency_p95_ms:
    max: 5.0
  memory_bytes:
    max: 17179869184

search_space:
  ef_search: [50, 100, 200, 400, 800, 1000]

evaluation:
  warmup_query_count: 100
  repeat: 3
  threads: 16
```

核心模块不应依赖这个 YAML 形状。YAML 只是工具层格式；核心应使用 typed request object 或稳定的
内部 JSON-like schema。

## 9. 报告模型

报告格式应从 P0 开始保持稳定。

```json
{
  "version": 1,
  "status": "success",
  "input_summary": {
    "index_name": "hgraph",
    "topk": 10,
    "query_count": 1000,
    "targets": {
      "recall_at_k_min": 0.95,
      "qps_min": 5000,
      "latency_p95_ms_max": 5.0,
      "memory_bytes_max": 17179869184
    }
  },
  "recommendation": {
    "reason": "minimal ef_search satisfying all hard constraints",
    "profile": {
      "search_config": {
        "ef_search": 200
      }
    },
    "metrics": {
      "recall_avg": 0.953,
      "qps": 5400,
      "latency_p95_ms": 4.7,
      "memory_bytes": 15676630630
    }
  },
  "best_effort": {
    "profile": {},
    "metrics": {},
    "reason": "used when no candidate satisfies all constraints"
  },
  "pareto_frontier": [],
  "trials": [
    {
      "trial_id": 1,
      "candidate": {
        "search_config": {
          "ef_search": 50
        }
      },
      "status": "completed",
      "metrics": {
        "recall_avg": 0.82,
        "recall_detail": {
          "p10": 0.7,
          "p50": 0.85,
          "p90": 0.93
        },
        "qps": 12000,
        "latency_avg_ms": 1.0,
        "latency_detail_ms": {
          "p50": 0.8,
          "p90": 1.1,
          "p95": 1.2,
          "p99": 1.5
        },
        "memory_bytes": 15676630630
      }
    }
  ],
  "skipped": [
    {
      "candidate": {},
      "reason": "requires raw vector but index does not have raw vector storage"
    }
  ],
  "environment": {
    "vsag_version": "",
    "hardware": "",
    "threads": 16
  }
}
```

## 10. 阶段规划

### 10.1 P0：`ef_search` 自动调优

范围：

- 固定 build config。
- 固定 representation config。
- 只调 `ef_search`。
- 使用真实 query execution 和 ground-truth recall。

执行流程：

```text
加载或构建索引
for ef in candidates:
    使用 {"hgraph": {"ef_search": ef}} 跑 query set
    记录 recall、QPS、latency、memory
选择满足 hard constraints 的最小 ef
输出完整报告
```

默认候选：

```text
[50, 100, 200, 400, 800, 1000]
```

这里选择 `1000` 作为保守默认值，是因为 HGraph 当前对 `ef_search` 有 topK 相关的上限校验。
更大的值可以支持，但必须先通过 index validation。

P0 验收标准：

- 给定 target recall，tuner 能找到满足目标的最小合法 `ef_search`。
- 如果没有 candidate 满足目标，tuner 输出 best measured candidate，并说明没有 feasible
  candidate。
- 输出包含所有 trial metrics，而不是只有最终推荐。
- 同一输入配置可重复执行，输出结果可比较。

当前内部 POC target：

```bash
cmake -B build \
  -DENABLE_EXAMPLES=ON \
  -DENABLE_TOOLS=ON \
  -DENABLE_INTEL_MKL=ON \
  -DENABLE_TESTS=ON \
  -DENABLE_MOCKIMPL=OFF
cmake --build build --target hgraph_auto_tuning_poc --parallel 96

# synthetic 小数据
./build/examples/cpp/hgraph_auto_tuning_poc

# SIFT128 真实数据子集
./build/examples/cpp/hgraph_auto_tuning_poc \
  --dataset /root/data/sift-128-euclidean.hdf5 \
  --base-count 10000 \
  --query-count 100 \
  --target-recall 0.90

# 同时写出机器可读 JSON report
./build/examples/cpp/hgraph_auto_tuning_poc \
  --json-output /tmp/hgraph_auto_tuning_report.json

# 通过 raw_dataset API 语义构建 baseline HGraph 后调 ef_search
./build/examples/cpp/hgraph_auto_tuning_poc \
  --source-type raw_dataset \
  --json-output /tmp/hgraph_auto_tuning_raw_report.json

# 只评估前 2 个合法 ef_search candidate，剩余合法 candidate 标记为 budget exceeded
./build/examples/cpp/hgraph_auto_tuning_poc \
  --max-trials 2 \
  --json-output /tmp/hgraph_auto_tuning_budget_report.json
```

该 POC 位于 `src/tuning/hgraph_auto_tuning_poc.cpp`。默认使用小型内存数据集；传入
`--dataset` 时从 HDF5 读取真实 dense float32 数据，例如 SIFT128。POC 会构建 HGraph，
在当前 base 子集内直接计算精确 L2 ground truth，然后调用 `AutoTuningPipeline` 输出输入摘要、
准备阶段耗时、tuning 总耗时、stage report、trial report 和 `ef_search` recommendation。
传入 `--json-output` 时，POC 会把 `AutoTuningReport` 的稳定 JSON 表达写到指定文件；这样可以
避开启动日志对 stdout 的影响，方便脚本解析。JSON report 里包含 `request` 摘要块，会回显
`index_type`、`source.type`、`topk`、目标 recall、`ef_search` 搜索空间、实际 query 数和预算，
但不会序列化真实的 `IndexPtr`、`DatasetPtr` 或向量内容。
`--source-type existing_index` 是默认值，表示 POC 在调用 tuning 前先构建存量索引；
`--source-type raw_dataset` 表示通过内部 JSON API 层用 `config.build_parameters` 构建 baseline
HGraph，然后复用同一条 `ef_search` tuning pipeline。
`--max-trials` 对应 `budget.max_trials`，只限制合法 candidate 的实际评估次数；静态非法 candidate
仍会被跳过并保留原始原因。
等 tuning API 迁到 public header 后，再移动到正式 `examples/cpp/` 示例。

当前 SIFT128 POC 验证结果：

```text
source = /root/data/sift-128-euclidean.hdf5
HDF5 train = (1000000, 128) float32
HDF5 test = (10000, 128) float32
base_count = 10000
query_count = 100
topk = 10
target_recall = 0.90
base_search_parameters = {"hgraph":{"factor":2}}
ef_search_candidates = {0, 10, 20, 40, 80, 160, 320, 1201}

load_elapsed_ms = 15.9445
build_index_elapsed_ms = 12167.7
ground_truth_elapsed_ms = 524.287
tuning_elapsed_ms = 513.664

ef_search = 0    -> skipped, ef_search must be greater than 0
ef_search = 10   -> recall = 0.884, latency_avg_ms = 0.286231, qps = 3493.68
ef_search = 20   -> recall = 0.959, latency_avg_ms = 0.348663, qps = 2868.1
ef_search = 40   -> recall = 0.985, latency_avg_ms = 0.474116, qps = 2109.19
ef_search = 80   -> recall = 0.997, latency_avg_ms = 0.720731, qps = 1387.48
ef_search = 160  -> recall = 0.999, latency_avg_ms = 1.17306, qps = 852.47
ef_search = 320  -> recall = 0.999, latency_avg_ms = 2.04497, qps = 489.005
ef_search = 1201 -> skipped, ef_search must be no greater than 1000

recommendation = ef_search 20
```

这个验证仍然是 `existing_index` 语义：`build_index_elapsed_ms` 是 POC 外层为制造输入索引而记录
的准备阶段耗时，不计入 `AutoTuningPipeline::Tune()` 的 `tuning_elapsed_ms`。

### 10.2 P1：representation + `ef_search` 联合调优

范围：

- 固定 build config。
- 生成有限 representation candidates。
- 对每个 representation candidate 扫 `ef_search`。

候选分类：

| 类别 | 示例 | 初始支持建议 |
| --- | --- | --- |
| Query-time search | `ef_search`、`factor`、`enable_reorder` | P0/P1 支持。 |
| Quantizer type switch | `base_quantization_type`、`precise_quantization_type` | 需要 raw vector。 |
| 同 quantizer 子参数 | `base_pq_dim`、RaBitQ bits、PCA/FHT 参数 | 当前标记为 future 或 rebuild-required。 |
| IO / storage 参数 | `base_io_type`、`precise_io_type`、file paths | future。 |

关键规则：

```text
如果底层 index 没有实际应用该 candidate，不能静默认为调优成功。
```

P1 执行流程：

```text
for representation_candidate in representation_candidates:
    应用 representation candidate
    if candidate 不能应用:
        记录 skipped reason
        continue
    在该 representation 上执行 P0 ef_search tuning
合并所有 trials
计算 Pareto frontier
推荐 best feasible profile
```

P1 验收标准：

- 报告清楚区分 completed、failed、skipped candidates。
- 不宣称支持 HGraph 实际没有应用的 representation 变化。
- 最终推荐来自真实 recall、QPS、latency、memory 指标。

### 10.3 P2：有限 build candidates

范围：

- 加入少量 build candidates。
- 每个 build candidate 都对应一次 fresh index build。
- 每个 build candidate 内部执行 P1。

示例：

```text
max_degree: [16, 32, 48]
ef_construction: [400]
alpha: [1.2]
```

P2 验收标准：

- 报告包含每个 build candidate 的 build time 和 build memory。
- 最终 Pareto frontier 可以跨 build config 比较。
- build 失败应作为 candidate failure 记录，而不是导致整个 tuner 失败。

## 11. 候选生成

候选生成应同时支持用户显式配置和 rule-based 默认空间。

P0 默认：

```text
ef_search: [50, 100, 200, 400, 800, 1000]
```

P1 保守默认：

```text
base_quantization_type: [fp32, fp16, bf16, sq8, sq4]
use_reorder: [false, true]
precise_quantization_type when use_reorder=true: [fp32, fp16]
ef_search: [50, 100, 200, 400, 800, 1000]
```

这只是第一版建议候选集，不等于完整 HGraph 参数空间。HGraph 还支持 `sq8_uniform`、
`sq4_uniform`、`pq`、`pqfs`、`rabitq`、`tq` 等量化器。它们应在 Tune 语义明确后再逐步加入。

## 12. 指标语义

初始指标：

| 指标 | 含义 |
| --- | --- |
| `recall_avg` | query set 上的平均 recall。 |
| `recall_detail` | query-level recall 分位，例如 `p10`、`p50`、`p90`。 |
| `qps` | measured queries 上的每秒查询数。 |
| `latency_avg_ms` | 平均单 query latency。 |
| `latency_detail_ms` | latency 分位，例如 `p50`、`p90`、`p95`、`p99`。 |
| `memory_bytes` | index memory usage 或 peak memory，取决于 evaluator 模式。 |
| `build_time_s` | build candidate 的构建时间。 |
| `tuning_elapsed_ms` | 调优流程总 wall-clock 时间，从进入 tuner 到生成报告。 |

报告必须明确 memory 表示的是当前 index memory、estimated memory，还是 peak process memory。

## 13. Pareto Frontier 与推荐策略

Hard constraints：

- recall 不低于目标值。
- memory 不超过预算。
- 如果设置 QPS 目标，则 QPS 不低于目标值。
- 如果设置 latency 目标，则 latency 不超过目标值。

初始推荐策略应保持确定性：

1. 先按 hard constraints 过滤候选。
2. 如果存在 feasible candidates，选择 primary cost 最小的候选。
3. P0 的 primary cost 是 `ef_search`。
4. P1 的 primary cost 可以由用户目标决定，例如 latency 或 memory。
5. 如果没有 feasible candidate，返回 `best_effort` 并说明原因。

Pareto 维度：

```text
maximize recall
maximize QPS
minimize latency
minimize memory
minimize build time when build candidates are present
```

## 14. 失败与跳过语义

每个 candidate 应进入下列状态之一：

| 状态 | 含义 |
| --- | --- |
| `completed` | trial 正常运行并产生指标。 |
| `skipped` | candidate 因已知不支持或非法而未运行。 |
| `failed` | trial 被尝试执行，但返回错误。 |
| `timeout` | trial 超过配置预算。 |
| `cached` | 复用了此前相同 candidate 的结果。 |

典型 skipped reason：

- `ef_search` 超出 index validation 范围。
- representation candidate 需要 raw vector，但当前索引没有 raw vector。
- candidate 修改了当前 HGraph `Tune()` 无法检测的同 quantizer 子参数。
- candidate 需要 rebuild，但当前阶段不允许 rebuild。

## 15. 降成本路线

降成本应该在 P0 正确性稳定后再推进。

| 技术 | 阶段 | 说明 |
| --- | --- | --- |
| Query sampling | P0/P1 | 先用少量 queries 评估，再对 finalist 做 full query validation。 |
| Trial cache | P0/P1 | 避免重复评测相同 candidate。 |
| Memory estimate pruning | P1/P2 | 对 hard memory budget 有用，但 estimate 需要足够保守。 |
| Successive halving | P1/P2 | 先低成本评估大量 candidates，再逐轮提升精度。 |
| Recall-ef 曲线拟合 | 后续 | 需要实验验证后再启用。 |
| Build candidate pruning | P2 | 用 fp32 large-ef recall ceiling 或 memory estimate 剪掉明显无望的 build。 |

核心原则：

```text
Proxy metrics prune. Full-chain metrics decide.
```

## 16. 测试策略

### 16.1 单元测试

- Candidate generation。
- Constraint filtering。
- Pareto frontier generation。
- Recommendation selection。
- JSON report serialization。
- skipped / failed 状态处理。

### 16.2 集成测试

- 在小型 deterministic HGraph dataset 上执行 P0 tuning。
- target recall 可达。
- target recall 不可达。
- 非法 `ef_search` candidate 能以清晰原因 skipped 或 failed。
- 没有 raw vector 时，需要 raw vector 的 representation candidate 被 skipped。

### 16.3 性能测试

- 验证小候选集下 tuning overhead 可控。
- 比较多次运行结果稳定性。
- 在至少一个代表性 dataset 上验证 sampling error，再决定是否默认启用 sampling。

## 17. 文档计划

开发阶段：

- 保留本文档作为工程设计文档。
- 如果新增 CLI，则增加本地工具说明。

功能稳定后：

- 在 `docs/docs/{en,zh}/src/` 下增加中英文用户文档。
- 明确记录已支持参数和不支持 candidates。
- 如果存在 public workflow，增加 `examples/cpp/` 或工具示例。

## 18. 开放问题

1. 第一版 CLI 应放在 `tools/tune/`，还是集成到 `tools/eval/`？
2. 可复用核心应直接放在 `src/tuning/`，还是 P0 先完全放在 `tools/`，稳定后再下沉？
3. 如果未来提供 public API，第一版 API 应该长什么样？
4. P1 是否强制要求 `store_raw_vector: true`，还是 tuner 自动检测并复用已有 fp32 base/precise
   storage？
5. tuner 如何验证 representation candidate 真的改变了 index 状态？
6. `factor` 是否应进入 P0，还是延后到 P1 与 reorder 相关候选一起处理？
7. 多个 feasible candidates 同时存在时，默认目标是什么：最低 latency、最低 memory、最小
   `ef_search`，还是最高 QPS？
8. 启用 sampling 后，final candidates 是否必须在完整 query set 上复评？
9. `eval_performance` 应直接复用多少，哪些部分需要在 core evaluator 中重新实现？
10. 报告如何记录机器和环境元数据，才能让不同运行结果可比较？

## 19. 建议 PR 拆分

1. 仅提交设计文档。
2. P0 CLI 与 `ef_search` tuning 核心数据模型。
3. 抽出可复用 evaluator 与 report schema。
4. 增加 Pareto frontier 和 recommendation 逻辑。
5. 增加 representation candidates，并做显式支持检查。
6. 增加 query sampling 和 trial cache。
7. 增加有限 build candidates。
8. workflow 稳定后补充用户文档和 examples。

## 20. 总结

推荐路径：

```text
P0: 固定 build + 固定 representation + ef_search tuning
P1: 固定 build + supported representation candidates + ef_search tuning
P2: limited build candidates + 每个 build candidate 内跑 P1
后续: proxy pruning、sampling、successive halving、adaptive search
```

这个路径能让第一版实现足够小、足够可靠，同时保留继续演进为完整自动调优框架的空间。
