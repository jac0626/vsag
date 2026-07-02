# AutoTune 框架设计草案

状态：正式设计草案，已对齐当前 `tools/autotune` 实现和 V1 自动化闭环目标。

本文档描述 VSAG AutoTune 的内部框架、模块边界、执行流程和演进方向。外部输入输出
契约见 [`autotune_api_v1_zh.md`](autotune_api_v1_zh.md)。本文档不是用户 API 文档，
而是后续实现、review 和拆分任务时使用的工程设计基准。

## 1. 背景

VSAG 现有 `tools/eval` 已经具备构建索引、执行搜索、计算召回率、延迟、QPS、内存等
指标的能力。实际调参时，用户通常仍然需要手工完成以下工作：

```text
选择索引类型
  -> 固定一部分经验参数
  -> 枚举 build/search 参数
  -> 生成多份 eval 配置
  -> 逐个执行 build/search
  -> 收集指标
  -> 按 recall、latency、memory、build time 等约束筛选
  -> 选择最终参数
```

AutoTune V1 的定位不是替代 eval，也不是实现智能优化器，而是在 eval 之上提供官方、
结构化、可复现的参数调优编排层。

## 2. 目标与非目标

### 2.1 V1 目标

V1 要完成的闭环是：

```text
AutoTune request
  -> 参数校验
  -> 默认候选补齐
  -> 参数候选展开
  -> build group / search trial 规划
  -> evaluation strategy 执行
  -> build/search 指标合并
  -> 约束过滤
  -> 推荐结果输出
```

V1 的核心判断标准是：

```text
人原来手写 eval 配置和 shell/python 网格搜索做的事，
现在 AutoTune 可以官方、可复现、结构化地完成。
```

具体目标：

- 复用 `tools/eval` 的 build/search 执行和指标计算能力。
- 支持用户用数组、`$range` 和 `$value` 描述参数候选。
- 支持索引 policy 为缺失参数补齐默认候选。
- 支持 HGraph 和 IVF 的基础候选空间。
- 支持按唯一 build candidate 复用同一个 build artifact。
- 支持已有索引上的 search-only 调优。
- 支持按硬约束筛选，并输出 `recommendation` 或 `best_effort`。
- 支持完整 build/trial 报告和结构化失败结果。

### 2.2 V1 非目标

V1 不解决以下问题：

- 不承诺比人工网格搜索更快。
- 不实现机器学习候选生成。
- 不要求 query sampling、successive halving、Hyperband 或 Bayesian optimization 等成本优化。
- 不实现自动选择索引类型。
- 不把 `Index::Tune()` 作为核心路径。
- 不实现跨请求完整 index artifact 复用。
- 不实现分布式执行。

这些能力属于 V2 及之后的成本优化、智能搜索和产品形态演进。

## 3. 产品演进路线

AutoTune 的产品演进分三步：先做可靠自动化，再降低调参成本，最后改变用户创建索引的
产品入口。

### 3.1 V1：参数调优自动化

用户显式给出数据集、索引集合、候选参数和约束。AutoTune 负责展开候选、编排 eval、
合并指标、过滤约束并输出推荐结果。

V1 优先保证：

- 输入语义稳定。
- 输出语义稳定。
- trial 可复现。
- 失败可解释。
- 指标来源可信。

V1 不追求聪明优化器，也不承诺比人工网格搜索更快。

### 3.2 V1.5：索引侧默认候选策略收口

目标：避免 AutoTune 变成第二套索引参数系统。

长期看，参数语义、索引默认值、默认候选建议和明显非法组合过滤应由索引侧维护；
AutoTune 框架只消费统一接口并执行调优流程。

所有权边界：

```text
索引侧：参数语义、索引默认值、默认候选建议、候选合法性
AutoTune：候选展开、执行编排、指标合并、约束过滤、结果选择
```

当前实现已经将 HGraph 和 IVF policy 拆成独立 provider，但这些 provider 仍位于
`tools/autotune` 内部。后续可以进一步迁移到索引侧暴露的接口。

### 3.3 V2：成本感知优化

目标：让自动调参变得更便宜、更快。

V2 可以加入确定性的成本优化：

- query sampling。
- successive halving。
- build-side pruning。
- search-side pruning。
- 完整 index artifact 复用。
- 预算控制，例如最多 trial 数、最多 build 数、最大耗时。
- 失败候选快速跳过。

关键约束：

```text
中间低成本结果只能用于剪枝；
最终 recommendation 必须来自 full query validation。
```

V2 解决“自动但是太慢、太贵”的问题，但不改变 V1 的核心输入结构。

### 3.4 V3：约束驱动建索引

目标：把产品形态从“用户选择参数”升级为“用户声明目标”。

当前用户创建索引时需要理解并填写：

```json
{
  "index_name": "hgraph",
  "index_param": {
    "base_quantization_type": "sq8_uniform",
    "max_degree": 32,
    "ef_construction": 300
  }
}
```

V3 希望用户只声明数据和约束：

```json
{
  "constraints": {
    "recall_at_k": 0.95,
    "latency_avg_ms": 2.0,
    "memory_peak_mb": 8192
  }
}
```

系统根据数据特征和约束选择索引类型、量化策略、build 参数和 search 参数，返回可复现的
建索引方案。最终可以沉淀为约束驱动的创建入口，例如
`CreateIndexWithConstraints(...)`。

## 4. 设计原则

AutoTune 框架遵循以下原则。

### 4.1 eval 是真实性能来源

AutoTune 不重新实现召回、延迟、内存等评估逻辑。真实指标来自 `tools/eval`。

### 4.2 保持 VSAG 参数结构

`create_params` 和 `search_params` 使用 VSAG 现有参数结构。AutoTune 不重新发明参数名。

### 4.3 用户显式输入优先

候选来源的优先级是：

```text
用户显式候选 > 系统建议候选 > index policy 默认候选 > 索引内部默认值
```

index policy 不是参数白名单。用户显式写出的任何参数候选都会由通用展开逻辑处理，
即使该参数不在默认候选空间中。

### 4.4 build 和 search 分离

AutoTune 必须区分 build-scoped 参数和 search-scoped 参数。相同
`index_name + create_params` 的候选应共享同一个 build group。

### 4.5 预测只能影响过程，不能直接决定结果

未来无论是 ML 候选、query sampling 还是 successive halving，都只能用于候选生成、
排序或剪枝。最终推荐必须来自真实 eval 结果。

### 4.6 失败必须结构化

validation、build、search 和 result 写入失败都应返回结构化 JSON，避免用户只能从日志中
理解失败原因。

## 5. 总体架构

当前框架可以抽象为：

```text
AutoTuneRequest
  -> RequestValidator
  -> ExecutionOptions
  -> IndexPolicyRegistry
  -> CandidateGenerator
  -> TrialPlanner
       -> AutoTunePlan(BuildSpec[] + TrialSpec[])
  -> EvaluationStrategy
       -> EvaluationRunner
  -> ConstraintEvaluator
  -> ResultSelector
  -> ReportWriter
```

当前代码中的主要模块如下：

| 模块 | 当前代码 | 职责 |
| --- | --- | --- |
| Public API | `tools/autotune/autotune.h` | 暴露 `RunAutoTune()`。 |
| CLI | `tools/autotune/main.cpp` | 读取 request JSON，调用 `RunAutoTune()`。 |
| Orchestrator | `tools/autotune/autotune.cpp` | 串联 validation、candidate、plan、strategy、selection。 |
| Request | `tools/autotune/autotune_request.cpp` | 校验 request，解析 execution/output。 |
| Planner | `tools/autotune/autotune_planner.cpp` | 展开候选，生成 build group 和 search trial。 |
| Evaluation | `tools/autotune/autotune_evaluation.cpp` | 调用 eval，提取并合并指标，计算约束。 |
| Strategy | `tools/autotune/autotune_strategy.cpp` | 执行 `EvaluationStrategy`，当前支持 `full_grid` 和 `query_sampling`。 |
| Result | `tools/autotune/autotune_result.cpp` | 选择推荐结果，写 JSON。 |
| Policy registry | `tools/autotune/autotune_index_policy.cpp` | 管理 index policy 映射和通用默认补齐。 |
| HGraph policy | `tools/autotune/policies/hgraph_policy.cpp` | HGraph 默认候选和基础校验。 |
| IVF policy | `tools/autotune/policies/ivf_policy.cpp` | IVF 默认候选、固定默认和基础校验。 |

## 6. 核心数据模型

### 6.1 `ExecutionOptions`

由 `execution` 和 `output` 字段解析得到，表示执行控制：

- `top_k`
- `search_mode`
- `search_query_count`
- `num_threads_building`
- `num_threads_searching`
- `workspace_path`
- `keep_intermediate`
- `max_trials`
- `evaluation_strategy`
- `sample_query_count`
- `finalist_count`
- `include_trials`
- `result_path`

这些字段是 benchmark workload 和执行控制的一部分，不是索引参数。

### 6.2 `CandidateSpec`

表示一组完整候选参数：

```text
CandidateSpec {
  index_name
  create_params
  search_params
}
```

`CandidateSpec` 不再包含数组候选或 `$range`。

### 6.3 `BuildSpec`

表示一次 build group：

```text
BuildSpec {
  build_id
  index_name
  index_path
  create_params
  use_existing_index
  cleanup_index_after_build_group
}
```

如果 `use_existing_index = true`，该 build group 不执行 build eval，只表示一个已有索引
artifact。

### 6.4 `TrialSpec`

表示一次 search trial：

```text
TrialSpec {
  trial_id
  build_id
  index_name
  eval_type
  index_path
  create_params
  search_params
}
```

当前 trial 固定表示 search 评估。

### 6.5 `AutoTunePlan`

表示完整执行计划：

```text
AutoTunePlan {
  builds: BuildSpec[]
  trials: TrialSpec[]
}
```

Plan 负责描述“如何执行”，而不是描述“参数空间是什么”。

### 6.6 `EvaluationResult`

表示一个 strategy 的执行结果：

```text
EvaluationResult {
  build_results
  trial_results
  executed_build_count
}
```

`executed_build_count` 不包含已有索引 search-only 场景中的伪 build group。

### 6.7 `IndexTunePolicy`

表示某个索引的 AutoTune 默认候选策略：

```text
IndexTunePolicy {
  name
  default_candidate_params
  fixed_defaults
  validate
}
```

policy 只决定系统默认探索什么，不限制用户显式写出的候选。

## 7. 执行流程

### 7.1 入口

CLI 读取 request JSON 后调用：

```text
vsag::autotune::RunAutoTune(request)
```

`RunAutoTune()` 捕获异常并返回结构化失败结果。

### 7.2 Request validation

`ValidateRequest()` 检查：

- request 必须是 object。
- `version` 当前只能是 `1`。
- `data_path` 必须存在。
- `indexes` 必须是非空数组。
- 每个 index name 必须被 `IndexPolicyRegistry` 支持。
- `constraints` 必须是非空 object。
- constraint name 必须是当前支持的指标。

`ParseExecutionOptions()` 解析执行控制，并校验：

- `top_k > 0`
- `num_threads_building > 0`
- `num_threads_searching > 0`
- `search_mode == "knn"`

### 7.3 默认候选补齐

`ApplyIndexDefaults()` 按 index policy 补齐缺失字段。

补齐规则是“字段缺失才写入”，不会覆盖用户显式输入。例如用户写了
`max_degree = 48`，HGraph policy 不会再把它替换成 `[16, 32]`。

当前 HGraph 默认候选：

```text
create_params/index_param/base_quantization_type -> ["fp32", "sq8_uniform"]
create_params/index_param/max_degree             -> [16, 32]
create_params/index_param/ef_construction        -> [100, 200]
search_params/hgraph/ef_search                   -> [40, 80, 120]
```

当前 IVF 默认候选：

```text
create_params/index_param/base_quantization_type -> ["fp32", "sq8_uniform"]
create_params/index_param/buckets_count          -> [1024, 2048]
search_params/ivf/scan_buckets_count             -> [16, 32, 64]
```

当前 IVF 固定默认：

```text
create_params/index_param/partition_strategy_type -> "ivf"
create_params/index_param/ivf_train_type          -> "kmeans"
```

### 7.4 候选展开

`ExpandJson()` 递归展开参数空间：

- 标量表示固定值。
- 数组表示候选集合。
- `$range` 表示闭区间枚举。
- `$value` 表示真实数组值，不作为候选集合展开。

`GenerateCandidates()` 对 `create_params` 和 `search_params` 做笛卡尔积，生成
`CandidateSpec[]`。

### 7.5 Trial planning

`PlanTrials()` 将 `CandidateSpec[]` 转换为 `AutoTunePlan`。

核心规则：

- build key 是 `index_name + create_params`。
- 相同 build key 只生成一个 `BuildSpec`。
- 每个 candidate 生成一个 `TrialSpec`。
- `TrialSpec` 通过 `build_id` 关联到对应 `BuildSpec`。
- 没有 `index_path` 时，临时 index path 写入 `workspace_path/trials/`。
- 有 `index_path` 时，要求展开后只有一个 build candidate。

这样可以保证同一请求内多个 search 参数候选复用同一个构建产物。

### 7.6 Evaluation strategy

`EvaluationStrategy` 决定一批 trial 如何被评估。

当前实现包含两个策略：

- `FullGridEvaluationStrategy`：默认策略，也是准确性 baseline。
- `QuerySamplingEvaluationStrategy`：V2 初始优化策略，先 sampled eval，再 full validate
  finalists。

`FullGridEvaluationStrategy`：

```text
for each BuildSpec:
  run build once or mark existing index
  for each TrialSpec attached to this build:
    run search
    merge build metrics into search trial result
  cleanup build artifact when needed
```

`QuerySamplingEvaluationStrategy`：

```text
for each BuildSpec:
  run build once or mark existing index
  run every TrialSpec(search) with sample_query_count
select finalists from sampled trial results
run finalists with full queries
select final recommendation only from full validation results
cleanup build artifact when needed
```

这个抽象是后续 successive halving、ML 迭代搜索等能力的插入点。

### 7.7 Evaluation runner

当前 runner 直接复用 eval 的进程内能力。

Build path：

```text
BuildSpec
  -> eval::EvalConfig(action_type = "build")
  -> eval::EvalCase::MakeInstance()
  -> EvalCase::Run()
  -> raw build eval json
  -> build metrics
```

Search path：

```text
TrialSpec
  -> eval::EvalConfig(action_type = "search")
  -> eval::EvalCase::MakeInstance()
  -> EvalCase::Run()
  -> raw search eval json
  -> search metrics
  -> merge(build metrics, search metrics)
```

已有索引 path：

```text
BuildSpec(use_existing_index = true)
  -> skip build eval
  -> use index_path directly
  -> run search trials
```

### 7.8 指标合并

Build metrics：

- `build_seconds`
- `memory_peak_mb`
- `index_size_mb`

Search metrics：

- `recall_at_k`
- `latency_avg_ms`
- `latency_p99_ms`
- `qps`
- `search_seconds`
- `memory_peak_mb`

合并规则：

- search 指标覆盖同名普通字段。
- `memory_peak_mb` 取 build/search 两侧峰值的较大值。
- 使用已有索引时没有 build eval，因此不会产生 `build_seconds`。

### 7.9 约束过滤

约束方向由字段固定：

- `recall_at_k`、`qps` 是下限。
- `latency_avg_ms`、`latency_p99_ms`、`memory_peak_mb`、`build_seconds`、
  `index_size_mb` 是上限。

缺失指标会产生 `missing_metric` violation。

### 7.10 结果选择

`SelectResult()` 的默认规则：

1. 保留成功 trial。
2. 在成功 trial 中保留满足所有约束的 trial。
3. 如果存在满足约束的 trial，选择 `latency_avg_ms` 最低者。
4. 如延迟相同，依次比较 `memory_peak_mb`、`build_seconds`、`trial_id`。
5. 如果没有 trial 满足约束，但存在成功 trial，返回 `no_candidate_satisfied`，
   并给出 `best_effort`。
6. 如果所有 trial 都失败，返回 `failed`。

`best_effort` 默认按 `recall_at_k` 最高、`latency_avg_ms` 最低排序。

## 8. 结果结构

成功或部分成功结果包含：

- `version`
- `status`
- `elapsed_seconds`
- `elapsed_breakdown_seconds`
- `recommendation`
- `best_effort`
- `trial_count`
- `build_count`
- `build_group_count`
- `failure`
- `builds`
- `trials`

当 `output.include_trials = false` 时，可以省略完整 `builds` 和 `trials`。

结构化失败结果至少包含：

- `version`
- `status = "failed"`
- `elapsed_seconds`
- `recommendation = null`
- `best_effort = null`
- `trial_count = 0`
- `build_count = 0`
- `build_group_count = 0`
- `failure.message`

## 9. 后续优化插入点

### 9.1 Candidate source

当前候选来源包括：

- 用户显式候选。
- index policy 默认候选。

后续可以扩展：

- 基于数据规模的数据画像候选。
- 基于历史 tuning report 的候选。
- 基于机器学习模型的候选。
- 基于约束的候选空间收缩。

推荐抽象：

```text
CandidateSource[]
  -> UserCandidateSource
  -> IndexPolicyCandidateSource
  -> DataProfileCandidateSource
  -> MLCandidateSource
  -> HistoryCandidateSource
```

合并规则仍应保持用户显式输入优先。

### 9.2 Candidate ranker / pruner

候选展开后、trial planning 前，可以插入：

- ML ranker。
- 静态约束剪枝。
- index policy 非法组合剪枝。
- dominance pruning。
- `max_trials` 预算截断。

这些优化只能决定“哪些候选优先评估或跳过”，不能直接产生最终 recommendation。

### 9.3 Evaluation strategy

`EvaluationStrategy` 是 V2 成本优化的主插入点。

后续策略示例：

```text
QuerySamplingThenFullValidationStrategy:
  run all trials with sampled queries
  keep finalists
  run finalists with full queries
  select final recommendation from full validation results
```

```text
SuccessiveHalvingStrategy:
  active candidates = all candidates
  for each round:
      run active candidates with this round budget
      keep top candidates
  run final full validation
```

关键不变量：

```text
低成本评估只能用于剪枝；
最终 recommendation 必须来自 full query validation。
```

### 9.4 Artifact store

当前只支持单次请求内的 build artifact 复用。后续可以引入完整 index artifact 级别的
复用：

```text
hash(index_name + create_params + dataset identity) -> index_path
```

该能力只复用完整索引文件，不引入额外的构建中间态假设。

### 9.5 Metrics store

后续可以将每次 tuning 的候选、数据特征、指标和失败原因持久化，用于：

- 复现历史调参。
- 给 ML candidate source 提供训练数据。
- 给 history candidate source 提供经验候选。
- 分析不同索引和参数组合的稳定性。

## 10. V1 验收标准

V1 的验收标准是“自动化闭环可用”，不是“调参成本已经优化”。

必须满足：

1. 可以读取 AutoTune request JSON。
2. 可以展开 HGraph 和 IVF 的 build 参数、量化参数和 search 参数候选。
3. 可以按唯一 build candidate 生成 build group。
4. 可以为每个 candidate 生成 search trial。
5. 可以复用 eval 真实构建索引并搜索。
6. 可以把 build metrics 合并到同 build group 的每个 search trial。
7. 可以解析 eval 指标。
8. 可以按约束筛选并输出推荐结果。
9. 可以输出所有 build/trial 的完整参数、指标、耗时和失败原因。
10. request validation 失败时可以返回结构化 JSON，不启动 eval。
11. 可以规划已有索引上的 search-only trial。
12. 可以拒绝 `index_path` 与多个 build candidates 同时出现的输入。
13. 可以使用真实 `sift-128-euclidean.hdf5` 数据集跑通一个非简单路径 example：
    - 包含 HGraph 和 IVF。
    - 每个索引至少两个 build 参数候选。
    - 每个索引至少两个量化候选。
    - 每个索引至少三个 search 参数候选。
    - 总候选数量不少于 20。
    - 实际 build group 数量小于 search trial 数量。

当前 V1 不要求：

- successive halving。
- 机器学习候选生成。
- 自动剪枝。
- 跨请求完整 index artifact 复用。
- 分布式执行。
- 自动选择索引集合。
- public API 稳定化。

## 11. 当前状态与未决问题

当前实现已经具备 V1 自动化闭环，并实现了 V2 的第一种优化策略
`query_sampling`。后续继续 V2 时，应优先处理以下问题。

### 11.1 与 baseline 的质量对比

`full_grid` 是准确性 baseline。任何优化策略都必须报告：

- 端到端耗时对比。
- evaluation 阶段耗时对比。
- 推荐 trial 是否一致。
- full validation recall 与 baseline recall 的差异。
- full validation latency 与 baseline latency 的差异。
- 是否仍满足 baseline 中声明的硬约束。

### 11.2 多阶段结果结构

当前 `query_sampling` 已使用 `evaluation_stage = sampled/full_validation` 表达两阶段结果。
后续 successive halving 还需要表达 round、budget、晋级关系和最终验证关系。

### 11.3 per-trial index 准备开销

真实 SIFT 对比中，`query_sampling` 能减少 full query validation 的 trial 数量，并且最终
recommendation 来自 full validation；但端到端 wall time 提升很小。原因是当前 search
trial 的 wall time 主要消耗在每个 trial 的 index 准备/加载和 eval 固定开销上，真实
query loop 只占较小部分。

因此 V2 下一步不应只继续调小 `sample_query_count`，而应优先做 search trial 级别的
artifact/loaded-index 复用，让同一个 build group 下的多个 search 参数候选在一次 index
加载后完成评估。否则 query sampling 对 wall time 的收益会被固定开销抵消。

### 11.4 eval query sampling 语义

当前 AutoTune 通过 eval 内部 `query_limit_count` 支持 deterministic query prefix subset。
后续如果需要更强采样语义，应继续明确：

- 是否支持随机 query subset。
- 是否支持 seed。
- 是否支持分层采样。
- 是否由 eval 原生支持，还是由 AutoTune 生成临时 sampled dataset。

无论采用哪种方式，最终 recommendation 都必须经过 full query validation。

### 11.5 index policy 合法性校验

当前 HGraph 和 IVF policy 主要维护默认候选和基础 create 参数校验。后续应逐步补充
明显非法组合过滤，但不能把 policy 变成用户显式候选的白名单。

### 11.5 正式用户文档

当前文档仍位于顶层 `docs/`，属于设计草案。API 稳定后，需要同步到
`docs/docs/{zh,en}/src/` 的用户文档中。
