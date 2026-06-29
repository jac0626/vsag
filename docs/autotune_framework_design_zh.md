# AutoTune 框架设计草案

状态：P6 草案，已对齐当前 `tools/autotune` 实现

本文档描述 VSAG AutoTune 的内部框架。外部输入输出契约见
[`autotune_api_v1_zh.md`](autotune_api_v1_zh.md)。本文档的目标是指导第一阶段实现：
先完成一个能替代人工网格搜索的官方调参执行框架，再逐步加入剪枝、采样和缓存复用。

## 1. 目标

第一阶段要完成的闭环：

```text
AutoTune request
  -> 参数校验
  -> 默认候选补齐
  -> 参数候选展开
  -> build group / search trial 规划
  -> build group 执行
  -> search trial 执行
  -> build/search 指标合并
  -> 约束过滤
  -> 推荐结果输出
```

核心定位：

- AutoTune 不重新实现召回、延迟、内存等评估逻辑。
- AutoTune 复用 `tools/eval` 的 build/search 能力。
- AutoTune 负责参数空间、评估编排、结果选择和报告。
- 第一阶段不追求更快，只追求自动化和可复现。

## 2. 模块分层

```text
AutoTuneRequest
  -> RequestValidator
  -> RequestNormalizer
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

### 2.1 `AutoTuneRequest`

用户原始输入。它可以包含：

- `data_path`
- `index_path`
- `indexes`
- `constraints`
- `execution`
- `output`

这个对象只代表用户意图，不保证完整，也不保证所有默认值都已经补齐。

### 2.2 `RequestValidator`

职责：在进入调优前失败得足够早。

检查内容：

- `version` 是否支持。
- `data_path` 是否存在。
- `indexes` 是否为非空数组。
- `constraints` 是否为非空对象。
- `execution.search_mode` 是否合法。
- 索引名是否被当前 AutoTune 支持。
- 参数候选表达是否合法，例如 `$range.start/stop/step` 是否完整。
- 候选总量是否超过 `execution.max_trials`。

输出：

- 合法 request 进入下一步。
- 非法 request 返回结构化错误，不启动 eval。

### 2.3 `RequestNormalizer`

职责：把用户输入规整成内部稳定结构。

典型工作：

- 补齐 `execution.top_k = 10`。
- 补齐 `execution.search_mode = "knn"`。
- 补齐 `execution.keep_intermediate = false`。
- 补齐 `execution.workspace_path`。
- 把路径标准化成绝对路径。

Normalizer 不展开候选组合，也不补索引参数默认候选。

输出：

```text
NormalizedTuneRequest
```

### 2.4 `IndexPolicyRegistry`

职责：隔离“通用调参框架”和“具体索引知识”。

Registry 维护索引到 policy 的映射：

```text
hgraph -> HGraphTunePolicy
ivf    -> IvfTunePolicy
sindi  -> SindiTunePolicy
```

当前只实现 `hgraph` 和 `ivf` 的 policy；`sindi` 是后续扩展示例。

每个 `IndexTunePolicy` 负责：

- 声明索引是否支持 AutoTune。
- 补齐缺失的默认候选参数。
- 判断参数组合是否明显非法。
- 维护该索引的默认候选空间。
- 声明每个默认候选参数属于 build scope 还是 search scope。

`IndexTunePolicy` 不是参数白名单。用户显式写出的任何参数候选都会由通用展开逻辑处理，
即使该参数不在默认候选空间中。默认候选空间只决定“用户没写时系统自动探索什么”；
用户没写、policy 也没声明的参数继续使用索引内部默认值。

例如 HGraph policy 可以补齐：

```json
{
  "index_param": {
    "base_quantization_type": ["fp32", "sq8_uniform"],
    "max_degree": [16, 32],
    "ef_construction": [100, 200]
  }
}
```

以及：

```json
{
  "hgraph": {
    "ef_search": [40, 80, 120]
  }
}
```

### 2.5 `CandidateGenerator`

职责：把参数空间展开成候选组合。

输入：

```text
NormalizedTuneRequest + IndexTunePolicy
```

工作：

1. 调用 index policy 补齐缺失的默认候选。
2. 展开数组候选。
3. 展开 `$range` 候选。
4. 处理 `$value`，避免真实数组参数被误认为候选集合。
5. 对 create/search 参数做笛卡尔积。
6. 调用 index policy 过滤明显非法组合。

展开阶段不检查候选字段是否属于默认候选空间。policy 只提供默认候选，不限制用户显式候选。

输出：

```text
CandidateSpec[]
```

`CandidateSpec` 是完整参数，不再包含数组候选或 `$range`。

### 2.6 `TrialPlanner`

职责：把候选参数变成 build group 和 search trial。

Candidate 只说明“参数是什么”。Plan 说明“如何执行这批参数”。

典型决策：

- 按 `index_name + create_params` 聚合唯一 build candidate。
- 没有 `index_path`：每个唯一 build candidate 生成一个 `BuildSpec`。
- 有 `index_path` 且只有一个 build candidate：生成一个 `BuildSpec`，标记为
  `use_existing_index = true`。
- 有 `index_path` 但存在多个 build candidates：validation 失败，避免静默 rebuild。
- 每个完整 candidate 生成一个 `TrialSpec`，trial 固定表示 search 评估。
- 生成 trial id。
- 生成 build id。
- 生成 build artifact index path。
- 决定 build group 与 search trial 的执行顺序。

输出：

```text
AutoTunePlan {
  builds: BuildSpec[]
  trials: TrialSpec[]
}
```

当前实现仍然做全量枚举，但会复用同一 build group 的索引产物。后续完整 index
artifact 复用、query sampling、successive halving 和剪枝都从 plan 或 strategy 层插入。

### 2.7 `EvaluationStrategy`

职责：决定一批 trial 如何被评估。

当前策略：

```text
OneShotFullEvaluationStrategy:
  AutoTunePlan
    -> for each BuildSpec:
         run build once or mark existing index
         run each TrialSpec(search) in this build group
         merge build metrics into each search trial result
    -> TrialResult[]
```

后续策略：

```text
QuerySamplingThenFullValidationStrategy:
  all TrialSpec -> sampled eval -> keep finalists -> full eval -> TrialResult[]
```

```text
SuccessiveHalvingStrategy:
  active candidates = all candidates
  for each round:
      run active candidates with this round budget
      keep top candidates
  run final full validation
```

### 2.8 `EvaluationRunner`

职责：执行 build group 和 search trial。

当前实现直接复用 eval 的进程内能力：

```text
BuildSpec
  -> eval::EvalConfig
  -> action_type = "build"
  -> eval::EvalCase::MakeInstance()
  -> EvalCase::Run()
  -> raw build eval json
  -> build metrics
```

```text
TrialSpec(search)
  -> eval::EvalConfig
  -> action_type = "search"
  -> eval::EvalCase::MakeInstance()
  -> EvalCase::Run()
  -> raw search eval json
  -> search metrics
  -> merge(build metrics, search metrics)
```

这样可以最大化复用现有 build/search、召回率、延迟、内存统计逻辑。

输出：

```text
BuildResult[]
TrialResult[]
```

BuildResult 包含：

- `build_id`
- `status`
- `index_name`
- `eval_type`，取值为 `build` 或 `existing_index`
- 完整 `create_params`
- build metrics
- `raw_eval_result`
- `artifacts.index_path`
- `failure`

每个结果包含：

- `trial_id`
- `build_id`
- `status`
- `index_name`
- `eval_type`
- 完整 `create_params`
- 完整 `search_params`
- 合并后的 `metrics`
- `elapsed_seconds`
- `raw_eval_result`
- `failure`

指标合并规则：

- `recall_at_k`、`latency_avg_ms`、`latency_p99_ms`、`qps` 来自 search eval。
- `build_seconds` 来自 build eval。
- `index_size_mb` 来自 build artifact。
- `memory_peak_mb` 取 build/search 两侧峰值内存的较大值。
- 使用已有索引时没有 build eval，因此不会产生 `build_seconds`。

### 2.9 `ConstraintEvaluator`

职责：判断 trial 是否满足用户约束。

约束方向由字段固定：

- `recall_at_k`、`qps` 是下限。
- `latency_avg_ms`、`latency_p99_ms`、`memory_peak_mb`、`build_seconds`、
  `index_size_mb` 是上限。

输出：

```text
EvaluatedTrialResult[]
```

其中每个 trial 标记：

- `satisfied_constraints`
- `violated_constraints`

### 2.10 `ResultSelector`

职责：选择推荐结果。

默认规则：

1. 过滤失败 trial。
2. 过滤不满足约束的 trial。
3. 如果存在满足约束的 trial，选择 `latency_avg_ms` 最低者。
4. 如果 `latency_avg_ms` 相同，依次比较 `memory_peak_mb`、`build_seconds`、trial id。
5. 如果没有 trial 满足约束，返回 `no_candidate_satisfied`，同时给出 `best_effort`。

`best_effort` 默认按 `recall_at_k` 最高、`latency_avg_ms` 最低排序。

### 2.11 `ReportWriter`

职责：输出最终 JSON。

输出内容：

- 顶层状态。
- 端到端耗时。
- 分阶段耗时。
- 推荐参数。
- `best_effort`。
- build group 数量和实际 build 次数。
- 完整 build 列表。
- 完整 trial 列表。
- 每个 trial 的失败原因和中间产物路径。

## 3. Query Sampling 插入方式

Query sampling 是 eval 的评估预算能力，不是新的参数搜索算法。

它的语义是：同一个 search trial 可以只用一部分 query 做低成本评估。

```text
TrialSpec(search) + EvaluationBudget(query_count=1000)
  -> sampled eval
```

当前 eval 的 `search_query_count` 不能表达真正的采样。现有逻辑会至少跑完整 query，
当 `search_query_count` 大于 query 数时还会重复 query。因此后续需要给 eval 增加
稳定采样能力，例如：

```json
{
  "query_sample": {
    "count": 1000,
    "seed": 42
  }
}
```

或者由 AutoTune 生成临时 sampled dataset。无论哪种实现，最终推荐结果必须经过 full
query validation。

## 4. Successive Halving 插入方式

Successive halving 是 AutoTune 的多轮调度策略。

它不需要 eval 理解 `successive halving`。eval 只需要能按指定 query budget 跑一次评估。

执行流程：

```text
active_candidates = all candidates

round 0:
  run active candidates with 500 sampled queries
  keep top 1/4

round 1:
  run active candidates with 2000 sampled queries
  keep top 1/4

round 2:
  run active candidates with full queries
  select final recommendation
```

对应到模块：

```text
CandidateGenerator
  -> TrialPlanner
  -> SuccessiveHalvingStrategy
       -> BuildRunner(needed build groups)
       -> SearchRunner(round 0 budget)
       -> EarlyPruner
       -> SearchRunner(round 1 budget)
       -> EarlyPruner
       -> SearchRunner(full budget)
  -> ResultSelector
```

关键约束：

- 中间轮结果只能用于剪枝。
- 最终 `recommendation` 必须来自 full query validation。
- build group 可以跨轮复用；search budget 逐轮提升。
- 用户输入结构不因为这个优化发生变化。

## 5. 第一阶段验收标准

当前第一阶段实现完成必须满足：

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

当前第一阶段不要求：

- query sampling。
- successive halving。
- 完整 index artifact 复用。
- 分布式执行。
- 自动选择索引集合。
