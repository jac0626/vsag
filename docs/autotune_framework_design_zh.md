# AutoTune 框架设计草案

状态：V1 实现评审稿

本文档描述 VSAG AutoTune 的内部职责、数据模型、执行基线和 V2/V3 演进缝。外部 JSON
契约以 [autotune_api_v1_zh.md](autotune_api_v1_zh.md) 为准。

V1 不是智能优化器。它先建立一个正确、隔离、可复现的完整评估基线：

~~~text
request validation
  -> local candidate generation
  -> candidate expansion
  -> build grouping
  -> complete trial planning
  -> build once
  -> reload artifact for every search trial
  -> evaluate the full dataset query set
  -> constraint evaluation
  -> result selection
  -> report
~~~

## 1. 目标和边界

### 1.1 V1 目标

- 支持 HGraph 和 IVF。
- 支持构建、量化和搜索参数候选。
- 用户可以显式给出完整候选，也可以省略一部分，由 CandidateGenerator 本地规则补齐。
- 同一个 index_name + create_params 只 build 一次。
- 一个请求只描述一个 KNN workload。
- constraints 同时作用于完整候选的 build 和 search metrics。
- 一个 objective 直接选择完整 create/search recommendation。
- 每个正式 search trial 使用完整 query 集，不做 sampling。
- 已有索引可以执行 search-only。
- 所有失败、参数、指标、耗时和 artifact 关系都进入结构化报告。

### 1.2 V1 非目标

- 不支持 range、filter、hybrid 或 query-aware adaptive search。
- 不自动增加用户未声明的索引类型。
- 不承诺比完整网格更快。
- 不使用 query sampling、successive halving、Bayesian optimization 或 ML 推荐。
- 不复用同一个已加载 index 对象执行多个正式 trial。
- 不做跨请求 artifact cache、分布式执行或 Index::Tune() 编排。
- 不在一个请求内调优多个 workload；多场景由调用方提交独立请求。

这些边界建立了后续优化可比较的 baseline，不限制 V2/V3 的能力。

## 2. 核心不变量

### 2.1 用户输入所有权

用户显式固定的索引、参数和值域不能被默认规则或模型覆盖。候选补全的优先关系是：

~~~text
user explicit values
  > CandidateGenerator rules for still-missing tunable fields
~~~

CandidateGenerator 只能为缺失字段生成 patch；尝试覆盖已有叶子值必须在评估前失败。V1 的
CandidateGenerator 是 AutoTune 内部独立组件，使用本地规则，不通过运行时接口向具体索引查询。
生成器没有生成、用户没有提供且索引原生允许省略的字段保持缺失，由具体索引使用原生默认值；
AutoTune 不复制另一份索引默认参数。dataset 元数据由请求规范化流程绑定，不属于
CandidateGenerator 或索引默认值。

### 2.2 参数知识边界

具体索引负责 parameter schema、原生参数解析、原生默认值和实际 build/search 语义。AutoTune 的
CandidateGenerator 只维护缺失字段的 patch 生成规则，不复制完整 schema，也不预判用户候选是否合法。
框架其余部分负责 request 校验、patch 合并、展开、预算、分组、执行、约束和选择。

### 2.3 build 与 search 分离

create_params 决定 build identity，search_params 不进入 build key。相同 build identity
共享一个序列化 artifact，但 V1 的每个 search trial 都从该 artifact 创建新的内存实例。

### 2.4 workload 与索引参数分离

top_k 和 concurrency 描述用户执行场景，不是索引参数。ef_search、
scan_buckets_count 等仍属于 indexes[].search_params 候选。

### 2.5 最终结果来自真实完整评估

V1 只接受完整 query 结果。未来预测、采样和剪枝可以改变评估顺序与成本，但不能直接成为
recommendation 的证据。

### 2.6 失败结构化

validation、candidate generation、build、load、search、selection 和 report write 中通过异常
返回给 AutoTune 的失败都带 stage、code 和 message。底层 eval 仍有 worker 直接终止进程的
历史行为；这属于 eval 的待修问题，不在 AutoTune 中复制 search loop。

## 3. 总体架构

~~~text
+-------------------------------+
| Request                       |
|                               |
| Data / Existing Index         |
| Workload                      |
| Index Specs                   |
| Constraints                   |
| Objective                     |
| Tuning Config / Output        |
+---------------+---------------+
                |
                v
+------------------------------------------------+
| AutoTune                                       |
|                                                |
| Request Validation                             |
| IndexTuningDescriptor / CandidateGenerator     |
| Trial Planning                                 |
| Evaluation Runner                              |
| Constraint Evaluation / Result Selection       |
| Report Writer                                  |
+---------------+----------------+---------------+
                |                |
                |                | recommendation / evidence
                |                v
                |       +-------------------------------+
                |       | Result                        |
                |       |                               |
                |       | CLI Summary                   |
                |       | Compact Full Report           |
                |       | Complete Recommendation       |
                |       | Artifact Evidence             |
                |       +-------------------------------+
                |
                | build / reload / search
                v
+-------------------------------+
| eval                          |
|                               |
| build / serialize             |
| deserialize / KNN             |
| one-pass metric collection    |
+---------------+---------------+
                |
                | invoke
                v
+-------------------------------+
| Concrete Index                |
|                               |
| HGraph / IVF                  |
+-------------------------------+
~~~

图中是逻辑职责，不要求每个框在实现中对应一个独立类。

## 4. 外部请求模型

V1 请求包含：

~~~text
AutoTuneRequest {
  version
  data_path
  optional index_path
  indexes[]
  workload {
    top_k
    concurrency
  }
  constraints
  objective { metric }
  tuning_config { workspace_path, keep_intermediate, max_trials = 1000 }
  output { optional result_path, include_raw_eval = false }
}
~~~

workload 中没有 searches、name、mode、query_count、filter 或 weight。V1 的 search 语义固定为
全量、无过滤 KNN。

## 5. 核心内部数据模型

### 5.1 WorkloadSpec

~~~text
WorkloadSpec {
  top_k
  concurrency
}
~~~

WorkloadSpec 是不可变输入，不包含 search_params、constraints 或 objective。

### 5.2 IndexCandidateSpace

~~~text
IndexCandidateSpace {
  index_name
  create_params_space
  search_params_space
}
~~~

用户显式 search_params_space 用于当前 WorkloadSpec；CandidateGenerator 的本地补齐规则可以依赖
workload.top_k。

### 5.3 CandidateSpec

参数展开先生成标准化候选：

~~~text
CandidateSpec {
  index_name
  create_params
  search_params
}
~~~

CandidateSpec 表示一个具体 create/search 参数组合，不再包含候选数组或 $range；workload、
constraints 和 objective 由请求统一提供。CandidateGenerator 未生成的索引原生默认字段可以保持
缺失，并在创建或搜索时由具体索引解析。

### 5.4 BuildSpec

~~~text
BuildSpec {
  build_id
  index_name
  create_params
  index_path
  use_existing_index
  cleanup_index_after_build_group
}
~~~

planner 内部用 index_name 与 create_params 的规范化 JSON 作为 build identity，不把该临时 key
保存到 BuildSpec。

### 5.5 TrialSpec

~~~text
TrialSpec {
  trial_id
  build_id
  search_params
}
~~~

一个 CandidateSpec 产生一个 TrialSpec。build 级状态通过 build_id 关联，workload 使用请求级
单一来源，不复制到规划对象。

### 5.6 BuildResult 与 TrialResult

~~~text
BuildResult {
  build_id
  status
  index_name
  create_params
  eval_type
  metrics
  constraint_evaluation
  artifacts
  elapsed_seconds
  failure
  optional raw_eval_result
}

TrialResult {
  trial_id
  build_id
  status
  top_k
  search_params
  metrics
  constraint_evaluation
  execution {
    query_count
    requested_concurrency
    index_instance_reuse
    load_policy
    reload_succeeded
    index_deserialize_count
  }
  elapsed_seconds
  failure
  optional raw_eval_result
}
~~~

TrialResult 通过 build_id 关联 index_name、create_params 和 artifact，不复制这些 build 级内容；
metrics 合并 build/search 指标，使完整候选可直接应用 constraints 和 objective。raw_eval_result
只有在 include_raw_eval=true 时出现；build 最多记录一份 build raw，trial 只记录自己的 search
raw。

### 5.7 Recommendation

~~~text
Recommendation {
  index_name
  create_params
  search_params
  workload
  metrics
  artifacts
  evidence { build_id, trial_id }
}
~~~

Recommendation 是 V1 选择单位，直接对应一个满足全部约束的完整 trial。best_effort 额外包含
constraint_evaluation 和 violation_summary，用于解释未满足约束的原因。

## 6. 模块职责

| 模块 | 职责 |
| --- | --- |
| Public API / CLI | 接收 JSON，调用 orchestrator，返回或写出 JSON。 |
| RequestValidator | 校验请求形态、workload、metric、objective、已有索引和预算。 |
| IndexTuningDescriptor | 在 AutoTune 内按 index name 注册 build/search proposal，使用静态生命周期。 |
| CandidateGenerator | 查找 descriptor，按本地规则为缺失字段生成 patch。 |
| CandidateExpander | 合并 patch，展开用户候选表达，并执行通用预算检查。 |
| TrialPlanner | 按 build key 分组，生成完整 create/search trials。 |
| EvaluationRunner | build、serialize、fresh deserialize、完整 query KNN 和指标采集。 |
| ConstraintEvaluator | 对每个完整 trial 计算 violations。 |
| ResultSelector | 按单一 objective 选择最终完整配置。 |
| ReportWriter | 生成 compact full report、可选写文件，并为 CLI 投影按状态裁剪的 summary。 |

模块名称是设计角色，不要求在实现中一一对应独立类；当前实现按这些职责保持边界。

## 7. Request validation

基础 validation 必须在创建 workspace 和启动 eval 前完成：

- request 是 object，version 等于 1。
- data_path 是可读普通文件；index_path 若存在也必须是可读普通文件。
- indexes 是非空数组，只含 hgraph / ivf。
- workload 只含 top_k 和 concurrency。
- top_k 在 1 到 1,000,000 之间且不超过 ground-truth 宽度；concurrency 在 1 到 200
  之间且不超过 Q；top_k × concurrency 不超过 1,000,000，避免 recall 结果缓冲随并发
  无界放大。
- constraints 是非空 object，metric 合法；阈值必须有限且非负，recall_at_k 在 [0, 1] 内。
- objective 必填且只包含 metric。
- tuning_config 只含 workspace_path、keep_intermediate、max_trials，后者默认 1000。
- output 只含 result_path 和 include_raw_eval；前者不能与 data_path 或 index_path 互为别名，
  后者默认 false；compact full report 中 builds 和 trials 始终输出。

V1 必须拒绝 mode、query_count、filter、weight、evaluation_strategy、sample_query_count 和
finalist_count，而不是忽略。

RequestValidator 不检查 create_params/search_params 的索引字段集合、类型、范围或依赖组合；这些
参数在真实 build/search 时由具体索引解释，错误写入对应 BuildResult 或 TrialResult.failure。

### 7.1 数据与 create_params 一致性

框架从数据集推断 dim、dtype 和 metric_type，并注入 create_params 中缺失的字段。用户显式
提供这些标量时必须与数据集一致。V1 的输入范围固定为 dense float32，不允许一个请求中的
index candidate 使用不同数据语义；其他 vector type 或 dtype 在候选生成前结构化失败。

### 7.2 existing-index 校验

提供 index_path 时：

- 请求必须恰好包含一个 indexes[] spec。
- 该 spec 展开后只有一个具体的 index_name + create_params。
- 允许多个 search_params，但 workload 仍只有一个。
- 输入 artifact 永远只读。

## 8. 候选生成

### 8.1 默认候选

CandidateGenerator 只为缺失字段生成 patch。内置实现通过 AutoTune-local
IndexTuningDescriptor 查找索引特定的 build/search proposal：

~~~text
GenerateBuildPatches(build_context)
  -> merge missing-only patches
  -> expand concrete create candidates
  -> GenerateSearchPatches(search_context)
  -> merge missing-only patches
  -> expand concrete search candidates
~~~

CandidateGenerator 的 HGraph 规则维护图结构、量化和 ef_search 的默认候选；IVF 规则维护
分桶、量化和 scan_buckets_count 的默认候选。具体列表是 AutoTune 的版本化实现细节。默认生成
分为 build 和 search 两步，因为 search 候选可以依赖数据集、具体 create candidate 和 workload。

HGraph 本地规则仅在 ef_search 缺失时根据 workload.top_k 生成默认候选。用户显式值保持
原样进入 trial，不会被改写或预先判定是否合法。ef_construction patch 同样可以依赖已经展开的
max_degree。

IVF 的 buckets_count patch 可以依赖 dataset.base_count，scan_buckets_count patch 可以依赖具体
buckets_count。partition_strategy_type 和 ivf_train_type 没有 patch 时使用 IVF 原生默认值；用户
显式提供的值，包括 GNO-IMI 相关参数，都保持原样进入 trial。

CandidateGenerator 不维护完整 parameter schema，也不执行索引参数的类型、范围、依赖组合或
safety 预校验。未知字段和可能非法的显式组合都进入真实 eval；原生 parser 拒绝 create 参数时
记录为 build failure，拒绝 search 参数时记录为对应 trial failure。AutoTune V1 不额外保证发现被
原生 parser 忽略的字段拼写错误。effective_request 只记录确定性的请求归一化和数据集证据；实际
展开参数和执行结果记录在 builds 与 trials 中。

构建线程直接使用索引原生创建参数表达：HGraph 对应
create_params.index_param.build_thread_count，IVF 对应 create_params.index_param.thread_count。
用户可以显式固定这些字段，也可以用数组或 range 参与候选展开；省略时由具体索引采用原生默认值。
AutoTune 不提供第二个通用构建线程字段，也不做索引特定的线程映射。

### 8.2 参数表达

- 标量是固定值。
- 数组是候选集合。
- $range 是闭区间枚举。

候选展开对对象字段、create/search 组合和多个 index 做笛卡尔积。

### 8.3 增量预算检查

设 index i 有 B_i 个 create candidates，每个 create candidate 对应 S_ib 个实际 search
candidates。计划 trial 数为：

~~~text
sum_i(sum_{b=1..B_i}(S_ib))
~~~

max_trials 默认 1000、V1 最大为 100000，并对展开后的 trial 数量生效。框架必须在递归
展开时累计并中止，不能先物化超限空间。V1 还分别限制单个候选表达最多展开 100000 个值，
整个请求最多执行 1000000 次展开工作，并最多访问 1000000 个
create/search 组合，防止嵌套表达或巨大组合绕过 max_trials 造成 CPU 和内存
放大。递归展开深度最多为 128，并同时计入 JSON 嵌套和对象字段遍历。

通用框架只校验 patch 不覆盖用户叶子，并合并、展开候选表达。每个完整 CandidateSpec 都进入
planner；索引参数是否合法由真实 build/search 判断，失败记录在对应 BuildResult 或 TrialResult。

## 9. Trial planning

### 9.1 build 分组

Planner 对 CandidateSpec 按 build key 分组：

~~~text
HGraph create_params X
  -> search_params A
  -> search_params B
~~~

只生成一个 BuildSpec 和两个 TrialSpec。

### 9.2 artifact 路径

无 index_path 时：

~~~text
workspace_path/runs/<run-id>/artifacts/<build-id>.index
~~~

每个通过基础 validation 的请求原子创建唯一 run 目录。keep_intermediate=false 时，请求级
RAII cleanup 只删除自己的目录。

有 index_path 时，BuildSpec 标记 use_existing_index=true，index_path 指向只读输入。

### 9.3 确定性

候选按用户数组和 generator patch 的首次出现顺序展开，重复候选保留第一次。build_id 和
trial_id 按该确定性顺序生成；相同输入在相同版本下应得到相同候选顺序和 plan 结构，run_id
与实际耗时可以不同。

## 10. EvaluationRunner

### 10.1 build path

~~~text
BuildSpec
  -> create concrete index
  -> build once
  -> collect build metrics
  -> serialize artifact
  -> measure artifact size
~~~

Build/shared metrics 至少包括 build_seconds、index_size_mb 和 index_memory_mb；`_mb` 按
1024 × 1024 bytes 换算。指标若尚不能可靠测量，必须明确标记 missing_metric，不能用近似值
冒充正式结果。

### 10.2 artifact reload baseline

对每个 TrialSpec：

~~~text
create empty concrete index
  -> deserialize BuildSpec artifact
  -> configure workload top_k / concurrency
  -> run eval with the full query set
  -> collect eval search metrics
  -> destroy index instance
~~~

同一 build 的 artifact 文件被复用，但内存实例不复用。这样 trial 间没有缓存、Tune 或统计
状态泄漏，也是 loaded-index reuse 优化的对照组。

### 10.3 full-query validation

设 query 集大小为 Q：

- 每个 trial 把完整 Q 条 query 交给 eval，不做 sampling。
- concurrency 映射为 eval 的 search worker 数。
- AutoTune 不为了延长 benchmark 主动循环 query。
- recall、latency 和 QPS 直接使用 eval 的输出口径。

当前 eval 可能为不同 monitor 分别执行完整 query 集。报告中的 query_count 表示正式统计覆盖
Q 条 query，不表示底层总搜索调用次数。合并 monitor pass 属于 eval 自身的后续优化，不在
AutoTune V1 中复制实现。

### 10.4 指标边界

- recall、QPS 和 latency 沿用 eval 的定义。
- search_seconds 优先使用 eval 输出；当前 eval 未输出该值时，使用完整 SearchEvalCase::Run
  的墙钟时间，可能包含 deserialize、多次 monitor pass 和结果处理。
- build_and_search_seconds = build_seconds + search_seconds，按 trial 计算。
- trial.elapsed_seconds 包含 AutoTune 的 trial 编排开销。
- index_memory_mb 来自具体 index 的内存占用报告，不使用进程峰值内存替代。

### 10.5 worker 错误

EvalCase::Run 通过异常报告的错误会被 runner 转换为结构化 trial failure，并继续后续 trial。
底层 eval 仍有 worker 错误直接终止进程的历史行为；该问题应在 eval 中单独修复，AutoTune
不复制一套 search loop 规避它。

## 11. Constraints 与结果选择

### 11.1 metric registry

MetricRegistry 固定每个 metric 的：

- 约束比较方式。
- objective 自然偏好。
- 指标在 build 阶段或 search 阶段产生的内部分类。
- 原始 eval 字段到标准字段的映射。

V1 方向：

| metric | constraint | objective |
| --- | --- | --- |
| recall_at_k、qps | 至少达到 | 越大越优 |
| latency_avg_ms、latency_p99_ms、search_seconds | 不超过 | 越小越优 |
| build_and_search_seconds | 不超过 | 越小越优 |
| build_seconds、index_size_mb、index_memory_mb | 不超过 | 越小越优 |

请求和 objective 中都不携带 direction。
V1 不接受用户自定义 metric、约束表达式、回调、多目标或 Pareto 输出。

### 11.2 constraints

顶层 constraints 可以同时包含 build 和 search metric。EvaluationRunner 把对应 build metrics
合并进每个 trial metrics，ConstraintEvaluator 随后对完整候选一次性应用全部 constraints。

### 11.3 ResultSelector

ResultSelector：

1. 要求每个成功 trial 都包含 objective metric，否则结构化失败。
2. 过滤不满足 constraints 的 trials。
3. 按 objective 的自然偏好选择 recommendation。
4. objective 相同则按稳定 trial_id 打破平局，不引入未声明的次级目标。

### 11.4 best_effort

没有可行 trial 时，只要存在包含 objective metric 的成功 trial，就输出完整 best_effort。
其排序使用：

1. missing metric 数。
2. violated constraint 数。
3. 归一化 violation 总量。
4. objective。
5. 稳定 ID。

best_effort 必须带完整 violations，且 status 仍为 no_candidate_satisfied。

## 12. existing-index 执行

existing-index 生成一个逻辑 BuildSpec：

~~~text
BuildSpec {
  index_path = request.index_path
  use_existing_index = true
  cleanup_index_after_build_group = false
}
~~~

runner 跳过 build，build_count=0。每个 TrialSpec 仍创建新 index 并从输入 artifact
deserialize。输入文件不进入 cleanup。

build_seconds 和 build_and_search_seconds 不存在，引用它们的 constraint 或 objective 在
validation 阶段失败；index_size_mb 可以从输入文件测量，index_memory_mb 可以在 load 后读取。
报告必须区分“没有执行 build”和“build 执行失败”。

## 13. 报告与证据

V1 生成两个投影：

- CLI stdout 是按状态裁剪的 summary：始终包含 version、status 和 elapsed_seconds，只保留
  recommendation、best_effort、failure 中适用的分支，并在报告成功写入时附加 report_path。
- RunAutoTune() 返回 compact full report；配置 result_path 时，同一 full report 写入该路径。

summary 面向直接消费，不包含请求快照、环境、builds、trials 或 raw eval。full report 包含：

- version、run_id、run_workspace_path 和 report_path。
- input_request 和 effective_request。
- status、elapsed_seconds、阶段耗时和 benchmark environment fingerprint。
- 固定为 full_grid/full_dataset/build-file-reuse 的 evaluation_strategy 证据。
- 顶层 objective。
- recommendation 或 best_effort。
- trial_count、build_count、build_group_count。
- builds 和 trials。
- failure。

full report 顶层字段集合固定。某阶段尚未产生的 run、effective request 或 objective 证据以 null
表示，数组/对象型计数证据使用空值；failure 说明实际停止阶段。

summary 和 full report 共享同一 status 及适用的结果分支。full report 保持固定字段集合；summary
省略不适用的 null 分支，result_path 未配置时也省略 report_path。full report 的 builds 和 trials
不提供关闭开关，避免 recommendation 的 evidence ID 失去引用对象。

### 13.1 请求快照

input_request 保留调用方原始 JSON。effective_request 记录：

- tuning_config/output 默认值。
- dataset 元数据绑定后的索引请求。
- 规范化后的 workload、constraints 和 objective。

### 13.2 artifact evidence

BuildResult 记录 artifact 的路径、来源、大小、是否为已有索引和 cleanup 计划。
TrialResult 不重复 artifact path，而是通过 build_id 引用 BuildResult；execution 记录
reload_succeeded、index_deserialize_count、index_instance_reuse 和 load_policy。同时记录数据集
query_count 和请求 requested_concurrency。底层 eval 当前不报告实际 worker 数，因此 V1 不
输出推测的 actual_concurrency；并发是否降级属于 eval 需要补充的运行期证据。

### 13.3 recommendation evidence

recommendation.evidence 必须包含 build_id 和 trial_id，使调用方能回到完整 build/trial 查看
参数、指标、constraints 和耗时。

### 13.4 environment evidence

报告记录 VSAG 版本、OS、CPU、内存和 SIMD 能力。search 并发记录在 effective_request 和
trial execution 中，索引原生构建线程配置记录在 create_params 中。V1 不预测跨机器性能；
使用 latency、QPS 或耗时指标时，benchmark 与
部署机器在 CPU/SIMD、核数、内存、VSAG 构建和并发配置等方面保持同规格是调用方契约。

顶层 elapsed_seconds 在选择完成后写入，覆盖 validation、workspace、候选生成、规划、评估和
选择，不包含可选报告文件写入和请求结束时的 workspace cleanup。

### 13.5 raw eval evidence

include_raw_eval 默认 false。关闭时 full report 直接省略 raw_eval_result 字段，不输出 null
占位；CLI summary 始终不含 raw。显式开启后，每个 BuildResult 最多保存一份本 build 的原始
eval 结果，每个 TrialResult 只保存本 search 的原始 eval 结果，不能把 build raw 复制到每个
trial。该开关不能改变标准化 metrics、constraint_evaluation、execution、failure 或选择结果。

## 14. CandidateGenerator 边界

V1 在 AutoTune 内定义独立的 CandidateGenerator，并用一个静态
IndexTuningDescriptor 表完成内置 proposer 的发现和生命周期管理：

~~~text
IndexTuningDescriptor {
  index_name
  propose_build_candidates
  propose_search_candidates
}
~~~

HGraph 和 IVF 各注册两个 proposal；request validation 和默认 generator 使用同一次 descriptor
查找，不再分别维护支持列表和 dispatch 分支。descriptor、proposer 和默认 generator 都是进程内
静态只读对象，不需要动态所有权协议或可变全局 registry。

该 descriptor 只存在于 AutoTune 模块，不扩展 core index registry，也不在候选生成时调用具体
索引。原因是 AutoTune 属于可选 tools target，且当前 AutoTune 与 core 使用不同 JSON 表示；V1
为了两个内置索引增加 core bridge 会扩大依赖面。以后出现第二个真实消费者或插件式索引时，可以把
同样的二方法 descriptor 提升为 core-neutral 接口，而无需改变 merge、expand、planner 或 eval。

proposal 接口为：

~~~text
GenerateBuildPatches(BuildContext)
GenerateSearchPatches(SearchContext)

BuildContext {
  request
  dataset
  index_name
  user_create_params
}

SearchContext {
  request
  dataset
  index_name
  concrete_create_params
  workload
  user_search_params
}
~~~

两个接口只返回缺失字段的 object patch，不声明 parameter schema，也不校验用户参数。它们不包含
构建线程绑定或 ValidateCandidate；构建线程是普通原生 create candidate，参数错误由真实
build/search 报告。search
context 包含具体 create candidate 和 workload，因此规则可以按上下文补齐 factor 等字段，且不会
与整段用户数组重新做无意的笛卡尔积。通用框架负责检查 patch 不覆盖用户叶子、合并 patch 和展开
候选；空 patch vector 等价于 generator abstain。

规则可以返回多个全标量 patch，以保留启发式或模型给出的参数 tuple 相关性。未来规则、历史或
ML generator 实现相同的 GenerateBuildPatches/GenerateSearchPatches 接口；替换实现不改变
TrialPlanner、EvaluationRunner、constraints 或完整结果。成本估计、artifact 复用和低成本
执行都在该接口之外。

## 15. V2 演进缝

### 15.1 候选来源

可以增加 DataProfileCandidateGenerator、HistoryCandidateGenerator 和 MLCandidateGenerator。
用户显式值仍不可覆盖。

### 15.2 评估优化

可以增加 query sampling、successive halving 和 dominance pruning。低成本阶段只产生排序或
淘汰证据；最终入选 recommendation 必须跑 full-query validation。

### 15.3 loaded-index reuse

V2 可以在同一 build 下复用已加载 index，但必须证明：

- 搜索参数切换没有跨 trial 可变状态。
- trial 顺序不影响结果。
- 资源和延迟指标口径未被 cache 污染。
- recommendation 与 artifact-reload baseline 一致。

无法证明时继续使用 V1 reload path。

### 15.4 ArtifactStore

跨请求缓存的 key 至少包含 dataset identity、index_name、完整 create_params、VSAG 版本和
artifact format 版本。V1 run-local artifact 语义不变。

### 15.5 索引低成本路径

HGraph build cache、Index::Tune()、训练结果复用或量化切换通过独立执行接口接入，不进入
CandidateGenerator 的本地候选规则。这些路径改变执行成本，不改变 workload、constraints、
objective 或最终 result shape。

## 16. V3 演进缝

V3 增加约束驱动创建入口：

~~~text
data + workload + constraints + objective
  -> generate Index Specs
  -> reuse AutoTune candidate/plan/eval/selection
  -> recommendation + validated metrics + artifact
~~~

V3 允许 indexes 省略，但 V1 的显式索引入口继续保留。CreateIndexWithConstraints 一类产品
API 只是 AutoTune 上层包装，不是新的索引实现。

## 17. V1 验收标准

必须满足：

1. HGraph 和 IVF 在真实 SIFT 上跑通。
2. 每种索引能展开 build、量化和 search 参数。
3. 单 workload 的 trial 数和 build group 数计算正确。
4. 相同 build key 只调用一次 Build 并只生成一个 artifact。
5. 每个 search trial 使用新 index 实例 reload artifact。
6. 每个 trial 使用完整 Q 条 query，不做 sampling；monitor pass 次数沿用 eval 实现。
7. concurrency=1 和 concurrency>1 均正确映射到 eval 配置并记录 requested_concurrency。
8. recommendation 同时包含一套 create_params 和 search_params。
9. 统一 constraints 与单一 objective 的选择可验证。
10. existing-index 路径 build_count=0 且不修改输入文件。
11. max_trials、单轴展开和总组合预算在物化过程中生效。
12. EvalCase::Run 通过异常报告的单 trial 失败不会阻止后续 trial。
13. 并发请求使用不同 run 目录。
14. 被原生 parser 拒绝的索引参数由真实 build/search 暴露，并记录在对应 build 或 trial failure 中。
15. recommendation 能通过 evidence.trial_id 追溯全部证据。
16. CLI stdout 只输出当前状态适用的结果分支和可用字段，RunAutoTune() 和 result_path 保留
    compact full report。
17. compact trial 通过 build_id 关联 build，不重复 create_params 或 artifact path。
18. include_raw_eval 默认关闭；开启时 build raw 只记录一次，trial 只记录 search raw。

V1 实现必须以本文档的 tuning_config、完整网格、artifact-reload 和单场景完整推荐为
唯一正式路径。任何 V2 实验能力都不能改变 V1 baseline。
