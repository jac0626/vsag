# AutoTune API v1 草案

状态：V1 实现评审稿

本文档定义 VSAG AutoTune V1 已实现的输入、输出和执行语义，是代码、测试和示例共同遵守的
评审契约。

V1 的目标是把用户手工编写 eval 配置、展开候选、复用构建、汇总指标和筛选结果的流程，
变成一个可复现的官方任务。V1 是完整评估基线，不承诺比人工网格搜索更快。

## 1. V1 范围

V1 支持：

- HGraph 和 IVF。
- 用户显式给出一个或多个索引及其全部或部分候选空间。
- 构建参数、量化参数和搜索参数联合调优。
- 每个请求一个 KNN search workload。
- 一组同时覆盖 build 和 search 指标的 constraints。
- 一个用于选择完整 create/search 配置的 objective。
- 相同构建配置只 build 一次。
- 已有索引上的 search-only 调优。
- CLI 默认输出精简 summary；C++ API 和可选 result_path 提供包含 builds、trials 和请求快照的
  compact full report。

V1 不支持：

- range search、filtered search、hybrid search 和 query-time adaptive search。
- 用户省略索引集合后由系统自动选型。
- query sampling、successive halving、Bayesian optimization 或 ML candidate generator。
- 跨请求 artifact cache、分布式执行和 Index::Tune() 热替换编排。
- 一次请求内动态改变数据集或 query 集。
- 一次请求内调优多个 workload；不同场景必须使用独立请求。

V1 输入中没有 mode、query_count 或 filter 字段。workload 隐式表示：
使用数据集中的全部 query 评测无过滤 KNN。

### 1.1 调用入口

CLI 接收一个 request JSON 文件，并把按状态裁剪的 summary 写到 stdout：

~~~bash
build-release/tools/autotune/autotune request.json
~~~

公共 C++ API 接收 request JSON 字符串并返回 compact full report JSON 字符串：

~~~cpp
#include <vsag/autotune.h>

const std::string report_json = vsag::autotune::RunAutoTune(request_json);
~~~

该接口由源码树内的可选 CMake target `vsag::autotune` 提供。构建 VSAG 时必须同时开启
`ENABLE_TOOLS=ON` 和 `ENABLE_CXX11_ABI=ON`。V1 不把该 library 或 CLI 加入安装包；AutoTune
是依赖 eval 和 HDF5 的 add-on，core `vsag` 不反向依赖它。

CLI 只有在 `status=failed` 或读取 request 文件失败时返回非零退出码。
`no_candidate_satisfied` 表示调优任务完整执行但约束不可满足，因此退出码仍为 0；调用方
必须读取 summary 中的 status，不能只用退出码判断是否得到 recommendation。

## 2. Request 总体结构

~~~json
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
  "workload": {
    "top_k": 10,
    "concurrency": 1
  },
  "constraints": {
    "recall_at_k": 0.95,
    "latency_p99_ms": 5.0,
    "build_seconds": 3600,
    "index_size_mb": 2048,
    "index_memory_mb": 8192
  },
  "objective": {
    "metric": "latency_avg_ms"
  },
  "tuning_config": {
    "workspace_path": "/tmp/vsag_autotune",
    "keep_intermediate": false,
    "max_trials": 1000
  },
  "output": {
    "result_path": "/tmp/vsag_autotune/report.json",
    "include_raw_eval": false
  }
}
~~~

## 3. 顶层字段

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| version | 是 | int | API 版本，V1 固定为 1。 |
| data_path | 是 | string | eval 可读取的普通文件，包含 base、query 和 ground truth。 |
| index_path | 否 | string | 可读的已有索引普通文件；存在时进入 search-only 路径。 |
| indexes | 是 | array | 用户允许评估的索引集合及候选参数。 |
| workload | 是 | object | 单个 KNN search workload。 |
| constraints | 是 | object | 对完整 create/search 候选生效的非空硬约束。 |
| objective | 是 | object | 在全部可行候选中选择最终 recommendation 的目标。 |
| tuning_config | 否 | object | 调优任务控制，不描述生产 workload。 |
| output | 否 | object | full report 的写入路径和诊断内容控制。 |

未知顶层字段必须在 validation 阶段失败，避免拼写错误被静默忽略。

## 4. 数据与已有索引

### 4.1 data_path

data_path 始终必填，并且必须指向可读的普通文件，目录、FIFO 等特殊文件会在 validation 阶段
失败。即使使用 index_path，AutoTune 仍需要 query 和 ground truth 计算 recall、latency 和
QPS。

V1 使用 eval 的 HDF5 数据集格式，并将支持范围固定为 dense float32。框架从数据集推断并补齐
索引创建必需的 dim、dtype 和 metric_type。repr 对非 sparse dtype 已由 VSAG 默认为 dense，
因此 AutoTune 不主动输出该冗余字段；用户显式提供 repr 时仍必须校验其与数据集一致，不能把它
作为候选轴。其他 vector type 或 dtype 在 validation
阶段结构化失败，避免索引默认量化候选与数据类型不匹配。V1 不提供 query subset；Q 条 query
就是一次正式评估的完整 query 集。
effective_request.dataset_description 记录推断出的 base_count、query_count 等数据集证据，
但这些值不是用户可控的 workload 字段。

### 4.2 index_path

index_path 必须指向可读的普通文件，并且是只读输入。AutoTune 不覆盖、不移动、不删除该文件。

使用已有索引时：

- 请求必须恰好包含一个 indexes[] spec；该 spec 展开后也必须只有一个具体的
  index_name + create_params。
- search_params 仍可包含多个候选。
- workload 仍然只描述一个场景。
- build_count 为 0，build_group_count 为 1。
- 每个 search trial 都从 index_path 创建新的 index 实例并完成 deserialize。
- create_params 仍然必需，因为 runner 需要先创建正确类型的空 index 再反序列化。

如果 index_path 对应的 create_params 展开出多个 build candidate，请求必须在
candidate_generation 阶段结构化失败，不能静默选择一个 candidate 或重新 build。

已有索引场景不会产生 build_seconds 或 build_and_search_seconds，因此 constraints 或 objective
引用这两个指标会直接返回结构化 validation failure。index_size_mb 从输入文件测量，
index_memory_mb 在成功加载后由具体索引报告。

## 5. workload

workload 是 V1 的正式 benchmark 契约，每个请求只包含一个。构建线程不是通用 workload 字段，
而是具体索引的原生创建参数：HGraph 使用
indexes[].create_params.index_param.build_thread_count，IVF 使用
indexes[].create_params.index_param.thread_count。用户可以把它写成标量、数组或 range 参与候选
展开；省略时使用具体索引的原生默认值。AutoTune 不再维护第二个线程配置入口，也不做字段映射。

### 5.1 workload 字段

workload 必须且只能包含以下字段：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| top_k | 是 | int | KNN 返回数量，范围为 1 到 1,000,000，且不能超过 ground-truth 宽度。 |
| concurrency | 是 | int | 同时执行 query 的 worker 数，范围为 1 到 200，且不超过 Q。 |

V1 不接受 searches、name、mode、query_count、filter、weight 或任意额外字段。为保证 recall
结果缓冲不会随并发无界放大，V1 还要求 top_k × concurrency 不超过
1,000,000 个 neighbor id；该限制与 top_k、concurrency 各自的上限同时生效。

示例：

~~~json
{
  "top_k": 10,
  "concurrency": 1
}
~~~

### 5.2 query 执行和并发语义

设数据集有 Q 条 query。对每个 build candidate 和 search_params candidate 的组合，runner：

1. 从对应 build artifact 创建并反序列化一个新的 index 实例。
2. 把 Q 条 query 全部交给底层 eval，不做 sampling。
3. 把 concurrency 映射为 eval 的 search worker 数。
4. 直接采用 eval 对完整 query 集输出的 recall、latency 和 QPS 口径。

底层 eval 当前可能为不同 monitor 分别执行完整 query 集，因此 query_count 只证明正式统计
覆盖 Q 条 query，不表示底层总共只调用 Q 次搜索。AutoTune V1 不改变 eval 的 QPS 和 latency
定义，也不会为了延长 benchmark 主动循环 query。concurrency 描述评测查询形态，不是搜索
参数候选，也不属于 indexes[].search_params。

同一组 query 会在不同 candidate 下分别执行。

## 6. indexes 与参数候选

indexes 是非空数组。V1 只接受 hgraph 和 ivf。

~~~json
{
  "name": "hgraph",
  "create_params": {
    "dim": 128,
    "dtype": "float32",
    "metric_type": "l2",
    "index_param": {
      "base_quantization_type": ["fp32", "sq8_uniform"],
      "max_degree": [16, 32],
      "ef_construction": 200
    }
  },
  "search_params": {
    "hgraph": {
      "ef_search": [40, 80, 120]
    }
  }
}
~~~

字段：

| 字段 | 必填 | 类型 | 说明 |
| --- | --- | --- | --- |
| name | 是 | string | hgraph 或 ivf。 |
| create_params | 否 | object | VSAG 原生创建参数及 build 候选。 |
| search_params | 否 | object | VSAG 原生搜索参数候选。 |

用户显式给出的 search_params 是当前 workload 的索引级候选池。AutoTune 内部独立的
CandidateGenerator 只为缺失字段生成 patch；规则可以依赖 dataset profile、已经展开的具体 create
candidate 和 workload。effective_request 只保存规范化请求和数据集证据，不复制完整候选空间；
合并 patch 并展开后的参数记录在 builds/trials 中。

内置 HGraph/IVF proposer 通过 AutoTune-local 的静态 IndexTuningDescriptor 表发现；descriptor
只包含 index name、build proposal 和 search proposal，不包含构建线程绑定或参数预校验。

HGraph 的 ef_search 只有在缺失时才由 CandidateGenerator 根据 top_k 生成默认 patch。用户显式的
标量、数组或范围会原样进入 trial，不会被截断、改写或预先判定是否合法；缺失的
ef_construction patch 可以依赖已经展开的 max_degree。

IVF 缺失的 buckets_count patch 根据 dataset.base_count 生成，缺失的 scan_buckets_count patch
可以依赖具体 buckets_count。partition_strategy_type 和 ivf_train_type 未生成时分别使用 IVF
原生默认值；用户显式提供的值，包括 GNO-IMI 相关参数，都原样进入 trial。

CandidateGenerator 不维护完整 parameter schema，也不执行索引参数的类型、范围、依赖组合或
safety 预校验。未知字段和可能非法的显式组合同样进入真实 eval；原生 parser 拒绝 create 参数时
记录为 build failure，拒绝 search 参数时记录为对应 trial failure，其他 build 和 trial 继续执行。
AutoTune V1 不额外保证发现被原生 parser 忽略的字段拼写错误。

### 6.1 候选叶子语义

| 写法 | 含义 |
| --- | --- |
| "max_degree": 32 | 固定值。 |
| "max_degree": [16, 32, 48] | 三个候选值。 |
| "ef_search": {"$range": {"start": 40, "stop": 200, "step": 40}} | 闭区间范围。 |

对象中的多个候选叶子做笛卡尔积。用户显式值优先；CandidateGenerator 只为缺失字段提供 patch。
它没有生成且索引原生允许省略的字段保持缺失，由具体索引使用原生默认值，AutoTune 不复制或提前
物化另一份索引默认参数。通用框架只校验 patch 不覆盖用户叶子，并负责合并、展开和 max_trials
预算；展开出的具体组合全部进入 build/trial。

具体索引是 parameter schema、原生类型和范围、缺省值及实际执行语义的唯一来源。AutoTune 不在
候选生成阶段查询或复制这套 schema；被原生 parser 拒绝的索引参数由真实 eval 返回并保存在
build/trial failure 中。

### 6.2 max_trials

每个具体的 index_name + create_params + search_params 组合产生一个 trial。max_trials 限制最终
完整 create/search trial 总数。

实现必须在递归展开过程中执行上限，不能先物化一个巨大笛卡尔积再检查。V1 的
max_trials 硬上限为 100000；单个候选表达最多展开 100000 个值，整个请求的展开工作量和
create/search 组合访问最多执行 1000000 次。这些通用上限独立于 trial 是否能被
具体索引执行，用于阻止嵌套表达或巨大组合造成 CPU 和内存放大。候选递归展开深度最多为 128；
该深度同时覆盖 JSON 嵌套层级和对象字段遍历，超过时在 candidate generation 阶段结构化失败。

## 7. constraints

constraints 是 metric 到阈值的对象。每个阈值都必须是有限、非负的数；recall_at_k 还必须在
[0, 1] 内。方向由 metric 固定，不需要也不允许用户填写 direction。

| metric | 约束语义 | objective 的自然偏好 |
| --- | --- | --- |
| recall_at_k | 大于等于阈值 | 越大越优 |
| qps | 大于等于阈值 | 越大越优 |
| latency_avg_ms | 小于等于阈值 | 越小越优 |
| latency_p99_ms | 小于等于阈值 | 越小越优 |
| build_seconds | 小于等于阈值 | 越小越优 |
| index_size_mb | 小于等于阈值 | 越小越优 |
| index_memory_mb | 小于等于阈值 | 越小越优 |
| search_seconds | 小于等于阈值 | 越小越优 |
| build_and_search_seconds | 小于等于阈值 | 越小越优 |

constraints 是非空顶层对象，可以同时包含 build 和 search metric，对完整 create/search 候选
生效：

- build_seconds 是该 build group 的一次真实构建耗时，不包含序列化。
- index_size_mb 是 artifact 文件大小，换算基数为 1024 × 1024 bytes。
- index_memory_mb 是具体 index 报告的内存占用，不是进程峰值内存。
- search_seconds 优先使用 eval 输出；当前 eval 未输出该值时，记录完整 search eval run 的
  墙钟时间，可能包含 deserialize、多次 monitor pass 和 eval 结果处理。
- 新构建场景下，build_and_search_seconds 等于 build_seconds 加 trial 的 search_seconds；
  已有索引场景无法提供该指标。

缺少约束要求的 metric 时，该 trial 不可行，并记录 missing_metric。

## 8. objective

V1 objective 必填且只包含一个 metric。方向由上一节的自然偏好决定：

~~~json
{
  "metric": "latency_avg_ms"
}
~~~

也可以引用 build metric：

~~~json
{
  "metric": "index_size_mb"
}
~~~

objective 不写 workload、direction、minimize 或 maximize。它用于在满足全部 constraints 的完整
create/search candidates 中选择 recommendation。V1 不支持多 objective、用户自定义 metric、
表达式、回调或 Pareto 输出。

## 9. tuning_config 与 output

### 9.1 tuning_config

tuning_config 只控制调优任务，不包含 workload 字段。

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| workspace_path | /tmp/vsag_autotune | run 根目录。 |
| keep_intermediate | false | 是否保留本次 run 的临时 artifact。 |
| max_trials | 1000 | 完整 create/search trial 数上限；V1 最大为 100000。 |

AutoTune 为每个通过基础 validation 的请求创建
workspace_path/runs/<run-id>/。并发请求不能共享或清理对方的 run 目录。已有 index_path
始终位于 run 目录之外并保持只读。

V1 tuning_config 不提供 evaluation_strategy、sample_query_count 或 finalist_count。完整网格和
全量 query 是唯一正式策略。

### 9.2 output

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| result_path | 空 | 可选 compact full report JSON 路径。 |
| include_raw_eval | false | 是否在 full report 中附带 eval 原始结果。 |

output 只能包含 result_path 和 include_raw_eval。若提供 result_path，它必须是非空字符串，且
不能与 data_path 或 index_path 指向同一文件（包括等价的规范化路径）；include_raw_eval 必须
是 bool。result_path 写入失败必须成为结构化 failure，不能只打印日志。

RunAutoTune() 的返回值和 result_path 文件都是 compact full report，builds 和 trials 始终完整
存在，不能由请求关闭。CLI stdout 固定输出精简 summary，不受 include_raw_eval 影响。
include_raw_eval=false 时，full report 省略所有 raw_eval_result 字段，而不是用 null 占位；只有
显式设为 true 才附带原始 eval 诊断内容。

## 10. 规划与执行语义

### 10.1 build group

build key 是规范化后的 index_name + create_params。相同 build key 只生成一个 BuildSpec：

~~~text
one build artifact
  -> search params 1
  -> search params 2
~~~

BuildSpec 成功后必须序列化 artifact。所有关联 search trials 复用该文件，不重复 build。

### 10.2 artifact reload

V1 为每个 search trial 创建新的 index 对象并从 artifact 反序列化。这样可以：

- 隔离不同 search_params trial 的可变状态。
- 让 trial 与真实 deserialize 后的部署路径一致。
- 建立后续 loaded-index reuse 优化的正确性基线。

artifact reuse 表示复用同一个已构建文件，不表示 V1 复用同一个内存 index 实例。

### 10.3 trial 规划

TrialSpec 至少包含：

~~~text
trial_id
build_id
search_params
~~~

index_name、create_params 和 artifact 由 build_id 关联 BuildSpec；top_k、concurrency 和
constraints 来自请求级 workload。每个 trial 对应一套完整的 index_name + create_params +
search_params 候选，但不复制这些共享状态。

## 11. 结果选择

V1 对完整 trial 做一层确定性选择：

1. 要求每个成功 trial 都包含 objective metric；缺失时结构化失败。
2. 对每个 trial 的合并 build/search metrics 应用全部 constraints。
3. 在可行 trials 中按 objective 的自然方向选择 recommendation。
4. objective 值相同则按稳定 trial_id 打破平局，不引入未声明的次级目标。

没有可行 trial 时返回 status = no_candidate_satisfied，recommendation = null。只要存在成功且
包含 objective metric 的 trial，就会提供一个完整 best_effort candidate。best_effort 只用于解释
失败，不能当作成功推荐；其排序先比较 missing metrics、违反约束数量和归一化违反程度，再比较
objective 和稳定 trial_id。

## 12. Response

V1 区分面向普通调用方的 CLI summary 和用于评审、复现的 compact full report。二者来自同一
次 run、共享同一 status、recommendation 和 best_effort，但不是同一个 JSON 投影。

### 12.1 CLI summary

CLI stdout 只输出当前状态有意义的字段。例如成功结果为：

~~~json
{
  "recommendation": {},
  "status": "success",
  "elapsed_seconds": 128.41,
  "report_path": "/tmp/vsag_autotune/report.json",
  "version": 1
}
~~~

CLI 的展示顺序按状态组织：success 先输出 recommendation，no_candidate_satisfied
先输出 best_effort，failed 先输出 failure，然后再输出 status 和其余字段。三个结果分支只输出
实际适用的一个，不使用 null 占位；report_path 也只在完整报告成功写入时输出。
recommendation 内部先输出 index_name、create_params 和 search_params，再输出 workload、指标
与证据。该顺序只用于提高命令行
可读性和复制便利性，JSON 调用方不能依赖对象字段顺序。compact full report 的结构和
序列化方式不受该展示逻辑影响。

report_path 是成功写入的 output.result_path。CLI 不把请求快照、环境、builds、trials 或 raw
eval 内容写到 stdout。

RunAutoTune() 不返回该 summary，而是返回下一节定义的 compact full report；提供 result_path
时，写入文件的内容与 RunAutoTune() 返回值相同。

### 12.2 recommendation

recommendation 在 summary 和 full report 中使用同一形态，必须包含：

- index_name 和实际传给索引的 create_params、search_params；原生默认字段可以省略。
- workload 的 top_k 和 concurrency。
- 合并后的 build/search metrics。
- build_id/trial_id evidence。
- index artifact 及其 response 后可用性证据。

success 时只输出 recommendation；no_candidate_satisfied 时只输出 best_effort。成功状态已经
证明全部约束满足，因此 recommendation 不重复空的 constraint_evaluation。best_effort 使用相同
完整候选形态，并额外保留 constraint_evaluation 和 violation_summary，用于解释失败与排序。

### 12.3 compact full report

full report 包含：

- version、run_id、run_workspace_path 和 report_path。
- input_request 和 effective_request。
- status、elapsed_seconds 和 elapsed_breakdown_seconds。
- environment、evaluation_strategy 和顶层 objective。
- recommendation、best_effort 和 failure。
- trial_count、build_count 和 build_group_count。
- compact builds 和 compact trials。

full report 顶层字段集合固定。某阶段尚未产生的 run、effective request 或 objective 证据以 null
表示，数组/对象型计数证据使用空值；failure 说明实际停止阶段。

[SIFT request 示例](../tools/autotune/examples/sift_hgraph_autotune_request.json)给出可运行的
输入形态；CLI summary 和 compact full report 的字段结构分别见 12.1 和 12.3。

### 12.4 builds

每个 compact build record 包含：

- build_id、status、index_name 和实际传给索引的 create_params；原生默认字段可以省略。
- eval_type：build 或 existing_index。
- build metrics，以及只检查当前阶段可测 build constraints 的 constraint_evaluation。
- elapsed_seconds。
- artifact path、是否为已有索引、是否计划清理及 response 后可用性。
- failure。

build_count 是实际 build 次数；existing-index 场景为 0。build_group_count 是规划中的唯一
build identity 数。build 参数只在 build record 记录一次；trial 的 metrics 会合并对应 build
metrics，使每个完整候选可以直接应用 constraints 和 objective。

### 12.5 trials

每个 compact trial record 固定包含：

- trial_id、build_id 和 status。
- workload 的 top_k 和完整 search_params。
- 合并后的 build/search metrics 和完整 constraint_evaluation。
- execution：query_count、requested_concurrency、index_instance_reuse、load_policy、
  reload_succeeded 和 index_deserialize_count。
- failure 和 elapsed_seconds。

trial 通过 build_id 引用对应 build，不重复 index_name、create_params 或 artifact path。
execution 保留验证完整 query 输入和 fresh deserialize 不变量所需的最小证据。底层 eval
当前不报告实际 worker 数，因此 V1 不输出推测的 actual_concurrency。
trial_count 是实际输出的完整 create/search trial
数；builds 和 trials 始终完整返回，V1 没有用于省略它们的 output 开关。

### 12.6 environment

environment 只出现在 full report，记录 VSAG 版本、操作系统、CPU、内存和 SIMD 能力；
workload 并发记录在 effective_request 和 trial execution 中，索引原生构建线程配置记录在
create_params 中。V1 不做跨机器性能推断；依赖
latency、QPS 或耗时指标的推荐必须在与部署环境同规格的机器上评估。调用方应保证 benchmark
与部署环境在 CPU/SIMD、核数、内存、VSAG 构建和并发配置等方面一致。

顶层 elapsed_seconds 在选择完成后记录，包含 validation、workspace、候选生成、规划、评估和
选择，但不包含可选 result_path 的文件写入以及返回前的 workspace cleanup。

构建线程若由用户指定，直接记录在具体索引的 create_params 中，并作为 build identity 和推荐
证据的一部分；省略时由具体索引采用原生默认值。底层 eval runner 按 V1 的进程内串行策略执行
trial。

### 12.7 raw eval

raw_eval_result 是诊断字段，不属于稳定的标准化指标契约。include_raw_eval=false 时，full
report 完全省略该字段；CLI summary 无论配置如何都不包含它。include_raw_eval=true 时：

- build record 最多包含一份本 build 的原始 eval 结果。
- trial record 只包含本 search 的原始 eval 结果。
- build raw 不复制到每个 trial；existing-index build 没有 build raw。

include_raw_eval 不改变 metrics、constraint_evaluation、failure、execution 或选择结果。

### 12.8 status 与 failure

| status | 说明 |
| --- | --- |
| success | 至少一个完整 trial 满足全部 constraints。 |
| no_candidate_satisfied | 有成功 trial，但没有候选满足全部约束。 |
| failed | validation 失败、所有 build 失败、没有可用 trial，或选择所需的 objective metric 缺失。 |

validation failure 时 recommendation 和 best_effort 都为 null；full report 的三个 count 均为
0，并提供 failure.stage、failure.code 和 failure.message。CLI summary 只投影 failure，不附带
空 builds、trials 或 count 字段。

如果 eval 返回了成功状态但缺少选择所需的 objective metric，不能把该 candidate 当成最优值；
报告以 `failure.stage=selection`、`failure.code=objective_metric_missing` 结构化失败。

## 13. 示例

### 13.1 最小 HGraph

~~~json
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
      },
      "search_params": {}
    }
  ],
  "workload": {
    "top_k": 10,
    "concurrency": 1
  },
  "constraints": {
    "recall_at_k": 0.95
  },
  "objective": {
    "metric": "latency_avg_ms"
  }
}
~~~

HGraph CandidateGenerator 可以为缺失的 build、量化和 search 参数生成默认 patch。

### 13.2 existing-index、search-only

~~~json
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
          "ef_search": [40, 80, 120]
        }
      }
    }
  ],
  "workload": {
    "top_k": 10,
    "concurrency": 32
  },
  "constraints": {
    "recall_at_k": 0.94,
    "index_size_mb": 2048,
    "index_memory_mb": 8192
  },
  "objective": {
    "metric": "qps"
  }
}
~~~

## 14. Validation 与结构化失败

V1 至少在启动 eval 前检查：

- version、必填字段和 JSON 类型。
- data_path 是可读普通文件，数据集 query 数大于 0；index_path 若存在也必须是可读普通文件。
- indexes 非空，只包含 HGraph / IVF。
- workload 只包含 top_k 和 concurrency，字段集合精确。
- top_k 在 1 到 1,000,000 之间且不超过 ground-truth 宽度；concurrency 在 1 到 200
  之间且不超过 Q；top_k × concurrency 不超过 1,000,000。
- 顶层 constraints 非空且 metric 合法；阈值有限且非负，recall_at_k 在 [0, 1] 内。
- 顶层 objective 合法且只包含 metric。
- 候选表达可以展开，且 max_trials、单轴展开值数、128 层递归展开深度和总组合访问次数不超限。
- index_path 只与一个 indexes[] spec 和一个具体 build candidate 共存。
- output 只含 result_path 和 include_raw_eval，且路径不覆盖任何输入文件。

该阶段不检查 create_params/search_params 的索引字段集合、类型、范围或依赖组合。展开后的用户
候选全部交给真实 eval；被原生 parser 拒绝的参数记录在对应 build 或 trial 的
failure 中。

失败对象至少包含：

~~~json
{
  "stage": "validation",
  "code": "invalid_request",
  "message": "request.workload.mode is unsupported"
}
~~~

build 或 search 通过 EvalCase::Run 异常报告的错误会转换为结构化 failure。build failure 标记
整个 build group；search failure 只影响当前 trial。底层 eval 仍有 worker 错误直接终止进程的
历史行为，该问题不由 AutoTune 复制 search loop 规避，需在 eval 中单独修复。

## 15. V2 / V3 演进缝

V2 可以在不改变 V1 单场景 workload 和完整候选结果形态的前提下增加：

- query sampling、successive halving 和 dominance pruning。
- loaded-index reuse；启用前必须证明与 V1 artifact-reload baseline 语义等价。
- 跨请求 artifact cache。
- build cache、Index::Tune() 和索引内部低成本路径。
- 预算控制、历史候选和 ML candidate generator。
- range、filter 等新的 workload schema；不能用含糊的 mode 字符串绕过独立字段设计。

V2 的硬规则是：低成本评估只能用于排序或剪枝，最终 recommendation 必须经过完整 query 集
验证。

V3 在 AutoTune 之上增加约束驱动创建入口。用户可以省略 indexes，由系统生成 Index Spec，
但仍复用 V1/V2 的 candidate、build group、workload evaluation、constraints 和完整结果。
V3 不是新的索引类型，也不要求重写调优器。

## 16. V1 验收

V1 验收必须持续证明：

1. HGraph 和 IVF 能在真实 SIFT 数据集上运行单 workload 调优。
2. build、量化和 search 参数都能展开。
3. 相同 index_name + create_params 只 build 一次并产出一个 artifact。
4. 每个 trial 从 artifact 重新加载 index。
5. 每个 trial 使用完整 query 集，不做 sampling；底层 monitor pass 次数沿用 eval 实现。
6. 同一个请求只能产生一套完整 create/search recommendation。
7. 统一 constraints 和单一 objective 都按本文语义工作。
8. existing-index search-only 不 build、不修改输入 artifact。
9. invalid request、unsupported index 和 max_trials 超限均在 eval 前结构化失败。
10. 被原生 parser 拒绝的索引参数由真实 build/search 暴露，并记录在对应 build 或 trial failure 中。
11. CLI stdout 只包含当前状态适用的结果分支和可用字段；RunAutoTune() 和 result_path 始终是
    compact full report。
12. compact trials 可通过 build_id 关联构建证据，不重复 build 参数。
13. recommendation、best_effort、builds、trials、请求快照和耗时足以复现实验。
14. include_raw_eval 默认关闭；开启后每个 build raw 只记录一次，每个 trial 只记录 search raw。
