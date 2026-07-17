# VSAG AutoTune

## 0. 文档说明

* **状态**：评审草案
* **更新时间**：2026-07-14
* **VSAG main 代码快照**：`efdaf17a10e96cdb5222baf558d50dfacbdc672e`
* **关联文档**：`autotune_api_v1_zh.md`

本文用于内部方向评审，主要讨论 VSAG AutoTune 要解决的问题、外部工作、VSAG 当前能力、项目目标、演进路线和总体设计。

具体 JSON 字段、默认值、状态码、错误语义和报告结构由 `autotune_api_v1_zh.md` 定义。候选展开、任务调度、缓存格式、指标采集和失败处理等实现问题不在本文展开。

---

## 1. 背景

### 1.1 当前调参流程

VSAG 已经支持多种索引、量化方式以及构建和搜索参数。为了得到一组适合当前数据和 workload 的配置，用户通常需要完成下面的工作：

```text
选择索引
  -> 固定一部分经验参数
  -> 枚举构建、量化和搜索参数
  -> 生成多份 eval 配置
  -> 执行 build 和 search
  -> 汇总 recall、latency、memory 等指标
  -> 筛选最终参数
```

`tools/eval` 可以对一份明确配置执行真实的 build 和 search，但多个候选之间的组织仍由用户处理。用户需要自己准备候选、生成配置、安排执行、复用构建产物并汇总结果。

例如：

```text
max_degree:       [32, 64]
ef_construction:  [100, 200]
quantization:     [sq8, fp32]
ef_search:        [40, 80, 120]
```

前三项产生 8 个构建配置，每个构建配置还要测试 3 个 `ef_search`，总计 24 个 search trial。只有搜索参数不同的候选可以共享同一个构建产物，实际只需要执行 8 次 build。

这套流程可以通过 shell 或 Python 脚本完成。当前的问题是，每个使用者都要重新实现一遍。不同脚本对候选空间、构建复用、指标口径、失败处理和结果选择的理解可能不同，试验过程也不容易复查和复现。

### 1.2 参数空间

索引参数之间存在相互影响。

提高图的连接数可能改善 recall，也会增加内存、索引大小和搜索开销。更激进的量化可以缩小索引，但为了达到相同 recall，可能需要增大搜索范围或增加重排。构建参数发生变化后，原来合适的搜索参数也可能需要重新选择。

随着索引类型、构建参数、量化方式和搜索参数继续增加，完整候选空间会快速扩大。主要成本不是生成配置，而是实际训练、构建、加载和查询。

不同参数的评估成本也不同。`ef_search`、`nprobe` 和重排规模等搜索参数通常可以在已有索引上反复测试；图结构、聚类数量或量化方式发生变化时，往往需要重新训练或重新构建。

这里包含两个不同的问题：

* 减少需要实际评估的候选；
* 降低每个候选的真实评估成本。

候选推荐和搜索策略主要处理前一个问题。采样、缓存、构建复用和索引内部能力主要处理后一个问题。

### 1.3 用户目标

索引参数的效果取决于数据和 workload，包括数据规模、维度、距离分布、局部密度、embedding 模型、query 难度、topK、线程数、硬件和索引版本。

一套参数在某个数据集上表现良好，换一批数据或部署环境后不一定仍然适用。仅提供一套默认参数不能覆盖所有场景。

多数用户最终希望表达的是性能和资源要求，例如：

* recall 至少达到 0.95；
* 平均查询延迟不超过 2 毫秒；
* QPS 不低于某个值；
* 峰值内存不超过 8 GB；
* 索引文件不超过 2 GB；
* 构建在 1 小时内完成。

高级用户仍然需要显式控制索引和参数，但参数不是多数用户的最终目标。如果每增加一种索引或量化方式，都继续向用户暴露更多底层配置，VSAG 支持的能力越多，用户承担的配置负担也越大。

### 1.4 要解决的问题

VSAG AutoTune 需要逐步处理三个问题。

当前首先缺少的是统一的调参任务。用户已经可以通过 `eval` 和脚本完成试验，但候选、执行、结果选择和试验证据没有统一入口。

参数空间扩大后，真实评估成本会成为主要问题。系统需要减少无效候选，并复用构建、训练和查询过程中已经完成的工作。

长期入口需要从参数逐步转向目标。用户可以继续显式控制参数，系统也应允许用户只声明数据、workload 和性能或资源要求。

---

## 2. 外部工作

### 2.1 工业界

| 系统或产品                           | 用户提供什么                    | 系统完成什么                       | 当前边界                              |
| ------------------------------- | ------------------------- | ---------------------------- | --------------------------------- |
| [Faiss ParameterSpace](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning#the-parameterspace-object)            | 已有索引、query 和 ground truth | 尝试固定索引的运行期参数，记录搜索时间和结果质量     | 索引类型、量化方式和大部分构建参数需要提前确定           |
| [AutoFaiss](https://github.com/criteo/autofaiss)                       | 向量数据、索引内存限制、可用内存和查询时间要求   | 选择并构建 Faiss 索引               | 绑定 Faiss；约束和试验报告没有形成通用契约          |
| [Amazon OpenSearch Auto-optimize](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-auto-optimize.html) | 数据、recall 和 latency 要求    | 执行采样和候选评估，返回 HNSW、量化、重排和搜索参数 | 主要面向托管环境和特定索引；候选空间和完整 trial 过程不开放 |
| [Milvus](https://milvus.io/docs/single-vector-search.md) / [Zilliz AUTOINDEX](https://docs.zilliz.com/docs/autoindex-explained)       | 数据和部署信息                   | 自动选择部分索引配置，隐藏部分底层参数          | 更接近自动选型和参数屏蔽，公开接口不强调完整 trial 证据   |
| [Elasticsearch dense vector 相关能力](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector/) | 数据和索引配置                   | 提供默认或自动的向量索引配置               | 公开能力更偏默认选择，离线试验过程不是主要接口           |
| [Weaviate dynamic ef](https://docs.weaviate.io/weaviate/concepts/vector-index#dynamic-ef)             | topK 等运行期信息               | 动态调整 ef                      | 处理运行期搜索预算，不选择构建参数和量化方式            |

现有产品已经出现从内存、查询时间和 recall 等目标出发选择配置的入口，说明这种使用方式存在实际需求。

开源工具通常更容易理解和复现，但覆盖范围较窄。托管产品覆盖了更多工程环节，内部候选生成和完整试验过程通常不可见。

### 2.2 学术界

| 研究方向  | 代表工作                                          | 解决的问题                             | 适用边界                      |
| ----- | --------------------------------------------- | --------------------------------- | ------------------------- |
| 候选选择  | [Manu](https://www.vldb.org/pvldb/vol15/p3548-yan.pdf)、[VDTuner](https://doi.org/10.1109/ICDE60146.2024.00332)                                  | 使用 BOHB、多目标贝叶斯优化等方法，根据已有试验选择下一批候选 | 可以减少无效候选，但选中的配置仍需要真实构建和测量 |
| 历史经验  | [Meta-learning configuration framework](https://doi.org/10.1016/j.is.2022.102123)、[PGTuner](https://doi.org/10.1145/3749179) | 根据数据规模、维度、局部结构、距离分布和历史试验推荐索引或参数   | 依赖历史数据、硬件和索引版本是否具有代表性     |
| 评估成本  | [FastPGT](https://arxiv.org/abs/2602.11573)、[CHAT](https://arxiv.org/abs/2607.04630)                                  | 利用参数结构、变化趋势或中间计算复用，减少重复构图和无效评估    | 通常依赖 HNSW 或图索引内部过程        |
| 查询期调整 | [DARTH](https://doi.org/10.1145/3749160)、[Ada-ef](https://doi.org/10.1145/3786639)                                  | 根据 query 难度或搜索过程中的质量信号调整搜索预算      | 不负责选择索引类型、构建参数和量化方式       |

这些工作处理的是调优过程中的不同成本。

BOHB、贝叶斯优化和学习型推荐主要减少候选数量。FastPGT、CHAT 和索引内部复用主要降低一次真实评估的成本。历史模型可以减少冷启动试验，但数据分布、硬件或索引实现变化后，仍然需要真实试验校准。

查询期自适应处理的是索引建成以后单条 query 的搜索量，与离线调优可以配合，但不是同一个任务。

在当前调研范围内，没有找到一个被广泛采用的通用开源实现，同时覆盖多种索引、构建参数、量化参数、搜索参数、多维约束、真实验证和完整试验证据。

---

## 3. VSAG 现状

### 3.1 已有能力

| 能力                 | 当前能够做什么                                                        | 还缺什么                                               |
| ------------------ | -------------------------------------------------------------- | -------------------------------------------------- |
| `tools/eval`       | 读取 HDF5 数据集，创建索引，执行 build 和 search，输出构建耗时、QPS、延迟、recall 和内存等指标 | 一次处理一份明确配置，不负责候选补全、多候选规划、构建复用和结果选择                 |
| `Index::Tune()`    | HGraph 可以保留已有图结构，重新训练并替换部分量化存储                                 | 不生成候选，不读取真实 workload，不计算 recall 和 latency，也不负责结果选择 |
| ELP                | 使用 mock query 调整部分 prefetch 参数                                 | 优化范围局部，不处理用户 workload 和多维约束                        |
| HGraph build cache | 导出和导入部分邻居信息，用于后续 build 的 warm start                            | 尚未由统一框架判断哪些候选可以复用以及如何进入试验流程                        |

`tools/eval` 调用的是真实索引实现，可以继续作为 AutoTune 的执行和测量基础。

部分资源指标还需要补齐。当前 build 路径中的 `memory_peak` 不是对完整构建过程持续采样得到的峰值，`eval` 也不直接输出索引文件大小。在指标口径明确之前，AutoTune 不能可靠判断对应约束。

### 3.2 当前缺口

VSAG 已经具备真实评估、部分量化替换、局部运行期优化和构建复用能力。当前缺少的是把这些能力连接起来的任务层。

这层需要处理：

* 接收数据、workload、索引、候选和目标；
* 补充和规划候选；
* 组织 build 和 search trial；
* 复用相同构建配置；
* 调用真实 `eval`；
* 根据指标和约束选择结果；
* 保存推荐结果和试验证据。

第一步不需要重新实现 `eval`，也不需要先选择某一种智能候选搜索算法。当前先要补上候选、执行、指标和结果之间的任务流程。

---

## 4. 项目目标

### 4.1 当前目标

当前先把用户通过 `eval` 配置和 shell/Python 脚本完成的调参流程，变成 VSAG 提供的正式任务。

用户在这一阶段仍然显式指定要评估的索引，并可以提供全部或部分候选参数。系统负责组织真实试验、复用相同构建配置、根据约束选择结果并保存试验证据。

完整候选评估作为第一阶段的质量基线。后续加入采样、剪枝、缓存或模型后，可以与这个基线比较总耗时和最终结果。

### 4.2 长期目标

长期入口允许用户只提供：

* 数据或已有索引；
* workload；
* 性能和资源约束；
* 优化目标。

系统自动选择索引及参数，并返回：

* 推荐配置；
* 正式验证指标；
* 支持结果的 trial 记录；
* 可直接使用的索引产物。

高级用户仍然可以固定索引或参数。约束驱动入口是在现有参数入口之上增加的一层能力，不替代显式配置。

### 4.3 范围

本文讨论的是离线索引调优：针对给定数据和 workload，对构建、量化和搜索配置进行评估和选择。

最终 `recommendation` 需要来自真实评估。使用 query sampling 或其他低成本策略时，入选候选仍需经过正式验证。

用户显式指定的索引、参数和值域不能被系统静默修改。

查询期自适应处理单条 query 的搜索预算，单独演进。

---

## 5. 演进路线

### 5.1 流程自动化

第一阶段处理当前依赖用户脚本的问题。

系统接收候选和约束，规划 build 和 search trial，调用真实 `eval`，筛选结果并保存试验证据。相同构建配置只执行一次 build，并服务多个 search trial。

这一阶段关注正确、稳定、可解释和可复现，不承诺比完整候选评估更快。

### 5.2 成本优化

任务流程稳定后，第二阶段逐步降低调优成本。

候选侧可以加入：

* 更多基于数据和索引规则的候选补全；
* 基于历史报告的候选推荐；
* 贝叶斯优化或学习型候选。

评估侧可以加入：

* query sampling；
* successive halving；
* pruning；
* 预算控制；
* 同一 build 下的 search 复用。

具体索引执行侧可以逐步提供：

* build/search 返回的结构化参数错误；
* 参数变化是否需要重新构建；
* 图结构、训练结果或量化存储复用；
* `Index::Tune()`；
* warm start 和 build cache；
* 候选的大致成本。

当前开发分支已经实现 query sampling 和同一 build group 内的 loaded-index 复用。这些能力属于第二阶段的早期尝试，完整候选评估仍用于检查最终结果是否发生变化。

### 5.3 约束驱动入口

当候选生成、成本控制、自动选型和真实验证稳定后，用户可以省略具体索引配置，只声明数据、workload、constraints 和 objective。

系统自动生成索引和参数候选，复用已有 AutoTune 流程完成评估和选择。

第三阶段不需要重新实现调优逻辑，主要增加：

* constraints-only 的上层入口；
* 自动生成 `Index Spec`；
* 保存最终推荐的完整参数；
* 保存或返回选中的索引产物。

使用方可以根据推荐配置重新执行 `create + build`，也可以直接反序列化 AutoTune 已经生成的索引。

### 5.4 查询期自适应

查询期自适应针对索引建成后的单条 query。

系统可以根据查询难度、候选稳定性、距离 gap 或访问点数量动态决定是否继续搜索。它可以使用离线 AutoTune 选出的索引和基础参数，但训练、信号采集和运行期决策属于另一条技术路线。

该方向可以独立推进，不阻塞前三个阶段。

---

## 6. 总体设计

### 6.1 架构

```text
+-------------------------------+
| Request                       |
|                               |
| Data / Existing Index         |
| Workload                      |
| Index Spec（可选）            |
| Constraints                   |
| Objective                     |
| Tuning Config                 |
+---------------+---------------+
                |
                v
+------------------------------------------------+
| AutoTune                                       |
|                                                |
| IndexTuningDescriptor / CandidateGenerator     |
| Trial Planning                                 |
| Evaluation Strategy                            |
| Constraint Evaluation / Result Selection       |
+---------------+----------------+---------------+
                |                |
                |                | recommendation / evidence
                |                v
                |       +-------------------------------+
                |       | Result                        |
                |       |                               |
                |       | Recommendation                |
                |       | Evidence                      |
                |       | Index Artifact                |
                |       +-------------------------------+
                |
                | build / search trials
                v
+-------------------------------+
| eval                          |
|                               |
| build / load / search         |
| metric collection             |
+---------------+---------------+
                |
                | invoke
                v
+-------------------------------+
| Concrete Index                |
|                               |
| HGraph / IVF / ...            |
+-------------------------------+
```

图中的模块表示逻辑职责，不要求在实现中一一对应独立类。

### 6.2 请求和结果

Request 只描述上层信息：

| 类别                    | 含义                                      |
| --------------------- | --------------------------------------- |
| Data / Existing Index | 要构建的数据，或可选的已有索引                         |
| Workload              | metric、topK、query、过滤和并发方式等评测条件          |
| Index Spec            | 用户指定的索引和候选参数；第三阶段可以省略                   |
| Constraints           | recall、latency、memory、index size 等可行性条件 |
| Objective             | 多个可行候选之间优先优化的指标                         |
| Tuning Config         | 评估策略、预算和工作目录等调优任务配置                     |

Constraints 用来判断候选是否可行，Objective 用来在多个可行候选之间做选择。第一阶段可以提供固定的默认 objective，接口上保留这一概念即可。

Result 包含三个部分：

| 类别             | 含义                            |
| -------------- | ----------------------------- |
| Recommendation | 选中的索引、完整参数和实测指标               |
| Evidence       | 支持结果的 builds、trials、有效请求和运行信息 |
| Index Artifact | 可选的最终索引产物，可用于反序列化或后续交付        |

具体字段由 API 文档定义。

### 6.3 职责划分

| 组件       | 主要职责                         |
| -------- | ---------------------------- |
| AutoTune | 本地候选生成、试验编排、评估策略、约束判断、结果选择和报告 |
| eval     | 真实 build、load、search 和指标采集   |
| index    | 原生参数解析、原生默认值、真实执行、复用和序列化能力 |

具体索引是 parameter schema、原生参数和执行语义的来源。AutoTune 的 CandidateGenerator 只按
本地规则为缺失字段生成 patch，不维护完整 schema，也不预先校验用户候选。内置 HGraph/IVF
proposal 由 AutoTune-local 的静态 IndexTuningDescriptor 表发现；descriptor 只承担 proposal
注册和生命周期，不包含构建线程映射或参数校验。具体索引也不分别实现一套完整的任务和结果流程。

### 6.4 阶段接入

| 阶段   | 对框架的影响                                                       |
| ---- | ------------------------------------------------------------ |
| 第一阶段 | 用户提供 `Index Spec`，使用简单 CandidateGenerator 和完整评估，输出推荐和试验证据       |
| 第二阶段 | 增强 CandidateGenerator、Evaluation Strategy 和独立执行优化接口 |
| 第三阶段 | `Index Spec` 可以省略，增加约束入口和索引产物交付                              |

第一阶段建立的候选、评估和结果流程在后续阶段继续复用。

第二阶段主要替换或增强内部策略：

* CandidateGenerator 决定为哪些缺失字段生成哪些 patch；
* Evaluation Strategy 决定如何分配评估预算；
* 独立执行优化接口承载索引复用和低成本执行能力，不进入候选生成链路。

第三阶段主要调整上层入口和结果交付，不重新设计调优流程。

### 6.5 设计原则

**用户输入优先。** 用户已经指定的索引、参数和值域不能被系统静默修改。系统只补充未指定的部分。

**候选生成和真实评估分开。** 候选可以来自静态默认、数据规则、历史报告或模型，但后面的 trial
执行、约束判断和结果结构保持一致。

**最终结果经过真实验证。** 采样、预测和剪枝可以降低成本，但最终 `recommendation` 需要有正式评估结果支持。

**参数知识边界清晰。** 具体索引维护 parameter schema、原生默认值和执行语义；AutoTune 的
CandidateGenerator 只维护缺失字段的 patch 规则。用户显式候选原样进入真实评估，被原生 parser
拒绝的参数记录为 build 或 trial failure。

**调优结果可以直接使用。** 结果包含实际使用的参数配置，索引原生默认字段可以保持省略；结果也
可以关联选中的索引产物，为后续 `create + build` 或 deserialize 提供输入。

---

## 7. 第一阶段

### 7.1 支持范围

第一阶段计划支持：

* HGraph 和 IVF；
* 真实 SIFT 数据集；
* 构建参数、量化参数和搜索参数；
* 用户显式指定索引；
* 用户提供全部或部分候选参数；
* AutoTune 的 CandidateGenerator 为缺失参数生成本地默认 patch；
* 相同构建配置服务多个 search trial；
* 已有索引上的 search-only；
* 性能和资源约束；
* 推荐结果和试验证据。

资源约束的实际支持范围取决于对应指标是否能够可靠测量。

### 7.2 不支持的能力

第一阶段不包含：

* 自动选择用户没有声明的索引；
* 一次性支持所有 VSAG 索引；
* 使用机器学习直接决定最终结果；
* 比完整候选评估更快的承诺；
* 跨请求缓存和分布式执行；
* query-time adaptive search。

### 7.3 验收标准

第一阶段完成时，应能够做到：

1. HGraph 和 IVF 可以在真实 SIFT 数据集上完成一轮调优。
2. 构建、量化和搜索参数可以同时参与候选，只有搜索参数不同的候选不会重复 build。
3. 已有索引可以执行 search-only；有可行候选时返回推荐，无可行候选时明确失败。
4. 推荐结果有对应的实测指标和 trial 记录，相同请求可以复现候选规划和选择逻辑。

---

## 8. 评审事项

本次评审需要确认：

1. 当前调参流程、参数空间和用户入口是否构成 AutoTune 要解决的主要问题。
2. VSAG 当前是否缺少连接候选、真实评估、约束和结果的任务层。
3. 流程自动化、成本优化和约束驱动入口的阶段划分是否合理。
4. 总体框架能否支撑三个阶段，不需要在后续重新设计调优流程。
5. AutoTune、`eval` 和具体索引的职责划分是否合理。
6. 第一阶段的范围和验收要求是否合适。

具体 API、指标口径、执行方式、缓存格式和执行优化接口在方向确认后分别评审。

---

## 参考资料

### 工业实现

* [Faiss runtime parameter tuning / ParameterSpace](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning#the-parameterspace-object)
* [AutoFaiss](https://github.com/criteo/autofaiss)
* [Amazon OpenSearch Auto-optimize](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-auto-optimize.html)
* [Milvus AUTOINDEX](https://milvus.io/docs/single-vector-search.md)
* [Zilliz AUTOINDEX](https://docs.zilliz.com/docs/autoindex-explained)
* [Weaviate dynamic ef](https://docs.weaviate.io/weaviate/concepts/vector-index#dynamic-ef)
* [Elasticsearch dense vector](https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector/)

### 学术工作

* [Manu](https://www.vldb.org/pvldb/vol15/p3548-yan.pdf)
* [Meta-learning configuration framework](https://doi.org/10.1016/j.is.2022.102123)
* [VDTuner](https://doi.org/10.1109/ICDE60146.2024.00332)
* [PGTuner](https://doi.org/10.1145/3749179)
* [FastPGT](https://arxiv.org/abs/2602.11573)
* [CHAT](https://arxiv.org/abs/2607.04630)
* [DARTH](https://doi.org/10.1145/3749160)
* [Distribution-Aware Exploration / Ada-ef](https://doi.org/10.1145/3786639)

正式合并前补充精确标题、版本日期和公开链接。
