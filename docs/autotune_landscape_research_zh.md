# 向量索引自动调优现状调研

- 状态：评审草案
- 更新时间：2026-07-13
- VSAG 代码快照：`main` 分支 `efdaf17a10e96cdb5222baf558d50dfacbdc672e`
  （2026-07-12）

向量索引通常同时暴露索引类型、构建参数、量化方式和搜索参数。这些参数共同影响召回率、
延迟、吞吐、内存、索引大小和构建时间，而且适合一个数据集的配置不一定适合另一个数据集。
自动调优要解决的，就是如何在可接受的试验成本内找到合适配置。

本文只描述当前公开可见的工业产品、开源项目、学术研究，以及 VSAG `main` 分支已有代码。
它不讨论 VSAG 的阶段规划，也不为某个具体设计方案作论证。

## 1. 摘要

向量索引自动调优并不是一个空白领域，但“AutoTune”这个名称覆盖了差异很大的能力。
有些产品只是替用户选择默认索引，有些工具只调整已有索引的搜索参数，还有一些系统会针对
给定数据执行多次构建和查询，再按照用户的性能要求给出推荐。

工业界已经把参数屏蔽和默认选择做得比较成熟。Faiss 可以探索固定索引的搜索参数并输出
速度与精度的 Pareto 前沿；AutoFaiss 可以在内存和查询性能限制下选择并构建 Faiss 索引；
Zilliz、Milvus 和 Elasticsearch 更偏向自动默认值或隐藏内部参数。Amazon OpenSearch
Auto-optimize 则更进一步：它接收 recall 和 latency 要求，实际运行优化任务并返回 HNSW、
量化和重排配置。在本文调研覆盖的公开产品中，它是较接近完整约束驱动调优的实现。

当前学术研究可以分为四条路线。Manu 和 VDTuner 根据当前试验结果选择更有希望的候选；
meta-learning configuration framework 和 PGTuner 尝试把历史数据迁移到新数据集；FastPGT
和 CHAT 分别通过共享建图工作和利用 HNSW 参数结构降低评估成本；DARTH 和 Ada-ef 则根据
单条 query 的难度动态决定搜索深度或终止时机。

本文调研尚未发现一个被广泛采用的通用开源实现，能够同时覆盖多类索引、构建参数、量化参数、
搜索参数、多维约束和完整试验证据。现有方案通常只覆盖其中一部分，并且大量研究集中在
HNSW 和 recall-latency 权衡上。

VSAG `main` 也处在这种“已有局部能力、尚无统一调优任务”的状态。仓库已经有真实性能评估
工具、HGraph 量化热替换、ELP 运行期参数优化和 HGraph 构建缓存，但还没有通用的候选生成、
试验编排、约束筛选和结果推荐模块。

## 2. “AutoTune”通常指什么

调研这类产品时，最容易出现的问题是把不同能力都叫作 AutoTune。公开实现大致可以分成
四类。

第一类是默认参数和参数屏蔽。系统根据数据类型、维度或部署规格选择一个经验配置，用户不再
直接看到复杂参数。Milvus AUTOINDEX、Zilliz AUTOINDEX 和 Elasticsearch 的自动量化默认值
主要属于这一类。

第二类是固定索引的搜索参数调优。索引已经构建完成，工具只调整 `ef_search`、`nprobe`、
重排规模等低成本参数，寻找速度和召回率之间的平衡。Faiss AutoTune 是最典型的实现。

第三类是离线的完整调优任务。系统接收数据、目标或约束，尝试不同的构建和搜索配置，并输出
经过真实评估的推荐。AutoFaiss 覆盖了其中一部分，OpenSearch Auto-optimize 是更完整的
托管产品形态。

第四类是查询期自适应搜索。系统不为所有 query 使用同一个搜索预算，而是根据当前 query 或
搜索过程信号动态选择 `ef`，或者在判断质量已经足够时提前停止。DARTH 和 Ada-ef 属于这一类。

这四类能力可以组合，但并不等价。尤其是“隐藏参数”不代表系统执行过针对当前数据集的真实
搜索，“动态 ef”也不代表系统调整了索引构建参数。

## 3. 工业界现状

### 3.1 Faiss：固定索引上的搜索参数探索

[Faiss AutoTune][faiss-autotune] 是较成熟、也经常被引用的基础实现。它把索引参数
分为构建期参数和运行期参数，自动调优只针对后者，例如 IVF 的 `nprobe`、HNSW 的
`efSearch`、`max_codes` 和重排候选规模。

调用方提供 query 和 ground truth，`ParameterSpace` 枚举可能的参数组合，
`AutoTuneCriterion` 计算搜索质量，`OperatingPoints` 保存性能和搜索时间。最终输出的不是
唯一答案，而是一组速度与精度互不支配的 operating points。

Faiss 也使用了一种简单但有效的剪枝：如果参数增大通常意味着更慢但更准确，就可以根据参数
偏序推导一个候选不可能成为新的 Pareto 点，从而跳过这次试验。

它的边界同样清楚：官方文档明确说明调优对象是运行期参数。索引类型、量化方式和大部分构建
参数必须在进入 AutoTune 之前确定。

### 3.2 AutoFaiss：带资源限制的 Faiss 索引构建

[AutoFaiss][autofaiss] 在 Faiss 之上提供了更接近用户目标的入口。用户可以给出 embeddings、
最大索引内存、当前可用内存和查询时间要求，工具选择适合的 Faiss 索引和参数，完成构建并
输出索引信息。

它解决了“我知道要用 Faiss，但不知道选哪种索引和参数”的问题，也支持大规模数据、磁盘映射
索引和 Spark 构建。不过，它仍然是 Faiss 专用包装，外部约束类型和试验报告没有形成通用
调优服务那样的契约。

AutoFaiss 不能简单描述为“已经停止维护”。截至本文更新时间，其
[release 页面][autofaiss-releases]显示最新版本为 2.18.0，发布时间为 2025-11-04。

### 3.3 OpenSearch：约束驱动调优的托管实现

[Amazon OpenSearch Auto-optimize][aws-auto] 接收存放在 S3 的 Parquet 或 JSONL 数据，
用户设置距离类型、维度、recall 和 latency 要求。服务在独立的托管 worker 上运行优化任务，
使用数据采样和并行评估，通常返回最多三个推荐。

它调整的内容包括 HNSW 的 `m`、`ef_construction`、`ef_search`，以及二值/标量量化、重排和
引擎参数。结果页面展示构建参数、搜索参数、预计性能和内存占用，用户还可以根据推荐继续
创建并灌入索引。

这说明约束驱动的向量索引优化已经进入工业产品。但它目前只支持 HNSW，运行在 AWS 托管
环境中，优化算法、候选空间和完整 trial 细节并不开放。

### 3.4 Milvus、Zilliz 和 Elasticsearch：自动选择多于透明调优

[Milvus AUTOINDEX][milvus-auto] 允许用户只指定 `AUTOINDEX` 和距离类型，由系统选择具体
索引。公开文档强调的是简化配置，并没有描述一个以用户 recall、latency 或 memory 约束为
输入、运行真实 trials、再返回证据的任务接口。因此，它更接近自动默认选择。

[Zilliz AUTOINDEX][zilliz-auto] 是专有实现。用户通常只需要选择 metric，云服务内部决定
索引构建、搜索、存储方式和动态量化。它降低了使用门槛，但不向用户公开候选空间、筛选过程
和完整试验结果。

[Elasticsearch dense_vector][elastic-vector] 会根据版本、向量类型和维度选择 HNSW 及
`int8`、`int4` 或 BBQ 等量化默认值。这也是产品默认策略，而不是针对每次创建请求执行的
黑盒优化任务。

### 3.5 Weaviate dynamic ef：动态，但不是 query difficulty

[Weaviate dynamic ef][weaviate-ef] 根据 query 的返回数量计算搜索列表长度，基本形式是：

```text
ef = clamp(query_limit * dynamicEfFactor, dynamicEfMin, dynamicEfMax)
```

它能减少用户在不同 `topK` 下手工设置 `ef` 的工作，但输入信号是 query limit，不是当前
query 的实际难度，也没有观察候选稳定性、距离 gap 或搜索过程中的 recall 信号。

### 3.6 工业界的整体状态

工业界已经比较成熟地解决了两件事：用经验规则隐藏复杂参数，以及在固定索引上寻找合适的
搜索参数。针对具体数据集执行构建、量化和搜索联合优化的产品正在出现，但目前公开实现少，
适用索引和约束范围也比较窄。

另一个明显差异是透明度。开源工具通常能看到参数和评估过程，但覆盖范围有限；云产品覆盖的
工程环节更多，却通常不公开候选生成、剪枝和每次 trial 的完整证据。

## 4. 学术界现状

向量索引调优借用了数据库 knob tuning 和机器学习超参数优化中的许多方法，包括贝叶斯优化、
Hyperband、强化学习和元学习。但向量索引还有一个特殊困难：改变构建参数或量化方式经常
意味着重新训练、重新编码或重新建图，一次评估本身就很昂贵。

现有研究并不是一条按时间依次替代的技术路线。按照它们主要降低哪一部分成本，可以分为
候选搜索、历史迁移、评估加速和查询期自适应四类。

### 4.1 用当前试验反馈选择候选

[Manu][manu] 在 PVLDB 2022 的系统论文中已经讨论了自动参数搜索。Manu 使用 BOHB，也就是
贝叶斯优化与 Hyperband 的组合。用户提供评价配置的 utility function 和总预算，系统优先
把更多预算分配给较有希望的区域，也可以只抽样 collection 的一部分降低成本。

[VDTuner][vdtuner] 发表于 ICDE 2024。它根据当前 workload 的真实试验结果训练多目标贝叶斯
优化模型，在搜索速度和 recall 之间寻找 Pareto 改善。它把不同索引类型及其参数放进一个
整体模型，并逐步减少对低收益索引类型的评估。[VDTuner 代码][vdtuner-code]已经开源。

这两项工作都根据当前调优任务的反馈决定下一步试什么。它们能减少无效候选，但不依赖历史
数据集为新任务直接推荐配置，也没有消除候选的真实构建和 workload replay 成本。

系统论文也展示了这类能力如何进入数据库内部。[SingleStore-V][singlestore-v] 在后台构建
索引，并基于 Faiss AutoTune 的网格搜索确定搜索参数。[VSAG 论文][vsag-paper]则重点利用
索引内部能力，避免为了调整部分参数反复重建图结构。这些工作体现的是系统集成和低成本变换，
不等同于一个覆盖多种索引的通用调优器。

### 4.2 用历史数据迁移配置经验

2023 年发表的 [meta-learning configuration framework][meta-config] 使用数据集描述符预测
图索引类型和参数。它既可以训练一个覆盖全部历史数据的全局模型，也可以先寻找相似数据集，
再使用对应模型；还支持在新数据集上做少量试验后微调。

这类方法的吸引力在于推理很快，不需要在每个新数据集上重新跑完整网格。但它把问题转化成了
模型泛化问题：训练数据是否覆盖新的 embedding 分布、硬件和索引版本，决定了推荐是否可靠。

[PGTuner][pgtuner] 是 2025 年发布的预印本。它进一步引入预训练的 query performance
predictor、强化学习参数推荐、分布外检测和主动学习，希望把旧数据集上的知识迁移到新数据集，
并在发现分布变化时用少量新试验校准。模型推荐的最终配置仍要真实构建和修正搜索参数。
[PGTuner 代码][pgtuner-code]也已公开。

### 4.3 新重点：减少昂贵的建图评估

学习型推荐能够减少候选数量，却不能消除候选的真实性能验证。对于 HNSW 等图索引，真正的
瓶颈往往是每个构建参数都要重新建图。

[IGS-HNSW][igs-hnsw] 是较早专门研究 HNSW 构建参数的预印本。它缩小需要测试的候选集合，
同时构建多个图，并利用候选配置之间的顺序关系选择参数。

[FastPGT][fastpgt] 是 2026 年发布的预印本，关注多份 proximity graph 构建之间的公共工作。
它共享距离计算和部分图结构，同时让推荐器一次提出一批候选，从而把候选推荐和批量评估配合
起来。这类优化必须进入索引构建器内部，单纯替换外部搜索算法无法获得同样收益。

[CHAT][chat] 是 2026 年 7 月刚发布的工作，作者在 arXiv 标注论文已被 SIGMOD 2027 接收。
它没有把 HNSW 完全看作黑盒，而是总结参数之间的单调性、主导单峰性和可分离性，据此推导
可行边界，并在完整建图前排除不可能满足资源约束的配置。

CHAT 支持 accuracy、latency、build time、index size 和 tuning budget 等约束，是目前与
“约束感知 HNSW 调优”最直接相关的研究之一。不过论文发布时间很近，其适用范围和复现结果
仍需要更多验证。

### 4.4 查询级自适应搜索

[DARTH][darth] 关注的是如何让用户直接声明目标 recall。它在 HNSW 或 IVF 搜索过程中周期性
调用 recall predictor，当模型判断当前结果已经达到目标时提前终止。它需要修改搜索循环并
获得距离计算次数、候选状态等内部信号。

[Ada-ef][ada-ef] 不直接预测搜索过程中的 recall，而是建立 query 与数据库向量距离分布的
统计模型，为 query 计算难度分数，再通过一个 difficulty-to-ef 映射选择尽可能小、同时能达到
目标 recall 的 `ef`。

这两项工作都试图解决固定搜索参数带来的过搜和欠搜，但它们不会替用户决定图的 `M`、
`ef_construction`、量化方式或索引类型。因此，它们属于查询运行期优化，而不是离线索引配置
搜索。

## 5. 目前已经解决什么，还缺什么

比较成熟的部分包括：标准数据集上的 recall/latency 评估、固定索引的搜索参数探索、基于
经验规则的默认参数，以及对 HNSW、IVF 等主流索引的基本性能权衡分析。

不同研究路线降低的不是同一种成本。贝叶斯优化减少需要尝试的候选，但仍依赖真实评估；
历史迁移降低新数据集的冷启动成本，但存在分布外风险；共享构建和结构感知剪枝可以直接减少
评估工作，却通常绑定具体索引；查询期自适应减少单条 query 的过搜和欠搜，但不负责构建参数
和索引类型的选择。它们可以组合，不能互相替代。

公开实现仍然比较分散。Faiss 只负责运行期参数，AutoFaiss 绑定 Faiss，VDTuner 和 PGTuner
更偏研究原型，FastPGT 和 CHAT 主要面向图索引，OpenSearch 则是闭源托管服务。跨 HNSW、
IVF 和量化方案统一处理 build、search、memory、index size 和 build time，且公开完整试验
证据的系统，在本文调研范围内仍然少见。

另一个未充分解决的问题是证据和复现。许多产品只返回最终配置，许多论文则在固定 benchmark
上报告结果。对于实际部署，硬件、线程数、query 分布、过滤比例和数据更新都会影响结论；
仅有参数推荐而没有评估环境和 trial 记录，很难判断结果能否复现。不同论文报告的加速比也
不能直接比较，因为数据集、硬件、预算和 recall 口径并不一致。

## 6. VSAG `main` 代码现状

本节只描述开头列出的 `main` 提交，不包含当前开发分支新增的 AutoTune 代码。该快照中没有
`tools/autotune` 目录，也没有统一的调优 request 或任务入口。

### 6.1 `tools/eval` 已经是真实性能来源

[`tools/eval`](../tools/eval/) 可以读取 HDF5 数据集，创建 VSAG 索引并执行 build、search 或
组合 case。build 路径调用 `Index::Build()` 并序列化索引；search 路径创建索引对象，从
`index_path` 反序列化，然后执行查询。

它能够输出 build duration、TPS、QPS、平均及分位 latency、平均及分位 recall、内存明细和
若干搜索过程统计。这些指标来自真实索引实现，而不是代价模型。

eval 的使用单位仍然是一个明确配置的 case。每个 case 都要写出 `index_name`、
`create_params` 和 `search_params`；多个 case 按顺序运行。它不会展开参数数组，不会判断两个
case 是否可以共享同一次 build，也不会根据约束筛选和推荐结果。

[`BuildSearchEvalCase`](../tools/eval/case/build_search_eval_case.h) 的 build 和 search 使用两个
case 对象，通过索引文件衔接。search-only 可以手工完成，但调用方仍要提供能创建索引对象的
`index_name` 和 `create_params`。

内存指标还需要注意口径。build 路径只在 `Build()` 返回后调用一次
`MemoryPeakMonitor::Record()`，所以当前名为 `memory_peak(build)` 的值实际上是构建结束时的
RSS 相对初始值，并不是构建过程中的真实峰值。eval 目前也不直接输出索引文件大小。

错误处理同时存在异常和进程退出路径，输出结果里没有统一的 trial failure 结构。这些都是
eval 作为单配置性能工具的现状，而不是指标计算本身的问题。

### 6.2 `Index::Tune()` 目前主要是 HGraph 量化热替换

公开接口位于 [`include/vsag/index.h`](../include/vsag/index.h)。默认实现返回
`UNSUPPORTED_INDEX_OPERATION`，`main` 中只有 HGraph 覆写了具体行为。

HGraph 的 [`Tune()`](../src/algorithm/hgraph/hgraph.cpp) 接收一份新的创建参数，比较 base 和
precise quantizer、reorder 设置，重新训练需要变化的 quantizer，再为全部向量生成新的 code
storage。准备完成后，它在全局锁保护下替换旧存储。

这个过程保留已经构建的图结构，因此切换量化表示可以比完整重建更便宜。它依赖索引中仍然
保留可恢复的向量；调用方还可以用 `disable_future_tuning=true` 删除后续调优所需的数据。

当前实现不生成候选，不读取 query 或 ground truth，也不计算 recall、latency 或约束。
它不能修改已经建好的 `max_degree`、`ef_construction` 等图结构参数。示例
[`318_feature_tune.cpp`](../examples/cpp/318_feature_tune.cpp) 展示的也是从一种 HGraph 量化配置
切换到另一种量化配置。

### 6.3 ELP 是独立的运行期微调器

HGraph 配置 `use_elp_optimizer=true` 后，会在 build 或 load 完成时调用
[`elp_optimize()`](../src/algorithm/hgraph/hgraph_build.cpp)。它固定 `ef=80` 和 `topk=10`，逐个
扫描 `prefetch_stride_codes`、`prefetch_stride_visit` 的 `1..10`，以 mock search 的运行时间
作为 loss。

mock query 来自索引内部向量的解码结果，而不是用户提供的 query workload。当前 ELP 不计算
recall，也不根据 recall-latency 约束选择参数。它解决的是 CPU prefetch 相关的运行期微调，
范围比通常所说的索引 AutoTune 窄得多。

### 6.4 HGraph build cache 和 IVF

`Index` 已公开 `ExportCache()` 和 `ImportCache()`，HGraph 可以导出与 source id 关联的邻居
信息。新索引导入 cache 后，下一次 `Build()` 会 warm start、区分命中和未命中节点，再执行
refine，并在 stats 中输出 cache hit 信息。

这个 cache 是索引内部的构建加速能力。`tools/eval` 不会自动导入、导出或判断哪些配置之间
可以复用 cache。

IVF 在 `main` 中已经支持真实 build、search、序列化和 eval，但没有覆写 `Tune()`，也没有
HGraph 这套 build cache 接口实现。

### 6.5 文档语义与代码存在差异

当前 [`优化器文档`](docs/zh/src/advanced/optimizer.md) 把 `Tune()` 描述为接收
`queries_dataset`、`target_recall` 和 `top_k`，并称历史 ELP 已经统一到 `Tune()` 背后。

这与 `main` 的代码和示例不一致：HGraph `Tune()` 实际解析创建参数并替换量化存储；ELP 仍由
`use_elp_optimizer` 单独触发，也不读取用户 query 或 recall 目标。阅读仓库现状时，应以当前
实现和测试为准。

### 6.6 当前代码的整体判断

`main` 已有一组与自动调优直接相关的组件：eval 负责真实指标，HGraph `Tune()` 提供低成本
量化切换，ELP 调整 prefetch 参数，build cache 复用部分构建工作。这些能力彼此独立，各有
明确用途。

仓库当前没有把它们组织成一个通用调优任务。参数候选仍由调用方准备，eval case 仍由调用方
编排，约束筛选和最终选择也需要外部脚本完成。这是 `main` 快照的代码事实。

## 7. 参考资料

### 7.1 工业实现

- [Faiss runtime parameter tuning][faiss-autotune]
- [Faiss AutoTune C++ API][faiss-api]
- [AutoFaiss][autofaiss]
- [Amazon OpenSearch Auto-optimize][aws-auto]
- [Amazon OpenSearch result semantics][aws-console]
- [Milvus Create Index / AUTOINDEX][milvus-auto]
- [Zilliz AUTOINDEX Explained][zilliz-auto]
- [Weaviate dynamic ef][weaviate-ef]
- [Elasticsearch dense_vector][elastic-vector]

### 7.2 学术工作

- [Manu][manu]
- [Meta-learning configuration framework][meta-config]
- [VDTuner][vdtuner]
- [SingleStore-V][singlestore-v]
- [PGTuner][pgtuner]
- [IGS-HNSW][igs-hnsw]
- [FastPGT][fastpgt]
- [CHAT][chat]
- [DARTH][darth]
- [Distribution-Aware Exploration / Ada-ef][ada-ef]
- [VSAG][vsag-paper]

[faiss-autotune]:
  https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning
[faiss-api]: https://faiss.ai/cpp_api/file/AutoTune_8h.html
[autofaiss]: https://github.com/criteo/autofaiss
[autofaiss-releases]: https://github.com/criteo/autofaiss/releases
[aws-auto]:
  https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-auto-optimize.html
[aws-console]:
  https://docs.aws.amazon.com/opensearch-service/latest/developerguide/auto-optimize-console.html
[milvus-auto]:
  https://milvus.io/api-reference/restful/v3.0.x/v2/Index%20%28v2%29/Create.md
[zilliz-auto]: https://docs.zilliz.com/docs/byoc/autoindex-explained
[weaviate-ef]: https://docs.weaviate.io/weaviate/concepts/vector-index#dynamic-ef
[elastic-vector]:
  https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/dense-vector/
[manu]: https://www.vldb.org/pvldb/vol15/p3548-yan.pdf
[meta-config]:
  https://www.sciencedirect.com/science/article/pii/S0306437922001016
[vdtuner]: https://arxiv.org/abs/2404.10413
[vdtuner-code]: https://github.com/tiannuo-yang/VDTuner
[singlestore-v]: https://vldb.org/pvldb/vol17/p3772-chen.pdf
[pgtuner]: https://arxiv.org/abs/2508.17886
[pgtuner-code]: https://github.com/hao-duan/PGTuner
[igs-hnsw]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4734062
[fastpgt]: https://arxiv.org/abs/2602.11573
[chat]: https://arxiv.org/abs/2607.04630
[darth]: https://arxiv.org/abs/2505.19001
[ada-ef]: https://arxiv.org/abs/2512.06636
[vsag-paper]:
  https://www.vldb.org/pvldb/vol18/p5017-cheng.pdf?file=p5017-cheng.pdf
