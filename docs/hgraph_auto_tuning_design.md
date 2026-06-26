# HGraph Auto Tuning Framework Design

> Status: draft for discussion.
>
> This document describes a staged engineering plan for an HGraph automatic tuning framework in
> VSAG. It is intentionally written as an implementation design, not as user-facing product
> documentation. After the feature becomes stable, the public workflow should be documented under
> `docs/docs/{en,zh}/src/`.

## 1. Background

HGraph exposes several classes of parameters:

| Layer | Example parameters | Change cost | Main impact |
| --- | --- | --- | --- |
| Build config | `max_degree`, `ef_construction`, `alpha`, graph type | High. Usually requires rebuilding the graph. | Recall ceiling, graph memory, build time, path quality. |
| Representation config | `base_quantization_type`, `precise_quantization_type`, `use_reorder`, storage parameters | Medium. May require rebuilding vector codes, retaining raw vectors, or reloading storage. | Memory, distance cost, recall, reorder cost. |
| Search config | `ef_search`, `factor`, `enable_reorder` | Low. Query-time parameters. | Recall, latency, QPS. |
| Runtime environment config | prefetch/runtime parameters | Low. Does not change graph structure. | Latency, QPS, cache behavior. |

Manual tuning usually follows this loop:

```text
choose parameters -> build or load index -> run queries -> compute metrics -> adjust parameters
```

This is slow, difficult to reproduce, and hard to compare across datasets. The goal of this
framework is to turn that loop into a reproducible workflow:

```text
user constraints -> candidate generation -> measured trials -> Pareto frontier -> recommendation
```

The first implementation should prioritize correctness, reproducibility, and clear reporting over
advanced search strategies.

## 2. Goals

- Provide a repeatable tuning workflow for HGraph.
- Start with fixed build config and tune low-cost search parameters.
- Produce complete trial records instead of only returning one recommended value.
- Select recommended configurations based on measured full-chain metrics.
- Support staged expansion from `ef_search` tuning to representation tuning and limited build
  candidates.
- Keep the implementation reusable by both CLI tools and future library APIs.

## 3. Non-goals

The initial implementation does not aim to:

- Replace all manual tuning.
- Search the full HGraph parameter space.
- Provide a learned or Bayesian optimizer.
- Promise that every representation parameter can be hot-swapped through `Index::Tune()`.
- Redesign the current `Index::Tune()` public API as the primary auto-tuning API.
- Automatically generate ground truth for all workloads.
- Guarantee that sampled-query results exactly match full-query results.

## 4. Current VSAG Capabilities and Caveats

### 4.1 Useful existing capabilities

- HGraph supports query-time `ef_search` through search parameters such as:

  ```json
  {"hgraph": {"ef_search": 100}}
  ```

- `tools/eval/eval_performance` already measures recall, recall percentiles, QPS, latency
  percentiles, memory, and build time.
- HGraph has an internal ELP optimizer using a small grid search over runtime prefetch parameters.
- HGraph has a `Tune()` implementation that can rebuild vector codes from raw vectors for some
  representation changes.
- HGraph exposes memory estimation and memory usage APIs.

### 4.2 Important caveats

- Current `Index::Tune()` returns `expected<bool, Error>`, so it cannot directly carry tuning
  curves, Pareto frontiers, or detailed reports.
- Current HGraph `Tune()` requires raw vector availability. If the index does not have a raw vector
  source, tuning returns `false`.
- Current HGraph `Tune()` mainly detects changes by quantizer name. Sub-parameters inside the same
  quantizer, such as PQ or RaBitQ parameters, may not trigger a rebuild unless the implementation is
  extended.
- `eval_performance` lives under `tools/eval/` and has tool-level dependencies such as HDF5 and
  YAML. The core library should not depend directly on these tool internals.
- ELP optimizer is useful as a grid-search precedent, but it is not a target-recall tuner. It does
  not consume user query sets or ground truth.

These caveats are not blockers, but they should shape the first implementation.

## 5. Design Principles

1. Full-chain measurements decide final recommendations.
2. Proxy metrics may prune candidates, but must not be the final source of truth.
3. The first implementation should be simple and deterministic.
4. Every skipped or failed candidate should be recorded with a reason.
5. Tuning should be reproducible from an input config and output report.
6. Representation tuning must distinguish hot-swappable parameters from rebuild-required
   parameters.
7. CLI and library-level code should share the same core data model.

## 6. High-level Architecture

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
  |      - search parameter application
  |      - representation tuning through HGraph Tune when supported
  |      - optional rebuild for build candidates in later phases
  |
  +--> Evaluator
         - search execution
         - recall calculation
         - latency and QPS calculation
         - memory collection
  |
  v
ResultStore
  |
  v
Recommender
  |
  +--> constraint filtering
  +--> Pareto frontier generation
  +--> best feasible selection
  |
  v
TuningReport
```

## 7. Proposed Module Boundaries

### 7.1 Core tuning module

Suggested location:

```text
src/tuning/
```

The core module should avoid HDF5, YAML, command-line parsing, and tool-only dependencies. It should
operate on in-memory objects:

- `IndexPtr`
- query `DatasetPtr`
- ground-truth `DatasetPtr`
- candidate definitions
- target constraints

Main components:

| Component | Responsibility |
| --- | --- |
| `TuningRequest` | User goals, data references, candidate space, budget, and evaluation options. |
| `CandidateGenerator` | Generate search, representation, and later build candidates. |
| `CandidateApplier` | Apply one candidate to an index or report why it cannot be applied. |
| `Evaluator` | Execute queries and compute metrics. |
| `TrialRunner` | Run candidates, collect errors, handle timeout and cache. |
| `ParetoFrontier` | Compute non-dominated candidates. |
| `Recommender` | Select best feasible candidate under user constraints. |
| `TuningReport` | Stable output data model. |

### 7.2 CLI/tool module

Suggested location:

```text
tools/tune/
```

or, if we want to keep it near the existing benchmark tool:

```text
tools/eval/tune/
```

The CLI module should handle:

- HDF5 dataset loading.
- YAML/JSON config parsing.
- File output.
- Optional progress display.
- Optional integration with existing `eval_performance` components.

The CLI should call the core tuning module rather than duplicating the tuning algorithm.

## 8. Request Model

A future CLI config may look like this:

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

The core request should not require this exact YAML shape. YAML is a tool-level format. The core
should use typed request objects or a stable JSON-like internal schema.

## 9. Report Model

The report should be stable from P0 onward.

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

## 10. Phase Plan

### 10.1 P0: `ef_search` tuning

Scope:

- Fixed build config.
- Fixed representation config.
- Tune only `ef_search`.
- Use real query execution and ground-truth recall.

Execution:

```text
load or build index
for ef in candidates:
    run query set with {"hgraph": {"ef_search": ef}}
    record recall, QPS, latency, memory
choose the smallest ef satisfying hard constraints
emit full report
```

Default candidate set:

```text
[50, 100, 200, 400, 800, 1000]
```

`1000` is chosen as a conservative default for `topk=10` because HGraph currently validates
`ef_search` against a topK-dependent upper bound. Larger values can be supported when they pass the
index validation rules.

P0 acceptance criteria:

- Given a target recall, the tuner finds the minimal valid `ef_search` that satisfies it.
- If no candidate satisfies the target, the tuner reports the best measured candidate and explains
  that no feasible candidate was found.
- The output contains all trial metrics, not only the final recommendation.
- The P0 workflow can be repeated and produces comparable reports.

### 10.2 P1: representation plus `ef_search` tuning

Scope:

- Fixed build config.
- Generate a bounded list of representation candidates.
- For each representation candidate, tune `ef_search`.

Candidate categories:

| Category | Examples | Initial support |
| --- | --- | --- |
| Query-time search | `ef_search`, `factor`, `enable_reorder` | Supported in P0/P1. |
| Quantizer type switch | `base_quantization_type`, `precise_quantization_type`, `use_reorder` | Supported only when HGraph `Tune()` can apply it and raw vectors are available. |
| Same-quantizer sub-parameters | `base_pq_dim`, RaBitQ bits, PCA/FHT options | Future or rebuild-required until HGraph `Tune()` handles these changes explicitly. |
| IO/storage parameters | `base_io_type`, `precise_io_type`, file paths | Future or rebuild/reload-required. |

Important rule:

```text
Do not silently treat a candidate as tuned if the underlying index did not apply the change.
```

P1 execution:

```text
for representation_candidate in representation_candidates:
    apply representation candidate
    if candidate cannot be applied:
        record skipped reason
        continue
    run P0 ef_search tuning on this representation
merge all trials
compute Pareto frontier
recommend best feasible profile
```

P1 acceptance criteria:

- The report clearly separates completed, failed, and skipped candidates.
- The tuner does not claim support for representation changes that are not applied by HGraph.
- The final recommendation is based on measured recall, QPS, latency, and memory.

### 10.3 P2: limited build candidates

Scope:

- Add a small number of build candidates.
- Each build candidate gets a fresh index build.
- Run P1 inside each build candidate.

Example:

```text
max_degree: [16, 32, 48]
ef_construction: [400]
alpha: [1.2]
```

P2 acceptance criteria:

- The report includes build time and build memory per build candidate.
- The final Pareto frontier can compare candidates across build configs.
- Build failures are reported as candidate failures, not global tuner failures.

## 11. Candidate Generation

Candidate generation should support both explicit user-provided spaces and rule-based defaults.

P0 default:

```text
ef_search: [50, 100, 200, 400, 800, 1000]
```

P1 conservative default:

```text
base_quantization_type: [fp32, fp16, bf16, sq8, sq4]
use_reorder: [false, true]
precise_quantization_type when use_reorder=true: [fp32, fp16]
ef_search: [50, 100, 200, 400, 800, 1000]
```

This is a suggested first candidate set, not the full HGraph parameter space. HGraph also supports
additional quantizers such as `sq8_uniform`, `sq4_uniform`, `pq`, `pqfs`, `rabitq`, and `tq`.
Those should be added only after their tuning semantics are clear.

## 12. Metric Semantics

Initial metrics:

| Metric | Meaning |
| --- | --- |
| `recall_avg` | Average recall over evaluated queries. |
| `recall_detail` | Query-level recall percentiles such as `p10`, `p50`, `p90`. |
| `qps` | Queries per second over measured queries. |
| `latency_avg_ms` | Average per-query latency. |
| `latency_detail_ms` | Latency percentiles such as `p50`, `p90`, `p95`, `p99`. |
| `memory_bytes` | Index memory usage or peak memory depending on evaluator mode. |
| `build_time_s` | Build time for build candidates. |

The report should make clear whether memory is current index memory, estimated memory, or peak
process memory.

## 13. Pareto Frontier and Recommendation

Hard constraints:

- Recall must be at least target.
- Memory must be at most budget.
- QPS must be at least target if provided.
- Latency must be at most target if provided.

The initial recommendation policy should be deterministic:

1. Filter candidates by hard constraints.
2. If feasible candidates exist, choose the one with the smallest primary cost.
3. P0 primary cost is `ef_search`.
4. P1 primary cost can be latency or memory depending on user objective.
5. If no feasible candidate exists, return `best_effort` with a reason.

Pareto dimensions:

```text
maximize recall
maximize QPS
minimize latency
minimize memory
minimize build time when build candidates are present
```

## 14. Failure and Skip Semantics

Every candidate should end in one of these states:

| State | Meaning |
| --- | --- |
| `completed` | Trial ran and produced metrics. |
| `skipped` | Candidate was not run because it is known to be unsupported or invalid. |
| `failed` | Trial was attempted but returned an error. |
| `timeout` | Trial exceeded the configured budget. |
| `cached` | Result was reused from a previous identical trial. |

Examples of skip reasons:

- `ef_search` is outside the index validation range.
- Representation candidate requires raw vector storage but raw vector is unavailable.
- Candidate changes same-quantizer sub-parameters that current HGraph `Tune()` cannot detect.
- Candidate requires rebuild but current phase does not allow rebuild.

## 15. Cost Reduction Roadmap

Cost reduction should come after P0 correctness.

Potential techniques:

| Technique | Phase | Notes |
| --- | --- | --- |
| Query sampling | P0/P1 | Evaluate on fewer queries, then optionally validate finalists on the full query set. |
| Trial cache | P0/P1 | Avoid repeated measurements of identical candidates. |
| Memory estimate pruning | P1/P2 | Safe for hard memory budgets if estimate is conservative. |
| Successive halving | P1/P2 | Run many candidates cheaply, then promote survivors. |
| Recall-ef curve fitting | Later | Useful only after empirical validation. |
| Build candidate pruning | P2 | Use fp32 large-ef recall ceiling or memory estimate to avoid hopeless builds. |

The guiding rule remains:

```text
Proxy metrics prune. Full-chain metrics decide.
```

## 16. Testing Strategy

### 16.1 Unit tests

- Candidate generation.
- Constraint filtering.
- Pareto frontier generation.
- Recommendation selection.
- JSON report serialization.
- Skip and failure state handling.

### 16.2 Integration tests

- P0 tuning on a small deterministic HGraph dataset.
- Target recall is reachable.
- Target recall is not reachable.
- Invalid `ef_search` candidate is skipped or failed with a clear reason.
- Representation candidate requiring raw vector is skipped when raw vectors are unavailable.

### 16.3 Performance tests

- Verify that tuning overhead is bounded for a small candidate set.
- Compare repeated runs for stability.
- Validate sampling error on at least one representative dataset before enabling sampling by
  default.

## 17. Documentation Plan

During development:

- Keep this document as the engineering design.
- Add local tool documentation if a CLI is introduced.

After feature stabilization:

- Add user-facing English and Chinese docs under `docs/docs/{en,zh}/src/`.
- Document supported parameters and unsupported candidates explicitly.
- Add examples under `examples/cpp/` or tool examples if there is a public workflow.

## 18. Open Questions

1. Should the first CLI live under `tools/tune/` or be integrated into `tools/eval/`?
2. Should the reusable core be added under `src/tuning/`, or should P0 start entirely under
   `tools/` and be promoted later?
3. What should the first public API look like, if any?
4. Should P1 require `store_raw_vector: true`, or should the tuner detect and use existing fp32
   base/precise storage as a raw-vector source?
5. How should the tuner verify that a representation candidate actually changed the index state?
6. Should `factor` be included in P0, or deferred to P1 with reorder-related candidates?
7. What is the default objective when multiple feasible candidates exist: lowest latency, lowest
   memory, smallest `ef_search`, or highest QPS?
8. Should final candidates always be re-evaluated on the full query set after sampling?
9. How much of `eval_performance` should be reused directly versus reimplemented in a core
   evaluator?
10. How should reports encode machine/environment metadata so that results are comparable across
    runs?

## 19. Suggested PR Breakdown

1. Design document only.
2. P0 CLI and core data model for `ef_search` tuning.
3. Extract reusable evaluator and report schema.
4. Add Pareto frontier and recommendation logic.
5. Add representation candidates with explicit support checks.
6. Add query sampling and trial cache.
7. Add limited build candidates.
8. Add public docs and examples after the workflow stabilizes.

## 20. Summary

The recommended path is:

```text
P0: fixed build + fixed representation + ef_search tuning
P1: fixed build + supported representation candidates + ef_search tuning
P2: limited build candidates + P1 inside each build candidate
Later: proxy pruning, sampling, successive halving, adaptive search
```

This plan keeps the first implementation small enough to land while preserving a clear path toward
a more complete automatic tuning framework.
