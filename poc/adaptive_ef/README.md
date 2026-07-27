# Adaptive-ef cosine matrix evaluator

`cosine_matrix_eval.cpp` loads an existing FP32 cosine HGraph (`max_degree=64`),
trains adaptive ef through `Index::EnableAdaptiveEf`, and saves the resulting
state as a separate index. If no base index exists, it builds and saves the base
graph once before training. Calibration covers top-k `10,50,100` and recall
targets `0.90,0.95,0.99`. The evaluator tests all nine adaptive combinations at
`alpha=0.05`, plus a fixed-ef frontier for each top-k.

The evaluator works with angular/cosine ANN-Benchmarks datasets such as
`glove-100-angular` and the published `lastfm-64-dot` file. Despite its name,
the ANN-Benchmarks LastFM HDF5 file stores a 65-dimensional MIPS-to-angular
transformation and declares `distance="angular"`; preserve that transformed
representation and evaluate it as cosine. Do not reinterpret those 65 columns
as raw inner-product vectors.

Build against the current release library without writing a binary into the
repository:

```bash
clang++ -std=c++17 -O3 -march=native -Iinclude \
    poc/adaptive_ef/cosine_matrix_eval.cpp \
    -Lbuild-release/src -lvsag \
    -Wl,-rpath,"$PWD/build-release/src" \
    -o /tmp/cosine_matrix_eval
```

The data directory must contain `train.fbin`, `test.fbin`, and `gt.ibin`. Each
file starts with signed 32-bit row and column counts followed by a row-major
payload (`float` for fbin, `int32_t` for ibin). Run all queries, or pass an
optional query count:

```bash
/tmp/cosine_matrix_eval /path/to/data_bin /path/to/base.idx \
    /tmp/adaptive.idx /tmp/matrix.csv 5000
```

Pass `query_count` and `only_topk` without `only_fixed_ef` to evaluate only the
adaptive targets; use `only_topk=0` for the complete top-k/recall matrix.
Supplying all three optional arguments evaluates one positive fixed-ef point
instead; it does not need to be one of the default frontier rungs. This makes it
possible to refine the fixed baseline to the first measured ef that matches an
adaptive pass fraction.

The evaluator validates all headers, payload sizes, dimensions, row counts, and
that ground truth has at least 100 neighbors. The CSV contains policy, top-k,
target recall, alpha, starting/fixed ef, QPS, pass fraction, average recall, and
mean `dist_cmp`, plus status and error columns. The calibration gate
integer-refines the sparse training frontier and directly measures a fixed-ef
policy that reaches at least the same held-out target pass rate. It fails closed
if measured pass counts or work violate the monotonic trend required by the
refinement, or if no fixed ef up to the cap reaches the adaptive pass count.
Costs at different pass rates are never compared. A rejected combination is
recorded as `rejected`; fixed-ef evaluation continues.
QPS times only `KnnSearch`; result checking and statistics parsing are outside
the timed region. The base index is never overwritten. Reuse the trained-index
path to skip both graph construction and adaptive-ef training, or choose a new
trained-index path after changing calibration code.
