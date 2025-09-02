#!/bin/bash

# --- 配置 ---
# 可执行的评估程序路径
export CFLAGS="-g -fno-omit-frame-pointer"
export CXXFLAGS="-g -fno-omit-frame-pointer"
make release
EVAL_BINARY="./build-release/tools/eval/eval_performance"
./build-release/tools/eval/eval_performance .github/build_index.yml

bash scripts/download_annbench_datasets.sh
# FlameGraph 脚本所在的目录
FLAMEGRAPH_DIR="./FlameGraph"

# 所有测试配置文件的列表 (相对于项目根目录)
TEST_CONFIGS=(
    ".github/fp32.yml"
    ".github/sq8.yml"
    ".github/sq8_uniform.yml"
    ".github/rabitq.yml"
    ".github/pq.yml"
    ".github/pqfs.yml"
)

# --- 脚本主体 ---

git clone https://github.com/brendangregg/FlameGraph.git

# 检查评估程序是否存在
if [ ! -f "$EVAL_BINARY" ]; then
    echo "错误: 评估程序 '$EVAL_BINARY' 未找到."
    echo "请确保您已经编译了项目."
    exit 1
fi

# 创建一个带时间戳的主输出目录，用于存放本次所有测试的结果
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="perf_results/${TIMESTAMP}"
mkdir -p "$BASE_OUTPUT_DIR"
echo "所有性能测试结果将保存在: ${BASE_OUTPUT_DIR}"
echo ""

# 循环执行每一个测试配置
for config_file in "${TEST_CONFIGS[@]}"; do
    TEST_NAME=$(basename "$config_file" .yml)
    echo "============================================================"
    echo "开始测试: ${TEST_NAME}"
    echo "============================================================"

    # 为当前测试创建一个独立的子目录
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TEST_NAME}"
    mkdir -p "$OUTPUT_DIR"

    # 要执行的命令
    CMD="$EVAL_BINARY $config_file"

    # --- 1. 使用 perf stat 收集缓存和流水线指标 ---
    STAT_OUTPUT_FILE="${OUTPUT_DIR}/perf_stat.log"
    echo "--> 正在收集统计数据 (缓存, IPC, ...)，结果保存至 ${STAT_OUTPUT_FILE}"
    # 选择关键事件：缓存引用/未命中，分支预测，指令数，周期数
    sudo perf stat -a \
    -e inst_retired \
    -e cycles \
    -e stall_backend_mem \
    -e l3d_cache_refill \
    -e dtlb_walk \
    -e sve_inst_spec \
    -e sve_pred_empty_spec \
    -o "$STAT_OUTPUT_FILE" \
    -- $CMD
    echo "--> 统计数据收集完成."
    echo ""


    # --- 2. 使用 perf record 收集火焰图数据 ---
    PERF_DATA_FILE="${OUTPUT_DIR}/perf.data"
    FLAMEGRAPH_FILE="${OUTPUT_DIR}/${TEST_NAME}_flamegraph.svg"
    echo "--> 正在记录火焰图数据，结果保存至 ${PERF_DATA_FILE}"

    # -F 99: 采样频率 99Hz. -g: 记录调用栈
    sudo perf record -F 99 -g -o "$PERF_DATA_FILE" -- $CMD
    echo "--> 火焰图数据记录完成."
    echo ""


    # --- 3. 生成火焰图 ---
    echo "--> 正在生成火焰图，结果保存至 ${FLAMEGRAPH_FILE}"
    # 使用 perf script 读取数据，然后通过管道传给 FlameGraph 工具
    sudo perf script -i "$PERF_DATA_FILE" | \
        "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" | \
        "$FLAMEGRAPH_DIR/flamegraph.pl" > "$FLAMEGRAPH_FILE"
    echo "--> 火焰图生成完毕."
    echo ""

    # 清理临时的 perf.data 文件，可选
    # sudo rm "$PERF_DATA_FILE"

    echo "测试 ${TEST_NAME} 完成. 结果位于: ${OUTPUT_DIR}"
    echo ""
done

echo "============================================================"
echo "所有性能测试已全部完成！"
echo "总结果目录: ${BASE_OUTPUT_DIR}"
echo "============================================================"