#!/bin/bash

# ==============================================================================
# PaMoRL vs RWKV v7 Atari 100k Benchmark 自动化脚本
# ==============================================================================

trap "echo '正在关闭所有实验...'; pkill -P $$; exit" SIGINT SIGTERM

# 1. 定义 26 个标准 Atari 100k 游戏列表
GAMES=(
    "Alien" "Amidar" "Assault" "Asterix" "BankHeist" "BattleZone" "Boxing" "Breakout"
    "ChopperCommand" "CrazyClimber" "DemonAttack" "Freeway" "Frostbite" "Gopher"
    "Hero" "Jamesbond" "Kangaroo" "Krull" "KungFuMaster" "MsPacman" "Pong"
    "PrivateEye" "Qbert" "RoadRunner" "Seaquest" "UpNDown"
)

# 2. 设置使用的 GPU 编号
GPU_A=3
GPU_B=2

LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR  # 创建日志根目录

# 3. 任务分配：将 26 个游戏平分为两组
GAMES_FOR_GPU_A=("${GAMES[@]:0:13}")  # 前 13 个
GAMES_FOR_GPU_B=("${GAMES[@]:13:13}") # 后 13 个

# 4. 定义单个 GPU 的执行逻辑函数
run_gpu_group() {
    local gpu_id=$1
    shift
    local task_list=("$@")

    for game in "${task_list[@]}"; do
        local env="ALE/${game}-v5"
        # 为当前游戏创建独立的日志子目录
        local game_log_dir="$LOG_DIR/${game}"

        echo "--------------------------------------------------"
        echo "[GPU $gpu_id] 正在开始游戏: $game"
        echo "--------------------------------------------------"

        # --- A. 运行基准组 (PSSM Baseline) ---
        local run_name_base="${game}-baseline-pssm-100k"
        echo "[GPU $gpu_id] [PSSM] 开始训练..."
        # 使用 2>&1 | tee 将输出同时显示在屏幕并记录到文件
        CUDA_VISIBLE_DEVICES=$gpu_id python -u -O train.py \
            -n "$run_name_base" \
            -seed 0 \
            -config_path "config_files/PWM.yaml" \
            -env_name "$env" \
            -device "cuda:0" \
            2>&1 | tee "$game_log_dir/pssm_train.log"

        echo "[GPU $gpu_id] [PSSM] 开始评估..."
        CUDA_VISIBLE_DEVICES=$gpu_id python -u -O eval.py \
            -env_name "$env" \
            -run_name "$run_name_base" \
            -seed 0 \
            -config_path "config_files/PWM.yaml" \
            -device "cuda:0" \
            2>&1 | tee "$game_log_dir/pssm_eval.log"

        # --- B. 运行实验组 (RWKV v7 Ours) ---
        local run_name_rwkv="${game}-ours-rwkv-100k"
        echo "[GPU $gpu_id] [RWKV] 开始训练..."
        CUDA_VISIBLE_DEVICES=$gpu_id python -u -O train.py \
            -n "$run_name_rwkv" \
            -seed 0 \
            -config_path "config_files/PWM.yaml" \
            -env_name "$env" \
            -device "cuda:0" \
            --use_rwkv \
            2>&1 | tee "$game_log_dir/rwkv_train.log"

        echo "[GPU $gpu_id] [RWKV] 开始评估..."
        CUDA_VISIBLE_DEVICES=$gpu_id python -u -O eval.py \
            -env_name "$env" \
            -run_name "$run_name_rwkv" \
            -seed 0 \
            -config_path "config_files/PWM.yaml" \
            -device "cuda:0" \
            --use_rwkv \
            2>&1 | tee "$game_log_dir/rwkv_eval.log"

        echo "[GPU $gpu_id] 已完成游戏 $game 的所有测试。"
    done
}

# 5. 并行启动两个 GPU 队列
echo "启动并行实验：GPU $GPU_A 负责组 A，GPU $GPU_B 负责组 B。"

run_gpu_group $GPU_A "${GAMES_FOR_GPU_A[@]}" &
PID_A=$!

run_gpu_group $GPU_B "${GAMES_FOR_GPU_B[@]}" &
PID_B=$!

# 等待所有后台任务完成
wait $PID_A
wait $PID_B

echo "=================================================="
echo "所有 26 个游戏的对比实验已全部完成！日志已保存至 $LOG_DIR 文件夹。"
echo "=================================================="