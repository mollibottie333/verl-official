#!/usr/bin/env bash
set -euo pipefail

# ================= Environment configuration =================
# 1) Rank is assigned by the platform (Head=0, Worker>0)
export RANK=${GEMINI_TASK_INDEX:-0}

# 2) Prefer the platform-provided head IP for MASTER_ADDR
export MASTER_ADDR=${GEMINI_IP_taskrole1_0:-"127.0.0.1"}
export MASTER_PORT=${GEMINI_TASK_PORT:-6379}

# 3) WORLD_SIZE should be the number of nodes
export WORLD_SIZE=${WORLD_SIZE:-2}

# Detect GPUs per node from env or system
detect_gpus_per_node() {
    if [ -n "${GPUS_PER_NODE:-}" ]; then
        echo "${GPUS_PER_NODE}"
        return
    fi
    if [ -n "${NGPUS_PER_NODE:-}" ]; then
        echo "${NGPUS_PER_NODE}"
        return
    fi
    if [ -n "${NUM_GPUS:-}" ]; then
        echo "${NUM_GPUS}"
        return
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        python3 - <<'PY'
import os
value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if not value:
    print(0)
else:
    devices = [d for d in value.split(",") if d.strip()]
    print(len(devices))
PY
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L | wc -l
        return
    fi
    echo 0
}

GPUS_PER_NODE="$(detect_gpus_per_node)"
if [ "${GPUS_PER_NODE}" -le 0 ]; then
    echo "[ERROR] Unable to detect GPUs per node."
    echo "Set GPUS_PER_NODE/NGPUS_PER_NODE/NUM_GPUS before launching."
    exit 1
fi

# Export common names used by recipes/configs
export GPUS_PER_NODE
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-$GPUS_PER_NODE}"
export NUM_GPUS="${NUM_GPUS:-$GPUS_PER_NODE}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$GPUS_PER_NODE}"
export NNODES="${NNODES:-$WORLD_SIZE}"
export NNODES_TRAIN="${NNODES_TRAIN:-$NNODES}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-$NNODES}"

# Ensure python connects to the existing Ray cluster
export RAY_ADDRESS="${RAY_ADDRESS:-${MASTER_ADDR}:${MASTER_PORT}}"

echo "================ NODE INFO ================"
echo "Node IP (Local): $(hostname -i)"
echo "MASTER_ADDR    : ${MASTER_ADDR}"
echo "MASTER_PORT    : ${MASTER_PORT}"
echo "RANK           : ${RANK}"
echo "WORLD_SIZE     : ${WORLD_SIZE}"
echo "GPUS_PER_NODE  : ${GPUS_PER_NODE}"
echo "NNODES_TRAIN   : ${NNODES_TRAIN}"
echo "NNODES_ROLLOUT : ${NNODES_ROLLOUT}"
echo "==========================================="

# Enter repo directory
REPO_DIR=${REPO_DIR:-/gfs/space/chatrl/users/jqli/code/verl-gemini}
cd "${REPO_DIR}" || exit 1

RECIPE_PATH=${RECIPE_PATH:-"${REPO_DIR}/recipe/fully_async/dapo_8b_math_fsdp2_8_8_mathv2.sh"}
if [ ! -f "${RECIPE_PATH}" ]; then
    echo "[ERROR] Recipe script not found: ${RECIPE_PATH}"
    echo "Set RECIPE_PATH to the correct training recipe."
    exit 1
fi

# ================= Node check helper (Python instead of grep) =================
check_ray_cluster_ready() {
    expected_nodes=$1
    python3 - <<PY
import ray
import time
import sys

try:
    ray.init(address='${MASTER_ADDR}:${MASTER_PORT}', ignore_reinit_error=True)
    alive_nodes = 0
    for _ in range(30):
        nodes = ray.nodes()
        alive_nodes = sum(1 for n in nodes if n.get('Alive'))
        if alive_nodes >= ${expected_nodes}:
            print(f'Ready: {alive_nodes}/{${expected_nodes}}')
            sys.exit(0)
        time.sleep(1)
    print(f'Waiting: {alive_nodes}/{${expected_nodes}}')
    sys.exit(1)
except Exception as e:
    print(f'Error connecting to Ray: {e}')
    sys.exit(1)
PY
}

# ================= Launch logic =================
if [ -z "${WORLD_SIZE}" ] || [ "${WORLD_SIZE}" -eq 1 ]; then
    echo "[Mode] Single-node training..."
    bash "${RECIPE_PATH}"
else
    if [ "${RANK}" = "0" ]; then
        echo "[Mode] Multi-node: Head Node (Rank 0)"

        ray start --head \
            --node-ip-address="${MASTER_ADDR}" \
            --port="${MASTER_PORT}" \
            --num-gpus="${GPUS_PER_NODE}" \
            --block &

        RAY_PID=$!
        echo "Ray Head started (pid=${RAY_PID}), waiting for workers..."
        sleep 10

        while true; do
            if check_ray_cluster_ready "${WORLD_SIZE}"; then
                echo ">>> All ${WORLD_SIZE} nodes are ready! Starting training..."
                break
            else
                echo ">>> Waiting for workers... (Ctrl+C to exit)"
                sleep 10
            fi
        done

        echo ">>> Executing training recipe..."
        bash "${RECIPE_PATH}"
    else
        echo "[Mode] Multi-node: Worker Node (Rank ${RANK})"
        sleep 10

        ray start --address="${MASTER_ADDR}:${MASTER_PORT}" \
            --num-gpus="${GPUS_PER_NODE}" \
            --block
    fi
fi
