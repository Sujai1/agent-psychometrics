#!/bin/bash
# Screen tasks with k=1 to find divergent tasks (gpt-5.2 passes, o4-mini fails)
# Runs both models in parallel on batches of 5 tasks

set -e

cd /Users/chrisge/Downloads/model_irt
source .venv/bin/activate

# 20 random tasks from medium difficulty range, split into 4 batches
BATCH1="pydata__xarray-3305,sphinx-doc__sphinx-9673,django__django-17087,django__django-15128,pydata__xarray-2905"
BATCH2="django__django-13279,pytest-dev__pytest-10081,matplotlib__matplotlib-23412,sympy__sympy-23534,pytest-dev__pytest-7432"
BATCH3="django__django-12304,django__django-14122,django__django-12039,django__django-16032,django__django-13012"
BATCH4="django__django-13343,sphinx-doc__sphinx-8120,django__django-14787,pydata__xarray-6461,django__django-12125"

run_batch() {
    local batch_num=$1
    local tasks=$2

    echo "=== Batch $batch_num: Running both models in parallel ==="

    # Run both models in parallel
    python -m experiment_pass_at_k.run_pass_k \
        --model openai/gpt-5.2 \
        --task_ids "$tasks" \
        --k 1 \
        --parallel-tasks 5 &
    PID1=$!

    python -m experiment_pass_at_k.run_pass_k \
        --model openai/o4-mini \
        --task_ids "$tasks" \
        --k 1 \
        --parallel-tasks 5 &
    PID2=$!

    # Wait for both to complete
    wait $PID1 $PID2

    echo "=== Batch $batch_num complete ==="
    echo ""
}

echo "=============================================="
echo "Screening 20 tasks with k=1 (parallel batches)"
echo "=============================================="
echo ""

run_batch 1 "$BATCH1"
run_batch 2 "$BATCH2"
run_batch 3 "$BATCH3"
run_batch 4 "$BATCH4"

echo ""
echo "Cleaning up Docker state..."
docker stop $(docker ps -q) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true
docker system prune -f 2>/dev/null || true

echo ""
echo "=============================================="
echo "Screening complete!"
echo "=============================================="
