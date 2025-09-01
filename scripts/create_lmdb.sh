#!/bin/bash
#SBATCH --job-name=train_tabasco_initial
#SBATCH --partition=goodarzilab_gpu_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=train_tabasco_%j.out
#SBATCH --error=train_tabasco_%j.err

# Print job information
echo "======================================"
echo "SLURM Job Information"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time Limit: $SLURM_TIMELIMIT"
echo "======================================"

# Load conda
echo "🔧 Setting up environment..."
CONDA_FOUND=false
for conda_path in ~/miniconda ~/miniconda3 ~/anaconda3 /opt/conda /opt/miniconda3; do
    if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
        echo "✅ Found conda at: $conda_path"
        source "$conda_path/etc/profile.d/conda.sh"
        CONDA_FOUND=true
        break
    fi
done

if [ "$CONDA_FOUND" = false ]; then
    echo "❌ Could not find conda installation"
    exit 1
fi

# Activate tabasco environment
echo "Activating tabasco environment..."
conda activate tabasco

# Set WandB API key
export WANDB_API_KEY="3b47e16a9067a52572c9e52c59f4692ddbe60a9f"
echo "✅ WandB API key configured"

# Verify environment
echo "Environment Verification:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo "======================================"

# Change to tabasco directory
cd /home/aidens/metagen/tabasco

# Check if data files exist
echo "Checking for .pt data files..."
ls -lh src/data/*.pt

echo ""
echo "======================================"
echo "Starting training (LMDB creation + initial epochs)..."
echo "======================================"
echo "Note: First run will create LMDB databases (~30-60 minutes)"
echo "Then will run 12 epochs with GPU monitoring"
echo ""

# Function to monitor GPU usage in background
monitor_gpu() {
    while true; do
        echo "[$(date +%H:%M:%S)] GPU Usage:"
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
        sleep 30
    done
}

# Start GPU monitoring in background
monitor_gpu > gpu_usage_${SLURM_JOB_ID}.log 2>&1 &
MONITOR_PID=$!
echo "📊 GPU monitoring started (PID: $MONITOR_PID)"

# Try larger batch size for better GPU utilization
# Start with 512, can increase based on GPU memory
BATCH_SIZE=512

echo "Configuration:"
echo "  Model: mild_geom (3.7M params)"
echo "  Batch size: $BATCH_SIZE"
echo "  Max epochs: 12"
echo "  Validation frequency: every 3 epochs"
echo ""

# Run training following the authors' approach
python src/train.py \
    experiment=mild_geom \
    trainer=gpu \
    trainer.max_epochs=12 \
    trainer.check_val_every_n_epoch=3 \
    trainer.log_every_n_steps=10 \
    datamodule.num_workers=15 \
    datamodule.batch_size=$BATCH_SIZE \
    logger.wandb.name="tabasco_initial_${SLURM_JOB_ID}" \
    logger.wandb.tags="[initial,lmdb_creation,gpu_monitoring]" \
    hydra.verbose=false

# Stop GPU monitoring
kill $MONITOR_PID 2>/dev/null

# Check exit status
exit_code=$?
echo "======================================"
echo "LMDB creation completed with exit code: $exit_code"
echo "Job completed at: $(date)"

# Check if LMDB files were created
echo ""
echo "Checking for created LMDB files:"
find src/data -name "*.lmdb" -o -name "*_stats.yaml" 2>/dev/null | while read f; do
    echo "  ✅ $(basename $f) - $(du -h $f | cut -f1)"
done

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "📊 GPU Usage Summary:"
    echo "Check gpu_usage_${SLURM_JOB_ID}.log for detailed GPU utilization"
    tail -5 gpu_usage_${SLURM_JOB_ID}.log 2>/dev/null || echo "No GPU log found"
    echo ""
    echo "🔍 Key Metrics to Check in WandB:"
    echo "  - train/loss: Should decrease over epochs"
    echo "  - val/loss: Should decrease (watch for overfitting)"
    echo "  - val/validity: % of valid molecules generated"
    echo "  - val/uniqueness: % unique molecules"
    echo "  - GPU memory usage: Check if we can increase batch size"
    echo ""
    echo "📈 WandB Dashboard:"
    echo "  https://wandb.ai/your-entity/tabasco"
    echo ""
    echo "💡 Optimization Tips Based on GPU Usage:"
    awk -F', ' '{
        gpu_util+=$1; mem_util+=$2; mem_used+=$3; mem_total+=$4; count++
    } END {
        if (count > 0) {
            avg_gpu = gpu_util/count
            avg_mem = mem_util/count
            avg_used = mem_used/count/1024
            avg_total = mem_total/count/1024
            printf "  Avg GPU Utilization: %.1f%%\n", avg_gpu
            printf "  Avg Memory Used: %.1f GB / %.1f GB\n", avg_used, avg_total
            if (avg_gpu < 80) print "  ⚠️  Low GPU usage - consider increasing batch size"
            if (avg_used < avg_total * 0.7) print "  💡 Memory available - try batch_size=1024 or higher"
        }
    }' gpu_usage_${SLURM_JOB_ID}.log 2>/dev/null || echo "  Unable to calculate averages"
else
    echo "❌ Training failed with exit code: $exit_code"
    echo "Check the error log for details"
fi

echo "======================================"