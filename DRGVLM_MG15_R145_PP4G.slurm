#!/bin/bash
#SBATCH --job-name=DRGVLM_MG15_R145_PP5G
#SBATCH --partition=8gpus
#SBATCH --account=MST114560
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yasaisenkiri@gmail.com

# CUDA
ml nvhpc-hpcx-cuda12/24.7

# PRINT
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Submit host: $SLURM_SUBMIT_HOST"
echo "Allocated node(s): $SLURM_NODELIST"
echo "Num nodes: $SLURM_JOB_NUM_NODES"
echo "Num GPUs: $SLURM_GPUS_PER_NODE"

# ACTIVATE
source /home/misaka13/anaconda3/etc/profile.d/conda.sh
conda activate MVLM_CLEE

# python /home/misaka13/R27_MVLM_v3.18.5/DRepGenVLM/R27_MVLM_trainDRGVLM_v0.0.py --cfg-path /home/misaka13/R27_MVLM_v3.18.5/DRepGenVLM/DRepGenVLM/projects/MG15_basic_R145_PP4G_config_hpc.json
python /home/misaka13/R27_MVLM_v3.18.5/DRepGenVLM/R27_MVLM_trainDRGVLM_v0.0.py --latest-checkpoint-path /work/misaka13/R27_MVLM_v3.19/checkpoints/260805140136 --use-this-dir
