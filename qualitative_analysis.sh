#!/bin/bash
#SBATCH --job-name=qualitative
#SBATCH --account=pi_shenoy_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/saddy_umass_edu/cs682-final-project/logs/qualitative_%j.out
#SBATCH --error=/home/saddy_umass_edu/cs682-final-project/logs/qualitative_%j.err

PROJ=/home/saddy_umass_edu/cs682-final-project
PYTHON=$PROJ/.venv/bin/python

echo "=== GASSL qualitative analysis ==="
$PYTHON $PROJ/qualitative_analysis.py \
    --backbone           gassl \
    --backbone_checkpoint $PROJ/checkpoints/gassl_mocov2_tp_resnet50.pth.tar \
    --model_checkpoint   $PROJ/checkpoints/best_model_gassl.pth \
    --test_dir           $PROJ/satellite_imagery_collection/data_dir/test \
    --labels_dir         $PROJ/labels \
    --n_tracts           6

echo "=== SatDINO qualitative analysis ==="
$PYTHON $PROJ/qualitative_analysis.py \
    --backbone           satdino \
    --backbone_checkpoint $PROJ/checkpoints/satdino-vit_small-16.pth \
    --model_checkpoint   $PROJ/checkpoints/best_model.pth \
    --test_dir           $PROJ/satellite_imagery_collection/data_dir/test \
    --labels_dir         $PROJ/labels \
    --n_tracts           6
