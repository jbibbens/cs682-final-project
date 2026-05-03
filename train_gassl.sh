#!/bin/bash
#SBATCH --job-name=train_gassl
#SBATCH --account=pi_shenoy_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/saddy_umass_edu/cs682-final-project/logs/train_gassl_%j.out
#SBATCH --error=/home/saddy_umass_edu/cs682-final-project/logs/train_gassl_%j.err

PROJ=/home/saddy_umass_edu/cs682-final-project
PYTHON=$PROJ/.venv/bin/python

$PYTHON $PROJ/train_gassl.py \
    --train_dir       $PROJ/satellite_imagery_collection/data_dir/train \
    --val_dir         $PROJ/satellite_imagery_collection/data_dir/val \
    --labels_dir      $PROJ/labels \
    --gassl_checkpoint $PROJ/checkpoints/gassl_mocov2_tp_resnet50.pth.tar \
    --batch_size      16 \
    --lr              1e-4 \
    --wd              1e-4 \
    --dropout         0.2 \
    --epochs          100 \
    --warmup_epochs   5 \
    --early_stop      10 \
    --output          $PROJ/checkpoints/best_model_gassl.pth
