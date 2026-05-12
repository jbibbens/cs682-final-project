#!/bin/bash
#SBATCH --job-name=eval_gassl
#SBATCH --account=pi_shenoy_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/saddy_umass_edu/cs682-final-project/logs/eval_gassl_%j.out
#SBATCH --error=/home/saddy_umass_edu/cs682-final-project/logs/eval_gassl_%j.err

PROJ=/home/saddy_umass_edu/cs682-final-project
PYTHON=$PROJ/.venv/bin/python

$PYTHON $PROJ/eval_poverty.py \
    --test_dir          $PROJ/satellite_imagery_collection/data_dir/test \
    --labels_dir        $PROJ/labels \
    --backbone          gassl \
    --gassl_checkpoint  $PROJ/checkpoints/gassl_mocov2_tp_resnet50.pth.tar \
    --model_checkpoint  $PROJ/checkpoints/best_model_gassl.pth \
    --batch_size        16 \
    --dropout           0.2
