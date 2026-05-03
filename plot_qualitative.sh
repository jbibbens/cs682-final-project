#!/bin/bash
#SBATCH --job-name=plot_qualitative
#SBATCH --account=pi_shenoy_umass_edu
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/saddy_umass_edu/cs682-final-project/logs/plot_qualitative_%j.out
#SBATCH --error=/home/saddy_umass_edu/cs682-final-project/logs/plot_qualitative_%j.err

PROJ=/home/saddy_umass_edu/cs682-final-project
PYTHON=$PROJ/.venv/bin/python

echo "=== GASSL ==="
$PYTHON $PROJ/plot_qualitative.py \
    --backbone  gassl \
    --test_dir  $PROJ/satellite_imagery_collection/data_dir/test \
    --labels_dir $PROJ/labels \
    --n_tracts  30

echo "=== SatDINO ==="
$PYTHON $PROJ/plot_qualitative.py \
    --backbone  satdino \
    --test_dir  $PROJ/satellite_imagery_collection/data_dir/test \
    --labels_dir $PROJ/labels \
    --n_tracts  30
