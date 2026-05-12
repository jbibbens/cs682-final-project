#!/bin/bash
#SBATCH --job-name=satellite_download
#SBATCH --account=pi_shenoy_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/saddy_umass_edu/cs682-final-project/logs/download_%j.out
#SBATCH --error=/home/saddy_umass_edu/cs682-final-project/logs/download_%j.err

cd /home/saddy_umass_edu/cs682-final-project/satellite_imagery_collection

/home/saddy_umass_edu/cs682-final-project/.venv/bin/python download_image.py
