#!/bin/bash
#SBATCH --job-name=1
#SBATCH --output=1-WS%j.out
#SBATCH --error=1-WS%j.err
export PYTHONUNBUFFERED=1
cd /HOME/scz0ru4/run/P11-DTU EEG-AAD/EDAN
/HOME/scz0ru4/run/conda_envs/torch/bin/python 1-WS.py
