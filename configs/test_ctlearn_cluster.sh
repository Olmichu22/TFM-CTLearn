#!/bin/bash
#SBATCH -J ctlearn_job            # Nombre del trabajo
#SBATCH -o result_test%j.out          # Archivo de salida (stdout)
#SBATCH --error=result_test%j.err     # Archivo de errores (stderr)
#SBATCH -A gpu
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ctlearn-cluster
export LD_LIBRARY_PATH=/fefs/aswg/workspace/olmo.arqueropeinazo/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDNN_PATH=/fefs/aswg/workspace/olmo.arqueropeinazo/local/cuda/lib64/libcudnn.so
export LD_LIBRARY_PATH=/home/olmo.arqueropeinazo/miniconda3/envs/ctlearn-cluster/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

ctlearn --config /home/olmo.arqueropeinazo/data/temp/logs/trial_complete_data/predicting_config_def.yml --reco type --tel_types LST_LST_LSTCam --output "/home/olmo.arqueropeinazo/data/temp/logs/trial_complete_data/" --mode predict

# Desactivar el entorno
conda deactivate
