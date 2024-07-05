#!/bin/bash
#SBATCH -J ctlearn_job            # Nombre del trabajo
#SBATCH -o result_%j.out          # Archivo de salida (stdout)
#SBATCH --error=result_%j.err     # Archivo de errores (stderr)
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


# Comando para ejecutar ctlearn
ctlearn -d TRN -i "/fefs/aswg/workspace/tjark.miener/DeepCrab/R1DL1/LSTProd2/TrainingDataset/GammaDiffuse/dec_2276/theta_16.087_az_108.090/" "/fefs/aswg/workspace/tjark.miener/DeepCrab/R1DL1/LSTProd2/TrainingDataset/Protons/dec_2276/theta_16.087_az_108.090/" -o "/home/olmo.arqueropeinazo/data/temp/logs/trial_complete_data/" -z 50 -e 15 -t LST_LST_LSTCam -b 24 -r type

# Desactivar el entorno
conda deactivate
