#!/bin/bash
#SBATCH -J ctlearn_job            # Nombre del trabajo
#SBATCH -o slurm_outputs/test/result_test_type_%j.out          # Archivo de salida (stdout)
#SBATCH --error=slurm_outputs/train/result_test_type_%j.err     # Archivo de errores (stderr)
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
ctlearn \
 -d TRN \
 -i /fefs/aswg/workspace/tjark.miener/DeepCrab_new/DL1/LSTProd2/TestDataset/Electrons/ -p "electron_theta_10.0_az_102.199*" \
 -y /fefs/home/olmo.arqueropeinazo/DL2/data/ \
 -o "/fefs/home/olmo.arqueropeinazo/TFM/TFM-CTLearn/logs/trial_complete_data_type_node/" \
 -t LST_LST_LSTCam \
 -b 64 \
 --clean \
 --log_to_file \
 --mode predict \
 --reco type

# Desactivar el entorno
conda deactivate
