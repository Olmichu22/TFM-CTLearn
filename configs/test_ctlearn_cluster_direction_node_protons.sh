#!/bin/bash
#SBATCH -J ctlearn_job            # Nombre del trabajo
#SBATCH -o slurm_outputs/test/result_test_direction_%j.out          # Archivo de salida (stdout)
#SBATCH --error=slurm_outputs/test/result_test_direction_%j.err     # Archivo de errores (stderr)
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
 --config_file "/fefs/home/olmo.arqueropeinazo/TFM/TFM-CTLearn/logs/trial_complete_data_direction_node/20240722_203850_config.yml" \
 -o "/fefs/home/olmo.arqueropeinazo/TFM/TFM-CTLearn/logs/trial_complete_data_direction_node/" \
 -t LST_LST_LSTCam \
 -b 64 \
 --clean \
 --log_to_file \
 --mode predict \
 --reco direction

# Desactivar el entorno
conda deactivate
