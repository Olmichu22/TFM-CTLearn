#!/bin/bash
#SBATCH -J ctlearn_job_PC_type            # Nombre del trabajo
#SBATCH -o slurm_outputs/train/result_train_type_PC_%j.out # Archivo de salida (stdout)
#SBATCH --error=slurm_outputs/train/result_train_type_PC_%j.err     # Archivo de errores (stderr)
#SBATCH -A gpu
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ctlearn-pc
export LD_LIBRARY_PATH=/fefs/aswg/workspace/olmo.arqueropeinazo/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDNN_PATH=/fefs/aswg/workspace/olmo.arqueropeinazo/local/cuda/lib64/libcudnn.so
export LD_LIBRARY_PATH=/home/olmo.arqueropeinazo/miniconda3/envs/ctlearn-pc/lib:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"


# Comando para ejecutar ctlearn
ctlearn --config_file "/fefs/home/olmo.arqueropeinazo/TFM/TFM-CTLearn/configs/ctlearn config/PN_complete_config.yml" -i /fefs/aswg/workspace/tjark.miener/DeepCrab_new/DL1/LSTProd2/TrainingDataset/GammasDiffuse/ /fefs/aswg/workspace/tjark.miener/DeepCrab_new/DL1/LSTProd2/TrainingDataset/Protons/ -p "*16.087_az_108.090*" "proton_theta_16.087_az_251.910*" "proton_theta_9.579_az_126.888*" "proton_theta_23.161_az_99.261*" -o "/fefs/home/olmo.arqueropeinazo/TFM/TFM-CTLearn/logs/pc_complete_data_type_node/" -z 50 -e 8 -t LST_LST_LSTCam -b 64 --log_to_file -r type 

# Desactivar el entorno
conda deactivate
