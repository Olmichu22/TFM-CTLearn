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
ctlearn --config_file "/fefs/home/olmo.arqueropeinazo/TFM/TFM-CTLearn/configs/ctlearn config/PN_complete_config.yml" -i "/fefs/aswg/workspace/tjark.miener/DeepCrab/R1DL1/LSTProd2/TrainingDataset/GammaDiffuse/dec_2276/theta_16.087_az_108.090/" "/fefs/aswg/workspace/tjark.miener/DeepCrab/R1DL1/LSTProd2/TrainingDataset/Protons/dec_2276/theta_16.087_az_108.090/" -o "/fefs/home/olmo.arqueropeinazo/TFM/TFM-CTLearn/logs/pc_complete_data_type_four_edgeconv/" -z 50 -e 15 -t LST_LST_LSTCam -b 24 --log_to_file -r type 

# Desactivar el entorno
conda deactivate
