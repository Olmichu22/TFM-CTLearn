#!/bin/bash
#SBATCH -J ctlearn_job            # Nombre del trabajo
#SBATCH -o result_%j.out          # Archivo de salida (stdout)
#SBATCH --error=result_%j.err     # Archivo de errores (stderr)
#SBATCH -t 00:30:00               # Tiempo máximo de ejecución (hh:mm:ss)
#SBATCH -A gpu
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ctlearn-tjark

module load cuda/11.1
export LD_LIBRARY_PATH=/home/olmo.arqueropeinazo/miniconda3/envs/ctlearn-tjark/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Comando para ejecutar ctlearn
ctlearn -d TRN -i "/fefs/aswg/workspace/tjark.miener/DeepCrab/R1DL1/LSTProd2/TrainingDataset/GammaDiffuse/dec_2276/theta_16.08_az_108.090/" "/fefs/aswg/workspace/tjark.miener/DeepCrab/R1DL1/LSTProd2/TrainingDataset/Protons/dec_2276/theta_16.087_az_108.090/" -o "/home/olmo.arqueropeinazo/data/temp/logs/trial_complete_data" -z 50 -e 15 -t LST_LST_LSTCam -b 24 -r type

# Desactivar el entorno
conda deactivate
