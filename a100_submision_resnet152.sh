#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=gpu22
#SBATCH --mail-type=ALL
#SBATCH --mem=82G
#SBATCH --mail-user=n.musembi@sheffield.ac.uk
#SBATCH --output=resnet50_bs_%x_%j.out

#parameters from the bash call
batchSize=$1
jobID=$SLURM_JOB_ID
model=$2
echo "==========================="
echo $model
echo "==========================="
eval {mkdir,cd}\ "a100_resnet"$model"_batch"$batchSize\;

#load the modules cuda, cudnn and conda
module load cuDNN/8.6.0.163-CUDA-11.8.0
module load Anaconda3/2022.10

#Tells tensor flow where the cuda libraries are
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME


#activate conda environment
source activate tensorflow

#run the benchmark
python ../benchmark_resnet_152_or_50.py -b $batchSize -j $jobID -m $model


mv ../*${jobID}* "resnet${model}_bs_${batchSize}_$SLURM_JOB_ID.out"
