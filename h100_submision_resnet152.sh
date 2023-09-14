#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu22
#SBATCH --reservation=gpu-h100
#SBATCH --mail-type=ALL
#SBATCH --mem=82G
#SBATCH --mail-user=n.musembi@sheffield.ac.uk
#SBATCH --output=resnet50_bs_%x_%j.out

#parameters from the bash call
batchSize=$1
jobID=$SLURM_JOB_ID
model=$2
runNumber=$3
echo "==========================="
echo $model
echo "==========================="
eval {mkdir,cd}\ $model"_"$batchSize\;

#load the modules cuda, cudnn and conda
module load cuDNN/8.6.0.163-CUDA-11.8.0
module load Anaconda3/2022.10
echo $hostname
#you need this as whoevere coded cuda looks in defauult plataces like /usr/local . wtf???
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME


#activate conda environment
source activate tensorflow

#run the benchmark
#python ../benchmark_resnet_152_or_50.py -b $batchSize -j $jobID -m $model -r $runNumber
python ../alter_benchmark_resnet_152_or_50.py -b $batchSize -j $jobID -m $model -r $runNumber

#To run the code type the following
#   batch=512
#   sbatch --job-name=$batch  h100_submision_resnet50.sh $batch
mv ../*${jobID}* "resnet${model}_bs_${batchSize}_$SLURM_JOB_ID.out"
