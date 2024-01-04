#!/bin/bash
#SBATCH -t 8:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=32G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 1-5
#SBATCH -o score_train_gmm_ansatz_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e score_train_gmm_ansatz_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--shape spiral --train_pnts 20  --lr 0.01 --gmm_components 1 2 3 4 5 6 7 8 10 12 14 16 18 20 24 28 32 36 40 45 50 64 96 128 192 256 384 512 1024
--shape spiral --train_pnts 50  --lr 0.01 --gmm_components 1 2 3 4 5 6 7 8 10 12 14 16 18 20 24 28 32 36 40 45 50 64 96 128 192 256 384 512 1024
--shape ring --train_pnts 5  --lr 0.01 --gmm_components 1 2 3 4 5 6 7 8 10 12 14 16 18 20 24 28 32 36 40 45 50 64 96 128 192 256 384 512 1024
--shape ring --train_pnts 10  --lr 0.01 --gmm_components 1 2 3 4 5 6 7 8 10 12 14 16 18 20 24 28 32 36 40 45 50 64 96 128 192 256 384 512 1024
--shape ring --train_pnts 20  --lr 0.01 --gmm_components 1 2 3 4 5 6 7 8 10 12 14 16 18 20 24 28 32 36 40 45 50 64 96 128 192 256 384 512 1024
'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python/3.10.9-fasrc01
module load cuda cudnn
mamba activate torch

# run code
cd /n/home12/binxuwang/Github/DiffusionMemorization
python core/score_ansatz_NN_ris_CLI.py $param_name
