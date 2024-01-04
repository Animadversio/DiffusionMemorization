#!/bin/bash
#SBATCH -t 8:00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=32G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1
#SBATCH --array 6-10
#SBATCH -o score_train_mlp_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e score_train_mlp_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--shape spiral --train_pnts 20 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 
--shape spiral --train_pnts 50 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 
--shape ring --train_pnts 5 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 
--shape ring --train_pnts 10 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 
--shape ring --train_pnts 20 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 
--shape spiral --train_pnts 20 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 --lr_scaling
--shape spiral --train_pnts 50 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 --lr_scaling
--shape ring --train_pnts 5 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 --lr_scaling
--shape ring --train_pnts 10 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 --lr_scaling
--shape ring --train_pnts 20 --time_embed_dim 16 32 64 128 --mlp_depth 2 3 4 6 8 --mlp_width 8 16 32 64 128 256 512 1024 2048 4096 --lr_scaling
'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"

# load modules
module load python/3.10.9-fasrc01
module load cuda cudnn
mamba activate torch

# run code
cd /n/home12/binxuwang/Github/DiffusionMemorization
python core/score_MLP_NN_ris_CLI.py $param_name
