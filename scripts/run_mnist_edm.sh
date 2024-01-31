DATASET="mnist"
CHANNEL_MULT="1 2 3 4"
MODEL_CHANNELS="16"
ATTN_RESOLUTIONS="0"
LAYERS_PER_BLOCK="1"
## training
python -u train_edm.py --dataset $DATASET \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 25000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 500 --train_progress_bar \
      --save_images_step 5000 \
      --save_model_iters 500 


DATASET="gabor_prime"
CHANNEL_MULT="1 2 3 4"
MODEL_CHANNELS="16"
ATTN_RESOLUTIONS="0"
LAYERS_PER_BLOCK="1"
## training
python -u train_edm.py --dataset $DATASET \
      --dataset_root /n/home12/binxuwang/Datasets/gabor_prime --grayscale \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 10000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 500 --train_progress_bar \
      --save_images_step 500 \
      --save_model_iters 500 

DATASET="gabor_sf"
CHANNEL_MULT="1 2 3 4"
MODEL_CHANNELS="16"
ATTN_RESOLUTIONS="0"
LAYERS_PER_BLOCK="1"
## training
python -u train_edm.py --dataset $DATASET \
      --dataset_root /n/home12/binxuwang/Datasets/gabor_classic_sf --grayscale \
      --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
      --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
      --train_batch_size 128 --num_steps 10000 \
      --learning_rate 2e-4 --accumulation_steps 1 \
      --log_step 500 --train_progress_bar \
      --save_images_step 500 \
      --save_model_iters 500 



DATASET="cifar10"
# Define model architecture parameters
CHANNEL_MULT="1 2 2"
MODEL_CHANNELS="96"
ATTN_RESOLUTIONS="16"
LAYERS_PER_BLOCK="2"
## training
python -u train_edm.py --dataset $DATASET \
        --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
        --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
        --train_batch_size 128 --num_steps 1000000 \
        --learning_rate 2e-4 --accumulation_steps 1 \
        --log_step 500 --train_progress_bar \
        --save_images_step 2000 \
        --save_model_iters 5000


DATASET="cifar10"
# Define model architecture parameters
CHANNEL_MULT="1 2 2"
MODEL_CHANNELS="96"
ATTN_RESOLUTIONS="16"
LAYERS_PER_BLOCK="2"
## training
python -u train_edm.py --dataset $DATASET \
        --channel_mult $CHANNEL_MULT --model_channels $MODEL_CHANNELS \
        --attn_resolutions $ATTN_RESOLUTIONS --layers_per_block $LAYERS_PER_BLOCK \
        --train_batch_size 128 --num_steps 50000 \
        --learning_rate 2e-4 --accumulation_steps 1 \
        --log_step 500 --train_progress_bar \
        --save_images_step 500 \
        --save_model_iters 500



