# train examples for nuscenes

# perldiff 256x384, trained by bs2x8, iter 60000
export TOKENIZERS_PARALLELISM=false
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
CUDA_VISIBLE_DEVICES="0"
OMP_NUM_THREADS=16 torchrun \
            --nproc_per_node=1 main.py \
            --training \
            --yaml_file=configs/nusc_text.yaml   \
            --batch_size=1 \
            --name=nusc_train_256x384_perldiff_bs2x8 \
            --guidance_scale_c=5 \
            --step=50 \
            --official_ckpt_name=sd-v1-4.ckpt \
            --total_iters=60000 \
            --save_every_iters=6000 \
            --val_ckpt_name=None \
         
            