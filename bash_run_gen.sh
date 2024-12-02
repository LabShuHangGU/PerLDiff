# generate validation examples for nuscenes

# perldiff 256x384, trained by bs2x8, iter 60000
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=16 torchrun \
            --nproc_per_node=2 main.py \
            --generation \
            --yaml_file=configs/nusc_text_with_path.yaml   \
            --batch_size=4 \
            --name=nusc_test_256x384_perldiff_bs2x8 \
            --guidance_scale_c=5 \
            --step=50 \
            --official_ckpt_name=sd-v1-4.ckpt \
            --total_iters=60000 \
            --save_every_iters=6000 \
            --val_ckpt_name=DATA/perldiff_256x384_lambda_5_bs2x8_model_checkpoint_00060000.pth \
            --gen_path=val_ddim50w5_256x384_perldiff_bs2x8 \

         