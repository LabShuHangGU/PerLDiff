# generate validation examples for kitti

# perldiff 384x1280, trained by bs1x8, iter 60000

         
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="8,9" OMP_NUM_THREADS=16 torchrun \
            --nproc_per_node=2 main.py \
            --validation \
            --yaml_file=configs/kitti_text.yaml   \
            --batch_size=2 \
            --name=kitti_test_384x1280_perldiff_bs1x8 \
            --guidance_scale_c=5 \
            --step=50 \
            --official_ckpt_name=sd-v1-4.ckpt \
            --total_iters=60000 \
            --save_every_iters=6000 \
            --val_ckpt_path=None \
            
