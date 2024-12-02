export TMPDIR="./tmp"
CUDA_VISIBLE_DEVICES=1  python main_gradio.py 
            --yaml_file=configs/nusc_text.yaml   \
            --batch_size=1 \
            --name=nusc_test_256x384_perldiff_bs2x8 \
            --guidance_scale_c=5 \
            --step=50 \
            --official_ckpt_name=sd-v1-4.ckpt \
            --total_iters=60000 \
            --save_every_iters=6000 \
            --val_ckpt_name=DATA/perldiff_256x384_lambda_5_bs2x8_model_checkpoint_00060000.pth \