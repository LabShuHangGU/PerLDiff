# 1.Use MagicDrive Code

CUDA_VISIBLE_DEVICES=1 python tools/fid_score_384.py cfg \
  resume_from_checkpoint=./pretrained/SDv1.5mv-rawbox_2023-09-07_18-39_224x400 \
  fid.rootb=data/nuscenes/val_ddim50w5_256x384_perldiff_lambda_5_bs2x8/samples  

# 2.Use CleanFid # pip install clean-fid
#2.1 First to generate real samples
# python scripts/get_nusc_real_img.py
#2.2 Then to calculate FID by generated samples
# python cleanfid_test_fid.py ./val_ddim50w5_256x384_perldiff_lambda_5_bs2x8/samples ./samples_256x384/samples
