conda create -n perldiff python=3.8 -y
conda activate perldiff

## we use pytorch1.12.0+cu113 in V100, CUDA 11.3

pip install albumentations==0.4.3 opencv-python pudb==2019.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 
pip install pytorch-lightning==1.4.2 omegaconf==2.1.1 test-tube>=0.7.5 streamlit>=0.73.1 einops==0.3.0 torch-fidelity==0.3.0 timm
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install protobuf torchmetrics==0.6.0 transformers==4.19.2 kornia==0.5.8 ftfy regex tqdm

# git+https://github.com/openai/CLIP.git 
cd ./CLIP
pip install .
cd ../
# pip install git+https://github.com/openai/CLIP.git
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0 scikit-image==0.18.0 ipdb gradio


# use "-i https://mirrors.aliyun.com/pypi/simple/" for pip install will be faster