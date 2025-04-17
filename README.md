# replicode-Vision-Transformer-and-MLP-Mixer-Architectures

cd "D:\My Projects\demo"
python310 --version # check python version and if it was added to system environment path successfully
nvidia-smi # check gpu
python310 -m venv cuda # create virtual environment named 'cuda'
cuda\Scripts\activate # activate virtual environment
git clone https://github.com/google-research/vision_transformer.git
cd vision_transformer
pip install -r vit_jax/requirements.txt # if using GPU
