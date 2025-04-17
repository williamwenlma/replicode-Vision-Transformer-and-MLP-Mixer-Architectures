# replicode-Vision-Transformer-and-MLP-Mixer-Architectures

```bash
# Navigate to project directory
cd "D:\My Projects\demo"

# Verify Python version and PATH configuration
python310 --version

# Check GPU availability
nvidia-smi

# Create and activate virtual environment
python310 -m venv cuda
cuda\Scripts\activate

# Clone repository and install dependencies
git clone https://github.com/google-research/vision_transformer.git
cd vision_transformer
pip install -r vit_jax/requirements.txt  # GPU users only
