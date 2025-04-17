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

# Prepare dataset.py with python
  # your_dataset.py（参考TFDS文档）
  import tensorflow_datasets as tfds
  class YourDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    def _split_generators(self, dl_manager):
      return [
        tfds.core.SplitGenerator(
          name=tfds.Split.TRAIN,
          gen_kwargs={"data_dir": "path/to/train"},
        ),
        tfds.core.SplitGenerator(
          name=tfds.Split.VALIDATION,
          gen_kwargs={"data_dir": "path/to/validation"},
        ),
      ]
    # ...

# regist datasets in input_pipeline.py
python -m vit_jax.main \
    --config=vit_jax/configs/vit.py:b16,your_dataset \
    --config.dataset=your_dataset \       # Cover dataset name
    --config.num_classes=10 \             # Cover class
    --config.image_size=384 \             # Cover size
    --config.batch=256 \
    --config.base_lr=0.03

# modify dataset parms
python -m vit_jax.main \
    --config=vit_jax/configs/vit.py:b16,your_dataset \
    --config.dataset=your_dataset \       # dataset name
    --config.num_classes=10 \             # classes number
    --config.image_size=384               # size

# modify training parms
python -m vit_jax.main \
    --config=vit_jax/configs/vit.py:b16,your_dataset \
    --config.batch=256 \             # training batch size
    --config.base_lr=0.03 \          # learning rate
    --config.total_steps=30000       # steps
# you can set following parms
  # config.batch = 512               # training batch size
  # config.eval_batch = 256          # evaluate batch size
  # config.base_lr = 0.01            # learning rate
  # config.total_steps = 20000       # steps
  # config.warmup_steps = 1000       # warmup steps
  # config.weight_decay = 0.03       # weight decay (L2 Regularization)

