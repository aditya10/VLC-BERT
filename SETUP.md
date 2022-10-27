# Setup

### Clone the repo and install requirements:

```bash
https://github.com/aditya10/VLC-BERT.git
cd VL-BERT
```

### Install requirements:

```bash
# Option 1: use venv
virtualenv -p python3 --no-download vl-bert
source vl-bert/bin/activate 

# Option 2: use conda
conda create -n vl-bert python=3.6 pip
conda activate vl-bert

# Install PyTorch, torchvision (this is for CUDA 11.X, check nvidia-smi)
# pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install requirements
pip install Cython
pip install -r requirements.txt

# Optional: Install Nvidia Apex (I skipped this because VQA config uses FP32)
git clone https://github.com/jackroos/apex
cd ./apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Initialize
./scripts/init.sh # if init.sh is not working, it means that ur CUDA versions did not match

# Make dir for checkpoints
mkdir ckpts
```

### Download data

Follow [PREPARE_DATA.md](./data/PREPARE_DATA.md) to download relevant data.

Updated links for commonsense components will be available very soon (3-5 days), please stay tuned!

