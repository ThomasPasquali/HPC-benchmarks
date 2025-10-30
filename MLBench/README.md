# Inference on Small ML Models

Pioneer: 

```bash
ml llvm/EPI-development openBLAS/ubuntu/0.3.29_llvmEPI1.0
export CC=$(which clang)
export CXX=$(which clang++)

# Setup
python -m venv venv_ml_pioneer

source ~/venv_ml_pioneer/bin/activate

# Setup
pip install transformers
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```