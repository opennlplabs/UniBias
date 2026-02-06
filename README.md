# UniBias
This repository contains the code for our [NeurIPS 2024 paper:](https://arxiv.org/abs/2405.20612)

**"UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation" (NeurIPS 2024)**


## Dependencies

We use python 3.8 and pytorch 2.0.1. You can use ```pip install -r requirements.txt``` to install the required libraries.

Additionally, install the `essential-generators` package:
```bash
pip install essential-generators
```

### Setup with Conda

```bash
# Create conda environment with Python 3.8
conda create -n unibias python=3.8 -y
conda activate unibias

# Install dependencies
pip install -r requirements.txt
pip install essential-generators
```

### Model Configuration

The code supports multiple models. Edit `main.py` to change the model:
- **TinyLlama** (default, ~2.2GB, free): `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Llama-2-7B** (requires HuggingFace token): `meta-llama/Llama-2-7b-hf`

### Platform Support

- **CUDA GPUs**: Full support with automatic detection
- **Apple Silicon (M1/M2/M3)**: Falls back to CPU mode (MPS has compatibility issues with some operations)
- **CPU**: Supported but slower


## Data

Running the code will automatically download datasets from Huggingface.


## Run Experiment

To run UniBias using the following command:

```bash
python main.py \
  --dataset_name <dataset> \
  --UniBias <True/False> \
  --Calibration <True/False> \
  --seed <random_seed> \
  --format_index <Generate prompts with different formats> \
  --order_index <Generate prompts with varying example orders>
```

### Example: Basic Evaluation

```bash
# Run basic ICL evaluation on SST-2 without UniBias or Calibration
python main.py --dataset_name sst2 --UniBias False --Calibration False

# Run with UniBias enabled
python main.py --dataset_name sst2 --UniBias True --Calibration False
```

## Acknowledgment

The code for calibration baselines evaluation is from [DC](https://github.com/fywalter/label-bias) and [PC](https://github.com/fywalter/label-bias). We appreciate their excellent contributions!

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{zhou2024unibias,
  title={UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation},
  author={Zhou, Hanzhang  and
    Feng, Zijian and
    Zhu, Zixiao  and
    Qian, Junlang  and
    Mao, Kezhi},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```

