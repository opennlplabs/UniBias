# UniBias
This repository contains the code for our [NeurIPS 2024 paper:](https://arxiv.org/abs/2405.20612)

**"UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation" (NeurIPS 2024)**


## Dependencies

We use Python 3.8+ and PyTorch 2.0+. You can install dependencies with:

```bash
pip install transformers torch datasets numpy pandas scikit-learn scipy tokenizers huggingface-hub essential-generators accelerate
```

### Setup with Conda

```bash
# Create conda environment
conda create -n unibias python=3.10 -y
conda activate unibias

# Install dependencies
pip install transformers torch datasets numpy pandas scikit-learn scipy tokenizers huggingface-hub essential-generators accelerate
```

### Model Configuration

The code supports multiple model architectures. Edit `main.py` to change the model:
- **GPT-2** (default, ~500MB, fast on CPU): `gpt2`
- **TinyLlama** (~2.2GB, free): `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Llama-2-7B** (requires HuggingFace token): `meta-llama/Llama-2-7b-hf`

### Supported Architectures

The codebase uses an architecture abstraction layer (`model_utils.py`) that supports:
- **GPT-2 family**: GPT-2, GPT-2 Medium, GPT-2 Large, GPT-2 XL
- **Llama family**: Llama-2, TinyLlama, and other Llama-based models

### Platform Support

- **CUDA GPUs**: Full support with automatic detection
- **Apple Silicon (M1/M2/M3)**: Falls back to CPU mode (MPS has compatibility issues with some operations)
- **CPU**: Supported (GPT-2 recommended for CPU-only systems)


## Project Structure

```
UniBias/
├── main.py                 # Entry point - model loading, pipeline orchestration
├── model_utils.py          # Architecture abstraction layer (GPT-2/Llama support)
├── FFN_manipulate.py       # FFN bias identification and elimination
├── attention_manipulate.py # Attention head bias identification and elimination
├── evaluation.py           # ICL evaluation and calibration methods
├── utils.py                # Dataset loading, prompt generation, utilities
├── results/                # Output results (JSON format)
└── README.md
```

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
