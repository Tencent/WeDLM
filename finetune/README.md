# WeDLM Fine-tuning

This directory contains the training framework for fine-tuning WeDLM models using Supervised Fine-Tuning (SFT).

> [!NOTE]
> **About This Code**
> 
> This fine-tuning codebase has been tested and verified to work correctly. However, it is **not** the internal training framework we used for our official SFT experiments — our internal pipeline relies on proprietary infrastructure.
> 
> This code is provided as a **functional reference implementation** to help researchers and developers fine-tune WeDLM on their own data.

---

## Requirements

### Base Dependencies

The fine-tuning code requires the following additional packages on top of the main WeDLM requirements:

```bash
pip install accelerate deepspeed pyyaml datasets
```

### Attention Backend

WeDLM training supports two attention backends:

| Backend | Description | Installation |
|:--------|:------------|:-------------|
| `dense` | PyTorch native SDPA with 2D mask | Works out-of-the-box, no extra installation |
| `magi` | MagiAttention flex flash attention | Requires separate installation (see below) |

#### Dense Backend (Default)

The `dense` backend uses PyTorch's `scaled_dot_product_attention` with explicit 2D attention masks. This works on any GPU with PyTorch 2.0+ and requires no additional installation.

```yaml
attention_backend: "dense"
```

#### MagiAttention Backend (Optimized)

For optimized training with flexible attention patterns, you can use [MagiAttention](https://github.com/SandAI-org/MagiAttention).

> [!IMPORTANT]
> MagiAttention installation involves CUDA kernel compilation and can be complex. Please follow the **official installation guide** at:
> 
> 👉 **https://github.com/SandAI-org/MagiAttention**
> 
> Ensure your CUDA toolkit version matches your PyTorch CUDA version before installation.

Once installed, enable it in your config:

```yaml
attention_backend: "magi"
```

---

## Quick Start

### 1. Prepare Data

Download a sample dataset (Alpaca):

```bash
cd finetune
python scripts/download_alpaca_sft.py
```

This creates `data/alpaca_cleaned_sft.jsonl` in chat format.

### 2. Configure Training

Edit `configs/example.yaml` or create your own config file:

```yaml
# Model
model_path: "tencent/WeDLM-8B-Base"

# Data
train_data: "data/alpaca_cleaned_sft.jsonl"
max_seq_length: 2048

# Attention backend: "magi" or "dense"
attention_backend: "dense"

# Training
output_dir: "outputs/my-sft-run"
num_train_epochs: 1
learning_rate: 3.0e-6
```

### 3. Launch Training

**Multi-GPU Training (Recommended):**

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    train.py --config configs/example.yaml
```

**Single GPU Training:**

```bash
python train.py --config configs/example.yaml
```

---

## Data Format

Training data should be in JSONL format, where each line is a JSON array of chat messages:

```json
[{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]
```

The trainer will:
1. Apply the tokenizer's chat template
2. Pack multiple samples into sequences of `max_seq_length`
3. Generate WeDLM-specific attention masks for training

---

## Configuration Reference

### Model Settings

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model_path` | str | `"tencent/WeDLM-8B-Instruct"` | HuggingFace model path or local path |
| `trust_remote_code` | bool | `true` | Trust remote code for model loading |

### Data Settings

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `train_data` | str | `"data/train.jsonl"` | Path to training data (JSONL) |
| `max_seq_length` | int | `2048` | Maximum sequence length per sample |

### WeDLM-Specific Settings

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `block_size` | int | `32` | Block size for mask generation |
| `mask_per_block` | bool | `true` | Whether to mask per block (matches paper) |
| `loss_weighting_scheme` | str | `"weighted"` | Loss weighting: `"weighted"` (1/γ) or `"uniform"` |
| `mask_eps` | float | `1e-8` | Epsilon for numerical stability in masking |
| `num_learnable_im_end` | int | `8` | Number of learnable `<\|im_end\|>` tokens |

### AR Loss Settings

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `enable_ar_loss` | bool | `true` | Enable auxiliary AR loss |
| `ar_loss_weight` | float | `1.0` | Weight for AR loss term |

### Attention Backend

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `attention_backend` | str | `"magi"` | Attention implementation: `"magi"` or `"dense"` |

### Training Hyperparameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `output_dir` | str | `"outputs"` | Directory for checkpoints and logs |
| `num_train_epochs` | int | `3` | Number of training epochs |
| `per_device_train_batch_size` | int | `1` | Batch size per GPU |
| `gradient_accumulation_steps` | int | `8` | Gradient accumulation steps |
| `learning_rate` | float | `3e-6` | Peak learning rate |
| `lr_scheduler_type` | str | `"cosine"` | LR scheduler type |
| `warmup_ratio` | float | `0.1` | Warmup ratio |
| `weight_decay` | float | `0.01` | Weight decay |
| `max_grad_norm` | float | `1.0` | Max gradient norm for clipping |

### DeepSpeed Settings

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `use_deepspeed` | bool | `false` | Enable DeepSpeed |
| `deepspeed_zero_stage` | int | `2` | ZeRO stage (1, 2, or 3) |
| `deepspeed_offload_optimizer` | bool | `false` | Offload optimizer to CPU |
| `deepspeed_offload_param` | bool | `false` | Offload parameters to CPU (ZeRO-3 only) |

### Logging & Checkpointing

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `logging_steps` | int | `10` | Log every N steps |
| `save_steps` | int | `500` | Save checkpoint every N steps |
| `save_total_limit` | int | `3` | Maximum checkpoints to keep |

### Device & Precision

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `bf16` | bool | `true` | Use bfloat16 precision |
| `seed` | int | `42` | Random seed |

### WandB (Optional)

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `use_wandb` | bool | `false` | Enable WandB logging |
| `wandb_project` | str | `null` | WandB project name |
| `wandb_team` | str | `null` | WandB team/entity |
| `wandb_group` | str | `null` | WandB run group |
| `wandb_host` | str | `null` | Custom WandB host (for private deployments) |
| `wandb_key` | str | `null` | WandB API key |

---

## Command Line Overrides

You can override config values via command line:

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    train.py \
    --config configs/example.yaml \
    --model_path "tencent/WeDLM-7B-Base" \
    --output_dir "outputs/custom-run" \
    --attention_backend "dense" \
    --rebuild_cache
```

---

## Output Structure

After training, the output directory will contain:

```
outputs/my-sft-run/
├── training_config.yaml      # Saved training configuration
├── deepspeed_config.json     # DeepSpeed config (if enabled)
├── .packed_cache/            # Cached preprocessed data
├── checkpoint-500/           # Intermediate checkpoint
├── checkpoint-1000/          # Intermediate checkpoint
└── final/                    # Final model checkpoint
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

---

## Tips

1. **Memory Optimization**: Use `deepspeed_offload_optimizer: true` to offload optimizer states to CPU, reducing GPU memory usage.

2. **Attention Backend Selection**:
   - Use `dense` for compatibility and debugging
   - Use `magi` for optimized training with complex attention patterns

3. **Data Caching**: The trainer caches preprocessed data. Use `--rebuild_cache` to force regeneration if you modify your dataset.

4. **Multi-node Training**: Configure accelerate for multi-node setup and adjust `gradient_accumulation_steps` accordingly.

