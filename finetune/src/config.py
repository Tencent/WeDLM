# coding=utf-8
"""Training configuration for WeDLM SFT."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class WeDLMTrainingConfig:
    """Configuration for WeDLM SFT training."""
    
    # Model
    model_path: str = "tencent/WeDLM-8B-Instruct"
    trust_remote_code: bool = True
    
    # Data
    train_data: str = "data/train.jsonl"
    max_seq_length: int = 2048
    
    # WeDLM specific
    block_size: int = 32
    mask_per_block: bool = True
    loss_weighting_scheme: str = "weighted"  # "weighted" (1/γ) or "uniform"
    mask_eps: float = 1e-8
    num_learnable_im_end: int = 8
    
    # AR loss
    enable_ar_loss: bool = True
    ar_loss_weight: float = 1.0
    
    # Attention backend
    attention_backend: str = "magi"  # "magi" or "dense"
    
    # Training
    output_dir: str = "outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-6
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Cache
    rebuild_cache: bool = False
    
    # DeepSpeed
    use_deepspeed: bool = False
    deepspeed_zero_stage: int = 2
    deepspeed_offload_optimizer: bool = False
    deepspeed_offload_param: bool = False
    deepspeed_offload_nvme: bool = False
    deepspeed_nvme_path: str = "/tmp/deepspeed_offload"
    deepspeed_pin_memory: bool = True
    deepspeed_overlap_comm: bool = True
    deepspeed_contiguous_gradients: bool = True
    deepspeed_reduce_bucket_size: int = 50000000
    deepspeed_stage3_prefetch_bucket_size: int = 50000000
    deepspeed_stage3_param_persistence_threshold: int = 100000
    deepspeed_stage3_max_live_parameters: int = 1000000000
    deepspeed_stage3_max_reuse_distance: int = 1000000000
    deepspeed_config_file: Optional[str] = None
    
    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Device & Seed
    bf16: bool = True
    seed: int = 42
    
    # WandB (optional)
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_team: Optional[str] = None  # entity
    wandb_group: Optional[str] = None
    wandb_host: Optional[str] = None  # for private deployment
    wandb_key: Optional[str] = None   # API key
    
    def __post_init__(self):
        if self.loss_weighting_scheme not in ["uniform", "weighted"]:
            raise ValueError(f"Unknown loss_weighting_scheme: {self.loss_weighting_scheme}")
        if not self.mask_per_block:
            import warnings
            warnings.warn("mask_per_block=False does not match the paper's design.", UserWarning)
    
    @classmethod
    def from_yaml(cls, path: str) -> "WeDLMTrainingConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def save_yaml(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(save_dict, f, default_flow_style=False)
    
    def get_batch_seq_length(self) -> int:
        return self.max_seq_length * self.per_device_train_batch_size
    
    def get_deepspeed_config(self) -> Optional[Dict[str, Any]]:
        if not self.use_deepspeed:
            return None
        if self.deepspeed_config_file and os.path.exists(self.deepspeed_config_file):
            import json
            with open(self.deepspeed_config_file, "r") as f:
                return json.load(f)
        
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.max_grad_norm,
            "steps_per_print": self.logging_steps,
            "wall_clock_breakdown": False,
        }
        
        ds_config["bf16" if self.bf16 else "fp16"] = {"enabled": True}
        
        zero_config = {
            "stage": self.deepspeed_zero_stage,
            "overlap_comm": self.deepspeed_overlap_comm,
            "contiguous_gradients": self.deepspeed_contiguous_gradients,
            "reduce_bucket_size": self.deepspeed_reduce_bucket_size,
            "allgather_bucket_size": self.deepspeed_reduce_bucket_size,
        }
        
        if self.deepspeed_offload_optimizer:
            device = "nvme" if self.deepspeed_offload_nvme else "cpu"
            zero_config["offload_optimizer"] = {"device": device, "pin_memory": self.deepspeed_pin_memory}
            if self.deepspeed_offload_nvme:
                zero_config["offload_optimizer"]["nvme_path"] = self.deepspeed_nvme_path
        
        if self.deepspeed_zero_stage == 3:
            zero_config.update({
                "stage3_prefetch_bucket_size": self.deepspeed_stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": self.deepspeed_stage3_param_persistence_threshold,
                "stage3_max_live_parameters": self.deepspeed_stage3_max_live_parameters,
                "stage3_max_reuse_distance": self.deepspeed_stage3_max_reuse_distance,
                "stage3_gather_16bit_weights_on_model_save": True,
            })
            if self.deepspeed_offload_param:
                device = "nvme" if self.deepspeed_offload_nvme else "cpu"
                zero_config["offload_param"] = {"device": device, "pin_memory": self.deepspeed_pin_memory}
        
        ds_config["zero_optimization"] = zero_config
        return ds_config

