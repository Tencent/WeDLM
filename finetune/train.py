#!/usr/bin/env python
# coding=utf-8
"""WeDLM SFT Training Entry Script.

Usage:
    accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 train.py --config configs/example.yaml
"""

import os
import argparse
import logging
import json

from accelerate import Accelerator
from accelerate.utils import set_seed, DeepSpeedPlugin

from src import WeDLMTrainingConfig, WeDLMTrainer

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WeDLM SFT Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--model_path", type=str, default=None, help="Override model path")
    parser.add_argument("--train_data", type=str, default=None, help="Override training data path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--attention_backend", type=str, choices=["magi", "dense"], default=None,
                        help="Attention backend: magi or dense")
    parser.add_argument("--loss_weighting_scheme", type=str, choices=["uniform", "weighted"], default=None,
                        help="Loss weighting scheme")
    parser.add_argument("--rebuild_cache", action="store_true", help="Rebuild data cache")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config = WeDLMTrainingConfig.from_yaml(args.config) if args.config else WeDLMTrainingConfig()
    
    # Override config with command line arguments
    for key in ["model_path", "train_data", "output_dir", "attention_backend", "loss_weighting_scheme"]:
        if getattr(args, key, None) is not None:
            setattr(config, key, getattr(args, key))
    config.rebuild_cache = args.rebuild_cache
    
    # Setup DeepSpeed if enabled
    deepspeed_plugin = None
    if config.use_deepspeed:
        ds_config = config.get_deepspeed_config()
        if ds_config:
            os.makedirs(config.output_dir, exist_ok=True)
            ds_path = os.path.join(config.output_dir, "deepspeed_config.json")
            with open(ds_path, "w") as f:
                json.dump(ds_config, f, indent=2)
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.bf16 else "no",
        deepspeed_plugin=deepspeed_plugin,
    )
    
    set_seed(config.seed)
    
    # Save config
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        config.save_yaml(os.path.join(config.output_dir, "training_config.yaml"))
    
    # Train
    trainer = WeDLMTrainer(config, accelerator)
    trainer.train()


if __name__ == "__main__":
    main()
