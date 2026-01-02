# coding=utf-8
"""WeDLM Trainer for SFT training."""

import os
import math
import logging
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from src.config import WeDLMTrainingConfig
from src.data import WeDLMPackedDataset, packed_collate_fn, get_im_end_token_id
from src.batch import WeDLMBatch, build_wedlm_batch
from src.model import wedlm_forward
from src.loss import compute_mlm_loss, compute_ar_loss
from src.attention import check_backend_available, get_available_backend, get_attention_wrapper

logger = logging.getLogger(__name__)

MASK_TOKEN_ID = 151665

# Lazy import wandb
_wandb = None

def _init_wandb(config: "WeDLMTrainingConfig", accelerator: Accelerator):
    """Initialize wandb if enabled (main process only)."""
    if not config.use_wandb or not accelerator.is_main_process:
        return None
    
    global _wandb
    try:
        import wandb
        _wandb = wandb
    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return None
    
    import os
    if config.wandb_host:
        os.environ["WANDB_BASE_URL"] = config.wandb_host
    if config.wandb_key:
        os.environ["WANDB_API_KEY"] = config.wandb_key
    
    wandb.init(
        project=config.wandb_project or "wedlm-sft",
        entity=config.wandb_team,
        group=config.wandb_group,
        config={k: v for k, v in config.__dict__.items() if not k.startswith('_')},
    )
    return wandb


class WeDLMTrainer:
    """Trainer for WeDLM SFT."""
    
    def __init__(self, config: WeDLMTrainingConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.wandb = _init_wandb(config, accelerator)
        self._setup()
        self._prepare_training()
    
    def _setup(self):
        """Initialize components."""
        if not check_backend_available(self.config.attention_backend):
            self.config.attention_backend = get_available_backend()
        logger.info(f"Attention backend: {self.config.attention_backend}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=self.config.trust_remote_code
        )
        self.im_end_token_id = get_im_end_token_id(self.tokenizer)
        self.tokenizer.pad_token_id = self.im_end_token_id
        
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float32,
            "attn_implementation": "eager",
        }
        if self.config.use_deepspeed and self.config.deepspeed_zero_stage == 3:
            model_kwargs["low_cpu_mem_usage"] = True
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path, **model_kwargs)
        
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        # 训练时 deterministic=False 更快
        self.attn_wrapper = get_attention_wrapper(
            self.config.attention_backend, 
            head_dim,
            deterministic=False,
        )
        # 将 wrapper 移到正确的设备
        if hasattr(self.attn_wrapper, 'to'):
            self.attn_wrapper = self.attn_wrapper.to(self.accelerator.device)
        
        self.train_dataset = WeDLMPackedDataset(
            data_path=self.config.train_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            num_learnable_im_end=self.config.num_learnable_im_end,
            cache_dir=os.path.join(self.config.output_dir, ".packed_cache"),
            seed=self.config.seed,
            rebuild_cache=self.config.rebuild_cache,
        )
        
        if self.accelerator.num_processes > 1:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
                seed=self.config.seed,
            )
            shuffle = False
            logger.info(f"Using DistributedSampler with {self.accelerator.num_processes} processes")
        else:
            self.train_sampler = None
            shuffle = True
        
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=1, 
            sampler=self.train_sampler,
            shuffle=shuffle,
            collate_fn=packed_collate_fn, 
            num_workers=4, 
            pin_memory=True
        )
    
    def _prepare_training(self):
        """Prepare optimizer, scheduler, and accelerator."""
        if self.train_sampler is not None:
            steps_per_epoch = len(self.train_dataloader)
        else:
            steps_per_epoch = len(self.train_dataset)
        
        num_update_steps_per_epoch = math.ceil(steps_per_epoch / self.config.gradient_accumulation_steps)
        self.num_training_steps = num_update_steps_per_epoch * self.config.num_train_epochs
        num_warmup_steps = int(self.num_training_steps * self.config.warmup_ratio)
        
        if self.accelerator.is_main_process:
            total_batches = len(self.train_dataset)
            logger.info(f"=== Training Configuration ===")
            logger.info(f"Number of GPUs: {self.accelerator.num_processes}")
            logger.info(f"Total batches in dataset: {total_batches}")
            logger.info(f"Batches per GPU per epoch: {steps_per_epoch}")
            logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
            logger.info(f"Update steps per epoch: {num_update_steps_per_epoch}")
            logger.info(f"Total training steps: {self.num_training_steps}")
            logger.info(f"Warmup steps: {num_warmup_steps}")
        
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_groups = [
            {"params": [p for n, p in self.model.named_parameters() 
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() 
                       if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_groups, lr=self.config.learning_rate)
        
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler_type, optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=self.num_training_steps
        )
        
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, self.lr_scheduler)
        
        self.global_step = 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single training step."""
        device = self.accelerator.device
        
        wedlm_batch = build_wedlm_batch(
            packed_input_ids=batch["packed_input_ids"].to(device),
            packed_labels=batch["packed_labels"].to(device),
            cum_seqlens=batch["cum_seqlens"].to(device),
            block_size=self.config.block_size,
            mask_token_id=MASK_TOKEN_ID,
            mask_per_block=self.config.mask_per_block,
            backend=self.config.attention_backend,
            eps=self.config.mask_eps,
        )
        
        logits = wedlm_forward(
            self.accelerator.unwrap_model(self.model),
            wedlm_batch, self.attn_wrapper, self.config.attention_backend
        )
        
        mlm_loss, mlm_logs = compute_mlm_loss(
            logits, wedlm_batch.original_ids, wedlm_batch.masked_indices,
            wedlm_batch.p_mask, self.config.loss_weighting_scheme, self.config.mask_eps
        )
        
        ar_loss, ar_logs = torch.tensor(0.0, device=device), {}
        if self.config.enable_ar_loss and self.config.ar_loss_weight > 0:
            ar_loss, ar_logs = self._compute_ar_loss(logits, batch["packed_labels"].to(device), wedlm_batch)
        
        ar_w = self.config.ar_loss_weight if self.config.enable_ar_loss else 0.0
        total_loss = (mlm_loss + ar_w * ar_loss) / (1.0 + ar_w) if ar_w > 0 else mlm_loss
        
        return total_loss, {"loss": total_loss.detach(), **mlm_logs, **ar_logs}
    
    def _compute_ar_loss(self, logits, packed_labels, batch: WeDLMBatch):
        """Extract x0 stream and compute AR loss."""
        device = logits.device
        bs = batch.base_cum_seqlens.numel() - 1
        
        x0_logits, x0_labels = [], []
        for si in range(bs):
            pst = batch.cum_seqlens[si].item()
            L = (batch.cum_seqlens[si + 1].item() - pst) // 2
            orig_st = batch.base_cum_seqlens[si].item()
            
            if L > 0:
                x0_logits.append(logits[pst:pst + L])
                x0_labels.append(packed_labels[orig_st:orig_st + L])
        
        if x0_logits:
            return compute_ar_loss(torch.cat(x0_logits), torch.cat(x0_labels))
        return torch.tensor(0.0, device=device), {}
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training: {len(self.train_dataloader)} batches per GPU, {self.num_training_steps} total update steps")
        
        progress_bar = tqdm(total=self.num_training_steps, disable=not self.accelerator.is_local_main_process)
        
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, logs = self.train_step(batch)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{logs['loss'].item():.4f}")
                    
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_metrics(logs, epoch)
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()
        
        progress_bar.close()
        self._save_checkpoint(final=True)
        if self.wandb:
            self.wandb.finish()
        logger.info("Training complete!")
    
    def _log_metrics(self, logs, epoch):
        if self.accelerator.is_main_process:
            log_str = f"Epoch {epoch} Step {self.global_step}: "
            log_str += ", ".join(f"{k}={v.item():.4f}" for k, v in logs.items() if isinstance(v, torch.Tensor))
            logger.info(log_str)
            
            if self.wandb:
                self.wandb.log({k: v.item() if hasattr(v, 'item') else v for k, v in logs.items()}, step=self.global_step)
    
    def _save_checkpoint(self, final=False):
        self.accelerator.wait_for_everyone()
        save_path = os.path.join(self.config.output_dir, "final" if final else f"checkpoint-{self.global_step}")
        
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            self.accelerator.unwrap_model(self.model).save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")

