# coding=utf-8
"""Data processing for WeDLM SFT with efficient packing."""

import json
import os
import pickle
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

DEFAULT_IM_END_TOKEN_ID = 151645


@dataclass
class PackedBatch:
    """A pre-packed batch of samples."""
    packed_input_ids: torch.Tensor  # [total_length]
    packed_labels: torch.Tensor     # [total_length]
    cum_seqlens: torch.Tensor       # [num_samples + 1]
    num_samples: int
    total_tokens: int


class WeDLMPackedDataset(Dataset):
    """Dataset that pre-packs samples into fixed-length batches.
    
    Key design:
    - batch_seq_length = max_seq_length * per_device_train_batch_size
    - Each batch is packed to approximately batch_seq_length tokens
    - Multiple samples are packed into one batch, last sample may be truncated
    - All batches are pre-built and cached for fast loading
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        per_device_train_batch_size: int = 2,
        num_learnable_im_end: int = 8,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        rebuild_cache: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.per_device_train_batch_size = per_device_train_batch_size
        self.batch_seq_length = max_seq_length * per_device_train_batch_size
        self.num_learnable_im_end = num_learnable_im_end
        self.im_end_token_id = get_im_end_token_id(tokenizer)
        self.seed = seed
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(data_path), ".packed_cache")
        self.cache_dir = cache_dir
        
        # Compute cache filename based on config hash
        config_hash = self._compute_config_hash(data_path)
        self.cache_file = os.path.join(cache_dir, f"packed_{config_hash}.pkl")
        self.meta_file = os.path.join(cache_dir, f"meta_{config_hash}.json")
        
        # Load or build packed batches
        if rebuild_cache and os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            if os.path.exists(self.meta_file):
                os.remove(self.meta_file)
        
        self.packed_batches, self.metadata = self._load_or_build_cache(data_path)
        
        logger.info(f"Loaded {len(self.packed_batches)} packed batches")
        logger.info(f"Total samples: {self.metadata['total_samples']}")
        logger.info(f"Total tokens: {self.metadata['total_tokens']}")
        logger.info(f"Batch seq length: {self.batch_seq_length}")
    
    def _compute_config_hash(self, data_path: str) -> str:
        """Compute a hash of the configuration for cache naming."""
        config_dict = {
            "data_path": data_path,
            "max_seq_length": self.max_seq_length,
            "batch_seq_length": self.batch_seq_length,
            "num_learnable_im_end": self.num_learnable_im_end,
            "seed": self.seed,
        }
        # Include file modification time for cache invalidation
        if os.path.exists(data_path):
            config_dict["mtime"] = os.path.getmtime(data_path)
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _load_or_build_cache(self, data_path: str) -> Tuple[List[PackedBatch], Dict]:
        """Load from cache or build packed batches."""
        if os.path.exists(self.cache_file) and os.path.exists(self.meta_file):
            logger.info(f"Loading cached packed batches from {self.cache_file}")
            try:
                with open(self.cache_file, "rb") as f:
                    packed_batches = pickle.load(f)
                with open(self.meta_file, "r") as f:
                    metadata = json.load(f)
                return packed_batches, metadata
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, rebuilding...")
        
        logger.info("Building packed batches (this may take a while)...")
        packed_batches, metadata = self._build_packed_batches(data_path)
        
        # Save cache
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            pickle.dump(packed_batches, f)
        with open(self.meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(packed_batches)} packed batches to {self.cache_file}")
        return packed_batches, metadata
    
    def _build_packed_batches(self, data_path: str) -> Tuple[List[PackedBatch], Dict]:
        """Build packed batches from raw data."""
        # 1. Load and tokenize all samples
        all_samples = self._load_and_tokenize_data(data_path)
        logger.info(f"Loaded {len(all_samples)} samples")
        
        # 2. Shuffle samples
        import random
        rng = random.Random(self.seed)
        rng.shuffle(all_samples)
        
        # 3. Pack into fixed-length batches
        packed_batches = self._pack_samples_into_batches(all_samples)
        
        # 4. Compute metadata
        total_samples = sum(b.num_samples for b in packed_batches)
        total_tokens = sum(b.total_tokens for b in packed_batches)
        
        metadata = {
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "num_batches": len(packed_batches),
            "batch_seq_length": self.batch_seq_length,
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
        }
        
        return packed_batches, metadata
    
    def _load_and_tokenize_data(self, data_path: str) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Load and tokenize all samples.
        
        Returns list of (input_ids, labels, original_length) tuples.
        """
        samples = []
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    messages = json.loads(line)
                    result = self._tokenize_messages(messages)
                    if result is not None:
                        samples.append(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: JSON decode error - {e}")
                except Exception as e:
                    logger.warning(f"Skipping line {line_num}: {e}")
        
        return samples
    
    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Tokenize chat messages into input_ids and labels.
        
        Returns (input_ids, labels, original_length) or None if invalid.
        """
        if not messages:
            return None
        
        # Apply chat template
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Reserve space for learnable im_end tokens
        reserved_for_im_end = max(0, self.num_learnable_im_end - 1)
        effective_max_len = self.max_seq_length - reserved_for_im_end
        
        # Tokenize with truncation
        full_ids = self.tokenizer.encode(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=effective_max_len
        )
        
        if len(full_ids) == 0:
            return None
        
        # Compute prompt length for label masking
        if len(messages) > 1:
            prompt_messages = messages[:-1]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = min(len(prompt_ids), len(full_ids))
        else:
            # Single message - mask everything as prompt (no loss)
            prompt_len = len(full_ids)
        
        # Add learnable im_end tokens
        if reserved_for_im_end > 0:
            full_ids = full_ids + [self.im_end_token_id] * reserved_for_im_end
        
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        original_length = len(full_ids)
        
        return (input_ids, labels, original_length)
    
    def _pack_samples_into_batches(
        self, 
        samples: List[Tuple[torch.Tensor, torch.Tensor, int]]
    ) -> List[PackedBatch]:
        """Pack samples into fixed-length batches.
        
        Strategy:
        - Fill each batch up to batch_seq_length tokens
        - If a sample doesn't fit completely, truncate it and discard the rest
        - A new batch starts fresh with the next complete sample
        """
        packed_batches = []
        
        current_input_ids: List[torch.Tensor] = []
        current_labels: List[torch.Tensor] = []
        current_seqlens = [0]
        current_length = 0
        
        for input_ids, labels, _ in samples:
            sample_len = input_ids.size(0)
            space_left = self.batch_seq_length - current_length
            
            if sample_len <= space_left:
                # Sample fits completely
                current_input_ids.append(input_ids)
                current_labels.append(labels)
                current_length += sample_len
                current_seqlens.append(current_length)
            else:
                # Sample doesn't fit completely
                if space_left > 0:
                    # Truncate and add the part that fits, discard the rest
                    current_input_ids.append(input_ids[:space_left])
                    current_labels.append(labels[:space_left])
                    current_length += space_left
                    current_seqlens.append(current_length)
                
                # Finalize current batch
                if current_length > 0:
                    batch = self._create_packed_batch(
                        current_input_ids, current_labels, current_seqlens
                    )
                    packed_batches.append(batch)
                
                # Start new batch (the current sample's remainder is discarded)
                current_input_ids = []
                current_labels = []
                current_seqlens = [0]
                current_length = 0
        
        # Handle last batch (may not be full)
        if current_length > 0:
            batch = self._create_packed_batch(
                current_input_ids, current_labels, current_seqlens
            )
            packed_batches.append(batch)
        
        return packed_batches
    
    def _create_packed_batch(
        self,
        input_ids_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        seqlens: List[int],
    ) -> PackedBatch:
        """Create a PackedBatch from lists of tensors."""
        packed_input_ids = torch.cat(input_ids_list, dim=0)
        packed_labels = torch.cat(labels_list, dim=0)
        cum_seqlens = torch.tensor(seqlens, dtype=torch.long)
        
        return PackedBatch(
            packed_input_ids=packed_input_ids,
            packed_labels=packed_labels,
            cum_seqlens=cum_seqlens,
            num_samples=len(input_ids_list),
            total_tokens=packed_input_ids.size(0),
        )
    
    def __len__(self) -> int:
        """Return number of batches (= number of training steps per epoch)."""
        return len(self.packed_batches)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a pre-packed batch."""
        batch = self.packed_batches[idx]
        return {
            "packed_input_ids": batch.packed_input_ids.clone(),
            "packed_labels": batch.packed_labels.clone(),
            "cum_seqlens": batch.cum_seqlens.clone(),
        }
    
    def get_total_samples(self) -> int:
        """Get total number of samples across all batches."""
        return self.metadata["total_samples"]
    
    def get_total_tokens(self) -> int:
        """Get total number of tokens across all batches."""
        return self.metadata["total_tokens"]
    
    def get_num_training_steps(self, num_epochs: int = 1) -> int:
        """Get total number of training steps for given epochs."""
        return len(self.packed_batches) * num_epochs


class WeDLMShuffledPackedDataset(Dataset):
    """A wrapper that shuffles batch order each epoch.
    
    Use this for multi-epoch training where you want different
    batch ordering each epoch.
    """
    
    def __init__(
        self,
        base_dataset: WeDLMPackedDataset,
        epoch: int = 0,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.epoch = epoch
        self.seed = seed
        self._shuffle_indices()
    
    def _shuffle_indices(self):
        """Shuffle indices based on epoch and seed."""
        import random
        rng = random.Random(self.seed + self.epoch)
        self.indices = list(range(len(self.base_dataset)))
        rng.shuffle(self.indices)
    
    def set_epoch(self, epoch: int):
        """Set epoch for reshuffling."""
        self.epoch = epoch
        self._shuffle_indices()
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.base_dataset[self.indices[idx]]


def packed_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for packed dataset.
    
    Since each item is already a complete packed batch, we just return it directly.
    DataLoader should use batch_size=1 with this dataset.
    """
    if len(batch) == 1:
        return batch[0]
    
    # If somehow batch_size > 1, we need to concatenate
    # This shouldn't happen in normal usage
    all_input_ids = []
    all_labels = []
    all_seqlens = [0]
    
    for item in batch:
        all_input_ids.append(item["packed_input_ids"])
        all_labels.append(item["packed_labels"])
        
        # Adjust cum_seqlens offsets
        offset = all_seqlens[-1]
        item_seqlens = item["cum_seqlens"][1:] + offset
        all_seqlens.extend(item_seqlens.tolist())
    
    return {
        "packed_input_ids": torch.cat(all_input_ids, dim=0),
        "packed_labels": torch.cat(all_labels, dim=0),
        "cum_seqlens": torch.tensor(all_seqlens, dtype=torch.long),
    }


def get_im_end_token_id(tokenizer: PreTrainedTokenizer) -> int:
    """Get im_end token id from tokenizer."""
    if hasattr(tokenizer, 'im_end_id'):
        return tokenizer.im_end_id
    
    try:
        tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if tokens:
            return tokens[0]
    except:
        pass
    
    if hasattr(tokenizer, 'added_tokens_encoder'):
        if "<|im_end|>" in tokenizer.added_tokens_encoder:
            return tokenizer.added_tokens_encoder["<|im_end|>"]
    
    return DEFAULT_IM_END_TOKEN_ID


# Legacy compatibility
@dataclass
class SFTSample:
    """A single SFT sample (for backward compatibility)."""
    input_ids: torch.Tensor
    labels: torch.Tensor


def collate_fn(
    batch: List[SFTSample],
    pad_token_id: int = DEFAULT_IM_END_TOKEN_ID,
) -> Dict[str, torch.Tensor]:
    """Legacy collate function (for backward compatibility)."""
    input_ids_list = []
    labels_list = []
    seq_lens = [0]
    
    for sample in batch:
        input_ids_list.append(sample.input_ids)
        labels_list.append(sample.labels)
        seq_lens.append(seq_lens[-1] + sample.input_ids.size(0))
    
    return {
        "packed_input_ids": torch.cat(input_ids_list, dim=0),
        "packed_labels": torch.cat(labels_list, dim=0),
        "cum_seqlens": torch.tensor(seq_lens, dtype=torch.long),
    }

