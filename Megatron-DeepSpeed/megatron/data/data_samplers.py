# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""


import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron import get_args, get_tokenizer
from megatron.core import mpu
from deepspeed.runtime.dataloader import RepeatingLoader


class DummyAwareDataset(Dataset):
    """Wrapper dataset that returns dummy samples for negative indices.
    
    Used for uneven batch size support where some slots need to be filled
    with dummy samples that will be masked out during loss computation.
    """
    def __init__(self, dataset, seq_length, eod_token):
        self.dataset = dataset
        self.seq_length = seq_length
        self.eod_token = eod_token
    
    def __getitem__(self, idx):
        if idx < 0:  # dummy index
            # Return all EOD tokens, which will be masked by eod_mask_loss
            return {'text': np.full(self.seq_length + 1, self.eod_token, dtype=np.int64)}
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()
    tokenizer = get_tokenizer()

    # Wrap dataset with DummyAwareDataset for uneven batch size support
    # This allows the sampler to use negative indices for dummy samples
    wrapped_dataset = DummyAwareDataset(dataset, args.seq_length, tokenizer.eod)

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            real_global_batch_size=args.real_global_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps)
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    loader = torch.utils.data.DataLoader(wrapped_dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True)
    if args.repeated_dataloader:
        loader=RepeatingLoader(loader)
    return loader

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size, real_global_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True, pipeline_model_parallel_size=1,
                 gradient_accumulation_steps=1):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.real_global_batch_size = real_global_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        # Total slots per training step = micro_batch * dp * gas
        self.slot_per_step = self.micro_batch_size * data_parallel_size * gradient_accumulation_steps
        self.drop_last = drop_last
        self.pipeline_model_parallel_size = pipeline_model_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def _allocate_samples_to_slots(self, real_indices):
        """Allocate real samples to slots across micro-steps and ranks.
        
        For uneven batch size, some slots may be filled with dummy indices (-1).
        
        Args:
            real_indices: List of real sample indices (length = real_global_batch_size)
            
        Returns:
            List of (gas) lists, each containing indices for each rank's micro-batch.
            Dummy slots are filled with -1.
        """
        N = len(real_indices)  # real_global_batch_size
        P = self.data_parallel_size
        gas = self.gradient_accumulation_steps
        mbsz = self.micro_batch_size
        
        # Allocate samples evenly across all slots
        # Total slots = mbsz * P * gas
        # We distribute N real samples across these slots
        
        # Result: gas micro-steps, each with P ranks, each with mbsz samples
        result = []
        sample_idx = 0
        
        for micro_step in range(gas):
            micro_step_batches = []
            for rank in range(P):
                rank_batch = []
                for _ in range(mbsz):
                    if sample_idx < N:
                        rank_batch.append(real_indices[sample_idx])
                        sample_idx += 1
                    else:
                        # Dummy sample
                        rank_batch.append(-1)
                micro_step_batches.append(rank_batch)
            result.append(micro_step_batches)
        
        return result

    def __iter__(self):
        batch = []
        # Collect real_global_batch_size samples for each training step
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.real_global_batch_size:
                # Allocate samples to slots
                allocated = self._allocate_samples_to_slots(batch)
                
                # Yield micro-batches for this rank across all gas steps
                for micro_step in range(self.gradient_accumulation_steps):
                    yield allocated[micro_step][self.data_parallel_rank]
                
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            # Pad with dummy indices if needed
            allocated = self._allocate_samples_to_slots(batch)
            for micro_step in range(self.gradient_accumulation_steps):
                yield allocated[micro_step][self.data_parallel_rank]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size
            
            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []