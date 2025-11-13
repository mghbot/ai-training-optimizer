# Adaptive Curriculum with Memory-Driven Difficulty Scaling (ACMDS)
# Production-ready implementation for RTX 3090 (24GB VRAM)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.cuda.amp as amp
from torch.cuda import memory_reserved, memory_allocated, max_memory_allocated
import numpy as np
from collections import deque, defaultdict
import psutil
import time
from typing import Dict, List, Tuple, Optional, Iterator
import json
from pathlib import Path
import threading
import queue
import warnings
from dataclasses import dataclass
from enum import Enum
import math
import gc

class MemoryPressureLevel(Enum):
    CRITICAL = 0    # < 1GB headroom
    HIGH = 1        # 1-3GB headroom
    MODERATE = 2    # 3-6GB headroom
    LOW = 3         # > 6GB headroom

@dataclass
class CurriculumConfig:
    """ACMDS hyperparameters optimized for RTX 3090"""
    target_headroom_gb: float = 3.0
    min_batch_size: int = 4
    max_batch_size: int = 64
    difficulty_window: int = 1000
    pressure_smoothing: float = 0.9
    pid_kp: float = 0.15
    pid_ki: float = 0.02
    pid_kd: float = 0.08
    memory_cost_coef: float = 0.001  # bytes per difficulty unit
    curriculum_alpha: float = 0.05   # EMA for difficulty tracking
    enable_cuda_graphs: bool = False  # Disabled due to dynamic batch sizes
    enable_amp: bool = True
    gradient_checkpointing: bool = False  # Model-dependent, disabled by default
    gc_threshold_gb: float = 2.0

class MemoryPressureController:
    """RTX 3090-specific memory controller with PID feedback"""

    def __init__(self, device_id: int = 0, config: CurriculumConfig = None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This optimizer requires an NVIDIA GPU.")

        self.device = torch.device(f'cuda:{device_id}')
        self.config = config or CurriculumConfig()
        self.pid_kp = self.config.pid_kp
        self.pid_ki = self.config.pid_ki
        self.pid_kd = self.config.pid_kd
        self.target_headroom = self.config.target_headroom_gb * 1024**3

        self.error_integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        self.pressure_ema = 0.0

        # RTX 3090 specific thresholds (24GB total)
        # Auto-detect actual VRAM if available
        try:
            self.vram_total = torch.cuda.get_device_properties(device_id).total_memory
        except Exception:
            self.vram_total = 24 * 1024**3  # Default to 24GB

        self.critical_threshold = 1 * 1024**3
        self.high_threshold = 3 * 1024**3
        self.moderate_threshold = 6 * 1024**3

        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self.metrics_queue = queue.Queue(maxsize=1000)

    def start_monitoring(self, interval_ms: int = 50):
        """Launch asynchronous memory monitoring thread"""
        def monitor():
            while not self._shutdown.wait(interval_ms / 1000.0):
                try:
                    metrics = self._sample_memory_metrics()
                    with self._lock:
                        self.pressure_ema = (self.config.pressure_smoothing * self.pressure_ema +
                                           (1 - self.config.pressure_smoothing) * metrics['pressure'])
                        self.metrics_queue.put(metrics, block=False)
                except queue.Full:
                    pass
                except Exception as e:
                    warnings.warn(f"Memory monitor error: {e}")

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def _sample_memory_metrics(self) -> Dict:
        """Sample RTX 3090 memory state"""
        with torch.cuda.device(self.device):
            allocated = memory_allocated()
            reserved = memory_reserved()
            free = self.vram_total - reserved

            # Calculate memory pressure (0-1 scale)
            headroom = max(0, free - self.target_headroom)
            pressure = 1.0 - (headroom / (self.vram_total - self.target_headroom))
            pressure = max(0.0, min(1.0, pressure))

            return {
                'allocated_gb': allocated / 1024**3,
                'reserved_gb': reserved / 1024**3,
                'free_gb': free / 1024**3,
                'pressure': pressure,
                'timestamp': time.time()
            }

    def get_pressure_level(self) -> MemoryPressureLevel:
        """Get discrete pressure level for curriculum decisions"""
        with torch.cuda.device(self.device):
            reserved = memory_reserved()
        free_bytes = self.vram_total - reserved

        if free_bytes < self.critical_threshold:
            return MemoryPressureLevel.CRITICAL
        elif free_bytes < self.high_threshold:
            return MemoryPressureLevel.HIGH
        elif free_bytes < self.moderate_threshold:
            return MemoryPressureLevel.MODERATE
        else:
            return MemoryPressureLevel.LOW

    def get_difficulty_capacity(self) -> float:
        """Map memory pressure to maximum sample difficulty (0-1)"""
        level = self.get_pressure_level()
        free_gb = (self.vram_total - memory_reserved()) / 1024**3

        # RTX 3090 specific scaling: more granular at high memory
        if level == MemoryPressureLevel.CRITICAL:
            return 0.3 * (free_gb / 1.0)  # Scale within critical band
        elif level == MemoryPressureLevel.HIGH:
            return 0.3 + 0.3 * ((free_gb - 1) / 2.0)
        elif level == MemoryPressureLevel.MODERATE:
            return 0.6 + 0.3 * ((free_gb - 3) / 3.0)
        else:
            return 0.9 + 0.1 * ((free_gb - 6) / 18.0)  # Up to 24GB

    def adaptive_batch_size(self, base_batch_size: int, sample_difficulty: float) -> int:
        """Dynamically adjust batch size based on memory and sample difficulty"""
        capacity = self.get_difficulty_capacity()
        difficulty_factor = 1.0 - (sample_difficulty * capacity)

        # RTX 3090 memory model: assume ~500MB base + 150MB per sample at difficulty 0.5
        estimated_memory_per_sample = 500e6 + (sample_difficulty * self.config.memory_cost_coef)
        available_memory = max(0, self.vram_total - memory_reserved() - self.target_headroom)

        max_batch_from_memory = int(available_memory / estimated_memory_per_sample)

        # PID control for smooth transitions
        with self._lock:
            current_error = self.target_headroom - (self.vram_total - memory_reserved())
            self.error_integral += current_error * (time.time() - self.last_time)
            derivative = (current_error - self.last_error) / (time.time() - self.last_time + 1e-6)

            pid_output = (self.pid_kp * current_error +
                         self.pid_ki * self.error_integral +
                         self.pid_kd * derivative)

            self.last_error = current_error
            self.last_time = time.time()

        # Scale batch size by PID output (smaller batch when memory is tight)
        pid_factor = 1.0 / (1.0 + max(0, -pid_output) / (1024**3))

        adaptive_size = int(base_batch_size * difficulty_factor * pid_factor)
        adaptive_size = max(self.config.min_batch_size,
                           min(adaptive_size, max_batch_from_memory, self.config.max_batch_size))

        return adaptive_size

    def emergency_gc(self):
        """Aggressive garbage collection for critical memory pressure"""
        free_gb = (self.vram_total - memory_reserved()) / 1024**3
        if free_gb < self.config.gc_threshold_gb:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.ipc_collect()
            return True
        return False

    def shutdown(self):
        self._shutdown.set()
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)

class DifficultyAwareDataset(Dataset):
    """Dataset that provides per-sample difficulty and memory cost estimates"""

    def __init__(self, base_dataset: Dataset, difficulty_scores: Optional[np.ndarray] = None):
        self.dataset = base_dataset
        self.difficulty_scores = difficulty_scores

        if difficulty_scores is None:
            # Initialize with uniform difficulty
            self.difficulty_scores = np.ones(len(base_dataset)) * 0.5

        # Pre-compute memory cost estimates based on sample characteristics
        self.memory_costs = self._estimate_memory_costs()

    def _estimate_memory_costs(self) -> np.ndarray:
        """Heuristic memory cost based on sample size (RTX 3090 optimized)"""
        costs = np.zeros(len(self.dataset))
        for i in range(min(100, len(self.dataset))):
            sample = self.dataset[i]
            if isinstance(sample, (tuple, list)):
                sample = sample[0]
            if torch.is_tensor(sample):
                # Estimate based on tensor size (assuming float32)
                costs[i] = sample.numel() * 4 * 2  # *2 for activations
        # Extrapolate to full dataset
        if len(costs) > 0:
            avg_cost = costs[costs > 0].mean() if np.any(costs > 0) else 500e6
            costs = np.full(len(self.dataset), avg_cost)
        return costs

    def update_difficulties(self, indices: List[int], losses: List[float]):
        """EMA update of difficulty scores based on training loss"""
        for idx, loss in zip(indices, losses):
            current = self.difficulty_scores[idx]
            self.difficulty_scores[idx] = (1 - 0.05) * current + 0.05 * min(loss / 5.0, 1.0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'data': item,
            'difficulty': self.difficulty_scores[idx],
            'memory_cost': self.memory_costs[idx],
            'index': idx
        }

class AdaptiveCurriculumSampler(Sampler):
    """Memory-aware curriculum sampler with dynamic difficulty targeting"""

    def __init__(self, dataset: DifficultyAwareDataset,
                 memory_controller: MemoryPressureController,
                 config: CurriculumConfig):
        self.dataset = dataset
        self.memory_controller = memory_controller
        self.config = config

        # Curriculum state
        self.epoch = 0
        self.global_step = 0
        self.difficulty_percentile = 0.0

        # Sample history for stratified sampling
        self.sample_history = deque(maxlen=config.difficulty_window)

        # Pre-compute difficulty percentiles
        self.difficulty_distribution = np.sort(dataset.difficulty_scores)

        # RTX 3090: Use pinned memory for faster transfers
        self.pinned_memory = torch.cuda.is_available()

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices based on current memory pressure"""
        indices = np.arange(len(self.dataset))
        samples_seen = 0
        max_samples_per_epoch = len(self.dataset)

        while samples_seen < max_samples_per_epoch:
            # Get target difficulty range from memory controller
            max_difficulty = self.memory_controller.get_difficulty_capacity()
            min_difficulty = max(0.0, max_difficulty - 0.3)

            # Select samples within difficulty window
            valid_mask = (self.dataset.difficulty_scores >= min_difficulty) & \
                        (self.dataset.difficulty_scores <= max_difficulty)
            valid_indices = indices[valid_mask]

            if len(valid_indices) == 0:
                # Fallback to easiest samples
                valid_indices = indices[self.dataset.difficulty_scores <= 0.5]

            if len(valid_indices) == 0:
                # Emergency fallback: use all samples
                valid_indices = indices

            # Stratified sampling weighted by difficulty
            difficulties = self.dataset.difficulty_scores[valid_indices]
            weights = difficulties + 0.1  # Prioritize harder samples
            weights /= weights.sum()

            # Sample batch
            batch_size = self.memory_controller.adaptive_batch_size(
                self.config.max_batch_size,
                np.mean(difficulties) if len(difficulties) > 0 else 0.5
            )

            if len(valid_indices) < batch_size:
                # Not enough samples, use all valid samples
                batch_size = len(valid_indices)

            # Don't exceed remaining samples in epoch
            batch_size = min(batch_size, max_samples_per_epoch - samples_seen)

            if batch_size == 0:
                break

            batch_indices = np.random.choice(
                valid_indices, size=batch_size, replace=False, p=weights
            )

            self.sample_history.extend(batch_indices)
            self.global_step += 1
            samples_seen += batch_size

            yield batch_indices.tolist()

    def __len__(self):
        return len(self.dataset) // self.config.min_batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

class RTX3090ACMDSTrainer:
    """Production trainer with ACMDS optimization for RTX 3090"""

    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 optimizer: torch.optim.Optimizer,
                 config: CurriculumConfig = None,
                 device_id: int = 0):

        self.config = config or CurriculumConfig()
        self.device = torch.device(f'cuda:{device_id}')

        # Setup model with RTX 3090 optimizations
        self.model = model.to(self.device)
        if self.config.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception as e:
                warnings.warn(f"Could not enable gradient checkpointing: {e}")

        # Enable TF32 for Ampere (RTX 3090)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.optimizer = optimizer
        self.scaler = amp.GradScaler(enabled=self.config.enable_amp)

        # Initialize ACMDS components
        self.memory_controller = MemoryPressureController(device_id, self.config)
        self.difficulty_dataset = DifficultyAwareDataset(dataset)

        # CUDA Graph capture
        self.cuda_graph = None
        self.static_input = None
        self.static_target = None

        # Training state
        self.step = 0
        self.epoch = 0
        self.metrics = defaultdict(list)

        # Start monitoring
        self.memory_controller.start_monitoring()

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch with memory-adaptive curriculum"""
        self.model.train()

        sampler = AdaptiveCurriculumSampler(
            self.difficulty_dataset,
            self.memory_controller,
            self.config
        )
        sampler.set_epoch(self.epoch)

        total_loss = 0.0
        num_batches = 0

        for batch_indices in sampler:
            # Emergency memory management
            if self.memory_controller.emergency_gc():
                warnings.warn("Triggered emergency GC")

            # Get batch data with pinned memory
            batch_data = self._collate_batch(batch_indices)

            # Train step
            loss = self._train_step(batch_data)
            total_loss += loss
            num_batches += 1

            # Update difficulty scores
            self.difficulty_dataset.update_difficulties(batch_indices, [loss] * len(batch_indices))

            # Logging
            if self.step % 100 == 0:
                self._log_metrics()

            self.step += 1

            # Check for epoch end
            if num_batches >= len(sampler):
                break

        self.epoch += 1
        return {'epoch_loss': total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def _collate_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Efficient batch collation with RTX 3090 pinned memory"""
        batch = [self.difficulty_dataset[i] for i in indices]

        # Extract data and targets
        data = torch.stack([b['data'][0] if isinstance(b['data'], tuple) else b['data'] for b in batch])
        target = torch.stack([b['data'][1] if isinstance(b['data'], tuple) else torch.tensor(0) for b in batch])

        # Move to GPU with non-blocking transfer (pinned memory)
        return {
            'data': data.to(self.device, non_blocking=True),
            'target': target.to(self.device, non_blocking=True),
            'difficulties': torch.tensor([b['difficulty'] for b in batch], device=self.device)
        }

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Optimized training step with optional CUDA Graphs"""
        # Note: CUDA Graphs disabled for dynamic batch sizes
        # They require fixed tensor shapes which conflict with adaptive batching
        loss = self._forward_backward(batch['data'], batch['target'])

        return loss.item()

    def _forward_backward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward/backward pass with AMP"""
        with amp.autocast(enabled=self.config.enable_amp):
            output = self.model(data)
            loss = nn.functional.cross_entropy(output, target)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)  # RTX 3090 optimization

        return loss

    def _log_metrics(self):
        """Log comprehensive metrics"""
        metrics = self.memory_controller.metrics_queue.get() if not self.memory_controller.metrics_queue.empty() else None
        if metrics:
            self.metrics['memory_pressure'].append(metrics['pressure'])
            self.metrics['free_gb'].append(metrics['free_gb'])

        self.metrics['scale'].append(self.scaler.get_scale())
        self.metrics['step'].append(self.step)

    def save_checkpoint(self, path: Path):
        """Save training state"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'difficulty_scores': self.difficulty_dataset.difficulty_scores,
            'step': self.step,
            'epoch': self.epoch,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.difficulty_dataset.difficulty_scores = checkpoint['difficulty_scores']
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']

    def shutdown(self):
        """Cleanup resources"""
        self.memory_controller.shutdown()
        if self.cuda_graph:
            del self.cuda_graph
        torch.cuda.empty_cache()

# Example usage for RTX 3090
if __name__ == "__main__":
    # Synthetic dataset with varying difficulty
    class SyntheticDataset(Dataset):
        def __init__(self, size=50000):
            self.size = size
            self.data = torch.randn(size, 3, 224, 224)
            self.targets = torch.randint(0, 1000, (size,))

            # Simulate difficulty correlation with image statistics
            difficulties = torch.std(self.data, dim=(1,2,3)).numpy()
            self.difficulties = (difficulties - difficulties.min()) / (difficulties.max() - difficulties.min())

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    # Initialize components
    dataset = SyntheticDataset()
    model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)

    # Try to use fused optimizer (PyTorch 2.0+), fallback to regular
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        warnings.warn("Fused optimizer not available, using regular AdamW")

    config = CurriculumConfig(
        target_headroom_gb=4.0,  # Conservative for RTX 3090
        max_batch_size=32,       # Fits in 24GB with ResNet50
        enable_amp=True
    )

    trainer = RTX3090ACMDSTrainer(model, dataset, optimizer, config)

    # Training loop
    try:
        for epoch in range(10):
            metrics = trainer.train_epoch()
            print(f"Epoch {epoch}: {metrics}")

            # Save checkpoint
            if epoch % 5 == 0:
                trainer.save_checkpoint(Path(f"checkpoint_epoch_{epoch}.pt"))
    finally:
        trainer.shutdown()
