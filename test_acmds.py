"""
Comprehensive test suite for ACMDS optimizer
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pytest
import time
from pathlib import Path
import tempfile
import warnings

from acmds_optimizer import (
    MemoryPressureLevel,
    CurriculumConfig,
    MemoryPressureController,
    DifficultyAwareDataset,
    AdaptiveCurriculumSampler,
    RTX3090ACMDSTrainer
)


# Test fixtures
class SimpleDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, size=1000, input_dim=32):
        self.size = size
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_dim=32, hidden_dim=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def config():
    """Default test config"""
    return CurriculumConfig(
        target_headroom_gb=1.0,
        min_batch_size=2,
        max_batch_size=16,
        enable_amp=True,
        gradient_checkpointing=False
    )


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# Config Tests
class TestCurriculumConfig:
    def test_default_config(self):
        """Test default config values"""
        config = CurriculumConfig()
        assert config.target_headroom_gb == 3.0
        assert config.min_batch_size == 4
        assert config.max_batch_size == 64
        assert config.enable_amp is True

    def test_custom_config(self):
        """Test custom config values"""
        config = CurriculumConfig(
            target_headroom_gb=5.0,
            max_batch_size=128
        )
        assert config.target_headroom_gb == 5.0
        assert config.max_batch_size == 128


# Memory Pressure Controller Tests
class TestMemoryPressureController:
    def test_init_requires_cuda(self):
        """Test that controller requires CUDA"""
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                MemoryPressureController()

    def test_init_with_cuda(self, skip_if_no_cuda, config):
        """Test controller initialization with CUDA"""
        controller = MemoryPressureController(device_id=0, config=config)
        assert controller.device.type == 'cuda'
        assert controller.vram_total > 0

    def test_pressure_levels(self, skip_if_no_cuda, config):
        """Test pressure level detection"""
        controller = MemoryPressureController(device_id=0, config=config)
        level = controller.get_pressure_level()
        assert isinstance(level, MemoryPressureLevel)

    def test_difficulty_capacity(self, skip_if_no_cuda, config):
        """Test difficulty capacity calculation"""
        controller = MemoryPressureController(device_id=0, config=config)
        capacity = controller.get_difficulty_capacity()
        assert 0.0 <= capacity <= 1.0

    def test_adaptive_batch_size(self, skip_if_no_cuda, config):
        """Test adaptive batch size computation"""
        controller = MemoryPressureController(device_id=0, config=config)
        batch_size = controller.adaptive_batch_size(32, 0.5)
        assert config.min_batch_size <= batch_size <= config.max_batch_size

    def test_emergency_gc(self, skip_if_no_cuda, config):
        """Test emergency garbage collection"""
        controller = MemoryPressureController(device_id=0, config=config)
        result = controller.emergency_gc()
        assert isinstance(result, bool)

    def test_monitoring_thread(self, skip_if_no_cuda, config):
        """Test memory monitoring thread"""
        controller = MemoryPressureController(device_id=0, config=config)
        controller.start_monitoring(interval_ms=100)
        time.sleep(0.3)  # Let it collect some metrics
        assert not controller.metrics_queue.empty()
        controller.shutdown()

    def test_shutdown(self, skip_if_no_cuda, config):
        """Test controller shutdown"""
        controller = MemoryPressureController(device_id=0, config=config)
        controller.start_monitoring()
        controller.shutdown()
        # Thread should be stopped
        time.sleep(0.2)
        assert controller._shutdown.is_set()


# Dataset Tests
class TestDifficultyAwareDataset:
    def test_init_with_base_dataset(self):
        """Test initialization with base dataset"""
        base_dataset = SimpleDataset(size=100)
        dataset = DifficultyAwareDataset(base_dataset)
        assert len(dataset) == 100
        assert dataset.difficulty_scores.shape == (100,)
        assert np.all((dataset.difficulty_scores >= 0) & (dataset.difficulty_scores <= 1))

    def test_init_with_custom_difficulties(self):
        """Test initialization with custom difficulty scores"""
        base_dataset = SimpleDataset(size=100)
        difficulties = np.random.rand(100)
        dataset = DifficultyAwareDataset(base_dataset, difficulties)
        assert np.array_equal(dataset.difficulty_scores, difficulties)

    def test_getitem(self):
        """Test getting items from dataset"""
        base_dataset = SimpleDataset(size=10)
        dataset = DifficultyAwareDataset(base_dataset)
        item = dataset[0]
        assert 'data' in item
        assert 'difficulty' in item
        assert 'memory_cost' in item
        assert 'index' in item

    def test_update_difficulties(self):
        """Test difficulty score updates"""
        base_dataset = SimpleDataset(size=10)
        dataset = DifficultyAwareDataset(base_dataset)
        initial_difficulties = dataset.difficulty_scores.copy()

        # Update with high losses
        dataset.update_difficulties([0, 1, 2], [10.0, 8.0, 12.0])

        # Difficulties should have changed
        assert not np.array_equal(initial_difficulties, dataset.difficulty_scores)
        # Updated samples should have higher difficulty
        assert dataset.difficulty_scores[0] > initial_difficulties[0]

    def test_memory_cost_estimation(self):
        """Test memory cost estimation"""
        base_dataset = SimpleDataset(size=100, input_dim=64)
        dataset = DifficultyAwareDataset(base_dataset)
        assert len(dataset.memory_costs) == 100
        assert np.all(dataset.memory_costs > 0)


# Sampler Tests
class TestAdaptiveCurriculumSampler:
    def test_init(self, skip_if_no_cuda, config):
        """Test sampler initialization"""
        base_dataset = SimpleDataset(size=100)
        dataset = DifficultyAwareDataset(base_dataset)
        controller = MemoryPressureController(device_id=0, config=config)
        sampler = AdaptiveCurriculumSampler(dataset, controller, config)
        assert sampler.epoch == 0
        assert sampler.global_step == 0

    def test_iter_yields_batches(self, skip_if_no_cuda, config):
        """Test that sampler yields batches"""
        base_dataset = SimpleDataset(size=100)
        dataset = DifficultyAwareDataset(base_dataset)
        controller = MemoryPressureController(device_id=0, config=config)
        sampler = AdaptiveCurriculumSampler(dataset, controller, config)

        batches = []
        for i, batch_indices in enumerate(sampler):
            batches.append(batch_indices)
            if i >= 5:  # Just test a few batches
                break

        assert len(batches) > 0
        for batch in batches:
            assert len(batch) >= config.min_batch_size
            assert len(batch) <= config.max_batch_size

    def test_set_epoch(self, skip_if_no_cuda, config):
        """Test epoch setting"""
        base_dataset = SimpleDataset(size=100)
        dataset = DifficultyAwareDataset(base_dataset)
        controller = MemoryPressureController(device_id=0, config=config)
        sampler = AdaptiveCurriculumSampler(dataset, controller, config)

        sampler.set_epoch(5)
        assert sampler.epoch == 5

    def test_sampler_length(self, skip_if_no_cuda, config):
        """Test sampler length calculation"""
        base_dataset = SimpleDataset(size=100)
        dataset = DifficultyAwareDataset(base_dataset)
        controller = MemoryPressureController(device_id=0, config=config)
        sampler = AdaptiveCurriculumSampler(dataset, controller, config)

        length = len(sampler)
        assert length == len(dataset) // config.min_batch_size


# Trainer Tests
class TestRTX3090ACMDSTrainer:
    def test_init(self, skip_if_no_cuda, config):
        """Test trainer initialization"""
        model = SimpleModel()
        dataset = SimpleDataset(size=100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer = RTX3090ACMDSTrainer(model, dataset, optimizer, config, device_id=0)
        assert trainer.model.training is False  # Model starts in eval mode
        assert trainer.device.type == 'cuda'
        assert trainer.epoch == 0
        assert trainer.step == 0

    def test_train_epoch(self, skip_if_no_cuda, config):
        """Test training for one epoch"""
        model = SimpleModel()
        dataset = SimpleDataset(size=50)  # Small dataset for quick test
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        config.max_batch_size = 8
        trainer = RTX3090ACMDSTrainer(model, dataset, optimizer, config, device_id=0)

        # Train for one epoch
        metrics = trainer.train_epoch()

        assert 'epoch_loss' in metrics
        assert isinstance(metrics['epoch_loss'], float)
        assert trainer.epoch == 1
        assert trainer.step > 0

    def test_checkpoint_save_load(self, skip_if_no_cuda, config):
        """Test checkpoint saving and loading"""
        model = SimpleModel()
        dataset = SimpleDataset(size=50)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer = RTX3090ACMDSTrainer(model, dataset, optimizer, config, device_id=0)

        # Train a bit
        trainer.train_epoch()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            # Create new trainer and load
            new_model = SimpleModel()
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            new_trainer = RTX3090ACMDSTrainer(
                new_model, dataset, new_optimizer, config, device_id=0
            )

            new_trainer.load_checkpoint(checkpoint_path)

            # Check states match
            assert new_trainer.epoch == trainer.epoch
            assert new_trainer.step == trainer.step

    def test_collate_batch(self, skip_if_no_cuda, config):
        """Test batch collation"""
        model = SimpleModel()
        dataset = SimpleDataset(size=50)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer = RTX3090ACMDSTrainer(model, dataset, optimizer, config, device_id=0)

        # Test collation
        batch_data = trainer._collate_batch([0, 1, 2, 3])

        assert 'data' in batch_data
        assert 'target' in batch_data
        assert 'difficulties' in batch_data
        assert batch_data['data'].device.type == 'cuda'
        assert batch_data['target'].device.type == 'cuda'

    def test_shutdown(self, skip_if_no_cuda, config):
        """Test trainer shutdown"""
        model = SimpleModel()
        dataset = SimpleDataset(size=50)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        trainer = RTX3090ACMDSTrainer(model, dataset, optimizer, config, device_id=0)
        trainer.shutdown()

        # Memory controller should be shut down
        assert trainer.memory_controller._shutdown.is_set()


# Integration Tests
class TestIntegration:
    def test_full_training_loop(self, skip_if_no_cuda):
        """Test a complete training loop"""
        model = SimpleModel()
        dataset = SimpleDataset(size=100)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        config = CurriculumConfig(
            target_headroom_gb=1.0,
            min_batch_size=4,
            max_batch_size=16,
            enable_amp=True
        )

        trainer = RTX3090ACMDSTrainer(model, dataset, optimizer, config, device_id=0)

        # Train for a few epochs
        try:
            for epoch in range(3):
                metrics = trainer.train_epoch()
                assert 'epoch_loss' in metrics
                print(f"Epoch {epoch}: {metrics}")
        finally:
            trainer.shutdown()

    def test_memory_adaptation(self, skip_if_no_cuda):
        """Test that batch sizes adapt to memory pressure"""
        model = SimpleModel(input_dim=512, hidden_dim=1024)  # Larger model
        dataset = SimpleDataset(size=200, input_dim=512)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        config = CurriculumConfig(
            target_headroom_gb=0.5,  # Tight memory constraint
            min_batch_size=2,
            max_batch_size=64,
            enable_amp=True
        )

        trainer = RTX3090ACMDSTrainer(model, dataset, optimizer, config, device_id=0)

        try:
            # Train and observe batch size adaptation
            metrics = trainer.train_epoch()
            assert trainer.step > 0
        finally:
            trainer.shutdown()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
