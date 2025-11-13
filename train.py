#!/usr/bin/env python3
"""
Command-line interface for ACMDS optimizer training
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

from acmds_optimizer import (
    CurriculumConfig,
    RTX3090ACMDSTrainer,
    DifficultyAwareDataset
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='ACMDS: Adaptive Curriculum with Memory-Driven Difficulty Scaling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and data arguments
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture (resnet50, resnet101, efficientnet_b0, custom)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset path or name (cifar10, cifar100, imagenet, custom)')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of output classes')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='Optimizer type')

    # ACMDS configuration
    parser.add_argument('--target-headroom-gb', type=float, default=3.0,
                        help='Target GPU memory headroom in GB')
    parser.add_argument('--min-batch-size', type=int, default=4,
                        help='Minimum batch size')
    parser.add_argument('--max-batch-size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('--difficulty-window', type=int, default=1000,
                        help='Window size for difficulty tracking')

    # PID controller parameters
    parser.add_argument('--pid-kp', type=float, default=0.15,
                        help='PID proportional gain')
    parser.add_argument('--pid-ki', type=float, default=0.02,
                        help='PID integral gain')
    parser.add_argument('--pid-kd', type=float, default=0.08,
                        help='PID derivative gain')

    # Optimization flags
    parser.add_argument('--enable-amp', action='store_true', default=True,
                        help='Enable automatic mixed precision')
    parser.add_argument('--no-amp', dest='enable_amp', action='store_false',
                        help='Disable automatic mixed precision')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing (model-dependent)')
    parser.add_argument('--fused-optimizer', action='store_true',
                        help='Use fused optimizer (PyTorch 2.0+)')

    # Device and system
    parser.add_argument('--device-id', type=int, default=0,
                        help='CUDA device ID')
    parser.add_argument('--gc-threshold-gb', type=float, default=2.0,
                        help='Trigger garbage collection when free memory below this (GB)')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log metrics every N steps')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to log file')

    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON configuration file (overrides CLI args)')

    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_curriculum_config(args) -> CurriculumConfig:
    """Create CurriculumConfig from arguments"""
    return CurriculumConfig(
        target_headroom_gb=args.target_headroom_gb,
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
        difficulty_window=args.difficulty_window,
        pid_kp=args.pid_kp,
        pid_ki=args.pid_ki,
        pid_kd=args.pid_kd,
        enable_amp=args.enable_amp,
        gradient_checkpointing=args.gradient_checkpointing,
        gc_threshold_gb=args.gc_threshold_gb
    )


def load_model(model_name: str, num_classes: int) -> nn.Module:
    """Load model architecture"""
    model_name = model_name.lower()

    if model_name == 'resnet50':
        model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'resnet101':
        model = torch.hub.load('pytorch/vision', 'resnet101', pretrained=False)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        model = torch.hub.load('pytorch/vision', 'efficientnet_b0', pretrained=False)
        if num_classes != 1000:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == 'custom':
        raise NotImplementedError(
            "Custom model support: Please modify load_model() to load your custom model"
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def load_dataset(dataset_name: str, dataset_path: Optional[str] = None) -> Dataset:
    """Load dataset"""
    dataset_name = dataset_name.lower()

    if dataset_name == 'cifar10':
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        return datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    elif dataset_name == 'cifar100':
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        return datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    elif dataset_name == 'imagenet':
        from torchvision import datasets, transforms
        if dataset_path is None:
            raise ValueError("ImageNet requires --dataset-path argument")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return datasets.ImageFolder(dataset_path, transform=transform)

    elif dataset_name == 'custom':
        raise NotImplementedError(
            "Custom dataset support: Please modify load_dataset() to load your custom dataset"
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Create optimizer"""
    optimizer_name = args.optimizer.lower()

    if optimizer_name == 'adamw':
        if args.fused_optimizer:
            try:
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    fused=True
                )
            except TypeError:
                warnings.warn("Fused optimizer not available, using regular AdamW")
                return torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )
        else:
            return torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

    elif optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def setup_logging(log_file: Optional[str] = None):
    """Setup logging"""
    import logging

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def main():
    """Main training loop"""
    args = parse_args()

    # Load config from file if provided
    if args.config:
        config_dict = load_config_from_file(args.config)
        # Update args with config file values
        for key, value in config_dict.items():
            setattr(args, key, value)

    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting ACMDS training")
    logger.info(f"Arguments: {vars(args)}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This optimizer requires an NVIDIA GPU.")
        sys.exit(1)

    # Load model and dataset
    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, args.num_classes)

    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)

    # Create optimizer
    logger.info(f"Creating optimizer: {args.optimizer}")
    optimizer = create_optimizer(model, args)

    # Create curriculum config
    config = create_curriculum_config(args)
    logger.info(f"Curriculum config: {config}")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    logger.info("Initializing ACMDS trainer")
    trainer = RTX3090ACMDSTrainer(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        config=config,
        device_id=args.device_id
    )

    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Training loop
    try:
        logger.info(f"Starting training for {args.epochs} epochs")
        for epoch in range(trainer.epoch, args.epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{args.epochs}")
            logger.info(f"{'='*60}")

            # Train one epoch
            metrics = trainer.train_epoch()

            # Log metrics
            logger.info(f"Epoch {epoch} completed:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")

            # Memory stats
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(args.device_id) / 1024**3
                reserved = torch.cuda.memory_reserved(args.device_id) / 1024**3
                logger.info(f"  GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            # Save checkpoint
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                trainer.save_checkpoint(checkpoint_path)

        # Final checkpoint
        final_checkpoint = checkpoint_dir / "checkpoint_final.pt"
        logger.info(f"Saving final checkpoint to {final_checkpoint}")
        trainer.save_checkpoint(final_checkpoint)

        logger.info("\nTraining completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        logger.info("Shutting down trainer")
        trainer.shutdown()


if __name__ == "__main__":
    main()
