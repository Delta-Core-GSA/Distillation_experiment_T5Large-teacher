"""
Essential utilities for the distillation project
"""
import os
import json
import random
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import torch
import numpy as np
import psutil

def set_seed(seed: int):
    """
    Set seed for reproducibility
    
    Args:
        seed: Seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Random seed set to: {seed}")

def setup_logging(log_dir: str):
    """
    Configure the logging system
    
    Args:
        log_dir: Directory for log files
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # File name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging (hardcoded to INFO)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging configured. File: {log_file}")

def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
    Returns:
        Dict with configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logging.info(f"Configuration loaded from: {config_path}")
    return config

def save_checkpoint(state: Dict, checkpoint_path: str, 
                   is_best: bool = False,
                   best_path: Optional[str] = None):
    """
    Save model checkpoint
    
    Args:
        state: State dict to save
        checkpoint_path: Checkpoint path
        is_best: Whether it's the best model so far
        best_path: Path to save the best model
    """
    torch.save(state, checkpoint_path)
    logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    if is_best and best_path:
        torch.save(state, best_path)
        logging.info(f"Best model updated: {best_path}")

def print_system_info():
    """Print system information"""
    logging.info("=" * 50)
    logging.info("SYSTEM INFO")
    logging.info("=" * 50)
    
    # Python version
    logging.info(f"Python: {sys.version}")
    
    # PyTorch version
    logging.info(f"PyTorch: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        logging.info(f"CUDA available: True")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logging.info(f"  Memory: {memory:.2f} GB")
    else:
        logging.info("CUDA available: False")
    
    # CPU info
    logging.info(f"CPU cores: {psutil.cpu_count()}")
    
    # Memory info
    memory = psutil.virtual_memory()
    logging.info(f"RAM: {memory.total / 1e9:.2f} GB")
    logging.info(f"Available RAM: {memory.available / 1e9:.2f} GB")
    
    logging.info("=" * 50)

class AverageMeter:
    """Calculates and stores progressive averages"""
    
    def __init__(self, name: str, fmt: str = ':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    """Shows training progress"""
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class EarlyStopping:
    """Early stopping for training"""
    
    def __init__(self, patience: int = 7, verbose: bool = False, 
                 delta: float = 0, mode: str = 'max'):
        """
        Args:
            patience: Epochs without improvement before stopping
            verbose: Whether to print messages
            delta: Minimum required improvement
            mode: 'max' or 'min' to maximize/minimize metric
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.mode == 'max':
            score = val_score
        else:
            score = -val_score
        
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                logging.info(f'Validation score initialized to {val_score:.4f}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                logging.info(f'Validation score improved to {val_score:.4f}')