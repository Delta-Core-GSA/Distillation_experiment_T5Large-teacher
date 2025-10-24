"""
Module for teacher training and student distillation
"""
import os
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq
)
from torch.cuda.amp import autocast, GradScaler

from data_loader import TextGenerationDataset, download_and_prepare_dataset
from metrics import compute_generation_metrics
from utils import (
    save_checkpoint,
    EarlyStopping,
    AverageMeter,
    ProgressMeter
)


class BaseTrainer:
    """Base class for training"""
    
    def __init__(self, config, device, output_dir, checkpoint_dir, 
                 num_workers=4, mixed_precision=False):
        
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.best_metric = float('-inf')
        self.current_epoch = 0
    
    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        """Log metrics to console"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"{prefix} | Step: {step} | {metrics_str}", flush=True)
    
    def prepare_data(self):
        """Prepare datasets"""
        print("Preparing datasets...", flush=True)
        
        # Download and prepare dataset
        data_path = download_and_prepare_dataset(
            self.config['dataset']['name'],
            self.config['dataset'].get('cache_dir', './data')
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        # Create datasets
        train_dataset = TextGenerationDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            split='train',
            max_length=self.config['dataset']['max_length'],
            max_samples=self.config['dataset'].get('max_train_samples', None)
        )
        
        val_dataset = TextGenerationDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            split='validation',
            max_length=self.config['dataset']['max_length'],
            max_samples=self.config['dataset'].get('max_val_samples', None)
        )
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=None,
            label_pad_token_id=-100
        )
        
        # Create DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=data_collator,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['eval_batch_size'],
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_collator,
            pin_memory=True
        )
        
        print(f"Datasets loaded - Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
        return len(train_dataset), len(val_dataset)


class TeacherTrainer(BaseTrainer):
    """Trainer for the teacher model"""
    
    def __init__(self, resume_from_checkpoint=None, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.resume_from_checkpoint = resume_from_checkpoint
        
    def setup_model(self):
        """Initialize the teacher model"""
        print(f"Loading teacher model: {self.config['model']['name']}", flush=True)
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['model']['name'],
            torch_dtype=torch.float16 if self.mixed_precision else torch.float32
        )
        
        self.model = self.model.to(self.device)
        
        # Log parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}", flush=True)
        print(f"Trainable parameters: {trainable_params:,}", flush=True)
        
    def setup_optimization(self, num_training_steps):
        """Setup optimizer and scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01),
            eps=self.config['training'].get('adam_epsilon', 1e-8)
        )
        
        num_warmup_steps = int(num_training_steps * self.config['training'].get('warmup_ratio', 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        if self.mixed_precision:
            self.scaler = GradScaler()
        
        print(f"Optimizer configured - LR: {self.config['training']['learning_rate']}", flush=True)
        print(f"Scheduler configured - Warmup steps: {num_warmup_steps}", flush=True)
    
    def train_epoch(self, epoch):
        """Training for a single epoch"""
        self.model.train()
        
        losses = AverageMeter('Loss', ':.4f')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses],
            prefix=f"Epoch: [{epoch}]"
        )
        
        end = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config['training'].get('max_grad_norm', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                              self.config['training'].get('max_grad_norm', 1.0))
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            losses.update(loss.item(), batch['input_ids'].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Log progress
            if batch_idx % self.config['training'].get('log_interval', 10) == 0:
                progress.display(batch_idx)
                
        return losses.avg
    
    def validate(self, epoch):
        """Validation loop"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_references = []
        
        print(f"Validating epoch {epoch}...", flush=True)
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                # Generate predictions (limit to 100 samples)
                if len(all_predictions) < 100:
                    generated = self.model.generate(
                        batch['input_ids'],
                        max_length=self.config['dataset']['max_length'],
                        num_beams=self.config['generation'].get('num_beams', 4),
                        early_stopping=True
                    )
                    
                    predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                    references = self.tokenizer.batch_decode(
                        batch['labels'].masked_fill(batch['labels'] == -100, 0),
                        skip_special_tokens=True
                    )
                    
                    all_predictions.extend(predictions)
                    all_references.extend(references)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_generation_metrics(all_predictions, all_references)
        metrics['loss'] = avg_loss
        
        self.log_metrics(metrics, epoch, prefix='Validation')
        
        return metrics
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded", flush=True)
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded", flush=True)
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded", flush=True)
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Scaler state loaded", flush=True)
        
        # Recover epoch and metrics
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'metrics' in checkpoint and 'rouge_l' in checkpoint['metrics']:
            self.best_metric = checkpoint['metrics']['rouge_l']
            print(f"Best metric (ROUGE-L): {self.best_metric:.4f}", flush=True)
        
        print(f"Resuming from epoch {start_epoch}", flush=True)
        return start_epoch
    
    def train(self):
        """Main training loop"""
        # Prepare data
        train_size, val_size = self.prepare_data()
        
        # Setup model
        self.setup_model()
        
        # Setup optimization
        num_epochs = self.config['training']['num_epochs']
        num_training_steps = num_epochs * len(self.train_loader)
        self.setup_optimization(num_training_steps)
        
        # Load checkpoint if specified
        start_epoch = 1
        if self.resume_from_checkpoint:
            start_epoch = self.load_checkpoint(self.resume_from_checkpoint)
            
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training'].get('early_stopping_patience', 3),
            verbose=True
        )
        
        # Training loop
        print("=" * 50, flush=True)
        print("Starting training", flush=True)
        print(f"Model: {self.config['model']['name']}", flush=True)
        print(f"Epochs: {num_epochs}", flush=True)
        print(f"Train samples: {train_size}", flush=True)
        print(f"Val samples: {val_size}", flush=True)
        print("=" * 50, flush=True)
        
        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            print(f"\n--- Epoch {epoch}/{num_epochs} ---", flush=True)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Log summary
            print(f"Epoch {epoch} Summary:", flush=True)
            print(f"  Train Loss: {train_loss:.4f}", flush=True)
            print(f"  Val Loss: {val_metrics['loss']:.4f}", flush=True)
            print(f"  Val ROUGE-L: {val_metrics.get('rouge_l', 0):.4f}", flush=True)
            
            # Save checkpoint
            is_best = val_metrics.get('rouge_l', 0) > self.best_metric
            if is_best:
                self.best_metric = val_metrics.get('rouge_l', 0)
                
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() if self.mixed_precision else None,
                    'metrics': val_metrics,
                    'config': self.config
                },
                checkpoint_path,
                is_best,
                self.checkpoint_dir / 'best_model.pt'
            )
            
            # Early stopping
            early_stopping(val_metrics.get('rouge_l', 0))
            if early_stopping.early_stop:
                print("Early stopping triggered!", flush=True)
                break
        
        print("Training completed!", flush=True)
        return self.best_metric
    
    def save_final_model(self):
        """Save the final teacher model"""
        model_path = self.output_dir / 'final_teacher_model'
        model_path.mkdir(exist_ok=True)
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save config
        with open(model_path / 'training_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Model saved to: {model_path}", flush=True)
        return str(model_path)
    
    def evaluate(self):
        """Final evaluation of the teacher"""
        if self.model is None:
            self.setup_model()
        
        metrics = self.validate(self.current_epoch)
        return metrics


class DistillationTrainer(BaseTrainer):
    """Trainer for knowledge distillation"""
    
    def __init__(self, teacher_checkpoint=None, student_config=None, 
                 resume_from_checkpoint=None, distiller_type='standard', **kwargs):
        super().__init__(**kwargs)
        
        self.teacher_model = None
        self.student_model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        self.teacher_checkpoint = teacher_checkpoint
        self.student_config = student_config or {}
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Distillation parameters
        self.distiller_type = distiller_type
        self.alpha = self.config['distillation'].get('alpha', 0.5)
        self.temperature = self.config['distillation'].get('temperature', 3.0)
        
    def setup_models(self):
        """Setup teacher and student models"""
        # Load teacher
        print(f"Loading teacher model: {self.config['model']['name']}", flush=True)
        self.teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['model']['name'],
            torch_dtype=torch.float16 if self.mixed_precision else torch.float32
        )
        
        # Load teacher checkpoint if provided
        if self.teacher_checkpoint:
            print(f"Loading teacher checkpoint: {self.teacher_checkpoint}", flush=True)
            checkpoint = torch.load(self.teacher_checkpoint, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
        
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()  # Teacher always in eval mode
        
        # Initialize student
        student_model_name = self.student_config.get('name', self.config['model']['name'])
        print(f"Initializing student model: {student_model_name}", flush=True)
        
        if 'architecture' in self.student_config:
            # Custom student architecture
            print("Creating custom student architecture", flush=True)
            from models import create_student_model
            self.student_model = create_student_model(
                teacher_config=self.teacher_model.config,
                **self.student_config['architecture']
            )
        else:
            # Use pretrained model as student
            self.student_model = AutoModelForSeq2SeqLM.from_pretrained(
                student_model_name,
                torch_dtype=torch.float16 if self.mixed_precision else torch.float32
            )
        
        self.student_model = self.student_model.to(self.device)
        
        # Log model sizes
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        print(f"Teacher parameters: {teacher_params:,}", flush=True)
        print(f"Student parameters: {student_params:,}", flush=True)
        print(f"Compression ratio: {teacher_params/student_params:.2f}x", flush=True)
    
    def compute_distillation_loss(self, student_outputs, teacher_outputs, labels):
        """Calculate the distillation loss"""
        # KL divergence loss between student and teacher
        kd_loss = nn.KLDivLoss(reduction='batchmean')(
            nn.functional.log_softmax(student_outputs.logits / self.temperature, dim=-1),
            nn.functional.softmax(teacher_outputs.logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = student_outputs.loss
        
        # Combined loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, kd_loss, ce_loss
    
    def distill_epoch(self, epoch):
        """Distillation for a single epoch"""
        self.student_model.train()
        self.teacher_model.eval()
        
        total_losses = AverageMeter('Total Loss', ':.4f')
        kd_losses = AverageMeter('KD Loss', ':.4f')
        ce_losses = AverageMeter('CE Loss', ':.4f')
        batch_time = AverageMeter('Time', ':6.3f')
        
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, total_losses, kd_losses, ce_losses],
            prefix=f"Distillation Epoch: [{epoch}]"
        )
        
        end = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Teacher forward pass (no gradient)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**batch)
            
            # Student forward pass
            if self.mixed_precision:
                with autocast():
                    student_outputs = self.student_model(**batch)
                    total_loss, kd_loss, ce_loss = self.compute_distillation_loss(
                        student_outputs, teacher_outputs, batch['labels']
                    )
            else:
                student_outputs = self.student_model(**batch)
                total_loss, kd_loss, ce_loss = self.compute_distillation_loss(
                    student_outputs, teacher_outputs, batch['labels']
                )
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.config['training'].get('max_grad_norm', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.config['training'].get('max_grad_norm', 1.0)
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            batch_size = batch['input_ids'].size(0)
            total_losses.update(total_loss.item(), batch_size)
            kd_losses.update(kd_loss.item(), batch_size)
            ce_losses.update(ce_loss.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Log progress
            if batch_idx % self.config['training'].get('log_interval', 10) == 0:
                progress.display(batch_idx)
        
        return total_losses.avg, kd_losses.avg, ce_losses.avg
    
    def validate_student(self, epoch):
        """Validate the student model"""
        self.student_model.eval()
        
        total_loss = 0
        all_predictions = []
        all_references = []
        
        print(f"Validating student model...", flush=True)
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.student_model(**batch)
                total_loss += outputs.loss.item()
                
                # Generate predictions (limit to 100)
                if len(all_predictions) < 100:
                    generated = self.student_model.generate(
                        batch['input_ids'],
                        max_length=self.config['dataset']['max_length'],
                        num_beams=self.config['generation'].get('num_beams', 4),
                        early_stopping=True
                    )
                    
                    predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                    references = self.tokenizer.batch_decode(
                        batch['labels'].masked_fill(batch['labels'] == -100, 0),
                        skip_special_tokens=True
                    )
                    
                    all_predictions.extend(predictions)
                    all_references.extend(references)
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_generation_metrics(all_predictions, all_references)
        metrics['loss'] = avg_loss
        
        self.log_metrics(metrics, epoch, prefix='student_val')
        return metrics
    
    def load_checkpoint_for_resume(self, checkpoint_path):
        """Load checkpoint to resume distillation"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading distillation checkpoint from: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load student states
        if 'model_state_dict' in checkpoint:
            self.student_model.load_state_dict(checkpoint['model_state_dict'])
            print("Student model state loaded", flush=True)
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded", flush=True)
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded", flush=True)
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Scaler state loaded", flush=True)
        
        # Recover epoch and metrics
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'metrics' in checkpoint and 'rouge_l' in checkpoint['metrics']:
            self.best_metric = checkpoint['metrics']['rouge_l']
            print(f"Best metric (ROUGE-L): {self.best_metric:.4f}", flush=True)
        
        # Log distillation losses if present
        if 'distillation_losses' in checkpoint:
            losses = checkpoint['distillation_losses']
            print(f"Distillation losses from last checkpoint:", flush=True)
            print(f"  Total: {losses.get('total', 'N/A'):.4f}", flush=True)
            print(f"  KD: {losses.get('kd', 'N/A'):.4f}", flush=True)
            print(f"  CE: {losses.get('ce', 'N/A'):.4f}", flush=True)
        
        print(f"Resuming distillation from epoch {start_epoch}", flush=True)
        return start_epoch
    
    def distill(self):
        """Main distillation loop"""
        # Setup
        train_size, val_size = self.prepare_data()
        self.setup_models()
        
        # Setup optimization for student
        num_epochs = self.config['training']['num_epochs']
        num_training_steps = num_epochs * len(self.train_loader)
        
        self.optimizer = AdamW(
            self.student_model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )
        
        num_warmup_steps = int(num_training_steps * self.config['training'].get('warmup_ratio', 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        if self.mixed_precision:
            self.scaler = GradScaler()
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training'].get('early_stopping_patience', 3),
            verbose=True
        )
        
        # Load checkpoint if specified
        start_epoch = 1
        if self.resume_from_checkpoint:
            start_epoch = self.load_checkpoint_for_resume(self.resume_from_checkpoint)

        # Distillation loop
        print("=" * 50, flush=True)
        print("Starting distillation", flush=True)
        if self.resume_from_checkpoint:
            print(f"Resuming from epoch {start_epoch}", flush=True)
        print(f"Distiller type: {self.distiller_type}", flush=True)
        print(f"Temperature: {self.config['distillation']['temperature']}", flush=True)
        print(f"Alpha: {self.alpha}", flush=True)
        print("=" * 50, flush=True)

        for epoch in range(start_epoch, num_epochs + 1):
            self.current_epoch = epoch
            print(f"\n--- Distillation Epoch {epoch}/{num_epochs} ---", flush=True)
            
            # Distill
            total_loss, kd_loss, ce_loss = self.distill_epoch(epoch)
            
            # Validate student
            val_metrics = self.validate_student(epoch)
            
            # Log summary
            print(f"Epoch {epoch} Summary:", flush=True)
            print(f"  Total Loss: {total_loss:.4f}", flush=True)
            print(f"  KD Loss: {kd_loss:.4f}", flush=True)
            print(f"  CE Loss: {ce_loss:.4f}", flush=True)
            print(f"  Val Loss: {val_metrics['loss']:.4f}", flush=True)
            print(f"  Val ROUGE-L: {val_metrics.get('rouge_l', 0):.4f}", flush=True)
            
            # Save checkpoint
            is_best = val_metrics.get('rouge_l', 0) > self.best_metric
            if is_best:
                self.best_metric = val_metrics.get('rouge_l', 0)
            
            checkpoint_path = self.checkpoint_dir / f'student_checkpoint_epoch_{epoch}.pt'
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'metrics': val_metrics,
                    'config': self.config,
                    'distillation_losses': {
                        'total': total_loss,
                        'kd': kd_loss,
                        'ce': ce_loss
                    }
                },
                checkpoint_path,
                is_best,
                self.checkpoint_dir / 'best_student.pt'
            )
            
            # Early stopping
            early_stopping(val_metrics.get('rouge_l', 0))
            if early_stopping.early_stop:
                print("Early stopping triggered!", flush=True)
                break
        
        print("Distillation completed!", flush=True)
        return self.best_metric
    
    def save_final_model(self):
        """Save the final student model"""
        model_path = self.output_dir / 'final_student_model'
        model_path.mkdir(exist_ok=True)
        
        self.student_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        with open(model_path / 'distillation_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Student model saved to: {model_path}", flush=True)
        return str(model_path)
    
    def evaluate(self):
        """Final evaluation of the student"""
        if self.student_model is None:
            self.setup_models()
        
        metrics = self.validate_student(self.current_epoch)
        return metrics