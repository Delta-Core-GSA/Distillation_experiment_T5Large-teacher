"""
Modulo per il training del teacher e la distillazione dello student
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
    """Classe base per il training"""
    
    def __init__(self, config, device, output_dir, checkpoint_dir, 
                 num_workers=4, mixed_precision=False):
        
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        
        # Crea directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.best_metric = float('-inf')
        self.current_epoch = 0
    
    def log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        """Log metrics su console"""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"{prefix} | Step: {step} | {metrics_str}", flush=True)
    
    def prepare_data(self):
        """Prepara i dataset"""
        print("Preparando i dataset...", flush=True)
        
        # Download e prepara il dataset
        data_path = download_and_prepare_dataset(
            self.config['dataset']['name'],
            self.config['dataset'].get('cache_dir', './data')
        )
        
        # Inizializza tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        # Crea dataset
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
        
        # Crea data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=None,
            label_pad_token_id=-100
        )
        
        # Crea DataLoaders
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
        
        print(f"Dataset caricati - Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
        return len(train_dataset), len(val_dataset)


class TeacherTrainer(BaseTrainer):
    """Trainer per il modello teacher"""
    
    def __init__(self, resume_from_checkpoint=None, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.resume_from_checkpoint = resume_from_checkpoint
        
    def setup_model(self):
        """Inizializza il modello teacher"""
        print(f"Caricando il modello teacher: {self.config['model']['name']}", flush=True)
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['model']['name'],
            torch_dtype=torch.float16 if self.mixed_precision else torch.float32
        )
        
        self.model = self.model.to(self.device)
        
        # Log parametri
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}", flush=True)
        print(f"Trainable parameters: {trainable_params:,}", flush=True)
        
    def setup_optimization(self, num_training_steps):
        """Setup optimizer e scheduler"""
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
        
        print(f"Optimizer configurato - LR: {self.config['training']['learning_rate']}", flush=True)
        print(f"Scheduler configurato - Warmup steps: {num_warmup_steps}", flush=True)
    
    def train_epoch(self, epoch):
        """Training per una singola epoca"""
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
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training'].get('max_grad_norm', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training'].get('max_grad_norm', 1.0)
                )
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
                global_step = epoch * len(self.train_loader) + batch_idx
                self.log_metrics(
                    {'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]},
                    global_step,
                    prefix='train'
                )
        
        return losses.avg
    
    def validate(self, epoch):
        """Validazione del modello"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", disable=True):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                # Generate predictions per metrics (limita a 100 per velocità)
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
        
        # Calcola metriche
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_generation_metrics(all_predictions, all_references)
        metrics['loss'] = avg_loss
        
        self.log_metrics(metrics, epoch, prefix='val')
        return metrics
    
    def load_checkpoint_for_resume(self, checkpoint_path):
        """Carica checkpoint per riprendere il training"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
        
        print(f"Caricando checkpoint da: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Carica states
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state caricato", flush=True)
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state caricato", flush=True)
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state caricato", flush=True)
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Scaler state caricato", flush=True)
        
        # Recupera epoca e metriche
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'metrics' in checkpoint and 'rouge_l' in checkpoint['metrics']:
            self.best_metric = checkpoint['metrics']['rouge_l']
            print(f"Best metric (ROUGE-L): {self.best_metric:.4f}", flush=True)
        
        print(f"Riprendendo dall'epoca {start_epoch}", flush=True)
        return start_epoch
    
    def train(self):
        """Main training loop"""
        # Setup
        train_size, val_size = self.prepare_data()
        self.setup_model()
        
        num_epochs = self.config['training']['num_epochs']
        num_training_steps = num_epochs * len(self.train_loader)
        self.setup_optimization(num_training_steps)
        
        # Carica checkpoint se specificato
        start_epoch = 1
        if self.resume_from_checkpoint:
            start_epoch = self.load_checkpoint_for_resume(self.resume_from_checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training'].get('early_stopping_patience', 3),
            verbose=True
        )
        
        # Training loop
        print("=" * 50, flush=True)
        print("Iniziando il training del Teacher", flush=True)
        if self.resume_from_checkpoint:
            print(f"Riprendendo dall'epoca {start_epoch}", flush=True)
        print(f"Epochs: {num_epochs}", flush=True)
        print(f"Batch size: {self.config['training']['batch_size']}", flush=True)
        print(f"Steps per epoch: {len(self.train_loader)}", flush=True)
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
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': val_metrics,
                'config': self.config,
                'best_metric': self.best_metric
            }
            
            if self.mixed_precision and self.scaler is not None:
                checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
            
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            save_checkpoint(
                checkpoint_data,
                checkpoint_path,
                is_best,
                self.checkpoint_dir / 'best_model.pt'
            )
            
            # Early stopping
            early_stopping(val_metrics.get('rouge_l', 0))
            if early_stopping.early_stop:
                print("Early stopping triggered!", flush=True)
                break
        
        print("Training completato!", flush=True)
        return self.best_metric
    
    def save_final_model(self):
        """Salva il modello finale"""
        model_path = self.output_dir / 'final_model'
        model_path.mkdir(exist_ok=True)
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        with open(model_path / 'training_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Modello salvato in: {model_path}", flush=True)
        return str(model_path)
    
    def evaluate(self):
        """Valutazione finale del modello"""
        if self.model is None:
            self.setup_model()
        
        metrics = self.validate(self.current_epoch)
        return metrics


class StudentDistiller(BaseTrainer):
    """Classe generica per la distillazione dello student"""
    
    def __init__(self, student_config, teacher_path, distiller_type='vanilla',resume_from_checkpoint =None ,**kwargs):
        super().__init__(config=student_config, **kwargs)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.teacher_path = teacher_path
        self.distiller_type = distiller_type
        self.teacher_model = None
        self.student_model = None
        self.distiller = None
    
    def setup_models(self):
        """Setup teacher e student models"""
        print("Setting up models for distillation...", flush=True)
        
        # 1. Carica Student Model
        student_model_name = self.config['model']['name']
        print(f"Loading student model: {student_model_name}", flush=True)
        
        self.student_model = AutoModelForSeq2SeqLM.from_pretrained(
            student_model_name,
            cache_dir=self.config['model'].get('cache_dir', None)
        )
        
        # 2. Carica Teacher Model da checkpoint
        teacher_checkpoint_path = self.config['distillation']['teacher_model_path']
        print(f"Loading teacher from: {teacher_checkpoint_path}", flush=True)
        
        if not os.path.exists(teacher_checkpoint_path):
            raise FileNotFoundError(f"Teacher checkpoint non trovato: {teacher_checkpoint_path}")
        
        checkpoint = torch.load(teacher_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Determina nome modello teacher
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            teacher_model_name = checkpoint['config']['model']['name']
        elif 'teacher_model_name' in self.config['distillation']:
            teacher_model_name = self.config['distillation']['teacher_model_name']
        else:
            teacher_model_name = student_model_name.replace('-small', '-base')
            print(f"Model name non trovato nel checkpoint, uso: {teacher_model_name}", flush=True)
        
        print(f"Loading teacher architecture: {teacher_model_name}", flush=True)
        self.teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
            teacher_model_name,
            cache_dir=self.config['model'].get('cache_dir', None)
        )
        
        # Carica weights dal checkpoint
        if 'model_state_dict' in checkpoint:
            self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            self.teacher_model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError("Checkpoint non contiene 'model_state_dict' o 'model'")
        
        # Teacher in eval mode (frozen)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Move to device
        self.student_model = self.student_model.to(self.device)
        self.teacher_model = self.teacher_model.to(self.device)
        
        # Log info
        student_params = sum(p.numel() for p in self.student_model.parameters())
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        print(f"Student: {student_params:,} params", flush=True)
        print(f"Teacher: {teacher_params:,} params", flush=True)
        print(f"Compression: {teacher_params / student_params:.2f}x", flush=True)
        
        # 3. Setup distiller dalla cartella distiller_zoo
        self._setup_distiller()
        
        print(f"Distiller setup complete: {self.distiller_type}", flush=True)
    
    def _setup_distiller(self):
        """Inizializza il distiller dalla cartella distiller_zoo"""
        distill_config = self.config['distillation']
        
        if self.distiller_type == 'vanilla':
            from distiller_zoo.vanillakd import DistillKL
            self.distiller = DistillKL(T=distill_config.get('temperature', 3.0))
            self.alpha = distill_config.get('alpha', 0.7)
            
        elif self.distiller_type == 'feature':
            from distiller_zoo.featurekd import FeatureDistiller
            
            student_dim = self.student_model.config.d_model
            teacher_dim = self.teacher_model.config.d_model
            
            self.distiller = FeatureDistiller(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                T=distill_config.get('temperature', 3.0),
                beta=distill_config.get('beta', 0.1)
            ).to(self.device)  # AGGIUNGI .tFo(self.device)
            self.alpha = distill_config.get('alpha', 0.7)
            
        elif self.distiller_type == 'attention':
            from distiller_zoo.attentionkd import AttentionDistiller
            self.distiller = AttentionDistiller(
                T=distill_config.get('temperature', 3.0),
                gamma=distill_config.get('gamma', 0.1)
            ).to(self.device)  # AGGIUNGI .to(self.device)
            self.alpha = distill_config.get('alpha', 0.7)


        elif self.distiller_type == 'norm':
            from distiller_zoo.normkd import NormDistiller
            
            student_dim = self.student_model.config.d_model
            teacher_dim = self.teacher_model.config.d_model
            
            self.distiller = NormDistiller(
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                T=distill_config.get('temperature', 3.0),
                gamma=distill_config.get('gamma', 0.1)
            ).to(self.device)
            self.alpha = distill_config.get('alpha', 0.7)
                    
        else:
            raise ValueError(f"Distiller type non supportato: {self.distiller_type}")
        
    def distill_epoch(self, epoch):
        """Distillazione per una singola epoca"""
        self.student_model.train()
        self.teacher_model.eval()
        
        losses = AverageMeter('Loss', ':.4f')
        kd_losses = AverageMeter('KD_Loss', ':.4f')
        ce_losses = AverageMeter('CE_Loss', ':.4f')
        
        total_batches = len(self.train_loader)
        print(f"Distilling Epoch {epoch} - Total batches: {total_batches}", flush=True)


            
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            total=total_batches,
            ncols=100,  # Larghezza fissa
            ascii=True,  # Usa caratteri ASCII semplici
            file=sys.stdout,
            disable=False,
            mininterval=10,
            miniters=50
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass student e teacher
            student_outputs = self.student_model(
                **batch, 
                output_hidden_states=True,
                output_attentions=True
            )
            ce_loss = student_outputs.loss
            
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    **batch,
                    output_hidden_states=True,
                    output_attentions=True
                )
            
            # PRE-PROCESSING: Maschera le posizioni di padding
            labels = batch['labels']
            mask = (labels != -100)
            
            # Flattened logits solo per posizioni valide
            student_logits_flat = student_outputs.logits[mask]
            teacher_logits_flat = teacher_outputs.logits[mask]
            
            # Verifica che ci siano posizioni valide
            if student_logits_flat.size(0) == 0:
                continue
            
            # Calcola KD loss in base al tipo di distiller
            if self.distiller_type == 'vanilla':
                kd_loss = self.distiller(
                    student_logits_flat,
                    teacher_logits_flat
                )
                
            elif self.distiller_type == 'feature':
                # Per feature distillation, maschera anche gli hidden states
                student_hidden = student_outputs.encoder_hidden_states[-1]
                teacher_hidden = teacher_outputs.encoder_hidden_states[-1]
                
                # Maschera gli hidden states dell'encoder
                encoder_mask = batch['attention_mask'].bool()
                student_hidden_flat = student_hidden[encoder_mask]
                teacher_hidden_flat = teacher_hidden[encoder_mask]
                
                kd_loss = self.distiller(
                    student_logits_flat,
                    teacher_logits_flat,
                    student_hidden_flat,
                    teacher_hidden_flat
                )
                
            elif self.distiller_type == 'attention':
                # Per attention distillation
                student_attn = student_outputs.encoder_attentions[-1]
                teacher_attn = teacher_outputs.encoder_attentions[-1]
                
                kd_loss = self.distiller(
                    student_logits_flat,
                    teacher_logits_flat,
                    student_attn,
                    teacher_attn
                )

            elif self.distiller_type == 'norm':
                # Estrai hidden states dell'encoder
                student_hidden = student_outputs.encoder_hidden_states[-1]
                teacher_hidden = teacher_outputs.encoder_hidden_states[-1]
                
                # Maschera encoder
                encoder_mask = batch['attention_mask'].bool()
                student_hidden_flat = student_hidden[encoder_mask]
                teacher_hidden_flat = teacher_hidden[encoder_mask]
                
                kd_loss = self.distiller(
                    student_logits_flat,
                    teacher_logits_flat,
                    student_hidden_flat,
                    teacher_hidden_flat
                )
            
            # Loss totale
            loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.config['training'].get('max_grad_norm', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.config['training'].get('max_grad_norm', 1.0)
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            losses.update(loss.item(), batch['input_ids'].size(0))
            kd_losses.update(kd_loss.item(), batch['input_ids'].size(0))
            ce_losses.update(ce_loss.item(), batch['input_ids'].size(0))
            

            progress_bar.set_postfix({
                        'Loss': f'{losses.avg:.4f}',
                        'KD': f'{kd_losses.avg:.4f}',
                        'CE': f'{ce_losses.avg:.4f}',
                        'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
                    })

            
        
        print(f"Epoch {epoch} completed - Avg Loss: {losses.avg:.4f}, KD: {kd_losses.avg:.4f}, CE: {ce_losses.avg:.4f}", flush=True)
        return losses.avg, kd_losses.avg, ce_losses.avg
    
    def validate_student(self, epoch):
        """Validazione dello student"""
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
                
                # Generate predictions (limita a 100)
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
        """Carica checkpoint per riprendere la distillazione"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
        
        print(f"Caricando checkpoint distillazione da: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Carica states dello student
        if 'model_state_dict' in checkpoint:
            self.student_model.load_state_dict(checkpoint['model_state_dict'])
            print("Student model state caricato", flush=True)
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state caricato", flush=True)
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state caricato", flush=True)
        
        if self.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Scaler state caricato", flush=True)
        
        # Recupera epoca e metriche
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'metrics' in checkpoint and 'rouge_l' in checkpoint['metrics']:
            self.best_metric = checkpoint['metrics']['rouge_l']
            print(f"Best metric (ROUGE-L): {self.best_metric:.4f}", flush=True)
        
        # Log distillation losses se presenti
        if 'distillation_losses' in checkpoint:
            losses = checkpoint['distillation_losses']
            print(f"Distillation losses dall'ultimo checkpoint:", flush=True)
            print(f"  Total: {losses.get('total', 'N/A'):.4f}", flush=True)
            print(f"  KD: {losses.get('kd', 'N/A'):.4f}", flush=True)
            print(f"  CE: {losses.get('ce', 'N/A'):.4f}", flush=True)
        
        print(f"Riprendendo distillazione dall'epoca {start_epoch}", flush=True)
        return start_epoch
    
    def distill(self):
        """Main distillation loop"""
        # Setup
        train_size, val_size = self.prepare_data()
        self.setup_models()
        
        # Setup optimization per student
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
        
        # Carica checkpoint se specificato
        start_epoch = 1
        if self.resume_from_checkpoint:
            start_epoch = self.load_checkpoint_for_resume(self.resume_from_checkpoint)

        # Distillation loop
        print("=" * 50, flush=True)
        print("Iniziando la distillazione", flush=True)
        if self.resume_from_checkpoint:
            print(f"Riprendendo dall'epoca {start_epoch}", flush=True)
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
        
        print("Distillazione completata!", flush=True)
        return self.best_metric
    
    def save_final_model(self):
        """Salva lo student model finale"""
        model_path = self.output_dir / 'final_student_model'
        model_path.mkdir(exist_ok=True)
        
        self.student_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        with open(model_path / 'distillation_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Student model salvato in: {model_path}", flush=True)
        return str(model_path)
    
    def evaluate(self):
        """Valutazione finale dello student"""
        if self.student_model is None:
            self.setup_models()
        
        metrics = self.validate_student(self.current_epoch)
        return metrics