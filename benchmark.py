"""
Modulo per il benchmarking comparativo di Teacher e Student
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from datetime import datetime
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from tqdm import tqdm
import pandas as pd

from data_loader import TextGenerationDataset, download_and_prepare_dataset
from metrics import compute_generation_metrics, compute_bleu_scores


class BenchmarkEvaluator:
    """Classe per valutazione e benchmark comparativo"""
        
    def __init__(self, teacher_path: str = None, student_path: str = None,
                device: str = 'cuda',
                output_dir: str = './benchmark_results',
                num_workers: int = 4,
                base_model: str = 't5-base'):
        """
        Args:
            teacher_path: Path al modello teacher (.pt locale o HF repo)
            student_path: Path al modello student (.pt locale o HF repo)
            device: Device da utilizzare
            output_dir: Directory per salvare i risultati
            num_workers: Numero di worker per DataLoader
            base_model: Modello base per caricare checkpoint .pt
        """
        self.teacher_path = teacher_path
        self.student_path = student_path
        self.device = device
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.base_model = base_model
        
        # Crea directory output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inizializza variabili
        self.teacher_model = None
        self.student_model = None
        self.teacher_params = 0
        self.student_params = 0
        self.tokenizer = None
        
        # Carica modelli disponibili
        self._load_models()
        
        print("BenchmarkEvaluator inizializzato", flush=True)

    def _is_local_checkpoint(self, path: str) -> bool:
        """Verifica se il path è un checkpoint locale .pt"""
        return path is not None and (path.endswith('.pt') or path.endswith('.pth')) and os.path.exists(path)
    
    def _detect_model_size_from_checkpoint(self, checkpoint_path: str) -> str:
        """Rileva automaticamente la dimensione del modello dal checkpoint"""
        print("Rilevando dimensioni modello dal checkpoint...", flush=True)
        
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=self.device,
            weights_only=False
        )
        # Estrai state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Rileva hidden_size
        if 'shared.weight' in state_dict:
            hidden_size = state_dict['shared.weight'].shape[1]
        elif 'encoder.embed_tokens.weight' in state_dict:
            hidden_size = state_dict['encoder.embed_tokens.weight'].shape[1]
        else:
            print("Non riesco a rilevare dimensioni, uso t5-base", flush=True)
            return 't5-base'
        
        # Mappa dimensioni → modelli
        size_mapping = {
            512: 't5-small',
            768: 't5-base',
            1024: 't5-large',
            2048: 't5-11b'
        }
        
        detected_model = size_mapping.get(hidden_size, 't5-base')
        print(f"Modello rilevato: {detected_model} (hidden_size: {hidden_size})", flush=True)
        return detected_model

    def _load_model_from_checkpoint(self, checkpoint_path: str, model_name: str):
        """Carica modello da checkpoint locale .pt"""
        print(f"Caricando {model_name} da checkpoint: {checkpoint_path}", flush=True)
        
        # Auto-detect size se base_model è default
        if self.base_model == 't5-base':
            base_model_to_use = self._detect_model_size_from_checkpoint(checkpoint_path)
        else:
            base_model_to_use = self.base_model
            
        print(f"Usando modello base: {base_model_to_use}", flush=True)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_to_use)
        model = model.to(self.device)
        print(f"Modello spostato su: {self.device}", flush=True)
        
        print(f"Caricando checkpoint sul device: {self.device}", flush=True)
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=self.device,
            weights_only=False
        )
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"{model_name} caricato con successo", flush=True)
        except Exception as e:
            print(f"Errore strict loading: {e}", flush=True)
            print("Tentativo non-strict...", flush=True)
            model.load_state_dict(state_dict, strict=False)

        if self.device == 'cuda' and next(model.parameters()).is_cuda:
            print(f"✓ {model_name} confermato su GPU", flush=True)
        elif self.device == 'cuda':
            print(f"⚠ {model_name} NON è su GPU! Tentativo di spostamento...", flush=True)
            model = model.to(self.device)
        
        return model
    
    def _load_model_from_hf(self, model_path: str, model_name: str):
        """Carica modello da Hugging Face"""
        print(f"Caricando {model_name} da HF: {model_path}", flush=True)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        )
        
        return model
    
    def _load_models(self):
        """Carica i modelli teacher e/o student disponibili"""
        print("Caricando modelli per benchmark...", flush=True)
        
        if self.device == 'cuda':
            if torch.cuda.is_available():
                print(f"✓ CUDA disponibile - GPU: {torch.cuda.get_device_name(0)}", flush=True)
                print(f"  Memoria GPU disponibile: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", flush=True)
            else:
                print("✗ CUDA NON disponibile! Il benchmark girerà su CPU", flush=True)
                self.device = 'cpu'

        # Load teacher
        if self.teacher_path:
            if self._is_local_checkpoint(self.teacher_path):
                self.teacher_model = self._load_model_from_checkpoint(self.teacher_path, "teacher")
                if not self.tokenizer:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            else:
                self.teacher_model = self._load_model_from_hf(self.teacher_path, "teacher")
                if not self.tokenizer:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_path)
            
            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()
            self.teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
            print(f"Teacher params: {self.teacher_params:,}", flush=True)
        
        # Load student
        if self.student_path:
            if self._is_local_checkpoint(self.student_path):
                self.student_model = self._load_model_from_checkpoint(self.student_path, "student")
                if not self.tokenizer:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            else:
                self.student_model = self._load_model_from_hf(self.student_path, "student")
                if not self.tokenizer:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.student_path)
            
            self.student_model = self.student_model.to(self.device)
            self.student_model.eval()
            self.student_params = sum(p.numel() for p in self.student_model.parameters())
            print(f"Student params: {self.student_params:,}", flush=True)
        
        # Log compression ratio
        if self.teacher_model and self.student_model:
            print(f"Compression ratio: {self.teacher_params/self.student_params:.2f}x", flush=True)
        
        if not self.tokenizer:
            raise ValueError("Nessun modello caricato. Fornire almeno un path valido.")
        
        # Set pad_token se mancante
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Impostato pad_token = eos_token", flush=True)
    
    def prepare_test_data(self, dataset_name: str = 'cnn_dailymail',
                         max_samples: Optional[int] = 1000):
        """Prepara il dataset di test"""
        print("Preparando dataset di test...", flush=True)
        
        data_path = download_and_prepare_dataset(dataset_name, './data')
        
        test_dataset = TextGenerationDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            split='test',
            max_length=512,
            max_samples=max_samples
        )
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=None,
            label_pad_token_id=-100
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=data_collator,
            pin_memory=True
        )
        
        print(f"Test dataset preparato: {len(test_dataset)} samples", flush=True)
        return len(test_dataset)
    
    def evaluate_model(self, model, model_name: str) -> Dict:
        """
        Valuta un singolo modello
        
        Args:
            model: Modello da valutare
            model_name: Nome del modello
        Returns:
            Dict con metriche
        """
        print(f"Valutando {model_name}...", flush=True)
        
        model.eval()
        
        all_predictions = []
        all_references = []
        total_loss = 0
        generation_times = []
        
        pbar = tqdm(self.test_loader, desc=f"Evaluating {model_name}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Calculate loss
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                
                # Generate predictions con timing
                start_time = time.time()
                generated = model.generate(
                    batch['input_ids'],
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                generation_time = time.time() - start_time
                generation_times.append(generation_time / batch['input_ids'].size(0))
                
                # Decode
                predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                references = self.tokenizer.batch_decode(
                    batch['labels'].masked_fill(batch['labels'] == -100, 0),
                    skip_special_tokens=True
                )
                
                all_predictions.extend(predictions)
                all_references.extend(references)
                
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': total_loss / (batch_idx + 1),
                        'gen_time': np.mean(generation_times[-10:]) if generation_times else 0
                    })
        
        # Calcola metriche
        avg_loss = total_loss / len(self.test_loader)
        avg_generation_time = np.mean(generation_times)
        
        print(f"Calcolando metriche per {model_name}...", flush=True)
        
        # ROUGE e altre metriche base
        metrics = compute_generation_metrics(all_predictions, all_references)
        
        # BLEU scores aggiuntivi
        bleu_scores = compute_bleu_scores(all_predictions, all_references)
        
        # Compila risultati
        results = {
            'loss': avg_loss,
            'generation_time_per_sample': avg_generation_time,
            'rouge': {k: v for k, v in metrics.items() if 'rouge' in k},
            'bleu': bleu_scores,
            'exact_match': metrics.get('exact_match', 0),
            'f1_score': metrics.get('f1_score', 0),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_samples_evaluated': len(all_predictions)
        }
        
        # Sample predictions
        sample_predictions = [
            {
                'input': self.tokenizer.decode(
                    self.test_loader.dataset[i]['input_ids'],
                    skip_special_tokens=True
                ),
                'reference': all_references[i],
                'prediction': all_predictions[i]
            }
            for i in range(min(10, len(all_predictions)))
        ]
        
        results['sample_predictions'] = sample_predictions
        
        print(f"{model_name} evaluation completato", flush=True)
        return results
    
    def compare_inference_speed(self, num_samples: int = 100) -> Dict:
        """Confronta velocità di inferenza"""
        print("Confrontando velocità di inferenza...", flush=True)
        
        test_inputs = []
        for i, batch in enumerate(self.test_loader):
            if i * batch['input_ids'].size(0) >= num_samples:
                break
            test_inputs.append(batch['input_ids'].to(self.device))
        
        def benchmark_model(model, name):
            times = []
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model.generate(test_inputs[0], max_length=128)
            
            # Benchmark
            with torch.no_grad():
                for input_ids in test_inputs:
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    start = time.time()
                    
                    _ = model.generate(
                        input_ids,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    end = time.time()
                    
                    times.append((end - start) / input_ids.size(0))
            
            return {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'median_time': np.median(times),
                'samples_per_second': 1.0 / np.mean(times)
            }
        
        teacher_speed = benchmark_model(self.teacher_model, "Teacher")
        student_speed = benchmark_model(self.student_model, "Student")
        
        speedup = teacher_speed['mean_time'] / student_speed['mean_time']
        
        return {
            'teacher': teacher_speed,
            'student': student_speed,
            'speedup': speedup,
            'relative_throughput': student_speed['samples_per_second'] / teacher_speed['samples_per_second']
        }
    
    def calculate_memory_footprint(self) -> Dict:
        """Calcola impronta di memoria modelli"""
        print("Calcolando impronta memoria...", flush=True)
        
        def get_model_size(model):
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            size_mb = (param_size + buffer_size) / 1024 / 1024
            
            return {
                'param_size_mb': param_size / 1024 / 1024,
                'buffer_size_mb': buffer_size / 1024 / 1024,
                'total_size_mb': size_mb
            }
        
        teacher_memory = get_model_size(self.teacher_model)
        student_memory = get_model_size(self.student_model)
        
        return {
            'teacher': teacher_memory,
            'student': student_memory,
            'compression_ratio': teacher_memory['total_size_mb'] / student_memory['total_size_mb']
        }
    
    def analyze_performance_tradeoffs(self, teacher_results: Dict, 
                                     student_results: Dict) -> Dict:
        """Analizza tradeoff performance/efficienza"""
        
        # Performance degradation
        rouge_degradation = {
            key: (teacher_results['rouge'][key] - student_results['rouge'][key]) / teacher_results['rouge'][key] * 100
            for key in teacher_results['rouge'].keys()
        }
        
        bleu_degradation = {
            key: (teacher_results['bleu'][key] - student_results['bleu'][key]) / teacher_results['bleu'][key] * 100
            for key in teacher_results['bleu'].keys()
        }
        
        # Efficiency gains
        param_reduction = (1 - self.student_params / self.teacher_params) * 100
        speedup = teacher_results['generation_time_per_sample'] / student_results['generation_time_per_sample']
        
        # Efficiency score
        avg_rouge_retention = 100 - np.mean(list(rouge_degradation.values()))
        efficiency_score = (avg_rouge_retention * 0.5 + param_reduction * 0.3 + speedup * 20)
        
        return {
            'performance_degradation': {
                'rouge': rouge_degradation,
                'bleu': bleu_degradation
            },
            'efficiency_gains': {
                'parameter_reduction_percent': param_reduction,
                'speedup_factor': speedup,
                'memory_reduction_factor': self.teacher_params / self.student_params
            },
            'overall_efficiency_score': efficiency_score,
            'performance_retention_percent': avg_rouge_retention
        }
    
    def run_full_evaluation(self) -> Dict:
        """Esegue valutazione completa entrambi i modelli"""
        print("=" * 50, flush=True)
        print("INIZIANDO BENCHMARK COMPLETO", flush=True)
        print("=" * 50, flush=True)
        
        self.prepare_test_data()
        
        print("\n--- Evaluating Teacher Model ---", flush=True)
        teacher_results = self.evaluate_model(self.teacher_model, "Teacher")
        
        print("\n--- Evaluating Student Model ---", flush=True)
        student_results = self.evaluate_model(self.student_model, "Student")
        
        print("\n--- Comparing Inference Speed ---", flush=True)
        speed_comparison = self.compare_inference_speed()
        
        print("\n--- Analyzing Memory Footprint ---", flush=True)
        memory_analysis = self.calculate_memory_footprint()
        
        print("\n--- Analyzing Performance Tradeoffs ---", flush=True)
        tradeoff_analysis = self.analyze_performance_tradeoffs(
            teacher_results, student_results
        )
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'models': {
                'teacher_path': self.teacher_path,
                'student_path': self.student_path
            },
            'teacher_results': teacher_results,
            'student_results': student_results,
            'speed_comparison': speed_comparison,
            'memory_analysis': memory_analysis,
            'tradeoff_analysis': tradeoff_analysis,
            'compression_ratio': {
                'parameters': self.teacher_params / self.student_params,
                'memory': memory_analysis['compression_ratio'],
                'inference_speedup': speed_comparison['speedup']
            }
        }
        
        return results
    
    def run_single_model_evaluation(self, model_type: str = 'teacher') -> Dict:
        """
        Esegue valutazione singolo modello
        
        Args:
            model_type: 'teacher' o 'student'
        Returns:
            Dict con risultati
        """
        print("=" * 50, flush=True)
        print(f"VALUTAZIONE SINGOLO MODELLO: {model_type.upper()}", flush=True)
        print("=" * 50, flush=True)
        
        self.prepare_test_data()
        
        if model_type == 'teacher':
            if not self.teacher_model:
                raise ValueError("Teacher model non caricato")
            model = self.teacher_model
            model_path = self.teacher_path
            params = self.teacher_params
        else:
            if not self.student_model:
                raise ValueError("Student model non caricato")
            model = self.student_model
            model_path = self.student_path
            params = self.student_params
        
        print(f"\n--- Evaluating {model_type.capitalize()} Model ---", flush=True)
        model_results = self.evaluate_model(model, model_type.capitalize())
        
        # Test velocità inferenza
        print("\n--- Testing Inference Speed ---", flush=True)
        test_inputs = []
        for i, batch in enumerate(self.test_loader):
            if i >= 5:
                break
            test_inputs.append(batch['input_ids'].to(self.device))
        
        speeds = []
        with torch.no_grad():
            for input_ids in test_inputs:
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                _ = model.generate(input_ids, max_length=128, num_beams=4)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                speeds.append((time.time() - start) / input_ids.size(0))
        
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Memoria
        param_size = params * 4 / 1024 / 1024
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'model_path': model_path,
            'model_results': model_results,
            'performance_summary': {
                'rouge_l': model_results['rouge'].get('rouge_l', 0),
                'bleu_4': model_results['bleu'].get('bleu_4', 0),
                'loss': model_results['loss'],
                'avg_generation_time': avg_speed,
                'samples_per_second': 1.0 / avg_speed if avg_speed > 0 else 0
            },
            'model_info': {
                'parameters': params,
                'parameters_millions': params / 1e6,
                'model_size_mb': param_size
            }
        }
        
        return results
    
    def generate_comparison_table(self, results: Dict) -> pd.DataFrame:
        """Genera tabella di confronto"""
        
        comparison_data = {
            'Metric': [],
            'Teacher': [],
            'Student': [],
            'Difference (%)': []
        }
        
        metrics_to_compare = [
            ('Loss', 'loss'),
            ('ROUGE-1', lambda r: r['rouge']['rouge_1']),
            ('ROUGE-2', lambda r: r['rouge']['rouge_2']),
            ('ROUGE-L', lambda r: r['rouge']['rouge_l']),
            ('BLEU-1', lambda r: r['bleu']['bleu_1']),
            ('BLEU-2', lambda r: r['bleu']['bleu_2']),
            ('BLEU-3', lambda r: r['bleu']['bleu_3']),
            ('BLEU-4', lambda r: r['bleu']['bleu_4']),
            ('Generation Time (s)', 'generation_time_per_sample'),
            ('Parameters (M)', lambda r: r['num_parameters'] / 1e6),
        ]
        
        for metric_name, metric_key in metrics_to_compare:
            comparison_data['Metric'].append(metric_name)
            
            if callable(metric_key):
                teacher_val = metric_key(results['teacher_results'])
                student_val = metric_key(results['student_results'])
            else:
                teacher_val = results['teacher_results'][metric_key]
                student_val = results['student_results'][metric_key]
            
            comparison_data['Teacher'].append(f"{teacher_val:.4f}")
            comparison_data['Student'].append(f"{student_val:.4f}")
            
            if teacher_val != 0:
                diff = ((student_val - teacher_val) / teacher_val) * 100
                comparison_data['Difference (%)'].append(f"{diff:+.2f}%")
            else:
                comparison_data['Difference (%)'].append("N/A")
        
        return pd.DataFrame(comparison_data)
    
    def save_report(self, results: Dict) -> str:
        """Salva report completo"""
        
        # JSON results
        json_path = self.output_dir / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            results_copy = results.copy()
            if 'teacher_results' in results_copy and 'sample_predictions' in results_copy['teacher_results']:
                del results_copy['teacher_results']['sample_predictions']
            if 'student_results' in results_copy and 'sample_predictions' in results_copy['student_results']:
                del results_copy['student_results']['sample_predictions']
            json.dump(results_copy, f, indent=2)
        
        # Comparison table
        if 'teacher_results' in results and 'student_results' in results:
            comparison_df = self.generate_comparison_table(results)
            csv_path = self.output_dir / 'comparison_table.csv'
            comparison_df.to_csv(csv_path, index=False)
        
        # Text report
        report_path = self.output_dir / 'benchmark_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(" KNOWLEDGE DISTILLATION BENCHMARK REPORT \n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {results['timestamp']}\n")
            
            if 'models' in results:
                f.write(f"Teacher Model: {results['models']['teacher_path']}\n")
                f.write(f"Student Model: {results['models']['student_path']}\n\n")
            elif 'model_path' in results:
                f.write(f"Model: {results['model_path']}\n\n")
            
            if 'teacher_results' in results and 'student_results' in results:
                f.write("=" * 80 + "\n")
                f.write(" MODEL COMPARISON \n")
                f.write("=" * 80 + "\n\n")
                comparison_df = self.generate_comparison_table(results)
                f.write(comparison_df.to_string(index=False))
                f.write("\n\n")
                
                f.write("=" * 80 + "\n")
                f.write(" EFFICIENCY ANALYSIS \n")
                f.write("=" * 80 + "\n\n")
                
                tradeoffs = results['tradeoff_analysis']
                f.write(f"Parameter Reduction: {tradeoffs['efficiency_gains']['parameter_reduction_percent']:.2f}%\n")
                f.write(f"Inference Speedup: {tradeoffs['efficiency_gains']['speedup_factor']:.2f}x\n")
                f.write(f"Memory Reduction: {tradeoffs['efficiency_gains']['memory_reduction_factor']:.2f}x\n")
                f.write(f"Performance Retention: {tradeoffs['performance_retention_percent']:.2f}%\n")
                f.write(f"Overall Efficiency Score: {tradeoffs['overall_efficiency_score']:.2f}\n\n")
        
        print(f"Report salvato in: {report_path}", flush=True)
        return str(report_path)

    def print_summary(self, results: Dict):
        """Stampa sommario risultati"""
        
        print("\n" + "=" * 80, flush=True)
        
        if 'model_type' in results:
            # Valutazione singola
            print(f" BENCHMARK SUMMARY - {results['model_type'].upper()} MODEL ", flush=True)
            print("=" * 80, flush=True)
            
            model_info = results['model_info']
            print(f"\nModel Info:", flush=True)
            print(f"  Path: {results['model_path']}", flush=True)
            print(f"  Parameters: {model_info['parameters']:,}", flush=True)
            print(f"  Size: {model_info['model_size_mb']:.2f} MB", flush=True)
            
            perf = results['performance_summary']
            print(f"\nPerformance Metrics:", flush=True)
            print(f"  ROUGE-L: {perf['rouge_l']:.4f}", flush=True)
            print(f"  BLEU-4: {perf['bleu_4']:.4f}", flush=True)
            print(f"  Loss: {perf['loss']:.4f}", flush=True)
            print(f"  Speed: {perf['samples_per_second']:.2f} samples/sec", flush=True)
            
        else:
            # Valutazione comparativa
            print(" BENCHMARK SUMMARY - COMPARATIVE ", flush=True)
            print("=" * 80, flush=True)
            
            print(f"\nModel Sizes:", flush=True)
            print(f"  Teacher: {self.teacher_params:,} parameters", flush=True)
            print(f"  Student: {self.student_params:,} parameters", flush=True)
            print(f"  Compression: {self.teacher_params/self.student_params:.2f}x", flush=True)
            
            print(f"\nPerformance Metrics:", flush=True)
            print(f"  Teacher ROUGE-L: {results['teacher_results']['rouge']['rouge_l']:.4f}", flush=True)
            print(f"  Student ROUGE-L: {results['student_results']['rouge']['rouge_l']:.4f}", flush=True)
            print(f"  Teacher BLEU-4: {results['teacher_results']['bleu']['bleu_4']:.4f}", flush=True)
            print(f"  Student BLEU-4: {results['student_results']['bleu']['bleu_4']:.4f}", flush=True)
            
            print(f"\nEfficiency:", flush=True)
            print(f"  Speedup: {results['speed_comparison']['speedup']:.2f}x", flush=True)
            print(f"  Memory Reduction: {results['memory_analysis']['compression_ratio']:.2f}x", flush=True)
            print(f"  Performance Retention: {results['tradeoff_analysis']['performance_retention_percent']:.2f}%", flush=True)
        
        print("=" * 80 + "\n", flush=True)