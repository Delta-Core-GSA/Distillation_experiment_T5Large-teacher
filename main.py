#!/usr/bin/env python3
"""
Main entry point per il progetto di distillazione T5
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
import torch
from datetime import datetime

from training import TeacherTrainer, StudentDistiller
from benchmark import BenchmarkEvaluator
from utils import setup_logging, load_config, set_seed, print_system_info


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='T5 Knowledge Distillation Project')
    
    # Modalità operative
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train_teacher', 'distill_student', 'benchmark', 'full_pipeline','benchmark_code_carbon'],
                       help='Modalità di esecuzione')

    # Percorsi modelli (per benchmark)
    parser.add_argument('--teacher_path', type=str, default=None,
                    help='Path al modello teacher')
    parser.add_argument('--student_path', type=str, default=None,
                    help='Path al modello student')
    


    # Percorsi modelli per benchmark codecarbon
    parser.add_argument('--teacher_path_benchmark', type=str, default=None,
                    help='Path al modello teacher finetuned')
    parser.add_argument('--student_distilled_path', type=str, default=None,
                    help='Path al modello student distillato')
    parser.add_argument('--student_finetuned_path', type=str, default=None,
                    help='Path al modello student finetuned (baseline)')
    

    # Configurazioni
    parser.add_argument('--teacher_config', type=str, default='configs/teacher_config.json',
                       help='Path al file di configurazione del teacher')
    parser.add_argument('--student_config', type=str, default='configs/student_config.json',
                       help='Path al file di configurazione dello student')
    
    # Opzioni generali
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed per riproducibilità')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device da utilizzare (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Numero di worker per il data loading')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Usa mixed precision training (FP16)')
    
    # Percorsi
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path al checkpoint da cui riprendere il training')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Directory principale per output')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory per i log')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory per i checkpoint')
    
   
    return parser.parse_args()





def setup_directories(args):
    """Crea le directory necessarie"""
    dirs = [args.output_dir, args.log_dir, args.checkpoint_dir]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory creata/verificata: {dir_path}")



def train_teacher(args, teacher_config):
    """Addestra il modello teacher"""
    logging.info("=" * 50)
    logging.info("INIZIANDO TRAINING DEL TEACHER")
    logging.info("=" * 50)
    
    if args.resume_from_checkpoint:
        logging.info(f"Riprendendo training da checkpoint: {args.resume_from_checkpoint}")
    
    trainer = TeacherTrainer(
        config=teacher_config,
        device=args.device,
        output_dir=os.path.join(args.output_dir, 'teacher'),
        checkpoint_dir=os.path.join(args.checkpoint_dir, 'teacher'),
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Avvia il training
    trainer.train()
    
    # Salva il modello finale
    final_model_path = trainer.save_final_model()
    logging.info(f"Teacher model salvato in: {final_model_path}")
    
    # Valuta il modello
    metrics = trainer.evaluate()
    logging.info(f"Metriche finali del Teacher: {json.dumps(metrics, indent=2)}")
    
    return final_model_path, metrics


def distill_student(args, student_config, teacher_path=None):
    """Distilla la conoscenza dal teacher allo student"""
    logging.info("=" * 50)
    logging.info("INIZIANDO DISTILLAZIONE DELLO STUDENT")
    logging.info("=" * 50)
    
    # Se non specificato, cerca il teacher più recente

    #quick fix (hard coded)
    teacher_path = student_config.get('distillation', {}).get('teacher_model_path', None)
    
    if teacher_path is None:
        teacher_dir = os.path.join(args.output_dir, 'teacher')
        if not os.path.exists(teacher_dir):
            raise ValueError(f"Teacher model non trovato in {teacher_dir}")
        teacher_path = teacher_dir


    distiller_type = student_config.get('distillation', {}).get('distillation_type', 'vanilla')
    print(f"Distillation type from config: {distiller_type}", flush=True)
    
    distiller = StudentDistiller(
        student_config=student_config,
        teacher_path=teacher_path,
        device=args.device,
        distiller_type=distiller_type,
        output_dir=os.path.join(args.output_dir, 'student'),
        checkpoint_dir=os.path.join(args.checkpoint_dir, 'student'),
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Avvia la distillazione
    distiller.distill()
    
    # Salva il modello finale
    final_model_path = distiller.save_final_model()
    logging.info(f"Student model salvato in: {final_model_path}")
    
    # Valuta il modello
    metrics = distiller.evaluate()
    logging.info(f"Metriche finali dello Student: {json.dumps(metrics, indent=2)}")
    
    return final_model_path, metrics



def run_benchmark(args, teacher_path=None, student_path=None):
    """Esegue benchmark su modelli specificati"""
    logging.info("=" * 50)
    logging.info("ESEGUENDO BENCHMARK")
    logging.info("=" * 50)
    
    # Validazione: almeno un modello deve essere specificato
    if not teacher_path and not student_path:
        raise ValueError("Specificare almeno un path: teacher_path o student_path")
    
    # Determina tipo di benchmark in base ai path forniti
    if teacher_path and student_path:
        # Benchmark comparativo
        logging.info("Benchmark comparativo: Teacher vs Student")
        evaluator = BenchmarkEvaluator(
            teacher_path=teacher_path,
            student_path=student_path,
            device=args.device,
            output_dir=os.path.join(args.output_dir, 'benchmark'),
            num_workers=args.num_workers
        )
        results = evaluator.run_full_evaluation()
        
    elif teacher_path:
        # Solo teacher
        logging.info("Benchmark singolo: Teacher")
        evaluator = BenchmarkEvaluator(
            teacher_path=teacher_path,
            student_path=None,
            device=args.device,
            output_dir=os.path.join(args.output_dir, 'benchmark'),
            num_workers=args.num_workers
        )
        results = evaluator.run_single_model_evaluation('teacher')
        
    else:
        # Solo student
        logging.info("Benchmark singolo: Student")
        evaluator = BenchmarkEvaluator(
            teacher_path=None,
            student_path=student_path,
            device=args.device,
            output_dir=os.path.join(args.output_dir, 'benchmark'),
            num_workers=args.num_workers
        )
        results = evaluator.run_single_model_evaluation('student')
    
    # Salva e visualizza risultati
    report_path = evaluator.save_report(results)
    logging.info(f"Report salvato in: {report_path}")
    evaluator.print_summary(results)
    
    return results

def run_benchmark_code_carbon(args):
    """Esegue benchmark su tutti e 3 i modelli"""
    logging.info("=" * 50)
    logging.info("ESEGUENDO BENCHMARK TRIPLO")
    logging.info("=" * 50)
    
    if not args.teacher_path or not args.student_distilled_path or not args.student_finetuned_path:
        raise ValueError("Per benchmark triplo servono tutti e 3 i path")
    
    evaluator = BenchmarkEvaluator(
        teacher_path=args.teacher_path,
        student_distilled_path=args.student_distilled_path,
        student_finetuned_path=args.student_finetuned_path,
        device=args.device,
        output_dir=os.path.join(args.output_dir, 'benchmark_code_carbon'),
        num_workers=args.num_workers,
        track_energy=True
    )
    
    results = evaluator.run_triple_evaluation()
    
    report_path = evaluator.save_report(results)
    logging.info(f"Report salvato in: {report_path}")
    evaluator.print_summary(results)
    
    return results


def run_full_pipeline(args, teacher_config, student_config):
    """Esegue l'intero pipeline: training, distillazione, benchmark"""
    logging.info("=" * 50)
    logging.info("ESEGUENDO PIPELINE COMPLETO")
    logging.info("=" * 50)
    
    start_time = datetime.now()
    
    # Step 1: Training Teacher
    logging.info("\n[STEP 1/3] Training del Teacher Model")
    teacher_path, teacher_metrics = train_teacher(args, teacher_config)
    
    # Step 2: Distillazione Student
    logging.info("\n[STEP 2/3] Distillazione dello Student Model")
    student_path, student_metrics = distill_student(args, student_config, teacher_path)
    
    # Step 3: Benchmark Comparativo
    logging.info("\n[STEP 3/3] Benchmark Comparativo")
    benchmark_results = run_benchmark(args, teacher_path, student_path)
    
    # Report finale
    duration = datetime.now() - start_time
    logging.info("\n" + "=" * 50)
    logging.info("PIPELINE COMPLETATO CON SUCCESSO")
    logging.info("=" * 50)
    logging.info(f"Tempo totale: {duration}")
    
    # Stampa metriche principali
    teacher_rouge = teacher_metrics.get('rouge_l', 0)
    student_rouge = student_metrics.get('rouge_l', 0)
    compression = benchmark_results.get('compression_ratio', {}).get('parameters', 0)
    speedup = benchmark_results.get('speedup', 0)
    
    logging.info(f"Teacher ROUGE-L: {teacher_rouge:.4f}")
    logging.info(f"Student ROUGE-L: {student_rouge:.4f}")
    logging.info(f"Compression ratio: {compression:.2f}x")
    logging.info(f"Speedup: {speedup:.2f}x")
    
    # Salva sommario
    summary = {
        'timestamp': datetime.now().isoformat(),
        'duration': str(duration),
        'teacher_metrics': teacher_metrics,
        'student_metrics': student_metrics,
        'benchmark_results': benchmark_results,
        'config': {
            'teacher': teacher_config,
            'student': student_config
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Sommario salvato in: {summary_path}")
    return summary


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup iniziale
    set_seed(args.seed)
    setup_directories(args)
    setup_logging(args.log_dir)  # Log level hardcoded
    
    # Log info sistema
    logging.info("Starting T5 Knowledge Distillation")
    print_system_info()
    logging.info(f"Arguments: {vars(args)}")
    
    try:
        if args.mode == 'train_teacher':
            teacher_config = load_config(args.teacher_config)
            train_teacher(args, teacher_config)
        
        elif args.mode == 'distill_student':
            student_config = load_config(args.student_config)
            distill_student(args, student_config)
        
        elif args.mode == 'benchmark':
            # Path devono essere specificati via CLI
            run_benchmark(args, args.teacher_path, args.student_path)
                
        elif args.mode == 'benchmark_code_carbon':
            run_benchmark_code_carbon(args)

        elif args.mode == 'full_pipeline':
            teacher_config = load_config(args.teacher_config)
            student_config = load_config(args.student_config)
            run_full_pipeline(args, teacher_config, student_config)
        
        else:
            raise ValueError(f"Modalità non supportata: {args.mode}")
        
        logging.info("Processo completato con successo!")
        
    except Exception as e:
        logging.error(f"Errore: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()