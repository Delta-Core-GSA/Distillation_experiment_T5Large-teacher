#!/usr/bin/env python3
"""
Main entry point for T5 distillation project
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
    
    # Operating modes
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train_teacher', 'distill_student', 'benchmark', 'full_pipeline','benchmark_code_carbon'],
                       help='Execution mode')
    
    # Model paths (for benchmark)
    parser.add_argument('--teacher_path', type=str, default=None,
                    help='Path to teacher model')
    parser.add_argument('--student_path', type=str, default=None,
                    help='Path to student model')
    
    # Model paths for codecarbon benchmark
    parser.add_argument('--teacher_path_benchmark', type=str, default=None,
                    help='Path to finetuned teacher model')
    parser.add_argument('--student_distilled_path', type=str, default=None,
                    help='Path to distilled student model')
    parser.add_argument('--student_finetuned_path', type=str, default=None,
                    help='Path to finetuned student model (baseline)')
    
    # Configurations
    parser.add_argument('--teacher_config', type=str, default='configs/teacher_config.json',
                       help='Path to teacher configuration file')
    parser.add_argument('--student_config', type=str, default='configs/student_config.json',
                       help='Path to student configuration file')
    
    # General options
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (FP16)')
    
    # Paths
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Main output directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    
    return parser.parse_args()


def setup_directories(args):
    """Create necessary directories"""
    dirs = [args.output_dir, args.log_dir, args.checkpoint_dir]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory created/verified: {dir_path}")


def train_teacher(args, teacher_config):
    """Train teacher model"""
    logging.info("=" * 50)
    logging.info("STARTING TEACHER TRAINING")
    logging.info("=" * 50)
    
    if args.resume_from_checkpoint:
        logging.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    
    trainer = TeacherTrainer(
        config=teacher_config,
        device=args.device,
        output_dir=os.path.join(args.output_dir, 'teacher'),
        checkpoint_dir=os.path.join(args.checkpoint_dir, 'teacher'),
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    final_model_path = trainer.save_final_model()
    logging.info(f"Teacher model saved at: {final_model_path}")
    
    # Evaluate model
    metrics = trainer.evaluate()
    logging.info(f"Teacher final metrics: {json.dumps(metrics, indent=2)}")
    
    return final_model_path, metrics


def distill_student(args, student_config, teacher_path=None):
    """Distill knowledge from teacher to student"""
    logging.info("=" * 50)
    logging.info("STARTING STUDENT DISTILLATION")
    logging.info("=" * 50)
    
    # If not specified, search for most recent teacher
    teacher_path = student_config.get('distillation', {}).get('teacher_model_path', None)
    
    if teacher_path is None:
        teacher_dir = os.path.join(args.output_dir, 'teacher')
        if not os.path.exists(teacher_dir):
            raise ValueError(f"Teacher model not found at {teacher_dir}")
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
    
    # Start distillation
    distiller.distill()
    
    # Save final model
    final_model_path = distiller.save_final_model()
    logging.info(f"Student model saved at: {final_model_path}")
    
    # Evaluate model
    metrics = distiller.evaluate()
    logging.info(f"Student final metrics: {json.dumps(metrics, indent=2)}")
    
    return final_model_path, metrics


def run_benchmark(args, teacher_path=None, student_path=None):
    """Run benchmark on specified models"""
    logging.info("=" * 50)
    logging.info("RUNNING BENCHMARK")
    logging.info("=" * 50)
    
    # Validation: at least one model must be specified
    if not teacher_path and not student_path:
        raise ValueError("Specify at least one path: teacher_path or student_path")
    
    # Determine benchmark type based on provided paths
    if teacher_path and student_path:
        # Comparative benchmark
        logging.info("Comparative benchmark: Teacher vs Student")
        evaluator = BenchmarkEvaluator(
            teacher_path=teacher_path,
            student_path=student_path,
            device=args.device,
            output_dir=os.path.join(args.output_dir, 'benchmark'),
            num_workers=args.num_workers
        )
        results = evaluator.run_full_evaluation()
        
    elif teacher_path:
        # Teacher only
        logging.info("Single benchmark: Teacher")
        evaluator = BenchmarkEvaluator(
            teacher_path=teacher_path,
            student_path=None,
            device=args.device,
            output_dir=os.path.join(args.output_dir, 'benchmark'),
            num_workers=args.num_workers
        )
        results = evaluator.run_single_model_evaluation('teacher')
        
    else:
        # Student only
        logging.info("Single benchmark: Student")
        evaluator = BenchmarkEvaluator(
            teacher_path=None,
            student_path=student_path,
            device=args.device,
            output_dir=os.path.join(args.output_dir, 'benchmark'),
            num_workers=args.num_workers
        )
        results = evaluator.run_single_model_evaluation('student')
    
    # Save and display results
    report_path = evaluator.save_report(results)
    logging.info(f"Report saved at: {report_path}")
    evaluator.print_summary(results)
    
    return results


def run_benchmark_code_carbon(args):
    """Run benchmark on all 3 models"""
    logging.info("=" * 50)
    logging.info("RUNNING TRIPLE BENCHMARK")
    logging.info("=" * 50)
    
    if not args.teacher_path or not args.student_distilled_path or not args.student_finetuned_path:
        raise ValueError("Triple benchmark requires all 3 paths")
    
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
    logging.info(f"Report saved at: {report_path}")
    evaluator.print_summary(results)
    
    return results


def run_full_pipeline(args, teacher_config, student_config):
    """Run full pipeline: training, distillation, benchmark"""
    logging.info("=" * 50)
    logging.info("RUNNING FULL PIPELINE")
    logging.info("=" * 50)
    
    start_time = datetime.now()
    
    # Step 1: Teacher Training
    logging.info("\n[STEP 1/3] Teacher Model Training")
    teacher_path, teacher_metrics = train_teacher(args, teacher_config)
    
    # Step 2: Student Distillation
    logging.info("\n[STEP 2/3] Student Model Distillation")
    student_path, student_metrics = distill_student(args, student_config, teacher_path)
    
    # Step 3: Comparative Benchmark
    logging.info("\n[STEP 3/3] Comparative Benchmark")
    benchmark_results = run_benchmark(args, teacher_path, student_path)
    
    # Final report
    duration = datetime.now() - start_time
    logging.info("\n" + "=" * 50)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("=" * 50)
    logging.info(f"Total time: {duration}")
    
    # Print main metrics
    teacher_rouge = teacher_metrics.get('rouge_l', 0)
    student_rouge = student_metrics.get('rouge_l', 0)
    compression = benchmark_results.get('compression_ratio', {}).get('parameters', 0)
    speedup = benchmark_results.get('speedup', 0)
    
    logging.info(f"Teacher ROUGE-L: {teacher_rouge:.4f}")
    logging.info(f"Student ROUGE-L: {student_rouge:.4f}")
    logging.info(f"Compression ratio: {compression:.2f}x")
    logging.info(f"Speedup: {speedup:.2f}x")
    
    # Save summary
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
    
    logging.info(f"Summary saved at: {summary_path}")
    return summary


def main():
    """Main function"""
    args = parse_arguments()
    
    # Initial setup
    set_seed(args.seed)
    setup_directories(args)
    setup_logging(args.log_dir)
    
    # Log system info
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
            # Paths must be specified via CLI
            run_benchmark(args, args.teacher_path, args.student_path)
                
        elif args.mode == 'benchmark_code_carbon':
            run_benchmark_code_carbon(args)
        
        elif args.mode == 'full_pipeline':
            teacher_config = load_config(args.teacher_config)
            student_config = load_config(args.student_config)
            run_full_pipeline(args, teacher_config, student_config)
        
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
        
        logging.info("Process completed successfully!")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()