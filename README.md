# T5 Knowledge Distillation for Text Summarization

Knowledge distillation framework for compressing T5 models on the CNN/DailyMail summarization task. Trains a large teacher model and distills its knowledge into a smaller, faster student model.

## Project Structure
new_code/
├── main.py                 # Main entry point
├── training.py             # Teacher training and student distillation
├── benchmark.py            # Model evaluation and comparison
├── data_loader.py          # Dataset handling
├── metrics.py              # Evaluation metrics
├── utils.py                # Utilities
├── config/
│   ├── teacher_config.json # Teacher configuration
│   └── student_config.json # Student configuration
└── distiller_zoo/          # Distillation methods
├── vanillakd.py        # KL divergence distillation
├── featurekd.py        # Feature-based distillation
└── attentionkd.py      # Attention-based distillation

## Features

- Teacher model training (T5-large or T5-base)
- Three distillation methods: vanilla KL, feature matching, attention transfer
- Student model training (T5-small,FLAN-T5-small)
- Comprehensive benchmarking with ROUGE, BLEU, and speed metrics
- Checkpoint resume capability
- Mixed precision training support



Key parameters:

distillation_type: vanilla, feature, or attention
temperature: softmax temperature for distillation
alpha: weight balance between distillation and task loss
num_epochs: training epochs
batch_size: training batch size

Usage
1. Train Teacher Model
bashpython main.py --mode train_teacher \
  --teacher_config configs/teacher_config.json \
  --device cuda \
  --output_dir ./output \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs
2. Distill Student Model
bashpython main.py --mode distill_student \
  --student_config configs/student_config.json \
  --device cuda \
  --output_dir ./output \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs
3. Resume from Checkpoint
bashpython main.py --mode distill_student \
  --student_config configs/student_config.json \
  --resume_from_checkpoint ./checkpoints/student/student_checkpoint_epoch_4.pt \
  --device cuda
4. Run Benchmark
Compare teacher and student models:
bashpython main.py --mode benchmark \
  --teacher_path ./checkpoints/teacher/checkpoint_epoch_5.pt \
  --student_path ./checkpoints/student/best_student.pt \
  --device cuda \
  --output_dir ./benchmark_results
Evaluate single model:
bashpython main.py --mode benchmark \
  --teacher_path ./checkpoints/teacher/checkpoint_epoch_5.pt \
  --device cuda
5. Full Pipeline
Run complete training, distillation, and benchmarking:
bashpython main.py --mode full_pipeline \
  --teacher_config configs/teacher_config.json \
  --student_config configs/student_config.json \
  --device cuda
Background Execution
For long training runs:
bashLOGFILE="training_$(date +%Y%m%d_%H%M%S).log"
nohup python -u main.py --mode distill_student \
  --student_config configs/student_config.json \
  --device cuda > "$LOGFILE" 2>&1 &
echo $! > training.pid
tail -f "$LOGFILE"
Output
Training produces:

checkpoints/: Model checkpoints (.pt files)
output/: Final trained models
logs/: Training logs
benchmark_results/: Evaluation reports and metrics

Benchmark generates:

benchmark_results.json: Complete metrics
comparison_table.csv: Side-by-side comparison
benchmark_report.txt: Human-readable report

Distillation Methods
Vanilla KD: Matches output logit distributions using KL divergence
Feature KD: Aligns intermediate hidden representations between models
Attention KD: Transfers attention patterns from teacher to student
Performance Metrics



Results are evaluated using:

ROUGE-1, ROUGE-2, ROUGE-L
BLEU-1, BLEU-2, BLEU-3, BLEU-4
Inference speed (samples/second)
Model compression ratio

Parameters: 12x reduction
Memory: 12x reduction
Speed: 2-3x faster inference
Performance retention: 95-97%

Dataset
Uses CNN/DailyMail dataset for abstractive summarization. The dataset is automatically downloaded on first run.


