# T5 Knowledge Distillation for Text Summarization

A knowledge distillation framework for compressing T5 models on the CNN/DailyMail summarization task, featuring four distillation strategies (vanilla KL divergence, feature matching, attention transfer, NORM), comprehensive benchmarking with ROUGE and BLEU metrics, checkpoint resume capability, and mixed precision training support for training smaller student models.


## Installation
```bash
# Clone repository
git clone <https://github.com/Delta-Core-GSA/Distillation_experiment_T5Large-teacher>
cd <Distillation_experiment_T5Large-teacher>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train Teacher Model
```bash
python main.py --mode train_teacher \
  --teacher_config configs/teacher_config.json \
  --device cuda \
  --output_dir ./output \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs
```

### Distill Student Model
```bash
python main.py --mode distill_student \
  --student_config configs/student_config.json \
  --device cuda \
  --output_dir ./output \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs
```

### Resume from Checkpoint
```bash
python main.py --mode distill_student \
  --student_config configs/student_config.json \
  --resume_from_checkpoint ./checkpoints/student/student_checkpoint_epoch_4.pt \
  --device cuda
```

### Run Benchmark

Compare teacher and student models:
```bash
python main.py --mode benchmark \
  --teacher_path ./checkpoints/teacher/checkpoint_epoch_5.pt \
  --student_path ./checkpoints/student/best_student.pt \
  --device cuda \
  --output_dir ./benchmark_results
```

Evaluate single model:
```bash
python main.py --mode benchmark \
  --teacher_path ./checkpoints/teacher/checkpoint_epoch_5.pt \
  --device cuda
```

### Full Pipeline
```bash
python main.py --mode full_pipeline \
  --teacher_config configs/teacher_config.json \
  --student_config configs/student_config.json \
  --device cuda
```

### Background Execution

For long training runs:
```bash
LOGFILE="training_$(date +%Y%m%d_%H%M%S).log"
nohup python -u main.py --mode distill_student \
  --student_config configs/student_config.json \
  --device cuda > "$LOGFILE" 2>&1 &
echo $! > training.pid
tail -f "$LOGFILE"
```

## Configuration

Edit configuration files to adjust hyperparameters:
```python
# configs/teacher_config.json or configs/student_config.json
{
  "distillation_type": "vanilla",  # vanilla, feature, attention or NORM
  "temperature": 2.0,               # Softmax temperature
  "alpha": 0.5,                     # Distillation/task loss weight
  "num_epochs": 5,                  # Training epochs
  "batch_size": 8                   # Training batch size
}
```

## Dataset Format

Uses CNN/DailyMail dataset for abstractive summarization. The dataset is automatically downloaded on first run.

## Architecture
```
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
    ├── normkd.py  
    └── attentionkd.py      # Attention-based distillation
```

## Key Features

- **Teacher training**: Supports T5-large and T5-base models
- **Multiple distillation methods**: Vanilla KL divergence, feature matching, attention transfer
- **Student models**: T5-small and FLAN-T5-small support
- **Comprehensive metrics**: ROUGE-1/2/L, BLEU-1/2/3/4, inference speed
- **Resume capability**: Checkpoint-based training recovery
- **Mixed precision**: FP16 training support

## Distillation Methods

- **Vanilla KD**: Matches output logit distributions using KL divergence
- **Feature KD**: Aligns intermediate hidden representations between models
- **Attention KD**: Transfers attention patterns from teacher to student
- **Norm KD**: Enforces similarity between layer normalization statistics of teacher and student models
## Evaluation metrics:
- ROUGE-1, ROUGE-2, ROUGE-L
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- Inference speed (samples/second)
- Model compression ratio

## Output Files

Training produces:
- `checkpoints/`: Model checkpoints (.pt files)
- `output/`: Final trained models
- `logs/`: Training logs
- `benchmark_results/`: Evaluation reports and metrics

Benchmark generates:
- `benchmark_results.json`: Complete metrics
- `comparison_table.csv`: Side-by-side comparison
- `benchmark_report.txt`: Human-readable report

## Technologies

- PyTorch 2.0+
- HuggingFace Transformers (T5, FLAN-T5)
- ROUGE and BLEU metrics
- CNN/DailyMail dataset



## Authors

@andreaeliia
