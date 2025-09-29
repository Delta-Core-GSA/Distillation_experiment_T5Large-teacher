"""
Data loader per dataset di text generation
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizer
import pandas as pd
from tqdm import tqdm
import numpy as np


def download_and_prepare_dataset(dataset_name: str, cache_dir: str) -> str:
    """
    Scarica e prepara il dataset
    
    Args:
        dataset_name: Nome dataset (es. 'cnn_dailymail')
        cache_dir: Directory cache
    Returns:
        Path al dataset processato
    """
    cache_path = Path(cache_dir) / dataset_name
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset_path = cache_path / 'processed'
    
    # Check cache
    if dataset_path.exists() and (dataset_path / 'dataset_dict.json').exists():
        logging.info(f"Dataset trovato in cache: {dataset_path}")
        return str(dataset_path)
    
    logging.info(f"Scaricando dataset {dataset_name}...")
    
    # Download
    if dataset_name == 'cnn_dailymail':
        dataset = load_dataset('cnn_dailymail', '3.0.0', cache_dir=cache_dir)
    else:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    # Save
    dataset.save_to_disk(str(dataset_path))
    
    # Save stats
    stats = {
        'dataset_name': dataset_name,
        'num_train': len(dataset['train']),
        'num_validation': len(dataset['validation']),
        'num_test': len(dataset['test']),
        'features': list(dataset['train'].features.keys())
    }
    
    with open(cache_path / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logging.info(f"Dataset salvato: {dataset_path}")
    logging.info(f"Statistics: {stats}")
    
    return str(dataset_path)


class TextGenerationDataset(Dataset):
    """Dataset per text generation (summarization, translation, etc.)"""
    
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 split: str = 'train',
                 max_length: int = 512,
                 max_target_length: Optional[int] = None,
                 max_samples: Optional[int] = None,
                 prefix: str = "summarize: "):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_target_length = max_target_length or max_length
        self.prefix = prefix
        
        # Carica dataset
        logging.info(f"Caricando dataset: {data_path}, split: {split}")
        dataset = load_from_disk(data_path)
        self.data = dataset[split]
        
        # Limita campioni se specificato
        if max_samples:
            self.data = self.data.select(range(min(max_samples, len(self.data))))
            logging.info(f"Dataset limitato a {len(self.data)} campioni")
        
        # Auto-detect colonne
        self._detect_columns()
        
    def _detect_columns(self):
        """Rileva colonne input/target automaticamente"""
        columns = self.data.column_names
        
        if 'article' in columns and 'highlights' in columns:
            self.input_col = 'article'
            self.target_col = 'highlights'
        elif 'text' in columns and 'summary' in columns:
            self.input_col = 'text'
            self.target_col = 'summary'
        else:
            # Fallback: prime due colonne
            self.input_col = columns[0]
            self.target_col = columns[1] if len(columns) > 1 else columns[0]
        
        logging.info(f"Colonne: input='{self.input_col}', target='{self.target_col}'")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Ritorna esempio tokenizzato (come liste, non tensori)"""
        example = self.data[idx]
        
        # Prepara testi
        input_text = self.prefix + example[self.input_col]
        target_text = example[self.target_col]
        
        # Tokenizza input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenizza target
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length'
        )
        
        # Labels: sostituisci padding con -100
        labels = [
            -100 if token == self.tokenizer.pad_token_id else token 
            for token in targets['input_ids']
        ]
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        }


class CustomTextDataset(Dataset):
    """Dataset per file CSV/TSV custom"""
    
    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512,
                 delimiter: str = '\t'):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Carica dati
        logging.info(f"Caricando file custom: {file_path}")
        self.data = pd.read_csv(file_path, delimiter=delimiter)
        
        # Check colonne (assume 'source' e 'target')
        if 'source' not in self.data.columns or 'target' not in self.data.columns:
            self.data.columns = ['source', 'target'] + list(self.data.columns[2:])
        
        logging.info(f"Caricati {len(self.data)} esempi")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenizza
        inputs = self.tokenizer(
            row['source'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        targets = self.tokenizer(
            row['target'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        # Labels
        labels = [
            -100 if token == self.tokenizer.pad_token_id else token 
            for token in targets['input_ids']
        ]
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels
        }


def create_data_splits(dataset, 
                      train_ratio: float = 0.8, 
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      seed: int = 42) -> Dict:
    """
    Crea split train/val/test
    
    Args:
        dataset: Dataset da splittare
        train_ratio: % training
        val_ratio: % validation
        test_ratio: % test
        seed: Random seed
    Returns:
        Dict con i tre split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # Shuffle e split
    dataset = dataset.shuffle(seed=seed)
    
    return {
        'train': dataset.select(range(train_size)),
        'validation': dataset.select(range(train_size, train_size + val_size)),
        'test': dataset.select(range(train_size + val_size, total_size))
    }


def get_dataset_statistics(dataset) -> Dict:
    """
    Calcola statistiche sul dataset
    
    Args:
        dataset: Dataset da analizzare
    Returns:
        Dict con statistiche (lunghezze, media, std, etc.)
    """
    lengths = []
    
    for example in tqdm(dataset, desc="Calcolando statistiche"):
        if isinstance(example, dict):
            for value in example.values():
                if isinstance(value, str):
                    lengths.append(len(value.split()))
    
    if not lengths:
        return {
            'num_examples': len(dataset),
            'avg_length': 0,
            'std_length': 0,
            'min_length': 0,
            'max_length': 0,
            'median_length': 0
        }
    
    return {
        'num_examples': len(dataset),
        'avg_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'min_length': int(np.min(lengths)),
        'max_length': int(np.max(lengths)),
        'median_length': float(np.median(lengths))
    }

#fase di testing con altri dataset
def inject_noise(text: str, noise_level: float = 0.1) -> str:
    """
    Data augmentation: inietta rumore nel testo
    
    Args:
        text: Testo originale
        noise_level: Livello di rumore (0-1)
    Returns:
        Testo con rumore
    """
    import random
    
    words = text.split()
    num_noise = int(len(words) * noise_level)
    
    for _ in range(num_noise):
        if not words:
            break
            
        idx = random.randint(0, len(words) - 1)
        noise_type = random.choice(['delete', 'duplicate', 'swap'])
        
        if noise_type == 'delete' and len(words) > 1:
            del words[idx]
        elif noise_type == 'duplicate':
            words.insert(idx, words[idx])
        elif noise_type == 'swap' and idx < len(words) - 1:
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
    
    return ' '.join(words)